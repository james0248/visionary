#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


PENDING_STATES = {"ACCEPTED", "WAITING_FOR_RESOURCES", "PROVISIONING", "CREATING"}
TERMINAL_RETRY_STATES = {"FAILED", "SUSPENDED"}
LIVE_STATES = PENDING_STATES | {"ACTIVE", "SUSPENDING", "DELETING"}
DELETE_IN_PROGRESS_STATES = {"SUSPENDING", "DELETING"}

TRC_ACCELERATOR_LIMITS = {
    ("v5e", "europe-west4-b", True): 64,
    ("v6e", "europe-west4-a", True): 64,
    ("v4", "us-central2-b", False): 32,
    ("v4", "us-central2-b", True): 32,
    ("v6e", "us-east1-d", True): 64,
    ("v5e", "us-central1-a", True): 64,
}

DEFAULT_RUNTIME_VERSION = {
    "v4": "tpu-ubuntu2204-base",
    "v5e": "v2-alpha-tpuv5-lite",
    "v6e": "v2-alpha-tpuv6e",
}

VALID_CHIP_COUNTS = {
    "v4": {4, 8, 16, 32},
    "v5e": {1, 4, 8, 16, 32, 64},
    "v6e": {1, 4, 8, 16, 32, 64},
}

FAMILY_ALIASES = {
    "v4": "v4",
    "v5e": "v5e",
    "v5litepod": "v5e",
    "v6e": "v6e",
}


def load_config(path: Path) -> dict[str, Any]:
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    default_spot = bool(cfg.get("job", {}).get("spot", True))
    has_machine = bool(cfg.get("machine"))
    has_candidates = bool(cfg.get("candidates"))
    if has_machine and has_candidates:
        raise ValueError("Use either `machine` or `candidates`, not both.")
    if has_machine:
        cfg["candidates"] = [normalize_machine(cfg["machine"], default_spot=default_spot)]
    elif has_candidates:
        cfg["candidates"] = [
            normalize_candidate({**candidate, "spot": candidate.get("spot", default_spot)})
            for candidate in cfg["candidates"]
        ]
    else:
        raise ValueError("Config must define either `machine` or `candidates`.")
    return cfg


def parse_accelerator_type(accelerator_type: str) -> tuple[str, int]:
    if match := re.fullmatch(r"v6e-(\d+)", accelerator_type):
        return "v6e", int(match.group(1))
    if match := re.fullmatch(r"v5litepod-(\d+)", accelerator_type):
        return "v5e", int(match.group(1))
    if match := re.fullmatch(r"v4-(\d+)", accelerator_type):
        tensorcore_count = int(match.group(1))
        if tensorcore_count % 2 != 0:
            raise ValueError(f"Unsupported v4 accelerator type: {accelerator_type}")
        return "v4", tensorcore_count // 2
    raise ValueError(f"Unsupported accelerator type: {accelerator_type}")


def normalize_family(family: str) -> str:
    normalized = FAMILY_ALIASES.get(str(family).lower())
    if normalized is None:
        raise ValueError(f"Unsupported TPU family: {family}")
    return normalized


def accelerator_type_from_family(family: str, chip_count: int) -> str:
    valid_chip_counts = VALID_CHIP_COUNTS[family]
    if chip_count not in valid_chip_counts:
        valid_values = ", ".join(str(value) for value in sorted(valid_chip_counts))
        raise ValueError(
            f"Unsupported chip count {chip_count} for {family}. Valid values: {valid_values}."
        )
    if family == "v4":
        return f"v4-{chip_count * 2}"
    if family == "v5e":
        return f"v5litepod-{chip_count}"
    return f"v6e-{chip_count}"


def allowed_zones_for_family(family: str, spot: bool) -> list[str]:
    return sorted(
        zone
        for candidate_family, zone, candidate_spot in TRC_ACCELERATOR_LIMITS
        if candidate_family == family and candidate_spot == spot
    )


def resolve_zone(*, family: str, spot: bool, region: str | None, zone: str | None) -> str:
    allowed_zones = allowed_zones_for_family(family, spot)
    if zone is not None:
        if zone not in allowed_zones:
            raise ValueError(
                f"Zone {zone} is not allowed for {family} with spot={spot}. "
                f"Allowed zones: {', '.join(allowed_zones)}."
            )
        if region is not None and not zone.startswith(f"{region}-"):
            raise ValueError(f"Zone {zone} does not belong to region {region}.")
        return zone

    if region is None:
        if len(allowed_zones) == 1:
            return allowed_zones[0]
        raise ValueError(
            f"Multiple zones are available for {family} with spot={spot}; specify `region` or `zone`."
        )

    matching_zones = [candidate_zone for candidate_zone in allowed_zones if candidate_zone.startswith(f"{region}-")]
    if not matching_zones:
        raise ValueError(
            f"No allowed zone found for {family} in region {region} with spot={spot}. "
            f"Allowed zones: {', '.join(allowed_zones)}."
        )
    if len(matching_zones) > 1:
        raise ValueError(
            f"Region {region} maps to multiple allowed zones for {family} with spot={spot}: "
            f"{', '.join(matching_zones)}. Specify `zone` instead."
        )
    return matching_zones[0]


def normalize_machine(machine: dict[str, Any], *, default_spot: bool) -> dict[str, Any]:
    machine = dict(machine)
    family = normalize_family(str(machine["family"]))
    chip_count = int(machine["chips"])
    spot = bool(machine.get("spot", default_spot))
    zone = resolve_zone(
        family=family,
        spot=spot,
        region=machine.get("region"),
        zone=machine.get("zone"),
    )

    candidate = {
        "zone": zone,
        "spot": spot,
        "accelerator_type": accelerator_type_from_family(family, chip_count),
        "runtime_version": machine.get("runtime_version", DEFAULT_RUNTIME_VERSION[family]),
    }
    for key in (
        "data_disk",
        "node_count",
        "node_id",
        "node_prefix",
        "valid_until_duration",
        "network",
        "subnetwork",
        "internal_ips",
        "labels",
    ):
        if key in machine:
            candidate[key] = machine[key]
    return normalize_candidate(candidate)


def normalize_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    candidate = dict(candidate)
    accelerator_type = candidate.get("accelerator_type")
    if not accelerator_type:
        raise ValueError("TRC watcher candidates must use accelerator_type.")

    family, chip_count = parse_accelerator_type(str(accelerator_type))
    zone = str(candidate["zone"])
    spot = bool(candidate.get("spot", False))
    max_chips = TRC_ACCELERATOR_LIMITS.get((family, zone, spot))
    if max_chips is None:
        raise ValueError(
            f"Candidate {accelerator_type} in {zone} with spot={spot} is outside the TRC allowlist."
        )
    if chip_count > max_chips:
        raise ValueError(
            f"Candidate {accelerator_type} requests {chip_count} chips, which exceeds the "
            f"TRC limit of {max_chips} chips for {family} in {zone} with spot={spot}."
        )

    candidate["family"] = family
    candidate["chip_count"] = chip_count
    candidate.setdefault("runtime_version", DEFAULT_RUNTIME_VERSION[family])
    return candidate


TRANSIENT_GCLOUD_ERROR_SUBSTRINGS = (
    "there was a problem refreshing your current auth tokens",
    "max retries exceeded with url",
    "sslerror(",
    "unexpected_eof_while_reading",
    "eof occurred in violation of protocol",
    "connection reset by peer",
    "connection aborted",
    "connection refused",
    "no route to host",
    "temporary failure in name resolution",
    "name or service not known",
    "timed out",
    "deadline exceeded",
    "service unavailable",
    "internal error encountered",
)


def resolve_gcloud_retry_policy(cfg: dict[str, Any] | None) -> tuple[int, float, float]:
    watcher_cfg = dict(cfg.get("watcher", {})) if cfg else {}
    attempts = max(int(watcher_cfg.get("gcloud_retry_attempts", 5)), 1)
    backoff_seconds = max(float(watcher_cfg.get("gcloud_retry_backoff_seconds", 5)), 0.0)
    max_backoff_seconds = max(
        float(watcher_cfg.get("gcloud_retry_max_backoff_seconds", 60)),
        backoff_seconds,
    )
    return attempts, backoff_seconds, max_backoff_seconds


def is_transient_gcloud_failure(
    cmd: list[str], result: subprocess.CompletedProcess[str]
) -> bool:
    if not cmd or Path(str(cmd[0])).name != "gcloud":
        return False

    combined_output = "\n".join(
        value for value in (result.stdout, result.stderr) if isinstance(value, str) and value
    ).lower()
    if not combined_output:
        return False

    if any(token in combined_output for token in TRANSIENT_GCLOUD_ERROR_SUBSTRINGS):
        return True

    if "http" in combined_output and any(
        marker in combined_output
        for marker in (
            "(429)",
            "(500)",
            "(502)",
            "(503)",
            "(504)",
            " 429 ",
            " 500 ",
            " 502 ",
            " 503 ",
            " 504 ",
        )
    ):
        return True

    return False


def run_subprocess(
    cmd: list[str],
    *,
    cfg: dict[str, Any] | None = None,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    attempts, backoff_seconds, max_backoff_seconds = resolve_gcloud_retry_policy(cfg)
    if not cmd or Path(str(cmd[0])).name != "gcloud":
        attempts = 1

    next_delay = backoff_seconds
    for attempt in range(1, attempts + 1):
        result = subprocess.run(
            cmd,
            text=True,
            capture_output=capture_output,
            check=False,
        )
        if result.returncode == 0:
            return result
        if attempt >= attempts or not is_transient_gcloud_failure(cmd, result):
            return result

        detail = (result.stderr or result.stdout or "").strip().splitlines()
        summary = detail[-1] if detail else f"exit code {result.returncode}"
        print(
            "[watcher] Transient gcloud failure; retrying in "
            f"{next_delay:.0f}s ({attempt}/{attempts - 1}). {summary}"
        )
        time.sleep(next_delay)
        next_delay = min(max_backoff_seconds, max(next_delay * 2, 1.0))

    raise RuntimeError("unreachable")


def run_command(
    cmd: list[str],
    *,
    cfg: dict[str, Any] | None = None,
    capture_output: bool = True,
    check: bool = True,
) -> str:
    result = run_subprocess(cmd, cfg=cfg, capture_output=capture_output)
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            cmd,
            output=result.stdout,
            stderr=result.stderr,
        )
    return result.stdout


def command_failure_detail(exc: subprocess.CalledProcessError) -> str:
    return (exc.stderr or exc.output or "").strip()


def log_command_failure(exc: subprocess.CalledProcessError, *, message: str) -> None:
    command = " ".join(str(part) for part in exc.cmd)
    print(f"[watcher] {message}: {command}", file=sys.stderr)
    detail = command_failure_detail(exc)
    if detail:
        print(detail, file=sys.stderr)


def gcloud_command(cfg: dict[str, Any], *args: str, json_output: bool = False) -> list[str]:
    watcher_cfg = cfg.get("watcher", {})
    cmd = ["gcloud", *args]
    impersonate = watcher_cfg.get("impersonate_service_account")
    if impersonate:
        cmd.append(f"--impersonate-service-account={impersonate}")
    if json_output:
        cmd.append("--format=json")
    return cmd


def maybe_describe_queued_resource(
    cfg: dict[str, Any], *, queued_resource_name: str, zone: str
) -> dict[str, Any] | None:
    cmd = gcloud_command(
        cfg,
        "compute",
        "tpus",
        "queued-resources",
        "describe",
        queued_resource_name,
        f"--project={cfg['project']}",
        f"--zone={zone}",
        json_output=True,
    )
    try:
        return json.loads(run_command(cmd, cfg=cfg))
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr or ""
        if "NOT_FOUND" in stderr or "not found" in stderr.lower():
            return None
        raise


def delete_queued_resource(cfg: dict[str, Any], *, queued_resource_name: str, zone: str) -> None:
    cmd = gcloud_command(
        cfg,
        "compute",
        "tpus",
        "queued-resources",
        "delete",
        queued_resource_name,
        f"--project={cfg['project']}",
        f"--zone={zone}",
        "--async",
        "--force",
        "--quiet",
    )
    try:
        run_command(cmd, cfg=cfg, capture_output=True)
    except subprocess.CalledProcessError as exc:
        detail = command_failure_detail(exc).lower()
        if "not_found" in detail or "not found" in detail:
            print(
                f"[watcher] Queued resource {queued_resource_name} is already absent.",
                file=sys.stderr,
            )
            return
        raise


def queued_resource_state(desc: dict[str, Any]) -> str:
    state = desc.get("state")
    if isinstance(state, dict):
        return str(state.get("state", "UNKNOWN"))
    return str(state or "UNKNOWN")


def queued_resource_create_time(desc: dict[str, Any]) -> datetime | None:
    raw = desc.get("createTime")
    if not raw:
        return None
    return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))


def metadata_to_dict(metadata: Any) -> dict[str, str]:
    if isinstance(metadata, dict):
        return {str(key): str(value) for key, value in metadata.items()}
    if isinstance(metadata, list):
        items: dict[str, str] = {}
        for entry in metadata:
            if not isinstance(entry, dict):
                continue
            key = entry.get("key", entry.get("name"))
            value = entry.get("value")
            if key is None or value is None:
                continue
            items[str(key)] = str(value)
        return items
    return {}


def queued_resource_node_specs(desc: dict[str, Any]) -> list[dict[str, Any]]:
    tpu_desc = desc.get("tpu", {})
    if not isinstance(tpu_desc, dict):
        return []
    node_specs = tpu_desc.get("nodeSpec", tpu_desc.get("node_spec", []))
    if not isinstance(node_specs, list):
        return []
    return [spec for spec in node_specs if isinstance(spec, dict)]


def queued_resource_metadata_mismatch_reason(
    desc: dict[str, Any], *, expected_startup_script: str, expected_payload_json: str
) -> str | None:
    node_specs = queued_resource_node_specs(desc)
    if not node_specs:
        return "missing nodeSpec metadata"

    for spec in node_specs:
        node = spec.get("node", {})
        if not isinstance(node, dict):
            return "missing node metadata"

        metadata = metadata_to_dict(node.get("metadata", {}))
        if metadata.get("startup-script") != expected_startup_script:
            return "startup-script metadata mismatch"
        if metadata.get("visionary-job-json") != expected_payload_json:
            return "visionary-job-json metadata mismatch"

    return None


def gcs_object_exists(cfg: dict[str, Any], uri: str) -> bool:
    if not uri:
        return False
    cmd = gcloud_command(cfg, "storage", "ls", uri)
    result = run_subprocess(cmd, cfg=cfg, capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr or ""
        if "matched no objects" in stderr.lower() or "does not exist" in stderr.lower():
            return False
        raise subprocess.CalledProcessError(
            result.returncode,
            cmd,
            output=result.stdout,
            stderr=result.stderr,
        )
    return bool(result.stdout.strip())


def gcs_read_text(cfg: dict[str, Any], uri: str) -> str:
    if not uri:
        return ""
    cmd = gcloud_command(cfg, "storage", "cat", uri)
    return run_command(cmd, cfg=cfg)


def gcs_list_objects(cfg: dict[str, Any], uri_prefix: str) -> list[str]:
    if not uri_prefix:
        return []
    cmd = gcloud_command(cfg, "storage", "ls", f"{uri_prefix.rstrip('/')}/**")
    result = run_subprocess(cmd, cfg=cfg, capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr or ""
        if "matched no objects" in stderr.lower() or "does not exist" in stderr.lower():
            return []
        raise subprocess.CalledProcessError(
            result.returncode,
            cmd,
            output=result.stdout,
            stderr=result.stderr,
        )
    return sorted(
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip().startswith("gs://")
    )


def gcs_write_text(cfg: dict[str, Any], uri: str, text: str) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        handle.write(text)
        payload_path = Path(handle.name)
    try:
        cmd = gcloud_command(cfg, "storage", "cp", payload_path.as_posix(), uri)
        run_command(cmd, cfg=cfg)
    finally:
        payload_path.unlink(missing_ok=True)


def sanitize_name(value: str, *, max_len: int = 40) -> str:
    value = re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")
    return value[:max_len].strip("-")


def queued_resource_name(job_name: str, candidate_index: int) -> str:
    return f"{sanitize_name(job_name)}-{candidate_index}"


def node_name(job_name: str, candidate_index: int) -> str:
    return f"{sanitize_name(job_name, max_len=30)}-{candidate_index}-node"


def disk_name_from_source(source: str) -> str:
    return source.rstrip("/").split("/")[-1]


def build_job_payload(
    cfg: dict[str, Any],
    candidate: dict[str, Any],
    *,
    queued_resource_name_value: str,
    candidate_index: int,
    attempt_id: int | None = None,
) -> dict[str, Any]:
    markers_cfg = dict(cfg.get("markers", {}))
    training_cfg = dict(cfg["training"])

    data_disk_cfg = dict(candidate.get("data_disk", {}))
    if data_disk_cfg.get("source") and "name" not in data_disk_cfg:
        data_disk_cfg["name"] = disk_name_from_source(data_disk_cfg["source"])

    return {
        "project": cfg["project"],
        "job_name": cfg["job"]["name"],
        "repo": cfg["repo"],
        "setup": cfg["setup"],
        "training": training_cfg,
        "markers": markers_cfg,
        "logs": cfg.get("logs", {}),
        "secrets": cfg.get("secrets", {}),
        "data_disk": data_disk_cfg,
        "queued_resource": {
            "name": queued_resource_name_value,
            "candidate_index": candidate_index,
            "zone": candidate["zone"],
            **({"attempt_id": attempt_id} if attempt_id is not None else {}),
        },
    }


def resolve_marker_paths(cfg: dict[str, Any]) -> tuple[str, str]:
    markers_cfg = dict(cfg.get("markers", {}))
    complete_uri = markers_cfg.get("complete_uri", "")
    failure_prefix = str(markers_cfg.get("failure_prefix", "")).strip()
    return complete_uri, failure_prefix


def resolve_state_uri(cfg: dict[str, Any]) -> str:
    watcher_cfg = dict(cfg.get("watcher", {}))
    return str(watcher_cfg.get("state_uri", "")).strip()


def resolve_failure_policy(cfg: dict[str, Any]) -> dict[str, Any]:
    failure_cfg = dict(cfg.get("failure_policy", {}))
    mode = str(failure_cfg.get("mode", "retry")).lower()
    if mode not in {"retry", "stop"}:
        raise ValueError("failure_policy.mode must be one of: retry, stop")

    max_retries = int(failure_cfg.get("max_retries", 3))
    retry_backoff_seconds = int(failure_cfg.get("retry_backoff_seconds", 30))
    if retry_backoff_seconds < 0:
        raise ValueError("failure_policy.retry_backoff_seconds must be >= 0")

    return {
        "mode": mode,
        "max_retries": max_retries,
        "retry_backoff_seconds": retry_backoff_seconds,
    }


def load_marker_payload(cfg: dict[str, Any], uri: str) -> dict[str, Any] | None:
    if not uri:
        return None
    try:
        payload = gcs_read_text(cfg, uri)
    except subprocess.CalledProcessError:
        return None
    if not payload.strip():
        return None
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def default_watcher_state() -> dict[str, Any]:
    return {
        "next_attempt_id": 1,
        "current_attempt_id": None,
        "current_candidate_index": None,
        "last_processed_failure_uri": "",
        "failure_retries": 0,
    }


def load_watcher_state(cfg: dict[str, Any]) -> dict[str, Any]:
    state = default_watcher_state()
    state_uri = resolve_state_uri(cfg)
    if not state_uri:
        return state

    try:
        payload = gcs_read_text(cfg, state_uri)
    except subprocess.CalledProcessError:
        return state
    if not payload.strip():
        return state

    try:
        raw = json.loads(payload)
    except json.JSONDecodeError:
        return state
    if not isinstance(raw, dict):
        return state

    for key in ("next_attempt_id", "failure_retries", "current_attempt_id", "current_candidate_index"):
        value = raw.get(key)
        if value is None:
            continue
        try:
            state[key] = int(value)
        except (TypeError, ValueError):
            continue
    last_processed_failure_uri = raw.get("last_processed_failure_uri")
    if isinstance(last_processed_failure_uri, str):
        state["last_processed_failure_uri"] = last_processed_failure_uri
    return state


def save_watcher_state(cfg: dict[str, Any], state: dict[str, Any]) -> None:
    state_uri = resolve_state_uri(cfg)
    if not state_uri:
        return
    payload = json.dumps(state, indent=2, sort_keys=True)
    gcs_write_text(cfg, state_uri, payload)


def queued_resource_job_payload(desc: dict[str, Any]) -> dict[str, Any] | None:
    for spec in queued_resource_node_specs(desc):
        node = spec.get("node", {})
        if not isinstance(node, dict):
            continue
        metadata = metadata_to_dict(node.get("metadata", {}))
        payload = metadata.get("visionary-job-json")
        if not payload:
            continue
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None


def queued_resource_attempt_id(desc: dict[str, Any]) -> int | None:
    payload = queued_resource_job_payload(desc)
    if not isinstance(payload, dict):
        return None
    queued_resource_cfg = payload.get("queued_resource", {})
    if not isinstance(queued_resource_cfg, dict):
        return None
    raw_attempt_id = queued_resource_cfg.get("attempt_id")
    if raw_attempt_id is None:
        return None
    try:
        return int(raw_attempt_id)
    except (TypeError, ValueError):
        return None


def next_failure_event_uri(cfg: dict[str, Any], failure_prefix: str, last_processed_uri: str) -> str:
    for uri in gcs_list_objects(cfg, failure_prefix):
        if uri > last_processed_uri:
            return uri
    return ""


def failure_candidate_index(
    payload: dict[str, Any] | None,
    *,
    existing_index: int | None,
    fallback_index: int,
    candidate_count: int,
) -> int:
    if payload is not None:
        raw_candidate_index = payload.get("candidate_index")
        if raw_candidate_index is not None:
            try:
                return int(raw_candidate_index) % candidate_count
            except (TypeError, ValueError):
                pass

        queued_name = payload.get("queued_resource_name")
        if isinstance(queued_name, str):
            match = re.search(r"-(\d+)$", queued_name)
            if match is not None:
                return int(match.group(1)) % candidate_count

    if existing_index is not None:
        return existing_index % candidate_count
    return fallback_index % candidate_count


def print_failure_details(cfg: dict[str, Any], failure_uri: str) -> dict[str, Any] | None:
    payload = load_marker_payload(cfg, failure_uri)
    if payload is None:
        print(f"[watcher] Failure event found at {failure_uri}.")
        return None

    details = []
    for key in ("job_name", "node_name", "queued_resource_name", "exit_code", "timestamp"):
        value = payload.get(key)
        if value not in (None, ""):
            details.append(f"{key}={value}")
    if details:
        print(f"[watcher] Failure details: {', '.join(details)}")
    else:
        print(f"[watcher] Failure event found at {failure_uri}.")

    log_uri = payload.get("log_uri")
    if isinstance(log_uri, str) and log_uri:
        print(f"[watcher] Full startup log: {log_uri}")

    archived_log_uri = payload.get("archived_log_uri")
    if isinstance(archived_log_uri, str) and archived_log_uri:
        print(f"[watcher] Archived failure log: {archived_log_uri}")

    upload_error = payload.get("log_upload_error")
    if isinstance(upload_error, str) and upload_error:
        print(f"[watcher] Log upload error: {upload_error}")

    log_tail = payload.get("log_tail")
    if isinstance(log_tail, str) and log_tail.strip():
        print("[watcher] Failure log tail:")
        print(log_tail.rstrip())
    return payload


def create_queued_resource(
    cfg: dict[str, Any],
    candidate: dict[str, Any],
    *,
    candidate_index: int,
    attempt_id: int,
    starter_script: Path,
) -> None:
    qr_name = queued_resource_name(cfg["job"]["name"], candidate_index)
    payload = build_job_payload(
        cfg,
        candidate,
        queued_resource_name_value=qr_name,
        candidate_index=candidate_index,
        attempt_id=attempt_id,
    )

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(payload, handle, indent=2)
        payload_path = Path(handle.name)

    cmd = gcloud_command(
        cfg,
        "compute",
        "tpus",
        "queued-resources",
        "create",
        qr_name,
        f"--project={cfg['project']}",
        f"--zone={candidate['zone']}",
        f"--runtime-version={candidate['runtime_version']}",
        f"--service-account={cfg['starter_service_account']['email']}",
        f"--scopes={','.join(cfg['starter_service_account'].get('scopes', ['https://www.googleapis.com/auth/cloud-platform']))}",
        (
            f"--metadata-from-file="
            f"startup-script={starter_script},visionary-job-json={payload_path}"
        ),
    )

    cmd.append(f"--accelerator-type={candidate['accelerator_type']}")

    node_count = int(candidate.get("node_count", 1))
    if node_count == 1:
        cmd.append(f"--node-id={candidate.get('node_id', node_name(cfg['job']['name'], candidate_index))}")
    else:
        cmd.extend(
            [
                f"--node-count={node_count}",
                f"--node-prefix={candidate.get('node_prefix', node_name(cfg['job']['name'], candidate_index))}",
            ]
        )

    if candidate.get("spot", cfg["job"].get("spot", True)):
        cmd.append("--spot")
    if candidate.get("valid_until_duration"):
        cmd.append(f"--valid-until-duration={candidate['valid_until_duration']}")
    if candidate.get("network"):
        cmd.append(f"--network={candidate['network']}")
    if candidate.get("subnetwork"):
        cmd.append(f"--subnetwork={candidate['subnetwork']}")
    if candidate.get("internal_ips"):
        cmd.append("--internal-ips")
    if candidate.get("labels"):
        labels = ",".join(f"{key}={value}" for key, value in candidate["labels"].items())
        cmd.append(f"--labels={labels}")
    if candidate.get("data_disk"):
        disk_cfg = candidate["data_disk"]
        disk_parts = [f"source={disk_cfg['source']}"]
        if disk_cfg.get("mode"):
            disk_parts.append(f"mode={disk_cfg['mode']}")
        cmd.append(f"--data-disk={','.join(disk_parts)}")

    try:
        run_command(cmd)
    finally:
        payload_path.unlink(missing_ok=True)


def discover_existing_resource(
    cfg: dict[str, Any], candidates: list[dict[str, Any]]
) -> tuple[int, dict[str, Any], dict[str, Any]] | None:
    matches = []
    for index, candidate in enumerate(candidates):
        desc = maybe_describe_queued_resource(
            cfg,
            queued_resource_name=queued_resource_name(cfg["job"]["name"], index),
            zone=candidate["zone"],
        )
        if desc is not None:
            matches.append((index, candidate, desc))
    if len(matches) > 1:
        raise RuntimeError("More than one managed queued resource exists; refusing to continue.")
    return matches[0] if matches else None


def wait_timed_out(desc: dict[str, Any], timeout_seconds: int) -> bool:
    if timeout_seconds <= 0:
        return False
    created_at = queued_resource_create_time(desc)
    if created_at is None:
        return False
    age = datetime.now(timezone.utc) - created_at.astimezone(timezone.utc)
    return age.total_seconds() >= timeout_seconds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continuously recreate Cloud TPU queued resources from an allowlisted config."
    )
    parser.add_argument("--config", required=True, help="Path to watcher YAML config.")
    parser.add_argument(
        "--starter-script",
        default=str(Path(__file__).with_name("starter.sh")),
        help="Startup script injected into the TPU VM.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    starter_script = Path(args.starter_script).resolve()
    starter_script_contents = starter_script.read_text()
    cfg = load_config(config_path)
    candidates = list(cfg["candidates"])
    if not candidates:
        raise ValueError("Config must define at least one candidate TPU.")

    state = load_watcher_state(cfg)
    current_candidate_index = state.get("current_candidate_index")
    if current_candidate_index is None:
        next_candidate_index = 0
    else:
        next_candidate_index = int(current_candidate_index) % len(candidates)
    poll_interval = int(cfg["job"].get("poll_interval_seconds", 60))
    queue_wait_timeout = int(cfg["job"].get("queue_wait_timeout_seconds", 3600))
    complete_marker_uri, failure_prefix = resolve_marker_paths(cfg)
    failure_policy = resolve_failure_policy(cfg)
    failure_retries = int(state["failure_retries"])

    print(f"[watcher] Managing job {cfg['job']['name']} from {config_path}")
    while True:
        existing = discover_existing_resource(cfg, candidates)

        if gcs_object_exists(cfg, complete_marker_uri):
            if existing is not None:
                index, candidate, _ = existing
                print(f"[watcher] Completion marker found; deleting queued resource {index}.")
                delete_queued_resource(
                    cfg,
                    queued_resource_name=queued_resource_name(cfg["job"]["name"], index),
                    zone=candidate["zone"],
                )
            state["current_attempt_id"] = None
            state["current_candidate_index"] = None
            state["failure_retries"] = 0
            save_watcher_state(cfg, state)
            print("[watcher] Training completed.")
            return

        failure_event_uri = next_failure_event_uri(
            cfg,
            failure_prefix,
            str(state.get("last_processed_failure_uri", "")),
        )
        if failure_event_uri:
            existing_index = existing[0] if existing is not None else None
            payload = print_failure_details(cfg, failure_event_uri)
            if existing is not None:
                index, candidate, _ = existing
                print(f"[watcher] Failure event found; deleting queued resource {index}.")
                delete_queued_resource(
                    cfg,
                    queued_resource_name=queued_resource_name(cfg["job"]["name"], index),
                    zone=candidate["zone"],
                )
            state["last_processed_failure_uri"] = failure_event_uri
            state["current_attempt_id"] = None
            state["current_candidate_index"] = None
            if failure_policy["mode"] == "retry" and (
                failure_policy["max_retries"] < 0 or failure_retries < failure_policy["max_retries"]
            ):
                failure_retries += 1
                state["failure_retries"] = failure_retries
                save_watcher_state(cfg, state)
                retry_target_index = failure_candidate_index(
                    payload,
                    existing_index=existing_index,
                    fallback_index=next_candidate_index,
                    candidate_count=len(candidates),
                )
                retries_label = (
                    "unbounded"
                    if failure_policy["max_retries"] < 0
                    else f"{failure_retries}/{failure_policy['max_retries']}"
                )
                print(
                    "[watcher] Failure event consumed; retrying candidate "
                    f"{retry_target_index} after {failure_policy['retry_backoff_seconds']}s "
                    f"(attempt {retries_label})."
                )
                next_candidate_index = retry_target_index
                time.sleep(failure_policy["retry_backoff_seconds"])
                continue

            if failure_policy["mode"] == "retry":
                save_watcher_state(cfg, state)
                max_retries = failure_policy["max_retries"]
                raise SystemExit(
                    "[watcher] Training failed and retry budget is exhausted "
                    f"({failure_retries}/{max_retries})."
                )
            save_watcher_state(cfg, state)
            raise SystemExit("[watcher] Training failed; see failure event details above.")

        if existing is None:
            candidate = candidates[next_candidate_index]
            attempt_id = int(state["next_attempt_id"])
            state["current_attempt_id"] = attempt_id
            state["current_candidate_index"] = next_candidate_index
            state["next_attempt_id"] = attempt_id + 1
            save_watcher_state(cfg, state)
            print(
                "[watcher] No queued resource exists; creating candidate "
                f"{next_candidate_index} in {candidate['zone']} for attempt {attempt_id}."
            )
            create_queued_resource(
                cfg,
                candidate,
                candidate_index=next_candidate_index,
                attempt_id=attempt_id,
                starter_script=starter_script,
            )
            next_candidate_index = (next_candidate_index + 1) % len(candidates)
            time.sleep(poll_interval)
            continue

        index, candidate, desc = existing
        attempt_id = state.get("current_attempt_id")
        if attempt_id is None:
            attempt_id = queued_resource_attempt_id(desc)
        expected_payload_json = json.dumps(
            build_job_payload(
                cfg,
                candidate,
                queued_resource_name_value=queued_resource_name(cfg["job"]["name"], index),
                candidate_index=index,
                attempt_id=attempt_id,
            ),
            indent=2,
        )
        queued_state = queued_resource_state(desc)
        mismatch_reason = queued_resource_metadata_mismatch_reason(
            desc,
            expected_startup_script=starter_script_contents,
            expected_payload_json=expected_payload_json,
        )
        if mismatch_reason is not None:
            if queued_state in DELETE_IN_PROGRESS_STATES:
                print(
                    "[watcher] Existing queued resource metadata mismatch "
                    f"({mismatch_reason}), but deletion is already in progress "
                    f"(state={queued_state})."
                )
                time.sleep(poll_interval)
                continue
            print(
                "[watcher] Existing queued resource metadata mismatch "
                f"({mismatch_reason}); deleting and recreating candidate {index}."
            )
            delete_queued_resource(
                cfg,
                queued_resource_name=queued_resource_name(cfg["job"]["name"], index),
                zone=candidate["zone"],
            )
            state["current_attempt_id"] = None
            state["current_candidate_index"] = None
            save_watcher_state(cfg, state)
            next_candidate_index = index
            time.sleep(poll_interval)
            continue

        print(f"[watcher] {queued_resource_name(cfg['job']['name'], index)} state={queued_state}")

        if queued_state in TERMINAL_RETRY_STATES:
            print(f"[watcher] Deleting terminal queued resource in state {queued_state}.")
            delete_queued_resource(
                cfg,
                queued_resource_name=queued_resource_name(cfg["job"]["name"], index),
                zone=candidate["zone"],
            )
            state["current_attempt_id"] = None
            state["current_candidate_index"] = None
            save_watcher_state(cfg, state)
            next_candidate_index = (index + 1) % len(candidates)
            time.sleep(poll_interval)
            continue

        if queued_state in PENDING_STATES and wait_timed_out(desc, queue_wait_timeout):
            print(
                "[watcher] Queued resource wait timed out; deleting and rotating "
                f"to candidate {(index + 1) % len(candidates)}."
            )
            delete_queued_resource(
                cfg,
                queued_resource_name=queued_resource_name(cfg["job"]["name"], index),
                zone=candidate["zone"],
            )
            state["current_attempt_id"] = None
            state["current_candidate_index"] = None
            save_watcher_state(cfg, state)
            next_candidate_index = (index + 1) % len(candidates)
            time.sleep(poll_interval)
            continue

        if queued_state not in LIVE_STATES:
            raise RuntimeError(f"Unhandled queued resource state: {queued_state}")

        time.sleep(poll_interval)


if __name__ == "__main__":
    while True:
        try:
            main()
            break
        except KeyboardInterrupt:
            print("[watcher] Interrupted.", file=sys.stderr)
            break
        except subprocess.CalledProcessError as exc:
            log_command_failure(exc, message="Ignoring command failure; retrying watcher loop")
            time.sleep(60)
