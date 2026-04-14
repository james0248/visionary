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


def run_command(cmd: list[str], *, capture_output: bool = True, check: bool = True) -> str:
    result = subprocess.run(
        cmd,
        text=True,
        capture_output=capture_output,
        check=False,
    )
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            cmd,
            output=result.stdout,
            stderr=result.stderr,
        )
    return result.stdout


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
        return json.loads(run_command(cmd))
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
        "--quiet",
    )
    run_command(cmd, capture_output=True)


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


def gcs_object_exists(cfg: dict[str, Any], uri: str) -> bool:
    if not uri:
        return False
    cmd = gcloud_command(cfg, "storage", "ls", uri)
    result = subprocess.run(cmd, text=True, capture_output=True, check=False)
    return result.returncode == 0 and bool(result.stdout.strip())


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
        "secrets": cfg.get("secrets", {}),
        "data_disk": data_disk_cfg,
        "queued_resource": {
            "name": queued_resource_name_value,
            "candidate_index": candidate_index,
            "zone": candidate["zone"],
        },
    }


def resolve_marker_uris(cfg: dict[str, Any]) -> tuple[str, str]:
    markers_cfg = dict(cfg.get("markers", {}))
    complete_uri = markers_cfg.get("complete_uri", "")
    failure_uri = markers_cfg.get("failure_uri", "")
    return complete_uri, failure_uri


def create_queued_resource(
    cfg: dict[str, Any],
    candidate: dict[str, Any],
    *,
    candidate_index: int,
    starter_script: Path,
) -> None:
    qr_name = queued_resource_name(cfg["job"]["name"], candidate_index)
    payload = build_job_payload(
        cfg,
        candidate,
        queued_resource_name_value=qr_name,
        candidate_index=candidate_index,
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
        f"--metadata-from-file=startup-script={starter_script}",
        f"--metadata-from-file=visionary-job-json={payload_path}",
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
    cfg = load_config(config_path)
    candidates = list(cfg["candidates"])
    if not candidates:
        raise ValueError("Config must define at least one candidate TPU.")

    next_candidate_index = 0
    poll_interval = int(cfg["job"].get("poll_interval_seconds", 60))
    queue_wait_timeout = int(cfg["job"].get("queue_wait_timeout_seconds", 3600))
    complete_marker_uri, failure_marker_uri = resolve_marker_uris(cfg)

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
            print("[watcher] Training completed.")
            return

        if gcs_object_exists(cfg, failure_marker_uri):
            if existing is not None:
                index, candidate, _ = existing
                print(f"[watcher] Failure marker found; deleting queued resource {index}.")
                delete_queued_resource(
                    cfg,
                    queued_resource_name=queued_resource_name(cfg["job"]["name"], index),
                    zone=candidate["zone"],
                )
            raise SystemExit("[watcher] Training failed; see failure marker for details.")

        if existing is None:
            candidate = candidates[next_candidate_index]
            print(
                "[watcher] No queued resource exists; creating candidate "
                f"{next_candidate_index} in {candidate['zone']}."
            )
            create_queued_resource(
                cfg,
                candidate,
                candidate_index=next_candidate_index,
                starter_script=starter_script,
            )
            next_candidate_index = (next_candidate_index + 1) % len(candidates)
            time.sleep(poll_interval)
            continue

        index, candidate, desc = existing
        state = queued_resource_state(desc)
        print(f"[watcher] {queued_resource_name(cfg['job']['name'], index)} state={state}")

        if state in TERMINAL_RETRY_STATES:
            print(f"[watcher] Deleting terminal queued resource in state {state}.")
            delete_queued_resource(
                cfg,
                queued_resource_name=queued_resource_name(cfg["job"]["name"], index),
                zone=candidate["zone"],
            )
            next_candidate_index = (index + 1) % len(candidates)
            time.sleep(poll_interval)
            continue

        if state in PENDING_STATES and wait_timed_out(desc, queue_wait_timeout):
            print(
                "[watcher] Queued resource wait timed out; deleting and rotating "
                f"to candidate {(index + 1) % len(candidates)}."
            )
            delete_queued_resource(
                cfg,
                queued_resource_name=queued_resource_name(cfg["job"]["name"], index),
                zone=candidate["zone"],
            )
            next_candidate_index = (index + 1) % len(candidates)
            time.sleep(poll_interval)
            continue

        if state not in LIVE_STATES:
            raise RuntimeError(f"Unhandled queued resource state: {state}")

        time.sleep(poll_interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[watcher] Interrupted.", file=sys.stderr)
