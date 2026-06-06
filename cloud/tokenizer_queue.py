#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import watcher


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_TOKENIZER_CONFIGS = [
    SCRIPT_DIR / "breakout_tokenizer.yaml",
    SCRIPT_DIR / "pacman_tokenizer.yaml",
    SCRIPT_DIR / "qbert_tokenizer.yaml",
    SCRIPT_DIR / "seaquest_tokenizer.yaml",
    SCRIPT_DIR / "space_invaders_tokenizer.yaml",
    SCRIPT_DIR / "enduro_tokenizer.yaml",
]
DEFAULT_DYNAMICS_CONFIGS = [
    SCRIPT_DIR / "breakout_dynamics.yaml",
    SCRIPT_DIR / "pacman_dynamics.yaml",
    SCRIPT_DIR / "qbert_dynamics.yaml",
    SCRIPT_DIR / "seaquest_dynamics.yaml",
    SCRIPT_DIR / "space_invaders_dynamics.yaml",
    SCRIPT_DIR / "enduro_dynamics.yaml",
]

DEFAULT_CONFIGS_BY_KIND = {
    "tokenizer": DEFAULT_TOKENIZER_CONFIGS,
    "dynamics": DEFAULT_DYNAMICS_CONFIGS,
}


@dataclass
class Job:
    config_path: Path
    cfg: dict[str, Any]
    candidates: list[dict[str, Any]]
    state: dict[str, Any]
    status: str = "WAITING"
    detail: str = ""
    queued_state: str = ""
    resource_name: str = ""
    zone: str = ""
    active: bool = False
    terminal: bool = False
    retry_after: float = 0.0
    last_error: str = ""
    last_failure_uri: str = ""
    updated_at: float = field(default_factory=time.time)

    @property
    def name(self) -> str:
        return str(self.cfg["job"]["name"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Queue TPU training jobs while keeping only a fixed number alive."
    )
    parser.add_argument(
        "--job-kind",
        choices=sorted(DEFAULT_CONFIGS_BY_KIND),
        default="tokenizer",
        help="Default job config set to queue when --config is omitted.",
    )
    parser.add_argument(
        "--config",
        dest="configs",
        action="append",
        type=Path,
        help=(
            "Watcher config. Repeat to override the default config list for --job-kind."
        ),
    )
    parser.add_argument(
        "--starter-script",
        type=Path,
        default=SCRIPT_DIR / "starter.sh",
        help="Startup script injected into each TPU VM.",
    )
    parser.add_argument(
        "--max-active",
        type=int,
        default=2,
        help="Maximum live queued resources at once.",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=int,
        default=60,
        help="Seconds between dashboard refreshes.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Print one dashboard snapshot and exit without creating jobs.",
    )
    parser.add_argument(
        "--use-active-account",
        action="store_true",
        help=(
            "Use the active gcloud account instead of watcher.impersonate_service_account. "
            "Useful when TPU queue admission is granted to the user principal, not the watcher service account."
        ),
    )
    return parser.parse_args()


def use_active_account(cfg: dict[str, Any]) -> None:
    watcher_cfg = cfg.get("watcher")
    if isinstance(watcher_cfg, dict):
        watcher_cfg.pop("impersonate_service_account", None)


def load_jobs(config_paths: list[Path], *, active_account: bool = False) -> list[Job]:
    jobs = []
    for config_path in config_paths:
        resolved_path = config_path.resolve()
        cfg = watcher.load_config(resolved_path)
        if active_account:
            use_active_account(cfg)
        candidates = list(cfg["candidates"])
        if len(candidates) != 1:
            raise ValueError(
                f"{resolved_path} must define exactly one candidate for this queue runner."
            )
        jobs.append(
            Job(
                config_path=resolved_path,
                cfg=cfg,
                candidates=candidates,
                state=watcher.load_watcher_state(cfg),
            )
        )
    return jobs


def failure_budget_exhausted(job: Job) -> bool:
    policy = watcher.resolve_failure_policy(job.cfg)
    if policy["mode"] == "stop":
        return True
    max_retries = int(policy["max_retries"])
    if max_retries < 0:
        return False
    return int(job.state.get("failure_retries", 0)) > max_retries


def set_status(
    job: Job,
    status: str,
    *,
    detail: str = "",
    queued_state: str = "",
    active: bool = False,
    terminal: bool = False,
) -> None:
    job.status = status
    job.detail = detail
    job.queued_state = queued_state
    job.active = active
    job.terminal = terminal
    job.updated_at = time.time()


def summarize_called_process(exc: subprocess.CalledProcessError) -> str:
    detail = watcher.command_failure_detail(exc)
    if detail:
        json_start = detail.find("{")
        if json_start >= 0:
            try:
                data = json.loads(detail[json_start:])
            except json.JSONDecodeError:
                data = None
            if isinstance(data, dict):
                message = find_message(data)
                if message:
                    return message

        lines = [line.strip() for line in detail.splitlines() if line.strip()]
        for line in reversed(lines):
            token = line.rstrip(",")
            if token in {"{", "}", "[", "]"}:
                continue
            if token.startswith('"message":'):
                raw_value = token.split(":", 1)[1].strip().rstrip(",")
                try:
                    parsed_value = json.loads(raw_value)
                except json.JSONDecodeError:
                    parsed_value = raw_value.strip('"')
                if parsed_value:
                    return str(parsed_value)
            return token
    return f"exit code {exc.returncode}: {' '.join(str(part) for part in exc.cmd)}"


def find_message(value: Any) -> str:
    if isinstance(value, dict):
        message = value.get("message")
        if isinstance(message, str) and message:
            return message
        for nested_value in value.values():
            nested_message = find_message(nested_value)
            if nested_message:
                return nested_message
    if isinstance(value, list):
        for item in value:
            nested_message = find_message(item)
            if nested_message:
                return nested_message
    return ""


def queue_state_status(queued_state: str) -> tuple[str, bool]:
    if queued_state == "ACTIVE":
        return "ALIVE", True
    if queued_state in watcher.PENDING_STATES:
        return "PENDING", True
    if queued_state in watcher.DELETE_IN_PROGRESS_STATES:
        return "DELETING", True
    if queued_state in watcher.TERMINAL_RETRY_STATES:
        return "REQUEUE", True
    return "UNKNOWN", False


def latest_failure_uri(cfg: dict[str, Any], failure_prefix: str) -> str:
    if not failure_prefix:
        return ""
    failures = watcher.gcs_list_objects(cfg, failure_prefix)
    return failures[-1] if failures else ""


def clear_current_attempt(job: Job) -> None:
    job.state["current_attempt_id"] = None
    job.state["current_candidate_index"] = None
    watcher.save_watcher_state(job.cfg, job.state)


def request_delete_existing(
    job: Job, existing: tuple[int, dict[str, Any], dict[str, Any]]
) -> str:
    index, candidate, desc = existing
    queued_state = watcher.queued_resource_state(desc)
    if queued_state not in watcher.DELETE_IN_PROGRESS_STATES:
        watcher.delete_queued_resource(
            job.cfg,
            queued_resource_name=watcher.queued_resource_name(job.name, index),
            zone=candidate["zone"],
        )
    return queued_state


def mark_complete(job: Job, existing: tuple[int, dict[str, Any], dict[str, Any]] | None) -> None:
    detail = "completion marker found"
    active = False
    if existing is not None:
        queued_state = request_delete_existing(job, existing)
        detail = f"{detail}; deleting resource state={queued_state}"
        active = True
    job.state["current_attempt_id"] = None
    job.state["current_candidate_index"] = None
    job.state["failure_retries"] = 0
    watcher.save_watcher_state(job.cfg, job.state)
    set_status(job, "DONE", detail=detail, active=active, terminal=True)


def refresh_terminal_cleanup(job: Job) -> None:
    existing = watcher.discover_existing_resource(job.cfg, job.candidates)
    base_detail = job.detail.split("; cleanup state=", 1)[0]
    if existing is None:
        set_status(job, job.status, detail=base_detail, active=False, terminal=True)
        return

    queued_state = request_delete_existing(job, existing)
    set_status(
        job,
        job.status,
        detail=f"{base_detail}; cleanup state={queued_state}",
        queued_state=queued_state,
        active=True,
        terminal=True,
    )


def consume_failure_event(
    job: Job,
    existing: tuple[int, dict[str, Any], dict[str, Any]] | None,
    failure_uri: str,
) -> None:
    policy = watcher.resolve_failure_policy(job.cfg)
    payload = watcher.load_marker_payload(job.cfg, failure_uri)
    active = False

    if existing is not None:
        request_delete_existing(job, existing)
        active = True

    job.last_failure_uri = failure_uri
    job.state["last_processed_failure_uri"] = failure_uri
    job.state["current_attempt_id"] = None
    job.state["current_candidate_index"] = None

    retries = int(job.state.get("failure_retries", 0))
    can_retry = policy["mode"] == "retry" and (
        int(policy["max_retries"]) < 0 or retries < int(policy["max_retries"])
    )
    if can_retry:
        retries += 1
        job.state["failure_retries"] = retries
        watcher.save_watcher_state(job.cfg, job.state)
        job.retry_after = time.time() + int(policy["retry_backoff_seconds"])
        label = (
            f"retry {retries}"
            if int(policy["max_retries"]) < 0
            else f"retry {retries}/{policy['max_retries']}"
        )
        set_status(
            job,
            "RETRY_WAIT",
            detail=f"{label} after failure marker {Path(failure_uri).name}",
            active=active,
        )
        return

    watcher.save_watcher_state(job.cfg, job.state)
    exit_code = ""
    if isinstance(payload, dict) and payload.get("exit_code") not in (None, ""):
        exit_code = f" exit={payload['exit_code']}"
    set_status(
        job,
        "CRASH",
        detail=f"failure marker {Path(failure_uri).name}{exit_code}",
        terminal=True,
    )


def refresh_existing_resource(
    job: Job,
    existing: tuple[int, dict[str, Any], dict[str, Any]],
    starter_script_contents: str,
) -> None:
    index, candidate, desc = existing
    qr_name = watcher.queued_resource_name(job.name, index)
    queued_state = watcher.queued_resource_state(desc)
    job.resource_name = qr_name
    job.zone = candidate["zone"]

    attempt_id = job.state.get("current_attempt_id")
    if attempt_id is None:
        attempt_id = watcher.queued_resource_attempt_id(desc)
    expected_payload_json = json.dumps(
        watcher.build_job_payload(
            job.cfg,
            candidate,
            queued_resource_name_value=qr_name,
            candidate_index=index,
            attempt_id=attempt_id,
        ),
        indent=2,
    )
    mismatch_reason = watcher.queued_resource_metadata_mismatch_reason(
        desc,
        expected_startup_script=starter_script_contents,
        expected_payload_json=expected_payload_json,
    )
    if mismatch_reason is not None:
        if queued_state not in watcher.DELETE_IN_PROGRESS_STATES:
            watcher.delete_queued_resource(job.cfg, qr_name, candidate["zone"])
            clear_current_attempt(job)
        set_status(
            job,
            "DELETING",
            detail=f"metadata mismatch: {mismatch_reason}",
            queued_state=queued_state,
            active=True,
        )
        return

    if queued_state in watcher.TERMINAL_RETRY_STATES:
        details = watcher.queued_resource_state_details(desc)
        watcher.delete_queued_resource(job.cfg, qr_name, candidate["zone"])
        clear_current_attempt(job)
        set_status(
            job,
            "REQUEUE",
            detail=details or f"terminal queued state {queued_state}",
            queued_state=queued_state,
            active=True,
        )
        return

    if queued_state not in watcher.LIVE_STATES:
        set_status(
            job,
            "CRASH",
            detail=f"unhandled queued state {queued_state}",
            queued_state=queued_state,
            terminal=True,
        )
        return

    status, active = queue_state_status(queued_state)
    set_status(job, status, queued_state=queued_state, active=active)


def refresh_job(job: Job, starter_script_contents: str) -> None:
    job.active = False
    job.terminal = False
    job.resource_name = ""
    job.zone = job.candidates[0]["zone"]
    complete_uri, failure_prefix = watcher.resolve_marker_paths(job.cfg)
    existing = watcher.discover_existing_resource(job.cfg, job.candidates)

    if watcher.gcs_object_exists(job.cfg, complete_uri):
        mark_complete(job, existing)
        return

    failure_uri = watcher.next_failure_event_uri(
        job.cfg,
        failure_prefix,
        str(job.state.get("last_processed_failure_uri", "")),
    )
    if failure_uri:
        consume_failure_event(job, existing, failure_uri)
        return

    if existing is not None:
        refresh_existing_resource(job, existing, starter_script_contents)
        return

    latest_failure = latest_failure_uri(job.cfg, failure_prefix)
    if latest_failure and latest_failure <= str(job.state.get("last_processed_failure_uri", "")):
        if failure_budget_exhausted(job):
            set_status(
                job,
                "CRASH",
                detail=f"retry budget exhausted after {Path(latest_failure).name}",
                terminal=True,
            )
            return

    if job.retry_after > time.time():
        remaining = int(job.retry_after - time.time())
        set_status(job, "RETRY_WAIT", detail=f"retrying in {remaining}s")
        return

    set_status(job, "WAITING")


def start_job(
    job: Job,
    starter_script: Path,
    before_create: Callable[[], None] | None = None,
) -> None:
    candidate_index = 0
    candidate = job.candidates[candidate_index]
    attempt_id = int(job.state["next_attempt_id"])
    job.state["current_attempt_id"] = attempt_id
    job.state["current_candidate_index"] = candidate_index
    job.state["next_attempt_id"] = attempt_id + 1
    watcher.save_watcher_state(job.cfg, job.state)

    job.resource_name = watcher.queued_resource_name(job.name, candidate_index)
    job.zone = candidate["zone"]
    set_status(
        job,
        "CREATING",
        detail=f"creating attempt {attempt_id}",
        queued_state="CREATING",
        active=True,
    )
    if before_create is not None:
        before_create()

    watcher.create_queued_resource(
        job.cfg,
        candidate,
        candidate_index=candidate_index,
        attempt_id=attempt_id,
        starter_script=starter_script,
    )
    set_status(
        job,
        "PENDING",
        detail=f"created attempt {attempt_id}",
        queued_state="CREATING",
        active=True,
    )


def trim(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    if width <= 1:
        return value[:width]
    return value[: width - 1] + "~"


def render_dashboard(
    jobs: list[Job],
    max_active: int,
    message: str = "",
    *,
    title: str = "Training queue",
) -> None:
    reserved_count = sum(1 for job in jobs if job.active)
    live_count = sum(
        1
        for job in jobs
        if job.status in {"ALIVE", "CREATING", "PENDING", "DELETING", "REQUEUE"}
    )
    complete_count = sum(1 for job in jobs if job.status == "DONE")
    crashed_count = sum(1 for job in jobs if job.status == "CRASH")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if sys.stdout.isatty():
        print("\033[2J\033[H", end="")
    print(f"{title} dashboard  {now}")
    print(
        f"reserved={reserved_count}/{max_active}  live={live_count}  "
        f"done={complete_count}/{len(jobs)}  crash={crashed_count}"
    )
    if message:
        print(f"last event: {message}")
    print()
    print(
        f"{'job':28} {'status':11} {'qr_state':22} {'zone':20} "
        f"{'attempts':9} detail"
    )
    print("-" * 116)
    for job in jobs:
        attempts = f"{job.state.get('failure_retries', 0)}/{watcher.resolve_failure_policy(job.cfg)['max_retries']}"
        detail = job.detail
        if job.resource_name:
            detail = f"{job.resource_name} {detail}".strip()
        print(
            f"{trim(job.name, 28):28} "
            f"{trim(job.status, 11):11} "
            f"{trim(job.queued_state, 22):22} "
            f"{trim(job.zone, 20):20} "
            f"{trim(attempts, 9):9} "
            f"{trim(detail, 80)}"
        )
    print(flush=True)


def terminal_exit_code(jobs: list[Job]) -> int:
    return 0 if all(job.status == "DONE" for job in jobs) else 1


def main() -> int:
    args = parse_args()
    if args.max_active < 1:
        raise ValueError("--max-active must be >= 1")
    poll_interval = max(int(args.poll_interval_seconds), 1)
    starter_script = args.starter_script.resolve()
    starter_script_contents = starter_script.read_text()
    job_kind = str(args.job_kind)
    config_paths = args.configs or DEFAULT_CONFIGS_BY_KIND[job_kind]
    dashboard_title = f"{job_kind.title()} queue"
    jobs = load_jobs(config_paths, active_account=args.use_active_account)
    message = "initializing"
    if args.use_active_account:
        message = "initializing with active gcloud account"
    for job in jobs:
        job.zone = job.candidates[0]["zone"]
        set_status(job, "CHECKING", detail="waiting for first cloud status check")
    render_dashboard(jobs, args.max_active, message=message, title=dashboard_title)

    while True:
        for job in jobs:
            try:
                if job.terminal and not job.active:
                    continue
                if job.terminal:
                    refresh_terminal_cleanup(job)
                else:
                    refresh_job(job, starter_script_contents)
            except subprocess.CalledProcessError as exc:
                summary = summarize_called_process(exc)
                if job.terminal:
                    set_status(
                        job,
                        job.status,
                        detail=f"{job.detail}; cleanup check unknown: {summary}",
                        queued_state="UNKNOWN",
                        active=True,
                        terminal=True,
                    )
                else:
                    set_status(job, "UNKNOWN", detail=summary, active=True)
                message = f"{job.name}: {summary}"

        active_count = sum(1 for job in jobs if job.active)
        if not args.once:
            for job in jobs:
                if active_count >= args.max_active:
                    break
                if job.status != "WAITING" or job.terminal or job.active:
                    continue
                try:
                    start_job(
                        job,
                        starter_script,
                        before_create=lambda: render_dashboard(
                            jobs,
                            args.max_active,
                            message=f"{job.name}: creating queued resource",
                            title=dashboard_title,
                        ),
                    )
                except subprocess.CalledProcessError as exc:
                    job.last_error = summarize_called_process(exc)
                    clear_current_attempt(job)
                    set_status(job, "CREATE_ERR", detail=job.last_error, active=True)
                    message = f"{job.name}: create failed: {job.last_error}"
                    active_count += 1
                    break
                active_count += 1
                message = f"{job.name}: queued in {job.zone}"

        render_dashboard(jobs, args.max_active, message=message, title=dashboard_title)

        if args.once:
            return (
                terminal_exit_code(jobs)
                if all(job.terminal for job in jobs) and not any(job.active for job in jobs)
                else 0
            )
        if all(job.terminal for job in jobs) and not any(job.active for job in jobs):
            return terminal_exit_code(jobs)
        time.sleep(poll_interval)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[tokenizer-queue] Interrupted.", file=sys.stderr)
        raise SystemExit(130)
