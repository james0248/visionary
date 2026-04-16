#!/usr/bin/env bash

set -euo pipefail

LOG_PATH="/var/log/visionary-starter.log"
mkdir -p "$(dirname "$LOG_PATH")"
exec > >(tee -a "$LOG_PATH") 2>&1

info()  { printf '[starter] %s\n' "$1"; }
error() { printf '[starter][error] %s\n' "$1" >&2; }

MODE="all"
ACCELERATOR_OVERRIDE=""

usage() {
    cat <<'EOF'
Usage:
  starter.sh [--mode all|setup-only] [--accelerator cpu|gpu|tpu]

Modes:
  all         Run full cloud bootstrap from instance metadata, then launch training.
  setup-only  Only install the local environment.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --accelerator)
            ACCELERATOR_OVERRIDE="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            error "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

case "$MODE" in
    all|setup-only) ;;
    *)
        error "Unsupported mode: $MODE"
        usage
        exit 1
        ;;
esac

resolve_home_dir() {
    if [[ -n "${HOME:-}" ]]; then
        printf '%s\n' "$HOME"
        return
    fi

    local current_user
    current_user="$(id -un)"
    if command -v getent >/dev/null 2>&1; then
        local candidate_home
        candidate_home="$(getent passwd "$current_user" | cut -d: -f6)"
        if [[ -n "$candidate_home" ]]; then
            printf '%s\n' "$candidate_home"
            return
        fi
    fi

    if command -v python3 >/dev/null 2>&1; then
        python3 - <<'PY'
import pathlib
print(pathlib.Path.home())
PY
        return
    fi

    printf '/root\n'
}

HOME="${HOME:-$(resolve_home_dir)}"
export HOME

run_setup() {
    local accelerator="$1"
    local jax_spec="jax"
    local jax_label="CPU"

    case "$accelerator" in
        cpu) ;;
        gpu)
            jax_spec="jax[cuda12]"
            jax_label="GPU (CUDA 12)"
            ;;
        tpu)
            jax_spec="jax[tpu]"
            jax_label="TPU"
            ;;
        *)
            error "Unsupported accelerator: $accelerator"
            exit 1
            ;;
    esac

    if command -v uv >/dev/null 2>&1; then
        info "uv is already installed ($(uv --version))"
    else
        info "Installing uv"
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi

    info "Running uv sync"
    uv sync

    info "Installing visionary package"
    uv pip install --python .venv/bin/python .

    info "Installing ${jax_label} JAX package (${jax_spec})"
    uv pip install --python .venv/bin/python --upgrade "$jax_spec"
}

if [[ "$MODE" == "setup-only" ]]; then
    if [[ -z "$ACCELERATOR_OVERRIDE" ]]; then
        error "--accelerator is required with --mode setup-only"
        exit 1
    fi
    run_setup "$ACCELERATOR_OVERRIDE"
    exit 0
fi

metadata_get() {
    curl -fsS -H "Metadata-Flavor: Google" \
        "http://metadata.google.internal/computeMetadata/v1/$1"
}

JOB_JSON="$(metadata_get "instance/attributes/visionary-job-json")"

json_get() {
    python3 - "$JOB_JSON" "$1" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
path = sys.argv[2]
value = payload
for part in path.split("."):
    if not part:
        continue
    if isinstance(value, dict) and part in value:
        value = value[part]
    else:
        print("", end="")
        raise SystemExit(0)
if value is None:
    print("", end="")
elif isinstance(value, bool):
    print("true" if value else "false")
elif isinstance(value, (dict, list)):
    print(json.dumps(value))
else:
    print(value)
PY
}

json_list() {
    python3 - "$JOB_JSON" "$1" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
path = sys.argv[2]
value = payload
for part in path.split("."):
    if not part:
        continue
    if isinstance(value, dict) and part in value:
        value = value[part]
    else:
        raise SystemExit(0)
if isinstance(value, list):
    for item in value:
        print(item)
PY
}

boot_disk_device() {
    local root_source=""
    root_source="$(findmnt -n -o SOURCE / 2>/dev/null || true)"
    if [[ -z "$root_source" || ! -b "$root_source" ]]; then
        return 0
    fi

    local parent_name=""
    parent_name="$(lsblk -no PKNAME "$root_source" 2>/dev/null | head -n1)"
    if [[ -n "$parent_name" ]]; then
        printf '/dev/%s\n' "$parent_name"
        return 0
    fi

    printf '%s\n' "$root_source"
}

log_disk_inventory() {
    info "Disk inventory from /dev/disk/by-id:"
    local listed_any="false"
    local entry=""
    shopt -s nullglob
    for entry in /dev/disk/by-id/google-* /dev/disk/by-id/scsi-0Google_*; do
        listed_any="true"
        ls -l "$entry"
    done
    shopt -u nullglob
    if [[ "$listed_any" == "false" ]]; then
        info "No Google disk symlinks found in /dev/disk/by-id."
    fi

    info "Disk inventory from lsblk:"
    lsblk -dnpo NAME,TYPE,SIZE,MODEL,SERIAL,MOUNTPOINT 2>/dev/null || true
}

resolve_data_disk_device() {
    local disk_name="$1"
    local candidate=""

    if [[ -n "$disk_name" ]]; then
        local direct_candidates=(
            "/dev/disk/by-id/google-${disk_name}"
            "/dev/disk/by-id/scsi-0Google_PersistentDisk_${disk_name}"
            "/dev/disk/by-id/scsi-0Google_Hyperdisk_${disk_name}"
        )
        for candidate in "${direct_candidates[@]}"; do
            if [[ -e "$candidate" ]]; then
                printf '%s\n' "$candidate"
                return 0
            fi
        done

        local prefixed_matches=()
        shopt -s nullglob
        prefixed_matches=(/dev/disk/by-id/google-"${disk_name}"*)
        shopt -u nullglob
        if [[ "${#prefixed_matches[@]}" -gt 0 ]]; then
            for candidate in "${prefixed_matches[@]}"; do
                if [[ "$candidate" != *-part* ]]; then
                    printf '%s\n' "$candidate"
                    return 0
                fi
            done
            printf '%s\n' "${prefixed_matches[0]}"
            return 0
        fi
    fi

    local boot_disk=""
    boot_disk="$(boot_disk_device)"
    local fallback_disks=()
    while read -r candidate; do
        [[ -z "$candidate" ]] && continue
        [[ "$candidate" == "$boot_disk" ]] && continue
        fallback_disks+=("$candidate")
    done < <(lsblk -dnpo NAME,TYPE 2>/dev/null | awk '$2 == "disk" {print $1}')

    if [[ "${#fallback_disks[@]}" -eq 1 ]]; then
        printf '%s\n' "${fallback_disks[0]}"
        return 0
    fi

    return 1
}

PROJECT="$(json_get project)"
JOB_NAME="$(json_get job_name)"
QUEUED_RESOURCE_NAME="$(json_get queued_resource.name)"
CANDIDATE_INDEX="$(json_get queued_resource.candidate_index)"
ATTEMPT_ID="$(json_get queued_resource.attempt_id)"
NODE_NAME="$(metadata_get "instance/name")"
COMPLETE_MARKER_URI="$(json_get markers.complete_uri)"
FAILURE_MARKER_PREFIX="$(json_get markers.failure_prefix)"
LOG_URI="$(json_get logs.uri)"
LOG_ARCHIVE_URI_PREFIX="$(json_get logs.archive_uri_prefix)"
LOG_TAIL_LINES="$(json_get logs.tail_lines)"
if [[ -z "$LOG_TAIL_LINES" ]]; then
    LOG_TAIL_LINES=120
fi
if [[ -z "$ATTEMPT_ID" ]]; then
    ATTEMPT_ID=0
fi

upload_log() {
    local uri="$1"
    if [[ -z "$uri" || ! -f "$LOG_PATH" ]]; then
        return 0
    fi
    gcloud storage cp "$LOG_PATH" "$uri" >/dev/null
}

build_archive_uri() {
    local prefix="$1"
    local suffix="$2"
    if [[ -z "$prefix" ]]; then
        return 0
    fi
    printf '%s/%s\n' "${prefix%/}" "$suffix"
}

write_marker() {
    local uri="$1"
    local status="$2"
    local exit_code="$3"
    local log_uri="$4"
    local archived_log_uri="$5"
    local log_upload_error="$6"
    local tmp
    if [[ -z "$uri" ]]; then
        return
    fi
    tmp="$(mktemp)"
    python3 - "$tmp" "$status" "$exit_code" "$NODE_NAME" "$QUEUED_RESOURCE_NAME" "$JOB_NAME" "$CANDIDATE_INDEX" "$ATTEMPT_ID" "$LOG_PATH" "$log_uri" "$archived_log_uri" "$LOG_TAIL_LINES" "$log_upload_error" <<'PY'
import collections
import json
import pathlib
import sys
import time

path = pathlib.Path(sys.argv[1])
status = sys.argv[2]
candidate_index = sys.argv[7]
attempt_id = sys.argv[8]
log_path = pathlib.Path(sys.argv[9])
log_uri = sys.argv[10]
archived_log_uri = sys.argv[11]
tail_lines = int(sys.argv[12])
log_upload_error = sys.argv[13]
payload = {
    "status": status,
    "exit_code": int(sys.argv[3]),
    "node_name": sys.argv[4],
    "queued_resource_name": sys.argv[5],
    "job_name": sys.argv[6],
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}
if candidate_index:
    payload["candidate_index"] = int(candidate_index)
if attempt_id:
    payload["attempt_id"] = int(attempt_id)
if log_uri:
    payload["log_uri"] = log_uri
if archived_log_uri:
    payload["archived_log_uri"] = archived_log_uri
if log_upload_error:
    payload["log_upload_error"] = log_upload_error
if status != "completed" and tail_lines > 0 and log_path.exists():
    recent_lines = collections.deque(maxlen=tail_lines)
    with log_path.open(errors="replace") as handle:
        for line in handle:
            recent_lines.append(line.rstrip("\n"))
    payload["log_tail"] = "\n".join(recent_lines)
path.write_text(json.dumps(payload, indent=2))
PY
    gcloud storage cp "$tmp" "$uri" >/dev/null
    rm -f "$tmp"
}

on_exit() {
    local exit_code="$1"
    local uploaded_log_uri=""
    local archived_log_uri=""
    local failure_marker_uri=""
    local log_upload_error=""
    local attempt_timestamp=""
    if [[ "$exit_code" -ne 0 ]]; then
        attempt_timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
    fi
    if [[ -n "$LOG_URI" ]]; then
        if upload_log "$LOG_URI"; then
            uploaded_log_uri="$LOG_URI"
            info "Uploaded starter log to $LOG_URI."
        else
            log_upload_error="Failed to upload starter log to $LOG_URI."
            error "$log_upload_error"
        fi
    fi
    if [[ "$exit_code" -ne 0 && -n "$LOG_ARCHIVE_URI_PREFIX" ]]; then
        archived_log_uri="$(build_archive_uri "$LOG_ARCHIVE_URI_PREFIX" "$(printf 'attempt-%06d_%s_%s.log' "$ATTEMPT_ID" "$attempt_timestamp" "$NODE_NAME")")"
        if upload_log "$archived_log_uri"; then
            info "Archived starter log to $archived_log_uri."
        else
            if [[ -n "$log_upload_error" ]]; then
                log_upload_error="$log_upload_error "
            fi
            log_upload_error="${log_upload_error}Failed to upload archived starter log to $archived_log_uri."
            error "Failed to upload archived starter log to $archived_log_uri."
            archived_log_uri=""
        fi
    fi
    if [[ "$exit_code" -eq 0 ]]; then
        write_marker "$COMPLETE_MARKER_URI" "completed" "$exit_code" "$uploaded_log_uri" "$archived_log_uri" "$log_upload_error"
        info "Training completed successfully."
    else
        failure_marker_uri="$(build_archive_uri "$FAILURE_MARKER_PREFIX" "$(printf 'attempt-%06d_%s_%s_exit%d.json' "$ATTEMPT_ID" "$attempt_timestamp" "$NODE_NAME" "$exit_code")")"
        write_marker "$failure_marker_uri" "failed" "$exit_code" "$uploaded_log_uri" "$archived_log_uri" "$log_upload_error"
        error "Training or startup failed with exit code $exit_code."
    fi
}

trap 'on_exit $?' EXIT

info "Node ${NODE_NAME} bootstrapping job ${JOB_NAME}."

WAND_SECRET="$(json_get secrets.wandb_secret)"
if [[ -n "$WAND_SECRET" ]]; then
    export WANDB_API_KEY
    WANDB_API_KEY="$(gcloud secrets versions access latest --secret="$WAND_SECRET" --project="$PROJECT")"
    info "Loaded W&B API key from Secret Manager."
fi

REPO_URL="$(json_get repo.url)"
REPO_BRANCH="$(json_get repo.branch)"
REPO_BASENAME="$(basename "${REPO_URL%.git}")"
REPO_DIR="$HOME/$REPO_BASENAME"
mkdir -p "$(dirname "$REPO_DIR")"

if [[ ! -d "$REPO_DIR/.git" ]]; then
    info "Cloning repository into $REPO_DIR"
    git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"
git remote set-url origin "$REPO_URL"
git fetch origin --prune
git checkout "$REPO_BRANCH"
git pull --ff-only origin "$REPO_BRANCH"
info "Repository ready at $(git rev-parse HEAD)"

DATA_DISK_NAME="$(json_get data_disk.name)"
DATA_MOUNT_PATH="$(json_get data_disk.mount_path)"
DATA_DISK_MODE="$(json_get data_disk.mode)"
if [[ -n "$DATA_DISK_NAME" ]]; then
    DEVICE_PATH="$(resolve_data_disk_device "$DATA_DISK_NAME" || true)"
    if [[ ! -e "$DEVICE_PATH" ]]; then
        error "Attached data disk ${DATA_DISK_NAME} could not be resolved to a block device."
        log_disk_inventory
        exit 1
    fi
    sudo mkdir -p "$DATA_MOUNT_PATH"
    if ! mountpoint -q "$DATA_MOUNT_PATH"; then
        if [[ "$DATA_DISK_MODE" == "read-only" ]]; then
            sudo mount -o ro,defaults "$DEVICE_PATH" "$DATA_MOUNT_PATH"
        else
            sudo mount -o discard,defaults "$DEVICE_PATH" "$DATA_MOUNT_PATH"
        fi
    fi
    info "Mounted ${DATA_DISK_NAME} from ${DEVICE_PATH} at ${DATA_MOUNT_PATH}"
fi

ACCELERATOR="${ACCELERATOR_OVERRIDE:-$(json_get setup.accelerator)}"
run_setup "$ACCELERATOR"

TRAIN_SCRIPT="$(json_get training.script)"
CONFIG_NAME="$(json_get training.config_name)"
AUTO_RESUME="$(json_get training.auto_resume)"
mapfile -t TRAIN_OVERRIDES < <(json_list "training.overrides")

TRAIN_CMD=(uv run python "$TRAIN_SCRIPT")
if [[ -n "$CONFIG_NAME" ]]; then
    TRAIN_CMD+=(--config-name "$CONFIG_NAME")
fi
for override in "${TRAIN_OVERRIDES[@]}"; do
    TRAIN_CMD+=("$override")
done

if [[ "$AUTO_RESUME" == "true" ]]; then
    has_resume_override="false"
    for override in "${TRAIN_OVERRIDES[@]}"; do
        if [[ "$override" == checkpoint.resume_step=* ]]; then
            has_resume_override="true"
            break
        fi
    done
    if [[ "$has_resume_override" == "false" ]]; then
        TRAIN_CMD+=("checkpoint.resume_step=latest")
    fi
fi

info "Starting training: ${TRAIN_CMD[*]}"
"${TRAIN_CMD[@]}"
