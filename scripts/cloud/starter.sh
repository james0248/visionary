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

PROJECT="$(json_get project)"
JOB_NAME="$(json_get job_name)"
QUEUED_RESOURCE_NAME="$(json_get queued_resource.name)"
NODE_NAME="$(metadata_get "instance/name")"
COMPLETE_MARKER_URI="$(json_get markers.complete_uri)"
FAILURE_MARKER_URI="$(json_get markers.failure_uri)"

write_marker() {
    local uri="$1"
    local status="$2"
    local exit_code="$3"
    local tmp
    if [[ -z "$uri" ]]; then
        return
    fi
    tmp="$(mktemp)"
    python3 - "$tmp" "$status" "$exit_code" "$NODE_NAME" "$QUEUED_RESOURCE_NAME" "$JOB_NAME" <<'PY'
import json
import pathlib
import sys
import time

path = pathlib.Path(sys.argv[1])
payload = {
    "status": sys.argv[2],
    "exit_code": int(sys.argv[3]),
    "node_name": sys.argv[4],
    "queued_resource_name": sys.argv[5],
    "job_name": sys.argv[6],
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}
path.write_text(json.dumps(payload, indent=2))
PY
    gcloud storage cp "$tmp" "$uri" >/dev/null
    rm -f "$tmp"
}

on_exit() {
    local exit_code="$1"
    if [[ "$exit_code" -eq 0 ]]; then
        write_marker "$COMPLETE_MARKER_URI" "completed" "$exit_code"
        info "Training completed successfully."
    else
        write_marker "$FAILURE_MARKER_URI" "failed" "$exit_code"
        error "Training or startup failed with exit code $exit_code."
    fi
}

trap 'on_exit $?' EXIT

info "Node ${NODE_NAME} bootstrapping job ${JOB_NAME}."
gcloud config set project "$PROJECT" >/dev/null

WAND_SECRET="$(json_get secrets.wandb_secret)"
if [[ -n "$WAND_SECRET" ]]; then
    export WANDB_API_KEY
    WANDB_API_KEY="$(gcloud secrets versions access latest --secret="$WAND_SECRET")"
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
    DEVICE_PATH="/dev/disk/by-id/google-${DATA_DISK_NAME}"
    if [[ ! -e "$DEVICE_PATH" ]]; then
        error "Attached data disk not found at $DEVICE_PATH"
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
    info "Mounted ${DATA_DISK_NAME} at ${DATA_MOUNT_PATH}"
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
