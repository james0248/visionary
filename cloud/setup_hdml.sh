#!/usr/bin/env bash

set -euo pipefail

PROJECT_ID="visionary-491008"
STAGING_SERVICE_ACCOUNT="visionary-starter@visionary-491008.iam.gserviceaccount.com"
MACHINE_TYPE="c4-standard-4"
BOOT_DISK_SIZE_GB="50"

DISK_NAME=""
ZONE=""
SIZE_GB=""
THROUGHPUT=""
GCS_PATHS=()

usage() {
    cat <<'EOF'
Usage:
  cloud/setup_hdml.sh \
    --disk-name DISK_NAME \
    --zone ZONE \
    --size-gb SIZE_GB \
    --throughput THROUGHPUT_MIBPS \
    --gcs-path GCS_PATH [--gcs-path GCS_PATH ...]

Example:
  cloud/setup_hdml.sh \
    --disk-name visionary-data-tokenizer-ue1d \
    --zone us-east1-d \
    --size-gb 200 \
    --throughput 2000 \
    --gcs-path 'gs://visionary-uc1/atari/data/*_tokenizer'

Example with exact sources:
  cloud/setup_hdml.sh \
    --disk-name visionary-data-dynamics-ue1d \
    --zone us-east1-d \
    --size-gb 200 \
    --throughput 2000 \
    --gcs-path gs://visionary-uc1/atari/data/qbert_dynamics/ \
    --gcs-path gs://visionary-uc1/atari/data/seaquest_dynamics/

The script:
  1. Deletes an existing detached disk with the same name, if present.
  2. Creates a writeable Hyperdisk ML disk.
  3. Creates a temporary C4 VM in ZONE.
  4. Runs three separate SSH commands: format/mount, copy, sync/unmount.
  5. Detaches the disk, deletes the VM, and switches the disk to READ_ONLY_MANY.
EOF
}

info() {
    printf '[setup-hdml] %s\n' "$1"
}

die() {
    printf '[setup-hdml][error] %s\n' "$1" >&2
    exit 1
}

require_value() {
    local flag="$1"
    local value="${2:-}"
    if [[ -z "$value" || "$value" == --* ]]; then
        die "$flag requires a value."
    fi
}

normalize_size_gb() {
    local value="$1"
    case "$value" in
        *GB) value="${value%GB}" ;;
        *gb) value="${value%gb}" ;;
        *G) value="${value%G}" ;;
        *g) value="${value%g}" ;;
    esac
    printf '%s\n' "$value"
}

sanitize_name() {
    printf '%s' "$1" \
        | tr '[:upper:]' '[:lower:]' \
        | sed -E 's/[^a-z0-9-]+/-/g; s/^-+//; s/-+$//; s/-+/-/g'
}

shell_quote() {
    printf "'"
    printf '%s' "$1" | sed "s/'/'\\\\''/g"
    printf "'"
}

log_gcloud() {
    printf '+ gcloud'
    local arg
    for arg in "$@"; do
        printf ' %q' "$arg"
    done
    printf '\n'
}

run_gcloud() {
    local args=("$@" "--project=$PROJECT_ID")
    log_gcloud "${args[@]}"
    gcloud "${args[@]}"
}

gcloud_quiet() {
    gcloud "$@" "--project=$PROJECT_ID"
}

ssh_command() {
    local command="$1"
    run_gcloud compute ssh "$VM_NAME" \
        --zone="$ZONE" \
        --command="$command"
}

disk_exists() {
    gcloud_quiet compute disks describe "$DISK_NAME" \
        --zone="$ZONE" \
        --format='value(name)' >/dev/null 2>&1
}

disk_users() {
    gcloud_quiet compute disks describe "$DISK_NAME" \
        --zone="$ZONE" \
        --format='value(users)' 2>/dev/null || true
}

wait_for_ssh() {
    local attempt
    for attempt in $(seq 1 30); do
        if gcloud_quiet compute ssh "$VM_NAME" \
            --zone="$ZONE" \
            --command='true' >/dev/null 2>&1; then
            return
        fi
        info "Waiting for SSH on $VM_NAME (${attempt}/30)."
        sleep 10
    done
    die "SSH did not become ready on $VM_NAME."
}

format_and_mount_command() {
    printf 'set -eu
DEVICE=/dev/disk/by-id/google-%s
MOUNT_PATH=/mnt/data
attempt=0
while [ "$attempt" -lt 60 ]; do
  if [ -e "$DEVICE" ]; then
    break
  fi
  sleep 2
  attempt=$((attempt + 1))
done
test -e "$DEVICE"
sudo mkfs.ext4 -F -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard "$DEVICE"
sudo mkdir -p "$MOUNT_PATH"
sudo mount -o discard,defaults "$DEVICE" "$MOUNT_PATH"
sudo chmod 777 "$MOUNT_PATH"
df -h "$MOUNT_PATH"
lsblk -f
' "$DISK_NAME"
}

copy_command() {
    printf 'set -eu
test -d /mnt/data
'
    local path
    for path in "${GCS_PATHS[@]}"; do
        printf 'gcloud storage cp --recursive %s /mnt/data/
' "$(shell_quote "$path")"
    done
    printf 'find /mnt/data -maxdepth 2 -type d | sort
df -h /mnt/data
'
}

unmount_command() {
    cat <<'EOF'
set -eu
sync
sudo umount /mnt/data
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --disk-name)
            require_value "$1" "${2:-}"
            DISK_NAME="$2"
            shift 2
            ;;
        --disk-name=*)
            DISK_NAME="${1#*=}"
            shift
            ;;
        --zone)
            require_value "$1" "${2:-}"
            ZONE="$2"
            shift 2
            ;;
        --zone=*)
            ZONE="${1#*=}"
            shift
            ;;
        --size-gb)
            require_value "$1" "${2:-}"
            SIZE_GB="$2"
            shift 2
            ;;
        --size-gb=*)
            SIZE_GB="${1#*=}"
            shift
            ;;
        --throughput)
            require_value "$1" "${2:-}"
            THROUGHPUT="$2"
            shift 2
            ;;
        --throughput=*)
            THROUGHPUT="${1#*=}"
            shift
            ;;
        --gcs-path)
            require_value "$1" "${2:-}"
            GCS_PATHS+=("$2")
            shift 2
            ;;
        --gcs-path=*)
            GCS_PATHS+=("${1#*=}")
            shift
            ;;
        *)
            die "Unknown argument: $1"
            ;;
    esac
done

command -v gcloud >/dev/null 2>&1 || die "gcloud is not installed or is not on PATH."
[[ -n "$DISK_NAME" ]] || die "DISK_NAME is required."
[[ -n "$ZONE" ]] || die "ZONE is required."
SIZE_GB="$(normalize_size_gb "$SIZE_GB")"
[[ "$SIZE_GB" =~ ^[0-9]+$ ]] || die "SIZE_GB must be an integer GB value."
[[ "$THROUGHPUT" =~ ^[0-9]+$ ]] || die "THROUGHPUT_MIBPS must be an integer."
((${#GCS_PATHS[@]} > 0)) || die "At least one --gcs-path is required."
for gcs_path in "${GCS_PATHS[@]}"; do
    [[ "$gcs_path" == gs://* ]] || die "GCS_PATH must start with gs://: $gcs_path"
done

if ! gcloud_quiet compute machine-types describe "$MACHINE_TYPE" \
    --zone="$ZONE" \
    --format='value(name)' >/dev/null 2>&1; then
    die "$MACHINE_TYPE is not available in $ZONE."
fi

VM_NAME="hdml-$(sanitize_name "$DISK_NAME")"
VM_NAME="${VM_NAME:0:55}"
while [[ "$VM_NAME" == *- ]]; do
    VM_NAME="${VM_NAME%-}"
done

info "Project: $PROJECT_ID"
info "Zone: $ZONE"
info "Disk: $DISK_NAME (${SIZE_GB}GB, ${THROUGHPUT} MiB/s)"
for gcs_path in "${GCS_PATHS[@]}"; do
    info "Source: $gcs_path"
done
info "Temporary VM: $VM_NAME"

if gcloud_quiet compute instances describe "$VM_NAME" \
    --zone="$ZONE" \
    --format='value(name)' >/dev/null 2>&1; then
    die "Temporary VM $VM_NAME already exists in $ZONE. Delete it first or edit VM_NAME in the script."
fi

if disk_exists; then
    existing_users="$(disk_users)"
    if [[ -n "$existing_users" ]]; then
        die "Disk $DISK_NAME is attached to: $existing_users. Detach it before replacing it."
    fi
    info "Deleting existing detached disk $DISK_NAME."
    run_gcloud compute disks delete "$DISK_NAME" \
        --zone="$ZONE" \
        --quiet
fi

info "Creating writeable Hyperdisk ML disk."
run_gcloud compute disks create "$DISK_NAME" \
    --zone="$ZONE" \
    --type=hyperdisk-ml \
    --size="${SIZE_GB}GB" \
    --provisioned-throughput="$THROUGHPUT" \
    --access-mode=READ_WRITE_SINGLE

info "Creating temporary C4 VM and attaching the disk."
run_gcloud compute instances create "$VM_NAME" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size="${BOOT_DISK_SIZE_GB}GB" \
    --boot-disk-type=hyperdisk-balanced \
    --service-account="$STAGING_SERVICE_ACCOUNT" \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --disk="name=$DISK_NAME,device-name=$DISK_NAME,mode=rw,boot=no,auto-delete=no"

wait_for_ssh

info "Formatting and mounting disk over SSH."
ssh_command "$(format_and_mount_command)"

info "Copying data over SSH."
ssh_command "$(copy_command)"

info "Syncing and unmounting disk over SSH."
ssh_command "$(unmount_command)"

info "Detaching data disk."
run_gcloud compute instances detach-disk "$VM_NAME" \
    --zone="$ZONE" \
    --disk="$DISK_NAME" \
    --quiet

info "Deleting temporary VM."
run_gcloud compute instances delete "$VM_NAME" \
    --zone="$ZONE" \
    --quiet

info "Switching disk to READ_ONLY_MANY."
run_gcloud compute disks update "$DISK_NAME" \
    --zone="$ZONE" \
    --access-mode=READ_ONLY_MANY \
    --quiet

info "Final disk state:"
run_gcloud compute disks describe "$DISK_NAME" \
    --zone="$ZONE" \
    --format='yaml(name,status,type,sizeGb,provisionedThroughput,accessMode,users)'

info "Done."
