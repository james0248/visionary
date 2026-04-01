#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK_DIR="${SCRIPT_DIR}/google-cloud-sdk"
SDK_ARCHIVE="${SCRIPT_DIR}/google-cloud-cli-linux-x86_64.tar.gz"
SDK_URL="https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz"

if command -v gcloud >/dev/null 2>&1; then
    echo "Using existing gcloud installation: $(command -v gcloud)"
elif [ -x "${SDK_DIR}/bin/gcloud" ]; then
    export PATH="${SDK_DIR}/bin:${PATH}"
    echo "Using local gcloud installation: ${SDK_DIR}/bin/gcloud"
else
    curl -L "${SDK_URL}" -o "${SDK_ARCHIVE}"
    tar -xf "${SDK_ARCHIVE}" -C "${SCRIPT_DIR}"
    "${SDK_DIR}/install.sh" --quiet
    export PATH="${SDK_DIR}/bin:${PATH}"
    rm -f "${SDK_ARCHIVE}"
fi

if [ -f "${SDK_DIR}/path.bash.inc" ]; then
    # shellcheck disable=SC1091
    . "${SDK_DIR}/path.bash.inc"
fi

gcloud init

PROJECT_ID="$(gcloud config get-value project 2>/dev/null)"
if [ -z "${PROJECT_ID}" ] || [ "${PROJECT_ID}" = "(unset)" ]; then
    echo "No active Google Cloud project is configured."
    echo "Run 'gcloud config set project <PROJECT_ID>' and rerun this script."
    exit 1
fi

gcloud auth application-default login
gcloud auth application-default set-quota-project "${PROJECT_ID}"

echo "Active Google Cloud project: ${PROJECT_ID}"
