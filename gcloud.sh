#!/usr/bin/env bash

set -euo pipefail

PROJECT_ID=""
LOGOUT=0
SDK_ROOT="${HOME}/google-cloud-sdk"
SDK_BIN="${SDK_ROOT}/bin"
SDK_BIN_GCLOUD="${SDK_BIN}/gcloud"
PATH_BLOCK_START="# >>> google-cloud-sdk >>>"
PATH_BLOCK_END="# <<< google-cloud-sdk <<<"

usage() {
    echo "Usage: ./gcloud.sh [PROJECT_ID]"
    echo "       ./gcloud.sh --logout"
}

parse_args() {
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --logout|-l)
                LOGOUT=1
                ;;
            --help|-h)
                usage
                exit 0
                ;;
            -*)
                echo "Unknown option: $1" >&2
                usage >&2
                exit 1
                ;;
            *)
                if [ -n "${PROJECT_ID}" ]; then
                    echo "Only one project id can be provided." >&2
                    usage >&2
                    exit 1
                fi
                PROJECT_ID="$1"
                ;;
        esac
        shift
    done

    if [ "${LOGOUT}" -eq 1 ] && [ -n "${PROJECT_ID}" ]; then
        echo "The --logout flag cannot be combined with a project id." >&2
        usage >&2
        exit 1
    fi
}

detect_archive_name() {
    local os arch
    os="$(uname -s)"
    arch="$(uname -m)"

    case "${os}" in
        Linux) os="linux" ;;
        Darwin) os="darwin" ;;
        *)
            echo "Unsupported OS: ${os}" >&2
            exit 1
            ;;
    esac

    case "${arch}" in
        x86_64|amd64) arch="x86_64" ;;
        arm64|aarch64)
            if [ "${os}" = "darwin" ]; then
                arch="arm"
            else
                arch="arm"
            fi
            ;;
        *)
            echo "Unsupported architecture: ${arch}" >&2
            exit 1
            ;;
    esac

    echo "google-cloud-cli-${os}-${arch}.tar.gz"
}

append_path_block() {
    local rc_file="$1"
    local path_block

    path_block="${PATH_BLOCK_START}
if [ -d \"${SDK_BIN}\" ]; then
    export PATH=\"${SDK_BIN}:\$PATH\"
fi
${PATH_BLOCK_END}"

    if [ -f "${rc_file}" ] && grep -Fq "${PATH_BLOCK_START}" "${rc_file}"; then
        return
    fi

    {
        printf "\n%s\n" "${path_block}"
    } >> "${rc_file}"
}

persist_path() {
    append_path_block "${HOME}/.zprofile"
    append_path_block "${HOME}/.zshrc"
    append_path_block "${HOME}/.bashrc"
    append_path_block "${HOME}/.profile"
}

ensure_gcloud() {
    local archive_name archive_path extract_dir

    if command -v gcloud >/dev/null 2>&1; then
        echo "Using existing gcloud installation: $(command -v gcloud)"
        return
    fi

    if [ -x "${SDK_BIN_GCLOUD}" ]; then
        export PATH="${SDK_BIN}:${PATH}"
        echo "Using local gcloud installation: ${SDK_BIN_GCLOUD}"
        return
    fi

    archive_name="$(detect_archive_name)"
    archive_path="/tmp/${archive_name}"
    extract_dir="$(mktemp -d /tmp/google-cloud-sdk.XXXXXX)"

    curl -L "https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/${archive_name}" -o "${archive_path}"
    tar -xf "${archive_path}" -C "${extract_dir}"
    rm -rf "${SDK_ROOT}"
    mv "${extract_dir}/google-cloud-sdk" "${SDK_ROOT}"
    rm -rf "${extract_dir}"
    rm -f "${archive_path}"

    "${SDK_ROOT}/install.sh" --quiet --path-update=false
    export PATH="${SDK_BIN}:${PATH}"
}

use_existing_gcloud() {
    if command -v gcloud >/dev/null 2>&1; then
        echo "Using existing gcloud installation: $(command -v gcloud)"
        return 0
    fi

    if [ -x "${SDK_BIN_GCLOUD}" ]; then
        export PATH="${SDK_BIN}:${PATH}"
        echo "Using local gcloud installation: ${SDK_BIN_GCLOUD}"
        return 0
    fi

    return 1
}

logout_gcloud() {
    if ! use_existing_gcloud; then
        echo "gcloud is not installed. Nothing to log out."
        exit 0
    fi

    gcloud auth revoke --all --quiet || true
    gcloud auth application-default revoke --quiet || true

    echo "Logged out from gcloud."
}

parse_args "$@"

if [ "${LOGOUT}" -eq 1 ]; then
    logout_gcloud
    exit 0
fi

ensure_gcloud
persist_path

if [ -n "${PROJECT_ID}" ]; then
    gcloud auth login
    gcloud config set project "${PROJECT_ID}"
else
    gcloud init
    PROJECT_ID="$(gcloud config get-value project 2>/dev/null)"
fi

if [ -z "${PROJECT_ID}" ] || [ "${PROJECT_ID}" = "(unset)" ]; then
    echo "No active Google Cloud project is configured."
    echo "Pass a project id like './gcloud.sh <PROJECT_ID>' or run 'gcloud config set project <PROJECT_ID>'."
    exit 1
fi

gcloud auth application-default login
gcloud auth application-default set-quota-project "${PROJECT_ID}"

echo "Active Google Cloud project: ${PROJECT_ID}"
echo "gcloud was added to your shell startup files. Open a new shell or run 'source ~/.zprofile' before using it interactively."
