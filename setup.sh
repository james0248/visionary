#!/usr/bin/env bash

set -euo pipefail

exec "$(dirname "$0")/cloud/starter.sh" --mode setup-only "$@"
