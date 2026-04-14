#!/usr/bin/env bash

set -euo pipefail

exec "$(dirname "$0")/scripts/cloud/starter.sh" --mode setup-only "$@"
