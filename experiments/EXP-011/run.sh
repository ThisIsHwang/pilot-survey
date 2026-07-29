#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PROFILE=${PROFILE:-pilot} bash "$ROOT/trace_go/run_c.sh" "$@"
