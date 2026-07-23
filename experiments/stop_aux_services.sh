#!/usr/bin/env bash
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/scripts/lib/runtime.sh"
RUNTIME_WORK_ROOT=${STACKPILOT_RUNTIME_ROOT:-$ROOT/work}
status=0
for name in mixed-router hybrid-rrf; do
  stop_managed_pid \
    "$RUNTIME_WORK_ROOT/experiments/services/${name}.pid" \
    "$ROOT/.venv-pilot/bin/python" "$ROOT" 1 || status=1
done
exit "$status"
