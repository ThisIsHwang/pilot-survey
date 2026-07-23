#!/usr/bin/env bash
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/scripts/lib/runtime.sh"
status=0
for name in mixed-router hybrid-rrf; do
  stop_managed_pid \
    "$ROOT/work/experiments/services/${name}.pid" \
    "$ROOT/.venv-pilot/bin/python" "$ROOT" 1 || status=1
done
exit "$status"
