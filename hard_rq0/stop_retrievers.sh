#!/usr/bin/env bash
set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/scripts/lib/runtime.sh"
WORK_ROOT=$ROOT/work/hard_rq0
status=0

stop_managed_pid "$WORK_ROOT/pids/e5.pid" "$ROOT/.venv-pilot/bin/python" "$ROOT" 1 || status=1
stop_managed_pid "$WORK_ROOT/pids/bm25.pid" "$ROOT/.venv-pilot/bin/python" "$ROOT" 1 || status=1
if [[ $status -eq 0 ]]; then
  echo "Managed Hard-RQ0 retrievers stopped."
else
  echo "At least one Hard-RQ0 process was left untouched because its PID file did not match." >&2
fi
exit "$status"
