#!/usr/bin/env bash
set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/scripts/lib/runtime.sh"
status=0

for work_dir in "$ROOT/work" "$ROOT/work_smoke"; do
  stop_managed_pid "$work_dir/pids/vllm.pid" "$ROOT/.venv-vllm/bin/vllm" "$ROOT" 1 || status=1
  stop_managed_pid "$work_dir/pids/colbert.pid" "$ROOT/.venv-pilot/bin/python" "$ROOT" 1 || status=1
  stop_managed_pid "$work_dir/pids/e5.pid" "$ROOT/.venv-pilot/bin/python" "$ROOT" 1 || status=1
  stop_managed_pid "$work_dir/pids/bm25.pid" "$ROOT/.venv-pilot/bin/python" "$ROOT" 1 || status=1
done

if [[ $status -eq 0 ]]; then
  echo "Managed pilot servers stopped."
else
  echo "At least one process was left untouched because its PID file did not match; the PID file was quarantined." >&2
fi
exit "$status"
