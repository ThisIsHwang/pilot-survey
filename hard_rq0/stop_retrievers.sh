#!/usr/bin/env bash
set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/scripts/lib/runtime.sh"
WORK_ROOT=$ROOT/work/hard_rq0
status=0

stop_managed_pid "$WORK_ROOT/pids/e5.pid" "$ROOT/.venv-pilot/bin/python" "$ROOT" 1 || status=1
stop_managed_pid "$WORK_ROOT/pids/bm25.pid" "$ROOT/.venv-pilot/bin/python" "$ROOT" 1 || status=1
exit "$status"
