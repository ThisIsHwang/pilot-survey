#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
source "$ROOT/scripts/lib/runtime.sh"
ensure_local_no_proxy

PILOT_PYTHON=$ROOT/.venv-pilot/bin/python
[[ -x "$PILOT_PYTHON" ]] || { echo "Run scripts/bootstrap.sh first" >&2; exit 1; }
MIXED_PORT=${MIXED_PORT:-8200}
BM25_PORT=${BM25_PORT:-8101}
E5_PORT=${E5_PORT:-8102}
READY_TIMEOUT=${MIXED_READY_TIMEOUT:-600}
PID_ROOT=$ROOT/work/experiments/services
LOG_ROOT=$ROOT/logs/experiments/services
mkdir -p "$PID_ROOT" "$LOG_ROOT"

bash "$ROOT/experiments/ensure_retrievers.sh"

PID_FILE=$PID_ROOT/mixed-router.pid
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Mixed router already running on port $MIXED_PORT"
  exit 0
fi
rm -f "$PID_FILE"
require_free_port "$PILOT_PYTHON" "$MIXED_PORT"
LOG_FILE=$LOG_ROOT/mixed-router.log
ASSIGNMENT_LOG=${ASSIGNMENT_LOG:-$ROOT/work/experiments/services/mixed-assignments.jsonl}
PID=$(CUDA_VISIBLE_DEVICES='' start_managed_process \
  "$PILOT_PYTHON" "$LOG_FILE" "$PILOT_PYTHON" -m stackpilot.mixed_retriever_server \
  --bm25-url "http://127.0.0.1:${BM25_PORT}/retrieve" \
  --e5-url "http://127.0.0.1:${E5_PORT}/retrieve" \
  --topk 3 --port "$MIXED_PORT" --assignment-log "$ASSIGNMENT_LOG")
echo "$PID" > "$PID_FILE"
wait_for_http "$PID" "http://127.0.0.1:${MIXED_PORT}/health" \
  "$READY_TIMEOUT" "$LOG_FILE" '"backend":"mixed"'
echo "Mixed router ready: http://127.0.0.1:${MIXED_PORT}/retrieve"
