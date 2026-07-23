#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
source "$ROOT/scripts/lib/runtime.sh"
ensure_local_no_proxy

PILOT_PYTHON=$ROOT/.venv-pilot/bin/python
[[ -x "$PILOT_PYTHON" ]] || { echo "Run scripts/bootstrap.sh first" >&2; exit 1; }
HYBRID_PORT=${HYBRID_PORT:-8300}
BM25_PORT=${BM25_PORT:-8101}
E5_PORT=${E5_PORT:-8102}
UPSTREAM_TOPK=${UPSTREAM_TOPK:-100}
RRF_CONSTANT=${RRF_CONSTANT:-60}
READY_TIMEOUT=${HYBRID_READY_TIMEOUT:-600}
PID_ROOT=$ROOT/work/experiments/services
LOG_ROOT=$ROOT/logs/experiments/services
mkdir -p "$PID_ROOT" "$LOG_ROOT"

bash "$ROOT/hard_rq0/launch_retrievers.sh"

PID_FILE=$PID_ROOT/hybrid-rrf.pid
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Hybrid RRF already running on port $HYBRID_PORT"
  exit 0
fi
rm -f "$PID_FILE"
require_free_port "$PILOT_PYTHON" "$HYBRID_PORT"
LOG_FILE=$LOG_ROOT/hybrid-rrf.log
PID=$(CUDA_VISIBLE_DEVICES='' start_managed_process \
  "$PILOT_PYTHON" "$LOG_FILE" "$PILOT_PYTHON" -m stackpilot.hybrid_rrf_server \
  --bm25-url "http://127.0.0.1:${BM25_PORT}/retrieve" \
  --e5-url "http://127.0.0.1:${E5_PORT}/retrieve" \
  --upstream-topk "$UPSTREAM_TOPK" --rrf-constant "$RRF_CONSTANT" \
  --topk 3 --port "$HYBRID_PORT")
echo "$PID" > "$PID_FILE"
wait_for_http "$PID" "http://127.0.0.1:${HYBRID_PORT}/health" \
  "$READY_TIMEOUT" "$LOG_FILE" '"backend":"hybrid-rrf"'
echo "Hybrid RRF ready: http://127.0.0.1:${HYBRID_PORT}/retrieve"
