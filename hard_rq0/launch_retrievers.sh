#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
source "$ROOT/scripts/lib/runtime.sh"
ensure_local_no_proxy

PYTHON=$ROOT/.venv-pilot/bin/python
[[ -x "$PYTHON" ]] || { echo "Run scripts/bootstrap.sh first" >&2; exit 1; }
ASSET_ROOT=${HARD_ASSET_ROOT:-$ROOT/work/hard_rq0/assets/wiki18}
WORK_ROOT=$ROOT/work/hard_rq0
PID_ROOT=$WORK_ROOT/pids
LOG_ROOT=$ROOT/logs/hard_rq0
BM25_PORT=${BM25_PORT:-8101}
E5_PORT=${E5_PORT:-8102}
E5_GPU=${E5_GPU:-7}
mkdir -p "$PID_ROOT" "$LOG_ROOT"

[[ -s "$ASSET_ROOT/wiki-18.jsonl" ]] || { echo "Run hard_rq0/download_assets.sh first" >&2; exit 1; }
[[ -d "$ASSET_ROOT/bm25" ]] || { echo "Missing BM25 index: $ASSET_ROOT/bm25" >&2; exit 1; }
[[ -s "$ASSET_ROOT/e5_Flat.index" ]] || { echo "Missing E5 index: $ASSET_ROOT/e5_Flat.index" >&2; exit 1; }

bash "$ROOT/hard_rq0/stop_retrievers.sh" || true
require_free_port "$PYTHON" "$BM25_PORT"
require_free_port "$PYTHON" "$E5_PORT"

BM25_LOG=$LOG_ROOT/bm25.log
BM25_PID=$(CUDA_VISIBLE_DEVICES='' start_managed_process \
  "$PYTHON" "$BM25_LOG" "$PYTHON" -m stackpilot.searchr1_server \
  --search-r1-root "$ROOT/upstream/Search-R1" \
  --index-path "$ASSET_ROOT/bm25" \
  --corpus-path "$ASSET_ROOT/wiki-18.jsonl" \
  --retriever-name bm25 --topk 5 --port "$BM25_PORT")
echo "$BM25_PID" > "$PID_ROOT/bm25.pid"
wait_for_http "$BM25_PID" "http://127.0.0.1:${BM25_PORT}/health" 600 "$BM25_LOG" '"status":"ok"'

E5_LOG=$LOG_ROOT/e5.log
E5_PID=$(CUDA_VISIBLE_DEVICES="$E5_GPU" start_managed_process \
  "$PYTHON" "$E5_LOG" "$PYTHON" -m stackpilot.searchr1_server \
  --search-r1-root "$ROOT/upstream/Search-R1" \
  --index-path "$ASSET_ROOT/e5_Flat.index" \
  --corpus-path "$ASSET_ROOT/wiki-18.jsonl" \
  --retriever-name e5 --retriever-model intfloat/e5-base-v2 \
  --topk 5 --port "$E5_PORT" --faiss-gpu)
echo "$E5_PID" > "$PID_ROOT/e5.pid"
wait_for_http "$E5_PID" "http://127.0.0.1:${E5_PORT}/health" 1800 "$E5_LOG" '"status":"ok"'

echo "Hard-RQ0 retrievers ready: BM25=$BM25_PORT, E5=$E5_PORT (GPU $E5_GPU)"
