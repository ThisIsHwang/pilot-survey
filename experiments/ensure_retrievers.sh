#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
BM25_PORT=${BM25_PORT:-8101}
E5_PORT=${E5_PORT:-8102}
E5_GPU=${E5_GPU:-7}
E5_MODEL_REVISION=${E5_MODEL_REVISION:-f52bf8ec8c7124536f0efb74aca902b2995e5bcd}
PILOT_PYTHON=$ROOT/.venv-pilot/bin/python
[[ -x "$PILOT_PYTHON" ]] || { echo "Run scripts/bootstrap.sh first" >&2; exit 1; }
ASSET_ROOT=${HARD_ASSET_ROOT:-$ROOT/work/hard_rq0/assets/wiki18}
CORPUS_PATH=$ASSET_ROOT/wiki-18.jsonl
BM25_INDEX=$ASSET_ROOT/bm25
E5_INDEX=$ASSET_ROOT/e5_Flat.index
EXPECTED_DOCUMENTS=$(
  "$PILOT_PYTHON" -c \
    'from stackpilot.hard_assets import EXPECTED_DOCUMENTS; print(EXPECTED_DOCUMENTS)'
)

healthy() {
  local port=$1
  local backend=$2
  local index_path=$3
  local payload
  payload=$(curl --noproxy '*' -fsS --connect-timeout 2 --max-time 10 \
    "http://127.0.0.1:${port}/health" 2>/dev/null) || return 1
  "$PILOT_PYTHON" - \
    "$payload" "$backend" "$index_path" "$CORPUS_PATH" \
    "$EXPECTED_DOCUMENTS" "$E5_GPU" "$E5_MODEL_REVISION" "$ROOT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

payload = json.loads(sys.argv[1])
expected = sys.argv[2]
index_path = str(Path(sys.argv[3]).resolve())
corpus_path = str(Path(sys.argv[4]).resolve())
documents = int(sys.argv[5])
e5_gpu = sys.argv[6]
e5_revision = sys.argv[7]
root = Path(sys.argv[8])
expected_server_files = {
    name: hashlib.sha256((root / "stackpilot" / name).read_bytes()).hexdigest()
    for name in ("faiss_gpu.py", "retrieval_concurrency.py", "searchr1_server.py")
}
common = (
    payload.get("status") == "ok"
    and payload.get("backend") == expected
    and payload.get("index_path") == index_path
    and payload.get("corpus_path") == corpus_path
    and payload.get("index_documents") == documents
    and payload.get("corpus_documents") == documents
    and payload.get("server_files") == expected_server_files
)
if not common:
    raise SystemExit(1)
if expected == "e5" and not (
    payload.get("faiss_gpu") is True
    and int(payload.get("faiss_gpu_count", 0)) == 1
    and payload.get("faiss_gpu_load_mode") == "paged-fp16-flat"
    and payload.get("faiss_storage_dtype") == "float16"
    and payload.get("cuda_visible_devices") == e5_gpu
    and payload.get("gpu_search_serialized") is True
    and payload.get("cuda_empty_cache_disabled") is True
    and payload.get("retriever_model_revision") == e5_revision
):
    raise SystemExit(1)
PY
}

if healthy "$BM25_PORT" bm25 "$BM25_INDEX" && \
   healthy "$E5_PORT" e5 "$E5_INDEX"; then
  echo "Reusing healthy hard-RQ0 BM25/E5 retrievers."
  exit 0
fi

echo "Hard-RQ0 retrievers are absent or unhealthy; starting them now."
bash "$ROOT/hard_rq0/launch_retrievers.sh"
