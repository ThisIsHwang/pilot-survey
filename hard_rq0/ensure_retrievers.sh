#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
source "$ROOT/scripts/lib/runtime.sh"
ensure_local_no_proxy
export HF_HOME=${HF_HOME:-$ROOT/.cache/huggingface}

PILOT_PYTHON=$ROOT/.venv-pilot/bin/python
[[ -x "$PILOT_PYTHON" ]] || {
  echo "Run scripts/bootstrap.sh first." >&2
  exit 1
}
BM25_PORT=${BM25_PORT:-8101}
E5_PORT=${E5_PORT:-8102}
E5_GPU=${E5_GPU:-7}
E5_FAISS_TEMP_MEMORY_MIB=${E5_FAISS_TEMP_MEMORY_MIB:-512}
PROBE_TIMEOUT=${HARD_RETRIEVER_PROBE_TIMEOUT:-300}
DEFAULT_E5_MODEL=intfloat/e5-base-v2
DEFAULT_E5_MODEL_REVISION=f52bf8ec8c7124536f0efb74aca902b2995e5bcd
E5_MODEL_SOURCE=${E5_MODEL:-$DEFAULT_E5_MODEL}
if [[ -z ${E5_MODEL_REVISION:-} ]]; then
  if [[ "$E5_MODEL_SOURCE" == "$DEFAULT_E5_MODEL" ]]; then
    E5_MODEL_REVISION=$DEFAULT_E5_MODEL_REVISION
  else
    E5_MODEL_REVISION=main
  fi
fi
for name in BM25_PORT E5_PORT E5_FAISS_TEMP_MEMORY_MIB PROBE_TIMEOUT; do
  value=${!name}
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
    echo "$name must be a positive integer; got '$value'." >&2
    exit 2
  }
done
if (( BM25_PORT > 65535 || E5_PORT > 65535 )) || [[ "$BM25_PORT" == "$E5_PORT" ]]; then
  echo "BM25_PORT and E5_PORT must be distinct valid ports; got $BM25_PORT and $E5_PORT." >&2
  exit 2
fi

ASSET_ROOT=${HARD_ASSET_ROOT:-$ROOT/work/hard_rq0/assets/wiki18}
RUNTIME_WORK_ROOT=${STACKPILOT_RUNTIME_ROOT:-$ROOT/work}
PID_ROOT=$RUNTIME_WORK_ROOT/hard_rq0/pids
CORPUS_PATH=$ASSET_ROOT/wiki-18.jsonl
BM25_INDEX=$ASSET_ROOT/bm25
E5_INDEX=$ASSET_ROOT/e5_Flat.index
EXPECTED_DOCUMENTS=$(
  "$PILOT_PYTHON" -c \
    'from stackpilot.hard_assets import EXPECTED_DOCUMENTS; print(EXPECTED_DOCUMENTS)'
)
EXPECTED_E5_INDEX_BYTES=$((EXPECTED_DOCUMENTS * 768 * 4))
E5_MODEL_PATH=$(unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE; \
  bash "$ROOT/scripts/resolve_hf_model.sh" \
    "$E5_MODEL_SOURCE" "$E5_MODEL_REVISION" "$PILOT_PYTHON")
E5_RESOLVED_REVISION=$(
  "$PILOT_PYTHON" - "$E5_MODEL_PATH" <<'PY'
import sys
from pathlib import Path

model = Path(sys.argv[1]).resolve()
print(model.name if model.parent.name == "snapshots" else "local")
PY
)

healthy() {
  local port=$1
  local backend=$2
  local index_path=$3
  local pid_file=$4
  local payload
  local expected_pid
  local cmdline
  [[ -r "$pid_file" ]] || return 1
  expected_pid=$(<"$pid_file")
  [[ "$expected_pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$expected_pid" 2>/dev/null || return 1
  cmdline=$(process_cmdline "$expected_pid")
  [[ "$cmdline" == *"$ROOT/.venv-pilot/bin/python"* \
    && "$cmdline" == *"stackpilot.searchr1_server"* ]] || return 1
  [[ "$(readlink -f "/proc/$expected_pid/cwd" 2>/dev/null || true)" == "$ROOT" ]] \
    || return 1
  payload=$(curl --noproxy '*' -fsS --connect-timeout 2 --max-time 10 \
    "http://127.0.0.1:${port}/health" 2>/dev/null) || return 1
  "$PILOT_PYTHON" - \
    "$payload" "$backend" "$index_path" "$CORPUS_PATH" \
    "$EXPECTED_DOCUMENTS" "$E5_GPU" "$E5_MODEL_PATH" \
    "$E5_RESOLVED_REVISION" "$E5_FAISS_TEMP_MEMORY_MIB" \
    "$EXPECTED_E5_INDEX_BYTES" "$expected_pid" "$ROOT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

(
    raw,
    expected_backend,
    expected_index,
    expected_corpus,
    expected_documents,
    expected_e5_gpu,
    expected_e5_model,
    expected_e5_revision,
    expected_temp_memory,
    expected_index_bytes,
    expected_pid,
    root_text,
) = sys.argv[1:]
payload = json.loads(raw)
root = Path(root_text)
expected_server_files = {
    name: hashlib.sha256((root / "stackpilot" / name).read_bytes()).hexdigest()
    for name in ("faiss_gpu.py", "retrieval_concurrency.py", "searchr1_server.py")
}
common = (
    payload.get("status") == "ok"
    and int(payload.get("process_id", -1)) == int(expected_pid)
    and payload.get("backend") == expected_backend
    and Path(str(payload.get("index_path", ""))).resolve()
    == Path(expected_index).resolve()
    and Path(str(payload.get("corpus_path", ""))).resolve()
    == Path(expected_corpus).resolve()
    and int(payload.get("index_documents", -1)) == int(expected_documents)
    and int(payload.get("corpus_documents", -1)) == int(expected_documents)
    and payload.get("server_files") == expected_server_files
)
if not common:
    raise SystemExit(1)
if expected_backend == "bm25":
    if payload.get("faiss_gpu") is not False:
        raise SystemExit(1)
else:
    checks = (
        payload.get("faiss_gpu") is True,
        int(payload.get("faiss_gpu_count", 0)) == 1,
        payload.get("faiss_gpu_load_mode") == "paged-fp32-flat",
        payload.get("faiss_storage_dtype") == "float32",
        int(payload.get("faiss_temp_memory_mib", 0)) == int(expected_temp_memory),
        int(payload.get("faiss_index_bytes", 0)) == int(expected_index_bytes),
        str(payload.get("cuda_visible_devices")) == expected_e5_gpu,
        payload.get("gpu_search_serialized") is True,
        payload.get("cuda_empty_cache_disabled") is True,
        Path(str(payload.get("retriever_model", ""))).resolve()
        == Path(expected_e5_model).resolve(),
        str(payload.get("retriever_model_revision") or "")
        == expected_e5_revision,
    )
    if not all(checks):
        raise SystemExit(1)
PY
}

probe() {
  local port=$1
  local payload
  payload=$(curl --noproxy '*' -fsS --connect-timeout 3 \
    --max-time "$PROBE_TIMEOUT" \
    -X POST "http://127.0.0.1:${port}/retrieve" \
    -H 'Content-Type: application/json' \
    -d '{"queries":["Who wrote Hamlet?"],"topk":1,"return_scores":true}' \
    2>/dev/null) || return 1
  "$PILOT_PYTHON" - "$payload" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
results = payload.get("result")
valid = (
    isinstance(results, list)
    and len(results) == 1
    and isinstance(results[0], list)
    and bool(results[0])
    and isinstance(results[0][0], dict)
    and isinstance(results[0][0].get("document"), dict)
)
raise SystemExit(0 if valid else 1)
PY
}

if healthy "$BM25_PORT" bm25 "$BM25_INDEX" "$PID_ROOT/bm25.pid" \
  && healthy "$E5_PORT" e5 "$E5_INDEX" "$PID_ROOT/e5.pid" \
  && probe "$BM25_PORT" \
  && probe "$E5_PORT"; then
  echo "Reusing verified hard-RQ0 BM25/E5 retrievers after live search probes."
  exit 0
fi

echo "Hard-RQ0 retrievers are absent, stale, or fail a live search; restarting them."
env \
  HARD_ASSET_ROOT="$ASSET_ROOT" \
  BM25_PORT="$BM25_PORT" \
  E5_PORT="$E5_PORT" \
  E5_GPU="$E5_GPU" \
  E5_MODEL="$E5_MODEL_SOURCE" \
  E5_MODEL_REVISION="$E5_MODEL_REVISION" \
  E5_FAISS_TEMP_MEMORY_MIB="$E5_FAISS_TEMP_MEMORY_MIB" \
  HARD_RETRIEVER_PROBE_TIMEOUT="$PROBE_TIMEOUT" \
  bash "$ROOT/hard_rq0/launch_retrievers.sh"
