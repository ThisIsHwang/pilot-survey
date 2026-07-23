#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
source "$ROOT/scripts/lib/bootstrap_java.sh"
source "$ROOT/scripts/lib/runtime.sh"
ensure_local_no_proxy
export HF_HOME=${HF_HOME:-$ROOT/.cache/huggingface}

PILOT_PYTHON=$ROOT/.venv-pilot/bin/python
[[ -x "$PILOT_PYTHON" ]] || { echo "Run scripts/bootstrap.sh first" >&2; exit 1; }
SEARCH_R1=${SEARCH_R1_ROOT:-$ROOT/upstream/Search-R1}
[[ -e "$SEARCH_R1/.git" ]] || {
  echo "Missing pinned Search-R1 checkout; run scripts/bootstrap.sh first." >&2
  exit 1
}
ensure_java "$ROOT"

ASSET_ROOT=${HARD_ASSET_ROOT:-$ROOT/work/hard_rq0/assets/wiki18}
RUNTIME_WORK_ROOT=${STACKPILOT_RUNTIME_ROOT:-$ROOT/work}
RUNTIME_LOG_ROOT=${STACKPILOT_LOG_ROOT:-$ROOT/logs}
WORK_ROOT=$RUNTIME_WORK_ROOT/hard_rq0
PID_ROOT=$WORK_ROOT/pids
LOG_ROOT=$RUNTIME_LOG_ROOT/hard_rq0
BM25_PORT=${BM25_PORT:-8101}
E5_PORT=${E5_PORT:-8102}
E5_GPU=${E5_GPU:-7}
E5_FAISS_TEMP_MEMORY_MIB=${E5_FAISS_TEMP_MEMORY_MIB:-512}
TRAIN_GPUS=${TRAIN_GPUS:-0,1,2,3,4,5,6}
N_GPUS=${N_GPUS:-7}
LLM_GPUS=${LLM_GPUS:-0,1,2,3,4,5,6}
TP=${TP:-1}
DP=${DP:-7}
LLM_PORT=${LLM_PORT:-9000}
READY_TIMEOUT=${HARD_RETRIEVER_READY_TIMEOUT:-14400}
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
mkdir -p "$PID_ROOT" "$LOG_ROOT"

validate_positive_integer() {
  local name=$1
  local value=$2
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$name must be a positive integer; got '$value'." >&2
    return 1
  fi
}

validate_port() {
  local name=$1
  local value=$2
  validate_positive_integer "$name" "$value" || return 1
  if (( value > 65535 )); then
    echo "$name must be at most 65535; got '$value'." >&2
    return 1
  fi
}

validate_port BM25_PORT "$BM25_PORT"
validate_port E5_PORT "$E5_PORT"
validate_port LLM_PORT "$LLM_PORT"
validate_positive_integer N_GPUS "$N_GPUS"
validate_positive_integer TP "$TP"
validate_positive_integer DP "$DP"
validate_positive_integer HARD_RETRIEVER_READY_TIMEOUT "$READY_TIMEOUT"
validate_positive_integer HARD_RETRIEVER_PROBE_TIMEOUT "$PROBE_TIMEOUT"
validate_positive_integer E5_FAISS_TEMP_MEMORY_MIB "$E5_FAISS_TEMP_MEMORY_MIB"
if [[ "$N_GPUS" != 7 ]]; then
  echo "Hard-RQ0 reserves one H100 for E5 and requires N_GPUS=7; got '$N_GPUS'." >&2
  exit 2
fi
if [[ "$BM25_PORT" == "$E5_PORT" || "$BM25_PORT" == "$LLM_PORT" || "$E5_PORT" == "$LLM_PORT" ]]; then
  echo "BM25_PORT, E5_PORT, and LLM_PORT must be distinct; got $BM25_PORT, $E5_PORT, $LLM_PORT." >&2
  exit 2
fi

validate_gpu_list "$E5_GPU" 1 "Hard-RQ0 E5 retrieval"
validate_gpu_list "$TRAIN_GPUS" "$N_GPUS" "Hard-RQ0 training"
validate_gpu_list "$LLM_GPUS" "$((TP * DP))" \
  "Hard-RQ0 vLLM TP=$TP x DP=$DP evaluation"
if [[ ",$TRAIN_GPUS," == *",$E5_GPU,"* ]]; then
  echo "E5_GPU=$E5_GPU overlaps TRAIN_GPUS=$TRAIN_GPUS." >&2
  exit 2
fi
if [[ ",$LLM_GPUS," == *",$E5_GPU,"* ]]; then
  echo "E5_GPU=$E5_GPU overlaps LLM_GPUS=$LLM_GPUS." >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required to validate the Hard-RQ0 GPU layout." >&2
  exit 1
fi
GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
for gpu_list in "$E5_GPU" "$TRAIN_GPUS" "$LLM_GPUS"; do
  IFS=',' read -r -a gpu_ids <<< "$gpu_list"
  for gpu_id in "${gpu_ids[@]}"; do
    if (( 10#$gpu_id >= GPU_COUNT )); then
      echo "GPU $gpu_id does not exist; this node exposes $GPU_COUNT GPUs." >&2
      exit 1
    fi
  done
done

CORPUS_PATH=$ASSET_ROOT/wiki-18.jsonl
BM25_INDEX=$ASSET_ROOT/bm25
E5_INDEX=$ASSET_ROOT/e5_Flat.index
ASSET_MANIFEST=$ASSET_ROOT/.hard-rq0-assets-manifest.json
EXPECTED_DOCUMENTS=$(
  "$PILOT_PYTHON" -c \
    'from stackpilot.hard_assets import EXPECTED_DOCUMENTS; print(EXPECTED_DOCUMENTS)'
)
[[ -s "$ASSET_MANIFEST" ]] || {
  echo "Missing Hard-RQ0 asset manifest; run hard_rq0/download_assets.sh first: $ASSET_MANIFEST" >&2
  exit 1
}
"$PILOT_PYTHON" -m stackpilot.hard_assets check --root "$ASSET_ROOT" >/dev/null
echo "Validated Hard-RQ0 assets: $ASSET_ROOT"

# Convert the mutable Hub reference to the pinned local snapshot before the
# long index load. Clear stale offline flags only inside the resolver process.
E5_MODEL=$(unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE; \
  bash "$ROOT/scripts/resolve_hf_model.sh" \
    "$E5_MODEL_SOURCE" "$E5_MODEL_REVISION" "$PILOT_PYTHON")
E5_RESOLVED_REVISION=$("$PILOT_PYTHON" - "$E5_MODEL" <<'PY'
import sys
from pathlib import Path

model = Path(sys.argv[1]).resolve()
print(model.name if model.parent.name == "snapshots" else "local")
PY
)
E5_MODEL_REVISION_ARGS=()
if [[ -n "$E5_RESOLVED_REVISION" ]]; then
  E5_MODEL_REVISION_ARGS=(--retriever-model-revision "$E5_RESOLVED_REVISION")
fi

bash "$ROOT/hard_rq0/stop_retrievers.sh" || true
for port in "$BM25_PORT" "$E5_PORT"; do
  if port_is_open "$PILOT_PYTHON" "$port"; then
    echo "Hard-RQ0 port $port is already in use after managed cleanup." >&2
    echo "Stop the owning process or choose consistent BM25_PORT/E5_PORT overrides." >&2
    exit 1
  fi
done

launch_complete=0
cleanup_started() {
  local status=$?
  trap - EXIT INT TERM
  if [[ $launch_complete -ne 1 ]]; then
    bash "$ROOT/hard_rq0/stop_retrievers.sh" || true
  fi
  exit "$status"
}
trap cleanup_started EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

BM25_LOG=$LOG_ROOT/bm25.log
BM25_PID=$(CUDA_VISIBLE_DEVICES='' start_managed_process \
  "$PILOT_PYTHON" "$BM25_LOG" "$PILOT_PYTHON" -m stackpilot.searchr1_server \
  --search-r1-root "$SEARCH_R1" \
  --index-path "$BM25_INDEX" \
  --corpus-path "$CORPUS_PATH" \
  --retriever-name bm25 --topk 10 --port "$BM25_PORT" \
  --expected-documents "$EXPECTED_DOCUMENTS")
echo "$BM25_PID" > "$PID_ROOT/bm25.pid"
wait_for_http "$BM25_PID" "http://127.0.0.1:${BM25_PORT}/health" \
  "$READY_TIMEOUT" "$BM25_LOG" '"backend":"bm25"'

E5_LOG=$LOG_ROOT/e5.log
E5_PID=$(HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  RETRIEVER_DISABLE_CUDA_EMPTY_CACHE=1 CUDA_VISIBLE_DEVICES="$E5_GPU" \
  start_managed_process \
  "$PILOT_PYTHON" "$E5_LOG" "$PILOT_PYTHON" -m stackpilot.searchr1_server \
  --search-r1-root "$SEARCH_R1" \
  --index-path "$E5_INDEX" \
  --corpus-path "$CORPUS_PATH" \
  --retriever-name e5 --retriever-model "$E5_MODEL" \
  "${E5_MODEL_REVISION_ARGS[@]}" \
  --topk 10 --port "$E5_PORT" --faiss-gpu --faiss-gpu-paged-load \
  --faiss-gpu-temp-memory-mib "$E5_FAISS_TEMP_MEMORY_MIB" \
  --expected-documents "$EXPECTED_DOCUMENTS")
echo "$E5_PID" > "$PID_ROOT/e5.pid"
wait_for_http "$E5_PID" "http://127.0.0.1:${E5_PORT}/health" \
  "$READY_TIMEOUT" "$E5_LOG" '"backend":"e5"'

probe_retriever() {
  local name=$1
  local port=$2
  local pid=$3
  local log_file=$4
  local expected_index=$5
  local expected_model=${6:-}
  local expected_model_revision=${7:-}
  local response
  local health

  health=$(curl --noproxy '*' -fsS --connect-timeout 3 --max-time 30 \
    "http://127.0.0.1:${port}/health")
  "$PILOT_PYTHON" - "$health" "$name" "$expected_index" "$CORPUS_PATH" \
    "$E5_GPU" "$expected_model" "$expected_model_revision" \
    "$EXPECTED_DOCUMENTS" "$E5_FAISS_TEMP_MEMORY_MIB" <<'PY'
import json
import sys
from pathlib import Path

(
    raw,
    expected_backend,
    expected_index,
    expected_corpus,
    expected_e5_gpu,
    expected_model,
    expected_model_revision,
    expected_documents,
    expected_faiss_temp_memory_mib,
) = sys.argv[1:]
payload = json.loads(raw)
if payload.get("status") != "ok" or payload.get("backend") != expected_backend:
    raise SystemExit(f"Unexpected {expected_backend} health response: {payload}")
if Path(str(payload.get("index_path", ""))).resolve() != Path(expected_index).resolve():
    raise SystemExit(f"{expected_backend} loaded an unexpected index: {payload}")
if Path(str(payload.get("corpus_path", ""))).resolve() != Path(expected_corpus).resolve():
    raise SystemExit(f"{expected_backend} loaded an unexpected corpus: {payload}")
if int(payload.get("index_documents", -1)) != int(expected_documents):
    raise SystemExit(f"{expected_backend} loaded an incomplete index: {payload}")
if int(payload.get("corpus_documents", -1)) != int(expected_documents):
    raise SystemExit(f"{expected_backend} loaded an incomplete corpus: {payload}")
if expected_backend == "e5":
    if payload.get("faiss_gpu") is not True or int(payload.get("faiss_gpu_count", 0)) != 1:
        raise SystemExit(f"E5 did not initialize one visible FAISS GPU: {payload}")
    if payload.get("faiss_gpu_load_mode") != "paged-fp16-flat":
        raise SystemExit(f"E5 did not use the memory-safe paged FAISS loader: {payload}")
    if payload.get("faiss_storage_dtype") != "float16":
        raise SystemExit(f"E5 did not use FP16 FAISS storage: {payload}")
    if int(payload.get("faiss_temp_memory_mib", 0)) != int(expected_faiss_temp_memory_mib):
        raise SystemExit(f"E5 used unexpected FAISS scratch memory: {payload}")
    if str(payload.get("cuda_visible_devices")) != expected_e5_gpu:
        raise SystemExit(f"E5 is running on an unexpected CUDA device: {payload}")
    if Path(str(payload.get("retriever_model", ""))).resolve() != Path(expected_model).resolve():
        raise SystemExit(f"E5 loaded an unexpected encoder model: {payload}")
    if str(payload.get("retriever_model_revision") or "") != expected_model_revision:
        raise SystemExit(f"E5 loaded an unexpected encoder revision: {payload}")
PY

  if ! response=$(curl --noproxy '*' -fsS --connect-timeout 3 --max-time "$PROBE_TIMEOUT" \
    -X POST "http://127.0.0.1:${port}/retrieve" \
    -H 'Content-Type: application/json' \
    -d '{"queries":["Who wrote Hamlet?"],"topk":1,"return_scores":true}'); then
    echo "$name retrieval probe failed." >&2
    show_log_tail "$log_file"
    return 1
  fi
  "$PILOT_PYTHON" - "$response" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
results = payload.get("result")
if (
    not results
    or not results[0]
    or not isinstance(results[0][0], dict)
    or not isinstance(results[0][0].get("document"), dict)
):
    raise SystemExit(f"Invalid retrieval probe response: {payload}")
PY
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "$name exited during its retrieval probe." >&2
    show_log_tail "$log_file"
    return 1
  fi
  echo "$name retriever ready on port $port"
}

probe_retriever bm25 "$BM25_PORT" "$BM25_PID" "$BM25_LOG" "$BM25_INDEX"
probe_retriever e5 "$E5_PORT" "$E5_PID" "$E5_LOG" "$E5_INDEX" \
  "$E5_MODEL" "$E5_RESOLVED_REVISION"

launch_complete=1
trap - EXIT INT TERM
echo "Hard-RQ0 retrievers ready: BM25=$BM25_PORT, E5=$E5_PORT (GPU $E5_GPU)"
