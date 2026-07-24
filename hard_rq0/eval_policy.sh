#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
source "$ROOT/scripts/lib/runtime.sh"
ensure_local_no_proxy
export HF_HOME=${HF_HOME:-$ROOT/.cache/huggingface}

PILOT_PYTHON=$ROOT/.venv-pilot/bin/python
[[ -x "$PILOT_PYTHON" ]] || { echo "Run scripts/bootstrap.sh first" >&2; exit 1; }
TAG=${TAG:?Set TAG, e.g. base-qwen or bm25-specialist}
SEED=${SEED:-0}
DEFAULT_MODEL_REF=Qwen/Qwen2.5-3B-Instruct
DEFAULT_MODEL_REVISION=aa8e72537993ba99e69dfaafa59ed015b17504d1
MODEL_REF=${MODEL_REF:-${MODEL_PATH:-${MODEL:-$DEFAULT_MODEL_REF}}}
if [[ -z ${MODEL_REVISION:-} ]]; then
  if [[ "$MODEL_REF" == "$DEFAULT_MODEL_REF" ]]; then
    MODEL_REVISION=$DEFAULT_MODEL_REVISION
  else
    MODEL_REVISION=main
  fi
fi
LIMIT=${LIMIT:-}
RESULT_SET=${RESULT_SET:-pilot}
DATA_FILE=${DATA_FILE:-$ROOT/work/hard_rq0/data/final_eval.jsonl}
ASSET_ROOT=${HARD_ASSET_ROOT:-$ROOT/work/hard_rq0/assets/wiki18}
BACKENDS=${BACKENDS:-"bm25 e5"}
TOPKS=${TOPKS:-"3 5 10"}
SPECIALIST_SEEDS=${SPECIALIST_SEEDS:-"13 42 87"}
BM25_PORT=${BM25_PORT:-8101}
E5_PORT=${E5_PORT:-8102}
E5_GPU=${E5_GPU:-7}
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
LLM_GPUS=${LLM_GPUS:-0,1,2,3,4,5,6}
TP=${TP:-1}
DP=${DP:-7}
VLLM_API_SERVER_COUNT=${VLLM_API_SERVER_COUNT:-$DP}
HARD_EVAL_WORKERS=${HARD_EVAL_WORKERS:-112}
VLLM_BATCH_INVARIANT=${VLLM_BATCH_INVARIANT:-1}
VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.88}
LLM_PORT=${LLM_PORT:-9000}
KEEP_VLLM=${KEEP_VLLM:-0}
RETRIEVER_PROBE_TIMEOUT=${HARD_RETRIEVER_PROBE_TIMEOUT:-300}

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Invalid TAG=$TAG" >&2; exit 2; }
[[ "$RESULT_SET" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Invalid RESULT_SET=$RESULT_SET" >&2; exit 2; }
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "SEED must be a non-negative integer; got '$SEED'." >&2; exit 2; }
if [[ -n "$LIMIT" && ! "$LIMIT" =~ ^[1-9][0-9]*$ ]]; then
  echo "LIMIT must be empty or a positive integer; got '$LIMIT'." >&2
  exit 2
fi
if [[ "$KEEP_VLLM" != 0 && "$KEEP_VLLM" != 1 ]]; then
  echo "KEEP_VLLM must be 0 or 1; got '$KEEP_VLLM'." >&2
  exit 2
fi
if [[ -z "$MODEL_REF" || -z "$MODEL_REVISION" ]]; then
  echo "MODEL_REF and MODEL_REVISION must not be empty." >&2
  exit 2
fi
[[ -s "$DATA_FILE" ]] || {
  echo "Missing or empty $DATA_FILE; run hard_rq0/prepare_data.sh" >&2
  exit 1
}
DATA_MANIFEST=$(dirname "$DATA_FILE")/.hard-rq0-data-manifest.json
[[ -s "$DATA_MANIFEST" ]] || {
  echo "Missing Hard-RQ0 data manifest; rerun hard_rq0/prepare_data.sh: $DATA_MANIFEST" >&2
  exit 1
}

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
validate_positive_integer HARD_RETRIEVER_PROBE_TIMEOUT "$RETRIEVER_PROBE_TIMEOUT"
if [[ "$BM25_PORT" == "$E5_PORT" || "$BM25_PORT" == "$LLM_PORT" || "$E5_PORT" == "$LLM_PORT" ]]; then
  echo "BM25_PORT, E5_PORT, and LLM_PORT must be distinct; got $BM25_PORT, $E5_PORT, $LLM_PORT." >&2
  exit 2
fi

validate_positive_integer TP "$TP"
validate_positive_integer DP "$DP"
validate_positive_integer VLLM_API_SERVER_COUNT "$VLLM_API_SERVER_COUNT"
validate_positive_integer HARD_EVAL_WORKERS "$HARD_EVAL_WORKERS"
if ! "$PILOT_PYTHON" - "$GPU_MEMORY_UTILIZATION" <<'PY'
import math
import sys

try:
    value = float(sys.argv[1])
except ValueError:
    raise SystemExit(1)
raise SystemExit(0 if math.isfinite(value) and 0.0 < value < 1.0 else 1)
PY
then
  echo "GPU_MEMORY_UTILIZATION must be a number strictly between 0 and 1; got '$GPU_MEMORY_UTILIZATION'." >&2
  exit 2
fi
if [[ "$VLLM_BATCH_INVARIANT" != 0 && "$VLLM_BATCH_INVARIANT" != 1 ]]; then
  echo "VLLM_BATCH_INVARIANT must be 0 or 1; got '$VLLM_BATCH_INVARIANT'." >&2
  exit 2
fi
if [[ "$VLLM_BATCH_INVARIANT" == 1 ]]; then
  case "$VLLM_ATTENTION_BACKEND" in
    FLASH_ATTN|TRITON_ATTN) ;;
    *)
      echo "Batch-invariant Qwen evaluation requires VLLM_ATTENTION_BACKEND=FLASH_ATTN or TRITON_ATTN; got '$VLLM_ATTENTION_BACKEND'." >&2
      exit 2
      ;;
  esac
fi
validate_gpu_list "$E5_GPU" 1 "Hard-RQ0 E5 retrieval"
validate_gpu_list "$LLM_GPUS" "$((TP * DP))" \
  "Hard-RQ0 vLLM TP=$TP x DP=$DP evaluation"
if [[ ",$LLM_GPUS," == *",$E5_GPU,"* ]]; then
  echo "E5_GPU=$E5_GPU overlaps LLM_GPUS=$LLM_GPUS." >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required to validate the Hard-RQ0 GPU layout." >&2
  exit 1
fi
GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
for gpu_list in "$E5_GPU" "$LLM_GPUS"; do
  IFS=',' read -r -a gpu_ids <<< "$gpu_list"
  for gpu_id in "${gpu_ids[@]}"; do
    if (( 10#$gpu_id >= GPU_COUNT )); then
      echo "GPU $gpu_id does not exist; this node exposes $GPU_COUNT GPUs." >&2
      exit 1
    fi
  done
done

read -r -a BACKEND_ARGS <<< "$BACKENDS"
read -r -a TOPK_ARGS <<< "$TOPKS"
if [[ ${#BACKEND_ARGS[@]} -eq 0 || ${#TOPK_ARGS[@]} -eq 0 ]]; then
  echo "BACKENDS and TOPKS must each contain at least one value." >&2
  exit 2
fi
declare -A SEEN_BACKENDS=()
for backend in "${BACKEND_ARGS[@]}"; do
  case "$backend" in
    bm25|e5) ;;
    *) echo "Unknown Hard-RQ0 backend: $backend" >&2; exit 2 ;;
  esac
  if [[ -n ${SEEN_BACKENDS[$backend]+x} ]]; then
    echo "Duplicate Hard-RQ0 backend: $backend" >&2
    exit 2
  fi
  SEEN_BACKENDS[$backend]=1
done
declare -A SEEN_TOPKS=()
for topk in "${TOPK_ARGS[@]}"; do
  case "$topk" in
    3|5|10) ;;
    *) echo "Hard-RQ0 TOPKS may contain only 3, 5, and 10; got '$topk'." >&2; exit 2 ;;
  esac
  if [[ -n ${SEEN_TOPKS[$topk]+x} ]]; then
    echo "Duplicate Hard-RQ0 top-k value: $topk" >&2
    exit 2
  fi
  SEEN_TOPKS[$topk]=1
done
read -r -a SPECIALIST_SEED_ARGS <<< "$SPECIALIST_SEEDS"
if [[ ${#SPECIALIST_SEED_ARGS[@]} -eq 0 ]]; then
  echo "SPECIALIST_SEEDS must contain at least one seed." >&2
  exit 2
fi
declare -A SEEN_SPECIALIST_SEEDS=()
for specialist_seed in "${SPECIALIST_SEED_ARGS[@]}"; do
  if [[ ! "$specialist_seed" =~ ^[1-9][0-9]*$ || -n ${SEEN_SPECIALIST_SEEDS[$specialist_seed]+x} ]]; then
    echo "SPECIALIST_SEEDS must contain unique positive integers; got '$SPECIALIST_SEEDS'." >&2
    exit 2
  fi
  SEEN_SPECIALIST_SEEDS[$specialist_seed]=1
done

"$PILOT_PYTHON" -m stackpilot.hard_assets check --root "$ASSET_ROOT" >/dev/null
E5_MODEL_PATH=$(unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE; \
  bash "$ROOT/scripts/resolve_hf_model.sh" \
    "$E5_MODEL_SOURCE" "$E5_MODEL_REVISION" "$PILOT_PYTHON")
MODEL_REF=$(unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE; \
  bash "$ROOT/scripts/resolve_hf_model.sh" "$MODEL_REF" "$MODEL_REVISION" "$PILOT_PYTHON")
unset MODEL MODEL_PATH MODEL_LOCAL_ONLY
export MODEL_PATH=$MODEL_REF
export MODEL_REVISION
export HARD_ASSET_ROOT=$ASSET_ROOT
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}
export LLM_GPUS TP DP VLLM_API_SERVER_COUNT GPU_MEMORY_UTILIZATION \
  LLM_PORT VLLM_BATCH_INVARIANT VLLM_ATTENTION_BACKEND
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
export VLLM_READY_TIMEOUT=${VLLM_READY_TIMEOUT:-1800}
if [[ -z "$SERVED_MODEL_NAME" || ! "$MAX_MODEL_LEN" =~ ^[1-9][0-9]*$ ]]; then
  echo "SERVED_MODEL_NAME must be non-empty and MAX_MODEL_LEN must be a positive integer." >&2
  exit 2
fi

EVAL_CONFIG=$ROOT/work/hard_rq0/configs/${RESULT_SET}.yaml
mkdir -p "$(dirname "$EVAL_CONFIG")"
"$PILOT_PYTHON" - configs/hard_rq0.yaml "$EVAL_CONFIG" "$ROOT" "$RESULT_SET" \
  "$BM25_PORT" "$E5_PORT" "$LLM_PORT" "$SERVED_MODEL_NAME" "$ASSET_ROOT" \
  "$E5_MODEL_PATH" "$E5_MODEL_REVISION" "$SPECIALIST_SEEDS" <<'PY'
import sys
import os
from pathlib import Path
import yaml

source = Path(sys.argv[1])
target = Path(sys.argv[2])
root = Path(sys.argv[3])
result_set = sys.argv[4]
bm25_port, e5_port, llm_port = map(int, sys.argv[5:8])
served_model_name = sys.argv[8]
config = yaml.safe_load(source.read_text(encoding="utf-8"))
config["work_dir"] = str(root / "work" / "hard_rq0" / "runs" / result_set)
config["assets"]["root"] = str(Path(sys.argv[9]).resolve())
config["assets"]["corpus_path"] = str(Path(sys.argv[9]).resolve() / "wiki-18.jsonl")
config["assets"]["bm25_index_path"] = str(Path(sys.argv[9]).resolve() / "bm25")
config["assets"]["e5_index_path"] = str(Path(sys.argv[9]).resolve() / "e5_Flat.index")
config["retrieval"]["bm25_port"] = bm25_port
config["retrieval"]["e5_port"] = e5_port
config["retrieval"]["e5_model"] = str(Path(sys.argv[10]).resolve())
config["retrieval"]["e5_model_revision"] = sys.argv[11]
config["training"]["seeds"] = [int(value) for value in sys.argv[12].split()]
config["llm"]["api_base"] = f"http://127.0.0.1:{llm_port}/v1"
config["llm"]["model"] = served_model_name
temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
temporary.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
os.replace(temporary, target)
PY

probe_retriever() {
  local backend=$1
  local port
  local health
  local response
  case "$backend" in
    bm25) port=$BM25_PORT ;;
    e5) port=$E5_PORT ;;
  esac
  if ! health=$(curl --noproxy '*' -fsS --connect-timeout 3 --max-time 30 \
    "http://127.0.0.1:${port}/health"); then
    echo "Hard-RQ0 $backend is not running on port $port; run hard_rq0/launch_retrievers.sh" >&2
    return 1
  fi
  "$PILOT_PYTHON" - "$health" "$backend" "$E5_GPU" "$E5_MODEL_PATH" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(sys.argv[1])
backend = sys.argv[2]
expected_e5_gpu = sys.argv[3]
expected_e5_model = Path(sys.argv[4]).resolve()
if payload.get("status") != "ok" or payload.get("backend") != backend:
    raise SystemExit(f"Unexpected {backend} health response: {payload}")
if backend == "e5":
    if payload.get("faiss_gpu") is not True or int(payload.get("faiss_gpu_count", 0)) != 1:
        raise SystemExit(f"E5 is not using one visible FAISS GPU: {payload}")
    if payload.get("faiss_gpu_load_mode") != "paged-fp16-flat":
        raise SystemExit(f"E5 is not using the memory-safe paged FAISS loader: {payload}")
    if payload.get("faiss_storage_dtype") != "float16":
        raise SystemExit(f"E5 is not using FP16 FAISS storage: {payload}")
    if str(payload.get("cuda_visible_devices")) != expected_e5_gpu:
        raise SystemExit(f"E5 is running on an unexpected CUDA device: {payload}")
    if Path(str(payload.get("retriever_model", ""))).resolve() != expected_e5_model:
        raise SystemExit(f"E5 is running an unexpected encoder model: {payload}")
PY
  response=$(curl --noproxy '*' -fsS --connect-timeout 3 --max-time "$RETRIEVER_PROBE_TIMEOUT" \
    -X POST "http://127.0.0.1:${port}/retrieve" \
    -H 'Content-Type: application/json' \
    -d '{"queries":["Who wrote Hamlet?"],"topk":1,"return_scores":true}')
  "$PILOT_PYTHON" - "$response" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
results = payload.get("result")
if not results or not results[0]:
    raise SystemExit(f"Invalid retrieval probe response: {payload}")
PY
}
for backend in "${BACKEND_ARGS[@]}"; do
  probe_retriever "$backend"
done

vllm_started=0
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ $vllm_started -eq 1 && "$KEEP_VLLM" != 1 ]]; then
    bash "$ROOT/scripts/stop_servers.sh" || true
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

bash "$ROOT/scripts/stop_servers.sh" || true
bash "$ROOT/scripts/launch_vllm_bg.sh"
vllm_started=1

ARGS=(
  --config "$EVAL_CONFIG"
  --data-file "$DATA_FILE"
  --tag "$TAG"
  --seed "$SEED"
  --workers "$HARD_EVAL_WORKERS"
)
if [[ -n "$LIMIT" ]]; then
  ARGS+=(--limit "$LIMIT")
fi
ARGS+=(--backends "${BACKEND_ARGS[@]}" --topks "${TOPK_ARGS[@]}")

"$PILOT_PYTHON" -m stackpilot.hard_policy_eval "${ARGS[@]}"

echo "Hard-RQ0 policy evaluation complete: $TAG seed=$SEED model=$MODEL_REF"
