#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
source "$ROOT/scripts/lib/runtime.sh"
ensure_local_no_proxy
export HF_HOME=${HF_HOME:-$ROOT/.cache/huggingface}

EXPERIMENT_ID=${EXPERIMENT_ID:?Set EXPERIMENT_ID}
TAG=${TAG:?Set TAG}
SEED=${SEED:-0}
PROFILE=${PROFILE:-pilot}
VARIANT=${VARIANT:-$TAG}
MODEL_REF=${MODEL_REF:-${MODEL_PATH:-}}
MODEL_REVISION=${MODEL_REVISION:-main}
LIMIT=${LIMIT:-}
BACKENDS=${BACKENDS:-"bm25 e5"}
TOPKS=${TOPKS:-"3 5 10"}
DATA_FILE=${DATA_FILE:-$ROOT/work/hard_rq0/data/eval_all.jsonl}
HARD_ASSET_ROOT=${HARD_ASSET_ROOT:-$ROOT/work/hard_rq0/assets/wiki18}
BM25_PORT=${BM25_PORT:-8101}
E5_PORT=${E5_PORT:-8102}
HYBRID_PORT=${HYBRID_PORT:-8300}
E5_GPU=${E5_GPU:-7}
LLM_GPUS=${LLM_GPUS:-0,1,2,3,4,5,6}
TP=${TP:-1}
DP=${DP:-7}
VLLM_API_SERVER_COUNT=${VLLM_API_SERVER_COUNT:-$DP}
NUMBERED_EVAL_WORKERS=${NUMBERED_EVAL_WORKERS:-112}
VLLM_BATCH_INVARIANT=${VLLM_BATCH_INVARIANT:-1}
VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.88}
LLM_PORT=${LLM_PORT:-9000}
KEEP_VLLM=${KEEP_VLLM:-0}
INJECT_BACKEND_ID=${INJECT_BACKEND_ID:-0}
HYBRID_UPSTREAM_TOPK=${HYBRID_UPSTREAM_TOPK:-${UPSTREAM_TOPK:-100}}
HYBRID_RRF_CONSTANT=${HYBRID_RRF_CONSTANT:-${RRF_CONSTANT:-60}}

[[ -n "$MODEL_REF" ]] || { echo "Set MODEL_REF or MODEL_PATH" >&2; exit 2; }
[[ "$INJECT_BACKEND_ID" == 0 || "$INJECT_BACKEND_ID" == 1 ]] || {
  echo "INJECT_BACKEND_ID must be 0 or 1" >&2; exit 2;
}
[[ "$KEEP_VLLM" == 0 || "$KEEP_VLLM" == 1 ]] || {
  echo "KEEP_VLLM must be 0 or 1" >&2; exit 2;
}
[[ "$VLLM_BATCH_INVARIANT" == 0 || "$VLLM_BATCH_INVARIANT" == 1 ]] || {
  echo "VLLM_BATCH_INVARIANT must be 0 or 1" >&2; exit 2;
}
if [[ "$VLLM_BATCH_INVARIANT" == 1 ]]; then
  case "$VLLM_ATTENTION_BACKEND" in
    FLASH_ATTN|TRITON_ATTN) ;;
    *)
      echo "Batch-invariant Qwen evaluation requires VLLM_ATTENTION_BACKEND=FLASH_ATTN or TRITON_ATTN; got '$VLLM_ATTENTION_BACKEND'." >&2
      exit 2
      ;;
  esac
fi
for value_name in TP DP VLLM_API_SERVER_COUNT NUMBERED_EVAL_WORKERS; do
  value=${!value_name}
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
    echo "$value_name must be a positive integer; got '$value'." >&2
    exit 2
  }
done
validate_gpu_list "$LLM_GPUS" "$((TP * DP))" \
  "numbered-policy vLLM TP=$TP x DP=$DP"
validate_gpu_list "$E5_GPU" 1 "numbered-policy E5 retrieval"
if [[ ",$LLM_GPUS," == *",$E5_GPU,"* ]]; then
  echo "E5_GPU=$E5_GPU overlaps LLM_GPUS=$LLM_GPUS." >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required to validate the numbered-evaluation GPU layout." >&2
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
PILOT_PYTHON=$ROOT/.venv-pilot/bin/python
[[ -x "$PILOT_PYTHON" ]] || { echo "Run scripts/bootstrap.sh first" >&2; exit 1; }
RUN_ID=${RUN_ID:-$(
  "$PILOT_PYTHON" -m stackpilot.experiment_registry run-id "$EXPERIMENT_ID" \
    --seed "$SEED" --profile "$PROFILE" --variant "$VARIANT"
)}
OUTPUT_DIR=$ROOT/work/experiments/$EXPERIMENT_ID/results/$RUN_ID
mkdir -p "$OUTPUT_DIR"

export HARD_ASSET_ROOT BM25_PORT E5_PORT HYBRID_PORT E5_GPU
export LLM_GPUS TP DP VLLM_API_SERVER_COUNT
bash "$ROOT/experiments/ensure_retrievers.sh"
if [[ " $BACKENDS " == *" hybrid "* ]]; then
  UPSTREAM_TOPK="$HYBRID_UPSTREAM_TOPK" \
    RRF_CONSTANT="$HYBRID_RRF_CONSTANT" \
    bash "$ROOT/experiments/launch_hybrid_rrf.sh"
fi

MODEL_REF=$(unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE; \
  bash "$ROOT/scripts/resolve_hf_model.sh" "$MODEL_REF" "$MODEL_REVISION" "$PILOT_PYTHON")
export MODEL_PATH=$MODEL_REF
export MODEL_REVISION
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-numbered-policy}
export HARD_ASSET_ROOT BM25_PORT E5_PORT HYBRID_PORT E5_GPU
export LLM_GPUS TP DP VLLM_API_SERVER_COUNT NUMBERED_EVAL_WORKERS
export VLLM_BATCH_INVARIANT VLLM_ATTENTION_BACKEND \
  GPU_MEMORY_UTILIZATION LLM_PORT
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}

bash "$ROOT/scripts/stop_servers.sh" || true
bash "$ROOT/scripts/launch_vllm_bg.sh"

cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [[ "$KEEP_VLLM" != 1 ]]; then
    bash "$ROOT/scripts/stop_servers.sh" >/dev/null 2>&1 || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

limit_args=()
if [[ -n "$LIMIT" ]]; then limit_args=(--limit "$LIMIT"); fi
backend_id_args=()
if [[ "$INJECT_BACKEND_ID" == 1 ]]; then backend_id_args=(--inject-backend-id); fi
read -r -a backend_args <<< "$BACKENDS"
read -r -a topk_args <<< "$TOPKS"

"$PILOT_PYTHON" -m stackpilot.numbered_policy_eval \
  --config configs/hard_rq0.yaml \
  --data-file "$DATA_FILE" --output-dir "$OUTPUT_DIR" \
  --experiment-id "$EXPERIMENT_ID" --run-id "$RUN_ID" \
  --tag "$TAG" --profile "$PROFILE" --variant "$VARIANT" --seed "$SEED" \
  --api-base "http://127.0.0.1:${LLM_PORT}/v1" --model "$SERVED_MODEL_NAME" \
  --workers "$NUMBERED_EVAL_WORKERS" \
  --bm25-port "$BM25_PORT" --e5-port "$E5_PORT" --hybrid-port "$HYBRID_PORT" \
  --hybrid-upstream-topk "$HYBRID_UPSTREAM_TOPK" \
  --hybrid-rrf-constant "$HYBRID_RRF_CONSTANT" \
  --backends "${backend_args[@]}" --topks "${topk_args[@]}" \
  "${backend_id_args[@]}" "${limit_args[@]}"

echo "Numbered results: $OUTPUT_DIR"
