#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

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
LLM_GPUS=${LLM_GPUS:-0,1}
TP=${TP:-2}
LLM_PORT=${LLM_PORT:-9000}
KEEP_VLLM=${KEEP_VLLM:-0}

[[ -n "$MODEL_REF" ]] || { echo "Set MODEL_REF or MODEL_PATH" >&2; exit 2; }
PILOT_PYTHON=$ROOT/.venv-pilot/bin/python
[[ -x "$PILOT_PYTHON" ]] || { echo "Run scripts/bootstrap.sh first" >&2; exit 1; }
RUN_ID=${RUN_ID:-$(
  "$PILOT_PYTHON" -m stackpilot.experiment_registry run-id "$EXPERIMENT_ID" \
    --seed "$SEED" --profile "$PROFILE" --variant "$VARIANT"
)}
OUTPUT_DIR=$ROOT/work/experiments/$EXPERIMENT_ID/results/$RUN_ID
mkdir -p "$OUTPUT_DIR"

bash "$ROOT/hard_rq0/launch_retrievers.sh"
if [[ " $BACKENDS " == *" hybrid "* ]]; then
  bash "$ROOT/experiments/launch_hybrid_rrf.sh"
fi

MODEL_REF=$(unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE; \
  bash "$ROOT/scripts/resolve_hf_model.sh" "$MODEL_REF" "$MODEL_REVISION" "$PILOT_PYTHON")
export MODEL_PATH=$MODEL_REF
export MODEL_REVISION
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-numbered-policy}
export LLM_GPUS TP LLM_PORT
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
read -r -a backend_args <<< "$BACKENDS"
read -r -a topk_args <<< "$TOPKS"

"$PILOT_PYTHON" -m stackpilot.numbered_policy_eval \
  --config configs/hard_rq0.yaml \
  --data-file "$DATA_FILE" --output-dir "$OUTPUT_DIR" \
  --experiment-id "$EXPERIMENT_ID" --run-id "$RUN_ID" \
  --tag "$TAG" --seed "$SEED" \
  --api-base "http://127.0.0.1:${LLM_PORT}/v1" --model "$SERVED_MODEL_NAME" \
  --backends "${backend_args[@]}" --topks "${topk_args[@]}" \
  "${limit_args[@]}"

echo "Numbered results: $OUTPUT_DIR"
