#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PILOT_PYTHON=$ROOT/.venv-pilot/bin/python
[[ -x "$PILOT_PYTHON" ]] || { echo "Run scripts/bootstrap.sh first." >&2; exit 1; }
[[ -x "$ROOT/.venv-vllm/bin/vllm" ]] || { echo "Run scripts/bootstrap_vllm.sh first." >&2; exit 1; }

export STACKPILOT_RUNTIME_ROOT=${STACKPILOT_RUNTIME_ROOT:-$ROOT/work/causal_query_audit/runtime}
export STACKPILOT_LOG_ROOT=${STACKPILOT_LOG_ROOT:-$ROOT/logs/causal_query_audit}
export HARD_ASSET_ROOT=${HARD_ASSET_ROOT:-$ROOT/work/hard_rq0/assets/wiki18}
export BM25_PORT=${BM25_PORT:-8101}
export E5_PORT=${E5_PORT:-8102}
export E5_GPU=${E5_GPU:-7}
export LLM_PORT=${LLM_PORT:-9000}
export LLM_GPUS=${LLM_GPUS:-0,1,2,3,4,5,6}
export TP=${TP:-1}
export DP=${DP:-7}
export VLLM_API_SERVER_COUNT=${VLLM_API_SERVER_COUNT:-7}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.88}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
export VLLM_BATCH_INVARIANT=${VLLM_BATCH_INVARIANT:-1}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}
MODEL_SOURCE=${CAUSAL_QUERY_BASE_MODEL:-${TRACE_BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}}
MODEL_REVISION=${CAUSAL_QUERY_MODEL_REVISION:-a09a35458c702b33eeacc393d103063234e8bc28}
MODEL_PATH=$(unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE; \
  bash "$ROOT/scripts/resolve_hf_model.sh" "$MODEL_SOURCE" "$MODEL_REVISION" "$PILOT_PYTHON")
export MODEL_PATH MODEL_REVISION

mkdir -p "$STACKPILOT_RUNTIME_ROOT" "$STACKPILOT_LOG_ROOT"
bash "$ROOT/hard_rq0/launch_retrievers.sh"
bash "$ROOT/scripts/launch_vllm_bg.sh"

echo "Causal-query services ready."
echo "MODEL_PATH=$MODEL_PATH"
echo "BM25=http://127.0.0.1:${BM25_PORT} E5=http://127.0.0.1:${E5_PORT} vLLM=http://127.0.0.1:${LLM_PORT}/v1"
