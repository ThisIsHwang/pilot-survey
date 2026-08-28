#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
source "$ROOT/scripts/lib/runtime.sh"
source "$ROOT/scripts/lib/bootstrap_java.sh"
ensure_local_no_proxy

PILOT_PYTHON=$ROOT/.venv-pilot/bin/python
[[ -x "$PILOT_PYTHON" ]] || { echo "Run scripts/bootstrap.sh first." >&2; exit 1; }
[[ -x "$ROOT/.venv-vllm/bin/vllm" ]] || { echo "Run scripts/bootstrap_vllm.sh first." >&2; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi is required." >&2; exit 1; }
GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')

export STACKPILOT_RUNTIME_ROOT=${STACKPILOT_RUNTIME_ROOT:-$ROOT/work/query_credit_weekend/runtime}
export STACKPILOT_LOG_ROOT=${STACKPILOT_LOG_ROOT:-$ROOT/logs/query_credit_weekend}
export HARD_ASSET_ROOT=${HARD_ASSET_ROOT:-$ROOT/work/hard_rq0/assets/wiki18}
export BM25_PORT=${BM25_PORT:-8101}
export E5_PORT=${E5_PORT:-8102}
export LLM_PORT=${LLM_PORT:-9000}
mkdir -p "$STACKPILOT_RUNTIME_ROOT" "$STACKPILOT_LOG_ROOT"

MODEL_SOURCE=${BASE_MODEL:-${CAUSAL_QUERY_BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}}
MODEL_REVISION=${MODEL_REVISION:-a09a35458c702b33eeacc393d103063234e8bc28}
MODEL_PATH=$(unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE; \
  bash "$ROOT/scripts/resolve_hf_model.sh" "$MODEL_SOURCE" "$MODEL_REVISION" "$PILOT_PYTHON")
export MODEL_PATH BASE_MODEL=$MODEL_PATH MODEL_REVISION
printf '%s\n' "$MODEL_PATH" > "$STACKPILOT_RUNTIME_ROOT/model_path"

if (( GPU_COUNT >= 8 )); then
  export E5_GPU=${E5_GPU:-7}
  export TRAIN_GPUS=${TRAIN_GPUS:-0,1,2,3,4,5,6}
  export N_GPUS=${N_GPUS:-7}
  export LLM_GPUS=${LLM_GPUS:-0,1,2,3,4,5,6}
  export TP=${TP:-1}
  export DP=${DP:-7}
  export VLLM_API_SERVER_COUNT=${VLLM_API_SERVER_COUNT:-7}
  export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.88}
  export MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
  export VLLM_BATCH_INVARIANT=${VLLM_BATCH_INVARIANT:-1}
  export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
  export CAUSAL_QUERY_BASE_MODEL=$MODEL_PATH
  bash "$ROOT/causal_query_audit/launch_services.sh"
  printf 'node8\n' > "$STACKPILOT_RUNTIME_ROOT/hardware_profile"
  echo "Weekend services ready in node8 mode: 7 GPUs for vLLM, GPU 7 for E5."
  exit 0
fi

if (( GPU_COUNT < 1 )); then
  echo "At least one H100-class CUDA GPU is required." >&2
  exit 1
fi

echo "Only $GPU_COUNT GPU detected; using the BM25-only single-GPU fallback."
SEARCH_R1=${SEARCH_R1_ROOT:-$ROOT/upstream/Search-R1}
[[ -e "$SEARCH_R1/.git" ]] || { echo "Missing pinned Search-R1 checkout." >&2; exit 1; }
ensure_java "$ROOT"
ASSET_ROOT=$HARD_ASSET_ROOT
CORPUS_PATH=$ASSET_ROOT/wiki-18.jsonl
BM25_INDEX=$ASSET_ROOT/bm25
"$PILOT_PYTHON" -m stackpilot.hard_assets check --root "$ASSET_ROOT" >/dev/null
EXPECTED_DOCUMENTS=$("$PILOT_PYTHON" -c 'from stackpilot.hard_assets import EXPECTED_DOCUMENTS; print(EXPECTED_DOCUMENTS)')
PID_ROOT=$STACKPILOT_RUNTIME_ROOT/hard_rq0/pids
LOG_ROOT=$STACKPILOT_LOG_ROOT/hard_rq0
mkdir -p "$PID_ROOT" "$LOG_ROOT"
BM25_PID_FILE=$PID_ROOT/bm25.pid
stop_managed_pid "$BM25_PID_FILE" "stackpilot.searchr1_server" "$ROOT" 1 || true
require_free_port "$PILOT_PYTHON" "$BM25_PORT"
BM25_LOG=$LOG_ROOT/bm25.log
BM25_PID=$(CUDA_VISIBLE_DEVICES='' start_managed_process \
  "$PILOT_PYTHON" "$BM25_LOG" "$PILOT_PYTHON" -m stackpilot.searchr1_server \
  --search-r1-root "$SEARCH_R1" \
  --index-path "$BM25_INDEX" \
  --corpus-path "$CORPUS_PATH" \
  --retriever-name bm25 --topk 10 --port "$BM25_PORT" \
  --expected-documents "$EXPECTED_DOCUMENTS")
echo "$BM25_PID" > "$BM25_PID_FILE"
wait_for_http "$BM25_PID" "http://127.0.0.1:${BM25_PORT}/health" \
  "${HARD_RETRIEVER_READY_TIMEOUT:-14400}" "$BM25_LOG"

export LLM_GPUS=0 TP=1 DP=1 VLLM_API_SERVER_COUNT=1
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.88}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
export VLLM_BATCH_INVARIANT=${VLLM_BATCH_INVARIANT:-1}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}
bash "$ROOT/scripts/launch_vllm_bg.sh"
printf 'single\n' > "$STACKPILOT_RUNTIME_ROOT/hardware_profile"
echo "Weekend services ready in single-GPU BM25 mode."
echo "BASE_MODEL=$BASE_MODEL"
