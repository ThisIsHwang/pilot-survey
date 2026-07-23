#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
source "$ROOT/scripts/lib/runtime.sh"
source "$ROOT/scripts/lib/vllm_launch.sh"
ensure_local_no_proxy
configure_vllm_launch "$ROOT"
require_free_port "$VLLM_PYTHON" "$LLM_PORT"

echo "Serving $MODEL_PATH as $SERVED_MODEL_NAME on GPUs $LLM_GPUS (TP=$TP, DP=$DP)."
exec env CUDA_VISIBLE_DEVICES="$LLM_GPUS" "$VLLM_BIN" "${VLLM_ARGS[@]}"
