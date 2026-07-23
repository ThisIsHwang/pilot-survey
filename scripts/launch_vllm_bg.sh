#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
source "$ROOT/scripts/lib/runtime.sh"
source "$ROOT/scripts/lib/vllm_launch.sh"
ensure_local_no_proxy
configure_vllm_launch "$ROOT"
RUNTIME_WORK_ROOT=${STACKPILOT_RUNTIME_ROOT:-$ROOT/work}
RUNTIME_LOG_ROOT=${STACKPILOT_LOG_ROOT:-$ROOT/logs}
mkdir -p "$RUNTIME_LOG_ROOT" "$RUNTIME_WORK_ROOT/pids"

PID_FILE=$RUNTIME_WORK_ROOT/pids/vllm.pid
LOG_FILE=$RUNTIME_LOG_ROOT/vllm.log
stop_managed_pid "$PID_FILE" "$ROOT/.venv-vllm/bin/vllm" "$ROOT" 1
require_free_port "$VLLM_PYTHON" "$LLM_PORT"

VLLM_PID=$(CUDA_VISIBLE_DEVICES=$LLM_GPUS start_managed_process \
  "$VLLM_PYTHON" "$LOG_FILE" "$VLLM_BIN" "${VLLM_ARGS[@]}")
echo "$VLLM_PID" > "$PID_FILE"

cleanup_vllm() {
  local status=$?
  trap - ERR INT TERM
  stop_managed_pid "$PID_FILE" "$ROOT/.venv-vllm/bin/vllm" "$ROOT" 1 || true
  exit "$status"
}
trap cleanup_vllm ERR INT TERM

if [[ $VLLM_MODEL_IS_LOCAL -eq 1 ]]; then
  READY_TIMEOUT_DEFAULT=900
  MODEL_SOURCE="local filesystem"
else
  READY_TIMEOUT_DEFAULT=14400
  MODEL_SOURCE="Hugging Face; cache=${HF_HOME:-$HOME/.cache/huggingface}"
fi
READY_TIMEOUT=${VLLM_READY_TIMEOUT:-$READY_TIMEOUT_DEFAULT}
echo "Loading $MODEL_PATH as $SERVED_MODEL_NAME on GPUs $LLM_GPUS (TP=$TP, DP=$DP)."
echo "Model source: $MODEL_SOURCE; readiness timeout: ${READY_TIMEOUT}s."
wait_for_http "$VLLM_PID" "http://127.0.0.1:${LLM_PORT}/v1/models" \
  "$READY_TIMEOUT" "$LOG_FILE"

models_response=$(curl --noproxy '*' -fsS --connect-timeout 3 --max-time 30 \
  "http://127.0.0.1:${LLM_PORT}/v1/models")
"$VLLM_PYTHON" -c \
  'import json,sys; ids={str(x.get("id")) for x in json.loads(sys.argv[1]).get("data", [])}; expected=sys.argv[2]; assert expected in ids, (expected, sorted(ids))' \
  "$models_response" "$SERVED_MODEL_NAME"

# Exercise one generation so model loading, NCCL, kernels, and the API model
# name are all verified before a long evaluation begins.
probe_payload=$("$VLLM_PYTHON" -c \
  'import json,sys; print(json.dumps({"model":sys.argv[1],"messages":[{"role":"user","content":"Reply with OK"}],"max_tokens":8,"temperature":0}))' \
  "$SERVED_MODEL_NAME")
response=$(curl --noproxy '*' -fsS --connect-timeout 3 \
  --max-time "${VLLM_PROBE_TIMEOUT:-300}" \
  -X POST "http://127.0.0.1:${LLM_PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "$probe_payload")
"$VLLM_PYTHON" -c \
  'import json,sys; p=json.loads(sys.argv[1]); assert p.get("choices"), p' "$response"

trap - ERR INT TERM
echo "vLLM ready on port $LLM_PORT (PID $VLLM_PID)."
