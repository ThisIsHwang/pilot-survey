#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
source "$ROOT/scripts/lib/runtime.sh"

TAG=${TAG:?Set TAG, e.g. base-qwen or bm25-specialist}
MODEL_PATH=${MODEL_PATH:?Set MODEL_PATH to a local or Hugging Face model path}
LIMIT=${LIMIT:-200}
BACKENDS=${BACKENDS:-"bm25 e5"}
VARIANTS=${VARIANTS:-blind}

export MODEL_PATH
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}
export LLM_GPUS=${LLM_GPUS:-0,1,2,3}
export TP=${TP:-4}

bash "$ROOT/scripts/stop_servers.sh" || true
bash "$ROOT/scripts/launch_retrievers.sh"
bash "$ROOT/scripts/launch_vllm_bg.sh"

# shellcheck disable=SC2086
"$ROOT/.venv-pilot/bin/python" -m stackpilot.policy_eval \
  --config configs/pilot.yaml \
  --tag "$TAG" \
  --limit "$LIMIT" \
  --backends $BACKENDS \
  --variants $VARIANTS

if [[ ${KEEP_SERVERS:-0} != 1 ]]; then
  bash "$ROOT/scripts/stop_servers.sh"
fi
