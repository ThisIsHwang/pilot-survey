#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PYTHON=$ROOT/.venv-pilot/bin/python
[[ -x "$PYTHON" ]] || { echo "Run scripts/bootstrap.sh first." >&2; exit 1; }
CONFIG=${BEHAVIOR_ALIAS_CONFIG:-configs/behavior_alias_pilot.yaml}
PROFILE=${PROFILE:-pilot}
MODEL_SOURCE=${BEHAVIOR_ALIAS_BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}
MODEL_REVISION=${BEHAVIOR_ALIAS_MODEL_REVISION:-a09a35458c702b33eeacc393d103063234e8bc28}
MODEL_PATH=$(unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE; \
  bash "$ROOT/scripts/resolve_hf_model.sh" "$MODEL_SOURCE" "$MODEL_REVISION" "$PYTHON")
"$PYTHON" -m stackpilot.behavior_alias_run \
  --config "$CONFIG" \
  --profile "$PROFILE" \
  --model "$MODEL_PATH" \
  "$@"
