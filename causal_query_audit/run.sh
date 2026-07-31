#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PYTHON=$ROOT/.venv-pilot/bin/python
[[ -x "$PYTHON" ]] || { echo "Run scripts/bootstrap.sh first." >&2; exit 1; }
CONFIG=${CAUSAL_QUERY_CONFIG:-configs/causal_query_audit.yaml}
PROFILE=${PROFILE:-pilot}
MODEL_SOURCE=${CAUSAL_QUERY_BASE_MODEL:-${TRACE_BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}}
MODEL_REVISION=${CAUSAL_QUERY_MODEL_REVISION:-a09a35458c702b33eeacc393d103063234e8bc28}
MODEL_PATH=$(unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE; \
  bash "$ROOT/scripts/resolve_hf_model.sh" "$MODEL_SOURCE" "$MODEL_REVISION" "$PYTHON")
"$PYTHON" -m stackpilot.causal_query_model_contract \
  --config "$CONFIG" \
  --model "$MODEL_PATH" \
  --output "$ROOT/work/causal_query_audit/model_contract.json"
"$PYTHON" -m stackpilot.causal_query_replay \
  --config "$CONFIG" \
  --profile "$PROFILE" \
  --model "$MODEL_PATH" \
  "$@"
