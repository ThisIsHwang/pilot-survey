#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PYTHON=$ROOT/.venv-trace/bin/python
[[ -x "$PYTHON" ]] || { echo "Run trace_go/bootstrap.sh first." >&2; exit 1; }
PROFILE=${PROFILE:-pilot}
CONFIG=${TRACE_FACTORIAL_CONFIG:-configs/trace_factorial.yaml}
BASE_MODEL=${TRACE_BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}
"$PYTHON" -m stackpilot.trace_factorial plan \
  --config "$CONFIG" \
  --profile "$PROFILE" \
  --base-model "$BASE_MODEL" \
  "$@"
