#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PYTHON=${TRACE_PYTHON:-$ROOT/.venv-trace/bin/python}
PROFILE=${PROFILE:-pilot}
MODEL=${QUERY_EQUIVALENCE_BASE_MODEL:-${TRACE_BASE_MODEL:-}}
args=()
[[ -n "$MODEL" ]] && args+=(--base-model "$MODEL")
"$PYTHON" -m stackpilot.query_equivalence_plan \
  --config "${QUERY_EQUIVALENCE_CONFIG:-configs/query_equivalence.yaml}" \
  --profile "$PROFILE" "${args[@]}"
