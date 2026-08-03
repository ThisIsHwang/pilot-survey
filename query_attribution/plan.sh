#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PYTHON=${TRACE_PYTHON:-$ROOT/.venv-trace/bin/python}
[[ -x "$PYTHON" ]] || { echo "Run trace_go/bootstrap.sh first." >&2; exit 1; }
PROFILE=${PROFILE:-pilot}
ARGS=()
[[ -n ${QUERY_ATTRIBUTION_BASE_MODEL:-} ]] && ARGS+=(--base-model "$QUERY_ATTRIBUTION_BASE_MODEL")
"$PYTHON" -m stackpilot.query_attribution_plan --config configs/query_attribution.yaml --profile "$PROFILE" "${ARGS[@]}"
