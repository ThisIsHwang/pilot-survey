#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PYTHON=${TRACE_PYTHON:-$ROOT/.venv-trace/bin/python}
[[ -x "$PYTHON" ]] || { echo "Run trace_go/bootstrap.sh first." >&2; exit 1; }
PROFILE=${PROFILE:-pilot}
ARGS=()
[[ -n ${MULTIPOSITIVE_BASE_MODEL:-} ]] && ARGS+=(--base-model "$MULTIPOSITIVE_BASE_MODEL")
[[ -n ${MULTIPOSITIVE_EXTERNAL_QUERIES:-} ]] && ARGS+=(--external-queries "$MULTIPOSITIVE_EXTERNAL_QUERIES")
"$PYTHON" -m stackpilot.multipositive_plan --config configs/multipositive_generalization.yaml --profile "$PROFILE" "${ARGS[@]}"
