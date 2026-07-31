#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PYTHON=$ROOT/.venv-trace/bin/python
[[ -x "$PYTHON" ]] || { echo "Run query_equivalence/bootstrap.sh first." >&2; exit 1; }
PROFILE=${PROFILE:-pilot}
ARGS=()
if [[ -n ${QUERY_EQUIVALENCE_BASE_MODEL:-} ]]; then
  ARGS+=(--base-model "$QUERY_EQUIVALENCE_BASE_MODEL")
fi
"$PYTHON" -m stackpilot.query_equivalence_plan \
  --config configs/query_equivalence.yaml \
  --profile "$PROFILE" \
  "${ARGS[@]}"
