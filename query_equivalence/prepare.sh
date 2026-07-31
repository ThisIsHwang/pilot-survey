#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PYTHON=$ROOT/.venv-trace/bin/python
[[ -x "$PYTHON" ]] || { echo "Run query_equivalence/bootstrap.sh first." >&2; exit 1; }
PROFILE=${PROFILE:-pilot}
ARGS=()
if [[ -n ${QUERY_EQUIVALENCE_INPUTS:-} ]]; then
  IFS=':' read -r -a INPUTS <<< "$QUERY_EQUIVALENCE_INPUTS"
  ARGS+=(--inputs "${INPUTS[@]}")
fi
"$PYTHON" -m stackpilot.query_equivalence_prepare \
  --config configs/query_equivalence.yaml \
  --profile "$PROFILE" \
  "${ARGS[@]}"
