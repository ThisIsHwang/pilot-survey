#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PYTHON=$ROOT/.venv-pilot/bin/python
[[ -x "$PYTHON" ]] || { echo "Run scripts/bootstrap.sh first." >&2; exit 1; }
CONFIG=${BEHAVIOR_ALIAS_CONFIG:-configs/behavior_alias_pilot.yaml}
PROFILE=${PROFILE:-pilot}
ARGS=(--config "$CONFIG" --profile "$PROFILE")
if [[ -n ${BEHAVIOR_ALIAS_STATES_FILE:-} ]]; then
  ARGS+=(--states-file "$BEHAVIOR_ALIAS_STATES_FILE")
fi
"$PYTHON" -m stackpilot.behavior_alias_prepare "${ARGS[@]}" "$@"
