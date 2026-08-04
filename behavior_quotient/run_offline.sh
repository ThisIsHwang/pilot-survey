#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
if [[ ! -x "$ROOT/.venv-pilot/bin/python" ]]; then
  bash scripts/bootstrap.sh
fi
PYTHON=${PILOT_PYTHON:-$ROOT/.venv-pilot/bin/python}
[[ -x "$PYTHON" ]] || { echo "Run scripts/bootstrap.sh first" >&2; exit 1; }
"$PYTHON" -m stackpilot.behavior_quotient_fixed_budget \
  --config configs/behavior_quotient.yaml --profile "$PROFILE"
"$PYTHON" -m stackpilot.behavior_signature_audit \
  --config configs/behavior_quotient.yaml --profile "$PROFILE"
