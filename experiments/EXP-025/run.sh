#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
PYTHON=${PILOT_PYTHON:-$ROOT/.venv-pilot/bin/python}
"$PYTHON" -m stackpilot.behavior_quotient_fixed_budget \
  --config configs/behavior_quotient.yaml --profile "$PROFILE"
