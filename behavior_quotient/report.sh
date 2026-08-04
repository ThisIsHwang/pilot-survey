#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
PYTHON=${PILOT_PYTHON:-$ROOT/.venv-pilot/bin/python}
[[ -x "$PYTHON" ]] || { echo "Run scripts/bootstrap.sh first" >&2; exit 1; }
"$PYTHON" -m stackpilot.behavior_quotient_telemetry \
  --config configs/behavior_quotient.yaml --profile "$PROFILE"
"$PYTHON" -m stackpilot.behavior_quotient_training_report \
  --config configs/behavior_quotient.yaml --profile "$PROFILE" \
  --experiment-id EXP-027
cat "work/behavior_quotient/reports/$PROFILE/EXP-028/EXP028_REPORT.md"
