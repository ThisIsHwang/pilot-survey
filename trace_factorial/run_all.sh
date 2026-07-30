#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
if [[ ${SKIP_BOOTSTRAP:-0} != 1 ]]; then
  bash trace_go/bootstrap.sh
fi
if [[ ${SKIP_BANK:-0} != 1 ]]; then
  bash trace_go/prepare_bank.sh
fi
PROFILE="$PROFILE" bash trace_factorial/plan.sh
PROFILE="$PROFILE" bash trace_factorial/run.sh
PROFILE="$PROFILE" bash trace_factorial/report.sh
