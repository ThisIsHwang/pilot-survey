#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
export PROFILE

if [[ ${SKIP_BOOTSTRAP:-0} != 1 ]]; then
  bash trace_go/bootstrap.sh
fi
if [[ ${SKIP_BANK:-0} != 1 ]]; then
  bash trace_go/prepare_bank.sh
fi
if [[ ${SKIP_PLAN:-0} != 1 ]]; then
  bash trace_go/plan.sh
fi
bash trace_go/run_a.sh
bash trace_go/run_b.sh
bash trace_go/run_c.sh
bash trace_go/report.sh
