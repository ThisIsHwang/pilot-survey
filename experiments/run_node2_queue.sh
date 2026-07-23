#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
RUN_EXP003=${RUN_EXP003:-1}
RUN_EXP004=${RUN_EXP004:-1}
RUN_EXP005=${RUN_EXP005:-0}
RUN_EXP006=${RUN_EXP006:-0}
RUN_REPORT=${RUN_REPORT:-1}

for flag in RUN_EXP003 RUN_EXP004 RUN_EXP005 RUN_EXP006 RUN_REPORT; do
  value=${!flag}
  [[ "$value" == 0 || "$value" == 1 ]] || {
    echo "$flag must be 0 or 1; got $value" >&2
    exit 2
  }
done

if [[ "$RUN_EXP003" == 1 ]]; then
  PROFILE="$PROFILE" SEEDS="${EXP003_SEEDS:-13 42 87}" \
    bash experiments/EXP-003/run.sh
fi
if [[ "$RUN_EXP004" == 1 ]]; then
  PROFILE="$PROFILE" SEEDS="${EXP004_SEEDS:-42}" \
    bash experiments/EXP-004/run.sh
fi
if [[ "$RUN_EXP005" == 1 ]]; then
  PROFILE="$PROFILE" SEEDS="${EXP005_SEEDS:-42}" \
    BACKEND_LIST="${EXP005_BACKENDS:-bm25 e5}" \
    bash experiments/EXP-005/run.sh
fi
if [[ "$RUN_EXP006" == 1 ]]; then
  PROFILE="$PROFILE" SEEDS="${EXP006_SEEDS:-13 42 87}" \
    ORACLE_SEEDS="${EXP006_ORACLE_SEEDS:-42}" \
    bash experiments/EXP-006/run.sh
fi
if [[ "$RUN_REPORT" == 1 ]]; then
  PROFILE="$PROFILE" bash experiments/make_report.sh
fi

echo "Second-node numbered experiment queue complete."
