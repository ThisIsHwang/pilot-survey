#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
PROFILE="$PROFILE" bash query_credit/run_labels.sh
PROFILE="$PROFILE" bash query_credit/run_report.sh
PROFILE="$PROFILE" bash query_credit/run_estimator.sh
PROFILE="$PROFILE" bash query_credit/run_gradient.sh
PROFILE="$PROFILE" bash query_credit/run_micro_update.sh
if [[ ${SKIP_GRPO:-0} != 1 ]]; then
  PROFILE="$PROFILE" bash query_credit/run_factorial.sh
fi
if [[ ${SKIP_ENDPOINT:-0} != 1 ]]; then
  PROFILE="$PROFILE" bash query_credit/merge_eval.sh
fi
