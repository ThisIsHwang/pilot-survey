#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
if [[ ${SKIP_BOOTSTRAP:-0} != 1 ]]; then
  bash trace_go/bootstrap.sh
fi
if [[ ${SKIP_PREPARE:-0} != 1 ]]; then
  PROFILE="$PROFILE" bash query_attribution/prepare.sh
fi
if [[ ${SKIP_PLAN:-0} != 1 ]]; then
  PROFILE="$PROFILE" bash query_attribution/plan.sh
fi
PROFILE="$PROFILE" bash query_attribution/run_jobs.sh
PROFILE="$PROFILE" bash query_attribution/report.sh
cat "work/query_attribution/reports/$PROFILE/EXP016_REPORT.md"
