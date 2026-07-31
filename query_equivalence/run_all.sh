#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
if [[ ${SKIP_BOOTSTRAP:-0} != 1 ]]; then
  bash query_equivalence/bootstrap.sh
fi
if [[ ${SKIP_PREPARE:-0} != 1 ]]; then
  PROFILE="$PROFILE" bash query_equivalence/prepare.sh
fi
if [[ ${SKIP_PLAN:-0} != 1 ]]; then
  PROFILE="$PROFILE" bash query_equivalence/plan.sh
fi
PROFILE="$PROFILE" bash query_equivalence/run_jobs.sh
PROFILE="$PROFILE" bash query_equivalence/report.sh
cat "work/query_equivalence/reports/$PROFILE/EXP015_REPORT.md"
