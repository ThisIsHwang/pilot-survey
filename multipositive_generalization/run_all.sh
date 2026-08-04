#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
[[ ${SKIP_BOOTSTRAP:-0} == 1 ]] || bash trace_go/bootstrap.sh
[[ ${SKIP_PREPARE:-0} == 1 ]] || PROFILE="$PROFILE" bash multipositive_generalization/prepare.sh
[[ ${SKIP_PLAN:-0} == 1 ]] || PROFILE="$PROFILE" bash multipositive_generalization/plan.sh
PROFILE="$PROFILE" bash multipositive_generalization/run_jobs.sh "$@"
PROFILE="$PROFILE" bash multipositive_generalization/report.sh
cat "work/multipositive_generalization/reports/$PROFILE/EXP024_026_REPORT.md"
