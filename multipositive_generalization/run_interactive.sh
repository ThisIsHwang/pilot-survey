#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
PROFILE="$PROFILE" bash multipositive_generalization/interactive_plan.sh
PROFILE="$PROFILE" bash multipositive_generalization/interactive_run.sh "$@"
PROFILE="$PROFILE" bash multipositive_generalization/interactive_report.sh
cat "work/multipositive_generalization/interactive_reports/$PROFILE/EXP027_REPORT.md"
