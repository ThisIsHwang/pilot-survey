#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
bash hard_rq0/launch_retrievers.sh
PROFILE="$PROFILE" bash query_attribution/interactive_plan.sh
PROFILE="$PROFILE" bash query_attribution/interactive_run.sh
PROFILE="$PROFILE" bash query_attribution/interactive_report.sh
cat "work/query_attribution/interactive_reports/$PROFILE/EXP019_REPORT.md"
