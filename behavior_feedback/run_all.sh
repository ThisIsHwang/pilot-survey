#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
PROFILE="$PROFILE" bash behavior_feedback/run_alias_lagged.sh
PROFILE="$PROFILE" bash behavior_feedback/run_feedback_audit.sh
PROFILE="$PROFILE" bash behavior_feedback/run_router.sh
PROFILE="$PROFILE" bash behavior_feedback/run_paired_grid.sh
PROFILE="$PROFILE" bash behavior_feedback/run_ctu_closure.sh
if [[ ${SKIP_GRPO:-0} != 1 ]]; then
  PROFILE="$PROFILE" bash behavior_feedback/run_factorial.sh
  PROFILE="$PROFILE" bash behavior_feedback/merge_eval.sh
fi
PROFILE="$PROFILE" bash behavior_feedback/report.sh
