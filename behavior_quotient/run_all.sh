#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
if [[ ${SKIP_BOOTSTRAP:-0} != 1 ]]; then
  bash scripts/bootstrap.sh
  bash scripts/bootstrap_vllm.sh
  bash scripts/bootstrap_searchr1.sh
  bash hard_rq0/download_assets.sh
  bash hard_rq0/prepare_data.sh
fi
if [[ ${RUN_OFFLINE:-1} == 1 ]]; then
  PROFILE="$PROFILE" bash behavior_quotient/run_offline.sh
fi
if [[ ${RUN_NATURAL:-1} == 1 ]]; then
  PROFILE="$PROFILE" bash behavior_quotient/run_natural_dynamics.sh
fi
if [[ ${RUN_MATRIX:-1} == 1 ]]; then
  BQ_SETUP_READY=1 PROFILE="$PROFILE" bash behavior_quotient/run_matrix.sh
fi
if [[ ${RUN_EVAL:-1} == 1 ]]; then
  PROFILE="$PROFILE" bash behavior_quotient/merge_eval.sh
fi
if [[ ${RUN_REPORT:-1} == 1 ]]; then
  PROFILE="$PROFILE" bash behavior_quotient/report.sh
fi
