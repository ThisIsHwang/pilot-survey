#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
BASE_MODEL_PATH=${BASE_MODEL_PATH:?Set BASE_MODEL_PATH to the base Qwen2.5-3B-Instruct checkpoint}
PROFILE=${PROFILE:-pilot}
RESULT_SET=${RESULT_SET:-$PROFILE}
LIMIT=${LIMIT:-}
RUN_REPORT=${RUN_REPORT:-1}

if [[ ${SKIP_ASSETS:-0} != 1 ]]; then
  bash hard_rq0/download_assets.sh
fi
if [[ ${SKIP_DATA:-0} != 1 ]]; then
  bash hard_rq0/prepare_data.sh
fi
bash hard_rq0/launch_retrievers.sh

base_eval=(
  TAG=base-qwen
  SEED=0
  MODEL_PATH=$BASE_MODEL_PATH
  RESULT_SET=$RESULT_SET
)
if [[ -n "$LIMIT" ]]; then
  base_eval+=(LIMIT=$LIMIT)
fi
env "${base_eval[@]}" bash hard_rq0/eval_policy.sh

PROFILE=$PROFILE RESULT_SET=$RESULT_SET BASE_MODEL=$BASE_MODEL_PATH LIMIT=$LIMIT \
  bash hard_rq0/run_three_seed_specialists.sh

if [[ "$RUN_REPORT" == 1 ]]; then
  RESULT_SET=$RESULT_SET bash hard_rq0/make_report.sh
else
  echo "RUN_REPORT=0: skipped the three-seed interaction report."
fi
bash hard_rq0/stop_retrievers.sh
