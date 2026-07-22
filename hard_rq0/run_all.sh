#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
BASE_MODEL_PATH=${BASE_MODEL_PATH:?Set BASE_MODEL_PATH to the base Qwen2.5-3B-Instruct checkpoint}
PROFILE=${PROFILE:-pilot}
LIMIT=${LIMIT:-}

if [[ ${SKIP_ASSETS:-0} != 1 ]]; then
  bash hard_rq0/download_assets.sh
fi
if [[ ${SKIP_DATA:-0} != 1 ]]; then
  bash hard_rq0/prepare_data.sh
fi
bash hard_rq0/launch_retrievers.sh

base_eval=(TAG=base-qwen SEED=0 MODEL_PATH=$BASE_MODEL_PATH)
if [[ -n "$LIMIT" ]]; then
  base_eval+=(LIMIT=$LIMIT)
fi
env "${base_eval[@]}" bash hard_rq0/eval_policy.sh

PROFILE=$PROFILE BASE_MODEL=$BASE_MODEL_PATH LIMIT=$LIMIT \
  bash hard_rq0/run_three_seed_specialists.sh

bash hard_rq0/make_report.sh
bash hard_rq0/stop_retrievers.sh
