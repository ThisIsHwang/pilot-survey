#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
DEFAULT_STAGE0_MODEL_REF=Qwen/Qwen2.5-7B-Instruct
DEFAULT_STAGE0_MODEL_REVISION=a09a35458c702b33eeacc393d103063234e8bc28
STAGE0_MODEL_REF=${STAGE0_MODEL_REF:-${MODEL_PATH:-${MODEL:-$DEFAULT_STAGE0_MODEL_REF}}}
if [[ -z ${STAGE0_MODEL_REVISION:-} ]]; then
  if [[ -n ${MODEL_REVISION:-} ]]; then
    STAGE0_MODEL_REVISION=$MODEL_REVISION
  elif [[ "$STAGE0_MODEL_REF" == "$DEFAULT_STAGE0_MODEL_REF" ]]; then
    STAGE0_MODEL_REVISION=$DEFAULT_STAGE0_MODEL_REVISION
  else
    STAGE0_MODEL_REVISION=main
  fi
fi
export HF_HOME=${HF_HOME:-$ROOT/.cache/huggingface}

RUN_STAGE0=${RUN_STAGE0:-1}
RUN_HARD_RQ0=${RUN_HARD_RQ0:-1}
for flag_name in RUN_STAGE0 RUN_HARD_RQ0; do
  flag_value=${!flag_name}
  if [[ "$flag_value" != 0 && "$flag_value" != 1 ]]; then
    echo "$flag_name must be 0 or 1; got '$flag_value'." >&2
    exit 2
  fi
done
if [[ -z "$STAGE0_MODEL_REF" ]]; then
  echo "STAGE0_MODEL_REF must not be empty." >&2
  exit 2
fi

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  "$ROOT/.venv-searchr1/bin/ray" stop --force >/dev/null 2>&1 || true
  bash "$ROOT/hard_rq0/stop_retrievers.sh" || true
  bash "$ROOT/scripts/stop_servers.sh" || true
  exit "$status"
}
trap cleanup EXIT INT TERM

# An interrupted run can leave processes importing these virtual environments
# or holding GPUs. Stop them before any bootstrap recreates an environment.
"$ROOT/.venv-searchr1/bin/ray" stop --force >/dev/null 2>&1 || true
bash "$ROOT/hard_rq0/stop_retrievers.sh" || true
bash "$ROOT/scripts/stop_servers.sh" || true

if [[ ${SKIP_BOOTSTRAP:-0} != 1 ]]; then
  bash "$ROOT/scripts/bootstrap.sh"
  bash "$ROOT/scripts/bootstrap_vllm.sh"
  bash "$ROOT/scripts/bootstrap_searchr1.sh"
fi

if [[ "$RUN_STAGE0" == 1 ]]; then
  (
    unset MODEL MODEL_PATH MODEL_LOCAL_ONLY MODEL_REVISION HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
    if [[ -d "$STAGE0_MODEL_REF" ]]; then
      export MODEL_PATH=$STAGE0_MODEL_REF
    else
      export MODEL=$STAGE0_MODEL_REF
    fi
    export MODEL_REVISION=$STAGE0_MODEL_REVISION
    export SERVED_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
    SKIP_BOOTSTRAP=1 bash "$ROOT/scripts/run_all.sh"
  )
fi

(
  unset MODEL MODEL_PATH MODEL_LOCAL_ONLY MODEL_REVISION SERVED_MODEL_NAME HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
  SKIP_BOOTSTRAP=1 bash "$ROOT/searchr1_stage2/run_all.sh"
)
if [[ "$RUN_HARD_RQ0" == 1 ]]; then
  (
    unset MODEL MODEL_PATH MODEL_LOCAL_ONLY MODEL_REVISION SERVED_MODEL_NAME \
      BASE_MODEL_REF BASE_MODEL_PATH BASE_MODEL_REVISION \
      HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
    hard_env=("SKIP_BOOTSTRAP=1")
    if [[ -n ${HARD_BASE_MODEL_REF:-} ]]; then
      hard_env+=("BASE_MODEL_REF=$HARD_BASE_MODEL_REF")
      if [[ -n ${HARD_BASE_MODEL_REVISION:-} ]]; then
        hard_env+=("BASE_MODEL_REVISION=$HARD_BASE_MODEL_REVISION")
      fi
    elif [[ -n ${TRAIN_BASE_MODEL:-} ]]; then
      hard_env+=("BASE_MODEL_REF=$TRAIN_BASE_MODEL")
      if [[ -n ${HARD_BASE_MODEL_REVISION:-} ]]; then
        hard_env+=("BASE_MODEL_REVISION=$HARD_BASE_MODEL_REVISION")
      elif [[ -n ${TRAIN_BASE_MODEL_REVISION:-} ]]; then
        hard_env+=("BASE_MODEL_REVISION=$TRAIN_BASE_MODEL_REVISION")
      fi
    elif [[ -n ${HARD_BASE_MODEL_REVISION:-} ]]; then
      hard_env+=("BASE_MODEL_REVISION=$HARD_BASE_MODEL_REVISION")
    fi
    if [[ ${SMOKE_ONLY:-0} == 1 ]]; then
      hard_env+=(
        "PROFILE=${HARD_PROFILE:-smoke}"
        "RESULT_SET=${HARD_RESULT_SET:-smoke}"
        "SEEDS=${HARD_SEEDS:-13}"
        "LIMIT=${HARD_LIMIT:-20}"
        "RUN_REPORT=0"
      )
    else
      hard_env+=(
        "PROFILE=${HARD_PROFILE:-pilot}"
        "RESULT_SET=${HARD_RESULT_SET:-pilot}"
        "SEEDS=${HARD_SEEDS:-13 42 87}"
      )
      if [[ -n ${HARD_LIMIT:-} ]]; then hard_env+=("LIMIT=$HARD_LIMIT"); fi
    fi
    env "${hard_env[@]}" bash "$ROOT/hard_rq0/run_all.sh"
  )
fi

completed="Stage-2"
if [[ "$RUN_STAGE0" == 1 ]]; then completed="Stage-0 + $completed"; fi
if [[ "$RUN_HARD_RQ0" == 1 ]]; then completed="$completed + hard-RQ0"; fi
echo "$completed pipeline completed."
