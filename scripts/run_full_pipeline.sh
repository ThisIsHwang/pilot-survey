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
if [[ "$RUN_STAGE0" != 0 && "$RUN_STAGE0" != 1 ]]; then
  echo "RUN_STAGE0 must be 0 or 1; got '$RUN_STAGE0'." >&2
  exit 2
fi
if [[ -z "$STAGE0_MODEL_REF" ]]; then
  echo "STAGE0_MODEL_REF must not be empty." >&2
  exit 2
fi

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  "$ROOT/.venv-searchr1/bin/ray" stop --force >/dev/null 2>&1 || true
  bash "$ROOT/scripts/stop_servers.sh" || true
  exit "$status"
}
trap cleanup EXIT INT TERM

# An interrupted run can leave processes importing these virtual environments
# or holding GPUs. Stop them before any bootstrap recreates an environment.
"$ROOT/.venv-searchr1/bin/ray" stop --force >/dev/null 2>&1 || true
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
if [[ "$RUN_STAGE0" == 1 ]]; then
  echo "Full Stage-0 + Stage-2 pipeline completed."
else
  echo "Stage-2 pipeline completed (RUN_STAGE0=0)."
fi
