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
PREFETCH_FUTURE_WORK=${PREFETCH_FUTURE_WORK:-1}
SKIP_BOOTSTRAP=${SKIP_BOOTSTRAP:-0}
SKIP_ASSETS=${SKIP_ASSETS:-0}
SMOKE_ONLY=${SMOKE_ONLY:-0}
for flag_name in \
  RUN_STAGE0 RUN_HARD_RQ0 PREFETCH_FUTURE_WORK SKIP_BOOTSTRAP \
  SKIP_ASSETS SMOKE_ONLY; do
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

for required_command in flock nice; do
  command -v "$required_command" >/dev/null 2>&1 || {
    echo "Required process-control command is missing: $required_command" >&2
    exit 1
  }
done
mkdir -p "$ROOT/work/locks"
exec {PIPELINE_LOCK_FD}>"$ROOT/work/locks/full-pipeline.lock"
if ! flock -n "$PIPELINE_LOCK_FD"; then
  echo "Another scripts/run_full_pipeline.sh is already active for $ROOT." >&2
  exit 1
fi

SEARCHR1_BOOTSTRAP_PID=
STAGE2_MODEL_PREFETCH_PID=
HARD_MODEL_PREFETCH_PID=
HARD_ASSET_PREFETCH_PID=
declare -A BACKGROUND_PROCESS_GROUPS=()

start_background_job() {
  local output_variable=$1
  local log_file=$2
  shift 2
  local attempt
  local group_pid
  local pid
  local pid_file
  mkdir -p "$(dirname "$log_file")"
  mkdir -p "$ROOT/work/pids"
  pid_file=$ROOT/work/pids/.background-${output_variable,,}-$$-$RANDOM.pid
  "$ROOT/.venv-pilot/bin/python" "$ROOT/scripts/session_runner.py" \
    --pid-file "$pid_file" -- "$@" >"$log_file" 2>&1 &
  printf -v "$output_variable" '%s' "$!"
  pid=${!output_variable}
  # session_runner replaces itself with the job, so its PID is also the
  # eventual session/process-group ID. Register it before the handshake.
  BACKGROUND_PROCESS_GROUPS[$pid]=$pid
  for attempt in {1..100}; do
    if [[ -s "$pid_file" ]]; then break; fi
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" 2>/dev/null || true
      echo "Background job failed before startup; tail of $log_file:" >&2
      tail -n 80 "$log_file" >&2 || true
      rm -f -- "$pid_file"
      printf -v "$output_variable" ''
      unset "BACKGROUND_PROCESS_GROUPS[$pid]"
      return 1
    fi
    sleep 0.05
  done
  if [[ ! -s "$pid_file" ]]; then
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    rm -f -- "$pid_file"
    printf -v "$output_variable" ''
    unset "BACKGROUND_PROCESS_GROUPS[$pid]"
    echo "Timed out while creating a process group for $log_file." >&2
    return 1
  fi
  read -r group_pid < "$pid_file"
  rm -f -- "$pid_file"
  if [[ ! "$group_pid" =~ ^[1-9][0-9]*$ || "$group_pid" != "$pid" ]]; then
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    printf -v "$output_variable" ''
    unset "BACKGROUND_PROCESS_GROUPS[$pid]"
    echo "Invalid process-group PID for $log_file: '$group_pid' (expected $pid)." >&2
    return 1
  fi
  BACKGROUND_PROCESS_GROUPS[$pid]=$group_pid
}

wait_background_job() {
  local pid_variable=$1
  local label=$2
  local log_file=$3
  local pid=${!pid_variable}
  local status
  [[ -n "$pid" ]] || return 0
  if wait "$pid"; then status=0; else status=$?; fi
  printf -v "$pid_variable" ''
  unset "BACKGROUND_PROCESS_GROUPS[$pid]"
  if [[ $status -ne 0 ]]; then
    echo "$label failed with status $status; tail of $log_file:" >&2
    tail -n 80 "$log_file" >&2 || true
    return "$status"
  fi
  echo "$label completed (log: $log_file)."
}

stop_background_jobs() {
  local -a pids=(
    "$SEARCHR1_BOOTSTRAP_PID"
    "$STAGE2_MODEL_PREFETCH_PID"
    "$HARD_MODEL_PREFETCH_PID"
    "$HARD_ASSET_PREFETCH_PID"
  )
  local attempt
  local alive
  local group_pid
  local pid
  for pid in "${pids[@]}"; do
    [[ -n "$pid" ]] || continue
    group_pid=${BACKGROUND_PROCESS_GROUPS[$pid]:-$pid}
    kill -TERM -- "-$group_pid" 2>/dev/null || \
      kill -TERM "$pid" 2>/dev/null || true
  done
  for attempt in {1..20}; do
    alive=0
    for pid in "${pids[@]}"; do
      [[ -n "$pid" ]] || continue
      group_pid=${BACKGROUND_PROCESS_GROUPS[$pid]:-$pid}
      if kill -0 -- "-$group_pid" 2>/dev/null || kill -0 "$pid" 2>/dev/null; then
        alive=1
      fi
    done
    [[ $alive -eq 0 ]] && break
    sleep 1
  done
  for pid in "${pids[@]}"; do
    [[ -n "$pid" ]] || continue
    group_pid=${BACKGROUND_PROCESS_GROUPS[$pid]:-$pid}
    if kill -0 -- "-$group_pid" 2>/dev/null; then
      kill -KILL -- "-$group_pid" 2>/dev/null || true
    elif kill -0 "$pid" 2>/dev/null; then
      kill -KILL "$pid" 2>/dev/null || true
    fi
    wait "$pid" 2>/dev/null || true
  done
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  stop_background_jobs
  if [[ -x "$ROOT/.venv-searchr1/bin/ray" ]]; then
    "$ROOT/.venv-searchr1/bin/ray" stop --force >/dev/null 2>&1 || true
  fi
  bash "$ROOT/hard_rq0/stop_retrievers.sh" || true
  bash "$ROOT/scripts/stop_servers.sh" || true
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# An interrupted run can leave processes importing these virtual environments
# or holding GPUs. Stop them before any bootstrap recreates an environment.
if [[ -x "$ROOT/.venv-searchr1/bin/ray" ]]; then
  "$ROOT/.venv-searchr1/bin/ray" stop --force >/dev/null 2>&1 || true
fi
bash "$ROOT/hard_rq0/stop_retrievers.sh" || true
bash "$ROOT/scripts/stop_servers.sh" || true

PREFETCH_LOG_ROOT=$ROOT/logs/prefetch
SEARCHR1_BOOTSTRAP_LOG=$PREFETCH_LOG_ROOT/searchr1-bootstrap.log
STAGE2_MODEL_PREFETCH_LOG=$PREFETCH_LOG_ROOT/stage2-models.log
HARD_MODEL_PREFETCH_LOG=$PREFETCH_LOG_ROOT/hard-models.log
HARD_ASSET_PREFETCH_LOG=$PREFETCH_LOG_ROOT/hard-assets.log
LOW_PRIORITY=(nice -n 10)
if command -v ionice >/dev/null 2>&1; then
  LOW_PRIORITY=(ionice -c 3 nice -n 10)
fi

if [[ "$SKIP_BOOTSTRAP" != 1 ]]; then
  bash "$ROOT/scripts/bootstrap.sh"
fi

if [[ "$PREFETCH_FUTURE_WORK" == 1 ]]; then
  if [[ "$RUN_HARD_RQ0" == 1 ]]; then
    if [[ "$SMOKE_ONLY" == 1 ]]; then
      HARD_PREFETCH_PROFILE=${HARD_PROFILE:-smoke}
    else
      HARD_PREFETCH_PROFILE=${HARD_PROFILE:-pilot}
    fi
    if [[ "$SKIP_ASSETS" == 1 ]]; then
      "$ROOT/.venv-pilot/bin/python" -m stackpilot.hard_assets check \
        --root "${HARD_ASSET_ROOT:-$ROOT/work/hard_rq0/assets/wiki18}" >/dev/null
    fi
    HARD_ROOT_EXTRA_GIB=${FULL_PIPELINE_STAGE_DISK_GIB:-60} \
      HARD_HF_RESERVE_GIB=${FULL_PIPELINE_HF_RESERVE_GIB:-30} \
      PROFILE=$HARD_PREFETCH_PROFILE \
      bash "$ROOT/hard_rq0/preflight_storage.sh"
  fi

  if [[ "$SKIP_BOOTSTRAP" != 1 ]]; then
    start_background_job SEARCHR1_BOOTSTRAP_PID "$SEARCHR1_BOOTSTRAP_LOG" \
      env CUDA_VISIBLE_DEVICES= SEARCHR1_DEFER_GPU_PROBE=1 \
      "${LOW_PRIORITY[@]}" bash "$ROOT/scripts/bootstrap_searchr1.sh"
    echo "Search-R1 installation is running behind the current stage."
  fi
  start_background_job \
    STAGE2_MODEL_PREFETCH_PID "$STAGE2_MODEL_PREFETCH_LOG" \
    env CUDA_VISIBLE_DEVICES= "${LOW_PRIORITY[@]}" \
    bash "$ROOT/scripts/prefetch_future_models.sh" --stage2
  echo "Stage-2 model downloads are running behind the current stage."
  if [[ "$RUN_HARD_RQ0" == 1 ]]; then
    start_background_job HARD_MODEL_PREFETCH_PID "$HARD_MODEL_PREFETCH_LOG" \
      env CUDA_VISIBLE_DEVICES= "${LOW_PRIORITY[@]}" \
      bash "$ROOT/scripts/prefetch_future_models.sh" --hard
    echo "Hard-RQ0 model downloads are running behind the current stage."
  fi
  if [[ "$RUN_HARD_RQ0" == 1 && "$SKIP_ASSETS" != 1 ]]; then
    start_background_job HARD_ASSET_PREFETCH_PID "$HARD_ASSET_PREFETCH_LOG" \
      env CUDA_VISIBLE_DEVICES= "${LOW_PRIORITY[@]}" \
      bash "$ROOT/hard_rq0/download_assets.sh"
    echo "Hard-RQ0 assets are downloading behind the current stage."
  fi
fi

if [[ "$SKIP_BOOTSTRAP" != 1 ]]; then
  bash "$ROOT/scripts/bootstrap_vllm.sh"
  if [[ "$PREFETCH_FUTURE_WORK" != 1 ]]; then
    bash "$ROOT/scripts/bootstrap_searchr1.sh"
  fi
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

if ! wait_background_job SEARCHR1_BOOTSTRAP_PID \
  "Search-R1 background installation" "$SEARCHR1_BOOTSTRAP_LOG"; then
  echo "Retrying Search-R1 installation synchronously." >&2
  bash "$ROOT/scripts/bootstrap_searchr1.sh"
fi
if ! wait_background_job STAGE2_MODEL_PREFETCH_PID \
  "Stage-2 model prefetch" "$STAGE2_MODEL_PREFETCH_LOG"; then
  echo "Stage-2 will retry and validate any missing model snapshot." >&2
fi

(
  unset MODEL MODEL_PATH MODEL_LOCAL_ONLY MODEL_REVISION SERVED_MODEL_NAME HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
  SKIP_BOOTSTRAP=1 bash "$ROOT/searchr1_stage2/run_all.sh"
)
if [[ "$RUN_HARD_RQ0" == 1 ]]; then
  if ! wait_background_job HARD_MODEL_PREFETCH_PID \
    "Hard-RQ0 model prefetch" "$HARD_MODEL_PREFETCH_LOG"; then
    echo "Hard-RQ0 will retry and validate any missing model snapshot." >&2
  fi
  if ! wait_background_job HARD_ASSET_PREFETCH_PID \
    "Hard-RQ0 asset prefetch" "$HARD_ASSET_PREFETCH_LOG"; then
    echo "The Hard-RQ0 consumer will retry the resumable download." >&2
  fi
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
