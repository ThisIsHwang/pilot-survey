#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

PROFILE=${PROFILE:-pilot}
RUN_EXP003=${RUN_EXP003:-1}
RUN_EXP004=${RUN_EXP004:-1}
RUN_EXP005=${RUN_EXP005:-1}
RUN_EXP006=${RUN_EXP006:-1}
RUN_REPORT=${RUN_REPORT:-1}
PREFETCH_MODELS=${PREFETCH_MODELS:-1}
OVERLAP_VLLM_SETUP=${OVERLAP_VLLM_SETUP:-1}
FORCE_TRAIN=${FORCE_TRAIN:-0}
KEEP_VLLM=${KEEP_VLLM:-0}
EXP002_RESULT_SET=${EXP002_RESULT_SET:-$PROFILE}
EXP002_WAIT_TIMEOUT=${EXP002_WAIT_TIMEOUT:-172800}
EXP002_POLL_SECONDS=${EXP002_POLL_SECONDS:-30}

for flag in \
  RUN_EXP003 RUN_EXP004 RUN_EXP005 RUN_EXP006 RUN_REPORT PREFETCH_MODELS \
  OVERLAP_VLLM_SETUP FORCE_TRAIN KEEP_VLLM; do
  value=${!flag}
  [[ "$value" == 0 || "$value" == 1 ]] || {
    echo "$flag must be 0 or 1; got '$value'." >&2
    exit 2
  }
done
if [[ "$KEEP_VLLM" == 1 ]]; then
  echo "KEEP_VLLM=1 is unsafe in the sequential node-2 queue because the next training stage needs GPUs 0-6; leave it at 0." >&2
  exit 2
fi
if [[ "$OVERLAP_VLLM_SETUP" == 1 && "$FORCE_TRAIN" == 1 ]]; then
  echo "FORCE_TRAIN=1 disables vLLM/training overlap to avoid training the first seed twice."
  OVERLAP_VLLM_SETUP=0
fi
case "$PROFILE" in
  smoke|pilot|full) ;;
  *) echo "PROFILE must be smoke, pilot, or full; got '$PROFILE'." >&2; exit 2 ;;
esac
for name in EXP002_WAIT_TIMEOUT EXP002_POLL_SECONDS; do
  value=${!name}
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
    echo "$name must be a positive integer; got '$value'." >&2
    exit 2
  }
done
[[ "$EXP002_RESULT_SET" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "EXP002_RESULT_SET may contain only letters, digits, dot, underscore, and dash." >&2
  exit 2
}

for command_name in flock hostname nice nvidia-smi nvcc realpath setsid; do
  command -v "$command_name" >/dev/null 2>&1 || {
    echo "Required command is missing: $command_name" >&2
    exit 1
  }
done
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo "The second-node queue requires Linux x86_64." >&2
  exit 1
fi
if ! nvcc --version | grep -Eq 'release 12\.9([, ]|$)'; then
  echo "The second-node queue requires the CUDA 12.9 toolkit; nvcc reports:" >&2
  nvcc --version >&2 || true
  exit 1
fi
mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
if [[ ${#GPU_NAMES[@]} -ne 8 ]]; then
  echo "Exactly 8 visible H100 GPUs are required; found ${#GPU_NAMES[@]}." >&2
  exit 1
fi
for gpu_name in "${GPU_NAMES[@]}"; do
  [[ "$gpu_name" == *H100* ]] || {
    echo "Every visible GPU must be an H100; found '$gpu_name'." >&2
    exit 1
  }
done

HOST_ID=${NODE2_HOST_ID:-$(hostname -s)}
HOST_ID=${HOST_ID//[^A-Za-z0-9._-]/-}
[[ -n "$HOST_ID" ]] || HOST_ID=unknown-host
NODE2_RUN_ID=${NODE2_RUN_ID:-$HOST_ID-$(date -u +%Y%m%dT%H%M%SZ)-$$}
[[ "$NODE2_RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "NODE2_RUN_ID may contain only letters, digits, dot, underscore, and dash." >&2
  exit 2
}

mkdir -p "$ROOT/work/locks"
# run_full_pipeline.sh takes this same lifetime lock. A second node pointed at
# the active shared checkout therefore fails before touching its venvs,
# Search-R1 checkout, outputs, or services.
exec {CHECKOUT_LOCK_FD}>"$ROOT/work/locks/full-pipeline.lock"
if ! flock -n "$CHECKOUT_LOCK_FD"; then
  cat >&2 <<EOF
This checkout is already running scripts/run_full_pipeline.sh or another
numbered queue: $ROOT
Use a separate clone on node 2. Each node must own its checkout, .venv-pilot,
.venv-vllm, .venv-searchr1, upstream/Search-R1, work, and logs directories.
Only immutable package/model caches and the read-only EXP-002 artifact root may
be shared.
EOF
  exit 1
fi

NODE2_LOCK_ROOT=${NODE2_LOCK_ROOT:-/tmp}
mkdir -p "$NODE2_LOCK_ROOT"
exec {HOST_LOCK_FD}>"$NODE2_LOCK_ROOT/stackpilot-node2-${USER:-unknown}-${HOST_ID}.lock"
if ! flock -n "$HOST_LOCK_FD"; then
  echo "Another second-node queue is already active on host '$HOST_ID'." >&2
  exit 1
fi
exec {RUN_LOCK_FD}> \
  "$NODE2_LOCK_ROOT/stackpilot-node2-${USER:-unknown}-${HOST_ID}-${NODE2_RUN_ID}.lock"
if ! flock -n "$RUN_LOCK_FD"; then
  echo "Second-node run ID is already active: $NODE2_RUN_ID" >&2
  exit 1
fi

export STACKPILOT_RUNTIME_ROOT=${STACKPILOT_RUNTIME_ROOT:-\
$ROOT/work/runtime/node2/$HOST_ID/$NODE2_RUN_ID}
export STACKPILOT_LOG_ROOT=${STACKPILOT_LOG_ROOT:-\
$ROOT/logs/runtime/node2/$HOST_ID/$NODE2_RUN_ID}
STACKPILOT_RUNTIME_ROOT=$(realpath -m "$STACKPILOT_RUNTIME_ROOT")
STACKPILOT_LOG_ROOT=$(realpath -m "$STACKPILOT_LOG_ROOT")
export STACKPILOT_RUNTIME_ROOT STACKPILOT_LOG_ROOT
if [[ "$STACKPILOT_RUNTIME_ROOT" == "$ROOT/work" || \
      "$STACKPILOT_LOG_ROOT" == "$ROOT/logs" ]]; then
  echo "Node 2 requires node-scoped STACKPILOT_RUNTIME_ROOT and STACKPILOT_LOG_ROOT." >&2
  exit 2
fi
mkdir -p "$STACKPILOT_RUNTIME_ROOT" "$STACKPILOT_LOG_ROOT"

EXP002_ROOT=
if [[ -n ${EXP002_ARTIFACT_ROOT:-} ]]; then
  EXP002_ROOT=$(realpath -m "$EXP002_ARTIFACT_ROOT")
  if [[ -d "$EXP002_ROOT/work/hard_rq0" ]]; then
    EXP002_ROOT=$(realpath "$EXP002_ROOT/work/hard_rq0")
  fi
  [[ -d "$EXP002_ROOT" ]] || {
    echo "EXP002_ARTIFACT_ROOT does not exist: $EXP002_ROOT" >&2
    exit 1
  }
  if [[ "$EXP002_ROOT" == "$(realpath -m "$ROOT/work/hard_rq0")" ]]; then
    echo "EXP002_ARTIFACT_ROOT must be the other node's read-only artifact root." >&2
    exit 2
  fi
elif [[ "$RUN_EXP006" == 1 || "$RUN_REPORT" == 1 ]]; then
  cat >&2 <<EOF
EXP002_ARTIFACT_ROOT is required when EXP-006 or the combined report is enabled.
No experiment has started.
Point it at the completed/in-progress EXP-002 directory from node 1, for example:
  EXP002_ARTIFACT_ROOT=/node1/pilot-survey/work/hard_rq0 \\
    bash experiments/run_node2_queue.sh
The directory is only read; EXP-002 is never launched by this queue.

To intentionally run only the independent EXP-003/004/005 stages:
  RUN_EXP006=0 RUN_REPORT=0 bash experiments/run_node2_queue.sh
EOF
  exit 2
fi

SEARCH_R1_ROOT=$(realpath -m "${SEARCH_R1_ROOT:-$ROOT/upstream/Search-R1}")
case "$SEARCH_R1_ROOT/" in
  "$ROOT/"*) ;;
  *)
    echo "SEARCH_R1_ROOT must be private to the node-2 checkout: $ROOT" >&2
    echo "Refusing to mutate external Search-R1 checkout: $SEARCH_R1_ROOT" >&2
    exit 2
    ;;
esac
export SEARCH_R1_ROOT

if [[ -n "$EXP002_ROOT" ]]; then
  for scoped_path in "$STACKPILOT_RUNTIME_ROOT" "$STACKPILOT_LOG_ROOT"; do
    case "$scoped_path/" in
      "$EXP002_ROOT/"*)
        echo "Runtime/log paths must not write inside EXP002_ARTIFACT_ROOT: $scoped_path" >&2
        exit 2
        ;;
    esac
  done
  export HARD_ASSET_ROOT=$ROOT/work/hard_rq0/assets/wiki18
fi

# A sibling isolated checkout can safely reuse content-addressed package/model
# caches from node 1. Virtual environments and editable checkouts are never
# shared. Explicit caller settings take precedence.
if [[ -n "$EXP002_ROOT" ]]; then
  EXP002_REPO_ROOT=$(realpath -m "$EXP002_ROOT/../..")
  if [[ -d "$EXP002_REPO_ROOT/.cache/huggingface" ]]; then
    export HF_HOME=${HF_HOME:-$EXP002_REPO_ROOT/.cache/huggingface}
  fi
  if [[ -d "$EXP002_REPO_ROOT/.cache/uv" ]]; then
    export UV_CACHE_DIR=${UV_CACHE_DIR:-$EXP002_REPO_ROOT/.cache/uv}
  fi
fi
export HF_HOME=${HF_HOME:-$ROOT/.cache/huggingface}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$ROOT/.cache/uv}
LOW_PRIORITY=(nice -n 10)
if command -v ionice >/dev/null 2>&1; then
  LOW_PRIORITY=(ionice -c 3 nice -n 10)
fi

declare -a BACKGROUND_GROUPS=()
EXP002_WATCH_PID=
EXP002_WATCH_WAITED=0
VLLM_BOOTSTRAP_PID=

remove_background_group() {
  local removed=$1
  local group_pid
  local -a remaining=()
  for group_pid in "${BACKGROUND_GROUPS[@]}"; do
    if [[ "$group_pid" != "$removed" ]]; then remaining+=("$group_pid"); fi
  done
  BACKGROUND_GROUPS=("${remaining[@]}")
}

stop_background_groups() {
  local group_pid
  local attempt
  local alive
  for group_pid in "${BACKGROUND_GROUPS[@]}"; do
    [[ -n "$group_pid" ]] || continue
    kill -TERM -- "-$group_pid" 2>/dev/null || true
  done
  for attempt in {1..20}; do
    alive=0
    for group_pid in "${BACKGROUND_GROUPS[@]}"; do
      [[ -n "$group_pid" ]] || continue
      if kill -0 -- "-$group_pid" 2>/dev/null; then alive=1; fi
    done
    [[ $alive -eq 0 ]] && break
    sleep 1
  done
  for group_pid in "${BACKGROUND_GROUPS[@]}"; do
    [[ -n "$group_pid" ]] || continue
    kill -KILL -- "-$group_pid" 2>/dev/null || true
    wait "$group_pid" 2>/dev/null || true
  done
  BACKGROUND_GROUPS=()
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  stop_background_groups
  bash "$ROOT/experiments/stop_aux_services.sh" >/dev/null 2>&1 || true
  bash "$ROOT/hard_rq0/stop_retrievers.sh" >/dev/null 2>&1 || true
  bash "$ROOT/scripts/stop_servers.sh" >/dev/null 2>&1 || true
  if [[ -x "$ROOT/.venv-searchr1/bin/ray" ]]; then
    "$ROOT/.venv-searchr1/bin/ray" stop --force >/dev/null 2>&1 || true
  fi
  if [[ $status -eq 0 ]]; then
    echo "Second-node cleanup complete; all managed services are stopped."
  else
    echo "Second-node queue failed with status $status; logs: $STACKPILOT_LOG_ROOT" >&2
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

start_exp002_watcher() {
  local log_file=$STACKPILOT_LOG_ROOT/setup/exp002-watcher.log
  mkdir -p "$(dirname "$log_file")"
  setsid env CUDA_VISIBLE_DEVICES= \
    "${LOW_PRIORITY[@]}" \
    bash "$ROOT/experiments/watch_exp002.sh" \
      --root "$EXP002_ROOT" \
      --profile "$PROFILE" \
      --result-set "$EXP002_RESULT_SET" \
      --seeds "${EXP006_SEEDS:-13 42 87}" \
      --ready-root "$STACKPILOT_RUNTIME_ROOT/exp002" \
      --timeout "$EXP002_WAIT_TIMEOUT" \
      --poll-seconds "$EXP002_POLL_SECONDS" \
      >"$log_file" 2>&1 &
  EXP002_WATCH_PID=$!
  BACKGROUND_GROUPS+=("$EXP002_WATCH_PID")
  echo "Watching EXP-002 in the background (PID $EXP002_WATCH_PID; log: $log_file)."
}

start_vllm_bootstrap() {
  local log_file=$STACKPILOT_LOG_ROOT/setup/vllm-bootstrap.log
  mkdir -p "$(dirname "$log_file")"
  setsid env CUDA_VISIBLE_DEVICES= VLLM_DEFER_GPU_PROBE=1 \
    "${LOW_PRIORITY[@]}" bash "$ROOT/scripts/bootstrap_vllm.sh" \
    >"$log_file" 2>&1 &
  VLLM_BOOTSTRAP_PID=$!
  BACKGROUND_GROUPS+=("$VLLM_BOOTSTRAP_PID")
  echo "Preparing the evaluation-only vLLM environment in the background (PID $VLLM_BOOTSTRAP_PID)."
}

wait_for_vllm_bootstrap() {
  local log_file=$STACKPILOT_LOG_ROOT/setup/vllm-bootstrap.log
  local status
  [[ -n "$VLLM_BOOTSTRAP_PID" ]] || return 0
  if wait "$VLLM_BOOTSTRAP_PID"; then status=0; else status=$?; fi
  remove_background_group "$VLLM_BOOTSTRAP_PID"
  VLLM_BOOTSTRAP_PID=
  if [[ $status -ne 0 ]]; then
    echo "Background vLLM setup failed with status $status." >&2
    tail -n 100 "$log_file" >&2 || true
    return "$status"
  fi
  echo "Background vLLM setup completed (log: $log_file)."
}

validate_vllm_hardware() {
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    "$ROOT/.venv-vllm/bin/python" - <<'PY'
import torch
import vllm

if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
    raise SystemExit(
        f"Expected 8 visible H100s, found {torch.cuda.device_count()} CUDA devices."
    )
for index in range(8):
    properties = torch.cuda.get_device_properties(index)
    if "H100" not in properties.name:
        raise SystemExit(f"GPU {index} is not an H100: {properties.name}")
    if properties.total_memory < 70 * 1024**3:
        raise SystemExit(
            f"GPU {index} appears to be a MIG slice "
            f"({properties.total_memory / 1024**3:.1f} GiB)."
        )
torch.ones(1, device="cuda").mul_(2)
print(
    f"vLLM {vllm.__version__}; PyTorch {torch.__version__}; "
    f"wheel CUDA {torch.version.cuda}; visible H100s: {torch.cuda.device_count()}"
)
PY
}

pretrain_first_mixed_policy() {
  local experiment_id
  local first_seed
  local seed_text
  local -a seeds
  if [[ "$RUN_EXP003" == 1 ]]; then
    experiment_id=EXP-003
    seed_text=${EXP003_SEEDS:-13 42 87}
  elif [[ "$RUN_EXP004" == 1 ]]; then
    experiment_id=EXP-004
    seed_text=${EXP004_SEEDS:-42}
  else
    echo "No EXP-003/004 training stage is enabled; skipping overlap pretraining."
    return 0
  fi
  read -r -a seeds <<<"$seed_text"
  first_seed=${seeds[0]:-}
  [[ "$first_seed" =~ ^[1-9][0-9]*$ ]] || {
    echo "The first $experiment_id seed must be a positive integer; got '$first_seed'." >&2
    return 2
  }

  echo "Pretraining $experiment_id seed $first_seed while vLLM setup runs."
  bash "$ROOT/experiments/reset_searchr1_experiment_files.sh"
  EXPERIMENT_ID="$experiment_id" SEED="$first_seed" PROFILE="$PROFILE" \
    BASE_MODEL="$BASE_MODEL" BASE_MODEL_REVISION="$BASE_MODEL_REVISION" \
    bash "$ROOT/experiments/train_mixed_policy.sh"
}

wait_for_watcher_marker() {
  local marker=$1
  local label=$2
  local log_file=$STACKPILOT_LOG_ROOT/setup/exp002-watcher.log
  while [[ ! -s "$marker" ]]; do
    if ! kill -0 "$EXP002_WATCH_PID" 2>/dev/null; then
      local status
      if wait "$EXP002_WATCH_PID"; then status=0; else status=$?; fi
      echo "$label was not published; watcher exited with status $status." >&2
      tail -n 100 "$log_file" >&2 || true
      return 1
    fi
    sleep 5
  done
  echo "$label ready."
}

wait_for_exp002_completion() {
  local log_file=$STACKPILOT_LOG_ROOT/setup/exp002-watcher.log
  local status
  if wait "$EXP002_WATCH_PID"; then status=0; else status=$?; fi
  remove_background_group "$EXP002_WATCH_PID"
  EXP002_WATCH_PID=
  EXP002_WATCH_WAITED=1
  if [[ $status -ne 0 ]]; then
    echo "EXP-002 watcher failed with status $status." >&2
    tail -n 100 "$log_file" >&2 || true
    return "$status"
  fi
  [[ -s "$STACKPILOT_RUNTIME_ROOT/exp002/complete.ready" ]] || {
    echo "EXP-002 watcher exited without a completion marker." >&2
    return 1
  }
  echo "Validated external EXP-002 completion."
}

link_read_only_component() {
  local component=$1
  local source=$EXP002_ROOT/$component
  local destination=$ROOT/work/hard_rq0/$component
  mkdir -p "$(dirname "$destination")"
  if [[ -L "$destination" ]]; then
    local current
    current=$(realpath -m "$destination")
    [[ "$current" == "$(realpath -m "$source")" ]] || {
      echo "Existing $destination points at $current, not $source." >&2
      return 1
    }
    return
  fi
  if [[ -e "$destination" ]]; then
    echo "Refusing to mix node-local and external EXP-002 $component data." >&2
    echo "Move or remove this node-2 path, then rerun so it can be linked read-only: $destination" >&2
    return 1
  fi
  ln -s "$source" "$destination"
  echo "Linked read-only EXP-002 $component: $destination -> $source"
}

# Remove only this run's stale managed services. The checkout lock above makes
# it impossible to race this cleanup against run_full_pipeline.sh.
bash "$ROOT/experiments/stop_aux_services.sh" >/dev/null 2>&1 || true
bash "$ROOT/hard_rq0/stop_retrievers.sh" >/dev/null 2>&1 || true
bash "$ROOT/scripts/stop_servers.sh" >/dev/null 2>&1 || true
if [[ -x "$ROOT/.venv-searchr1/bin/ray" ]]; then
  "$ROOT/.venv-searchr1/bin/ray" stop --force >/dev/null 2>&1 || true
fi

if [[ -n "$EXP002_ROOT" ]]; then
  link_read_only_component assets
  link_read_only_component data
  link_read_only_component searchr1
fi

echo "Preparing node-isolated environments; verified caches are reused."
bash "$ROOT/scripts/bootstrap.sh"

if [[ -n "$EXP002_ROOT" ]]; then
  start_exp002_watcher
fi

# Search-R1 is required by the first training stage. The independent vLLM
# environment is evaluation-only, so its CPU/network setup may safely continue
# at low priority behind training; its GPU probe runs after that training exits.
bash "$ROOT/scripts/bootstrap_searchr1.sh"
if [[ "$OVERLAP_VLLM_SETUP" == 1 ]]; then
  start_vllm_bootstrap
else
  bash "$ROOT/scripts/bootstrap_vllm.sh"
fi

DEFAULT_NUMBERED_MODEL=Qwen/Qwen2.5-3B-Instruct
DEFAULT_NUMBERED_REVISION=aa8e72537993ba99e69dfaafa59ed015b17504d1
BASE_MODEL=${BASE_MODEL:-$DEFAULT_NUMBERED_MODEL}
if [[ -z ${BASE_MODEL_REVISION:-} ]]; then
  if [[ "$BASE_MODEL" == "$DEFAULT_NUMBERED_MODEL" ]]; then
    BASE_MODEL_REVISION=$DEFAULT_NUMBERED_REVISION
  else
    BASE_MODEL_REVISION=main
  fi
fi
export BASE_MODEL BASE_MODEL_REVISION

EXP006_BASE_MODEL=${BASE_MODEL_REF:-$DEFAULT_NUMBERED_MODEL}
if [[ -n ${BASE_MODEL_REF_REVISION:-} ]]; then
  EXP006_BASE_REVISION=$BASE_MODEL_REF_REVISION
elif [[ "$EXP006_BASE_MODEL" == "$BASE_MODEL" ]]; then
  EXP006_BASE_REVISION=$BASE_MODEL_REVISION
elif [[ "$EXP006_BASE_MODEL" == "$DEFAULT_NUMBERED_MODEL" ]]; then
  EXP006_BASE_REVISION=$DEFAULT_NUMBERED_REVISION
else
  EXP006_BASE_REVISION=main
fi

if [[ "$PREFETCH_MODELS" == 1 ]]; then
  env CUDA_VISIBLE_DEVICES= "${LOW_PRIORITY[@]}" \
    bash "$ROOT/scripts/resolve_hf_model.sh" \
      "$BASE_MODEL" "$BASE_MODEL_REVISION" \
      "$ROOT/.venv-pilot/bin/python" >/dev/null

  DEFAULT_E5_MODEL=intfloat/e5-base-v2
  DEFAULT_E5_REVISION=f52bf8ec8c7124536f0efb74aca902b2995e5bcd
  E5_MODEL_SOURCE=${E5_MODEL:-$DEFAULT_E5_MODEL}
  if [[ -n ${E5_MODEL_REVISION:-} ]]; then
    E5_REVISION=$E5_MODEL_REVISION
  elif [[ "$E5_MODEL_SOURCE" == "$DEFAULT_E5_MODEL" ]]; then
    E5_REVISION=$DEFAULT_E5_REVISION
  else
    E5_REVISION=main
  fi
  env CUDA_VISIBLE_DEVICES= "${LOW_PRIORITY[@]}" \
    bash "$ROOT/scripts/resolve_hf_model.sh" \
      "$E5_MODEL_SOURCE" "$E5_REVISION" "$ROOT/.venv-pilot/bin/python" \
      >/dev/null

  if [[ "$RUN_EXP006" == 1 && \
        "$EXP006_BASE_MODEL@$EXP006_BASE_REVISION" != \
        "$BASE_MODEL@$BASE_MODEL_REVISION" ]]; then
    env CUDA_VISIBLE_DEVICES= "${LOW_PRIORITY[@]}" \
      bash "$ROOT/scripts/resolve_hf_model.sh" \
        "$EXP006_BASE_MODEL" "$EXP006_BASE_REVISION" \
        "$ROOT/.venv-pilot/bin/python" >/dev/null
  fi
fi

if [[ -n "$EXP002_ROOT" ]]; then
  wait_for_watcher_marker \
    "$STACKPILOT_RUNTIME_ROOT/exp002/inputs.ready" \
    "External Hard-RQ0 assets and prepared data"
  "$ROOT/.venv-pilot/bin/python" -m stackpilot.hard_assets check \
    --root "$ROOT/work/hard_rq0/assets/wiki18"
  "$ROOT/.venv-pilot/bin/python" -m stackpilot.prepare_hard_rq0 \
    --config "$ROOT/configs/hard_rq0.yaml" --check
else
  bash "$ROOT/hard_rq0/download_assets.sh"
  bash "$ROOT/hard_rq0/prepare_data.sh"
fi

export NUMBERED_SETUP_READY=1
export TRAIN_GPUS=${TRAIN_GPUS:-0,1,2,3,4,5,6}
export N_GPUS=${N_GPUS:-7}
export E5_GPU=${E5_GPU:-7}

# Give the GPUs useful work while the independent evaluation environment is
# still installing. The normal EXP runner revalidates and reuses this exact
# completed training marker before it merges and evaluates the first seed.
if [[ "$OVERLAP_VLLM_SETUP" == 1 ]]; then
  pretrain_first_mixed_policy
  wait_for_vllm_bootstrap
  validate_vllm_hardware
fi

# Experiment stages are deliberately sequential. E5 may occupy reserved GPU 7
# while GRPO/evaluation uses GPUs 0-6, but two experiment GPU stages never
# overlap.
if [[ "$RUN_EXP003" == 1 ]]; then
  PROFILE="$PROFILE" SEEDS="${EXP003_SEEDS:-13 42 87}" \
    bash "$ROOT/experiments/EXP-003/run.sh"
fi
if [[ "$RUN_EXP004" == 1 ]]; then
  PROFILE="$PROFILE" SEEDS="${EXP004_SEEDS:-42}" \
    bash "$ROOT/experiments/EXP-004/run.sh"
fi
if [[ "$RUN_EXP005" == 1 ]]; then
  PROFILE="$PROFILE" SEEDS="${EXP005_SEEDS:-42}" \
    BACKEND_LIST="${EXP005_BACKENDS:-bm25 e5}" \
    bash "$ROOT/experiments/EXP-005/run.sh"
fi
if [[ "$RUN_EXP006" == 1 ]]; then
  wait_for_exp002_completion
  PROFILE="$PROFILE" SEEDS="${EXP006_SEEDS:-13 42 87}" \
    ORACLE_SEEDS="${EXP006_ORACLE_SEEDS:-42}" \
    EXP002_ROOT="$EXP002_ROOT" EXP002_RESULT_SET="$EXP002_RESULT_SET" \
    BASE_MODEL_REF="$EXP006_BASE_MODEL" \
    BASE_MODEL_REVISION="$EXP006_BASE_REVISION" \
    bash "$ROOT/experiments/EXP-006/run.sh"
fi
if [[ "$RUN_REPORT" == 1 ]]; then
  if [[ -n "$EXP002_ROOT" && "$EXP002_WATCH_WAITED" != 1 ]]; then
    wait_for_exp002_completion
  fi
  report_args=(PROFILE="$PROFILE")
  if [[ -n "$EXP002_ROOT" ]]; then
    report_args+=(
      "HARD_RESULTS=$EXP002_ROOT/runs/$EXP002_RESULT_SET/results/policies"
    )
  fi
  env "${report_args[@]}" bash "$ROOT/experiments/make_report.sh"
fi

echo "Second-node numbered experiment queue complete: $NODE2_RUN_ID"
exit 0
