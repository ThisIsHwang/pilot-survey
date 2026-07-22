#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
SEARCH_R1=$ROOT/upstream/Search-R1
SEARCH_R1_PYTHON=$ROOT/.venv-searchr1/bin/python
SEARCH_R1_COMMIT=${SEARCH_R1_COMMIT:-598e61bd1d36895726d28a8d06b3a15bed19f5d3}
BACKEND=${BACKEND:-bm25}
PROFILE=${PROFILE:-smoke}
DEFAULT_BASE_MODEL=Qwen/Qwen2.5-3B-Instruct
DEFAULT_BASE_MODEL_REVISION=aa8e72537993ba99e69dfaafa59ed015b17504d1
BASE_MODEL=${BASE_MODEL:-$DEFAULT_BASE_MODEL}
if [[ -z ${BASE_MODEL_REVISION:-} ]]; then
  if [[ "$BASE_MODEL" == "$DEFAULT_BASE_MODEL" ]]; then
    BASE_MODEL_REVISION=$DEFAULT_BASE_MODEL_REVISION
  else
    BASE_MODEL_REVISION=main
  fi
fi
N_AGENT=${N_AGENT:-4}
AUTO_LAUNCH_RETRIEVER=${AUTO_LAUNCH_RETRIEVER:-1}

if [[ "$BASE_MODEL" == \~/* ]]; then
  BASE_MODEL="$HOME/${BASE_MODEL#\~/}"
fi
if [[ -d "$BASE_MODEL" ]]; then
  BASE_MODEL=$(readlink -f "$BASE_MODEL")
elif [[ "$BASE_MODEL" == /* || "$BASE_MODEL" == ./* || "$BASE_MODEL" == ../* || -e "$BASE_MODEL" ]]; then
  echo "Local BASE_MODEL does not exist or is not a directory: $BASE_MODEL" >&2
  exit 2
fi

case "$BACKEND" in
  bm25) DEFAULT_PORT=8001; PORT=${PORT:-$DEFAULT_PORT} ;;
  e5) DEFAULT_PORT=8002; PORT=${PORT:-$DEFAULT_PORT} ;;
  *) echo "BACKEND must be bm25 or e5" >&2; exit 2 ;;
esac

DEFAULT_TOTAL_UPDATES=
case "$PROFILE" in
  smoke)
    TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
    TRAIN_BATCH=${TRAIN_BATCH:-32}
    VAL_BATCH=${VAL_BATCH:-32}
    MINI_BATCH=${MINI_BATCH:-16}
    MICRO_BATCH=${MICRO_BATCH:-8}
    TRAIN_DATA_NUM=${TRAIN_DATA_NUM:-32}
    VAL_DATA_NUM=${VAL_DATA_NUM:-32}
    DEFAULT_TOTAL_UPDATES=1
    VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-false}
    ;;
  pilot)
    TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
    TRAIN_BATCH=${TRAIN_BATCH:-128}
    VAL_BATCH=${VAL_BATCH:-128}
    MINI_BATCH=${MINI_BATCH:-64}
    MICRO_BATCH=${MICRO_BATCH:-8}
    TRAIN_DATA_NUM=${TRAIN_DATA_NUM:-null}
    VAL_DATA_NUM=${VAL_DATA_NUM:-null}
    VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-false}
    ;;
  *) echo "PROFILE must be smoke or pilot" >&2; exit 2 ;;
esac

if [[ -n ${TOTAL_UPDATES+x} && -n ${TOTAL_STEPS+x} && "$TOTAL_UPDATES" != "$TOTAL_STEPS" ]]; then
  echo "TOTAL_UPDATES and legacy TOTAL_STEPS disagree: $TOTAL_UPDATES != $TOTAL_STEPS" >&2
  exit 2
fi
if [[ -n ${TOTAL_UPDATES+x} ]]; then
  TOTAL_UPDATES_VALUE=$TOTAL_UPDATES
elif [[ -n ${TOTAL_STEPS+x} ]]; then
  TOTAL_UPDATES_VALUE=$TOTAL_STEPS
else
  TOTAL_UPDATES_VALUE=$DEFAULT_TOTAL_UPDATES
fi
TOTAL_UPDATES=$TOTAL_UPDATES_VALUE

EXP=${EXP:-hotpot-${BACKEND}-${PROFILE}-grpo}
if [[ ! "$EXP" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "EXP may contain only letters, digits, dot, underscore, and dash: $EXP" >&2
  exit 2
fi
CHECKPOINT_DIR=${CHECKPOINT_DIR:-$ROOT/work/checkpoints/$EXP}
LOG_FILE=$ROOT/logs/${EXP}.log
TRAIN_FILE=$ROOT/work/searchr1_hotpot/train.parquet
VAL_FILE=$ROOT/work/searchr1_hotpot/test.parquet
INDEX_MANIFEST=$ROOT/work/indexes/$BACKEND/.pilot-manifest.json
COMPLETE_MARKER=$CHECKPOINT_DIR/.complete.json
mkdir -p "$ROOT/logs"

[[ -x "$SEARCH_R1_PYTHON" ]] || {
  echo "Missing .venv-searchr1; run bash scripts/bootstrap_searchr1.sh" >&2
  exit 1
}
[[ -d "$SEARCH_R1/.git" ]] || {
  echo "Missing $SEARCH_R1; run bash scripts/bootstrap.sh first" >&2
  exit 1
}
for data_file in "$TRAIN_FILE" "$VAL_FILE"; do
  [[ -s "$data_file" ]] || {
    echo "Missing Search-R1 data: $data_file" >&2
    echo "Run: $ROOT/.venv-pilot/bin/python searchr1_stage2/make_hotpot_searchr1_data.py --work-dir work" >&2
    exit 1
  }
done
[[ -s "$INDEX_MANIFEST" ]] || {
  echo "Missing $BACKEND index manifest: $INDEX_MANIFEST" >&2
  echo "Run: bash scripts/build_indexes.sh" >&2
  exit 1
}

TRAIN_MIN_DISK_GIB=${TRAIN_MIN_DISK_GIB:-15}
if [[ ! "$TRAIN_MIN_DISK_GIB" =~ ^[1-9][0-9]*$ ]]; then
  echo "TRAIN_MIN_DISK_GIB must be a positive integer; got '$TRAIN_MIN_DISK_GIB'." >&2
  exit 2
fi
STAGE2_MIN_DISK_GIB=$TRAIN_MIN_DISK_GIB \
  bash "$ROOT/scripts/preflight_searchr1.sh"
source "$ROOT/scripts/lib/runtime.sh"
GPU_IDS=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
validate_gpu_list "$GPU_IDS" 8 "Search-R1 GRPO" || exit 1
for numeric_name in TOTAL_EPOCHS TRAIN_BATCH VAL_BATCH MINI_BATCH MICRO_BATCH N_AGENT PORT; do
  numeric_value=${!numeric_name}
  if [[ ! "$numeric_value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$numeric_name must be a positive integer; got '$numeric_value'." >&2
    exit 2
  fi
done
for subset_name in TRAIN_DATA_NUM VAL_DATA_NUM; do
  subset_value=${!subset_name}
  if [[ "$subset_value" != null && ! "$subset_value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$subset_name must be 'null' or a positive integer; got '$subset_value'." >&2
    exit 2
  fi
done
if [[ "$VAL_BEFORE_TRAIN" != true && "$VAL_BEFORE_TRAIN" != false ]]; then
  echo "VAL_BEFORE_TRAIN must be true or false; got '$VAL_BEFORE_TRAIN'." >&2
  exit 2
fi
if [[ "$AUTO_LAUNCH_RETRIEVER" != 0 && "$AUTO_LAUNCH_RETRIEVER" != 1 ]]; then
  echo "AUTO_LAUNCH_RETRIEVER must be 0 or 1; got '$AUTO_LAUNCH_RETRIEVER'." >&2
  exit 2
fi
if [[ "$AUTO_LAUNCH_RETRIEVER" == 1 && "$PORT" != "$DEFAULT_PORT" ]]; then
  echo "AUTO_LAUNCH_RETRIEVER=1 uses fixed $BACKEND port $DEFAULT_PORT; got PORT=$PORT." >&2
  echo "Use PORT=$DEFAULT_PORT or manage the custom retriever with AUTO_LAUNCH_RETRIEVER=0." >&2
  exit 2
fi
if (( MINI_BATCH % 8 != 0 || MICRO_BATCH % 8 != 0 )); then
  echo "MINI_BATCH and MICRO_BATCH must both be divisible by 8; got $MINI_BATCH and $MICRO_BATCH." >&2
  exit 2
fi
if (( MINI_BATCH / 8 < MICRO_BATCH / 8 || (MINI_BATCH / 8) % (MICRO_BATCH / 8) != 0 )); then
  echo "Per-GPU MINI_BATCH must be a positive multiple of per-GPU MICRO_BATCH." >&2
  exit 2
fi
if (( (TRAIN_BATCH * N_AGENT) % 8 != 0 )); then
  echo "TRAIN_BATCH*N_AGENT must be divisible by 8; got $TRAIN_BATCH*$N_AGENT." >&2
  exit 2
fi
if (( (TRAIN_BATCH * N_AGENT) % MINI_BATCH != 0 )); then
  echo "TRAIN_BATCH*N_AGENT must be divisible by MINI_BATCH; got $TRAIN_BATCH*$N_AGENT/$MINI_BATCH." >&2
  exit 2
fi

AVAILABLE_UPDATES=$("$SEARCH_R1_PYTHON" - \
  "$TRAIN_FILE" "$TRAIN_DATA_NUM" "$TRAIN_BATCH" "$TOTAL_EPOCHS" \
  "$VAL_FILE" "$VAL_DATA_NUM" "$VAL_BATCH" <<'PY'
import sys
import pyarrow.parquet as pq

train_path, requested, batch, epochs, val_path, val_requested, val_batch = sys.argv[1:]
batch, epochs, val_batch = int(batch), int(epochs), int(val_batch)
rows = pq.ParquetFile(train_path).metadata.num_rows
used = rows if requested == "null" else min(rows, int(requested))
val_rows = pq.ParquetFile(val_path).metadata.num_rows
val_used = val_rows if val_requested == "null" else min(val_rows, int(val_requested))
steps_per_epoch = used // batch
if steps_per_epoch < 1:
    raise SystemExit(f"Training subset {used} is smaller than TRAIN_BATCH={batch}")
if val_used < val_batch:
    raise SystemExit(f"Validation subset {val_used} is smaller than VAL_BATCH={val_batch}")
print(steps_per_epoch * epochs)
PY
)
if [[ -z "$TOTAL_UPDATES" ]]; then
  TOTAL_UPDATES=$AVAILABLE_UPDATES
elif [[ ! "$TOTAL_UPDATES" =~ ^[1-9][0-9]*$ ]]; then
  echo "TOTAL_UPDATES must be a positive integer; got '$TOTAL_UPDATES'." >&2
  exit 2
fi
if (( TOTAL_UPDATES > AVAILABLE_UPDATES )); then
  echo "Requested $TOTAL_UPDATES updates, but this data/profile provides only $AVAILABLE_UPDATES." >&2
  exit 2
fi
# This pinned trainer starts at global_steps=1, increments after each update,
# and returns on >= total_training_steps. N updates therefore require N+1.
TRAINER_STOP_STEP=$((TOTAL_UPDATES + 1))
SAVE_FREQ=${SAVE_FREQ:-$TOTAL_UPDATES}
TEST_FREQ=${TEST_FREQ:--1}
if [[ ! "$SAVE_FREQ" =~ ^[1-9][0-9]*$ ]] || \
  (( SAVE_FREQ > TOTAL_UPDATES || TOTAL_UPDATES % SAVE_FREQ != 0 )); then
  echo "SAVE_FREQ must divide TOTAL_UPDATES and include the final update; got $SAVE_FREQ/$TOTAL_UPDATES." >&2
  exit 2
fi
if [[ ! "$TEST_FREQ" =~ ^-1$|^[1-9][0-9]*$ ]]; then
  echo "TEST_FREQ must be -1 or a positive integer; got '$TEST_FREQ'." >&2
  exit 2
fi

if [[ "$BACKEND" == e5 ]]; then
  ROLLOUT_GPU_MEMORY=${ROLLOUT_GPU_MEMORY:-0.50}
else
  ROLLOUT_GPU_MEMORY=${ROLLOUT_GPU_MEMORY:-0.55}
fi
if ! awk -v value="$ROLLOUT_GPU_MEMORY" 'BEGIN { exit !(value > 0 && value < 1) }'; then
  echo "ROLLOUT_GPU_MEMORY must be between 0 and 1; got '$ROLLOUT_GPU_MEMORY'." >&2
  exit 2
fi

# Standalone training supports repository IDs too, but always converts them to
# a concrete snapshot so completion markers cannot outlive mutable Hub refs.
BASE_MODEL=$(bash "$ROOT/scripts/resolve_hf_model.sh" \
  "$BASE_MODEL" "$BASE_MODEL_REVISION" "$SEARCH_R1_PYTHON")
SEARCH_R1_DIRTY_SHA=$(git -C "$SEARCH_R1" diff --no-ext-diff --binary HEAD | sha256sum | awk '{print $1}')

TRAIN_SIGNATURE=$("$SEARCH_R1_PYTHON" - \
  "$TRAIN_FILE" "$VAL_FILE" "$BASE_MODEL" "$BACKEND" "$PROFILE" \
  "$TRAIN_BATCH" "$VAL_BATCH" "$MINI_BATCH" "$MICRO_BATCH" \
  "$TRAIN_DATA_NUM" "$VAL_DATA_NUM" "$TOTAL_UPDATES" "$N_AGENT" \
  "$TOTAL_EPOCHS" "$SAVE_FREQ" "$TEST_FREQ" "$VAL_BEFORE_TRAIN" \
  "$ROLLOUT_GPU_MEMORY" "$SEARCH_R1_COMMIT" \
  "$ROOT/searchr1_stage2/searchr1-runtime.patch" "$INDEX_MANIFEST" \
  "$ROOT/searchr1_stage2/run_single_stack_grpo.sh" "$SEARCH_R1_DIRTY_SHA" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

(
    train_file,
    val_file,
    base_model,
    backend,
    profile,
    train_batch,
    val_batch,
    mini_batch,
    micro_batch,
    train_data_num,
    val_data_num,
    total_updates,
    n_agent,
    total_epochs,
    save_freq,
    test_freq,
    val_before_train,
    rollout_gpu_memory,
    search_r1_commit,
    runtime_patch,
    index_manifest,
    training_wrapper,
    search_r1_dirty_sha,
) = sys.argv[1:]

def digest(path: str) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

model_path = Path(base_model)
model_identity: dict = {"reference": base_model}
if model_path.is_dir():
    model_identity["resolved_path"] = str(model_path.resolve())
    model_identity["files"] = [
        {"name": path.name, "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        for pattern in ("config.json", "tokenizer*", "*.index.json", "*.safetensors", "*.bin")
        for path in sorted(model_path.glob(pattern))
        if path.is_file()
    ]

payload = {
    "schema": 3,
    "base_model": model_identity,
    "backend": backend,
    "profile": profile,
    "train_sha256": digest(train_file),
    "val_sha256": digest(val_file),
    "train_batch": int(train_batch),
    "val_batch": int(val_batch),
    "mini_batch": int(mini_batch),
    "micro_batch": int(micro_batch),
    "train_data_num": train_data_num,
    "val_data_num": val_data_num,
    "total_updates": int(total_updates),
    "n_agent": int(n_agent),
    "total_epochs": int(total_epochs),
    "save_freq": int(save_freq),
    "test_freq": int(test_freq),
    "val_before_train": val_before_train == "true",
    "rollout_gpu_memory": float(rollout_gpu_memory),
    "search_r1_commit": search_r1_commit,
    "search_r1_dirty_sha256": search_r1_dirty_sha,
    "runtime_patch_sha256": digest(runtime_patch),
    "index_manifest_sha256": digest(index_manifest),
    "training_wrapper_sha256": digest(training_wrapper),
}
canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
print(hashlib.sha256(canonical.encode()).hexdigest())
PY
)

validate_checkpoint() {
  local checkpoint=$1
  "$SEARCH_R1_PYTHON" - "$checkpoint" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
if not (root / "config.json").is_file():
    raise SystemExit(1)
index_files = sorted(root.glob("*.safetensors.index.json"))
if len(index_files) > 1:
    raise SystemExit(1)
if index_files:
    try:
        index = json.loads(index_files[0].read_text(encoding="utf-8"))
        weights = [root / name for name in sorted(set(index["weight_map"].values()))]
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        raise SystemExit(1)
else:
    weights = sorted(root.glob("*.safetensors")) or sorted(root.glob("*.bin"))
if not weights or any(
    not path.is_file() or path.stat().st_size < 1024 * 1024 for path in weights
):
    raise SystemExit(1)
if not (root / "tokenizer.json").is_file() and not (root / "tokenizer.model").is_file():
    raise SystemExit(1)
json.loads((root / "config.json").read_text(encoding="utf-8"))
PY
}

latest_checkpoint() {
  local candidate
  candidate="$CHECKPOINT_DIR/actor/global_step_$TOTAL_UPDATES"
  [[ -d "$candidate" ]] || return 1
  validate_checkpoint "$candidate" || return 1
  printf '%s\n' "$candidate"
}

# A prior interrupted training may have left Ray workers owning GPUs. This must
# also run on the completed-checkpoint fast path below.
"$ROOT/.venv-searchr1/bin/ray" stop --force >/dev/null 2>&1 || true

if [[ ${FORCE_TRAIN:-0} != 1 && -f "$COMPLETE_MARKER" ]]; then
  marker_signature=$("$SEARCH_R1_PYTHON" - "$COMPLETE_MARKER" <<'PY'
import json
import sys
from pathlib import Path

try:
    print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["training_signature"])
except (OSError, KeyError, TypeError, json.JSONDecodeError):
    print("")
PY
  )
  if [[ "$marker_signature" == "$TRAIN_SIGNATURE" ]] && completed_checkpoint=$(latest_checkpoint); then
    echo "Reusing completed $EXP checkpoint: $completed_checkpoint"
    exit 0
  fi
  echo "Completion marker does not match this training configuration; training will restart." >&2
fi

if [[ -e "$CHECKPOINT_DIR" ]]; then
  if [[ "$CHECKPOINT_DIR" != "$ROOT/work/checkpoints/"* ]]; then
    echo "Refusing to rotate a nonstandard existing CHECKPOINT_DIR: $CHECKPOINT_DIR" >&2
    echo "Choose a new directory or move it manually." >&2
    exit 1
  fi
  if [[ ${FORCE_TRAIN:-0} == 1 ]]; then
    backup_kind=previous
  else
    backup_kind=incomplete
  fi
  checkpoint_backup="${CHECKPOINT_DIR}.${backup_kind}.$(date +%Y%m%d-%H%M%S)"
  mv -- "$CHECKPOINT_DIR" "$checkpoint_backup"
  echo "Moved prior training output to: $checkpoint_backup"
fi
mkdir -p "$CHECKPOINT_DIR"

LOGGER="['console']"
if [[ ${ENABLE_WANDB:-0} == 1 ]]; then
  LOGGER="['console','wandb']"
else
  export WANDB_MODE=disabled
fi

ensure_local_no_proxy
export CUDA_VISIBLE_DEVICES=$GPU_IDS
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export SEARCH_R1_RETRIEVER_TIMEOUT=${SEARCH_R1_RETRIEVER_TIMEOUT:-120}
# Retriever models may still need to populate their own Hugging Face cache.
# Apply local-only flags later, after the selected retriever has started.
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

owns_retriever=0
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  "$ROOT/.venv-searchr1/bin/ray" stop --force >/dev/null 2>&1 || true
  if [[ $owns_retriever -eq 1 && ${KEEP_RETRIEVER:-0} != 1 ]]; then
    bash "$ROOT/scripts/stop_servers.sh" || true
  fi
  if [[ $status -ne 0 ]]; then
    echo "Search-R1 training failed; log: $LOG_FILE" >&2
    tail -n 100 "$LOG_FILE" >&2 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

"$ROOT/.venv-searchr1/bin/ray" stop --force >/dev/null 2>&1 || true
if [[ "$AUTO_LAUNCH_RETRIEVER" == 1 ]]; then
  bash "$ROOT/scripts/stop_servers.sh" || true
  E5_GPU=${E5_GPU:-7} RETRIEVER_BACKENDS="$BACKEND" \
    bash "$ROOT/scripts/launch_retrievers.sh"
  owns_retriever=1
fi

# Fail before loading eight model replicas if an externally managed retriever
# is absent or returns an invalid result.
probe_response=$(curl --noproxy '*' -fsS --connect-timeout 3 --max-time 120 \
  -X POST "http://127.0.0.1:${PORT}/retrieve" \
  -H 'Content-Type: application/json' \
  -d '{"queries":["Who wrote Hamlet?"],"topk":1,"return_scores":true}')
"$SEARCH_R1_PYTHON" -c \
  'import json,sys; p=json.loads(sys.argv[1]); assert p.get("result") and p["result"][0], p' \
  "$probe_response"

if [[ -d "$BASE_MODEL" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
else
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
fi

echo "Training $EXP: backend=$BACKEND profile=$PROFILE updates=$TOTAL_UPDATES GPUs=$GPU_IDS"
echo "Base model: $BASE_MODEL; checkpoints: $CHECKPOINT_DIR"

cd "$SEARCH_R1"
"$SEARCH_R1_PYTHON" -m verl.trainer.main_ppo \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_data_num="$TRAIN_DATA_NUM" data.val_data_num="$VAL_DATA_NUM" \
  data.train_batch_size="$TRAIN_BATCH" data.val_batch_size="$VAL_BATCH" \
  data.max_prompt_length=4096 data.max_response_length=500 \
  data.max_start_length=2048 data.max_obs_length=700 \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.model.path="$BASE_MODEL" \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.ppo_mini_batch_size="$MINI_BATCH" \
  actor_rollout_ref.actor.ppo_micro_batch_size="$MICRO_BATCH" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=16 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY" \
  actor_rollout_ref.ref.log_prob_micro_batch_size=16 \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  algorithm.no_think_rl=false \
  actor_rollout_ref.rollout.n_agent="$N_AGENT" \
  actor_rollout_ref.rollout.temperature=1 \
  actor_rollout_ref.actor.state_masking=true \
  trainer.logger="$LOGGER" \
  +trainer.val_only=false +trainer.val_before_train="$VAL_BEFORE_TRAIN" \
  trainer.n_gpus_per_node=8 trainer.nnodes=1 \
  trainer.save_freq="$SAVE_FREQ" trainer.test_freq="$TEST_FREQ" \
  trainer.project_name=StackAdaptPilot trainer.experiment_name="$EXP" \
  trainer.total_epochs="$TOTAL_EPOCHS" trainer.total_training_steps="$TRAINER_STOP_STEP" \
  trainer.default_local_dir="$CHECKPOINT_DIR" trainer.default_hdfs_dir=null \
  max_turns=3 retriever.url="http://127.0.0.1:${PORT}/retrieve" retriever.topk=10 \
  2>&1 | tee "$LOG_FILE"

completed_checkpoint=$(latest_checkpoint) || {
  echo "Training exited successfully but no valid final actor checkpoint was found." >&2
  exit 1
}
"$SEARCH_R1_PYTHON" - "$COMPLETE_MARKER" "$completed_checkpoint" "$EXP" "$BACKEND" "$PROFILE" "$TOTAL_UPDATES" "$TRAINER_STOP_STEP" "$TRAIN_SIGNATURE" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

marker, checkpoint, experiment, backend, profile, updates, stop_step, signature = sys.argv[1:]
payload = {
    "schema": 1,
    "experiment": experiment,
    "backend": backend,
    "profile": profile,
    "total_updates": int(updates),
    "trainer_stop_step": int(stop_step),
    "training_signature": signature,
    "checkpoint": str(Path(checkpoint).resolve()),
    "completed_at": datetime.now(timezone.utc).isoformat(),
}
marker_path = Path(marker)
temporary = marker_path.with_name(f"{marker_path.name}.tmp.{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, marker_path)
PY
echo "Completed $EXP: $completed_checkpoint"
