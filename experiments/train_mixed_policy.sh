#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

EXPERIMENT_ID=${EXPERIMENT_ID:?Set EXPERIMENT_ID=EXP-003 or EXP-004}
SEED=${SEED:?Set SEED, e.g. 13}
PROFILE=${PROFILE:-pilot}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}
BASE_MODEL_REVISION=${BASE_MODEL_REVISION:-aa8e72537993ba99e69dfaafa59ed015b17504d1}
TRAIN_GPUS=${TRAIN_GPUS:-0,1,2,3,4,5,6}
N_GPUS=${N_GPUS:-7}
E5_GPU=${E5_GPU:-7}
MIXED_PORT=${MIXED_PORT:-8200}
TOPK=${TOPK:-3}
N_AGENT=${N_AGENT:-4}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-100}
MAX_PROMPT_LENGTH=4096
MAX_RESPONSE_LENGTH=500
MAX_START_LENGTH=2048
MAX_OBS_LENGTH=500
MAX_TURNS=4

case "$EXPERIMENT_ID" in
  EXP-003) MIXED_MODE=blind; VARIANT=blind ;;
  EXP-004) MIXED_MODE=oracle; VARIANT=backend-id ;;
  *) echo "train_mixed_policy.sh supports EXP-003 and EXP-004; got $EXPERIMENT_ID" >&2; exit 2 ;;
esac
case "$PROFILE" in
  smoke)
    TOTAL_UPDATES=${TOTAL_UPDATES:-20}; TRAIN_BATCH=${TRAIN_BATCH:-28}; VAL_BATCH=${VAL_BATCH:-28}
    MINI_BATCH=${MINI_BATCH:-14}; MICRO_BATCH=${MICRO_BATCH:-7}; SAVE_FREQ=${SAVE_FREQ:-20}; TEST_FREQ=${TEST_FREQ:-20}
    ;;
  pilot)
    TOTAL_UPDATES=${TOTAL_UPDATES:-200}; TRAIN_BATCH=${TRAIN_BATCH:-112}; VAL_BATCH=${VAL_BATCH:-112}
    MINI_BATCH=${MINI_BATCH:-56}; MICRO_BATCH=${MICRO_BATCH:-7}; SAVE_FREQ=${SAVE_FREQ:-100}; TEST_FREQ=${TEST_FREQ:-100}
    ;;
  full)
    TOTAL_UPDATES=${TOTAL_UPDATES:-500}; TRAIN_BATCH=${TRAIN_BATCH:-112}; VAL_BATCH=${VAL_BATCH:-112}
    MINI_BATCH=${MINI_BATCH:-56}; MICRO_BATCH=${MICRO_BATCH:-7}; SAVE_FREQ=${SAVE_FREQ:-100}; TEST_FREQ=${TEST_FREQ:-100}
    ;;
  *) echo "PROFILE must be smoke, pilot, or full" >&2; exit 2 ;;
esac

PILOT_PYTHON=$ROOT/.venv-pilot/bin/python
SEARCH_R1_PYTHON=$ROOT/.venv-searchr1/bin/python
SEARCH_R1=$ROOT/upstream/Search-R1
for executable in "$PILOT_PYTHON" "$SEARCH_R1_PYTHON"; do
  [[ -x "$executable" ]] || { echo "Missing environment; run bootstrap scripts first" >&2; exit 1; }
done
if [[ "$N_GPUS" != 7 || "$TRAIN_GPUS" != "0,1,2,3,4,5,6" || "$E5_GPU" != 7 ]]; then
  echo "Numbered mixed experiments reserve GPUs 0-6 for GRPO and GPU 7 for E5" >&2
  exit 2
fi
if [[ ! "$SEED" =~ ^[1-9][0-9]*$ ]]; then
  echo "SEED must be a positive integer" >&2
  exit 2
fi

RUN_ID=$(
  "$PILOT_PYTHON" -m stackpilot.experiment_registry run-id "$EXPERIMENT_ID" \
    --seed "$SEED" --profile "$PROFILE" --variant "$VARIANT"
)
EXPERIMENT_ROOT=$ROOT/work/experiments/$EXPERIMENT_ID
CHECKPOINT_DIR=$EXPERIMENT_ROOT/checkpoints/$RUN_ID
LOG_DIR=$ROOT/logs/experiments/$EXPERIMENT_ID
LOG_FILE=$LOG_DIR/${RUN_ID}.log
COMPLETE_MARKER=$CHECKPOINT_DIR/.complete.json
mkdir -p "$LOG_DIR"

SOURCE_TRAIN=$ROOT/work/hard_rq0/searchr1/train.parquet
SOURCE_VAL=$ROOT/work/hard_rq0/searchr1/test.parquet
for path in "$SOURCE_TRAIN" "$SOURCE_VAL"; do
  [[ -s "$path" ]] || { echo "Missing hard-RQ0 data: $path" >&2; exit 1; }
done
if [[ "$MIXED_MODE" == oracle ]]; then
  DATA_DIR=$EXPERIMENT_ROOT/data/seed-${SEED}
  TRAIN_DATA=$DATA_DIR/train.parquet
  VAL_DATA=$DATA_DIR/test.parquet
  if [[ ! -s "$TRAIN_DATA" ]]; then
    "$PILOT_PYTHON" -m stackpilot.prepare_mixed_data \
      --input "$SOURCE_TRAIN" --output "$TRAIN_DATA" --seed "$SEED"
  fi
  if [[ ! -s "$VAL_DATA" ]]; then
    "$PILOT_PYTHON" -m stackpilot.prepare_mixed_data \
      --input "$SOURCE_VAL" --output "$VAL_DATA" --seed "$SEED"
  fi
else
  TRAIN_DATA=$SOURCE_TRAIN
  VAL_DATA=$SOURCE_VAL
fi

bash "$ROOT/scripts/bootstrap_searchr1.sh"
"$SEARCH_R1_PYTHON" "$ROOT/hard_rq0/patch_searchr1_seed.py" --search-r1-root "$SEARCH_R1"
"$SEARCH_R1_PYTHON" "$ROOT/hard_rq0/patch_searchr1_mixed.py" --search-r1-root "$SEARCH_R1"
bash "$ROOT/experiments/launch_mixed_router.sh"

BASE_MODEL=$(unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE; \
  bash "$ROOT/scripts/resolve_hf_model.sh" "$BASE_MODEL" "$BASE_MODEL_REVISION" "$SEARCH_R1_PYTHON")
TRAINER_STOP_STEP=$((TOTAL_UPDATES + 1))
LOG_PROB_MICRO_BATCH=${LOG_PROB_MICRO_BATCH:-14}
ROLLOUT_GPU_MEMORY=${ROLLOUT_GPU_MEMORY:-0.55}
LOGGER=${LOGGER:-"['console']"}

TRAIN_SIGNATURE=$(
  "$SEARCH_R1_PYTHON" - "$TRAIN_DATA" "$VAL_DATA" "$BASE_MODEL" \
    "$EXPERIMENT_ID" "$RUN_ID" "$MIXED_MODE" "$SEED" "$PROFILE" "$TOTAL_UPDATES" \
    "$ROOT/hard_rq0/patch_searchr1_mixed.py" "$ROOT/experiments/train_mixed_policy.sh" <<'PY'
import hashlib, json, sys
from pathlib import Path
train, val, model, experiment_id, run_id, mode, seed, profile, updates, patch, wrapper = sys.argv[1:]
def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()
payload = {
    'schema': 1,
    'experiment_id': experiment_id,
    'run_id': run_id,
    'mode': mode,
    'seed': int(seed),
    'profile': profile,
    'total_updates': int(updates),
    'train_sha256': digest(train),
    'val_sha256': digest(val),
    'model_path': str(Path(model).resolve()),
    'mixed_patch_sha256': digest(patch),
    'wrapper_sha256': digest(wrapper),
}
print(hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest())
PY
)

if [[ -f "$COMPLETE_MARKER" && ${FORCE_TRAIN:-0} != 1 ]]; then
  marker_signature=$(
    "$SEARCH_R1_PYTHON" -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["training_signature"])' \
      "$COMPLETE_MARKER" 2>/dev/null || true
  )
  if [[ "$marker_signature" == "$TRAIN_SIGNATURE" ]]; then
    echo "Reusing completed run: $RUN_ID"
    exit 0
  fi
  echo "Completion marker exists but does not match this configuration: $COMPLETE_MARKER" >&2
  exit 1
fi
if [[ -e "$CHECKPOINT_DIR" ]]; then
  if [[ ${FORCE_TRAIN:-0} != 1 ]]; then
    echo "Checkpoint directory already exists without a reusable marker: $CHECKPOINT_DIR" >&2
    exit 1
  fi
  mv "$CHECKPOINT_DIR" "${CHECKPOINT_DIR}.previous.$(date +%Y%m%d-%H%M%S)"
fi
mkdir -p "$CHECKPOINT_DIR"

source "$ROOT/scripts/lib/runtime.sh"
ensure_local_no_proxy
health=$(curl --noproxy '*' -fsS "http://127.0.0.1:${MIXED_PORT}/health")
"$PILOT_PYTHON" -c \
  'import json,sys; p=json.loads(sys.argv[1]); assert p.get("status")=="ok" and p.get("backend")=="mixed", p' \
  "$health"

export CUDA_VISIBLE_DEVICES=$TRAIN_GPUS
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export TOKENIZERS_PARALLELISM=false
export RQ0_SEED=$SEED
export PYTHONHASHSEED=$SEED
export SEARCH_R1_MIXED_MODE=$MIXED_MODE
export PYTHONPATH="$ROOT/hard_rq0:$SEARCH_R1:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export SEARCH_R1_RETRIEVER_TIMEOUT=${SEARCH_R1_RETRIEVER_TIMEOUT:-180}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=${WANDB_MODE:-disabled}

cleanup() {
  status=$?
  trap - EXIT INT TERM
  "$ROOT/.venv-searchr1/bin/ray" stop --force >/dev/null 2>&1 || true
  if [[ $status -ne 0 ]]; then
    echo "Mixed GRPO failed; log: $LOG_FILE" >&2
    tail -n 100 "$LOG_FILE" >&2 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

cd "$SEARCH_R1"
echo "Training $RUN_ID mode=$MIXED_MODE seed=$SEED updates=$TOTAL_UPDATES"
"$SEARCH_R1_PYTHON" -m verl.trainer.main_ppo \
  data.train_files="$TRAIN_DATA" data.val_files="$VAL_DATA" \
  data.train_batch_size="$TRAIN_BATCH" data.val_batch_size="$VAL_BATCH" \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.max_start_length="$MAX_START_LENGTH" data.max_obs_length="$MAX_OBS_LENGTH" \
  data.shuffle_train_dataloader=true algorithm.adv_estimator=grpo \
  actor_rollout_ref.model.path="$BASE_MODEL" \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
  actor_rollout_ref.actor.use_kl_loss=true actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.ppo_mini_batch_size="$MINI_BATCH" \
  actor_rollout_ref.actor.ppo_micro_batch_size="$MICRO_BATCH" \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.grad_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.rollout.log_prob_micro_batch_size="$LOG_PROB_MICRO_BATCH" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY" \
  actor_rollout_ref.rollout.n_agent="$N_AGENT" actor_rollout_ref.rollout.temperature=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size="$LOG_PROB_MICRO_BATCH" \
  actor_rollout_ref.ref.fsdp_config.param_offload=true actor_rollout_ref.actor.state_masking=true \
  algorithm.no_think_rl=false trainer.logger="$LOGGER" \
  +trainer.val_only=false +trainer.val_before_train=true \
  trainer.n_gpus_per_node="$N_GPUS" trainer.nnodes=1 \
  trainer.save_freq="$SAVE_FREQ" trainer.test_freq="$TEST_FREQ" \
  trainer.project_name=StackAdaptNumberedExperiments trainer.experiment_name="$RUN_ID" \
  trainer.total_epochs="$TOTAL_EPOCHS" trainer.total_training_steps="$TRAINER_STOP_STEP" \
  trainer.default_local_dir="$CHECKPOINT_DIR" trainer.default_hdfs_dir=null \
  max_turns="$MAX_TURNS" retriever.url="http://127.0.0.1:${MIXED_PORT}/retrieve" \
  retriever.topk="$TOPK" 2>&1 | tee "$LOG_FILE"

FINAL_CHECKPOINT=$CHECKPOINT_DIR/actor/global_step_$TOTAL_UPDATES
[[ -d "$FINAL_CHECKPOINT" && -s "$FINAL_CHECKPOINT/config.json" ]] || {
  echo "Exact final checkpoint is missing: $FINAL_CHECKPOINT" >&2; exit 1;
}
"$SEARCH_R1_PYTHON" - "$COMPLETE_MARKER" "$FINAL_CHECKPOINT" "$RUN_ID" \
  "$EXPERIMENT_ID" "$MIXED_MODE" "$SEED" "$PROFILE" "$TOTAL_UPDATES" "$TRAIN_SIGNATURE" <<'PY'
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path
marker, checkpoint, run_id, experiment_id, mode, seed, profile, updates, signature = sys.argv[1:]
payload = {
    'schema': 1,
    'experiment': run_id,
    'experiment_id': experiment_id,
    'routing_mode': mode,
    'seed': int(seed),
    'profile': profile,
    'total_updates': int(updates),
    'training_signature': signature,
    'checkpoint': str(Path(checkpoint).resolve()),
    'completed_at': datetime.now(timezone.utc).isoformat(),
}
path = Path(marker); path.parent.mkdir(parents=True, exist_ok=True)
tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
tmp.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
os.replace(tmp, path)
PY
echo "Completed $RUN_ID: $FINAL_CHECKPOINT"
