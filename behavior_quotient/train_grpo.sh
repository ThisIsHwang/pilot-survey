#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

EXPERIMENT_ID=${EXPERIMENT_ID:-EXP-027}
PROFILE=${PROFILE:-pilot}
SEED=${SEED:?Set SEED}
BACKEND=${BACKEND:?Set BACKEND=bm25 or e5}
METHOD=${METHOD:?Set METHOD=standard, random-surface, balanced-surface, random-quotient, or balanced-quotient}
CONFIG=${CONFIG:-$ROOT/configs/behavior_quotient.yaml}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}
BASE_MODEL_REVISION=${BASE_MODEL_REVISION:-a09a35458c702b33eeacc393d103063234e8bc28}
TRAIN_GPUS=${TRAIN_GPUS:-0,1,2,3,4,5,6}
N_GPUS=${N_GPUS:-7}
E5_GPU=${E5_GPU:-7}

if [[ ${BQ_SETUP_READY:-0} != 1 ]]; then
  bash scripts/bootstrap.sh
  bash scripts/bootstrap_searchr1.sh
  bash hard_rq0/download_assets.sh
  bash hard_rq0/prepare_data.sh
fi

PILOT_PYTHON=$ROOT/.venv-pilot/bin/python
SEARCH_R1_PYTHON=$ROOT/.venv-searchr1/bin/python
SEARCH_R1=${SEARCH_R1_ROOT:-$ROOT/upstream/Search-R1}
for executable in "$PILOT_PYTHON" "$SEARCH_R1_PYTHON"; do
  [[ -x "$executable" ]] || { echo "Missing environment after bootstrap" >&2; exit 1; }
done
[[ -f "$CONFIG" ]] || { echo "Missing config: $CONFIG" >&2; exit 1; }

# Load only scalar preregistered settings; shlex.quote prevents YAML values from
# becoming shell syntax.
eval "$("$PILOT_PYTHON" - "$CONFIG" "$PROFILE" "$METHOD" <<'PY'
import shlex
import sys
import yaml
config, profile_name, method = sys.argv[1:]
cfg = yaml.safe_load(open(config, encoding='utf-8'))
if profile_name not in cfg['profiles']:
    raise SystemExit(f'unknown profile: {profile_name}')
if method not in cfg['training']['variants']:
    raise SystemExit(f'unknown method: {method}')
profile = cfg['profiles'][profile_name]
training = cfg['training']
variant = training['variants'][method]
values = {
    'TOTAL_UPDATES': profile['total_updates'],
    'TRAIN_BATCH': profile['train_batch'],
    'VAL_BATCH': profile['val_batch'],
    'MINI_BATCH': profile['mini_batch'],
    'MICRO_BATCH': profile['micro_batch'],
    'SAVE_FREQ': profile['save_freq'],
    'TEST_FREQ': profile['test_freq'],
    'N_AGENT': training['n_agent'],
    'TOPK': training['topk'],
    'MAX_TURNS': training['max_turns'],
    'MAX_PROMPT_LENGTH': training['max_prompt_length'],
    'MAX_RESPONSE_LENGTH': training['max_response_length'],
    'MAX_START_LENGTH': training['max_start_length'],
    'MAX_OBS_LENGTH': training['max_obs_length'],
    'LEARNING_RATE': training['learning_rate'],
    'ROLLOUT_GPU_MEMORY': training['rollout_gpu_memory'],
    'BQ_ADVANTAGE_MODE': variant['advantage_mode'],
    'BQ_SELECTION_MODE': variant['selection_mode'],
    'BQ_UPDATE_PER_PROMPT': variant['update_per_prompt'],
}
for key, value in values.items():
    print(f'{key}={shlex.quote(str(value))}')
PY
)"

if [[ "$BACKEND" == bm25 ]]; then
  RETRIEVER_PORT=${BM25_PORT:-8101}
elif [[ "$BACKEND" == e5 ]]; then
  RETRIEVER_PORT=${E5_PORT:-8102}
else
  echo "BACKEND must be bm25 or e5" >&2
  exit 2
fi
for value_name in SEED TOTAL_UPDATES TRAIN_BATCH VAL_BATCH MINI_BATCH MICRO_BATCH N_AGENT TOPK; do
  value=${!value_name}
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "$value_name must be a positive integer" >&2; exit 2; }
done
if [[ "$N_GPUS" != 7 || "$TRAIN_GPUS" != "0,1,2,3,4,5,6" || "$E5_GPU" != 7 ]]; then
  echo "BQ-GRPO reserves GPUs 0-6 for Search-R1 and GPU 7 for E5" >&2
  exit 2
fi
if (( TRAIN_BATCH % N_GPUS || VAL_BATCH % N_GPUS || MINI_BATCH % N_GPUS || MICRO_BATCH % N_GPUS )); then
  echo "All batch sizes must be divisible by N_GPUS=$N_GPUS" >&2
  exit 2
fi
if (( (TRAIN_BATCH * N_AGENT) % MINI_BATCH )); then
  echo "TRAIN_BATCH*N_AGENT must be divisible by MINI_BATCH" >&2
  exit 2
fi
if (( BQ_UPDATE_PER_PROMPT > N_AGENT )); then
  echo "update_per_prompt cannot exceed n_agent" >&2
  exit 2
fi

export E5_GPU
if [[ ${SKIP_RETRIEVER_LAUNCH:-0} != 1 ]]; then
  bash hard_rq0/launch_retrievers.sh
fi
bash experiments/reset_searchr1_experiment_files.sh
"$SEARCH_R1_PYTHON" hard_rq0/patch_searchr1_seed.py --search-r1-root "$SEARCH_R1"
"$SEARCH_R1_PYTHON" hard_rq0/patch_searchr1_worker_cuda.py --search-r1-root "$SEARCH_R1"
"$SEARCH_R1_PYTHON" hard_rq0/patch_searchr1_validation.py --search-r1-root "$SEARCH_R1"
"$SEARCH_R1_PYTHON" hard_rq0/patch_searchr1_action_protocol.py --search-r1-root "$SEARCH_R1"
"$SEARCH_R1_PYTHON" hard_rq0/patch_searchr1_observation_geometry.py --search-r1-root "$SEARCH_R1"
"$SEARCH_R1_PYTHON" hard_rq0/patch_searchr1_reward_protocol.py --search-r1-root "$SEARCH_R1"
"$SEARCH_R1_PYTHON" hard_rq0/patch_searchr1_behavior_quotient.py --search-r1-root "$SEARCH_R1"
if [[ ${STACKPILOT_CR_PATCH:-0} == 1 ]]; then
  "$SEARCH_R1_PYTHON" hard_rq0/patch_searchr1_credit_routing.py --search-r1-root "$SEARCH_R1"
fi

TRAIN_DATA=$ROOT/work/hard_rq0/searchr1/train.parquet
VAL_DATA=$ROOT/work/hard_rq0/searchr1/dev.parquet
for path in "$TRAIN_DATA" "$VAL_DATA"; do
  [[ -s "$path" ]] || { echo "Missing Hard-RQ0 data: $path" >&2; exit 1; }
done
BASE_MODEL=$(unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE; \
  bash scripts/resolve_hf_model.sh "$BASE_MODEL" "$BASE_MODEL_REVISION" "$SEARCH_R1_PYTHON")

VARIANT="${BACKEND}-${METHOD}"
RUN_ID=$("$PILOT_PYTHON" -m stackpilot.experiment_registry run-id "$EXPERIMENT_ID" \
  --seed "$SEED" --profile "$PROFILE" --variant "$VARIANT")
EXPERIMENT_ROOT=$ROOT/work/experiments/$EXPERIMENT_ID
CHECKPOINT_DIR=$EXPERIMENT_ROOT/checkpoints/$RUN_ID
TELEMETRY_PATH=$ROOT/work/behavior_quotient/telemetry/$PROFILE/$RUN_ID/telemetry.jsonl
LOG_DIR=$ROOT/logs/experiments/$EXPERIMENT_ID
LOG_FILE=$LOG_DIR/${RUN_ID}.log
CONTRACT=$CHECKPOINT_DIR/training_contract.json
COMPLETE_MARKER=$CHECKPOINT_DIR/.complete.json
mkdir -p "$CHECKPOINT_DIR" "$(dirname "$TELEMETRY_PATH")" "$LOG_DIR"

"$PILOT_PYTHON" - "$ROOT" "$SEARCH_R1" "$CONFIG" "$TRAIN_DATA" "$VAL_DATA" \
  "$BASE_MODEL" "$CONTRACT" "$EXPERIMENT_ID" "$RUN_ID" "$PROFILE" "$SEED" \
  "$BACKEND" "$METHOD" "$TOTAL_UPDATES" "$TRAIN_BATCH" "$VAL_BATCH" \
  "$MINI_BATCH" "$MICRO_BATCH" "$N_AGENT" "$TOPK" "$MAX_TURNS" \
  "$MAX_PROMPT_LENGTH" "$MAX_RESPONSE_LENGTH" "$MAX_START_LENGTH" \
  "$MAX_OBS_LENGTH" "$LEARNING_RATE" "$ROLLOUT_GPU_MEMORY" \
  "$BQ_ADVANTAGE_MODE" "$BQ_SELECTION_MODE" "$BQ_UPDATE_PER_PROMPT" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
(
    root_text, search_r1_text, config, train, val, model, output,
    experiment_id, run_id, profile, seed, backend, method, updates,
    train_batch, val_batch, mini_batch, micro_batch, n_agent, topk, max_turns,
    max_prompt, max_response, max_start, max_obs, learning_rate, rollout_memory,
    advantage_mode, selection_mode, update_k,
) = sys.argv[1:]
root = Path(root_text).resolve()
search_r1 = Path(search_r1_text).resolve()

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def model_identity(path):
    model_root = Path(path).resolve()
    files = {}
    for pattern in ('config.json', 'tokenizer*', '*.index.json', '*.safetensors', '*.bin'):
        for item in sorted(model_root.glob(pattern)):
            if item.is_file():
                files[item.name] = {
                    'size': item.stat().st_size,
                    'mtime_ns': item.stat().st_mtime_ns,
                }
    if not files:
        raise SystemExit(f'no model artifacts under {model_root}')
    return {'path': str(model_root), 'files': files}

code_files = [
    root / 'behavior_quotient/train_grpo.sh',
    root / 'stackpilot/behavior_quotient_runtime.py',
    root / 'hard_rq0/patch_searchr1_behavior_quotient.py',
    root / 'hard_rq0/patch_searchr1_action_protocol.py',
    root / 'hard_rq0/patch_searchr1_reward_protocol.py',
]
search_r1_diff = subprocess.run(
    ['git', '-C', str(search_r1), 'diff', '--binary', 'HEAD'],
    check=True,
    stdout=subprocess.PIPE,
).stdout
payload = {
    'schema': 2,
    'experiment_id': experiment_id,
    'run_id': run_id,
    'profile': profile,
    'seed': int(seed),
    'backend': backend,
    'method': method,
    'parameters': {
        'updates': int(updates),
        'train_batch': int(train_batch),
        'val_batch': int(val_batch),
        'mini_batch': int(mini_batch),
        'micro_batch': int(micro_batch),
        'n_agent': int(n_agent),
        'topk': int(topk),
        'max_turns': int(max_turns),
        'max_prompt': int(max_prompt),
        'max_response': int(max_response),
        'max_start': int(max_start),
        'max_obs': int(max_obs),
        'learning_rate': str(learning_rate),
        'rollout_memory': float(rollout_memory),
        'advantage_mode': advantage_mode,
        'selection_mode': selection_mode,
        'update_per_prompt': int(update_k),
        'signature_mode': 'trajectory-ranked',
    },
    'config_sha256': digest(config),
    'train_sha256': digest(train),
    'val_sha256': digest(val),
    'model': model_identity(model),
    'code_sha256': {str(path.relative_to(root)): digest(path) for path in code_files},
    'search_r1_diff_sha256': hashlib.sha256(search_r1_diff).hexdigest(),
}
path = Path(output)
if path.exists():
    previous = json.loads(path.read_text(encoding='utf-8'))
    if previous != payload:
        raise SystemExit(f'stale BQ training contract: {path}')
else:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)
PY
if [[ -f "$COMPLETE_MARKER" ]]; then
  echo "BQ-GRPO already complete: $RUN_ID"
  exit 0
fi
rm -f "$TELEMETRY_PATH"

export CUDA_VISIBLE_DEVICES="$TRAIN_GPUS"
export PYTHONPATH="$ROOT:$SEARCH_R1:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
export STACKPILOT_EXPERIMENT_ID="$EXPERIMENT_ID"
export STACKPILOT_EXPERIMENT_VARIANT="$VARIANT"
export STACKPILOT_BQ_BACKEND="$BACKEND"
export STACKPILOT_BQ_RUN_SEED="$SEED"
export STACKPILOT_BQ_ADVANTAGE_MODE="$BQ_ADVANTAGE_MODE"
export STACKPILOT_BQ_SELECTION_MODE="$BQ_SELECTION_MODE"
export STACKPILOT_BQ_UPDATE_PER_PROMPT="$BQ_UPDATE_PER_PROMPT"
export STACKPILOT_BQ_SIGNATURE_MODE=trajectory-ranked
export STACKPILOT_BQ_SELECTION_SEED="$SEED"
export STACKPILOT_BQ_TELEMETRY_PATH="$TELEMETRY_PATH"
export SEARCH_R1_REWARD_MODE=answer
RETRIEVER_URL=${RETRIEVER_URL_OVERRIDE:-http://127.0.0.1:${RETRIEVER_PORT}/retrieve}

cd "$SEARCH_R1"
"$SEARCH_R1_PYTHON" -m verl.trainer.main_ppo \
  data.train_files="$TRAIN_DATA" \
  data.val_files="$VAL_DATA" \
  data.train_batch_size="$TRAIN_BATCH" \
  data.val_batch_size="$VAL_BATCH" \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.max_start_length="$MAX_START_LENGTH" \
  data.max_obs_length="$MAX_OBS_LENGTH" \
  data.shuffle_train_dataloader=true \
  algorithm.adv_estimator=grpo \
  algorithm.no_think_rl=false \
  actor_rollout_ref.model.path="$BASE_MODEL" \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.actor.optim.lr="$LEARNING_RATE" \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.ppo_mini_batch_size="$MINI_BATCH" \
  actor_rollout_ref.actor.ppo_micro_batch_size="$MICRO_BATCH" \
  actor_rollout_ref.actor.state_masking=true \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=14 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY" \
  actor_rollout_ref.rollout.n_agent="$N_AGENT" \
  actor_rollout_ref.rollout.temperature=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size=14 \
  trainer.logger="['console']" \
  +trainer.val_only=false \
  +trainer.val_before_train=true \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.nnodes=1 \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.project_name=BehaviorQuotientRL \
  trainer.experiment_name="$RUN_ID" \
  trainer.total_epochs=999 \
  trainer.total_training_steps="$TOTAL_UPDATES" \
  trainer.default_local_dir="$CHECKPOINT_DIR" \
  max_turns="$MAX_TURNS" \
  retriever.url="$RETRIEVER_URL" \
  retriever.topk="$TOPK" \
  2>&1 | tee "$LOG_FILE"

cd "$ROOT"
"$PILOT_PYTHON" - "$CHECKPOINT_DIR" "$TELEMETRY_PATH" "$COMPLETE_MARKER" "$CONTRACT" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
checkpoint_root, telemetry_text, marker_text, contract_text = sys.argv[1:]
root = Path(checkpoint_root)
telemetry = Path(telemetry_text)
marker = Path(marker_text)
contract = Path(contract_text)
checkpoints = sorted(root.glob('actor/global_step_*'))
if not checkpoints:
    raise SystemExit(f'no actor checkpoint found under {root}')
if not telemetry.is_file() or telemetry.stat().st_size == 0:
    raise SystemExit(f'empty behavior telemetry: {telemetry}')
payload = {
    'schema': 1,
    'completed_at': datetime.now(timezone.utc).isoformat(),
    'latest_checkpoint': str(checkpoints[-1].resolve()),
    'telemetry_path': str(telemetry.resolve()),
    'telemetry_sha256': hashlib.sha256(telemetry.read_bytes()).hexdigest(),
    'training_contract_sha256': hashlib.sha256(contract.read_bytes()).hexdigest(),
}
temporary = marker.with_suffix('.tmp')
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
os.replace(temporary, marker)
PY

echo "Completed $RUN_ID"
