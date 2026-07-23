#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
SEEDS=${SEEDS:-"42"}
BACKEND_LIST=${BACKEND_LIST:-"bm25 e5"}
if [[ -z ${LIMIT:-} ]]; then
  if [[ "$PROFILE" == smoke ]]; then LIMIT=20; else LIMIT=500; fi
fi
TOPKS=${TOPKS:-"3 5 10"}
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
export ANSWER_REWARD_WEIGHT=${ANSWER_REWARD_WEIGHT:-1.0}
export EVIDENCE_REWARD_WEIGHT=${EVIDENCE_REWARD_WEIGHT:-0.5}
export SEARCH_COST_WEIGHT=${SEARCH_COST_WEIGHT:-0.02}

if [[ ${NUMBERED_SETUP_READY:-0} != 1 ]]; then
  bash scripts/bootstrap.sh
  bash scripts/bootstrap_vllm.sh
  bash scripts/bootstrap_searchr1.sh
  bash hard_rq0/download_assets.sh
  bash hard_rq0/prepare_data.sh
fi
bash experiments/ensure_retrievers.sh

.venv-pilot/bin/python - \
  "$ANSWER_REWARD_WEIGHT" "$EVIDENCE_REWARD_WEIGHT" "$SEARCH_COST_WEIGHT" <<'PY'
import math, sys
names = ("ANSWER_REWARD_WEIGHT", "EVIDENCE_REWARD_WEIGHT", "SEARCH_COST_WEIGHT")
for name, value in zip(names, sys.argv[1:]):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise SystemExit(f"{name} must be finite and non-negative; got {value!r}")
PY
reward_suffix="a${ANSWER_REWARD_WEIGHT}-e${EVIDENCE_REWARD_WEIGHT}-c${SEARCH_COST_WEIGHT}"

for backend in $BACKEND_LIST; do
  for seed in $SEEDS; do
    bash experiments/reset_searchr1_experiment_files.sh
    if [[ ${NUMBERED_SETUP_READY:-0} != 1 ]]; then
      bash scripts/bootstrap_searchr1.sh
    fi
    .venv-searchr1/bin/python hard_rq0/patch_searchr1_evidence_reward.py \
      --search-r1-root "${SEARCH_R1_ROOT:-upstream/Search-R1}"
    variant="${backend}-evidence-${reward_suffix}"
    run_id=$(
      .venv-pilot/bin/python -m stackpilot.experiment_registry run-id EXP-005 \
        --seed "$seed" --profile "$PROFILE" --variant "$variant"
    )
    trainer_checkpoint="$ROOT/work/hard_rq0/checkpoints/$run_id"
    numbered_checkpoint="$ROOT/work/experiments/EXP-005/checkpoints/$run_id"
    EXP="$run_id" \
    CHECKPOINT_DIR="$trainer_checkpoint" \
    LOG_FILE="$ROOT/logs/experiments/EXP-005/${run_id}.log" \
    BACKEND="$backend" SEED="$seed" PROFILE="$PROFILE" \
    BASE_MODEL="$BASE_MODEL" BASE_MODEL_REVISION="$BASE_MODEL_REVISION" \
      bash hard_rq0/train_specialist.sh

    mkdir -p "$(dirname "$numbered_checkpoint")"
    if [[ -e "$numbered_checkpoint" && ! -L "$numbered_checkpoint" ]]; then
      echo "Refusing to replace non-symlink numbered checkpoint: $numbered_checkpoint" >&2
      exit 1
    fi
    ln -sfn "$trainer_checkpoint" "$numbered_checkpoint"

    EXPERIMENT_ID=EXP-005 SEED="$seed" PROFILE="$PROFILE" VARIANT="$variant" \
      RUN_ID="$run_id" bash experiments/merge_numbered_checkpoint.sh
    EXPERIMENT_ID=EXP-005 TAG="evidence-${backend}" SEED="$seed" \
      PROFILE="$PROFILE" VARIANT="$variant" RUN_ID="$run_id" \
      MODEL_REF="$ROOT/work/experiments/EXP-005/merged/$run_id" \
      LIMIT="$LIMIT" TOPKS="$TOPKS" BACKENDS="bm25 e5" \
      bash experiments/eval_numbered_policy.sh
  done
done

echo "EXP-005 complete: work/experiments/EXP-005"
