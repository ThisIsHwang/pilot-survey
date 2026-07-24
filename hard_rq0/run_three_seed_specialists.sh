#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
RESULT_SET=${RESULT_SET:-$PROFILE}
SEEDS=${SEEDS:-"13 42 87"}
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
LIMIT=${LIMIT:-}
BM25_PORT=${BM25_PORT:-8101}
E5_PORT=${E5_PORT:-8102}
E5_GPU=${E5_GPU:-7}
BACKENDS=${BACKENDS:-"bm25 e5"}
TOPKS=${TOPKS:-"3 5 10"}

case "$PROFILE" in smoke|pilot|full) ;; *)
  echo "PROFILE must be smoke, pilot, or full; got '$PROFILE'." >&2; exit 2 ;;
esac
[[ "$RESULT_SET" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Invalid RESULT_SET=$RESULT_SET" >&2; exit 2; }
if [[ -n "$LIMIT" && ! "$LIMIT" =~ ^[1-9][0-9]*$ ]]; then
  echo "LIMIT must be empty or a positive integer; got '$LIMIT'." >&2
  exit 2
fi
read -r -a seed_args <<< "$SEEDS"
if [[ ${#seed_args[@]} -eq 0 ]]; then echo "SEEDS must not be empty." >&2; exit 2; fi
declare -A seen=()
for seed in "${seed_args[@]}"; do
  if [[ ! "$seed" =~ ^[1-9][0-9]*$ || -n ${seen[$seed]+x} ]]; then
    echo "SEEDS must contain unique positive integers; got '$SEEDS'." >&2
    exit 2
  fi
  seen[$seed]=1
done

BASE_MODEL=$(unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE; \
  bash "$ROOT/scripts/resolve_hf_model.sh" \
    "$BASE_MODEL" "$BASE_MODEL_REVISION" "$ROOT/.venv-pilot/bin/python")

BM25_PORT="$BM25_PORT" E5_PORT="$E5_PORT" E5_GPU="$E5_GPU" \
  bash "$ROOT/hard_rq0/ensure_retrievers.sh"

for backend in bm25 e5; do
  if [[ "$backend" == bm25 ]]; then port=$BM25_PORT; else port=$E5_PORT; fi
  for seed in "${seed_args[@]}"; do
    echo "=== train ${backend} specialist, seed ${seed}, profile ${PROFILE} ==="
    BACKEND=$backend PORT=$port SEED=$seed PROFILE=$PROFILE \
      BASE_MODEL=$BASE_MODEL BASE_MODEL_REVISION=$BASE_MODEL_REVISION \
      SEARCH_R1_REWARD_MODE=answer \
      bash "$ROOT/hard_rq0/train_specialist.sh"

    BACKEND=$backend SEED=$seed PROFILE=$PROFILE \
      bash "$ROOT/hard_rq0/merge_specialist.sh"

    exp=hard-rq0-${backend}-seed${seed}-${PROFILE}
    model_path=$ROOT/work/hard_rq0/merged/$exp
    eval_args=(
      "TAG=${backend}-specialist"
      "SEED=$seed"
      "MODEL_REF=$model_path"
      "RESULT_SET=$RESULT_SET"
      "BM25_PORT=$BM25_PORT"
      "E5_PORT=$E5_PORT"
      "SPECIALIST_SEEDS=$SEEDS"
      "BACKENDS=$BACKENDS"
      "TOPKS=$TOPKS"
    )
    if [[ -n "$LIMIT" ]]; then eval_args+=("LIMIT=$LIMIT"); fi
    env "${eval_args[@]}" bash "$ROOT/hard_rq0/eval_policy.sh"
  done
done
