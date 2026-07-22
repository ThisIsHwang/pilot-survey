#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
PROFILE=${PROFILE:-pilot}
RESULT_SET=${RESULT_SET:-$PROFILE}
SEEDS=${SEEDS:-"13 42 87"}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}
LIMIT=${LIMIT:-}

curl --noproxy '*' -fsS http://127.0.0.1:8101/health >/dev/null || {
  echo "Run hard_rq0/launch_retrievers.sh first" >&2
  exit 1
}
curl --noproxy '*' -fsS http://127.0.0.1:8102/health >/dev/null || {
  echo "Run hard_rq0/launch_retrievers.sh first" >&2
  exit 1
}

for backend in bm25 e5; do
  for seed in $SEEDS; do
    echo "=== train ${backend} specialist, seed ${seed}, profile ${PROFILE} ==="
    BACKEND=$backend SEED=$seed PROFILE=$PROFILE BASE_MODEL=$BASE_MODEL \
      bash "$ROOT/hard_rq0/train_specialist.sh"

    BACKEND=$backend SEED=$seed PROFILE=$PROFILE \
      bash "$ROOT/hard_rq0/merge_specialist.sh"

    exp=hard-rq0-${backend}-seed${seed}-${PROFILE}
    model_path=$ROOT/work/hard_rq0/merged/$exp
    eval_args=(
      TAG=${backend}-specialist
      SEED=$seed
      MODEL_PATH=$model_path
      RESULT_SET=$RESULT_SET
    )
    if [[ -n "$LIMIT" ]]; then
      eval_args+=(LIMIT=$LIMIT)
    fi
    env "${eval_args[@]}" bash "$ROOT/hard_rq0/eval_policy.sh"
  done
done
