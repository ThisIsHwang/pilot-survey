#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
export HF_HOME=${HF_HOME:-$ROOT/.cache/huggingface}
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

PYTHON=$ROOT/.venv-pilot/bin/python
[[ -x "$PYTHON" ]] || {
  echo "Missing .venv-pilot; run scripts/bootstrap.sh first." >&2
  exit 1
}
MODE=${1:---all}
case "$MODE" in
  --all|--stage2|--hard) ;;
  *)
    echo "Usage: bash scripts/prefetch_future_models.sh [--all|--stage2|--hard]" >&2
    exit 2
    ;;
esac

DEFAULT_BASE_MODEL=Qwen/Qwen2.5-3B-Instruct
DEFAULT_BASE_REVISION=aa8e72537993ba99e69dfaafa59ed015b17504d1
DEFAULT_OFFICIAL_MODEL=PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo-v0.3
DEFAULT_OFFICIAL_REVISION=4b8abb9beb4efc748019e956266ada547e276381
DEFAULT_E5_MODEL=intfloat/e5-base-v2
DEFAULT_E5_REVISION=f52bf8ec8c7124536f0efb74aca902b2995e5bcd
DEFAULT_QUERY_MODEL=sentence-transformers/all-MiniLM-L6-v2
DEFAULT_QUERY_REVISION=1110a243fdf4706b3f48f1d95db1a4f5529b4d41

default_revision() {
  local ref=$1
  local default_ref=$2
  local pinned_revision=$3
  if [[ "$ref" == "$default_ref" ]]; then
    printf '%s\n' "$pinned_revision"
  else
    printf '%s\n' main
  fi
}

declare -A SEEN_MODELS=()
prefetch_model() {
  local label=$1
  local ref=$2
  local revision=$3
  local key="${ref}@${revision}"
  if [[ -n ${SEEN_MODELS[$key]+x} ]]; then
    echo "Reusing prefetch request for $label: $key"
    return
  fi
  SEEN_MODELS[$key]=1
  echo "Prefetching $label: $key"
  bash "$ROOT/scripts/resolve_hf_model.sh" "$ref" "$revision" "$PYTHON" >/dev/null
}

base_policy_model=${BASE_POLICY_MODEL:-$DEFAULT_BASE_MODEL}
base_policy_revision=${BASE_POLICY_MODEL_REVISION:-$(
  default_revision "$base_policy_model" "$DEFAULT_BASE_MODEL" "$DEFAULT_BASE_REVISION"
)}
official_model=${OFFICIAL_POLICY_MODEL:-$DEFAULT_OFFICIAL_MODEL}
official_revision=${OFFICIAL_POLICY_MODEL_REVISION:-$(
  default_revision \
    "$official_model" "$DEFAULT_OFFICIAL_MODEL" "$DEFAULT_OFFICIAL_REVISION"
)}
train_model=${TRAIN_BASE_MODEL:-$DEFAULT_BASE_MODEL}
train_revision=${TRAIN_BASE_MODEL_REVISION:-$(
  default_revision "$train_model" "$DEFAULT_BASE_MODEL" "$DEFAULT_BASE_REVISION"
)}
e5_model=${E5_MODEL:-$DEFAULT_E5_MODEL}
e5_revision=${E5_MODEL_REVISION:-$(
  default_revision "$e5_model" "$DEFAULT_E5_MODEL" "$DEFAULT_E5_REVISION"
)}

if [[ "$MODE" == --all || "$MODE" == --stage2 ]]; then
  prefetch_model "Stage-2 base policy" "$base_policy_model" "$base_policy_revision"
  prefetch_model \
    "Stage-2 official policy" "$official_model" "$official_revision"
  prefetch_model "Stage-2 training base" "$train_model" "$train_revision"
  prefetch_model "E5 encoder" "$e5_model" "$e5_revision"
fi

if [[ "$MODE" == --all || "$MODE" == --hard ]]; then
  if [[ -n ${HARD_BASE_MODEL_REF:-} ]]; then
    hard_model=$HARD_BASE_MODEL_REF
    hard_revision=${HARD_BASE_MODEL_REVISION:-$(
      default_revision \
        "$hard_model" "$DEFAULT_BASE_MODEL" "$DEFAULT_BASE_REVISION"
    )}
  elif [[ -n ${TRAIN_BASE_MODEL:-} ]]; then
    hard_model=$TRAIN_BASE_MODEL
    if [[ -n ${HARD_BASE_MODEL_REVISION:-} ]]; then
      hard_revision=$HARD_BASE_MODEL_REVISION
    elif [[ -n ${TRAIN_BASE_MODEL_REVISION:-} ]]; then
      hard_revision=$TRAIN_BASE_MODEL_REVISION
    else
      hard_revision=$(
        default_revision \
          "$hard_model" "$DEFAULT_BASE_MODEL" "$DEFAULT_BASE_REVISION"
      )
    fi
  else
    hard_model=$DEFAULT_BASE_MODEL
    hard_revision=${HARD_BASE_MODEL_REVISION:-$DEFAULT_BASE_REVISION}
  fi
  query_model=${QUERY_MODEL:-$DEFAULT_QUERY_MODEL}
  query_revision=${QUERY_MODEL_REVISION:-$(
    default_revision \
      "$query_model" "$DEFAULT_QUERY_MODEL" "$DEFAULT_QUERY_REVISION"
  )}
  prefetch_model "hard-RQ0 training/evaluation base" "$hard_model" "$hard_revision"
  prefetch_model "hard-RQ0 report encoder" "$query_model" "$query_revision"
fi

echo "$MODE model prefetch complete; consumers will revalidate every snapshot."
