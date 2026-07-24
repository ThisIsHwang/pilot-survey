#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
SEEDS=${SEEDS:-"13 42 87"}
ORACLE_SEEDS=${ORACLE_SEEDS:-"42"}
if [[ -z ${LIMIT:-} ]]; then
  if [[ "$PROFILE" == smoke ]]; then LIMIT=20; else LIMIT=500; fi
fi
TOPKS=${TOPKS:-"3 5 10"}
DEFAULT_BASE_MODEL_REF=Qwen/Qwen2.5-3B-Instruct
DEFAULT_BASE_MODEL_REVISION=aa8e72537993ba99e69dfaafa59ed015b17504d1
BASE_MODEL_REF=${BASE_MODEL_REF:-$DEFAULT_BASE_MODEL_REF}
if [[ -z ${BASE_MODEL_REVISION:-} ]]; then
  if [[ "$BASE_MODEL_REF" == "$DEFAULT_BASE_MODEL_REF" ]]; then
    BASE_MODEL_REVISION=$DEFAULT_BASE_MODEL_REVISION
  else
    BASE_MODEL_REVISION=main
  fi
fi
REQUIRE_ALL=${REQUIRE_ALL:-1}
EXP002_ROOT=${EXP002_ROOT:-$ROOT/work/hard_rq0}
EXP002_RESULT_SET=${EXP002_RESULT_SET:-$PROFILE}
EXP002_COMPLETE_MARKER=${EXP002_COMPLETE_MARKER:-$EXP002_ROOT/runs/$EXP002_RESULT_SET/.complete.json}

if [[ ${NUMBERED_SETUP_READY:-0} != 1 ]]; then
  bash scripts/bootstrap.sh
  bash scripts/bootstrap_vllm.sh
  bash hard_rq0/download_assets.sh
  bash hard_rq0/prepare_data.sh
fi

if [[ "$REQUIRE_ALL" == 1 ]]; then
  .venv-pilot/bin/python - \
    "$EXP002_COMPLETE_MARKER" "$EXP002_ROOT" "$EXP002_RESULT_SET" \
    "$PROFILE" "$SEEDS" <<'PY'
import json
import sys
from pathlib import Path

from stackpilot.exp002_completion import validate_run_completion

marker_path, exp002_root, result_set, profile, seeds_text = sys.argv[1:]
marker = Path(marker_path)
root = Path(exp002_root)
try:
    required_seeds = {int(value) for value in seeds_text.split()}
    validate_run_completion(
        root,
        profile,
        result_set,
        required_seeds,
        marker_path=marker,
    )
except RuntimeError as exc:
    raise SystemExit(
        f"EXP-002 completion is absent, stale, or invalid at {marker}: {exc}"
    ) from exc


def validate_model(path: Path) -> None:
    resolved = path.resolve()
    config = resolved / "config.json"
    try:
        json.loads(config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid EXP-002 model config {config}: {exc}") from exc
    indexes = sorted(resolved.glob("*.safetensors.index.json")) or sorted(
        resolved.glob("*.bin.index.json")
    )
    if len(indexes) > 1:
        raise SystemExit(f"Multiple weight indexes in EXP-002 model: {resolved}")
    if indexes:
        try:
            index = json.loads(indexes[0].read_text(encoding="utf-8"))
            weights = [
                resolved / name
                for name in sorted(set(index["weight_map"].values()))
            ]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise SystemExit(f"Invalid EXP-002 weight index: {indexes[0]}") from exc
    else:
        weights = sorted(resolved.glob("*.safetensors")) or sorted(
            resolved.glob("*.bin")
        )
    if not weights or any(
        not item.is_file() or item.stat().st_size < 1024 * 1024
        for item in weights
    ):
        raise SystemExit(f"Missing or incomplete EXP-002 weights: {resolved}")
    if not (resolved / "tokenizer.json").is_file() and not (
        resolved / "tokenizer.model"
    ).is_file():
        raise SystemExit(f"Missing EXP-002 tokenizer: {resolved}")


for seed in sorted(required_seeds):
    for backend in ("bm25", "e5"):
        validate_model(
            root / "merged" / f"hard-rq0-{backend}-seed{seed}-{profile}"
        )
PY
fi

bash experiments/launch_hybrid_rrf.sh

run_eval() {
  local tag=$1
  local seed=$2
  local variant=$3
  local model_ref=$4
  local model_revision=${5:-main}
  local inject_backend_id=${6:-0}
  if [[ ! -e "$model_ref" && "$model_ref" == /* ]]; then
    if [[ "$REQUIRE_ALL" == 1 ]]; then
      echo "Missing source policy for EXP-006: $model_ref" >&2
      exit 1
    fi
    echo "Skipping missing source policy: $model_ref" >&2
    return
  fi
  local run_id
  run_id=$(
    .venv-pilot/bin/python -m stackpilot.experiment_registry run-id EXP-006 \
      --seed "$seed" --profile "$PROFILE" --variant "$variant"
  )
  EXPERIMENT_ID=EXP-006 TAG="$tag" SEED="$seed" PROFILE="$PROFILE" \
    VARIANT="$variant" RUN_ID="$run_id" MODEL_REF="$model_ref" \
    MODEL_REVISION="$model_revision" LIMIT="$LIMIT" TOPKS="$TOPKS" \
    BACKENDS="hybrid" INJECT_BACKEND_ID="$inject_backend_id" \
    bash experiments/eval_numbered_policy.sh
}

run_eval base-qwen 0 base-qwen "$BASE_MODEL_REF" "$BASE_MODEL_REVISION"

for seed in $SEEDS; do
  for backend in bm25 e5; do
    specialist="$EXP002_ROOT/merged/hard-rq0-${backend}-seed${seed}-${PROFILE}"
    run_eval "${backend}-specialist" "$seed" "exp002-${backend}-specialist" "$specialist"
  done
  mixed_run=$(
    .venv-pilot/bin/python -m stackpilot.experiment_registry run-id EXP-003 \
      --seed "$seed" --profile "$PROFILE" --variant blind
  )
  run_eval mixed-blind "$seed" exp003-mixed-blind \
    "$ROOT/work/experiments/EXP-003/merged/$mixed_run"
done

for seed in $ORACLE_SEEDS; do
  oracle_run=$(
    .venv-pilot/bin/python -m stackpilot.experiment_registry run-id EXP-004 \
      --seed "$seed" --profile "$PROFILE" --variant backend-id
  )
  # "hybrid" is intentionally out of the oracle policy's BM25/E5 training labels.
  run_eval mixed-backend-id "$seed" exp004-backend-id \
    "$ROOT/work/experiments/EXP-004/merged/$oracle_run" main 1
done

echo "EXP-006 complete: work/experiments/EXP-006"
