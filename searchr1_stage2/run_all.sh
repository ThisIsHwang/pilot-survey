#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

DEFAULT_BASE_MODEL=Qwen/Qwen2.5-3B-Instruct
DEFAULT_BASE_MODEL_REVISION=aa8e72537993ba99e69dfaafa59ed015b17504d1
DEFAULT_OFFICIAL_MODEL=PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo-v0.3
DEFAULT_OFFICIAL_MODEL_REVISION=4b8abb9beb4efc748019e956266ada547e276381
BASE_POLICY_MODEL=${BASE_POLICY_MODEL:-$DEFAULT_BASE_MODEL}
OFFICIAL_POLICY_MODEL=${OFFICIAL_POLICY_MODEL:-$DEFAULT_OFFICIAL_MODEL}
TRAIN_BASE_MODEL=${TRAIN_BASE_MODEL:-$DEFAULT_BASE_MODEL}
if [[ -z ${BASE_POLICY_MODEL_REVISION:-} ]]; then
  if [[ "$BASE_POLICY_MODEL" == "$DEFAULT_BASE_MODEL" ]]; then
    BASE_POLICY_MODEL_REVISION=$DEFAULT_BASE_MODEL_REVISION
  else
    BASE_POLICY_MODEL_REVISION=main
  fi
fi
if [[ -z ${OFFICIAL_POLICY_MODEL_REVISION:-} ]]; then
  if [[ "$OFFICIAL_POLICY_MODEL" == "$DEFAULT_OFFICIAL_MODEL" ]]; then
    OFFICIAL_POLICY_MODEL_REVISION=$DEFAULT_OFFICIAL_MODEL_REVISION
  else
    OFFICIAL_POLICY_MODEL_REVISION=main
  fi
fi
if [[ -z ${TRAIN_BASE_MODEL_REVISION:-} ]]; then
  if [[ "$TRAIN_BASE_MODEL" == "$DEFAULT_BASE_MODEL" ]]; then
    TRAIN_BASE_MODEL_REVISION=$DEFAULT_BASE_MODEL_REVISION
  else
    TRAIN_BASE_MODEL_REVISION=main
  fi
fi
POLICY_LIMIT=${POLICY_LIMIT:-300}
RUN_SMOKE=${RUN_SMOKE:-1}
SMOKE_ONLY=${SMOKE_ONLY:-0}
export HF_HOME=${HF_HOME:-$ROOT/.cache/huggingface}
# Model roles above are explicit. Clear stale serving/offline variables so
# index/model downloads cannot inherit a previous manual Stage-0 launch.
unset MODEL MODEL_PATH MODEL_LOCAL_ONLY MODEL_REVISION SERVED_MODEL_NAME HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

if [[ ! "$POLICY_LIMIT" =~ ^[1-9][0-9]*$ ]]; then
  echo "POLICY_LIMIT must be a positive integer; got '$POLICY_LIMIT'." >&2
  exit 2
fi
for flag_name in RUN_SMOKE SMOKE_ONLY; do
  flag_value=${!flag_name}
  if [[ "$flag_value" != 0 && "$flag_value" != 1 ]]; then
    echo "$flag_name must be 0 or 1; got '$flag_value'." >&2
    exit 2
  fi
done
if [[ "$SMOKE_ONLY" == 1 && "$RUN_SMOKE" != 1 ]]; then
  echo "SMOKE_ONLY=1 requires RUN_SMOKE=1." >&2
  exit 2
fi

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  "$ROOT/.venv-searchr1/bin/ray" stop --force >/dev/null 2>&1 || true
  bash "$ROOT/scripts/stop_servers.sh" || true
  if [[ $status -ne 0 ]]; then
    echo "Stage-2 pipeline failed. Inspect logs/ and work/results/policies/." >&2
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

"$ROOT/.venv-searchr1/bin/ray" stop --force >/dev/null 2>&1 || true
bash "$ROOT/scripts/stop_servers.sh" || true
if [[ ${SKIP_BOOTSTRAP:-0} != 1 ]]; then
  bash "$ROOT/scripts/bootstrap.sh"
  bash "$ROOT/scripts/bootstrap_vllm.sh"
  bash "$ROOT/scripts/bootstrap_searchr1.sh"
fi

bash "$ROOT/scripts/preflight.sh"
bash "$ROOT/scripts/preflight_searchr1.sh"

# Pin every remote role to a concrete cached snapshot once. In particular, the
# base evaluation and GRPO initialization cannot silently move to different
# commits during a long run.
BASE_POLICY_SOURCE=$BASE_POLICY_MODEL
TRAIN_BASE_SOURCE=$TRAIN_BASE_MODEL
BASE_POLICY_MODEL=$(bash "$ROOT/scripts/resolve_hf_model.sh" \
  "$BASE_POLICY_SOURCE" "$BASE_POLICY_MODEL_REVISION" "$ROOT/.venv-pilot/bin/python")
OFFICIAL_POLICY_MODEL=$(bash "$ROOT/scripts/resolve_hf_model.sh" \
  "$OFFICIAL_POLICY_MODEL" "$OFFICIAL_POLICY_MODEL_REVISION" "$ROOT/.venv-pilot/bin/python")
if [[ "$TRAIN_BASE_SOURCE" == "$BASE_POLICY_SOURCE" && \
      "$TRAIN_BASE_MODEL_REVISION" == "$BASE_POLICY_MODEL_REVISION" ]]; then
  TRAIN_BASE_MODEL=$BASE_POLICY_MODEL
else
  TRAIN_BASE_MODEL=$(bash "$ROOT/scripts/resolve_hf_model.sh" \
    "$TRAIN_BASE_SOURCE" "$TRAIN_BASE_MODEL_REVISION" "$ROOT/.venv-pilot/bin/python")
fi

bash "$ROOT/scripts/prepare_data.sh" --config configs/pilot.yaml
bash "$ROOT/scripts/build_indexes.sh"
"$ROOT/.venv-pilot/bin/python" searchr1_stage2/make_hotpot_searchr1_data.py \
  --work-dir work

run_policy_eval() {
  local tag=$1
  local model_ref=$2
  echo "== Policy evaluation: $tag =="
  TAG="$tag" MODEL_REF="$model_ref" SERVED_MODEL_NAME="$tag" \
    LIMIT="$POLICY_LIMIT" BACKENDS="bm25 e5" VARIANTS=blind \
    bash "$ROOT/searchr1_stage2/eval_policy.sh"
}

run_training() {
  local backend=$1
  local profile=$2
  echo "== GRPO training: $backend / $profile =="
  BACKEND="$backend" PROFILE="$profile" BASE_MODEL="$TRAIN_BASE_MODEL" \
    E5_GPU=${STAGE2_E5_GPU:-7} \
    bash "$ROOT/searchr1_stage2/run_single_stack_grpo.sh"
}

run_policy_eval base-qwen "$BASE_POLICY_MODEL"
run_policy_eval official-searchr1 "$OFFICIAL_POLICY_MODEL"

if [[ "$RUN_SMOKE" == 1 ]]; then
  run_training bm25 smoke
  run_training e5 smoke
fi
if [[ "$SMOKE_ONLY" == 1 ]]; then
  echo "Stage-2 smoke runs completed. Pilot training and report were intentionally skipped."
  exit 0
fi

run_training bm25 pilot
run_training e5 pilot

EXP=hotpot-bm25-pilot-grpo \
  bash "$ROOT/searchr1_stage2/merge_latest_checkpoint.sh"
EXP=hotpot-e5-pilot-grpo \
  bash "$ROOT/searchr1_stage2/merge_latest_checkpoint.sh"

run_policy_eval bm25-specialist "$ROOT/work/merged/hotpot-bm25-pilot-grpo"
run_policy_eval e5-specialist "$ROOT/work/merged/hotpot-e5-pilot-grpo"

bash "$ROOT/searchr1_stage2/make_rq0_report.sh"
"$ROOT/.venv-pilot/bin/python" - "$ROOT/work/results/rq0/.complete.json" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

marker = Path(sys.argv[1])
marker.parent.mkdir(parents=True, exist_ok=True)
report = marker.parent / "RQ0_REPORT.md"
if not report.is_file() or not report.stat().st_size:
    raise SystemExit(f"Missing RQ0 report: {report}")
marker.write_text(
    json.dumps(
        {"schema": 1, "completed_at": datetime.now(timezone.utc).isoformat(), "report": str(report.resolve())},
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY

echo "Stage-2 pipeline completed: $ROOT/work/results/rq0/RQ0_REPORT.md"
