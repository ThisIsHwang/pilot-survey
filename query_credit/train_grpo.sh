#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
export STACKPILOT_EXPERIMENT_REGISTRY_OVERLAY=$ROOT/experiments/registry.query_credit.json
PROFILE=${PROFILE:-pilot}
SEED=${SEED:?Set SEED}
BACKEND=${BACKEND:?Set BACKEND=bm25 or e5}
METHOD=${METHOD:?Set METHOD=outcome, doc-to-action, alias-normalized, or shuffled-doc}
QC_CONFIG=${QC_CONFIG:-$ROOT/configs/query_credit.yaml}
PYTHON=${PILOT_PYTHON:-$ROOT/.venv-pilot/bin/python}
[[ -x "$PYTHON" ]] || bash scripts/bootstrap.sh
MODEL_PATH=${QUERY_CREDIT_MODEL:-$ROOT/work/query_credit/reports/$PROFILE/EXP-054/document_utility_model.json}
[[ -s "$MODEL_PATH" ]] || {
  echo "Missing frozen document-utility estimator: $MODEL_PATH" >&2
  echo "Run PROFILE=$PROFILE bash experiments/EXP-054/run.sh first." >&2
  exit 1
}
case "$METHOD" in
  outcome) QC_MODE=outcome ;;
  doc-to-action) QC_MODE=doc-to-action ;;
  alias-normalized) QC_MODE=alias-normalized ;;
  shuffled-doc) QC_MODE=shuffled-doc ;;
  *) echo "Unknown METHOD=$METHOD" >&2; exit 2 ;;
esac
RUNTIME_DIR=$ROOT/work/query_credit/runtime/$PROFILE
mkdir -p "$RUNTIME_DIR"
DERIVED_CONFIG=$RUNTIME_DIR/behavior_quotient_query_credit.yaml
"$PYTHON" - "$ROOT/configs/behavior_quotient.yaml" "$QC_CONFIG" "$DERIVED_CONFIG" <<'PY'
import sys, yaml
base_path, qc_path, output_path = sys.argv[1:]
base = yaml.safe_load(open(base_path, encoding='utf-8'))
qc = yaml.safe_load(open(qc_path, encoding='utf-8'))
for profile_name, source in qc['profiles'].items():
    target = base['profiles'].setdefault(profile_name, {})
    for key in (
        'total_updates', 'train_batch', 'val_batch', 'mini_batch',
        'micro_batch', 'save_freq', 'test_freq', 'seeds', 'eval_limit',
        'bootstrap_samples',
    ):
        if key in source:
            target[key] = source[key]
training = base['training']
for method in ('outcome', 'doc-to-action', 'alias-normalized', 'shuffled-doc'):
    training['variants'][method] = {
        'advantage_mode': 'surface',
        'selection_mode': 'all',
        'update_per_prompt': 0,
    }
with open(output_path, 'w', encoding='utf-8') as handle:
    yaml.safe_dump(base, handle, sort_keys=False)
PY
export STACKPILOT_QUERY_CREDIT_PATCH=1
export STACKPILOT_QC_MODE="$QC_MODE"
export STACKPILOT_QC_MODEL="$MODEL_PATH"
export STACKPILOT_QC_AGGREGATION=${STACKPILOT_QC_AGGREGATION:-positive-sum}
export STACKPILOT_QC_ALPHA=${STACKPILOT_QC_ALPHA:-$("$PYTHON" - "$QC_CONFIG" <<'PY'
import sys,yaml
print(yaml.safe_load(open(sys.argv[1],encoding='utf-8'))['training']['action_bonus_weight'])
PY
)}
export STACKPILOT_QC_TELEMETRY_PATH=$ROOT/work/query_credit/telemetry/$PROFILE/${BACKEND}-${METHOD}-seed-${SEED}.jsonl
export STACKPILOT_RF_ROLLOUT_MODE=iid
export STACKPILOT_RF_VALIDATION_MODE=iid
EXPERIMENT_ID=EXP-055 CONFIG="$DERIVED_CONFIG" PROFILE="$PROFILE" SEED="$SEED" \
BACKEND="$BACKEND" METHOD="$METHOD" BQ_SETUP_READY=${BQ_SETUP_READY:-0} \
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct} \
  bash behavior_quotient/train_grpo.sh
