#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
PYTHON=${PILOT_PYTHON:-$ROOT/.venv-pilot/bin/python}
[[ -x "$PYTHON" ]] || bash scripts/bootstrap.sh
SEEDS=${SEEDS:-$("$PYTHON" - configs/credit_routing.yaml "$PROFILE" <<'PY'
import sys,yaml
cfg=yaml.safe_load(open(sys.argv[1],encoding='utf-8'))
print(' '.join(map(str,cfg['profiles'][sys.argv[2]]['seeds'])))
PY
)}
LIMIT=${LIMIT:-$("$PYTHON" - configs/credit_routing.yaml "$PROFILE" <<'PY'
import sys,yaml
cfg=yaml.safe_load(open(sys.argv[1],encoding='utf-8'))
print(cfg['profiles'][sys.argv[2]]['eval_limit'])
PY
)}
METHODS=${METHODS:-"outcome-only action-route observation-route both"}
BACKENDS=${SOURCE_BACKENDS:-"bm25 e5"}
ARTIFACT=${UTILITY_ARTIFACT:-$ROOT/work/credit_routing/models/$PROFILE/document_utility_ridge.json}
CHECKPOINT_MANIFEST=${CREDIT_ROUTING_CHECKPOINT_MANIFEST:-$ROOT/work/credit_routing/checkpoints/$PROFILE/selected_checkpoints.csv}
if [[ -n ${REQUIRE_SELECTED_CHECKPOINTS:-} ]]; then
  REQUIRE_SELECTED=$REQUIRE_SELECTED_CHECKPOINTS
elif [[ "$PROFILE" == full ]]; then
  REQUIRE_SELECTED=1
else
  REQUIRE_SELECTED=0
fi
[[ "$REQUIRE_SELECTED" == 0 || "$REQUIRE_SELECTED" == 1 ]] || { echo "REQUIRE_SELECTED_CHECKPOINTS must be 0 or 1" >&2; exit 2; }
EVALUATED_MANIFEST=$ROOT/work/credit_routing/checkpoints/$PROFILE/evaluated_checkpoints.csv
mkdir -p "$(dirname "$EVALUATED_MANIFEST")"
printf 'source_backend,method,seed,checkpoint_role,model_ref\n' > "$EVALUATED_MANIFEST"
ENDPOINT_DATA=${CREDIT_ROUTING_ENDPOINT_DATA:-$ROOT/work/credit_routing/data/$PROFILE/final_eval_test.jsonl}
PROFILE="$PROFILE" CREDIT_ROUTING_ENDPOINT_DATA="$ENDPOINT_DATA" \
  bash credit_routing/prepare_endpoint_data.sh >/dev/null
[[ -s "$ENDPOINT_DATA" ]] || { echo "Missing held-out endpoint data: $ENDPOINT_DATA" >&2; exit 1; }
for source_backend in $BACKENDS; do
  for method in $METHODS; do
    observation_route=$("$PYTHON" - configs/credit_routing.yaml "$method" <<'PY'
import sys,yaml
cfg=yaml.safe_load(open(sys.argv[1],encoding='utf-8'))
print(cfg['routing']['variants'][sys.argv[2]]['observation_route'])
PY
)
    for seed in $SEEDS; do
      variant="${source_backend}-${method}"
      checkpoint_role=latest-fallback
      model_ref=
      if [[ -s "$CHECKPOINT_MANIFEST" ]]; then
        readarray -t selected < <("$PYTHON" - "$CHECKPOINT_MANIFEST" "$source_backend" "$method" "$seed" <<'PY'
import csv,sys
path,backend,method,seed=sys.argv[1:]
rows=[]
with open(path,encoding='utf-8',newline='') as handle:
    for row in csv.DictReader(handle):
        if (row.get('source_backend')==backend and row.get('method')==method
                and str(row.get('seed'))==str(seed)):
            rows.append(row)
if len(rows)!=1:
    raise SystemExit(
        f'expected exactly one selected checkpoint for {backend}/{method}/seed-{seed}; found {len(rows)}'
    )
print(rows[0].get('checkpoint_role') or 'selected')
print(rows[0]['model_ref'])
PY
)
        checkpoint_role=${selected[0]}
        model_ref=${selected[1]}
      elif [[ "$REQUIRE_SELECTED" == 1 ]]; then
        echo "Full endpoint evaluation requires CREDIT_ROUTING_CHECKPOINT_MANIFEST: $CHECKPOINT_MANIFEST" >&2
        exit 1
      else
        EXPERIMENT_ID=EXP-047 SEED="$seed" PROFILE="$PROFILE" VARIANT="$variant" \
          bash experiments/merge_numbered_checkpoint.sh
        source_run_id=$("$PYTHON" -m stackpilot.experiment_registry run-id EXP-047 \
          --seed "$seed" --profile "$PROFILE" --variant "$variant")
        model_ref=$ROOT/work/experiments/EXP-047/merged/$source_run_id
      fi
      [[ -e "$model_ref" ]] || { echo "Missing selected model: $model_ref" >&2; exit 1; }
      printf '%s,%s,%s,%s,%s\n' "$source_backend" "$method" "$seed" "$checkpoint_role" "$model_ref" >> "$EVALUATED_MANIFEST"
      PROFILE="$PROFILE" OBSERVATION_ROUTE="$observation_route" UTILITY_ARTIFACT="$ARTIFACT" \
        bash credit_routing/launch_routing_stack.sh
      cleanup_one(){ PROFILE="$PROFILE" OBSERVATION_ROUTE="$observation_route" bash credit_routing/stop_routing_stack.sh || true; }
      trap cleanup_one EXIT INT TERM
      EXPERIMENT_ID=EXP-048 TAG="cr-${source_backend}-${method}" \
        SEED="$seed" PROFILE="$PROFILE" VARIANT="$variant" MODEL_REF="$model_ref" \
        LIMIT="$LIMIT" DATA_FILE="$ENDPOINT_DATA" BACKENDS="bm25 e5 hybrid" TOPKS="3" \
        BM25_PORT=8101 E5_PORT=8102 HYBRID_PORT=8300 \
        bash experiments/eval_numbered_policy.sh
      cleanup_one
      trap - EXIT INT TERM
    done
  done
done
"$PYTHON" -m stackpilot.credit_routing_report \
  --config configs/credit_routing.yaml --profile "$PROFILE"
