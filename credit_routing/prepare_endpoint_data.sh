#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
PYTHON=${PILOT_PYTHON:-$ROOT/.venv-pilot/bin/python}
[[ -x "$PYTHON" ]] || bash scripts/bootstrap.sh
DEFAULT_SOURCE=$("$PYTHON" - configs/credit_routing.yaml <<'PY'
import pathlib,sys,yaml
cfg=yaml.safe_load(open(sys.argv[1],encoding='utf-8'))
value=pathlib.Path(cfg['endpoint']['data_file'])
print(value.resolve())
PY
)
SOURCE=${CREDIT_ROUTING_EVAL_DATA:-$DEFAULT_SOURCE}
OUTPUT=${CREDIT_ROUTING_ENDPOINT_DATA:-$ROOT/work/credit_routing/data/$PROFILE/final_eval_test.jsonl}
[[ -s "$SOURCE" ]] || { echo "Missing endpoint source data: $SOURCE" >&2; exit 1; }
PYTHONPATH="$ROOT:${PYTHONPATH:-}" "$PYTHON" -m stackpilot.credit_routing_eval_split \
  --config configs/credit_routing.yaml --input "$SOURCE" --output "$OUTPUT"
echo "$OUTPUT"
