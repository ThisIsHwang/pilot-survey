#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
source "$ROOT/.venv-pilot/bin/activate"

RESULT_SET=${RESULT_SET:-pilot}
SEEDS=${SEEDS:-"13 42 87"}
BOOTSTRAP_SAMPLES=${BOOTSTRAP_SAMPLES:-10000}
THRESHOLD=${THRESHOLD:-0.05}
QUERY_DEVICE=${QUERY_DEVICE:-cpu}
QUERY_MODEL=${QUERY_MODEL:-sentence-transformers/all-MiniLM-L6-v2}
RESULT_ROOT=work/hard_rq0/runs/$RESULT_SET/results

# shellcheck disable=SC2206
SEED_ARGS=($SEEDS)
python -m stackpilot.validate_hard_results \
  --results-dir "$RESULT_ROOT/policies" \
  --seeds "${SEED_ARGS[@]}"

python -m stackpilot.normalize_hard_results \
  --results-dir "$RESULT_ROOT/policies"

python -m stackpilot.hard_rq0_report \
  --results-dir "$RESULT_ROOT/policies" \
  --output-dir "$RESULT_ROOT/report" \
  --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
  --threshold "$THRESHOLD"

python -m stackpilot.hard_query_analysis \
  --results-dir "$RESULT_ROOT/policies" \
  --output-dir "$RESULT_ROOT/report" \
  --difficulty-file "$RESULT_ROOT/report/difficulty_matching.csv" \
  --model "$QUERY_MODEL" \
  --device "$QUERY_DEVICE"

python -m stackpilot.hard_query_report \
  --summary "$RESULT_ROOT/report/query_turn_summary.csv" \
  --output "$RESULT_ROOT/report/QUERY_BEHAVIOR.md"

echo "Hard-RQ0 report: $ROOT/$RESULT_ROOT/report/HARD_RQ0_REPORT.md"
echo "Query report: $ROOT/$RESULT_ROOT/report/QUERY_BEHAVIOR.md"
