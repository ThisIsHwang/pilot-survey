#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
source "$ROOT/.venv-pilot/bin/activate"

BOOTSTRAP_SAMPLES=${BOOTSTRAP_SAMPLES:-10000}
THRESHOLD=${THRESHOLD:-0.05}
QUERY_DEVICE=${QUERY_DEVICE:-cpu}
QUERY_MODEL=${QUERY_MODEL:-sentence-transformers/all-MiniLM-L6-v2}

python -m stackpilot.hard_query_analysis \
  --results-dir work/hard_rq0/results/policies \
  --output-dir work/hard_rq0/results/report \
  --model "$QUERY_MODEL" \
  --device "$QUERY_DEVICE"

python -m stackpilot.hard_rq0_report \
  --results-dir work/hard_rq0/results/policies \
  --output-dir work/hard_rq0/results/report \
  --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
  --threshold "$THRESHOLD"

echo "Hard-RQ0 report: $ROOT/work/hard_rq0/results/report/HARD_RQ0_REPORT.md"
