#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
source "$ROOT/.venv-pilot/bin/activate"

python -m stackpilot.query_stats \
  --results-dir work/results/policies \
  --output-dir work/results/rq0
python -m stackpilot.rq0_report \
  --results-dir work/results/policies \
  --output-dir work/results/rq0

echo "RQ0 report: $ROOT/work/results/rq0/RQ0_REPORT.md"
