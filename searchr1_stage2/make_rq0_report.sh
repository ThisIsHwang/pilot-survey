#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PILOT_PYTHON=$ROOT/.venv-pilot/bin/python
[[ -x "$PILOT_PYTHON" ]] || {
  echo "Missing .venv-pilot; run scripts/bootstrap.sh" >&2
  exit 1
}

"$PILOT_PYTHON" -m stackpilot.query_stats \
  --results-dir work/results/policies \
  --output-dir work/results/rq0
"$PILOT_PYTHON" -m stackpilot.rq0_report \
  --results-dir work/results/policies \
  --output-dir work/results/rq0

[[ -s "$ROOT/work/results/rq0/RQ0_REPORT.md" ]] || {
  echo "RQ0 report was not produced." >&2
  exit 1
}
echo "RQ0 report: $ROOT/work/results/rq0/RQ0_REPORT.md"
