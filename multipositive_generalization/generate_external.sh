#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PYTHON=${TRACE_PYTHON:-$ROOT/.venv-trace/bin/python}
PROFILE=${PROFILE:-pilot}
OUTPUT=${MULTIPOSITIVE_EXTERNAL_OUTPUT:-$ROOT/work/multipositive_generalization/external/$PROFILE/queries.jsonl}
"$PYTHON" -m stackpilot.multipositive_external_generate --config configs/multipositive_generalization.yaml --profile "$PROFILE" --output "$OUTPUT" "$@"
echo "export MULTIPOSITIVE_EXTERNAL_QUERIES='$OUTPUT'"
