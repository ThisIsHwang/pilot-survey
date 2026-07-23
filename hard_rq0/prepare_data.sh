#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PYTHON=$ROOT/.venv-pilot/bin/python
[[ -x "$PYTHON" ]] || {
  echo "Missing .venv-pilot; run bash scripts/bootstrap.sh first." >&2
  exit 1
}
export HF_HOME=${HF_HOME:-$ROOT/.cache/huggingface}
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
"$PYTHON" -m stackpilot.prepare_hard_rq0 --config "$ROOT/configs/hard_rq0.yaml"
