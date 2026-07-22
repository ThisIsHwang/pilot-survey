#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
source "$ROOT/.venv-pilot/bin/activate"
python -m stackpilot.prepare_hard_rq0 --config configs/hard_rq0.yaml
