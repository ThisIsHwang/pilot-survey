#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
source .venv-pilot/bin/activate
python -m stackpilot.retrieval_matrix --config configs/pilot.yaml "$@"
