#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
source .venv-pilot/bin/activate
python -m stackpilot.react_agent_eval --config configs/pilot.yaml "$@"
python -m stackpilot.make_report --config configs/pilot.yaml
