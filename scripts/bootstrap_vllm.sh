#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
python -m venv .venv-vllm
source .venv-vllm/bin/activate
python -m pip install --upgrade pip wheel setuptools
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.6.3
echo "vLLM environment ready: source $ROOT/.venv-vllm/bin/activate"
