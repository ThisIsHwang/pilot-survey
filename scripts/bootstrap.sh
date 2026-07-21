#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
mkdir -p upstream work logs

SEARCH_R1_COMMIT=${SEARCH_R1_COMMIT:-598e61bd1d36895726d28a8d06b3a15bed19f5d3}
if [[ ! -d upstream/Search-R1/.git ]]; then
  git clone https://github.com/PeterGriffinJin/Search-R1.git upstream/Search-R1
fi
git -C upstream/Search-R1 fetch --all --tags
git -C upstream/Search-R1 checkout "$SEARCH_R1_COMMIT"

python -m venv .venv-pilot
source .venv-pilot/bin/activate
python -m pip install --upgrade pip wheel setuptools
# Retrieval environment: keep it independent from the official Search-R1/vLLM environment.
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-pilot.txt
pip install -e .

if ! command -v java >/dev/null 2>&1; then
  echo "WARNING: Java is missing. Install OpenJDK 21 before building the BM25 index." >&2
else
  java -version 2>&1 | head -1
fi

cat <<MSG
Bootstrap complete.
Activate with: source $ROOT/.venv-pilot/bin/activate
Search-R1 pinned at: $SEARCH_R1_COMMIT

For Search-R1's native GRPO stage, create its official separate environment later;
the zero-shot pilot deliberately keeps the dependency surface smaller.
MSG
