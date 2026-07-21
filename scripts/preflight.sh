#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

echo '== GPU =='
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

echo '== Java =='
java -version 2>&1 | head -2 || { echo 'OpenJDK 21 is required for Pyserini BM25 indexing.'; exit 1; }

echo '== Python environments =='
[[ -x .venv-pilot/bin/python ]] && .venv-pilot/bin/python -V || echo 'missing .venv-pilot'
[[ -x .venv-vllm/bin/python ]] && .venv-vllm/bin/python -V || echo 'missing .venv-vllm'

echo '== FAISS GPU =='
if [[ -x .venv-pilot/bin/python ]]; then
  .venv-pilot/bin/python - <<'PY'
import faiss

if not hasattr(faiss, "StandardGpuResources"):
    raise SystemExit("FAISS was installed without GPU support")
print(f"FAISS {faiss.__version__}; visible GPUs: {faiss.get_num_gpus()}")
PY
fi

echo '== Upstream =='
if [[ -d upstream/Search-R1/.git ]]; then
  git -C upstream/Search-R1 rev-parse HEAD
else
  echo 'missing upstream/Search-R1'
fi

echo '== Ports =='
for port in 8001 8002 8003 9000; do
  if ss -ltn 2>/dev/null | grep -q ":${port} "; then
    echo "port $port is in use"
  else
    echo "port $port is free"
  fi
done
