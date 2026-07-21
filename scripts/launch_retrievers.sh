#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
source .venv-pilot/bin/activate
source "$ROOT/scripts/lib/bootstrap_java.sh"
ensure_java "$ROOT"
mkdir -p logs work/pids

CORPUS=$ROOT/work/data/corpus.jsonl
BM25_INDEX=$ROOT/work/indexes/bm25/bm25
E5_INDEX=$ROOT/work/indexes/e5/e5_Flat.index
COLBERT_INDEX=$ROOT/work/indexes/colbert/colbert/indexes/hotpot_pilot_colbert

if [[ ! -d "$BM25_INDEX" ]]; then echo "Missing BM25 index: $BM25_INDEX"; exit 1; fi
if [[ ! -f "$E5_INDEX" ]]; then echo "Missing E5 index: $E5_INDEX"; exit 1; fi
if [[ ! -d "$COLBERT_INDEX" ]]; then
  COLBERT_INDEX=$(find "$ROOT/work/indexes/colbert" -type d -name hotpot_pilot_colbert | head -1)
fi
if [[ -z "$COLBERT_INDEX" || ! -d "$COLBERT_INDEX" ]]; then echo "Missing ColBERT index"; exit 1; fi

CUDA_VISIBLE_DEVICES=${BM25_GPU:-5} nohup python -m stackpilot.searchr1_server \
  --search-r1-root upstream/Search-R1 \
  --index-path "$BM25_INDEX" --corpus-path "$CORPUS" \
  --retriever-name bm25 --topk 10 --port 8001 \
  > logs/bm25.log 2>&1 & echo $! > work/pids/bm25.pid

CUDA_VISIBLE_DEVICES=${E5_GPU:-5} nohup python -m stackpilot.searchr1_server \
  --search-r1-root upstream/Search-R1 \
  --index-path "$E5_INDEX" --corpus-path "$CORPUS" \
  --retriever-name e5 --retriever-model intfloat/e5-base-v2 \
  --faiss-gpu --topk 10 --port 8002 \
  > logs/e5.log 2>&1 & echo $! > work/pids/e5.pid

CUDA_VISIBLE_DEVICES=${COLBERT_GPU:-6} nohup python -m stackpilot.colbert_server \
  --index-path "$COLBERT_INDEX" --topk 10 --port 8003 \
  > logs/colbert.log 2>&1 & echo $! > work/pids/colbert.pid

sleep 15
for port in 8001 8002 8003; do
  curl -fsS -X POST "http://127.0.0.1:${port}/retrieve" \
    -H 'Content-Type: application/json' \
    -d '{"queries":["test query"],"topk":1,"return_scores":true}' >/dev/null
  echo "retriever on port $port ready"
done
