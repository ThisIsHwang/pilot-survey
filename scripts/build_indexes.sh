#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
source .venv-pilot/bin/activate
source "$ROOT/scripts/lib/bootstrap_java.sh"
ensure_java "$ROOT"

CORPUS=$ROOT/work/data/corpus.jsonl
INDEX_ROOT=$ROOT/work/indexes
mkdir -p "$INDEX_ROOT"

# Search-R1 creates this directory before importing Pyserini. Remove only this
# disposable staging directory when a previous Java/Pyserini startup failed.
BM25_TEMP=$INDEX_ROOT/bm25/bm25/temp
if [[ -d "$BM25_TEMP" ]]; then
  echo "Removing stale BM25 staging directory: $BM25_TEMP"
  rm -rf -- "$BM25_TEMP"
fi

# Reuse Search-R1's official index builder for BM25 and E5.
python upstream/Search-R1/search_r1/search/index_builder.py \
  --retrieval_method bm25 \
  --corpus_path "$CORPUS" \
  --save_dir "$INDEX_ROOT/bm25"

DENSE_GPUS=${DENSE_GPUS:-0,1,2,3}
CUDA_VISIBLE_DEVICES=$DENSE_GPUS python upstream/Search-R1/search_r1/search/index_builder.py \
  --retrieval_method e5 \
  --model_path intfloat/e5-base-v2 \
  --corpus_path "$CORPUS" \
  --save_dir "$INDEX_ROOT/e5" \
  --use_fp16 \
  --max_length 256 \
  --batch_size 512 \
  --pooling_method mean \
  --faiss_type Flat \
  --faiss_gpu \
  --save_embedding

COLBERT_GPU=${COLBERT_GPU:-4}
CUDA_VISIBLE_DEVICES=$COLBERT_GPU python -m stackpilot.build_colbert \
  --corpus "$CORPUS" \
  --index-root "$INDEX_ROOT/colbert" \
  --index-name hotpot_pilot_colbert \
  --model colbert-ir/colbertv2.0
