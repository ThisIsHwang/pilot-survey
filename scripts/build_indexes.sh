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

if [[ ! -f "$CORPUS" ]]; then
  echo "Missing corpus: $CORPUS" >&2
  echo "Run bash scripts/prepare_data.sh first." >&2
  exit 1
fi

BM25_INDEX=$INDEX_ROOT/bm25/bm25
BM25_TEMP=$BM25_INDEX/temp
BM25_DONE=$INDEX_ROOT/bm25/.pilot-complete
if [[ -f "$BM25_DONE" ]] && compgen -G "$BM25_INDEX/segments_*" >/dev/null; then
  echo "Reusing completed BM25 index: $BM25_INDEX"
else
  # The upstream wrapper ignores Pyserini's subprocess return code and prints
  # "Finish!" after a JVM failure. Run Pyserini directly so set -e can stop on
  # an actual indexing error, and rebuild only this derived index directory.
  echo "Building BM25 index: $BM25_INDEX"
  rm -f -- "$BM25_DONE"
  rm -rf -- "$BM25_INDEX"
  mkdir -p "$BM25_TEMP"
  cp "$CORPUS" "$BM25_TEMP/corpus.jsonl"
  if ! python -m pyserini.index.lucene \
    --collection JsonCollection \
    --input "$BM25_TEMP" \
    --index "$BM25_INDEX" \
    --generator DefaultLuceneDocumentGenerator \
    --threads 1; then
    rm -rf -- "$BM25_TEMP"
    echo "BM25 indexing failed; E5 and ColBERT were not started." >&2
    exit 1
  fi
  rm -rf -- "$BM25_TEMP"
  if ! compgen -G "$BM25_INDEX/segments_*" >/dev/null; then
    echo "BM25 command exited successfully but no Lucene segments were created." >&2
    exit 1
  fi
  touch "$BM25_DONE"
fi

E5_INDEX=$INDEX_ROOT/e5/e5_Flat.index
if [[ -s "$E5_INDEX" ]]; then
  echo "Reusing completed E5 index: $E5_INDEX"
else
  # Search-R1 uses torch.nn.DataParallel whenever more than one GPU is
  # visible. One H100 is ample for this 33k-document corpus. The stackpilot
  # wrapper also forces eager attention to avoid cuDNN SDPA initialization
  # failures in this CUDA 12.9 environment.
  DENSE_GPUS=${DENSE_GPUS:-0}
  E5_BATCH_SIZE=${E5_BATCH_SIZE:-256}
  echo "Building E5 index on CUDA_VISIBLE_DEVICES=$DENSE_GPUS"
  CUDA_VISIBLE_DEVICES=$DENSE_GPUS python -m stackpilot.build_e5 \
    --retrieval_method e5 \
    --model_path intfloat/e5-base-v2 \
    --corpus_path "$CORPUS" \
    --save_dir "$INDEX_ROOT/e5" \
    --use_fp16 \
    --max_length 256 \
    --batch_size "$E5_BATCH_SIZE" \
    --pooling_method mean \
    --faiss_type Flat \
    --faiss_gpu \
    --save_embedding
  if [[ ! -s "$E5_INDEX" ]]; then
    echo "E5 command exited successfully but no FAISS index was created." >&2
    exit 1
  fi
fi

COLBERT_GPU=${COLBERT_GPU:-4}
CUDA_VISIBLE_DEVICES=$COLBERT_GPU python -m stackpilot.build_colbert \
  --corpus "$CORPUS" \
  --index-root "$INDEX_ROOT/colbert" \
  --index-name hotpot_pilot_colbert \
  --model colbert-ir/colbertv2.0
