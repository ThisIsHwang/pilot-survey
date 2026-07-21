#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

if [[ ! -x "$ROOT/.venv-pilot/bin/python" ]]; then
  echo "Missing .venv-pilot. Run: bash scripts/bootstrap.sh" >&2
  exit 1
fi
source "$ROOT/.venv-pilot/bin/activate"
source "$ROOT/scripts/lib/bootstrap_java.sh"
ensure_java "$ROOT"

PILOT_PYTHON=$ROOT/.venv-pilot/bin/python
TORCH_EXTENSIONS_DIR=$ROOT/.cache/torch_extensions
export TORCH_EXTENSIONS_DIR
mkdir -p "$TORCH_EXTENSIONS_DIR"
CORPUS=$ROOT/work/data/corpus.jsonl
INDEX_ROOT=$ROOT/work/indexes
mkdir -p "$INDEX_ROOT"

if [[ ! -s "$CORPUS" ]]; then
  echo "Missing corpus: $CORPUS" >&2
  echo "Run bash scripts/prepare_data.sh first." >&2
  exit 1
fi

index_state() {
  "$PILOT_PYTHON" -m stackpilot.index_state "$@"
}

BM25_INDEX=$INDEX_ROOT/bm25/bm25
BM25_MANIFEST=$INDEX_ROOT/bm25/.pilot-manifest.json
if index_state check --kind bm25 --corpus "$CORPUS" --index "$BM25_INDEX" \
  --manifest "$BM25_MANIFEST" --model "pyserini-0.25.0:DefaultEnglishAnalyzer"; then
  :
else
  BM25_TEMP=$INDEX_ROOT/bm25/input
  echo "Building BM25 index: $BM25_INDEX"
  rm -rf -- "$INDEX_ROOT/bm25"
  mkdir -p "$BM25_TEMP"
  cp "$CORPUS" "$BM25_TEMP/corpus.jsonl"
  if ! "$PILOT_PYTHON" -m pyserini.index.lucene \
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
  index_state record --kind bm25 --corpus "$CORPUS" --index "$BM25_INDEX" \
    --manifest "$BM25_MANIFEST" --model "pyserini-0.25.0:DefaultEnglishAnalyzer"
fi

E5_MODEL=${E5_MODEL:-intfloat/e5-base-v2}
E5_INDEX=$INDEX_ROOT/e5/e5_Flat.index
E5_MANIFEST=$INDEX_ROOT/e5/.pilot-manifest.json
E5_SIGNATURE="$E5_MODEL:mean:max256:flat:fp16"
if index_state check --kind e5 --corpus "$CORPUS" --index "$E5_INDEX" \
  --manifest "$E5_MANIFEST" --model "$E5_SIGNATURE"; then
  :
else
  DENSE_GPUS=${DENSE_GPUS:-0}
  E5_BATCH_SIZE=${E5_BATCH_SIZE:-256}
  echo "Building E5 index on CUDA_VISIBLE_DEVICES=$DENSE_GPUS"
  rm -rf -- "$INDEX_ROOT/e5"
  mkdir -p "$INDEX_ROOT/e5"
  CUDA_VISIBLE_DEVICES=$DENSE_GPUS "$PILOT_PYTHON" -m stackpilot.build_e5 \
    --retrieval_method e5 \
    --model_path "$E5_MODEL" \
    --corpus_path "$CORPUS" \
    --save_dir "$INDEX_ROOT/e5" \
    --use_fp16 \
    --max_length 256 \
    --batch_size "$E5_BATCH_SIZE" \
    --pooling_method mean \
    --faiss_type Flat \
    --faiss_gpu \
    --save_embedding
  index_state record --kind e5 --corpus "$CORPUS" --index "$E5_INDEX" \
    --manifest "$E5_MANIFEST" --model "$E5_SIGNATURE"
fi

COLBERT_MODEL=${COLBERT_MODEL:-colbert-ir/colbertv2.0}
COLBERT_NAME=hotpot_pilot_colbert
COLBERT_INDEX=$INDEX_ROOT/colbert/colbert/indexes/$COLBERT_NAME
COLBERT_MANIFEST=$INDEX_ROOT/colbert/.pilot-manifest.json
COLBERT_SIGNATURE="$COLBERT_MODEL:doc256:nbits2:faiss"
if index_state check --kind colbert --corpus "$CORPUS" --index "$COLBERT_INDEX" \
  --manifest "$COLBERT_MANIFEST" --model "$COLBERT_SIGNATURE"; then
  :
else
  COLBERT_GPU=${COLBERT_GPU:-4}
  COLBERT_BATCH_SIZE=${COLBERT_BATCH_SIZE:-32}
  echo "Building and warming ColBERT index on CUDA_VISIBLE_DEVICES=$COLBERT_GPU"
  rm -rf -- "$INDEX_ROOT/colbert"
  mkdir -p "$INDEX_ROOT/colbert"
  CUDA_VISIBLE_DEVICES=$COLBERT_GPU "$PILOT_PYTHON" -m stackpilot.build_colbert \
    --corpus "$CORPUS" \
    --index-root "$INDEX_ROOT/colbert" \
    --index-name "$COLBERT_NAME" \
    --model "$COLBERT_MODEL" \
    --batch-size "$COLBERT_BATCH_SIZE"
  index_state record --kind colbert --corpus "$CORPUS" --index "$COLBERT_INDEX" \
    --manifest "$COLBERT_MANIFEST" --model "$COLBERT_SIGNATURE"
fi

echo "All indexes are complete and validated."
