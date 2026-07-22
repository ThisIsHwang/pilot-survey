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
source "$ROOT/scripts/lib/runtime.sh"
ensure_java "$ROOT"
ensure_local_no_proxy

PILOT_PYTHON=$ROOT/.venv-pilot/bin/python
TORCH_EXTENSIONS_DIR=$ROOT/.cache/torch_extensions
export TORCH_EXTENSIONS_DIR
mkdir -p "$ROOT/logs" "$ROOT/work/pids" "$TORCH_EXTENSIONS_DIR"

CORPUS=$ROOT/work/data/corpus.jsonl
BM25_INDEX=$ROOT/work/indexes/bm25/bm25
E5_INDEX=$ROOT/work/indexes/e5/e5_Flat.index
COLBERT_INDEX=$ROOT/work/indexes/colbert/colbert/indexes/hotpot_pilot_colbert
DEFAULT_E5_MODEL=intfloat/e5-base-v2
DEFAULT_E5_MODEL_REVISION=f52bf8ec8c7124536f0efb74aca902b2995e5bcd
DEFAULT_COLBERT_MODEL=colbert-ir/colbertv2.0
DEFAULT_COLBERT_MODEL_REVISION=c1e84128e85ef755c096a95bdb06b47793b13acf
E5_MODEL_SOURCE=${E5_MODEL:-$DEFAULT_E5_MODEL}
COLBERT_MODEL_SOURCE=${COLBERT_MODEL:-$DEFAULT_COLBERT_MODEL}

read -r -a requested_backends <<< "${RETRIEVER_BACKENDS:-bm25 e5 colbert}"
if [[ ${#requested_backends[@]} -eq 0 ]]; then
  echo "RETRIEVER_BACKENDS must contain at least one backend." >&2
  exit 2
fi
declare -A selected=()
for backend in "${requested_backends[@]}"; do
  case "$backend" in
    bm25|e5|colbert) ;;
    *) echo "Unknown retriever backend: $backend" >&2; exit 2 ;;
  esac
  if [[ -n ${selected[$backend]+x} ]]; then
    echo "Duplicate retriever backend: $backend" >&2
    exit 2
  fi
  selected[$backend]=1
done

if [[ -n ${selected[e5]+x} ]]; then
  if [[ -z ${E5_MODEL_REVISION:-} ]]; then
    if [[ "$E5_MODEL_SOURCE" == "$DEFAULT_E5_MODEL" ]]; then
      E5_MODEL_REVISION=$DEFAULT_E5_MODEL_REVISION
    else
      E5_MODEL_REVISION=main
    fi
  fi
  E5_MODEL=$(bash "$ROOT/scripts/resolve_hf_model.sh" \
    "$E5_MODEL_SOURCE" "$E5_MODEL_REVISION" "$PILOT_PYTHON")
fi
if [[ -n ${selected[colbert]+x} ]]; then
  if [[ -z ${COLBERT_MODEL_REVISION:-} ]]; then
    if [[ "$COLBERT_MODEL_SOURCE" == "$DEFAULT_COLBERT_MODEL" ]]; then
      COLBERT_MODEL_REVISION=$DEFAULT_COLBERT_MODEL_REVISION
    else
      COLBERT_MODEL_REVISION=main
    fi
  fi
  COLBERT_MODEL=$(bash "$ROOT/scripts/resolve_hf_model.sh" \
    "$COLBERT_MODEL_SOURCE" "$COLBERT_MODEL_REVISION" "$PILOT_PYTHON")
fi

if [[ -n ${selected[bm25]+x} ]]; then
  "$PILOT_PYTHON" -m stackpilot.index_state check --kind bm25 --corpus "$CORPUS" \
    --index "$BM25_INDEX" --manifest "$ROOT/work/indexes/bm25/.pilot-manifest.json" \
    --model "pyserini-0.25.0:DefaultEnglishAnalyzer"
fi
if [[ -n ${selected[e5]+x} ]]; then
  "$PILOT_PYTHON" -m stackpilot.index_state check --kind e5 --corpus "$CORPUS" \
    --index "$E5_INDEX" --manifest "$ROOT/work/indexes/e5/.pilot-manifest.json" \
    --model "$E5_MODEL:mean:max256:flat:fp16"
fi
if [[ -n ${selected[colbert]+x} ]]; then
  "$PILOT_PYTHON" -m stackpilot.index_state check --kind colbert --corpus "$CORPUS" \
    --index "$COLBERT_INDEX" --manifest "$ROOT/work/indexes/colbert/.pilot-manifest.json" \
    --model "$COLBERT_MODEL:doc256:nbits2:faiss"
fi

# A selective launch still stops this checkout's other retrievers first. This
# prevents an old E5/ColBERT process from silently consuming training GPUs.
stop_managed_pid "$ROOT/work/pids/bm25.pid" "$ROOT/.venv-pilot/bin/python" "$ROOT" 1
stop_managed_pid "$ROOT/work/pids/e5.pid" "$ROOT/.venv-pilot/bin/python" "$ROOT" 1
stop_managed_pid "$ROOT/work/pids/colbert.pid" "$ROOT/.venv-pilot/bin/python" "$ROOT" 1

if [[ -n ${selected[bm25]+x} ]]; then
  require_free_port "$PILOT_PYTHON" 8001
fi
if [[ -n ${selected[e5]+x} ]]; then
  require_free_port "$PILOT_PYTHON" 8002
fi
if [[ -n ${selected[colbert]+x} ]]; then
  require_free_port "$PILOT_PYTHON" 8003
fi

cleanup_started() {
  local status=$?
  trap - ERR INT TERM
  stop_managed_pid "$ROOT/work/pids/colbert.pid" "$ROOT/.venv-pilot/bin/python" "$ROOT" 1 || true
  stop_managed_pid "$ROOT/work/pids/e5.pid" "$ROOT/.venv-pilot/bin/python" "$ROOT" 1 || true
  stop_managed_pid "$ROOT/work/pids/bm25.pid" "$ROOT/.venv-pilot/bin/python" "$ROOT" 1 || true
  exit "$status"
}
trap cleanup_started ERR INT TERM

BM25_PID=
E5_PID=
COLBERT_PID=
if [[ -n ${selected[bm25]+x} ]]; then
  BM25_PID=$(CUDA_VISIBLE_DEVICES='' start_managed_process "$PILOT_PYTHON" "$ROOT/logs/bm25.log" \
    "$PILOT_PYTHON" -m stackpilot.searchr1_server \
    --search-r1-root "$ROOT/upstream/Search-R1" \
    --index-path "$BM25_INDEX" --corpus-path "$CORPUS" \
    --retriever-name bm25 --topk 10 --port 8001)
  echo "$BM25_PID" > "$ROOT/work/pids/bm25.pid"
fi

if [[ -n ${selected[e5]+x} ]]; then
  E5_PID=$(CUDA_VISIBLE_DEVICES=${E5_GPU:-5} start_managed_process "$PILOT_PYTHON" "$ROOT/logs/e5.log" \
    "$PILOT_PYTHON" -m stackpilot.searchr1_server \
    --search-r1-root "$ROOT/upstream/Search-R1" \
    --index-path "$E5_INDEX" --corpus-path "$CORPUS" \
    --retriever-name e5 --retriever-model "$E5_MODEL" \
    --faiss-gpu --topk 10 --port 8002)
  echo "$E5_PID" > "$ROOT/work/pids/e5.pid"
fi

if [[ -n ${selected[colbert]+x} ]]; then
  COLBERT_PID=$(CUDA_VISIBLE_DEVICES=${COLBERT_GPU:-6} start_managed_process "$PILOT_PYTHON" "$ROOT/logs/colbert.log" \
    "$PILOT_PYTHON" -m stackpilot.colbert_server \
    --index-path "$COLBERT_INDEX" --topk 10 --port 8003)
  echo "$COLBERT_PID" > "$ROOT/work/pids/colbert.pid"
fi

READY_TIMEOUT=${RETRIEVER_READY_TIMEOUT:-600}
if [[ -n ${selected[bm25]+x} ]]; then
  wait_for_http "$BM25_PID" "http://127.0.0.1:8001/health" "$READY_TIMEOUT" \
    "$ROOT/logs/bm25.log" '"backend":"bm25"'
fi
if [[ -n ${selected[e5]+x} ]]; then
  wait_for_http "$E5_PID" "http://127.0.0.1:8002/health" "$READY_TIMEOUT" \
    "$ROOT/logs/e5.log" '"backend":"e5"'
fi
if [[ -n ${selected[colbert]+x} ]]; then
  wait_for_http "$COLBERT_PID" "http://127.0.0.1:8003/health" "$READY_TIMEOUT" \
    "$ROOT/logs/colbert.log" '"backend":"colbert"'
fi

probe_retriever() {
  local name=$1
  local port=$2
  local pid=$3
  local log_file=$4
  local response
  if ! kill -0 "$pid" 2>/dev/null; then
    show_log_tail "$log_file"
    return 1
  fi
  if ! response=$(curl --noproxy '*' -fsS --connect-timeout 3 \
    --max-time "${RETRIEVER_PROBE_TIMEOUT:-300}" \
    -X POST "http://127.0.0.1:${port}/retrieve" \
    -H 'Content-Type: application/json' \
    -d '{"queries":["Who wrote Hamlet?"],"topk":1,"return_scores":true}'); then
    echo "$name retrieval probe failed." >&2
    show_log_tail "$log_file"
    return 1
  fi
  "$PILOT_PYTHON" -c \
    'import json,sys; p=json.loads(sys.argv[1]); assert p.get("result") and p["result"][0], p' \
    "$response"
  echo "$name retriever ready on port $port"
}

if [[ -n ${selected[bm25]+x} ]]; then
  probe_retriever bm25 8001 "$BM25_PID" "$ROOT/logs/bm25.log"
fi
if [[ -n ${selected[e5]+x} ]]; then
  probe_retriever e5 8002 "$E5_PID" "$ROOT/logs/e5.log"
fi
if [[ -n ${selected[colbert]+x} ]]; then
  probe_retriever colbert 8003 "$COLBERT_PID" "$ROOT/logs/colbert.log"
fi

trap - ERR INT TERM
echo "Selected retrievers are ready: ${requested_backends[*]}"
