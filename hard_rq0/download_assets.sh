#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
[[ -x "$ROOT/.venv-pilot/bin/python" ]] || { echo "Run scripts/bootstrap.sh first" >&2; exit 1; }

ASSET_ROOT=${HARD_ASSET_ROOT:-$ROOT/work/hard_rq0/assets/wiki18}
mkdir -p "$ASSET_ROOT" "$ASSET_ROOT/bm25-download"

"$ROOT/.venv-pilot/bin/python" - "$ASSET_ROOT" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

root = Path(sys.argv[1])
for filename in ("part_aa", "part_ab"):
    hf_hub_download(
        repo_id="PeterJinGo/wiki-18-e5-index",
        filename=filename,
        repo_type="dataset",
        local_dir=root,
    )
hf_hub_download(
    repo_id="PeterJinGo/wiki-18-corpus",
    filename="wiki-18.jsonl.gz",
    repo_type="dataset",
    local_dir=root,
)
snapshot_download(
    repo_id="PeterJinGo/wiki-18-bm25-index",
    repo_type="dataset",
    local_dir=root / "bm25-download",
)
PY

if [[ ! -s "$ASSET_ROOT/e5_Flat.index" || "$ASSET_ROOT/part_aa" -nt "$ASSET_ROOT/e5_Flat.index" || "$ASSET_ROOT/part_ab" -nt "$ASSET_ROOT/e5_Flat.index" ]]; then
  cat "$ASSET_ROOT/part_aa" "$ASSET_ROOT/part_ab" > "$ASSET_ROOT/e5_Flat.index.tmp"
  mv "$ASSET_ROOT/e5_Flat.index.tmp" "$ASSET_ROOT/e5_Flat.index"
fi

if [[ ! -s "$ASSET_ROOT/wiki-18.jsonl" || "$ASSET_ROOT/wiki-18.jsonl.gz" -nt "$ASSET_ROOT/wiki-18.jsonl" ]]; then
  "$ROOT/.venv-pilot/bin/python" - "$ASSET_ROOT/wiki-18.jsonl.gz" "$ASSET_ROOT/wiki-18.jsonl" <<'PY'
import gzip
import shutil
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
temporary = target.with_suffix(target.suffix + ".tmp")
with gzip.open(source, "rb") as src, temporary.open("wb") as dst:
    shutil.copyfileobj(src, dst)
temporary.replace(target)
PY
fi

BM25_SOURCE=$(find "$ASSET_ROOT/bm25-download" -type f -name 'segments_*' -printf '%h\n' | head -n 1)
[[ -n "$BM25_SOURCE" ]] || { echo "Could not locate a Lucene index under $ASSET_ROOT/bm25-download" >&2; exit 1; }
rm -f "$ASSET_ROOT/bm25"
ln -s "$(realpath --relative-to="$ASSET_ROOT" "$BM25_SOURCE")" "$ASSET_ROOT/bm25"

[[ -s "$ASSET_ROOT/wiki-18.jsonl" ]] || { echo "wiki-18 corpus is missing" >&2; exit 1; }
[[ -s "$ASSET_ROOT/e5_Flat.index" ]] || { echo "E5 index is missing" >&2; exit 1; }
[[ -d "$ASSET_ROOT/bm25" ]] || { echo "BM25 index is missing" >&2; exit 1; }

echo "Hard-RQ0 assets ready: $ASSET_ROOT"
du -sh "$ASSET_ROOT/wiki-18.jsonl" "$ASSET_ROOT/e5_Flat.index" "$BM25_SOURCE"
