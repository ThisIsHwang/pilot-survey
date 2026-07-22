#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
EXP=${EXP:?Set EXP to the Search-R1 experiment name}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-$ROOT/work/checkpoints/$EXP}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/work/merged/$EXP}

LATEST=$(find "$CHECKPOINT_ROOT" -maxdepth 1 -type d -name 'global_step_*' | sort -V | tail -n 1)
[[ -n "$LATEST" ]] || { echo "No global_step_* checkpoint under $CHECKPOINT_ROOT" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR"

cd "$ROOT/upstream/Search-R1"
if python3 -m verl.model_merger --help >/dev/null 2>&1; then
  python3 -m verl.model_merger merge --backend fsdp --local_dir "$LATEST/actor" --target_dir "$OUTPUT_DIR"
elif [[ -f scripts/model_merger.py ]]; then
  python3 scripts/model_merger.py --local_dir "$LATEST/actor" --target_dir "$OUTPUT_DIR"
elif [[ -f scripts/legacy_model_merger.py ]]; then
  python3 scripts/legacy_model_merger.py --local_dir "$LATEST/actor" --target_dir "$OUTPUT_DIR"
else
  echo "Could not find a supported veRL model merger in $ROOT/upstream/Search-R1" >&2
  exit 1
fi

echo "$OUTPUT_DIR"
