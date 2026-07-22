#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
BACKEND=${BACKEND:?Set BACKEND=bm25 or e5}
SEED=${SEED:?Set SEED}
PROFILE=${PROFILE:-pilot}
EXP=${EXP:-hard-rq0-${BACKEND}-seed${SEED}-${PROFILE}}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-$ROOT/work/hard_rq0/checkpoints/$EXP}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/work/hard_rq0/merged/$EXP}

EXP="$EXP" CHECKPOINT_ROOT="$CHECKPOINT_ROOT" OUTPUT_DIR="$OUTPUT_DIR" \
  bash "$ROOT/searchr1_stage2/merge_latest_checkpoint.sh"
