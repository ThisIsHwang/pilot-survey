#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
EXPERIMENT_ID=${EXPERIMENT_ID:?Set EXPERIMENT_ID}
SEED=${SEED:?Set SEED}
PROFILE=${PROFILE:-pilot}
VARIANT=${VARIANT:?Set VARIANT}
PILOT_PYTHON=$ROOT/.venv-pilot/bin/python
[[ -x "$PILOT_PYTHON" ]] || { echo "Run scripts/bootstrap.sh first" >&2; exit 1; }
RUN_ID=${RUN_ID:-$(
  "$PILOT_PYTHON" -m stackpilot.experiment_registry run-id "$EXPERIMENT_ID" \
    --seed "$SEED" --profile "$PROFILE" --variant "$VARIANT"
)}
CHECKPOINT_ROOT=$ROOT/work/experiments/$EXPERIMENT_ID/checkpoints/$RUN_ID
OUTPUT_DIR=$ROOT/work/experiments/$EXPERIMENT_ID/merged/$RUN_ID
EXP=$RUN_ID CHECKPOINT_ROOT=$CHECKPOINT_ROOT OUTPUT_DIR=$OUTPUT_DIR \
  bash "$ROOT/searchr1_stage2/merge_latest_checkpoint.sh"
"$PILOT_PYTHON" - \
  "$CHECKPOINT_ROOT/.complete.json" "$OUTPUT_DIR" "${OUTPUT_DIR}.complete.json" \
  "$EXPERIMENT_ID" "$RUN_ID" "$SEED" "$PROFILE" "$VARIANT" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    training_marker_text,
    output_text,
    marker_text,
    experiment_id,
    run_id,
    seed,
    profile,
    variant,
) = sys.argv[1:]
training_marker = Path(training_marker_text)
output = Path(output_text)
marker = Path(marker_text)
if not output.is_symlink():
    raise SystemExit(f"Merged numbered model is not a symlink: {output}")
target = output.resolve()
if not target.is_dir():
    raise SystemExit(f"Merged numbered model target is missing: {target}")
digest = hashlib.sha256(training_marker.read_bytes()).hexdigest()
payload = {
    "schema": 1,
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "experiment_id": experiment_id,
    "run_id": run_id,
    "seed": int(seed),
    "profile": profile,
    "variant": variant,
    "model_path": str(target),
    "training_marker_sha256": digest,
}
marker.parent.mkdir(parents=True, exist_ok=True)
temporary = marker.with_name(f".{marker.name}.{os.getpid()}.tmp")
temporary.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
os.replace(temporary, marker)
PY
echo "$OUTPUT_DIR"
