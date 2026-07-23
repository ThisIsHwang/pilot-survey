#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PYTHON=$ROOT/.venv-pilot/bin/python
[[ -x "$PYTHON" ]] || { echo "Missing .venv-pilot; run scripts/bootstrap.sh." >&2; exit 1; }

RESULT_SET=${RESULT_SET:-pilot}
SEEDS=${SEEDS:-"13 42 87"}
BOOTSTRAP_SAMPLES=${BOOTSTRAP_SAMPLES:-10000}
BOOTSTRAP_SEED=${BOOTSTRAP_SEED:-2026}
THRESHOLD=${THRESHOLD:-0.05}
QUERY_DEVICE=${QUERY_DEVICE:-cpu}
BACKENDS=${BACKENDS:-"bm25 e5"}
TOPKS=${TOPKS:-"3 5 10"}
DEFAULT_QUERY_MODEL=sentence-transformers/all-MiniLM-L6-v2
DEFAULT_QUERY_MODEL_REVISION=1110a243fdf4706b3f48f1d95db1a4f5529b4d41
QUERY_MODEL=${QUERY_MODEL:-$DEFAULT_QUERY_MODEL}
if [[ -z ${QUERY_MODEL_REVISION:-} ]]; then
  if [[ "$QUERY_MODEL" == "$DEFAULT_QUERY_MODEL" ]]; then
    QUERY_MODEL_REVISION=$DEFAULT_QUERY_MODEL_REVISION
  else
    QUERY_MODEL_REVISION=main
  fi
fi
RESULT_ROOT=$ROOT/work/hard_rq0/runs/$RESULT_SET/results
REPORT_MARKER=$RESULT_ROOT/report/.complete.json

[[ "$RESULT_SET" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Invalid RESULT_SET=$RESULT_SET" >&2; exit 2; }
[[ "$BOOTSTRAP_SAMPLES" =~ ^[1-9][0-9]*$ ]] || {
  echo "BOOTSTRAP_SAMPLES must be a positive integer." >&2; exit 2;
}
[[ "$BOOTSTRAP_SEED" =~ ^[0-9]+$ ]] || {
  echo "BOOTSTRAP_SEED must be a non-negative integer." >&2; exit 2;
}
"$PYTHON" -c \
  'import math,sys; value=float(sys.argv[1]); sys.exit(None if math.isfinite(value) and value >= 0 else 1)' \
  "$THRESHOLD" || {
  echo "THRESHOLD must be a finite non-negative number; got '$THRESHOLD'." >&2
  exit 2
}
read -r -a seed_args <<< "$SEEDS"
read -r -a backend_args <<< "$BACKENDS"
read -r -a topk_args <<< "$TOPKS"
if [[ ${#seed_args[@]} -lt 3 ]]; then
  echo "The hard-RQ0 report requires at least three specialist seeds." >&2
  exit 2
fi
declare -A seen=()
for seed in "${seed_args[@]}"; do
  if [[ ! "$seed" =~ ^[1-9][0-9]*$ || -n ${seen[$seed]+x} ]]; then
    echo "SEEDS must contain unique positive integers; got '$SEEDS'." >&2
    exit 2
  fi
  seen[$seed]=1
done

rm -f -- "$REPORT_MARKER"

QUERY_MODEL=$(unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE; \
  bash "$ROOT/scripts/resolve_hf_model.sh" \
    "$QUERY_MODEL" "$QUERY_MODEL_REVISION" "$PYTHON")
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

"$PYTHON" -m stackpilot.validate_hard_results \
  --results-dir "$RESULT_ROOT/policies" --seeds "${seed_args[@]}" \
  --backends "${backend_args[@]}" --topks "${topk_args[@]}"

"$PYTHON" -m stackpilot.hard_rq0_report \
  --results-dir "$RESULT_ROOT/policies" \
  --output-dir "$RESULT_ROOT/report" \
  --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
  --bootstrap-seed "$BOOTSTRAP_SEED" \
  --threshold "$THRESHOLD"

"$PYTHON" -m stackpilot.hard_query_analysis \
  --results-dir "$RESULT_ROOT/policies" \
  --output-dir "$RESULT_ROOT/report" \
  --difficulty-file "$RESULT_ROOT/report/difficulty_matching.csv" \
  --model "$QUERY_MODEL" --device "$QUERY_DEVICE"

"$PYTHON" -m stackpilot.hard_query_report \
  --summary "$RESULT_ROOT/report/query_turn_summary.csv" \
  --output "$RESULT_ROOT/report/QUERY_BEHAVIOR.md"

"$PYTHON" - "$REPORT_MARKER" "$RESULT_ROOT/report" \
  "$RESULT_SET" "$SEEDS" "$BACKENDS" "$TOPKS" \
  "$BOOTSTRAP_SAMPLES" "$BOOTSTRAP_SEED" "$THRESHOLD" \
  "$QUERY_MODEL" "$QUERY_MODEL_REVISION" "$QUERY_DEVICE" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    marker_text,
    report_root_text,
    result_set,
    seeds,
    backends,
    topks,
    bootstrap_samples,
    bootstrap_seed,
    threshold,
    query_model,
    query_model_revision,
    query_device,
) = sys.argv[1:]
marker = Path(marker_text)
report_root = Path(report_root_text).resolve()
report_names = (
    "absolute_summary.csv",
    "gain_over_base.csv",
    "home_backend_excess.csv",
    "base_backend_gap.csv",
    "difficulty_matching.csv",
    "matched_hard_question_ids.json",
    "matched_hard_units.json",
    "query_turns.csv",
    "query_turn_summary.csv",
    "query_shift_by_turn.csv",
    "HARD_RQ0_REPORT.md",
    "QUERY_BEHAVIOR.md",
)
reports = [report_root / name for name in report_names]
if any(not path.is_file() or not path.stat().st_size for path in reports):
    raise SystemExit(f"Missing final hard-RQ0 report: {reports}")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


payload = {
    "schema": 3,
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "result_set": result_set,
    "seeds": [int(value) for value in seeds.split()],
    "backends": backends.split(),
    "topks": [int(value) for value in topks.split()],
    "bootstrap_samples": int(bootstrap_samples),
    "bootstrap_seed": int(bootstrap_seed),
    "specialization_threshold": float(threshold),
    "query_model": {
        "resolved_path": str(Path(query_model).resolve()),
        "requested_revision": query_model_revision,
        "resolved_revision": (
            Path(query_model).resolve().name
            if Path(query_model).resolve().parent.name == "snapshots"
            else "local"
        ),
        "device": query_device,
    },
    "reports": {path.name: digest(path) for path in reports},
}
temporary = marker.with_name(f".{marker.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, marker)
PY

echo "Hard-RQ0 report: $RESULT_ROOT/report/HARD_RQ0_REPORT.md"
echo "Query report: $RESULT_ROOT/report/QUERY_BEHAVIOR.md"
