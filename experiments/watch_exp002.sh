#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

usage() {
  cat <<'EOF'
Usage: watch_exp002.sh --root PATH --profile PROFILE --result-set NAME
       --seeds "13 42 87" --ready-root PATH --timeout SECONDS
       --poll-seconds SECONDS
EOF
}

EXP002_ROOT=
PROFILE=
RESULT_SET=
SEEDS=
READY_ROOT=
TIMEOUT=
POLL_SECONDS=
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) EXP002_ROOT=${2:?}; shift 2 ;;
    --profile) PROFILE=${2:?}; shift 2 ;;
    --result-set) RESULT_SET=${2:?}; shift 2 ;;
    --seeds) SEEDS=${2:?}; shift 2 ;;
    --ready-root) READY_ROOT=${2:?}; shift 2 ;;
    --timeout) TIMEOUT=${2:?}; shift 2 ;;
    --poll-seconds) POLL_SECONDS=${2:?}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for name in EXP002_ROOT PROFILE RESULT_SET SEEDS READY_ROOT TIMEOUT POLL_SECONDS; do
  [[ -n ${!name} ]] || { echo "Missing required argument: $name" >&2; exit 2; }
done
for name in TIMEOUT POLL_SECONDS; do
  value=${!name}
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
    echo "$name must be a positive integer; got '$value'." >&2
    exit 2
  }
done

PILOT_PYTHON=$ROOT/.venv-pilot/bin/python
[[ -x "$PILOT_PYTHON" ]] || {
  echo "Missing .venv-pilot; the queue must bootstrap it before this watcher." >&2
  exit 1
}
mkdir -p "$READY_ROOT"
rm -f -- "$READY_ROOT/inputs.ready" "$READY_ROOT/complete.ready"
deadline=$((SECONDS + TIMEOUT))

timed_out() {
  if (( SECONDS >= deadline )); then
    echo "Timed out after ${TIMEOUT}s waiting for external EXP-002: $EXP002_ROOT" >&2
    exit 1
  fi
}

inputs_ready() {
  "$PILOT_PYTHON" -m stackpilot.hard_assets check \
    --root "$EXP002_ROOT/assets/wiki18" >/dev/null 2>&1 &&
  "$PILOT_PYTHON" -m stackpilot.prepare_hard_rq0 \
    --config "$ROOT/configs/hard_rq0.yaml" --check >/dev/null 2>&1
}

completion_ready() {
  "$PILOT_PYTHON" - \
    "$EXP002_ROOT" "$PROFILE" "$RESULT_SET" "$SEEDS" <<'PY'
import sys
from pathlib import Path

from stackpilot.exp002_completion import validate_run_completion

root, profile, result_set, raw_seeds = sys.argv[1:]
root = Path(root)
seeds = [int(value) for value in raw_seeds.split()]
try:
    validate_run_completion(root, profile, result_set, seeds)
except RuntimeError:
    raise SystemExit(1)
for seed in seeds:
    for backend in ("bm25", "e5"):
        model = root / "merged" / f"hard-rq0-{backend}-seed{seed}-{profile}"
        if not (model / "config.json").is_file():
            raise SystemExit(1)
        weights = list(model.glob("*.safetensors")) + list(model.glob("*.bin"))
        if not weights or any(path.stat().st_size < 1024 * 1024 for path in weights):
            raise SystemExit(1)
PY
}

echo "Waiting for verified Hard-RQ0 assets/data in $EXP002_ROOT."
until inputs_ready; do
  timed_out
  sleep "$POLL_SECONDS"
done
printf 'ready\n' > "$READY_ROOT/inputs.ready"
echo "External Hard-RQ0 inputs are ready; waiting for EXP-002 completion."

until completion_ready; do
  timed_out
  sleep "$POLL_SECONDS"
done
printf 'ready\n' > "$READY_ROOT/complete.ready"
echo "External EXP-002 completion is ready."
