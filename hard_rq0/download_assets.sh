#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PYTHON=$ROOT/.venv-pilot/bin/python
[[ -x "$PYTHON" ]] || {
  echo "Missing .venv-pilot; run bash scripts/bootstrap.sh first." >&2
  exit 1
}

export HF_HOME=${HF_HOME:-$ROOT/.cache/huggingface}
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
ASSET_ROOT=${HARD_ASSET_ROOT:-$ROOT/work/hard_rq0/assets/wiki18}
MIN_FREE_GIB=${HARD_ASSET_MIN_FREE_GIB:-150}
if [[ ! "$MIN_FREE_GIB" =~ ^[1-9][0-9]*$ ]]; then
  echo "HARD_ASSET_MIN_FREE_GIB must be a positive integer; got '$MIN_FREE_GIB'." >&2
  exit 2
fi

args=(download --root "$ASSET_ROOT" --min-free-gib "$MIN_FREE_GIB")
if [[ ${KEEP_HARD_SOURCE_ARCHIVES:-0} == 1 ]]; then
  args+=(--keep-source-archives)
elif [[ ${KEEP_HARD_SOURCE_ARCHIVES:-0} != 0 ]]; then
  echo "KEEP_HARD_SOURCE_ARCHIVES must be 0 or 1." >&2
  exit 2
fi

"$PYTHON" -m stackpilot.hard_assets "${args[@]}"
