#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
source "$ROOT/.venv-pilot/bin/activate"
source "$ROOT/scripts/lib/runtime.sh"
ensure_local_no_proxy

CONFIG=${CONFIG:-configs/pilot.yaml}
ARGS=()
while (( $# )); do
  case "$1" in
    --config)
      if (( $# < 2 )); then
        echo "--config requires a path" >&2
        exit 2
      fi
      CONFIG=$2
      shift 2
      ;;
    --config=*)
      CONFIG=${1#--config=}
      shift
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

python -m stackpilot.retrieval_matrix --config "$CONFIG" "${ARGS[@]}"
