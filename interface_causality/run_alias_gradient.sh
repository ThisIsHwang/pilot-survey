#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PYTHON=${TRACE_PYTHON:-$ROOT/.venv-trace/bin/python}
[[ -x "$PYTHON" ]] || { echo "Run trace_go/bootstrap.sh first." >&2; exit 1; }
PROFILE=${PROFILE:-pilot}
GPU=${INTERFACE_GRADIENT_GPU:-0}
ARGS=()
if [[ -n ${INTERFACE_CAUSAL_INPUTS:-} ]]; then
  IFS=':' read -r -a INPUTS <<< "$INTERFACE_CAUSAL_INPUTS"
  ARGS+=(--inputs "${INPUTS[@]}")
fi
[[ -n ${INTERFACE_BASE_MODEL:-} ]] && ARGS+=(--base-model "$INTERFACE_BASE_MODEL")
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -m stackpilot.interface_alias_gradient \
  --config configs/interface_causality.yaml \
  --profile "$PROFILE" "${ARGS[@]}"
