#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

TAG=${TAG:?Set TAG, e.g. base-qwen or bm25-specialist}
MODEL_REF=${MODEL_REF:-${MODEL_PATH:-${MODEL:-}}}
MODEL_REVISION=${MODEL_REVISION:-main}
[[ -n "$MODEL_REF" ]] || {
  echo "Set MODEL_REF to a local Hugging Face directory or repository ID." >&2
  exit 2
}
if [[ ! "$TAG" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "TAG may contain only letters, digits, dot, underscore, and dash: $TAG" >&2
  exit 2
fi

LIMIT=${LIMIT:-200}
BACKENDS=${BACKENDS:-"bm25 e5"}
VARIANTS=${VARIANTS:-blind}
read -r -a backend_args <<< "$BACKENDS"
read -r -a variant_args <<< "$VARIANTS"
if [[ ${#backend_args[@]} -eq 0 || ${#variant_args[@]} -eq 0 ]]; then
  echo "BACKENDS and VARIANTS must not be empty." >&2
  exit 2
fi
if [[ ! "$LIMIT" =~ ^[1-9][0-9]*$ ]]; then
  echo "LIMIT must be a positive integer; got '$LIMIT'." >&2
  exit 2
fi
declare -A seen_backends=()
for backend in "${backend_args[@]}"; do
  case "$backend" in
    bm25|e5|colbert) ;;
    *) echo "Unknown backend: $backend" >&2; exit 2 ;;
  esac
  if [[ -n ${seen_backends[$backend]+x} ]]; then
    echo "Duplicate backend: $backend" >&2
    exit 2
  fi
  seen_backends[$backend]=1
done
declare -A seen_variants=()
for variant in "${variant_args[@]}"; do
  case "$variant" in
    blind|oracle_guidance) ;;
    *) echo "Unknown policy variant: $variant" >&2; exit 2 ;;
  esac
  if [[ -n ${seen_variants[$variant]+x} ]]; then
    echo "Duplicate policy variant: $variant" >&2
    exit 2
  fi
  seen_variants[$variant]=1
done

# Convert remote repository IDs to immutable local snapshot directories before
# launch. The resolved commit is then part of the evaluator's model identity.
MODEL_REF=$(bash "$ROOT/scripts/resolve_hf_model.sh" \
  "$MODEL_REF" "$MODEL_REVISION" "$ROOT/.venv-pilot/bin/python")

# Preserve the model source separately from its API alias. MODEL_PATH is local
# when it resolves to a directory; repository IDs are allowed to download into
# the normal Hugging Face cache while vLLM starts.
unset MODEL_PATH MODEL MODEL_LOCAL_ONLY
if [[ -d "$MODEL_REF" ]]; then
  export MODEL_PATH=$MODEL_REF
else
  export MODEL=$MODEL_REF
fi
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-$TAG}
export LLM_GPUS=${LLM_GPUS:-0,1,2,3}
export TP=${TP:-4}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ ${KEEP_SERVERS:-0} != 1 ]]; then
    bash "$ROOT/scripts/stop_servers.sh" || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

bash "$ROOT/scripts/stop_servers.sh" || true
RETRIEVER_BACKENDS="$BACKENDS" bash "$ROOT/scripts/launch_retrievers.sh"
bash "$ROOT/scripts/launch_vllm_bg.sh"

"$ROOT/.venv-pilot/bin/python" -m stackpilot.policy_eval \
  --config configs/pilot.yaml \
  --tag "$TAG" \
  --limit "$LIMIT" \
  --backends "${backend_args[@]}" \
  --variants "${variant_args[@]}"

echo "Policy evaluation complete: $TAG ($MODEL_REF)"
