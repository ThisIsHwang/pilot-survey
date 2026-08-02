#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
if [[ -n ${QUERY_ATTRIBUTION_INPUTS:-} ]]; then
  export QUERY_EQUIVALENCE_INPUTS="$QUERY_ATTRIBUTION_INPUTS"
fi
PROFILE="$PROFILE" bash query_equivalence/prepare.sh
