#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
case "021" in
  020) exec bash interface_causality/run_alias.sh "$@" ;;
  021) exec bash interface_causality/run_granularity.sh "$@" ;;
  022) exec bash interface_causality/run_expressivity.sh "$@" ;;
  023) exec bash interface_causality/run_predictor.sh "$@" ;;
esac
