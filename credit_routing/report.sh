#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
for experiment in EXP-045 EXP-046 EXP-047 EXP-048 EXP-049; do
  compact=${experiment/EXP-/EXP}
  path=$ROOT/work/credit_routing/reports/$PROFILE/$experiment/${compact}_REPORT.md
  [[ -f "$path" ]] && { echo; cat "$path"; }
done
