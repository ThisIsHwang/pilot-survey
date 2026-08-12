#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PROFILE=${PROFILE:-pilot}
for experiment in EXP-051 EXP-052 EXP-053 EXP-054 EXP-056; do
  compact=${experiment/EXP-/EXP}
  path=$ROOT/work/query_credit/reports/$PROFILE/$experiment/${compact}_REPORT.md
  [[ -f "$path" ]] && { echo; cat "$path"; }
done
