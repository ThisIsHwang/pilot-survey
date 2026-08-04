#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
for experiment in EXP-029 EXP-030 EXP-031 EXP-032 EXP-033 EXP-034; do
  path=$ROOT/work/behavior_feedback/reports/$PROFILE/$experiment/${experiment//-/}_REPORT.md
  if [[ ! -f "$path" ]]; then
    # Canonical files use EXP029_REPORT.md rather than EXP_029.
    compact=${experiment/EXP-/EXP}
    path=$ROOT/work/behavior_feedback/reports/$PROFILE/$experiment/${compact}_REPORT.md
  fi
  [[ -f "$path" ]] && { echo; cat "$path"; }
done
