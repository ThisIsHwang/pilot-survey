#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

CONFIG=${WEEKEND_CONFIG:-configs/query_credit_weekend.yaml}
PYTHON=${QWEN35_PYTHON:-$ROOT/.venv-qwen35/bin/python}
[[ -x "$PYTHON" ]] || {
  echo "Run scripts/bootstrap_qwen35.sh first." >&2
  exit 1
}

# Keep every Qwen3.5 label, score, gradient, checkpoint, runtime contract, and
# log physically separate from the earlier Qwen2.5 pilot namespace.
export STACKPILOT_RUNTIME_ROOT=${STACKPILOT_RUNTIME_ROOT:-$ROOT/work/query_credit_weekend_qwen35/runtime}
export STACKPILOT_LOG_ROOT=${STACKPILOT_LOG_ROOT:-$ROOT/logs/query_credit_weekend_qwen35}
export STACKPILOT_QWEN35_NO_THINK=1
mkdir -p "$STACKPILOT_RUNTIME_ROOT" "$STACKPILOT_LOG_ROOT"

"$PYTHON" - "$CONFIG" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import yaml

path = Path(sys.argv[1])
cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
model = cfg["model"]
if model.get("base_model") != "Qwen/Qwen3.5-9B":
    raise SystemExit(f"Expected Qwen/Qwen3.5-9B, found {model.get('base_model')!r}")
if model.get("enable_thinking") is not False:
    raise SystemExit("enable_thinking must be false")
if model.get("require_non_thinking") is not True:
    raise SystemExit("require_non_thinking must be true")
if model.get("chat_template_kwargs", {}).get("enable_thinking") is not False:
    raise SystemExit("chat_template_kwargs.enable_thinking must be false")
if cfg.get("source", {}).get("cross_model_artifact_policy") != "reject":
    raise SystemExit("cross_model_artifact_policy must be reject")
work_dir = str(cfg.get("work_dir", ""))
if Path(work_dir).as_posix() != "work/query_credit_weekend_qwen35":
    raise SystemExit(
        "Qwen3.5 results must use work/query_credit_weekend_qwen35; "
        f"found {work_dir!r}"
    )
if sum(int(value) for value in cfg["budget"].values()) != 120:
    raise SystemExit("The Qwen3.5 experiment budget must total 120 hours")
print("Qwen3.5 fresh-artifact and non-thinking contract passed.")
PY

exec bash "$ROOT/query_credit/run_weekend_h100.sh" "$@"
