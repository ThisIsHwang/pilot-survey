#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

DEFAULT_BASE_MODEL=Qwen/Qwen2.5-3B-Instruct
DEFAULT_BASE_MODEL_REVISION=aa8e72537993ba99e69dfaafa59ed015b17504d1
BASE_MODEL_REF=${BASE_MODEL_REF:-${BASE_MODEL_PATH:-$DEFAULT_BASE_MODEL}}
if [[ -z ${BASE_MODEL_REVISION:-} ]]; then
  if [[ "$BASE_MODEL_REF" == "$DEFAULT_BASE_MODEL" ]]; then
    BASE_MODEL_REVISION=$DEFAULT_BASE_MODEL_REVISION
  else
    BASE_MODEL_REVISION=main
  fi
fi
PROFILE=${PROFILE:-pilot}
RESULT_SET=${RESULT_SET:-$PROFILE}
SEEDS=${SEEDS:-"13 42 87"}
BACKENDS=${BACKENDS:-"bm25 e5"}
TOPKS=${TOPKS:-"3 5 10"}
if [[ -z ${LIMIT+x} ]]; then
  if [[ "$PROFILE" == smoke ]]; then LIMIT=20; else LIMIT=; fi
fi
if [[ -z ${RUN_REPORT+x} ]]; then
  if [[ "$PROFILE" == smoke ]]; then RUN_REPORT=0; else RUN_REPORT=1; fi
fi
export HF_HOME=${HF_HOME:-$ROOT/.cache/huggingface}
export BM25_PORT=${BM25_PORT:-8101}
export E5_PORT=${E5_PORT:-8102}
export E5_GPU=${E5_GPU:-7}

case "$PROFILE" in smoke|pilot|full) ;; *)
  echo "PROFILE must be smoke, pilot, or full; got '$PROFILE'." >&2; exit 2 ;;
esac
[[ "$RESULT_SET" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "RESULT_SET may contain only letters, digits, dot, underscore, and dash." >&2
  exit 2
}
if [[ -n "$LIMIT" && ! "$LIMIT" =~ ^[1-9][0-9]*$ ]]; then
  echo "LIMIT must be empty or a positive integer; got '$LIMIT'." >&2
  exit 2
fi
for flag_name in \
  SKIP_BOOTSTRAP SKIP_ASSETS SKIP_DATA RUN_REPORT KEEP_HARD_SERVERS KEEP_VLLM; do
  flag_value=${!flag_name:-0}
  if [[ "$flag_value" != 0 && "$flag_value" != 1 ]]; then
    echo "$flag_name must be 0 or 1; got '$flag_value'." >&2
    exit 2
  fi
done
if [[ ${KEEP_VLLM:-0} == 1 ]]; then
  echo "KEEP_VLLM=1 is unsafe in the sequential Hard-RQ0 pipeline because the next training stage needs GPUs 0-6; leave it at 0." >&2
  exit 2
fi
read -r -a seed_args <<< "$SEEDS"
if [[ ${#seed_args[@]} -eq 0 ]]; then
  echo "SEEDS must contain at least one integer." >&2
  exit 2
fi
declare -A seen_seeds=()
for seed in "${seed_args[@]}"; do
  if [[ ! "$seed" =~ ^[1-9][0-9]*$ || -n ${seen_seeds[$seed]+x} ]]; then
    echo "SEEDS must contain unique positive integers; got '$SEEDS'." >&2
    exit 2
  fi
  seen_seeds[$seed]=1
done
if [[ "$RUN_REPORT" == 1 && ${#seed_args[@]} -lt 3 ]]; then
  echo "RUN_REPORT=1 requires at least three specialist seeds; got '$SEEDS'." >&2
  exit 2
fi
read -r -a backend_args <<< "$BACKENDS"
read -r -a topk_args <<< "$TOPKS"
if [[ "${backend_args[*]}" != "bm25 e5" && "${backend_args[*]}" != "e5 bm25" ]]; then
  echo "The complete hard-RQ0 run requires BACKENDS='bm25 e5'; got '$BACKENDS'." >&2
  exit 2
fi
declare -A seen_topks=()
if [[ ${#topk_args[@]} -ne 3 ]]; then
  echo "TOPKS must contain each of 3, 5, and 10 exactly once; got '$TOPKS'." >&2
  exit 2
fi
for topk in "${topk_args[@]}"; do
  if [[ "$topk" != 3 && "$topk" != 5 && "$topk" != 10 ]] || [[ -n ${seen_topks[$topk]+x} ]]; then
    echo "TOPKS must contain each of 3, 5, and 10 exactly once; got '$TOPKS'." >&2
    exit 2
  fi
  seen_topks[$topk]=1
done
if [[ "$RUN_REPORT" == 1 && -n "$LIMIT" && "$LIMIT" -lt 2 ]]; then
  echo "RUN_REPORT=1 requires LIMIT>=2 so both benchmark datasets are represented." >&2
  exit 2
fi

RUN_COMPLETE_MARKER=$ROOT/work/hard_rq0/runs/$RESULT_SET/.complete.json
REPORT_COMPLETE_MARKER=$ROOT/work/hard_rq0/runs/$RESULT_SET/results/report/.complete.json
# A failed rerun must never leave an older success marker behind.
rm -f -- "$RUN_COMPLETE_MARKER" "$REPORT_COMPLETE_MARKER"

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ ${KEEP_HARD_SERVERS:-0} != 1 ]]; then
    bash "$ROOT/hard_rq0/stop_retrievers.sh" || true
    bash "$ROOT/scripts/stop_servers.sh" || true
  fi
  if [[ -x "$ROOT/.venv-searchr1/bin/ray" ]]; then
    "$ROOT/.venv-searchr1/bin/ray" stop --force >/dev/null 2>&1 || true
  fi
  if [[ $status -ne 0 ]]; then
    echo "Hard-RQ0 pipeline failed. Inspect logs/hard_rq0 and work/hard_rq0." >&2
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

if [[ -x "$ROOT/.venv-searchr1/bin/ray" ]]; then
  "$ROOT/.venv-searchr1/bin/ray" stop --force >/dev/null 2>&1 || true
fi
bash "$ROOT/hard_rq0/stop_retrievers.sh" || true
bash "$ROOT/scripts/stop_servers.sh" || true

if [[ ${SKIP_BOOTSTRAP:-0} != 1 ]]; then
  bash "$ROOT/scripts/bootstrap.sh"
  bash "$ROOT/scripts/bootstrap_vllm.sh"
  bash "$ROOT/scripts/bootstrap_searchr1.sh"
fi

PROFILE=$PROFILE bash "$ROOT/hard_rq0/preflight.sh"

BASE_MODEL=$(unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE; \
  bash "$ROOT/scripts/resolve_hf_model.sh" \
    "$BASE_MODEL_REF" "$BASE_MODEL_REVISION" "$ROOT/.venv-pilot/bin/python")

if [[ ${SKIP_ASSETS:-0} != 1 ]]; then
  bash "$ROOT/hard_rq0/download_assets.sh"
else
  "$ROOT/.venv-pilot/bin/python" -m stackpilot.hard_assets check \
    --root "${HARD_ASSET_ROOT:-$ROOT/work/hard_rq0/assets/wiki18}" >/dev/null
fi
if [[ ${SKIP_DATA:-0} != 1 ]]; then
  bash "$ROOT/hard_rq0/prepare_data.sh"
else
  "$ROOT/.venv-pilot/bin/python" -m stackpilot.prepare_hard_rq0 \
    --config "$ROOT/configs/hard_rq0.yaml" --check
fi

expected_policy_files=(base-qwen-seed0.jsonl)
for specialist_tag in bm25-specialist e5-specialist; do
  for seed in "${seed_args[@]}"; do
    expected_policy_files+=("${specialist_tag}-seed${seed}.jsonl")
  done
done
"$ROOT/.venv-pilot/bin/python" - \
  "$ROOT/work/hard_rq0/runs/$RESULT_SET/results/policies" \
  "${expected_policy_files[@]}" <<'PY'
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

root = Path(sys.argv[1])
expected = set(sys.argv[2:])
extras = [path for path in root.glob("*.jsonl") if path.name not in expected]
if extras:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    archive = root / "archive" / f"unselected-{stamp}-{os.getpid()}"
    archive.mkdir(parents=True, exist_ok=False)
    for path in extras:
        shutil.move(str(path), archive / path.name)
    print(f"Archived {len(extras)} unselected policy files: {archive}")
PY

bash "$ROOT/hard_rq0/launch_retrievers.sh"

base_eval=(
  "TAG=base-qwen"
  "SEED=0"
  "MODEL_REF=$BASE_MODEL"
  "MODEL_REVISION=$BASE_MODEL_REVISION"
  "RESULT_SET=$RESULT_SET"
  "SPECIALIST_SEEDS=$SEEDS"
  "BACKENDS=$BACKENDS"
  "TOPKS=$TOPKS"
)
if [[ -n "$LIMIT" ]]; then base_eval+=("LIMIT=$LIMIT"); fi
env "${base_eval[@]}" bash "$ROOT/hard_rq0/eval_policy.sh"

specialist_run=(
  "PROFILE=$PROFILE"
  "RESULT_SET=$RESULT_SET"
  "SEEDS=$SEEDS"
  "BASE_MODEL=$BASE_MODEL"
  "BASE_MODEL_REVISION=$BASE_MODEL_REVISION"
  "LIMIT=$LIMIT"
  "SPECIALIST_SEEDS=$SEEDS"
  "BACKENDS=$BACKENDS"
  "TOPKS=$TOPKS"
)
env "${specialist_run[@]}" bash "$ROOT/hard_rq0/run_three_seed_specialists.sh"

if [[ "$RUN_REPORT" == 1 ]]; then
  RESULT_SET=$RESULT_SET SEEDS="$SEEDS" BACKENDS="$BACKENDS" TOPKS="$TOPKS" \
    bash "$ROOT/hard_rq0/make_report.sh"
else
  echo "RUN_REPORT=0: specialist evaluation is complete; the multi-seed report was skipped."
fi

"$ROOT/.venv-pilot/bin/python" - \
  "$RUN_COMPLETE_MARKER" \
  "$PROFILE" "$RESULT_SET" "$BASE_MODEL" "$SEEDS" "$RUN_REPORT" \
  "$LIMIT" "$BACKENDS" "$TOPKS" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

marker, profile, result_set, model, seeds, report, limit, backends, topks = sys.argv[1:]
path = Path(marker)
path.parent.mkdir(parents=True, exist_ok=True)
policy_dir = path.parent / "results" / "policies"
expected_names = ["base-qwen-seed0.jsonl"] + [
    f"{tag}-seed{seed}.jsonl"
    for tag in ("bm25-specialist", "e5-specialist")
    for seed in seeds.split()
]
policy_files = [policy_dir / name for name in expected_names]
if any(not item.is_file() or not item.stat().st_size for item in policy_files):
    raise SystemExit(f"Missing completed policy result: {policy_files}")


def digest(item: Path) -> str:
    value = hashlib.sha256()
    with item.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


evaluation_signatures = set()
for item in policy_files:
    with item.open("r", encoding="utf-8") as handle:
        first = next((line for line in handle if line.strip()), "")
    if not first:
        raise SystemExit(f"Empty policy result: {item}")
    evaluation_signatures.add(str(json.loads(first).get("evaluation_signature", "")))
if len(evaluation_signatures) != 1 or not next(iter(evaluation_signatures)):
    raise SystemExit(f"Policy results do not share one evaluation signature: {evaluation_signatures}")
report_marker = path.parent / "results" / "report" / ".complete.json"
if report == "1" and (not report_marker.is_file() or not report_marker.stat().st_size):
    raise SystemExit(f"Missing report completion marker: {report_marker}")
payload = {
    "schema": 2,
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "profile": profile,
    "result_set": result_set,
    "base_model": str(Path(model).resolve()),
    "seeds": [int(value) for value in seeds.split()],
    "limit": int(limit) if limit else None,
    "backends": backends.split(),
    "topks": [int(value) for value in topks.split()],
    "evaluation_signature": next(iter(evaluation_signatures)),
    "policy_files": {item.name: digest(item) for item in policy_files},
    "report_generated": report == "1",
    "report_marker_sha256": digest(report_marker) if report == "1" else None,
}
temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY

echo "Hard-RQ0 pipeline completed: $ROOT/work/hard_rq0/runs/$RESULT_SET"
