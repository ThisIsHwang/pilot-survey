#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

PROFILE=${PROFILE:-pilot}
SEED=${SEED:?Set SEED}
BACKEND=${BACKEND:?Set BACKEND=bm25 or e5}
METHOD=${METHOD:?Set METHOD to one of the six response-feedback factorial variants}
CONFIG=${CONFIG:-$ROOT/configs/behavior_feedback.yaml}
PYTHON=${PILOT_PYTHON:-$ROOT/.venv-pilot/bin/python}
if [[ ! -x "$PYTHON" ]]; then
  bash scripts/bootstrap.sh
fi

# Resolve the preregistered rollout and advantage factors before invoking the
# already-audited Search-R1 training launcher from PR #19.
eval "$("$PYTHON" - "$CONFIG" "$METHOD" <<'PY'
import shlex, sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding='utf-8'))
method = sys.argv[2]
variant = cfg['training']['variants'].get(method)
if variant is None:
    raise SystemExit(f'unknown response-feedback method: {method}')
values = {
    'RF_ROLLOUT_MODE': variant['rollout_mode'],
    'RF_FIRST_COUNT': cfg['training']['feedback_first_count'],
    'RF_MAX_TITLES': cfg['training']['feedback_max_titles'],
    'RF_MAX_CHARS': cfg['training']['feedback_max_chars'],
    'RF_PROMPT_TOKEN_BUDGET': cfg['training']['feedback_prompt_token_budget'],
}
for key, value in values.items():
    print(f'{key}={shlex.quote(str(value))}')
PY
)"

CONTRACT_ROOT=$ROOT/work/experiments/EXP-031/feedback_contracts
mkdir -p "$CONTRACT_ROOT"
CONTRACT=$CONTRACT_ROOT/${BACKEND}-${METHOD}-seed-${SEED}-${PROFILE}.json
"$PYTHON" - "$ROOT" "$CONFIG" "$CONTRACT" "$BACKEND" "$METHOD" "$SEED" "$PROFILE" \
  "$RF_ROLLOUT_MODE" "$RF_FIRST_COUNT" "$RF_MAX_TITLES" "$RF_MAX_CHARS" "$RF_PROMPT_TOKEN_BUDGET" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
(
    root_text, config_text, output_text, backend, method, seed, profile,
    rollout_mode, first_count, max_titles, max_chars, prompt_budget,
) = sys.argv[1:]
root = Path(root_text).resolve()

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
files = [
    root / 'configs/behavior_feedback.yaml',
    root / 'behavior_feedback/train_grpo.sh',
    root / 'stackpilot/response_feedback_runtime.py',
    root / 'hard_rq0/patch_searchr1_response_feedback.py',
    root / 'hard_rq0/patch_searchr1_behavior_quotient.py',
]
payload = {
    'schema': 1,
    'experiment_id': 'EXP-031',
    'backend': backend,
    'method': method,
    'seed': int(seed),
    'profile': profile,
    'rollout_mode': rollout_mode,
    'first_count': int(first_count),
    'maximum_feedback_titles': int(max_titles),
    'maximum_feedback_chars': int(max_chars),
    'prompt_token_budget': int(prompt_budget),
    'code_sha256': {str(path.relative_to(root)): digest(path) for path in files},
}
path = Path(output_text)
if path.exists():
    previous = json.loads(path.read_text(encoding='utf-8'))
    if previous != payload:
        raise SystemExit(f'stale response-feedback contract: {path}')
else:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(temporary, path)
PY

export STACKPILOT_RF_ROLLOUT_MODE="$RF_ROLLOUT_MODE"
# Validation remains ordinary IID search so the checkpoint is judged without
# privileged sibling response feedback.
export STACKPILOT_RF_VALIDATION_MODE=iid
export STACKPILOT_RF_FIRST_COUNT="$RF_FIRST_COUNT"
export STACKPILOT_RF_MAX_TITLES="$RF_MAX_TITLES"
export STACKPILOT_RF_MAX_CHARS="$RF_MAX_CHARS"
export STACKPILOT_RF_PROMPT_TOKEN_BUDGET="$RF_PROMPT_TOKEN_BUDGET"

EXPERIMENT_ID=EXP-031 CONFIG="$CONFIG" PROFILE="$PROFILE" SEED="$SEED" \
BACKEND="$BACKEND" METHOD="$METHOD" \
  bash behavior_quotient/train_grpo.sh
