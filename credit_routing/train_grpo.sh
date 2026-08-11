#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
SEED=${SEED:?Set SEED}
BACKEND=${BACKEND:?Set BACKEND=bm25 or e5}
METHOD=${METHOD:?Set METHOD=outcome-only, action-route, observation-route, or both}
CONFIG=${CONFIG:-$ROOT/configs/credit_routing.yaml}
PYTHON=${PILOT_PYTHON:-$ROOT/.venv-pilot/bin/python}
[[ -x "$PYTHON" ]] || bash scripts/bootstrap.sh
bash scripts/bootstrap_searchr1.sh
bash hard_rq0/download_assets.sh
bash hard_rq0/prepare_data.sh
ARTIFACT=${UTILITY_ARTIFACT:-$ROOT/work/credit_routing/models/$PROFILE/document_utility_ridge.json}
[[ -s "$ARTIFACT" ]] || { echo "Missing utility artifact: $ARTIFACT" >&2; exit 1; }

eval "$("$PYTHON" - "$CONFIG" "$METHOD" <<'PY'
import shlex,sys,yaml
cfg=yaml.safe_load(open(sys.argv[1],encoding='utf-8'))
method=sys.argv[2]
variant=cfg['routing']['variants'].get(method)
if variant is None: raise SystemExit(f'unknown credit-routing method: {method}')
values={
 'ACTION_ROUTE':variant['action_route'],
 'OBSERVATION_ROUTE':variant['observation_route'],
 'ACTION_COEFFICIENT':cfg['routing']['action_coefficient'],
 'ACTION_CLIP':cfg['routing']['action_clip'],
 'TRAJECTORY_AGGREGATION':cfg['routing']['trajectory_aggregation'],
}
for key,value in values.items(): print(f'{key}={shlex.quote(str(value))}')
PY
)"

KEEP_ROUTING_STACK=1 PROFILE="$PROFILE" OBSERVATION_ROUTE="$OBSERVATION_ROUTE" \
  UTILITY_ARTIFACT="$ARTIFACT" bash credit_routing/launch_routing_stack.sh
cleanup(){ status=$?; if [[ ${KEEP_ROUTING_STACK:-0} != 1 ]]; then PROFILE="$PROFILE" OBSERVATION_ROUTE="$OBSERVATION_ROUTE" bash credit_routing/stop_routing_stack.sh || true; fi; exit $status; }
trap cleanup EXIT INT TERM

CONTRACT_DIR=$ROOT/work/credit_routing/contracts/$PROFILE
mkdir -p "$CONTRACT_DIR"
CONTRACT=$CONTRACT_DIR/${BACKEND}-${METHOD}-seed-${SEED}.json
"$PYTHON" - "$ROOT" "$CONFIG" "$ARTIFACT" "$CONTRACT" "$BACKEND" "$METHOD" "$SEED" "$PROFILE" "$ACTION_ROUTE" "$OBSERVATION_ROUTE" "$ACTION_COEFFICIENT" "$ACTION_CLIP" "$TRAJECTORY_AGGREGATION" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
(root_text,config,artifact,output,backend,method,seed,profile,action_route,observation_route,coefficient,clip,aggregation)=sys.argv[1:]
root=Path(root_text).resolve()
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
files=[
 root/'configs/credit_routing.yaml',
 root/'credit_routing/train_grpo.sh',
 root/'stackpilot/credit_routing_runtime.py',
 root/'stackpilot/credit_routing_proxy.py',
 root/'hard_rq0/patch_searchr1_credit_routing.py',
]
payload={
 'schema':1,'experiment_id':'EXP-047','backend':backend,'method':method,
 'seed':int(seed),'profile':profile,'action_route':action_route,
 'observation_route':observation_route,'action_coefficient':float(coefficient),
 'action_clip':float(clip),'trajectory_aggregation':aggregation,
 'artifact_sha256':digest(artifact),'code_sha256':{str(p.relative_to(root)):digest(p) for p in files},
}
path=Path(output)
if path.exists() and json.loads(path.read_text()) != payload: raise SystemExit(f'stale routing contract: {path}')
path.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
PY

export STACKPILOT_CR_PATCH=1
export STACKPILOT_CR_ACTION_ROUTE="$ACTION_ROUTE"
export STACKPILOT_CR_ACTION_COEFFICIENT="$ACTION_COEFFICIENT"
export STACKPILOT_CR_ACTION_CLIP="$ACTION_CLIP"
export STACKPILOT_CR_TRAJECTORY_AGGREGATION="$TRAJECTORY_AGGREGATION"
export SKIP_RETRIEVER_LAUNCH=1
export RETRIEVER_URL_OVERRIDE="http://127.0.0.1:$([[ "$BACKEND" == bm25 ]] && echo 8101 || echo 8102)/retrieve"
export BQ_SETUP_READY=1
EXPERIMENT_ID=EXP-047 CONFIG="$CONFIG" PROFILE="$PROFILE" SEED="$SEED" \
  BACKEND="$BACKEND" METHOD="$METHOD" \
  bash behavior_quotient/train_grpo.sh
