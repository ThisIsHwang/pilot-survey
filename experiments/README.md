# Numbered experiment suite

Every scientific experiment receives an immutable `EXP-###` identifier. The identifier describes a scientific question, not an individual seed or checkpoint.

## Naming rules

- experiment ID: `EXP-003`
- run ID: `EXP-003__seed-013__profile-pilot__variant-blind`
- checkpoint root: `work/experiments/EXP-003/checkpoints/<run-id>`
- merged model root: `work/experiments/EXP-003/merged/<run-id>`
- results root: `work/experiments/EXP-003/results/<run-id>`
- logs: `logs/experiments/EXP-003/<run-id>.log`

Experiment numbers are never reused or renumbered after results have been produced. Seeds, profiles, reward weights, and variants are encoded in the run ID and completion manifest rather than receiving new experiment numbers.

## Current map

| ID | Name | Purpose |
|---|---|---|
| EXP-001 | Original RQ0 | Small-corpus zero-shot/query-style pilot |
| EXP-002 | Hard RQ0 | Full-wiki BM25/E5 specialist transfer |
| EXP-003 | Mixed-blind GRPO | Shared policy, episode-stable backend assignment, no metadata |
| EXP-004 | Mixed backend-ID oracle | Shared policy with explicit lexical/semantic metadata |
| EXP-005 | Evidence-aware reward | Answer reward plus terminal supporting-evidence recall |
| EXP-006 | Held-out hybrid RRF | Transfer to a BM25+E5 reciprocal-rank-fusion backend |

The canonical machine-readable registry is `experiments/registry.json`. Validate it with:

```bash
python -m stackpilot.experiment_registry validate
python -m stackpilot.experiment_registry list
python -m stackpilot.experiment_registry run-id EXP-003 --seed 13 --profile pilot --variant blind
```

EXP-003 assigns one hidden backend to an entire `n_agent` GRPO rollout group;
BM25/E5 balance is enforced across prompt groups in each training batch. It
never compares BM25 and E5 trajectories inside one advantage-normalization
group. EXP-004 duplicates each source question into explicit BM25 and E5
conditions, but gives the two rows backend-qualified GRPO UIDs while preserving
their common `source_index`. Generated mixed-data parquet files carry a
source/config/output digest sidecar and are rebuilt only when that identity
changes.

Trainer validation uses `searchr1/dev.parquet`, held out from the source
training split, while final reporting alone uses `data/final_eval.jsonl` from
the official pinned development split. Validation consumes all rows
(`drop_last=false`) and uses hidden row-level routing for EXP-003. Manifested
roles are checked before training/evaluation, and the data-manifest SHA is part
of every checkpoint signature. Older incompatible completion directories are
archived with a `.stale.<timestamp>` suffix and retrained rather than silently
reused.

All policy training and evaluation paths share one strict action parser:
optional complete `<think>` blocks plus exactly one `<search>` or `<answer>`
action. Malformed final output has an empty primary prediction and cannot earn
primary EM/F1; its legacy raw-text score is emitted only as a separately named
robustness metric. Training reward reads the rollout's structured terminal
answer rather than reparsing the prompt-plus-response string. EXP-005 likewise
uses structured executed-search counts and retriever-returned titles, so prompt
examples or model-authored control blocks cannot inflate evidence or search
counts. Parser, reward, and result-schema identities are part of the relevant
signatures, forcing incompatible cached work to rerun. Specialist training
defaults to explicit `answer` reward mode; EXP-005 alone selects `evidence`.
Each run first restores the answer-only implementation, then applies the
evidence extension when requested, preventing shared-checkout contamination.
Truncated trajectories receive exactly zero reward in either mode.

## Scientific ordering

1. Complete EXP-002 before interpreting EXP-003/004.
2. Run EXP-003 with seeds 13, 42, and 87.
3. Run EXP-004 at seed 42 first; expand to three seeds only when metadata value is at least 0.03.
4. EXP-005 is a reward-diagnostic experiment, not the main mixed-policy baseline.
5. EXP-006 is evaluation-only and uses checkpoints from EXP-002 through EXP-004.

## Main decision pattern

The original latent-adaptation idea is supported only when:

```text
specialist oracle > mixed-blind
mixed + backend ID ~= specialist oracle
```

If mixed-blind already matches both specialists, a separate online stack-identification module is not justified.

## Run EXP-003 through EXP-006 on a second node

Use a separate clone for node 2. Do not run this queue from the same mutable
checkout that is currently executing `scripts/run_full_pipeline.sh`. The two
runners share a lifetime checkout lock and will fail before changing an
environment if they are pointed at the same checkout.

From the isolated node-2 clone, point the queue at node 1's Hard-RQ0 work
directory:

```bash
EXP002_ARTIFACT_ROOT=/group-volume/teo.hwang/pilot-survey/work/hard_rq0 \
  bash experiments/run_node2_queue.sh
```

That one command runs EXP-003, EXP-004, EXP-005, EXP-006, and the combined
report in order. It never launches EXP-002 or `run_full_pipeline.sh`.
`EXP002_ARTIFACT_ROOT` is consumed read-only: assets and prepared data are
reused through links, and a low-priority CPU watcher waits for node 1's
validated EXP-002 completion while EXP-003 through EXP-005 use GPUs 0-6 for
training/evaluation and GPU 7 for E5. EXP-006 starts only after the external
completion manifest and all six specialist models pass validation.

The queue creates node/run-scoped service PID and log roots, reuses valid uv
and Hugging Face caches, skips completed numbered training checkpoints through
their signatures, performs common setup once, and runs GPU stages
sequentially. By default, the independent evaluation-only vLLM environment is
installed at low CPU/I/O priority behind the first EXP-003 training seed (or
EXP-004 when EXP-003 is disabled); its deferred GPU probe runs only after that
training releases the GPUs. Set `OVERLAP_VLLM_SETUP=0` for synchronous setup.
On success or failure the queue stops managed routers, retrievers, vLLM, Ray,
and background process groups while preserving the original exit status. A
normal completed run explicitly exits with status 0.

Useful overrides:

```bash
# Smoke profile, or disable selected experiments explicitly.
PROFILE=smoke RUN_EXP005=0 RUN_EXP006=0 RUN_REPORT=0 \
  bash experiments/run_node2_queue.sh

# Node 1 used a result-set name different from PROFILE.
EXP002_ARTIFACT_ROOT=/path/to/node1/work/hard_rq0 \
EXP002_RESULT_SET=pilot \
  bash experiments/run_node2_queue.sh
```

All `RUN_EXP003`, `RUN_EXP004`, `RUN_EXP005`, `RUN_EXP006`, and `RUN_REPORT`
flags default to `1`. `EXP002_WAIT_TIMEOUT` defaults to 48 hours. Set
`HF_HOME` or `UV_CACHE_DIR` explicitly to choose another shared
content-addressed cache; virtual environments and `upstream/Search-R1` must
remain private to each checkout. `FORCE_TRAIN=1` automatically disables vLLM
setup overlap so the first seed is not forcibly trained twice. When EXP-006
uses a `BASE_MODEL_REF` different from the training `BASE_MODEL`, set its
revision with `BASE_MODEL_REF_REVISION`.
