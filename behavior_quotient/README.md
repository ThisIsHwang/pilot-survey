# Behavior-Quotient RL experiment suite (EXP-024 through EXP-028)

This stacked suite follows the accepted interface-causality branch (PR #17).
Experiment labels that appeared only in the rejected PR #18 are not canonical;
the repository registry intentionally resumes at EXP-024.

The prior results show
that free-form query aliases are common enough to consume rollout budget, while
query-level and behavior-class counterfactual credit are almost identical. The
new question is therefore not whether a behavior class deserves a different
reward, but whether **surface aliases waste fixed rollout budget and distort
prompt-group normalization during real on-policy training**.

The suite keeps the free-form `<search>...</search>` interface. Behavior classes
are induced *after retrieval* from the visible ranked title transition. No
semantic-only equivalence classifier is trusted by the primary method.

## Experiments

### EXP-024 — Natural on-policy alias dynamics

Train the standard surface-GRPO policy while logging, for every prompt group and
training step:

- generated surface trajectory count;
- unique ranked retrieval-trajectory count;
- alias fraction and largest-class share;
- effective behavior count;
- selected behavior coverage and duplicate rate;
- surface- versus class-level reward variance;
- non-zero-advantage row fraction.

The experiment tests whether natural, non-injected aliases increase during
training and whether the effective behavior count or useful advantage coverage
falls with training progress.

### EXP-025 — Fixed-budget behavior selection

Using cached causal-audit candidates and retrieval results, compare under the
same K-query budget:

- random surface selection;
- lexical text-diversity selection;
- behavior-balanced selection;
- reward-oracle selection as an unattainable upper bound.

The primary comparison is behavior-balanced minus random surface selection for
unique behavior coverage and union immediate support recall. Selection never
uses reward in the proposed method.

### EXP-026 — Response-signature robustness

The primary behavior identity is the exact ranked **full retrieval trajectory**.
The audit tests cheaper approximate signatures:

- unordered document sets per turn;
- top-1 document per turn;
- top-3 ranked documents per turn;
- title-token Jaccard thresholds 0.50, 0.75, and 0.90.

A deployable approximation must preserve pair precision, pair recall, and the
highest-reward exact class in both BM25 and E5. This experiment determines
whether exact document IDs/ranks are required or a cheaper response signature
is safe.

### EXP-027 — Real 2×2 Search-R1 GRPO factorial

Every controlled method generates eight trajectories per prompt and exposes the
same eight retrieval trajectories to the experiment. Four rows are allowed to
contribute actor loss. The factors are:

| Selection | Advantage normalization | Variant |
|---|---|---|
| random surface rows | surface trajectory | `random-surface` |
| behavior-balanced rows | surface trajectory | `balanced-surface` |
| random surface rows | behavior quotient | `random-quotient` |
| behavior-balanced rows | behavior quotient | `balanced-quotient` |

A standard eight-row surface-GRPO run from EXP-024 is retained as the conventional
reference. The selected-row mask is applied to actor loss, so unselected rows do
not contribute PPO or actor KL loss. All controlled cells retain identical
rollout/retrieval calls, generation tokens, model, data, reward, and K=4 actor
rows.

Quotient normalization computes one normalized advantage per response-induced
behavior class, then distributes equal total advantage mass across the selected
members of each class. It does not change the reward and does not use gold
support labels to form classes.

### EXP-028 — Seen, cross-retriever, and hybrid confirmation

Merge the EXP-024 standard and EXP-027 controlled policies and evaluate each on:

- the source retriever;
- the other retriever;
- held-out BM25+E5 RRF hybrid retrieval.

Primary policy metrics are observed supporting-title recall, answer F1, and
search calls. Training telemetry is joined to the evaluation so performance per
retrieval call can be related to behavior coverage, duplicate rate, and
non-zero-advantage coverage.

## One-node layout

The default layout is one CUDA 12.9 node with eight full H100 GPUs:

```text
GPUs 0–6: Search-R1 Qwen2.5-7B FSDP/rollout workers
GPU 7:    E5 FAISS retrieval
CPU:      BM25 retrieval and the driver
```

Default checkpoint:

```text
Qwen/Qwen2.5-7B-Instruct
a09a35458c702b33eeacc393d103063234e8bc28
```

## Inputs

EXP-025 and EXP-026 consume the completed causal-query state JSON files:

```bash
export BEHAVIOR_QUOTIENT_INPUTS=\
'/absolute/path/to/causal_query_audit/results/full/states/*/*.json'
```

EXP-024 and EXP-027 use the pinned Hard-RQ0 train/dev parquet files and live
BM25/E5 services.

## Smoke

```bash
export BASE_MODEL=/absolute/path/to/Qwen2.5-7B-Instruct
export BEHAVIOR_QUOTIENT_INPUTS=\
'/absolute/path/to/causal_query_audit/results/full/states/*/*.json'

PROFILE=smoke bash behavior_quotient/run_all.sh
```

## Pilot

```bash
SKIP_BOOTSTRAP=1 \
PROFILE=pilot \
BASE_MODEL=/absolute/path/to/Qwen2.5-7B-Instruct \
  bash behavior_quotient/run_all.sh
```

## Individual experiments

```bash
PROFILE=pilot bash experiments/EXP-024/run.sh
PROFILE=pilot bash experiments/EXP-025/run.sh
PROFILE=pilot bash experiments/EXP-026/run.sh
PROFILE=pilot bash experiments/EXP-027/run.sh
PROFILE=pilot bash experiments/EXP-028/run.sh
```

Run only the controlled matrix after EXP-024 standard checkpoints exist:

```bash
BQ_SETUP_READY=1 PROFILE=pilot bash behavior_quotient/run_matrix.sh
```

Merge and evaluate:

```bash
PROFILE=pilot bash behavior_quotient/merge_eval.sh
PROFILE=pilot bash behavior_quotient/report.sh
```

## Profiles

| Profile | Seeds | Standard updates | Controlled jobs | Evaluation questions |
|---|---:|---:|---:|---:|
| smoke | 1 | 10 | 8 | 20 |
| pilot | 3 | 100 | 24 | 500 |
| full | 5 | 300 | 40 | 1,000 |

Controlled job count is `2 source backends × 4 cells × seeds`. EXP-024 adds one
standard run per source backend and seed. Standard checkpoints are reused in
EXP-028 instead of being retrained inside EXP-027.

## Outputs

```text
work/behavior_quotient/
├── telemetry/<profile>/<run-id>/telemetry.jsonl
└── reports/<profile>/
    ├── EXP-024/
    │   ├── step_summary.csv
    │   ├── run_slopes.csv
    │   └── EXP024_REPORT.md
    ├── EXP-025/
    │   ├── selection_rows.csv
    │   ├── paired_contrasts.csv
    │   └── EXP025_REPORT.md
    ├── EXP-026/
    │   ├── signature_rows.csv
    │   ├── signature_summary.csv
    │   └── EXP026_REPORT.md
    └── EXP-028/
        ├── evaluation_episodes.csv
        ├── variant_means.csv
        ├── paired_contrasts.csv
        ├── telemetry_means.csv
        └── EXP028_REPORT.md
```

## Decision logic

The paper direction proceeds only when all of the following survive:

1. natural aliasing rises or effective behavior coverage falls during standard
   on-policy GRPO;
2. behavior-balanced selection improves fixed-budget behavior coverage and
   evidence union without reward leakage;
3. the selected response signature has high precision and recall relative to
   exact ranked trajectory identity;
4. the controlled 2×2 shows positive behavior-coverage and evidence effects,
   with answer F1 non-inferior;
5. the joint method improves standard GRPO on seen/cross retrieval and on the
   held-out hybrid without increasing search calls.

An offline alias-injection result or a query-NLL result alone is not a GO.
