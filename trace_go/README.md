# TRACE project go/no-go suite (EXP-009 through EXP-011)

Experiment numbers 007 and 008 are intentionally not reused. When raw EXP-008
trajectory artifacts exist, this suite can consume them as input without
changing their original scientific identity.

This suite tests the three structural premises required before implementing full
TRACE/GRPO. It is designed for one Linux node with eight full NVIDIA H100 GPUs,
CUDA 12.9, and Python 3.12.

The suite **does not** claim final interactive-agent improvement. It performs
small, equal-budget Qwen2.5-7B LoRA updates on query-reformulation examples and
measures held-out evidence-gaining query negative log-likelihood (NLL). This
makes dozens of controlled training interventions feasible on one node. A
positive NLL gain means that a curriculum made useful held-out reformulations
more likely. Only if all three conditions pass should the project spend compute
on full recovery-aware GRPO and interactive held-out-retriever validation.

## Scientific conditions

### EXP-009 / Condition A — recoverability is not generic difficulty

For calibration cells drawn from one retriever, train an equal-budget LoRA
adapter and evaluate evidence-gaining reformulations on the other retriever.
Regress held-out NLL gain on:

```text
portable recovery proxy
+ first-turn success
+ reward variance
+ search depth
+ question difficulty
+ source retriever
```

The portable-recovery coefficient must be positive with a bootstrap interval
above zero and must add predictive R² beyond the controls.

### EXP-010 / Condition B — recovery is better than depth alone

On the same source retriever and under matched dataset/view/difficulty/token
marginals, compare:

```text
short-recovered:
first search missed; a turn-2 reformulation added evidence

versus

deep-unrecovered:
at least three searches; no evidence recovery
```

Both curricula use the same example count, LoRA optimizer steps, model, and
held-out target-retriever probe. The short-recovered curriculum must produce a
larger held-out NLL improvement.

### EXP-011 / Condition C — same-question pairing matters

Compare two equal-budget curricula from the same source retriever:

```text
paired:
source-view recovery is selected only when the same question is also
recoverable in the target retrieval view

unpaired:
globally high-recovery examples selected without using the paired target view
```

The paired curriculum must transfer better and, for the default criterion, must
not have larger across-seed variability.

## Required input

TRACE consumes **raw per-episode JSONL**, not summary CSVs. Each episode must
contain the Hard-RQ0 fields below:

```text
question_id, question, dataset, policy_tag, seed, backend, topk,
search_count, turns[]
```

Each executed-search turn must include `query`, `support_recall`,
`evidence_gain`, and preferably `observed_titles`. Current Hard-RQ0 and numbered
policy evaluation outputs satisfy this contract.

Default discovery patterns include:

```text
work/hard_rq0/runs/*/results/policies/*.jsonl
work/experiments/EXP-002/results/**/*.jsonl
work/experiments/EXP-008/results/**/*.jsonl
```

For another location, use colon- or newline-separated patterns:

```bash
export TRACE_INPUTS='/path/exp008/*.jsonl:/path/other/**/*.jsonl'
```

A primary matrix or report alone is insufficient because Conditions B and C
need the actual turn-level query transitions.

## One-node execution

### 1. Bootstrap the isolated CUDA 12.9 environment

```bash
bash trace_go/bootstrap.sh
```

This creates `.venv-trace` and verifies:

- exactly 8 visible non-MIG H100 GPUs;
- CUDA toolkit 12.9;
- PyTorch CUDA 12.9 wheel;
- pinned Transformers, PEFT, and Accelerate versions.

The default model is `Qwen/Qwen2.5-7B-Instruct`. Before any GPU worker starts,
`trace_go/run_jobs.sh` reconstructs the configured architecture on meta tensors
and requires an exact parameter count between 6 and 9 billion. A stale 3B plan,
a mixed-model job list, or a config changed after planning fails before GPU
allocation.

The 7B defaults use BF16 LoRA with gradient checkpointing, microbatch 2,
gradient accumulation 8, and evaluation batch 4. The effective training batch
therefore remains 16, matching the original diagnostic design while retaining
additional H100 memory headroom.

### 2. Build the paired trajectory bank

```bash
TRACE_INPUTS='/absolute/path/to/raw/*.jsonl' \
  bash trace_go/prepare_bank.sh
```

Outputs:

```text
work/trace_go/bank/episodes.jsonl
work/trace_go/bank/transitions.jsonl
work/trace_go/bank/manifest.json
```

Question IDs are hash-split into train, calibration, and held-out sets. The
bank computes easy/recoverable/unrecoverable labels, recovery score, reward
variance, question difficulty, and a paired portable-recovery proxy.

### 3. Plan equal-budget jobs

Use a local model snapshot when the cluster is offline:

```bash
PROFILE=smoke \
TRACE_BASE_MODEL=/absolute/path/to/Qwen2.5-7B-Instruct \
  bash trace_go/plan.sh
```

Profiles:

| Profile | Purpose | A jobs | B jobs | C jobs |
|---|---|---:|---:|---:|
| smoke | code-path validation | 10 | 4 | 4 |
| pilot | project decision | 48 | 12 | 12 |
| full | stronger replication | 72 | 12 | 12 |

The exact count follows the two predeclared transfer directions and profile
seeds. Every job has a content signature and is resumable. Planning must be
rerun after changing the model or configuration because those identities are
part of each job signature.

### 4. Run the experiments

All eight GPUs run independent one-GPU Qwen2.5-7B LoRA jobs in parallel:

```bash
PROFILE=smoke bash trace_go/run_a.sh
PROFILE=smoke bash trace_go/run_b.sh
PROFILE=smoke bash trace_go/run_c.sh
```

For the project decision:

```bash
PROFILE=pilot bash trace_go/run_a.sh
PROFILE=pilot bash trace_go/run_b.sh
PROFILE=pilot bash trace_go/run_c.sh
```

Override allocation when necessary:

```bash
TRACE_GPUS='0 1 2 3 4 5 6 7' TRACE_WORKERS=8 \
PROFILE=pilot bash trace_go/run_jobs.sh --experiment EXP-009
```

Each process sees one physical GPU through `CUDA_VISIBLE_DEVICES`. No retriever
server is needed during the LoRA phase because retrieval interactions are
already frozen in the signed trajectory bank. A successful launch writes the
verified model identity and parameter count to
`work/trace_go/plans/<profile>/model_contract.json`.

The micro-update uses **query-only signed credit**. A reformulation that added
new observed supporting evidence receives positive likelihood weight. An
executed reformulation with zero evidence gain receives a smaller negative
weight, so its likelihood is reduced rather than imitated. This approximates
the direction of policy-gradient credit while holding model, optimizer steps,
example count, and target-retriever probe fixed across Conditions B and C.

### 5. Generate the decision report

```bash
PROFILE=pilot bash trace_go/report.sh
cat work/trace_go/reports/pilot/TRACE_GO_REPORT.md
```

Primary outputs:

```text
condition_a_coefficients.csv
condition_a.json
condition_b_directions.csv
condition_b.json
condition_c_directions.csv
condition_c.json
job_summary.csv
decision.json
TRACE_GO_REPORT.md
```

## End-to-end command

```bash
TRACE_INPUTS='/absolute/path/to/exp008/raw/*.jsonl' \
TRACE_BASE_MODEL=/absolute/path/to/Qwen2.5-7B-Instruct \
PROFILE=pilot \
  bash trace_go/run_all.sh
```

Rerun without reinstalling or rebuilding the bank:

```bash
SKIP_BOOTSTRAP=1 SKIP_BANK=1 SKIP_PLAN=1 PROFILE=pilot \
  bash trace_go/run_all.sh
```

## Default go criteria

- **A:** standardized portable-recovery coefficient ≥ 0.05, bootstrap lower
  bound > 0, incremental R² ≥ 0.01.
- **B:** short-recovered minus deep-unrecovered held-out gain ≥ 0.02 nats per
  target token, hierarchical bootstrap lower bound > 0.
- **C:** paired minus unpaired held-out gain ≥ 0.01 nats per target token,
  hierarchical bootstrap lower bound > 0, and paired seed standard deviation no
  larger than unpaired.

These thresholds are project gates, not claims of statistical confirmation.
After a pass, the next experiment must train a full search policy and confirm
turn-wise evidence gain on a held-out retriever under matched search-call
budgets.
