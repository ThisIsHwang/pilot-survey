# EXP-012: Positive-only recovery × depth factorial

EXP-012 is the minimum follow-up after EXP-009 through EXP-011 produced two
non-finite contrasts and did not support the original TRACE premises. It tests
only the surviving scientific question:

> Does a query reformulation that actually adds supporting evidence teach a
> held-out retriever better than an executed reformulation with no evidence
> gain, after search depth is controlled?

This is a one-node CUDA 12.9 / 8xH100 diagnostic using
`Qwen/Qwen2.5-7B-Instruct` LoRA micro-updates. It is not full GRPO and it does
not claim final interactive-agent improvement.

## Why this experiment is different

The prior EXP-010 compared a short recovered trajectory with a deep failed
trajectory while also assigning a negative likelihood weight to zero-gain
queries. That comparison confounded recovery with depth and made the negative
cell numerically unbounded.

EXP-012 removes both problems:

1. every training query receives positive imitation weight `+1.0`;
2. recovery and depth are crossed in a complete 2×2 design.

| Cell | First useful/selected reformulation | Evidence gain |
|---|---:|---:|
| short recovered | turn 2 or earlier | positive |
| short unrecovered | final query by turn 2 | zero |
| deep recovered | first positive-gain query at turn 3+ | positive |
| deep unrecovered | final query at turn 3+ | zero |

The four curricula have equal example counts, optimizer steps, base model, LoRA
configuration, and held-out target-retriever query grid. Dataset, source
backend, top-k, continuous question difficulty, and approximate prompt/target
length are matched before training.

## Primary effect

For held-out NLL gain `G`, the preregistered primary contrast is

```text
Recovery main effect
= 0.5 * [(G_short,recovered - G_short,unrecovered)
       + (G_deep,recovered  - G_deep,unrecovered)]
```

The secondary effects are:

```text
Depth main effect
= 0.5 * [(G_deep,recovered   - G_short,recovered)
       + (G_deep,unrecovered - G_short,unrecovered)]

Recovery × depth interaction
= (G_deep,recovered - G_deep,unrecovered)
  - (G_short,recovered - G_short,unrecovered)
```

The primary GO rule is:

- combined recovery main effect ≥ `0.02` nats per target token;
- hierarchical-bootstrap lower bound > 0;
- recovered curricula have positive absolute mean held-out gain;
- both BM25→E5 and E5→BM25 recovery point estimates are non-negative.

A GO justifies a later interactive evidence-gain curriculum experiment. A
NO-GO means the observed portable gain is more likely explained by generic good
query imitation or data diversity than by recovery status.

## Inputs

The experiment reuses the signed raw trajectory bank from PR #9:

```text
work/trace_go/bank/episodes.jsonl
work/trace_go/bank/transitions.jsonl
work/trace_go/bank/manifest.json
```

Build it from raw Hard-RQ0/EXP-008 JSONL if needed:

```bash
TRACE_INPUTS='/absolute/path/to/raw/episodes/*.jsonl' \
  bash trace_go/prepare_bank.sh
```

## Smoke run

```bash
export TRACE_INPUTS='/absolute/path/to/raw/episodes/*.jsonl'
export TRACE_BASE_MODEL='/absolute/path/to/Qwen2.5-7B-Instruct'

PROFILE=smoke bash trace_factorial/run_all.sh
cat work/trace_factorial/reports/smoke/EXP012_REPORT.md
```

Smoke uses one seed and 8 examples per factorial cell. It validates the code
path only.

## Pilot run

```bash
SKIP_BOOTSTRAP=1 \
SKIP_BANK=1 \
TRACE_BASE_MODEL='/absolute/path/to/Qwen2.5-7B-Instruct' \
PROFILE=pilot \
  bash trace_factorial/run_all.sh
```

Pilot configuration:

```text
seeds: 13, 42, 87
transfer directions: BM25→E5 and E5→BM25
factorial cells: 4
examples per cell: 48
optimizer steps per job: 24
held-out target queries per direction: 192
jobs: 2 × 4 × 3 = 24
```

Eight one-GPU BF16 LoRA jobs run concurrently. The effective training batch is
16 through microbatch 2 and gradient accumulation 8.

## Resume

Every job is content-signed. Re-run the same command; completed jobs with the
same signature are reused.

```bash
SKIP_BOOTSTRAP=1 SKIP_BANK=1 PROFILE=pilot \
  bash trace_factorial/run_all.sh
```

Use `--force` only when intentionally rerunning all jobs:

```bash
PROFILE=pilot bash trace_factorial/run.sh --force
```

## Fail-closed behavior

Before GPU launch, the model contract requires one shared 6B–9B checkpoint and
verifies the plan/config hashes. Every training file must contain strictly
positive finite weights. After each job, baseline NLL, adapted NLL, held-out
gain, and training metrics must all be finite. A non-finite job removes its
completion marker, writes `invalid.json`, and makes the scheduler fail. The
report also refuses incomplete grids or variant-specific base-model NLL drift.

## Outputs

```text
work/trace_factorial/reports/<profile>/
├── cell_summary.csv
├── factorial_effect_rows.csv
├── factorial_effects.csv
├── decision.json
└── EXP012_REPORT.md
```

The query-NLL endpoint remains a project-selection proxy. If EXP-012 passes, the
next experiment must train an evidence-gain curriculum search policy and measure
interactive held-out-retriever `EvidenceGain@2/3`, recovery, answer quality, and
search cost under the same call budget.
