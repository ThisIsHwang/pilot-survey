# EXP-014 — Query equivalence and credit concentration

EXP-013's causal query audit found that immediate evidence gain usually selects a final-utility-optimal query, while many direct-evidence queries are replaceable by state-matched alternatives. EXP-014 tests whether action-specific first-exposure credit is therefore unnecessarily concentrated on one wording.

The experiment has two layers:

1. **Offline functional-equivalence audit** over existing EXP-013 suffix-replay branches.
2. **Equal-budget Qwen2.5-7B LoRA micro-updates** comparing one-hot and equivalence-aware credit.

It does not rerun retrieval or suffix replay. It consumes the completed causal-audit state JSON files.

## Functional equivalence

Two candidate queries from the same intervention state are equivalent when they produce:

- exactly the same observed gold-support title set;
- the same answer EM result;
- answer F1 within `0.05`;
- the same total search count;
- the same protocol-validity status.

The strict definition is intentional. It treats two wordings as interchangeable only when they lead to the same evidence and operational outcome, rather than merely similar scalar reward.

The audit reports:

- fraction of direct-evidence candidates with an equivalent alternative;
- fraction of states containing a non-trivial direct equivalence class;
- fraction of factual direct queries that are replaceable;
- expected over-allocation from giving one query unit credit instead of splitting credit across its class;
- BM25/E5 stability of pairwise style-equivalence relations for matched question states.

## LoRA credit variants

For each source-retriever training state, the factual query must be direct and belong to an equivalence class of size at least two. All variants use the **same states**, optimizer steps, base model, and held-out target-retriever class grid.

### `factual-onehot`

Give all state credit to the factual query that happened to appear in the recorded trajectory.

### `random-onehot`

Choose one member of the factual query's equivalence class by training seed and give it all state credit. This estimates wording-selection variance.

### `equivalence-normalized`

Train on all class members. The grouped loss first averages member NLLs within each state and then averages states, so a class of size four carries the same total credit as a one-member class.

## Evaluation

Training direction is crossed:

```text
BM25 equivalence classes -> E5 held-out equivalence classes
E5 equivalence classes   -> BM25 held-out equivalence classes
```

Every held-out class contributes all of its members. Per class, the report computes:

- mean NLL improvement across equivalent wordings;
- worst-member NLL improvement;
- within-class gain standard deviation;
- fraction of class members with positive gain.

Primary comparisons are paired by seed and held-out class:

```text
equivalence-normalized - factual-onehot
equivalence-normalized - random-onehot
```

## Pilot gate

A GO requires all of the following:

- replaceable direct-query rate at least `0.40`;
- equivalence-normalized class-mean gain advantage over factual one-hot at least `0.010` nats/token with bootstrap lower bound above zero;
- worst-member advantage at least `0.005` with bootstrap lower bound above zero;
- within-class gain standard deviation no larger than factual one-hot;
- class-mean advantage over random one-hot at least `0.005` with bootstrap lower bound above zero.

A pass only justifies an interactive equivalence-aware policy-credit experiment. It is not a final paper result.

## One-node CUDA 12.9 run

This experiment reuses PR #9's isolated `.venv-trace`, 7B model contract, and CUDA 12.9 packages.

```bash
git checkout agent/add-query-equivalence-credit

export QUERY_EQUIVALENCE_INPUTS='/absolute/path/to/causal_query_audit/results/full/states/*/*.json'
export QUERY_EQUIVALENCE_BASE_MODEL='/absolute/path/to/Qwen2.5-7B-Instruct'

PROFILE=smoke bash query_equivalence/run_all.sh
```

After smoke succeeds:

```bash
SKIP_BOOTSTRAP=1 \
SKIP_PREPARE=1 \
QUERY_EQUIVALENCE_BASE_MODEL='/absolute/path/to/Qwen2.5-7B-Instruct' \
PROFILE=pilot \
  bash query_equivalence/run_all.sh
```

Pilot profile:

```text
2 directions x 3 variants x 3 seeds = 18 jobs
24 source states per direction
32 held-out classes per direction
24 optimizer steps
```

Eight H100s run up to eight independent one-GPU LoRA jobs in parallel.

## Outputs

```text
work/query_equivalence/
├── prepared/
│   ├── states.jsonl
│   ├── classes.jsonl
│   ├── candidates.jsonl
│   ├── paired_states.jsonl
│   └── manifest.json
├── plans/<profile>/
│   ├── jobs.jsonl
│   ├── job_specs/
│   ├── data/
│   └── outputs/
└── reports/<profile>/
    ├── offline_audit.csv
    ├── cross_backend_pairs.csv
    ├── cross_backend_summary.json
    ├── eval_losses.csv
    ├── class_metrics.csv
    ├── contrasts.csv
    ├── seed_stability.csv
    ├── decision.json
    └── EXP014_REPORT.md
```

A strong offline redundancy rate without a LoRA advantage means the phenomenon is real but action-specific credit is harmless at training time. A LoRA advantage in class-mean and worst-member gain, together with lower wording sensitivity, supports equivalence-aware credit as a method direction.
