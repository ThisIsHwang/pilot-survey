# EXP-016 — Attribution-controlled query-class supervision

EXP-015 found a large cross-retriever benefit from training on multiple members
of a functional query-equivalence class, but the original comparison did not
fully separate equivalence structure from generic multi-target augmentation,
direct-evidence filtering, or target-token compute. EXP-016 is the attribution
correction.

The scientific question is:

> Does functional equivalence itself improve portable query supervision after
> the number of target sequences, state credit, source states, optimizer steps,
> tokenizer-level target budget, and direct-evidence quality are controlled?

This is a project-selection diagnostic for one CUDA 12.9 node with eight full
H100 GPUs. It reuses EXP-015's strict counterfactual equivalence states and
Qwen2.5-7B grouped LoRA runner. It does not yet train a full interactive search
policy.

## Four fixed-K curricula

Every source state contributes exactly `K=2` target sequences and exactly one
unit of total state credit in every variant.

| Variant | Targets from the same state |
|---|---|
| `factual-replicated-k` | factual query repeated K times |
| `random-k` | factual query plus K-1 random valid non-equivalent alternatives |
| `all-direct-k` | factual query plus K-1 direct-evidence queries outside the best equivalence class |
| `equivalence-k` | factual query plus K-1 members of its functional equivalence class |

All target weights equal `1/K`. Therefore:

- every state has the same number of target forward passes;
- every state has the same total optimization weight;
- factual replication controls target count and nominal compute;
- random-K controls generic multi-query diversity;
- all-direct-K controls evidence quality without using equivalence structure;
- equivalence-K is the proposed attribution rule.

The planner uses the exact Qwen tokenizer, filters states whose target-token
budgets differ too much across variants or seeds, and writes the final token
budget into every job manifest.

## State requirements

A training state must contain:

- one protocol-valid factual query that is direct and belongs to the best class;
- at least one other best-class member;
- at least one protocol-valid direct query outside the best class;
- at least one protocol-valid non-equivalent alternative;
- exact tokenizer target totals within the configured 25% state-level bound
  across all variants and all predeclared seeds.

The strict outside-class direct requirement ensures `all-direct-k` is not merely
a relabeled equivalence curriculum.

## Cross-retriever design

```text
BM25 source states -> E5 held-out states
E5 source states   -> BM25 held-out states
```

Source and target question IDs are hash-split by EXP-015 before EXP-016 sees
them. Every variant in a direction uses the same source states and exact same
held-out target grid.

## Corrected robustness metrics

EXP-015 reported the standard deviation of **NLL gain** across class members.
That statistic can increase even when final query likelihoods become more
uniform. EXP-016 instead reports:

```text
baseline class NLL std
adapted class NLL std
final dispersion reduction = baseline std - adapted std
```

It also reports:

- class-mean NLL gain;
- worst final-class NLL improvement;
- non-factual class-member gain;
- positive member coverage;
- class versus off-class direct-query margin;
- style-specific gain;
- actual target-count and tokenizer-token balance;
- across-seed variability.

## Primary contrasts

The attribution claims depend on two comparisons:

```text
equivalence-k - random-k
equivalence-k - all-direct-k
```

`equivalence-k - factual-replicated-k` is descriptive only: it confirms the
multi-target effect but cannot isolate equivalence structure.

## Project gate

EXP-016 is a GO only when all of the following hold:

1. equivalence-K beats random-K in class-mean gain by at least `0.020`
   nats/token with a hierarchical-bootstrap lower bound above zero;
2. equivalence-K beats all-direct-K in class-mean gain by at least `0.010`
   with lower bound above zero;
3. equivalence-K improves worst final-member NLL over random-K by at least
   `0.005` with lower bound above zero;
4. equivalence-K improves final within-class dispersion over random-K by at
   least `0.005` with lower bound above zero;
5. equivalence-K is not worse than all-direct-K on dispersion reduction;
6. both transfer directions have non-negative point estimates for the two
   primary mean-gain contrasts;
7. equivalence-K has no greater seed variability than random-K;
8. the final plan remains inside the preregistered target-token tolerance.

A GO justifies independently generated query evaluation and an interactive
held-out-retriever experiment. It does not by itself justify a full GRPO paper
claim.

## One-node execution

This PR is stacked on EXP-015. First prepare the EXP-015 equivalence states from
raw EXP-014 state JSON files:

```bash
git checkout agent/add-attribution-controlled-equivalence

export QUERY_ATTRIBUTION_INPUTS='/absolute/path/to/causal_query_audit/results/full/states/*/*.json'
export QUERY_ATTRIBUTION_BASE_MODEL='/absolute/path/to/Qwen2.5-7B-Instruct'

PROFILE=smoke bash query_attribution/run_all.sh
```

After smoke succeeds:

```bash
SKIP_BOOTSTRAP=1 \
SKIP_PREPARE=1 \
QUERY_ATTRIBUTION_BASE_MODEL='/absolute/path/to/Qwen2.5-7B-Instruct' \
PROFILE=pilot \
  bash query_attribution/run_all.sh
```

Pilot geometry:

```text
Directions: 2
Variants: 4
Seeds: 13, 42, 87
Source states per direction: 24
Held-out states per direction: 32
Optimizer steps: 24
Total one-GPU jobs: 24
```

Full geometry:

```text
Seeds: 13, 42, 87, 101, 131
Source states per direction: 48
Held-out states per direction: 64
Optimizer steps: 48
Total one-GPU jobs: 40
```

Eight H100s execute up to eight independent Qwen2.5-7B LoRA jobs in parallel.

## Outputs

```text
work/query_attribution/
├── plans/<profile>/
│   ├── jobs.jsonl
│   ├── manifest.json
│   ├── data/
│   └── outputs/
└── reports/<profile>/
    ├── compute_budget.csv
    ├── target_losses.csv
    ├── state_metrics.csv
    ├── variant_summary.csv
    ├── style_summary.csv
    ├── contrast_rows.csv
    ├── contrasts.csv
    ├── decision.json
    └── EXP016_REPORT.md
```

## Interpretation

- **GO:** equivalence structure contributes beyond matched target count,
  generic query diversity, and direct-evidence filtering. Proceed to an
  independent-generator and interactive retrieval confirmation.
- **Equivalence beats factual only:** multi-query augmentation is useful, but
  equivalence-aware credit is not uniquely supported.
- **Equivalence equals all-direct:** evidence-quality filtering explains the
  result; stop equivalence-specific method development.
- **No dispersion advantage:** class-average likelihood can improve without
  reducing wording sensitivity; do not claim robustness.
