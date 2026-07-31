# EXP-015 — Many Queries, Same Evidence

This experiment follows the EXP-014 causal-query audit. That audit found that
immediate support gain selected a final-best query in roughly 94% of states,
bridge queries were rare, but many direct-evidence queries were counterfactually
replaceable. EXP-015 tests the surviving hypothesis:

> When several distinct query wordings expose the same evidence and lead to the
> same outcome, should query credit be assigned to the observed factual action
> alone, or shared across the functional equivalence class?

This is a one-node CUDA 12.9 / 8×H100 project-selection diagnostic. It uses
Qwen2.5-7B LoRA micro-updates and held-out query NLL, not full interactive GRPO.
A GO must be followed by generated-query retrieval evaluation.

## Functional equivalence

For each EXP-014 state, candidate queries are grouped by a strict evidence
signature:

```text
same immediate gold-support title set
+ same final gold-support title set
+ same answer-EM outcome
```

The primary class must:

- contain at least two queries;
- contain the factual trajectory query;
- contain at least one meaningfully different wording with content-token
  Jaccard at most 0.80;
- contain only protocol-valid direct-evidence candidates.

The highest-utility class is selected by final evidence recall, answer EM,
search cost, and class size. All compared training variants use exactly the
same source states and held-out target states.

## Equal-budget curricula

Four variants are trained in both BM25→E5 and E5→BM25 directions:

| Variant | State credit |
|---|---|
| `first-exposure` | factual query alone receives weight 1 |
| `random-member` | one random class member receives weight 1 |
| `equivalence-pool` | every best-class member receives weight `1 / class_size` |
| `all-direct-pool` | every direct-evidence query receives weight `1 / direct_count` |

The LoRA loss is normalized **per state**, not per query. Therefore a state with
four equivalent queries has the same total credit as a singleton state. The
`all-direct-pool` variant controls for the possibility that simply imitating
more queries, rather than equivalence structure, explains the result.

## Primary outcomes

On held-out target-retriever states, every valid candidate query is scored before
and after the micro-update. The report computes:

- mean NLL gain across the best equivalence class;
- worst-member class gain;
- fraction of class members with positive gain;
- within-class NLL dispersion reduction;
- class versus off-class direct-query margin;
- across-seed variability.

The primary contrast is:

```text
equivalence-pool class-mean gain
- first-exposure class-mean gain
```

The key robustness contrast is:

```text
equivalence-pool worst-member gain
- random-member worst-member gain
```

## Input

Use the raw per-state JSON results from EXP-014, not the Markdown report or
`candidate_metrics.csv`:

```bash
export QUERY_EQUIVALENCE_INPUTS='/absolute/path/to/causal_query_audit/results/full/states/*/*.json'
export QUERY_EQUIVALENCE_BASE_MODEL='/absolute/path/to/Qwen2.5-7B-Instruct'
```

Every file must contain the `state`, `prefix`, and `candidates` payload written
by `stackpilot.causal_query_replay`. Mixing multiple EXP-014 run signatures
fails closed.

## Smoke run

```bash
git checkout agent/add-query-equivalence-credit-audit

QUERY_EQUIVALENCE_INPUTS='/absolute/path/to/states/*/*.json' \
QUERY_EQUIVALENCE_BASE_MODEL='/absolute/path/to/Qwen2.5-7B-Instruct' \
PROFILE=smoke \
  bash query_equivalence/run_all.sh
```

Smoke uses one seed, four source states and four held-out states per direction,
and four optimizer steps. It checks only data availability, grouped loss,
model loading, scheduling, and report generation.

## Pilot run

```bash
SKIP_BOOTSTRAP=1 \
QUERY_EQUIVALENCE_INPUTS='/absolute/path/to/states/*/*.json' \
QUERY_EQUIVALENCE_BASE_MODEL='/absolute/path/to/Qwen2.5-7B-Instruct' \
PROFILE=pilot \
  bash query_equivalence/run_all.sh
```

Pilot geometry:

```text
Directions: BM25→E5, E5→BM25
Variants: 4
Seeds: 13, 42, 87
Training states per direction: 24
Held-out states per direction: 24
Optimizer steps: 24
Total jobs: 24
```

The eight H100 GPUs run one independent 7B LoRA job each. Microbatch is one
state group, gradient accumulation is 16, and an equivalence class may contain
up to four query targets inside that group.

## Outputs

```text
work/query_equivalence/
├── prepared/<profile>/
│   ├── state_audit.jsonl
│   ├── eligible_states.jsonl
│   └── manifest.json
├── plans/<profile>/
│   ├── jobs.jsonl
│   ├── manifest.json
│   ├── data/
│   └── outputs/
└── reports/<profile>/
    ├── audit_summary.csv
    ├── target_losses.csv
    ├── state_metrics.csv
    ├── variant_summary.csv
    ├── contrast_rows.csv
    ├── contrasts.csv
    ├── decision.json
    └── EXP015_REPORT.md
```

## Project gate

EXP-015 is a GO only when all of the following hold:

1. at least 20% of direct-evidence states contain a nontrivial equivalence class;
2. equivalence pooling improves class-mean NLL gain over factual-only credit by
   at least 0.01 nats/token with a hierarchical-bootstrap lower bound above 0;
3. equivalence pooling improves worst-member gain over random representative
   credit by at least 0.005 with a lower bound above 0;
4. equivalence pooling is not worse than pooling all direct queries;
5. the factual-credit contrast is non-negative in both transfer directions;
6. equivalence pooling has no larger across-seed variance than random-member
   credit.

These are project thresholds, not final statistical claims.

## Interpretation

- **GO:** implement an interactive equivalence-aware query-credit policy and
  compare it with first-exposure provenance, immediate evidence gain, and
  outcome-only GRPO under matched retrieval-call budgets.
- **NO-GO:** counterfactual query redundancy is descriptive but does not justify
  a separate credit-assignment method. Retain the audit as analysis and stop
  method development.
