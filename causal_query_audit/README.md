# EXP-013: Causal Query Signal Audit

EXP-013 is the low-cost prerequisite for a top-conference project on causal
credit assignment for multi-step search queries. It does **not** train a new
search policy. It asks whether immediate evidence gain is sufficiently different
from a query's downstream trajectory effect to justify developing a replay-based
causal estimator.

The experiment is designed for one Linux node with eight full NVIDIA H100 GPUs,
CUDA 12.9, and Qwen2.5-7B-Instruct.

## Scientific question

At a fixed search state, an executed query can contribute in three ways:

```text
direct query:
  immediately retrieves new supporting evidence

bridge query:
  retrieves no supporting evidence immediately
  but exposes an entity used by the next query
  and the next search retrieves evidence

redundant direct query:
  immediately retrieves evidence
  but state-matched alternatives reach the same or better final outcome
```

The audit tests whether local evidence gain is an incomplete proxy for the
query's total downstream utility.

## State-matched intervention

For every selected state before search turn 2 or 3, EXP-013 constructs:

```text
factual query from the recorded trajectory
+ lexical alternative
+ semantic alternative
+ discovered-entity alternative
```

All candidates receive the exact same:

- question;
- prefix queries;
- reconstructed prefix retrieval observations;
- retriever and top-k;
- observation-token budget;
- frozen Qwen2.5-7B continuation policy;
- maximum total search budget.

Each candidate is executed, then the remaining suffix is replayed from the
resulting observation. The intervention therefore changes the query action, not
the prefix state or retriever snapshot.

The factual query is replayed as a provenance check. Its observed titles,
immediate support gain, and recall after the intervention must match the raw
trajectory. A mismatch aborts the state instead of silently changing the causal
treatment.

## Query effects

For candidate `i` in a state with `K` candidates:

```text
Support TQE_i
  = final support recall_i
  - mean(final support recall of the other K-1 candidates)

Direct effect_i
  = immediate support gain_i
  - mean(immediate gain of the other candidates)

Downstream effect_i
  = post-intervention support gain_i
  - mean(post-intervention support gain of the other candidates)
```

The implementation checks:

```text
Support TQE = Direct effect + Downstream effect
```

up to floating-point tolerance.

A **mediated bridge** candidate satisfies all of the following:

1. immediate support gain is zero;
2. the continuation policy issues another search query;
3. that next query gains supporting evidence;
4. the next query contains a content token newly exposed in the intervention's
   observed document titles.

A **redundant direct** candidate has positive immediate support gain but
non-positive total support effect relative to its state-matched alternatives.

## Primary project gates

The causal-credit project proceeds only if the pilot shows all of the following:

1. Spearman correlation between direct effect and total support effect is at
   most `0.40`;
2. mediated bridge queries are at least `15%` of zero-immediate-gain candidate
   branches;
3. redundant direct queries are at least `5%` of direct-gain candidate branches;
4. mediated bridge queries have mean downstream effect of at least `0.02` with
   a state-cluster bootstrap lower bound above zero;
5. bridge examples occur under both BM25 and E5.

These are project-selection thresholds, not final confirmatory claims.

## Required raw input

The input must be raw Hard-RQ0 or EXP-008 episode JSONL, not a summary CSV. Every
row must contain:

```text
question_id
question
answers
support_titles
dataset
backend
topk
policy_tag
seed
turns[]
```

Every turn needs at least:

```text
query
observed_titles or retrieved_titles
support_recall
evidence_gain
```

Default discovery patterns are defined in `configs/causal_query_audit.yaml`.
Override them with a colon- or newline-separated value:

```bash
export CAUSAL_QUERY_INPUTS='/path/a/*.jsonl:/path/b/**/*.jsonl'
```

## Hardware layout

```text
GPU 0-6: vLLM data-parallel Qwen2.5-7B replicas
GPU 7:   E5 FAISS retrieval
CPU:     BM25 retrieval and orchestration
```

The vLLM suffix policy is deterministic by default (`temperature=0`). Alternative
query generation uses controlled sampling and retries, then rejects duplicates,
invalid lengths, missing known entities, tags, and malformed JSON.

## Smoke run

```bash
git checkout agent/add-causal-query-signal-audit

export CAUSAL_QUERY_INPUTS='/absolute/path/to/raw/episodes/*.jsonl'
export CAUSAL_QUERY_BASE_MODEL='/absolute/path/to/Qwen2.5-7B-Instruct'

PROFILE=smoke bash causal_query_audit/run_all.sh
```

Smoke configuration:

```text
6 intervention states per backend
2 alternatives per state
4 concurrent orchestration workers
200 bootstrap samples
```

Read:

```bash
cat work/causal_query_audit/reports/smoke/CAUSAL_QUERY_AUDIT_REPORT.md
```

Smoke validates the code path only.

## Pilot run

After smoke passes:

```bash
SKIP_BOOTSTRAP=1 \
SKIP_ASSETS=1 \
CAUSAL_QUERY_INPUTS='/absolute/path/to/raw/episodes/*.jsonl' \
CAUSAL_QUERY_BASE_MODEL='/absolute/path/to/Qwen2.5-7B-Instruct' \
PROFILE=pilot \
  bash causal_query_audit/run_all.sh
```

Pilot configuration:

```text
120 intervention states per backend
3 alternatives per state
240 total states
960 candidate branches
28 concurrent orchestration workers
5,000 state-cluster bootstrap samples
```

The actual number of LLM requests depends on whether the suffix policy answers
or searches. Every branch has one forced intervention search and can use only
the remaining portion of the common four-search budget.

## Reuse existing services

When compatible retriever and vLLM services are already running on the configured
ports:

```bash
SKIP_BOOTSTRAP=1 \
SKIP_ASSETS=1 \
SKIP_PREPARE=1 \
SKIP_SERVICES=1 \
PROFILE=pilot \
  bash causal_query_audit/run_all.sh
```

## Resume

Each state result is independently content-signed and written atomically. Re-run
the same command after interruption. Compatible completed states are reused;
stale results are renamed with a timestamp.

## Outputs

```text
work/causal_query_audit/
├── states/<profile>/
│   ├── states.jsonl
│   └── manifest.json
├── results/<profile>/
│   ├── run_summary.json
│   └── states/<backend>/<state-id>.json
└── reports/<profile>/
    ├── candidate_metrics.csv
    ├── proxy_correlations.csv
    ├── prevalence.csv
    ├── subgroups.csv
    ├── style_summary.csv
    ├── bridge_examples.csv
    ├── redundant_examples.csv
    ├── bootstrap_metrics.json
    ├── decision.json
    └── CAUSAL_QUERY_AUDIT_REPORT.md
```

## Interpretation

### GO

A GO means immediate evidence gain is not enough to explain query utility, and
there is a nontrivial population of query actions whose value is mediated through
later queries. The next project stage should then:

1. collect replay labels at larger scale;
2. train a low-cost total/direct/bridge-effect estimator;
3. apply predicted query credit to query tokens during policy optimization;
4. compare with immediate evidence gain, STAMP-style provenance, information
   gain, LLM-critic credit, and outcome-only GRPO;
5. evaluate interactive held-out retrievers under equal search-call budgets.

### NO-GO

A NO-GO means immediate evidence gain or simpler observable proxies explain most
of the useful action ordering, or bridge/redundant actions are too rare to justify
expensive counterfactual replay and causal mediation modeling.

## Limitations of this audit

- Alternative queries are generated rather than randomized human actions.
- The continuation policy is frozen Qwen2.5-7B, so causal effects are policy
  dependent.
- The primary reward is support-title recall; final answer F1 and search cost are
  secondary components.
- A successful audit establishes the existence and prevalence of a signal gap,
  not that a learned causal-credit policy will improve final performance.
