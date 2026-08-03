# Interface Causality Suite — EXP-020 through EXP-023

This suite is a top-conference project selector built on the completed causal-query state audit. It does not assume that query equivalence itself is novel. Instead, it tests four stronger claims that remain distinguishable from finite-menu systems such as Harness-G and provenance-based credit methods such as STAMP.

## EXP-020 — Surface-alias invariance

Question: if one behavior is represented by more natural-language aliases but the environment and behavior rewards are unchanged, does standard surface-level GRPO normalization change its class-level update?

The experiment injects alias multiplicities 1, 2, 4, 8, and 16 into exact retrieval-transition classes. It compares:

- surface-action advantage normalization;
- behavior-quotient advantage normalization;
- fixed-budget surface sampling;
- fixed-budget quotient sampling.

Primary outcomes are class-level advantage drift, best-class rank flips, effective behavior count, behavior coverage, and best-behavior discovery. A second 7B LoRA gradient probe computes the actual update vector for every query target and measures how much its direction rotates when only alias multiplicity changes.

## EXP-021 — Credit granularity

Question: should evidence credit be assigned to the first observed query action, the individual counterfactual query, the behavior class, or the retrieved document observation?

The base audit compares:

- first-exposure immediate credit;
- query-level counterfactual support TQE;
- behavior-class mean TQE.

An optional document-omission runner removes individual top-k documents, replays the suffix with the same frozen 7B policy, and attaches document CTU labels. This enables observation-versus-action-versus-class comparisons on the same states.

## EXP-022 — Interface expressivity

Question: does a finite entity/title menu reduce aliases by removing useful open-ended actions?

For the same prefix and backend, the experiment executes:

- free-form factual and state-matched alternative queries;
- deterministic finite-menu queries from observed titles and unresolved relation tokens;
- their hybrid union.

It reports oracle immediate evidence gain, behavior coverage, alias rate, menu miss rate, and the free-form advantage on states where the needed gold entity is absent from the prefix.

## EXP-023 — Environment-conditioned equivalence

Question: is behavioral query equivalence a semantic relation, or a state- and retriever-conditioned relation?

Pairwise labels are induced by actual retrieval transitions. Four predictors are compared:

- semantic-only surface features;
- state-conditioned features;
- backend-conditioned features;
- response-conditioned oracle features.

The report includes mixed held-out and BM25→E5/E5→BM25 transfer, plus equivalence-edge stability for matched questions across backends.

## One-node usage

Input is the raw per-state JSON output from the causal-query audit:

```bash
export INTERFACE_CAUSAL_INPUTS='/absolute/path/to/causal_query_audit/results/full/states/*/*.json'
```

Offline experiments:

```bash
PROFILE=pilot bash interface_causality/run_offline.sh
```

Actual update-direction probe on one H100:

```bash
export INTERFACE_BASE_MODEL=/absolute/path/to/Qwen2.5-7B-Instruct
INTERFACE_GRADIENT_GPU=0 PROFILE=pilot \
  bash interface_causality/run_alias_gradient.sh
```

Finite-menu retrieval audit requires the BM25 and E5 services from the causal-query setup:

```bash
bash causal_query_audit/launch_services.sh
PROFILE=pilot bash interface_causality/run_expressivity.sh
```

Optional document CTU labels require the same services plus the frozen Qwen2.5-7B vLLM service:

```bash
PROFILE=pilot bash interface_causality/run_document_ctu.sh
PROFILE=pilot bash interface_causality/run_granularity_with_documents.sh
```

Aggregate all completed decisions:

```bash
PROFILE=pilot bash interface_causality/report.sh
```

## Decision logic

A method paper is justified only when at least one mechanistic audit passes—surface aliasing materially changes the learning signal or action/class credit differs—and at least one open-interface result passes—finite menus miss useful behaviors or equivalence is demonstrably environment-conditioned.

Possible outcomes:

- **EXP-020 + EXP-022:** develop a free-form quotient policy and compare directly with Harness-G.
- **EXP-020 + EXP-023:** develop an environment-conditioned quotient predictor and alias-invariant group optimization.
- **EXP-021 only:** write a causal-credit granularity analysis; do not propose another generic dense reward.
- **EXP-022 only:** focus on the expressivity–creditability tradeoff between free-form and finite action spaces.
- **all fail:** stop equivalence/alias method development and retain the results as boundary-condition analysis.
