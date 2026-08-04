# Response-feedback search experiments (EXP-029 through EXP-034)

This stacked suite extends PR #19. It is motivated by the completed causal and
interface audits:

- strict query-equivalence credit did not outperform all-direct supervision;
- query-level and behavior-class counterfactual utility were nearly identical;
- free-form search retained substantially more oracle retrieval utility than a
  finite menu;
- surface alias multiplicity changed real Qwen2.5-7B gradients and reduced
  effective behavior coverage;
- response-conditioned retrieval signatures identified behavior classes far
  more reliably than pre-action semantic or state features.

The suite therefore tests **behavior-efficient rollout generation**, not another
query-credit estimator. The primary method gives the second half of a rollout
group only document titles that were visibly retrieved by the first half. It
never exposes gold support labels, answers, rewards, or hidden retriever scores.

## Experiments

### EXP-029 — Natural on-policy alias dynamics

Reanalyzes standard surface-GRPO telemetry from EXP-024. It measures alias
growth, effective-behavior decline, and fixed-effect lagged associations with
the next-step reward improvement. The method direction proceeds only if the
synthetic gradient mechanism is also visible in natural on-policy training.

### EXP-030 — Fixed-call response-feedback audit

From the same held-out state, Qwen2.5-7B generates eight next-query candidates
under three conditions:

1. `iid`: all eight queries are sampled independently;
2. `text-feedback`: the second four are told only to avoid previous sibling
   wordings;
3. `response-feedback`: the second four see only the document titles returned
   by the first four and are asked to seek a different retrieval outcome.

Every condition receives exactly eight live target-retriever calls. Primary
metrics are unique behavior coverage, duplicate rate, union support recall,
best evidence gain, and invalid-query rate.

### EXP-031 — Sampling × normalization 3×2 factorial

Runs real Search-R1 GRPO with three rollout strategies crossed with two
advantage normalizations:

| Sampling | Surface GRPO | Behavior-quotient GRPO |
|---|---|---|
| IID | `iid-surface` | `iid-quotient` |
| post-hoc behavior-balanced | `posthoc-surface` | `posthoc-quotient` |
| two-phase response-feedback | `feedback-surface` | `feedback-quotient` |

Each controlled cell generates eight trajectories and permits exactly four rows
to update the actor. The retrieval-call budget, generated-token budget, reward,
training questions, model, optimizer steps, and actor-row count are matched.
Validation is always ordinary IID generation, so a feedback-trained checkpoint
must transfer to the normal search-agent interface.

### EXP-032 — Adaptive hybrid interface router

For one free-form query and one deterministic finite-menu query from the same
state, both outcomes are executed to create an automatic training label. A
small question-heldout router uses only pre-retrieval state features to choose
one interface under a single-call budget. It is compared with always-free,
always-menu, random, and an oracle upper bound.

### EXP-033 — Fully paired multi-retriever behavior grid

The exact same normalized query bank is executed against BM25, E5, a BM25+E5
RRF hybrid, and optional ColBERT. It reports behavior-edge agreement, positive
edge Jaccard, best-query agreement, and fixed-budget behavior-balanced sampling
gain per backend. No style label is treated as query identity.

### EXP-034 — Document–action credit closure

Joins document-omission CTU with factual-query TQE and response-induced
behavior-class TQE on the same states. A disagreement prevalence below 10%
closes the causal-credit direction; at least 20% reopens only an observation-
level analysis. It does not automatically justify query-level GRPO.

## One-node layout

```text
GPUs 0–6: Qwen2.5-7B Search-R1 actor/rollout workers
GPU 7:    E5 FAISS retrieval
CPU:      BM25, RRF fusion, and orchestration
```

## Inputs

```bash
export BASE_MODEL=/absolute/path/to/Qwen2.5-7B-Instruct
export BEHAVIOR_FEEDBACK_INPUTS=\
'/absolute/path/to/causal_query_audit/results/full/states/*/*.json'
```

An optional live ColBERT endpoint can be attached for EXP-033:

```bash
export COLBERT_URL=http://127.0.0.1:8103/retrieve
```

## Smoke

```bash
PROFILE=smoke SKIP_GRPO=1 bash behavior_feedback/run_all.sh
```

Run the 3×2 GRPO matrix after the offline smoke checks:

```bash
PROFILE=smoke bash behavior_feedback/run_factorial.sh
PROFILE=smoke bash behavior_feedback/merge_eval.sh
```

## Pilot

```bash
PROFILE=pilot SKIP_GRPO=1 bash behavior_feedback/run_all.sh
PROFILE=pilot bash behavior_feedback/run_factorial.sh
PROFILE=pilot bash behavior_feedback/merge_eval.sh
```

## Individual entry points

```bash
PROFILE=pilot bash experiments/EXP-029/run.sh
PROFILE=pilot bash experiments/EXP-030/run.sh
PROFILE=pilot bash experiments/EXP-031/run.sh
PROFILE=pilot bash experiments/EXP-032/run.sh
PROFILE=pilot bash experiments/EXP-033/run.sh
PROFILE=pilot bash experiments/EXP-034/run.sh
```

## Decision rule

A top-conference method direction requires all of the following:

1. natural alias growth or behavior-count decline predicts weaker next-step
   learning progress;
2. response feedback improves behavior coverage and union evidence under the
   same eight retrieval calls;
3. the feedback-sampling main effect improves actual support recall in every
   BM25/E5 seen and cross direction without answer or search-cost regression;
4. the adaptive router preserves free-form utility while reducing unnecessary
   free-form use;
5. behavior-aware sampling is positive on at least three fully paired retrieval
   environments;
6. document–action CTU either closes the old causal-credit direction or clearly
   isolates a separate observation-level phenomenon.

Query NLL, synthetic alias injection, or oracle candidate-set utility alone are
never GO criteria.
