# Retrieval-derived query-credit audit (EXP-050–056)

This suite keeps the original Search-R1 interface unchanged:

```text
query → one top-k=3 retrieval call → all three documents are shown to the agent
```

No post-retrieval selector, document dropping, reranking, or larger candidate pool is introduced in the training comparison. The suite tests a narrower claim:

> A useful retrieved document can provide a dense learning signal without being a causally valid estimate of the producing query's marginal value.

## Scientific chain

### EXP-050 — matched query and document interventions

For every frozen search state, execute several state-matched query candidates under common continuation seeds. For each query, execute the complete top-3 observation and three document-omission branches.

Outputs:

- query reward under the fixed suffix policy;
- query indispensability: reward minus the best alternative-query reward;
- centered query-action advantage;
- document omission utility for every returned document;
- document-derived query credit under multiple aggregations;
- retrieval-equivalence class and class size.

### EXP-051 — surrogate validity and alias stratification

Tests whether document-derived query credit agrees with query indispensability. Primary diagnostics are correlation, sign agreement, false-positive action credit, top-query agreement, and excess over-credit in alias classes of size three or greater.

### EXP-052 — search-span gradient audit

Using the same Qwen2.5-7B LoRA parameters and the same state-query examples, compute gradients under:

- state-relative query-intervention credit;
- positive document utility propagated to the query;
- alias-normalized document credit;
- shuffled document credit;
- outcome reward.

The loss covers only the generated `<search>…</search>` action. A multiplicity stress duplicates retrieval-equivalent surface queries while holding the underlying document utility fixed.

### EXP-053 — matched LoRA micro-update

Every method starts from the same base checkpoint and sees the same training examples. On question-heldout states, the updated model scores the same candidate query bank; the highest-likelihood query is evaluated with its already executed retrieval trajectory. This turns the gradient diagnostic into an actual query-selection result without relying on query NLL alone.

### EXP-054 — frozen document-utility estimator

Trains a question-heldout ridge ensemble from document-omission labels using only post-retrieval information available online. The JSON artifact is shared by every GRPO condition.

### EXP-055 — real Search-R1 GRPO

All cells use ordinary IID rollout generation, top-k=3, and the same trajectory reward. Only query-span credit differs:

| Method | Additional search-span signal |
|---|---|
| `outcome` | none |
| `doc-to-action` | normalized predicted document utility |
| `alias-normalized` | one class-level credit mass divided across retrieval-equivalent aliases |
| `shuffled-doc` | state-shuffled document utility control |

### EXP-056 — held-out endpoint and paper gate

Evaluates BM25- and E5-trained checkpoints on BM25, E5, and BM25+E5 RRF. A method GO requires the alias-normalized signal to improve seen and cross support recall without answer, search-call, or protocol regression.

## Run

```bash
export BASE_MODEL=/absolute/path/to/Qwen2.5-7B-Instruct
export QUERY_CREDIT_INPUTS='/absolute/path/to/causal_query_audit/results/full/states/*/*.json'

PROFILE=smoke SKIP_GRPO=1 SKIP_ENDPOINT=1 bash query_credit/run_all.sh
PROFILE=pilot bash query_credit/run_all.sh
```

Individual stages:

```bash
PROFILE=pilot bash experiments/EXP-050/run.sh
PROFILE=pilot bash experiments/EXP-051/run.sh
PROFILE=pilot bash experiments/EXP-052/run.sh
PROFILE=pilot bash experiments/EXP-053/run.sh
PROFILE=pilot bash experiments/EXP-054/run.sh
PROFILE=pilot bash experiments/EXP-055/run.sh
PROFILE=pilot bash experiments/EXP-056/run.sh
```

## Interpretation

The paper story is supported only if:

1. document-derived query credit frequently praises replaceable queries;
2. over-credit increases with retrieval-alias class size;
3. document-derived search-span gradients diverge from query-intervention gradients and drift with alias multiplicity;
4. query-intervention supervision selects better held-out queries than document-derived supervision;
5. alias-normalized online credit improves actual held-out retrieval over raw document-to-action credit.

If document credit agrees with query intervention and trains equally good policies, the correct conclusion is that causal attribution is imperfect but unnecessary for optimization.
