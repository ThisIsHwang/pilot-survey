# CTU credit-routing experiments (EXP-045 through EXP-049)

This suite tests a stronger claim than “useful documents should be retained.” The
same learned document-utility signal is routed to two different policy variables:

1. **action-side routing** — aggregate the utility of documents produced by a
   search query and add it to the producing trajectory's GRPO reward;
2. **observation routing** — use the utility only to decide which retrieved
   documents enter the agent context.

All four cells make one upstream retrieval call for eight candidates and expose
exactly three documents to the agent. Search calls, upstream depth, observation
count, observation token budget, model, prompts, and optimizer geometry are
matched.

| Variant | CTU → producing-policy reward | CTU → document selection |
|---|---:|---:|
| `outcome-only` | no | no; rank top-3 |
| `action-route` | yes | no; rank top-3 |
| `observation-route` | no | yes; utility top-3 |
| `both` | yes | yes; utility top-3 |

The paper hypothesis is not merely that document utility exists. It is that
routing the same utility signal back to the producing action is ineffective or
harmful when multiple replaceable queries can expose the same evidence, whereas
routing it to the observation decision improves end-to-end search.

## EXP-045 — Fixed-budget document utility

For each factual search state, retrieve eight candidates and enumerate all 56
three-document contexts.  A document is scored by exact matched swaps: for every
other candidate and every shared two-document context, compare replacing the
other candidate with this document.  Companion documents, context cardinality,
query, prefix, continuation model, and suffix random seed are therefore fixed.

The resulting relative utility combines support recall, answer F1, extra search
calls, and invalid actions.  Unlike a single rank-anchored omission, every
document is evaluated against the same symmetric fixed-budget design.

## EXP-046 — Shared utility estimator

A query-only ridge ensemble predicts the EXP-045 utility from information
available at retrieval time: rank, normalized retriever score, query/document
overlap, exact and numeric matches, lengths, duplicate titles, and backend.
Question IDs are hash-split into train, validation, and test. The same frozen
artifact is used by both routing factors, and endpoint evaluation is restricted
to the same globally held-out `test` hash split.

## EXP-047 — Real 2×2 Search-R1 GRPO factorial

The four variants are trained separately on BM25 and E5 for every configured
seed. A retrieval proxy always fetches top-8 and emits top-3. It also attaches a
query-level aggregate of the same predicted document utilities.

When action-side routing is enabled, the aggregate is normalized within each
GRPO prompt group and added to the final valid reward token.  This is a
trajectory-level policy-reward intervention, not a claim that the exact query
token span has been independently identified. When observation routing
is enabled, the proxy selects the three highest-utility documents. No gold
support, answer, CTU label, or future trajectory information is visible online.

## EXP-048 — Seen, cross, and hybrid endpoint

Pre-stop healthy validation-best checkpoints are evaluated on BM25, E5, and
BM25+E5 RRF.  Full-profile evaluation requires a checkpoint manifest so a
safety-stopped run is never silently replaced by its final checkpoint.  Smoke
and pilot runs may explicitly allow a latest-checkpoint fallback for plumbing.
See `credit_routing/checkpoint_manifest.example.csv`. The
primary factorial contrasts are:

```text
action main effect
observation main effect
action × observation interaction
observation-route − action-route
both − outcome-only
```

A method claim requires observation routing to outperform routing the same
signal to the query policy in every BM25/E5 seen and cross direction without
answer, search-call, or protocol regression.

## EXP-049 — Paper gate

The final gate combines:

- prevalence of the document–action mismatch from EXP-034;
- availability of fixed-budget document utility from EXP-045;
- held-out estimator performance from EXP-046;
- end-to-end routing results from EXP-048.

Possible dispositions are `CREDIT-ROUTING-METHOD-GO`, `ANALYSIS-PAPER-ONLY`,
`PREREQUISITE-PENDING`, or `NO-GO`.

## Run

```bash
export BASE_MODEL=/absolute/path/to/Qwen2.5-7B-Instruct
export CREDIT_ROUTING_INPUTS='/absolute/path/to/causal_query_audit/results/full/states/*/*.json'

PROFILE=smoke SKIP_GRPO=1 SKIP_ENDPOINT=1 bash credit_routing/run_all.sh
PROFILE=pilot bash credit_routing/run_all.sh
```

Individual stages:

```bash
PROFILE=pilot bash experiments/EXP-045/run.sh
PROFILE=pilot bash experiments/EXP-046/run.sh
PROFILE=pilot bash experiments/EXP-047/run.sh
PROFILE=pilot bash experiments/EXP-048/run.sh
PROFILE=pilot bash experiments/EXP-049/run.sh
```

## Checkpoint selection contract

For paper-quality `full` evaluation, create:

```text
work/credit_routing/checkpoints/full/selected_checkpoints.csv
```

with columns `source_backend,method,seed,checkpoint_role,model_ref`.  Every one
of the 2 backends × 4 cells × configured seeds must appear exactly once.  The
recommended role is `validation-best`; a common-step checkpoint can be supplied
in a separate evaluation run.  Set `REQUIRE_SELECTED_CHECKPOINTS=0` only for a
non-confirmatory smoke or pilot plumbing check.
