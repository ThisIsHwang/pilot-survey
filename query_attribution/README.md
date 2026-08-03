# EXP-016–019 — Attribution-controlled query-equivalence matrix

EXP-015 showed a large benefit from training on multiple functionally equivalent query wordings, but did not isolate why that benefit appeared. This suite tests all plausible explanations in one controlled matrix before any full GRPO implementation.

## Hypotheses

| ID | Hypothesis | Primary comparison |
|---|---|---|
| H1 | Functional equivalence adds value beyond generic multi-query augmentation | `strict-uniform - random-k` |
| H2 | Functional equivalence adds value beyond direct-evidence filtering | `strict-uniform - all-direct-k` |
| H3 | Functional equivalence adds value beyond lexical diversity/length matching | `strict-uniform - diversity-matched-k` |
| H4 | The EXP-015 gain is mostly generic multi-target augmentation | `strict-uniform - factual-replicated-k` |
| H5 | Worst-member-focused training improves the weakest equivalent wording | `strict-hardmax - strict-uniform` |
| H6 | Consistency regularization reduces final within-class NLL dispersion | `strict-consistency - strict-uniform` |
| H7 | Immediate evidence equivalence is sufficient; suffix replay is unnecessary | `immediate-only ≈ strict-immediate-control` |
| H8 | Final outcome equivalence is sufficient; immediate matching is unnecessary | `final-only ≈ strict-final-control` |
| H9 | The effect transfers across retrievers rather than only fitting the source view | seen/cross portability |
| H10 | The effect is not caused only by the synthetic alternative-query generator | factual/synthetic subgroup gain |
| H11 | The effect is stronger for larger or more lexically diverse classes | class-size/diversity subgroups |
| H12 | NLL improvements produce actual held-out evidence gain | EXP-019 interactive retrieval |

## Compute and data controls

Within each hypothesis family, the suite fixes source states, held-out grids, two target sequences per state, total state credit one, tokenizer target-token imbalance at most 25%, model, LoRA rank, optimizer steps, seeds, and positive-only credit. Non-finite values and incomplete paired grids fail closed.

`factual-replicated-k` matches target forwards without adding information. `diversity-matched-k` chooses a non-equivalent query with lexical distance and length closest to the strict partner. `all-direct-k` controls evidence quality without equivalence structure.

## Class definitions

- strict: same immediate support set, final support set, and answer EM;
- immediate: same immediate support set only;
- final: same final support set and answer EM.

## Objectives

- `strict-uniform`: mean member NLL;
- `strict-hardmax`: smooth maximum NLL;
- `strict-consistency`: mean NLL plus within-class NLL variance.

Final wording sensitivity is `baseline class NLL std - adapted class NLL std`, not the standard deviation of gains.

## Run

The branch is stacked on EXP-015, and EXP-015 prepared states must exist.

```bash
export QUERY_ATTRIBUTION_BASE_MODEL=/absolute/path/to/Qwen2.5-7B-Instruct
PROFILE=smoke bash query_attribution/run_all.sh
PROFILE=pilot bash query_attribution/run_all.sh
```

To rebuild EXP-015 preparation from raw causal-audit state files:

```bash
export QUERY_ATTRIBUTION_INPUTS='/absolute/path/to/causal_query_audit/results/full/states/*/*.json'
REFRESH_EQUIVALENCE_PREPARE=1 PROFILE=pilot bash query_attribution/run_all.sh
```

Run one family:

```bash
PROFILE=pilot bash experiments/EXP-016/run.sh
PROFILE=pilot bash experiments/EXP-017/run.sh
PROFILE=pilot bash experiments/EXP-018/run.sh
```

Interactive confirmation:

```bash
PROFILE=pilot bash experiments/EXP-019/run.sh
```

Full sequence:

```bash
PROFILE=pilot bash query_attribution/run_everything.sh
```

## Scale

There are 11 NLL variants. Pilot uses two directions and three seeds, for 66 one-GPU jobs. Full uses five seeds, for 110 jobs. EXP-019 evaluates six selected adapter variants and reserves GPU 7 for E5 FAISS.

## Interpretation

- H1/H2/H3 pass: equivalence structure itself is supported.
- H4 only: generic multi-query augmentation explains EXP-015.
- H5: use a worst-member-aware objective.
- H6: consistency regularization is the robustness mechanism.
- H7: immediate support equivalence is enough; remove suffix replay.
- H8: final outcome equivalence is enough; immediate matching is unnecessary.
- NLL pass but EXP-019 fail: stop because the proxy does not improve real retrieval.
- EXP-019 pass: proceed to full equivalence-aware query-token GRPO under matched search-call budgets.
