# EXP-024–027 — Multi-positive query generalization suite

The preceding attribution matrix produced three clear findings:

1. compute-matched multi-query supervision was much stronger than repeating the factual query;
2. strict functional equivalence did not consistently beat random, diversity-matched, or all-direct positives;
3. a mean-plus-variance objective reduced final within-class NLL dispersion, but most of the gain appeared on synthetic query styles.

This suite tests the stronger evidence needed for a top-conference claim. It does **not** assume that strict query equivalence is useful.

## EXP-024 — Style- and generator-heldout generalization

For each lexical, semantic, and entity style fold, the training target selector is forbidden from using the held-out style. The same source states and target-count budget compare:

- `all-direct-uniform`;
- `all-direct-consistency`;
- `strict-consistency`.

The primary metric is NLL gain on the excluded style. An optional independent query generator uses a different model and a style-free prompt before planning. Its outputs are added to every held-out evaluation grid as `external-generator` targets.

A useful method must improve excluded-style and independent-generator targets, not only the synthetic styles observed during training.

## EXP-025 — Where does consistency help?

Consistency regularization is crossed with four positive-set constructions:

| Positive set | Uniform | Consistency |
|---|---:|---:|
| random state-matched query | yes | yes |
| diversity-maximized query | yes | yes |
| all direct-evidence queries | yes | yes |
| strict equivalence query | yes | yes |

Every state contributes exactly two targets and one total credit unit. The key question is whether consistency is a generic multi-positive regularizer or depends on functional equivalence.

If `all-direct-consistency` matches or beats `strict-consistency`, the equivalence-specific direction should stop.

## EXP-026 — Set-objective trade-off

The same all-direct pair is trained with:

- uniform mean NLL;
- consistency regularization;
- smooth hardmax;
- set-mass likelihood;
- set-mass plus consistency.

For member NLLs `L_i`, set mass is

```text
L_set = -tau * log sum_i w_i exp(-L_i / tau)
```

It rewards probability mass assigned to any valid query, while consistency and hardmax encourage coverage of weaker members. The experiment tests the expected trade-off between point-policy performance and set coverage.

## EXP-027 — Interactive retrieval and behavior diversity

Selected adapters generate queries on identical held-out target-retriever states under two budgets:

```text
1 generated query / state
4 generated queries / state
```

Every generated query is executed against BM25 or E5. The report measures:

- mean and best evidence gain;
- union evidence gain under four calls;
- number of unique ranked retrieval transitions;
- duplicate-behavior rate;
- lexical query diversity;
- invalid-query rate.

This distinguishes a policy that merely raises target likelihood from one that actually acquires more evidence without collapsing retrieval behavior.

## Run order

```bash
export MULTIPOSITIVE_BASE_MODEL=/absolute/path/to/Qwen2.5-7B-Instruct

git checkout agent/add-multipositive-generalization-suite

PROFILE=smoke bash multipositive_generalization/run_all.sh
PROFILE=pilot bash multipositive_generalization/run_all.sh
```

Optional independent generator:

```bash
export MULTIPOSITIVE_GENERATOR_API_BASE=http://127.0.0.1:9100/v1
export MULTIPOSITIVE_GENERATOR_MODEL=/absolute/path/to/a/different/model

PROFILE=pilot bash multipositive_generalization/generate_external.sh
export MULTIPOSITIVE_EXTERNAL_QUERIES=$PWD/work/multipositive_generalization/external/pilot/queries.jsonl

SKIP_PREPARE=1 PROFILE=pilot bash multipositive_generalization/run_all.sh
```

Interactive retrieval after training:

```bash
bash causal_query_audit/launch_services.sh
PROFILE=pilot bash multipositive_generalization/run_interactive.sh
```

## Pilot geometry

```text
Core variants: 12
Style-heldout variants: 3 methods x 3 styles = 9
Directions: 2
Seeds: 3
Total LoRA jobs: 126
Training states per direction/family: 24
Held-out target states per direction: 32
Optimizer steps: 24
```

The jobs are independent one-GPU Qwen2.5-7B LoRA updates and are scheduled eight at a time on one H100 node.

## Project decisions

- **Style-heldout pass:** consistency improves excluded-style targets by at least 0.02 nats/token with a positive interval.
- **Independent-generator pass:** the effect survives a different query generator and prompt.
- **Equivalence needed:** strict consistency must outperform all-direct consistency. Otherwise retain the cheaper all-direct method.
- **Generic consistency:** dispersion improvement must occur for random, diversity, and all-direct positive sets.
- **Set-objective pass:** set mass must improve interactive one-call or four-call evidence utility without reducing unique retrieval behaviors.
- **Final pass:** NLL gains must survive EXP-027 under matched retrieval-call budgets.

A paper claim should be based on held-out generators and interactive retrieval. In-distribution query NLL alone is not sufficient.
