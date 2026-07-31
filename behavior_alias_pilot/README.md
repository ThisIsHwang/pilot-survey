# EXP-015: Behavioral alias injection pilot

This experiment is the structural go/no-go test before implementing
Behavior-Quotient GRPO (BQ-GRPO).

The preceding causal-query audit found that immediate support gain was already a
strong query-selection signal, while more than half of direct-evidence queries
were redundant relative to state-matched alternatives. EXP-015 therefore asks a
new question:

> Do multiple text strings that induce the same retrieval transition consume
> rollout budget and reduce useful behavior coverage, and does quotient-class
> selection prevent that loss?

`EXP-014` is intentionally not reused because the completed causal-query report
artifact already carries that identifier. The repository registry records this
new experiment as `EXP-015`.

## What this pilot does

For each unresolved turn-2 or turn-3 state from the causal-query audit:

1. frozen Qwen2.5-7B samples many candidate next-query strings;
2. every query is executed on the original BM25 or E5 backend;
3. queries are placed in the same behavior class only when the visible ranked
   document-ID transition is exactly identical (with title fallback);
4. natural aliasing is measured from the sampled query pool;
5. one non-best behavior class is injected with multiplicity 1, 2, 4, or 8;
6. three fixed-call-budget selectors are compared:
   - `surface`: sample text strings from the alias-expanded pool;
   - `text-diverse`: greedily maximize lexical query distance;
   - `quotient`: balance environment-induced behavior classes.

This is not policy training. It tests whether the structural premise is strong
enough to justify implementing an equivalence predictor and BQ-GRPO.

## Primary metrics

- natural state alias rate;
- unique behavior-class coverage under eight calls;
- effective behavior sample size;
- duplicate-call fraction;
- union supporting-evidence gain;
- best immediate evidence gain found in the rollout group;
- reward variance available to a group-relative update.

The primary alias-injection contrasts compare multiplicity 1 with multiplicity
8 and compare quotient selection with surface selection at multiplicity 8.

## Project gate

The BQ-GRPO project proceeds only when all conditions hold:

- at least 30% of states contain a natural exact-transition alias;
- surface selection loses at least 20 percentage points of normalized behavior
  coverage from multiplicity 1 to 8, with a bootstrap interval above zero;
- quotient selection loses at most 5 percentage points;
- quotient selection recovers at least 20 percentage points of behavior
  coverage at multiplicity 8;
- quotient selection improves union evidence gain by at least 3 percentage
  points, with a bootstrap interval above zero;
- the utility contrast is non-negative for both BM25 and E5.

A pass justifies a learned outcome-equivalence predictor and a true BQ-GRPO
training experiment. A fail means redundancy is descriptively common but not an
important fixed-budget RL problem.

## One-node layout

The default CUDA 12.9 / 8xH100 layout reuses the strict causal-audit service
launcher:

```text
GPU 0-6: seven Qwen2.5-7B vLLM data-parallel replicas
GPU 7:   E5 FAISS retrieval
CPU:     BM25 retrieval and orchestration
```

The model contract rejects checkpoints outside 6B-9B parameters before the GPU
services start.

## Inputs

EXP-015 consumes the prepared state bank from the causal-query audit:

```text
work/causal_query_audit/states/pilot/states.jsonl
```

Override it with:

```bash
export BEHAVIOR_ALIAS_STATES_FILE=/absolute/path/to/states.jsonl
```

The source states must include the exact prior query/result prefix, support
titles, original factual query, and factual replay diagnostics.

## Smoke run

```bash
git checkout agent/add-behavior-quotient-pilot

export BEHAVIOR_ALIAS_BASE_MODEL=/absolute/path/to/Qwen2.5-7B-Instruct
PROFILE=smoke bash behavior_alias_pilot/run_all.sh

cat work/behavior_alias_pilot/reports/smoke/BEHAVIOR_ALIAS_PILOT_REPORT.md
```

Smoke uses six states per backend, eight candidate strings per state, and 100
selection simulations. It validates only the code and service path.

## Pilot run

```bash
SKIP_BOOTSTRAP=1 \
SKIP_ASSETS=1 \
BEHAVIOR_ALIAS_BASE_MODEL=/absolute/path/to/Qwen2.5-7B-Instruct \
PROFILE=pilot \
  bash behavior_alias_pilot/run_all.sh
```

Pilot defaults:

```text
states:          80 BM25 + 80 E5
queries/state:   24
candidate calls: 3,840 actual retrievals
multiplicities:  1, 2, 4, 8
call budget:     8 per simulated rollout group
simulations:     1,000 per state/method/multiplicity
bootstrap:       5,000 state-cluster draws
```

The retrieval results are cached per state, so the large simulation grid does
not issue additional environment calls.

## Resume

State results are content-signed and atomic. Re-run the same command to reuse
completed states. Keep services alive across reruns with:

```bash
KEEP_SERVICES=1 PROFILE=pilot bash behavior_alias_pilot/run_all.sh
```

When preparation and services are already complete:

```bash
SKIP_BOOTSTRAP=1 \
SKIP_ASSETS=1 \
SKIP_PREPARE=1 \
SKIP_SERVICES=1 \
PROFILE=pilot \
  bash behavior_alias_pilot/run_all.sh
```

## Outputs

```text
work/behavior_alias_pilot/
├── states/<profile>/
│   ├── states.jsonl
│   └── manifest.json
├── results/<profile>/
│   ├── run_summary.json
│   └── states/<backend>/<state-id>.json
└── reports/<profile>/
    ├── natural_state_metrics.csv
    ├── natural_alias_summary.csv
    ├── simulation_state_means.csv
    ├── cell_summary.csv
    ├── contrasts.csv
    ├── backend_utility.csv
    ├── qualitative_examples.csv
    ├── decision.json
    └── BEHAVIOR_ALIAS_PILOT_REPORT.md
```

## Interpretation

A GO supports only the following premise:

> Surface-string multiplicity reduces fixed-budget retrieval behavior coverage,
> and an oracle quotient operation avoids that loss.

It does not establish that a learned equivalence detector is accurate or that
BQ-GRPO improves a trained agent. Those are the next two experiments.
