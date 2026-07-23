# Numbered experiment suite

Every scientific experiment receives an immutable `EXP-###` identifier. The identifier describes a scientific question, not an individual seed or checkpoint.

## Naming rules

- experiment ID: `EXP-003`
- run ID: `EXP-003__seed-013__profile-pilot__variant-blind`
- checkpoint root: `work/experiments/EXP-003/checkpoints/<run-id>`
- merged model root: `work/experiments/EXP-003/merged/<run-id>`
- results root: `work/experiments/EXP-003/results/<run-id>`
- logs: `logs/experiments/EXP-003/<run-id>.log`

Experiment numbers are never reused or renumbered after results have been produced. Seeds, profiles, reward weights, and variants are encoded in the run ID and completion manifest rather than receiving new experiment numbers.

## Current map

| ID | Name | Purpose |
|---|---|---|
| EXP-001 | Original RQ0 | Small-corpus zero-shot/query-style pilot |
| EXP-002 | Hard RQ0 | Full-wiki BM25/E5 specialist transfer |
| EXP-003 | Mixed-blind GRPO | Shared policy, episode-stable backend assignment, no metadata |
| EXP-004 | Mixed backend-ID oracle | Shared policy with explicit lexical/semantic metadata |
| EXP-005 | Evidence-aware reward | Answer reward plus terminal supporting-evidence recall |
| EXP-006 | Held-out hybrid RRF | Transfer to a BM25+E5 reciprocal-rank-fusion backend |

The canonical machine-readable registry is `experiments/registry.json`. Validate it with:

```bash
python -m stackpilot.experiment_registry validate
python -m stackpilot.experiment_registry list
python -m stackpilot.experiment_registry run-id EXP-003 --seed 13 --profile pilot --variant blind
```

## Scientific ordering

1. Complete EXP-002 before interpreting EXP-003/004.
2. Run EXP-003 with seeds 13, 42, and 87.
3. Run EXP-004 at seed 42 first; expand to three seeds only when metadata value is at least 0.03.
4. EXP-005 is a reward-diagnostic experiment, not the main mixed-policy baseline.
5. EXP-006 is evaluation-only and uses checkpoints from EXP-002 through EXP-004.

## Main decision pattern

The original latent-adaptation idea is supported only when:

```text
specialist oracle > mixed-blind
mixed + backend ID ~= specialist oracle
```

If mixed-blind already matches both specialists, a separate online stack-identification module is not justified.
