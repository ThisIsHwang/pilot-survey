# Hard RQ0: retrieval-driven specialization under multi-turn search

The first RQ0 used a small HotpotQA context corpus, top-k 10, and usually one search turn. It found a strong general RL improvement but almost no Policy × Backend interaction. This follow-up tests the same hypothesis in an environment where query reformulation can matter.

## What changes

- corpus: full Search-R1 `wiki-18`, not the small per-example context pool
- datasets: 2WikiMultiHopQA and MuSiQue from the existing FlashRAG benchmark collection
- retrieval depth: top-k 3 and 5
- search budget: up to four searches, with turn 1/2/3 metrics recorded separately
- policies: base Qwen plus BM25 and E5 specialists trained from the same initialization
- seeds: 13, 42, and 87 for each specialist
- primary analysis: gain over base Qwen and home-backend excess gain
- hard subset: base Qwen first-turn support recall <= 0.5 on both backends, with evidence recoverable by turn 3

The central interaction is:

```text
(specialist_home - base_home) - (specialist_away - base_away)
```

A specialist improving both backends equally is a general RL effect. It is not retriever specialization.

## Resource layout on one 8xH100 node

During GRPO training:

- GPUs 0-6: Search-R1 actor/rollout/reference workers
- GPU 7: E5 query encoder and flat FAISS index
- CPU: BM25 Lucene server

During evaluation:

- GPUs 0-1: vLLM policy server, tensor parallel size 2
- GPU 7: E5
- CPU: BM25

BM25 and E5 servers stay up throughout the sequential specialist runs. Training top-k is 3; every checkpoint is evaluated at top-k 3 and 5.

## 1. Bootstrap the existing environments

```bash
bash scripts/bootstrap.sh
bash scripts/bootstrap_vllm.sh
bash scripts/preflight.sh
```

## 2. Download the official full-wiki assets

```bash
bash hard_rq0/download_assets.sh
```

This downloads Search-R1's official wiki-18 corpus, BM25 Lucene index, and E5 flat index. Existing Hugging Face caches are reused.

## 3. Prepare existing benchmark annotations

```bash
bash hard_rq0/prepare_data.sh
cat work/hard_rq0/data/SUMMARY.txt
```

No new human annotation is created. The script uses the existing answers and supporting-document metadata. It intentionally stops with diagnostic examples if a dataset revision no longer exposes supporting titles.

## 4. Start full-wiki retrieval servers

```bash
bash hard_rq0/launch_retrievers.sh
```

Health endpoints:

```text
BM25: http://127.0.0.1:8101/health
E5:   http://127.0.0.1:8102/health
```

## 5. Smoke test the evaluation path

```bash
TAG=base-qwen \
SEED=0 \
RESULT_SET=smoke \
MODEL_PATH=/absolute/path/to/Qwen2.5-3B-Instruct \
LIMIT=20 \
  bash hard_rq0/eval_policy.sh
```

The raw result stores, per question and backend:

- recall after turns 1, 2, and 3
- marginal evidence gain at turns 2 and 3
- recovery and full recovery after a first-turn miss
- all generated queries and retrieved titles
- lexical query features at each turn

## 6. Smoke test one specialist

Run this before starting all six specialist jobs:

```bash
BACKEND=bm25 SEED=13 PROFILE=smoke \
BASE_MODEL=/absolute/path/to/Qwen2.5-3B-Instruct \
  bash hard_rq0/train_specialist.sh

BACKEND=bm25 SEED=13 PROFILE=smoke \
  bash hard_rq0/merge_specialist.sh

TAG=bm25-specialist \
SEED=13 \
RESULT_SET=smoke \
MODEL_PATH=$PWD/work/hard_rq0/merged/hard-rq0-bm25-seed13-smoke \
LIMIT=20 \
  bash hard_rq0/eval_policy.sh
```

Smoke checkpoints are only pipeline checks and must not be mixed with pilot/full results. `RESULT_SET` gives each profile a separate output directory and the report validator rejects mixed model signatures.

## 7. Evaluate the deterministic base policy

```bash
TAG=base-qwen \
SEED=0 \
RESULT_SET=pilot \
MODEL_PATH=/absolute/path/to/Qwen2.5-3B-Instruct \
  bash hard_rq0/eval_policy.sh
```

The base policy is evaluated on exactly the same questions and backends. These results are subtracted from every specialist result before specialization is assessed.

## 8. Train and cross-evaluate three seeds

```bash
PROFILE=pilot \
RESULT_SET=pilot \
SEEDS="13 42 87" \
BASE_MODEL=/absolute/path/to/Qwen2.5-3B-Instruct \
  bash hard_rq0/run_three_seed_specialists.sh
```

`pilot` uses 200 optimizer steps. After the pipeline is stable, use `PROFILE=full RESULT_SET=full` for 500 steps or override `TOTAL_STEPS` explicitly.

The loop performs, sequentially:

```text
BM25 seed 13 -> merge -> BM25/E5 cross-evaluation
BM25 seed 42 -> merge -> BM25/E5 cross-evaluation
BM25 seed 87 -> merge -> BM25/E5 cross-evaluation
E5 seed 13   -> merge -> BM25/E5 cross-evaluation
E5 seed 42   -> merge -> BM25/E5 cross-evaluation
E5 seed 87   -> merge -> BM25/E5 cross-evaluation
```

## 9. Generate query and interaction reports

```bash
RESULT_SET=pilot bash hard_rq0/make_report.sh
cat work/hard_rq0/runs/pilot/results/report/HARD_RQ0_REPORT.md
```

Outputs under `work/hard_rq0/runs/<result-set>/results/report/`:

```text
absolute_summary.csv
gain_over_base.csv
home_backend_excess.csv
base_backend_gap.csv
difficulty_matching.csv
matched_hard_question_ids.json
query_turns.csv
query_turn_summary.csv
query_shift_by_turn.csv
HARD_RQ0_REPORT.md
```

The query analysis uses a fixed MiniLM encoder on CPU by default. Override with `QUERY_MODEL` or `QUERY_DEVICE`.

## Matched-hard subset

For each dataset and top-k, a question enters the matched-hard subset when:

1. base Qwen turn-1 support recall is at most 0.5 on BM25;
2. base Qwen turn-1 support recall is at most 0.5 on E5; and
3. at least one evaluated policy improves support recall by turn 3.

This removes questions that E5 already solves immediately and questions for which none of the policies can recover evidence. Because recoverability uses evaluated-policy outcomes, matched-hard is an explicit diagnostic subset; the all-question results remain in every report.

## Go / no-go rule

The main gate is evaluated on the matched-hard subset. Continue only when, across three specialist seeds:

- home-backend excess gain is at least 0.05 on support recall, turn-2/3 marginal evidence gain, or recovery;
- the hierarchical bootstrap 95% interval is above zero; and
- turn-1 interaction is small while turn-2/3 interaction becomes positive.

If the interaction remains below 0.03, the hidden-retriever adaptation hypothesis should be rejected for this setup. A large gain over base on both backends instead supports a general search/reasoning or hard-retriever curriculum effect.

## End-to-end command

After the two environments are installed and the base model is available locally:

```bash
BASE_MODEL_PATH=/absolute/path/to/Qwen2.5-3B-Instruct \
PROFILE=pilot RESULT_SET=pilot \
  bash hard_rq0/run_all.sh
```

For a quick code-path check, prepare assets and data first, then use:

```bash
BASE_MODEL_PATH=/absolute/path/to/Qwen2.5-3B-Instruct \
SKIP_ASSETS=1 SKIP_DATA=1 RUN_REPORT=0 \
PROFILE=smoke RESULT_SET=smoke LIMIT=20 SEEDS="13" \
  bash hard_rq0/run_all.sh
```

The one-seed smoke command exercises training and evaluation without producing the final report. The final report intentionally requires all three specialist seeds.
