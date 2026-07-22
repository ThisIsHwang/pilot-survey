# Hard RQ0: retrieval-driven specialization under multi-turn search

The first RQ0 used a small HotpotQA context corpus, top-k 10, and usually one search turn. It found a strong general RL improvement but almost no Policy × Backend interaction. This follow-up tests the same hypothesis in an environment where query reformulation can matter.

## Protocol

- corpus: the official full Search-R1 `wiki-18` corpus
- datasets: 2WikiMultiHopQA and MuSiQue from the existing FlashRAG collection
- hard retrieval depths: top-k 3 and 5
- easy control: top-k 10
- search budget: up to four searches, with turns 1, 2, and 3 analyzed separately
- policies: deterministic base Qwen plus BM25 and E5 specialists from the same initialization
- specialist seeds: 13, 42, and 87
- primary analysis: gain over base Qwen and home-backend excess gain
- diagnostic subset: questions that are difficult for base Qwen on both retrievers but recoverable by turn 3

The central difference-in-differences interaction is:

```text
(specialist_home - base_home) - (specialist_away - base_away)
```

A specialist that improves BM25 and E5 equally shows a general RL effect, not retriever specialization.

## One-node GPU layout

During GRPO training:

- GPUs 0-6: Search-R1 actor, rollout, and reference workers
- GPU 7: E5 query encoder and flat FAISS index
- CPU: BM25 Lucene server

During evaluation:

- GPUs 0-1: vLLM policy server with tensor parallel size 2
- GPU 7: E5 retrieval
- CPU: BM25 retrieval

Training uses top-k 3. Every policy is evaluated at top-k 3, 5, and 10.

## 1. Bootstrap

```bash
bash scripts/bootstrap.sh
bash scripts/bootstrap_vllm.sh
bash scripts/preflight.sh
```

## 2. Download full-wiki assets

```bash
bash hard_rq0/download_assets.sh
```

This downloads and verifies Search-R1's official wiki-18 corpus, BM25 index, and E5 flat index. Existing Hugging Face caches are reused.

## 3. Prepare existing benchmark annotations

```bash
bash hard_rq0/prepare_data.sh
cat work/hard_rq0/data/SUMMARY.txt
```

No new human annotation is created. The script uses existing answers and supporting-document metadata. It stops with diagnostic examples if a dataset revision does not expose supporting titles in a recognized format.

## 4. Start full-wiki retrieval servers

```bash
bash hard_rq0/launch_retrievers.sh
```

```text
BM25: http://127.0.0.1:8101/health
E5:   http://127.0.0.1:8102/health
```

## 5. Smoke-test base evaluation

```bash
TAG=base-qwen \
SEED=0 \
RESULT_SET=smoke \
MODEL_PATH=/absolute/path/to/Qwen2.5-3B-Instruct \
LIMIT=20 \
  bash hard_rq0/eval_policy.sh
```

Each raw episode stores:

- support recall after turns 1, 2, and 3
- marginal evidence gain at turns 2 and 3
- recovery and full recovery after a first-turn miss
- generated queries and retrieved titles
- lexical query features at each turn

## 6. Smoke-test one specialist

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

Smoke checkpoints are pipeline checks only. `RESULT_SET` isolates smoke, pilot, and full outputs so they cannot be combined accidentally.

## 7. Evaluate the deterministic base policy

```bash
TAG=base-qwen \
SEED=0 \
RESULT_SET=pilot \
MODEL_PATH=/absolute/path/to/Qwen2.5-3B-Instruct \
  bash hard_rq0/eval_policy.sh
```

Base Qwen is evaluated on the same questions, backends, and top-k values as every specialist. Its same-backend score is subtracted before specialization is assessed.

## 8. Train and cross-evaluate three specialist seeds

```bash
PROFILE=pilot \
RESULT_SET=pilot \
SEEDS="13 42 87" \
BASE_MODEL=/absolute/path/to/Qwen2.5-3B-Instruct \
  bash hard_rq0/run_three_seed_specialists.sh
```

`pilot` uses 200 optimizer steps. After the path is stable, use `PROFILE=full RESULT_SET=full` for 500 steps or override `TOTAL_STEPS`.

The loop runs sequentially:

```text
BM25 seed 13 -> merge -> BM25/E5 evaluation at k=3/5/10
BM25 seed 42 -> merge -> BM25/E5 evaluation at k=3/5/10
BM25 seed 87 -> merge -> BM25/E5 evaluation at k=3/5/10
E5 seed 13   -> merge -> BM25/E5 evaluation at k=3/5/10
E5 seed 42   -> merge -> BM25/E5 evaluation at k=3/5/10
E5 seed 87   -> merge -> BM25/E5 evaluation at k=3/5/10
```

## 9. Generate the interaction and query reports

```bash
RESULT_SET=pilot bash hard_rq0/make_report.sh
cat work/hard_rq0/runs/pilot/results/report/HARD_RQ0_REPORT.md
cat work/hard_rq0/runs/pilot/results/report/QUERY_BEHAVIOR.md
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
QUERY_BEHAVIOR.md
```

The query analysis uses a fixed MiniLM encoder on CPU by default. Override `QUERY_MODEL` or `QUERY_DEVICE` when needed.

## Matched-hard diagnostic subset

For each dataset and top-k, a question enters this subset when:

1. base Qwen turn-1 support recall is at most 0.5 on BM25;
2. base Qwen turn-1 support recall is at most 0.5 on E5; and
3. at least one evaluated policy improves support recall by turn 3.

This removes questions that E5 already solves immediately and questions that none of the evaluated policies can recover. Because recoverability uses evaluated-policy outcomes, this is explicitly a diagnostic subset. All-question results remain primary and are always reported.

## Go / no-go rule

Continue only when, across three specialist seeds:

- matched-hard home-backend excess is at least 0.05 on support recall, turn-2/3 marginal evidence gain, or recovery;
- the hierarchical bootstrap 95% interval is above zero; and
- turn-1 interaction is small while turn-2/3 interaction becomes positive.

If the interaction remains below 0.03, reject the hidden-retriever specialization hypothesis for this setup. A large gain over base on both backends instead supports a general search/reasoning or hard-retriever curriculum effect.

## End-to-end command

```bash
BASE_MODEL_PATH=/absolute/path/to/Qwen2.5-3B-Instruct \
PROFILE=pilot RESULT_SET=pilot \
  bash hard_rq0/run_all.sh
```

Quick code-path check after assets and data are prepared:

```bash
BASE_MODEL_PATH=/absolute/path/to/Qwen2.5-3B-Instruct \
SKIP_ASSETS=1 SKIP_DATA=1 RUN_REPORT=0 \
PROFILE=smoke RESULT_SET=smoke LIMIT=20 SEEDS="13" \
  bash hard_rq0/run_all.sh
```

A one-seed smoke run intentionally skips the final report. The final report validator requires all three specialist seeds and identical evaluation units across policies.
