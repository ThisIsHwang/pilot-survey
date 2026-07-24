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
- trainer validation: 504 held-out rows per dataset (1,008 total)
- final evaluation: a different 500 held-out rows per dataset (1,000 total)
- primary analysis: gain over base Qwen and home-backend excess gain
- diagnostic subset: questions that are difficult on both retrievers at base-Qwen turn 1 but recoverable by that same fixed base policy by turn 3

The central difference-in-differences interaction is:

```text
(specialist_home - base_home) - (specialist_away - base_away)
```

A specialist that improves BM25 and E5 equally shows a general RL effect, not retriever specialization.

## Recommended one-command run

On the allocated Linux node with CUDA 12.9 and eight full H100s, run:

```bash
HF_HOME=/group-volume/teo.hwang/huggingface-cache \
  bash hard_rq0/run_all.sh
```

No model path or manual service launch is required. The default is the complete
three-seed `pilot` profile. It bootstraps all three isolated environments,
performs CUDA/FAISS/Java/Search-R1/capacity preflights, downloads pinned assets
and models, prepares data, launches both retrievers, evaluates base Qwen, trains
and cross-evaluates six specialists, builds both reports, and cleans up. A rerun
reuses verified environments, downloads, data, completed evaluations, and exact
final checkpoints.

Plan at least 270 GiB free before an uncached pilot-only hard-RQ0 run, at least
192 GiB available host RAM, and at least 22 affinity-visible physical CPU
cores. The script checks these before downloading.
Set `HF_HOME` to an existing shared cache if desired; omitting it uses
`$PWD/.cache/huggingface`.

Bootstrap reruns validate and reuse `.venv-pilot`, `.venv-vllm`, and
`.venv-searchr1`. The persistent `uv` package cache defaults to
`$PWD/.cache/uv`; missing packages are repaired in place and only the uncached
delta is downloaded. Set `UV_CACHE_DIR` to share that cache, use
`FORCE_BOOTSTRAP=1` for an intentional clean rebuild, or use `UV_OFFLINE=1` to
require every Python package to be present in the cache. Offline package mode
does not disable model or dataset downloads.

Pinned Hugging Face commit snapshots are checked in the local `HF_HOME` cache
first and downloaded only when missing or incomplete. A mutable custom branch
or tag still needs the Hub to resolve its current commit.

When this experiment is reached through `scripts/run_full_pipeline.sh`, its
model snapshots and roughly 100 GB of full-wiki assets begin downloading as
low-priority, GPU-hidden background work during Stage 0. The hard stage waits
for that resumable job only at its own boundary. Use
`PREFETCH_FUTURE_WORK=0` for sequential preparation; logs are in
`logs/prefetch/`.

## One-node GPU layout

During GRPO training:

- GPUs 0-6: Search-R1 actor, rollout, and reference workers
- GPU 7: E5 query encoder and flat FAISS index
- CPU: BM25 Lucene server

During evaluation:

- GPUs 0-6: seven vLLM policy replicas (`TP=1`, `DP=7`)
- GPU 7: E5 retrieval
- CPU: BM25 retrieval

Training uses top-k 3. Every policy is evaluated at top-k 3, 5, and 10. The
default evaluator runs 112 independent episodes concurrently to feed all seven
replicas. Turns within an episode remain sequential, seeded output is protected
with vLLM batch invariance, and E5 GPU-FAISS calls are serialized around the
single GPU resource. Override concurrency with `HARD_EVAL_WORKERS`; a custom
serving layout must provide exactly `TP * DP` IDs in `LLM_GPUS`.

The H100 profile keeps actor parameters, gradients, optimizer state, and the
reference shard on GPU by default; the 3B model fits this layout and avoids
CPU/GPU transfer stalls. `ACTOR_PARAM_OFFLOAD`, `ACTOR_GRAD_OFFLOAD`,
`ACTOR_OPTIMIZER_OFFLOAD`, and `REF_PARAM_OFFLOAD` remain explicit boolean
escape hatches. Training also fails immediately unless every Ray FSDP worker
sees exactly one assigned GPU. This prevents Python startup hooks from fixing
CUDA visibility before Ray narrows `CUDA_VISIBLE_DEVICES`, which otherwise
appears later as a ten-minute NCCL/FSDP broadcast timeout.

### Search-R1 rollout limits

Specialist training uses the retrieval-context geometry from the pinned
Search-R1 recipe: a 4096-token prompt limit, 500-token generated responses,
a 2048-token initial prompt, at most 500 retrieved-content tokens per search,
four search turns, and top-k 3 retrieval.

The upstream message
`[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, N & 500` is
expected when any retrieval result in a rollout batch exceeds that budget. The
pinned implementation concatenates ranked passages and keeps the first 500
tokens; it does not indicate a failed or stalled training step. Raising the
limit merely to silence the warning would change the paper-aligned protocol.
The separate Stage-2 transfer pilot intentionally retains its historical
top-k-10, three-turn protocol.

## Manual execution

The remaining sections are for inspecting individual phases. They are not
needed for the recommended command.

### 1. Bootstrap

```bash
bash scripts/bootstrap.sh
bash scripts/bootstrap_vllm.sh
bash scripts/bootstrap_searchr1.sh
PROFILE=pilot bash hard_rq0/preflight.sh
```

### 2. Download full-wiki assets

```bash
bash hard_rq0/download_assets.sh
```

This downloads immutable Search-R1 wiki-18 corpus, BM25-index, and E5-index
revisions. The E5 split parts total 64.6 GB and are assembled incrementally;
each source part is removed after a durable assembly checkpoint and the
compressed corpus is removed after atomic promotion.
The completion manifest tracks the corpus, BM25 index, and E5 index
independently. Subsequent runs validate and reuse each good component and
download or rebuild only a missing or invalid one; for example, a missing BM25
index does not force the 64.6 GB E5 index to be assembled again. The downloader
holds an exclusive Linux file lock, so a foreground retry safely waits for an
already-running background assembly instead of corrupting its temporary files.

### 3. Prepare existing benchmark annotations

```bash
bash hard_rq0/prepare_data.sh
cat work/hard_rq0/data/SUMMARY.txt
```

No new human annotation is created. The script uses existing answers and supporting-document metadata. It stops with diagnostic examples if a dataset revision does not expose supporting titles in a recognized format.
Trainer validation and final policy evaluation are drawn without overlap from
the pinned development split. The historical
`work/hard_rq0/searchr1/test.parquet` name now denotes trainer validation only;
`work/hard_rq0/data/eval_all.jsonl` remains the final evaluation set.

### 4. Start full-wiki retrieval servers

```bash
bash hard_rq0/launch_retrievers.sh
```

```text
BM25: http://127.0.0.1:8101/health
E5:   http://127.0.0.1:8102/health
```

### 5. Smoke-test base evaluation

```bash
TAG=base-qwen \
SEED=0 \
RESULT_SET=smoke \
LIMIT=20 \
  bash hard_rq0/eval_policy.sh
```

Each raw episode stores:

- support recall after turns 1, 2, and 3
- marginal evidence gain at turns 2 and 3
- recovery and full recovery after a first-turn miss
- generated queries and retrieved titles
- lexical query features at each turn

### 6. Smoke-test one specialist

```bash
BACKEND=bm25 SEED=13 PROFILE=smoke \
  bash hard_rq0/train_specialist.sh

BACKEND=bm25 SEED=13 PROFILE=smoke \
  bash hard_rq0/merge_specialist.sh

TAG=bm25-specialist \
SEED=13 \
RESULT_SET=smoke \
MODEL_REF=$PWD/work/hard_rq0/merged/hard-rq0-bm25-seed13-smoke \
LIMIT=20 \
  bash hard_rq0/eval_policy.sh
```

Smoke checkpoints are pipeline checks only. `RESULT_SET` isolates smoke, pilot, and full outputs so they cannot be combined accidentally.

### 7. Evaluate the deterministic base policy

```bash
TAG=base-qwen \
SEED=0 \
RESULT_SET=pilot \
  bash hard_rq0/eval_policy.sh
```

Base Qwen is evaluated on the same questions, backends, and top-k values as every specialist. Its same-backend score is subtracted before specialization is assessed.

### 8. Train and cross-evaluate three specialist seeds

```bash
PROFILE=pilot \
RESULT_SET=pilot \
SEEDS="13 42 87" \
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

### 9. Generate the interaction and query reports

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
matched_hard_units.json
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
3. the fixed base policy's best turn-3 support recall is higher than its best
   turn-1 recall across the two retrievers.

This removes questions that E5 already solves immediately and questions that the fixed base policy cannot recover. Specialist outcomes never select the subset, avoiding outcome-conditioned specialization estimates. All-question results remain primary and are always reported.

## Go / no-go rule

Continue only when, across three specialist seeds:

- matched-hard home-backend excess is at least 0.05 on support recall, turn-2/3 marginal evidence gain, or recovery;
- the crossed seed/question bootstrap 95% interval is above zero; and
- turn-1 interaction is small while turn-2/3 interaction becomes positive.

If the interaction remains below 0.03, reject the hidden-retriever specialization hypothesis for this setup. A large gain over base on both backends instead supports a general search/reasoning or hard-retriever curriculum effect.

## Overrides

```bash
BASE_MODEL_REF=/models/Qwen2.5-3B-Instruct \
  bash hard_rq0/run_all.sh
```

For a one-seed smoke/code-path check:

```bash
PROFILE=smoke RESULT_SET=smoke LIMIT=20 SEEDS="13" \
  bash hard_rq0/run_all.sh
```

A smoke run defaults to `LIMIT=20` and skips the final report. It still needs the
full-wiki assets. `SKIP_BOOTSTRAP=1`, `SKIP_ASSETS=1`, and `SKIP_DATA=1` are
available for an already verified installation; skipped assets and data are
still validated before use. Normal bootstrap reuse does not require
`SKIP_BOOTSTRAP=1`. `FORCE_TRAIN=1` archives a completed checkpoint and starts
that run again. A custom remote model should also set its full immutable
`BASE_MODEL_REVISION`; the bundled Qwen model is already pinned.
