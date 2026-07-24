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
- trainer development: 500 rows per dataset held out from the source training
  split after the 5,000 training rows (1,000 total)
- final evaluation: 500 rows per dataset from the official pinned development
  split, reserved exclusively for reporting (1,000 total)
- primary analysis: all-question, top-k 3 observed support-title recall,
  summarized as equal-weight home-backend excess across specialist×dataset
  strata within each training seed
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

Training and Hard-RQ0 evaluation share one token-budget renderer: a 4096-token
prompt limit, 500-token generated responses, a 2048-token initial prompt, and
exactly 500 tokenizer tokens for the complete ranked retrieval bundle on each
search turn. The budget is total, not per document, and is fixed for top-k 3,
5, and 10. Increasing top-k therefore changes retrieval depth without silently
increasing the amount of context the policy can read. Later-ranked document
titles can be retrieved but truncated before reaching the model.

Every episode records both `retrieved_support_title_recall` over all returned
titles and `observed_support_title_recall` over titles whose complete headers
fit in the 500-token prompt. The historical `support_recall` and turn-specific
aliases now mean observed support-title recall. The observed metric is the
primary evidence endpoint; the retrieved metric diagnoses ranking independently
of context truncation. Stage-2 uses the same bounded renderer while retaining
its historical top-k-10, three-turn protocol.

### Action and reward protocol

Training and evaluation use the same strict action parser. A model turn may
contain optional complete `<think>...</think>` blocks and exactly one nonempty
`<search>...</search>` or `<answer>...</answer>` action; additional text,
multiple or nested actions, and incomplete tags are invalid. Search-R1
preserves the complete generated turn before parsing, so an output is not made
valid by truncating it at the first closing action tag.

If the forced final evaluation turn is malformed, `prediction` is empty,
`protocol_failure=1`, and primary EM/F1 are zero. The former permissive
raw-text score is retained only in `raw_text_prediction`, `raw_text_em`, and
`raw_text_f1` as a robustness diagnostic. `invalid_action_count` separately
records recoverable malformed turns.

Every rollout exports its protocol-valid terminal answer as structured
per-example metadata. The trainer scores that answer directly, so malformed
trajectories cannot earn EM by leaving a plausible `<answer>` somewhere in the
prompt-plus-response string. Training EM uses the same normalization as final
evaluation. `train_specialist.sh` defaults to
`SEARCH_R1_REWARD_MODE=answer`; only EXP-005 sets the mode to `evidence`, after
first restoring the canonical answer-only trainer. The mode, all three reward
weights, and both reward patch hashes are bound into the checkpoint signature,
so an evidence-patched shared Search-R1 tree cannot contaminate a later
answer-only run. EXP-005 also consumes the structured executed-search count
and titles actually visible after observation truncation. Prompt examples,
truncated documents, and model-authored fake control blocks therefore cannot
add a search or evidence hit. A malformed
terminal action receives no answer/evidence bonus (while costs from searches
that actually ran still apply), and an empty response fails immediately instead
of writing reward at index `-1`. If the accumulated trajectory was truncated
before PPO scoring, both answer-only and evidence modes assign exactly zero
reward because the credited action sequence is no longer fully represented.

These rules are included in training and evaluation signatures. Existing
checkpoints or result rows made under the older parser, reward, or data
protocol are not reused; resumable runners archive incompatible completion
state and recompute it.

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

No new human annotation is created. The script uses existing answers and
supporting-document metadata. It stops with diagnostic examples if a dataset
revision does not expose supporting titles in a recognized format. Requested
splits and counts are strict: there is no automatic split fallback or silent
shortening, including smoke runs. IDs and NFKC/lowercase/whitespace-normalized
questions must be unique within and disjoint across trainer train, trainer dev,
and final evaluation.
Training and trainer development rows are disjoint deterministic slices of the
pinned source training split. The official pinned development split is never
passed to the trainer and is reserved for final reporting. The canonical files
are `work/hard_rq0/searchr1/train.parquet`,
`work/hard_rq0/searchr1/dev.parquet`, and
`work/hard_rq0/data/final_eval.jsonl`. Their manifest records each artifact's
role, requested and actual source split/count, ID and normalized-question
hashes, and file SHA-256. Training and evaluation
entry points fail closed if an input has the wrong manifested role; changing a
manifest also invalidates the matching checkpoint signature.

Gold support titles are normalized exactly as in evaluation and checked
against the pinned wiki-18 corpus in one streaming pass. Zero- and
partial-coverage questions are retained under the fixed
`retain-and-report-all-gold` policy; missing titles remain in the denominator.
Per-row coverage and role/dataset coverage distributions are committed to the
data manifest so corpus limitations are visible rather than selection-filtered.

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

- retrieved and actually observed support-title recall
- observed support-title recall after turns 1, 2, and 3
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

## Primary decision and secondary analyses

There is one pre-registered primary endpoint: the all-question, top-k 3
observed support-title-recall home-backend excess. Question cells are averaged
within each specialist×dataset stratum, strata are equally averaged within
each training seed, and uncertainty is a Student-t 95% interval over those seed
means. Every seed value is printed.

Three seeds are exploratory and cannot produce a confirmatory GO; at least
eight predeclared training seeds are required for that label. Other metrics,
top-k values, datasets, specialists, and the matched-hard subset are secondary.
They use exact one-sided seed sign-flip p-values with a single Holm correction
family. Crossed seed/question bootstrap intervals remain descriptive only.

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
