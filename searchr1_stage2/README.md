# Stage 2: Search-R1 transfer-gap pilot

Run this stage only after the zero-shot pilot completes. Its purpose is not to claim a new adaptation method; it tests whether a policy trained against one retrieval backend develops a measurable portability gap on another backend.

## 1. Prepare Search-R1 data

```bash
source .venv-pilot/bin/activate
python searchr1_stage2/make_hotpot_searchr1_data.py --work-dir work
```

Create the official Search-R1 environment at the commit pinned by the root README. Run all GRPO commands inside that environment.

## 2. Evaluate the untrained base policy

Start BM25 and E5, then serve the base model and evaluate the same policy on both backends:

```bash
TAG=base-qwen \
MODEL_PATH=/path/to/Qwen2.5-3B-Instruct \
LIMIT=300 \
  bash searchr1_stage2/eval_policy.sh
```

This estimates the raw backend-quality gap before RL specialization.

## 3. Evaluate the official Search-R1 policy

```bash
TAG=official-searchr1 \
MODEL_PATH=PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo-v0.3 \
LIMIT=300 \
  bash searchr1_stage2/eval_policy.sh
```

Use a local snapshot path when the cluster cannot download from Hugging Face at runtime.

## 4. Train specialist policies

Run a smoke test first:

```bash
BACKEND=bm25 PROFILE=smoke bash searchr1_stage2/run_single_stack_grpo.sh
BACKEND=e5   PROFILE=smoke bash searchr1_stage2/run_single_stack_grpo.sh
```

Then run the pilot profiles:

```bash
BACKEND=bm25 PROFILE=pilot bash searchr1_stage2/run_single_stack_grpo.sh
BACKEND=e5   PROFILE=pilot bash searchr1_stage2/run_single_stack_grpo.sh
```

Both runs initialize from `Qwen/Qwen2.5-3B-Instruct` unless `BASE_MODEL` is overridden. Keep the corresponding retriever service alive for the complete run.

## 5. Merge and cross-evaluate specialists

```bash
EXP=hotpot-bm25-pilot-grpo bash searchr1_stage2/merge_latest_checkpoint.sh
EXP=hotpot-e5-pilot-grpo   bash searchr1_stage2/merge_latest_checkpoint.sh

TAG=bm25-specialist \
MODEL_PATH=$PWD/work/merged/hotpot-bm25-pilot-grpo \
LIMIT=300 \
  bash searchr1_stage2/eval_policy.sh

TAG=e5-specialist \
MODEL_PATH=$PWD/work/merged/hotpot-e5-pilot-grpo \
LIMIT=300 \
  bash searchr1_stage2/eval_policy.sh
```

## 6. Generate query statistics and the transfer report

```bash
bash searchr1_stage2/make_rq0_report.sh
cat work/results/rq0/RQ0_REPORT.md
```

Outputs include:

- `work/results/policies/<tag>.jsonl`
- `work/results/policies/<tag>_summary.csv`
- `work/results/rq0/transfer_matrix_*.csv`
- `work/results/rq0/query_stats_summary.csv`
- `work/results/rq0/RQ0_REPORT.md`

## Decision rule

Continue to a mixed-stack or latent-adaptation method only when all of the following are true:

1. BM25 and E5 specialists show meaningful diagonal dominance, preferably at least five percentage points in support recall or answer F1.
2. The specialist diagonal is larger than the base-policy BM25/E5 gap.
3. The policies exhibit backend-dependent query behavior rather than only a raw retriever-quality difference.
4. A fixed query ensemble from Stage 0 does not already close most of the gap.

A mixed-stack GRPO baseline should be implemented only after this RQ0 gate passes. Search-R1's stock retrieval request does not expose a stable episode identifier, so a per-request random proxy would silently switch backends within an episode and would not be a valid episode-level domain-randomization baseline.
