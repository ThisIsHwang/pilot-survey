# Stage 2: Search-R1 transfer-gap pilot

This stage tests whether a policy trained against BM25 or E5 develops a
measurable portability gap on the other backend. It targets one Linux node
with eight full H100s, a CUDA 12.9 host toolkit, and Python 3.12.

## One-command Stage-2 run

After Stage 0 has produced its data and indexes, the recommended command is:

```bash
HF_HOME=/group-volume/teo.hwang/huggingface-cache \
  bash searchr1_stage2/run_all.sh
```

The runner creates and validates all three environments when needed, prepares
the data and indexes, evaluates the base and official policies, runs BM25 and
E5 smoke training, trains both pilot specialists, cross-evaluates them, and
writes `work/results/rq0/RQ0_REPORT.md`. At run start, remote model refs are
resolved and downloaded into concrete `HF_HOME` snapshot directories; the base
evaluation and GRPO initialization therefore use the same exact weights. Use
`SKIP_BOOTSTRAP=1` only after all bootstraps have completed successfully at
least once.

For only the short training path:

```bash
POLICY_LIMIT=20 SMOKE_ONLY=1 bash searchr1_stage2/run_all.sh
```

The corresponding remote-revision controls are
`BASE_POLICY_MODEL_REVISION`, `OFFICIAL_POLICY_MODEL_REVISION`, and
`TRAIN_BASE_MODEL_REVISION`. The bundled models already default to pinned
commits; a custom remote ID defaults to `main`, which is resolved once to a
concrete snapshot. Local model directories ignore these controls.

The smoke profile uses 32 training and 32 validation examples for one update.
The pilot profile uses all 3,000/500 examples for three epochs by default. The
pinned trainer has an off-by-one stop condition; the wrapper compensates for it
and verifies an exact final `global_step_<updates>` checkpoint before recording
completion.

Default remote models can be replaced with local Hugging Face directories:

```bash
BASE_POLICY_MODEL=/models/Qwen2.5-3B-Instruct \
OFFICIAL_POLICY_MODEL=/models/SearchR1-official \
TRAIN_BASE_MODEL=/models/Qwen2.5-3B-Instruct \
  bash searchr1_stage2/run_all.sh
```

## Runtime and GPU allocation

`scripts/bootstrap_searchr1.sh` creates `.venv-searchr1` with the versions
required by the pinned Search-R1/veRL code: PyTorch 2.4.0's CUDA 12.1
compatibility wheel and vLLM 0.6.3. It is intentionally isolated from the
CUDA-12.9 `.venv-pilot` and `.venv-vllm` environments. The host driver and
`nvcc` remain CUDA 12.9, and `flash-attn` is compiled locally for H100. Run the
standalone checks with:

```bash
bash scripts/bootstrap_searchr1.sh
bash scripts/preflight_searchr1.sh
```

The bootstrap applies the tracked `searchr1-runtime.patch`, adding a bounded
retrieval request and HTTP status checking. Consequently
`upstream/Search-R1/search_r1/llm_agent/generation.py` appears intentionally
modified; set `SEARCH_R1_RETRIEVER_TIMEOUT` to override its 120-second timeout.

Policy evaluation uses vLLM on GPUs 0-3, BM25 on CPU, and E5 with GPU FAISS on
GPU 5. BM25 training gives GPUs 0-7 to Search-R1 and leaves retrieval on CPU.
E5 training also gives GPUs 0-7 to Search-R1 while sharing GPU 7 with E5/FAISS;
the rollout memory fraction is reduced to 0.50. Every phase is sequential.

## Manual sequence

Prepare data, then evaluate the two fixed policies. `MODEL_REF` accepts either
a local directory or a Hugging Face repository ID:

```bash
.venv-pilot/bin/python searchr1_stage2/make_hotpot_searchr1_data.py --work-dir work

TAG=base-qwen MODEL_REF=Qwen/Qwen2.5-3B-Instruct LIMIT=300 \
  bash searchr1_stage2/eval_policy.sh

TAG=official-searchr1 \
MODEL_REF=PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo-v0.3 \
LIMIT=300 bash searchr1_stage2/eval_policy.sh
```

Set `MODEL_REVISION=<full-commit-sha>` to pin either manual remote evaluation.
Standalone remote training uses `BASE_MODEL_REVISION` in the same way. Both
commands download the selected revision while running, then pass the concrete
snapshot directory to vLLM/Search-R1.

Train and select the specialists:

```bash
BACKEND=bm25 PROFILE=smoke bash searchr1_stage2/run_single_stack_grpo.sh
BACKEND=e5   PROFILE=smoke bash searchr1_stage2/run_single_stack_grpo.sh
BACKEND=bm25 PROFILE=pilot bash searchr1_stage2/run_single_stack_grpo.sh
BACKEND=e5   PROFILE=pilot bash searchr1_stage2/run_single_stack_grpo.sh

EXP=hotpot-bm25-pilot-grpo bash searchr1_stage2/merge_latest_checkpoint.sh
EXP=hotpot-e5-pilot-grpo   bash searchr1_stage2/merge_latest_checkpoint.sh
```

The retriever is selected, launched, probed, and stopped automatically. Set
`AUTO_LAUNCH_RETRIEVER=0` only when managing the corresponding fixed-port
service externally. WandB is off by default; opt in with `ENABLE_WANDB=1`.

The so-called merge step does not copy or convert weights. Search-R1 already
writes a complete Hugging Face model under `actor/global_step_*`; the helper
validates that model and creates a symlink under `work/merged`. If an old
nonempty directory occupies that output path, `FORCE_MODEL_LINK=1` archives it
with a timestamp before creating the link.

Cross-evaluate with the exact report tags and the same `LIMIT` for all four
policies:

```bash
TAG=bm25-specialist MODEL_REF=$PWD/work/merged/hotpot-bm25-pilot-grpo \
LIMIT=300 bash searchr1_stage2/eval_policy.sh

TAG=e5-specialist MODEL_REF=$PWD/work/merged/hotpot-e5-pilot-grpo \
LIMIT=300 bash searchr1_stage2/eval_policy.sh

bash searchr1_stage2/make_rq0_report.sh
```

The report requires BM25/E5 blind rows for exactly `base-qwen`,
`official-searchr1`, `bm25-specialist`, and `e5-specialist`, with the same
question selection. Partial or stale matrices fail instead of being averaged.

## Restart behavior and outputs

Matching completed training is reused only after `.complete.json` and its exact
actor checkpoint validate. The pinned trainer does not persist optimizer state,
so interrupted output is moved to `.incomplete.<timestamp>` and restarted from
the base model. `FORCE_TRAIN=1` similarly archives a completed run before
retraining.

Primary outputs are:

- `work/checkpoints/<experiment>/actor/global_step_*`
- `work/merged/<experiment>`
- `work/results/policies/<tag>.jsonl`
- `work/results/policies/<tag>_summary.csv`
- `work/results/rq0/query_stats_summary.csv`
- `work/results/rq0/transfer_matrix_*.csv`
- `work/results/rq0/RQ0_REPORT.md`

Interpret the RQ0 report alongside Stage 0's `work/results/REPORT.md`; the RQ0
generator does not automatically import Stage-0 RRF/oracle results.
