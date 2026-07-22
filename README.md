# Retrieval-stack adaptation pilot

This repository runs the zero-shot retrieval-stack pilot on one Linux node with
8 full NVIDIA H100 GPUs and CUDA 12.9. It compares BM25, E5, and ColBERT on a
controlled HotpotQA corpus, then evaluates the same Qwen 2.5 search agent against
all three backends.

Search-R1 is pinned to commit
`598e61bd1d36895726d28a8d06b3a15bed19f5d3`. BM25 and E5 reuse its official
indexing/retrieval implementation; ColBERT uses pinned RAGatouille/ColBERT
packages.

## Required node

- Linux x86_64 with glibc 2.31 or newer
- exactly 8 full (non-MIG) H100 GPUs, with at least 60 GiB free on each
- Python 3.12 available as `python3.12`, including its development headers
  (`Python.h`; commonly provided by `python3.12-dev`)
- an NVIDIA driver capable of running CUDA 12.9 binaries
- the CUDA 12.9 toolkit, including `nvcc`, on `PATH`
- `g++`, `git`, and `curl`
- at least 8 GiB free in `/dev/shm`; keep at least 60 GiB free when Stage 2
  starts, and plan roughly 100 GiB for a completely uncached full bootstrap,
  model download, and four specialist checkpoints

On a module-based cluster, load its CUDA 12.9 toolkit before running the pilot,
for example `module load cuda/12.9`. Confirm that `nvcc --version` reports
release 12.9. A driver alone is insufficient because the preflight and ColBERT
warm-up compile CUDA extensions.

## Recommended one-command run

The full runner executes the zero-shot Stage 0 and the newly merged Search-R1
Stage 2 in sequence. By default it downloads the 7B Stage-0 model, the 3B base
model, and the official Search-R1 policy from Hugging Face while the relevant
step is starting.

```bash
cd /group-volume/teo.hwang/pilot-survey
git pull --ff-only origin main

HF_HOME=/group-volume/teo.hwang/huggingface-cache \
  bash scripts/run_full_pipeline.sh
```

`HF_HOME` is optional; without it the runner uses
`$PWD/.cache/huggingface`. Before a service or trainer starts, each remote
branch/tag is downloaded to its concrete Hugging Face snapshot directory. The
download progress is therefore visible and is not hidden behind vLLM's old
900-second readiness limit. Subsequent runs reuse the Hugging Face, dataset,
index, policy-evaluation, and completed-training caches. A direct remote
`launch_vllm_bg.sh` call still has a four-hour first-load fallback.

The three entry points are deliberately explicit:

- `scripts/run_all.sh`: Stage 0 only
- `searchr1_stage2/run_all.sh`: Stage 2 only
- `scripts/run_full_pipeline.sh`: Stage 0 followed by Stage 2

The full runner performs all three environment bootstraps, strict preflights,
data preparation, all indexes, server warm-ups, Stage-0 evaluations, two
Search-R1 specialist trainings and cross-evaluations, both reports, and final
cleanup. It always proceeds to Stage 2; inspect the Stage-0 report separately
if a manual go/no-go decision is required first.

To use existing local model snapshots, point each role at its concrete Hugging
Face model directory containing `config.json`, tokenizer files, and weight
files:

```bash
STAGE0_MODEL_REF=/models/Qwen2.5-7B-Instruct \
BASE_POLICY_MODEL=/models/Qwen2.5-3B-Instruct \
OFFICIAL_POLICY_MODEL=/models/SearchR1-official \
TRAIN_BASE_MODEL=/models/Qwen2.5-3B-Instruct \
  bash scripts/run_full_pipeline.sh
```

The bundled Qwen/Search-R1, E5, and ColBERT models are pinned to concrete Hub
commits. A custom remote model defaults to `main`, which is resolved once to an
immutable commit at run start. To select another exact policy-model revision, set
`STAGE0_MODEL_REVISION`, `BASE_POLICY_MODEL_REVISION`,
`OFFICIAL_POLICY_MODEL_REVISION`, and `TRAIN_BASE_MODEL_REVISION` to full Hub
commit hashes. Local directories ignore these revision controls.
Retriever overrides use `E5_MODEL`/`E5_MODEL_REVISION` and
`COLBERT_MODEL`/`COLBERT_MODEL_REVISION` in the same way.

For a quick end-to-end smoke/code-path run:

```bash
RETRIEVAL_LIMIT=20 AGENT_LIMIT=5 POLICY_LIMIT=20 SMOKE_ONLY=1 \
  bash scripts/run_full_pipeline.sh
```

This still prepares the configured 3,000/500 examples and builds the full pilot
indexes when they are not cached. It shortens all evaluation limits, runs only
the two one-update smoke trainings, and intentionally skips pilot training,
specialist cross-evaluation, and the Stage-2 report. It is not a complete
experiment.

After all three bootstraps have succeeded once, avoid recreating the
environments on a rerun:

```bash
SKIP_BOOTSTRAP=1 \
  bash scripts/run_full_pipeline.sh
```

Useful Stage-2 controls are `RUN_STAGE0=0`, `RUN_SMOKE=0`, `SMOKE_ONLY=1`,
`POLICY_LIMIT=<n>`, and `FORCE_TRAIN=1`. The last option archives a completed
checkpoint directory and trains again; it does not delete it.

## Manual Stage-0 run

Use this sequence when observing each phase separately:

```bash
export MODEL_PATH=/absolute/path/to/Qwen2.5-7B-Instruct

bash scripts/bootstrap.sh
bash scripts/bootstrap_vllm.sh

bash scripts/preflight.sh

bash scripts/prepare_data.sh --config configs/pilot.yaml
bash scripts/build_indexes.sh
bash scripts/launch_retrievers.sh

bash scripts/launch_vllm_bg.sh

bash scripts/run_retrieval_matrix.sh --limit 500
bash scripts/run_agent_eval.sh --limit 200

cat work/results/REPORT.md
bash scripts/stop_servers.sh
```

The foreground vLLM alternative is:

```bash
MODEL_PATH=/absolute/path/to/Qwen2.5-7B-Instruct \
  bash scripts/launch_vllm.sh
```

## CUDA and dependency choices

The pilot environment uses PyTorch 2.11.0 with its CUDA 12.9 wheel and
GPU-enabled FAISS. The vLLM environment is separate and pins vLLM 0.19.0. That
release's normal PyPI wheel is built for CUDA 12.9; newer vLLM releases switched
their default PyPI binary starting with 0.20 to CUDA 13 and place their alternate
CUDA 12.9 wheel on GitHub release storage, which this cluster's egress proxy
returns as HTTP 403. The pinned version avoids that URL entirely.

All three virtual environments are created by `uv`; none of the scripts calls
`ensurepip`. If `uv` is absent, its pinned Linux wheel is fetched directly from
PyPI into `.bootstrap-tools`. Package installation uses copy mode so a cache on
a different filesystem from the group volume does not emit hard-link failures.

Stage 2 uses a third, isolated `.venv-searchr1`. The pinned Search-R1/veRL code
requires vLLM 0.6.3 and PyTorch 2.4.0's CUDA 12.1 compatibility wheel. This is
expected: the host driver and `nvcc`, `.venv-pilot`, and `.venv-vllm` remain
CUDA 12.9. `flash-attn` is built against the host CUDA 12.9 toolkit after
PyTorch is installed, so the first Stage-2 bootstrap can take a while.

Python/RAGatouille/LangChain/ColBERT versions are pinned. Java 21 comes from the
`jdk4py` PyPI wheel, so no Temurin or Corretto archive download is needed.

## GPU layout

Index construction is sequential:

- GPU 0: E5 encoding and GPU FAISS construction
- GPU 4: ColBERT indexing, GPU FAISS clustering, and warm-up search

Evaluation uses:

- GPUs 0-3: Qwen 2.5 vLLM, tensor parallel size 4
- GPU 5: E5 encoder and GPU FAISS search
- GPU 6: ColBERT search
- GPUs 4 and 7: free
- CPU: BM25/Lucene

Stage-2 policy evaluation starts only BM25 and E5: vLLM uses GPUs 0-3, E5 and
GPU FAISS use GPU 5, and BM25 remains on CPU. BM25-specialist training uses all
eight GPUs for Search-R1 with the CPU BM25 service. E5-specialist training also
uses all eight GPUs and intentionally shares GPU 7 with the comparatively small
E5/FAISS service; its rollout memory utilization is reduced to 0.50. Override
that service GPU with `STAGE2_E5_GPU` only when the replacement remains visible
and has enough free memory. These phases run sequentially.

The main overrides are `DENSE_GPUS`, `COLBERT_GPU`, `E5_GPU`, `LLM_GPUS`, and
`TP`. `LLM_GPUS` must contain exactly `TP` unique GPU IDs.

## Cache and restart behavior

- Hugging Face datasets/models retain their normal cache behavior. `HF_HOME`
  may point at an existing shared cache before running the scripts. Remote
  model refs are converted to concrete `snapshots/<commit>` directories before
  evaluation or training, so a moved `main` cannot reuse results from different
  weights.
- Prepared data has a configuration-and-SHA-256 manifest and is reused only
  when its counts, settings, and output files match. Existing valid pilot data
  is upgraded to the stronger manifest without downloading it again. Use
  `FORCE_PREPARE=1` to regenerate it.
- BM25, E5, and ColBERT each store a corpus/model manifest. Reuse requires a
  matching SHA-256 corpus fingerprint and an actual index document-count check;
  ColBERT also verifies every codec, IVF, and chunk artifact.
- ColBERT runs one real search before it is marked complete, so compiler/runtime
  failures cannot leave a false-success cache.
- Retrieval-matrix and agent-evaluation JSONL files are append-only checkpoints.
  A rerun resumes completed backend/episode units instead of losing hours of
  work after a transient request failure. Interrupted tail writes are repaired
  before appending, and reports refuse to combine different run signatures.
- Stage-2 policy JSONL files use the same signature-based resume behavior.
  Matching GRPO runs are reused only when `.complete.json` and the final actor
  checkpoint both validate. The pinned trainer does not save optimizer state,
  so an interrupted run is moved to `.incomplete.<timestamp>` and restarted
  from the base model instead of pretending to resume.

## Logs and cleanup

Runtime logs are written to:

```text
logs/bm25.log
logs/e5.log
logs/colbert.log
logs/vllm.log
logs/hotpot-bm25-smoke-grpo.log
logs/hotpot-e5-smoke-grpo.log
logs/hotpot-bm25-pilot-grpo.log
logs/hotpot-e5-pilot-grpo.log
```

Stop only the processes recorded by this checkout:

```bash
bash scripts/stop_servers.sh
.venv-searchr1/bin/ray stop --force
```

PID command lines are verified before signals are sent, so a stale PID file
cannot kill an unrelated process. Launchers also refuse an unknown process
already occupying ports 8001, 8002, 8003, or 9000.

## Outputs

- `work/results/retrieval_matrix_summary.csv`
- `work/results/retrieval_style_oracle.csv`
- `work/results/agent_eval_summary.csv`
- `work/results/REPORT.md`
- `work/checkpoints/hotpot-{bm25,e5}-pilot-grpo/actor/global_step_*`
- `work/merged/hotpot-{bm25,e5}-pilot-grpo` (validated model symlinks)
- `work/results/policies/*.jsonl` and `*_summary.csv`
- `work/results/rq0/RQ0_REPORT.md`

The query-style oracle excludes the RRF ensemble; the ensemble remains a
separate fixed baseline.

## What the pilot measures

1. Supporting-title recall for semantic, keyword, exact-phrase, decomposed, and
   RRF-fused queries across BM25, E5, and ColBERT.
2. Blind ReAct-style search-agent performance on each hidden backend.
3. The gain from backend-specific query guidance as an adaptation upper bound.

Proceed to the separate Stage-2 Search-R1 specialist work only if at least two
of these hold:

- blind answer/support performance differs by at least 5 percentage points
  across backends;
- backend guidance improves the weakest backend by at least 3 points;
- the best fixed query style differs by backend and the per-question style
  oracle is materially stronger;
- the fixed RRF ensemble does not already close most of the gap.

Stage 2 remains outside the Stage-0-only `scripts/run_all.sh` because it needs a
separate runtime and GPU allocation. Use `scripts/run_full_pipeline.sh` to run
both stages, or `searchr1_stage2/run_all.sh` to run only the new stage.
