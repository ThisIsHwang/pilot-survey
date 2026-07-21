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
- at least 1 GiB free in `/dev/shm`

On a module-based cluster, load its CUDA 12.9 toolkit before running the pilot,
for example `module load cuda/12.9`. Confirm that `nvcc --version` reports
release 12.9. A driver alone is insufficient because the preflight and ColBERT
warm-up compile CUDA extensions.

## Recommended one-command run

Point `MODEL_PATH` directly at the Qwen directory that already contains
`config.json` and the `*.safetensors` files. For a Hugging Face cache, this is
the concrete `snapshots/<revision>` directory, not the parent
`models--Qwen--...` directory.

```bash
cd /group-volume/teo.hwang/pilot-survey
git pull origin main

MODEL_PATH=/absolute/path/to/Qwen2.5-7B-Instruct \
  bash scripts/run_all.sh
```

This is the single recommended model-cache method. The local filesystem path is
used to load weights, while the API name remains
`Qwen/Qwen2.5-7B-Instruct`, matching `configs/pilot.yaml`. Therefore a local
path cannot accidentally cause an OpenAI API `model not found` error.

`run_all.sh` performs both bootstraps, strict hardware/runtime preflight, data
preparation, all three indexes, real server warm-ups, the retrieval matrix, the
agent evaluation, report generation, and server cleanup. It stops immediately
with the relevant log tail when a service dies. Set `KEEP_SERVERS=1` only when
the servers should remain running after completion.

For a quick evaluation code-path run:

```bash
MODEL_PATH=/absolute/path/to/Qwen2.5-7B-Instruct \
RETRIEVAL_LIMIT=20 AGENT_LIMIT=5 \
  bash scripts/run_all.sh
```

This still prepares the configured 3,000/500 examples and builds the full pilot
indexes when they are not cached; only the retrieval and agent-evaluation
limits are shortened.

After a successful bootstrap, avoid recreating the environments on a rerun:

```bash
MODEL_PATH=/absolute/path/to/Qwen2.5-7B-Instruct \
SKIP_BOOTSTRAP=1 \
  bash scripts/run_all.sh
```

## Manual run

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

Both virtual environments are created by `uv`; neither script calls
`ensurepip`. If `uv` is absent, its pinned Linux wheel is fetched directly from
PyPI into `.bootstrap-tools`. Package installation uses copy mode so a cache on
a different filesystem from the group volume does not emit hard-link failures.

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

The main overrides are `DENSE_GPUS`, `COLBERT_GPU`, `E5_GPU`, `LLM_GPUS`, and
`TP`. `LLM_GPUS` must contain exactly `TP` unique GPU IDs.

## Cache and restart behavior

- Hugging Face datasets/models retain their normal cache behavior. `HF_HOME`
  may point at an existing shared cache before running the scripts.
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

## Logs and cleanup

Runtime logs are written to:

```text
logs/bm25.log
logs/e5.log
logs/colbert.log
logs/vllm.log
```

Stop only the processes recorded by this checkout:

```bash
bash scripts/stop_servers.sh
```

PID command lines are verified before signals are sent, so a stale PID file
cannot kill an unrelated process. Launchers also refuse an unknown process
already occupying ports 8001, 8002, 8003, or 9000.

## Outputs

- `work/results/retrieval_matrix_summary.csv`
- `work/results/retrieval_style_oracle.csv`
- `work/results/agent_eval_summary.csv`
- `work/results/REPORT.md`

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

Stage 2 is deliberately not part of `run_all.sh`: it requires Search-R1's
separate training environment and a different GPU allocation from the zero-shot
pilot.
