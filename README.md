# Retrieval-Stack Adaptation Pilot

This is a **go/no-go experiment** for the paper idea: a search agent trained or prompted on one retrieval stack may fail when attached to another stack. It deliberately reuses the official [`PeterGriffinJin/Search-R1`](https://github.com/PeterGriffinJin/Search-R1) implementation for BM25/E5 indexing and retrieval, and uses RAGatouille for ColBERT.

It does **not** yet implement the proposed latent-belief robust RL. First establish that the problem is real and not solved by a fixed query ensemble.

## What it measures

1. **Retrieval-only query-style matrix**
   - Backends: BM25, E5, ColBERT
   - Styles: semantic, keyword, exact phrase, decomposed first-hop
   - Metric: HotpotQA supporting-title recall
   - Output: best fixed style and per-question style oracle
2. **Blind search-agent evaluation**
   - The same ReAct-style Qwen agent is connected to each hidden backend.
3. **Backend-guideline oracle**
   - The model is told the backend-specific query guideline.
   - The gap estimates whether online stack adaptation has useful headroom.
4. **Stage-2 Search-R1 specialists**
   - Scripts are included to train BM25/E5/ColBERT specialist baselines after the pilot passes.

## Upstream reuse

- Search-R1 is pinned to commit `598e61bd1d36895726d28a8d06b3a15bed19f5d3`.
- Search-R1 already supports BM25 and dense local retrievers behind a FastAPI endpoint and GRPO on one 8-GPU node.
- This package adds a port-configurable wrapper, ColBERT-compatible endpoint, controlled HotpotQA corpus, and cross-stack evaluation.

## Hardware layout for 8×H100

During index construction:

- GPUs 0–3: E5 encoding
- GPU 4: ColBERT indexing

During evaluation:

- GPU 5: E5 query encoder and GPU FAISS search
- GPU 6: ColBERT server
- GPUs 0–3: vLLM Qwen-7B, tensor parallel 4
- GPU 7: free / monitoring / larger vLLM TP if desired

Change GPU assignments through environment variables in the scripts. BM25 indexing requires OpenJDK 21 for Pyserini.
If Java 21 is not already available, the scripts install the `jdk4py` Java 21
runtime wheel from PyPI into `.venv-pilot` and export `JAVA_HOME` and `JVM_PATH`
automatically. This avoids direct JDK archive downloads that cluster egress
proxies may block.

## Run

```bash
unzip stack_adapt_pilot.zip
cd stack_adapt_pilot

bash scripts/bootstrap.sh
bash scripts/bootstrap_vllm.sh
source .venv-pilot/bin/activate

bash scripts/prepare_data.sh
bash scripts/build_indexes.sh
bash scripts/launch_retrievers.sh

# Fastest experiment; no LLM server required.
bash scripts/run_retrieval_matrix.sh --limit 500

# In a second terminal, start the LLM server.
# This script activates the separate .venv-vllm environment.
bash scripts/launch_vllm.sh
# Or background mode: bash scripts/launch_vllm_bg.sh

# Then evaluate the blind agent and oracle-guidance upper bound.
bash scripts/run_agent_eval.sh --limit 200

cat work/results/REPORT.md
```

Both bootstrap scripts target CUDA 12.9 (`cu129`) and recreate their virtual
environment on each run so an incomplete install cannot leak into the next one.
They use the `python` command by default; select a specific interpreter with, for
example, `PYTHON_BIN=python3.12 bash scripts/bootstrap_vllm.sh`.
Virtual environments are created by `uv`, without relying on the system
`venv`/`ensurepip` packages. If `uv` is absent, a standalone binary is installed
under `.bootstrap-tools/` automatically.

The pilot environment installs GPU-enabled FAISS. E5 index construction uses
all GPUs listed in `DENSE_GPUS` and writes the resulting portable CPU index to
disk; the E5 server moves that index onto the GPU selected by `E5_GPU` at startup.

Stop retrieval servers:

```bash
bash scripts/stop_servers.sh
```

## Faster smoke test

Edit `configs/pilot.yaml`:

```yaml
data:
  train_examples: 500
  eval_examples: 100
agent:
  eval_examples: 30
```

Then run the same commands.

## Full query generation with the LLM

The default query-style matrix uses deterministic heuristics so it can run before vLLM. After the server is up, set:

```yaml
query_generation:
  source: vllm
```

and rerun `scripts/run_retrieval_matrix.sh`.

## Go / no-go

Proceed to the RL method only when at least two conditions hold:

- Blind answer/support performance differs by at least 5 percentage points across backends.
- Backend-guideline oracle improves the weakest backend by at least 3 points.
- The best query style differs by backend and per-question style oracle clearly beats the best fixed style.
- A fixed query ensemble does not already close most of the gap.

If these fail, the latent stack-belief contribution is unlikely to justify a top-conference paper.

## Important limitations of this pilot

- It uses a global corpus aggregated from HotpotQA distractor contexts, not the full Wikipedia corpus. This is intentional for a fast first run.
- The backend-guideline oracle is a cheap upper bound, not a trained specialist.
- RAGatouille dependency compatibility can vary. If ColBERT installation fails, run BM25/E5 first and address ColBERT separately.
- The final paper must repeat the study on a fixed full-corpus benchmark such as BrowseComp-Plus or BEIR HotpotQA.

## Smoke configuration

A tiny configuration is included at `configs/smoke.yaml`. All Python commands accept `--config configs/smoke.yaml`. The launch scripts use the main config paths, so for the first full end-to-end run prefer `configs/pilot.yaml`; the smoke config is most useful for data preparation and code-level validation.
