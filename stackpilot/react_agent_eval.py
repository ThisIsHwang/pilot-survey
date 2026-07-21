from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from stackpilot.common import (
    answer_em,
    answer_f1,
    append_jsonl,
    ensure_dir,
    load_config,
    read_jsonl,
    read_jsonl_tolerant,
)
from stackpilot.retrieval_clients import RetrievalClient
from stackpilot.service_checks import (
    check_retrievers,
    check_vllm,
    effective_model_name,
)

RESULT_SCHEMA = 2

SYSTEM_PROMPT = """You are a search agent. Solve the question using the search tool.
At each turn output exactly one action:
<search>your query</search>
or, when ready,
<answer>short answer</answer>
Use returned evidence and do not include explanations inside <answer>.
"""

GUIDANCE = {
    "bm25": "The search system is lexical. Prefer concise entity and relation keywords, rare terms, and exact phrases.",
    "e5": "The search system is semantic. Prefer clear natural-language descriptions and paraphrased relations.",
    "colbert": "The search system uses token-level late interaction. Preserve exact entities and important relation terms in a fluent query.",
}


def file_digest(path: Path) -> str:
    if not path.is_file():
        raise RuntimeError(f"Required run-identity file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_identity(cfg: dict) -> dict:
    configured_model = str(cfg["llm"]["model"])
    model_path = (
        os.environ.get("MODEL_PATH") or os.environ.get("MODEL") or configured_model
    )
    identity = {
        "configured_model": configured_model,
        "model_path": model_path,
        "served_model_name": os.environ.get("SERVED_MODEL_NAME") or configured_model,
    }
    local_path = Path(model_path).expanduser()
    if local_path.is_dir():
        identity["resolved_model_path"] = str(local_path.resolve())
        model_files = {
            path
            for pattern in (
                "config.json",
                "generation_config.json",
                "tokenizer.json",
                "tokenizer.model",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "chat_template*",
                "*.index.json",
                "*.safetensors",
                "*.bin",
            )
            for path in local_path.glob(pattern)
            if path.is_file()
        }
        local_files = []
        for path in sorted(model_files):
            state = {
                "name": path.name,
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            if path.suffix not in {".safetensors", ".bin"}:
                state["sha256"] = file_digest(path)
            local_files.append(state)
        identity["local_files"] = local_files
    return identity


def run_identity(cfg: dict, work_dir: Path) -> tuple[str, dict]:
    artifacts = {
        "corpus": work_dir / "data" / "corpus.jsonl",
        "queries_eval": work_dir / "data" / "queries_eval.jsonl",
        "data_manifest": work_dir / "data" / ".pilot-manifest.json",
        "bm25_manifest": work_dir / "indexes" / "bm25" / ".pilot-manifest.json",
        "e5_manifest": work_dir / "indexes" / "e5" / ".pilot-manifest.json",
        "colbert_manifest": work_dir / "indexes" / "colbert" / ".pilot-manifest.json",
    }
    payload = {
        "schema": RESULT_SCHEMA,
        "config": cfg,
        "artifacts": {name: file_digest(path) for name, path in artifacts.items()},
        "model": model_identity(cfg),
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest(), payload


def parse_action(text: str) -> tuple[str, str]:
    search = re.search(r"<search>(.*?)</search>", text, flags=re.S | re.I)
    if search:
        return "search", search.group(1).strip()
    answer = re.search(r"<answer>(.*?)</answer>", text, flags=re.S | re.I)
    if answer:
        return "answer", answer.group(1).strip()
    return "invalid", text.strip()


def format_results(results: list[dict], max_chars: int) -> str:
    parts = []
    for result in results:
        text = result["text"].replace("\n", " ")[:max_chars]
        parts.append(f"[{result['rank']}] {result['title']}: {text}")
    return "\n".join(parts)


def completion(
    client: OpenAI,
    model: str,
    messages: list[dict],
    cfg: dict,
    request_seed: int,
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=float(cfg["llm"]["temperature"]),
        max_tokens=int(cfg["llm"]["max_tokens"]),
        seed=request_seed,
    )
    return response.choices[0].message.content or ""


def run_episode(
    client: OpenAI,
    model: str,
    retriever: RetrievalClient,
    item: dict,
    cfg: dict,
    oracle: bool,
) -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    user = f"Question: {item['question']}"
    if oracle:
        user += "\n" + GUIDANCE[retriever.name]
    messages.append({"role": "user", "content": user})
    all_titles: list[str] = []
    searches: list[str] = []
    prediction = ""
    search_budget = int(cfg["agent"]["max_search_turns"])
    max_attempts = search_budget * 2 + 2

    attempts = 0
    while len(searches) < search_budget and attempts < max_attempts:
        attempts += 1
        seed_payload = (
            f"{cfg['seed']}:{item['id']}:{retriever.name}:{int(oracle)}:{attempts}"
        )
        request_seed = int.from_bytes(
            hashlib.sha256(seed_payload.encode("utf-8")).digest()[:4], "big"
        ) % (2**31)
        content = completion(client, model, messages, cfg, request_seed)
        messages.append({"role": "assistant", "content": content})
        action, value = parse_action(content)
        if action == "answer":
            prediction = value
            break
        if action != "search" or not value:
            messages.append(
                {
                    "role": "user",
                    "content": "Invalid format. Output <search>...</search> or <answer>...</answer>.",
                }
            )
            continue
        results = retriever.search(value, int(cfg["retrieval"]["topk"]))
        searches.append(value)
        all_titles.extend(result["title"] for result in results)
        observation = format_results(results, int(cfg["agent"]["result_snippet_chars"]))
        messages.append(
            {"role": "user", "content": f"<information>\n{observation}\n</information>"}
        )

    if not prediction:
        messages.append(
            {
                "role": "user",
                "content": "The search budget is exhausted. Give your best final answer now as <answer>short answer</answer>.",
            }
        )
        seed_payload = (
            f"{cfg['seed']}:{item['id']}:{retriever.name}:{int(oracle)}:final"
        )
        request_seed = int.from_bytes(
            hashlib.sha256(seed_payload.encode("utf-8")).digest()[:4], "big"
        ) % (2**31)
        content = completion(client, model, messages, cfg, request_seed)
        action, value = parse_action(content)
        prediction = value if action == "answer" else content.strip()

    gold_titles = {value.lower().strip() for value in item["support_titles"]}
    got_titles = {value.lower().strip() for value in all_titles}
    support_recall = len(gold_titles & got_titles) / max(1, len(gold_titles))
    return {
        "question_id": str(item["id"]),
        "backend": retriever.name,
        "variant": "oracle_guidance" if oracle else "blind",
        "prediction": prediction,
        "gold": item["answer"],
        "em": answer_em(prediction, item["answer"]),
        "f1": answer_f1(prediction, item["answer"]),
        "support_recall": support_recall,
        "search_count": len(searches),
        "queries": searches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    check_retrievers(cfg)
    check_vllm(cfg)

    work_dir = Path(cfg["work_dir"]).resolve()
    out_dir = ensure_dir(work_dir / "results")
    output_path = out_dir / "agent_eval.jsonl"
    run_signature, _ = run_identity(cfg, work_dir)
    queries = read_jsonl(work_dir / "data" / "queries_eval.jsonl")
    limit = int(cfg["agent"]["eval_examples"]) if args.limit is None else args.limit
    queries = queries[:limit]

    r_cfg = cfg["retrieval"]
    retrievers = [
        RetrievalClient("bm25", f"http://127.0.0.1:{r_cfg['bm25_port']}/retrieve"),
        RetrievalClient("e5", f"http://127.0.0.1:{r_cfg['e5_port']}/retrieve"),
        RetrievalClient(
            "colbert", f"http://127.0.0.1:{r_cfg['colbert_port']}/retrieve"
        ),
    ]
    llm = OpenAI(
        base_url=cfg["llm"]["api_base"],
        api_key=cfg["llm"]["api_key"],
        timeout=180.0,
        max_retries=5,
    )
    effective_model = effective_model_name(cfg)
    selected_ids = {str(item["id"]) for item in queries}
    existing = read_jsonl_tolerant(output_path)
    row_by_key = {
        (
            str(row.get("question_id")),
            str(row.get("backend")),
            str(row.get("variant")),
        ): row
        for row in existing
        if row.get("schema") == RESULT_SCHEMA
        and row.get("run_signature") == run_signature
        and str(row.get("question_id")) in selected_ids
    }

    total = len(queries) * len(retrievers) * 2
    progress = tqdm(total=total, desc="agent eval")
    for item in queries:
        for retriever in retrievers:
            for oracle in (False, True):
                variant = "oracle_guidance" if oracle else "blind"
                key = (str(item["id"]), retriever.name, variant)
                if key not in row_by_key:
                    row = run_episode(
                        llm, effective_model, retriever, item, cfg, oracle
                    )
                    row["schema"] = RESULT_SCHEMA
                    row["run_signature"] = run_signature
                    append_jsonl(output_path, [row])
                    row_by_key[key] = row
                progress.update(1)
    progress.close()

    rows = list(row_by_key.values())
    if not rows:
        raise RuntimeError("No agent-evaluation rows were produced")
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(["backend", "variant"])[
            ["em", "f1", "support_recall", "search_count"]
        ]
        .mean()
        .reset_index()
    )
    summary.insert(0, "run_signature", run_signature)
    summary.insert(1, "n_questions", len(queries))
    summary.to_csv(out_dir / "agent_eval_summary.csv", index=False)
    print(summary.round(4).to_string(index=False))
    print(f"Resumable episode results: {output_path}")


if __name__ == "__main__":
    main()
