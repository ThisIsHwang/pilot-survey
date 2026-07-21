from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from stackpilot.common import answer_em, answer_f1, ensure_dir, load_config, read_jsonl, write_jsonl
from stackpilot.retrieval_clients import RetrievalClient

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
        parts.append(f'[{result["rank"]}] {result["title"]}: {text}')
    return "\n".join(parts)


def run_episode(client: OpenAI, model: str, retriever: RetrievalClient, item: dict, cfg: dict, oracle: bool) -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    user = f'Question: {item["question"]}'
    if oracle:
        user += "\n" + GUIDANCE[retriever.name]
    messages.append({"role": "user", "content": user})
    all_titles: list[str] = []
    searches = []
    prediction = ""

    for _ in range(int(cfg["agent"]["max_search_turns"]) + 1):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=float(cfg["llm"]["temperature"]),
            max_tokens=int(cfg["llm"]["max_tokens"]),
        )
        content = response.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": content})
        action, value = parse_action(content)
        if action == "answer":
            prediction = value
            break
        if action != "search":
            messages.append({"role": "user", "content": "Invalid format. Output <search>...</search> or <answer>...</answer>."})
            continue
        results = retriever.search(value, int(cfg["retrieval"]["topk"]))
        searches.append(value)
        all_titles.extend(r["title"] for r in results)
        observation = format_results(results, int(cfg["agent"]["result_snippet_chars"]))
        messages.append({"role": "user", "content": f"<information>\n{observation}\n</information>"})

    gold_titles = {x.lower().strip() for x in item["support_titles"]}
    got_titles = {x.lower().strip() for x in all_titles}
    support_recall = len(gold_titles & got_titles) / max(1, len(gold_titles))
    return {
        "question_id": item["id"],
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
    work_dir = Path(cfg["work_dir"]).resolve()
    out_dir = ensure_dir(work_dir / "results")
    queries = read_jsonl(work_dir / "data" / "queries_eval.jsonl")
    limit = args.limit or int(cfg["agent"]["eval_examples"])
    queries = queries[:limit]

    r_cfg = cfg["retrieval"]
    retrievers = [
        RetrievalClient("bm25", f'http://127.0.0.1:{r_cfg["bm25_port"]}/retrieve'),
        RetrievalClient("e5", f'http://127.0.0.1:{r_cfg["e5_port"]}/retrieve'),
        RetrievalClient("colbert", f'http://127.0.0.1:{r_cfg["colbert_port"]}/retrieve'),
    ]
    llm = OpenAI(base_url=cfg["llm"]["api_base"], api_key=cfg["llm"]["api_key"])
    rows = []
    for item in tqdm(queries, desc="agent eval"):
        for retriever in retrievers:
            rows.append(run_episode(llm, cfg["llm"]["model"], retriever, item, cfg, oracle=False))
            rows.append(run_episode(llm, cfg["llm"]["model"], retriever, item, cfg, oracle=True))

    write_jsonl(out_dir / "agent_eval.jsonl", rows)
    frame = pd.DataFrame(rows)
    summary = frame.groupby(["backend", "variant"])[["em", "f1", "support_recall", "search_count"]].mean().reset_index()
    summary.to_csv(out_dir / "agent_eval_summary.csv", index=False)
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
