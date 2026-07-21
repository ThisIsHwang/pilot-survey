from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from stackpilot.common import ensure_dir, load_config, read_jsonl, write_jsonl
from stackpilot.query_styles import LLMQueryGenerator, heuristic_candidates
from stackpilot.retrieval_clients import RetrievalClient


def support_recall(results: list[dict], support_titles: list[str]) -> float:
    gold = {x.strip().lower() for x in support_titles}
    if not gold:
        return 0.0
    retrieved = {str(x.get("title", "")).strip().lower() for x in results}
    return len(gold & retrieved) / len(gold)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    work_dir = Path(cfg["work_dir"]).resolve()
    out_dir = ensure_dir(work_dir / "results")
    queries = read_jsonl(work_dir / "data" / "queries_eval.jsonl")
    if args.limit:
        queries = queries[: args.limit]

    r_cfg = cfg["retrieval"]
    clients = {
        "bm25": RetrievalClient("bm25", f'http://127.0.0.1:{r_cfg["bm25_port"]}/retrieve'),
        "e5": RetrievalClient("e5", f'http://127.0.0.1:{r_cfg["e5_port"]}/retrieve'),
        "colbert": RetrievalClient("colbert", f'http://127.0.0.1:{r_cfg["colbert_port"]}/retrieve'),
    }
    source = cfg["query_generation"]["source"]
    generator = None
    if source == "vllm":
        generator = LLMQueryGenerator(
            api_base=cfg["llm"]["api_base"],
            api_key=cfg["llm"]["api_key"],
            model=cfg["llm"]["model"],
            temperature=cfg["llm"]["temperature"],
        )

    rows = []
    for item in tqdm(queries, desc="retrieval matrix"):
        candidates = generator.generate(item["question"]) if generator else heuristic_candidates(item["question"])
        for backend, client in clients.items():
            style_results = {}
            for style, query in candidates.items():
                results = client.search(query, int(r_cfg["topk"]))
                style_results[style] = results
                recall = support_recall(results, item["support_titles"])
                rows.append(
                    {
                        "question_id": item["id"],
                        "backend": backend,
                        "style": style,
                        "query": query,
                        "support_recall": recall,
                        "support_hit": float(recall > 0),
                        "full_support": float(recall == 1.0),
                        "retrieved_titles": [r["title"] for r in results],
                    }
                )

            # Strong simple baseline: run all query styles and fuse rankings with RRF.
            rrf_scores = defaultdict(float)
            doc_payload = {}
            for results in style_results.values():
                for result in results:
                    key = result["id"] or result["title"].strip().lower()
                    if not key:
                        continue
                    rrf_scores[key] += 1.0 / (60.0 + result["rank"])
                    doc_payload[key] = result
            fused = [doc_payload[key] for key, _ in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[: int(r_cfg["topk"])]]
            recall = support_recall(fused, item["support_titles"])
            rows.append(
                {
                    "question_id": item["id"],
                    "backend": backend,
                    "style": "rrf_ensemble",
                    "query": " | ".join(candidates.values()),
                    "support_recall": recall,
                    "support_hit": float(recall > 0),
                    "full_support": float(recall == 1.0),
                    "retrieved_titles": [r["title"] for r in fused],
                }
            )

    write_jsonl(out_dir / "retrieval_matrix.jsonl", rows)
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(["backend", "style"])[["support_recall", "support_hit", "full_support"]]
        .mean()
        .reset_index()
    )
    summary.to_csv(out_dir / "retrieval_matrix_summary.csv", index=False)

    oracle = frame.groupby(["question_id", "backend"])["support_recall"].max().groupby("backend").mean()
    fixed = frame.groupby(["backend", "style"])["support_recall"].mean().unstack()
    print("\nMean supporting-title recall")
    print(fixed.round(4).to_string())
    print("\nPer-question query-style oracle")
    print(oracle.round(4).to_string())
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
