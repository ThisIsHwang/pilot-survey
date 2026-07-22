from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from stackpilot.common import ensure_dir, read_jsonl_tolerant


def query_features(query: str) -> dict[str, float]:
    tokens = re.findall(r"\b\w+\b", query)
    quoted = re.findall(r'"[^"]+"', query)
    return {
        "query_chars": float(len(query)),
        "query_tokens": float(len(tokens)),
        "quoted_spans": float(len(quoted)),
        "uppercase_tokens": float(sum(token.isupper() and len(token) > 1 for token in tokens)),
        "digit_tokens": float(sum(any(ch.isdigit() for ch in token) for token in tokens)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="work/results/policies")
    parser.add_argument("--output-dir", default="work/results/rq0")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = ensure_dir(Path(args.output_dir))
    rows: list[dict] = []

    for path in sorted(results_dir.glob("*.jsonl")):
        policy_tag = path.stem
        for episode in read_jsonl_tolerant(path):
            queries = episode.get("queries") or []
            for turn, query in enumerate(queries, start=1):
                row = {
                    "policy_tag": episode.get("policy_tag", policy_tag),
                    "backend": episode.get("backend"),
                    "variant": episode.get("variant"),
                    "question_id": episode.get("question_id"),
                    "turn": turn,
                    "query": query,
                }
                row.update(query_features(str(query)))
                rows.append(row)

    if not rows:
        raise RuntimeError(f"No policy JSONL results found under {results_dir}")

    frame = pd.DataFrame(rows)
    detail_path = output_dir / "query_stats_detail.csv"
    summary_path = output_dir / "query_stats_summary.csv"
    frame.to_csv(detail_path, index=False)

    summary = (
        frame.groupby(["policy_tag", "backend", "variant"])[
            [
                "query_chars",
                "query_tokens",
                "quoted_spans",
                "uppercase_tokens",
                "digit_tokens",
            ]
        ]
        .mean()
        .reset_index()
    )
    counts = (
        frame.groupby(["policy_tag", "backend", "variant"])
        .size()
        .rename("n_queries")
        .reset_index()
    )
    summary = counts.merge(summary, on=["policy_tag", "backend", "variant"])
    summary.to_csv(summary_path, index=False)
    print(summary.round(4).to_string(index=False))
    print(json.dumps({"detail": str(detail_path), "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
