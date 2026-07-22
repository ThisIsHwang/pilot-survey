from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from stackpilot.common import ensure_dir, read_jsonl_tolerant

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.linalg.norm(left, axis=1, keepdims=True)
    right_norm = np.linalg.norm(right, axis=1, keepdims=True)
    return (left * right).sum(axis=1) / np.maximum(1e-12, left_norm[:, 0] * right_norm[:, 0])


def load_turns(results_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(results_dir.glob("*.jsonl")):
        for episode in read_jsonl_tolerant(path):
            question = str(episode.get("question", ""))
            previous_query = None
            for turn in episode.get("turns", []):
                query = str(turn.get("query", ""))
                tokens = TOKEN_RE.findall(query)
                rows.append(
                    {
                        "policy_tag": episode.get("policy_tag"),
                        "seed": episode.get("seed"),
                        "question_id": episode.get("question_id"),
                        "dataset": episode.get("dataset"),
                        "backend": episode.get("backend"),
                        "topk": episode.get("topk"),
                        "turn": int(turn.get("turn", 0)),
                        "question": question,
                        "query": query,
                        "previous_query": previous_query or "",
                        "query_token_count": len(tokens),
                        "query_question_overlap": float(turn.get("query_question_overlap", 0.0)),
                        "query_has_quotes": float(turn.get("query_has_quotes", 0.0)),
                        "query_capitalized_ratio": float(turn.get("query_capitalized_ratio", 0.0)),
                        "query_numeric_ratio": float(turn.get("query_numeric_ratio", 0.0)),
                        "query_lexical_change": float(turn.get("query_lexical_change", 0.0)),
                        "support_recall": float(turn.get("support_recall", 0.0)),
                        "evidence_gain": float(turn.get("evidence_gain", 0.0)),
                    }
                )
                previous_query = query
    if not rows:
        raise RuntimeError(f"No turn records found under {results_dir}")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="work/hard_rq0/results/policies")
    parser.add_argument("--output-dir", default="work/hard_rq0/results/report")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    output_dir = ensure_dir(Path(args.output_dir))
    turns = load_turns(Path(args.results_dir))
    encoder = SentenceTransformer(args.model, device=args.device)
    questions = encoder.encode(
        turns["question"].tolist(),
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    queries = encoder.encode(
        turns["query"].tolist(),
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    turns["query_question_semantic_similarity"] = cosine_rows(queries, questions)

    previous_mask = turns["previous_query"].str.len().gt(0).to_numpy()
    semantic_change = np.zeros(len(turns), dtype=np.float64)
    if previous_mask.any():
        previous = encoder.encode(
            turns.loc[previous_mask, "previous_query"].tolist(),
            batch_size=args.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        semantic_change[previous_mask] = 1.0 - cosine_rows(queries[previous_mask], previous)
    turns["query_semantic_change"] = semantic_change

    turns.to_csv(output_dir / "query_turns.csv", index=False)
    metric_columns = [
        "query_token_count",
        "query_question_overlap",
        "query_question_semantic_similarity",
        "query_has_quotes",
        "query_capitalized_ratio",
        "query_numeric_ratio",
        "query_lexical_change",
        "query_semantic_change",
        "support_recall",
        "evidence_gain",
    ]
    summary = (
        turns.groupby(
            ["policy_tag", "seed", "dataset", "backend", "topk", "turn"],
            as_index=False,
        )[metric_columns]
        .mean()
        .sort_values(["dataset", "topk", "policy_tag", "backend", "seed", "turn"])
    )
    summary.to_csv(output_dir / "query_turn_summary.csv", index=False)

    turn_pivot = summary.pivot_table(
        index=["policy_tag", "seed", "dataset", "backend", "topk"],
        columns="turn",
        values=[
            "query_question_semantic_similarity",
            "query_has_quotes",
            "query_capitalized_ratio",
            "query_lexical_change",
            "query_semantic_change",
        ],
        aggfunc="mean",
    )
    turn_pivot.columns = [f"{metric}_turn{turn}" for metric, turn in turn_pivot.columns]
    turn_pivot.reset_index().to_csv(output_dir / "query_shift_by_turn.csv", index=False)
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
