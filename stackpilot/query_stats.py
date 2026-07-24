from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from stackpilot.common import ensure_dir, read_jsonl_tolerant
from stackpilot.react_agent_eval import RESULT_SCHEMA

DEFAULT_TAGS = (
    "base-qwen",
    "official-searchr1",
    "bm25-specialist",
    "e5-specialist",
)
FEATURE_COLUMNS = (
    "query_chars",
    "query_tokens",
    "quoted_spans",
    "uppercase_tokens",
    "digit_tokens",
)


def query_features(query: str) -> dict[str, float]:
    tokens = re.findall(r"\b\w+\b", query)
    quoted = re.findall(r'"[^"]+"', query)
    return {
        "query_chars": float(len(query)),
        "query_tokens": float(len(tokens)),
        "quoted_spans": float(len(quoted)),
        "uppercase_tokens": float(
            sum(token.isupper() and len(token) > 1 for token in tokens)
        ),
        "digit_tokens": float(
            sum(any(ch.isdigit() for ch in token) for token in tokens)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="work/results/policies")
    parser.add_argument("--output-dir", default="work/results/rq0")
    parser.add_argument("--tags", nargs="+", default=DEFAULT_TAGS)
    args = parser.parse_args()
    if len(set(args.tags)) != len(args.tags):
        raise ValueError(f"Duplicate policy tags are not allowed: {args.tags}")

    results_dir = Path(args.results_dir)
    output_dir = ensure_dir(Path(args.output_dir))
    rows: list[dict] = []
    summary_rows: list[dict] = []

    for policy_tag in args.tags:
        path = results_dir / f"{policy_tag}.jsonl"
        if not path.is_file():
            raise RuntimeError(f"Missing policy episodes: {path}")
        summary_path = results_dir / f"{policy_tag}_summary.csv"
        if not summary_path.is_file():
            raise RuntimeError(
                f"Missing current-run summary for {path}: {summary_path}"
            )
        summary = pd.read_csv(summary_path)
        if "run_signature" not in summary.columns:
            raise RuntimeError(f"Summary has no run_signature column: {summary_path}")
        signatures = {
            str(value) for value in summary["run_signature"].dropna().unique().tolist()
        }
        if len(signatures) != 1:
            raise RuntimeError(
                f"Expected one current run signature in {summary_path}, found {sorted(signatures)}"
            )
        run_signature = next(iter(signatures))
        required_summary_columns = {
            "evaluation_signature",
            "n_questions",
            "backend",
            "variant",
        }
        missing = required_summary_columns - set(summary.columns)
        if missing:
            raise RuntimeError(f"{summary_path} is missing columns: {sorted(missing)}")
        allowed_pairs = {
            (str(row.backend), str(row.variant)) for row in summary.itertuples()
        }
        if not allowed_pairs:
            raise RuntimeError(f"No backend/variant rows in {summary_path}")
        n_values = pd.to_numeric(summary["n_questions"], errors="coerce")
        if n_values.isna().any() or (n_values <= 0).any() or (n_values % 1 != 0).any():
            raise RuntimeError(f"Invalid n_questions in {summary_path}")
        expected_questions = {int(value) for value in n_values.unique()}
        if len(expected_questions) != 1:
            raise RuntimeError(f"Mixed n_questions values in {summary_path}")
        expected_count = next(iter(expected_questions))
        evaluation_signatures = {
            str(value)
            for value in summary["evaluation_signature"].dropna().unique()
            if str(value)
        }
        if len(evaluation_signatures) != 1:
            raise RuntimeError(f"Expected one evaluation signature in {summary_path}")
        evaluation_signature = next(iter(evaluation_signatures))
        episode_by_key = {}
        for episode in read_jsonl_tolerant(path):
            pair = (str(episode.get("backend")), str(episode.get("variant")))
            if (
                episode.get("schema") == RESULT_SCHEMA
                and str(episode.get("run_signature")) == run_signature
                and str(episode.get("evaluation_signature")) == evaluation_signature
                and str(episode.get("policy_tag", policy_tag)) == policy_tag
                and pair in allowed_pairs
            ):
                queries = episode.get("queries")
                if not isinstance(queries, list) or not all(
                    isinstance(query, str) for query in queries
                ):
                    raise RuntimeError(
                        f"Current-run episode has invalid queries in {path}: "
                        f"question_id={episode.get('question_id')!r}"
                    )
                key = (str(episode.get("question_id")), *pair)
                if key in episode_by_key:
                    raise RuntimeError(
                        f"Duplicate current-run episode in {path}: {key}"
                    )
                episode_by_key[key] = episode
        episodes = list(episode_by_key.values())
        question_ids_by_pair = {}
        for backend, variant in sorted(allowed_pairs):
            pair_episodes = [
                episode
                for episode in episodes
                if str(episode.get("backend")) == backend
                and str(episode.get("variant")) == variant
            ]
            if len(pair_episodes) != expected_count:
                raise RuntimeError(
                    f"{path} has {len(pair_episodes)} current episodes for "
                    f"{backend}/{variant}; expected {expected_count}"
                )
            question_ids_by_pair[(backend, variant)] = {
                str(episode.get("question_id")) for episode in pair_episodes
            }
            feature_rows = [
                query_features(str(query))
                for episode in pair_episodes
                for query in (episode.get("queries") or [])
            ]
            aggregate = {
                "policy_tag": policy_tag,
                "run_signature": run_signature,
                "evaluation_signature": evaluation_signature,
                "backend": backend,
                "variant": variant,
                "n_episodes": len(pair_episodes),
                "n_queries": len(feature_rows),
                "zero_search_rate": sum(
                    not (episode.get("queries") or []) for episode in pair_episodes
                )
                / len(pair_episodes),
            }
            for column in FEATURE_COLUMNS:
                aggregate[column] = (
                    sum(row[column] for row in feature_rows) / len(feature_rows)
                    if feature_rows
                    else float("nan")
                )
            summary_rows.append(aggregate)
        selected_question_sets = list(question_ids_by_pair.values())
        if selected_question_sets and any(
            question_ids != selected_question_sets[0]
            for question_ids in selected_question_sets[1:]
        ):
            raise RuntimeError(
                f"Current-run backend/variant rows use different question IDs: {path}"
            )
        for episode in episodes:
            queries = episode.get("queries") or []
            for turn, query in enumerate(queries, start=1):
                row = {
                    "policy_tag": episode.get("policy_tag", policy_tag),
                    "run_signature": run_signature,
                    "evaluation_signature": evaluation_signature,
                    "backend": episode.get("backend"),
                    "variant": episode.get("variant"),
                    "question_id": episode.get("question_id"),
                    "turn": turn,
                    "query": query,
                }
                row.update(query_features(str(query)))
                rows.append(row)

    detail_path = output_dir / "query_stats_detail.csv"
    summary_path = output_dir / "query_stats_summary.csv"
    detail_columns = [
        "policy_tag",
        "run_signature",
        "evaluation_signature",
        "backend",
        "variant",
        "question_id",
        "turn",
        "query",
        *FEATURE_COLUMNS,
    ]
    frame = pd.DataFrame(rows, columns=detail_columns)
    frame.to_csv(detail_path, index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(summary_path, index=False)
    print(summary.round(4).to_string(index=False))
    print(
        json.dumps({"detail": str(detail_path), "summary": str(summary_path)}, indent=2)
    )


if __name__ == "__main__":
    main()
