from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SPECIALISTS = {"bm25-specialist": "bm25", "e5-specialist": "e5"}
STYLE_METRICS = [
    "query_token_count",
    "query_question_overlap",
    "query_question_semantic_similarity",
    "query_has_quotes",
    "query_capitalized_ratio",
    "query_lexical_change",
    "query_semantic_change",
    "evidence_gain",
]


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No matched-hard query rows._"
    display = frame.copy()
    for column in display.select_dtypes(include=["number"]).columns:
        display[column] = display[column].map(lambda value: f"{float(value):.4f}")
    headers = list(display.columns)
    rows = [[str(value) for value in row] for row in display.itertuples(index=False, name=None)]
    widths = [max(len(str(headers[i])), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    lines = [
        "| " + " | ".join(str(headers[i]).ljust(widths[i]) for i in range(len(headers))) + " |",
        "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |",
    ]
    lines.extend(
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for row in rows
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    summary = pd.read_csv(args.summary)
    matched = summary[
        (summary["subset"] == "matched-hard")
        & summary["policy_tag"].isin(SPECIALISTS)
        & summary["turn"].isin((1, 2, 3))
    ].copy()
    aggregate = (
        matched.groupby(
            ["policy_tag", "dataset", "backend", "topk", "turn"],
            as_index=False,
        )[STYLE_METRICS]
        .mean()
        .sort_values(["dataset", "topk", "policy_tag", "backend", "turn"])
    )

    excess_rows = []
    for (policy_tag, dataset, topk, turn), group in aggregate.groupby(
        ["policy_tag", "dataset", "topk", "turn"]
    ):
        indexed = group.set_index("backend")
        if not {"bm25", "e5"}.issubset(indexed.index):
            continue
        home = SPECIALISTS[policy_tag]
        away = "e5" if home == "bm25" else "bm25"
        row = {
            "policy_tag": policy_tag,
            "dataset": dataset,
            "topk": topk,
            "turn": turn,
            "home_backend": home,
        }
        for metric in STYLE_METRICS:
            row[f"{metric}_home_minus_away"] = float(
                indexed.loc[home, metric] - indexed.loc[away, metric]
            )
        excess_rows.append(row)
    excess = pd.DataFrame(excess_rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Matched-hard query behavior",
        "",
        "All values are averaged over specialist seeds. Home-minus-away differences show how the same learned policy changes when connected to its training retriever versus the other retriever.",
        "",
        "## Per-turn query statistics",
        "",
        markdown_table(aggregate),
        "",
        "## Home-minus-away query-style differences",
        "",
        markdown_table(excess),
        "",
        "A convincing specialization pattern should couple positive home evidence gain with backend-appropriate reformulation changes after turn 1; style differences without evidence gain are not sufficient.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(output_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
