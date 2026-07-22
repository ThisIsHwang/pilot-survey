from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from stackpilot.common import ensure_dir


def load_summaries(results_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(results_dir.glob("*_summary.csv")):
        frame = pd.read_csv(path)
        if "policy_tag" not in frame.columns:
            frame.insert(0, "policy_tag", path.stem.removesuffix("_summary"))
        frames.append(frame)
    if not frames:
        raise RuntimeError(f"No policy summary CSV files found under {results_dir}")
    return pd.concat(frames, ignore_index=True)


def pivot_metric(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    blind = frame[frame["variant"] == "blind"].copy()
    return blind.pivot_table(index="policy_tag", columns="backend", values=metric, aggfunc="mean")


def diagonal_score(matrix: pd.DataFrame) -> float | None:
    required_rows = {"bm25-specialist", "e5-specialist"}
    required_cols = {"bm25", "e5"}
    if not required_rows.issubset(matrix.index) or not required_cols.issubset(matrix.columns):
        return None
    return float(
        (
            matrix.loc["bm25-specialist", "bm25"]
            - matrix.loc["bm25-specialist", "e5"]
            + matrix.loc["e5-specialist", "e5"]
            - matrix.loc["e5-specialist", "bm25"]
        )
        / 2.0
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="work/results/policies")
    parser.add_argument("--output-dir", default="work/results/rq0")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = ensure_dir(Path(args.output_dir))
    frame = load_summaries(results_dir)
    frame.to_csv(output_dir / "all_policy_summaries.csv", index=False)

    metrics = ["em", "f1", "support_recall", "search_count"]
    matrices = {metric: pivot_metric(frame, metric) for metric in metrics}
    for metric, matrix in matrices.items():
        matrix.to_csv(output_dir / f"transfer_matrix_{metric}.csv")

    lines = ["# RQ0 retrieval-stack transfer report", ""]
    lines.append("This report tests whether a search policy trained or selected for one backend transfers to another backend.")
    lines.append("")

    for metric, matrix in matrices.items():
        lines.extend([f"## Blind-policy {metric}", "", matrix.round(4).to_markdown(), ""])

    support_d = diagonal_score(matrices["support_recall"])
    answer_d = diagonal_score(matrices["f1"])
    lines.extend(["## Go / no-go checks", ""])
    if support_d is None:
        lines.append("- Specialist rows are incomplete; diagonal dominance could not be computed.")
    else:
        lines.append(f"- Specialist support-recall diagonal dominance: **{support_d:.4f}**")
        lines.append(f"- Specialist answer-F1 diagonal dominance: **{answer_d:.4f}**")
        lines.append(
            "- Suggested go criterion: continue only if diagonal dominance is at least 0.05 and is not explained by the base-policy backend gap."
        )

    base = matrices["support_recall"]
    if "base-qwen" in base.index and {"bm25", "e5"}.issubset(base.columns):
        gap = float(abs(base.loc["base-qwen", "bm25"] - base.loc["base-qwen", "e5"]))
        lines.append(f"- Base-policy BM25/E5 support-recall gap: **{gap:.4f}**")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A large specialist diagonal with a small base-policy backend gap supports policy–retriever specialization. ",
            "If the base-policy gap is already as large as the specialist gap, the result is more likely a raw retriever-quality difference. ",
            "If a mixed-stack policy matches both specialists, a separate online adaptation module may not be necessary.",
            "",
        ]
    )

    report_path = output_dir / "RQ0_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
