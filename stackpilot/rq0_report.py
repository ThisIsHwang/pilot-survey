from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from stackpilot.common import ensure_dir

RQ0_TAGS = (
    "base-qwen",
    "official-searchr1",
    "bm25-specialist",
    "e5-specialist",
)


def load_summaries(results_dir: Path) -> pd.DataFrame:
    frames = []
    required_columns = {
        "policy_tag",
        "run_signature",
        "evaluation_signature",
        "n_questions",
        "backend",
        "variant",
        "em",
        "f1",
        "support_recall",
        "search_count",
    }
    for policy_tag in RQ0_TAGS:
        path = results_dir / f"{policy_tag}_summary.csv"
        if not path.is_file():
            raise RuntimeError(f"Missing required RQ0 policy summary: {path}")
        frame = pd.read_csv(path)
        expected_tag = path.stem.removesuffix("_summary")
        if "policy_tag" not in frame.columns:
            frame.insert(0, "policy_tag", expected_tag)
        missing = required_columns - set(frame.columns)
        if missing:
            raise RuntimeError(f"{path} is missing columns: {sorted(missing)}")
        if frame["run_signature"].nunique(dropna=False) != 1:
            raise RuntimeError(f"{path} mixes multiple run signatures")
        run_signatures = frame["run_signature"].dropna().astype(str)
        if (
            len(run_signatures) != len(frame)
            or not run_signatures.str.len().gt(0).all()
        ):
            raise RuntimeError(f"{path} has an empty run signature")
        evaluation_signatures = frame["evaluation_signature"].dropna().astype(str)
        if (
            len(evaluation_signatures) != len(frame)
            or evaluation_signatures.nunique() != 1
            or not evaluation_signatures.str.len().gt(0).all()
        ):
            raise RuntimeError(f"{path} has an invalid evaluation signature")
        if frame["n_questions"].nunique(dropna=False) != 1:
            raise RuntimeError(f"{path} mixes multiple evaluation limits")
        n_questions = pd.to_numeric(frame["n_questions"], errors="coerce")
        if (
            n_questions.isna().any()
            or (n_questions <= 0).any()
            or (n_questions % 1 != 0).any()
        ):
            raise RuntimeError(f"{path} has invalid n_questions values")
        for column in ("em", "f1", "support_recall", "search_count"):
            numeric = pd.to_numeric(frame[column], errors="coerce")
            if not np.isfinite(numeric).all():
                raise RuntimeError(f"{path} has non-finite {column} values")
            frame[column] = numeric
        for column in ("em", "f1", "support_recall"):
            if ((frame[column] < 0) | (frame[column] > 1)).any():
                raise RuntimeError(f"{path} has out-of-range {column} values")
        if (frame["search_count"] < 0).any():
            raise RuntimeError(f"{path} has negative search_count values")
        actual_tags = {str(value) for value in frame["policy_tag"].unique()}
        if actual_tags != {expected_tag}:
            raise RuntimeError(
                f"{path} contains policy tags {sorted(actual_tags)}, "
                f"expected only {expected_tag!r}"
            )
        frames.append(frame)
    if not frames:
        raise RuntimeError(f"No policy summary CSV files found under {results_dir}")
    return pd.concat(frames, ignore_index=True)


def pivot_metric(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    blind = frame[frame["variant"] == "blind"].copy()
    return blind.pivot_table(
        index="policy_tag", columns="backend", values=metric, aggfunc="mean"
    )


def validate_complete_matrix(frame: pd.DataFrame) -> None:
    required_tags = set(RQ0_TAGS)
    required_pairs = {
        (tag, backend, "blind") for tag in required_tags for backend in ("bm25", "e5")
    }
    actual_pairs = {
        (str(row.policy_tag), str(row.backend), str(row.variant))
        for row in frame.itertuples()
    }
    missing = sorted(required_pairs - actual_pairs)
    if missing:
        raise RuntimeError(f"RQ0 policy matrix is incomplete; missing rows: {missing}")
    unexpected = sorted(actual_pairs - required_pairs)
    if unexpected:
        raise RuntimeError(f"RQ0 policy matrix contains unexpected rows: {unexpected}")
    if len(frame) != len(required_pairs):
        raise RuntimeError(
            f"RQ0 policy matrix must contain exactly {len(required_pairs)} rows; "
            f"found {len(frame)}"
        )
    required = frame[
        frame["policy_tag"].isin(required_tags)
        & frame["backend"].isin(("bm25", "e5"))
        & (frame["variant"] == "blind")
    ]
    duplicate_counts = required.groupby(
        ["policy_tag", "backend", "variant"], dropna=False
    ).size()
    duplicates = duplicate_counts[duplicate_counts != 1]
    if not duplicates.empty:
        raise RuntimeError(
            f"RQ0 policy matrix contains duplicate rows: {duplicates.to_dict()}"
        )
    question_counts = {int(value) for value in required["n_questions"].unique()}
    if len(question_counts) != 1:
        raise RuntimeError(
            "RQ0 policy evaluations use different question counts: "
            f"{sorted(question_counts)}"
        )
    evaluation_signatures = {
        str(value) for value in required["evaluation_signature"].unique()
    }
    if len(evaluation_signatures) != 1:
        raise RuntimeError(
            "RQ0 policies were evaluated on different question/backend selections: "
            f"{sorted(evaluation_signatures)}"
        )


def diagonal_score(matrix: pd.DataFrame) -> float | None:
    required_rows = {"bm25-specialist", "e5-specialist"}
    required_cols = {"bm25", "e5"}
    if not required_rows.issubset(matrix.index) or not required_cols.issubset(
        matrix.columns
    ):
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
    validate_complete_matrix(frame)
    frame.to_csv(output_dir / "all_policy_summaries.csv", index=False)

    metrics = ["em", "f1", "support_recall", "search_count"]
    matrices = {metric: pivot_metric(frame, metric) for metric in metrics}
    for metric, matrix in matrices.items():
        matrix.to_csv(output_dir / f"transfer_matrix_{metric}.csv")

    lines = ["# RQ0 retrieval-stack transfer report", ""]
    lines.append(
        "This report tests whether a search policy trained or selected for one backend transfers to another backend."
    )
    lines.append("")

    for metric, matrix in matrices.items():
        lines.extend(
            [
                f"## Blind-policy {metric}",
                "",
                "```text",
                matrix.round(4).to_string(),
                "```",
                "",
            ]
        )

    support_d = diagonal_score(matrices["support_recall"])
    answer_d = diagonal_score(matrices["f1"])
    lines.extend(["## Go / no-go checks", ""])
    if support_d is None or answer_d is None:
        lines.append(
            "- Specialist rows are incomplete; diagonal dominance could not be computed."
        )
    else:
        lines.append(
            f"- Specialist support-recall diagonal dominance: **{support_d:.4f}**"
        )
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
            "A large specialist diagonal with a small base-policy backend gap supports policy-retriever specialization. ",
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
