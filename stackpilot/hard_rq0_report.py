from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stackpilot.common import ensure_dir, read_jsonl_tolerant

METRICS = [
    "em",
    "f1",
    "support_recall",
    "turn1_support_recall",
    "turn2_support_recall",
    "turn3_support_recall",
    "turn2_evidence_gain",
    "turn3_evidence_gain",
    "recovery_at_2",
    "recovery_at_3",
    "full_recovery_at_2",
    "full_recovery_at_3",
]
SPECIALISTS = {
    "bm25-specialist": "bm25",
    "e5-specialist": "e5",
}


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for column in display.select_dtypes(include=["number"]).columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{float(value):.{digits}f}"
        )
    headers = [str(column) for column in display.columns]
    rows = [
        [str(value) for value in row]
        for row in display.itertuples(index=False, name=None)
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    lines = [
        "| "
        + " | ".join(
            headers[index].ljust(widths[index]) for index in range(len(headers))
        )
        + " |",
        "| " + " | ".join("-" * widths[index] for index in range(len(headers))) + " |",
    ]
    lines.extend(
        "| "
        + " | ".join(row[index].ljust(widths[index]) for index in range(len(headers)))
        + " |"
        for row in rows
    )
    return "\n".join(lines)


def load_raw_results(results_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(results_dir.glob("*.jsonl")):
        rows.extend(read_jsonl_tolerant(path))
    if not rows:
        raise RuntimeError(f"No JSONL policy results found under {results_dir}")
    frame = pd.DataFrame(rows)
    required = {
        "policy_tag",
        "seed",
        "question_id",
        "dataset",
        "backend",
        "topk",
        *METRICS,
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Raw result columns are missing: {sorted(missing)}")
    frame["topk"] = pd.to_numeric(frame["topk"], errors="raise").astype(int)
    return frame


def specialist_metrics_available(frame: pd.DataFrame) -> None:
    tags = set(frame["policy_tag"].astype(str))
    missing = {"base-qwen", *SPECIALISTS} - tags
    if missing:
        raise RuntimeError(f"Required policy results are missing: {sorted(missing)}")
    specialist_seed_counts = (
        frame[frame["policy_tag"].isin(SPECIALISTS)]
        .groupby("policy_tag")["seed"]
        .nunique()
    )
    too_small = specialist_seed_counts[specialist_seed_counts < 3]
    if not too_small.empty:
        raise RuntimeError(
            f"Hard-RQ0 requires three specialist seeds; observed {too_small.to_dict()}"
        )


def matched_hard_question_ids(frame: pd.DataFrame) -> pd.DataFrame:
    base = frame[frame["policy_tag"] == "base-qwen"].copy()
    base_turn1 = base.pivot_table(
        index=["question_id", "dataset", "topk"],
        columns="backend",
        values="turn1_support_recall",
        aggfunc="mean",
    ).dropna(subset=["bm25", "e5"])
    base_turn1["base_hard"] = (base_turn1["bm25"] <= 0.5) & (base_turn1["e5"] <= 0.5)

    # Define recoverability from the fixed base policy only. Selecting the
    # diagnostic subset using specialist outcomes would condition the reported
    # treatment effect on the outcome being tested.
    recoverability = (
        base.groupby(["question_id", "dataset", "topk"])["turn3_support_recall"]
        .max()
        .rename("base_best_turn3_recall")
    )
    matched = base_turn1.join(recoverability, how="inner").reset_index()
    matched["recoverable"] = matched["base_best_turn3_recall"] > matched[
        ["bm25", "e5"]
    ].max(axis=1)
    matched["matched_hard"] = matched["base_hard"] & matched["recoverable"]
    return matched


def add_subset_labels(frame: pd.DataFrame, matched: pd.DataFrame) -> pd.DataFrame:
    all_rows = frame.copy()
    all_rows["subset"] = "all"
    matched_keys = matched[matched["matched_hard"]][["question_id", "dataset", "topk"]]
    hard_rows = frame.merge(
        matched_keys,
        on=["question_id", "dataset", "topk"],
        how="inner",
        validate="many_to_many",
    )
    hard_rows["subset"] = "matched-hard"
    return pd.concat([all_rows, hard_rows], ignore_index=True)


def absolute_summary(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = METRICS + ["search_count"]
    return (
        frame.groupby(
            ["subset", "policy_tag", "dataset", "backend", "topk"],
            as_index=False,
        )[metrics]
        .mean()
        .sort_values(["subset", "dataset", "topk", "policy_tag", "backend"])
    )


def gain_over_base(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    base = frame[frame["policy_tag"] == "base-qwen"][
        ["subset", "question_id", "dataset", "backend", "topk", metric]
    ].rename(columns={metric: "base_score"})
    specialists = frame[frame["policy_tag"].isin(SPECIALISTS)][
        [
            "subset",
            "policy_tag",
            "seed",
            "question_id",
            "dataset",
            "backend",
            "topk",
            metric,
        ]
    ].rename(columns={metric: "specialist_score"})
    merged = specialists.merge(
        base,
        on=["subset", "question_id", "dataset", "backend", "topk"],
        how="inner",
        validate="many_to_one",
    )
    merged["metric"] = metric
    merged["gain_over_base"] = merged["specialist_score"] - merged["base_score"]
    return merged


def home_excess(gains: pd.DataFrame) -> pd.DataFrame:
    pivot = gains.pivot_table(
        index=[
            "subset",
            "policy_tag",
            "seed",
            "question_id",
            "dataset",
            "topk",
            "metric",
        ],
        columns="backend",
        values="gain_over_base",
        aggfunc="mean",
    ).reset_index()
    pivot = pivot.dropna(subset=["bm25", "e5"]).copy()
    pivot["home_backend"] = pivot["policy_tag"].map(SPECIALISTS)
    pivot["home_gain"] = np.where(
        pivot["home_backend"] == "bm25", pivot["bm25"], pivot["e5"]
    )
    pivot["away_gain"] = np.where(
        pivot["home_backend"] == "bm25", pivot["e5"], pivot["bm25"]
    )
    pivot["home_excess_gain"] = pivot["home_gain"] - pivot["away_gain"]
    return pivot


def crossed_cluster_bootstrap(
    frame: pd.DataFrame,
    value_column: str,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    if frame.empty:
        return np.nan, np.nan, np.nan
    try:
        matrix = (
            frame.pivot(index="seed", columns="question_id", values=value_column)
            .sort_index(axis=0)
            .sort_index(axis=1)
        )
    except ValueError as exc:
        raise RuntimeError(
            "Bootstrap input must have one value per seed/question cell"
        ) from exc
    if matrix.empty or matrix.isna().any().any():
        raise RuntimeError(
            "Bootstrap input must contain a complete crossed seed/question grid"
        )
    values = matrix.to_numpy(dtype=np.float64)
    seed_count, question_count = values.shape
    observed = float(values.mean(axis=1).mean())
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled_seeds = rng.integers(0, seed_count, size=seed_count)
        # Questions are crossed with seeds, so resample the same question IDs
        # for every sampled seed instead of treating them as nested observations.
        sampled_questions = rng.integers(0, question_count, size=question_count)
        draws[index] = float(values[np.ix_(sampled_seeds, sampled_questions)].mean())
    low, high = np.quantile(draws, [0.025, 0.975])
    return observed, float(low), float(high)


def home_excess_summary(home: pd.DataFrame, samples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    group_columns = ["subset", "policy_tag", "dataset", "topk", "metric"]
    for keys, group in home.groupby(group_columns):
        observed, low, high = crossed_cluster_bootstrap(
            group, "home_excess_gain", samples, rng
        )
        per_seed = group.groupby("seed")["home_excess_gain"].mean()
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "n_questions": int(group["question_id"].nunique()),
                "n_seeds": int(per_seed.shape[0]),
                "home_excess_gain": observed,
                "seed_std": float(per_seed.std(ddof=1))
                if len(per_seed) > 1
                else np.nan,
                "ci_low": low,
                "ci_high": high,
            }
        )
    if not rows:
        return pd.DataFrame(columns=group_columns)
    return pd.DataFrame(rows).sort_values(group_columns)


def gain_summary(gains: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["subset", "policy_tag", "dataset", "backend", "topk", "metric"]
    per_seed = gains.groupby(group_columns + ["seed"], as_index=False)[
        "gain_over_base"
    ].mean()
    return (
        per_seed.groupby(group_columns)["gain_over_base"]
        .agg(gain_over_base="mean", seed_std="std")
        .reset_index()
        .sort_values(group_columns)
    )


def base_backend_gap(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    base = frame[frame["policy_tag"] == "base-qwen"]
    pivot = base.pivot_table(
        index=["subset", "question_id", "dataset", "topk"],
        columns="backend",
        values=metric,
        aggfunc="mean",
    ).dropna(subset=["bm25", "e5"])
    pivot["e5_minus_bm25"] = pivot["e5"] - pivot["bm25"]
    return pivot.reset_index()


def report_section(
    lines: list[str],
    absolute: pd.DataFrame,
    gains: pd.DataFrame,
    interactions: pd.DataFrame,
    base_gaps: pd.DataFrame,
    subset_name: str,
    dataset: str,
    topk: int,
) -> None:
    lines.extend([f"## {subset_name}: {dataset}, top-k={topk}", ""])
    absolute_subset = absolute[
        (absolute["subset"] == subset_name)
        & (absolute["dataset"] == dataset)
        & (absolute["topk"] == topk)
    ]
    lines.extend(["### Absolute performance", "", markdown_table(absolute_subset), ""])
    gain_subset = gains[
        (gains["subset"] == subset_name)
        & (gains["dataset"] == dataset)
        & (gains["topk"] == topk)
    ]
    lines.extend(["### Gain over base Qwen", "", markdown_table(gain_subset), ""])
    interaction_subset = interactions[
        (interactions["subset"] == subset_name)
        & (interactions["dataset"] == dataset)
        & (interactions["topk"] == topk)
    ]
    lines.extend(
        [
            "### Home-backend excess gain (Policy × Backend interaction)",
            "",
            markdown_table(interaction_subset),
            "",
        ]
    )
    gap_subset = base_gaps[
        (base_gaps["subset"] == subset_name)
        & (base_gaps["dataset"] == dataset)
        & (base_gaps["topk"] == topk)
    ]
    lines.extend(["### Base backend gap", "", markdown_table(gap_subset), ""])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="work/hard_rq0/results/policies")
    parser.add_argument("--output-dir", default="work/hard_rq0/results/report")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--threshold", type=float, default=0.05)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = ensure_dir(Path(args.output_dir))
    raw = load_raw_results(results_dir)
    specialist_metrics_available(raw)
    matched = matched_hard_question_ids(raw)
    expanded = add_subset_labels(raw, matched)
    absolute = absolute_summary(expanded)
    all_gains = pd.concat(
        [gain_over_base(expanded, metric) for metric in METRICS], ignore_index=True
    )
    gains = gain_summary(all_gains)
    interactions = home_excess_summary(
        home_excess(all_gains), args.bootstrap_samples, args.bootstrap_seed
    )
    base_gap_rows = pd.concat(
        [
            base_backend_gap(expanded, metric).assign(metric=metric)
            for metric in METRICS
        ],
        ignore_index=True,
    )
    base_gap_summary = (
        base_gap_rows.groupby(["subset", "dataset", "topk", "metric"])["e5_minus_bm25"]
        .agg(base_e5_minus_bm25="mean", question_std="std")
        .reset_index()
        .sort_values(["subset", "dataset", "topk", "metric"])
    )

    absolute.to_csv(output_dir / "absolute_summary.csv", index=False)
    gains.to_csv(output_dir / "gain_over_base.csv", index=False)
    interactions.to_csv(output_dir / "home_backend_excess.csv", index=False)
    base_gap_summary.to_csv(output_dir / "base_backend_gap.csv", index=False)
    matched.to_csv(output_dir / "difficulty_matching.csv", index=False)
    matched_units = matched.loc[
        matched["matched_hard"], ["question_id", "dataset", "topk"]
    ].copy()
    matched_units["question_id"] = matched_units["question_id"].astype(str)
    matched_units["dataset"] = matched_units["dataset"].astype(str)
    matched_units["topk"] = matched_units["topk"].astype(int)
    matched_unit_records = matched_units.to_dict(orient="records")
    matched_ids = sorted(matched_units["question_id"].unique().tolist())
    (output_dir / "matched_hard_question_ids.json").write_text(
        json.dumps(matched_ids, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "matched_hard_units.json").write_text(
        json.dumps(matched_unit_records, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Hard-RQ0 retrieval-stack transfer report",
        "",
        "This report separates general RL improvement from retriever-specific specialization. ",
        "For each specialist and test backend it reports gain over the same base Qwen policy. ",
        "The home-backend excess is the difference-in-differences Policy × Backend interaction:",
        "",
        "`(specialist_home - base_home) - (specialist_away - base_away)`.",
        "",
        "The matched-hard subset contains evaluation units for which base Qwen has first-turn support recall <= 0.5 on both BM25 and E5, and base Qwen itself improves support recall by turn 3. Specialist outcomes are not used to select this subset.",
        "",
        f"Matched-hard evaluation units: **{len(matched_units)}** / **{len(matched)}**; unique questions: **{len(matched_ids)}** / **{matched['question_id'].nunique()}**.",
        "",
    ]

    for subset_name in ("all", "matched-hard"):
        subset_frame = absolute[absolute["subset"] == subset_name]
        for dataset in sorted(subset_frame["dataset"].unique()):
            for topk in sorted(
                subset_frame.loc[subset_frame["dataset"] == dataset, "topk"].unique()
            ):
                report_section(
                    lines,
                    absolute,
                    gains,
                    interactions,
                    base_gap_summary,
                    subset_name,
                    dataset,
                    int(topk),
                )

    lines.extend(["## Go / no-go checks", ""])
    target_metrics = {
        "support_recall",
        "turn2_evidence_gain",
        "turn3_evidence_gain",
        "recovery_at_2",
        "recovery_at_3",
    }
    target_rows = interactions[
        (interactions["subset"] == "matched-hard")
        & interactions["metric"].isin(target_metrics)
    ]
    if target_rows.empty:
        lines.append(
            "- Matched-hard specialist results are incomplete; no interaction estimate is available."
        )
    else:
        for row in target_rows.itertuples(index=False):
            passes = bool(row.home_excess_gain >= args.threshold and row.ci_low > 0)
            lines.append(
                f"- {row.policy_tag}, {row.dataset}, top-k={int(row.topk)}, {row.metric}: "
                f"home excess={row.home_excess_gain:.4f}, 95% CI=[{row.ci_low:.4f}, {row.ci_high:.4f}] "
                f"→ {'GO' if passes else 'NO-GO'}"
            )
    lines.extend(
        [
            "",
            "## Interpretation rules",
            "",
            "- A large gain over base on both backends is a general RL effect, not specialization.",
            "- A positive home-backend excess isolates retriever-specific improvement after subtracting the base backend gap.",
            "- The strongest evidence is near-zero turn-1 interaction followed by positive turn-2/3 evidence-gain or recovery interaction.",
            "- The hard-RQ0 gate requires at least 0.05 home-backend excess with a crossed seed/question bootstrap CI above zero on the matched-hard subset.",
            "- If this remains below 0.03, the hidden-retriever adaptation hypothesis should be rejected for this setup.",
            "",
        ]
    )

    report_path = output_dir / "HARD_RQ0_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
