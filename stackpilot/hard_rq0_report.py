from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from stackpilot.common import ensure_dir, read_jsonl_tolerant

METRICS = ["em", "f1", "support_recall"]
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
    rows = [[str(value) for value in row] for row in display.itertuples(index=False, name=None)]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    lines = [
        "| " + " | ".join(headers[index].ljust(widths[index]) for index in range(len(headers))) + " |",
        "| " + " | ".join("-" * widths[index] for index in range(len(headers))) + " |",
    ]
    lines.extend(
        "| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(headers))) + " |"
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
    return frame


def absolute_summary(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = METRICS + [
        "turn1_support_recall",
        "turn2_evidence_gain",
        "recovered_after_first_miss",
        "fully_recovered_after_first_miss",
        "search_count",
    ]
    available = [metric for metric in metrics if metric in frame.columns]
    return (
        frame.groupby(["policy_tag", "dataset", "backend", "topk"], as_index=False)[available]
        .mean()
        .sort_values(["dataset", "topk", "policy_tag", "backend"])
    )


def gain_over_base(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    base = frame[frame["policy_tag"] == "base-qwen"][
        ["question_id", "dataset", "backend", "topk", metric]
    ].rename(columns={metric: "base_score"})
    specialists = frame[frame["policy_tag"].isin(SPECIALISTS)][
        ["policy_tag", "seed", "question_id", "dataset", "backend", "topk", metric]
    ].rename(columns={metric: "specialist_score"})
    merged = specialists.merge(
        base,
        on=["question_id", "dataset", "backend", "topk"],
        how="inner",
        validate="many_to_one",
    )
    merged["metric"] = metric
    merged["gain_over_base"] = merged["specialist_score"] - merged["base_score"]
    return merged


def home_excess(gains: pd.DataFrame) -> pd.DataFrame:
    pivot = gains.pivot_table(
        index=["policy_tag", "seed", "question_id", "dataset", "topk", "metric"],
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


def hierarchical_bootstrap(
    frame: pd.DataFrame,
    value_column: str,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    if frame.empty:
        return np.nan, np.nan, np.nan
    seeds = sorted(frame["seed"].unique())
    observed = float(frame.groupby("seed")[value_column].mean().mean())
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        seed_means = []
        for seed in sampled_seeds:
            values = frame.loc[frame["seed"] == seed, value_column].to_numpy(dtype=float)
            sampled_values = rng.choice(values, size=len(values), replace=True)
            seed_means.append(float(sampled_values.mean()))
        draws[index] = float(np.mean(seed_means))
    low, high = np.quantile(draws, [0.025, 0.975])
    return observed, float(low), float(high)


def home_excess_summary(home: pd.DataFrame, samples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    group_columns = ["policy_tag", "dataset", "topk", "metric"]
    for keys, group in home.groupby(group_columns):
        observed, low, high = hierarchical_bootstrap(
            group, "home_excess_gain", samples, rng
        )
        per_seed = group.groupby("seed")["home_excess_gain"].mean()
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "n_seeds": int(per_seed.shape[0]),
                "home_excess_gain": observed,
                "seed_std": float(per_seed.std(ddof=1)) if len(per_seed) > 1 else np.nan,
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame(rows).sort_values(group_columns)


def gain_summary(gains: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["policy_tag", "dataset", "backend", "topk", "metric"]
    per_seed = gains.groupby(group_columns + ["seed"], as_index=False)["gain_over_base"].mean()
    return (
        per_seed.groupby(group_columns)["gain_over_base"]
        .agg(gain_over_base="mean", seed_std="std")
        .reset_index()
        .sort_values(group_columns)
    )


def base_backend_gap(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    base = frame[frame["policy_tag"] == "base-qwen"]
    pivot = base.pivot_table(
        index=["question_id", "dataset", "topk"],
        columns="backend",
        values=metric,
        aggfunc="mean",
    ).dropna(subset=["bm25", "e5"])
    pivot["e5_minus_bm25"] = pivot["e5"] - pivot["bm25"]
    return pivot.reset_index()


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
    absolute = absolute_summary(raw)
    all_gains = pd.concat([gain_over_base(raw, metric) for metric in METRICS], ignore_index=True)
    gains = gain_summary(all_gains)
    home = home_excess(all_gains)
    interactions = home_excess_summary(home, args.bootstrap_samples, args.bootstrap_seed)
    base_gaps = pd.concat(
        [base_backend_gap(raw, metric).assign(metric=metric) for metric in METRICS],
        ignore_index=True,
    )
    base_gap_summary = (
        base_gaps.groupby(["dataset", "topk", "metric"])["e5_minus_bm25"]
        .agg(base_e5_minus_bm25="mean", question_std="std")
        .reset_index()
        .sort_values(["dataset", "topk", "metric"])
    )

    absolute.to_csv(output_dir / "absolute_summary.csv", index=False)
    gains.to_csv(output_dir / "gain_over_base.csv", index=False)
    interactions.to_csv(output_dir / "home_backend_excess.csv", index=False)
    base_gap_summary.to_csv(output_dir / "base_backend_gap.csv", index=False)

    lines = [
        "# Hard-RQ0 retrieval-stack transfer report",
        "",
        "This report separates a general RL improvement from retriever-specific specialization. ",
        "For each specialist and test backend it reports gain over the same base Qwen policy. ",
        "The home-backend excess is a difference-in-differences estimate of the Policy × Backend interaction:",
        "",
        "`(specialist_home - base_home) - (specialist_away - base_away)`.",
        "",
    ]

    for dataset in sorted(absolute["dataset"].unique()):
        for topk in sorted(absolute.loc[absolute["dataset"] == dataset, "topk"].unique()):
            lines.extend([f"## {dataset}, top-k={topk}", ""])
            subset = absolute[(absolute["dataset"] == dataset) & (absolute["topk"] == topk)]
            lines.extend(["### Absolute performance", "", markdown_table(subset), ""])
            gain_subset = gains[(gains["dataset"] == dataset) & (gains["topk"] == topk)]
            lines.extend(["### Gain over base Qwen", "", markdown_table(gain_subset), ""])
            interaction_subset = interactions[
                (interactions["dataset"] == dataset) & (interactions["topk"] == topk)
            ]
            lines.extend(
                [
                    "### Home-backend excess gain (Policy × Backend interaction)",
                    "",
                    markdown_table(interaction_subset),
                    "",
                ]
            )
            gap_subset = base_gap_summary[
                (base_gap_summary["dataset"] == dataset)
                & (base_gap_summary["topk"] == topk)
            ]
            lines.extend(["### Base backend gap", "", markdown_table(gap_subset), ""])

    lines.extend(["## Go / no-go checks", ""])
    if interactions.empty:
        lines.append("- Specialist runs are incomplete; no interaction estimate is available.")
    else:
        for row in interactions.itertuples(index=False):
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
            "- The hard-RQ0 gate requires a home-backend excess of at least the configured threshold and a bootstrap CI above zero.",
            "- Turn-1 recall, turn-2 evidence gain, and recovery after a first-turn miss quantify whether the setting leaves room for online adaptation.",
            "",
        ]
    )

    report_path = output_dir / "HARD_RQ0_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
