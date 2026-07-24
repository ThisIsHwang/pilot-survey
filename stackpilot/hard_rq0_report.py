from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from stackpilot.common import ensure_dir, read_jsonl_tolerant

METRICS = [
    "em",
    "f1",
    "retrieved_support_title_recall",
    "observed_support_title_recall",
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
PRIMARY_METRIC = "observed_support_title_recall"
PRIMARY_SUBSET = "all"
PRIMARY_TOPK = 3
MIN_CONFIRMATORY_SEEDS = 8

T_CRITICAL_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def exact_outer_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    on: list[str],
    left_name: str,
    right_name: str,
) -> pd.DataFrame:
    try:
        merged = left.merge(
            right,
            how="outer",
            on=on,
            validate="one_to_one",
            indicator=True,
        )
    except pd.errors.MergeError as exc:
        raise RuntimeError(
            f"{left_name} and {right_name} do not have one row per matched key {on}"
        ) from exc
    bad = merged[merged["_merge"] != "both"]
    if not bad.empty:
        raise RuntimeError(
            f"incomplete matched grid between {left_name} and {right_name}:\n"
            f"{bad[on + ['_merge']].head(20).to_string(index=False)}"
        )
    return merged.drop(columns="_merge")


def seed_mean_interval(values: pd.Series) -> dict[str, object]:
    ordered = values.sort_index().astype(float)
    count = len(ordered)
    mean = float(ordered.mean()) if count else np.nan
    std = float(ordered.std(ddof=1)) if count > 1 else np.nan
    if count > 1:
        critical = T_CRITICAL_975.get(count - 1, 1.96)
        half_width = critical * std / math.sqrt(count)
        low, high = mean - half_width, mean + half_width
    else:
        low = high = np.nan
    seed_values = {str(seed): float(value) for seed, value in ordered.items()}
    return {
        "mean": mean,
        "seed_std": std,
        "n_seeds": count,
        "seed_ci_low": float(low),
        "seed_ci_high": float(high),
        "seed_values": json.dumps(seed_values, sort_keys=True),
    }


def exact_one_sided_sign_flip(values: pd.Series) -> float:
    observed = np.asarray(values, dtype=np.float64)
    if observed.size == 0 or not np.isfinite(observed).all():
        return np.nan
    observed_mean = float(observed.mean())
    exceedances = 0
    total = 2 ** int(observed.size)
    for signs in itertools.product((-1.0, 1.0), repeat=int(observed.size)):
        if float((observed * np.asarray(signs)).mean()) >= observed_mean - 1e-12:
            exceedances += 1
    return exceedances / total


def holm_adjust(p_values: pd.Series) -> pd.Series:
    values = p_values.astype(float)
    order = values.sort_values().index
    adjusted = pd.Series(index=values.index, dtype=float)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * float(values.loc[index]))
        adjusted.loc[index] = min(1.0, running)
    return adjusted


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
    frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)
    return frame


def specialist_metrics_available(frame: pd.DataFrame) -> None:
    tags = set(frame["policy_tag"].astype(str))
    missing = {"base-qwen", *SPECIALISTS} - tags
    if missing:
        raise RuntimeError(f"Required policy results are missing: {sorted(missing)}")
    specialist_rows = frame[frame["policy_tag"].isin(SPECIALISTS)]
    seed_sets = {
        tag: set(
            specialist_rows.loc[specialist_rows["policy_tag"] == tag, "seed"].astype(
                int
            )
        )
        for tag in SPECIALISTS
    }
    if len({frozenset(values) for values in seed_sets.values()}) != 1:
        raise RuntimeError(
            f"Specialists must use identical training seed sets; observed {seed_sets}"
        )
    specialist_seed_counts = pd.Series(
        {tag: len(values) for tag, values in seed_sets.items()}
    )
    too_small = specialist_seed_counts[specialist_seed_counts < 3]
    if not too_small.empty:
        raise RuntimeError(
            f"Hard-RQ0 requires three specialist seeds; observed {too_small.to_dict()}"
        )


def matched_hard_question_ids(frame: pd.DataFrame) -> pd.DataFrame:
    base = frame[frame["policy_tag"] == "base-qwen"].copy()
    keys = ["question_id", "dataset", "topk"]
    bm25 = base[base["backend"] == "bm25"][keys + ["turn1_support_recall"]].rename(
        columns={"turn1_support_recall": "bm25"}
    )
    e5 = base[base["backend"] == "e5"][keys + ["turn1_support_recall"]].rename(
        columns={"turn1_support_recall": "e5"}
    )
    base_turn1 = exact_outer_merge(
        bm25,
        e5,
        on=keys,
        left_name="base BM25 first-turn cells",
        right_name="base E5 first-turn cells",
    )
    base_turn1["base_hard"] = (base_turn1["bm25"] <= 0.5) & (base_turn1["e5"] <= 0.5)

    # Define recoverability from the fixed base policy only. Selecting the
    # diagnostic subset using specialist outcomes would condition the reported
    # treatment effect on the outcome being tested.
    recoverability = (
        base.groupby(["question_id", "dataset", "topk"])["turn3_support_recall"]
        .max()
        .rename("base_best_turn3_recall")
        .reset_index()
    )
    matched = exact_outer_merge(
        base_turn1,
        recoverability,
        on=keys,
        left_name="base first-turn pairs",
        right_name="base turn-three cells",
    )
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
    keys = ["subset", "question_id", "dataset", "backend", "topk"]
    base = frame[frame["policy_tag"] == "base-qwen"][keys + [metric]].rename(
        columns={metric: "base_score"}
    )
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
    merged_groups = []
    for (policy_tag, seed), group in specialists.groupby(
        ["policy_tag", "seed"], sort=True
    ):
        merged_groups.append(
            exact_outer_merge(
                group,
                base,
                on=keys,
                left_name=f"{policy_tag}/seed{seed}",
                right_name="base-qwen",
            )
        )
    if not merged_groups:
        raise RuntimeError("No specialist rows are available for gain-over-base")
    merged = pd.concat(merged_groups, ignore_index=True)
    merged["metric"] = metric
    merged["gain_over_base"] = merged["specialist_score"] - merged["base_score"]
    return merged


def home_excess(gains: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "subset",
        "policy_tag",
        "seed",
        "question_id",
        "dataset",
        "topk",
        "metric",
    ]
    bm25 = gains[gains["backend"] == "bm25"][keys + ["gain_over_base"]].rename(
        columns={"gain_over_base": "bm25"}
    )
    e5 = gains[gains["backend"] == "e5"][keys + ["gain_over_base"]].rename(
        columns={"gain_over_base": "e5"}
    )
    pivot = exact_outer_merge(
        bm25,
        e5,
        on=keys,
        left_name="BM25 gain cells",
        right_name="E5 gain cells",
    )
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
        _observed, low, high = crossed_cluster_bootstrap(
            group, "home_excess_gain", samples, rng
        )
        per_seed = group.groupby("seed")["home_excess_gain"].mean()
        interval = seed_mean_interval(per_seed)
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "n_questions": int(group["question_id"].nunique()),
                "n_seeds": interval["n_seeds"],
                "home_excess_gain": interval["mean"],
                "seed_std": interval["seed_std"],
                "ci_low": interval["seed_ci_low"],
                "ci_high": interval["seed_ci_high"],
                "seed_values": interval["seed_values"],
                "crossed_question_ci_low": low,
                "crossed_question_ci_high": high,
                "p_one_sided": exact_one_sided_sign_flip(per_seed),
            }
        )
    if not rows:
        return pd.DataFrame(columns=group_columns)
    result = pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)
    result["p_holm"] = holm_adjust(result["p_one_sided"])
    result["holm_significant"] = result["p_holm"] <= 0.05
    return result


def gain_summary(gains: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["subset", "policy_tag", "dataset", "backend", "topk", "metric"]
    per_seed = gains.groupby(group_columns + ["seed"], as_index=False)[
        "gain_over_base"
    ].mean()
    rows = []
    for keys, group in per_seed.groupby(group_columns):
        interval = seed_mean_interval(group.set_index("seed")["gain_over_base"])
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "gain_over_base": interval["mean"],
                "seed_std": interval["seed_std"],
                "n_seeds": interval["n_seeds"],
                "ci_low": interval["seed_ci_low"],
                "ci_high": interval["seed_ci_high"],
                "seed_values": interval["seed_values"],
            }
        )
    return pd.DataFrame(rows).sort_values(group_columns)


def primary_interaction(
    home: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    selected = home[
        (home["subset"] == PRIMARY_SUBSET)
        & (home["topk"] == PRIMARY_TOPK)
        & (home["metric"] == PRIMARY_METRIC)
    ].copy()
    if selected.empty:
        raise RuntimeError(
            "The pre-registered primary interaction is missing: "
            f"subset={PRIMARY_SUBSET}, topk={PRIMARY_TOPK}, metric={PRIMARY_METRIC}"
        )
    stratum_columns = ["policy_tag", "dataset"]
    per_stratum_seed = (
        selected.groupby(["seed", *stratum_columns], as_index=False)["home_excess_gain"]
        .mean()
        .sort_values(["seed", *stratum_columns])
    )
    expected_strata = {
        tuple(values)
        for values in per_stratum_seed[stratum_columns]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    }
    for seed, group in per_stratum_seed.groupby("seed"):
        actual = {
            tuple(values)
            for values in group[stratum_columns].itertuples(index=False, name=None)
        }
        if actual != expected_strata:
            raise RuntimeError(
                "incomplete primary specialist/dataset grid for "
                f"seed {seed}: missing={sorted(expected_strata - actual)}, "
                f"extra={sorted(actual - expected_strata)}"
            )
    per_seed = per_stratum_seed.groupby("seed")["home_excess_gain"].mean()
    interval = seed_mean_interval(per_seed)
    confirmatory = int(interval["n_seeds"]) >= MIN_CONFIRMATORY_SEEDS
    signal = bool(
        float(interval["mean"]) >= threshold and float(interval["seed_ci_low"]) > 0
    )
    if confirmatory:
        decision = "GO" if signal else "NO-GO"
        status = "confirmatory"
    else:
        decision = "EXPLORATORY-SIGNAL" if signal else "EXPLORATORY-NO-SIGNAL"
        status = "exploratory"
    return pd.DataFrame(
        [
            {
                "subset": PRIMARY_SUBSET,
                "topk": PRIMARY_TOPK,
                "metric": PRIMARY_METRIC,
                "estimand": (
                    "equal-weight mean home-backend excess across "
                    "specialist×dataset strata"
                ),
                "n_strata": len(expected_strata),
                "n_seeds": interval["n_seeds"],
                "home_excess_gain": interval["mean"],
                "seed_std": interval["seed_std"],
                "ci_low": interval["seed_ci_low"],
                "ci_high": interval["seed_ci_high"],
                "seed_values": interval["seed_values"],
                "threshold": threshold,
                "inference_status": status,
                "decision": decision,
            }
        ]
    )


def base_backend_gap(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    base = frame[frame["policy_tag"] == "base-qwen"]
    keys = ["subset", "question_id", "dataset", "topk"]
    bm25 = base[base["backend"] == "bm25"][keys + [metric]].rename(
        columns={metric: "bm25"}
    )
    e5 = base[base["backend"] == "e5"][keys + [metric]].rename(columns={metric: "e5"})
    pivot = exact_outer_merge(
        bm25,
        e5,
        on=keys,
        left_name=f"base BM25 {metric} cells",
        right_name=f"base E5 {metric} cells",
    )
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
    home = home_excess(all_gains)
    interactions = home_excess_summary(
        home, args.bootstrap_samples, args.bootstrap_seed
    )
    primary = primary_interaction(home, args.threshold)
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
    primary.to_csv(output_dir / "primary_interaction.csv", index=False)
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

    primary_row = primary.iloc[0]
    lines.extend(
        [
            "## Pre-registered primary decision",
            "",
            (
                f"- {primary_row['metric']}, subset={primary_row['subset']}, "
                f"top-k={int(primary_row['topk'])}: "
                f"home excess={primary_row['home_excess_gain']:.4f}, "
                f"seed-level 95% t CI=[{primary_row['ci_low']:.4f}, "
                f"{primary_row['ci_high']:.4f}], "
                f"seeds={primary_row['seed_values']} "
                f"? **{primary_row['decision']}** "
                f"({primary_row['inference_status']})."
            ),
            "",
            "## Holm-adjusted secondary interactions",
            "",
            markdown_table(interactions),
            "",
        ]
    )
    lines.extend(
        [
            "",
            "## Interpretation rules",
            "",
            "- A large gain over base on both backends is a general RL effect, not specialization.",
            "- A positive home-backend excess isolates retriever-specific improvement after subtracting the base backend gap.",
            "- The single primary endpoint is all-question, top-k=3, observed support-title recall; specialist×dataset strata are equally weighted within each seed.",
            "- Authoritative uncertainty is the Student-t interval over training-seed means. Crossed question intervals are descriptive only.",
            "- Runs with fewer than 8 training seeds are explicitly exploratory and cannot produce a confirmatory GO.",
            "- All policy/dataset/top-k/metric cells are secondary and use exact one-sided seed sign-flip p-values with one Holm correction family.",
            "",
        ]
    )

    report_path = output_dir / "HARD_RQ0_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
