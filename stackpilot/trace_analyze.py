from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stackpilot.trace_common import (
    atomic_write_json,
    hierarchical_bootstrap_difference,
    load_trace_config,
    read_jsonl,
    ridge_fit,
    r_squared,
    standardize_matrix,
)


A_PREDICTORS = [
    "portable_recovery_proxy",
    "turn1_recall",
    "reward_variance",
    "search_count",
    "question_difficulty",
]


def _load_completed_jobs(jobs_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    jobs = read_jsonl(jobs_path)
    metrics = []
    missing = []
    for job in jobs:
        metrics_path = Path(job["output_dir"]) / "metrics.json"
        if not metrics_path.is_file():
            missing.append(job["job_id"])
            continue
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if payload.get("job_signature") != job.get("job_signature"):
            raise RuntimeError(f"Stale TRACE metrics for {job['job_id']}: {metrics_path}")
        metrics.append(payload)
    if missing:
        raise RuntimeError(
            f"TRACE analysis is missing {len(missing)} jobs; first missing: {missing[:5]}"
        )
    return jobs, metrics


def _job_summary(metrics: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        metadata = dict(metric.get("metadata", {}))
        rows.append(
            {
                "experiment_id": metric["experiment_id"],
                "job_id": metric["job_id"],
                "variant": metric["variant"],
                "seed": int(metric["seed"]),
                "baseline_nll": float(metric["baseline_nll"]),
                "adapted_nll": float(metric["adapted_nll"]),
                "heldout_gain": float(metric["heldout_gain"]),
                "train_examples": int(metric["train_examples"]),
                "eval_examples": int(metric["eval_examples"]),
                **metadata,
            }
        )
    return pd.DataFrame(rows)


def _design(frame: pd.DataFrame, predictors: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    numeric = frame[predictors].to_numpy(dtype=np.float64)
    standardized, _, _ = standardize_matrix(numeric)
    source_e5 = (frame["source_backend"].astype(str) == "e5").astype(float).to_numpy()
    x = np.column_stack([np.ones(len(frame)), standardized, source_e5])
    names = ["intercept", *predictors, "source_is_e5"]
    y = frame["heldout_gain"].to_numpy(dtype=np.float64)
    return x, y, names


def analyze_a(
    frame: pd.DataFrame,
    *,
    ridge: float,
    bootstrap_samples: int,
    random_seed: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    subset = frame[frame["experiment_id"] == "EXP-009"].copy()
    if len(subset) < len(A_PREDICTORS) + 3:
        raise RuntimeError("Condition A has too few completed cells for regression")
    x, y, names = _design(subset, A_PREDICTORS)
    coefficients = ridge_fit(x, y, ridge)
    prediction = x @ coefficients
    full_r2 = r_squared(y, prediction)
    reduced_predictors = [name for name in A_PREDICTORS if name != "portable_recovery_proxy"]
    x_reduced, _, _ = _design(subset, reduced_predictors)
    reduced_coef = ridge_fit(x_reduced, y, ridge)
    reduced_r2 = r_squared(y, x_reduced @ reduced_coef)
    partial_r2 = max(0.0, full_r2 - reduced_r2)

    # Cell-level cluster bootstrap keeps micro-update seeds together.
    subset["cell_id"] = subset["variant"].astype(str)
    cells = sorted(subset["cell_id"].unique())
    rng = np.random.default_rng(random_seed)
    coefficient_draws = np.empty(bootstrap_samples, dtype=np.float64)
    partial_draws = np.empty(bootstrap_samples, dtype=np.float64)
    for draw_index in range(bootstrap_samples):
        sampled_cells = rng.choice(cells, size=len(cells), replace=True)
        sampled_parts = []
        for new_index, cell in enumerate(sampled_cells):
            part = subset[subset["cell_id"] == cell].copy()
            part["bootstrap_cell"] = new_index
            sampled_parts.append(part)
        sample = pd.concat(sampled_parts, ignore_index=True)
        sample_x, sample_y, sample_names = _design(sample, A_PREDICTORS)
        sample_coef = ridge_fit(sample_x, sample_y, ridge)
        coefficient_draws[draw_index] = sample_coef[
            sample_names.index("portable_recovery_proxy")
        ]
        sample_reduced, _, _ = _design(sample, reduced_predictors)
        full = r_squared(sample_y, sample_x @ sample_coef)
        reduced = r_squared(
            sample_y,
            sample_reduced @ ridge_fit(sample_reduced, sample_y, ridge),
        )
        partial_draws[draw_index] = max(0.0, full - reduced)

    coefficient = float(coefficients[names.index("portable_recovery_proxy")])
    low, high = np.quantile(coefficient_draws, [0.025, 0.975])
    partial_low, partial_high = np.quantile(partial_draws, [0.025, 0.975])
    coefficient_rows = pd.DataFrame(
        {"term": names, "standardized_coefficient": coefficients}
    )
    summary = {
        "portable_recovery_coefficient": coefficient,
        "coefficient_ci_low": float(low),
        "coefficient_ci_high": float(high),
        "full_r2": full_r2,
        "reduced_r2": reduced_r2,
        "partial_r2": partial_r2,
        "partial_r2_ci_low": float(partial_low),
        "partial_r2_ci_high": float(partial_high),
        "n_jobs": float(len(subset)),
        "n_cells": float(len(cells)),
    }
    return coefficient_rows, summary


def _loss_rows(jobs: list[dict[str, Any]], experiment_id: str) -> list[dict[str, Any]]:
    output = []
    for job in jobs:
        if job["experiment_id"] != experiment_id:
            continue
        path = Path(job["output_dir"]) / "eval_losses.jsonl"
        for row in read_jsonl(path):
            metadata = job["metadata"]
            output.append(
                {
                    **row,
                    "seed": int(job["seed"]),
                    "curriculum": metadata.get("curriculum"),
                    "source_backend": metadata.get("source_backend"),
                    "target_backend": metadata.get("target_backend"),
                    "direction": f"{metadata.get('source_backend')}-to-{metadata.get('target_backend')}",
                    "paired_example_id": f"{metadata.get('target_backend')}::{row['example_id']}",
                }
            )
    return output


def analyze_contrast(
    jobs: list[dict[str, Any]],
    *,
    experiment_id: str,
    positive_variant: str,
    negative_variant: str,
    bootstrap_samples: int,
    random_seed: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = _loss_rows(jobs, experiment_id)
    if not rows:
        raise RuntimeError(f"No per-example losses found for {experiment_id}")
    frame = pd.DataFrame(rows)
    direction_rows = []
    for direction, group in frame.groupby("direction"):
        result = hierarchical_bootstrap_difference(
            group.to_dict("records"),
            value_key="heldout_gain",
            variant_key="curriculum",
            positive_variant=positive_variant,
            negative_variant=negative_variant,
            seed_key="seed",
            example_key="paired_example_id",
            samples=bootstrap_samples,
            random_seed=random_seed + len(direction_rows),
        )
        direction_rows.append({"direction": direction, **result})
    combined = hierarchical_bootstrap_difference(
        frame.to_dict("records"),
        value_key="heldout_gain",
        variant_key="curriculum",
        positive_variant=positive_variant,
        negative_variant=negative_variant,
        seed_key="seed",
        example_key="paired_example_id",
        samples=bootstrap_samples,
        random_seed=random_seed + 100,
    )
    per_seed = (
        frame.groupby(["curriculum", "seed"], as_index=False)["heldout_gain"]
        .mean()
        .groupby("curriculum")["heldout_gain"]
        .agg(["mean", "std"])
        .reset_index()
    )
    std_lookup = {
        str(row.curriculum): float(row.std) if np.isfinite(row.std) else 0.0
        for row in per_seed.itertuples(index=False)
    }
    summary = {
        **combined,
        "positive_seed_std": std_lookup.get(positive_variant, 0.0),
        "negative_seed_std": std_lookup.get(negative_variant, 0.0),
    }
    return pd.DataFrame(direction_rows), summary


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze TRACE project go conditions A/B/C.")
    parser.add_argument("--config", default="configs/trace_go.yaml")
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), default="pilot")
    parser.add_argument("--plan-root", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_trace_config(args.config)
    profile = cfg["profiles"][args.profile]
    work_root = Path(cfg["work_dir"]).resolve()
    plan_root = Path(args.plan_root or work_root / "plans" / args.profile)
    output_dir = Path(args.output_dir or work_root / "reports" / args.profile)
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs, metrics = _load_completed_jobs(plan_root / "jobs.jsonl")
    summary_frame = _job_summary(metrics)
    summary_frame.to_csv(output_dir / "job_summary.csv", index=False)

    a_coefficients, a_summary = analyze_a(
        summary_frame,
        ridge=float(cfg["condition_a"]["ridge"]),
        bootstrap_samples=int(profile["bootstrap_samples"]),
        random_seed=12009,
    )
    b_directions, b_summary = analyze_contrast(
        jobs,
        experiment_id="EXP-010",
        positive_variant="short-recovered",
        negative_variant="deep-unrecovered",
        bootstrap_samples=int(profile["bootstrap_samples"]),
        random_seed=12010,
    )
    c_directions, c_summary = analyze_contrast(
        jobs,
        experiment_id="EXP-011",
        positive_variant="paired",
        negative_variant="unpaired",
        bootstrap_samples=int(profile["bootstrap_samples"]),
        random_seed=12011,
    )

    a_coefficients.to_csv(output_dir / "condition_a_coefficients.csv", index=False)
    b_directions.to_csv(output_dir / "condition_b_directions.csv", index=False)
    c_directions.to_csv(output_dir / "condition_c_directions.csv", index=False)
    atomic_write_json(output_dir / "condition_a.json", a_summary)
    atomic_write_json(output_dir / "condition_b.json", b_summary)
    atomic_write_json(output_dir / "condition_c.json", c_summary)

    a_pass = bool(
        a_summary["portable_recovery_coefficient"]
        >= float(cfg["condition_a"]["minimum_standardized_coefficient"])
        and a_summary["coefficient_ci_low"] > 0.0
        and a_summary["partial_r2"] >= float(cfg["condition_a"]["minimum_partial_r2"])
    )
    b_pass = bool(
        b_summary["estimate"] >= float(cfg["condition_b"]["minimum_nll_gain"])
        and b_summary["ci_low"] > 0.0
    )
    c_pass = bool(
        c_summary["estimate"] >= float(cfg["condition_c"]["minimum_nll_gain"])
        and c_summary["ci_low"] > 0.0
        and (
            not bool(cfg["condition_c"]["require_non_increasing_seed_std"])
            or c_summary["positive_seed_std"] <= c_summary["negative_seed_std"]
        )
    )
    decision = {
        "condition_a": a_pass,
        "condition_b": b_pass,
        "condition_c": c_pass,
        "all_conditions": a_pass and b_pass and c_pass,
    }
    atomic_write_json(output_dir / "decision.json", decision)

    report = [
        "# TRACE project go/no-go report",
        "",
        f"Profile: `{args.profile}`. Primary diagnostic endpoint: held-out evidence-gaining query NLL improvement after an equal-budget Qwen LoRA micro-update.",
        "",
        "This is a project-selection diagnostic. Passing all three conditions justifies implementing full TRACE/GRPO; it is not itself the final paper result.",
        "",
        "## Condition A — recoverability beyond generic difficulty",
        "",
        f"Portable-recovery standardized coefficient: **{a_summary['portable_recovery_coefficient']:.4f}** "
        f"(95% cluster bootstrap CI [{a_summary['coefficient_ci_low']:.4f}, {a_summary['coefficient_ci_high']:.4f}]).",
        f"Incremental R² beyond initial success, reward variance, depth, question difficulty, and source backend: **{a_summary['partial_r2']:.4f}**.",
        f"Decision: **{'GO' if a_pass else 'NO-GO'}**.",
        "",
        markdown_table(a_coefficients),
        "",
        "## Condition B — recovered trajectories versus merely deep failures",
        "",
        f"Held-out NLL-gain contrast (short recovered − deep unrecovered): **{b_summary['estimate']:.4f}** "
        f"(95% hierarchical bootstrap CI [{b_summary['ci_low']:.4f}, {b_summary['ci_high']:.4f}]).",
        f"Decision: **{'GO' if b_pass else 'NO-GO'}**.",
        "",
        markdown_table(b_directions),
        "",
        "## Condition C — same-question paired design versus unpaired selection",
        "",
        f"Held-out NLL-gain contrast (paired − unpaired): **{c_summary['estimate']:.4f}** "
        f"(95% hierarchical bootstrap CI [{c_summary['ci_low']:.4f}, {c_summary['ci_high']:.4f}]).",
        f"Seed standard deviations: paired={c_summary['positive_seed_std']:.4f}, unpaired={c_summary['negative_seed_std']:.4f}.",
        f"Decision: **{'GO' if c_pass else 'NO-GO'}**.",
        "",
        markdown_table(c_directions),
        "",
        "## Overall decision",
        "",
        f"**{'GO: implement full TRACE/GRPO.' if decision['all_conditions'] else 'NO-GO: at least one structural premise is not supported.'}**",
        "",
        "A later confirmatory experiment must replace the query-NLL proxy with interactive held-out retriever evidence gain under matched search-call budgets.",
        "",
    ]
    report_path = output_dir / "TRACE_GO_REPORT.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
