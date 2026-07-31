from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stackpilot.query_equivalence_common import (
    atomic_write_json,
    file_sha256,
    load_config,
    read_jsonl,
)

VARIANTS = ("first-exposure", "random-member", "equivalence-pool", "all-direct-pool")


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
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    lines = [
        "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |",
        "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |",
    ]
    lines.extend(
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for row in rows
    )
    return "\n".join(lines)


def _finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"Non-finite {label}: {value!r}")
    return number


def load_job_results(plan_root: Path) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    jobs_path = plan_root / "jobs.jsonl"
    manifest_path = plan_root / "manifest.json"
    if not jobs_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(f"Missing EXP-015 plan under {plan_root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("jobs_sha256") != file_sha256(jobs_path):
        raise RuntimeError("EXP-015 jobs do not match the plan manifest")
    jobs = read_jsonl(jobs_path)
    rows: list[dict[str, Any]] = []
    missing = []
    for job in jobs:
        output_dir = Path(job["output_dir"])
        metrics_path = output_dir / "metrics.json"
        losses_path = output_dir / "eval_losses.jsonl"
        if not metrics_path.is_file() or not losses_path.is_file():
            missing.append(job["job_id"])
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("job_signature") != job.get("job_signature"):
            raise RuntimeError(f"Stale metrics for {job['job_id']}")
        for name in ("baseline_nll", "adapted_nll", "heldout_gain"):
            _finite(metrics[name], f"{job['job_id']}.{name}")
        for row in read_jsonl(losses_path):
            rows.append(
                {
                    **row,
                    "direction": str(job["direction"]),
                    "variant": str(job["variant"]),
                    "seed": int(job["seed"]),
                    "baseline_nll": _finite(row["baseline_nll"], "baseline_nll"),
                    "adapted_nll": _finite(row["adapted_nll"], "adapted_nll"),
                    "heldout_gain": _finite(row["heldout_gain"], "heldout_gain"),
                }
            )
    if missing:
        raise RuntimeError(
            f"EXP-015 is missing {len(missing)} jobs; first missing: {missing[:5]}"
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("EXP-015 has no per-target evaluation losses")
    return jobs, frame


def state_metrics(frame: pd.DataFrame, baseline_tolerance: float) -> pd.DataFrame:
    required = set(VARIANTS)
    output = []
    for (direction, seed, state_id), group in frame.groupby(
        ["direction", "seed", "state_id"], sort=True
    ):
        variants = set(group["variant"].astype(str))
        if variants != required:
            raise RuntimeError(
                f"Incomplete variant grid for {direction}, seed={seed}, state={state_id}: {sorted(variants)}"
            )
        baseline_by_target = group.pivot_table(
            index="target_id", columns="variant", values="baseline_nll", aggfunc="first"
        )
        if baseline_by_target.isna().any().any():
            raise RuntimeError(f"Missing target baseline for state {state_id}")
        baseline_range = float((baseline_by_target.max(axis=1) - baseline_by_target.min(axis=1)).max())
        if baseline_range > baseline_tolerance:
            raise RuntimeError(
                f"Base NLL drift across variants for {state_id}: {baseline_range:.6g}"
            )
        for variant, variant_rows in group.groupby("variant", sort=True):
            class_rows = variant_rows[variant_rows["best_class_member"].astype(int) == 1]
            if len(class_rows) < 2:
                raise RuntimeError(f"Held-out state {state_id} has fewer than two class members")
            offclass = variant_rows[
                (variant_rows["direct"].astype(int) == 1)
                & (variant_rows["best_class_member"].astype(int) == 0)
            ]
            class_mean = float(class_rows["heldout_gain"].mean())
            offclass_mean = float(offclass["heldout_gain"].mean()) if len(offclass) else float("nan")
            output.append(
                {
                    "direction": str(direction),
                    "seed": int(seed),
                    "state_id": str(state_id),
                    "dataset": str(class_rows.iloc[0]["dataset"]),
                    "backend": str(class_rows.iloc[0]["backend"]),
                    "variant": str(variant),
                    "class_size": len(class_rows),
                    "class_mean_gain": class_mean,
                    "class_worst_gain": float(class_rows["heldout_gain"].min()),
                    "class_positive_coverage": float((class_rows["heldout_gain"] > 0.0).mean()),
                    "class_baseline_std": float(class_rows["baseline_nll"].std(ddof=0)),
                    "class_adapted_std": float(class_rows["adapted_nll"].std(ddof=0)),
                    "class_dispersion_reduction": float(
                        class_rows["baseline_nll"].std(ddof=0)
                        - class_rows["adapted_nll"].std(ddof=0)
                    ),
                    "offclass_mean_gain": offclass_mean,
                    "class_offclass_margin": class_mean - offclass_mean if math.isfinite(offclass_mean) else float("nan"),
                }
            )
    return pd.DataFrame(output)


def paired_effect_rows(
    metrics: pd.DataFrame,
    *,
    left: str,
    right: str,
    value: str,
) -> pd.DataFrame:
    pivot = metrics.pivot_table(
        index=["direction", "seed", "state_id"],
        columns="variant",
        values=value,
        aggfunc="first",
    )
    if left not in pivot or right not in pivot:
        raise RuntimeError(f"Missing variants for contrast {left} - {right}")
    result = pivot[[left, right]].dropna().reset_index()
    result["effect"] = result[left] - result[right]
    result["metric"] = value
    result["contrast"] = f"{left}-minus-{right}"
    return result


def hierarchical_bootstrap(
    frame: pd.DataFrame,
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    if frame.empty:
        raise RuntimeError("Cannot bootstrap an empty contrast")
    by_seed = {
        int(seed_value): group["effect"].to_numpy(dtype=np.float64)
        for seed_value, group in frame.groupby("seed")
    }
    if any(not np.isfinite(values).all() for values in by_seed.values()):
        raise RuntimeError("Non-finite contrast reached bootstrap")
    seeds = sorted(by_seed)
    estimate = float(np.mean([values.mean() for values in by_seed.values()]))
    generator = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled_seeds = generator.choice(seeds, size=len(seeds), replace=True)
        seed_means = []
        for sampled_seed in sampled_seeds:
            values = by_seed[int(sampled_seed)]
            sampled_values = generator.choice(values, size=len(values), replace=True)
            seed_means.append(float(sampled_values.mean()))
        draws[index] = float(np.mean(seed_means))
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "estimate": estimate,
        "ci_low": float(low),
        "ci_high": float(high),
        "n_seeds": float(len(seeds)),
        "n_rows": float(len(frame)),
    }


def summarize_variants(metrics: pd.DataFrame) -> pd.DataFrame:
    per_seed = (
        metrics.groupby(["direction", "variant", "seed"], as_index=False)[
            [
                "class_mean_gain",
                "class_worst_gain",
                "class_positive_coverage",
                "class_dispersion_reduction",
                "class_offclass_margin",
            ]
        ]
        .mean()
    )
    summary = (
        per_seed.groupby(["direction", "variant"])
        .agg(
            class_mean_gain=("class_mean_gain", "mean"),
            class_mean_seed_std=("class_mean_gain", "std"),
            class_worst_gain=("class_worst_gain", "mean"),
            class_positive_coverage=("class_positive_coverage", "mean"),
            class_dispersion_reduction=("class_dispersion_reduction", "mean"),
            class_offclass_margin=("class_offclass_margin", "mean"),
        )
        .reset_index()
    )
    summary["class_mean_seed_std"] = summary["class_mean_seed_std"].fillna(0.0)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Report EXP-015 equivalence-aware credit diagnostic.")
    parser.add_argument("--config", default="configs/query_equivalence.yaml")
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), default="pilot")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    profile_cfg = cfg["profiles"][args.profile]
    work_root = Path(cfg["work_dir"]).resolve()
    prepared_root = work_root / "prepared" / args.profile
    plan_root = work_root / "plans" / args.profile
    output_dir = Path(args.output_dir or work_root / "reports" / args.profile).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_manifest = json.loads((prepared_root / "manifest.json").read_text(encoding="utf-8"))
    audit = pd.DataFrame(read_jsonl(prepared_root / "state_audit.jsonl"))
    _jobs, losses = load_job_results(plan_root)
    metrics = state_metrics(losses, float(cfg["analysis"]["baseline_tolerance"]))
    metrics.to_csv(output_dir / "state_metrics.csv", index=False)
    losses.to_csv(output_dir / "target_losses.csv", index=False)
    variant_summary = summarize_variants(metrics)
    variant_summary.to_csv(output_dir / "variant_summary.csv", index=False)

    contrast_specs = [
        ("equivalence-pool", "first-exposure", "class_mean_gain"),
        ("equivalence-pool", "random-member", "class_worst_gain"),
        ("equivalence-pool", "all-direct-pool", "class_mean_gain"),
        ("equivalence-pool", "first-exposure", "class_dispersion_reduction"),
    ]
    summaries = []
    contrast_rows = []
    for contrast_index, (left, right, value) in enumerate(contrast_specs):
        frame = paired_effect_rows(metrics, left=left, right=right, value=value)
        contrast_rows.append(frame)
        for direction_index, (direction, group) in enumerate(frame.groupby("direction", sort=True)):
            summaries.append(
                {
                    "scope": str(direction),
                    "contrast": f"{left}-minus-{right}",
                    "metric": value,
                    **hierarchical_bootstrap(
                        group,
                        samples=int(profile_cfg["bootstrap_samples"]),
                        seed=15000 + contrast_index * 20 + direction_index,
                    ),
                }
            )
        summaries.append(
            {
                "scope": "combined",
                "contrast": f"{left}-minus-{right}",
                "metric": value,
                **hierarchical_bootstrap(
                    frame,
                    samples=int(profile_cfg["bootstrap_samples"]),
                    seed=15100 + contrast_index,
                ),
            }
        )
    contrast_frame = pd.concat(contrast_rows, ignore_index=True)
    contrast_frame.to_csv(output_dir / "contrast_rows.csv", index=False)
    contrast_summary = pd.DataFrame(summaries)
    contrast_summary.to_csv(output_dir / "contrasts.csv", index=False)

    audit_direct = audit[audit["direct_candidate_count"].astype(int) > 0]
    phenomenon_rate = float(prepared_manifest["nontrivial_equivalence_rate_among_direct"])
    factual_rate = float(prepared_manifest["factual_replaceability_rate_among_direct"])
    class_sizes = audit_direct.loc[audit_direct["best_class_size"] >= 2, "best_class_size"]
    audit_summary = pd.DataFrame(
        [
            {
                "audit_states": len(audit),
                "direct_states": len(audit_direct),
                "eligible_states": int(prepared_manifest["eligible_states"]),
                "nontrivial_equivalence_rate_among_direct": phenomenon_rate,
                "factual_replaceability_rate_among_direct": factual_rate,
                "mean_multiquery_class_size": float(class_sizes.mean()) if len(class_sizes) else 0.0,
                "mean_best_class_style_count": float(
                    audit_direct.loc[audit_direct["best_class_size"] >= 2, "best_class_style_count"].mean()
                ) if int((audit_direct["best_class_size"] >= 2).sum()) else 0.0,
            }
        ]
    )
    audit_summary.to_csv(output_dir / "audit_summary.csv", index=False)

    def combined(contrast: str, metric: str) -> pd.Series:
        subset = contrast_summary[
            (contrast_summary["scope"] == "combined")
            & (contrast_summary["contrast"] == contrast)
            & (contrast_summary["metric"] == metric)
        ]
        if len(subset) != 1:
            raise RuntimeError(f"Missing combined contrast {contrast}/{metric}")
        return subset.iloc[0]

    eq_first = combined("equivalence-pool-minus-first-exposure", "class_mean_gain")
    eq_random = combined("equivalence-pool-minus-random-member", "class_worst_gain")
    eq_all = combined("equivalence-pool-minus-all-direct-pool", "class_mean_gain")
    direction_eq_first = contrast_summary[
        (contrast_summary["scope"] != "combined")
        & (contrast_summary["contrast"] == "equivalence-pool-minus-first-exposure")
        & (contrast_summary["metric"] == "class_mean_gain")
    ]
    seed_summary = variant_summary.groupby("variant", as_index=False)["class_mean_seed_std"].mean()
    seed_std = dict(zip(seed_summary["variant"], seed_summary["class_mean_seed_std"], strict=True))
    gate = cfg["gate"]
    decision = bool(
        phenomenon_rate >= float(gate["minimum_nontrivial_equivalence_rate"])
        and float(eq_first["estimate"]) >= float(gate["minimum_mean_class_gain_advantage"])
        and float(eq_first["ci_low"]) > 0.0
        and float(eq_random["estimate"]) >= float(gate["minimum_worst_class_gain_advantage"])
        and float(eq_random["ci_low"]) > 0.0
        and float(eq_all["estimate"]) >= float(gate["minimum_all_direct_margin"])
        and bool((direction_eq_first["estimate"] >= 0.0).all())
        and float(seed_std.get("equivalence-pool", float("inf")))
        <= float(seed_std.get("random-member", float("-inf")))
    )
    decision_payload = {
        "schema": 1,
        "experiment_id": "EXP-015",
        "profile": args.profile,
        "go": decision,
        "phenomenon_rate": phenomenon_rate,
        "factual_replaceability_rate": factual_rate,
        "equivalence_pool_vs_first_exposure": eq_first.to_dict(),
        "equivalence_pool_vs_random_member_worst": eq_random.to_dict(),
        "equivalence_pool_vs_all_direct": eq_all.to_dict(),
        "seed_std": {key: float(value) for key, value in seed_std.items()},
    }
    atomic_write_json(output_dir / "decision.json", decision_payload)

    report = [
        "# EXP-015 Query-equivalence credit report",
        "",
        f"Profile: `{args.profile}`. The experiment tests whether functionally interchangeable evidence queries should share one state-normalized credit unit.",
        "",
        "## Equivalence audit",
        "",
        markdown_table(audit_summary),
        "",
        "A nontrivial equivalence class contains at least two distinct query wordings that expose the same immediate and final gold-support sets and have the same answer-EM outcome. The factual query must belong to the best class for the controlled LoRA matrix.",
        "",
        "## Variant means",
        "",
        markdown_table(variant_summary),
        "",
        "## Paired contrasts",
        "",
        markdown_table(contrast_summary),
        "",
        "Primary comparison:",
        "",
        "```text",
        "equivalence-pool class mean gain - first-exposure class mean gain",
        "```",
        "",
        f"Decision: **{'GO' if decision else 'NO-GO'}**.",
        "",
        "GO requires a prevalent nontrivial equivalence phenomenon, positive class-wide improvement over exclusive factual credit, better worst-member improvement than arbitrary representative credit, no disadvantage versus pooling all direct queries, non-negative effects in both transfer directions, and no greater seed variability than random representative selection.",
        "",
        "This is a grouped-query NLL micro-update diagnostic. A GO must be confirmed by generating queries and executing them against held-out retrievers under a matched search-call budget.",
        "",
    ]
    (output_dir / "EXP015_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
