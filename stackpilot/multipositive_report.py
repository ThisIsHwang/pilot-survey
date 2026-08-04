from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from stackpilot.multipositive_common import atomic_write_json, file_sha256, hierarchical_bootstrap, load_config, read_jsonl


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for column in display.select_dtypes(include=["number"]).columns:
        display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.{digits}f}")
    headers = [str(column) for column in display.columns]
    rows = [[str(value) for value in row] for row in display.itertuples(index=False, name=None)]
    widths = [max(len(headers[index]), *(len(row[index]) for row in rows)) for index in range(len(headers))]
    lines = [
        "| " + " | ".join(headers[index].ljust(widths[index]) for index in range(len(headers))) + " |",
        "| " + " | ".join("-" * widths[index] for index in range(len(headers))) + " |",
    ]
    lines.extend("| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(headers))) + " |" for row in rows)
    return "\n".join(lines)


def load_results(plan_root: Path) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    jobs_path = plan_root / "jobs.jsonl"
    manifest_path = plan_root / "manifest.json"
    if not jobs_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(f"Missing plan under {plan_root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["jobs_sha256"] != file_sha256(jobs_path):
        raise RuntimeError("Jobs do not match plan manifest")
    jobs = read_jsonl(jobs_path)
    rows = []
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
            raise RuntimeError(f"Stale output for {job['job_id']}")
        heldout_style = str(job.get("metadata", {}).get("heldout_style", ""))
        for row in read_jsonl(losses_path):
            values = {"baseline_nll": float(row["baseline_nll"]), "adapted_nll": float(row["adapted_nll"]), "heldout_gain": float(row["heldout_gain"])}
            if not all(math.isfinite(value) for value in values.values()):
                raise RuntimeError(f"Non-finite loss in {job['job_id']}")
            rows.append({**row, **values, "experiment_id": str(job["experiment_id"]), "family": str(job["family"]), "direction": str(job["direction"]), "variant": str(job["variant"]), "seed": int(job["seed"]), "heldout_style": heldout_style})
    if missing:
        raise RuntimeError(f"Missing {len(missing)} jobs; first={missing[:5]}")
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No evaluation losses found")
    return jobs, frame


def state_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    output = []
    for (direction, variant, seed, state_id), group in frame.groupby(["direction", "variant", "seed", "state_id"], sort=True):
        strict = group[group["best_class_member"].astype(int) == 1]
        direct = group[group["direct"].astype(int) == 1]
        factual = group[group["factual"].astype(int) == 1]
        synthetic = group[group["synthetic"].astype(int) == 1]
        external_column = group["external_generator"] if "external_generator" in group else pd.Series(0, index=group.index)
        external = group[external_column.astype(int) == 1]
        heldout_style = str(group.iloc[0].get("heldout_style", ""))
        heldout = group[group["style"].astype(str) == heldout_style] if heldout_style else group.iloc[0:0]
        if len(strict) < 2:
            raise RuntimeError(f"State {state_id} has fewer than two strict class targets")
        output.append({
            "direction": str(direction),
            "variant": str(variant),
            "experiment_id": str(group.iloc[0]["experiment_id"]),
            "family": str(group.iloc[0]["family"]),
            "seed": int(seed),
            "state_id": str(state_id),
            "question_id": str(group.iloc[0]["question_id"]),
            "dataset": str(group.iloc[0]["dataset"]),
            "backend": str(group.iloc[0]["backend"]),
            "heldout_style": heldout_style,
            "class_mean_gain": float(strict["heldout_gain"].mean()),
            "class_worst_gain": float(strict["heldout_gain"].min()),
            "class_worst_final_nll": float(strict["adapted_nll"].max()),
            "dispersion_reduction": float(strict["baseline_nll"].std(ddof=0) - strict["adapted_nll"].std(ddof=0)),
            "positive_member_rate": float((strict["heldout_gain"] > 0.0).mean()),
            "all_direct_mean_gain": float(direct["heldout_gain"].mean()) if len(direct) else float("nan"),
            "factual_gain": float(factual["heldout_gain"].mean()) if len(factual) else float("nan"),
            "synthetic_gain": float(synthetic["heldout_gain"].mean()) if len(synthetic) else float("nan"),
            "heldout_style_gain": float(heldout["heldout_gain"].mean()) if len(heldout) else float("nan"),
            "external_gain": float(external["heldout_gain"].mean()) if len(external) else float("nan"),
        })
    return pd.DataFrame(output)


def canonical_method(variant: str) -> str:
    return variant.split("--holdout-", 1)[0]


def paired_rows(metrics: pd.DataFrame, *, left: str, right: str, metric: str, reverse: bool = False, style_fold: bool = False) -> pd.DataFrame:
    frame = metrics.copy()
    frame["method"] = frame["variant"].map(canonical_method)
    if style_fold:
        frame = frame[frame["family"] == "style-heldout"]
        index = ["direction", "seed", "state_id", "question_id", "heldout_style"]
    else:
        frame = frame[frame["family"] != "style-heldout"]
        index = ["direction", "seed", "state_id", "question_id"]
    pivot = frame.pivot_table(index=index, columns="method", values=metric, aggfunc="first")
    if left not in pivot or right not in pivot:
        return pd.DataFrame()
    result = pivot[[left, right]].dropna().reset_index()
    result["effect"] = result[right] - result[left] if reverse else result[left] - result[right]
    result["metric"] = metric
    result["contrast"] = f"{right}-minus-{left}" if reverse else f"{left}-minus-{right}"
    return result


def summarize_variant_means(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (direction, variant), group in metrics.groupby(["direction", "variant"], sort=True):
        seed_means = group.groupby("seed")["class_mean_gain"].mean()
        rows.append({
            "direction": direction,
            "variant": variant,
            "class_mean_gain": float(group["class_mean_gain"].mean()),
            "class_worst_final_nll": float(group["class_worst_final_nll"].mean()),
            "dispersion_reduction": float(group["dispersion_reduction"].mean()),
            "heldout_style_gain": float(group["heldout_style_gain"].mean()),
            "external_gain": float(group["external_gain"].mean()),
            "seed_std": float(seed_means.std(ddof=1)) if len(seed_means) > 1 else 0.0,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report multi-positive generalization experiments.")
    parser.add_argument("--config", default="configs/multipositive_generalization.yaml")
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), default="pilot")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    profile_cfg = cfg["profiles"][args.profile]
    work_root = Path(cfg["work_dir"]).resolve()
    plan_root = work_root / "plans" / args.profile
    output_dir = Path(args.output_dir or work_root / "reports" / args.profile).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _jobs, losses = load_results(plan_root)
    losses.to_csv(output_dir / "target_losses.csv", index=False)
    metrics = state_metrics(losses)
    metrics.to_csv(output_dir / "state_metrics.csv", index=False)
    means = summarize_variant_means(metrics)
    means.to_csv(output_dir / "variant_summary.csv", index=False)

    contrast_specs = [
        ("H1_style_heldout_consistency", "all-direct-consistency", "all-direct-uniform", "heldout_style_gain", False, True),
        ("H2_style_heldout_strict_vs_direct", "strict-consistency", "all-direct-consistency", "heldout_style_gain", False, True),
        ("H3_all_direct_consistency", "all-direct-consistency", "all-direct-uniform", "dispersion_reduction", False, False),
        ("H3_random_consistency", "random-consistency", "random-uniform", "dispersion_reduction", False, False),
        ("H3_diversity_consistency", "diversity-consistency", "diversity-uniform", "dispersion_reduction", False, False),
        ("H3_strict_consistency", "strict-consistency", "strict-uniform", "dispersion_reduction", False, False),
        ("H4_strict_vs_direct_consistency", "strict-consistency", "all-direct-consistency", "class_mean_gain", False, False),
        ("H5_setmass_mean", "all-direct-setmass", "all-direct-uniform", "class_mean_gain", False, False),
        ("H6_setmass_consistency_worst", "all-direct-setmass-consistency", "all-direct-consistency", "class_worst_final_nll", True, False),
        ("H7_hardmax_worst_final", "all-direct-hardmax", "all-direct-uniform", "class_worst_final_nll", True, False),
        ("H8_external_generator", "all-direct-consistency", "all-direct-uniform", "external_gain", False, False),
    ]
    summary_rows = []
    raw_rows = []
    samples = int(profile_cfg["bootstrap_samples"])
    for index, (hypothesis, left, right, metric, reverse, style_fold) in enumerate(contrast_specs):
        frame = paired_rows(metrics, left=left, right=right, metric=metric, reverse=reverse, style_fold=style_fold)
        if frame.empty:
            summary_rows.append({"hypothesis": hypothesis, "estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_rows": 0.0})
            continue
        frame["hypothesis"] = hypothesis
        raw_rows.append(frame)
        result = hierarchical_bootstrap(frame.to_dict("records"), value_key="effect", samples=samples, random_seed=18000 + index)
        summary_rows.append({"hypothesis": hypothesis, **result})
    contrast_rows = pd.concat(raw_rows, ignore_index=True) if raw_rows else pd.DataFrame()
    contrast_rows.to_csv(output_dir / "contrast_rows.csv", index=False)
    summaries = pd.DataFrame(summary_rows)
    summaries.to_csv(output_dir / "hypotheses.csv", index=False)

    gates = cfg["hypothesis_gates"]
    by_name = {str(row["hypothesis"]): row for _, row in summaries.iterrows()}

    def passed(name: str, threshold: float) -> bool:
        row = by_name.get(name)
        return bool(row is not None and math.isfinite(float(row["estimate"])) and float(row["estimate"]) >= threshold and float(row["ci_low"]) > 0.0)

    decisions = {
        "style_heldout_consistency": passed("H1_style_heldout_consistency", float(gates["style_heldout_consistency"]["minimum_gain"])),
        "consistency_beyond_equivalence": bool(by_name.get("H4_strict_vs_direct_consistency") is not None and math.isfinite(float(by_name["H4_strict_vs_direct_consistency"]["ci_low"])) and float(by_name["H4_strict_vs_direct_consistency"]["ci_low"]) >= float(gates["consistency_beyond_equivalence"]["minimum_direct_minus_strict_margin"])),
        "generic_consistency_dispersion": all(passed(name, float(gates["generic_consistency_dispersion"]["minimum_reduction"])) for name in ("H3_all_direct_consistency", "H3_random_consistency", "H3_diversity_consistency")),
        "setmass_mean_gain": passed("H5_setmass_mean", float(gates["setmass_mean_gain"]["minimum_gain"])),
        "setmass_consistency_worst_final": passed("H6_setmass_consistency_worst", float(gates["setmass_consistency_worst_final"]["minimum_improvement"])),
        "independent_generator_gain": passed("H8_external_generator", float(gates["independent_generator_gain"]["minimum_gain"])),
    }
    atomic_write_json(output_dir / "decision.json", {"schema": 1, "suite_id": cfg["suite_id"], "profile": args.profile, "decisions": decisions})

    report = [
        "# EXP-024–026 Multi-positive generalization report",
        "",
        f"Profile: `{args.profile}`. This suite tests whether the large multi-query NLL gain survives held-out query styles, whether consistency requires functional equivalence, and which set objective best balances coverage and worst-member robustness.",
        "",
        "## Variant means",
        "",
        markdown_table(means),
        "",
        "## Hypothesis contrasts",
        "",
        markdown_table(summaries),
        "",
        "## Decisions",
        "",
    ]
    report.extend(f"- {name}: **{'PASS' if value else 'FAIL'}**" for name, value in decisions.items())
    report.extend(["", "The style-heldout and external-generator contrasts are the primary safeguards against synthetic query-distribution leakage. A consistency result that also appears for random and diversity-matched pairs is a generic multi-positive regularization effect rather than evidence for functional equivalence.", "", "NLL results remain diagnostic until EXP-027 executes generated queries against held-out retrievers under matched one-call and four-call budgets.", ""])
    (output_dir / "EXP024_026_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
