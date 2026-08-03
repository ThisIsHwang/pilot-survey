from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stackpilot.query_attribution_common import atomic_write_json, file_sha256, hierarchical_bootstrap, load_config, read_jsonl


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for column in display.select_dtypes(include=["number"]).columns:
        display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.{digits}f}")
    headers = [str(column) for column in display.columns]
    rows = [[str(value) for value in row] for row in display.itertuples(index=False, name=None)]
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    lines = ["| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |", "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"]
    lines.extend("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |" for row in rows)
    return "\n".join(lines)


def finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"Non-finite {label}: {value!r}")
    return number


def load_results(plan_root: Path) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    jobs_path = plan_root / "jobs.jsonl"; manifest_path = plan_root / "manifest.json"
    if not jobs_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(f"Missing plan under {plan_root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["jobs_sha256"] != file_sha256(jobs_path):
        raise RuntimeError("Jobs file does not match plan manifest")
    jobs = read_jsonl(jobs_path); losses = []; metrics_rows = []; missing = []
    for job in jobs:
        root = Path(job["output_dir"]); metrics_path = root / "metrics.json"; losses_path = root / "eval_losses.jsonl"
        if not metrics_path.is_file() or not losses_path.is_file():
            missing.append(job["job_id"]); continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("job_signature") != job.get("job_signature"):
            raise RuntimeError(f"Stale output for {job['job_id']}")
        for name in ("baseline_nll", "adapted_nll", "heldout_gain"):
            finite(metrics[name], f"{job['job_id']}.{name}")
        metrics_rows.append(metrics)
        for row in read_jsonl(losses_path):
            for name in ("baseline_nll", "adapted_nll", "heldout_gain"):
                row[name] = finite(row[name], f"{job['job_id']}.{name}")
            losses.append(row)
    if missing:
        raise RuntimeError(f"Missing {len(missing)} jobs; first={missing[:5]}")
    return jobs, pd.DataFrame(losses), pd.DataFrame(metrics_rows)


def validate_baseline_grid(losses: pd.DataFrame, tolerance: float) -> None:
    ranges = losses.groupby(["direction", "seed", "eval_scope", "state_id", "target_id"], sort=True)["baseline_nll"].agg(lambda values: float(values.max() - values.min()))
    if len(ranges) and float(ranges.max()) > tolerance:
        raise RuntimeError(f"Base-model NLL differs across variants for {ranges.idxmax()}: range={float(ranges.max()):.6g}")


def state_metrics(losses: pd.DataFrame, baseline_tolerance: float) -> pd.DataFrame:
    rows = []
    for (direction, variant, seed, eval_scope, state_id), group in losses.groupby(["direction", "variant", "seed", "eval_scope", "state_id"], sort=True):
        class_rows = group[group["best_class_member"].astype(int) == 1]
        if len(class_rows) < 2:
            raise RuntimeError(f"State {state_id} has fewer than two strict class members")
        baseline_range = float(group.groupby("target_id")["baseline_nll"].agg(lambda values: values.max() - values.min()).max())
        if baseline_range > baseline_tolerance:
            raise RuntimeError(f"Baseline NLL mismatch within job grid at {state_id}: {baseline_range}")
        offclass = group[(group["direct"].astype(int) == 1) & (group["best_class_member"].astype(int) == 0)]
        factual = class_rows[class_rows["factual"].astype(int) == 1]
        synthetic = class_rows[class_rows["synthetic"].astype(int) == 1]
        baseline_std = float(class_rows["baseline_nll"].std(ddof=0)); adapted_std = float(class_rows["adapted_nll"].std(ddof=0)); class_mean = float(class_rows["heldout_gain"].mean())
        offclass_mean = float(offclass["heldout_gain"].mean()) if len(offclass) else float("nan")
        rows.append({"direction": str(direction), "variant": str(variant), "seed": int(seed), "eval_scope": str(eval_scope), "state_id": str(state_id), "question_id": str(class_rows.iloc[0]["question_id"]), "dataset": str(class_rows.iloc[0]["dataset"]), "backend": str(class_rows.iloc[0]["backend"]), "class_size": int(class_rows.iloc[0].get("strict_class_size", len(class_rows))), "class_min_jaccard": float(class_rows.iloc[0].get("strict_min_jaccard", 1.0)), "class_mean_gain": class_mean, "class_worst_gain": float(class_rows["heldout_gain"].min()), "class_mean_final_nll": float(class_rows["adapted_nll"].mean()), "class_worst_final_nll": float(class_rows["adapted_nll"].max()), "baseline_class_std": baseline_std, "adapted_class_std": adapted_std, "dispersion_reduction": baseline_std - adapted_std, "positive_member_rate": float((class_rows["heldout_gain"] > 0).mean()), "factual_gain": float(factual["heldout_gain"].mean()) if len(factual) else float("nan"), "synthetic_gain": float(synthetic["heldout_gain"].mean()) if len(synthetic) else float("nan"), "offclass_direct_gain": offclass_mean, "class_offclass_margin": class_mean - offclass_mean if math.isfinite(offclass_mean) else float("nan")})
    return pd.DataFrame(rows)


def variant_summary(frame: pd.DataFrame) -> pd.DataFrame:
    per_seed = frame.groupby(["direction", "variant", "eval_scope", "seed"], as_index=False)[["class_mean_gain", "class_worst_gain", "class_mean_final_nll", "class_worst_final_nll", "dispersion_reduction", "positive_member_rate", "factual_gain", "synthetic_gain", "class_offclass_margin"]].mean()
    return per_seed.groupby(["direction", "variant", "eval_scope"]).agg(class_mean_gain=("class_mean_gain", "mean"), seed_std=("class_mean_gain", "std"), class_worst_gain=("class_worst_gain", "mean"), class_worst_final_nll=("class_worst_final_nll", "mean"), dispersion_reduction=("dispersion_reduction", "mean"), positive_member_rate=("positive_member_rate", "mean"), factual_gain=("factual_gain", "mean"), synthetic_gain=("synthetic_gain", "mean"), class_offclass_margin=("class_offclass_margin", "mean")).reset_index().fillna({"seed_std": 0.0})


def paired_effects(frame: pd.DataFrame, left: str, right: str, metric: str, scope: str = "cross") -> pd.DataFrame:
    subset = frame[frame["eval_scope"] == scope]
    pivot = subset.pivot_table(index=["direction", "seed", "question_id", "state_id"], columns="variant", values=metric, aggfunc="first")
    if left not in pivot or right not in pivot:
        raise RuntimeError(f"Missing variants for {left} - {right} on {metric}")
    output = pivot[[left, right]].dropna().reset_index(); output["effect"] = output[left] - output[right]; output["contrast"] = f"{left}-minus-{right}"; output["metric"] = metric; output["eval_scope"] = scope
    return output


def direction_summaries(rows: pd.DataFrame, samples: int, seed: int) -> list[dict[str, Any]]:
    output = []
    for index, (direction, group) in enumerate(rows.groupby("direction", sort=True)):
        output.append({"scope": str(direction), **hierarchical_bootstrap(group.to_dict("records"), value_key="effect", samples=samples, random_seed=seed + index)})
    output.append({"scope": "combined", **hierarchical_bootstrap(rows.to_dict("records"), value_key="effect", samples=samples, random_seed=seed + 100)})
    return output


def hypothesis_results(cfg: dict[str, Any], frame: pd.DataFrame, samples: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    summaries = []; decisions = {}
    for index, (name, gate) in enumerate(cfg["hypothesis_gates"].items()):
        left, right = map(str, gate["contrast"]); metric = str(gate["metric"])
        effects = paired_effects(frame, left, right, metric, scope="cross")
        result_rows = direction_summaries(effects, samples, 17000 + 200 * index)
        for row in result_rows:
            summaries.append({"hypothesis": name, "contrast": f"{left}-minus-{right}", "metric": metric, **row})
        combined = next(row for row in result_rows if row["scope"] == "combined"); direction_rows = [row for row in result_rows if row["scope"] != "combined"]
        if "equivalence_margin" in gate:
            margin = float(gate["equivalence_margin"]); passed = bool(abs(float(combined["estimate"])) <= margin and float(combined["ci_low"]) >= -margin and float(combined["ci_high"]) <= margin); criterion = f"equivalent within ±{margin}"
        else:
            minimum = float(gate["minimum"]); passed = bool(float(combined["estimate"]) >= minimum and (not bool(gate.get("ci_low_positive", False)) or float(combined["ci_low"]) > 0.0) and all(float(row["estimate"]) >= 0.0 for row in direction_rows)); criterion = f">= {minimum} with positive CI/directions"
        decisions[name] = {"pass": passed, "criterion": criterion, "combined": combined, "directions": direction_rows}
    return pd.DataFrame(summaries), decisions


def subgroup_summary(frame: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    copy = frame.copy(); copy["class_size_group"] = np.where(copy["class_size"] >= int(cfg["analysis"]["class_large_threshold"]), "large", "size-2"); copy["diversity_group"] = np.where(copy["class_min_jaccard"] <= float(cfg["analysis"]["diversity_high_threshold"]), "high-diversity", "low-diversity")
    return copy.groupby(["variant", "eval_scope", "dataset", "class_size_group", "diversity_group"], as_index=False)[["class_mean_gain", "dispersion_reduction", "factual_gain", "synthetic_gain"]].mean()


def portability(frame: pd.DataFrame) -> pd.DataFrame:
    per_seed = frame.groupby(["direction", "variant", "seed", "eval_scope"], as_index=False)["class_mean_gain"].mean()
    pivot = per_seed.pivot_table(index=["direction", "variant", "seed"], columns="eval_scope", values="class_mean_gain").reset_index()
    if "seen" in pivot and "cross" in pivot: pivot["portability_drop"] = pivot["seen"] - pivot["cross"]
    return pivot


def main() -> None:
    parser = argparse.ArgumentParser(description="Report the multi-hypothesis attribution matrix.")
    parser.add_argument("--config", default="configs/query_attribution.yaml"); parser.add_argument("--profile", choices=("smoke", "pilot", "full"), default="pilot")
    args = parser.parse_args(); cfg = load_config(args.config); profile_cfg = cfg["profiles"][args.profile]; root = Path(cfg["work_dir"]).resolve(); report_root = root / "reports" / args.profile; report_root.mkdir(parents=True, exist_ok=True)
    _jobs, losses, job_metrics = load_results(root / "plans" / args.profile); validate_baseline_grid(losses, float(cfg["analysis"]["baseline_tolerance"])); state_frame = state_metrics(losses, float(cfg["analysis"]["baseline_tolerance"])); summary = variant_summary(state_frame); hypotheses, decisions = hypothesis_results(cfg, state_frame, int(profile_cfg["bootstrap_samples"])); subgroups = subgroup_summary(state_frame, cfg); portability_frame = portability(state_frame)
    budget = job_metrics[["experiment_id", "direction", "variant", "seed", "train_groups", "train_targets", "processed_targets", "elapsed_seconds", "peak_allocated_gib"]].copy()
    losses.to_csv(report_root / "target_losses.csv", index=False); state_frame.to_csv(report_root / "state_metrics.csv", index=False); summary.to_csv(report_root / "variant_summary.csv", index=False); hypotheses.to_csv(report_root / "hypotheses.csv", index=False); subgroups.to_csv(report_root / "subgroups.csv", index=False); portability_frame.to_csv(report_root / "portability.csv", index=False); budget.to_csv(report_root / "compute_budget.csv", index=False); atomic_write_json(report_root / "decision.json", {"profile": args.profile, "hypotheses": decisions})
    decision_rows = pd.DataFrame([{"hypothesis": name, "pass": int(payload["pass"]), "criterion": payload["criterion"], "estimate": payload["combined"]["estimate"], "ci_low": payload["combined"]["ci_low"], "ci_high": payload["combined"]["ci_high"]} for name, payload in decisions.items()])
    report = ["# EXP-016–018 Query-attribution hypothesis matrix", "", f"Profile: `{args.profile}`. All variants use Qwen2.5-7B, state-normalized positive credit, matched source states within each hypothesis family, and identical seen/cross grids.", "", "## Variant means", "", markdown_table(summary), "", "## Hypothesis decisions", "", markdown_table(decision_rows), "", "## Full paired estimates", "", markdown_table(hypotheses), "", "## Compute audit", "", markdown_table(budget.groupby(["experiment_id", "variant"], as_index=False)[["train_targets", "processed_targets", "elapsed_seconds", "peak_allocated_gib"]].mean()), "", "Interpretation: H1/H2/H3 isolate equivalence structure; H4 identifies generic multi-query augmentation; H5/H6 test class-aware objectives; H7/H8 test cheaper class definitions. Any NLL pass must survive EXP-019 interactive retrieval.", ""]
    text = "\n".join(report); (report_root / "EXP016_018_REPORT.md").write_text(text, encoding="utf-8"); print(text)


if __name__ == "__main__":
    main()
