from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

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
    lines = ["| " + " | ".join(headers[index].ljust(widths[index]) for index in range(len(headers))) + " |", "| " + " | ".join("-" * widths[index] for index in range(len(headers))) + " |"]
    lines.extend("| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(headers))) + " |" for row in rows)
    return "\n".join(lines)


def load_results(root: Path) -> pd.DataFrame:
    jobs_path = root / "jobs.jsonl"
    manifest_path = root / "manifest.json"
    if not jobs_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(f"Missing interactive plan under {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["jobs_sha256"] != file_sha256(jobs_path):
        raise RuntimeError("Interactive jobs do not match manifest")
    rows = []
    missing = []
    for job in read_jsonl(jobs_path):
        output_dir = Path(job["output_dir"])
        metrics_path = output_dir / "metrics.json"
        results_path = output_dir / "results.jsonl"
        if not metrics_path.is_file() or not results_path.is_file():
            missing.append(job["job_id"])
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("job_signature") != job.get("job_signature"):
            raise RuntimeError(f"Stale interactive output for {job['job_id']}")
        rows.extend(read_jsonl(results_path))
    if missing:
        raise RuntimeError(f"Missing {len(missing)} interactive jobs; first={missing[:5]}")
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No interactive rows")
    return frame


def paired_rows(frame: pd.DataFrame, *, left: str, right: str, metric: str, budget: int) -> pd.DataFrame:
    subset = frame[frame["sample_budget"].astype(int) == budget]
    pivot = subset.pivot_table(index=["direction", "seed", "state_id", "question_id"], columns="variant", values=metric, aggfunc="first")
    if left not in pivot or right not in pivot:
        return pd.DataFrame()
    output = pivot[[left, right]].dropna().reset_index()
    output["effect"] = output[left] - output[right]
    output["contrast"] = f"{left}-minus-{right}"
    output["metric"] = metric
    output["sample_budget"] = budget
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Report interactive multi-positive retrieval evaluation.")
    parser.add_argument("--config", default="configs/multipositive_generalization.yaml")
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), default="pilot")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    work_root = Path(cfg["work_dir"]).resolve()
    plan_root = work_root / "interactive_plans" / args.profile
    output_dir = Path(args.output_dir or work_root / "interactive_reports" / args.profile).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_results(plan_root)
    frame.to_csv(output_dir / "results.csv", index=False)
    summary = frame.groupby(["direction", "variant", "sample_budget"], as_index=False).agg(mean_evidence_gain=("mean_evidence_gain", "mean"), best_evidence_gain=("best_evidence_gain", "mean"), union_evidence_gain=("union_evidence_gain", "mean"), unique_behavior_count=("unique_behavior_count", "mean"), duplicate_behavior_rate=("duplicate_behavior_rate", "mean"), query_diversity=("query_diversity", "mean"), invalid_rate=("invalid_rate", "mean"))
    summary.to_csv(output_dir / "variant_summary.csv", index=False)

    specs = [
        ("I1_consistency_single_call", "all-direct-consistency", "all-direct-uniform", "mean_evidence_gain", 1),
        ("I2_consistency_union", "all-direct-consistency", "all-direct-uniform", "union_evidence_gain", 4),
        ("I3_consistency_behavior_coverage", "all-direct-consistency", "all-direct-uniform", "unique_behavior_count", 4),
        ("I4_strict_vs_direct_consistency", "strict-consistency", "all-direct-consistency", "union_evidence_gain", 4),
        ("I5_setmass_single_call", "all-direct-setmass", "all-direct-uniform", "mean_evidence_gain", 1),
        ("I6_setmass_union", "all-direct-setmass", "all-direct-uniform", "union_evidence_gain", 4),
        ("I7_setmass_consistency_union", "all-direct-setmass-consistency", "all-direct-consistency", "union_evidence_gain", 4),
        ("I8_multiquery_vs_repetition", "all-direct-consistency", "factual-replicated-uniform", "union_evidence_gain", 4),
    ]
    raw = []
    summaries = []
    samples = int(cfg["interactive"]["bootstrap_samples"][args.profile])
    for index, (hypothesis, left, right, metric, budget) in enumerate(specs):
        rows = paired_rows(frame, left=left, right=right, metric=metric, budget=budget)
        if rows.empty:
            summaries.append({"hypothesis": hypothesis, "estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_rows": 0.0})
            continue
        rows["hypothesis"] = hypothesis
        raw.append(rows)
        result = hierarchical_bootstrap(rows.to_dict("records"), value_key="effect", samples=samples, random_seed=19000 + index)
        summaries.append({"hypothesis": hypothesis, **result})
    contrast_rows = pd.concat(raw, ignore_index=True) if raw else pd.DataFrame()
    contrast_rows.to_csv(output_dir / "contrast_rows.csv", index=False)
    hypothesis_frame = pd.DataFrame(summaries)
    hypothesis_frame.to_csv(output_dir / "hypotheses.csv", index=False)
    by_name = {str(row["hypothesis"]): row for _, row in hypothesis_frame.iterrows()}
    gate = cfg["interactive"]["gate"]

    def passed(name: str, threshold: float) -> bool:
        row = by_name.get(name)
        return bool(row is not None and math.isfinite(float(row["estimate"])) and float(row["estimate"]) >= threshold and float(row["ci_low"]) > 0.0)

    decisions = {
        "consistency_single_call": passed("I1_consistency_single_call", float(gate["minimum_consistency_evidence_gain"])),
        "consistency_union_gain": passed("I2_consistency_union", float(gate["minimum_consistency_union_gain"])),
        "consistency_behavior_coverage": passed("I3_consistency_behavior_coverage", float(gate["minimum_unique_behavior_gain"])),
        "setmass_single_call": passed("I5_setmass_single_call", 0.0),
        "setmass_union_gain": passed("I6_setmass_union", 0.0),
        "multiquery_over_repetition": passed("I8_multiquery_vs_repetition", 0.0),
    }
    atomic_write_json(output_dir / "decision.json", {"schema": 1, "experiment_id": "EXP-027", "profile": args.profile, "decisions": decisions})
    report = ["# EXP-027 Interactive multi-positive retrieval report", "", f"Profile: `{args.profile}`. Each adapter is evaluated on identical held-out target-retriever states with one-call and four-call budgets.", "", "## Variant means", "", markdown_table(summary), "", "## Paired contrasts", "", markdown_table(hypothesis_frame), "", "## Decisions", ""]
    report.extend(f"- {name}: **{'PASS' if value else 'FAIL'}**" for name, value in decisions.items())
    report.extend(["", "A top-conference method claim requires improvements in actual evidence gain without reducing unique retrieval behaviors or increasing invalid queries. Query-NLL improvements alone are insufficient.", ""])
    (output_dir / "EXP027_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
