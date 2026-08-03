from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from stackpilot.query_attribution_common import atomic_write_json, file_sha256, hierarchical_bootstrap, load_config, read_jsonl
from stackpilot.query_attribution_report import markdown_table


def load_results(root: Path, profile: str) -> pd.DataFrame:
    plan_root = root / "interactive_plans" / profile; jobs_path = plan_root / "jobs.jsonl"; manifest_path = plan_root / "manifest.json"
    if not jobs_path.is_file() or not manifest_path.is_file(): raise RuntimeError("Interactive plan is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["jobs_sha256"] != file_sha256(jobs_path): raise RuntimeError("Interactive jobs do not match manifest")
    rows = []; missing = []
    for job in read_jsonl(jobs_path):
        output = Path(job["output_dir"]); metrics = output / "metrics.json"; results = output / "results.jsonl"
        if not metrics.is_file() or not results.is_file(): missing.append(job["job_id"]); continue
        payload = json.loads(metrics.read_text(encoding="utf-8"))
        if payload.get("job_signature") != job.get("job_signature"): raise RuntimeError(f"Stale interactive output for {job['job_id']}")
        rows.extend(read_jsonl(results))
    if missing: raise RuntimeError(f"Missing {len(missing)} interactive jobs; first={missing[:5]}")
    return pd.DataFrame(rows)


def paired_effects(frame: pd.DataFrame, left: str, right: str, metric: str) -> pd.DataFrame:
    pivot = frame.pivot_table(index=["direction", "seed", "question_id", "state_id"], columns="variant", values=metric, aggfunc="first")
    if left not in pivot or right not in pivot: raise RuntimeError(f"Missing variants for interactive contrast {left}-{right}")
    result = pivot[[left, right]].dropna().reset_index(); result["effect"] = result[left] - result[right]; result["metric"] = metric; result["contrast"] = f"{left}-minus-{right}"; return result


def summarize_effects(frame: pd.DataFrame, samples: int) -> pd.DataFrame:
    specs = [("strict-uniform", "diversity-matched-k", "evidence_gain"), ("strict-uniform", "all-direct-k", "evidence_gain"), ("strict-hardmax", "strict-uniform", "evidence_gain"), ("strict-consistency", "strict-uniform", "evidence_gain"), ("strict-uniform", "diversity-matched-k", "invalid_query")]
    rows = []
    for index, (left, right, metric) in enumerate(specs):
        effects = paired_effects(frame, left, right, metric)
        for direction_index, (direction, group) in enumerate(effects.groupby("direction", sort=True)):
            rows.append({"scope": str(direction), "contrast": f"{left}-minus-{right}", "metric": metric, **hierarchical_bootstrap(group.to_dict("records"), value_key="effect", samples=samples, random_seed=19500 + index * 20 + direction_index)})
        rows.append({"scope": "combined", "contrast": f"{left}-minus-{right}", "metric": metric, **hierarchical_bootstrap(effects.to_dict("records"), value_key="effect", samples=samples, random_seed=19600 + index)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report EXP-019 interactive retrieval confirmation."); parser.add_argument("--config", default="configs/query_attribution.yaml"); parser.add_argument("--profile", choices=("smoke", "pilot", "full"), default="pilot"); args = parser.parse_args(); cfg = load_config(args.config); root = Path(cfg["work_dir"]).resolve(); frame = load_results(root, args.profile); output = root / "interactive_reports" / args.profile; output.mkdir(parents=True, exist_ok=True)
    summary = frame.groupby(["direction", "variant", "seed"], as_index=False)[["evidence_gain", "final_support_recall", "invalid_query"]].mean().groupby(["direction", "variant"]).agg(evidence_gain=("evidence_gain", "mean"), seed_std=("evidence_gain", "std"), final_support_recall=("final_support_recall", "mean"), invalid_rate=("invalid_query", "mean")).reset_index().fillna({"seed_std": 0.0})
    effects = summarize_effects(frame, int(cfg["interactive"]["bootstrap_samples"][args.profile])); frame.to_csv(output / "results.csv", index=False); summary.to_csv(output / "variant_summary.csv", index=False); effects.to_csv(output / "contrasts.csv", index=False)
    def combined(contrast: str, metric: str) -> pd.Series:
        subset = effects[(effects["scope"] == "combined") & (effects["contrast"] == contrast) & (effects["metric"] == metric)]
        if len(subset) != 1: raise RuntimeError(f"Missing interactive contrast {contrast}/{metric}")
        return subset.iloc[0]
    strict_div = combined("strict-uniform-minus-diversity-matched-k", "evidence_gain"); strict_direct = combined("strict-uniform-minus-all-direct-k", "evidence_gain"); invalid = combined("strict-uniform-minus-diversity-matched-k", "invalid_query"); gate = cfg["interactive"]["gate"]
    go = bool(float(strict_div["estimate"]) >= float(gate["minimum_strict_vs_diversity_evidence_gain"]) and float(strict_div["ci_low"]) > 0.0 and float(strict_direct["estimate"]) >= float(gate["minimum_strict_vs_all_direct_evidence_gain"]) and float(strict_direct["ci_low"]) > 0.0 and float(invalid["estimate"]) <= float(gate["maximum_invalid_rate_increase"]))
    atomic_write_json(output / "decision.json", {"experiment_id": "EXP-019", "profile": args.profile, "go": go, "strict_vs_diversity": strict_div.to_dict(), "strict_vs_all_direct": strict_direct.to_dict(), "invalid_rate_difference": invalid.to_dict()})
    report = ["# EXP-019 Interactive query-retrieval confirmation", "", f"Profile: `{args.profile}`. Each adapter generates one next query from the same held-out state and receives exactly one target-retriever call.", "", "## Variant means", "", markdown_table(summary), "", "## Paired contrasts", "", markdown_table(effects), "", f"Decision: **{'GO' if go else 'NO-GO'}**.", "", "A GO requires strict equivalence training to improve actual evidence gain over diversity-matched and all-direct controls without increasing invalid queries.", ""]
    text = "\n".join(report); (output / "EXP019_REPORT.md").write_text(text, encoding="utf-8"); print(text)


if __name__ == "__main__":
    main()
