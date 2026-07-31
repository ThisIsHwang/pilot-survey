from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from stackpilot.query_equivalence_common import (
    atomic_write_json,
    hierarchical_bootstrap_difference,
    load_equivalence_config,
    read_jsonl,
)


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
    output = [
        "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |",
        "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |",
    ]
    output.extend(
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for row in rows
    )
    return "\n".join(output)


def bootstrap_states(rows, *, state_key: str, statistic: Callable, samples: int, seed: int):
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row[state_key]), []).append(dict(row))
    states = sorted(grouped)
    if not states:
        raise RuntimeError("No rows for state bootstrap")
    observed = float(statistic([item for state in states for item in grouped[state]]))
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled = rng.choice(states, size=len(states), replace=True)
        draws[index] = float(statistic([item for draw in sampled for item in grouped[str(draw)]]))
    if not np.isfinite(draws).all():
        raise RuntimeError("Non-finite state bootstrap draws")
    low, high = np.quantile(draws, [0.025, 0.975])
    return {"estimate": observed, "ci_low": float(low), "ci_high": float(high), "n_states": float(len(states)), "n_rows": float(len(rows))}


def offline_audit(prepared: Path, *, samples: int):
    states = read_jsonl(prepared / "states.jsonl")
    classes = read_jsonl(prepared / "classes.jsonl")
    candidates = read_jsonl(prepared / "candidates.jsonl")
    pairs = read_jsonl(prepared / "paired_states.jsonl")
    class_size = {str(row["class_id"]): int(row["class_size"]) for row in classes}
    for row in candidates:
        row["class_size"] = class_size[str(row["class_id"])]
        row["direct"] = int(float(row["immediate_support_gain"]) > 1e-8)
        row["replaceable_direct"] = int(row["direct"] and row["class_size"] >= 2)
    metrics = {
        "replaceable_direct_rate": bootstrap_states(
            candidates, state_key="state_id",
            statistic=lambda rows: float(np.mean([row["replaceable_direct"] for row in rows if row["direct"]])),
            samples=samples, seed=14001,
        ),
        "factual_direct_replaceability": bootstrap_states(
            [row for row in states if row["factual_direct"]], state_key="state_id",
            statistic=lambda rows: float(np.mean([int(int(row["factual_class_size"]) >= 2) for row in rows])),
            samples=samples, seed=14002,
        ),
        "state_redundancy_rate": bootstrap_states(
            [row for row in states if int(row["direct_candidate_count"]) > 0], state_key="state_id",
            statistic=lambda rows: float(np.mean([int(int(row["nontrivial_direct_class_count"]) > 0) for row in rows])),
            samples=samples, seed=14003,
        ),
        "factual_credit_overallocation": bootstrap_states(
            [row for row in states if row["factual_direct"] and int(row["factual_class_size"]) >= 2],
            state_key="state_id",
            statistic=lambda rows: float(np.mean([1.0 - 1.0 / int(row["factual_class_size"]) for row in rows])),
            samples=samples, seed=14004,
        ),
    }
    audit = pd.DataFrame([{"metric": name, **payload} for name, payload in metrics.items()])
    pair_frame = pd.DataFrame(pairs)
    summary = {
        "paired_states": len(pairs),
        "mean_edge_jaccard": float(pair_frame["edge_jaccard"].mean()) if len(pair_frame) else None,
        "mean_relation_agreement": float(pair_frame["relation_agreement"].mean()) if len(pair_frame) else None,
    }
    return audit, pair_frame, summary


def load_job_losses(plan_root: Path):
    jobs = read_jsonl(plan_root / "jobs.jsonl")
    rows, missing = [], []
    for job in jobs:
        output_dir = Path(job["output_dir"])
        metrics_path, losses_path = output_dir / "metrics.json", output_dir / "eval_losses.jsonl"
        if not metrics_path.is_file() or not losses_path.is_file():
            missing.append(job["job_id"])
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("job_signature") != job.get("job_signature"):
            raise RuntimeError(f"Stale job output: {job['job_id']}")
        for row in read_jsonl(losses_path):
            for name in ("baseline_nll", "adapted_nll", "heldout_gain"):
                if not math.isfinite(float(row[name])):
                    raise RuntimeError(f"Non-finite {name} in {job['job_id']}")
            rows.append(row)
    if missing:
        raise RuntimeError(f"Missing {len(missing)} EXP-014 jobs; first={missing[:5]}")
    return jobs, pd.DataFrame(rows)


def class_metrics(losses: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (direction, variant, seed, class_id), group in losses.groupby(
        ["direction", "variant", "seed", "class_id"], sort=True
    ):
        if float(group["baseline_nll"].max() - group["baseline_nll"].min()) > 20.0:
            raise RuntimeError(f"Implausible baseline NLL range for class {class_id}")
        gains = group["heldout_gain"].to_numpy(dtype=np.float64)
        factual = group[group["origin"] == "factual"]
        rows.append({
            "direction": str(direction), "variant": str(variant), "seed": int(seed),
            "class_id": str(class_id), "probe_id": f"{direction}:{class_id}",
            "member_count": len(group), "class_mean_gain": float(gains.mean()),
            "class_worst_gain": float(gains.min()),
            "class_gain_std": float(gains.std(ddof=0)),
            "positive_member_rate": float(np.mean(gains > 0.0)),
            "factual_gain": float(factual["heldout_gain"].mean()) if len(factual) else float("nan"),
        })
    return pd.DataFrame(rows)


def contrast_rows(class_frame: pd.DataFrame, *, samples: int) -> pd.DataFrame:
    output, seed = [], 14500
    comparisons = [
        ("equivalence-normalized", "factual-onehot"),
        ("equivalence-normalized", "random-onehot"),
    ]
    metrics = ("class_mean_gain", "class_worst_gain", "class_gain_std", "positive_member_rate")
    scopes = [("combined", class_frame)] + [(str(direction), group) for direction, group in class_frame.groupby("direction")]
    for scope, frame in scopes:
        for positive, negative in comparisons:
            for metric in metrics:
                result = hierarchical_bootstrap_difference(
                    frame.to_dict("records"), positive_variant=positive,
                    negative_variant=negative, value_key=metric,
                    seed_key="seed", example_key="probe_id", samples=samples,
                    random_seed=seed,
                )
                seed += 1
                output.append({"scope": scope, "contrast": f"{positive}-minus-{negative}", "metric": metric, **result})
    return pd.DataFrame(output)


def seed_stability(class_frame: pd.DataFrame) -> pd.DataFrame:
    seed_means = class_frame.groupby(["direction", "variant", "seed"], as_index=False)["class_mean_gain"].mean()
    return seed_means.groupby(["direction", "variant"])["class_mean_gain"].agg(seed_mean="mean", seed_std="std").reset_index()


def _series_numbers(row: pd.Series) -> dict[str, float]:
    return {name: float(row[name]) for name in ("estimate", "ci_low", "ci_high", "n_seeds", "n_pairs")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Report EXP-014 equivalence-aware credit audit.")
    parser.add_argument("--config", default="configs/query_equivalence.yaml")
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), default="pilot")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    cfg = load_equivalence_config(args.config)
    profile = cfg["profiles"][args.profile]
    work_root = Path(cfg["work_dir"]).resolve()
    output_dir = work_root / "reports" / args.profile
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = int(profile["bootstrap_samples"])
    audit, pair_frame, pair_summary = offline_audit(work_root / "prepared", samples=samples)
    audit.to_csv(output_dir / "offline_audit.csv", index=False)
    pair_frame.to_csv(output_dir / "cross_backend_pairs.csv", index=False)
    atomic_write_json(output_dir / "cross_backend_summary.json", pair_summary)
    report = [
        "# EXP-014 Query-equivalence credit report", "", f"Profile: `{args.profile}`.", "",
        "## Offline equivalence audit", "", markdown_table(audit), "",
        "## Cross-backend relation stability", "", "```text",
        json.dumps(pair_summary, indent=2, sort_keys=True), "```", "",
    ]
    decision: dict[str, Any] = {
        "experiment_id": "EXP-014", "profile": args.profile,
        "offline": {row["metric"]: {name: float(row[name]) for name in ("estimate", "ci_low", "ci_high")} for row in audit.to_dict("records")},
        "cross_backend": pair_summary,
    }
    if args.audit_only:
        decision["status"] = "audit-only"
        atomic_write_json(output_dir / "decision.json", decision)
        (output_dir / "EXP014_REPORT.md").write_text("\n".join(report), encoding="utf-8")
        print("\n".join(report))
        return

    jobs, losses = load_job_losses(work_root / "plans" / args.profile)
    classes = class_metrics(losses)
    contrasts = contrast_rows(classes, samples=samples)
    stability = seed_stability(classes)
    losses.to_csv(output_dir / "eval_losses.csv", index=False)
    classes.to_csv(output_dir / "class_metrics.csv", index=False)
    contrasts.to_csv(output_dir / "contrasts.csv", index=False)
    stability.to_csv(output_dir / "seed_stability.csv", index=False)
    report.extend([
        "## LoRA class metrics", "",
        markdown_table(classes.groupby(["direction", "variant"], as_index=False)[["class_mean_gain", "class_worst_gain", "class_gain_std", "positive_member_rate"]].mean()), "",
        "## Paired contrasts", "", markdown_table(contrasts), "",
        "## Seed stability", "", markdown_table(stability), "",
    ])

    def lookup(contrast: str, metric: str) -> pd.Series:
        rows = contrasts[(contrasts["scope"] == "combined") & (contrasts["contrast"] == contrast) & (contrasts["metric"] == metric)]
        if len(rows) != 1:
            raise RuntimeError(f"Missing combined {contrast} {metric}")
        return rows.iloc[0]

    eq_factual_mean = lookup("equivalence-normalized-minus-factual-onehot", "class_mean_gain")
    eq_factual_worst = lookup("equivalence-normalized-minus-factual-onehot", "class_worst_gain")
    eq_factual_std = lookup("equivalence-normalized-minus-factual-onehot", "class_gain_std")
    eq_random_mean = lookup("equivalence-normalized-minus-random-onehot", "class_mean_gain")
    gate = cfg["gate"]
    go = bool(
        float(eq_factual_mean["estimate"]) >= float(gate["minimum_mean_gain_advantage"])
        and float(eq_factual_mean["ci_low"]) > 0.0
        and float(eq_factual_worst["estimate"]) >= float(gate["minimum_worst_gain_advantage"])
        and float(eq_factual_worst["ci_low"]) > 0.0
        and float(eq_factual_std["estimate"]) <= 0.0
        and float(eq_random_mean["estimate"]) >= float(gate["minimum_random_gain_advantage"])
        and float(eq_random_mean["ci_low"]) > 0.0
        and float(audit.loc[audit["metric"] == "replaceable_direct_rate", "estimate"].iloc[0]) >= float(gate["minimum_replaceable_direct_rate"])
    )
    decision.update({
        "status": "valid", "go": go,
        "primary": {
            "mean_gain_advantage": _series_numbers(eq_factual_mean),
            "worst_gain_advantage": _series_numbers(eq_factual_worst),
            "gain_std_advantage": _series_numbers(eq_factual_std),
            "versus_random_mean_gain": _series_numbers(eq_random_mean),
        },
        "jobs": len(jobs),
    })
    atomic_write_json(output_dir / "decision.json", decision)
    report.extend([
        "## Decision", "", f"**{'GO' if go else 'NO-GO'}**", "",
        "A GO requires equivalence-normalized credit to improve class-average and worst-member held-out query NLL gain, reduce wording sensitivity, beat random one-hot selection, and arise in a sufficiently common redundancy regime.", "",
        "This is a LoRA micro-update diagnostic. A GO must be followed by an interactive query-policy comparison under matched search-call budgets.", "",
    ])
    report_text = "\n".join(report)
    (output_dir / "EXP014_REPORT.md").write_text(report_text, encoding="utf-8")
    print(report_text)


if __name__ == "__main__":
    main()
