from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from stackpilot.behavior_quotient_common import atomic_write_json, load_config, markdown_table, read_jsonl
from stackpilot.behavior_quotient_telemetry import discover_telemetry, load_telemetry

VARIANTS = (
    "standard",
    "random-surface",
    "balanced-surface",
    "random-quotient",
    "balanced-quotient",
)
PRIMARY_METRICS = (
    "observed_support_title_recall",
    "f1",
    "search_count",
    "protocol_failure",
)


def _discover_episode_paths(config: dict[str, Any], experiment_id: str, patterns: Sequence[str] | None) -> list[Path]:
    if patterns:
        raw_patterns = list(patterns)
    else:
        environment = os.environ.get("BEHAVIOR_QUOTIENT_RESULTS", "").strip()
        if environment:
            raw_patterns = [part for part in environment.replace("\n", os.pathsep).split(os.pathsep) if part]
        else:
            experiment_ids = [experiment_id]
            if experiment_id == "EXP-027":
                experiment_ids.insert(0, "EXP-024")
            raw_patterns = [
                str(Path("work/experiments") / current / "results" / "**" / "episodes.jsonl")
                for current in experiment_ids
            ]
    output: dict[str, Path] = {}
    for pattern in raw_patterns:
        for raw in glob.glob(os.path.expanduser(pattern), recursive=True):
            path = Path(raw).resolve()
            if path.is_file():
                output[str(path)] = path
    return [output[key] for key in sorted(output)]


def _parse_source_and_method(variant: str) -> tuple[str, str]:
    for backend in ("bm25", "e5"):
        prefix = backend + "-"
        if variant.startswith(prefix):
            method = variant[len(prefix):]
            if method not in VARIANTS:
                raise RuntimeError(f"Unknown BQ method in variant {variant!r}")
            return backend, method
    raise RuntimeError(f"BQ variant must start with bm25- or e5-: {variant!r}")


def load_episodes(paths: Sequence[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for row in read_jsonl(path):
            variant = str(row.get("variant", ""))
            source_backend, method = _parse_source_and_method(variant)
            copy = dict(row)
            copy["source_backend"] = source_backend
            copy["method"] = method
            copy["result_path"] = str(path)
            rows.append(copy)
    if not rows:
        raise RuntimeError("No BQ numbered evaluation episodes were found")
    frame = pd.DataFrame(rows)
    required = {
        "question_id", "dataset", "backend", "topk", "seed", "variant",
        "observed_support_title_recall", "f1", "search_count",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"BQ evaluation rows miss {sorted(missing)}")
    for column in PRIMARY_METRICS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column]).all():
            raise RuntimeError(f"Non-finite BQ evaluation metric: {column}")
    frame["direction"] = frame["source_backend"] + "-to-" + frame["backend"].astype(str)
    return frame


def hierarchical_paired_bootstrap(
    rows: pd.DataFrame,
    *,
    metric: str,
    left: str,
    right: str,
    samples: int,
    seed: int,
) -> dict[str, float]:
    pivot = rows.pivot_table(
        index=["seed", "question_id", "dataset", "backend", "topk"],
        columns="method",
        values=metric,
        aggfunc="mean",
    )
    if left not in pivot.columns or right not in pivot.columns:
        raise RuntimeError(f"Missing paired methods {left!r} or {right!r} for {metric}")
    paired = pivot[[left, right]].dropna().reset_index()
    paired["difference"] = paired[left] - paired[right]
    if paired.empty:
        raise RuntimeError(f"No paired rows for {left} minus {right}")
    seed_groups = {
        int(seed_value): group.copy()
        for seed_value, group in paired.groupby("seed")
    }
    seeds = sorted(seed_groups)
    estimate = float(np.mean([group["difference"].mean() for group in seed_groups.values()]))
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for draw_index in range(samples):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        seed_means = []
        for sampled_seed in sampled_seeds:
            group = seed_groups[int(sampled_seed)]
            question_ids = group["question_id"].astype(str).unique()
            sampled_questions = rng.choice(question_ids, size=len(question_ids), replace=True)
            values = []
            for question_id in sampled_questions:
                values.extend(
                    group.loc[group["question_id"].astype(str) == str(question_id), "difference"].tolist()
                )
            seed_means.append(float(np.mean(values)))
        draws[draw_index] = float(np.mean(seed_means))
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "estimate": estimate,
        "ci_low": float(low),
        "ci_high": float(high),
        "n_seeds": float(len(seeds)),
        "n_questions": float(paired["question_id"].nunique()),
        "n_rows": float(len(paired)),
    }


def _telemetry_variant_means(config: dict[str, Any], profile: str) -> pd.DataFrame:
    paths = discover_telemetry(config, profile)
    if not paths:
        return pd.DataFrame()
    frame = load_telemetry(paths)
    if "variant" not in frame.columns:
        return pd.DataFrame()
    return (
        frame.groupby(["variant", "backend"], as_index=False)
        .agg(
            alias_fraction=("alias_fraction", "mean"),
            effective_behavior_count=("effective_behavior_count", "mean"),
            selected_behavior_coverage=("selected_behavior_coverage", "mean"),
            duplicate_rate=("selected_duplicate_rate", "mean"),
            nonzero_advantage_fraction=("nonzero_advantage_fraction", "mean"),
        )
    )


def report(
    config: dict[str, Any],
    profile: str,
    *,
    experiment_id: str = "EXP-027",
    patterns: Sequence[str] | None = None,
) -> dict[str, Any]:
    paths = _discover_episode_paths(config, experiment_id, patterns)
    frame = load_episodes(paths)
    output_dir = Path(config["work_dir"]).resolve() / "reports" / profile / "EXP-028"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "evaluation_episodes.csv", index=False)
    means = (
        frame.groupby(["source_backend", "method", "backend", "topk"], as_index=False)
        .agg(
            observed_support_recall=("observed_support_title_recall", "mean"),
            f1=("f1", "mean"),
            search_count=("search_count", "mean"),
            protocol_failure=("protocol_failure", "mean"),
            questions=("question_id", "nunique"),
        )
    )
    means.to_csv(output_dir / "variant_means.csv", index=False)

    contrast_specs = (
        ("balanced-surface", "random-surface", "selection_effect_surface"),
        ("balanced-quotient", "random-quotient", "selection_effect_quotient"),
        ("random-quotient", "random-surface", "normalization_effect_random"),
        ("balanced-quotient", "balanced-surface", "normalization_effect_balanced"),
        ("balanced-quotient", "random-surface", "joint_effect"),
        ("balanced-quotient", "standard", "joint_minus_standard"),
    )
    contrast_rows: list[dict[str, Any]] = []
    samples = int(config["profiles"][profile]["bootstrap_samples"])
    for source_backend in ("bm25", "e5"):
        source_rows = frame[frame["source_backend"] == source_backend]
        for target_backend in sorted(source_rows["backend"].unique()):
            direction_rows = source_rows[
                (source_rows["backend"] == target_backend)
                & (source_rows["topk"] == int(config["training"]["topk"]))
            ]
            for left, right, label in contrast_specs:
                for metric_index, metric in enumerate(PRIMARY_METRICS):
                    try:
                        result = hierarchical_paired_bootstrap(
                            direction_rows,
                            metric=metric,
                            left=left,
                            right=right,
                            samples=samples,
                            seed=28028 + len(contrast_rows) + metric_index,
                        )
                    except RuntimeError:
                        continue
                    contrast_rows.append(
                        {
                            "source_backend": source_backend,
                            "target_backend": target_backend,
                            "contrast": label,
                            "left": left,
                            "right": right,
                            "metric": metric,
                            **result,
                        }
                    )
    contrasts = pd.DataFrame(contrast_rows)
    contrasts.to_csv(output_dir / "paired_contrasts.csv", index=False)
    telemetry = _telemetry_variant_means(config, profile)
    if not telemetry.empty:
        telemetry.to_csv(output_dir / "telemetry_means.csv", index=False)

    training_gate = config["gates"]["EXP-027"]
    evaluation_gate = config["gates"]["EXP-028"]
    joint = contrasts[contrasts["contrast"] == "joint_effect"]
    base_targets = joint[joint["target_backend"].isin(["bm25", "e5"])]
    support = base_targets[base_targets["metric"] == "observed_support_title_recall"]
    f1 = base_targets[base_targets["metric"] == "f1"]
    selection_ok = bool(
        len(support) >= 4
        and (support["estimate"] >= float(training_gate["minimum_evidence_gain"])).all()
        and (support["ci_low"] > 0).all()
        and len(f1) >= 4
        and (f1["estimate"] >= -float(training_gate["maximum_answer_f1_regression"])).all()
    )
    behavior_ok = False
    if not telemetry.empty:
        def method_rows(method: str) -> pd.DataFrame:
            return telemetry[telemetry["variant"].astype(str).str.endswith("-" + method)]
        balanced = method_rows("balanced-quotient")["selected_behavior_coverage"].mean()
        random_surface = method_rows("random-surface")["selected_behavior_coverage"].mean()
        behavior_ok = bool(
            math.isfinite(float(balanced))
            and math.isfinite(float(random_surface))
            and float(balanced - random_surface) >= float(training_gate["minimum_behavior_coverage_gain"])
        )
    versus_standard = contrasts[contrasts["contrast"] == "joint_minus_standard"]
    support_vs_standard = versus_standard[
        versus_standard["metric"] == "observed_support_title_recall"
    ]
    f1_vs_standard = versus_standard[versus_standard["metric"] == "f1"]
    search_vs_standard = versus_standard[versus_standard["metric"] == "search_count"]
    protocol_vs_standard = versus_standard[
        versus_standard["metric"] == "protocol_failure"
    ]
    non_hybrid_support = support_vs_standard[
        support_vs_standard["target_backend"].isin(["bm25", "e5"])
    ]
    hybrid_support = support_vs_standard[
        support_vs_standard["target_backend"] == "hybrid"
    ]
    end_to_end_ok = bool(
        len(non_hybrid_support) >= 4
        and (non_hybrid_support["estimate"] >= float(
            evaluation_gate["minimum_seen_or_cross_support_gain"]
        )).all()
        and (non_hybrid_support["ci_low"] > 0).all()
        and len(hybrid_support) >= 2
        and (hybrid_support["estimate"] >= float(
            evaluation_gate["minimum_hybrid_support_gain"]
        )).all()
        and (hybrid_support["ci_low"] > 0).all()
        and len(f1_vs_standard) >= 6
        and (f1_vs_standard["estimate"] >= -float(
            evaluation_gate["maximum_answer_f1_regression"]
        )).all()
        and len(search_vs_standard) >= 6
        and (search_vs_standard["estimate"] <= float(
            evaluation_gate["maximum_search_call_increase"]
        )).all()
        and len(protocol_vs_standard) >= 6
        and (protocol_vs_standard["estimate"] <= float(
            evaluation_gate["maximum_protocol_failure_increase"]
        )).all()
    )
    decision = selection_ok and behavior_ok and end_to_end_ok
    payload = {
        "schema": 1,
        "experiment_id": "EXP-028",
        "training_experiment_id": experiment_id,
        "profile": profile,
        "evaluation_files": len(paths),
        "runs": int(frame[["variant", "seed"]].drop_duplicates().shape[0]),
        "selection_utility_go": selection_ok,
        "behavior_coverage_go": behavior_ok,
        "end_to_end_vs_standard_go": end_to_end_ok,
        "go": decision,
    }
    atomic_write_json(output_dir / "decision.json", payload)
    report_lines = [
        "# EXP-028 End-to-end behavior-quotient evaluation",
        "",
        f"Profile: `{profile}`. All primary 2×2 variants generate eight trajectories per prompt and update from four rows; only selection and GRPO normalization differ.",
        "",
        "## Evaluation means",
        "",
        markdown_table(means),
        "",
        "## Paired contrasts",
        "",
        markdown_table(contrasts),
        "",
        "## Training telemetry",
        "",
        markdown_table(telemetry),
        "",
        f"Decision: **{'GO' if decision else 'NO-GO'}**.",
        "",
    ]
    (output_dir / "EXP028_REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/behavior_quotient.yaml")
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), default="pilot")
    parser.add_argument("--experiment-id", default="EXP-027")
    parser.add_argument("--result", action="append", default=None)
    args = parser.parse_args()
    payload = report(
        load_config(args.config),
        args.profile,
        experiment_id=args.experiment_id,
        patterns=args.result,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
