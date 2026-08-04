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


def discover_telemetry(config: dict[str, Any], profile: str, patterns: Sequence[str] | None = None) -> list[Path]:
    if patterns:
        raw_patterns = list(patterns)
    else:
        environment = os.environ.get("BEHAVIOR_QUOTIENT_TELEMETRY", "").strip()
        if environment:
            raw_patterns = [part for part in environment.replace("\n", os.pathsep).split(os.pathsep) if part]
        else:
            raw_patterns = [
                str(Path(config["work_dir"]).resolve() / "telemetry" / profile / "**" / "*.jsonl")
            ]
    output: dict[str, Path] = {}
    for pattern in raw_patterns:
        for raw in glob.glob(os.path.expanduser(pattern), recursive=True):
            path = Path(raw).resolve()
            if path.is_file():
                output[str(path)] = path
    return [output[key] for key in sorted(output)]


def load_telemetry(paths: Sequence[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for row in read_jsonl(path):
            copy = dict(row)
            copy["telemetry_path"] = str(path)
            rows.append(copy)
    if not rows:
        raise RuntimeError("No behavior-quotient telemetry rows were found")
    frame = pd.DataFrame(rows)
    required = {
        "global_step",
        "uid",
        "alias_fraction",
        "effective_behavior_count",
        "selected_behavior_coverage",
        "selected_duplicate_rate",
        "nonzero_advantage_fraction",
        "surface_reward_variance",
        "class_reward_variance",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Telemetry is missing columns: {sorted(missing)}")
    for column in required - {"uid"}:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column]).all():
            raise RuntimeError(f"Telemetry contains non-finite {column}")
    for optional, default in (
        ("experiment_id", "unknown"),
        ("variant", "unknown"),
        ("backend", "unknown"),
        ("seed", 0),
        ("advantage_mode", "surface"),
        ("selection_mode", "all"),
    ):
        if optional not in frame.columns:
            frame[optional] = default
    return frame


def _slope(values: pd.Series, steps: pd.Series) -> float:
    x = steps.to_numpy(dtype=np.float64)
    y = values.to_numpy(dtype=np.float64)
    if len(x) < 2 or float(np.max(x) - np.min(x)) <= 0:
        return 0.0
    normalized = (x - x.min()) / (x.max() - x.min())
    return float(np.polyfit(normalized, y, 1)[0])


def run_dynamics(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    step_summary = (
        frame.groupby(
            ["experiment_id", "variant", "backend", "seed", "global_step"],
            as_index=False,
        )
        .agg(
            alias_fraction=("alias_fraction", "mean"),
            effective_behavior_count=("effective_behavior_count", "mean"),
            behavior_coverage=("selected_behavior_coverage", "mean"),
            duplicate_rate=("selected_duplicate_rate", "mean"),
            nonzero_advantage_fraction=("nonzero_advantage_fraction", "mean"),
            surface_reward_variance=("surface_reward_variance", "mean"),
            class_reward_variance=("class_reward_variance", "mean"),
            groups=("uid", "nunique"),
        )
    )
    slopes: list[dict[str, Any]] = []
    metric_names = (
        "alias_fraction",
        "effective_behavior_count",
        "behavior_coverage",
        "duplicate_rate",
        "nonzero_advantage_fraction",
        "surface_reward_variance",
        "class_reward_variance",
    )
    for keys, group in step_summary.groupby(
        ["experiment_id", "variant", "backend", "seed"], sort=True
    ):
        row = dict(zip(("experiment_id", "variant", "backend", "seed"), keys))
        for metric in metric_names:
            row[f"{metric}_slope"] = _slope(group[metric], group["global_step"])
        row["steps"] = int(group["global_step"].nunique())
        slopes.append(row)
    return step_summary, pd.DataFrame(slopes)


def report(config: dict[str, Any], profile: str, patterns: Sequence[str] | None = None) -> dict[str, Any]:
    paths = discover_telemetry(config, profile, patterns)
    frame = load_telemetry(paths)
    step_summary, slopes = run_dynamics(frame)
    output_dir = Path(config["work_dir"]).resolve() / "reports" / profile / "EXP-024"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "telemetry_rows.csv", index=False)
    step_summary.to_csv(output_dir / "step_summary.csv", index=False)
    slopes.to_csv(output_dir / "run_slopes.csv", index=False)

    standard = slopes[
        (slopes["experiment_id"].astype(str) == "EXP-024")
        & slopes["variant"].astype(str).str.contains(
            "standard", case=False, regex=False
        )
    ]
    if standard.empty:
        standard = slopes[
            (slopes["advantage_mode"].astype(str) == "surface")
            if "advantage_mode" in slopes.columns
            else np.ones(len(slopes), dtype=bool)
        ]
    gate = config["gates"]["EXP-024"]
    alias_growth = float(standard["alias_fraction_slope"].mean()) if not standard.empty else 0.0
    behavior_decline = float(standard["effective_behavior_count_slope"].mean()) if not standard.empty else 0.0
    nonzero_decline = float(standard["nonzero_advantage_fraction_slope"].mean()) if not standard.empty else 0.0
    decision = bool(
        alias_growth >= float(gate["minimum_alias_growth"])
        and behavior_decline <= -float(gate["minimum_behavior_count_decline"])
        and nonzero_decline <= 0.0
    )
    payload = {
        "schema": 1,
        "experiment_id": "EXP-024",
        "profile": profile,
        "telemetry_files": len(paths),
        "runs": int(len(slopes)),
        "standard_alias_growth": alias_growth,
        "standard_effective_behavior_count_slope": behavior_decline,
        "standard_nonzero_advantage_slope": nonzero_decline,
        "go": decision,
    }
    atomic_write_json(output_dir / "decision.json", payload)
    means = (
        step_summary.groupby(["variant", "backend"], as_index=False)
        .agg(
            alias_fraction=("alias_fraction", "mean"),
            effective_behavior_count=("effective_behavior_count", "mean"),
            behavior_coverage=("behavior_coverage", "mean"),
            duplicate_rate=("duplicate_rate", "mean"),
            nonzero_advantage_fraction=("nonzero_advantage_fraction", "mean"),
        )
    )
    report_lines = [
        "# EXP-024 Natural on-policy alias dynamics",
        "",
        f"Profile: `{profile}`. This report uses natural rollout aliases; no virtual alias multiplicity is injected.",
        "",
        "## Run means",
        "",
        markdown_table(means),
        "",
        "## Per-run normalized-step slopes",
        "",
        markdown_table(slopes),
        "",
        f"Decision: **{'GO' if decision else 'NO-GO'}**.",
        "",
        "A GO means standard surface GRPO exhibits increasing natural aliasing, falling effective behavior count, and no compensating increase in non-zero advantage coverage during actual training.",
        "",
    ]
    (output_dir / "EXP024_REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/behavior_quotient.yaml")
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), default="pilot")
    parser.add_argument("--telemetry", action="append", default=None)
    args = parser.parse_args()
    payload = report(load_config(args.config), args.profile, args.telemetry)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
