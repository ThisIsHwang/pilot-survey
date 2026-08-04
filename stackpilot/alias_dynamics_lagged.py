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

EXPERIMENT_ID = "EXP-029"


def telemetry_paths(cfg: dict[str, Any], profile: str, provided: Sequence[str] | None = None) -> list[Path]:
    patterns = list(provided or [])
    if not patterns:
        environment = os.environ.get("BEHAVIOR_FEEDBACK_TELEMETRY", "").strip()
        if environment:
            patterns = [part for part in environment.replace("\n", os.pathsep).split(os.pathsep) if part]
        else:
            patterns = [str(value).format(profile=profile) for value in cfg["source"]["telemetry_globs"]]
    output: dict[str, Path] = {}
    for pattern in patterns:
        for raw in glob.glob(os.path.expanduser(pattern), recursive=True):
            path = Path(raw).resolve()
            if path.is_file():
                output[str(path)] = path
    return [output[key] for key in sorted(output)]


def load_step_frame(paths: Sequence[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for row in read_jsonl(path):
            class_rewards = row.get("class_rewards", [])
            if not isinstance(class_rewards, list) or not class_rewards:
                continue
            numeric_rewards = [float(value) for value in class_rewards if math.isfinite(float(value))]
            if not numeric_rewards:
                continue
            copy = dict(row)
            copy["class_mean_reward"] = float(np.mean(numeric_rewards))
            copy["class_best_reward"] = float(np.max(numeric_rewards))
            copy["telemetry_path"] = str(path)
            rows.append(copy)
    if not rows:
        raise RuntimeError("No usable behavior telemetry rows were found")
    frame = pd.DataFrame(rows)
    required = {
        "global_step", "variant", "backend", "seed", "alias_fraction",
        "effective_behavior_count", "nonzero_advantage_fraction", "class_mean_reward",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Telemetry rows miss {sorted(missing)}")
    numeric = [
        "global_step", "seed", "alias_fraction", "effective_behavior_count",
        "nonzero_advantage_fraction", "class_mean_reward", "class_best_reward",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column]).all():
            raise RuntimeError(f"Non-finite telemetry column: {column}")
    frame["run_id"] = (
        frame["variant"].astype(str) + "|" + frame["backend"].astype(str) + "|" + frame["seed"].astype(int).astype(str)
    )
    step = (
        frame.groupby(["run_id", "variant", "backend", "seed", "global_step"], as_index=False)
        .agg(
            alias_fraction=("alias_fraction", "mean"),
            effective_behavior_count=("effective_behavior_count", "mean"),
            nonzero_advantage_fraction=("nonzero_advantage_fraction", "mean"),
            class_mean_reward=("class_mean_reward", "mean"),
            class_best_reward=("class_best_reward", "mean"),
            prompt_groups=("uid", "count"),
        )
        .sort_values(["run_id", "global_step"])
    )
    return step


def run_slopes(step: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run_id, group in step.groupby("run_id"):
        group = group.sort_values("global_step")
        first = group.iloc[0]
        last = group.iloc[-1]
        initial_behavior = max(float(first["effective_behavior_count"]), 1e-8)
        rows.append(
            {
                "run_id": run_id,
                "variant": first["variant"],
                "backend": first["backend"],
                "seed": int(first["seed"]),
                "steps": len(group),
                "alias_growth": float(last["alias_fraction"] - first["alias_fraction"]),
                "relative_behavior_decline": float(
                    (first["effective_behavior_count"] - last["effective_behavior_count"]) / initial_behavior
                ),
                "nonzero_advantage_change": float(
                    last["nonzero_advantage_fraction"] - first["nonzero_advantage_fraction"]
                ),
                "reward_change": float(last["class_mean_reward"] - first["class_mean_reward"]),
            }
        )
    return pd.DataFrame(rows)


def lagged_rows(step: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run_id, group in step.groupby("run_id"):
        group = group.sort_values("global_step").reset_index(drop=True)
        for index in range(len(group) - 1):
            current = group.iloc[index]
            following = group.iloc[index + 1]
            rows.append(
                {
                    "run_id": run_id,
                    "backend": current["backend"],
                    "variant": current["variant"],
                    "seed": int(current["seed"]),
                    "step": float(current["global_step"]),
                    "alias_fraction": float(current["alias_fraction"]),
                    "effective_behavior_count": float(current["effective_behavior_count"]),
                    "nonzero_advantage_fraction": float(current["nonzero_advantage_fraction"]),
                    "current_reward": float(current["class_mean_reward"]),
                    "next_reward_delta": float(following["class_mean_reward"] - current["class_mean_reward"]),
                    "next_nonzero_delta": float(
                        following["nonzero_advantage_fraction"] - current["nonzero_advantage_fraction"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    standard = float(values.std(ddof=0))
    if standard <= 1e-12:
        return np.zeros_like(values)
    return (values - float(values.mean())) / standard


PREDICTORS = (
    "alias_fraction",
    "effective_behavior_count",
    "nonzero_advantage_fraction",
    "current_reward",
    "step",
)


def _run_sufficient_statistics(
    frame: pd.DataFrame, outcome: str
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    run_ids: list[str] = []
    xtx: list[np.ndarray] = []
    xty: list[np.ndarray] = []
    xss: list[np.ndarray] = []
    yss: list[float] = []
    counts: list[int] = []
    for run_id, group in frame.groupby("run_id"):
        x = group[list(PREDICTORS)].to_numpy(dtype=np.float64)
        y = group[outcome].to_numpy(dtype=np.float64)
        x = x - x.mean(axis=0, keepdims=True)
        y = y - y.mean()
        run_ids.append(str(run_id))
        xtx.append(x.T @ x)
        xty.append(x.T @ y)
        xss.append(np.square(x).sum(axis=0))
        yss.append(float(np.square(y).sum()))
        counts.append(int(len(group)))
    if len(run_ids) < 2:
        raise RuntimeError("Lagged fixed-effect analysis needs at least two runs")
    return (
        run_ids,
        np.stack(xtx),
        np.stack(xty),
        np.stack(xss),
        np.asarray(yss, dtype=np.float64),
        np.asarray(counts, dtype=np.float64),
    )


def _coefficient_from_run_weights(
    weights: np.ndarray,
    xtx: np.ndarray,
    xty: np.ndarray,
    xss: np.ndarray,
    yss: np.ndarray,
    counts: np.ndarray,
) -> dict[str, float]:
    total_n = float(weights @ counts)
    if total_n <= 0:
        raise RuntimeError("Bootstrap selected no lagged rows")
    total_xtx = np.tensordot(weights, xtx, axes=(0, 0))
    total_xty = np.tensordot(weights, xty, axes=(0, 0))
    total_xss = weights @ xss
    total_yss = float(weights @ yss)
    x_scale = np.sqrt(np.maximum(total_xss / total_n, 0.0))
    y_scale = math.sqrt(max(total_yss / total_n, 0.0))
    active = x_scale > 1e-12
    coefficients = np.zeros(len(PREDICTORS), dtype=np.float64)
    if y_scale > 1e-12 and bool(active.any()):
        active_xtx = total_xtx[np.ix_(active, active)]
        active_scale = x_scale[active]
        standardized_xtx = active_xtx / (
            active_scale[:, None] * active_scale[None, :]
        )
        standardized_xty = total_xty[active] / (active_scale * y_scale)
        solution, *_ = np.linalg.lstsq(
            standardized_xtx, standardized_xty, rcond=None
        )
        coefficients[active] = solution
    return {
        "intercept": 0.0,
        **{
            name: float(value)
            for name, value in zip(PREDICTORS, coefficients)
        },
    }


def fixed_effect_coefficients(frame: pd.DataFrame, outcome: str) -> dict[str, float]:
    _runs, xtx, xty, xss, yss, counts = _run_sufficient_statistics(frame, outcome)
    return _coefficient_from_run_weights(
        np.ones(len(xtx), dtype=np.float64), xtx, xty, xss, yss, counts
    )


def bootstrap_coefficients(
    frame: pd.DataFrame, *, outcome: str, samples: int, seed: int
) -> pd.DataFrame:
    runs, xtx, xty, xss, yss, counts = _run_sufficient_statistics(frame, outcome)
    observed = _coefficient_from_run_weights(
        np.ones(len(runs), dtype=np.float64), xtx, xty, xss, yss, counts
    )
    rng = np.random.default_rng(seed)
    bootstrap_weights = rng.multinomial(
        len(runs), np.full(len(runs), 1.0 / len(runs)), size=int(samples)
    ).astype(np.float64)
    draws: dict[str, np.ndarray] = {
        key: np.empty(int(samples), dtype=np.float64) for key in observed
    }
    for index, weights in enumerate(bootstrap_weights):
        coefficients = _coefficient_from_run_weights(
            weights, xtx, xty, xss, yss, counts
        )
        for key, value in coefficients.items():
            draws[key][index] = float(value)
    rows = []
    for key, estimate in observed.items():
        low, high = np.quantile(draws[key], [0.025, 0.975])
        rows.append(
            {
                "outcome": outcome,
                "term": key,
                "estimate": estimate,
                "ci_low": float(low),
                "ci_high": float(high),
                "runs": len(runs),
                "rows": len(frame),
            }
        )
    return pd.DataFrame(rows)


def run(cfg: dict[str, Any], profile_name: str, provided: Sequence[str] | None = None) -> dict[str, Any]:
    profile = cfg["profiles"][profile_name]
    step = load_step_frame(telemetry_paths(cfg, profile_name, provided))
    standard = step[step["variant"].astype(str).str.endswith("-standard")].copy()
    if standard.empty:
        standard = step[step["variant"].astype(str).str.contains("standard")].copy()
    if standard.empty:
        raise RuntimeError("No standard on-policy telemetry found for EXP-029")
    slopes = run_slopes(standard)
    lagged = lagged_rows(standard)
    reward_coefficients = bootstrap_coefficients(
        lagged,
        outcome="next_reward_delta",
        samples=int(profile["bootstrap_samples"]),
        seed=29029,
    )
    nonzero_coefficients = bootstrap_coefficients(
        lagged,
        outcome="next_nonzero_delta",
        samples=int(profile["bootstrap_samples"]),
        seed=29030,
    )
    coefficients = pd.concat([reward_coefficients, nonzero_coefficients], ignore_index=True)

    output_dir = Path(cfg["work_dir"]).resolve() / "reports" / profile_name / EXPERIMENT_ID
    output_dir.mkdir(parents=True, exist_ok=True)
    step.to_csv(output_dir / "step_summary.csv", index=False)
    slopes.to_csv(output_dir / "run_slopes.csv", index=False)
    lagged.to_csv(output_dir / "lagged_rows.csv", index=False)
    coefficients.to_csv(output_dir / "lagged_coefficients.csv", index=False)

    gate = cfg["gates"][EXPERIMENT_ID]
    alias_term = reward_coefficients[reward_coefficients["term"] == "alias_fraction"].iloc[0]
    behavior_term = reward_coefficients[reward_coefficients["term"] == "effective_behavior_count"].iloc[0]
    go = bool(
        (
            slopes["alias_growth"].mean() >= float(gate["minimum_alias_growth"])
            or slopes["relative_behavior_decline"].mean() >= float(gate["minimum_behavior_count_decline"])
        )
        and float(alias_term["estimate"]) <= float(gate["maximum_standardized_alias_to_next_reward"])
        and float(alias_term["ci_high"]) < 0.0
        and float(behavior_term["estimate"]) >= float(gate["minimum_standardized_behavior_to_next_reward"])
        and float(behavior_term["ci_low"]) > 0.0
    )
    payload = {
        "schema": 1,
        "experiment_id": EXPERIMENT_ID,
        "profile": profile_name,
        "runs": int(slopes["run_id"].nunique()),
        "steps": int(len(step)),
        "mean_alias_growth": float(slopes["alias_growth"].mean()),
        "mean_relative_behavior_decline": float(slopes["relative_behavior_decline"].mean()),
        "go": go,
    }
    atomic_write_json(output_dir / "decision.json", payload)
    report = [
        "# EXP-029 Natural alias dynamics and lagged learning signal",
        "",
        f"Profile: `{profile_name}`. The analysis uses only standard on-policy GRPO telemetry and tests whether alias occupancy precedes weaker next-step reward improvement.",
        "",
        "## Run-level changes",
        "",
        markdown_table(slopes),
        "",
        "## Lagged fixed-effect coefficients",
        "",
        markdown_table(coefficients),
        "",
        f"Decision: **{'GO' if go else 'NO-GO'}**.",
        "",
    ]
    (output_dir / "EXP029_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/behavior_feedback.yaml")
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), default="pilot")
    parser.add_argument("--telemetry", action="append", default=None)
    args = parser.parse_args()
    payload = run(load_config(args.config), args.profile, args.telemetry)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
