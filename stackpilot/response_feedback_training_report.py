from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from stackpilot.behavior_quotient_common import (
    atomic_write_json,
    load_config,
    markdown_table,
    read_jsonl,
)

EXPERIMENT_ID = "EXP-031"
METHODS = (
    "standard",
    "iid-surface",
    "posthoc-surface",
    "feedback-surface",
    "iid-quotient",
    "posthoc-quotient",
    "feedback-quotient",
)
PRIMARY_METRICS = (
    "observed_support_title_recall",
    "f1",
    "search_count",
    "protocol_failure",
)


def discover_episode_paths(
    cfg: dict[str, Any], provided: Sequence[str] | None = None
) -> list[Path]:
    patterns = list(provided or [])
    if not patterns:
        environment = os.environ.get("BEHAVIOR_FEEDBACK_RESULTS", "").strip()
        if environment:
            patterns = [
                part
                for part in environment.replace("\n", os.pathsep).split(os.pathsep)
                if part
            ]
        else:
            patterns = [str(value) for value in cfg["source"]["episode_globs"]]
    output: dict[str, Path] = {}
    for pattern in patterns:
        for raw in glob.glob(os.path.expanduser(pattern), recursive=True):
            path = Path(raw).resolve()
            if path.is_file():
                output[str(path)] = path
    return [output[key] for key in sorted(output)]


def parse_variant(value: str) -> tuple[str, str]:
    variant = str(value)
    for backend in ("bm25", "e5"):
        prefix = backend + "-"
        if variant.startswith(prefix):
            method = variant[len(prefix) :]
            if method not in METHODS:
                raise RuntimeError(
                    f"Unknown response-feedback method: {method!r}"
                )
            return backend, method
    raise RuntimeError(
        f"Variant must start with bm25- or e5-: {variant!r}"
    )


def load_episodes(paths: Sequence[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for row in read_jsonl(path):
            source, method = parse_variant(str(row.get("variant", "")))
            copy = dict(row)
            copy["source_backend"] = source
            copy["method"] = method
            copy["result_path"] = str(path)
            rows.append(copy)
    if not rows:
        raise RuntimeError(
            "No response-feedback evaluation episodes were found"
        )
    frame = pd.DataFrame(rows)
    required = {
        "question_id",
        "dataset",
        "backend",
        "topk",
        "seed",
        "variant",
        *PRIMARY_METRICS,
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Evaluation episodes miss {sorted(missing)}")
    for column in PRIMARY_METRICS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column]).all():
            raise RuntimeError(f"Non-finite evaluation metric: {column}")
    return frame


def _paired_cell_frame(rows: pd.DataFrame, metric: str) -> pd.DataFrame:
    index = ["seed", "question_id", "dataset", "backend", "topk"]
    pivot = rows.pivot_table(
        index=index, columns="method", values=metric, aggfunc="mean"
    )
    return pivot.reset_index()


def _bootstrap_contrast(
    rows: pd.DataFrame,
    *,
    metric: str,
    statistic: Callable[[pd.Series], float],
    required: Sequence[str],
    samples: int,
    seed: int,
) -> dict[str, float]:
    pivot = _paired_cell_frame(rows, metric)
    missing = [method for method in required if method not in pivot.columns]
    if missing:
        raise RuntimeError(f"Missing methods {missing} for {metric}")
    paired = pivot.dropna(subset=list(required)).copy()
    if paired.empty:
        raise RuntimeError(f"No paired evaluation rows for {metric}")
    paired["contrast_value"] = paired.apply(statistic, axis=1)
    groups = {
        int(seed_value): group.copy()
        for seed_value, group in paired.groupby("seed")
    }
    seeds = sorted(groups)
    observed = float(
        np.mean(
            [group["contrast_value"].mean() for group in groups.values()]
        )
    )
    rng = np.random.default_rng(seed)
    number_of_draws = int(samples)
    seed_draw_means = np.empty(
        (number_of_draws, len(seeds)), dtype=np.float64
    )
    chunk_size = 512
    for seed_position, seed_value in enumerate(seeds):
        group = groups[seed_value]
        question_values = (
            group.groupby(group["question_id"].astype(str))["contrast_value"]
            .mean()
            .to_numpy(dtype=np.float64)
        )
        if question_values.size == 0:
            raise RuntimeError(
                f"Seed {seed_value} has no question clusters"
            )
        for offset in range(0, number_of_draws, chunk_size):
            stop = min(number_of_draws, offset + chunk_size)
            indices = rng.integers(
                0,
                question_values.size,
                size=(stop - offset, question_values.size),
            )
            seed_draw_means[offset:stop, seed_position] = question_values[
                indices
            ].mean(axis=1)
    sampled_seed_positions = rng.integers(
        0, len(seeds), size=(number_of_draws, len(seeds))
    )
    draws = seed_draw_means[
        np.arange(number_of_draws)[:, None], sampled_seed_positions
    ].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "estimate": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "n_seeds": float(len(seeds)),
        "n_questions": float(paired["question_id"].nunique()),
        "n_rows": float(len(paired)),
    }


def contrast_specs() -> list[
    tuple[str, Sequence[str], Callable[[pd.Series], float]]
]:
    return [
        (
            "feedback_sampling_main",
            (
                "feedback-surface",
                "iid-surface",
                "feedback-quotient",
                "iid-quotient",
            ),
            lambda row: 0.5
            * (
                (row["feedback-surface"] - row["iid-surface"])
                + (row["feedback-quotient"] - row["iid-quotient"])
            ),
        ),
        (
            "posthoc_sampling_main",
            (
                "posthoc-surface",
                "iid-surface",
                "posthoc-quotient",
                "iid-quotient",
            ),
            lambda row: 0.5
            * (
                (row["posthoc-surface"] - row["iid-surface"])
                + (row["posthoc-quotient"] - row["iid-quotient"])
            ),
        ),
        (
            "normalization_main",
            (
                "iid-quotient",
                "iid-surface",
                "posthoc-quotient",
                "posthoc-surface",
                "feedback-quotient",
                "feedback-surface",
            ),
            lambda row: (
                (row["iid-quotient"] - row["iid-surface"])
                + (row["posthoc-quotient"] - row["posthoc-surface"])
                + (
                    row["feedback-quotient"]
                    - row["feedback-surface"]
                )
            )
            / 3.0,
        ),
        (
            "feedback_x_quotient_interaction",
            (
                "feedback-quotient",
                "iid-quotient",
                "feedback-surface",
                "iid-surface",
            ),
            lambda row: (
                row["feedback-quotient"] - row["iid-quotient"]
            )
            - (row["feedback-surface"] - row["iid-surface"]),
        ),
        (
            "posthoc_x_quotient_interaction",
            (
                "posthoc-quotient",
                "iid-quotient",
                "posthoc-surface",
                "iid-surface",
            ),
            lambda row: (
                row["posthoc-quotient"] - row["iid-quotient"]
            )
            - (row["posthoc-surface"] - row["iid-surface"]),
        ),
        (
            "joint_feedback_quotient_minus_standard",
            ("feedback-quotient", "standard"),
            lambda row: row["feedback-quotient"] - row["standard"],
        ),
    ]


def report(
    cfg: dict[str, Any],
    profile_name: str,
    patterns: Sequence[str] | None = None,
) -> dict[str, Any]:
    paths = discover_episode_paths(cfg, patterns)
    frame = load_episodes(paths)
    expected_topk = int(cfg["training"]["topk"])
    frame = frame[frame["topk"] == expected_topk].copy()
    if frame.empty:
        raise RuntimeError(f"No evaluation rows use top-k {expected_topk}")

    output_dir = (
        Path(cfg["work_dir"]).resolve() / "reports" / profile_name / EXPERIMENT_ID
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "evaluation_episodes.csv", index=False)
    means = frame.groupby(
        ["source_backend", "method", "backend"], as_index=False
    ).agg(
        observed_support_recall=(
            "observed_support_title_recall",
            "mean",
        ),
        answer_f1=("f1", "mean"),
        search_count=("search_count", "mean"),
        protocol_failure=("protocol_failure", "mean"),
        questions=("question_id", "nunique"),
    )
    means.to_csv(output_dir / "variant_means.csv", index=False)

    rows: list[dict[str, Any]] = []
    samples = int(cfg["profiles"][profile_name]["bootstrap_samples"])
    for source in ("bm25", "e5"):
        source_rows = frame[frame["source_backend"] == source]
        for target in sorted(source_rows["backend"].astype(str).unique()):
            direction = source_rows[
                source_rows["backend"].astype(str) == str(target)
            ]
            for contrast_index, (
                name,
                required,
                statistic,
            ) in enumerate(contrast_specs()):
                for metric_index, metric in enumerate(PRIMARY_METRICS):
                    try:
                        effect = _bootstrap_contrast(
                            direction,
                            metric=metric,
                            statistic=statistic,
                            required=required,
                            samples=samples,
                            seed=(
                                31031
                                + len(rows)
                                + contrast_index
                                + metric_index
                            ),
                        )
                    except RuntimeError:
                        continue
                    rows.append(
                        {
                            "source_backend": source,
                            "target_backend": str(target),
                            "contrast": name,
                            "metric": metric,
                            **effect,
                        }
                    )
    contrasts = pd.DataFrame(rows)
    if contrasts.empty:
        raise RuntimeError(
            "No complete 3x2 factorial contrast could be computed"
        )
    contrasts.to_csv(output_dir / "factorial_contrasts.csv", index=False)

    gate = cfg["gates"][EXPERIMENT_ID]
    feedback = contrasts[
        contrasts["contrast"] == "feedback_sampling_main"
    ]
    base_targets = feedback[
        feedback["target_backend"].isin(["bm25", "e5"])
    ]
    support = base_targets[
        base_targets["metric"] == "observed_support_title_recall"
    ]
    f1 = base_targets[base_targets["metric"] == "f1"]
    searches = base_targets[base_targets["metric"] == "search_count"]
    protocol = base_targets[
        base_targets["metric"] == "protocol_failure"
    ]
    go = bool(
        len(support) >= 4
        and (
            support["estimate"] >= float(gate["minimum_support_gain"])
        ).all()
        and (support["ci_low"] > 0).all()
        and len(f1) >= 4
        and (
            f1["estimate"]
            >= -float(gate["maximum_answer_f1_regression"])
        ).all()
        and len(searches) >= 4
        and (
            searches["estimate"]
            <= float(gate["maximum_search_call_increase"])
        ).all()
        and len(protocol) >= 4
        and (protocol["estimate"] <= 0.0).all()
    )
    payload = {
        "schema": 1,
        "experiment_id": EXPERIMENT_ID,
        "profile": profile_name,
        "evaluation_files": len(paths),
        "methods": sorted(frame["method"].unique()),
        "sources": sorted(frame["source_backend"].unique()),
        "targets": sorted(frame["backend"].astype(str).unique()),
        "go": go,
    }
    atomic_write_json(output_dir / "decision.json", payload)
    report_lines = [
        "# EXP-031 Sampling × normalization factorial",
        "",
        f"Profile: `{profile_name}`. The six controlled cells form a 3×2 design: IID, post-hoc behavior-balanced, or response-feedback sampling crossed with surface or behavior-quotient advantage normalization.",
        "",
        "## Evaluation means",
        "",
        markdown_table(means),
        "",
        "## Paired factorial effects",
        "",
        markdown_table(contrasts),
        "",
        f"Decision: **{'GO' if go else 'NO-GO'}**.",
        "",
        "A GO requires the feedback-sampling main effect to improve actual support recall in every BM25/E5 seen and cross direction under identical retrieval-call budgets, without answer, search-count, or protocol regression.",
        "",
    ]
    (output_dir / "EXP031_REPORT.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/behavior_feedback.yaml")
    parser.add_argument(
        "--profile", choices=("smoke", "pilot", "full"), default="pilot"
    )
    parser.add_argument("--result", action="append", default=None)
    args = parser.parse_args()
    payload = report(
        load_config(args.config), args.profile, args.result
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
