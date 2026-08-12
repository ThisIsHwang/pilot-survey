from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from stackpilot.query_credit_common import atomic_write_json, load_config, markdown_table, read_jsonl

EXPERIMENT_ID = "EXP-056"
METHODS = ("outcome", "doc-to-action", "alias-normalized", "shuffled-doc")
METRICS = ("observed_support_title_recall", "f1", "search_count", "protocol_failure")


def discover(paths: Sequence[str] | None, cfg: dict[str, Any], profile: str) -> list[Path]:
    patterns = list(paths or [])
    if not patterns:
        environment = os.environ.get("QUERY_CREDIT_RESULTS", "").strip()
        if environment:
            patterns = [part for part in environment.split(os.pathsep) if part]
        else:
            patterns = [
                str(Path("work/experiments/EXP-055/results") / "**" / "episodes.jsonl"),
                str(Path("work/experiments/EXP-055/results") / "**" / "*.jsonl"),
            ]
    output: dict[str, Path] = {}
    for pattern in patterns:
        for raw in glob.glob(os.path.expanduser(pattern), recursive=True):
            path = Path(raw).resolve()
            if path.is_file():
                output[str(path)] = path
    return [output[key] for key in sorted(output)]


def parse_variant(value: str) -> tuple[str, str]:
    text = str(value)
    for backend in ("bm25", "e5"):
        prefix = backend + "-"
        if text.startswith(prefix):
            method = text[len(prefix) :]
            if method in METHODS:
                return backend, method
    raise RuntimeError(f"Unknown query-credit variant: {text}")


def load_episodes(paths: Sequence[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        for row in read_jsonl([path]):
            try:
                source, method = parse_variant(row.get("variant", ""))
            except RuntimeError:
                continue
            copy = dict(row)
            copy["source_backend"] = source
            copy["method"] = method
            copy["result_path"] = str(path)
            rows.append(copy)
    if not rows:
        raise RuntimeError("No EXP-055 endpoint episodes were found")
    frame = pd.DataFrame(rows)
    required = {"seed", "question_id", "dataset", "backend", "topk", *METRICS}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Endpoint episodes miss {sorted(missing)}")
    for metric in METRICS:
        frame[metric] = pd.to_numeric(frame[metric], errors="raise")
    return frame


def hierarchical_bootstrap(
    paired: pd.DataFrame,
    *,
    value_column: str,
    samples: int,
    seed: int,
) -> dict[str, float]:
    groups = {int(seed_value): group.copy() for seed_value, group in paired.groupby("seed")}
    seeds = sorted(groups)
    if not seeds:
        raise RuntimeError("No seed groups for endpoint bootstrap")
    observed = float(np.mean([group[value_column].mean() for group in groups.values()]))
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    question_arrays = {}
    for seed_value, group in groups.items():
        question_arrays[seed_value] = (
            group.groupby(group["question_id"].astype(str))[value_column]
            .mean()
            .to_numpy(dtype=np.float64)
        )
    for draw in range(samples):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        seed_means = []
        for seed_value in sampled_seeds:
            values = question_arrays[int(seed_value)]
            sampled = values[rng.integers(0, len(values), size=len(values))]
            seed_means.append(float(sampled.mean()))
        draws[draw] = float(np.mean(seed_means))
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "estimate": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "n_seeds": float(len(seeds)),
        "n_questions": float(paired["question_id"].nunique()),
        "n_rows": float(len(paired)),
    }


def paired_contrast(
    frame: pd.DataFrame,
    *,
    left: str,
    right: str,
    metric: str,
    samples: int,
    seed: int,
) -> dict[str, float]:
    index = ["seed", "question_id", "dataset", "source_backend", "backend", "topk"]
    pivot = frame.pivot_table(index=index, columns="method", values=metric, aggfunc="mean").reset_index()
    paired = pivot.dropna(subset=[left, right]).copy()
    paired["contrast"] = paired[left] - paired[right]
    return hierarchical_bootstrap(paired, value_column="contrast", samples=samples, seed=seed)


def _read_decision(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run(cfg: dict[str, Any], profile: str, provided: Sequence[str] | None = None) -> dict[str, Any]:
    paths = discover(provided, cfg, profile)
    frame = load_episodes(paths)
    frame = frame[frame["topk"] == int(cfg["training"]["topk"])].copy()
    means = (
        frame.groupby(["source_backend", "method", "backend"], as_index=False)
        .agg(
            support_recall=("observed_support_title_recall", "mean"),
            answer_f1=("f1", "mean"),
            search_count=("search_count", "mean"),
            protocol_failure=("protocol_failure", "mean"),
            questions=("question_id", "nunique"),
            seeds=("seed", "nunique"),
        )
    )
    samples = int(cfg["profiles"][profile]["bootstrap_samples"])
    contrasts = []
    pairs = (
        ("doc-to-action", "outcome"),
        ("alias-normalized", "outcome"),
        ("alias-normalized", "doc-to-action"),
        ("shuffled-doc", "doc-to-action"),
    )
    counter = 0
    for source in ("bm25", "e5"):
        source_frame = frame[frame["source_backend"] == source]
        for target in sorted(source_frame["backend"].astype(str).unique()):
            subset = source_frame[source_frame["backend"].astype(str) == target]
            for left, right in pairs:
                for metric in METRICS:
                    try:
                        effect = paired_contrast(
                            subset,
                            left=left,
                            right=right,
                            metric=metric,
                            samples=samples,
                            seed=56000 + counter,
                        )
                    except RuntimeError:
                        counter += 1
                        continue
                    contrasts.append(
                        {
                            "source_backend": source,
                            "target_backend": target,
                            "contrast": f"{left}-minus-{right}",
                            "metric": metric,
                            **effect,
                        }
                    )
                    counter += 1
    output_dir = Path(cfg["work_dir"]).resolve() / "reports" / profile / EXPERIMENT_ID
    output_dir.mkdir(parents=True, exist_ok=True)
    means.to_csv(output_dir / "endpoint_means.csv", index=False)
    pd.DataFrame(contrasts).to_csv(output_dir / "endpoint_contrasts.csv", index=False)
    gate = cfg["gates"][EXPERIMENT_ID]
    normalized = [
        row
        for row in contrasts
        if row["contrast"] == "alias-normalized-minus-outcome"
        and row["target_backend"] in {"bm25", "e5"}
    ]
    support = [row for row in normalized if row["metric"] == "observed_support_title_recall"]
    f1 = [row for row in normalized if row["metric"] == "f1"]
    searches = [row for row in normalized if row["metric"] == "search_count"]
    protocol = [row for row in normalized if row["metric"] == "protocol_failure"]
    endpoint_go = bool(
        len(support) >= 4
        and all(float(row["estimate"]) >= float(gate["minimum_alias_normalized_support_gain"]) and float(row["ci_low"]) > 0 for row in support)
        and len(f1) >= 4
        and all(float(row["estimate"]) >= -float(gate["maximum_answer_f1_regression"]) for row in f1)
        and len(searches) >= 4
        and all(float(row["estimate"]) <= float(gate["maximum_search_increase"]) for row in searches)
        and len(protocol) >= 4
        and all(float(row["estimate"]) <= 0.0 for row in protocol)
    )
    previous = {}
    for experiment in ("EXP-051", "EXP-052", "EXP-053", "EXP-054"):
        decision = _read_decision(Path(cfg["work_dir"]) / "reports" / profile / experiment / "decision.json")
        previous[experiment] = decision
    prerequisites = all(previous.get(exp) and bool(previous[exp].get("go", previous[exp].get("primary_go", False))) for exp in ("EXP-051", "EXP-052", "EXP-053", "EXP-054"))
    if prerequisites and endpoint_go:
        disposition = "ALIAS-CALIBRATED-CREDIT-GO"
    elif previous.get("EXP-051") and bool(previous["EXP-051"].get("primary_go", False)):
        disposition = "ANALYSIS-PAPER-ONLY"
    else:
        disposition = "NO-GO"
    payload = {
        "schema": 1,
        "experiment_id": EXPERIMENT_ID,
        "profile": profile,
        "episode_files": len(paths),
        "endpoint_go": endpoint_go,
        "prerequisites_go": prerequisites,
        "disposition": disposition,
        "go": disposition == "ALIAS-CALIBRATED-CREDIT-GO",
    }
    atomic_write_json(output_dir / "decision.json", payload)
    report = [
        "# EXP-056 Held-out endpoint and paper gate",
        "",
        f"Profile: `{profile}`. All methods retain the original Search-R1 top-k=3 observation interface. Only the query-span credit signal differs.",
        "",
        "## Endpoint means",
        "",
        markdown_table(means.to_dict("records")),
        "",
        "## Paired contrasts",
        "",
        markdown_table(contrasts),
        "",
        f"Disposition: **{disposition}**.",
        "",
    ]
    (output_dir / "EXP056_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/query_credit.yaml")
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), default="pilot")
    parser.add_argument("--result", action="append", default=None)
    args = parser.parse_args()
    print(json.dumps(run(load_config(args.config), args.profile, args.result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
