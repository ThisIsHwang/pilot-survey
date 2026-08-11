from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from stackpilot.credit_routing_common import (
    atomic_write_json,
    discover_paths,
    env_patterns,
    load_config,
    markdown_table,
    read_jsonl,
    stable_seed,
)

TRAIN_EXPERIMENT = "EXP-047"
ENDPOINT_EXPERIMENT = "EXP-048"
GATE_EXPERIMENT = "EXP-049"
METHODS = ("outcome-only", "action-route", "observation-route", "both")
METRICS = (
    "observed_support_title_recall",
    "f1",
    "search_count",
    "protocol_failure",
)


def parse_variant(value: str) -> tuple[str, str]:
    text = str(value)
    for backend in ("bm25", "e5"):
        prefix = backend + "-"
        if text.startswith(prefix):
            method = text[len(prefix) :]
            if method not in METHODS:
                raise RuntimeError(f"Unknown credit-routing method: {method}")
            return backend, method
    raise RuntimeError(f"Credit-routing variant must start with bm25- or e5-: {text}")


def load_episodes(paths: Sequence[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for raw in read_jsonl(path):
            if str(raw.get("experiment_id", ENDPOINT_EXPERIMENT)) != ENDPOINT_EXPERIMENT:
                continue
            source, method = parse_variant(str(raw.get("variant", "")))
            row = dict(raw)
            row["source_backend"] = source
            row["method"] = method
            row["source_path"] = str(path)
            rows.append(row)
    if not rows:
        raise RuntimeError("No EXP-048 credit-routing endpoint episodes were found")
    frame = pd.DataFrame(rows)
    required = {"question_id", "backend", "seed", "topk", *METRICS}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Endpoint rows miss {sorted(missing)}")
    for column in METRICS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column]).all():
            raise RuntimeError(f"Non-finite endpoint metric: {column}")
    frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)
    return frame


def paired_pivot(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    index = ["seed", "question_id", "backend", "topk"]
    if "dataset" in frame.columns:
        index.insert(2, "dataset")
    return frame.pivot_table(index=index, columns="method", values=metric, aggfunc="mean").reset_index()


def hierarchical_bootstrap(
    frame: pd.DataFrame,
    *,
    metric: str,
    required: Sequence[str],
    statistic: Callable[[pd.Series], float],
    samples: int,
    seed: int,
) -> dict[str, float]:
    pivot = paired_pivot(frame, metric)
    missing = [name for name in required if name not in pivot.columns]
    if missing:
        raise RuntimeError(f"Missing methods for {metric}: {missing}")
    paired = pivot.dropna(subset=list(required)).copy()
    if paired.empty:
        raise RuntimeError(f"No paired rows for {metric}")
    paired["contrast"] = paired.apply(statistic, axis=1)
    seed_groups = {int(value): group for value, group in paired.groupby("seed")}
    seed_values = sorted(seed_groups)
    observed = float(
        np.mean([group["contrast"].mean() for group in seed_groups.values()])
    )
    number_of_draws = int(samples)
    rng = np.random.default_rng(seed)
    seed_draw_means = np.empty(
        (number_of_draws, len(seed_values)), dtype=np.float64
    )
    chunk_size = 512
    for seed_position, seed_value in enumerate(seed_values):
        values = (
            seed_groups[seed_value]
            .groupby(seed_groups[seed_value]["question_id"].astype(str))["contrast"]
            .mean()
            .to_numpy(dtype=np.float64)
        )
        if values.size == 0:
            raise RuntimeError(f"Seed {seed_value} has no question clusters")
        for offset in range(0, number_of_draws, chunk_size):
            stop = min(number_of_draws, offset + chunk_size)
            indices = rng.integers(
                0, values.size, size=(stop - offset, values.size)
            )
            seed_draw_means[offset:stop, seed_position] = values[indices].mean(
                axis=1
            )
    sampled_seed_positions = rng.integers(
        0, len(seed_values), size=(number_of_draws, len(seed_values))
    )
    draws = seed_draw_means[
        np.arange(number_of_draws)[:, None], sampled_seed_positions
    ].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "estimate": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "n_seeds": float(len(seed_values)),
        "n_questions": float(paired["question_id"].nunique()),
        "n_rows": float(len(paired)),
    }


def contrast_specs() -> list[tuple[str, Sequence[str], Callable[[pd.Series], float]]]:
    return [
        (
            "action_routing_main",
            METHODS,
            lambda row: 0.5 * (
                (row["action-route"] - row["outcome-only"])
                + (row["both"] - row["observation-route"])
            ),
        ),
        (
            "observation_routing_main",
            METHODS,
            lambda row: 0.5 * (
                (row["observation-route"] - row["outcome-only"])
                + (row["both"] - row["action-route"])
            ),
        ),
        (
            "action_x_observation_interaction",
            METHODS,
            lambda row: (row["both"] - row["action-route"])
            - (row["observation-route"] - row["outcome-only"]),
        ),
        (
            "observation_minus_action",
            ("observation-route", "action-route"),
            lambda row: row["observation-route"] - row["action-route"],
        ),
        (
            "observation_minus_outcome",
            ("observation-route", "outcome-only"),
            lambda row: row["observation-route"] - row["outcome-only"],
        ),
        (
            "action_minus_outcome",
            ("action-route", "outcome-only"),
            lambda row: row["action-route"] - row["outcome-only"],
        ),
        (
            "both_minus_outcome",
            ("both", "outcome-only"),
            lambda row: row["both"] - row["outcome-only"],
        ),
    ]


def optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def run(cfg: dict[str, Any], profile_name: str, *, inputs: Sequence[str] | None = None) -> dict[str, Any]:
    patterns = env_patterns("CREDIT_ROUTING_RESULTS", cfg["source"]["episode_globs"], inputs)
    paths = discover_paths(patterns, suffixes=(".jsonl",))
    frame = load_episodes(paths)
    expected_topk = int(cfg["labeling"]["observation_k"])
    frame = frame[frame["topk"] == expected_topk].copy()
    if frame.empty:
        raise RuntimeError(f"No endpoint rows use top-k {expected_topk}")

    output_root = Path(cfg["work_dir"]).resolve() / "reports" / profile_name
    train_dir = output_root / TRAIN_EXPERIMENT
    endpoint_dir = output_root / ENDPOINT_EXPERIMENT
    gate_dir = output_root / GATE_EXPERIMENT
    for path in (train_dir, endpoint_dir, gate_dir):
        path.mkdir(parents=True, exist_ok=True)

    means = frame.groupby(["source_backend", "method", "backend"], as_index=False).agg(
        support_recall=("observed_support_title_recall", "mean"),
        answer_f1=("f1", "mean"),
        search_count=("search_count", "mean"),
        protocol_failure=("protocol_failure", "mean"),
        seeds=("seed", "nunique"),
        questions=("question_id", "nunique"),
    )
    means.to_csv(endpoint_dir / "variant_means.csv", index=False)
    frame.to_csv(endpoint_dir / "endpoint_episodes.csv", index=False)

    samples = int(cfg["profiles"][profile_name]["bootstrap_samples"])
    contrast_rows: list[dict[str, Any]] = []
    for source in ("bm25", "e5"):
        source_rows = frame[frame["source_backend"] == source]
        for target in sorted(source_rows["backend"].astype(str).unique()):
            direction = source_rows[source_rows["backend"].astype(str) == target]
            for contrast_name, required, statistic in contrast_specs():
                for metric in METRICS:
                    try:
                        estimate = hierarchical_bootstrap(
                            direction,
                            metric=metric,
                            required=required,
                            statistic=statistic,
                            samples=samples,
                            seed=stable_seed("credit-routing-report", source, target, contrast_name, metric),
                        )
                    except RuntimeError:
                        continue
                    contrast_rows.append(
                        {
                            "source_backend": source,
                            "target_backend": target,
                            "contrast": contrast_name,
                            "metric": metric,
                            **estimate,
                        }
                    )
    contrasts = pd.DataFrame(contrast_rows)
    if contrasts.empty:
        raise RuntimeError("No complete credit-routing factorial contrast was available")
    contrasts.to_csv(endpoint_dir / "factorial_contrasts.csv", index=False)

    gate = cfg["gates"][ENDPOINT_EXPERIMENT]
    base_targets = contrasts[contrasts["target_backend"].isin(["bm25", "e5"])]
    def contrast_metric(name: str, metric: str) -> pd.DataFrame:
        return base_targets[
            (base_targets["contrast"] == name)
            & (base_targets["metric"] == metric)
        ]

    observation_over_action = contrast_metric(
        "observation_minus_action", "observed_support_title_recall"
    )
    observation_over_outcome = contrast_metric(
        "observation_minus_outcome", "observed_support_title_recall"
    )
    action_over_outcome = contrast_metric(
        "action_minus_outcome", "observed_support_title_recall"
    )
    safety_frames = []
    for contrast_name in (
        "observation_minus_action",
        "observation_minus_outcome",
    ):
        for metric in ("f1", "search_count", "protocol_failure"):
            one = contrast_metric(contrast_name, metric).copy()
            one["safety_contrast"] = contrast_name
            safety_frames.append(one)
    safety = pd.concat(safety_frames, ignore_index=True)
    f1 = safety[safety["metric"] == "f1"]
    searches = safety[safety["metric"] == "search_count"]
    protocol = safety[safety["metric"] == "protocol_failure"]
    endpoint_go = bool(
        len(observation_over_action) >= 4
        and (
            observation_over_action["estimate"]
            >= float(gate["minimum_observation_over_action_support_gain"])
        ).all()
        and (observation_over_action["ci_low"] > 0.0).all()
        and len(observation_over_outcome) >= 4
        and (
            observation_over_outcome["estimate"]
            >= float(gate["minimum_observation_over_outcome_support_gain"])
        ).all()
        and (observation_over_outcome["ci_low"] > 0.0).all()
        and len(action_over_outcome) >= 4
        and (
            action_over_outcome["estimate"]
            <= float(gate["maximum_action_over_outcome_support_gain"])
        ).all()
        and len(f1) >= 8
        and (f1["estimate"] >= -float(gate["maximum_answer_f1_regression"])).all()
        and len(searches) >= 8
        and (searches["estimate"] <= float(gate["maximum_search_call_increase"])).all()
        and len(protocol) >= 8
        and (protocol["estimate"] <= float(gate["maximum_protocol_failure_increase"])).all()
    )
    endpoint_decision = {
        "schema": 1,
        "experiment_id": ENDPOINT_EXPERIMENT,
        "profile": profile_name,
        "episode_files": len(paths),
        "methods": sorted(frame["method"].unique()),
        "sources": sorted(frame["source_backend"].unique()),
        "targets": sorted(frame["backend"].astype(str).unique()),
        "go": endpoint_go,
    }
    atomic_write_json(endpoint_dir / "decision.json", endpoint_decision)

    training_inventory = frame.groupby(["source_backend", "method"], as_index=False).agg(
        seeds=("seed", "nunique"),
        evaluated_questions=("question_id", "nunique"),
    )
    training_complete = bool(
        set(training_inventory["method"]) == set(METHODS)
        and set(training_inventory["source_backend"]) == {"bm25", "e5"}
    )
    training_decision = {
        "schema": 1,
        "experiment_id": TRAIN_EXPERIMENT,
        "profile": profile_name,
        "factorial_cells_observed": int(len(training_inventory)),
        "complete_design": training_complete,
        "go": training_complete,
    }
    atomic_write_json(train_dir / "decision.json", training_decision)
    training_inventory.to_csv(train_dir / "training_inventory.csv", index=False)

    exp45 = optional_json(output_root / "EXP-045" / "decision.json")
    exp46 = optional_json(output_root / "EXP-046" / "decision.json")
    exp34 = optional_json(
        Path("work/behavior_feedback/reports").resolve()
        / profile_name
        / "EXP-034"
        / "decision.json"
    )
    mismatch = None
    if exp34 and isinstance(exp34.get("all_state_disagreement"), dict):
        mismatch = float(exp34["all_state_disagreement"].get("estimate", 0.0))
    mismatch_go = mismatch is not None and mismatch >= float(
        cfg["gates"][GATE_EXPERIMENT]["minimum_nonconservation_evidence"]
    )
    if bool(exp45 and exp45.get("go")) and bool(exp46 and exp46.get("go")) and endpoint_go and mismatch_go:
        disposition = "CREDIT-ROUTING-METHOD-GO"
    elif mismatch_go and bool(exp45 and exp45.get("go")):
        disposition = "ANALYSIS-PAPER-ONLY"
    elif exp45 is None or exp46 is None:
        disposition = "PREREQUISITE-PENDING"
    else:
        disposition = "NO-GO"
    gate_decision = {
        "schema": 1,
        "experiment_id": GATE_EXPERIMENT,
        "profile": profile_name,
        "exp045_go": bool(exp45 and exp45.get("go")),
        "exp046_go": bool(exp46 and exp46.get("go")),
        "exp048_go": endpoint_go,
        "document_action_mismatch": mismatch,
        "mismatch_go": mismatch_go,
        "disposition": disposition,
        "go": disposition == "CREDIT-ROUTING-METHOD-GO",
    }
    atomic_write_json(gate_dir / "decision.json", gate_decision)

    (train_dir / "EXP047_REPORT.md").write_text(
        "\n".join(
            [
                "# EXP-047 CTU credit-routing 2×2 training factorial",
                "",
                f"Profile: `{profile_name}`. The same learned document-utility score is crossed as an action-reward factor and an observation-selection factor.",
                "",
                markdown_table(training_inventory),
                "",
                f"Complete design observed in endpoint artifacts: **{training_complete}**.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (endpoint_dir / "EXP048_REPORT.md").write_text(
        "\n".join(
            [
                "# EXP-048 Credit-routing endpoint confirmation",
                "",
                f"Profile: `{profile_name}`. All cells retrieve top-8 once and expose exactly three documents. Only the location of the shared utility signal differs.",
                "",
                "## Variant means",
                "",
                markdown_table(means),
                "",
                "## Paired factorial contrasts",
                "",
                markdown_table(contrasts),
                "",
                f"Decision: **{'GO' if endpoint_go else 'NO-GO'}**.",
                "",
                "The method claim requires observation routing to beat both outcome-only training and action-side routing in every BM25/E5 seen and cross direction, while action-side routing alone provides no material support-recall gain and all observation contrasts preserve answer, search-call, and protocol metrics.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (gate_dir / "EXP049_REPORT.md").write_text(
        "\n".join(
            [
                "# EXP-049 Credit-routing paper gate",
                "",
                f"Disposition: **{disposition}**.",
                "",
                "A method-paper GO requires a prevalent document–action mismatch, a held-out utility estimator that beats rank, and an end-to-end advantage for routing utility to observation selection rather than to the producing action.",
                "",
                "```json",
                json.dumps(gate_decision, indent=2, sort_keys=True),
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return gate_decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Report the CTU credit-routing factorial.")
    parser.add_argument("--config", default="configs/credit_routing.yaml")
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), default="pilot")
    parser.add_argument("--result", action="append", default=None)
    args = parser.parse_args()
    payload = run(load_config(args.config), args.profile, inputs=args.result)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
