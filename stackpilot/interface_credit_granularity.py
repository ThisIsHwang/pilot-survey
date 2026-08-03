from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stackpilot.interface_causality_common import (
    atomic_write_json,
    balanced_state_subset,
    cluster_bootstrap,
    group_candidates,
    load_config,
    load_state_results,
    markdown_table,
    source_patterns,
)

EXPERIMENT_ID = "EXP-021"


def _factual(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next(
        (
            row
            for row in candidates
            if str(row.get("origin", "")) == "factual"
            or str(row.get("style", "")) == "factual"
        ),
        None,
    )


def state_credit_rows(result: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    state = result["state"]
    candidates = [
        dict(row)
        for row in result["candidates"]
        if int(row.get("protocol_failure", 0)) == 0
    ]
    classes = group_candidates(
        state,
        candidates,
        mode=str(cfg["credit_granularity"]["behavior_signature"]),
    )
    factual = _factual(candidates)
    if factual is None:
        return []
    class_index = next(
        (
            index
            for index, members in enumerate(classes)
            if any(str(row["candidate_id"]) == str(factual["candidate_id"]) for row in members)
        ),
        None,
    )
    if class_index is None:
        return []
    factual_class = classes[class_index]
    class_tqe = float(np.mean([float(row.get("support_tqe", 0.0)) for row in factual_class]))
    class_composite = float(np.mean([float(row.get("composite_tqe", 0.0)) for row in factual_class]))
    class_direct = float(np.mean([float(row.get("immediate_support_gain", 0.0)) for row in factual_class]))
    class_final = float(np.mean([float(row.get("final_support_recall", 0.0)) for row in factual_class]))
    class_size = len(factual_class)
    rows = []
    for candidate in candidates:
        candidate_class_index = next(
            index
            for index, members in enumerate(classes)
            if any(str(row["candidate_id"]) == str(candidate["candidate_id"]) for row in members)
        )
        candidate_class = classes[candidate_class_index]
        candidate_class_tqe = float(
            np.mean([float(row.get("support_tqe", 0.0)) for row in candidate_class])
        )
        rows.append(
            {
                "state_id": str(state["state_id"]),
                "question_id": str(state["question_id"]),
                "backend": str(state["backend"]),
                "dataset": str(state["dataset"]),
                "source_turn": int(state["source_turn"]),
                "candidate_id": str(candidate["candidate_id"]),
                "style": str(candidate.get("style", "")),
                "factual": int(str(candidate["candidate_id"]) == str(factual["candidate_id"])),
                "class_index": candidate_class_index,
                "class_size": len(candidate_class),
                "first_exposure_credit": (
                    float(candidate.get("immediate_support_gain", 0.0))
                    if str(candidate["candidate_id"]) == str(factual["candidate_id"])
                    else 0.0
                ),
                "query_tqe": float(candidate.get("support_tqe", 0.0)),
                "query_composite_tqe": float(candidate.get("composite_tqe", 0.0)),
                "class_tqe": candidate_class_tqe,
                "immediate_gain": float(candidate.get("immediate_support_gain", 0.0)),
                "final_support_recall": float(candidate.get("final_support_recall", 0.0)),
                "factual_class_tqe": class_tqe,
                "factual_class_composite_tqe": class_composite,
                "factual_class_direct": class_direct,
                "factual_class_final": class_final,
                "factual_class_size": class_size,
            }
        )
    return rows


def _spearman(left: list[float], right: list[float]) -> float:
    if len(left) < 2:
        return 0.0
    left_series = pd.Series(left).rank(method="average")
    right_series = pd.Series(right).rank(method="average")
    value = left_series.corr(right_series)
    return 0.0 if pd.isna(value) else float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-021: compare first-exposure, action-level, and class-level query credit.")
    parser.add_argument("--config", default="configs/interface_causality.yaml")
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), default="pilot")
    parser.add_argument("--inputs", nargs="*", default=None)
    parser.add_argument("--document-ctu", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    profile = cfg["profiles"][args.profile]
    results = load_state_results(source_patterns(cfg, args.inputs))
    results = balanced_state_subset(results, int(profile["granularity_states"]))
    rows = [row for result in results for row in state_credit_rows(result, cfg)]
    if not rows:
        raise RuntimeError("No factual-query credit rows were available")
    frame = pd.DataFrame(rows)
    factual = frame[frame["factual"] == 1].copy()
    epsilon = float(cfg["credit_granularity"]["epsilon"])
    factual["first_exposure_overattributes"] = (
        (factual["first_exposure_credit"] > epsilon)
        & (factual["query_tqe"] <= epsilon)
    ).astype(int)
    factual["class_rescues_action"] = (
        (factual["query_tqe"] <= epsilon)
        & (factual["factual_class_tqe"] > epsilon)
    ).astype(int)
    factual["action_class_sign_disagreement"] = (
        np.sign(factual["query_tqe"]) != np.sign(factual["factual_class_tqe"])
    ).astype(int)
    factual["credit_granularity_gap"] = factual["factual_class_tqe"] - factual["query_tqe"]

    document_frame = pd.DataFrame()
    if args.document_ctu:
        document_frame = pd.read_json(args.document_ctu, lines=True)
        required = {"state_id", "document_ctu"}
        if not required.issubset(document_frame.columns):
            raise RuntimeError(f"Document CTU file misses {sorted(required - set(document_frame.columns))}")
        document_summary = document_frame.groupby("state_id", as_index=False)["document_ctu"].max()
        factual = factual.merge(document_summary, on="state_id", how="left")
        factual["document_positive_query_nonpositive"] = (
            (factual["document_ctu"] > epsilon) & (factual["query_tqe"] <= epsilon)
        ).astype(float)

    output_dir = Path(args.output_dir or Path(cfg["work_dir"]) / "reports" / args.profile / "EXP-021").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "candidate_credit_levels.csv", index=False)
    factual.to_csv(output_dir / "factual_credit_granularity.csv", index=False)

    metrics = {
        "first_exposure_query_tqe_spearman": _spearman(
            factual["first_exposure_credit"].tolist(), factual["query_tqe"].tolist()
        ),
        "first_exposure_class_tqe_spearman": _spearman(
            factual["first_exposure_credit"].tolist(), factual["factual_class_tqe"].tolist()
        ),
        "query_class_tqe_spearman": _spearman(
            factual["query_tqe"].tolist(), factual["factual_class_tqe"].tolist()
        ),
        "first_exposure_overattribution_rate": float(factual["first_exposure_overattributes"].mean()),
        "class_rescues_action_rate": float(factual["class_rescues_action"].mean()),
        "action_class_sign_disagreement_rate": float(factual["action_class_sign_disagreement"].mean()),
        "mean_credit_granularity_gap": float(factual["credit_granularity_gap"].mean()),
        "mean_factual_class_size": float(factual["factual_class_size"].mean()),
    }
    if "document_ctu" in factual:
        finite = factual.dropna(subset=["document_ctu"])
        metrics["document_query_tqe_spearman"] = _spearman(
            finite["document_ctu"].tolist(), finite["query_tqe"].tolist()
        )
        metrics["document_positive_query_nonpositive_rate"] = float(
            finite["document_positive_query_nonpositive"].mean()
        )

    over_bootstrap = cluster_bootstrap(
        factual.to_dict("records"),
        cluster_key="state_id",
        statistic=lambda records: float(np.mean([row["first_exposure_overattributes"] for row in records])),
        samples=int(profile["bootstrap_samples"]),
        seed=21101,
    )
    disagreement_bootstrap = cluster_bootstrap(
        factual.to_dict("records"),
        cluster_key="state_id",
        statistic=lambda records: float(np.mean([row["action_class_sign_disagreement"] for row in records])),
        samples=int(profile["bootstrap_samples"]),
        seed=21102,
    )
    gates = cfg["gates"]["EXP-021"]
    go = bool(
        over_bootstrap["estimate"] >= float(gates["minimum_first_exposure_overattribution"])
        and over_bootstrap["ci_low"] > 0.0
        and disagreement_bootstrap["estimate"] >= float(gates["minimum_action_class_disagreement"])
        and disagreement_bootstrap["ci_low"] > 0.0
        and metrics["first_exposure_class_tqe_spearman"]
        <= float(gates["maximum_first_exposure_class_correlation"])
    )
    by_backend = (
        factual.groupby("backend", as_index=False)[
            [
                "first_exposure_overattributes", "class_rescues_action",
                "action_class_sign_disagreement", "credit_granularity_gap",
            ]
        ].mean()
    )
    decision = {
        "experiment_id": EXPERIMENT_ID,
        "profile": args.profile,
        "go": go,
        "metrics": metrics,
        "overattribution_bootstrap": over_bootstrap,
        "sign_disagreement_bootstrap": disagreement_bootstrap,
        "states": int(factual["state_id"].nunique()),
        "document_ctu_attached": bool(args.document_ctu),
    }
    atomic_write_json(output_dir / "decision.json", decision)
    report = [
        "# EXP-021 Credit-granularity audit",
        "",
        f"Profile: `{args.profile}`. The report compares provenance-like first exposure, query-level counterfactual utility, and behavior-class utility on the same intervention states.",
        "",
        "## Backend summary",
        "",
        markdown_table(by_backend),
        "",
        "## Primary metrics",
        "",
        "```text",
        json.dumps(decision, indent=2, sort_keys=True),
        "```",
        "",
        f"Decision: **{'GO' if go else 'NO-GO'}**.",
        "",
        "A GO supports the claim that credit granularity—not merely reward density—is a material modeling choice. Optional document-CTU input extends the same audit across observation, action, and behavior-class levels.",
        "",
    ]
    (output_dir / "EXP021_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
