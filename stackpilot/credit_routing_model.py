from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from stackpilot.credit_routing_common import (
    FEATURE_NAMES,
    apply_standardizer,
    atomic_write_json,
    discover_paths,
    env_patterns,
    fit_ridge,
    fit_standardizer,
    load_config,
    markdown_table,
    matrix_from_feature_rows,
    predict_ridge,
    question_split,
    read_jsonl,
    safe_spearman,
    selection_indices,
    stable_seed,
)

EXPERIMENT_ID = "EXP-046"


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return 0.5
    comparisons = positives[:, None] - negatives[None, :]
    return float((comparisons > 0).mean() + 0.5 * (comparisons == 0).mean())


def ndcg_at_k(truth: np.ndarray, scores: np.ndarray, k: int) -> float:
    k = min(int(k), len(truth))
    if k <= 0:
        return 0.0
    relevance = np.maximum(np.asarray(truth, dtype=np.float64), 0.0)
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="mergesort")[:k]
    ideal = np.argsort(-relevance, kind="mergesort")[:k]
    discounts = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))
    dcg = float(np.sum((2.0 ** relevance[order] - 1.0) * discounts))
    idcg = float(np.sum((2.0 ** relevance[ideal] - 1.0) * discounts))
    return 0.0 if idcg <= 1e-12 else dcg / idcg


def bootstrap_mean(values: Sequence[float], *, samples: int, seed: int) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise RuntimeError("Cannot bootstrap an empty contrast")
    rng = np.random.default_rng(seed)
    chunk = 512
    draws = np.empty(int(samples), dtype=np.float64)
    for offset in range(0, int(samples), chunk):
        stop = min(int(samples), offset + chunk)
        indices = rng.integers(0, len(array), size=(stop - offset, len(array)))
        draws[offset:stop] = array[indices].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "estimate": float(array.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "n": float(len(array)),
    }


def load_frame(paths: Sequence[Path], cfg: dict[str, Any]) -> pd.DataFrame:
    rows = [row for path in paths for row in read_jsonl(path)]
    if not rows:
        raise RuntimeError("No credit-routing utility labels were found")
    frame = pd.DataFrame(rows)
    required = {
        "state_id",
        "question_id",
        "backend",
        "document_rank",
        "document_utility",
        *FEATURE_NAMES,
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Utility label rows miss {sorted(missing)}")
    for column in ("document_rank", "document_utility", *FEATURE_NAMES):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column]).all():
            raise RuntimeError(f"Non-finite label column: {column}")
    frame["split"] = [
        question_split(
            str(question_id),
            salt=str(cfg["estimator"]["split_salt"]),
            train_fraction=float(cfg["estimator"]["train_fraction"]),
            validation_fraction=float(cfg["estimator"]["validation_fraction"]),
        )
        for question_id in frame["question_id"]
    ]
    return frame.reset_index(drop=True)


def fit_artifact(frame: pd.DataFrame, cfg: dict[str, Any], profile_name: str) -> dict[str, Any]:
    train = frame[frame["split"] == "train"].copy()
    if train.empty:
        raise RuntimeError("Question split produced no training rows")
    x_train = matrix_from_feature_rows(train.to_dict("records"))
    mean, scale = fit_standardizer(x_train)
    standardized = apply_standardizer(x_train, mean, scale)
    question_groups = {
        str(question_id): group.index.to_numpy(dtype=np.int64)
        for question_id, group in train.groupby("question_id", sort=True)
    }
    questions = sorted(question_groups)
    seeds = [int(value) for value in cfg["profiles"][profile_name]["estimator_seeds"]]
    weights: list[list[float]] = []
    targets = train["document_utility"].to_numpy(dtype=np.float64)
    for estimator_seed in seeds:
        rng = np.random.default_rng(stable_seed("credit-routing-ridge", estimator_seed))
        sampled_questions = rng.choice(questions, size=len(questions), replace=True)
        indices = np.concatenate([question_groups[str(value)] for value in sampled_questions])
        local_positions = train.index.get_indexer(indices)
        if (local_positions < 0).any():
            raise RuntimeError("Bootstrap question indices drifted")
        one = fit_ridge(
            standardized[local_positions],
            targets[local_positions],
            l2=float(cfg["estimator"]["ridge_l2"]),
        )
        weights.append(one.tolist())
    return {
        "schema": 1,
        "experiment_id": EXPERIMENT_ID,
        "feature_names": list(FEATURE_NAMES),
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "weights": weights,
        "estimator_seeds": seeds,
        "training_questions": len(questions),
        "training_rows": len(train),
        "aggregation": str(cfg["routing"]["action_aggregation"]),
        "upstream_topk": int(cfg["labeling"]["upstream_topk"]),
        "observation_k": int(cfg["labeling"]["observation_k"]),
    }


def predict_frame(frame: pd.DataFrame, artifact: dict[str, Any]) -> np.ndarray:
    matrix = matrix_from_feature_rows(frame.to_dict("records"))
    standardized = apply_standardizer(
        matrix,
        np.asarray(artifact["feature_mean"], dtype=np.float64),
        np.asarray(artifact["feature_scale"], dtype=np.float64),
    )
    draws = [
        predict_ridge(standardized, np.asarray(weights, dtype=np.float64))
        for weights in artifact["weights"]
    ]
    return np.mean(draws, axis=0)


def state_metrics(frame: pd.DataFrame, *, k: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for state_id, group in frame.groupby("state_id", sort=True):
        local = group.sort_values("document_rank", kind="mergesort").reset_index(drop=True)
        truth = local["document_utility"].to_numpy(dtype=np.float64)
        predicted = local["predicted_utility"].to_numpy(dtype=np.float64)
        rank_indices = selection_indices(predicted, k, mode="rank")
        learned_indices = selection_indices(predicted, k, mode="utility")
        oracle_indices = selection_indices(truth, k, mode="utility")
        rows.append(
            {
                "state_id": str(state_id),
                "question_id": str(local.iloc[0]["question_id"]),
                "backend": str(local.iloc[0]["backend"]),
                "dataset": str(local.iloc[0].get("dataset", "")),
                "rank_mean_utility": float(truth[rank_indices].mean()),
                "learned_mean_utility": float(truth[learned_indices].mean()),
                "oracle_mean_utility": float(truth[oracle_indices].mean()),
                "rank_sum_utility": float(truth[rank_indices].sum()),
                "learned_sum_utility": float(truth[learned_indices].sum()),
                "oracle_sum_utility": float(truth[oracle_indices].sum()),
                "learned_ndcg": ndcg_at_k(truth, predicted, k),
                "rank_ndcg": ndcg_at_k(truth, -local["document_rank"].to_numpy(dtype=float), k),
                "prediction_spearman": safe_spearman(truth, predicted),
            }
        )
    return pd.DataFrame(rows)


def run(cfg: dict[str, Any], profile_name: str, *, inputs: Sequence[str] | None = None) -> dict[str, Any]:
    patterns = env_patterns(
        "CREDIT_ROUTING_LABELS",
        cfg["source"]["label_globs"],
        inputs,
    )
    paths = discover_paths(patterns, suffixes=(".jsonl",))
    if not paths:
        raise RuntimeError(f"No utility labels matched {patterns}")
    frame = load_frame(paths, cfg)
    artifact = fit_artifact(frame, cfg, profile_name)
    frame["predicted_utility"] = predict_frame(frame, artifact)
    output_dir = Path(cfg["work_dir"]).resolve() / "reports" / profile_name / EXPERIMENT_ID
    artifact_dir = Path(cfg["work_dir"]).resolve() / "models" / profile_name
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "document_utility_ridge.json"
    atomic_write_json(artifact_path, artifact)
    frame.to_csv(output_dir / "document_predictions.csv", index=False)

    test = frame[frame["split"] == "test"].copy()
    if test.empty:
        raise RuntimeError("Question split produced no test rows")
    positive_threshold = float(cfg["estimator"]["positive_threshold"])
    auc = safe_auc(
        (test["document_utility"].to_numpy(dtype=float) > positive_threshold).astype(int),
        test["predicted_utility"].to_numpy(dtype=float),
    )
    metrics = state_metrics(test, k=int(cfg["labeling"]["observation_k"]))
    metrics.to_csv(output_dir / "state_selection_metrics.csv", index=False)
    metrics["learned_minus_rank"] = metrics["learned_mean_utility"] - metrics["rank_mean_utility"]
    question_effects = metrics.groupby("question_id")["learned_minus_rank"].mean().to_numpy(dtype=float)
    effect = bootstrap_mean(
        question_effects,
        samples=int(cfg["profiles"][profile_name]["bootstrap_samples"]),
        seed=46046,
    )
    backend_rows = []
    for backend, group in metrics.groupby("backend"):
        values = group.groupby("question_id")["learned_minus_rank"].mean().to_numpy(dtype=float)
        one = bootstrap_mean(
            values,
            samples=int(cfg["profiles"][profile_name]["bootstrap_samples"]),
            seed=stable_seed("credit-routing-backend", backend),
        )
        backend_rows.append({"backend": backend, **one})
    backend_frame = pd.DataFrame(backend_rows)
    backend_frame.to_csv(output_dir / "backend_selection_effects.csv", index=False)
    gate = cfg["gates"][EXPERIMENT_ID]
    go = bool(
        effect["estimate"] >= float(gate["minimum_mean_utility_gain"])
        and effect["ci_low"] > 0.0
        and auc >= float(gate["minimum_positive_auc"])
        and not backend_frame.empty
        and (backend_frame["estimate"] >= 0.0).all()
    )
    payload = {
        "schema": 1,
        "experiment_id": EXPERIMENT_ID,
        "profile": profile_name,
        "label_files": len(paths),
        "test_documents": len(test),
        "test_states": int(test["state_id"].nunique()),
        "positive_auc": auc,
        "learned_minus_rank": effect,
        "artifact": str(artifact_path),
        "go": go,
    }
    atomic_write_json(output_dir / "decision.json", payload)
    summary = pd.DataFrame(
        [
            {
                "metric": "positive_auc",
                "estimate": auc,
                "ci_low": float("nan"),
                "ci_high": float("nan"),
            },
            {"metric": "learned_minus_rank_mean_utility", **effect},
        ]
    )
    report = [
        "# EXP-046 Shared document-utility estimator",
        "",
        f"Profile: `{profile_name}`. A query-only ridge ensemble predicts the same fixed-budget document utility used by both the action-routing and observation-routing factors.",
        "",
        "## Primary metrics",
        "",
        markdown_table(summary),
        "",
        "## Backend effects",
        "",
        markdown_table(backend_frame),
        "",
        f"Decision: **{'GO' if go else 'NO-GO'}**.",
        "",
        "A GO requires the learned document score to beat rank-based selection on held-out questions, remain non-negative on BM25 and E5, and discriminate positive utility documents above the configured AUC threshold.",
        "",
    ]
    (output_dir / "EXP046_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the shared credit-routing utility estimator.")
    parser.add_argument("--config", default="configs/credit_routing.yaml")
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), default="pilot")
    parser.add_argument("--input", action="append", default=None)
    args = parser.parse_args()
    payload = run(load_config(args.config), args.profile, inputs=args.input)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
