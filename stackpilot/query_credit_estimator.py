from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stackpilot.query_credit_common import (
    FEATURE_NAMES,
    atomic_write_json,
    average_precision,
    document_features,
    fit_ridge_ensemble,
    load_config,
    markdown_table,
    question_split,
    read_jsonl,
    roc_auc,
    spearman,
)

EXPERIMENT_ID = "EXP-054"


def _paths(cfg: dict[str, Any], profile: str, document_file: str | None) -> tuple[Path, Path]:
    source = Path(document_file or Path(cfg["work_dir"]) / "labels" / profile / "document_credits.jsonl").resolve()
    output = Path(cfg["work_dir"]).resolve() / "reports" / profile / EXPERIMENT_ID
    return source, output


def run(cfg: dict[str, Any], profile: str, document_file: str | None = None) -> dict[str, Any]:
    source, output_dir = _paths(cfg, profile, document_file)
    rows = read_jsonl([source])
    if not rows:
        raise RuntimeError(f"No document-credit rows in {source}")
    salt = str(cfg["estimator"]["split_salt"])
    for row in rows:
        row["split"] = question_split(str(row["question_id"]), salt)
    frame = pd.DataFrame(rows)
    train = frame[frame["split"] == "train"].copy()
    validation = frame[frame["split"] == "validation"].copy()
    test = frame[frame["split"] == "test"].copy()
    if train.empty or test.empty:
        raise RuntimeError("Question-heldout estimator split is empty")
    x_train = np.stack([document_features(row) for row in train.to_dict("records")])
    y_train = train["document_utility"].to_numpy(dtype=np.float64)
    model = fit_ridge_ensemble(
        x_train,
        y_train,
        alphas=[float(value) for value in cfg["estimator"]["ridge_alphas"]],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model_payload = model.to_json()
    model_payload.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "profile": profile,
            "source": str(source),
            "split_salt": salt,
        }
    )
    atomic_write_json(output_dir / "document_utility_model.json", model_payload)
    predictions = []
    for row in rows:
        copy = dict(row)
        copy["predicted_document_utility"] = model.predict_row(row)
        predictions.append(copy)
    prediction_frame = pd.DataFrame(predictions)
    prediction_frame.to_csv(output_dir / "document_predictions.csv", index=False)
    threshold = float(cfg["estimator"]["positive_threshold"])
    metrics = []
    for split in ("train", "validation", "test"):
        subset = prediction_frame[prediction_frame["split"] == split]
        if subset.empty:
            continue
        labels = (subset["document_utility"].to_numpy(dtype=np.float64) > threshold).astype(int)
        scores = subset["predicted_document_utility"].to_numpy(dtype=np.float64)
        auc = roc_auc(labels.tolist(), scores.tolist())
        ap = average_precision(labels.tolist(), scores.tolist())
        metrics.append(
            {
                "split": split,
                "rows": len(subset),
                "questions": subset["question_id"].nunique(),
                "spearman": spearman(subset["document_utility"].tolist(), scores.tolist()),
                "rmse": float(np.sqrt(np.mean((scores - subset["document_utility"].to_numpy(dtype=np.float64)) ** 2))),
                "positive_auc": 0.5 if not np.isfinite(auc) else float(auc),
                "average_precision": 0.0 if not np.isfinite(ap) else float(ap),
            }
        )
    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(output_dir / "estimator_metrics.csv", index=False)
    test_metric = next(row for row in metrics if row["split"] == "test")
    gate = cfg["gates"][EXPERIMENT_ID]
    go = bool(
        float(test_metric["spearman"]) >= float(gate["minimum_test_spearman"])
        and float(test_metric["positive_auc"]) >= float(gate["minimum_positive_auc"])
    )
    payload = {
        "schema": 1,
        "experiment_id": EXPERIMENT_ID,
        "profile": profile,
        "source": str(source),
        "model": str(output_dir / "document_utility_model.json"),
        "test_metrics": test_metric,
        "go": go,
    }
    atomic_write_json(output_dir / "decision.json", payload)
    report = [
        "# EXP-054 Document-utility estimator",
        "",
        f"Profile: `{profile}`. The estimator uses only information available after retrieval and before future agent actions. Question IDs are split 60/20/20.",
        "",
        "## Metrics",
        "",
        markdown_table(metrics),
        "",
        f"Decision: **{'GO' if go else 'NO-GO'}**.",
        "",
    ]
    (output_dir / "EXP054_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/query_credit.yaml")
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), default="pilot")
    parser.add_argument("--document-file")
    args = parser.parse_args()
    print(json.dumps(run(load_config(args.config), args.profile, args.document_file), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
