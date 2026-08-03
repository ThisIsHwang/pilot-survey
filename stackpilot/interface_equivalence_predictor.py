from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stackpilot.interface_causality_common import (
    atomic_write_json,
    balanced_state_subset,
    behavior_signature,
    classification_metrics,
    fit_logistic,
    jaccard,
    load_config,
    load_state_results,
    markdown_table,
    ngram_set,
    normalize_title,
    predict_logistic,
    source_patterns,
    stable_hash,
    token_set,
)

EXPERIMENT_ID = "EXP-023"
DATASETS = ("2wikimultihopqa", "hotpotqa", "musique", "nq", "popqa", "triviaqa")
STYLES = ("factual", "lexical", "semantic", "entity")


def _prefix_titles(result: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for record in result.get("prefix", {}).get("records", []) or []:
        if isinstance(record, dict):
            raw = record.get("observed_titles", [])
            if isinstance(raw, list):
                values.extend(str(value) for value in raw if str(value).strip())
    if not values:
        for record in result["state"].get("prior_turns", []) or []:
            if isinstance(record, dict):
                raw = record.get("observed_titles", [])
                if isinstance(raw, list):
                    values.extend(str(value) for value in raw if str(value).strip())
    return values


def _observed_titles(candidate: dict[str, Any]) -> list[str]:
    raw = candidate.get("intervention_observed_titles", [])
    if isinstance(raw, list):
        return [str(value) for value in raw if str(value).strip()]
    return []


def _base_pair_features(result: dict[str, Any], left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    state = result["state"]
    left_query = str(left["query"])
    right_query = str(right["query"])
    left_tokens = token_set(left_query, content_only=True)
    right_tokens = token_set(right_query, content_only=True)
    question_tokens = token_set(str(state["question"]), content_only=True)
    prefix_tokens = token_set(" ".join(_prefix_titles(result)), content_only=True)
    left_length = max(1, len(left_tokens))
    right_length = max(1, len(right_tokens))
    features: dict[str, float] = {
        "token_jaccard": jaccard(left_tokens, right_tokens),
        "char3_jaccard": jaccard(ngram_set(left_query), ngram_set(right_query)),
        "length_ratio": min(left_length, right_length) / max(left_length, right_length),
        "style_equal": float(str(left.get("style", "")) == str(right.get("style", ""))),
        "left_question_overlap": jaccard(left_tokens, question_tokens),
        "right_question_overlap": jaccard(right_tokens, question_tokens),
        "question_overlap_difference": abs(
            jaccard(left_tokens, question_tokens) - jaccard(right_tokens, question_tokens)
        ),
        "left_prefix_overlap": jaccard(left_tokens, prefix_tokens),
        "right_prefix_overlap": jaccard(right_tokens, prefix_tokens),
        "prefix_overlap_difference": abs(
            jaccard(left_tokens, prefix_tokens) - jaccard(right_tokens, prefix_tokens)
        ),
        "source_turn": float(state["source_turn"]),
        "topk": float(state["topk"]),
        "immediate_gain_difference": abs(
            float(left.get("immediate_support_gain", 0.0))
            - float(right.get("immediate_support_gain", 0.0))
        ),
        "final_recall_difference": abs(
            float(left.get("final_support_recall", 0.0))
            - float(right.get("final_support_recall", 0.0))
        ),
        "answer_f1_difference": abs(
            float(left.get("answer_f1", 0.0)) - float(right.get("answer_f1", 0.0))
        ),
        "backend_e5": float(str(state["backend"]) == "e5"),
        "backend_bm25": float(str(state["backend"]) == "bm25"),
    }
    for dataset in DATASETS:
        features[f"dataset__{dataset}"] = float(str(state["dataset"]) == dataset)
    for style in STYLES:
        features[f"left_style__{style}"] = float(str(left.get("style", "")) == style)
        features[f"right_style__{style}"] = float(str(right.get("style", "")) == style)
    left_titles = {normalize_title(value) for value in _observed_titles(left)}
    right_titles = {normalize_title(value) for value in _observed_titles(right)}
    features.update(
        {
            "response_title_jaccard": jaccard(left_titles, right_titles),
            "response_size_ratio": min(len(left_titles), len(right_titles))
            / max(1, max(len(left_titles), len(right_titles))),
            "response_exact": float(left_titles == right_titles),
        }
    )
    return features


def pair_rows(results: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    mode = str(cfg["equivalence_predictor"]["label_signature"])
    for result in results:
        state = result["state"]
        candidates = [
            row for row in result["candidates"] if int(row.get("protocol_failure", 0)) == 0
        ]
        for left, right in itertools.combinations(candidates, 2):
            left_signature = behavior_signature(left, state, mode=mode)
            right_signature = behavior_signature(right, state, mode=mode)
            features = _base_pair_features(result, left, right)
            rows.append(
                {
                    "pair_id": stable_hash(state["state_id"], left["candidate_id"], right["candidate_id"]),
                    "state_id": str(state["state_id"]),
                    "question_id": str(state["question_id"]),
                    "backend": str(state["backend"]),
                    "dataset": str(state["dataset"]),
                    "source_turn": int(state["source_turn"]),
                    "left_id": str(left["candidate_id"]),
                    "right_id": str(right["candidate_id"]),
                    "left_style": str(left.get("style", "")),
                    "right_style": str(right.get("style", "")),
                    "label": int(left_signature == right_signature),
                    **features,
                }
            )
    return rows


def balance_pairs(rows: list[dict[str, Any]], *, maximum_negative_ratio: float, seed: int) -> list[dict[str, Any]]:
    positives = [row for row in rows if int(row["label"]) == 1]
    negatives = [row for row in rows if int(row["label"]) == 0]
    if not positives or not negatives:
        return list(rows)
    maximum_negatives = max(1, int(round(len(positives) * maximum_negative_ratio)))
    if len(negatives) <= maximum_negatives:
        return list(rows)
    generator = np.random.default_rng(seed)
    indices = generator.choice(len(negatives), size=maximum_negatives, replace=False)
    selected = positives + [negatives[int(index)] for index in indices]
    return sorted(selected, key=lambda row: str(row["pair_id"]))


def split_rows(rows: list[dict[str, Any]], seed: int, train_ratio: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train, test = [], []
    for row in rows:
        raw = int(stable_hash(seed, row["question_id"], length=16), 16) / float(16**16 - 1)
        (train if raw < train_ratio else test).append(row)
    return train, test


def feature_sets(columns: list[str]) -> dict[str, list[str]]:
    semantic = [
        "token_jaccard", "char3_jaccard", "length_ratio", "style_equal",
    ]
    state = semantic + [
        "left_question_overlap", "right_question_overlap", "question_overlap_difference",
        "left_prefix_overlap", "right_prefix_overlap", "prefix_overlap_difference",
        "source_turn", "topk",
    ] + [column for column in columns if column.startswith("dataset__") or column.startswith("left_style__") or column.startswith("right_style__")]
    backend = state + ["backend_e5", "backend_bm25"]
    response = backend + [
        "immediate_gain_difference", "final_recall_difference", "answer_f1_difference",
        "response_title_jaccard", "response_size_ratio", "response_exact",
    ]
    return {
        "semantic-only": semantic,
        "state-conditioned": state,
        "backend-conditioned": backend,
        "response-conditioned": response,
    }


def matrix(rows: list[dict[str, Any]], features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([[float(row[name]) for name in features] for row in rows], dtype=np.float64),
        np.asarray([int(row["label"]) for row in rows], dtype=np.int64),
    )


def evaluate_model(
    train: list[dict[str, Any]],
    test: list[dict[str, Any]],
    *,
    features: list[str],
    cfg: dict[str, Any],
) -> dict[str, float]:
    train_x, train_y = matrix(train, features)
    test_x, test_y = matrix(test, features)
    if len(np.unique(train_y)) < 2:
        probabilities = np.full(len(test_y), float(train_y.mean()), dtype=np.float64)
    else:
        weights, mean, standard = fit_logistic(
            train_x,
            train_y,
            learning_rate=float(cfg["equivalence_predictor"]["learning_rate"]),
            l2=float(cfg["equivalence_predictor"]["l2"]),
            steps=int(cfg["equivalence_predictor"]["steps"]),
        )
        probabilities = predict_logistic(test_x, weights, mean, standard)
    from stackpilot.interface_causality_common import classification_metrics

    return classification_metrics(test_y, probabilities)


def cross_backend_relation_stability(results: list[dict[str, Any]], cfg: dict[str, Any]) -> pd.DataFrame:
    mode = str(cfg["equivalence_predictor"]["label_signature"])
    by_key: dict[tuple[Any, ...], dict[str, dict[tuple[str, str], int]]] = defaultdict(dict)
    for result in results:
        state = result["state"]
        candidates = [
            row for row in result["candidates"] if int(row.get("protocol_failure", 0)) == 0
        ]
        relation = {}
        for left, right in itertools.combinations(candidates, 2):
            style_pair = tuple(sorted((str(left.get("style", "")), str(right.get("style", "")))))
            relation[style_pair] = int(
                behavior_signature(left, state, mode=mode)
                == behavior_signature(right, state, mode=mode)
            )
        key = (
            str(state["question_id"]),
            str(state["dataset"]),
            int(state["source_turn"]),
            int(state["topk"]),
            str(state.get("policy_tag", "")),
            int(state.get("policy_seed", 0)),
        )
        by_key[key][str(state["backend"])] = relation
    rows = []
    for key, backends in by_key.items():
        if "bm25" not in backends or "e5" not in backends:
            continue
        pairs = sorted(set(backends["bm25"]) | set(backends["e5"]))
        agreements = [int(backends["bm25"].get(pair, 0) == backends["e5"].get(pair, 0)) for pair in pairs]
        positive_union = {
            pair for pair in pairs if backends["bm25"].get(pair, 0) or backends["e5"].get(pair, 0)
        }
        positive_intersection = {
            pair for pair in pairs if backends["bm25"].get(pair, 0) and backends["e5"].get(pair, 0)
        }
        rows.append(
            {
                "question_id": key[0],
                "dataset": key[1],
                "source_turn": key[2],
                "topk": key[3],
                "policy_tag": key[4],
                "policy_seed": key[5],
                "relation_agreement": float(np.mean(agreements)) if agreements else 1.0,
                "positive_edge_jaccard": len(positive_intersection) / len(positive_union) if positive_union else 1.0,
                "pair_count": len(pairs),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-023: predict state- and backend-conditioned query equivalence.")
    parser.add_argument("--config", default="configs/interface_causality.yaml")
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), default="pilot")
    parser.add_argument("--inputs", nargs="*", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    profile = cfg["profiles"][args.profile]
    results = load_state_results(source_patterns(cfg, args.inputs))
    results = balanced_state_subset(results, int(profile["predictor_states"]))
    rows = pair_rows(results, cfg)
    if not rows:
        raise RuntimeError("No pairwise equivalence examples were generated")
    rows = balance_pairs(
        rows,
        maximum_negative_ratio=float(cfg["equivalence_predictor"].get("maximum_negative_ratio", 4.0)),
        seed=int(cfg["equivalence_predictor"]["split_seed"]),
    )
    train, test = split_rows(
        rows,
        seed=int(cfg["equivalence_predictor"]["split_seed"]),
        train_ratio=float(cfg["equivalence_predictor"]["train_ratio"]),
    )
    columns = list(rows[0])
    sets = feature_sets(columns)
    summary_rows = []
    for name, features in sets.items():
        metrics = evaluate_model(train, test, features=features, cfg=cfg)
        summary_rows.append({"scope": "mixed-heldout", "model": name, **metrics})
        for source_backend, target_backend in (("bm25", "e5"), ("e5", "bm25")):
            source_train = [row for row in train if row["backend"] == source_backend]
            target_test = [row for row in test if row["backend"] == target_backend]
            if source_train and target_test:
                cross_metrics = evaluate_model(source_train, target_test, features=features, cfg=cfg)
                summary_rows.append(
                    {
                        "scope": f"{source_backend}-to-{target_backend}",
                        "model": name,
                        **cross_metrics,
                    }
                )
    summary = pd.DataFrame(summary_rows)
    stability = cross_backend_relation_stability(results, cfg)
    output_dir = Path(args.output_dir or Path(cfg["work_dir"]) / "reports" / args.profile / "EXP-023").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "pair_examples.csv", index=False)
    summary.to_csv(output_dir / "predictor_metrics.csv", index=False)
    stability.to_csv(output_dir / "cross_backend_stability.csv", index=False)

    mixed = summary[summary["scope"] == "mixed-heldout"].set_index("model")
    semantic_auc = float(mixed.loc["semantic-only", "auc"])
    state_auc = float(mixed.loc["state-conditioned", "auc"])
    backend_auc = float(mixed.loc["backend-conditioned", "auc"])
    response_auc = float(mixed.loc["response-conditioned", "auc"])
    mean_relation_agreement = float(stability["relation_agreement"].mean()) if len(stability) else float("nan")
    mean_edge_jaccard = float(stability["positive_edge_jaccard"].mean()) if len(stability) else float("nan")
    gates = cfg["gates"]["EXP-023"]
    minimum_paired = int(gates.get("minimum_paired_backend_states", 10))
    go = bool(
        state_auc - semantic_auc >= float(gates["minimum_state_auc_gain"])
        and response_auc >= float(gates["minimum_response_auc"])
        and len(stability) >= minimum_paired
        and np.isfinite(mean_relation_agreement)
        and mean_relation_agreement <= float(gates["maximum_cross_backend_relation_agreement"])
    )
    decision = {
        "experiment_id": EXPERIMENT_ID,
        "profile": args.profile,
        "go": go,
        "semantic_auc": semantic_auc,
        "state_conditioned_auc": state_auc,
        "backend_conditioned_auc": backend_auc,
        "response_conditioned_auc": response_auc,
        "state_minus_semantic_auc": state_auc - semantic_auc,
        "backend_minus_state_auc": backend_auc - state_auc,
        "mean_cross_backend_relation_agreement": (mean_relation_agreement if np.isfinite(mean_relation_agreement) else None),
        "mean_cross_backend_positive_edge_jaccard": (mean_edge_jaccard if np.isfinite(mean_edge_jaccard) else None),
        "paired_backend_states": int(len(stability)),
        "train_pairs": len(train),
        "test_pairs": len(test),
    }
    atomic_write_json(output_dir / "decision.json", decision)
    report = [
        "# EXP-023 Environment-conditioned equivalence report",
        "",
        f"Profile: `{args.profile}`. Pair labels are induced by actual retrieval behavior, not semantic paraphrase labels.",
        "",
        "## Predictor metrics",
        "",
        markdown_table(summary),
        "",
        "## Decision statistics",
        "",
        "```text",
        json.dumps(decision, indent=2, sort_keys=True),
        "```",
        "",
        f"Decision: **{'GO' if go else 'NO-GO'}**.",
        "",
        "A GO supports environment-conditioned action abstraction: semantic similarity alone is inadequate, while state or response information predicts behavioral equivalence and the equivalence relation itself changes across retrievers.",
        "",
    ]
    (output_dir / "EXP023_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
