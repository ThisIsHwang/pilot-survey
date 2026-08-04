from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from stackpilot.behavior_quotient_common import (
    atomic_write_json,
    balanced_subset,
    cluster_bootstrap,
    jaccard,
    load_config,
    load_state_results,
    markdown_table,
    normalize_title,
    source_patterns,
    stable_hash,
    token_set,
    word_tokens,
)
from stackpilot.interface_expressivity_audit import menu_queries, prefix_titles
from stackpilot.retrieval_clients import RetrievalClient

EXPERIMENT_ID = "EXP-032"
FEATURES = (
    "question_tokens",
    "capitalized_tokens",
    "prefix_titles",
    "prefix_unique_titles",
    "prefix_duplicate_rate",
    "menu_size",
    "source_turn",
    "question_prefix_overlap",
    "free_menu_overlap",
    "free_query_tokens",
    "menu_query_tokens",
)


def factual_query(result: dict[str, Any]) -> str:
    for candidate in result["candidates"]:
        if int(candidate.get("protocol_failure", 0)) != 0:
            continue
        if str(candidate.get("origin", "")) == "factual" or str(candidate.get("style", "")) == "factual":
            query = " ".join(str(candidate.get("query", "")).split())
            if query:
                return query
    for candidate in result["candidates"]:
        if int(candidate.get("protocol_failure", 0)) == 0:
            query = " ".join(str(candidate.get("query", "")).split())
            if query:
                return query
    return ""


def _query_result(
    client: RetrievalClient,
    query: str,
    *,
    state: dict[str, Any],
    prefix: Sequence[str],
) -> dict[str, Any]:
    if not query:
        return {"query": "", "invalid": 1, "titles": [], "gain": 0.0, "recall": 0.0}
    results = client.search(query, int(state["topk"]))
    titles = [str(row.get("title", "")).strip() for row in results]
    titles = [title for title in titles if title]
    gold = {normalize_title(value) for value in state.get("support_titles", [])}
    prefix_set = {normalize_title(value) for value in prefix}
    observed = {normalize_title(value) for value in titles}
    before = len(gold & prefix_set) / max(1, len(gold))
    after = len(gold & (prefix_set | observed)) / max(1, len(gold))
    return {"query": query, "invalid": 0, "titles": titles, "gain": float(after - before), "recall": float(after)}


def state_features(result: dict[str, Any], *, menu: list[dict[str, str]], free: str, menu_query: str) -> dict[str, float]:
    state = result["state"]
    question = str(state["question"])
    titles = prefix_titles(result)
    normalized = [normalize_title(value) for value in titles]
    question_tokens = word_tokens(question)
    capitalized = [token for token in question.split() if token[:1].isupper()]
    prefix_tokens = token_set(" ".join(titles))
    question_set = token_set(question)
    return {
        "question_tokens": float(len(question_tokens)),
        "capitalized_tokens": float(len(capitalized)),
        "prefix_titles": float(len(titles)),
        "prefix_unique_titles": float(len(set(normalized))),
        "prefix_duplicate_rate": float(1.0 - len(set(normalized)) / max(1, len(normalized))),
        "menu_size": float(len(menu)),
        "source_turn": float(state.get("source_turn", 0)),
        "question_prefix_overlap": float(jaccard(question_set, prefix_tokens)),
        "free_menu_overlap": float(jaccard(token_set(free), token_set(menu_query))),
        "free_query_tokens": float(len(word_tokens(free))),
        "menu_query_tokens": float(len(word_tokens(menu_query))),
    }


def build_rows(results: Sequence[dict[str, Any]], cfg: dict[str, Any]) -> pd.DataFrame:
    clients = {
        "bm25": RetrievalClient("bm25", str(cfg["retrieval"]["bm25_url"]), timeout=int(cfg["retrieval"]["timeout"]), retries=int(cfg["retrieval"]["retries"])),
        "e5": RetrievalClient("e5", str(cfg["retrieval"]["e5_url"]), timeout=int(cfg["retrieval"]["timeout"]), retries=int(cfg["retrieval"]["retries"])),
    }
    rows = []
    for result in results:
        state = result["state"]
        backend = str(state["backend"])
        free = factual_query(result)
        menu = menu_queries(result, {
            "interface_audit": {
                "maximum_prefix_titles": 8,
                "maximum_menu_queries": 16,
                "relation_tokens": 6,
            }
        })
        if not free or not menu:
            continue
        menu_member = next((row for row in menu if row["style"] == "menu-title-relation"), menu[0])
        menu_query = str(menu_member["query"])
        prefix = prefix_titles(result)
        free_result = _query_result(clients[backend], free, state=state, prefix=prefix)
        menu_result = _query_result(clients[backend], menu_query, state=state, prefix=prefix)
        features = state_features(result, menu=menu, free=free, menu_query=menu_query)
        label = int(
            (free_result["gain"], free_result["recall"])
            > (menu_result["gain"], menu_result["recall"])
        )
        if (
            free_result["gain"] == menu_result["gain"]
            and free_result["recall"] == menu_result["recall"]
            and str(cfg["router"]["tie_break"]) == "free-form"
        ):
            label = 1
        rows.append(
            {
                "state_id": str(state["state_id"]),
                "question_id": str(state["question_id"]),
                "backend": backend,
                "dataset": str(state["dataset"]),
                "free_query": free,
                "menu_query": menu_query,
                "free_gain": free_result["gain"],
                "menu_gain": menu_result["gain"],
                "free_recall": free_result["recall"],
                "menu_recall": menu_result["recall"],
                "free_invalid": free_result["invalid"],
                "menu_invalid": menu_result["invalid"],
                "label_free": label,
                **features,
            }
        )
    if not rows:
        raise RuntimeError("Adaptive router had no states with both interfaces")
    return pd.DataFrame(rows)


def split_mask(frame: pd.DataFrame, fraction: float) -> np.ndarray:
    threshold = int(float(fraction) * 10_000)
    return np.asarray(
        [int(stable_hash("router-split", value, length=12), 16) % 10_000 < threshold for value in frame["question_id"].astype(str)],
        dtype=bool,
    )


def fit_logistic(frame: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    x = frame[list(FEATURES)].to_numpy(dtype=np.float64)
    y = frame["label_free"].to_numpy(dtype=np.float64)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    z = (x - mean) / scale
    weights = np.zeros(z.shape[1], dtype=np.float64)
    bias = 0.0
    lr = float(cfg["router"]["learning_rate"])
    l2 = float(cfg["router"]["l2"])
    for _ in range(int(cfg["router"]["iterations"])):
        logits = np.clip(z @ weights + bias, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        error = probability - y
        weights -= lr * ((z.T @ error) / len(z) + l2 * weights)
        bias -= lr * float(error.mean())
    return {"mean": mean, "scale": scale, "weights": weights, "bias": bias}


def predict(model: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    x = frame[list(FEATURES)].to_numpy(dtype=np.float64)
    z = (x - model["mean"]) / model["scale"]
    logits = np.clip(z @ model["weights"] + float(model["bias"]), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-logits))


def evaluate(frame: pd.DataFrame, probabilities: np.ndarray, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    random_free = rng.integers(0, 2, size=len(frame))
    methods = {
        "always-free": np.ones(len(frame), dtype=np.int64),
        "always-menu": np.zeros(len(frame), dtype=np.int64),
        "random": random_free,
        "learned-router": (probabilities >= 0.5).astype(np.int64),
        "oracle": frame["label_free"].to_numpy(dtype=np.int64),
    }
    rows = []
    for method, choose_free in methods.items():
        for position, (_, row) in enumerate(frame.iterrows()):
            free = int(choose_free[position])
            rows.append(
                {
                    "state_id": row["state_id"],
                    "question_id": row["question_id"],
                    "backend": row["backend"],
                    "dataset": row["dataset"],
                    "method": method,
                    "choose_free": free,
                    "gain": float(row["free_gain"] if free else row["menu_gain"]),
                    "recall": float(row["free_recall"] if free else row["menu_recall"]),
                    "invalid": int(row["free_invalid"] if free else row["menu_invalid"]),
                    "correct_interface": int(free == int(row["label_free"])),
                }
            )
    return pd.DataFrame(rows)


def paired_contrast(frame: pd.DataFrame, *, left: str, right: str, metric: str, samples: int, seed: int) -> dict[str, float]:
    pivot = frame.pivot_table(index="state_id", columns="method", values=metric, aggfunc="first")
    paired = pivot[[left, right]].dropna().reset_index()
    rows = [{"cluster": str(row["state_id"]), "difference": float(row[left] - row[right])} for _, row in paired.iterrows()]
    return cluster_bootstrap(
        rows,
        cluster_key="cluster",
        statistic=lambda values: float(np.mean([item["difference"] for item in values])),
        samples=samples,
        seed=seed,
    )


def run(cfg: dict[str, Any], profile_name: str, inputs: Sequence[str] | None = None) -> dict[str, Any]:
    profile = cfg["profiles"][profile_name]
    results = balanced_subset(load_state_results(source_patterns(cfg, inputs)), int(profile["router_states"]))
    frame = build_rows(results, cfg)
    train_mask = split_mask(frame, float(cfg["router"]["train_fraction"]))
    if train_mask.all() or (~train_mask).all():
        raise RuntimeError("Router question split produced an empty partition")
    train = frame[train_mask].copy()
    test = frame[~train_mask].copy()
    model = fit_logistic(train, cfg)
    probabilities = predict(model, test)
    evaluation = evaluate(test, probabilities, int(cfg["router"]["random_seed"]))

    output_dir = Path(cfg["work_dir"]).resolve() / "reports" / profile_name / EXPERIMENT_ID
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "interface_rows.csv", index=False)
    evaluation.to_csv(output_dir / "router_evaluation.csv", index=False)
    means = evaluation.groupby(["backend", "method"], as_index=False).agg(
        gain=("gain", "mean"),
        recall=("recall", "mean"),
        invalid=("invalid", "mean"),
        free_usage=("choose_free", "mean"),
        interface_accuracy=("correct_interface", "mean"),
    )
    means.to_csv(output_dir / "variant_means.csv", index=False)

    contrasts = []
    for backend in sorted(evaluation["backend"].unique()):
        subset = evaluation[evaluation["backend"] == backend]
        for right in ("always-menu", "always-free"):
            for metric in ("gain", "recall"):
                result = paired_contrast(
                    subset,
                    left="learned-router",
                    right=right,
                    metric=metric,
                    samples=int(profile["bootstrap_samples"]),
                    seed=32032 + len(contrasts),
                )
                contrasts.append({"backend": backend, "contrast": f"learned-router-minus-{right}", "metric": metric, **result})
    contrast_frame = pd.DataFrame(contrasts)
    contrast_frame.to_csv(output_dir / "paired_contrasts.csv", index=False)

    gate = cfg["gates"][EXPERIMENT_ID]
    versus_menu = contrast_frame[(contrast_frame["contrast"] == "learned-router-minus-always-menu") & (contrast_frame["metric"] == "gain")]
    versus_free = contrast_frame[(contrast_frame["contrast"] == "learned-router-minus-always-free") & (contrast_frame["metric"] == "gain")]
    router_means = means[means["method"] == "learned-router"]
    go = bool(
        len(versus_menu) >= 2
        and (versus_menu["estimate"] >= float(gate["minimum_router_vs_menu_gain"])).all()
        and (versus_menu["ci_low"] > 0).all()
        and len(versus_free) >= 2
        and (versus_free["estimate"] >= -float(gate["maximum_router_vs_free_regression"])).all()
        and len(router_means) >= 2
        and (router_means["free_usage"] <= float(gate["maximum_free_usage_rate"])).all()
    )
    payload = {
        "schema": 1,
        "experiment_id": EXPERIMENT_ID,
        "profile": profile_name,
        "train_states": int(len(train)),
        "test_states": int(len(test)),
        "go": go,
        "feature_names": list(FEATURES),
        "weights": {name: float(value) for name, value in zip(FEATURES, model["weights"])},
        "bias": float(model["bias"]),
    }
    atomic_write_json(output_dir / "decision.json", payload)
    report = [
        "# EXP-032 Budget-matched adaptive interface router",
        "",
        f"Profile: `{profile_name}`. The learned router chooses between one previously generated free-form query and one deterministic finite-menu query before a single target-retriever call.",
        "",
        "## Means",
        "",
        markdown_table(means),
        "",
        "## Paired contrasts",
        "",
        markdown_table(contrast_frame),
        "",
        f"Decision: **{'GO' if go else 'NO-GO'}**.",
        "",
    ]
    (output_dir / "EXP032_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/behavior_feedback.yaml")
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), default="pilot")
    parser.add_argument("--input", action="append", default=None)
    args = parser.parse_args()
    payload = run(load_config(args.config), args.profile, args.input)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
