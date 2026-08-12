from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stackpilot.query_credit_common import (
    atomic_write_json,
    bootstrap_by_cluster,
    load_config,
    markdown_table,
    question_split,
    read_jsonl,
)
from stackpilot.query_credit_modeling import (
    batch_indices,
    build_tokenized_example,
    collate_examples,
    load_lora_model,
    signal_values,
    weighted_query_loss,
)

EXPERIMENT_ID = "EXP-053"


def _score_examples(model: Any, tokenizer: Any, examples: list[dict[str, Any]], batch_size: int) -> list[float]:
    import torch

    model.eval()
    output: list[float] = []
    with torch.no_grad():
        for offset in range(0, len(examples), batch_size):
            subset = examples[offset : offset + batch_size]
            batch = collate_examples(tokenizer, subset, [1.0] * len(subset), next(model.parameters()).device)
            _, log_probs = weighted_query_loss(model, batch)
            output.extend(float(value) for value in log_probs.detach().cpu().tolist())
    model.train()
    return output


def _paired_bootstrap(rows: list[dict[str, Any]], method: str, baseline: str, metric: str, samples: int, seed: int) -> dict[str, float]:
    by_state: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        by_state[str(row["state_id"])][str(row["method"])] = float(row[metric])
    paired = [
        {"state_id": state_id, "value": values[method] - values[baseline]}
        for state_id, values in by_state.items()
        if method in values and baseline in values
    ]
    return bootstrap_by_cluster(
        paired,
        cluster_key="state_id",
        statistic=lambda values: float(np.mean([row["value"] for row in values])),
        samples=samples,
        seed=seed,
    )


def run(cfg: dict[str, Any], profile: str, candidate_file: str | None = None, prefix_file: str | None = None) -> dict[str, Any]:
    root = Path(cfg["work_dir"]) / "labels" / profile
    candidates = read_jsonl([candidate_file or root / "candidate_credits.jsonl"])
    prefixes = read_jsonl([prefix_file or root / "state_prefixes.jsonl"])
    prefix_by_state = {str(row["state_id"]): row["prefix_messages"] for row in prefixes}
    salt = str(cfg["micro_update"]["split_salt"])
    for row in candidates:
        row["split"] = question_split(str(row["question_id"]), salt)
    train_states = sorted({str(row["state_id"]) for row in candidates if row["split"] == "train"})[: int(cfg["profiles"][profile]["micro_train_states"])]
    test_states = sorted({str(row["state_id"]) for row in candidates if row["split"] == "test"})[: int(cfg["profiles"][profile]["micro_test_states"])]
    train_rows = [row for row in candidates if str(row["state_id"]) in train_states and str(row["state_id"]) in prefix_by_state]
    test_rows = [row for row in candidates if str(row["state_id"]) in test_states and str(row["state_id"]) in prefix_by_state]
    if not train_rows or not test_rows:
        raise RuntimeError("Micro-update train/test split is empty")
    maximum_length = int(cfg["micro_update"]["maximum_length"])
    batch_size = int(cfg["micro_update"]["batch_size"])
    methods = [str(value) for value in cfg["micro_update"]["methods"]]
    seeds = [int(value) for value in cfg["profiles"][profile]["seeds"]]
    results: list[dict[str, Any]] = []

    for seed in seeds:
        for method_index, method in enumerate(methods):
            import torch

            torch.manual_seed(seed)
            tokenizer, model = load_lora_model(cfg)
            examples = [
                build_tokenized_example(
                    tokenizer,
                    list(prefix_by_state[str(row["state_id"])]),
                    str(row["query"]),
                    maximum_length,
                )
                for row in train_rows
            ]
            weights = signal_values(train_rows, method, shuffle_seed=seed + method_index)
            optimizer = torch.optim.AdamW(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                lr=float(cfg["micro_update"]["learning_rate"]),
            )
            for epoch in range(int(cfg["micro_update"]["epochs"])):
                for indices in batch_indices(len(examples), batch_size, seed + epoch):
                    batch_examples = [examples[index] for index in indices]
                    batch_weights = [weights[index] for index in indices]
                    batch = collate_examples(tokenizer, batch_examples, batch_weights, next(model.parameters()).device)
                    loss, _ = weighted_query_loss(model, batch)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
            heldout_examples = [
                build_tokenized_example(
                    tokenizer,
                    list(prefix_by_state[str(row["state_id"])]),
                    str(row["query"]),
                    maximum_length,
                )
                for row in test_rows
            ]
            scores = _score_examples(model, tokenizer, heldout_examples, batch_size)
            grouped: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
            for row, score in zip(test_rows, scores, strict=True):
                grouped[str(row["state_id"])].append((row, score))
            for state_id, state_candidates in grouped.items():
                selected, selected_score = max(
                    state_candidates,
                    key=lambda item: (float(item[1]), -int(item[0]["candidate_index"])),
                )
                best_reward = max(float(item[0]["full_reward"]) for item in state_candidates)
                results.append(
                    {
                        "seed": seed,
                        "method": method,
                        "state_id": state_id,
                        "question_id": str(selected["question_id"]),
                        "backend": str(selected["backend"]),
                        "dataset": str(selected["dataset"]),
                        "selected_candidate_index": int(selected["candidate_index"]),
                        "selected_log_probability": float(selected_score),
                        "selected_reward": float(selected["full_reward"]),
                        "selected_query_indispensability": float(selected["query_indispensability"]),
                        "selected_alias_size": int(selected["alias_class_size"]),
                        "reward_regret": float(best_reward - float(selected["full_reward"])),
                    }
                )
            del model, tokenizer, optimizer
            gc.collect()
            torch.cuda.empty_cache()

    frame = pd.DataFrame(results)
    output_dir = Path(cfg["work_dir"]).resolve() / "reports" / profile / EXPERIMENT_ID
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "heldout_query_selection.csv", index=False)
    means = (
        frame.groupby("method", as_index=False)
        .agg(
            selected_reward=("selected_reward", "mean"),
            selected_query_indispensability=("selected_query_indispensability", "mean"),
            reward_regret=("reward_regret", "mean"),
            selected_alias_size=("selected_alias_size", "mean"),
            seeds=("seed", "nunique"),
            states=("state_id", "nunique"),
        )
        .to_dict("records")
    )
    samples = int(cfg["profiles"][profile]["bootstrap_samples"])
    contrasts = []
    for method in methods:
        if method == "doc-positive-sum":
            continue
        for metric in ("selected_reward", "selected_query_indispensability", "reward_regret"):
            effect = _paired_bootstrap(results, method, "doc-positive-sum", metric, samples, 53000 + len(contrasts))
            contrasts.append({"method": method, "baseline": "doc-positive-sum", "metric": metric, **effect})
    pd.DataFrame(contrasts).to_csv(output_dir / "paired_contrasts.csv", index=False)
    gate = cfg["gates"][EXPERIMENT_ID]
    oracle_reward = next(row for row in contrasts if row["method"] == "query-oracle" and row["metric"] == "selected_reward")
    oracle_iqu = next(row for row in contrasts if row["method"] == "query-oracle" and row["metric"] == "selected_query_indispensability")
    go = bool(
        float(oracle_reward["estimate"]) >= float(gate["minimum_query_oracle_over_doc_reward"])
        and float(oracle_iqu["estimate"]) >= float(gate["minimum_query_oracle_over_doc_iqu"])
    )
    payload = {
        "schema": 1,
        "experiment_id": EXPERIMENT_ID,
        "profile": profile,
        "train_states": len(train_states),
        "test_states": len(test_states),
        "methods": methods,
        "go": go,
    }
    atomic_write_json(output_dir / "decision.json", payload)
    report = [
        "# EXP-053 Matched LoRA micro-update",
        "",
        f"Profile: `{profile}`. Every method starts from the same base model and sees the same state-query examples. The update signal alone changes. Held-out evaluation chooses the highest-likelihood query from state-matched candidates and scores its already executed retrieval trajectory.",
        "",
        "## Means",
        "",
        markdown_table(means),
        "",
        "## Paired contrasts against document-derived query credit",
        "",
        markdown_table(contrasts),
        "",
        f"Decision: **{'GO' if go else 'NO-GO'}**.",
        "",
    ]
    (output_dir / "EXP053_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/query_credit.yaml")
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), default="pilot")
    parser.add_argument("--candidate-file")
    parser.add_argument("--prefix-file")
    args = parser.parse_args()
    print(json.dumps(run(load_config(args.config), args.profile, args.candidate_file, args.prefix_file), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
