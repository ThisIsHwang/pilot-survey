from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stackpilot.query_credit_common import atomic_write_json, load_config, markdown_table, read_jsonl
from stackpilot.query_credit_modeling import (
    build_tokenized_example,
    collate_examples,
    cosine,
    load_lora_model,
    signal_values,
    trainable_gradient_vector,
    weighted_query_loss,
)

EXPERIMENT_ID = "EXP-052"


def _load(cfg: dict[str, Any], profile: str, candidate_file: str | None, prefix_file: str | None):
    root = Path(cfg["work_dir"]) / "labels" / profile
    candidates = read_jsonl([candidate_file or root / "candidate_credits.jsonl"])
    prefixes = read_jsonl([prefix_file or root / "state_prefixes.jsonl"])
    by_state = {str(row["state_id"]): row["prefix_messages"] for row in prefixes}
    maximum_states = int(cfg["profiles"][profile]["gradient_states"])
    selected_states = sorted({str(row["state_id"]) for row in candidates})[:maximum_states]
    rows = [row for row in candidates if str(row["state_id"]) in selected_states and str(row["state_id"]) in by_state]
    return rows, by_state


def _gradient(model: Any, tokenizer: Any, examples: list[dict[str, Any]], weights: list[float], batch_size: int) -> Any:
    model.zero_grad(set_to_none=True)
    total = max(1, len(examples))
    for offset in range(0, len(examples), batch_size):
        subset = examples[offset : offset + batch_size]
        subweights = weights[offset : offset + batch_size]
        batch = collate_examples(tokenizer, subset, subweights, next(model.parameters()).device)
        loss, _ = weighted_query_loss(model, batch)
        (loss * (len(subset) / total)).backward()
    return trainable_gradient_vector(model)


def _expanded_alias_rows(rows: list[dict[str, Any]], multiplicity: int) -> list[dict[str, Any]]:
    by_state: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_state[str(row["state_id"])][str(row["behavior_signature"])].append(row)
    output: list[dict[str, Any]] = []
    for classes in by_state.values():
        targets = [group for group in classes.values() if len(group) >= 2]
        if not targets or len(classes) < 2:
            continue
        target = max(targets, key=len)
        for index in range(multiplicity):
            output.append(dict(target[index % len(target)]))
        for signature, group in classes.items():
            if group is target:
                continue
            output.append(dict(group[0]))
    return output


def run(cfg: dict[str, Any], profile: str, candidate_file: str | None = None, prefix_file: str | None = None) -> dict[str, Any]:
    rows, prefixes = _load(cfg, profile, candidate_file, prefix_file)
    if len(rows) < 2:
        raise RuntimeError("Too few rows for gradient audit")
    tokenizer, model = load_lora_model(cfg)
    maximum_length = int(cfg["gradient"]["maximum_length"])
    examples = [
        build_tokenized_example(
            tokenizer,
            list(prefixes[str(row["state_id"])]),
            str(row["query"]),
            maximum_length,
        )
        for row in rows
    ]
    gradients: dict[str, Any] = {}
    norms = []
    batch_size = int(cfg["gradient"]["batch_size"])
    for signal_index, signal in enumerate(cfg["gradient"]["signals"]):
        weights = signal_values(rows, str(signal), shuffle_seed=52000 + signal_index)
        gradient = _gradient(model, tokenizer, examples, weights, batch_size)
        gradients[str(signal)] = gradient
        norms.append({"signal": signal, "gradient_norm": float(gradient.norm())})
    pair_rows = []
    names = list(gradients)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            pair_rows.append(
                {
                    "left": left,
                    "right": right,
                    "cosine": cosine(gradients[left], gradients[right]),
                    "norm_ratio": float(gradients[left].norm() / gradients[right].norm().clamp_min(1e-12)),
                }
            )
    alias_rows = []
    multiplicities = [int(value) for value in cfg["gradient"]["alias_multiplicities"]]
    baseline_doc = None
    baseline_normalized = None
    for multiplicity in multiplicities:
        expanded = _expanded_alias_rows(rows, multiplicity)
        if not expanded:
            continue
        expanded_examples = [
            build_tokenized_example(
                tokenizer,
                list(prefixes[str(row["state_id"])]),
                str(row["query"]),
                maximum_length,
            )
            for row in expanded
        ]
        doc_weights = signal_values(expanded, "doc-positive-sum")
        normalized_weights = signal_values(expanded, "doc-alias-normalized")
        doc_gradient = _gradient(model, tokenizer, expanded_examples, doc_weights, batch_size)
        normalized_gradient = _gradient(model, tokenizer, expanded_examples, normalized_weights, batch_size)
        if baseline_doc is None:
            baseline_doc = doc_gradient
            baseline_normalized = normalized_gradient
        alias_rows.append(
            {
                "multiplicity": multiplicity,
                "rows": len(expanded),
                "doc_gradient_cosine_to_m1": cosine(baseline_doc, doc_gradient),
                "normalized_gradient_cosine_to_m1": cosine(baseline_normalized, normalized_gradient),
                "doc_gradient_norm_ratio": float(doc_gradient.norm() / baseline_doc.norm().clamp_min(1e-12)),
                "normalized_gradient_norm_ratio": float(normalized_gradient.norm() / baseline_normalized.norm().clamp_min(1e-12)),
            }
        )
    output_dir = Path(cfg["work_dir"]).resolve() / "reports" / profile / EXPERIMENT_ID
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(norms).to_csv(output_dir / "gradient_norms.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(output_dir / "gradient_pairs.csv", index=False)
    pd.DataFrame(alias_rows).to_csv(output_dir / "alias_gradient_stress.csv", index=False)
    doc_oracle = next(
        row for row in pair_rows if {row["left"], row["right"]} == {"query-oracle", "doc-positive-sum"}
    )
    last_alias = alias_rows[-1] if alias_rows else {
        "doc_gradient_cosine_to_m1": 1.0,
        "normalized_gradient_cosine_to_m1": 1.0,
    }
    doc_drift = 1.0 - float(last_alias["doc_gradient_cosine_to_m1"])
    normalized_drift = 1.0 - float(last_alias["normalized_gradient_cosine_to_m1"])
    recovery = doc_drift - normalized_drift
    gate = cfg["gates"][EXPERIMENT_ID]
    go = bool(
        float(doc_oracle["cosine"]) <= float(gate["maximum_doc_oracle_gradient_cosine"])
        and doc_drift >= float(gate["minimum_alias_gradient_drift"])
        and recovery >= float(gate["minimum_normalization_recovery"])
    )
    payload = {
        "schema": 1,
        "experiment_id": EXPERIMENT_ID,
        "profile": profile,
        "rows": len(rows),
        "states": len({row["state_id"] for row in rows}),
        "doc_oracle_cosine": float(doc_oracle["cosine"]),
        "alias_doc_drift": doc_drift,
        "alias_normalization_recovery": recovery,
        "go": go,
    }
    atomic_write_json(output_dir / "decision.json", payload)
    report = [
        "# EXP-052 Search-span gradient audit",
        "",
        f"Profile: `{profile}`. Every gradient uses the same Qwen LoRA parameters, state-query examples, and search-action loss; only the state-relative credit changes.",
        "",
        "## Gradient pairs",
        "",
        markdown_table(pair_rows),
        "",
        "## Alias multiplicity stress",
        "",
        markdown_table(alias_rows),
        "",
        f"Decision: **{'GO' if go else 'NO-GO'}**.",
        "",
    ]
    (output_dir / "EXP052_REPORT.md").write_text("\n".join(report), encoding="utf-8")
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
