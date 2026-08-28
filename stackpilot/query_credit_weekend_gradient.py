from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stackpilot.query_credit_common import atomic_write_json, load_config, markdown_table, read_jsonl, stable_hash
from stackpilot.query_credit_modeling import (
    build_tokenized_example,
    collate_examples,
    cosine,
    load_lora_model,
    trainable_gradient_vector,
    weighted_query_loss,
)
from stackpilot.query_credit_weekend_common import apply_model_override, state_standardize, two_way_paired_bootstrap


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


def _signal_values(rows: list[dict[str, Any]], reward_view: str, state_id: str) -> dict[str, list[float]]:
    action_half_a = []
    action_half_b = []
    swap_half_a = []
    swap_half_b = []
    for row in rows:
        seed_values = np.asarray(row["full_seed_rewards"][reward_view], dtype=np.float64)
        swap_seed_values = np.asarray(
            row["swap_credit"][reward_view]["per_seed_signed_mean"],
            dtype=np.float64,
        )
        if seed_values.size < 4 or swap_seed_values.size < 4:
            raise RuntimeError("Gradient reliability audit needs four continuation seeds")
        action_half_a.append(float(seed_values[::2].mean()))
        action_half_b.append(float(seed_values[1::2].mean()))
        swap_half_a.append(float(swap_seed_values[::2].mean()))
        swap_half_b.append(float(swap_seed_values[1::2].mean()))
    rng_a = np.random.default_rng(
        int(stable_hash("weekend-gradient-shuffle-a", state_id, length=8), 16)
    )
    rng_b = np.random.default_rng(
        int(stable_hash("weekend-gradient-shuffle-b", state_id, length=8), 16)
    )
    shuffled_a = list(np.asarray(swap_half_a)[rng_a.permutation(len(swap_half_a))])
    shuffled_b = list(np.asarray(swap_half_b)[rng_b.permutation(len(swap_half_b))])
    output = {
        "action-half-a": state_standardize(action_half_a).tolist(),
        "action-half-b": state_standardize(action_half_b).tolist(),
        "swap-half-a": state_standardize(swap_half_a).tolist(),
        "swap-half-b": state_standardize(swap_half_b).tolist(),
        "shuffled-half-a": state_standardize(shuffled_a).tolist(),
        "shuffled-half-b": state_standardize(shuffled_b).tolist(),
    }
    if all("ig_score" in row for row in rows):
        output["ig"] = state_standardize(
            [float(row["ig_score"]) for row in rows]
        ).tolist()
    return output


def _load_rows(cfg: dict[str, Any], profile_name: str) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    root = Path(cfg["work_dir"]).resolve() / profile_name / "data"
    candidates = read_jsonl([root / "candidate_credits.jsonl"])
    prefixes = read_jsonl([root / "state_prefixes.jsonl"])
    ig_path = Path(cfg["work_dir"]).resolve() / profile_name / "reports" / "ig" / "ig_scores.csv"
    ig_scores: dict[tuple[str, int], float] = {}
    if ig_path.is_file() and ig_path.stat().st_size > 0:
        ig_frame = pd.read_csv(ig_path)
        ig_scores = {
            (str(row.state_id), int(row.candidate_index)): float(row.ig_score)
            for row in ig_frame.itertuples(index=False)
        }
    for row in candidates:
        key = (str(row["state_id"]), int(row["candidate_index"]))
        if key in ig_scores:
            row["ig_score"] = ig_scores[key]
    prefix_by_state = {str(row["state_id"]): row["prefix_messages"] for row in prefixes}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        state_id = str(row["state_id"])
        if state_id in prefix_by_state:
            grouped[state_id].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: int(row["candidate_index"]))
    maximum = int(cfg["profiles"][profile_name]["gradient_states"])
    selected = sorted(
        grouped,
        key=lambda state_id: stable_hash("weekend-gradient-state-v1", state_id, length=32),
    )[:maximum]
    return {state_id: grouped[state_id] for state_id in selected}, prefix_by_state


def run_worker(cfg: dict[str, Any], profile_name: str, init_seed: int) -> dict[str, Any]:
    import torch

    grouped, prefixes = _load_rows(cfg, profile_name)
    if not grouped:
        raise RuntimeError("No states are available for the gradient audit")
    torch.manual_seed(int(init_seed))
    np.random.seed(int(init_seed))
    tokenizer, model = load_lora_model(cfg)
    maximum_length = int(cfg["gradient"]["maximum_length"])
    batch_size = int(cfg["gradient"]["batch_size"])
    reward_view = str(cfg["analysis"]["primary_reward_view"])
    output = []
    for state_id, rows in grouped.items():
        examples = [
            build_tokenized_example(
                tokenizer,
                list(prefixes[state_id]),
                str(row["query"]),
                maximum_length,
            )
            for row in rows
        ]
        signals = _signal_values(rows, reward_view, state_id)
        required = (
            "action-half-a",
            "action-half-b",
            "swap-half-a",
            "swap-half-b",
            "shuffled-half-a",
            "shuffled-half-b",
        )
        if any(float(np.std(signals[name])) <= 1e-12 for name in required):
            continue
        if "ig" in signals and float(np.std(signals["ig"])) <= 1e-12:
            signals.pop("ig")
        gradients = {
            name: _gradient(model, tokenizer, examples, values, batch_size)
            for name, values in signals.items()
        }
        self_cosine = cosine(gradients["action-half-a"], gradients["action-half-b"])
        action_swap = float(
            np.mean(
                [
                    cosine(gradients["action-half-a"], gradients["swap-half-b"]),
                    cosine(gradients["action-half-b"], gradients["swap-half-a"]),
                ]
            )
        )
        action_shuffled = float(
            np.mean(
                [
                    cosine(gradients["action-half-a"], gradients["shuffled-half-b"]),
                    cosine(gradients["action-half-b"], gradients["shuffled-half-a"]),
                ]
            )
        )
        action_ig = (
            float(
                np.mean(
                    [
                        cosine(gradients["action-half-a"], gradients["ig"]),
                        cosine(gradients["action-half-b"], gradients["ig"]),
                    ]
                )
            )
            if "ig" in gradients
            else float("nan")
        )
        output.append(
            {
                "init_seed": int(init_seed),
                "state_id": state_id,
                "question_id": str(rows[0]["question_id"]),
                "dataset": str(rows[0]["dataset"]),
                "backend": str(rows[0]["backend"]),
                "candidate_count": len(rows),
                "action_self_cosine": self_cosine,
                "action_swap_cosine": action_swap,
                "action_shuffled_cosine": action_shuffled,
                "action_ig_cosine": action_ig,
                "self_minus_swap": self_cosine - action_swap,
                "self_minus_ig": self_cosine - action_ig,
                "swap_minus_shuffled": action_swap - action_shuffled,
                "action_gradient_norm": float(
                    0.5 * (gradients["action-half-a"].norm() + gradients["action-half-b"].norm())
                ),
                "swap_gradient_norm": float(
                    0.5 * (gradients["swap-half-a"].norm() + gradients["swap-half-b"].norm())
                ),
            }
        )
        del gradients
    if not output:
        raise RuntimeError("No non-degenerate states remained for the gradient worker")
    report_root = Path(cfg["work_dir"]).resolve() / profile_name / "reports" / "gradient"
    report_root.mkdir(parents=True, exist_ok=True)
    path = report_root / f"gradient_seed_{int(init_seed)}.csv"
    pd.DataFrame(output).to_csv(path, index=False)
    payload = {
        "schema": 1,
        "init_seed": int(init_seed),
        "states": len(output),
        "path": str(path),
    }
    atomic_write_json(report_root / f"gradient_seed_{int(init_seed)}.json", payload)
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return payload


def run_report(cfg: dict[str, Any], profile_name: str) -> dict[str, Any]:
    report_root = Path(cfg["work_dir"]).resolve() / profile_name / "reports" / "gradient"
    frames = []
    for seed in cfg["gradient"]["init_seeds"]:
        path = report_root / f"gradient_seed_{int(seed)}.csv"
        if path.is_file() and path.stat().st_size > 0:
            frames.append(pd.read_csv(path))
    if not frames:
        raise RuntimeError("No gradient worker outputs were found")
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(report_root / "gradient_state_results.csv", index=False)
    rows = frame.to_dict("records")
    samples = int(cfg["gradient"]["bootstrap_samples"])
    effects = []
    for index, metric in enumerate(
        (
            "action_self_cosine",
            "action_swap_cosine",
            "action_ig_cosine",
            "action_shuffled_cosine",
            "self_minus_swap",
            "self_minus_ig",
            "swap_minus_shuffled",
        )
    ):
        effect = two_way_paired_bootstrap(
            rows,
            seed_key="init_seed",
            item_key="question_id",
            value_key=metric,
            samples=samples,
            seed=62000 + index,
        )
        effects.append({"metric": metric, **effect})
    pd.DataFrame(effects).to_csv(report_root / "gradient_summary.csv", index=False)
    lookup = {row["metric"]: row for row in effects}
    gate = cfg["gates"]["gradient"]
    conditions = {
        "all_initialization_seeds_finished": int(frame["init_seed"].nunique())
        >= len(cfg["gradient"]["init_seeds"]),
        "action_document_alignment_gap": float(lookup["self_minus_swap"]["estimate"])
        >= float(gate["minimum_alignment_gap"]),
        "gap_ci_is_positive": float(lookup["self_minus_swap"]["ci_low"]) > 0,
        "shuffled_control_is_low": float(lookup["action_shuffled_cosine"]["estimate"])
        <= float(gate["maximum_shuffled_alignment"]),
    }
    go = bool(all(conditions.values()))
    decision = {
        "schema": 1,
        "profile": profile_name,
        "initialization_seeds": int(frame["init_seed"].nunique()),
        "states": int(frame["state_id"].nunique()),
        "effects": {row["metric"]: row for row in effects},
        "conditions": conditions,
        "supports_gradient_claim": go,
    }
    atomic_write_json(report_root / "decision.json", decision)
    report = [
        "# 주말 H100 그래디언트 감사",
        "",
        "`그래디언트`는 모델 파라미터를 어느 방향으로 바꿀지 정하는 학습 화살표입니다.",
        "행동 점수를 실행 시드 절반씩 계산한 두 화살표의 일치도를 기준선으로 삼고, 문서 교체 점수의 화살표가 그 기준선보다 얼마나 멀어지는지 측정했습니다.",
        "",
        markdown_table(effects),
        "",
        "## 판정",
        "",
        markdown_table([{"조건": key, "통과": value} for key, value in conditions.items()]),
        "",
        f"그래디언트 주장 허용: **{'YES' if go else 'NO'}**.",
        "",
    ]
    (report_root / "GRADIENT_REPORT_KO.md").write_text("\n".join(report), encoding="utf-8")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/query_credit_weekend.yaml")
    parser.add_argument("--profile", choices=("smoke", "single", "node8"), default="node8")
    parser.add_argument("--init-seed", type=int)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    cfg = apply_model_override(load_config(args.config))
    if args.report:
        payload = run_report(cfg, args.profile)
    elif args.init_seed is not None:
        payload = run_worker(cfg, args.profile, args.init_seed)
    else:
        raise SystemExit("Provide --init-seed for a worker or --report to aggregate workers")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
