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
    batch_indices,
    build_tokenized_example,
    collate_examples,
    load_lora_model,
    weighted_query_loss,
)
from stackpilot.query_credit_weekend_common import apply_model_override, shaped_signal, state_standardize, two_way_paired_bootstrap


def _balanced_state_split(
    grouped: dict[str, list[dict[str, Any]]],
    *,
    cfg: dict[str, Any],
    profile_name: str,
) -> tuple[list[str], list[str]]:
    profile = cfg["profiles"][profile_name]
    backends = [str(value) for value in profile["backends"]]
    datasets = [str(value) for value in cfg["collection"]["datasets"]]
    cells = len(backends) * len(datasets)
    train_target = int(profile["micro_train_states"])
    test_target = int(profile["micro_test_states"])
    if train_target % cells or test_target % cells:
        raise RuntimeError(
            "micro_train_states and micro_test_states must be divisible by dataset/backend cells"
        )
    train_per_cell = train_target // cells
    test_per_cell = test_target // cells
    indexed: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for state_id, rows in grouped.items():
        row = rows[0]
        indexed[(str(row["dataset"]), str(row["backend"]))][
            str(row["question_id"])
        ] = state_id
    train_states: list[str] = []
    test_states: list[str] = []
    salt = str(cfg["micro_update"]["split_salt"])
    for dataset in datasets:
        question_sets = [set(indexed[(dataset, backend)]) for backend in backends]
        common = set.intersection(*question_sets) if question_sets else set()
        ordered = sorted(
            common,
            key=lambda question_id: stable_hash(
                salt, dataset, question_id, length=32
            ),
        )
        needed = train_per_cell + test_per_cell
        if len(ordered) < needed:
            raise RuntimeError(
                f"Micro split needs {needed} paired questions for {dataset}, found {len(ordered)}"
            )
        train_questions = ordered[:train_per_cell]
        test_questions = ordered[train_per_cell:needed]
        for backend in backends:
            train_states.extend(indexed[(dataset, backend)][qid] for qid in train_questions)
            test_states.extend(indexed[(dataset, backend)][qid] for qid in test_questions)
    return train_states, test_states


def _load(cfg: dict[str, Any], profile_name: str):
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
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["candidate_index"]))
    train_states, test_states = _balanced_state_split(
        grouped, cfg=cfg, profile_name=profile_name
    )
    return grouped, prefix_by_state, train_states, test_states


def _method_weights(
    rows: list[dict[str, Any]],
    *,
    method: str,
    reward_view: str,
    alpha: float,
    state_id: str,
) -> list[float]:
    outcome = [float(row["mean_reward"][reward_view]) for row in rows]
    signed = [float(row["swap_credit"][reward_view]["signed_mean"]) for row in rows]
    positive = [float(row["swap_credit"][reward_view]["positive_sum"]) for row in rows]
    if method == "outcome-only":
        return state_standardize(outcome).tolist()
    if method == "outcome-plus-swap":
        return shaped_signal(outcome, signed, alpha=alpha).tolist()
    if method == "outcome-plus-positive":
        return shaped_signal(outcome, positive, alpha=alpha).tolist()
    if method == "outcome-plus-ig":
        if not all("ig_score" in row for row in rows):
            raise RuntimeError(
                f"IG scores are incomplete for state {state_id}; run the IG stage first"
            )
        information_gain = [float(row["ig_score"]) for row in rows]
        return shaped_signal(outcome, information_gain, alpha=alpha).tolist()
    if method == "outcome-plus-shuffled":
        rng = np.random.default_rng(
            int(stable_hash("weekend-micro-shuffle", state_id, length=8), 16)
        )
        shuffled = list(np.asarray(signed)[rng.permutation(len(signed))])
        return shaped_signal(outcome, shuffled, alpha=alpha).tolist()
    raise ValueError(f"Unknown micro-update method: {method}")


def _score_examples(model: Any, tokenizer: Any, examples: list[dict[str, Any]], batch_size: int) -> list[float]:
    import torch

    model.eval()
    output: list[float] = []
    with torch.no_grad():
        for offset in range(0, len(examples), batch_size):
            subset = examples[offset : offset + batch_size]
            batch = collate_examples(
                tokenizer,
                subset,
                [1.0] * len(subset),
                next(model.parameters()).device,
            )
            _, log_probs = weighted_query_loss(model, batch)
            output.extend(float(value) for value in log_probs.detach().cpu().tolist())
    model.train()
    return output


def run_worker(cfg: dict[str, Any], profile_name: str, run_seed: int, method: str) -> dict[str, Any]:
    import torch

    if method not in cfg["micro_update"]["methods"]:
        raise ValueError(f"Unknown method {method}")
    grouped, prefixes, train_states, test_states = _load(cfg, profile_name)
    maximum_length = int(cfg["micro_update"]["maximum_length"])
    batch_size = int(cfg["micro_update"]["batch_size"])
    reward_view = str(cfg["analysis"]["primary_reward_view"])
    alpha = float(cfg["micro_update"]["alpha"])

    train_rows = [row for state_id in train_states for row in grouped[state_id]]
    torch.manual_seed(int(run_seed))
    torch.cuda.manual_seed_all(int(run_seed))
    np.random.seed(int(run_seed))
    tokenizer, model = load_lora_model(cfg)
    train_examples = [
        build_tokenized_example(
            tokenizer,
            list(prefixes[str(row["state_id"])]),
            str(row["query"]),
            maximum_length,
        )
        for row in train_rows
    ]
    weights_by_key: dict[tuple[str, int], float] = {}
    for state_id in train_states:
        state_rows = grouped[state_id]
        state_weights = _method_weights(
            state_rows,
            method=method,
            reward_view=reward_view,
            alpha=alpha,
            state_id=state_id,
        )
        for row, weight in zip(state_rows, state_weights, strict=True):
            weights_by_key[(state_id, int(row["candidate_index"]))] = float(weight)
    train_weights = [
        weights_by_key[(str(row["state_id"]), int(row["candidate_index"]))]
        for row in train_rows
    ]
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(cfg["micro_update"]["learning_rate"]),
    )
    update_count = 0
    for epoch in range(int(cfg["micro_update"]["epochs"])):
        # Method-independent ordering makes the optimization dose identical.
        for indices in batch_indices(len(train_examples), batch_size, int(run_seed) + epoch):
            batch = collate_examples(
                tokenizer,
                [train_examples[index] for index in indices],
                [train_weights[index] for index in indices],
                next(model.parameters()).device,
            )
            loss, _ = weighted_query_loss(model, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            update_count += 1

    test_rows = [row for state_id in test_states for row in grouped[state_id]]
    test_examples = [
        build_tokenized_example(
            tokenizer,
            list(prefixes[str(row["state_id"])]),
            str(row["query"]),
            maximum_length,
        )
        for row in test_rows
    ]
    scores = _score_examples(model, tokenizer, test_examples, batch_size)
    scored: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for row, score in zip(test_rows, scores, strict=True):
        scored[str(row["state_id"])].append((row, score))
    results = []
    for state_id, values in scored.items():
        selected, selected_score = max(
            values,
            key=lambda value: (float(value[1]), -int(value[0]["candidate_index"])),
        )
        rewards = [float(value[0]["mean_reward"][reward_view]) for value in values]
        best_reward = max(rewards)
        selected_reward = float(selected["mean_reward"][reward_view])
        results.append(
            {
                "run_seed": int(run_seed),
                "method": method,
                "state_id": state_id,
                "question_id": str(selected["question_id"]),
                "dataset": str(selected["dataset"]),
                "backend": str(selected["backend"]),
                "selected_candidate_index": int(selected["candidate_index"]),
                "selected_log_probability": float(selected_score),
                "selected_reward": selected_reward,
                "reward_regret": float(best_reward - selected_reward),
                "selected_swap_credit": float(selected["swap_credit"][reward_view]["signed_mean"]),
            }
        )
    report_root = Path(cfg["work_dir"]).resolve() / profile_name / "reports" / "micro"
    report_root.mkdir(parents=True, exist_ok=True)
    safe_method = method.replace("/", "-")
    path = report_root / f"micro_seed_{int(run_seed)}_{safe_method}.csv"
    pd.DataFrame(results).to_csv(path, index=False)
    payload = {
        "schema": 1,
        "run_seed": int(run_seed),
        "method": method,
        "train_states": len(train_states),
        "test_states": len(test_states),
        "training_rows": len(train_rows),
        "updates": update_count,
        "nonzero_weight_rate": float(np.mean(np.abs(train_weights) > 1e-12)),
        "weight_rms": float(np.sqrt(np.mean(np.asarray(train_weights) ** 2))),
        "path": str(path),
    }
    atomic_write_json(report_root / f"micro_seed_{int(run_seed)}_{safe_method}.json", payload)
    del optimizer, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return payload


def _paired_rows(frame: pd.DataFrame, method: str, metric: str) -> list[dict[str, Any]]:
    baseline = frame[frame["method"] == "outcome-only"]
    treatment = frame[frame["method"] == method]
    merged = treatment.merge(
        baseline,
        on=["run_seed", "state_id"],
        suffixes=("_method", "_baseline"),
        how="inner",
    )
    return [
        {
            "run_seed": int(row.run_seed),
            "state_id": str(row.state_id),
            "question_id": str(row.question_id_method),
            "difference": float(getattr(row, f"{metric}_method") - getattr(row, f"{metric}_baseline")),
        }
        for row in merged.itertuples(index=False)
    ]


def run_report(cfg: dict[str, Any], profile_name: str) -> dict[str, Any]:
    report_root = Path(cfg["work_dir"]).resolve() / profile_name / "reports" / "micro"
    frames = []
    for seed in cfg["micro_update"]["seeds"]:
        for method in cfg["micro_update"]["methods"]:
            path = report_root / f"micro_seed_{int(seed)}_{str(method).replace('/', '-')}.csv"
            if path.is_file() and path.stat().st_size > 0:
                frames.append(pd.read_csv(path))
    if not frames:
        raise RuntimeError("No micro-update worker outputs were found")
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(report_root / "micro_all_results.csv", index=False)
    seed_sets = {
        str(method): set(
            int(value)
            for value in frame.loc[frame["method"] == method, "run_seed"].unique()
        )
        for method in cfg["micro_update"]["methods"]
    }
    core_methods = [str(value) for value in cfg["micro_update"]["core_methods"]
    core_seed_sets = [seed_sets[method] for method in core_methods]
    complete_seed_set = (
        set.intersection(*core_seed_sets) if core_seed_sets else set()
    )
    analysis_frame = frame[frame["run_seed"].isin(sorted(complete_seed_set))].copy()
    if analysis_frame.empty:
        raise RuntimeError("No training seed completed every micro-update method")

    dose_rows = []
    for seed in sorted(complete_seed_set):
        for method in cfg["micro_update"]["methods"]:
            safe_method = str(method).replace("/", "-")
            path = report_root / f"micro_seed_{int(seed)}_{safe_method}.json"
            if path.is_file():
                dose_rows.append(json.loads(path.read_text(encoding="utf-8")))
    pd.DataFrame(dose_rows).to_csv(report_root / "micro_dose.csv", index=False)

    means = (
        analysis_frame.groupby("method", as_index=False)
        .agg(
            selected_reward=("selected_reward", "mean"),
            reward_regret=("reward_regret", "mean"),
            selected_swap_credit=("selected_swap_credit", "mean"),
            seeds=("run_seed", "nunique"),
            states=("state_id", "nunique"),
        )
        .to_dict("records")
    )
    effects = []
    for method_index, method in enumerate(cfg["micro_update"]["methods"]):
        if method == "outcome-only" or method not in set(analysis_frame["method"]):
            continue
        for metric_index, metric in enumerate(("selected_reward", "reward_regret")):
            paired = _paired_rows(analysis_frame, str(method), metric)
            if not paired:
                continue
            effect = two_way_paired_bootstrap(
                paired,
                seed_key="run_seed",
                item_key="question_id",
                value_key="difference",
                samples=int(cfg["micro_update"]["bootstrap_samples"]),
                seed=63000 + method_index * 10 + metric_index,
            )
            effects.append(
                {
                    "method": method,
                    "baseline": "outcome-only",
                    "metric": metric,
                    **effect,
                }
            )
    pd.DataFrame(means).to_csv(report_root / "micro_means.csv", index=False)
    pd.DataFrame(effects).to_csv(report_root / "micro_effects.csv", index=False)

    swap_effect = next(
        row
        for row in effects
        if row["method"] == "outcome-plus-swap" and row["metric"] == "selected_reward"
    )
    gate = cfg["gates"]["micro"]
    seed_counts = {method: len(values) for method, values in seed_sets.items()}
    completed_seeds = len(complete_seed_set)
    dose_by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in dose_rows:
        if str(row["method"]) in core_methods:
            dose_by_seed[int(row["run_seed"])].append(row)
    dose_geometry_matched = bool(dose_by_seed) and all(
        {str(row["method"]) for row in rows} == set(core_methods)
        and len({int(row["updates"]) for row in rows}) == 1
        and len({int(row["training_rows"]) for row in rows}) == 1
        for rows in dose_by_seed.values()
    )
    rms_spreads = []
    nonzero_spreads = []
    for rows in dose_by_seed.values():
        rms = [float(row["weight_rms"]) for row in rows]
        nonzero = [float(row["nonzero_weight_rate"]) for row in rows]
        rms_spreads.append((max(rms) - min(rms)) / max(1e-12, float(np.mean(rms))))
        nonzero_spreads.append(max(nonzero) - min(nonzero))
    maximum_rms_spread = max(rms_spreads, default=float("inf"))
    maximum_nonzero_spread = max(nonzero_spreads, default=float("inf"))
    dose_magnitude_matched = (
        maximum_rms_spread <= float(gate["maximum_relative_weight_rms_spread"])
        and maximum_nonzero_spread <= float(gate["maximum_nonzero_rate_spread"])
    )
    material = float(gate["material_reward_effect"])
    estimate = float(swap_effect["estimate"])
    if (
        completed_seeds < int(gate["minimum_seeds"])
        or not dose_geometry_matched
        or not dose_magnitude_matched
    ):
        conclusion = "insufficient-seeds-or-dose-mismatch"
    elif estimate >= material and float(swap_effect["ci_low"]) > 0:
        conclusion = "document-shaping-helps"
    elif estimate <= -material and float(swap_effect["ci_high"]) < 0:
        conclusion = "document-shaping-hurts"
    else:
        conclusion = "no-clear-optimization-effect"
    decision = {
        "schema": 1,
        "profile": profile_name,
        "completed_seeds_across_core_methods": completed_seeds,
        "core_methods": core_methods,
        "seed_counts": seed_counts,
        "methods": sorted(analysis_frame["method"].unique().tolist()),
        "states": int(analysis_frame["state_id"].nunique()),
        "dose_geometry_matched": dose_geometry_matched,
        "maximum_relative_weight_rms_spread": maximum_rms_spread,
        "maximum_nonzero_rate_spread": maximum_nonzero_spread,
        "dose_magnitude_matched": dose_magnitude_matched,
        "effects": effects,
        "document_shaping_conclusion": conclusion,
        "supports_harm_claim": conclusion == "document-shaping-hurts",
        "supports_help_claim": conclusion == "document-shaping-helps",
    }
    atomic_write_json(report_root / "decision.json", decision)
    report = [
        "# 주말 H100 일치 학습 실험",
        "",
        "모든 조건은 같은 초기 모델, 같은 검색어 예시, 같은 미니배치 순서, 같은 업데이트 횟수를 사용했습니다. 달라지는 것은 최종 결과 점수에 어떤 문서 보조점수를 더하는가뿐입니다.",
        "",
        "## 평균",
        "",
        markdown_table(means),
        "",
        "## outcome-only 대비 차이",
        "",
        markdown_table(effects),
        "",
        "## 학습량 일치 검사",
        "",
        markdown_table(
            [
                {
                    "완료 공통 시드": completed_seeds,
                    "업데이트·행 수 일치": dose_geometry_matched,
                    "최대 RMS 상대 차이": maximum_rms_spread,
                    "최대 비영점 비율 차이": maximum_nonzero_spread,
                    "학습 신호 크기 일치": dose_magnitude_matched,
                }
            ]
        ),
        "",
        f"판정: **{conclusion}**.",
        "",
        "이 결과는 인과적 정확성과 학습 유용성을 분리해서 해석해야 합니다. 문서 점수가 행동 원인을 정확히 나타내지 않더라도 학습 힌트로 도움될 수 있고, 반대로 방해할 수도 있습니다.",
        "",
    ]
    (report_root / "MICRO_REPORT_KO.md").write_text("\n".join(report), encoding="utf-8")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/query_credit_weekend.yaml")
    parser.add_argument("--profile", choices=("smoke", "single", "node8"), default="node8")
    parser.add_argument("--run-seed", type=int)
    parser.add_argument("--method")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    cfg = apply_model_override(load_config(args.config))
    if args.report:
        payload = run_report(cfg, args.profile)
    elif args.run_seed is not None and args.method:
        payload = run_worker(cfg, args.profile, args.run_seed, args.method)
    else:
        raise SystemExit("Provide --run-seed and --method for a worker, or --report")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
