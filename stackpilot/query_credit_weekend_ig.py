from __future__ import annotations

import argparse
import gc
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from stackpilot.observation_geometry import render_retrieval_observation
from stackpilot.query_credit_common import (
    atomic_write_json,
    load_config,
    markdown_table,
    read_jsonl,
    spearman,
    stable_hash,
)
from stackpilot.query_credit_modeling import collate_examples, weighted_query_loss
from stackpilot.query_credit_weekend_common import (
    apply_model_override,
    cluster_mean_bootstrap,
    pairwise_preference_accuracy,
    state_audit_metrics,
    top1_regret,
)


def _candidate_key(row: Mapping[str, Any]) -> tuple[str, int]:
    return str(row["state_id"]), int(row["candidate_index"])


def _choose_random_contexts(
    rows: Sequence[dict[str, Any]],
    *,
    tokenizer: Any,
    observation_token_budget: int,
) -> dict[tuple[str, int], dict[str, Any]]:
    """Choose an outcome-blind, length-matched random context from another question."""
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    lengths: dict[tuple[str, int], int] = {}
    for row in rows:
        key = _candidate_key(row)
        rendered = render_retrieval_observation(
            [dict(value) for value in row["visible_documents"]],
            tokenizer,
            int(observation_token_budget),
        )
        lengths[key] = int(rendered.token_count)
        by_cell[(str(row["dataset"]), str(row["backend"]))].append(row)

    output: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = _candidate_key(row)
        pool = [
            other
            for other in by_cell[(str(row["dataset"]), str(row["backend"]))]
            if str(other["question_id"]) != str(row["question_id"])
        ]
        if not pool:
            raise RuntimeError(
                f"No cross-question random context is available for {key}"
            )
        target_length = lengths[key]
        chosen = min(
            pool,
            key=lambda other: (
                abs(lengths[_candidate_key(other)] - target_length),
                stable_hash(
                    "weekend-ig-random-context-v1",
                    row["state_id"],
                    row["candidate_index"],
                    other["state_id"],
                    other["candidate_index"],
                    length=32,
                ),
            ),
        )
        random_length = lengths[_candidate_key(chosen)]
        output[key] = {
            "documents": [dict(value) for value in chosen["visible_documents"]],
            "random_state_id": str(chosen["state_id"]),
            "random_candidate_index": int(chosen["candidate_index"]),
            "actual_observation_tokens": int(target_length),
            "random_observation_tokens": int(random_length),
            "relative_length_difference": float(
                abs(random_length - target_length) / max(1, target_length)
            ),
        }
    return output


def _answer_example(
    tokenizer: Any,
    *,
    prefix_messages: Sequence[Mapping[str, str]],
    query: str,
    documents: Sequence[Mapping[str, Any]],
    answer: str,
    observation_token_budget: int,
    maximum_length: int,
) -> dict[str, Any]:
    observation = render_retrieval_observation(
        [dict(value) for value in documents],
        tokenizer,
        int(observation_token_budget),
    )
    context = [dict(value) for value in prefix_messages]
    context.append({"role": "assistant", "content": f"<search>{query}</search>"})
    context.append({"role": "user", "content": observation.visible_text})
    completion = f"<answer>{answer}</answer>"
    full = context + [{"role": "assistant", "content": completion}]
    prefix_ids = tokenizer.apply_chat_template(
        context,
        tokenize=True,
        add_generation_prompt=True,
    )
    full_ids = tokenizer.apply_chat_template(
        full,
        tokenize=True,
        add_generation_prompt=False,
    )
    prefix_ids = [int(value) for value in prefix_ids]
    full_ids = [int(value) for value in full_ids]
    common = 0
    for left, right in zip(prefix_ids, full_ids):
        if left != right:
            break
        common += 1
    if common == 0:
        raise RuntimeError("Answer prompt and completion have no common token prefix")
    dropped = max(0, len(full_ids) - int(maximum_length))
    full_ids = full_ids[dropped:]
    answer_start = max(0, common - dropped)
    if answer_start >= len(full_ids):
        raise RuntimeError("Maximum length removed the complete gold answer")
    labels = [-100] * len(full_ids)
    labels[answer_start:] = full_ids[answer_start:]
    return {
        "input_ids": full_ids,
        "labels": labels,
    }


def _load_model(cfg: dict[str, Any]) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_cfg = cfg["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["base_model"],
        revision=model_cfg.get("revision"),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model"],
        revision=model_cfg.get("revision"),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda"},
    )
    model.config.use_cache = False
    model.eval()
    return tokenizer, model


def _score_examples(
    model: Any,
    tokenizer: Any,
    examples: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
) -> list[float]:
    import torch

    output: list[float] = []
    with torch.no_grad():
        for offset in range(0, len(examples), int(batch_size)):
            subset = list(examples[offset : offset + int(batch_size)])
            batch = collate_examples(
                tokenizer,
                subset,
                [1.0] * len(subset),
                next(model.parameters()).device,
            )
            _, log_probs = weighted_query_loss(model, batch)
            output.extend(float(value) for value in log_probs.detach().cpu().tolist())
    return output


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def run_worker(
    cfg: dict[str, Any],
    profile_name: str,
    shard_index: int,
    shard_count: int,
) -> dict[str, Any]:
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("Invalid IG shard geometry")
    root = Path(cfg["work_dir"]).resolve() / profile_name
    data_root = root / "data"
    candidates = read_jsonl([data_root / "candidate_credits.jsonl"])
    prefixes = read_jsonl([data_root / "state_prefixes.jsonl"])
    prefix_by_state = {str(row["state_id"]): row["prefix_messages"] for row in prefixes}
    tokenizer, model = _load_model(cfg)
    ig_cfg = cfg["ig"]
    random_contexts = _choose_random_contexts(
        candidates,
        tokenizer=tokenizer,
        observation_token_budget=int(ig_cfg["observation_token_budget"]),
    )
    selected = [
        row
        for row in candidates
        if int(
            stable_hash(
                "weekend-ig-shard-v1",
                row["state_id"],
                row["candidate_index"],
                length=8,
            ),
            16,
        )
        % int(shard_count)
        == int(shard_index)
    ]
    examples: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for row in selected:
        answers = [str(value) for value in row.get("answers", []) if str(value).strip()]
        answers = answers[: int(ig_cfg["maximum_answer_aliases"])]
        if not answers:
            continue
        random_context = random_contexts[_candidate_key(row)]
        for condition, documents in (
            ("actual", row["visible_documents"]),
            ("random", random_context["documents"]),
        ):
            for answer_index, answer in enumerate(answers):
                examples.append(
                    _answer_example(
                        tokenizer,
                        prefix_messages=prefix_by_state[str(row["state_id"])],
                        query=str(row["query"]),
                        documents=documents,
                        answer=answer,
                        observation_token_budget=int(ig_cfg["observation_token_budget"]),
                        maximum_length=int(ig_cfg["maximum_length"]),
                    )
                )
                metadata.append(
                    {
                        "state_id": str(row["state_id"]),
                        "question_id": str(row["question_id"]),
                        "dataset": str(row["dataset"]),
                        "backend": str(row["backend"]),
                        "candidate_index": int(row["candidate_index"]),
                        "condition": condition,
                        "answer_index": answer_index,
                        "random_state_id": random_context["random_state_id"],
                        "random_candidate_index": random_context[
                            "random_candidate_index"
                        ],
                        "relative_random_length_difference": random_context[
                            "relative_length_difference"
                        ],
                    }
                )
    if not examples:
        raise RuntimeError(f"IG shard {shard_index}/{shard_count} has no examples")
    scores = _score_examples(
        model,
        tokenizer,
        examples,
        batch_size=int(ig_cfg["batch_size"]),
    )
    grouped: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    metadata_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for row, score in zip(metadata, scores, strict=True):
        key = str(row["state_id"]), int(row["candidate_index"])
        grouped[key][str(row["condition"])].append(float(score))
        metadata_by_key[key] = row
    output = []
    for key, conditions in sorted(grouped.items()):
        if not conditions.get("actual") or not conditions.get("random"):
            continue
        actual = max(conditions["actual"])
        random = max(conditions["random"])
        meta = metadata_by_key[key]
        output.append(
            {
                "state_id": key[0],
                "question_id": str(meta["question_id"]),
                "dataset": str(meta["dataset"]),
                "backend": str(meta["backend"]),
                "candidate_index": key[1],
                "actual_gold_log_probability": actual,
                "random_gold_log_probability": random,
                "ig_score": float(actual - random),
                "random_state_id": str(meta["random_state_id"]),
                "random_candidate_index": int(meta["random_candidate_index"]),
                "relative_random_length_difference": float(
                    meta["relative_random_length_difference"]
                ),
            }
        )
    report_root = root / "reports" / "ig"
    path = report_root / f"ig_shard_{int(shard_index):02d}.csv"
    _atomic_csv(path, pd.DataFrame(output))
    payload = {
        "schema": 1,
        "profile": profile_name,
        "shard_index": int(shard_index),
        "shard_count": int(shard_count),
        "candidate_rows": len(output),
        "scored_examples": len(examples),
        "path": str(path),
    }
    atomic_write_json(report_root / f"ig_shard_{int(shard_index):02d}.json", payload)
    del model, tokenizer
    gc.collect()
    import torch

    torch.cuda.empty_cache()
    return payload


def run_report(cfg: dict[str, Any], profile_name: str) -> dict[str, Any]:
    root = Path(cfg["work_dir"]).resolve() / profile_name
    report_root = root / "reports" / "ig"
    frames = [
        pd.read_csv(path)
        for path in sorted(report_root.glob("ig_shard_*.csv"))
        if path.stat().st_size > 0
    ]
    if not frames:
        raise RuntimeError("No IG worker outputs were found")
    frame = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["state_id", "candidate_index"], keep="last"
    )
    _atomic_csv(report_root / "ig_scores.csv", frame)
    candidates = read_jsonl([root / "data" / "candidate_credits.jsonl"])
    score_by_key = {
        (str(row.state_id), int(row.candidate_index)): float(row.ig_score)
        for row in frame.itertuples(index=False)
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        if _candidate_key(row) in score_by_key:
            grouped[str(row["state_id"])].append(row)
    state_rows = []
    reward_view = str(cfg["analysis"]["primary_reward_view"])
    epsilon = float(cfg["analysis"]["preference_epsilon"])
    for state_id, rows in grouped.items():
        rows.sort(key=lambda value: int(value["candidate_index"]))
        if len(rows) < 2:
            continue
        truth = [float(row["mean_reward"][reward_view]) for row in rows]
        ig_values = [score_by_key[_candidate_key(row)] for row in rows]
        audit = state_audit_metrics(
            rows,
            reward_view=reward_view,
            document_signal=str(cfg["analysis"]["primary_document_signal"]),
            epsilon=epsilon,
        )
        regret = top1_regret(truth, ig_values)
        ig_pairwise = pairwise_preference_accuracy(truth, ig_values, epsilon=epsilon)
        state_rows.append(
            {
                "state_id": state_id,
                "question_id": str(rows[0]["question_id"]),
                "dataset": str(rows[0]["dataset"]),
                "backend": str(rows[0]["backend"]),
                "action_self_pairwise": float(audit["action_self_pairwise"]),
                "ig_action_pairwise": float(ig_pairwise),
                "action_self_minus_ig": float(
                    audit["action_self_pairwise"] - ig_pairwise
                ),
                "ig_action_spearman": float(spearman(truth, ig_values)),
                "ig_normalized_regret": float(regret["normalized_regret"]),
            }
        )
    if not state_rows:
        raise RuntimeError("No complete state remained for the IG report")
    pd.DataFrame(state_rows).to_csv(report_root / "ig_state_metrics.csv", index=False)
    effects = []
    for index, metric in enumerate(
        (
            "ig_action_pairwise",
            "action_self_minus_ig",
            "ig_action_spearman",
            "ig_normalized_regret",
        )
    ):
        effect = cluster_mean_bootstrap(
            state_rows,
            value_key=metric,
            cluster_key="question_id",
            samples=int(cfg["ig"]["bootstrap_samples"]),
            seed=64000 + index,
        )
        effects.append({"metric": metric, **effect})
    pd.DataFrame(effects).to_csv(report_root / "ig_summary.csv", index=False)
    expected = len(candidates)
    observed = len(score_by_key)
    coverage = observed / max(1, expected)
    mean_length_difference = float(
        frame["relative_random_length_difference"].mean()
    )
    gate = cfg["gates"]["ig"]
    conditions = {
        "candidate_coverage": coverage >= float(gate["minimum_candidate_coverage"]),
        "random_context_length_is_matched": mean_length_difference
        <= float(gate["maximum_mean_relative_random_length_difference"]),
    }
    decision = {
        "schema": 1,
        "profile": profile_name,
        "expected_candidates": expected,
        "scored_candidates": observed,
        "candidate_coverage": coverage,
        "mean_relative_random_length_difference": mean_length_difference,
        "effects": {row["metric"]: row for row in effects},
        "conditions": conditions,
        "supports_ig_baseline": bool(all(conditions.values())),
        "baseline_scope": (
            "Teacher-forced IG-Search-style gold-answer confidence difference; "
            "not an exact reproduction of the paper's full online GRPO pipeline."
        ),
    }
    atomic_write_json(report_root / "decision.json", decision)
    report = [
        "# 정보이득 기준선",
        "",
        "`정보이득(information gain)`은 실제 검색 문서를 봤을 때 정답 문자열의 모델 로그확률이, 다른 질문에서 가져온 길이 맞춤 무작위 문서를 봤을 때보다 얼마나 높아졌는지를 뜻합니다.",
        "이 구현은 IG-Search의 핵심 점수 아이디어를 오프라인 후보 실험에 맞춘 기준선이며, 원 논문의 전체 온라인 GRPO 학습을 그대로 재현한 것은 아닙니다.",
        "",
        markdown_table(effects),
        "",
        markdown_table(
            [
                {
                    "후보 점수 완성률": coverage,
                    "무작위 문맥 평균 상대 길이 차이": mean_length_difference,
                    "기준선 사용 가능": bool(all(conditions.values())),
                }
            ]
        ),
        "",
    ]
    (report_root / "IG_REPORT_KO.md").write_text("\n".join(report), encoding="utf-8")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/query_credit_weekend.yaml")
    parser.add_argument("--profile", choices=("smoke", "single", "node8"), default="node8")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    cfg = apply_model_override(load_config(args.config))
    if args.report:
        payload = run_report(cfg, args.profile)
    elif args.shard_index is not None and args.shard_count is not None:
        payload = run_worker(cfg, args.profile, args.shard_index, args.shard_count)
    else:
        raise SystemExit("Provide --shard-index/--shard-count or --report")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
