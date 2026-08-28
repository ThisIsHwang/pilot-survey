from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stackpilot.query_credit_common import (
    atomic_write_json,
    load_config,
    markdown_table,
    read_jsonl,
)
from stackpilot.query_credit_weekend_common import (
    apply_model_override,
    cluster_mean_bootstrap,
    state_audit_metrics,
)


def _group_by_state(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["state_id"])].append(row)
    for state_rows in grouped.values():
        state_rows.sort(key=lambda row: int(row["candidate_index"]))
    return grouped


def _state_rows(
    candidates: list[dict[str, Any]],
    *,
    reward_views: list[str],
    signals: list[str],
    epsilon: float,
) -> list[dict[str, Any]]:
    output = []
    for state_id, state_candidates in _group_by_state(candidates).items():
        base = {
            "state_id": state_id,
            "question_id": str(state_candidates[0]["question_id"]),
            "dataset": str(state_candidates[0]["dataset"]),
            "backend": str(state_candidates[0]["backend"]),
        }
        for reward_view in reward_views:
            for signal in signals:
                metrics = state_audit_metrics(
                    state_candidates,
                    reward_view=reward_view,
                    document_signal=signal,
                    epsilon=epsilon,
                )
                output.append(
                    {
                        **base,
                        "reward_view": reward_view,
                        "document_signal": signal,
                        **metrics,
                    }
                )
    return output


def _summaries(
    rows: list[dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    metrics = (
        "action_self_pairwise",
        "document_action_pairwise",
        "reliability_gap",
        "within_state_spearman",
        "normalized_regret",
        "agreement",
    )
    output = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["reward_view"]), str(row["document_signal"]))].append(row)
    for group_index, ((reward_view, signal), values) in enumerate(sorted(groups.items())):
        for metric_index, metric in enumerate(metrics):
            effect = cluster_mean_bootstrap(
                values,
                value_key=metric,
                cluster_key="question_id",
                samples=samples,
                seed=seed + group_index * 100 + metric_index,
            )
            output.append(
                {
                    "reward_view": reward_view,
                    "document_signal": signal,
                    "metric": metric,
                    **effect,
                }
            )
    return output


def _corpus_summary(manifest: list[dict[str, Any]]) -> dict[str, float]:
    rows = [
        probe
        for state in manifest
        for probe in state.get("corpus_probe", [])
        if "found_at_100" in probe
    ]
    return {
        "support_titles": float(len(rows)),
        "support_title_retrieval_coverage_at_100": (
            float(np.mean([float(row["found_at_100"]) for row in rows])) if rows else float("nan")
        ),
    }


def _replacement_balance(candidates: list[dict[str, Any]]) -> dict[str, float]:
    rows = [plan for candidate in candidates for plan in candidate.get("replacement_plan", [])]
    relative = [
        float(row["absolute_length_difference"]) / max(1.0, float(row["original_token_length"]))
        for row in rows
    ]
    return {
        "swaps": float(len(rows)),
        "mean_absolute_token_difference": (
            float(np.mean([float(row["absolute_length_difference"]) for row in rows])) if rows else float("nan")
        ),
        "mean_relative_token_difference": float(np.mean(relative)) if relative else float("nan"),
        "p95_relative_token_difference": float(np.quantile(relative, 0.95)) if relative else float("nan"),
    }


def _omission_sensitivity(candidates: list[dict[str, Any]], reward_view: str) -> dict[str, float]:
    swap = []
    omission = []
    for row in candidates:
        value = row.get("omission_credit", {}).get(reward_view)
        if not value:
            continue
        swap.append(float(row["swap_credit"][reward_view]["signed_mean"]))
        omission.append(float(value["signed_mean"]))
    if len(swap) < 3 or float(np.std(swap)) <= 1e-12 or float(np.std(omission)) <= 1e-12:
        correlation = float("nan")
    else:
        correlation = float(np.corrcoef(swap, omission)[0, 1])
    return {
        "candidate_rows": float(len(swap)),
        "swap_omission_pearson": correlation,
        "same_sign_rate": (
            float(np.mean([np.sign(left) == np.sign(right) for left, right in zip(swap, omission, strict=True)]))
            if swap
            else float("nan")
        ),
    }


def run(cfg: dict[str, Any], profile_name: str) -> dict[str, Any]:
    root = Path(cfg["work_dir"]).resolve() / profile_name
    data_root = root / "data"
    candidate_path = data_root / "candidate_credits.jsonl"
    manifest_path = data_root / "state_manifest.jsonl"
    if not candidate_path.is_file():
        raise RuntimeError(f"Missing collection output: {candidate_path}")
    candidates = read_jsonl([candidate_path])
    manifest = read_jsonl([manifest_path])
    analysis = cfg["analysis"]
    reward_views = [str(value) for value in analysis["reward_views"]]
    signals = [str(value) for value in analysis["document_signals"]]
    state_rows = _state_rows(
        candidates,
        reward_views=reward_views,
        signals=signals,
        epsilon=float(analysis["preference_epsilon"]),
    )
    summaries = _summaries(
        state_rows,
        samples=int(analysis["bootstrap_samples"]),
        seed=int(analysis["bootstrap_seed"]),
    )
    report_root = root / "reports" / "audit"
    report_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(state_rows).to_csv(report_root / "audit_state_metrics.csv", index=False)
    pd.DataFrame(summaries).to_csv(report_root / "audit_summary.csv", index=False)

    states = len({str(row["state_id"]) for row in candidates})
    direct_fraction = float(
        np.mean(
            [
                row.get("origin") in {"factual", "direct-policy-sibling"}
                for row in candidates
            ]
        )
    )
    corpus = _corpus_summary(manifest)
    omission = _omission_sensitivity(candidates, str(analysis["primary_reward_view"]))
    replacement_balance = _replacement_balance(candidates)
    completed_by_cell: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in manifest:
        completed_by_cell[(str(row["dataset"]), str(row["backend"]))].add(
            str(row["state_id"])
        )
    expected_cell_count = len(cfg["collection"]["datasets"]) * len(
        cfg["profiles"][profile_name]["backends"]
    )
    minimum_per_cell = int(cfg["profiles"][profile_name]["minimum_states_per_cell"])

    lookup = {
        (row["reward_view"], row["document_signal"], row["metric"]): row
        for row in summaries
    }
    primary_view = str(analysis["primary_reward_view"])
    primary_signal = str(analysis["primary_document_signal"])
    self_effect = lookup[(primary_view, primary_signal, "action_self_pairwise")]
    pair_effect = lookup[(primary_view, primary_signal, "document_action_pairwise")]
    gap_effect = lookup[(primary_view, primary_signal, "reliability_gap")]
    regret_effect = lookup[(primary_view, primary_signal, "normalized_regret")]

    robust_views = 0
    for reward_view in reward_views:
        effect = lookup[(reward_view, primary_signal, "reliability_gap")]
        if float(effect["ci_low"]) > 0:
            robust_views += 1

    gate = cfg["gates"]["audit"]
    conditions = {
        "enough_completed_states": states >= int(cfg["profiles"][profile_name]["minimum_completed_states"]),
        "balanced_cell_completion": (
            len(completed_by_cell) == expected_cell_count
            and all(len(values) >= minimum_per_cell for values in completed_by_cell.values())
        ),
        "action_measurement_is_stable": float(self_effect["estimate"]) >= float(gate["minimum_action_self_pairwise"]),
        "document_action_gap_is_positive": float(gap_effect["ci_low"]) > float(gate["minimum_reliability_gap_ci_low"]),
        "top1_regret_is_material": float(regret_effect["estimate"]) >= float(gate["minimum_normalized_regret"]),
        "robust_across_reward_views": robust_views >= int(gate["minimum_robust_reward_views"]),
        "corpus_is_usable": float(corpus["support_title_retrieval_coverage_at_100"])
        >= float(gate["minimum_corpus_coverage_at_100"]),
        "replacement_length_is_matched": float(replacement_balance["mean_relative_token_difference"])
        <= float(gate["maximum_mean_relative_replacement_length_difference"]),
        "policy_supported_candidate_share": direct_fraction
        >= float(gate["minimum_direct_policy_candidate_fraction"]),
    }
    go = bool(all(conditions.values()))
    decision = {
        "schema": 1,
        "profile": profile_name,
        "states": states,
        "candidate_rows": len(candidates),
        "direct_policy_candidate_fraction": direct_fraction,
        "corpus": corpus,
        "omission_sensitivity": omission,
        "replacement_balance": replacement_balance,
        "primary": {
            "reward_view": primary_view,
            "document_signal": primary_signal,
            "action_self_pairwise": self_effect,
            "document_action_pairwise": pair_effect,
            "reliability_gap": gap_effect,
            "normalized_regret": regret_effect,
        },
        "robust_reward_views": robust_views,
        "conditions": conditions,
        "go_to_optimization_audit": go,
    }
    atomic_write_json(report_root / "decision.json", decision)

    cell_rows = (
        pd.DataFrame(manifest)
        .groupby(["dataset", "backend"], as_index=False)
        .agg(
            states=("state_id", "nunique"),
            mean_candidates=("candidate_count", "mean"),
            direct_policy_fraction=("direct_policy_candidate_fraction", "mean"),
        )
        .to_dict("records")
        if manifest
        else []
    )
    primary_summary = [
        row
        for row in summaries
        if row["reward_view"] == primary_view and row["document_signal"] == primary_signal
    ]
    report = [
        "# 주말 H100 행동–문서 크레딧 감사",
        "",
        "`크레딧`은 최종 결과의 공을 중간 행동이나 문서에 나눠 주는 점수입니다.",
        "이 보고서는 같은 검색 직전 상태에서 검색어를 비교한 행동 점수와, 문서 수를 유지한 채 한 문서만 교체한 문서 점수가 같은 검색어를 선호하는지 검사합니다.",
        "",
        "## 실행 범위",
        "",
        markdown_table(cell_rows),
        "",
        f"완료 상태: **{states}개**. 직접 정책 샘플 또는 실제 factual query 비율: **{direct_fraction:.1%}**.",
        f"정답 근거 제목 BM25 exact-title recall@100 환경 점검: **{corpus['support_title_retrieval_coverage_at_100']:.1%}**.",
        "",
        "## 주 결과",
        "",
        markdown_table(primary_summary),
        "",
        "- `action_self_pairwise`: 실행 시드를 반으로 나눴을 때 행동 순위가 스스로 얼마나 일치하는지입니다.",
        "- `document_action_pairwise`: 문서 점수와 행동 점수가 같은 검색어를 더 좋다고 고르는 비율입니다.",
        "- `reliability_gap`: 위 두 값의 차이입니다. 양수이면 단순 생성 잡음보다 행동–문서 차이가 더 큽니다.",
        "- `normalized_regret`: 문서 점수로 고른 검색어가 실제 최선 검색어보다 얼마나 손해인지, 그 상태의 보상 범위로 나눈 값입니다.",
        "",
        "## 교체 문서 길이 균형",
        "",
        markdown_table([replacement_balance]),
        "",
        "## 문서 제거 민감도",
        "",
        markdown_table([omission]),
        "",
        "## 사전등록형 판정",
        "",
        markdown_table(
            [
                {"조건": key, "통과": bool(value)}
                for key, value in conditions.items()
            ]
        ),
        "",
        f"다음 단계 판정: **{'GO' if go else 'STOP'}**.",
        "",
        "이 단계가 통과해도 ‘문서 점수는 항상 나쁘다’고 주장하지 않습니다. 허용되는 주장은 검사한 설정에서 문서 점수가 행동의 인과적 기여도를 충실하게 대체하지 못했다는 것입니다.",
        "",
    ]
    (report_root / "AUDIT_REPORT_KO.md").write_text("\n".join(report), encoding="utf-8")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/query_credit_weekend.yaml")
    parser.add_argument("--profile", choices=("smoke", "single", "node8"), default="node8")
    args = parser.parse_args()
    print(json.dumps(run(apply_model_override(load_config(args.config)), args.profile), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
