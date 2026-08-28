from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from stackpilot.query_credit_common import atomic_write_json, load_config, markdown_table
from stackpilot.query_credit_weekend_common import apply_model_override


def _read(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run(cfg: dict[str, Any], profile_name: str) -> dict[str, Any]:
    root = Path(cfg["work_dir"]).resolve() / profile_name / "reports"
    audit = _read(root / "audit" / "decision.json")
    information_gain = _read(root / "ig" / "decision.json")
    gradient = _read(root / "gradient" / "decision.json")
    micro = _read(root / "micro" / "decision.json")
    claims = {
        "document_credit_is_not_a_reliable_action_surrogate": bool(
            audit and audit.get("go_to_optimization_audit")
        ),
        "transport_changes_the_learning_direction": bool(
            gradient and gradient.get("supports_gradient_claim")
        ),
        "transport_hurts_matched_micro_training": bool(
            micro and micro.get("supports_harm_claim")
        ),
        "transport_helps_matched_micro_training": bool(
            micro and micro.get("supports_help_claim")
        ),
    }
    payload = {
        "schema": 1,
        "profile": profile_name,
        "audit_available": audit is not None,
        "ig_available": information_gain is not None,
        "ig_baseline_valid": bool(
            information_gain and information_gain.get("supports_ig_baseline")
        ),
        "gradient_available": gradient is not None,
        "micro_available": micro is not None,
        "claims": claims,
        "paper_status": (
            "mechanism-plus-optimization"
            if claims["document_credit_is_not_a_reliable_action_surrogate"]
            and claims["transport_changes_the_learning_direction"]
            and (
                claims["transport_hurts_matched_micro_training"]
                or claims["transport_helps_matched_micro_training"]
            )
            else "mechanism"
            if claims["document_credit_is_not_a_reliable_action_surrogate"]
            and claims["transport_changes_the_learning_direction"]
            else "audit-only"
            if claims["document_credit_is_not_a_reliable_action_surrogate"]
            else "insufficient"
        ),
    }
    atomic_write_json(root / "weekend_decision.json", payload)
    rows = [
        {
            "논문에서 허용되는 주장": "문서 점수는 검색 행동 점수의 신뢰할 만한 대체값이 아니다",
            "허용": claims["document_credit_is_not_a_reliable_action_surrogate"],
        },
        {
            "논문에서 허용되는 주장": "문서 점수를 행동에 전달하면 학습 방향이 달라진다",
            "허용": claims["transport_changes_the_learning_direction"],
        },
        {
            "논문에서 허용되는 주장": "문서 전달 점수가 일치 학습 성능을 악화한다",
            "허용": claims["transport_hurts_matched_micro_training"],
        },
        {
            "논문에서 허용되는 주장": "인과적으로 부정확한 문서 점수도 학습 힌트로는 도움된다",
            "허용": claims["transport_helps_matched_micro_training"],
        },
    ]
    report = [
        "# 3일 H100 실험 최종 판정",
        "",
        markdown_table(rows),
        "",
        f"정보이득 기준선 사용 가능: **{payload['ig_baseline_valid']}**.",
        "",
        f"현재 논문 상태: **{payload['paper_status']}**.",
        "",
        "항상 금지되는 과장: ‘문서 점수는 쓸모없다’, ‘모든 에이전트에서 항상 실패한다’, ‘문서 보상은 언제나 성능을 떨어뜨린다’.",
        "",
        "`audit-only`도 실패가 아닙니다. 이 경우 논문은 학습 성능 우열보다 행동–관찰 사이의 인과적 측정 차이를 중심으로 써야 합니다.",
        "",
    ]
    (root / "WEEKEND_DECISION_KO.md").write_text("\n".join(report), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/query_credit_weekend.yaml")
    parser.add_argument("--profile", choices=("smoke", "single", "node8"), default="node8")
    args = parser.parse_args()
    print(json.dumps(run(apply_model_override(load_config(args.config)), args.profile), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
