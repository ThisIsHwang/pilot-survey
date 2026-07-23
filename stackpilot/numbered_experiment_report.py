from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from stackpilot.common import ensure_dir, read_jsonl_tolerant
from stackpilot.hard_rq0_report import markdown_table, matched_hard_question_ids

REPORT_METRICS = [
    "em",
    "f1",
    "support_recall",
    "turn2_evidence_gain",
    "turn3_evidence_gain",
    "recovery_at_2",
    "recovery_at_3",
]
EVIDENCE_TO_ANSWER_POLICY = {
    "evidence-bm25": "bm25-specialist",
    "evidence-e5": "e5-specialist",
}


def load_jsonl_tree(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.rglob("*.jsonl")):
        if "archive" in path.parts:
            continue
        rows.extend(read_jsonl_tolerant(path))
    if not rows:
        raise RuntimeError(f"No JSONL results under {root}")
    frame = pd.DataFrame(rows)
    required = {"policy_tag", "seed", "question_id", "dataset", "backend", "topk", *REPORT_METRICS}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Missing result columns under {root}: {sorted(missing)}")
    frame["topk"] = pd.to_numeric(frame["topk"], errors="raise").astype(int)
    frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)
    return frame


def add_subsets(frame: pd.DataFrame, matched: pd.DataFrame) -> pd.DataFrame:
    all_rows = frame.copy()
    all_rows["subset"] = "all"
    keys = matched[matched["matched_hard"]][["question_id", "dataset", "topk"]]
    hard = frame.merge(keys, on=["question_id", "dataset", "topk"], how="inner")
    hard["subset"] = "matched-hard"
    return pd.concat([all_rows, hard], ignore_index=True)


def specialist_oracle(hard: pd.DataFrame, metric: str) -> pd.DataFrame:
    selected = pd.concat(
        [
            hard[(hard["policy_tag"] == "bm25-specialist") & (hard["backend"] == "bm25")],
            hard[(hard["policy_tag"] == "e5-specialist") & (hard["backend"] == "e5")],
        ],
        ignore_index=True,
    )
    return (
        selected.groupby(
            ["subset", "question_id", "dataset", "backend", "topk"], as_index=False
        )[metric]
        .mean()
        .rename(columns={metric: "specialist_oracle"})
    )


def mixed_regret(hard: pd.DataFrame, mixed: pd.DataFrame, metric: str) -> pd.DataFrame:
    oracle = specialist_oracle(hard, metric)
    policy = mixed[mixed["policy_tag"] == "mixed-blind"][
        ["subset", "seed", "question_id", "dataset", "backend", "topk", metric]
    ].rename(columns={metric: "mixed_score"})
    merged = policy.merge(
        oracle,
        on=["subset", "question_id", "dataset", "backend", "topk"],
        validate="many_to_one",
    )
    merged["metric"] = metric
    merged["mixed_regret"] = merged["specialist_oracle"] - merged["mixed_score"]
    return merged


def metadata_value(blind: pd.DataFrame, oracle: pd.DataFrame, metric: str) -> pd.DataFrame:
    left = blind[blind["policy_tag"] == "mixed-blind"][
        ["subset", "seed", "question_id", "dataset", "backend", "topk", metric]
    ].rename(columns={metric: "blind_score"})
    right = oracle[oracle["policy_tag"] == "mixed-backend-id"][
        ["subset", "seed", "question_id", "dataset", "backend", "topk", metric]
    ].rename(columns={metric: "id_score"})
    common_seeds = sorted(set(left["seed"]) & set(right["seed"]))
    if not common_seeds:
        raise RuntimeError("EXP-003 and EXP-004 have no common seeds")
    merged = left[left["seed"].isin(common_seeds)].merge(
        right[right["seed"].isin(common_seeds)],
        on=["subset", "seed", "question_id", "dataset", "backend", "topk"],
        validate="one_to_one",
    )
    merged["metric"] = metric
    merged["metadata_value"] = merged["id_score"] - merged["blind_score"]
    return merged


def evidence_reward_value(hard: pd.DataFrame, evidence: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    for evidence_tag, answer_tag in EVIDENCE_TO_ANSWER_POLICY.items():
        evidence_rows = evidence[evidence["policy_tag"] == evidence_tag][
            ["subset", "seed", "question_id", "dataset", "backend", "topk", metric]
        ].rename(columns={metric: "evidence_reward_score"})
        answer_rows = hard[hard["policy_tag"] == answer_tag][
            ["subset", "seed", "question_id", "dataset", "backend", "topk", metric]
        ].rename(columns={metric: "answer_only_score"})
        common_seeds = sorted(set(evidence_rows["seed"]) & set(answer_rows["seed"]))
        if not common_seeds:
            continue
        merged = evidence_rows[evidence_rows["seed"].isin(common_seeds)].merge(
            answer_rows[answer_rows["seed"].isin(common_seeds)],
            on=["subset", "seed", "question_id", "dataset", "backend", "topk"],
            validate="one_to_one",
        )
        merged["evidence_policy"] = evidence_tag
        merged["answer_policy"] = answer_tag
        merged["metric"] = metric
        merged["evidence_reward_value"] = (
            merged["evidence_reward_score"] - merged["answer_only_score"]
        )
        rows.append(merged)
    if not rows:
        raise RuntimeError("EXP-005 has no seed-matched answer-only specialist results")
    return pd.concat(rows, ignore_index=True)


def aggregate(frame: pd.DataFrame, value: str, extra_groups: list[str] | None = None) -> pd.DataFrame:
    groups = ["subset", "dataset", "topk", "metric", *(extra_groups or [])]
    return (
        frame.groupby(groups, as_index=False)[value]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": value, "std": f"{value}_std", "count": "cells"})
        .sort_values(groups)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hard-results", required=True)
    parser.add_argument("--exp003-results", required=True)
    parser.add_argument("--exp004-results", required=True)
    parser.add_argument("--exp005-results")
    parser.add_argument("--exp006-results")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    hard = load_jsonl_tree(Path(args.hard_results))
    matched = matched_hard_question_ids(hard)
    hard = add_subsets(hard, matched)
    exp003 = add_subsets(load_jsonl_tree(Path(args.exp003_results)), matched)
    exp004 = add_subsets(load_jsonl_tree(Path(args.exp004_results)), matched)

    regrets = pd.concat(
        [mixed_regret(hard, exp003, metric) for metric in REPORT_METRICS], ignore_index=True
    )
    values = pd.concat(
        [metadata_value(exp003, exp004, metric) for metric in REPORT_METRICS], ignore_index=True
    )
    regret_summary = aggregate(regrets, "mixed_regret")
    metadata_summary = aggregate(values, "metadata_value")

    output_dir = ensure_dir(args.output_dir)
    regrets.to_csv(Path(output_dir) / "mixed_regret_cells.csv", index=False)
    values.to_csv(Path(output_dir) / "metadata_value_cells.csv", index=False)
    regret_summary.to_csv(Path(output_dir) / "mixed_regret_summary.csv", index=False)
    metadata_summary.to_csv(Path(output_dir) / "metadata_value_summary.csv", index=False)

    lines = [
        "# Numbered mixed-policy experiment report",
        "",
        "## EXP-003 specialist-oracle regret",
        "",
        "Positive values mean the specialist oracle remains better than mixed-blind.",
        "",
        markdown_table(regret_summary),
        "",
        "## EXP-004 backend-metadata value",
        "",
        "Positive values mean explicit backend identity improves the same shared-policy setup.",
        "",
        markdown_table(metadata_summary),
        "",
    ]

    if args.exp005_results:
        exp005 = add_subsets(load_jsonl_tree(Path(args.exp005_results)), matched)
        evidence_values = pd.concat(
            [evidence_reward_value(hard, exp005, metric) for metric in REPORT_METRICS],
            ignore_index=True,
        )
        evidence_summary = aggregate(
            evidence_values, "evidence_reward_value", extra_groups=["evidence_policy"]
        )
        evidence_values.to_csv(Path(output_dir) / "evidence_reward_value_cells.csv", index=False)
        evidence_summary.to_csv(Path(output_dir) / "evidence_reward_value_summary.csv", index=False)
        lines.extend(
            [
                "## EXP-005 evidence-aware reward value",
                "",
                "Positive values mean evidence-aware reward improves over the seed-matched answer-only specialist.",
                "",
                markdown_table(evidence_summary),
                "",
            ]
        )

    if args.exp006_results:
        hybrid = load_jsonl_tree(Path(args.exp006_results))
        hybrid_summary = (
            hybrid.groupby(["policy_tag", "seed", "dataset", "topk"], as_index=False)[REPORT_METRICS]
            .mean()
            .sort_values(["dataset", "topk", "policy_tag", "seed"])
        )
        hybrid_summary.to_csv(Path(output_dir) / "hybrid_summary.csv", index=False)
        lines.extend(["## EXP-006 held-out hybrid RRF", "", markdown_table(hybrid_summary), ""])

    lines.extend(
        [
            "## Decision rule",
            "",
            "The latent stack-identification direction is supported when specialist-oracle regret is material and backend-metadata value closes most of that regret. If mixed-blind already has near-zero regret, online identification is unnecessary. If metadata value is near zero while regret remains large, the issue is optimization or capacity rather than missing backend information. EXP-005 diagnoses whether answer-only reward hid retrieval specialization; it is not a substitute for EXP-003.",
            "",
        ]
    )
    report = Path(output_dir) / "NUMBERED_EXPERIMENT_REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
