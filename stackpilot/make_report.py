from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from stackpilot.common import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    results = Path(cfg["work_dir"]).resolve() / "results"
    lines = ["# Stack-adaptation pilot report", ""]
    run_signatures: dict[str, str] = {}

    def read_summary(path: Path, label: str) -> pd.DataFrame:
        frame = pd.read_csv(path)
        if "run_signature" not in frame or "n_questions" not in frame:
            raise RuntimeError(
                f"{path} predates resumable run identities; rerun the corresponding evaluation"
            )
        signatures = {str(value) for value in frame["run_signature"].dropna()}
        if len(signatures) != 1:
            raise RuntimeError(f"{path} contains multiple or missing run signatures")
        run_signatures[label] = signatures.pop()
        return frame

    matrix_path = results / "retrieval_matrix_summary.csv"
    if matrix_path.exists():
        matrix = read_summary(matrix_path, "retrieval matrix")
        lines += [
            "## Retrieval-only query-style matrix",
            "",
            matrix.drop(columns="run_signature").to_markdown(index=False),
            "",
        ]
        style_matrix = matrix[matrix["style"] != "rrf_ensemble"]
        pivot = style_matrix.pivot(
            index="backend", columns="style", values="support_recall"
        )
        gap = (pivot.max(axis=1) - pivot.min(axis=1)).sort_values(ascending=False)
        lines += [
            "### Query-style sensitivity",
            "",
            gap.rename("max_minus_min_recall").to_frame().to_markdown(),
            "",
        ]

    style_oracle_path = results / "retrieval_style_oracle.csv"
    if style_oracle_path.exists():
        style_oracle = read_summary(style_oracle_path, "retrieval oracle")
        lines += [
            "### Per-question style oracle (RRF ensemble excluded)",
            "",
            style_oracle.drop(columns="run_signature").to_markdown(index=False),
            "",
        ]

    agent_path = results / "agent_eval_summary.csv"
    if agent_path.exists():
        agent = read_summary(agent_path, "agent evaluation")
        lines += [
            "## Blind vs backend-guideline oracle",
            "",
            agent.drop(columns="run_signature").to_markdown(index=False),
            "",
        ]
        blind = agent[agent.variant == "blind"].set_index("backend")
        oracle = agent[agent.variant == "oracle_guidance"].set_index("backend")
        joined = oracle[["em", "support_recall"]].join(
            blind[["em", "support_recall"]], lsuffix="_oracle", rsuffix="_blind"
        )
        joined["em_oracle_gap"] = joined["em_oracle"] - joined["em_blind"]
        joined["support_oracle_gap"] = (
            joined["support_recall_oracle"] - joined["support_recall_blind"]
        )
        lines += ["### Oracle gaps", "", joined.to_markdown(), ""]

    lines += [
        "## Go / no-go criteria",
        "",
        "Proceed to RL only when at least two hold:",
        "",
        "- Blind agent answer or support-recall spread across backends is >= 5 percentage points.",
        "- Backend-guideline oracle improves the weakest backend by >= 3 percentage points.",
        "- Best query style differs by backend and per-question style oracle is materially above the best fixed style.",
        "- Query ensemble does not already close most of the gap.",
        "",
    ]
    distinct_signatures = set(run_signatures.values())
    if len(distinct_signatures) > 1:
        details = ", ".join(
            f"{label}={signature}" for label, signature in run_signatures.items()
        )
        raise RuntimeError(f"Refusing to mix summaries from different runs: {details}")
    if distinct_signatures:
        signature = next(iter(distinct_signatures))
        lines[2:2] = [f"Run signature: `{signature}`", ""]
    output = results / "REPORT.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
