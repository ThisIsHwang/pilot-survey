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

    matrix_path = results / "retrieval_matrix_summary.csv"
    if matrix_path.exists():
        matrix = pd.read_csv(matrix_path)
        lines += ["## Retrieval-only query-style matrix", "", matrix.to_markdown(index=False), ""]
        pivot = matrix.pivot(index="backend", columns="style", values="support_recall")
        gap = (pivot.max(axis=1) - pivot.min(axis=1)).sort_values(ascending=False)
        lines += ["### Query-style sensitivity", "", gap.rename("max_minus_min_recall").to_frame().to_markdown(), ""]

    agent_path = results / "agent_eval_summary.csv"
    if agent_path.exists():
        agent = pd.read_csv(agent_path)
        lines += ["## Blind vs backend-guideline oracle", "", agent.to_markdown(index=False), ""]
        blind = agent[agent.variant == "blind"].set_index("backend")
        oracle = agent[agent.variant == "oracle_guidance"].set_index("backend")
        joined = oracle[["em", "support_recall"]].join(blind[["em", "support_recall"]], lsuffix="_oracle", rsuffix="_blind")
        joined["em_oracle_gap"] = joined["em_oracle"] - joined["em_blind"]
        joined["support_oracle_gap"] = joined["support_recall_oracle"] - joined["support_recall_blind"]
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
    output = results / "REPORT.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
