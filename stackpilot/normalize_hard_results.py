from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from stackpilot.common import read_jsonl_tolerant


def recall_at(turns: list[dict], turn_number: int) -> float:
    if not turns:
        return 0.0
    index = turn_number - 1
    if index < len(turns):
        return float(turns[index].get("support_recall", 0.0))
    return float(turns[-1].get("support_recall", 0.0))


def gain_at(turns: list[dict], turn_number: int) -> float:
    index = turn_number - 1
    if index < len(turns):
        return float(turns[index].get("evidence_gain", 0.0))
    return 0.0


def normalize(row: dict) -> dict:
    turns = list(row.get("turns") or [])
    turn1 = recall_at(turns, 1)
    turn2 = recall_at(turns, 2)
    turn3 = recall_at(turns, 3)
    first_miss = turn1 < 1.0
    row.update(
        {
            "retrieved_support_title_recall": (
                float(turns[-1].get("retrieved_support_title_recall", turn1))
                if turns
                else 0.0
            ),
            "observed_support_title_recall": (
                float(turns[-1].get("observed_support_title_recall", turn1))
                if turns
                else 0.0
            ),
            "turn1_support_recall": turn1,
            "turn2_support_recall": turn2,
            "turn3_support_recall": turn3,
            "turn2_evidence_gain": gain_at(turns, 2),
            "turn3_evidence_gain": gain_at(turns, 3),
            "recovery_at_2": float(first_miss and turn2 > turn1),
            "recovery_at_3": float(first_miss and turn3 > turn1),
            "full_recovery_at_2": float(first_miss and turn2 >= 1.0),
            "full_recovery_at_3": float(first_miss and turn3 >= 1.0),
        }
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="work/hard_rq0/results/policies")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    paths = sorted(results_dir.glob("*.jsonl"))
    if not paths:
        raise RuntimeError(f"No JSONL results found under {results_dir}")
    for path in paths:
        rows = [normalize(row) for row in read_jsonl_tolerant(path)]
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        print(f"Normalized {len(rows)} rows: {path}")


if __name__ == "__main__":
    main()
