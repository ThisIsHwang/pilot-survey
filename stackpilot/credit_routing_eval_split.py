from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from stackpilot.credit_routing_common import (
    atomic_write_json,
    atomic_write_jsonl,
    load_config,
    question_split,
)


def question_identifier(row: dict[str, Any]) -> str:
    for key in ("question_id", "id", "_id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    extra = row.get("extra_info")
    if isinstance(extra, dict):
        for key in ("question_id", "index", "id"):
            value = extra.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    raise RuntimeError("Endpoint row has no stable question identifier")


def run(config_path: str, input_path: str, output_path: str) -> dict[str, Any]:
    cfg = load_config(config_path)
    desired = str(cfg["endpoint"]["split"])
    source = Path(input_path).resolve()
    destination = Path(output_path).resolve()
    rows: list[dict[str, Any]] = []
    counts = {"train": 0, "validation": 0, "test": 0}
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(f"{source}:{line_number} is not a JSON object")
            question_id = question_identifier(row)
            split = question_split(
                question_id,
                salt=str(cfg["estimator"]["split_salt"]),
                train_fraction=float(cfg["estimator"]["train_fraction"]),
                validation_fraction=float(cfg["estimator"]["validation_fraction"]),
            )
            counts[split] += 1
            if split == desired:
                copy = dict(row)
                copy["credit_routing_split"] = split
                copy["credit_routing_question_id"] = question_id
                rows.append(copy)
    if not rows:
        raise RuntimeError(f"No endpoint rows were assigned to split {desired!r}")
    atomic_write_jsonl(destination, rows)
    manifest = {
        "schema": 1,
        "source": str(source),
        "output": str(destination),
        "selected_split": desired,
        "selected_rows": len(rows),
        "split_counts": counts,
        "split_salt": str(cfg["estimator"]["split_salt"]),
    }
    atomic_write_json(destination.with_suffix(".manifest.json"), manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/credit_routing.yaml")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.input, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
