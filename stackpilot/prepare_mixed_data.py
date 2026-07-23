from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

MARKER_TEMPLATE = "<retrieval_environment>{backend}</retrieval_environment>\n"


def add_marker(prompt: list[dict[str, Any]], backend: str) -> list[dict[str, Any]]:
    if backend not in {"bm25", "e5"}:
        raise ValueError(f"backend must be bm25 or e5; got {backend!r}")
    updated = copy.deepcopy(prompt)
    for message in updated:
        if str(message.get("role", "")) == "user":
            message["content"] = MARKER_TEMPLATE.format(backend=backend) + str(
                message.get("content", "")
            )
            return updated
    raise ValueError("Search-R1 row has no user prompt to annotate")


def duplicate_row(row: dict[str, Any], backend: str) -> dict[str, Any]:
    output = dict(row)
    output["prompt"] = add_marker(list(row["prompt"]), backend)
    extra_info = dict(row.get("extra_info") or {})
    extra_info["routing_backend"] = backend
    output["extra_info"] = extra_info
    return output


def prepare(input_path: Path, output_path: Path, seed: int) -> None:
    from datasets import Dataset, load_dataset

    source = load_dataset("parquet", data_files=str(input_path), split="train")
    rows: list[dict[str, Any]] = []
    for row in source:
        base = dict(row)
        rows.append(duplicate_row(base, "bm25"))
        rows.append(duplicate_row(base, "e5"))
    dataset = Dataset.from_list(rows).shuffle(seed=seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(output_path))
    print(f"Wrote {len(dataset):,} backend-ID rows to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(Path(args.input), Path(args.output), args.seed)


if __name__ == "__main__":
    main()
