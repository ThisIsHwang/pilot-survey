from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset

from stackpilot.common import read_jsonl

PROMPT = """Answer the given question. You must conduct reasoning inside <think> and </think> every time you get new information. If you lack knowledge, call search using <search> query </search>. The search result appears inside <information> and </information>. When ready, output only <answer> short answer </answer>. Question: {question}\n"""


def convert(input_path: Path, output_path: Path, split: str) -> None:
    rows = []
    for idx, item in enumerate(read_jsonl(input_path)):
        rows.append(
            {
                "data_source": "hotpotqa",
                "prompt": [{"role": "user", "content": PROMPT.format(question=item["question"])}],
                "ability": "fact-reasoning",
                "reward_model": {"style": "rule", "ground_truth": {"target": [item["answer"]]}},
                "extra_info": {"split": split, "index": idx, "question_id": item["id"]},
            }
        )
    Dataset.from_list(rows).to_parquet(str(output_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", default="work")
    args = parser.parse_args()
    root = Path(args.work_dir)
    output = root / "searchr1_hotpot"
    output.mkdir(parents=True, exist_ok=True)
    convert(root / "data/queries_train.jsonl", output / "train.parquet", "train")
    convert(root / "data/queries_eval.jsonl", output / "test.parquet", "test")
