from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from datasets import Dataset, concatenate_datasets, load_dataset

from stackpilot.common import ensure_dir, load_config, write_jsonl

PROMPT = (
    "Answer the given question. You must reason inside <think> and </think> whenever "
    "you receive new information. If knowledge is missing, call search with "
    "<search> query </search>. Search results appear inside <information> and "
    "</information>. When ready, output only <answer> short answer </answer>. "
    "Question: {question}\n"
)


def unique_strings(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def extract_support_titles(metadata: dict[str, Any]) -> list[str]:
    supporting = metadata.get("supporting_facts") or {}
    if isinstance(supporting, dict) and isinstance(supporting.get("title"), list):
        titles = unique_strings(supporting["title"])
        if titles:
            return titles

    paragraphs = metadata.get("paragraphs")
    if isinstance(paragraphs, list):
        titles = []
        for paragraph in paragraphs:
            if not isinstance(paragraph, dict):
                continue
            is_supporting = bool(
                paragraph.get("is_supporting")
                or paragraph.get("supporting")
                or paragraph.get("is_support")
            )
            if is_supporting:
                title = paragraph.get("title") or paragraph.get("wikipedia_title")
                if title:
                    titles.append(str(title))
        titles = unique_strings(titles)
        if titles:
            return titles

    context = metadata.get("context")
    if isinstance(context, list):
        titles = []
        for paragraph in context:
            if isinstance(paragraph, dict) and bool(
                paragraph.get("is_supporting") or paragraph.get("supporting")
            ):
                title = paragraph.get("title")
                if title:
                    titles.append(str(title))
        return unique_strings(titles)
    return []


def to_query(row: dict[str, Any], dataset_name: str, split: str) -> dict[str, Any]:
    answers = unique_strings(row.get("golden_answers") or [])
    if not answers:
        raise ValueError(f"{dataset_name}:{row.get('id')} has no golden answer")
    metadata = row.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        "id": f"{dataset_name}:{row['id']}",
        "dataset": dataset_name,
        "split": split,
        "question": str(row["question"]).strip(),
        "answers": answers,
        "answer": answers[0],
        "support_titles": extract_support_titles(metadata),
        "type": str(metadata.get("type", "")),
    }


def to_searchr1_row(item: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "data_source": item["dataset"],
        "prompt": [
            {
                "role": "user",
                "content": PROMPT.format(question=item["question"]),
            }
        ],
        "ability": "fact-reasoning",
        "reward_model": {
            "style": "rule",
            "ground_truth": {"target": item["answers"]},
        },
        "extra_info": {
            "split": item["split"],
            "index": index,
            "question_id": item["id"],
            "support_titles": item["support_titles"],
        },
    }


def prepare(config_path: str) -> None:
    cfg = load_config(config_path)
    work_dir = ensure_dir(Path(cfg["work_dir"]).resolve())
    data_root = ensure_dir(work_dir / "data")
    searchr1_root = ensure_dir(work_dir / "searchr1")
    data_cfg = cfg["data"]

    train_datasets: list[Dataset] = []
    eval_datasets: list[Dataset] = []
    all_eval_rows: list[dict[str, Any]] = []
    summary_lines = []

    for offset, dataset_name in enumerate(data_cfg["datasets"]):
        dataset = load_dataset(data_cfg["repo_id"], dataset_name)
        train_split = dataset[data_cfg["split_train"]]
        eval_split_name = data_cfg["split_eval"]
        if eval_split_name not in dataset:
            eval_split_name = "test" if "test" in dataset else "dev"
        eval_split = dataset[eval_split_name]

        train_count = min(int(data_cfg["train_examples_per_dataset"]), len(train_split))
        eval_count = min(int(data_cfg["eval_examples_per_dataset"]), len(eval_split))
        train_selected = train_split.shuffle(seed=int(cfg["seed"]) + offset).select(
            range(train_count)
        )
        eval_selected = eval_split.shuffle(seed=int(cfg["seed"]) + 100 + offset).select(
            range(eval_count)
        )

        train_rows = [to_query(dict(row), dataset_name, "train") for row in train_selected]
        eval_rows = [to_query(dict(row), dataset_name, "eval") for row in eval_selected]
        missing_support = sum(not row["support_titles"] for row in eval_rows)
        if missing_support:
            raise RuntimeError(
                f"{dataset_name}: {missing_support}/{len(eval_rows)} evaluation rows have "
                "no supporting-title metadata. Inspect the dataset schema before running."
            )

        dataset_dir = ensure_dir(data_root / dataset_name)
        write_jsonl(dataset_dir / "train.jsonl", train_rows)
        write_jsonl(dataset_dir / "eval.jsonl", eval_rows)
        all_eval_rows.extend(eval_rows)

        train_datasets.append(
            Dataset.from_list([to_searchr1_row(row, idx) for idx, row in enumerate(train_rows)])
        )
        eval_datasets.append(
            Dataset.from_list([to_searchr1_row(row, idx) for idx, row in enumerate(eval_rows)])
        )
        summary_lines.append(
            f"{dataset_name}: train={len(train_rows)}, eval={len(eval_rows)}, "
            f"mean_support_titles={sum(len(x['support_titles']) for x in eval_rows)/len(eval_rows):.2f}"
        )

    write_jsonl(data_root / "eval_all.jsonl", all_eval_rows)
    concatenate_datasets(train_datasets).shuffle(seed=int(cfg["seed"])).to_parquet(
        str(searchr1_root / "train.parquet")
    )
    concatenate_datasets(eval_datasets).to_parquet(str(searchr1_root / "test.parquet"))
    (data_root / "SUMMARY.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))
    print(f"Search-R1 parquet: {searchr1_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hard_rq0.yaml")
    args = parser.parse_args()
    prepare(args.config)
