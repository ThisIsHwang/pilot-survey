from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

from stackpilot.common import ensure_dir, load_config, set_seed, stable_id, write_jsonl


def context_rows(example: dict) -> list[tuple[str, str]]:
    titles = example["context"]["title"]
    sentence_lists = example["context"]["sentences"]
    return [(str(title), " ".join(sentences).strip()) for title, sentences in zip(titles, sentence_lists)]


def prepare(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(int(cfg["seed"]))
    data_cfg = cfg["data"]
    work_dir = ensure_dir(Path(cfg["work_dir"]).resolve())
    data_dir = ensure_dir(work_dir / "data")

    dataset = load_dataset(data_cfg["dataset_name"], data_cfg["dataset_config"])
    train = dataset[data_cfg["split_train"]].shuffle(seed=cfg["seed"]).select(range(data_cfg["train_examples"]))
    evaluation = dataset[data_cfg["split_eval"]].shuffle(seed=cfg["seed"] + 1).select(range(data_cfg["eval_examples"]))

    corpus_by_id: dict[str, dict] = {}
    dedup_to_doc_id: dict[str, str] = {}
    title_to_doc_ids: dict[str, list[str]] = defaultdict(list)

    def add_corpus(example: dict) -> None:
        for title, text in context_rows(example):
            dedup_key = stable_id(title, text)
            if dedup_key not in dedup_to_doc_id:
                # Search-R1's fallback corpus loader expects numeric document IDs.
                doc_id = str(len(corpus_by_id))
                dedup_to_doc_id[dedup_key] = doc_id
                corpus_by_id[doc_id] = {
                    "id": doc_id,
                    "title": title,
                    "text": text,
                    "contents": f'"{title}"\n{text}',
                }
                title_to_doc_ids[title].append(doc_id)

    for ex in train:
        add_corpus(ex)
    for ex in evaluation:
        add_corpus(ex)

    def make_query(example: dict, split: str) -> dict:
        support_titles = list(dict.fromkeys(str(x) for x in example["supporting_facts"]["title"]))
        support_doc_ids = []
        for title in support_titles:
            support_doc_ids.extend(title_to_doc_ids.get(title, []))
        return {
            "id": str(example["id"]),
            "split": split,
            "question": str(example["question"]).strip(),
            "answer": str(example["answer"]).strip(),
            "support_titles": support_titles,
            "support_doc_ids": list(dict.fromkeys(support_doc_ids)),
            "type": str(example.get("type", "")),
            "level": str(example.get("level", "")),
        }

    train_queries = [make_query(ex, "train") for ex in train]
    eval_queries = [make_query(ex, "eval") for ex in evaluation]

    write_jsonl(data_dir / "corpus.jsonl", corpus_by_id.values())
    write_jsonl(data_dir / "queries_train.jsonl", train_queries)
    write_jsonl(data_dir / "queries_eval.jsonl", eval_queries)

    print(f"Wrote {len(corpus_by_id):,} documents")
    print(f"Wrote {len(train_queries):,} train queries")
    print(f"Wrote {len(eval_queries):,} eval queries")
    missing = sum(1 for q in train_queries + eval_queries if not q["support_doc_ids"])
    if missing:
        raise RuntimeError(f"{missing} queries have no mapped supporting documents")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    args = parser.parse_args()
    prepare(args.config)
