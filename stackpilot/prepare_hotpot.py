from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

from stackpilot.common import ensure_dir, load_config, set_seed, stable_id, write_jsonl


def context_rows(example: dict) -> list[tuple[str, str]]:
    titles = example["context"]["title"]
    sentence_lists = example["context"]["sentences"]
    return [
        (str(title), " ".join(sentences).strip())
        for title, sentences in zip(titles, sentence_lists)
    ]


def prepare(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(int(cfg["seed"]))
    data_cfg = cfg["data"]
    work_dir = ensure_dir(Path(cfg["work_dir"]).resolve())
    data_dir = ensure_dir(work_dir / "data")

    manifest_path = data_dir / ".pilot-manifest.json"
    incomplete_path = data_dir / ".pilot-prepare-incomplete"
    manifest_config = {
        "schema": 2,
        "seed": int(cfg["seed"]),
        "dataset_name": data_cfg["dataset_name"],
        "dataset_config": data_cfg["dataset_config"],
        "split_train": data_cfg["split_train"],
        "split_eval": data_cfg["split_eval"],
        "train_examples": int(data_cfg["train_examples"]),
        "eval_examples": int(data_cfg["eval_examples"]),
    }
    output_paths = {
        "corpus": data_dir / "corpus.jsonl",
        "train": data_dir / "queries_train.jsonl",
        "eval": data_dir / "queries_eval.jsonl",
    }

    def count_rows(path: Path) -> int:
        if not path.is_file():
            return -1
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())

    def output_state(path: Path) -> dict[str, int | str]:
        digest = hashlib.sha256()
        rows = 0
        with path.open("rb") as handle:
            for line in handle:
                digest.update(line)
                if line.strip():
                    rows += 1
        return {"rows": rows, "sha256": digest.hexdigest()}

    def completed_manifest() -> dict:
        return {
            **manifest_config,
            "outputs": {
                name: output_state(path) for name, path in output_paths.items()
            },
        }

    outputs_match = (
        count_rows(output_paths["corpus"]) > 0
        and count_rows(output_paths["train"]) == manifest_config["train_examples"]
        and count_rows(output_paths["eval"]) == manifest_config["eval_examples"]
    )
    manifest_matches = False
    legacy_manifest_matches = False
    if manifest_path.is_file():
        try:
            current_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if outputs_match and not incomplete_path.exists():
                manifest_matches = current_manifest == completed_manifest()
                legacy_config = {**manifest_config, "schema": 1}
                legacy_manifest_matches = current_manifest == legacy_config
        except (UnicodeDecodeError, json.JSONDecodeError, OSError):
            manifest_matches = False
    legacy_outputs_match = (
        not manifest_path.exists() and not incomplete_path.exists() and outputs_match
    )
    if os.environ.get("FORCE_PREPARE") != "1" and (
        manifest_matches or legacy_manifest_matches or legacy_outputs_match
    ):
        if not manifest_matches:
            temporary_manifest = manifest_path.with_name(
                f".{manifest_path.name}.{os.getpid()}.tmp"
            )
            temporary_manifest.write_text(
                json.dumps(completed_manifest(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary_manifest, manifest_path)
        print(f"Reusing prepared data: {data_dir}")
        print(f"Corpus documents: {count_rows(output_paths['corpus']):,}")
        return

    # Prevent an interrupted promotion from being mistaken for a complete
    # legacy cache on the next run.
    incomplete_path.write_text(f"pid={os.getpid()}\n", encoding="utf-8")
    if manifest_path.exists():
        manifest_path.unlink()

    dataset = load_dataset(data_cfg["dataset_name"], data_cfg["dataset_config"])
    train = (
        dataset[data_cfg["split_train"]]
        .shuffle(seed=cfg["seed"])
        .select(range(data_cfg["train_examples"]))
    )
    evaluation = (
        dataset[data_cfg["split_eval"]]
        .shuffle(seed=cfg["seed"] + 1)
        .select(range(data_cfg["eval_examples"]))
    )

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
        support_titles = list(
            dict.fromkeys(str(x) for x in example["supporting_facts"]["title"])
        )
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

    missing = sum(1 for q in train_queries + eval_queries if not q["support_doc_ids"])
    if missing:
        raise RuntimeError(f"{missing} queries have no mapped supporting documents")

    with tempfile.TemporaryDirectory(prefix=".data-", dir=work_dir) as temp_name:
        temp_dir = Path(temp_name)
        temporary = {
            "corpus": temp_dir / "corpus.jsonl",
            "train": temp_dir / "queries_train.jsonl",
            "eval": temp_dir / "queries_eval.jsonl",
        }
        write_jsonl(temporary["corpus"], corpus_by_id.values())
        write_jsonl(temporary["train"], train_queries)
        write_jsonl(temporary["eval"], eval_queries)
        for name, target in output_paths.items():
            os.replace(temporary[name], target)

    temporary_manifest = manifest_path.with_name(
        f".{manifest_path.name}.{os.getpid()}.tmp"
    )
    temporary_manifest.write_text(
        json.dumps(completed_manifest(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)
    incomplete_path.unlink(missing_ok=True)
    print(f"Wrote {len(corpus_by_id):,} documents")
    print(f"Wrote {len(train_queries):,} train queries")
    print(f"Wrote {len(eval_queries):,} eval queries")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    args = parser.parse_args()
    prepare(args.config)
