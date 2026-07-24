from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from stackpilot.common import ensure_dir, load_config, set_seed, stable_id, write_jsonl

PILOT_DATA_MANIFEST_SCHEMA = 3
QUERY_OUTPUT_SPECS = {
    "train": {
        "filename": "queries_train.jsonl",
        "role": "trainer_train",
        "row_split": "train",
    },
    "dev": {
        "filename": "queries_dev.jsonl",
        "role": "trainer_validation",
        "row_split": "trainer_dev",
    },
    "eval": {
        "filename": "queries_eval.jsonl",
        "role": "final_evaluation",
        "row_split": "final_eval",
    },
}


def context_rows(example: dict) -> list[tuple[str, str]]:
    titles = example["context"]["title"]
    sentence_lists = example["context"]["sentences"]
    return [
        (str(title), " ".join(sentences).strip())
        for title, sentences in zip(titles, sentence_lists)
    ]


def load_source_dataset(name: str, config: str, revision: str) -> Any:
    from datasets import load_dataset

    return load_dataset(name, config, revision=revision)


def file_state(path: Path) -> dict[str, int | str]:
    digest = hashlib.sha256()
    rows = 0
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            if line.strip():
                rows += 1
    return {"rows": rows, "sha256": digest.hexdigest()}


def question_ids_sha256(question_ids: list[str]) -> str:
    canonical = json.dumps(
        question_ids, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _query_file_state(
    path: Path, *, expected_split: str
) -> tuple[dict[str, int | str], list[str]]:
    state = file_state(path)
    question_ids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON in {path} at line {line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise RuntimeError(
                    f"Expected an object in {path} at line {line_number}"
                )
            question_id = str(row.get("id", "")).strip()
            if not question_id:
                raise RuntimeError(
                    f"Missing question ID in {path} at line {line_number}"
                )
            if row.get("split") != expected_split:
                raise RuntimeError(
                    f"{path} line {line_number} has split={row.get('split')!r}; "
                    f"expected {expected_split!r}"
                )
            question_ids.append(question_id)
    if len(question_ids) != len(set(question_ids)):
        raise RuntimeError(f"Duplicate question IDs in {path}")
    state["question_ids_sha256"] = question_ids_sha256(question_ids)
    return state, question_ids


def pilot_data_configuration(cfg: dict[str, Any]) -> dict[str, Any]:
    data_cfg = cfg["data"]
    required = {
        "dataset_name",
        "dataset_config",
        "revision",
        "split_train",
        "split_eval",
        "train_examples",
        "trainer_dev_examples",
        "eval_examples",
    }
    missing = sorted(required - set(data_cfg))
    if missing:
        raise ValueError(
            "Missing required Hotpot data config fields: " + ", ".join(missing)
        )
    values = {
        "seed": int(cfg["seed"]),
        "dataset_name": str(data_cfg["dataset_name"]),
        "dataset_config": str(data_cfg["dataset_config"]),
        "revision": str(data_cfg["revision"]),
        "split_train": str(data_cfg["split_train"]),
        "split_eval": str(data_cfg["split_eval"]),
        "train_examples": int(data_cfg["train_examples"]),
        "trainer_dev_examples": int(data_cfg["trainer_dev_examples"]),
        "eval_examples": int(data_cfg["eval_examples"]),
    }
    for name in ("train_examples", "trainer_dev_examples", "eval_examples"):
        if values[name] < 1:
            raise ValueError(f"data.{name} must be positive; got {values[name]}")
    if values["split_train"] == values["split_eval"]:
        raise ValueError(
            "data.split_train and data.split_eval must be different so final "
            "evaluation is isolated from trainer-visible examples"
        )
    if re.fullmatch(r"[0-9a-f]{40}", values["revision"]) is None:
        raise ValueError(
            "data.revision must be a full immutable 40-character commit SHA; "
            f"got {values['revision']!r}"
        )
    return values


def _build_completed_manifest(
    data_dir: Path, configuration: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    corpus_path = data_dir / "corpus.jsonl"
    if not corpus_path.is_file():
        raise RuntimeError(f"Missing prepared corpus: {corpus_path}")
    corpus_state = file_state(corpus_path)
    if int(corpus_state["rows"]) < 1:
        raise RuntimeError(f"Prepared corpus is empty: {corpus_path}")

    outputs: dict[str, dict[str, Any]] = {
        "corpus": {
            "path": "corpus.jsonl",
            "role": "retrieval_corpus",
            **corpus_state,
        }
    }
    ids_by_split: dict[str, list[str]] = {}
    source_splits = {
        "train": configuration["split_train"],
        "dev": configuration["split_train"],
        "eval": configuration["split_eval"],
    }
    expected_counts = {
        "train": configuration["train_examples"],
        "dev": configuration["trainer_dev_examples"],
        "eval": configuration["eval_examples"],
    }
    for name, spec in QUERY_OUTPUT_SPECS.items():
        path = data_dir / spec["filename"]
        if not path.is_file():
            raise RuntimeError(f"Missing prepared query artifact: {path}")
        state, question_ids = _query_file_state(
            path, expected_split=spec["row_split"]
        )
        if state["rows"] != expected_counts[name]:
            raise RuntimeError(
                f"{path} has {state['rows']} rows; expected {expected_counts[name]}"
            )
        ids_by_split[name] = question_ids
        outputs[name] = {
            "path": spec["filename"],
            "role": spec["role"],
            "source_split": source_splits[name],
            **state,
        }

    names = tuple(QUERY_OUTPUT_SPECS)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = set(ids_by_split[left]) & set(ids_by_split[right])
            if overlap:
                example = sorted(overlap)[0]
                raise RuntimeError(
                    f"Prepared {left}/{right} question IDs overlap "
                    f"({len(overlap)} rows; example {example})"
                )

    train_count = configuration["train_examples"]
    dev_count = configuration["trainer_dev_examples"]
    selections = {
        "train": {
            "role": "trainer_train",
            "source_split": configuration["split_train"],
            "shuffle_seed": configuration["seed"],
            "slice_start": 0,
            "slice_stop": train_count,
            "count": train_count,
            "question_ids_sha256": outputs["train"]["question_ids_sha256"],
        },
        "dev": {
            "role": "trainer_validation",
            "source_split": configuration["split_train"],
            "shuffle_seed": configuration["seed"],
            "slice_start": train_count,
            "slice_stop": train_count + dev_count,
            "count": dev_count,
            "question_ids_sha256": outputs["dev"]["question_ids_sha256"],
        },
        "eval": {
            "role": "final_evaluation",
            "source_split": configuration["split_eval"],
            "shuffle_seed": configuration["seed"] + 1,
            "slice_start": 0,
            "slice_stop": configuration["eval_examples"],
            "count": configuration["eval_examples"],
            "question_ids_sha256": outputs["eval"]["question_ids_sha256"],
        },
    }
    return (
        {
            "schema": PILOT_DATA_MANIFEST_SCHEMA,
            "configuration": configuration,
            "selections": selections,
            "outputs": outputs,
        },
        ids_by_split,
    )


def validate_pilot_data_manifest(
    data_dir: str | Path,
    *,
    expected_configuration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data_dir = Path(data_dir)
    manifest_path = data_dir / ".pilot-manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(
            f"Missing Stage-0 data manifest: {manifest_path}. "
            "Rerun scripts/prepare_data.sh."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"Invalid Stage-0 data manifest: {manifest_path}") from exc
    if manifest.get("schema") != PILOT_DATA_MANIFEST_SCHEMA:
        raise RuntimeError(
            f"Unsupported Stage-0 data manifest schema "
            f"{manifest.get('schema')!r}; expected {PILOT_DATA_MANIFEST_SCHEMA}. "
            "Rerun scripts/prepare_data.sh to rebuild the split-safe cache."
        )
    configuration = manifest.get("configuration")
    if not isinstance(configuration, dict):
        raise RuntimeError(f"Missing configuration in {manifest_path}")
    if expected_configuration is not None and configuration != expected_configuration:
        raise RuntimeError(
            f"Prepared data configuration does not match the requested config: "
            f"{manifest_path}"
        )
    completed, _ = _build_completed_manifest(data_dir, configuration)
    if manifest != completed:
        raise RuntimeError(
            f"Prepared data artifacts or provenance do not match {manifest_path}. "
            "Rerun scripts/prepare_data.sh."
        )
    return manifest


def prepare(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(int(cfg["seed"]))
    configuration = pilot_data_configuration(cfg)
    work_dir = ensure_dir(Path(cfg["work_dir"]).resolve())
    data_dir = ensure_dir(work_dir / "data")

    manifest_path = data_dir / ".pilot-manifest.json"
    incomplete_path = data_dir / ".pilot-prepare-incomplete"
    output_paths = {
        "corpus": data_dir / "corpus.jsonl",
        "train": data_dir / "queries_train.jsonl",
        "dev": data_dir / "queries_dev.jsonl",
        "eval": data_dir / "queries_eval.jsonl",
    }
    manifest_matches = False
    if manifest_path.is_file() and not incomplete_path.exists():
        try:
            validate_pilot_data_manifest(
                data_dir, expected_configuration=configuration
            )
            manifest_matches = True
        except RuntimeError:
            manifest_matches = False
    if os.environ.get("FORCE_PREPARE") != "1" and manifest_matches:
        print(f"Reusing prepared data: {data_dir}")
        print(f"Corpus documents: {file_state(output_paths['corpus'])['rows']:,}")
        return

    # Old schemas and interrupted promotions are deliberately not reusable:
    # neither proves that trainer development and final evaluation are disjoint.
    incomplete_path.write_text(f"pid={os.getpid()}\n", encoding="utf-8")
    if manifest_path.exists():
        manifest_path.unlink()

    dataset = load_source_dataset(
        configuration["dataset_name"],
        configuration["dataset_config"],
        configuration["revision"],
    )
    for split_name in (configuration["split_train"], configuration["split_eval"]):
        if split_name not in dataset:
            raise RuntimeError(f"Dataset is missing required split {split_name!r}")
    train_source = dataset[configuration["split_train"]]
    eval_source = dataset[configuration["split_eval"]]
    required_train = (
        configuration["train_examples"] + configuration["trainer_dev_examples"]
    )
    if len(train_source) < required_train:
        raise RuntimeError(
            f"Source split {configuration['split_train']!r} has "
            f"{len(train_source):,} rows; exactly {required_train:,} are required "
            "for disjoint trainer train/dev selections"
        )
    if len(eval_source) < configuration["eval_examples"]:
        raise RuntimeError(
            f"Source split {configuration['split_eval']!r} has "
            f"{len(eval_source):,} rows; exactly "
            f"{configuration['eval_examples']:,} final-evaluation rows are required"
        )

    shuffled_train = train_source.shuffle(seed=configuration["seed"])
    train = list(
        shuffled_train.select(range(0, configuration["train_examples"]))
    )
    trainer_dev = list(
        shuffled_train.select(
            range(configuration["train_examples"], required_train)
        )
    )
    evaluation = list(
        eval_source.shuffle(seed=configuration["seed"] + 1).select(
            range(configuration["eval_examples"])
        )
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
    for ex in trainer_dev:
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
    dev_queries = [make_query(ex, "trainer_dev") for ex in trainer_dev]
    eval_queries = [make_query(ex, "final_eval") for ex in evaluation]

    all_queries = train_queries + dev_queries + eval_queries
    missing = sum(1 for q in all_queries if not q["support_doc_ids"])
    if missing:
        raise RuntimeError(f"{missing} queries have no mapped supporting documents")
    query_ids = [[str(query["id"]) for query in rows] for rows in (
        train_queries,
        dev_queries,
        eval_queries,
    )]
    for name, ids in zip(("train", "dev", "eval"), query_ids):
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"Source {name} selection has duplicate question IDs")
    for index, left in enumerate(query_ids):
        for right in query_ids[index + 1 :]:
            if set(left) & set(right):
                raise RuntimeError(
                    "Source selections are not disjoint by question ID; refusing "
                    "to contaminate trainer validation or final evaluation"
                )

    with tempfile.TemporaryDirectory(prefix=".data-", dir=work_dir) as temp_name:
        temp_dir = Path(temp_name)
        temporary = {
            "corpus": temp_dir / "corpus.jsonl",
            "train": temp_dir / "queries_train.jsonl",
            "dev": temp_dir / "queries_dev.jsonl",
            "eval": temp_dir / "queries_eval.jsonl",
        }
        write_jsonl(temporary["corpus"], corpus_by_id.values())
        write_jsonl(temporary["train"], train_queries)
        write_jsonl(temporary["dev"], dev_queries)
        write_jsonl(temporary["eval"], eval_queries)
        for name, target in output_paths.items():
            os.replace(temporary[name], target)

    completed_manifest, _ = _build_completed_manifest(data_dir, configuration)
    temporary_manifest = manifest_path.with_name(
        f".{manifest_path.name}.{os.getpid()}.tmp"
    )
    temporary_manifest.write_text(
        json.dumps(completed_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)
    incomplete_path.unlink(missing_ok=True)
    validate_pilot_data_manifest(data_dir, expected_configuration=configuration)
    print(f"Wrote {len(corpus_by_id):,} documents")
    print(f"Wrote {len(train_queries):,} train queries")
    print(f"Wrote {len(dev_queries):,} trainer-dev queries")
    print(f"Wrote {len(eval_queries):,} final-eval queries")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    args = parser.parse_args()
    prepare(args.config)
