from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stackpilot.common import ensure_dir, load_config, write_jsonl

DATA_MANIFEST_NAME = ".hard-rq0-data-manifest.json"
DATA_PREP_SCHEMA = 4
PINNED_REVISION_RE = re.compile(r"[0-9a-fA-F]{40}")

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


def titles_from_supporting_facts(value: Any) -> list[str]:
    if isinstance(value, dict):
        titles = value.get("title") or value.get("titles")
        if isinstance(titles, list):
            return unique_strings(titles)
        if isinstance(titles, str):
            return [titles.strip()] if titles.strip() else []
    if isinstance(value, list):
        titles = []
        for fact in value:
            if isinstance(fact, dict):
                title = fact.get("title") or fact.get("wikipedia_title")
                if title:
                    titles.append(str(title))
            elif isinstance(fact, (list, tuple)) and fact:
                titles.append(str(fact[0]))
        return unique_strings(titles)
    return []


def extract_support_titles(metadata: dict[str, Any]) -> list[str]:
    for key in ("supporting_titles", "support_titles", "gold_titles"):
        value = metadata.get(key)
        if isinstance(value, list):
            titles = unique_strings(value)
            if titles:
                return titles

    titles = titles_from_supporting_facts(metadata.get("supporting_facts"))
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
                or paragraph.get("is_gold")
            )
            if is_supporting:
                title = (
                    paragraph.get("title")
                    or paragraph.get("wikipedia_title")
                    or paragraph.get("page_title")
                )
                if title:
                    titles.append(str(title))
        titles = unique_strings(titles)
        if titles:
            return titles

    context = metadata.get("context")
    if isinstance(context, dict):
        context_titles = context.get("title") or context.get("titles")
        support_flags = (
            context.get("is_supporting")
            or context.get("supporting")
            or context.get("is_support")
        )
        if isinstance(context_titles, list) and isinstance(support_flags, list):
            return unique_strings(
                title
                for title, flag in zip(context_titles, support_flags)
                if bool(flag)
            )
    if isinstance(context, list):
        titles = []
        for paragraph in context:
            if isinstance(paragraph, dict) and bool(
                paragraph.get("is_supporting")
                or paragraph.get("supporting")
                or paragraph.get("is_support")
            ):
                title = paragraph.get("title") or paragraph.get("wikipedia_title")
                if title:
                    titles.append(str(title))
        titles = unique_strings(titles)
        if titles:
            return titles

    decomposition = metadata.get("question_decomposition")
    if isinstance(decomposition, list):
        titles = []
        for step in decomposition:
            if not isinstance(step, dict):
                continue
            paragraph = step.get("support_paragraph")
            if not isinstance(paragraph, dict):
                continue
            if paragraph.get("is_supporting") is False:
                continue
            title = (
                paragraph.get("title")
                or paragraph.get("wikipedia_title")
                or paragraph.get("page_title")
            )
            if title:
                titles.append(str(title))
        titles = unique_strings(titles)
        if titles:
            return titles
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


def to_searchr1_row(item: dict[str, Any]) -> dict[str, Any]:
    extra_info = {
        "split": item["split"],
        # Search-R1 copies this field to GRPO's uid. It must be globally
        # unique across every concatenated source dataset so unrelated
        # questions are never normalized as one rollout group.
        "index": item["id"],
        "question_id": item["id"],
        "support_titles": item["support_titles"],
    }
    routing_backend = str(item.get("routing_backend", "")).strip().lower()
    if routing_backend:
        if routing_backend not in {"bm25", "e5"}:
            raise ValueError(
                f"Unsupported hidden validation routing backend: {routing_backend!r}"
            )
        extra_info["routing_backend"] = routing_backend
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
        "extra_info": extra_info,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_request(cfg: dict[str, Any]) -> dict[str, Any]:
    data_cfg = cfg["data"]
    revision = str(data_cfg.get("revision") or "").strip()
    if PINNED_REVISION_RE.fullmatch(revision) is None:
        raise ValueError(
            "data.revision must be an immutable 40-character "
            f"Hugging Face commit SHA; got {revision!r}"
        )
    datasets = [str(value) for value in data_cfg["datasets"]]
    if not datasets or len(set(datasets)) != len(datasets):
        raise ValueError(f"data.datasets must be nonempty and unique; got {datasets}")
    invalid_names = [
        value for value in datasets if re.fullmatch(r"[A-Za-z0-9._-]+", value) is None
    ]
    if invalid_names:
        raise ValueError(f"Unsafe dataset names: {invalid_names}")
    train_count = int(data_cfg["train_examples_per_dataset"])
    eval_count = int(data_cfg["eval_examples_per_dataset"])
    # Older configs used the final evaluation rows for trainer validation as
    # well. Preserve config compatibility while assigning the two purposes
    # disjoint source rows under the new manifest schema.
    validation_count = int(data_cfg.get("validation_examples_per_dataset", eval_count))
    if train_count < 1 or validation_count < 1 or eval_count < 1:
        raise ValueError(
            "Hard-RQ0 train/validation/eval example counts must be positive; "
            f"got train={train_count}, validation={validation_count}, "
            f"eval={eval_count}"
        )
    return {
        "schema": DATA_PREP_SCHEMA,
        "repo_id": str(data_cfg["repo_id"]),
        "revision": revision,
        "datasets": datasets,
        "train_examples_per_dataset": train_count,
        "validation_examples_per_dataset": validation_count,
        "eval_examples_per_dataset": eval_count,
        "split_train": str(data_cfg["split_train"]),
        "split_eval": str(data_cfg["split_eval"]),
        "seed": int(cfg["seed"]),
        "prompt_sha256": hashlib.sha256(PROMPT.encode("utf-8")).hexdigest(),
        "preparer_sha256": file_sha256(Path(__file__)),
    }


def expected_artifacts(request: dict[str, Any]) -> list[str]:
    paths = [
        "data/eval_all.jsonl",
        "data/SUMMARY.txt",
        "searchr1/train.parquet",
        "searchr1/test.parquet",
    ]
    for dataset_name in request["datasets"]:
        paths.extend(
            (
                f"data/{dataset_name}/train.jsonl",
                f"data/{dataset_name}/validation.jsonl",
                f"data/{dataset_name}/eval.jsonl",
            )
        )
    return sorted(paths)


def artifact_records(root: Path, relative_paths: Iterable[str]) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for relative_path in relative_paths:
        path = root / relative_path
        if not path.is_file():
            raise RuntimeError(f"Prepared artifact is missing: {path}")
        records[relative_path] = {
            "size": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    return records


def prepared_cache_valid(
    manifest_path: Path, work_dir: Path, request: dict[str, Any]
) -> bool:
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if manifest.get("schema") != DATA_PREP_SCHEMA or manifest.get("request") != request:
        return False
    records = manifest.get("artifacts")
    expected = expected_artifacts(request)
    if not isinstance(records, dict) or set(records) != set(expected):
        return False
    for relative_path in expected:
        record = records.get(relative_path)
        path = work_dir / relative_path
        if not isinstance(record, dict) or not path.is_file():
            return False
        if record.get("size") != path.stat().st_size:
            return False
        if record.get("sha256") != file_sha256(path):
            return False
    return True


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_unique_rows(rows: list[dict[str, Any]], label: str) -> None:
    identifiers = [str(row.get("id", "")).strip() for row in rows]
    if any(not identifier for identifier in identifiers):
        raise RuntimeError(f"{label} contains an empty question ID")
    if len(set(identifiers)) != len(identifiers):
        raise RuntimeError(f"{label} contains duplicate question IDs")


def validate_disjoint_rows(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    left_label: str,
    right_label: str,
) -> None:
    left_ids = {str(row.get("id", "")).strip() for row in left}
    right_ids = {str(row.get("id", "")).strip() for row in right}
    overlap = sorted(left_ids & right_ids)
    if overlap:
        raise RuntimeError(
            f"{left_label} and {right_label} question IDs overlap; "
            f"examples={overlap[:10]}"
        )


def validate_support_titles(
    rows: list[dict[str, Any]],
    dataset_name: str,
    purpose: str,
    work_dir: Path,
) -> None:
    if purpose not in {"validation", "evaluation"}:
        raise ValueError(f"Unsupported support-title validation purpose: {purpose!r}")
    missing_rows = [row for row in rows if not row["support_titles"]]
    diagnostic_path = (
        work_dir / f"{dataset_name}_{purpose}_missing_support_examples.json"
    )
    if missing_rows:
        atomic_write_json(
            diagnostic_path,
            {"count": len(missing_rows), "examples": missing_rows[:10]},
        )
        raise RuntimeError(
            f"{dataset_name}: {len(missing_rows)}/{len(rows)} {purpose} "
            "rows have no supporting-title metadata. "
            f"Examples: {diagnostic_path}"
        )
    diagnostic_path.unlink(missing_ok=True)


def prepare(config_path: str) -> None:
    from datasets import Dataset, concatenate_datasets, load_dataset

    cfg = load_config(config_path)
    work_dir = ensure_dir(Path(cfg["work_dir"]).resolve())
    data_root = ensure_dir(work_dir / "data")
    manifest_path = data_root / DATA_MANIFEST_NAME
    request = prepare_request(cfg)
    if prepared_cache_valid(manifest_path, work_dir, request):
        print(f"Reusing verified hard-RQ0 data: {manifest_path}")
        print((data_root / "SUMMARY.txt").read_text(encoding="utf-8").rstrip())
        return

    revision = request["revision"]

    with tempfile.TemporaryDirectory(prefix=".hard-rq0-prepare-", dir=work_dir) as tmp:
        staging_root = Path(tmp)
        staging_data_root = ensure_dir(staging_root / "data")
        staging_searchr1_root = ensure_dir(staging_root / "searchr1")
        train_datasets: list[Any] = []
        validation_datasets: list[Any] = []
        all_train_rows: list[dict[str, Any]] = []
        all_validation_rows: list[dict[str, Any]] = []
        all_eval_rows: list[dict[str, Any]] = []
        summary_lines = [f"source_revision={revision}"]

        for offset, dataset_name in enumerate(request["datasets"]):
            dataset = load_dataset(request["repo_id"], dataset_name, revision=revision)
            train_split_name = request["split_train"]
            if train_split_name not in dataset:
                raise RuntimeError(
                    f"{dataset_name}: missing requested train split {train_split_name!r}; "
                    f"available={sorted(dataset)}"
                )
            train_split = dataset[train_split_name]
            eval_split_name = request["split_eval"]
            if eval_split_name not in dataset:
                eval_split_name = "test" if "test" in dataset else "dev"
            if eval_split_name not in dataset:
                raise RuntimeError(
                    f"{dataset_name}: missing requested eval split {request['split_eval']!r}; "
                    f"available={sorted(dataset)}"
                )
            eval_split = dataset[eval_split_name]

            train_count = request["train_examples_per_dataset"]
            validation_count = request["validation_examples_per_dataset"]
            eval_count = request["eval_examples_per_dataset"]
            if len(train_split) < train_count:
                raise RuntimeError(
                    f"{dataset_name}: requested {train_count} training rows from "
                    f"{train_split_name!r}, but the pinned split has only "
                    f"{len(train_split)}"
                )
            required_eval_rows = validation_count + eval_count
            if len(eval_split) < required_eval_rows:
                raise RuntimeError(
                    f"{dataset_name}: trainer validation and final evaluation "
                    "must use disjoint rows, requiring "
                    f"{validation_count}+{eval_count}={required_eval_rows} rows "
                    f"from {eval_split_name!r}; the pinned split has only "
                    f"{len(eval_split)}"
                )
            train_selected = train_split.shuffle(seed=request["seed"] + offset).select(
                range(train_count)
            )
            shuffled_eval = eval_split.shuffle(seed=request["seed"] + 100 + offset)
            # Preserve the original Hard-RQ0 final benchmark exactly. Earlier
            # schemas used the first eval_count rows of this pinned shuffle.
            # Trainer validation is a new, disjoint slice after those rows.
            eval_selected = shuffled_eval.select(range(eval_count))
            validation_selected = shuffled_eval.select(
                range(eval_count, eval_count + validation_count)
            )

            train_rows = [
                to_query(dict(row), dataset_name, "train") for row in train_selected
            ]
            validation_rows = [
                to_query(dict(row), dataset_name, "validation")
                for row in validation_selected
            ]
            # Mixed-policy validation is not n_agent-expanded upstream. Store a
            # hidden, deterministic route on each row so its environment is
            # explicit without exposing the backend in the prompt.
            for validation_index, validation_row in enumerate(validation_rows):
                validation_row["routing_backend"] = (
                    "bm25" if validation_index % 2 == 0 else "e5"
                )
            eval_rows = [
                to_query(dict(row), dataset_name, "eval") for row in eval_selected
            ]
            validate_unique_rows(train_rows, f"{dataset_name} train")
            validate_unique_rows(validation_rows, f"{dataset_name} validation")
            validate_unique_rows(eval_rows, f"{dataset_name} eval")
            validate_disjoint_rows(
                train_rows,
                validation_rows,
                f"{dataset_name} train",
                f"{dataset_name} validation",
            )
            validate_disjoint_rows(
                train_rows,
                eval_rows,
                f"{dataset_name} train",
                f"{dataset_name} eval",
            )
            validate_disjoint_rows(
                validation_rows,
                eval_rows,
                f"{dataset_name} validation",
                f"{dataset_name} eval",
            )
            validate_support_titles(
                validation_rows,
                dataset_name,
                "validation",
                work_dir,
            )
            validate_support_titles(
                eval_rows,
                dataset_name,
                "evaluation",
                work_dir,
            )

            dataset_dir = ensure_dir(staging_data_root / dataset_name)
            write_jsonl(dataset_dir / "train.jsonl", train_rows)
            write_jsonl(dataset_dir / "validation.jsonl", validation_rows)
            write_jsonl(dataset_dir / "eval.jsonl", eval_rows)
            all_train_rows.extend(train_rows)
            all_validation_rows.extend(validation_rows)
            all_eval_rows.extend(eval_rows)

            train_datasets.append(
                Dataset.from_list([to_searchr1_row(row) for row in train_rows])
            )
            validation_datasets.append(
                Dataset.from_list([to_searchr1_row(row) for row in validation_rows])
            )
            summary_lines.append(
                f"{dataset_name}: train={len(train_rows)}, "
                f"validation={len(validation_rows)}, eval={len(eval_rows)}, "
                f"eval_split={eval_split_name}, "
                "mean_support_titles="
                f"{sum(len(x['support_titles']) for x in eval_rows) / len(eval_rows):.2f}"
            )

        validate_unique_rows(all_train_rows, "combined hard-RQ0 train")
        validate_unique_rows(all_validation_rows, "combined hard-RQ0 validation")
        validate_unique_rows(all_eval_rows, "combined hard-RQ0 eval")
        validate_disjoint_rows(
            all_train_rows,
            all_validation_rows,
            "combined hard-RQ0 train",
            "combined hard-RQ0 validation",
        )
        validate_disjoint_rows(
            all_train_rows,
            all_eval_rows,
            "combined hard-RQ0 train",
            "combined hard-RQ0 eval",
        )
        validate_disjoint_rows(
            all_validation_rows,
            all_eval_rows,
            "combined hard-RQ0 validation",
            "combined hard-RQ0 eval",
        )
        write_jsonl(staging_data_root / "eval_all.jsonl", all_eval_rows)
        concatenate_datasets(train_datasets).shuffle(seed=request["seed"]).to_parquet(
            str(staging_searchr1_root / "train.parquet")
        )
        # Keep the historical filename because every training wrapper consumes
        # it, but it now contains trainer-validation rows only.
        concatenate_datasets(validation_datasets).to_parquet(
            str(staging_searchr1_root / "test.parquet")
        )
        (staging_data_root / "SUMMARY.txt").write_text(
            "\n".join(summary_lines) + "\n", encoding="utf-8", newline="\n"
        )

        relative_paths = expected_artifacts(request)
        records = artifact_records(staging_root, relative_paths)
        for relative_path in relative_paths:
            source = staging_root / relative_path
            target = work_dir / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, target)

    manifest = {
        "schema": DATA_PREP_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "request": request,
        "artifacts": records,
    }
    atomic_write_json(manifest_path, manifest)
    print("\n".join(summary_lines))
    print(f"Search-R1 parquet: {work_dir / 'searchr1'}")
    print(f"Data manifest: {manifest_path}")


def check_prepared(config_path: str) -> None:
    cfg = load_config(config_path)
    work_dir = Path(cfg["work_dir"]).resolve()
    manifest_path = work_dir / "data" / DATA_MANIFEST_NAME
    request = prepare_request(cfg)
    if not prepared_cache_valid(manifest_path, work_dir, request):
        raise RuntimeError(
            "Hard-RQ0 prepared data is missing, stale, or modified; rerun "
            "bash hard_rq0/prepare_data.sh"
        )
    print(f"Verified hard-RQ0 prepared data: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hard_rq0.yaml")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check_prepared(args.config)
    else:
        prepare(args.config)
