from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stackpilot.common import ensure_dir, load_config, write_jsonl

DATA_MANIFEST_NAME = ".hard-rq0-data-manifest.json"
DATA_PREP_SCHEMA = 5
PINNED_REVISION_RE = re.compile(r"[0-9a-fA-F]{40}")
TRAIN_ROLE = "trainer_train"
VALIDATION_ROLE = "trainer_validation"
FINAL_EVAL_ROLE = "final_evaluation"

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


def question_id_fingerprint(identifiers: Iterable[Any]) -> dict[str, Any]:
    ordered = [str(identifier).strip() for identifier in identifiers]
    if any(not identifier for identifier in ordered):
        raise RuntimeError("Cannot fingerprint an empty question ID")
    unique = sorted(set(ordered))

    def digest(values: list[str]) -> str:
        payload = json.dumps(
            values,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    return {
        "algorithm": "sha256-json-string-list-v1",
        "count": len(ordered),
        "unique_count": len(unique),
        "ordered_sha256": digest(ordered),
        "set_sha256": digest(unique),
    }


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
    canonical_dev_count = data_cfg.get("trainer_dev_examples_per_dataset")
    legacy_dev_count = data_cfg.get("validation_examples_per_dataset")
    if (
        canonical_dev_count is not None
        and legacy_dev_count is not None
        and int(canonical_dev_count) != int(legacy_dev_count)
    ):
        raise ValueError(
            "data.trainer_dev_examples_per_dataset and legacy "
            "data.validation_examples_per_dataset disagree; remove the legacy "
            "key or give both the same value"
        )
    # Accept the old spelling so existing server configs can be regenerated
    # safely, but write only the unambiguous trainer-dev name to schema 5.
    validation_count = int(
        canonical_dev_count
        if canonical_dev_count is not None
        else (legacy_dev_count if legacy_dev_count is not None else eval_count)
    )
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
        "trainer_dev_examples_per_dataset": validation_count,
        "eval_examples_per_dataset": eval_count,
        "split_train": str(data_cfg["split_train"]),
        "split_eval": str(data_cfg["split_eval"]),
        "seed": int(cfg["seed"]),
        "prompt_sha256": hashlib.sha256(PROMPT.encode("utf-8")).hexdigest(),
        "preparer_sha256": file_sha256(Path(__file__)),
    }


def expected_artifacts(request: dict[str, Any]) -> list[str]:
    paths = [
        "data/final_eval.jsonl",
        "data/eval_all.jsonl",
        "data/SUMMARY.txt",
        "searchr1/train.parquet",
        "searchr1/dev.parquet",
        "searchr1/test.parquet",
    ]
    for dataset_name in request["datasets"]:
        paths.extend(
            (
                f"data/{dataset_name}/train.jsonl",
                f"data/{dataset_name}/dev.jsonl",
                f"data/{dataset_name}/validation.jsonl",
                f"data/{dataset_name}/final_eval.jsonl",
                f"data/{dataset_name}/eval.jsonl",
            )
        )
    return sorted(paths)


def expected_artifact_metadata(request: dict[str, Any]) -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {
        "data/final_eval.jsonl": {"role": FINAL_EVAL_ROLE},
        "data/eval_all.jsonl": {
            "role": FINAL_EVAL_ROLE,
            "alias_of": "data/final_eval.jsonl",
        },
        "data/SUMMARY.txt": {"role": "metadata"},
        "searchr1/train.parquet": {"role": TRAIN_ROLE},
        "searchr1/dev.parquet": {"role": VALIDATION_ROLE},
        "searchr1/test.parquet": {
            "role": VALIDATION_ROLE,
            "alias_of": "searchr1/dev.parquet",
        },
    }
    for dataset_name in request["datasets"]:
        prefix = f"data/{dataset_name}"
        metadata.update(
            {
                f"{prefix}/train.jsonl": {"role": TRAIN_ROLE},
                f"{prefix}/dev.jsonl": {"role": VALIDATION_ROLE},
                f"{prefix}/validation.jsonl": {
                    "role": VALIDATION_ROLE,
                    "alias_of": f"{prefix}/dev.jsonl",
                },
                f"{prefix}/final_eval.jsonl": {"role": FINAL_EVAL_ROLE},
                f"{prefix}/eval.jsonl": {
                    "role": FINAL_EVAL_ROLE,
                    "alias_of": f"{prefix}/final_eval.jsonl",
                },
            }
        )
    return metadata


def artifact_records(
    root: Path,
    relative_paths: Iterable[str],
    *,
    metadata: dict[str, dict[str, Any]] | None = None,
    question_ids: dict[str, Iterable[Any]] | None = None,
) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for relative_path in relative_paths:
        path = root / relative_path
        if not path.is_file():
            raise RuntimeError(f"Prepared artifact is missing: {path}")
        record: dict[str, Any] = {
            "size": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        if metadata is not None:
            record.update(metadata.get(relative_path, {}))
        if question_ids is not None and relative_path in question_ids:
            record["question_ids"] = question_id_fingerprint(
                question_ids[relative_path]
            )
        records[relative_path] = record
    return records


def _valid_question_id_fingerprint(
    value: Any,
    *,
    expected_count: int | None = None,
) -> bool:
    if not isinstance(value, dict):
        return False
    if value.get("algorithm") != "sha256-json-string-list-v1":
        return False
    if expected_count is not None and value.get("count") != expected_count:
        return False
    if value.get("unique_count") != value.get("count"):
        return False
    return all(
        isinstance(value.get(key), str)
        and re.fullmatch(r"[0-9a-f]{64}", value[key]) is not None
        for key in ("ordered_sha256", "set_sha256")
    )


def _split_contract_valid_unchecked(
    contract: Any,
    request: dict[str, Any],
) -> bool:
    if not isinstance(contract, dict) or contract.get("schema") != 1:
        return False
    roles = contract.get("roles")
    if not isinstance(roles, dict) or set(roles) != {
        TRAIN_ROLE,
        VALIDATION_ROLE,
        FINAL_EVAL_ROLE,
    }:
        return False

    per_dataset_counts = {
        TRAIN_ROLE: request["train_examples_per_dataset"],
        VALIDATION_ROLE: request["trainer_dev_examples_per_dataset"],
        FINAL_EVAL_ROLE: request["eval_examples_per_dataset"],
    }
    for role, count in per_dataset_counts.items():
        role_record = roles.get(role)
        if not isinstance(role_record, dict):
            return False
        datasets = role_record.get("datasets")
        if not isinstance(datasets, dict) or set(datasets) != set(request["datasets"]):
            return False
        expected_total = count * len(request["datasets"])
        if role_record.get(
            "count"
        ) != expected_total or not _valid_question_id_fingerprint(
            role_record.get("question_ids"),
            expected_count=expected_total,
        ):
            return False
        for offset, dataset_name in enumerate(request["datasets"]):
            dataset_record = datasets.get(dataset_name)
            if not isinstance(dataset_record, dict):
                return False
            if role == TRAIN_ROLE:
                source_split = request["split_train"]
                shuffle_seed = request["seed"] + offset
                start = 0
            elif role == VALIDATION_ROLE:
                source_split = request["split_train"]
                shuffle_seed = request["seed"] + offset
                start = request["train_examples_per_dataset"]
            else:
                source_split = request["split_eval"]
                shuffle_seed = request["seed"] + 100 + offset
                start = 0
            if (
                dataset_record.get("source_split") != source_split
                or dataset_record.get("shuffle_seed") != shuffle_seed
                or dataset_record.get("selection")
                != {"start": start, "stop": start + count}
                or dataset_record.get("count") != count
                or not _valid_question_id_fingerprint(
                    dataset_record.get("question_ids"),
                    expected_count=count,
                )
            ):
                return False
    return True


def split_contract_valid(contract: Any, request: dict[str, Any]) -> bool:
    try:
        return _split_contract_valid_unchecked(contract, request)
    except (KeyError, TypeError, ValueError):
        return False


def _artifact_question_ids_match_contract(
    relative_path: str,
    record: dict[str, Any],
    contract: dict[str, Any],
    request: dict[str, Any],
) -> bool:
    role = record.get("role")
    if role == "metadata":
        return "question_ids" not in record
    if role not in {TRAIN_ROLE, VALIDATION_ROLE, FINAL_EVAL_ROLE}:
        return False
    role_record = contract["roles"][role]
    dataset_name = next(
        (
            name
            for name in request["datasets"]
            if relative_path.startswith(f"data/{name}/")
        ),
        None,
    )
    expected = (
        role_record["datasets"][dataset_name]["question_ids"]
        if dataset_name is not None
        else role_record["question_ids"]
    )
    return record.get("question_ids") == expected


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
    contract = manifest.get("split_contract")
    if not split_contract_valid(contract, request):
        return False
    records = manifest.get("artifacts")
    expected = expected_artifacts(request)
    expected_metadata = expected_artifact_metadata(request)
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
        if any(
            record.get(key) != value
            for key, value in expected_metadata[relative_path].items()
        ):
            return False
        if not _artifact_question_ids_match_contract(
            relative_path,
            record,
            contract,
            request,
        ):
            return False
        alias_of = record.get("alias_of")
        if alias_of is not None:
            canonical = records.get(alias_of)
            if (
                not isinstance(canonical, dict)
                or record.get("size") != canonical.get("size")
                or record.get("sha256") != canonical.get("sha256")
                or record.get("question_ids") != canonical.get("question_ids")
            ):
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
        artifact_question_ids: dict[str, list[str]] = {}
        contract_datasets: dict[str, dict[str, Any]] = {
            TRAIN_ROLE: {},
            VALIDATION_ROLE: {},
            FINAL_EVAL_ROLE: {},
        }
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
                raise RuntimeError(
                    f"{dataset_name}: missing requested eval split {request['split_eval']!r}; "
                    f"available={sorted(dataset)}"
                )
            eval_split = dataset[eval_split_name]

            train_count = request["train_examples_per_dataset"]
            validation_count = request["trainer_dev_examples_per_dataset"]
            eval_count = request["eval_examples_per_dataset"]
            required_train_rows = train_count + validation_count
            if len(train_split) < required_train_rows:
                raise RuntimeError(
                    f"{dataset_name}: trainer train and trainer dev must use "
                    "disjoint source-train rows, requiring "
                    f"{train_count}+{validation_count}={required_train_rows} rows "
                    f"from {train_split_name!r}; the pinned split has only "
                    f"{len(train_split)}"
                )
            if len(eval_split) < eval_count:
                raise RuntimeError(
                    f"{dataset_name}: requested {eval_count} final-evaluation "
                    f"rows from {eval_split_name!r}, but the pinned split has only "
                    f"{len(eval_split)}"
                )
            train_shuffle_seed = request["seed"] + offset
            shuffled_train = train_split.shuffle(seed=train_shuffle_seed)
            train_selected = shuffled_train.select(range(train_count))
            validation_selected = shuffled_train.select(
                range(train_count, train_count + validation_count)
            )
            eval_shuffle_seed = request["seed"] + 100 + offset
            shuffled_eval = eval_split.shuffle(seed=eval_shuffle_seed)
            # Preserve the original Hard-RQ0 final benchmark exactly. Earlier
            # schemas used the first eval_count rows of this pinned shuffle.
            eval_selected = shuffled_eval.select(range(eval_count))

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
            write_jsonl(dataset_dir / "dev.jsonl", validation_rows)
            shutil.copyfile(
                dataset_dir / "dev.jsonl",
                dataset_dir / "validation.jsonl",
            )
            write_jsonl(dataset_dir / "final_eval.jsonl", eval_rows)
            shutil.copyfile(
                dataset_dir / "final_eval.jsonl",
                dataset_dir / "eval.jsonl",
            )
            all_train_rows.extend(train_rows)
            all_validation_rows.extend(validation_rows)
            all_eval_rows.extend(eval_rows)

            train_ids = [row["id"] for row in train_rows]
            validation_ids = [row["id"] for row in validation_rows]
            eval_ids = [row["id"] for row in eval_rows]
            prefix = f"data/{dataset_name}"
            artifact_question_ids.update(
                {
                    f"{prefix}/train.jsonl": train_ids,
                    f"{prefix}/dev.jsonl": validation_ids,
                    f"{prefix}/validation.jsonl": validation_ids,
                    f"{prefix}/final_eval.jsonl": eval_ids,
                    f"{prefix}/eval.jsonl": eval_ids,
                }
            )
            contract_datasets[TRAIN_ROLE][dataset_name] = {
                "source_split": train_split_name,
                "shuffle_seed": train_shuffle_seed,
                "selection": {"start": 0, "stop": train_count},
                "count": len(train_ids),
                "question_ids": question_id_fingerprint(train_ids),
            }
            contract_datasets[VALIDATION_ROLE][dataset_name] = {
                "source_split": train_split_name,
                "shuffle_seed": train_shuffle_seed,
                "selection": {
                    "start": train_count,
                    "stop": train_count + validation_count,
                },
                "count": len(validation_ids),
                "question_ids": question_id_fingerprint(validation_ids),
            }
            contract_datasets[FINAL_EVAL_ROLE][dataset_name] = {
                "source_split": eval_split_name,
                "shuffle_seed": eval_shuffle_seed,
                "selection": {"start": 0, "stop": eval_count},
                "count": len(eval_ids),
                "question_ids": question_id_fingerprint(eval_ids),
            }

            train_datasets.append(
                Dataset.from_list([to_searchr1_row(row) for row in train_rows])
            )
            validation_datasets.append(
                Dataset.from_list([to_searchr1_row(row) for row in validation_rows])
            )
            summary_lines.append(
                f"{dataset_name}: train={len(train_rows)}, "
                f"trainer_dev={len(validation_rows)}, "
                f"final_eval={len(eval_rows)}, "
                f"final_eval_split={eval_split_name}, "
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
        write_jsonl(staging_data_root / "final_eval.jsonl", all_eval_rows)
        shutil.copyfile(
            staging_data_root / "final_eval.jsonl",
            staging_data_root / "eval_all.jsonl",
        )
        combined_train_dataset = concatenate_datasets(train_datasets).shuffle(
            seed=request["seed"]
        )
        combined_validation_dataset = concatenate_datasets(validation_datasets)
        combined_train_dataset.to_parquet(str(staging_searchr1_root / "train.parquet"))
        combined_validation_dataset.to_parquet(
            str(staging_searchr1_root / "dev.parquet")
        )
        # Compatibility aliases remain byte-identical, while manifest roles
        # prevent their historical names from obscuring the split contract.
        shutil.copyfile(
            staging_searchr1_root / "dev.parquet",
            staging_searchr1_root / "test.parquet",
        )
        (staging_data_root / "SUMMARY.txt").write_text(
            "\n".join(summary_lines) + "\n", encoding="utf-8", newline="\n"
        )

        all_train_ids = [row["id"] for row in all_train_rows]
        all_validation_ids = [row["id"] for row in all_validation_rows]
        all_eval_ids = [row["id"] for row in all_eval_rows]
        train_parquet_ids = [
            str(row["extra_info"]["question_id"]) for row in combined_train_dataset
        ]
        validation_parquet_ids = [
            str(row["extra_info"]["question_id"]) for row in combined_validation_dataset
        ]
        artifact_question_ids.update(
            {
                "data/final_eval.jsonl": all_eval_ids,
                "data/eval_all.jsonl": all_eval_ids,
                "searchr1/train.parquet": train_parquet_ids,
                "searchr1/dev.parquet": validation_parquet_ids,
                "searchr1/test.parquet": validation_parquet_ids,
            }
        )
        split_contract = {
            "schema": 1,
            "roles": {
                TRAIN_ROLE: {
                    "count": len(all_train_ids),
                    "question_ids": question_id_fingerprint(train_parquet_ids),
                    "datasets": contract_datasets[TRAIN_ROLE],
                },
                VALIDATION_ROLE: {
                    "count": len(all_validation_ids),
                    "question_ids": question_id_fingerprint(all_validation_ids),
                    "datasets": contract_datasets[VALIDATION_ROLE],
                },
                FINAL_EVAL_ROLE: {
                    "count": len(all_eval_ids),
                    "question_ids": question_id_fingerprint(all_eval_ids),
                    "datasets": contract_datasets[FINAL_EVAL_ROLE],
                },
            },
        }
        relative_paths = expected_artifacts(request)
        records = artifact_records(
            staging_root,
            relative_paths,
            metadata=expected_artifact_metadata(request),
            question_ids=artifact_question_ids,
        )
        for relative_path in relative_paths:
            source = staging_root / relative_path
            target = work_dir / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, target)

    manifest = {
        "schema": DATA_PREP_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "request": request,
        "split_contract": split_contract,
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


def _read_verified_manifest(manifest_path: Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = manifest_path.resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Hard-RQ0 data manifest is missing or invalid: {manifest_path}"
        ) from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != DATA_PREP_SCHEMA:
        raise RuntimeError(
            f"Hard-RQ0 data manifest must use schema {DATA_PREP_SCHEMA}: "
            f"{manifest_path}"
        )
    request = manifest.get("request")
    if not isinstance(request, dict):
        raise RuntimeError(  # noqa: TRY004
            f"Hard-RQ0 data manifest has no valid preparation request: {manifest_path}"
        )
    # The manifest lives at WORK_ROOT/data/.hard-rq0-data-manifest.json.
    work_dir = manifest_path.parent.parent.resolve()
    if not prepared_cache_valid(manifest_path, work_dir, request):
        raise RuntimeError(
            "Hard-RQ0 data manifest, split contract, or registered artifacts "
            f"are stale or modified: {manifest_path}"
        )
    return work_dir, manifest


def validate_manifest_artifact(
    manifest_path: Path,
    artifact_path: Path,
    expected_role: str,
) -> dict[str, Any]:
    """Verify one exact manifest-registered artifact and its protocol role."""

    if expected_role not in {TRAIN_ROLE, VALIDATION_ROLE, FINAL_EVAL_ROLE}:
        raise ValueError(f"Unsupported hard-RQ0 artifact role: {expected_role!r}")
    work_dir, manifest = _read_verified_manifest(Path(manifest_path))
    resolved = Path(artifact_path).resolve()
    try:
        relative_path = resolved.relative_to(work_dir).as_posix()
    except ValueError as exc:
        raise RuntimeError(
            f"Artifact is outside the manifest work root {work_dir}: {resolved}"
        ) from exc
    record = manifest["artifacts"].get(relative_path)
    if not isinstance(record, dict):
        raise RuntimeError(  # noqa: TRY004
            "Artifact is not registered in the hard-RQ0 data manifest "
            f"(repacked or foreign files are not trusted): {resolved}"
        )
    if record.get("role") != expected_role:
        raise RuntimeError(
            f"Artifact role is {record.get('role')!r}, expected {expected_role!r}: "
            f"{resolved}"
        )
    if (
        not resolved.is_file()
        or record.get("size") != resolved.stat().st_size
        or record.get("sha256") != file_sha256(resolved)
    ):
        raise RuntimeError(f"Manifest artifact bytes do not match: {resolved}")
    return {"relative_path": relative_path, **record}


def _parquet_training_metadata(path: Path) -> list[dict[str, str]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required to validate Search-R1 training inputs"
        ) from exc
    try:
        table = pq.read_table(path, columns=["extra_info"])
        values = table.column("extra_info").to_pylist()
    except Exception as exc:
        raise RuntimeError(
            f"Cannot read Search-R1 extra_info metadata from parquet: {path}"
        ) from exc
    metadata: list[dict[str, str]] = []
    for row_number, value in enumerate(values):
        if not isinstance(value, dict):
            raise RuntimeError(  # noqa: TRY004
                f"{path}: row {row_number} has no extra_info object"
            )
        question_id = str(value.get("question_id") or "").strip()
        index = str(value.get("index") or "").strip()
        if not question_id or not index:
            raise RuntimeError(
                f"{path}: row {row_number} has an empty question_id or GRPO index"
            )
        metadata.append(
            {
                "question_id": question_id,
                "index": index,
                "source_index": str(value.get("source_index") or "").strip(),
                "routing_backend": str(value.get("routing_backend") or "")
                .strip()
                .lower(),
            }
        )
    return metadata


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Missing or invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(  # noqa: TRY004
            f"{label} must contain a JSON object: {path}"
        )
    return payload


def _validate_training_parquet(
    path: Path,
    *,
    expected_role: str,
    manifest_path: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    from stackpilot.prepare_mixed_data import MODES
    from stackpilot.prepare_mixed_data import (
        manifest_path as mixed_manifest_path,
    )
    from stackpilot.prepare_mixed_data import (
        preparation_request as mixed_preparation_request,
    )
    from stackpilot.prepare_mixed_data import (
        prepared_cache_valid as mixed_cache_valid,
    )

    path = path.resolve()
    direct_record: dict[str, Any] | None = None
    derived_source_record: dict[str, Any] | None = None
    derived_mode: str | None = None
    try:
        direct_record = validate_manifest_artifact(
            manifest_path,
            path,
            expected_role,
        )
    except RuntimeError as direct_error:
        sidecar_path = mixed_manifest_path(path)
        if not sidecar_path.is_file():
            raise RuntimeError(
                f"{path} is neither a registered {expected_role} artifact nor "
                "a verified prepare_mixed_data derivative"
            ) from direct_error
        sidecar = _read_json_object(sidecar_path, "mixed-data sidecar")
        request = sidecar.get("request")
        if not isinstance(request, dict):
            raise RuntimeError(  # noqa: TRY004
                f"Mixed-data sidecar has no valid request: {sidecar_path}"
            )
        source_path = Path(str(request.get("input_path") or "")).resolve()
        # Only a one-step derivative of an exact manifest artifact is trusted.
        # This makes a derivative of final evaluation fail at the role check.
        derived_source_record = validate_manifest_artifact(
            manifest_path,
            source_path,
            expected_role,
        )
        try:
            derived_mode = str(request["mode"])
            seed = int(request["seed"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Mixed-data sidecar has invalid mode/seed: {sidecar_path}"
            ) from exc
        if derived_mode not in MODES:
            raise RuntimeError(
                f"Mixed-data sidecar has unsupported mode {derived_mode!r}: "
                f"{sidecar_path}"
            )
        expected_request = mixed_preparation_request(
            source_path,
            seed,
            derived_mode,
        )
        if request != expected_request or not mixed_cache_valid(path, expected_request):
            raise RuntimeError(
                f"mixed-policy data derivative or provenance is stale/modified: {path}"
            )

    rows = _parquet_training_metadata(path)
    identifiers = [row["question_id"] for row in rows]
    actual = question_id_fingerprint(identifiers)
    expected = manifest["split_contract"]["roles"][expected_role]["question_ids"]
    final_ids = manifest["split_contract"]["roles"][FINAL_EVAL_ROLE]["question_ids"]
    if actual["set_sha256"] != expected["set_sha256"]:
        if actual["set_sha256"] == final_ids["set_sha256"]:
            raise RuntimeError(
                f"Final-evaluation question IDs cannot be used as {expected_role}: "
                f"{path}"
            )
        raise RuntimeError(
            f"{path} does not contain the exact {expected_role} question-ID set"
        )

    counts = Counter(identifiers)
    if direct_record is not None:
        if actual != direct_record.get("question_ids"):
            raise RuntimeError(
                f"Registered {expected_role} parquet question IDs do not match "
                f"its manifest record: {path}"
            )
        if any(count != 1 for count in counts.values()):
            raise RuntimeError(f"{path} repeats a direct training question ID")
        if any(
            row["index"] != row["question_id"] or row["source_index"] for row in rows
        ):
            raise RuntimeError(
                f"{path} has unexpected derived GRPO indices without a sidecar"
            )
        return direct_record

    if derived_source_record is None or derived_mode is None:
        raise AssertionError("mixed-data provenance state is incomplete")
    if len(rows) != 2 * expected["count"] or set(counts.values()) != {2}:
        raise RuntimeError(
            f"Mixed-data {expected_role} must contain exactly two backend rows "
            f"per source question: {path}"
        )
    routes: dict[str, set[str]] = {}
    for row in rows:
        question_id = row["question_id"]
        backend = row["routing_backend"]
        if (
            row["source_index"] != question_id
            or backend not in {"bm25", "e5"}
            or row["index"] != f"{question_id}::retrieval_backend={backend}"
        ):
            raise RuntimeError(
                f"Mixed-data row provenance is inconsistent for {question_id}: {path}"
            )
        routes.setdefault(question_id, set()).add(backend)
    if any(backends != {"bm25", "e5"} for backends in routes.values()):
        raise RuntimeError(
            f"Mixed-data {expected_role} is missing a BM25/E5 pair: {path}"
        )
    return {
        "relative_path": str(path),
        "role": expected_role,
        "derived_from": derived_source_record["relative_path"],
        "mode": derived_mode,
        "size": path.stat().st_size,
        "sha256": file_sha256(path),
        "question_ids": actual,
    }


def validate_training_inputs(
    config_path: str,
    train_file: Path,
    val_file: Path,
) -> dict[str, dict[str, Any]]:
    cfg = load_config(config_path)
    work_dir = Path(cfg["work_dir"]).resolve()
    manifest_path = work_dir / "data" / DATA_MANIFEST_NAME
    request = prepare_request(cfg)
    verified_work_dir, manifest = _read_verified_manifest(manifest_path)
    if verified_work_dir != work_dir or manifest.get("request") != request:
        raise RuntimeError(
            "Hard-RQ0 training inputs do not match the configured data request"
        )
    result = {
        "train": _validate_training_parquet(
            Path(train_file),
            expected_role=TRAIN_ROLE,
            manifest_path=manifest_path,
            manifest=manifest,
        ),
        "validation": _validate_training_parquet(
            Path(val_file),
            expected_role=VALIDATION_ROLE,
            manifest_path=manifest_path,
            manifest=manifest,
        ),
    }
    print(
        "Verified hard-RQ0 training inputs: "
        f"train={Path(train_file).resolve()}, "
        f"validation={Path(val_file).resolve()}"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hard_rq0.yaml")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--validate-training-inputs", action="store_true")
    parser.add_argument("--train-file")
    parser.add_argument("--val-file")
    args = parser.parse_args()
    if args.check and args.validate_training_inputs:
        parser.error("--check and --validate-training-inputs are mutually exclusive")
    if args.validate_training_inputs:
        if not args.train_file or not args.val_file:
            parser.error(
                "--validate-training-inputs requires --train-file and --val-file"
            )
        validate_training_inputs(
            args.config,
            Path(args.train_file),
            Path(args.val_file),
        )
    elif args.check:
        check_prepared(args.config)
    else:
        if args.train_file or args.val_file:
            parser.error("--train-file/--val-file require --validate-training-inputs")
        prepare(args.config)
