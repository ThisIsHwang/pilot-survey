from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from stackpilot.common import read_jsonl
from stackpilot.prepare_hotpot import (
    question_ids_sha256,
    validate_pilot_data_manifest,
)

PROMPT = """Answer the given question. You must conduct reasoning inside <think> and </think> every time you get new information. If you lack knowledge, call search using <search> query </search>. The search result appears inside <information> and </information>. When ready, output only <answer> short answer </answer>. Question: {question}\n"""
STAGE2_DATA_MANIFEST_SCHEMA = 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(input_path: Path, output_path: Path, split: str) -> list[str]:
    from datasets import Dataset

    rows = []
    question_ids: list[str] = []
    for idx, item in enumerate(read_jsonl(input_path)):
        question_id = str(item["id"])
        question_ids.append(question_id)
        rows.append(
            {
                "data_source": "hotpotqa",
                "prompt": [
                    {
                        "role": "user",
                        "content": PROMPT.format(question=item["question"]),
                    }
                ],
                "ability": "fact-reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {"target": [item["answer"]]},
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "question_id": question_id,
                },
            }
        )
    if not rows:
        raise RuntimeError(f"Cannot convert an empty query artifact: {input_path}")
    if len(question_ids) != len(set(question_ids)):
        raise RuntimeError(f"Duplicate question IDs in {input_path}")
    Dataset.from_list(rows).to_parquet(str(output_path))
    return question_ids


def _parquet_state(path: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    return {
        "rows": pq.ParquetFile(path).metadata.num_rows,
        "sha256": sha256_file(path),
    }


def _validated_existing_manifest(
    output: Path,
    *,
    source_manifest_sha256: str,
    converter_sha256: str,
) -> dict[str, Any] | None:
    manifest_path = output / ".pilot-manifest.json"
    incomplete_path = output / ".pilot-prepare-incomplete"
    if not manifest_path.is_file() or incomplete_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, OSError):
        return None
    if (
        manifest.get("schema") != STAGE2_DATA_MANIFEST_SCHEMA
        or manifest.get("source_data_manifest_sha256") != source_manifest_sha256
        or manifest.get("converter_sha256") != converter_sha256
    ):
        return None
    expected = {
        "train": ("train.parquet", "trainer_train"),
        "dev": ("dev.parquet", "trainer_validation"),
        "test": ("test.parquet", "legacy_trainer_validation_alias"),
    }
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != set(expected):
        return None
    for name, (filename, role) in expected.items():
        record = outputs.get(name)
        path = output / filename
        if (
            not isinstance(record, dict)
            or record.get("path") != filename
            or record.get("role") != role
            or not path.is_file()
        ):
            return None
        try:
            state = _parquet_state(path)
        except (OSError, ValueError):
            return None
        if state["rows"] != record.get("rows") or state["sha256"] != record.get(
            "sha256"
        ):
            return None
    dev = outputs["dev"]
    test = outputs["test"]
    if (
        test.get("alias_of") != "dev"
        or test.get("sha256") != dev.get("sha256")
        or test.get("question_ids_sha256") != dev.get("question_ids_sha256")
    ):
        return None
    return manifest


def prepare(work_dir: str | Path, *, force: bool = False) -> dict[str, Any]:
    root = Path(work_dir).resolve()
    data_dir = root / "data"
    data_manifest = validate_pilot_data_manifest(data_dir)
    source_manifest_path = data_dir / ".pilot-manifest.json"
    source_manifest_sha256 = sha256_file(source_manifest_path)
    converter_sha256 = sha256_file(Path(__file__))

    output = root / "searchr1_hotpot"
    output.mkdir(parents=True, exist_ok=True)
    if not force:
        existing = _validated_existing_manifest(
            output,
            source_manifest_sha256=source_manifest_sha256,
            converter_sha256=converter_sha256,
        )
        if existing is not None:
            print(f"Reusing split-safe Search-R1 data: {output}")
            return existing

    incomplete_path = output / ".pilot-prepare-incomplete"
    manifest_path = output / ".pilot-manifest.json"
    incomplete_path.write_text(f"pid={os.getpid()}\n", encoding="utf-8")
    manifest_path.unlink(missing_ok=True)

    source_outputs = data_manifest["outputs"]
    with tempfile.TemporaryDirectory(prefix=".searchr1-hotpot-", dir=root) as name:
        temporary = Path(name)
        train_path = temporary / "train.parquet"
        dev_path = temporary / "dev.parquet"
        test_path = temporary / "test.parquet"
        train_ids = convert(
            data_dir / source_outputs["train"]["path"], train_path, "train"
        )
        dev_ids = convert(
            data_dir / source_outputs["dev"]["path"], dev_path, "trainer_dev"
        )
        if set(train_ids) & set(dev_ids):
            raise RuntimeError(
                "Search-R1 trainer train/dev question IDs overlap; refusing to "
                "create contaminated training data"
            )
        if question_ids_sha256(train_ids) != source_outputs["train"].get(
            "question_ids_sha256"
        ):
            raise RuntimeError("Search-R1 train conversion changed question selection")
        if question_ids_sha256(dev_ids) != source_outputs["dev"].get(
            "question_ids_sha256"
        ):
            raise RuntimeError(
                "Search-R1 trainer-dev conversion changed question selection"
            )

        # Preserve the historical filename for external tooling, but never use it
        # as final evaluation: it is a byte-identical trainer-dev alias.
        shutil.copyfile(dev_path, test_path)
        for filename in ("train.parquet", "dev.parquet", "test.parquet"):
            os.replace(temporary / filename, output / filename)

    train_state = _parquet_state(output / "train.parquet")
    dev_state = _parquet_state(output / "dev.parquet")
    test_state = _parquet_state(output / "test.parquet")
    manifest = {
        "schema": STAGE2_DATA_MANIFEST_SCHEMA,
        "source_data_manifest_sha256": source_manifest_sha256,
        "converter_sha256": converter_sha256,
        "outputs": {
            "train": {
                "path": "train.parquet",
                "role": "trainer_train",
                "source_path": source_outputs["train"]["path"],
                "source_split": source_outputs["train"]["source_split"],
                "source_sha256": source_outputs["train"]["sha256"],
                "question_ids_sha256": question_ids_sha256(train_ids),
                **train_state,
            },
            "dev": {
                "path": "dev.parquet",
                "role": "trainer_validation",
                "source_path": source_outputs["dev"]["path"],
                "source_split": source_outputs["dev"]["source_split"],
                "source_sha256": source_outputs["dev"]["sha256"],
                "question_ids_sha256": question_ids_sha256(dev_ids),
                **dev_state,
            },
            "test": {
                "path": "test.parquet",
                "role": "legacy_trainer_validation_alias",
                "alias_of": "dev",
                "source_path": source_outputs["dev"]["path"],
                "source_split": source_outputs["dev"]["source_split"],
                "source_sha256": source_outputs["dev"]["sha256"],
                "question_ids_sha256": question_ids_sha256(dev_ids),
                **test_state,
            },
        },
    }
    if test_state != dev_state:
        raise RuntimeError("Legacy test.parquet alias is not byte-identical to dev.parquet")

    temporary_manifest = manifest_path.with_name(
        f".{manifest_path.name}.{os.getpid()}.tmp"
    )
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)
    incomplete_path.unlink(missing_ok=True)
    validated = _validated_existing_manifest(
        output,
        source_manifest_sha256=source_manifest_sha256,
        converter_sha256=converter_sha256,
    )
    if validated != manifest:
        raise RuntimeError("Failed to validate prepared Search-R1 data manifest")
    print(f"Wrote {train_state['rows']:,} Search-R1 train rows")
    print(f"Wrote {dev_state['rows']:,} Search-R1 trainer-dev rows")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", default="work")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    prepare(args.work_dir, force=args.force)
