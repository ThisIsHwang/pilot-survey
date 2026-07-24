from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from stackpilot.common import read_jsonl
from stackpilot.prepare_hotpot import (
    PILOT_DATA_MANIFEST_SCHEMA,
    prepare,
    validate_pilot_data_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
HOTPOT_REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"


class FakeSplit:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def shuffle(self, seed: int) -> FakeSplit:
        rows = list(self.rows)
        random.Random(seed).shuffle(rows)
        return FakeSplit(rows)

    def select(self, indexes) -> FakeSplit:
        return FakeSplit([self.rows[index] for index in indexes])


class SizedSplit:
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def shuffle(self, seed: int):
        raise AssertionError(f"short split must fail before shuffle(seed={seed})")


def example(question_id: str) -> dict:
    title = f"Title {question_id}"
    return {
        "id": question_id,
        "question": f"Question {question_id}?",
        "answer": f"Answer {question_id}",
        "context": {"title": [title], "sentences": [[f"Text for {question_id}."]]},
        "supporting_facts": {"title": [title]},
        "type": "bridge",
        "level": "easy",
    }


def write_config(
    path: Path,
    work_dir: Path,
    *,
    train_examples: int = 2,
    trainer_dev_examples: int = 2,
    eval_examples: int = 2,
) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "seed": 42,
                "work_dir": str(work_dir),
                "data": {
                    "dataset_name": "hotpotqa/hotpot_qa",
                    "dataset_config": "distractor",
                    "revision": HOTPOT_REVISION,
                    "train_examples": train_examples,
                    "trainer_dev_examples": trainer_dev_examples,
                    "eval_examples": eval_examples,
                    "split_train": "train",
                    "split_eval": "validation",
                },
            }
        ),
        encoding="utf-8",
    )


def test_hotpot_preparation_isolates_trainer_dev_and_final_eval() -> None:
    train_rows = [example(f"train-{index}") for index in range(5)]
    eval_rows = [example(f"eval-{index}") for index in range(3)]
    dataset = {
        "train": FakeSplit(train_rows),
        "validation": FakeSplit(eval_rows),
    }
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        config_path = root / "config.yaml"
        write_config(config_path, root / "work")
        with patch(
            "stackpilot.prepare_hotpot.load_source_dataset",
            return_value=dataset,
        ) as loader:
            prepare(str(config_path))

        loader.assert_called_once_with(
            "hotpotqa/hotpot_qa", "distractor", HOTPOT_REVISION
        )
        data_dir = root / "work" / "data"
        train = read_jsonl(data_dir / "queries_train.jsonl")
        dev = read_jsonl(data_dir / "queries_dev.jsonl")
        final_eval = read_jsonl(data_dir / "queries_eval.jsonl")
        assert [row["split"] for row in train] == ["train", "train"]
        assert [row["split"] for row in dev] == [
            "trainer_dev",
            "trainer_dev",
        ]
        assert [row["split"] for row in final_eval] == [
            "final_eval",
            "final_eval",
        ]
        selections = [{row["id"] for row in rows} for rows in (train, dev, final_eval)]
        assert selections[0].isdisjoint(selections[1])
        assert selections[0].isdisjoint(selections[2])
        assert selections[1].isdisjoint(selections[2])
        assert len(read_jsonl(data_dir / "corpus.jsonl")) == 6

        manifest = validate_pilot_data_manifest(data_dir)
        assert manifest["schema"] == PILOT_DATA_MANIFEST_SCHEMA
        assert manifest["configuration"]["revision"] == HOTPOT_REVISION
        assert manifest["outputs"]["train"]["role"] == "trainer_train"
        assert manifest["outputs"]["dev"]["role"] == "trainer_validation"
        assert manifest["outputs"]["eval"]["role"] == "final_evaluation"
        assert manifest["outputs"]["train"]["source_split"] == "train"
        assert manifest["outputs"]["dev"]["source_split"] == "train"
        assert manifest["outputs"]["eval"]["source_split"] == "validation"
        assert manifest["selections"]["dev"]["slice_start"] == 2
        assert manifest["selections"]["dev"]["slice_stop"] == 4

        with patch(
            "stackpilot.prepare_hotpot.load_source_dataset",
            side_effect=AssertionError("valid cache should be reused"),
        ):
            prepare(str(config_path))


def test_old_hotpot_manifest_schema_is_rebuilt_not_reused() -> None:
    dataset = {
        "train": FakeSplit([example(f"train-{index}") for index in range(5)]),
        "validation": FakeSplit([example(f"eval-{index}") for index in range(3)]),
    }
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        config_path = root / "config.yaml"
        write_config(config_path, root / "work")
        with patch(
            "stackpilot.prepare_hotpot.load_source_dataset", return_value=dataset
        ):
            prepare(str(config_path))
        manifest_path = root / "work" / "data" / ".pilot-manifest.json"
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        old["schema"] = PILOT_DATA_MANIFEST_SCHEMA - 1
        manifest_path.write_text(json.dumps(old), encoding="utf-8")

        with patch(
            "stackpilot.prepare_hotpot.load_source_dataset", return_value=dataset
        ) as loader:
            prepare(str(config_path))
        loader.assert_called_once()
        assert (
            json.loads(manifest_path.read_text(encoding="utf-8"))["schema"]
            == PILOT_DATA_MANIFEST_SCHEMA
        )


@pytest.mark.parametrize("config_name", ["pilot.yaml", "smoke.yaml"])
def test_checked_in_profiles_fail_when_train_cannot_cover_dev(
    config_name: str,
) -> None:
    config = yaml.safe_load((ROOT / "configs" / config_name).read_text("utf-8"))
    required = (
        int(config["data"]["train_examples"])
        + int(config["data"]["trainer_dev_examples"])
    )
    dataset = {
        config["data"]["split_train"]: SizedSplit(required - 1),
        config["data"]["split_eval"]: SizedSplit(
            int(config["data"]["eval_examples"])
        ),
    }
    with tempfile.TemporaryDirectory() as temporary:
        config["work_dir"] = str(Path(temporary) / "work")
        config_path = Path(temporary) / config_name
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        with (
            patch(
                "stackpilot.prepare_hotpot.load_source_dataset",
                return_value=dataset,
            ),
            pytest.raises(RuntimeError, match="disjoint trainer train/dev"),
        ):
            prepare(str(config_path))


def test_hotpot_preparation_fails_when_official_eval_is_too_small() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        config_path = root / "config.yaml"
        write_config(config_path, root / "work")
        dataset = {
            "train": SizedSplit(4),
            "validation": SizedSplit(1),
        }
        with (
            patch(
                "stackpilot.prepare_hotpot.load_source_dataset",
                return_value=dataset,
            ),
            pytest.raises(RuntimeError, match="final-evaluation rows are required"),
        ):
            prepare(str(config_path))


def test_stage2_training_uses_trainer_dev_not_final_eval() -> None:
    training = (
        ROOT / "searchr1_stage2" / "run_single_stack_grpo.sh"
    ).read_text(encoding="utf-8")
    converter = (
        ROOT / "searchr1_stage2" / "make_hotpot_searchr1_data.py"
    ).read_text(encoding="utf-8")
    policy_eval = (ROOT / "stackpilot" / "policy_eval.py").read_text(
        encoding="utf-8"
    )

    assert "VAL_FILE=$ROOT/work/searchr1_hotpot/dev.parquet" in training
    assert '("dev", "trainer_validation", dev_path)' in training
    assert '"schema": 7' in training
    assert "queries_eval.jsonl" not in converter
    assert '"role": "legacy_trainer_validation_alias"' in converter
    assert '"source_path": source_outputs["dev"]["path"]' in converter
    assert "validate_pilot_data_manifest(" in policy_eval
    assert 'final_eval.get("role") != "final_evaluation"' in policy_eval
    assert '"queries_eval_question_ids_sha256"' in policy_eval
