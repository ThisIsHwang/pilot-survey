from __future__ import annotations

import copy
import json
import random
import sys
import tempfile
import types
import unittest
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

from stackpilot.prepare_mixed_data import (
    MARKER_TEMPLATE,
    duplicate_row,
    manifest_path,
    paired_rows,
    prepare,
)


def source_row(index: str = "musique:q1") -> dict:
    return {
        "data_source": "musique",
        "prompt": [{"role": "user", "content": "Question: where?"}],
        "extra_info": {
            "index": index,
            "question_id": index,
            "support_titles": ["Answer"],
        },
    }


class FakeDataset:
    fail_writes = False

    def __init__(self, rows: list[dict]):
        self.rows = copy.deepcopy(rows)

    @classmethod
    def from_list(cls, rows: list[dict]) -> FakeDataset:
        return cls(rows)

    def shuffle(self, seed: int) -> FakeDataset:
        rows = copy.deepcopy(self.rows)
        random.Random(seed).shuffle(rows)
        return type(self)(rows)

    def to_parquet(self, path: str) -> None:
        target = Path(path)
        if self.fail_writes:
            target.write_text("partial", encoding="utf-8")
            raise RuntimeError("simulated parquet failure")
        target.write_text(
            json.dumps(self.rows, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )

    def __len__(self) -> int:
        return len(self.rows)


class PrepareMixedDataTests(unittest.TestCase):
    def test_backend_id_pairs_have_condition_specific_grpo_uids(self) -> None:
        original = source_row()
        before = copy.deepcopy(original)

        rows = paired_rows([original], expose_backend=True)

        self.assertEqual(original, before)
        self.assertEqual(len(rows), 2)
        by_backend = {
            row["extra_info"]["routing_backend"]: row for row in rows
        }
        self.assertEqual(set(by_backend), {"bm25", "e5"})
        for backend, row in by_backend.items():
            info = row["extra_info"]
            self.assertEqual(info["question_id"], "musique:q1")
            self.assertEqual(info["source_index"], "musique:q1")
            self.assertEqual(
                info["index"],
                f"musique:q1::retrieval_backend={backend}",
            )
            content = row["prompt"][0]["content"]
            self.assertTrue(content.startswith(MARKER_TEMPLATE.format(backend=backend)))
            self.assertEqual(content.count("<retrieval_environment>"), 1)
        self.assertNotEqual(
            by_backend["bm25"]["extra_info"]["index"],
            by_backend["e5"]["extra_info"]["index"],
        )

        uid_backends: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            uid_backends[row["extra_info"]["index"]].add(
                row["extra_info"]["routing_backend"]
            )
        self.assertTrue(
            all(len(backends) == 1 for backends in uid_backends.values())
        )

    def test_hidden_pairs_keep_prompts_identical_and_unmarked(self) -> None:
        original = source_row()

        rows = paired_rows([original], expose_backend=False)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["prompt"], original["prompt"])
        self.assertEqual(rows[1]["prompt"], original["prompt"])
        self.assertEqual(rows[0]["prompt"], rows[1]["prompt"])
        for row in rows:
            info = row["extra_info"]
            backend = info["routing_backend"]
            self.assertNotIn(
                "<retrieval_environment>",
                row["prompt"][0]["content"],
            )
            self.assertEqual(info["source_index"], "musique:q1")
            self.assertEqual(
                info["index"],
                f"musique:q1::retrieval_backend={backend}",
            )

    def test_duplicate_row_is_canonical_across_repeated_conversion(self) -> None:
        first = duplicate_row(source_row(), "e5")
        repeated = duplicate_row(first, "e5")
        switched = duplicate_row(repeated, "bm25")
        hidden = duplicate_row(switched, "e5", expose_backend=False)

        self.assertEqual(repeated, first)
        self.assertEqual(
            switched["extra_info"]["source_index"],
            "musique:q1",
        )
        self.assertEqual(
            switched["extra_info"]["index"],
            "musique:q1::retrieval_backend=bm25",
        )
        self.assertEqual(
            switched["prompt"][0]["content"].count(
                "<retrieval_environment>"
            ),
            1,
        )
        self.assertEqual(hidden["prompt"], source_row()["prompt"])
        self.assertEqual(
            hidden["extra_info"]["index"],
            "musique:q1::retrieval_backend=e5",
        )

    def test_duplicate_row_rejects_missing_or_invalid_identity(self) -> None:
        missing = source_row()
        missing["extra_info"].pop("index")
        with self.assertRaisesRegex(ValueError, "extra_info.index"):
            paired_rows([missing], expose_backend=True)
        with self.assertRaisesRegex(ValueError, "backend must be"):
            duplicate_row(source_row(), "colbert")

    def test_prepare_is_deterministic_and_atomically_rerunnable(self) -> None:
        controlled = [source_row("q1"), source_row("q2")]
        datasets = types.ModuleType("datasets")
        datasets.Dataset = FakeDataset
        datasets.load_dataset = (
            lambda *args, **kwargs: copy.deepcopy(controlled)
        )
        FakeDataset.fail_writes = False

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.parquet"
            source.write_bytes(b"pinned-source")
            output = Path(temporary) / "nested" / "mixed.parquet"
            with patch.dict(sys.modules, {"datasets": datasets}):
                self.assertFalse(prepare(source, output, 42, "backend-id"))
                first = output.read_bytes()
                self.assertTrue(prepare(source, output, 42, "backend-id"))
                self.assertEqual(output.read_bytes(), first)
                self.assertTrue(manifest_path(output).is_file())

                FakeDataset.fail_writes = True
                with self.assertRaisesRegex(RuntimeError, "simulated parquet"):
                    prepare(source, output, 42, "backend-id", force=True)
                self.assertEqual(output.read_bytes(), first)

            leftovers = list(output.parent.glob(".mixed.parquet.tmp.*"))
            self.assertEqual(leftovers, [])
        FakeDataset.fail_writes = False

    def test_prepare_rebuilds_a_corrupt_cached_output(self) -> None:
        controlled = [source_row("q1")]
        datasets = types.ModuleType("datasets")
        datasets.Dataset = FakeDataset
        datasets.load_dataset = (
            lambda *args, **kwargs: copy.deepcopy(controlled)
        )
        FakeDataset.fail_writes = False

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.parquet"
            source.write_bytes(b"pinned-source")
            output = Path(temporary) / "mixed.parquet"
            with patch.dict(sys.modules, {"datasets": datasets}):
                self.assertFalse(prepare(source, output, 13, "hidden-paired"))
                expected = output.read_bytes()
                output.write_bytes(b"corrupt")
                self.assertFalse(prepare(source, output, 13, "hidden-paired"))
            self.assertEqual(output.read_bytes(), expected)


if __name__ == "__main__":
    unittest.main()
