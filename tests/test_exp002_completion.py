from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from stackpilot.exp002_completion import (
    RUN_COMPLETION_SCHEMA,
    current_input_provenance,
    validate_run_completion,
)
from stackpilot.prepare_hard_rq0 import DATA_PREP_SCHEMA


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class Exp002CompletionTests(unittest.TestCase):
    def make_tree(self, root: Path) -> Path:
        hard_root = root / "hard_rq0"
        write_json(
            hard_root / "data" / ".hard-rq0-data-manifest.json",
            {"schema": DATA_PREP_SCHEMA, "request": {"seed": 42}},
        )
        for backend in ("bm25", "e5"):
            write_json(
                hard_root
                / "checkpoints"
                / f"hard-rq0-{backend}-seed13-pilot"
                / ".complete.json",
                {
                    "schema": 2,
                    "backend": backend,
                    "seed": 13,
                    "profile": "pilot",
                    "training_signature": f"{backend}-signature",
                },
            )
        provenance = current_input_provenance(hard_root, "pilot", [13])
        marker = hard_root / "runs" / "pilot" / ".complete.json"
        write_json(
            marker,
            {
                "schema": RUN_COMPLETION_SCHEMA,
                "profile": "pilot",
                "result_set": "pilot",
                "seeds": [13],
                "input_provenance": provenance,
            },
        )
        return hard_root

    def test_current_data_and_specialist_markers_are_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            hard_root = self.make_tree(Path(temporary))
            payload = validate_run_completion(
                hard_root,
                "pilot",
                "pilot",
                [13],
            )
            self.assertEqual(payload["schema"], RUN_COMPLETION_SCHEMA)

    def test_changed_data_manifest_rejects_stale_run_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            hard_root = self.make_tree(Path(temporary))
            write_json(
                hard_root / "data" / ".hard-rq0-data-manifest.json",
                {
                    "schema": DATA_PREP_SCHEMA,
                    "request": {"seed": 42, "validation_rows": 1008},
                },
            )
            with self.assertRaisesRegex(RuntimeError, "stale.*data"):
                validate_run_completion(hard_root, "pilot", "pilot", [13])

    def test_changed_specialist_marker_rejects_stale_run_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            hard_root = self.make_tree(Path(temporary))
            marker = (
                hard_root
                / "checkpoints"
                / "hard-rq0-e5-seed13-pilot"
                / ".complete.json"
            )
            payload = json.loads(marker.read_text(encoding="utf-8"))
            payload["training_signature"] = "new-e5-signature"
            write_json(marker, payload)
            with self.assertRaisesRegex(RuntimeError, "stale.*e5-seed13"):
                validate_run_completion(hard_root, "pilot", "pilot", [13])

    def test_legacy_run_marker_is_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            hard_root = self.make_tree(Path(temporary))
            marker = hard_root / "runs" / "pilot" / ".complete.json"
            payload = json.loads(marker.read_text(encoding="utf-8"))
            payload["schema"] = RUN_COMPLETION_SCHEMA - 1
            write_json(marker, payload)
            with self.assertRaisesRegex(RuntimeError, "required schema"):
                validate_run_completion(hard_root, "pilot", "pilot", [13])

    def test_pipeline_writer_and_consumers_share_the_provenance_validator(
        self,
    ) -> None:
        root = Path(__file__).resolve().parents[1]
        writer = (root / "hard_rq0" / "run_all.sh").read_text("utf-8")
        watcher = (root / "experiments" / "watch_exp002.sh").read_text("utf-8")
        exp006 = (root / "experiments" / "EXP-006" / "run.sh").read_text("utf-8")

        self.assertIn('"input_provenance": current_input_provenance(', writer)
        self.assertIn('"schema": RUN_COMPLETION_SCHEMA', writer)
        self.assertIn("validate_run_completion", watcher)
        self.assertIn("validate_run_completion", exp006)


if __name__ == "__main__":
    unittest.main()
