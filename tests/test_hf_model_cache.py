from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from stackpilot.hf_model_cache import resolve_snapshot, validate_snapshot


def write_snapshot(root: Path, *, complete: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text("{}\n", encoding="utf-8")
    (root / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    if complete:
        (root / "model.safetensors").write_bytes(b"weights")
    return root


class HuggingFaceModelCacheTests(unittest.TestCase):
    def test_local_model_is_validated_without_a_download_callback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = write_snapshot(Path(temporary) / "model")
            self.assertEqual(
                resolve_snapshot(str(snapshot), "main", "local"), snapshot.resolve()
            )

    def test_immutable_revision_uses_complete_local_cache_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = write_snapshot(Path(temporary) / "cached")
            calls: list[dict[str, object]] = []

            def download(**kwargs: object) -> str:
                calls.append(kwargs)
                return str(snapshot)

            resolved = resolve_snapshot("owner/model", "a" * 40, "hub", download)
            self.assertEqual(resolved, snapshot.resolve())
            self.assertEqual(
                calls,
                [
                    {
                        "repo_id": "owner/model",
                        "revision": "a" * 40,
                        "local_files_only": True,
                    }
                ],
            )

    def test_incomplete_cached_snapshot_downloads_only_the_missing_delta(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            incomplete = write_snapshot(root / "incomplete", complete=False)
            complete = write_snapshot(root / "complete")
            calls: list[dict[str, object]] = []

            def download(**kwargs: object) -> str:
                calls.append(kwargs)
                if kwargs.get("local_files_only"):
                    return str(incomplete)
                return str(complete)

            resolved = resolve_snapshot("owner/model", "b" * 40, "hub", download)
            self.assertEqual(resolved, complete.resolve())
            self.assertEqual(len(calls), 2)
            self.assertTrue(calls[0]["local_files_only"])
            self.assertNotIn("local_files_only", calls[1])

    def test_mutable_revision_resolves_online_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = write_snapshot(Path(temporary) / "online")
            calls: list[dict[str, object]] = []

            def download(**kwargs: object) -> str:
                calls.append(kwargs)
                return str(snapshot)

            self.assertEqual(
                resolve_snapshot("owner/model", "main", "hub", download),
                snapshot.resolve(),
            )
            self.assertEqual(
                calls, [{"repo_id": "owner/model", "revision": "main"}]
            )

    def test_sharded_snapshot_requires_every_indexed_weight(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = write_snapshot(Path(temporary), complete=False)
            (snapshot / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "layer.0": "model-00001-of-00002.safetensors",
                            "layer.1": "model-00002-of-00002.safetensors",
                        }
                    }
                ),
                encoding="utf-8",
            )
            (snapshot / "model-00001-of-00002.safetensors").write_bytes(b"one")
            with self.assertRaisesRegex(RuntimeError, "incomplete model weights"):
                validate_snapshot(snapshot)


if __name__ == "__main__":
    unittest.main()
