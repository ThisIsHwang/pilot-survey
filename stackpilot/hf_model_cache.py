from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path


IMMUTABLE_REVISION = re.compile(r"[0-9a-fA-F]{40,64}")
SnapshotDownload = Callable[..., str]


def validate_snapshot(snapshot: Path) -> None:
    config_path = snapshot / "config.json"
    try:
        json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid model config in {snapshot}: {exc}") from exc

    index_files = sorted(snapshot.glob("*.safetensors.index.json")) or sorted(
        snapshot.glob("*.bin.index.json")
    )
    if len(index_files) > 1:
        raise RuntimeError(
            f"Multiple model weight indexes found in {snapshot}: {index_files}"
        )
    if index_files:
        try:
            index = json.loads(index_files[0].read_text(encoding="utf-8"))
            weights = [
                snapshot / name
                for name in sorted(set(index["weight_map"].values()))
            ]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Invalid model weight index {index_files[0]}: {exc}"
            ) from exc
    else:
        weights = sorted(snapshot.glob("*.safetensors")) or sorted(
            snapshot.glob("*.bin")
        )

    bad_weights = [
        path for path in weights if not path.is_file() or path.stat().st_size == 0
    ]
    if not weights or bad_weights:
        raise RuntimeError(
            f"Missing or incomplete model weights in {snapshot}: "
            f"{bad_weights or 'none found'}"
        )
    if not (snapshot / "tokenizer.json").is_file() and not (
        snapshot / "tokenizer.model"
    ).is_file():
        raise RuntimeError(f"Tokenizer files are missing from {snapshot}")


def resolve_snapshot(
    model_ref: str,
    revision: str,
    model_kind: str,
    snapshot_download: SnapshotDownload | None = None,
) -> Path:
    if model_kind == "local":
        snapshot = Path(model_ref).expanduser().resolve()
        validate_snapshot(snapshot)
        return snapshot
    if model_kind != "hub":
        raise ValueError(f"Unknown model kind: {model_kind}")

    if snapshot_download is None:
        from huggingface_hub import snapshot_download as hub_snapshot_download

        snapshot_download = hub_snapshot_download

    # A commit hash is immutable, so the cache can be checked without a Hub
    # metadata request. Branches and tags remain online to avoid stale refs.
    if IMMUTABLE_REVISION.fullmatch(revision):
        try:
            cached = Path(
                snapshot_download(
                    repo_id=model_ref,
                    revision=revision,
                    local_files_only=True,
                )
            ).resolve()
            validate_snapshot(cached)
            print(
                f"Reusing cached {model_ref}@{revision} from {cached}",
                file=sys.stderr,
            )
            return cached
        except Exception as exc:
            print(
                f"Cached {model_ref}@{revision} is absent or incomplete "
                f"({exc}); downloading the missing files.",
                file=sys.stderr,
            )

    snapshot = Path(
        snapshot_download(repo_id=model_ref, revision=revision)
    ).resolve()
    validate_snapshot(snapshot)
    return snapshot


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve and validate an HF snapshot")
    parser.add_argument("model_ref")
    parser.add_argument("revision")
    parser.add_argument("model_kind", choices=("hub", "local"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        snapshot = resolve_snapshot(args.model_ref, args.revision, args.model_kind)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(snapshot)
    if args.model_kind == "hub":
        print(
            f"Pinned {args.model_ref}@{args.revision} to snapshot {snapshot.name}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
