from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO, Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - the production downloader runs on Linux
    fcntl = None  # type: ignore[assignment]


SCHEMA = 2
HASH_CHUNK_SIZE = 16 * 1024 * 1024
EDGE_BYTES = 1024 * 1024

E5_REPO = "PeterJinGo/wiki-18-e5-index"
E5_REVISION = "a4d31160a035f30764604f4827cd8f1d0315eb86"
E5_PARTS = (
    (
        "part_aa",
        42_949_672_960,
        "a8a6a246951da4bbc8771a223283ef61963882a32864d9044ec00abb90fc3023",
    ),
    (
        "part_ab",
        21_609_402_413,
        "b6d9bc943626fe7cb44de4c849e9379e7f272ab216c0552acbcf2390cc033c11",
    ),
)
E5_INDEX_SIZE = sum(size for _, size, _ in E5_PARTS)

CORPUS_REPO = "PeterJinGo/wiki-18-corpus"
CORPUS_REVISION = "69c1c00ffe7c5554c68d8548355cb22e46aabc51"
CORPUS_ARCHIVE = "wiki-18.jsonl.gz"
CORPUS_ARCHIVE_SIZE = 5_123_307_260
CORPUS_ARCHIVE_SHA256 = (
    "7abd929223399cd63c52b499f289bf4f9039be1e9f8c43e1cb3938305b2317db"
)
EXPECTED_DOCUMENTS = 21_015_324

BM25_REPO = "PeterJinGo/wiki-18-bm25-index"
BM25_REVISION = "2c7554f25f425038c4bcb155735a0f831851fd78"
BM25_FILES = (
    ("_0.fdm", 621, "a95f7bb2104b12292c06ee4f4f3f582503863c457474348a9126b67b0ba25dfb"),
    (
        "_0.fdt",
        54_155_986,
        "48b1fe798dbc81f9df11339bfc2ca4b1b5305b8ad46bb5ac74584a6d790e872e",
    ),
    (
        "_0.fdx",
        21_869,
        "3064b8f8ed2376363df7f3330ba54d54973f57ec7ca5861cb71f8a05147e6459",
    ),
    ("_0.fnm", 322, "6e7ac467aa3dc918ba4f4480ba2d8d06a48fb1902a50ac41c2d9d4b74cd18aca"),
    (
        "_0.nvd",
        12_068_224,
        "04b7cd27447ab324ee288272917a94b521b1fcfd716c7c434d524a77797f6fd8",
    ),
    ("_0.nvm", 103, "b078face9aac8fbaebc5522e6856e1bb765ad84730b8f0e87416ede545ba2d18"),
    ("_0.si", 477, "2998d9a8b39c9919b357f89788e4dd26bf8769f42c031dd0e36e2ff9275ebc40"),
    (
        "_0_Lucene90_0.doc",
        1_071_202_432,
        "7e8532e78cf408f28a7886c32c0091bca96c9c747b4a36d854c7809683544895",
    ),
    (
        "_0_Lucene90_0.dvd",
        85_958_571,
        "1bc9a0ed11432dc7cee302879084a554ba5329858130021434bdce048e0dddef",
    ),
    (
        "_0_Lucene90_0.dvm",
        4_035,
        "8209f7acbfa3e6f6b71941b36c917626b6d4fa74ef826632ad0bad3f1bc42dda",
    ),
    (
        "_0_Lucene90_0.tim",
        93_697_175,
        "921723929339bc0c93178fd1d9d4442fe4a50a9c1c50db59c8c2e3391523b4be",
    ),
    (
        "_0_Lucene90_0.tip",
        2_661_050,
        "20a5e03674c2b7d6731fcdb6e702f68465d6bdb6d241c96d184c125ac44842a1",
    ),
    (
        "_0_Lucene90_0.tmd",
        295,
        "8bfc370b4666f68b3e3b6131bb5ac653836943afe773484358361a14ecd8d0d6",
    ),
    ("_1.fdm", 495, "24d26fbb31eaa94dae867a772862b34673a6215a63e830ff2b9f8cc74db2f0a9"),
    (
        "_1.fdt",
        40_464_480,
        "1fd0017b76f7180ad661b58f8d3a4f0f7f9a61b8989e510cedf0e7cdfa243fbb",
    ),
    (
        "_1.fdx",
        16_320,
        "9b37312224087cb1ed63702bd3548aa2599c3d353e7bf3528c5b2f5965ec0185",
    ),
    ("_1.fnm", 322, "daec2d15e89ce61c331acec8f2d3f47df6d278d3d6bcd8386ef4e73fc504c2db"),
    (
        "_1.nvd",
        8_947_218,
        "908fe8a323addab721442cf50a4a8f707a28254b0ed654543bccd392b01e4504",
    ),
    ("_1.nvm", 103, "476da7979ffdb40e58b278b5e1a4ebc540cb80f1da210e6445fe015c006b98cb"),
    ("_1.si", 477, "8445053549810e5aa1c8cf4052f4ecd5dd7404c19a4169bb931ece843c8f56b6"),
    (
        "_1_Lucene90_0.doc",
        792_865_581,
        "53492b1e601b359b99085093db473bb235ee67edfa7cc052c4366370f64308bb",
    ),
    (
        "_1_Lucene90_0.dvd",
        71_577_345,
        "2cb1278d6cc61690b56a7a0d6faf306002b36e476dc26229294e2d8b8b439a29",
    ),
    (
        "_1_Lucene90_0.dvm",
        133,
        "ef242e73aedf4b7e1e9c1da236039f9511829712c848a8ea81156d2515dfcee7",
    ),
    (
        "_1_Lucene90_0.tim",
        61_786_947,
        "b3ee12e0eaafb1d429a77e724b65ef77c38db52119aacda7e8958a8e3e353d6f",
    ),
    (
        "_1_Lucene90_0.tip",
        2_180_221,
        "9323fa705e1bd80113cd890f4ebf421cacff2894edb33360dd49b6e75bc6b470",
    ),
    (
        "_1_Lucene90_0.tmd",
        301,
        "ac58a8224681ee63f273239b078aaf569c71011900818235674cd8e7e3ae476c",
    ),
    (
        "segments_1",
        236,
        "a7ffed5cae6aa8c23315fe429df494390941127aa20546c828d35f68e4e88e41",
    ),
    (
        "write.lock",
        0,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    ),
)
MANIFEST_NAME = ".hard-rq0-assets-manifest.json"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def copy_chunks(source: BinaryIO, target: BinaryIO) -> int:
    copied = 0
    while chunk := source.read(HASH_CHUNK_SIZE):
        target.write(chunk)
        copied += len(chunk)
    return copied


def copy_and_hash(source: BinaryIO, target: BinaryIO) -> tuple[int, str]:
    """Copy a stream while calculating the digest of the exact bytes copied."""
    copied = 0
    digest = hashlib.sha256()
    while chunk := source.read(HASH_CHUNK_SIZE):
        target.write(chunk)
        copied += len(chunk)
        digest.update(chunk)
    return copied, digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(
    path: Path, *, expected_size: int, expected_sha256: str, label: str
) -> dict[str, Any]:
    """Verify a downloaded source against independently pinned metadata."""
    actual_size = path.stat().st_size if path.is_file() else -1
    if actual_size != expected_size:
        raise RuntimeError(
            f"{label} has size {actual_size:,}; expected {expected_size:,}: {path}"
        )
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"{label} SHA-256 is {actual_sha256}; expected {expected_sha256}: {path}"
        )
    return {"size": actual_size, "sha256": actual_sha256}


def edge_sha256(path: Path, *, edge_bytes: int = EDGE_BYTES) -> dict[str, str]:
    """Cheaply fingerprint both ends of a large immutable artifact."""
    if edge_bytes < 1:
        raise ValueError("edge_bytes must be positive")
    size = path.stat().st_size
    with path.open("rb") as handle:
        first = handle.read(edge_bytes)
        handle.seek(max(0, size - edge_bytes))
        last = handle.read(edge_bytes)
    return {
        "first_edge_sha256": hashlib.sha256(first).hexdigest(),
        "last_edge_sha256": hashlib.sha256(last).hexdigest(),
    }


def _last_nonempty_line(path: Path) -> bytes:
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell()
        if position == 0:
            return b""
        handle.seek(-1, os.SEEK_END)
        if handle.read(1) != b"\n":
            raise RuntimeError(f"wiki-18 corpus has an incomplete final row: {path}")
        buffer = b""
        while position > 0:
            amount = min(EDGE_BYTES, position)
            position -= amount
            handle.seek(position)
            buffer = handle.read(amount) + buffer
            lines = [line for line in buffer.splitlines() if line.strip()]
            if position == 0 or len(lines) >= 2:
                return lines[-1] if lines else b""
    return b""


def _validate_corpus_row(raw: bytes, label: str, path: Path) -> None:
    try:
        row = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"wiki-18 corpus has an invalid {label} row: {path}"
        ) from exc
    if not isinstance(row, dict) or not (
        row.get("contents") or row.get("text") or row.get("content")
    ):
        raise RuntimeError(f"wiki-18 corpus {label} row has no document text: {path}")


def validate_corpus(path: Path) -> dict[str, Any]:
    """Perform a fast reuse check without rescanning all 21 million rows."""
    if not path.is_file() or path.stat().st_size <= CORPUS_ARCHIVE_SIZE:
        raise RuntimeError(f"wiki-18 corpus is missing or truncated: {path}")
    with path.open("rb") as handle:
        first = handle.readline()
    if not first.endswith(b"\n"):
        raise RuntimeError(f"wiki-18 corpus has an incomplete first row: {path}")
    last = _last_nonempty_line(path)
    if not last:
        raise RuntimeError(f"wiki-18 corpus has no readable final row: {path}")
    _validate_corpus_row(first, "first", path)
    _validate_corpus_row(last, "last", path)
    return {
        "path": path.name,
        "size": path.stat().st_size,
        "first_line_sha256": hashlib.sha256(first).hexdigest(),
        "last_line_sha256": hashlib.sha256(last).hexdigest(),
    }


def validate_e5(path: Path) -> dict[str, Any]:
    actual = path.stat().st_size if path.is_file() else -1
    if actual != E5_INDEX_SIZE:
        raise RuntimeError(
            f"E5 flat index has size {actual:,}; expected {E5_INDEX_SIZE:,}: {path}"
        )
    return {"path": path.name, "size": actual, **edge_sha256(path)}


def validate_bm25(
    path: Path, root: Path, *, hash_files: bool = False
) -> dict[str, Any]:
    if not path.is_dir():
        raise RuntimeError(f"BM25 Lucene index is missing: {path}")
    files = sorted(
        (item for item in path.iterdir() if item.is_file()), key=lambda p: p.name
    )
    expected = {name: (size, digest) for name, size, digest in BM25_FILES}
    actual_sizes = {item.name: item.stat().st_size for item in files}
    expected_sizes = {name: size for name, (size, _) in expected.items()}
    if actual_sizes != expected_sizes:
        missing = sorted(set(expected_sizes) - set(actual_sizes))
        extra = sorted(set(actual_sizes) - set(expected_sizes))
        wrong_size = sorted(
            name
            for name in set(actual_sizes) & set(expected_sizes)
            if actual_sizes[name] != expected_sizes[name]
        )
        raise RuntimeError(
            "BM25 index does not match the pinned file set: "
            f"missing={missing}, extra={extra}, wrong_size={wrong_size}: {path}"
        )
    total = sum(actual_sizes.values())
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(
            f"BM25 index resolves outside the asset root: {path}"
        ) from exc
    records = []
    for item in files:
        record: dict[str, Any] = {"name": item.name, "size": item.stat().st_size}
        if hash_files:
            actual_digest = sha256_file(item)
            expected_digest = expected[item.name][1]
            if actual_digest != expected_digest:
                raise RuntimeError(
                    f"BM25 file {item.name} SHA-256 is {actual_digest}; "
                    f"expected {expected_digest}: {item}"
                )
            record["sha256"] = actual_digest
        records.append(record)
    return {"path": relative.as_posix(), "size": total, "files": records}


def source_identity() -> dict[str, Any]:
    return {
        "e5": {
            "repo_id": E5_REPO,
            "revision": E5_REVISION,
            "parts": [
                {"name": name, "size": size, "sha256": digest}
                for name, size, digest in E5_PARTS
            ],
        },
        "corpus": {
            "repo_id": CORPUS_REPO,
            "revision": CORPUS_REVISION,
            "archive": {
                "name": CORPUS_ARCHIVE,
                "size": CORPUS_ARCHIVE_SIZE,
                "sha256": CORPUS_ARCHIVE_SHA256,
            },
            "documents": EXPECTED_DOCUMENTS,
        },
        "bm25": {
            "repo_id": BM25_REPO,
            "revision": BM25_REVISION,
            "files": [
                {"name": name, "size": size, "sha256": digest}
                for name, size, digest in BM25_FILES
            ],
        },
    }


def locate_bm25(root: Path) -> Path:
    link = root / "bm25"
    canonical = root / f"bm25-pinned-{BM25_REVISION}" / "bm25"
    preferred = root / "bm25-download" / "bm25"
    if link.is_dir():
        return link
    if canonical.is_dir():
        return canonical
    if preferred.is_dir():
        return preferred
    raise RuntimeError(f"BM25 Lucene index is missing under {root}")


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Hard-RQ0 asset manifest is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Hard-RQ0 asset manifest is invalid: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Hard-RQ0 asset manifest must be a JSON object: {path}")
    if payload.get("schema") != SCHEMA:
        raise RuntimeError(
            f"Hard-RQ0 asset manifest schema {payload.get('schema')!r} is not the "
            f"provenance-safe schema {SCHEMA}; rebuild the assets"
        )
    return payload


def _without_hashes(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": record.get("path"),
        "size": record.get("size"),
        "files": [
            {"name": item.get("name"), "size": item.get("size")}
            for item in record.get("files", [])
            if isinstance(item, dict)
        ],
    }


def _validate_recorded_hashes(record: dict[str, Any], label: str) -> None:
    files = record.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError(f"{label} manifest has no file provenance")
    for item in files:
        digest = item.get("sha256") if isinstance(item, dict) else None
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise RuntimeError(
                f"{label} manifest has an invalid SHA-256 record: {item}"
            )


def _validate_recorded_e5(root: Path, record: Any, manifest: Path) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise RuntimeError(f"E5 provenance is missing from: {manifest}")
    actual = validate_e5(root / "e5_Flat.index")
    expected = {
        **actual,
        "assembled_from": source_identity()["e5"]["parts"],
    }
    if record != expected:
        raise RuntimeError(f"E5 index does not match its completion manifest: {manifest}")
    return record


def _validate_recorded_corpus(
    root: Path, record: Any, manifest: Path
) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise RuntimeError(f"wiki-18 corpus provenance is missing from: {manifest}")
    actual = validate_corpus(root / "wiki-18.jsonl")
    expected = {
        **actual,
        "documents": EXPECTED_DOCUMENTS,
        "source_archive": source_identity()["corpus"]["archive"],
    }
    if record != expected:
        raise RuntimeError(
            f"wiki-18 corpus does not match its completion manifest: {manifest}"
        )
    return record


def _validate_recorded_bm25(
    root: Path, record: Any, manifest: Path
) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise RuntimeError(f"BM25 provenance is missing from: {manifest}")
    _validate_recorded_hashes(record, "BM25")
    if record.get("files") != source_identity()["bm25"]["files"]:
        raise RuntimeError(f"BM25 hashes do not match pinned provenance: {manifest}")
    recorded_path = record.get("path")
    if not isinstance(recorded_path, str) or not recorded_path:
        raise RuntimeError(f"BM25 manifest path is invalid: {manifest}")
    actual = validate_bm25(root / recorded_path, root)
    if _without_hashes(record) != actual:
        raise RuntimeError(
            f"BM25 index does not match its completion manifest: {manifest}"
        )
    return record


def reusable_artifacts(
    root: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Return independently reusable artifacts from a trusted current manifest."""
    manifest = root / MANIFEST_NAME
    try:
        recorded = _load_manifest(manifest)
        if recorded.get("sources") != source_identity():
            raise RuntimeError(
                f"Hard-RQ0 asset source provenance does not match: {manifest}"
            )
        artifacts = recorded.get("artifacts")
        if not isinstance(artifacts, dict):
            raise RuntimeError(f"Hard-RQ0 asset manifest has no artifacts: {manifest}")
    except RuntimeError as exc:
        message = str(exc)
        return {}, {name: message for name in ("corpus", "bm25", "e5")}

    validators = {
        "corpus": _validate_recorded_corpus,
        "bm25": _validate_recorded_bm25,
        "e5": _validate_recorded_e5,
    }
    reusable: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for name, validator in validators.items():
        try:
            reusable[name] = validator(root, artifacts.get(name), manifest)
        except (OSError, RuntimeError) as exc:
            errors[name] = str(exc)
    return reusable, errors


def check(root: Path, *, adopt_legacy: bool = False) -> dict[str, Any]:
    """Validate a schema-2 cache cheaply; never bless unverified legacy files."""
    # Retained only for CLI/API compatibility; adoption is intentionally disabled.
    del adopt_legacy
    reusable, errors = reusable_artifacts(root)
    if errors or set(reusable) != {"corpus", "e5", "bm25"}:
        raise RuntimeError(f"Hard-RQ0 asset validation failed: {errors}")
    link = root / "bm25"
    recorded_bm25 = root / str(reusable["bm25"]["path"])
    if not link.is_dir() or link.resolve() != recorded_bm25.resolve():
        raise RuntimeError(
            f"BM25 compatibility link is missing or stale: {link} -> {recorded_bm25}"
        )
    return {
        "schema": SCHEMA,
        "sources": source_identity(),
        "artifacts": reusable,
    }


def ensure_free_space(root: Path, minimum_gib: int) -> None:
    if minimum_gib < 1:
        raise ValueError("--min-free-gib must be positive")
    available = shutil.disk_usage(root).free
    required = minimum_gib * 1024**3
    if available < required:
        raise RuntimeError(
            f"Remaining full-wiki assets require at least {minimum_gib} GiB free under "
            f"{root}; found {available / 1024**3:.1f} GiB. Set HARD_ASSET_MIN_FREE_GIB "
            "only after confirming the remaining asset sizes."
        )


def _required_free_gib(missing: set[str], full_min_gib: int) -> int:
    if full_min_gib < 1:
        raise ValueError("--full-min-gib must be positive")
    if not missing:
        return 0
    if "e5" in missing:
        return full_min_gib
    # The caller's full-cache override remains an upper bound so an operator who
    # deliberately lowers it is not silently overruled for a smaller component.
    component_default = 40 if "corpus" in missing else 10
    return min(full_min_gib, component_default)


def required_free_gib(root: Path, *, full_min_gib: int) -> int:
    """Return conservative free-space headroom for only the missing components."""
    reusable, _ = reusable_artifacts(root)
    return _required_free_gib(
        {"corpus", "bm25", "e5"} - set(reusable), full_min_gib
    )


def _e5_assembly_state(completed: list[str]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "source": source_identity()["e5"],
        "completed": completed,
    }


def repair_e5_assembly_prefix(
    temporary: Path,
    completed: list[str],
    parts: tuple[tuple[str, int, str], ...] = E5_PARTS,
) -> tuple[list[str], int]:
    """Keep verified parts while discarding only an interrupted partial part."""
    valid_names = [name for name, _, _ in parts]
    if completed != valid_names[: len(completed)]:
        completed = []
    expected_prefix = sum(
        size for name, size, _ in parts if name in set(completed)
    )
    if not temporary.is_file():
        completed = []
        expected_prefix = 0
        temporary.touch()
    else:
        actual_size = temporary.stat().st_size
        if actual_size < expected_prefix:
            completed = []
            expected_prefix = 0
        if actual_size != expected_prefix:
            with temporary.open("r+b") as handle:
                handle.truncate(expected_prefix)
    return completed, expected_prefix


def assemble_e5(root: Path, *, keep_sources: bool) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    target = root / "e5_Flat.index"
    temporary = root / ".e5_Flat.index.assembling"
    state_path = root / ".e5-assembly.json"
    completed: list[str] = []
    if state_path.is_file():
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            if (
                payload.get("schema") == SCHEMA
                and payload.get("source") == source_identity()["e5"]
            ):
                completed = [str(value) for value in payload.get("completed", [])]
        except (OSError, json.JSONDecodeError, AttributeError):
            completed = []
    completed, expected_prefix = repair_e5_assembly_prefix(temporary, completed)
    atomic_json(state_path, _e5_assembly_state(completed))

    for name, expected_size, expected_sha256 in E5_PARTS[len(completed) :]:
        downloaded = Path(
            hf_hub_download(
                repo_id=E5_REPO,
                filename=name,
                repo_type="dataset",
                revision=E5_REVISION,
                local_dir=root,
            )
        )
        if not downloaded.is_file() or downloaded.stat().st_size != expected_size:
            raise RuntimeError(
                f"Downloaded {name} has size "
                f"{downloaded.stat().st_size if downloaded.is_file() else -1:,}; "
                f"expected {expected_size:,}"
            )
        with downloaded.open("rb") as source, temporary.open("ab") as destination:
            copied, actual_sha256 = copy_and_hash(source, destination)
            destination.flush()
            os.fsync(destination.fileno())
        if copied != expected_size or actual_sha256 != expected_sha256:
            with temporary.open("r+b") as handle:
                handle.truncate(expected_prefix)
            downloaded.unlink(missing_ok=True)
            raise RuntimeError(
                f"Downloaded {name} failed pinned verification: size={copied:,}, "
                f"sha256={actual_sha256}; expected size={expected_size:,}, "
                f"sha256={expected_sha256}"
            )
        expected_prefix += copied
        completed.append(name)
        atomic_json(state_path, _e5_assembly_state(completed))
        if not keep_sources:
            downloaded.unlink()

    if temporary.stat().st_size != E5_INDEX_SIZE:
        raise RuntimeError(
            f"Assembled E5 index has size {temporary.stat().st_size:,}; "
            f"expected {E5_INDEX_SIZE:,}"
        )
    os.replace(temporary, target)
    state_path.unlink(missing_ok=True)
    if not keep_sources:
        for name, _, _ in E5_PARTS:
            (root / name).unlink(missing_ok=True)
    return {
        **validate_e5(target),
        "assembled_from": source_identity()["e5"]["parts"],
    }


def decompress_gzip_counted(source: Path, target: Path) -> tuple[int, int]:
    """Decompress a JSONL gzip stream and return (bytes, newline-delimited rows)."""
    copied = 0
    rows = 0
    final_byte = b""
    with gzip.open(source, "rb") as compressed, target.open("wb") as destination:
        while chunk := compressed.read(HASH_CHUNK_SIZE):
            destination.write(chunk)
            copied += len(chunk)
            rows += chunk.count(b"\n")
            final_byte = chunk[-1:]
        destination.flush()
        os.fsync(destination.fileno())
    if copied == 0 or final_byte != b"\n":
        raise RuntimeError(
            f"Decompressed corpus is empty or lacks a final newline: {target}"
        )
    return copied, rows


def decompress_corpus(root: Path, *, keep_sources: bool) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    target = root / "wiki-18.jsonl"
    archive = Path(
        hf_hub_download(
            repo_id=CORPUS_REPO,
            filename=CORPUS_ARCHIVE,
            repo_type="dataset",
            revision=CORPUS_REVISION,
            local_dir=root,
        )
    )
    try:
        verify_file(
            archive,
            expected_size=CORPUS_ARCHIVE_SIZE,
            expected_sha256=CORPUS_ARCHIVE_SHA256,
            label="Downloaded wiki-18 corpus archive",
        )
    except RuntimeError:
        archive.unlink(missing_ok=True)
        raise
    temporary = root / ".wiki-18.jsonl.decompressing"
    temporary.unlink(missing_ok=True)
    _, rows = decompress_gzip_counted(archive, temporary)
    if rows != EXPECTED_DOCUMENTS:
        raise RuntimeError(
            f"wiki-18 corpus has {rows:,} documents; expected {EXPECTED_DOCUMENTS:,}"
        )
    metadata = {
        **validate_corpus(temporary),
        "documents": rows,
        "source_archive": source_identity()["corpus"]["archive"],
    }
    metadata["path"] = target.name
    os.replace(temporary, target)
    if not keep_sources:
        archive.unlink()
    return metadata


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)


def _unused_backup_path(path: Path, label: str) -> Path:
    base = path.parent / f".{path.name}.{label}.{os.getpid()}"
    candidate = base
    suffix = 0
    while candidate.exists() or candidate.is_symlink():
        suffix += 1
        candidate = base.with_name(f"{base.name}.{suffix}")
    return candidate


def ensure_bm25_link(root: Path, installed_index: Path) -> None:
    """Atomically point the compatibility path at a verified installed index."""
    installed_index = installed_index.resolve()
    if not installed_index.is_dir():
        raise RuntimeError(f"Verified BM25 index directory is missing: {installed_index}")
    try:
        relative_target = installed_index.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(
            f"BM25 index resolves outside the asset root: {installed_index}"
        ) from exc

    link = root / "bm25"
    if link.is_dir() and link.resolve() == installed_index:
        return

    legacy_backup: Path | None = None
    if link.is_symlink() or link.is_file():
        link.unlink()
    elif link.is_dir():
        legacy_backup = _unused_backup_path(link, "unselected")
        link.replace(legacy_backup)

    temporary_link = root / f".bm25-link.{os.getpid()}.tmp"
    temporary_link.unlink(missing_ok=True)
    try:
        temporary_link.symlink_to(relative_target, target_is_directory=True)
        os.replace(temporary_link, link)
    except BaseException:
        temporary_link.unlink(missing_ok=True)
        if legacy_backup is not None and not link.exists() and not link.is_symlink():
            legacy_backup.replace(link)
        raise

    if legacy_backup is not None:
        print(f"Removing unverified legacy BM25 index: {legacy_backup}")
        _remove_path(legacy_backup)


def download_bm25(root: Path) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    staging = root / f".bm25-download-{BM25_REVISION}"
    canonical = root / f"bm25-pinned-{BM25_REVISION}"
    installed_index = canonical / "bm25"

    # A crash after installation but before the manifest write must not trigger
    # another network transfer. Full pinned hashes make this safe to adopt.
    try:
        installed_record = validate_bm25(installed_index, root, hash_files=True)
        ensure_bm25_link(root, installed_index)
        _remove_path(staging)
        return installed_record
    except RuntimeError:
        pass

    # A stable local_dir lets huggingface_hub retain and resume partial blobs.
    staging.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=BM25_REPO,
        repo_type="dataset",
        revision=BM25_REVISION,
        local_dir=staging,
        allow_patterns=["bm25/*"],
    )
    staged_index = staging / "bm25"
    staged_record = validate_bm25(staged_index, staging, hash_files=True)

    canonical_backup: Path | None = None
    if canonical.exists() or canonical.is_symlink():
        canonical_backup = _unused_backup_path(canonical, "previous")
        canonical.replace(canonical_backup)
    try:
        staging.replace(canonical)
    except BaseException:
        if canonical_backup is not None and not canonical.exists():
            canonical_backup.replace(canonical)
        raise

    try:
        fast_record = validate_bm25(installed_index, root)
        staged_record["path"] = fast_record["path"]
        if _without_hashes(staged_record) != fast_record:
            raise RuntimeError("Installed BM25 index changed after pinned verification")
        ensure_bm25_link(root, installed_index)
    except BaseException:
        # Keep the fully downloaded canonical tree (and old backup, if any). A
        # retry validates it locally and only repeats the interrupted link step.
        raise
    if canonical_backup is not None:
        _remove_path(canonical_backup)
    return staged_record


@contextmanager
def asset_download_lock(root: Path) -> Iterator[None]:
    """Prevent two download/assembly processes from sharing temporary files."""
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".hard-rq0-assets.lock"
    with lock_path.open("a+b") as handle:
        if fcntl is None:
            yield
            return
        print(f"Acquiring Hard-RQ0 asset lock: {lock_path}")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _download(
    root: Path, *, min_free_gib: int, keep_sources: bool
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / MANIFEST_NAME
    reusable, errors = reusable_artifacts(root)
    missing = {"corpus", "bm25", "e5"} - set(reusable)
    state = {
        "schema": SCHEMA,
        "sources": source_identity(),
        "artifacts": dict(reusable),
    }

    if reusable:
        print("Reusing provenance-verified hard-RQ0 components: " + ", ".join(reusable))
    for name in sorted(missing):
        print(f"Hard-RQ0 component {name} requires completion: {errors.get(name)}")

    if "bm25" in reusable:
        ensure_bm25_link(root, root / str(reusable["bm25"]["path"]))

    if not missing:
        atomic_json(manifest, state)
        checked = check(root)
        (root / ".hard-rq0-assets-incomplete").unlink(missing_ok=True)
        print(f"Reusing provenance-verified hard-RQ0 assets: {root}")
        return checked

    ensure_free_space(root, _required_free_gib(missing, min_free_gib))
    incomplete = root / ".hard-rq0-assets-incomplete"
    incomplete.write_text(
        f"schema={SCHEMA}\npid={os.getpid()}\nmissing={','.join(sorted(missing))}\n",
        encoding="utf-8",
    )
    # Persist every already-verified component before doing network or assembly
    # work. Each following component is committed independently for restart.
    atomic_json(manifest, state)
    # Finish the smaller corpus/BM25 downloads first. E5 is assembled last so
    # a later download failure cannot force a verified 64.6 GB index rebuild;
    # interrupted E5 assembly itself resumes from its durable part checkpoint.
    builders = {
        "corpus": lambda: decompress_corpus(root, keep_sources=keep_sources),
        "bm25": lambda: download_bm25(root),
        "e5": lambda: assemble_e5(root, keep_sources=keep_sources),
    }
    for name in ("corpus", "bm25", "e5"):
        if name not in missing:
            continue
        artifact = builders[name]()
        state["artifacts"][name] = artifact
        atomic_json(manifest, state)
        if name == "bm25":
            ensure_bm25_link(root, root / str(artifact["path"]))

    checked = check(root)
    incomplete.unlink(missing_ok=True)
    print(f"Hard-RQ0 assets ready with pinned provenance: {root}")
    for name, artifact in checked["artifacts"].items():
        print(f"  {name}: {artifact['size'] / 1024**3:.1f} GiB")
    return checked


def download(root: Path, *, min_free_gib: int, keep_sources: bool) -> dict[str, Any]:
    with asset_download_lock(root):
        return _download(
            root, min_free_gib=min_free_gib, keep_sources=keep_sources
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and validate hard-RQ0 assets"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("download", "check", "required-free"):
        child = subparsers.add_parser(name)
        child.add_argument("--root", required=True)
    downloader = subparsers.choices["download"]
    downloader.add_argument("--min-free-gib", type=int, default=150)
    downloader.add_argument("--keep-source-archives", action="store_true")
    subparsers.choices["check"].add_argument(
        "--adopt-legacy",
        action="store_true",
        help="Deprecated compatibility flag; legacy assets are never auto-adopted.",
    )
    subparsers.choices["required-free"].add_argument(
        "--full-min-gib",
        type=int,
        default=150,
        help="Free-space threshold used when the E5 component is missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    if args.command == "download":
        download(
            root,
            min_free_gib=args.min_free_gib,
            keep_sources=args.keep_source_archives,
        )
    elif args.command == "check":
        state = check(root, adopt_legacy=args.adopt_legacy)
        print(json.dumps(state, sort_keys=True))
    else:
        print(required_free_gib(root, full_min_gib=args.full_min_gib))


if __name__ == "__main__":
    main()
