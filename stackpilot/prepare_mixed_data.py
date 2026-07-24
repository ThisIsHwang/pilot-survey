from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

MIXED_DATA_SCHEMA = 3
MARKER_TEMPLATE = "<retrieval_environment>{backend}</retrieval_environment>\n"
MARKER_PATTERN = re.compile(
    r"\A(?:<retrieval_environment>\s*(?:bm25|e5)\s*</retrieval_environment>\r?\n)+",
    re.IGNORECASE,
)
BACKENDS = ("bm25", "e5")
MODES = ("backend-id", "hidden-paired")


def validate_backend(backend: str) -> str:
    normalized = str(backend).strip().lower()
    if normalized not in BACKENDS:
        raise ValueError(f"backend must be bm25 or e5; got {backend!r}")
    return normalized


def canonicalize_prompt(
    prompt: list[dict[str, Any]], backend: str | None
) -> list[dict[str, Any]]:
    updated = copy.deepcopy(prompt)
    for message in updated:
        if str(message.get("role", "")) == "user":
            content = MARKER_PATTERN.sub("", str(message.get("content", "")))
            if backend is not None:
                content = MARKER_TEMPLATE.format(backend=backend) + content
            message["content"] = content
            return updated
    raise ValueError("Search-R1 row has no user prompt to annotate")


def add_marker(prompt: list[dict[str, Any]], backend: str) -> list[dict[str, Any]]:
    return canonicalize_prompt(prompt, validate_backend(backend))


def backend_qualified_index(source_index: Any, backend: str) -> str:
    normalized = validate_backend(backend)
    if source_index is None or not str(source_index).strip():
        raise ValueError("Search-R1 row has an empty extra_info.index")
    return f"{str(source_index).strip()}::retrieval_backend={normalized}"


def duplicate_row(
    row: dict[str, Any],
    backend: str,
    *,
    expose_backend: bool = True,
    require_index: bool = False,
) -> dict[str, Any]:
    normalized = validate_backend(backend)
    output = copy.deepcopy(row)
    prompt = list(row["prompt"])
    output["prompt"] = canonicalize_prompt(
        prompt, normalized if expose_backend else None
    )
    extra_info = dict(row.get("extra_info") or {})
    source_index = str(
        extra_info.get("source_index") or extra_info.get("index") or ""
    ).strip()
    if not source_index:
        if require_index:
            raise ValueError("Search-R1 row is missing a non-empty extra_info.index")
    else:
        extra_info["source_index"] = source_index
        extra_info["index"] = backend_qualified_index(source_index, normalized)
    extra_info["routing_backend"] = normalized
    output["extra_info"] = extra_info
    return output


def paired_rows(
    source: list[dict[str, Any]], *, expose_backend: bool
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in source:
        base = dict(row)
        rows.extend(
            duplicate_row(
                base,
                backend,
                expose_backend=expose_backend,
                require_index=True,
            )
            for backend in BACKENDS
        )
    return rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path(output_path: Path) -> Path:
    return output_path.with_name(f".{output_path.name}.stackpilot.json")


def preparation_request(
    input_path: Path,
    seed: int,
    mode: str,
    *,
    source_rows: int | None = None,
) -> dict[str, Any]:
    source = input_path.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Mixed-policy source parquet is missing: {source}")
    request = {
        "schema": MIXED_DATA_SCHEMA,
        "input_path": str(source),
        "input_size": source.stat().st_size,
        "input_sha256": file_sha256(source),
        "seed": int(seed),
        "mode": mode,
        "preparer_sha256": file_sha256(Path(__file__).resolve()),
    }
    if source_rows is not None:
        if source_rows < 0:
            raise ValueError("source_rows must be non-negative")
        request["source_rows"] = int(source_rows)
    return request


def prepared_cache_valid(
    output_path: Path,
    request: dict[str, Any],
) -> bool:
    sidecar = manifest_path(output_path)
    if not output_path.is_file() or not sidecar.is_file():
        return False
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if payload.get("schema") != MIXED_DATA_SCHEMA:
        return False
    if payload.get("request") != request:
        return False
    output = payload.get("output")
    if not isinstance(output, dict):
        return False
    if output.get("size") != output_path.stat().st_size:
        return False
    source_rows = request.get("source_rows")
    if not isinstance(source_rows, int) or source_rows < 0:
        return False
    if output.get("rows") != source_rows * len(BACKENDS):
        return False
    return output.get("sha256") == file_sha256(output_path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare(
    input_path: Path,
    output_path: Path,
    seed: int,
    mode: str = "backend-id",
    *,
    force: bool = False,
) -> bool:
    from datasets import Dataset, load_dataset

    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}; got {mode!r}")
    source = load_dataset("parquet", data_files=str(input_path), split="train")
    request = preparation_request(
        input_path,
        seed,
        mode,
        source_rows=len(source),
    )
    if not force and prepared_cache_valid(output_path, request):
        print(f"Reusing verified {mode} rows: {output_path}")
        return True

    rows = paired_rows(
        list(source),
        expose_backend=mode == "backend-id",
    )
    dataset = Dataset.from_list(rows).shuffle(seed=seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    try:
        dataset.to_parquet(str(temporary))
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    atomic_write_json(
        manifest_path(output_path),
        {
            "schema": MIXED_DATA_SCHEMA,
            "request": request,
            "output": {
                "rows": len(dataset),
                "size": output_path.stat().st_size,
                "sha256": file_sha256(output_path),
            },
        },
    )
    print(f"Wrote {len(dataset):,} {mode} rows to {output_path}")
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=MODES, default="backend-id")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    prepare(
        Path(args.input),
        Path(args.output),
        args.seed,
        args.mode,
        force=args.force,
    )


if __name__ == "__main__":
    main()
