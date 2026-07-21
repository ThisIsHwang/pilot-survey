from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


def corpus_state(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    documents = 0
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            if line.strip():
                documents += 1
    if documents == 0:
        raise ValueError(f"Corpus is empty: {path}")
    return {"sha256": digest.hexdigest(), "documents": documents}


def colbert_artifact_state(index_path: Path) -> list[dict[str, Any]]:
    metadata_path = index_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    num_chunks = int(metadata.get("num_chunks", 0))
    if num_chunks <= 0:
        raise ValueError(
            f"Invalid ColBERT chunk count in {metadata_path}: {num_chunks}"
        )

    names = [
        "metadata.json",
        "collection.json",
        "pid_docid_map.json",
        "docid_metadata_map.json",
        "plan.json",
        "centroids.pt",
        "avg_residual.pt",
        "buckets.pt",
        "ivf.pid.pt",
    ]
    for chunk in range(num_chunks):
        names.extend(
            [
                f"{chunk}.codes.pt",
                f"{chunk}.residuals.pt",
                f"{chunk}.metadata.json",
                f"doclens.{chunk}.json",
            ]
        )

    artifacts = []
    for name in names:
        path = index_path / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"ColBERT index artifact is missing or empty: {path}")
        artifacts.append({"path": name, "size": path.stat().st_size})
    return artifacts


def validate_index(kind: str, index_path: Path, expected_documents: int) -> None:
    if kind == "bm25":
        from pyserini.index.lucene import IndexReader

        stats = IndexReader(str(index_path)).stats()
        actual = int(stats["documents"])
    elif kind == "e5":
        import faiss

        actual = int(faiss.read_index(str(index_path)).ntotal)
    elif kind == "colbert":
        colbert_artifact_state(index_path)
        collection = json.loads(
            (index_path / "collection.json").read_text(encoding="utf-8")
        )
        pid_map = json.loads(
            (index_path / "pid_docid_map.json").read_text(encoding="utf-8")
        )
        actual = len(collection)
        if len(pid_map) != actual:
            raise ValueError(
                f"ColBERT collection/pid map mismatch: {actual} vs {len(pid_map)}"
            )
        metadata = json.loads(
            (index_path / "metadata.json").read_text(encoding="utf-8")
        )
        passage_count = 0
        embedding_count = 0
        for chunk in range(int(metadata["num_chunks"])):
            chunk_metadata = json.loads(
                (index_path / f"{chunk}.metadata.json").read_text(encoding="utf-8")
            )
            doclens = json.loads(
                (index_path / f"doclens.{chunk}.json").read_text(encoding="utf-8")
            )
            chunk_passages = int(chunk_metadata["num_passages"])
            chunk_embeddings = int(chunk_metadata["num_embeddings"])
            if (
                len(doclens) != chunk_passages
                or sum(map(int, doclens)) != chunk_embeddings
            ):
                raise ValueError(f"ColBERT chunk {chunk} metadata/doclens mismatch")
            passage_count += chunk_passages
            embedding_count += chunk_embeddings
        if passage_count != actual or embedding_count != int(
            metadata["num_embeddings"]
        ):
            raise ValueError(
                "ColBERT aggregate metadata mismatch: "
                f"passages={passage_count}/{actual}, "
                f"embeddings={embedding_count}/{metadata['num_embeddings']}"
            )
    else:
        raise ValueError(f"Unknown index kind: {kind}")

    if actual != expected_documents:
        raise ValueError(
            f"{kind} index has {actual:,} documents; expected {expected_documents:,}"
        )


def expected_manifest(args: argparse.Namespace) -> dict[str, Any]:
    manifest = {
        "schema": 2 if args.kind == "colbert" else 1,
        "kind": args.kind,
        "model": args.model,
        "corpus": corpus_state(Path(args.corpus)),
    }
    if args.kind == "colbert":
        manifest["index_artifacts"] = colbert_artifact_state(Path(args.index))
    return manifest


def record(args: argparse.Namespace) -> None:
    manifest = expected_manifest(args)
    validate_index(args.kind, Path(args.index), manifest["corpus"]["documents"])
    output = Path(args.manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output)
    print(f"Recorded validated {args.kind} index: {args.index}")


def check(args: argparse.Namespace) -> None:
    path = Path(args.manifest)
    if not path.is_file():
        raise ValueError(f"completion manifest is missing: {path}")
    current = json.loads(path.read_text(encoding="utf-8"))
    expected = expected_manifest(args)
    if current != expected:
        raise ValueError("corpus or index model settings changed")
    validate_index(args.kind, Path(args.index), expected["corpus"]["documents"])
    print(f"Reusing validated {args.kind} index: {args.index}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("check", "record"))
    parser.add_argument("--kind", choices=("bm25", "e5", "colbert"), required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    try:
        (check if args.action == "check" else record)(args)
    except Exception as exc:
        print(f"{args.kind} index validation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
