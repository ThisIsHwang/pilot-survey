from __future__ import annotations

import argparse
import hashlib
import os
import sys
import threading
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from stackpilot.cuda_compat import (
    configure_cuda_attention,
    load_e5_with_eager_attention,
)
from stackpilot.faiss_gpu import paged_flat_gpu_loader
from stackpilot.retrieval_concurrency import batch_search


class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False


def source_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-r1-root", required=True)
    parser.add_argument("--index-path", required=True)
    parser.add_argument("--corpus-path", required=True)
    parser.add_argument("--retriever-name", required=True)
    parser.add_argument("--retriever-model", default="intfloat/e5-base-v2")
    parser.add_argument("--retriever-model-revision", default=None)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--faiss-gpu", action="store_true")
    parser.add_argument("--faiss-gpu-paged-load", action="store_true")
    parser.add_argument("--faiss-gpu-temp-memory-mib", type=int, default=512)
    parser.add_argument("--expected-documents", type=int, default=None)
    args = parser.parse_args()

    if args.expected_documents is not None and args.expected_documents <= 0:
        parser.error("--expected-documents must be positive")
    if args.faiss_gpu_temp_memory_mib <= 0:
        parser.error("--faiss-gpu-temp-memory-mib must be positive")
    if args.faiss_gpu_paged_load and not (
        args.retriever_name.lower() == "e5" and args.faiss_gpu
    ):
        parser.error("--faiss-gpu-paged-load requires an E5 --faiss-gpu server")
    disable_empty_cache_value = os.environ.get(
        "RETRIEVER_DISABLE_CUDA_EMPTY_CACHE", "0"
    )
    if disable_empty_cache_value not in {"0", "1"}:
        parser.error("RETRIEVER_DISABLE_CUDA_EMPTY_CACHE must be 0 or 1")
    disable_empty_cache = (
        disable_empty_cache_value == "1"
        and args.retriever_name.lower() == "e5"
        and args.faiss_gpu
    )

    sys.path.insert(0, str(Path(args.search_r1_root).resolve()))
    from search_r1.search import retrieval_server

    if args.retriever_name.lower() == "e5":
        configure_cuda_attention()
        retrieval_server.load_model = load_e5_with_eager_attention
        print("E5 attention backend: eager (cuDNN SDPA disabled)")

    config = retrieval_server.Config(
        retrieval_method=args.retriever_name,
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        retrieval_topk=args.topk,
        faiss_gpu=args.faiss_gpu,
        retrieval_model_path=args.retriever_model,
        retrieval_pooling_method="mean",
        retrieval_query_max_length=256,
        retrieval_use_fp16=True,
        retrieval_batch_size=256,
    )
    faiss_load = None
    if args.faiss_gpu_paged_load:
        with paged_flat_gpu_loader(
            retrieval_server.faiss,
            temp_memory_mib=args.faiss_gpu_temp_memory_mib,
        ) as faiss_load:
            retriever = retrieval_server.get_retriever(config)
        if faiss_load.documents is None:
            raise RuntimeError(
                "Search-R1 did not invoke the memory-safe FAISS GPU loader"
            )
        # FAISS GPU indexes borrow their resource object. Keep it alive for the
        # full retriever lifetime after the temporary monkeypatch is restored.
        retriever._stackpilot_faiss_resources = faiss_load.resources
    else:
        retriever = retrieval_server.get_retriever(config)
    if disable_empty_cache:
        # The hard-RQ0 E5 server owns GPU 7 exclusively. Retaining its allocator
        # cache avoids two synchronizing empty_cache calls per query batch.
        retrieval_server.torch.cuda.empty_cache = lambda: None
        print("E5 CUDA allocator cache retained between retrieval batches")
    index_documents: int | None = None
    corpus_documents: int | None = None
    if args.expected_documents is not None:
        if args.retriever_name.lower() == "bm25":
            from pyserini.index.lucene import IndexReader

            index_documents = int(IndexReader(args.index_path).stats()["documents"])
            corpus_documents = (
                index_documents
                if getattr(retriever, "contain_doc", False)
                else len(retriever.corpus)
            )
        elif args.retriever_name.lower() == "e5":
            index_documents = int(retriever.index.ntotal)
            corpus_documents = len(retriever.corpus)
        else:
            raise ValueError(
                "--expected-documents is supported only for bm25 and e5 retrievers"
            )
        if index_documents != args.expected_documents:
            raise ValueError(
                f"{args.retriever_name} index has {index_documents:,} documents; "
                f"expected {args.expected_documents:,}"
            )
        if corpus_documents != args.expected_documents:
            raise ValueError(
                f"{args.retriever_name} corpus has {corpus_documents:,} documents; "
                f"expected {args.expected_documents:,}"
            )
    retriever_model: str | None = None
    retriever_model_revision: str | None = None
    if args.retriever_name.lower() == "e5":
        model_path = Path(args.retriever_model).expanduser()
        if model_path.is_dir():
            model_path = model_path.resolve()
            retriever_model = str(model_path)
            if model_path.parent.name == "snapshots":
                retriever_model_revision = model_path.name
        else:
            retriever_model = args.retriever_model
        if args.retriever_model_revision:
            if (
                retriever_model_revision is not None
                and retriever_model_revision != args.retriever_model_revision
            ):
                raise ValueError(
                    "Retriever model revision does not match its local snapshot: "
                    f"{args.retriever_model_revision} != {retriever_model_revision}"
                )
            retriever_model_revision = args.retriever_model_revision
    faiss_gpu_count = (
        int(retrieval_server.faiss.get_num_gpus())
        if args.retriever_name.lower() == "e5"
        else 0
    )
    index_class = type(getattr(retriever, "index", None)).__name__
    search_lock = (
        threading.Lock()
        if args.retriever_name.lower() == "e5" and args.faiss_gpu
        else None
    )
    server_files = {
        path.name: source_digest(path)
        for path in (
            Path(__file__).resolve(),
            Path(__file__).with_name("faiss_gpu.py").resolve(),
            Path(__file__).with_name("retrieval_concurrency.py").resolve(),
        )
    }
    app = FastAPI()

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "process_id": os.getpid(),
            "backend": args.retriever_name,
            "index_path": str(Path(args.index_path).resolve()),
            "corpus_path": str(Path(args.corpus_path).resolve()),
            "faiss_gpu": bool(args.faiss_gpu),
            "faiss_gpu_count": faiss_gpu_count,
            "faiss_gpu_load_mode": (
                faiss_load.mode
                if faiss_load is not None
                else ("clone" if args.faiss_gpu else "cpu")
            ),
            "faiss_storage_dtype": (
                faiss_load.storage_dtype if faiss_load is not None else None
            ),
            "faiss_temp_memory_mib": (
                faiss_load.temp_memory_mib if faiss_load is not None else None
            ),
            "faiss_index_bytes": (
                faiss_load.index_bytes if faiss_load is not None else None
            ),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "index_class": index_class,
            "index_documents": index_documents,
            "corpus_documents": corpus_documents,
            "retriever_model": retriever_model,
            "retriever_model_revision": retriever_model_revision,
            "gpu_search_serialized": search_lock is not None,
            "cuda_empty_cache_disabled": disable_empty_cache,
            "server_files": server_files,
        }

    @app.post("/retrieve")
    def retrieve(request: QueryRequest):
        topk = request.topk or args.topk
        results, scores = batch_search(
            retriever, request.queries, topk, search_lock
        )
        payload = []
        for docs, doc_scores in zip(results, scores):
            combined = []
            for doc, score in zip(docs, doc_scores):
                combined.append({"document": doc, "score": float(score)})
            payload.append(combined)
        return {"result": payload}

    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
