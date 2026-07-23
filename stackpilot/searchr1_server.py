from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from stackpilot.cuda_compat import (
    configure_cuda_attention,
    load_e5_with_eager_attention,
)


class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False


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
    parser.add_argument("--expected-documents", type=int, default=None)
    args = parser.parse_args()

    if args.expected_documents is not None and args.expected_documents <= 0:
        parser.error("--expected-documents must be positive")

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
    retriever = retrieval_server.get_retriever(config)
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
    app = FastAPI()

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "backend": args.retriever_name,
            "index_path": str(Path(args.index_path).resolve()),
            "corpus_path": str(Path(args.corpus_path).resolve()),
            "faiss_gpu": bool(args.faiss_gpu),
            "faiss_gpu_count": faiss_gpu_count,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "index_class": index_class,
            "index_documents": index_documents,
            "corpus_documents": corpus_documents,
            "retriever_model": retriever_model,
            "retriever_model_revision": retriever_model_revision,
        }

    @app.post("/retrieve")
    def retrieve(request: QueryRequest):
        topk = request.topk or args.topk
        results, scores = retriever.batch_search(request.queries, topk, True)
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
