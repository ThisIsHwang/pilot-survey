from __future__ import annotations

import argparse
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
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--faiss-gpu", action="store_true")
    args = parser.parse_args()

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
    app = FastAPI()

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "backend": args.retriever_name,
            "index_path": str(Path(args.index_path).resolve()),
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
