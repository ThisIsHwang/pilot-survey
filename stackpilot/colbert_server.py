from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from stackpilot.cuda_compat import configure_cuda_attention
from stackpilot.ragatouille_compat import install_langchain_retriever_compat


class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False


def main() -> None:
    configure_cuda_attention()
    install_langchain_retriever_compat()
    from ragatouille import RAGPretrainedModel

    parser = argparse.ArgumentParser()
    parser.add_argument("--index-path", required=True)
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    rag = RAGPretrainedModel.from_index(args.index_path)
    warmed = rag.search("pilot server warmup", k=1)
    if not warmed:
        raise RuntimeError("ColBERT server warmup search returned no result")
    app = FastAPI()

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "backend": "colbert",
            "index_path": str(Path(args.index_path).resolve()),
        }

    @app.post("/retrieve")
    def retrieve(request: QueryRequest):
        k = request.topk or args.topk
        all_results = rag.search(request.queries, k=k)
        if (
            request.queries
            and len(request.queries) == 1
            and all_results
            and isinstance(all_results[0], dict)
        ):
            all_results = [all_results]
        payload = []
        for results in all_results:
            single = []
            for result in results:
                document = {
                    "id": str(result.get("document_id", "")),
                    "title": str(result.get("document_metadata", {}).get("title", "")),
                    "text": str(result.get("content", "")),
                    "contents": str(result.get("content", "")),
                }
                single.append(
                    {"document": document, "score": float(result.get("score", 0.0))}
                )
            payload.append(single)
        return {"result": payload}

    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
