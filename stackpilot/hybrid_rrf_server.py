from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from typing import Any

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class QueryRequest(BaseModel):
    queries: list[str]
    topk: int | None = None
    return_scores: bool = True


def post_batch(url: str, queries: list[str], topk: int, timeout: float) -> list[list[dict[str, Any]]]:
    response = requests.post(
        url,
        json={"queries": queries, "topk": topk, "return_scores": True},
        timeout=timeout,
    )
    response.raise_for_status()
    results = response.json().get("result")
    if not isinstance(results, list) or len(results) != len(queries):
        raise RuntimeError("Upstream retriever returned an invalid batch")
    return results


def document_key(item: dict[str, Any]) -> str:
    document = item.get("document", item)
    for field in ("id", "document_id", "docid"):
        value = document.get(field)
        if value is not None and str(value):
            return f"id:{value}"
    contents = str(document.get("contents") or document.get("text") or "")
    return "sha256:" + hashlib.sha256(contents.encode("utf-8")).hexdigest()


def fuse(
    bm25: list[dict[str, Any]],
    e5: list[dict[str, Any]],
    *,
    topk: int,
    rrf_constant: float,
) -> list[dict[str, Any]]:
    scores: defaultdict[str, float] = defaultdict(float)
    payloads: dict[str, dict[str, Any]] = {}
    routes: defaultdict[str, set[str]] = defaultdict(set)
    for backend, ranking in (("bm25", bm25), ("e5", e5)):
        for rank, item in enumerate(ranking, start=1):
            key = document_key(item)
            scores[key] += 1.0 / (rrf_constant + rank)
            payloads.setdefault(key, item.get("document", item))
            routes[key].add(backend)
    ordered = sorted(scores, key=lambda key: (-scores[key], key))[:topk]
    return [
        {
            "document": payloads[key],
            "score": float(scores[key]),
            "sources": sorted(routes[key]),
        }
        for key in ordered
    ]


def create_app(
    *,
    bm25_url: str,
    e5_url: str,
    upstream_topk: int,
    default_topk: int,
    rrf_constant: float,
    timeout: float,
) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    def health() -> dict[str, Any]:
        upstreams = {}
        for name, url in (("bm25", bm25_url), ("e5", e5_url)):
            try:
                response = requests.get(url.rsplit("/", 1)[0] + "/health", timeout=10)
                response.raise_for_status()
                upstreams[name] = response.json()
            except Exception as exc:  # pragma: no cover - requires live services
                raise HTTPException(status_code=503, detail=f"{name} unavailable: {exc}") from exc
        return {
            "status": "ok",
            "backend": "hybrid-rrf",
            "rrf_constant": rrf_constant,
            "upstream_topk": upstream_topk,
            "upstreams": upstreams,
        }

    @app.post("/retrieve")
    def retrieve(request: QueryRequest) -> dict[str, Any]:
        if not request.queries:
            return {"result": []}
        final_topk = int(request.topk or default_topk)
        candidate_topk = max(upstream_topk, final_topk)
        try:
            bm25_results = post_batch(bm25_url, request.queries, candidate_topk, timeout)
            e5_results = post_batch(e5_url, request.queries, candidate_topk, timeout)
        except Exception as exc:  # pragma: no cover - requires live services
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {
            "result": [
                fuse(bm25, e5, topk=final_topk, rrf_constant=rrf_constant)
                for bm25, e5 in zip(bm25_results, e5_results)
            ]
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bm25-url", default="http://127.0.0.1:8101/retrieve")
    parser.add_argument("--e5-url", default="http://127.0.0.1:8102/retrieve")
    parser.add_argument("--upstream-topk", type=int, default=100)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--rrf-constant", type=float, default=60.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--port", type=int, default=8300)
    args = parser.parse_args()
    app = create_app(
        bm25_url=args.bm25_url,
        e5_url=args.e5_url,
        upstream_topk=args.upstream_topk,
        default_topk=args.topk,
        rrf_constant=args.rrf_constant,
        timeout=args.timeout,
    )
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
