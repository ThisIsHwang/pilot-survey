from __future__ import annotations

import argparse
import json
import threading
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class QueryRequest(BaseModel):
    queries: list[str]
    topk: int | None = None
    return_scores: bool = True
    backend_ids: list[str] | None = None


def post_batch(url: str, queries: list[str], topk: int, timeout: float) -> list[list[dict[str, Any]]]:
    response = requests.post(
        url,
        json={"queries": queries, "topk": topk, "return_scores": True},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    results = payload.get("result")
    if not isinstance(results, list) or len(results) != len(queries):
        raise RuntimeError(
            f"Upstream retriever returned {len(results) if isinstance(results, list) else type(results)} "
            f"results for {len(queries)} queries"
        )
    return results


def create_app(
    *,
    bm25_url: str,
    e5_url: str,
    default_topk: int,
    timeout: float,
    assignment_log: Path | None,
) -> FastAPI:
    app = FastAPI()
    upstreams = {"bm25": bm25_url, "e5": e5_url}
    counters: Counter[str] = Counter()
    lock = threading.Lock()

    def record(routes: list[str], queries: list[str]) -> None:
        with lock:
            counters.update(routes)
            if assignment_log is None:
                return
            assignment_log.parent.mkdir(parents=True, exist_ok=True)
            with assignment_log.open("a", encoding="utf-8") as handle:
                for route, query in zip(routes, queries):
                    handle.write(json.dumps({"backend": route, "query": query}, ensure_ascii=False) + "\n")

    @app.get("/health")
    def health() -> dict[str, Any]:
        upstream_health: dict[str, Any] = {}
        for name, url in upstreams.items():
            health_url = url.rsplit("/", 1)[0] + "/health"
            try:
                response = requests.get(health_url, timeout=10)
                response.raise_for_status()
                upstream_health[name] = response.json()
            except Exception as exc:  # pragma: no cover - requires live services
                raise HTTPException(status_code=503, detail=f"{name} upstream unavailable: {exc}") from exc
        return {
            "status": "ok",
            "backend": "mixed",
            "routing": "episode-stable-explicit",
            "upstreams": upstream_health,
            "counts": dict(counters),
        }

    @app.get("/stats")
    def stats() -> dict[str, Any]:
        return {"counts": dict(counters), "total": sum(counters.values())}

    @app.post("/retrieve")
    def retrieve(request: QueryRequest) -> dict[str, Any]:
        if not request.queries:
            return {"result": [], "routes": []}
        if request.backend_ids is None:
            raise HTTPException(
                status_code=422,
                detail="mixed retriever requires one episode-stable backend_id per query",
            )
        if len(request.backend_ids) != len(request.queries):
            raise HTTPException(
                status_code=422,
                detail="backend_ids and queries must have equal length",
            )
        routes = [str(value).strip().lower() for value in request.backend_ids]
        invalid = sorted(set(routes) - set(upstreams))
        if invalid:
            raise HTTPException(status_code=422, detail=f"unsupported backend_ids: {invalid}")

        topk = int(request.topk or default_topk)
        grouped_indices: dict[str, list[int]] = defaultdict(list)
        for index, route in enumerate(routes):
            grouped_indices[route].append(index)

        ordered: list[list[dict[str, Any]] | None] = [None] * len(request.queries)
        try:
            for route, indices in grouped_indices.items():
                route_queries = [request.queries[index] for index in indices]
                route_results = post_batch(upstreams[route], route_queries, topk, timeout)
                for index, result in zip(indices, route_results):
                    ordered[index] = result
        except Exception as exc:  # pragma: no cover - requires live services
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        if any(result is None for result in ordered):
            raise HTTPException(status_code=500, detail="router failed to restore batch order")
        record(routes, request.queries)
        return {"result": ordered, "routes": routes}

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bm25-url", default="http://127.0.0.1:8101/retrieve")
    parser.add_argument("--e5-url", default="http://127.0.0.1:8102/retrieve")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--port", type=int, default=8200)
    parser.add_argument("--assignment-log")
    args = parser.parse_args()
    app = create_app(
        bm25_url=args.bm25_url,
        e5_url=args.e5_url,
        default_topk=args.topk,
        timeout=args.timeout,
        assignment_log=Path(args.assignment_log).resolve() if args.assignment_log else None,
    )
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
