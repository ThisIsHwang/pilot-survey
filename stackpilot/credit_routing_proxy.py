from __future__ import annotations

import argparse
import copy
import json
import threading
import time
from pathlib import Path
from typing import Any

import requests
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from stackpilot.credit_routing_common import (
    FEATURE_NAMES,
    aggregate_document_utility,
    atomic_write_text,
    score_artifact,
    selection_indices,
)


class QueryRequest(BaseModel):
    queries: list[str]
    topk: int | None = None
    return_scores: bool = False


def load_artifact(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "feature_names",
        "feature_mean",
        "feature_scale",
        "weights",
        "upstream_topk",
        "observation_k",
    }
    missing = required - set(payload)
    if missing:
        raise RuntimeError(f"Utility artifact misses {sorted(missing)}")
    return payload


def post_upstream(url: str, queries: list[str], topk: int, timeout: int) -> list[list[dict[str, Any]]]:
    response = requests.post(
        url,
        json={"queries": queries, "topk": int(topk), "return_scores": True},
        timeout=timeout,
    )
    response.raise_for_status()
    batches = response.json().get("result")
    if not isinstance(batches, list) or len(batches) != len(queries):
        raise RuntimeError("Credit-routing upstream returned an invalid batch")
    return batches


def attach_metadata(
    item: dict[str, Any],
    *,
    predicted_utility: float,
    action_utility: float,
    upstream_rank: int,
    observation_route: str,
) -> dict[str, Any]:
    output = copy.deepcopy(item)
    output["stackpilot_predicted_document_utility"] = float(predicted_utility)
    output["stackpilot_action_utility"] = float(action_utility)
    output["stackpilot_upstream_rank"] = int(upstream_rank)
    output["stackpilot_observation_route"] = str(observation_route)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-budget credit-routing retrieval proxy.")
    parser.add_argument("--backend", choices=("bm25", "e5"), required=True)
    parser.add_argument("--upstream-url", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--observation-route", choices=("rank", "utility"), required=True)
    parser.add_argument("--upstream-topk", type=int, default=8)
    parser.add_argument("--output-topk", type=int, default=3)
    parser.add_argument("--action-aggregation", choices=("mean-topk", "max", "mean", "positive-sum"), default="mean-topk")
    parser.add_argument("--action-aggregation-k", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--log", default="")
    args = parser.parse_args()
    if args.output_topk > args.upstream_topk:
        parser.error("output-topk cannot exceed upstream-topk")
    artifact = load_artifact(args.artifact)
    if list(artifact["feature_names"]) != list(FEATURE_NAMES):
        parser.error("artifact feature schema does not match the runtime scorer")
    if int(artifact["upstream_topk"]) != args.upstream_topk:
        parser.error("artifact and proxy upstream-topk disagree")
    if int(artifact["observation_k"]) != args.output_topk:
        parser.error("artifact and proxy output-topk disagree")

    app = FastAPI()
    log_lock = threading.Lock()
    upstream_health = args.upstream_url.rsplit("/", 1)[0] + "/health"

    @app.get("/health")
    def health() -> dict[str, Any]:
        response = requests.get(upstream_health, timeout=15)
        response.raise_for_status()
        payload = dict(response.json())
        payload.update(
            {
                "credit_routing_proxy": True,
                "credit_routing_observation_route": args.observation_route,
                "credit_routing_upstream_topk": args.upstream_topk,
                "credit_routing_output_topk": args.output_topk,
                "credit_routing_artifact": str(Path(args.artifact).resolve()),
            }
        )
        return payload

    @app.post("/retrieve")
    def retrieve(request: QueryRequest) -> dict[str, Any]:
        requested_topk = int(request.topk or args.output_topk)
        if requested_topk != args.output_topk:
            raise ValueError(
                f"Credit-routing proxy is pinned to top-k {args.output_topk}; received {requested_topk}"
            )
        batches = post_upstream(
            args.upstream_url,
            request.queries,
            args.upstream_topk,
            args.timeout,
        )
        output: list[list[dict[str, Any]]] = []
        logs: list[dict[str, Any]] = []
        for query, items in zip(request.queries, batches, strict=True):
            if len(items) < args.output_topk:
                raise RuntimeError(
                    f"Upstream returned {len(items)} items for query {query!r}"
                )
            scores = score_artifact(query, items, args.backend, artifact)
            action_utility = aggregate_document_utility(
                scores,
                k=args.action_aggregation_k,
                mode=args.action_aggregation,
            )
            indices = selection_indices(
                scores,
                args.output_topk,
                mode=args.observation_route,
            )
            selected = [
                attach_metadata(
                    items[index],
                    predicted_utility=float(scores[index]),
                    action_utility=action_utility,
                    upstream_rank=index + 1,
                    observation_route=args.observation_route,
                )
                for index in indices
            ]
            output.append(selected)
            logs.append(
                {
                    "timestamp": time.time(),
                    "backend": args.backend,
                    "query": query,
                    "observation_route": args.observation_route,
                    "upstream_topk": args.upstream_topk,
                    "output_topk": args.output_topk,
                    "selected_indices": indices,
                    "predicted_utilities": [float(value) for value in scores],
                    "action_utility": action_utility,
                }
            )
        if args.log:
            with log_lock:
                path = Path(args.log)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as handle:
                    for row in logs:
                        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        return {"result": output}

    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
