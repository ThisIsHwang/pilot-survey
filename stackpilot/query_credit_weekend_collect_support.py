from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any, Sequence

import requests

from stackpilot.action_protocol import parse_action
from stackpilot.causal_query_common import normalize_title
from stackpilot.causal_query_replay import _complete
from stackpilot.query_credit_common import (
    composite_reward,
    normalize_text,
    stable_hash,
    stable_seed,
)

SCHEMA = 2


def discover_inputs(cfg: dict[str, Any], provided: Sequence[str] | None) -> list[str]:
    patterns = list(provided or [])
    if not patterns:
        environment = os.environ.get("QUERY_CREDIT_INPUTS", "").strip()
        if environment:
            patterns = [value for value in environment.split(os.pathsep) if value]
        else:
            patterns = [str(value) for value in cfg["source"]["state_globs"]]
    paths: set[str] = set()
    for pattern in patterns:
        paths.update(
            str(Path(value).resolve())
            for value in glob.glob(os.path.expanduser(pattern), recursive=True)
            if Path(value).is_file()
        )
    if not paths:
        raise RuntimeError("No causal-query state files were found")
    return sorted(paths)


def _service_check(causal_cfg: dict[str, Any], backends: Sequence[str]) -> dict[str, Any]:
    identities: dict[str, Any] = {}
    for backend in backends:
        port = int(causal_cfg["retrieval"][f"{backend}_port"])
        response = requests.get(f"http://127.0.0.1:{port}/health", timeout=30)
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") != "ok" or payload.get("backend") != backend:
            raise RuntimeError(f"Unexpected {backend} health response: {payload}")
        identities[str(backend)] = payload
    models = requests.get(
        str(causal_cfg["model"]["api_base"]).rstrip("/") + "/models", timeout=30
    )
    models.raise_for_status()
    model_ids = {str(row.get("id")) for row in models.json().get("data", [])}
    expected = str(causal_cfg["model"]["served_model_name"])
    if expected not in model_ids:
        raise RuntimeError(f"vLLM does not serve {expected}; available={sorted(model_ids)}")
    identities["served_models"] = sorted(model_ids)
    return identities


def _reward_views(replay: dict[str, Any], cfg: dict[str, Any]) -> dict[str, float]:
    output = {
        "answer_f1": float(replay["answer_f1"]),
        "support_recall": float(replay["final_support_recall"]),
    }
    for name, weights in cfg["collection"]["reward_views"].items():
        output[str(name)] = composite_reward(
            support_recall=float(replay["final_support_recall"]),
            answer_f1=float(replay["answer_f1"]),
            search_count=float(replay["search_count"]),
            invalid_action_count=float(replay["invalid_action_count"]),
            weights={key: float(value) for key, value in weights.items()},
        )
    return output


def _serializable_documents(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for rank, item in enumerate(results, start=1):
        document = item.get("document") if isinstance(item, dict) else None
        if isinstance(document, dict):
            contents = str(document.get("contents") or "")
            title, separator, text = contents.partition("\n")
            if not separator:
                text = ""
            identifier = str(document.get("id") or document.get("document_id") or "")
            score = item.get("score", item.get("retrieval_score", 0.0))
        else:
            title = str(item.get("title") or "")
            text = str(item.get("text") or item.get("content") or "")
            identifier = str(item.get("id") or item.get("document_id") or "")
            score = item.get("score", item.get("retrieval_score", 0.0))
        try:
            numeric_score = float(score or 0.0)
        except (TypeError, ValueError):
            numeric_score = 0.0
        output.append(
            {
                "rank": rank,
                "id": identifier,
                "title": title.strip(),
                "text": text.strip(),
                "score": numeric_score,
            }
        )
    return output


def _compact_replay(replay: dict[str, Any], rewards: dict[str, float]) -> dict[str, Any]:
    return {
        "continuation_seed": int(replay["continuation_seed"]),
        "reward_views": rewards,
        "prediction": str(replay.get("prediction", "")),
        "answer_em": float(replay.get("answer_em", 0.0)),
        "answer_f1": float(replay.get("answer_f1", 0.0)),
        "final_support_recall": float(replay.get("final_support_recall", 0.0)),
        "search_count": int(replay.get("search_count", 0)),
        "invalid_action_count": int(replay.get("invalid_action_count", 0)),
        "observed_titles": list(replay.get("observed_titles", [])),
        "retrieved_titles": list(replay.get("retrieved_titles", [])),
        "observation_token_count": int(replay.get("observation_token_count", 0)),
        "observation_truncated": int(replay.get("observation_truncated", 0)),
        "records": list(replay.get("records", [])),
    }


def _candidate_bank(
    result: dict[str, Any],
    *,
    state: dict[str, Any],
    prefix: dict[str, Any],
    causal_cfg: dict[str, Any],
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    target = int(profile["candidates_per_state"])
    minimum = int(profile["minimum_candidates_per_state"])
    seen: set[str] = set()
    candidates: list[dict[str, Any]] = []

    def add(query: str, origin: str, source_seed: int | None = None) -> None:
        normalized = normalize_text(query)
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        candidates.append(
            {
                "query": str(query).strip(),
                "origin": origin,
                "source_seed": source_seed,
            }
        )

    add(str(state["factual_query"]), "factual")
    generation = cfg_generation = profile["sibling_generation"]
    for attempt in range(int(cfg_generation["attempts"])):
        if len(candidates) >= target:
            break
        generation_seed = stable_seed(
            "weekend-policy-sibling", state["state_id"], attempt
        )
        content = _complete(
            causal_cfg,
            list(prefix["messages"]),
            temperature=float(generation["temperature"]),
            max_tokens=int(generation["max_tokens"]),
            seed=generation_seed,
        )
        action, value = parse_action(content)
        if action == "search" and value:
            add(value, "direct-policy-sibling", generation_seed)

    # Controlled alternatives in the source artifact were generated by the
    # same model before outcomes were observed. They are a declared fallback,
    # never human-written or reward-selected.
    if len(candidates) < target and bool(profile.get("allow_controlled_fallback", True)):
        for row in result.get("candidates", []):
            if len(candidates) >= target:
                break
            if int(row.get("protocol_failure", 0)) != 0:
                continue
            add(str(row.get("query", "")), "controlled-same-model-fallback")

    if len(candidates) < minimum:
        raise RuntimeError(
            f"State {state['state_id']} has only {len(candidates)} policy/model-generated queries"
        )
    return candidates[:target]


def _corpus_probe(state: dict[str, Any], retriever: Any) -> list[dict[str, Any]]:
    rows = []
    for title in state.get("support_titles", []):
        results = retriever.search(str(title), 100)
        retrieved = []
        for item in results:
            document = item.get("document") if isinstance(item, dict) else None
            if isinstance(document, dict):
                contents = str(document.get("contents") or "")
                value = contents.split("\n", 1)[0]
            else:
                value = str(item.get("title", "")) if isinstance(item, dict) else ""
            retrieved.append(normalize_title(value))
        rows.append(
            {
                "support_title": str(title),
                "probe_backend": "bm25",
                "found_at_100": int(normalize_title(str(title)) in set(retrieved)),
            }
        )
    return rows


def _sensitivity_state_ids(
    selected: Sequence[dict[str, Any]],
    *,
    per_cell: int,
    salt: str,
) -> set[str]:
    grouped: dict[tuple[str, str], list[str]] = {}
    for result in selected:
        state = result["state"]
        key = (str(state["dataset"]).lower(), str(state["backend"]).lower())
        grouped.setdefault(key, []).append(str(state["state_id"]))
    chosen: set[str] = set()
    for key, state_ids in grouped.items():
        ordered = sorted(state_ids, key=lambda value: stable_hash(salt, *key, value))
        chosen.update(ordered[: int(per_cell)])
    return chosen
