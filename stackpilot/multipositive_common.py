from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from stackpilot.query_attribution_common import (
    candidate_valid,
    class_members,
    factual_candidate,
    query_jaccard,
    strict_class_ids,
)

SCHEMA = 1


def load_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or int(payload.get("schema", -1)) != SCHEMA:
        raise ValueError(f"Unsupported multipositive config: {path}")
    return payload


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def signature(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def stable_hash(*parts: object, length: int = 24) -> str:
    return hashlib.sha256("\n".join(map(str, parts)).encode("utf-8")).hexdigest()[:length]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: str | Path, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_json(path: str | Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_write_text(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def valid_candidates(state: dict[str, Any], *, excluded_style: str | None = None) -> list[dict[str, Any]]:
    output = []
    for row in state["candidates"]:
        if not candidate_valid(row):
            continue
        if excluded_style and str(row.get("style", "")) == excluded_style:
            continue
        output.append(dict(row))
    return output


def direct_candidates(state: dict[str, Any], *, excluded_style: str | None = None) -> list[dict[str, Any]]:
    return [row for row in valid_candidates(state, excluded_style=excluded_style) if int(row.get("direct", 0)) == 1]


def partner_by_utility(factual: dict[str, Any], rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("No partner candidate")
    return dict(max(rows, key=lambda row: (
        float(row.get("immediate_support_gain", 0.0)),
        float(row.get("final_support_recall", 0.0)),
        float(row.get("answer_em", 0.0)),
        -query_jaccard(str(factual["query"]), str(row["query"])),
        str(row["candidate_id"]),
    )))


def partner_by_diversity(factual: dict[str, Any], rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("No diversity partner")
    return dict(min(rows, key=lambda row: (
        query_jaccard(str(factual["query"]), str(row["query"])),
        abs(len(str(factual["query"]).split()) - len(str(row["query"]).split())),
        str(row["candidate_id"]),
    )))


def select_pair(
    state: dict[str, Any],
    *,
    selector: str,
    seed: int,
    excluded_style: str | None,
    maximum_random_query_jaccard: float,
) -> list[dict[str, Any]]:
    factual = factual_candidate(state)
    candidates = [row for row in valid_candidates(state, excluded_style=excluded_style) if str(row["candidate_id"]) != str(factual["candidate_id"])]
    if selector == "factual_replicated":
        return [dict(factual), {**dict(factual), "candidate_id": f"{factual['candidate_id']}::replica"}]
    if selector == "all_direct":
        pool = [row for row in candidates if int(row.get("direct", 0)) == 1]
        return [dict(factual), partner_by_utility(factual, pool)]
    if selector == "strict":
        strict_ids = strict_class_ids(state)
        pool = [row for row in candidates if str(row["candidate_id"]) in strict_ids]
        return [dict(factual), partner_by_diversity(factual, pool)]
    if selector == "random":
        pool = [row for row in candidates if query_jaccard(str(factual["query"]), str(row["query"])) <= maximum_random_query_jaccard]
        if not pool:
            raise RuntimeError("No random partner")
        index = int(stable_hash(seed, state["state_id"], excluded_style or "none", length=8), 16) % len(pool)
        return [dict(factual), dict(sorted(pool, key=lambda row: str(row["candidate_id"]))[index])]
    if selector == "diversity":
        return [dict(factual), partner_by_diversity(factual, candidates)]
    raise ValueError(f"Unknown selector {selector}")


def state_supports(
    state: dict[str, Any],
    *,
    selectors: Sequence[str],
    excluded_style: str | None,
    maximum_random_query_jaccard: float,
) -> bool:
    try:
        for selector in selectors:
            select_pair(
                state,
                selector=selector,
                seed=13,
                excluded_style=excluded_style,
                maximum_random_query_jaccard=maximum_random_query_jaccard,
            )
        return True
    except (RuntimeError, ValueError):
        return False


def target_payload(row: dict[str, Any], *, weight: float) -> dict[str, Any]:
    return {
        "target_id": str(row["candidate_id"]),
        "text": str(row["query"]),
        "weight": float(weight),
        "style": str(row.get("style", "unknown")),
        "origin": str(row.get("origin", "unknown")),
        "direct": int(row.get("direct", 0)),
        "factual": int(row.get("factual", 0) or row.get("origin") == "factual"),
        "immediate_support_gain": float(row.get("immediate_support_gain", 0.0)),
        "final_support_recall": float(row.get("final_support_recall", 0.0)),
        "answer_em": float(row.get("answer_em", 0.0)),
    }


def external_query_map(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    source = Path(path)
    if not source.is_file():
        raise RuntimeError(f"External query file does not exist: {source}")
    output: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(source):
        state_id = str(row["state_id"])
        query = str(row["query"]).strip()
        if not query or state_id in output:
            raise RuntimeError(f"Invalid or duplicate external query for {state_id}")
        output[state_id] = dict(row)
    return output


def hierarchical_bootstrap(
    rows: Sequence[dict[str, Any]],
    *,
    value_key: str,
    samples: int,
    random_seed: int,
) -> dict[str, float]:
    if not rows:
        raise RuntimeError("Cannot bootstrap an empty row set")
    cells: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = float(row[value_key])
        if not math.isfinite(value):
            raise RuntimeError(f"Non-finite {value_key}: {value}")
        cells[(str(row["direction"]), int(row["seed"]))].append(dict(row))
    directions = sorted({key[0] for key in cells})
    seeds = {direction: sorted(key[1] for key in cells if key[0] == direction) for direction in directions}
    estimate = float(np.mean([
        np.mean([np.mean([float(row[value_key]) for row in cells[(direction, seed)]]) for seed in seeds[direction]])
        for direction in directions
    ]))
    generator = np.random.default_rng(random_seed)
    draws = np.empty(samples, dtype=np.float64)
    for draw_index in range(samples):
        sampled_directions = generator.choice(directions, size=len(directions), replace=True)
        direction_means = []
        for direction in sampled_directions:
            local_seeds = seeds[str(direction)]
            sampled_seeds = generator.choice(local_seeds, size=len(local_seeds), replace=True)
            seed_means = []
            for seed in sampled_seeds:
                by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for row in cells[(str(direction), int(seed))]:
                    by_question[str(row["question_id"])].append(row)
                question_ids = sorted(by_question)
                sampled_questions = generator.choice(question_ids, size=len(question_ids), replace=True)
                seed_means.append(float(np.mean([
                    np.mean([float(row[value_key]) for row in by_question[str(question_id)]])
                    for question_id in sampled_questions
                ])))
            direction_means.append(float(np.mean(seed_means)))
        draws[draw_index] = float(np.mean(direction_means))
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "estimate": estimate,
        "ci_low": float(low),
        "ci_high": float(high),
        "n_directions": float(len(directions)),
        "n_rows": float(len(rows)),
    }
