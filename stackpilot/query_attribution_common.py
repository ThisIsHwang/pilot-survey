from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

SCHEMA = 2
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")
STOPWORDS = {"a","an","and","are","as","at","be","by","did","do","does","for","from","had","has","have","how","in","is","it","of","on","or","that","the","this","to","was","were","what","when","where","which","who","whom","whose","why","with"}


def load_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or int(payload.get("schema", -1)) != SCHEMA:
        raise ValueError(f"Unsupported query-attribution config: {path}")
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
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def word_tokens(value: str, *, content_only: bool = False) -> list[str]:
    tokens = [token.lower() for token in TOKEN_RE.findall(str(value))]
    return [token for token in tokens if token not in STOPWORDS and len(token) > 1] if content_only else tokens


def token_set(value: str, *, content_only: bool = False) -> set[str]:
    return set(word_tokens(value, content_only=content_only))


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    return 1.0 if not union else len(left_set & right_set) / len(union)


def query_jaccard(left: str, right: str) -> float:
    return jaccard(token_set(left, content_only=True), token_set(right, content_only=True))


def candidate_valid(row: dict[str, Any]) -> bool:
    return int(row.get("protocol_failure", 0)) == 0 and bool(str(row.get("query", "")).strip())


def factual_candidate(state: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in state["candidates"] if candidate_valid(row) and (int(row.get("factual", 0)) == 1 or row.get("origin") == "factual")]
    if len(rows) != 1:
        raise RuntimeError(f"State {state['state_id']} needs one factual candidate; found {len(rows)}")
    return dict(rows[0])


def candidate_key(row: dict[str, Any], definition: str) -> tuple[Any, ...]:
    immediate = tuple(sorted(str(value) for value in row.get("immediate_support_set", [])))
    final = tuple(sorted(str(value) for value in row.get("final_support_set", [])))
    answer = int(float(row.get("answer_em", 0.0)) > 0.5)
    if definition == "strict":
        return immediate, final, answer
    if definition == "immediate":
        return (immediate,)
    if definition == "final":
        return final, answer
    raise ValueError(f"Unknown class definition {definition}")


def class_members(state: dict[str, Any], definition: str) -> list[dict[str, Any]]:
    factual = factual_candidate(state)
    key = candidate_key(factual, definition)
    return sorted([dict(row) for row in state["candidates"] if candidate_valid(row) and int(row.get("direct", 0)) == 1 and candidate_key(row, definition) == key], key=lambda row: str(row["candidate_id"]))


def strict_class_ids(state: dict[str, Any]) -> set[str]:
    return {str(row["candidate_id"]) for row in class_members(state, "strict")}


def outside_strict_valid(state: dict[str, Any]) -> list[dict[str, Any]]:
    strict_ids = strict_class_ids(state)
    factual_id = str(factual_candidate(state)["candidate_id"])
    return sorted([dict(row) for row in state["candidates"] if candidate_valid(row) and str(row["candidate_id"]) not in strict_ids and str(row["candidate_id"]) != factual_id], key=lambda row: str(row["candidate_id"]))


def outside_strict_direct(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in outside_strict_valid(state) if int(row.get("direct", 0)) == 1]


def partner_by_distinctness(factual: dict[str, Any], rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("No candidate partner available")
    return dict(min(rows, key=lambda row: (query_jaccard(str(factual["query"]), str(row["query"])), abs(len(word_tokens(str(factual["query"]))) - len(word_tokens(str(row["query"])))), str(row["candidate_id"]))))


def select_targets(state: dict[str, Any], *, selector: str, seed: int, maximum_random_query_jaccard: float) -> list[dict[str, Any]]:
    factual = factual_candidate(state)
    strict = [row for row in class_members(state, "strict") if row["candidate_id"] != factual["candidate_id"]]
    strict_partner = partner_by_distinctness(factual, strict) if strict else None
    if selector == "factual_replicated":
        return [dict(factual), {**dict(factual), "candidate_id": f"{factual['candidate_id']}::replica"}]
    if selector == "strict":
        if strict_partner is None:
            raise RuntimeError(f"State {state['state_id']} has no strict partner")
        return [dict(factual), strict_partner]
    if selector == "random_outside_strict":
        pool = [row for row in outside_strict_valid(state) if query_jaccard(str(factual["query"]), str(row["query"])) <= maximum_random_query_jaccard]
        if not pool:
            raise RuntimeError(f"State {state['state_id']} has no random outside-strict partner")
        index = int(stable_hash(seed, state["state_id"], selector, length=8), 16) % len(pool)
        return [dict(factual), dict(pool[index])]
    if selector == "direct_outside_strict":
        pool = outside_strict_direct(state)
        if not pool:
            raise RuntimeError(f"State {state['state_id']} has no direct outside-strict partner")
        partner = max(pool, key=lambda row: (float(row.get("final_support_recall", 0.0)), float(row.get("answer_em", 0.0)), -int(row.get("total_search_count", 0)), -query_jaccard(str(factual["query"]), str(row["query"]))))
        return [dict(factual), dict(partner)]
    if selector == "diversity_matched_outside_strict":
        if strict_partner is None:
            raise RuntimeError(f"State {state['state_id']} has no strict partner")
        target_jaccard = query_jaccard(str(factual["query"]), str(strict_partner["query"]))
        target_length = len(word_tokens(str(strict_partner["query"])))
        pool = outside_strict_valid(state)
        if not pool:
            raise RuntimeError(f"State {state['state_id']} has no diversity-matched pool")
        partner = min(pool, key=lambda row: (abs(query_jaccard(str(factual["query"]), str(row["query"])) - target_jaccard), abs(len(word_tokens(str(row["query"]))) - target_length), str(row["candidate_id"])))
        return [dict(factual), dict(partner)]
    if selector == "immediate_only":
        strict_ids = strict_class_ids(state)
        pool = [row for row in class_members(state, "immediate") if row["candidate_id"] != factual["candidate_id"] and str(row["candidate_id"]) not in strict_ids]
        return [dict(factual), partner_by_distinctness(factual, pool)]
    if selector == "final_only":
        strict_ids = strict_class_ids(state)
        pool = [row for row in class_members(state, "final") if row["candidate_id"] != factual["candidate_id"] and str(row["candidate_id"]) not in strict_ids]
        return [dict(factual), partner_by_distinctness(factual, pool)]
    raise ValueError(f"Unknown selector {selector}")


def target_token_total(tokenizer: Any, rows: Sequence[dict[str, Any]]) -> int:
    return sum(len(tokenizer.encode(str(row["query"]).strip() + (tokenizer.eos_token or ""), add_special_tokens=False)) for row in rows)


def relative_imbalance(values: Sequence[int]) -> float:
    return float("inf") if not values or min(values) <= 0 else (max(values) - min(values)) / min(values)


def deterministic_order(rows: Sequence[dict[str, Any]], *parts: object) -> list[dict[str, Any]]:
    return sorted((dict(row) for row in rows), key=lambda row: stable_hash(*parts, row["question_id"], row["state_id"]))


def balanced_sample(rows: Sequence[dict[str, Any]], count: int, *, seed: int) -> list[dict[str, Any]]:
    if count > len(rows):
        raise RuntimeError(f"Requested {count} states from pool of {len(rows)}")
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row.get("dataset", "")), int(row.get("source_turn", 0)), str(row.get("policy_tag", "")), int(row.get("policy_seed", 0)))
        buckets[key].append(dict(row))
    for key in buckets:
        buckets[key] = deterministic_order(buckets[key], seed, key)
    active = sorted(buckets, key=repr)
    selected = []
    cursor = 0
    while active and len(selected) < count:
        key = active[cursor % len(active)]
        selected.append(buckets[key].pop(0))
        if not buckets[key]:
            active.remove(key)
            if active:
                cursor %= len(active)
        else:
            cursor += 1
    if len(selected) != count:
        raise RuntimeError(f"Only selected {len(selected)}/{count} states")
    return selected


def hierarchical_bootstrap(rows: Sequence[dict[str, Any]], *, value_key: str, samples: int, random_seed: int) -> dict[str, float]:
    if not rows:
        raise RuntimeError("Cannot bootstrap an empty row set")
    by_direction_seed: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = float(row[value_key])
        if not math.isfinite(value):
            raise RuntimeError(f"Non-finite {value_key}: {value}")
        by_direction_seed[(str(row["direction"]), int(row["seed"]))].append(dict(row))
    directions = sorted({key[0] for key in by_direction_seed})
    seeds_by_direction = {direction: sorted(seed for (row_direction, seed) in by_direction_seed if row_direction == direction) for direction in directions}
    estimate = float(np.mean([np.mean([np.mean([float(row[value_key]) for row in by_direction_seed[(direction, seed)]]) for seed in seeds_by_direction[direction]]) for direction in directions]))
    generator = np.random.default_rng(random_seed)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled_directions = generator.choice(directions, size=len(directions), replace=True)
        direction_means = []
        for direction in sampled_directions:
            seeds = seeds_by_direction[str(direction)]
            sampled_seeds = generator.choice(seeds, size=len(seeds), replace=True)
            seed_means = []
            for seed in sampled_seeds:
                cluster_rows = by_direction_seed[(str(direction), int(seed))]
                by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for row in cluster_rows:
                    by_question[str(row["question_id"])].append(row)
                questions = sorted(by_question)
                sampled_questions = generator.choice(questions, size=len(questions), replace=True)
                seed_means.append(float(np.mean([np.mean([float(row[value_key]) for row in by_question[str(question)]]) for question in sampled_questions])))
            direction_means.append(float(np.mean(seed_means)))
        draws[index] = float(np.mean(direction_means))
    low, high = np.quantile(draws, [0.025, 0.975])
    return {"estimate": estimate, "ci_low": float(low), "ci_high": float(high), "n_directions": float(len(directions)), "n_rows": float(len(rows))}
