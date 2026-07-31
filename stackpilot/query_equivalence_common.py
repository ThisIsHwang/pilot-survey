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

import yaml

SCHEMA = 1
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "how", "in", "is", "it", "of", "on",
    "or", "that", "the", "this", "to", "was", "were", "what", "when", "where",
    "which", "who", "whom", "whose", "why", "with",
}


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or int(payload.get("schema", -1)) != SCHEMA:
        raise ValueError(f"Unsupported query-equivalence config: {config_path}")
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
    atomic_write_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
    )


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


def discover_paths(patterns: Sequence[str]) -> list[Path]:
    import glob

    paths: dict[str, Path] = {}
    for pattern in patterns:
        for raw in glob.glob(os.path.expanduser(pattern), recursive=True):
            path = Path(raw).resolve()
            if path.is_file():
                paths[str(path)] = path
    return [paths[key] for key in sorted(paths)]


def normalize_title(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def word_tokens(value: str, *, content_only: bool = False) -> list[str]:
    tokens = [token.lower() for token in TOKEN_RE.findall(str(value))]
    if content_only:
        return [token for token in tokens if token not in STOPWORDS and len(token) > 1]
    return tokens


def token_set(value: str, *, content_only: bool = False) -> set[str]:
    return set(word_tokens(value, content_only=content_only))


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    return 1.0 if not union else len(left_set & right_set) / len(union)


def support_set(titles: Iterable[str], gold_titles: Sequence[str]) -> tuple[str, ...]:
    gold = {normalize_title(value) for value in gold_titles if str(value).strip()}
    observed = {normalize_title(value) for value in titles if str(value).strip()}
    return tuple(sorted(gold & observed))


def final_observed_titles(candidate: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for record in candidate.get("branch_turns", []):
        if isinstance(record, dict):
            raw = record.get("observed_titles", [])
            if isinstance(raw, list):
                values.extend(str(value) for value in raw if str(value).strip())
    if not values:
        raw = candidate.get("intervention_observed_titles", [])
        if isinstance(raw, list):
            values.extend(str(value) for value in raw if str(value).strip())
    return values


def query_jaccard(left: str, right: str) -> float:
    return jaccard(token_set(left, content_only=True), token_set(right, content_only=True))


def candidate_record(
    candidate: dict[str, Any],
    *,
    gold_titles: Sequence[str],
    epsilon: float,
) -> dict[str, Any]:
    immediate_titles = candidate.get("intervention_observed_titles", [])
    if not isinstance(immediate_titles, list):
        immediate_titles = []
    immediate_support = support_set(immediate_titles, gold_titles)
    final_support = support_set(final_observed_titles(candidate), gold_titles)
    query = str(candidate.get("query", "")).strip()
    if not query:
        raise RuntimeError("Causal-query candidate has an empty query")
    record = {
        "candidate_id": str(candidate.get("candidate_id") or stable_hash(query)),
        "query": query,
        "style": str(candidate.get("style", "unknown")),
        "origin": str(candidate.get("origin", "unknown")),
        "immediate_support_gain": float(candidate.get("immediate_support_gain", 0.0)),
        "immediate_support_set": list(immediate_support),
        "final_support_set": list(final_support),
        "final_support_recall": float(candidate.get("final_support_recall", 0.0)),
        "answer_em": float(candidate.get("answer_em", 0.0)),
        "answer_f1": float(candidate.get("answer_f1", 0.0)),
        "total_search_count": int(candidate.get("total_search_count", 0)),
        "suffix_search_count": int(candidate.get("suffix_search_count", 0)),
        "protocol_failure": int(candidate.get("protocol_failure", 0)),
        "support_tqe": float(candidate.get("support_tqe", 0.0)),
        "composite_tqe": float(candidate.get("composite_tqe", 0.0)),
        "direct": int(float(candidate.get("immediate_support_gain", 0.0)) > epsilon),
    }
    for name in (
        "immediate_support_gain",
        "final_support_recall",
        "answer_em",
        "answer_f1",
        "support_tqe",
        "composite_tqe",
    ):
        if not math.isfinite(float(record[name])):
            raise RuntimeError(f"Non-finite candidate field {name}: {record[name]!r}")
    return record


def evidence_signature(candidate: dict[str, Any], *, include_answer: bool) -> tuple[Any, ...]:
    value: list[Any] = [
        tuple(candidate["immediate_support_set"]),
        tuple(candidate["final_support_set"]),
    ]
    if include_answer:
        value.append(int(float(candidate["answer_em"]) > 0.5))
    return tuple(value)


def group_equivalence_classes(
    candidates: Sequence[dict[str, Any]],
    *,
    include_answer: bool,
) -> list[list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[evidence_signature(candidate, include_answer=include_answer)].append(
            dict(candidate)
        )
    classes = list(grouped.values())
    classes.sort(
        key=lambda rows: (
            -max(float(row["final_support_recall"]) for row in rows),
            -max(float(row["answer_em"]) for row in rows),
            min(float(row["total_search_count"]) for row in rows),
            -len(rows),
            min(str(row["candidate_id"]) for row in rows),
        )
    )
    return classes


def class_is_nontrivial(
    rows: Sequence[dict[str, Any]],
    *,
    maximum_query_jaccard: float,
) -> bool:
    if len(rows) < 2:
        return False
    for index, left in enumerate(rows):
        for right in rows[index + 1 :]:
            if query_jaccard(str(left["query"]), str(right["query"])) <= maximum_query_jaccard:
                return True
    return False


def build_prompt(result: dict[str, Any]) -> str:
    state = result.get("state", {})
    prefix = result.get("prefix", {})
    lines = [
        f"Question: {state['question']}",
        "Search history:",
    ]
    for record in prefix.get("records", []):
        lines.append(f"- Query {record.get('turn')}: {record.get('query', '')}")
        titles = record.get("observed_titles", [])
        if isinstance(titles, list) and titles:
            lines.append("  Observed titles: " + " | ".join(map(str, titles[:10])))
    lines.extend(
        [
            "Generate the next search query that best advances the unresolved information need.",
            "Return only the query, without XML tags or explanation.",
        ]
    )
    return "\n".join(lines)


def inspect_equivalence_state(
    result: dict[str, Any],
    *,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    state = result.get("state")
    candidates = result.get("candidates")
    if not isinstance(state, dict) or not isinstance(candidates, list):
        raise RuntimeError("Causal-query state result has no state/candidates payload")
    eq_cfg = cfg["equivalence"]
    epsilon = float(eq_cfg["epsilon"])
    records = [
        candidate_record(candidate, gold_titles=state["support_titles"], epsilon=epsilon)
        for candidate in candidates
        if isinstance(candidate, dict)
    ]
    valid = [row for row in records if int(row["protocol_failure"]) == 0]
    direct = [row for row in valid if int(row["direct"]) == 1]
    classes = group_equivalence_classes(
        direct,
        include_answer=bool(eq_cfg.get("include_answer_in_signature", True)),
    ) if direct else []
    best_class = classes[0] if classes else []
    factual = next((row for row in direct if row["origin"] == "factual"), None)
    factual_in_best = bool(
        factual is not None
        and any(row["candidate_id"] == factual["candidate_id"] for row in best_class)
    )
    nontrivial = class_is_nontrivial(
        best_class,
        maximum_query_jaccard=float(eq_cfg["maximum_nontrivial_query_jaccard"]),
    ) if best_class else False
    best_ids = {row["candidate_id"] for row in best_class}
    for row in records:
        row["best_class_member"] = int(row["candidate_id"] in best_ids)
        row["factual"] = int(row["origin"] == "factual")
    class_pair_jaccards = [
        query_jaccard(str(left["query"]), str(right["query"]))
        for index, left in enumerate(best_class)
        for right in best_class[index + 1 :]
    ]
    eligible = bool(
        len(best_class) >= int(eq_cfg["minimum_class_size"])
        and (not bool(eq_cfg.get("require_nontrivial_class", True)) or nontrivial)
        and (not bool(eq_cfg.get("require_factual_in_best_class", True)) or factual_in_best)
    )
    return {
        "schema": SCHEMA,
        "state_id": str(state["state_id"]),
        "question_id": str(state["question_id"]),
        "question": str(state["question"]),
        "dataset": str(state["dataset"]),
        "backend": str(state["backend"]),
        "topk": int(state["topk"]),
        "source_turn": int(state["source_turn"]),
        "policy_tag": str(state.get("policy_tag", "unknown")),
        "policy_seed": int(state.get("policy_seed", 0)),
        "prompt": build_prompt(result),
        "best_class_ids": sorted(best_ids),
        "best_class_size": len(best_class),
        "best_class_style_count": len({row["style"] for row in best_class}),
        "best_class_min_query_jaccard": min(class_pair_jaccards) if class_pair_jaccards else 1.0,
        "factual_in_best_class": int(factual_in_best),
        "nontrivial_best_class": int(nontrivial),
        "eligible": int(eligible),
        "direct_candidate_count": len(direct),
        "candidate_count": len(records),
        "candidates": records,
        "source_run_signature": str(result.get("run_signature", "")),
        "source_state_signature": str(result.get("state_signature", "")),
    }


def build_equivalence_state(
    result: dict[str, Any],
    *,
    cfg: dict[str, Any],
) -> dict[str, Any] | None:
    inspected = inspect_equivalence_state(result, cfg=cfg)
    return inspected if int(inspected["eligible"]) == 1 else None


def split_value(question_id: str, seed: int) -> float:
    raw = int(stable_hash(seed, question_id, length=16), 16)
    return raw / float(16**16 - 1)


def deterministic_order(rows: Sequence[dict[str, Any]], *parts: object) -> list[dict[str, Any]]:
    return sorted(
        (dict(row) for row in rows),
        key=lambda row: stable_hash(*parts, row["question_id"], row["state_id"]),
    )
