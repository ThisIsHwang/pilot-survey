from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass


STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "who",
    "what",
    "when",
    "where",
    "why",
    "how",
    "which",
    "of",
    "in",
    "on",
    "at",
    "to",
    "for",
    "from",
    "with",
    "and",
    "or",
    "did",
    "does",
    "do",
    "has",
    "have",
    "had",
    "be",
    "been",
    "being",
    "that",
    "this",
    "these",
    "those",
    "by",
}


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", text)


def keyword_query(question: str) -> str:
    tokens = [w for w in words(question) if w.lower() not in STOPWORDS]
    return " ".join(tokens[:14]) or question


def exact_query(question: str) -> str:
    phrases = re.findall(r'"([^"]+)"', question)
    capitalized = re.findall(r"\b(?:[A-Z][\w'\-]*)(?:\s+[A-Z][\w'\-]*)*\b", question)
    candidates = []
    for p in phrases + capitalized:
        p = p.strip()
        if len(p) > 2 and p.lower() not in {
            "what",
            "who",
            "when",
            "where",
            "which",
            "how",
        }:
            candidates.append(p)
    quoted = " ".join(f'"{x}"' for x in list(dict.fromkeys(candidates))[:3])
    tail = keyword_query(question)
    return f"{quoted} {tail}".strip() if quoted else tail


def decompose_query(question: str) -> str:
    # Conservative heuristic: retain the main entity-rich segment and remove generic question framing.
    q = re.sub(r"^(who|what|when|where|which|how)\s+", "", question.strip(), flags=re.I)
    q = re.sub(r"\?$", "", q)
    return keyword_query(q)


def heuristic_candidates(question: str) -> dict[str, str]:
    return {
        "semantic": question.strip(),
        "keyword": keyword_query(question),
        "exact": exact_query(question),
        "decompose": decompose_query(question),
    }


@dataclass
class LLMQueryGenerator:
    api_base: str
    api_key: str
    model: str
    temperature: float = 0.1
    max_tokens: int = 300
    seed: int = 42

    def __post_init__(self) -> None:
        from openai import OpenAI

        self.client = OpenAI(
            base_url=self.api_base,
            api_key=self.api_key,
            timeout=180.0,
            max_retries=5,
        )

    def generate(self, question: str) -> dict[str, str]:
        prompt = f"""Create four search-engine queries for the question below.
Return strict JSON with exactly these keys: semantic, keyword, exact, decompose.
- semantic: a natural-language semantic query
- keyword: concise entity and relation keywords
- exact: quoted rare entities or phrases plus keywords
- decompose: the best first-hop subquestion for multi-hop retrieval
Question: {question}
"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=(
                self.seed
                + int.from_bytes(
                    hashlib.sha256(question.encode("utf-8")).digest()[:4], "big"
                )
            )
            % (2**31),
        )
        content = response.choices[0].message.content or "{}"
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, flags=re.S)
            try:
                parsed = json.loads(match.group(0)) if match else {}
            except json.JSONDecodeError:
                parsed = {}
        if not isinstance(parsed, dict):
            parsed = {}
        fallback = heuristic_candidates(question)
        return {key: str(parsed.get(key) or fallback[key]).strip() for key in fallback}
