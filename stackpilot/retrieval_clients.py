from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import requests


def normalize_document(doc: dict[str, Any]) -> tuple[str, str]:
    metadata = doc.get("document_metadata") or {}
    title = str(doc.get("title") or metadata.get("title") or "").strip()
    contents = str(doc.get("contents") or "")
    text = str(doc.get("text") or doc.get("content") or contents)

    # Search-R1's wiki-18 dense corpus commonly stores only `contents` as
    # '"Wikipedia title"\npassage text'. Recover the title so BM25 and E5 use
    # the same supporting-title evaluation rather than silently scoring E5 as
    # title-less.
    if not title and contents:
        first_line, separator, remainder = contents.partition("\n")
        candidate = first_line.strip().strip('"').strip()
        if candidate:
            title = candidate
            if separator and remainder.strip():
                text = remainder.strip()
    return title, text


@dataclass
class RetrievalClient:
    name: str
    url: str
    timeout: int = 120
    retries: int = 4

    def search(self, query: str, topk: int) -> list[dict[str, Any]]:
        response = None
        for attempt in range(self.retries + 1):
            try:
                response = requests.post(
                    self.url,
                    json={"queries": [query], "topk": topk, "return_scores": True},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                break
            except requests.RequestException:
                if attempt >= self.retries:
                    raise
                time.sleep(min(8.0, 0.5 * (2**attempt)))
        assert response is not None
        payload = response.json()["result"][0]
        results = []
        for rank, item in enumerate(payload, start=1):
            if "document" in item:
                doc = item["document"]
                score = item.get("score")
            else:
                doc = item
                score = item.get("score")
            title, text = normalize_document(doc)
            results.append(
                {
                    "rank": rank,
                    "score": score,
                    "id": str(doc.get("id") or doc.get("document_id") or ""),
                    "title": title,
                    "text": text,
                }
            )
        return results
