from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import requests


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
            results.append(
                {
                    "rank": rank,
                    "score": score,
                    "id": str(doc.get("id") or doc.get("document_id") or ""),
                    "title": str(
                        doc.get("title")
                        or doc.get("document_metadata", {}).get("title")
                        or ""
                    ),
                    "text": str(
                        doc.get("text")
                        or doc.get("content")
                        or doc.get("contents")
                        or ""
                    ),
                }
            )
        return results
