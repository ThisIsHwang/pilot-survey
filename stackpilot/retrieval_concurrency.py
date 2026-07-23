from __future__ import annotations

from typing import Any


def batch_search(
    retriever: Any,
    queries: list[str],
    topk: int,
    search_lock: Any | None = None,
):
    """Run a retriever batch, optionally under its shared GPU-resource lock."""
    if search_lock is None:
        return retriever.batch_search(queries, topk, True)
    with search_lock:
        return retriever.batch_search(queries, topk, True)
