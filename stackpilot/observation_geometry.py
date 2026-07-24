from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RenderedObservation:
    full_text: str
    visible_text: str
    token_count: int
    truncated: bool
    retrieved_titles: tuple[str, ...]
    observed_titles: tuple[str, ...]


def _input_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    values = encoded["input_ids"]
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list):
        if len(values) != 1:
            raise RuntimeError(
                "Tokenizer returned more than one row for one observation"
            )
        values = values[0]
    return [int(value) for value in values]


def prefix_token_ids(tokenizer: Any, text: str, max_tokens: int) -> list[int]:
    if isinstance(max_tokens, bool) or max_tokens < 1:
        raise ValueError(
            f"Observation token budget must be positive; got {max_tokens!r}"
        )
    return _input_ids(tokenizer, text)[:max_tokens]


def tokenize_observation_batch(
    tokenizer: Any,
    texts: list[str],
    max_tokens: int,
) -> Any:
    """Tokenize and truncate each observation before batch padding.

    Search-R1 previously padded the full batch first and then sliced the tensor.
    With left padding, one unusually long peer could therefore remove another
    row's real prefix. Per-row truncation makes the 500-token contract invariant
    to batch composition.
    """

    rows = [prefix_token_ids(tokenizer, text, max_tokens) for text in texts]
    return tokenizer.pad(
        {"input_ids": rows},
        padding="longest",
        return_tensors="pt",
    )["input_ids"]


def _document_parts(item: dict[str, Any], rank: int) -> tuple[str, str, int]:
    document = item.get("document")
    if isinstance(document, dict):
        contents = str(document.get("contents") or "")
        title, separator, text = contents.partition("\n")
        if not separator:
            text = ""
        return title, text, rank
    title = str(item.get("title") or "")
    text = str(item.get("text") or item.get("content") or "")
    return title, text, rank


def render_retrieval_observation(
    results: list[dict[str, Any]],
    tokenizer: Any,
    max_tokens: int,
) -> RenderedObservation:
    passages: list[str] = []
    titles: list[str] = []
    header_end_offsets: list[int] = []
    body = ""
    prefix = "\n\n<information>"
    for rank, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            raise TypeError(f"Retriever result {rank} is not an object")
        title, text, display_rank = _document_parts(item, rank)
        if not title.strip():
            raise RuntimeError(f"Retriever result {rank} has an empty title")
        title = title.strip()
        text = text.strip()
        header = f"Doc {display_rank}(Title: {title})"
        separator = " " if text else ""
        passage = f"{header}{separator}{text}\n"
        header_end_offsets.append(len(prefix) + len(body) + len(header))
        passages.append(passage)
        titles.append(title)
        body += passage

    # This is byte-for-byte the wrapper used by the patched Search-R1 rollout.
    full_text = prefix + body.strip() + "</information>\n\n"
    all_ids = _input_ids(tokenizer, full_text)
    visible_ids = all_ids[:max_tokens]
    visible_text = tokenizer.decode(
        visible_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    observed = [
        title
        for title, end_offset in zip(titles, header_end_offsets, strict=True)
        if len(_input_ids(tokenizer, full_text[:end_offset])) <= max_tokens
    ]
    return RenderedObservation(
        full_text=full_text,
        visible_text=visible_text,
        token_count=len(all_ids),
        truncated=len(all_ids) > max_tokens,
        retrieved_titles=tuple(titles),
        observed_titles=tuple(observed),
    )
