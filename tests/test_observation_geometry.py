from __future__ import annotations

import numpy as np

from stackpilot.observation_geometry import (
    render_retrieval_observation,
    tokenize_observation_batch,
)


class CharacterTokenizer:
    padding_side = "left"
    pad_token_id = 0

    def __call__(self, text: str, *, add_special_tokens: bool = False, **kwargs):
        assert not add_special_tokens
        return {"input_ids": [ord(character) for character in text]}

    def decode(self, values, **kwargs) -> str:
        return "".join(chr(int(value)) for value in values if int(value) != 0)

    def pad(self, payload, *, padding: str, return_tensors: str):
        assert padding == "longest"
        assert return_tensors == "pt"
        rows = payload["input_ids"]
        width = max((len(row) for row in rows), default=0)
        padded = [[0] * (width - len(row)) + list(row) for row in rows]
        return {"input_ids": np.asarray(padded, dtype=np.int64)}


def result(rank: int, title: str, text: str) -> dict:
    return {"rank": rank, "title": title, "text": text}


def test_fixed_budget_is_independent_of_topk() -> None:
    tokenizer = CharacterTokenizer()
    top3 = [result(index, f"Title {index}", "x" * 200) for index in range(1, 4)]
    top10 = [result(index, f"Title {index}", "x" * 200) for index in range(1, 11)]

    rendered3 = render_retrieval_observation(top3, tokenizer, 500)
    rendered10 = render_retrieval_observation(top10, tokenizer, 500)

    assert len(rendered3.visible_text) <= 500
    assert len(rendered10.visible_text) <= 500
    assert rendered10.retrieved_titles == tuple(f"Title {i}" for i in range(1, 11))
    assert len(rendered10.observed_titles) < len(rendered10.retrieved_titles)


def test_short_bundle_observes_every_retrieved_title() -> None:
    rendered = render_retrieval_observation(
        [result(1, "First", "short"), result(2, "Second", "short")],
        CharacterTokenizer(),
        500,
    )

    assert not rendered.truncated
    assert rendered.observed_titles == rendered.retrieved_titles


def test_batch_truncation_happens_before_left_padding() -> None:
    tokenizer = CharacterTokenizer()
    encoded = tokenize_observation_batch(tokenizer, ["abc", "x" * 20], 5)

    assert encoded.shape == (2, 5)
    assert tokenizer.decode(encoded[0]) == "abc"
    assert tokenizer.decode(encoded[1]) == "xxxxx"


def test_raw_searchr1_and_normalized_results_render_identically() -> None:
    tokenizer = CharacterTokenizer()
    raw = [{"document": {"contents": "Title\nbody text"}}]
    normalized = [result(1, "Title", "body text")]

    raw_rendered = render_retrieval_observation(raw, tokenizer, 500)
    normalized_rendered = render_retrieval_observation(normalized, tokenizer, 500)

    assert raw_rendered.full_text == normalized_rendered.full_text
    assert raw_rendered.visible_text == normalized_rendered.visible_text
