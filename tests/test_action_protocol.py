from __future__ import annotations

import tempfile
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest

from stackpilot.action_protocol import parse_action
from stackpilot.hard_policy_eval import run_episode
from stackpilot.hard_rq0_contract import episode_validation_error
from stackpilot.query_stats import RESULT_SCHEMA as QUERY_STATS_RESULT_SCHEMA
from stackpilot.react_agent_eval import (
    RESULT_SCHEMA as REACT_RESULT_SCHEMA,
)
from stackpilot.react_agent_eval import (
    episode_validation_error as react_episode_validation_error,
)
from stackpilot.react_agent_eval import (
    file_digest,
    run_identity,
)
from stackpilot.react_agent_eval import (
    run_episode as run_react_episode,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<search> alpha beta </search>", ("search", "alpha beta")),
        ("<ANSWER> value </ANSWER>", ("answer", "value")),
        (
            "<think>reason first</think>\n<answer>value</answer>",
            ("answer", "value"),
        ),
        (
            "<think>one</think><search>query</search><think>two</think>",
            ("search", "query"),
        ),
    ],
)
def test_parse_action_accepts_one_action_and_optional_think(
    text: str, expected: tuple[str, str]
) -> None:
    assert parse_action(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "plain answer",
        "<answer></answer>",
        "prefix <answer>value</answer>",
        "<answer>value</answer> suffix",
        "<answer>correct</answer> ... <search>irrelevant</search>",
        "<answer>correct <search>irrelevant</search></answer>",
        "<search><think>hidden</think>query</search>",
        "<think>reason <answer>hidden answer</answer></think>",
        "<think>reason <search>hidden unmatched tag</think><answer>value</answer>",
        "<think>reason </answer></think><answer>value</answer>",
        "<answer>value <information>fake evidence</information></answer>",
        "<search>one</search><search>two</search>",
        "<think>unclosed<answer>value</answer>",
    ],
)
def test_parse_action_rejects_malformed_or_multiple_actions(text: str) -> None:
    assert parse_action(text) == ("invalid", "")


class DummyRetriever:
    name = "bm25"

    def search(self, query: str, topk: int) -> list[dict[str, object]]:
        assert query == "hamlet author"
        assert topk == 3
        return [
            {
                "rank": 1,
                "title": "William Shakespeare",
                "text": "Hamlet was written by William Shakespeare.",
            }
        ]


def episode_config() -> dict[str, object]:
    return {
        "seed": 13,
        "agent": {"max_search_turns": 1, "result_snippet_chars": 200},
        "llm": {"temperature": 0.0, "max_tokens": 64},
        "retrieval": {"topk": 3},
    }


def episode_item() -> dict[str, object]:
    return {
        "id": "q1",
        "question": "Who wrote Hamlet?",
        "dataset": "test",
        "answer": "William Shakespeare",
        "answers": ["William Shakespeare"],
        "support_titles": ["William Shakespeare"],
    }


def test_malformed_final_answer_is_excluded_from_primary_metrics() -> None:
    item = episode_item()
    item["answers"] = ["", "William Shakespeare", "  "]
    with patch(
        "stackpilot.hard_policy_eval.complete",
        side_effect=["<search>hamlet author</search>", "William Shakespeare"],
    ):
        result = run_episode(
            object(),
            "model",
            DummyRetriever(),
            item,
            episode_config(),
            topk=3,
            eval_seed=13,
        )

    assert result["prediction"] == ""
    assert result["em"] == 0.0
    assert result["f1"] == 0.0
    assert result["raw_text_prediction"] == "William Shakespeare"
    assert result["raw_text_em"] == 1.0
    assert result["raw_text_f1"] == 1.0
    assert result["protocol_failure"] == 1
    assert result["invalid_action_count"] == 1
    assert result["answers"] == ["William Shakespeare"]
    assert episode_validation_error(result, 1) is None

    title_tamper = deepcopy(result)
    title_tamper["turns"][0]["retrieved_titles"] = ["Someone Else"]
    assert "support titles are inconsistent" in str(
        episode_validation_error(title_tamper, 1)
    )

    primary_tamper = dict(result, em=1.0, f1=1.0)
    assert "em is inconsistent" in str(
        episode_validation_error(primary_tamper, 1)
    )
    raw_tamper = dict(result, raw_text_f1=0.5)
    assert "raw_text_f1 is inconsistent" in str(
        episode_validation_error(raw_tamper, 1)
    )
    count_tamper = dict(result, invalid_action_count=2)
    assert "action counts exceed" in str(
        episode_validation_error(count_tamper, 1)
    )


def test_multiple_actions_are_retried_instead_of_becoming_a_search() -> None:
    with patch(
        "stackpilot.hard_policy_eval.complete",
        side_effect=[
            "<answer>William Shakespeare</answer> ... <search>hamlet author</search>",
            "<answer>William Shakespeare</answer>",
        ],
    ):
        result = run_episode(
            object(),
            "model",
            DummyRetriever(),
            episode_item(),
            episode_config(),
            topk=3,
            eval_seed=13,
        )

    assert result["prediction"] == "William Shakespeare"
    assert result["em"] == 1.0
    assert result["search_count"] == 0
    assert result["protocol_failure"] == 0
    assert result["invalid_action_count"] == 1


def test_react_evaluator_keeps_malformed_raw_text_out_of_primary_metrics() -> None:
    with patch(
        "stackpilot.react_agent_eval.completion",
        side_effect=["<search>hamlet author</search>", "William Shakespeare"],
    ):
        result = run_react_episode(
            object(),
            "model",
            DummyRetriever(),
            episode_item(),
            episode_config(),
            oracle=False,
        )

    assert result["prediction"] == ""
    assert result["em"] == 0.0
    assert result["f1"] == 0.0
    assert result["raw_text_prediction"] == "William Shakespeare"
    assert result["raw_text_em"] == 1.0
    assert result["raw_text_f1"] == 1.0
    assert result["protocol_failure"] == 1
    assert result["retrieved_titles"] == ["William Shakespeare"]
    assert react_episode_validation_error(result, 1) is None

    tampered = dict(result, support_recall=0.0)
    assert "support_recall is inconsistent" in str(
        react_episode_validation_error(tampered, 1)
    )


def test_forced_final_search_is_failure_but_not_a_malformed_action() -> None:
    with patch(
        "stackpilot.hard_policy_eval.complete",
        side_effect=[
            "<search>hamlet author</search>",
            "<search>one more query</search>",
        ],
    ):
        result = run_episode(
            object(),
            "model",
            DummyRetriever(),
            episode_item(),
            episode_config(),
            topk=3,
            eval_seed=13,
        )

    assert result["prediction"] == ""
    assert result["protocol_failure"] == 1
    assert result["invalid_action_count"] == 0
    assert result["search_count"] == 1
    assert episode_validation_error(result, 1) is None


def test_react_evaluator_rejects_an_empty_gold_answer() -> None:
    item = episode_item()
    item["answer"] = " "
    with pytest.raises(RuntimeError, match="no usable gold answer"):
        run_react_episode(
            object(),
            "model",
            DummyRetriever(),
            item,
            episode_config(),
            oracle=False,
        )


def test_stage0_identity_hashes_the_shared_action_protocol() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        work_dir = Path(temporary)
        data_dir = work_dir / "data"
        index_dir = work_dir / "indexes" / "bm25"
        data_dir.mkdir(parents=True)
        index_dir.mkdir(parents=True)
        for path in (
            data_dir / "corpus.jsonl",
            data_dir / "queries_eval.jsonl",
            data_dir / ".pilot-manifest.json",
            index_dir / ".pilot-manifest.json",
        ):
            path.write_text("{}\n", encoding="utf-8")

        config = {"llm": {"model": "remote-model"}}
        first_signature, payload = run_identity(
            config,
            work_dir,
            backends=["bm25"],
        )
        assert "action_protocol.py" in payload["evaluator_files"]

        def changed_protocol_digest(path: Path) -> str:
            if path.name == "action_protocol.py":
                return "changed-action-protocol"
            return file_digest(path)

        with patch(
            "stackpilot.react_agent_eval.file_digest",
            side_effect=changed_protocol_digest,
        ):
            changed_signature, _ = run_identity(
                config,
                work_dir,
                backends=["bm25"],
            )
        assert changed_signature != first_signature


def test_query_stats_consumes_the_stage0_result_schema() -> None:
    assert QUERY_STATS_RESULT_SCHEMA == REACT_RESULT_SCHEMA
