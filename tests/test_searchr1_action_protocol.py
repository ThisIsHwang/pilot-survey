from __future__ import annotations

import ast
import tempfile
from pathlib import Path
from typing import Any, List, Tuple

import pytest

from hard_rq0.patch_searchr1_action_protocol import MARKER, patch
from hard_rq0.patch_searchr1_evidence_reward import (
    MARKER as EVIDENCE_REWARD_MARKER,
)
from hard_rq0.patch_searchr1_evidence_reward import patch as patch_evidence
from hard_rq0.patch_searchr1_experiment_env import patch as patch_environment
from hard_rq0.patch_searchr1_mixed import patch as patch_mixed
from hard_rq0.patch_searchr1_reward_protocol import patch as patch_reward
from hard_rq0.patch_searchr1_validation import patch as patch_validation
from stackpilot.action_protocol import parse_action


def test_searchr1_action_protocol_patch_is_idempotent_and_compiles() -> None:
    source = Path("upstream/Search-R1/search_r1/llm_agent/generation.py").read_text(
        encoding="utf-8"
    )
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        target = root / "search_r1" / "llm_agent" / "generation.py"
        target.parent.mkdir(parents=True)
        target.write_text(source, encoding="utf-8")

        patch(root)
        first = target.read_text(encoding="utf-8")
        patch(root)
        second = target.read_text(encoding="utf-8")

    assert first == second
    assert MARKER in first
    assert "from stackpilot.action_protocol import parse_action" in first
    assert "action, content = parse_action(prediction)" in first
    assert "resp.split('</search>')" not in first
    for field in (
        "stackpilot_terminal_answer",
        "stackpilot_protocol_failure",
        "stackpilot_invalid_action_count",
        "stackpilot_search_count",
        "stackpilot_trajectory_truncated",
        "stackpilot_retrieved_titles",
    ):
        assert field in first
    assert "non_tensors=protocol_non_tensors" in first
    assert "elif action == 'search' and do_search:" in first
    assert "execute_predictions(do_search=False)" in first
    assert first.count("search_results.pop(0)") == 2
    assert first.count("search_title_batches.pop(0)") == 2
    assert "self._stackpilot_last_search_titles" in first
    assert "contents.split(\"\\n\", 1)[0].strip()" in first
    assert 'r"Doc\\s+' not in first
    assert "truncated_rows = response_lengths > self.config.max_prompt_length" in first
    assert "self._stackpilot_trajectory_truncated[row_index] = 1" in first
    compile(first, str(target), "exec")


def test_strict_action_and_mixed_routing_patches_compose() -> None:
    generation_source = Path(
        "upstream/Search-R1/search_r1/llm_agent/generation.py"
    ).read_text(encoding="utf-8")
    trainer_source = Path("upstream/Search-R1/verl/trainer/main_ppo.py").read_text(
        encoding="utf-8"
    )
    for strict_first in (True, False):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generation = root / "search_r1" / "llm_agent" / "generation.py"
            trainer = root / "verl" / "trainer" / "main_ppo.py"
            generation.parent.mkdir(parents=True)
            trainer.parent.mkdir(parents=True)
            generation.write_text(generation_source, encoding="utf-8")
            trainer.write_text(trainer_source, encoding="utf-8")

            if strict_first:
                patch(root)
                patch_mixed(root)
            else:
                patch_mixed(root)
                patch(root)

            patched = generation.read_text(encoding="utf-8")
            assert MARKER in patched
            assert "STACKPILOT_MIXED_ROUTING_V2" in patched
            assert "action, content = parse_action(prediction)" in patched
            assert (
                "from hard_rq0.patch_searchr1_mixed import (" in patched
            )
            assert "from patch_searchr1_mixed import (" not in patched
            compile(patched, str(generation), "exec")


def test_mixed_patch_migrates_the_legacy_top_level_helper_import() -> None:
    generation_source = Path(
        "upstream/Search-R1/search_r1/llm_agent/generation.py"
    ).read_text(encoding="utf-8")
    trainer_source = Path("upstream/Search-R1/verl/trainer/main_ppo.py").read_text(
        encoding="utf-8"
    )
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        generation = root / "search_r1" / "llm_agent" / "generation.py"
        trainer = root / "verl" / "trainer" / "main_ppo.py"
        generation.parent.mkdir(parents=True)
        trainer.parent.mkdir(parents=True)
        generation.write_text(generation_source, encoding="utf-8")
        trainer.write_text(trainer_source, encoding="utf-8")

        patch_mixed(root)
        legacy = generation.read_text(encoding="utf-8").replace(
            "from hard_rq0.patch_searchr1_mixed import (",
            "from patch_searchr1_mixed import (",
            1,
        )
        generation.write_text(legacy, encoding="utf-8")
        patch_mixed(root)
        migrated = generation.read_text(encoding="utf-8")

    assert "from hard_rq0.patch_searchr1_mixed import (" in migrated
    assert "from patch_searchr1_mixed import (" not in migrated
    compile(migrated, str(generation), "exec")


def test_forced_final_search_consumes_placeholders_without_retrieval() -> None:
    source = Path("upstream/Search-R1/search_r1/llm_agent/generation.py").read_text(
        encoding="utf-8"
    )
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        target = root / "search_r1" / "llm_agent" / "generation.py"
        target.parent.mkdir(parents=True)
        target.write_text(source, encoding="utf-8")
        patch(root)
        tree = ast.parse(target.read_text(encoding="utf-8"))

    upstream_manager = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LLMGenerationManager"
    )
    methods = [
        node
        for node in upstream_manager.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"execute_predictions", "postprocess_predictions"}
    ]
    test_manager = ast.ClassDef(
        name="TestManager",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[test_manager], type_ignores=[]))
    namespace: dict[str, object] = {
        "Any": Any,
        "List": List,
        "Tuple": Tuple,
        "parse_action": parse_action,
    }
    exec(compile(module, "<patched-generation-methods>", "exec"), namespace)

    manager = namespace["TestManager"]()
    manager._stackpilot_terminal_answers = [""]
    manager._stackpilot_protocol_failures = [1]
    manager._stackpilot_invalid_action_counts = [0]
    manager._stackpilot_executed_search_counts = [0]
    manager._stackpilot_trajectory_truncated = [0]
    manager._stackpilot_retrieved_titles = [[]]
    manager.batch_search = lambda *_args, **_kwargs: pytest.fail(
        "forced final search must not call the retriever"
    )

    next_obs, dones, valid_actions, searches = manager.execute_predictions(
        ["<search>unused query</search>"],
        "<pad>",
        [True],
        do_search=False,
    )

    assert next_obs == [""]
    assert dones == [1]
    assert valid_actions == [1]
    assert searches == [0]
    assert manager._stackpilot_protocol_failures == [1]
    assert manager._stackpilot_invalid_action_counts == [0]
    assert manager._stackpilot_executed_search_counts == [0]
    assert manager._stackpilot_retrieved_titles == [[]]


@pytest.mark.parametrize("experiment", ["mixed", "evidence"])
def test_complete_searchr1_patch_stacks_are_idempotent(
    experiment: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = {
        Path("search_r1/llm_agent/generation.py"): Path(
            "upstream/Search-R1/search_r1/llm_agent/generation.py"
        ),
        Path("verl/trainer/main_ppo.py"): Path(
            "upstream/Search-R1/verl/trainer/main_ppo.py"
        ),
        Path("verl/trainer/ppo/ray_trainer.py"): Path(
            "upstream/Search-R1/verl/trainer/ppo/ray_trainer.py"
        ),
    }
    with tempfile.TemporaryDirectory() as temporary:
        if experiment == "evidence":
            monkeypatch.setenv("SEARCH_R1_REWARD_MODE", "evidence")
        root = Path(temporary)
        for relative, source in sources.items():
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

        if experiment == "mixed":
            installers = (
                patch_validation,
                patch,
                patch_reward,
                patch_environment,
                patch_mixed,
            )
        else:
            installers = (
                patch_validation,
                patch,
                patch_reward,
                patch_environment,
                patch_evidence,
            )

        for installer in installers:
            installer(root)
        first = {
            relative: (root / relative).read_text(encoding="utf-8")
            for relative in sources
        }
        for installer in installers:
            installer(root)
        second = {
            relative: (root / relative).read_text(encoding="utf-8")
            for relative in sources
        }

        assert first == second
        if experiment == "evidence":
            assert EVIDENCE_REWARD_MARKER in first[
                Path("verl/trainer/main_ppo.py")
            ]
        for relative, source in first.items():
            compile(source, str(root / relative), "exec")
