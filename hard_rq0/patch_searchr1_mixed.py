from __future__ import annotations

import argparse
import random
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

try:
    from hard_rq0.patch_searchr1_experiment_env import patch as patch_experiment_env
    from hard_rq0.patch_searchr1_observation_geometry import (
        MARKER as OBSERVATION_GEOMETRY_MARKER,
    )
    from hard_rq0.patch_searchr1_observation_geometry import (
        validate_patched as validate_observation_geometry,
    )
except ModuleNotFoundError:  # direct `python hard_rq0/...py` execution
    from patch_searchr1_experiment_env import patch as patch_experiment_env
    from patch_searchr1_observation_geometry import (
        MARKER as OBSERVATION_GEOMETRY_MARKER,
    )
    from patch_searchr1_observation_geometry import (
        validate_patched as validate_observation_geometry,
    )

LEGACY_MARKER = "# STACKPILOT_MIXED_ROUTING_V1"
MARKER = "# STACKPILOT_MIXED_ROUTING_V2"
TRAINER_MARKER = "# STACKPILOT_MIXED_ROUTING_METADATA_V2"
VALID_BACKENDS = ("bm25", "e5")


def assign_training_backend_ids(
    routing_uids: Sequence[Any],
    *,
    n_agent: int,
    seed: int,
    mixed_step: int,
) -> list[str]:
    """Assign one hidden backend per GRPO group, balanced across prompt groups."""
    if n_agent <= 0:
        raise ValueError("blind mixed routing requires a positive SEARCH_R1_N_AGENT")
    uids = [str(uid) for uid in routing_uids]
    if len(uids) % n_agent != 0:
        raise ValueError(
            f"rollout batch size {len(uids)} is not divisible by n_agent={n_agent}"
        )

    group_uids: list[str] = []
    seen_uids: set[str] = set()
    for group_start in range(0, len(uids), n_agent):
        group = uids[group_start : group_start + n_agent]
        if len(set(group)) != 1:
            raise RuntimeError(
                "GRPO repeat ordering is not UID-homogeneous at rollout rows "
                f"{group_start}:{group_start + n_agent}: {group}"
            )
        uid = group[0]
        if uid in seen_uids:
            raise RuntimeError(
                f"GRPO UID {uid!r} appears in more than one prompt group"
            )
        seen_uids.add(uid)
        group_uids.append(uid)

    n_groups = len(group_uids)
    if n_groups == 0 or n_groups % 2 != 0:
        raise ValueError(
            "blind mixed routing requires an even positive number of source "
            f"prompt groups per batch; got {n_groups}"
        )

    group_backends = ["bm25"] * (n_groups // 2) + ["e5"] * (n_groups // 2)
    rng = random.Random((int(seed) << 32) ^ int(mixed_step))
    rng.shuffle(group_backends)
    backend_ids = [backend for backend in group_backends for _ in range(n_agent)]

    for group_start in range(0, len(backend_ids), n_agent):
        group = backend_ids[group_start : group_start + n_agent]
        if len(set(group)) != 1:
            raise AssertionError(
                "internal error: one GRPO UID was assigned multiple backends"
            )
    expected_per_backend = len(backend_ids) // 2
    if any(
        backend_ids.count(backend) != expected_per_backend for backend in VALID_BACKENDS
    ):
        raise AssertionError("internal error: mixed-routing batch is not 50:50")
    return backend_ids


def validate_validation_backend_ids(
    routing_uids: Sequence[Any],
    backend_ids: Sequence[Any],
) -> list[str]:
    """Validate row-level hidden routes without applying GRPO grouping."""
    uids = list(routing_uids)
    normalized = [str(backend).strip().lower() for backend in backend_ids]
    if len(uids) != len(normalized):
        raise ValueError(
            "validation routing_uid and routing_backend must have equal length; "
            f"got {len(uids)} and {len(normalized)}"
        )
    invalid = sorted(set(normalized) - set(VALID_BACKENDS))
    if invalid:
        raise ValueError(f"invalid validation routing backends: {invalid}")
    if not normalized:
        raise ValueError("validation mixed routing received an empty batch")
    return normalized


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one {label} block, found {count}")
    return text.replace(old, new, 1)


def patch(search_r1_root: Path) -> None:
    patch_experiment_env(search_r1_root)
    target = search_r1_root / "search_r1" / "llm_agent" / "generation.py"
    text = target.read_text(encoding="utf-8")
    legacy_helper_import = """from patch_searchr1_mixed import (
    assign_training_backend_ids as _stackpilot_assign_training_backend_ids,
    validate_validation_backend_ids as _stackpilot_validate_validation_backend_ids,
)
"""
    helper_import = """from hard_rq0.patch_searchr1_mixed import (
    assign_training_backend_ids as _stackpilot_assign_training_backend_ids,
    validate_validation_backend_ids as _stackpilot_validate_validation_backend_ids,
)
"""
    import_anchor = "from typing import List, Dict, Any, Tuple\n"
    if legacy_helper_import in text:
        text = replace_once(
            text,
            legacy_helper_import,
            helper_import,
            "legacy mixed-routing helper import",
        )
    elif MARKER not in text and helper_import not in text:
        text = replace_once(
            text,
            import_anchor,
            import_anchor + helper_import,
            "mixed-routing helper import",
        )
    elif MARKER in text and helper_import not in text:
        raise RuntimeError(
            f"Mixed-routing marker has no package-qualified helper import in {target}"
        )

    loop_anchor = '''    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
'''
    loop_replacement = '''    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        # STACKPILOT_MIXED_ROUTING_V1
        # STACKPILOT_MIXED_ROUTING_V2
        mixed_mode = os.environ.get("SEARCH_R1_MIXED_MODE", "").strip().lower()
        batch_size = int(initial_input_ids.shape[0])
        self._episode_backend_ids = [None] * batch_size
        if mixed_mode:
            routing_uids = gen_batch.non_tensor_batch.get("routing_uid")
            if self.is_validation:
                backend_ids = gen_batch.non_tensor_batch.get("routing_backend")
                if routing_uids is None or backend_ids is None:
                    raise RuntimeError(
                        "mixed validation requires hidden row-level routing_uid "
                        "and routing_backend metadata"
                    )
                backend_ids = _stackpilot_validate_validation_backend_ids(
                    routing_uids, backend_ids
                )
                if len(backend_ids) != batch_size:
                    raise RuntimeError(
                        f"validation routing metadata has {len(backend_ids)} rows "
                        f"for generation batch size {batch_size}"
                    )
                self._episode_backend_ids = backend_ids
            elif mixed_mode == "blind":
                n_agent = int(os.environ.get("SEARCH_R1_N_AGENT", "0"))
                seed = int(os.environ.get("RQ0_SEED", "0"))
                if routing_uids is None:
                    raise RuntimeError(
                        "blind mixed training requires routing_uid metadata copied "
                        "from the repeated Search-R1 batch"
                    )
                mixed_step = int(getattr(self, "_mixed_step", 0))
                backend_ids = _stackpilot_assign_training_backend_ids(
                    routing_uids,
                    n_agent=n_agent,
                    seed=seed,
                    mixed_step=mixed_step,
                )
                if len(backend_ids) != batch_size:
                    raise RuntimeError(
                        f"training routing metadata has {len(backend_ids)} rows "
                        f"for generation batch size {batch_size}"
                    )
                self._episode_backend_ids = backend_ids
                self._mixed_step = mixed_step + 1
            elif mixed_mode == "oracle":
                prompts = self.tokenizer.batch_decode(
                    initial_input_ids, skip_special_tokens=True
                )
                backend_ids = []
                pattern = re.compile(
                    r"<retrieval_environment>\\s*(bm25|e5)\\s*</retrieval_environment>",
                    re.IGNORECASE,
                )
                for index, prompt in enumerate(prompts):
                    match = pattern.search(prompt)
                    if match is None:
                        raise RuntimeError(
                            "SEARCH_R1_MIXED_MODE=oracle requires a retrieval_environment "
                            f"marker in every prompt; missing at row {index}"
                        )
                    backend_ids.append(match.group(1).lower())
                self._episode_backend_ids = backend_ids
            else:
                raise ValueError(
                    "SEARCH_R1_MIXED_MODE must be empty, blind, or oracle; "
                    f"got {mixed_mode!r}"
                )

        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
'''
    new_body_start = loop_replacement.index(
        '        mixed_mode = os.environ.get("SEARCH_R1_MIXED_MODE", "").strip().lower()'
    )
    new_body_end = loop_replacement.index(
        "        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}"
    )
    # V1 used a different assignment body. Keep the exact old block here so a
    # direct rerun can migrate an already-patched Search-R1 checkout safely.
    legacy_body = """        mixed_mode = os.environ.get("SEARCH_R1_MIXED_MODE", "").strip().lower()
        batch_size = int(initial_input_ids.shape[0])
        self._episode_backend_ids = [None] * batch_size
        if mixed_mode:
            if mixed_mode == "blind":
                n_agent = int(os.environ.get("SEARCH_R1_N_AGENT", "0"))
                seed = int(os.environ.get("RQ0_SEED", "0"))
                if n_agent <= 0 or n_agent % 2 != 0:
                    raise ValueError(
                        "blind mixed routing requires an even positive SEARCH_R1_N_AGENT"
                    )
                if batch_size % n_agent != 0:
                    raise ValueError(
                        f"rollout batch size {batch_size} is not divisible by n_agent={n_agent}"
                    )
                backend_ids = []
                for group_start in range(0, batch_size, n_agent):
                    # Every GRPO group receives the same number of BM25 and E5
                    # rollouts. The starting backend varies by prompt and seed so
                    # backend identity is not confounded with rollout sample index.
                    prompt_sum = int(initial_input_ids[group_start].long().sum().item())
                    offset = (prompt_sum + seed) % 2
                    backend_ids.extend(
                        "bm25" if (rollout_index + offset) % 2 == 0 else "e5"
                        for rollout_index in range(n_agent)
                    )
                self._episode_backend_ids = backend_ids
            elif mixed_mode == "oracle":
                prompts = self.tokenizer.batch_decode(
                    initial_input_ids, skip_special_tokens=True
                )
                backend_ids = []
                pattern = re.compile(
                    r"<retrieval_environment>\\s*(bm25|e5)\\s*</retrieval_environment>",
                    re.IGNORECASE,
                )
                for index, prompt in enumerate(prompts):
                    match = pattern.search(prompt)
                    if match is None:
                        raise RuntimeError(
                            "SEARCH_R1_MIXED_MODE=oracle requires a retrieval_environment "
                            f"marker in every prompt; missing at row {index}"
                        )
                    backend_ids.append(match.group(1).lower())
                self._episode_backend_ids = backend_ids
            else:
                raise ValueError(
                    "SEARCH_R1_MIXED_MODE must be empty, blind, or oracle; "
                    f"got {mixed_mode!r}"
                )

"""
    if MARKER in text:
        pass
    elif LEGACY_MARKER in text:
        text = replace_once(
            text,
            legacy_body,
            loop_replacement[new_body_start:new_body_end],
            "legacy mixed-routing loop",
        )
        text = text.replace(
            LEGACY_MARKER,
            LEGACY_MARKER + "\n        " + MARKER,
            1,
        )
    else:
        text = replace_once(text, loop_anchor, loop_replacement, "run_llm_loop anchor")

    search_anchor = """        search_queries = [content for action, content in zip(cur_actions, contents) if action == 'search']
        if do_search:
            search_results = self.batch_search(search_queries)
            assert len(search_results) == sum([1 for action in cur_actions if action == 'search'])
        else:
            search_results = [''] * sum([1 for action in cur_actions if action == 'search'])
"""
    search_replacement = """        search_indices = [
            index
            for index, (action, active) in enumerate(zip(cur_actions, active_mask))
            if active and action == 'search'
        ]
        search_queries = [contents[index] for index in search_indices]
        backend_ids = None
        if getattr(self, "_episode_backend_ids", None) is not None:
            selected = [self._episode_backend_ids[index] for index in search_indices]
            if any(value is not None for value in selected):
                if any(value not in {"bm25", "e5"} for value in selected):
                    raise RuntimeError(f"Invalid episode backend assignment: {selected}")
                backend_ids = selected
        if do_search:
            search_results = self.batch_search(search_queries, backend_ids=backend_ids)
            assert len(search_results) == len(search_indices)
        else:
            search_results = [''] * len(search_indices)
"""
    if MARKER not in target.read_text(encoding="utf-8") and search_anchor in text:
        # A pristine checkout still needs the request-routing changes. V1 and V2
        # checkouts already contain them.
        text = replace_once(
            text,
            search_anchor,
            search_replacement,
            "execute_predictions search",
        )

    function_pattern = re.compile(
        r"    def batch_search\(self, queries: List\[str\] = None\) -> str:\n"
        r".*?"
        r"    def _passages2string\(self, retrieval_result\):",
        re.DOTALL,
    )
    replacement = '''    def batch_search(self, queries: List[str] = None, backend_ids=None) -> str:
        """Batchified search with optional episode-stable backend IDs."""
        results = self._batch_search(queries, backend_ids=backend_ids)['result']
        self._stackpilot_last_search_titles = []
        self._stackpilot_last_observed_titles = []
        for result_index, retrieval_result in enumerate(results):
            titles = []
            for document_index, item in enumerate(retrieval_result):
                if not isinstance(item, dict) or not isinstance(
                    item.get('document'), dict
                ):
                    raise RuntimeError(
                        "retriever result is missing document metadata at "
                        f"result={result_index}, document={document_index}"
                    )
                contents = item['document'].get('contents')
                if not isinstance(contents, str) or not contents.strip():
                    raise RuntimeError(
                        "retriever document has no contents at "
                        f"result={result_index}, document={document_index}"
                    )
                title = contents.split("\\n", 1)[0].strip()
                if not title:
                    raise RuntimeError(
                        "retriever document has an empty title at "
                        f"result={result_index}, document={document_index}"
                    )
                titles.append(title)
            self._stackpilot_last_search_titles.append(titles)
        return [self._passages2string(result) for result in results]

    def _batch_search(self, queries, backend_ids=None):
        payload = {
            "queries": queries,
            "topk": self.config.topk,
            "return_scores": True,
        }
        if backend_ids is not None:
            if len(backend_ids) != len(queries):
                raise ValueError("backend_ids and queries must have equal length")
            payload["backend_ids"] = backend_ids
        response = requests.post(
            self.config.search_url,
            json=payload,
            timeout=float(os.environ.get("SEARCH_R1_RETRIEVER_TIMEOUT", "120")),
        )
        response.raise_for_status()
        return response.json()

    def _passages2string(self, retrieval_result):'''
    if function_pattern.search(text) is not None:
        match = function_pattern.search(text)
        assert match is not None
        text = text[: match.start()] + replacement + text[match.end() :]
    elif (
        "def batch_search(self, queries: List[str] = None, backend_ids=None)"
        not in text
    ):
        raise RuntimeError(
            "Could not locate Search-R1 batch_search/_batch_search block"
        )
    if OBSERVATION_GEOMETRY_MARKER in text:
        validate_observation_geometry(text, target)
    target.write_text(text, encoding="utf-8")

    trainer_target = search_r1_root / "verl" / "trainer" / "ppo" / "ray_trainer.py"
    if trainer_target.exists():
        trainer_text = trainer_target.read_text(encoding="utf-8")
        if TRAINER_MARKER not in trainer_text:
            training_anchor = """                gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
"""
            training_replacement = """                gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                # STACKPILOT_MIXED_ROUTING_METADATA_V2
                if os.environ.get("SEARCH_R1_MIXED_MODE", "").strip():
                    routing_uids = batch.non_tensor_batch.get('index')
                    if routing_uids is None or len(routing_uids) != len(batch.batch):
                        raise RuntimeError(
                            "mixed training requires one source index per repeated rollout"
                        )
                    gen_batch.non_tensor_batch['routing_uid'] = routing_uids.copy()
"""
            trainer_text = replace_once(
                trainer_text,
                training_anchor,
                training_replacement,
                "training routing metadata",
            )
            validation_anchor = """                test_gen_batch = test_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
"""
            validation_replacement = """                test_gen_batch = test_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                if os.environ.get("SEARCH_R1_MIXED_MODE", "").strip():
                    extra_infos = test_batch.non_tensor_batch.get('extra_info')
                    routing_uids = test_batch.non_tensor_batch.get('index')
                    if (
                        extra_infos is None
                        or routing_uids is None
                        or len(extra_infos) != len(test_batch.batch)
                        or len(routing_uids) != len(test_batch.batch)
                    ):
                        raise RuntimeError(
                            "mixed validation requires one extra_info/index record per row"
                        )
                    validation_backends = []
                    for row_index, extra_info in enumerate(extra_infos):
                        if not isinstance(extra_info, dict):
                            raise RuntimeError(
                                f"mixed validation extra_info row {row_index} is not a mapping"
                            )
                        backend = str(
                            extra_info.get('routing_backend', '')
                        ).strip().lower()
                        if backend not in {'bm25', 'e5'}:
                            raise RuntimeError(
                                "mixed validation requires hidden routing_backend=bm25/e5 "
                                f"at row {row_index}; got {backend!r}"
                            )
                        validation_backends.append(backend)
                    test_gen_batch.non_tensor_batch['routing_uid'] = routing_uids.copy()
                    test_gen_batch.non_tensor_batch['routing_backend'] = np.asarray(
                        validation_backends, dtype=object
                    )
"""
            trainer_text = replace_once(
                trainer_text,
                validation_anchor,
                validation_replacement,
                "validation routing metadata",
            )
            trainer_target.write_text(trainer_text, encoding="utf-8")
    print(f"Applied UID-homogeneous mixed-routing patch: {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-r1-root", required=True)
    args = parser.parse_args()
    patch(Path(args.search_r1_root).resolve())
