from __future__ import annotations

import argparse
import re
from pathlib import Path

try:
    from hard_rq0.patch_searchr1_experiment_env import patch as patch_experiment_env
except ModuleNotFoundError:  # direct `python hard_rq0/...py` execution
    from patch_searchr1_experiment_env import patch as patch_experiment_env

MARKER = "# STACKPILOT_MIXED_ROUTING_V1"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one {label} block, found {count}")
    return text.replace(old, new, 1)


def patch(search_r1_root: Path) -> None:
    patch_experiment_env(search_r1_root)
    target = search_r1_root / "search_r1" / "llm_agent" / "generation.py"
    text = target.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"Mixed-routing patch already present: {target}")
        return

    loop_anchor = '''    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
'''
    loop_replacement = '''    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        # STACKPILOT_MIXED_ROUTING_V1
        mixed_mode = os.environ.get("SEARCH_R1_MIXED_MODE", "").strip().lower()
        batch_size = int(initial_input_ids.shape[0])
        self._episode_backend_ids = [None] * batch_size
        if mixed_mode:
            if mixed_mode == "blind":
                # Search-R1 expands each question into multiple rollouts. Alternating
                # by rollout row gives an exact balanced mixture while keeping a
                # trajectory on one backend for every later search turn.
                self._episode_backend_ids = [
                    "bm25" if index % 2 == 0 else "e5"
                    for index in range(batch_size)
                ]
            elif mixed_mode == "oracle":
                prompts = self.tokenizer.batch_decode(
                    initial_input_ids, skip_special_tokens=True
                )
                backend_ids = []
                pattern = re.compile(
                    r"<retrieval_environment>\s*(bm25|e5)\s*</retrieval_environment>",
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
    text = replace_once(text, loop_anchor, loop_replacement, "run_llm_loop anchor")

    search_anchor = '''        search_queries = [content for action, content in zip(cur_actions, contents) if action == 'search']
        if do_search:
            search_results = self.batch_search(search_queries)
            assert len(search_results) == sum([1 for action in cur_actions if action == 'search'])
        else:
            search_results = [''] * sum([1 for action in cur_actions if action == 'search'])
'''
    search_replacement = '''        search_indices = [
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
'''
    text = replace_once(text, search_anchor, search_replacement, "execute_predictions search")

    function_pattern = re.compile(
        r"    def batch_search\(self, queries: List\[str\] = None\) -> str:\n"
        r".*?"
        r"    def _passages2string\(self, retrieval_result\):",
        re.DOTALL,
    )
    match = function_pattern.search(text)
    if match is None:
        raise RuntimeError("Could not locate Search-R1 batch_search/_batch_search block")
    replacement = '''    def batch_search(self, queries: List[str] = None, backend_ids=None) -> str:
        """Batchified search with optional episode-stable backend IDs."""
        results = self._batch_search(queries, backend_ids=backend_ids)['result']
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
    text = text[: match.start()] + replacement + text[match.end() :]
    target.write_text(text, encoding="utf-8")
    print(f"Applied episode-stable mixed-routing patch: {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-r1-root", required=True)
    args = parser.parse_args()
    patch(Path(args.search_r1_root).resolve())
