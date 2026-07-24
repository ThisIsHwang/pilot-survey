from __future__ import annotations

import argparse
from pathlib import Path

LEGACY_MARKER = "# STACKPILOT_STRICT_ACTION_PROTOCOL_V1"
MARKER = "# STACKPILOT_STRICT_ACTION_PROTOCOL_V2"

TRUNCATION_BLOCK = (
    "        responses_str = [resp.split('</search>')[0] + '</search>'\n"
    "                 if '</search>' in resp \n"
    "                 else resp.split('</answer>')[0] + '</answer>'\n"
    "                 if '</answer>' in resp \n"
    "                 else resp\n"
    "                 for resp in responses_str]\n"
)

PARSER_BLOCK = """            if isinstance(prediction, str): # for llm output
                pattern = r'<(search|answer)>(.*?)</\\1>'
                match = re.search(pattern, prediction, re.DOTALL)
                if match:
                    content = match.group(2).strip()  # Return only the content inside the tags
                    action = match.group(1)
                else:
                    content = ''
                    action = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
"""

STRICT_PARSER_BLOCK = """            if isinstance(prediction, str): # for llm output
                action, content = parse_action(prediction)
                if action == 'invalid':
                    action = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
"""

PROTOCOL_STATE = """        # Protocol state is kept per source row, including rows that become
        # inactive before later generation turns. RewardManager consumes these
        # fields instead of reparsing the concatenated prompt/trajectory.
        protocol_batch_size = int(initial_input_ids.shape[0])
        self._stackpilot_terminal_answers = [""] * protocol_batch_size
        self._stackpilot_protocol_failures = [1] * protocol_batch_size
        self._stackpilot_invalid_action_counts = [0] * protocol_batch_size
        self._stackpilot_executed_search_counts = [0] * protocol_batch_size
        self._stackpilot_trajectory_truncated = [0] * protocol_batch_size
        self._stackpilot_retrieved_titles = [
            [] for _ in range(protocol_batch_size)
        ]

"""

TITLE_METADATA_RETURN = """        self._stackpilot_last_search_titles = []
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
"""

TITLE_BATCH_SETUP = """        if do_search:
            search_title_batches = getattr(
                self, '_stackpilot_last_search_titles', None
            )
            if (
                not isinstance(search_title_batches, list)
                or len(search_title_batches) != len(search_results)
            ):
                raise RuntimeError(
                    "retriever title metadata does not match search results"
                )
            search_title_batches = [
                list(titles) for titles in search_title_batches
            ]
        else:
            search_title_batches = [[] for _ in search_results]

"""

ORIGINAL_ACTION_BLOCK = """                if action == 'answer':
                    next_obs.append('')
                    dones.append(1)
                    valid_action.append(1)
                    is_search.append(0)
                elif action == 'search':
                    next_obs.append(f'\\n\\n<information>{search_results.pop(0).strip()}</information>\\n\\n')
                    dones.append(0)
                    valid_action.append(1)
                    is_search.append(1)
                else:
                    next_obs.append(f'\\nMy previous action is invalid. \\
If I want to search, I should put the query between <search> and </search>. \\
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\\n')
                    dones.append(0)
                    valid_action.append(0)
                    is_search.append(0)
"""

STRUCTURED_ACTION_BLOCK = """                if action == 'answer':
                    self._stackpilot_terminal_answers[i] = contents[i]
                    self._stackpilot_protocol_failures[i] = 0
                    next_obs.append('')
                    dones.append(1)
                    valid_action.append(1)
                    is_search.append(0)
                elif action == 'search' and do_search:
                    search_result = search_results.pop(0).strip()
                    self._stackpilot_executed_search_counts[i] += 1
                    self._stackpilot_retrieved_titles[i].extend(
                        search_title_batches.pop(0)
                    )
                    next_obs.append(
                        f'\\n\\n<information>{search_result}</information>\\n\\n'
                    )
                    dones.append(0)
                    valid_action.append(1)
                    is_search.append(1)
                elif action == 'search':
                    # The forced final turn cannot execute another retrieval.
                    # It is syntactically valid but has no terminal answer.
                    # Consume the placeholder result/title entries created by
                    # execute_predictions(do_search=False), so its exhaustion
                    # invariants remain true for this valid-but-terminal action.
                    search_results.pop(0)
                    search_title_batches.pop(0)
                    next_obs.append('')
                    dones.append(1)
                    valid_action.append(1)
                    is_search.append(0)
                else:
                    self._stackpilot_invalid_action_counts[i] += 1
                    next_obs.append(f'\\nMy previous action is invalid. \\
If I want to search, I should put the query between <search> and </search>. \\
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\\n')
                    dones.append(0 if do_search else 1)
                    valid_action.append(0)
                    is_search.append(0)
"""

ORIGINAL_RIGHT_SIDE_LIMIT = (
    "        effective_len = "
    "self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()\n"
    "        max_len = min(self.config.max_prompt_length, effective_len)\n"
    "        \n"
    "        return {'responses': responses[:, :max_len], "
    "'responses_with_info_mask': responses_with_info_mask[:, :max_len]}\n"
)

TRACKED_RIGHT_SIDE_LIMIT = """        response_lengths = self.tensor_fn.create_attention_mask(responses).sum(dim=1)
        truncated_rows = response_lengths > self.config.max_prompt_length
        if bool(truncated_rows.any()):
            if len(truncated_rows) != len(self._stackpilot_trajectory_truncated):
                raise RuntimeError(
                    "trajectory truncation state does not match rollout batch size"
                )
            for row_index, truncated in enumerate(truncated_rows.tolist()):
                if truncated:
                    self._stackpilot_trajectory_truncated[row_index] = 1
        effective_len = response_lengths.max()
        max_len = min(self.config.max_prompt_length, effective_len)

        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}
"""

ORIGINAL_FINAL_OUTPUT = """        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
"""

STRUCTURED_FINAL_OUTPUT = """        protocol_values = {
            'stackpilot_terminal_answer': self._stackpilot_terminal_answers,
            'stackpilot_protocol_failure': self._stackpilot_protocol_failures,
            'stackpilot_invalid_action_count': self._stackpilot_invalid_action_counts,
            'stackpilot_search_count': self._stackpilot_executed_search_counts,
            'stackpilot_trajectory_truncated': self._stackpilot_trajectory_truncated,
            'stackpilot_retrieved_titles': self._stackpilot_retrieved_titles,
        }
        protocol_non_tensors = {}
        for key, values in protocol_values.items():
            value_array = np.empty(len(values), dtype=object)
            value_array[:] = values
            protocol_non_tensors[key] = value_array

        final_output = DataProto.from_dict(
            final_output,
            non_tensors=protocol_non_tensors,
        )
        final_output.meta_info.update(meta_info)
"""


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one {label}, found {count}")
    return text.replace(old, new, 1)


def validate_patched(text: str, target: Path) -> None:
    required = (
        "import numpy as np",
        "from stackpilot.action_protocol import parse_action",
        "action, content = parse_action(prediction)",
        "self._stackpilot_terminal_answers",
        "self._stackpilot_protocol_failures",
        "self._stackpilot_executed_search_counts",
        "self._stackpilot_trajectory_truncated",
        "self._stackpilot_last_search_titles",
        "'stackpilot_retrieved_titles'",
        "non_tensors=protocol_non_tensors",
    )
    missing = [value for value in required if value not in text]
    legacy_fragments = (
        "resp.split('</search>')",
        "re.search(pattern, prediction",
        ORIGINAL_FINAL_OUTPUT,
    )
    remaining = [value for value in legacy_fragments if value in text]
    if missing or remaining:
        raise RuntimeError(
            f"Incomplete strict action protocol patch in {target}: "
            f"missing={missing}, legacy={remaining}"
        )


def patch(search_r1_root: Path) -> None:
    target = search_r1_root / "search_r1" / "llm_agent" / "generation.py"
    text = target.read_text(encoding="utf-8")
    if MARKER in text:
        validate_patched(text, target)
        print(f"Strict action protocol patch already present: {target}")
        return

    migrating_v1 = LEGACY_MARKER in text
    if migrating_v1:
        v1_required = (
            "from stackpilot.action_protocol import parse_action",
            "action, content = parse_action(prediction)",
        )
        missing = [value for value in v1_required if value not in text]
        if missing or "resp.split('</search>')" in text:
            raise RuntimeError(
                f"Cannot migrate incomplete strict action V1 patch in {target}: "
                f"{missing}"
            )

    if "import numpy as np\n" not in text:
        text = replace_once(
            text,
            "import requests\n",
            "import requests\nimport numpy as np\n",
            "numpy import anchor",
        )
    if "from stackpilot.action_protocol import parse_action\n" not in text:
        text = replace_once(
            text,
            "import requests\n",
            "import requests\nfrom stackpilot.action_protocol import parse_action\n",
            "action protocol import anchor",
        )

    if migrating_v1:
        text = replace_once(
            text,
            LEGACY_MARKER,
            MARKER,
            "legacy action protocol marker",
        )
    else:
        preservation = f"""        {MARKER}
        # Preserve the complete model output. The shared parser below rejects
        # multiple actions and non-<think> text outside the single action.
"""
        text = replace_once(
            text,
            TRUNCATION_BLOCK,
            preservation,
            "response truncation block",
        )
        text = replace_once(
            text,
            PARSER_BLOCK,
            STRICT_PARSER_BLOCK,
            "upstream action parser",
        )

    state_anchor = (
        "        original_right_side = {'responses': initial_input_ids[:, []], "
        "'responses_with_info_mask': initial_input_ids[:, []]}\n"
    )
    text = replace_once(
        text,
        state_anchor,
        state_anchor + PROTOCOL_STATE,
        "protocol state anchor",
    )
    text = replace_once(
        text,
        ORIGINAL_ACTION_BLOCK,
        STRUCTURED_ACTION_BLOCK,
        "structured action tracking",
    )
    text = replace_once(
        text,
        "        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):\n",
        TITLE_BATCH_SETUP
        + "        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):\n",
        "retrieval title metadata setup",
    )
    text = replace_once(
        text,
        "        assert len(search_results) == 0\n",
        "        assert len(search_results) == 0\n"
        "        assert len(search_title_batches) == 0\n",
        "retrieval title metadata exhaustion",
    )
    if "self._stackpilot_last_search_titles = []" not in text:
        text = replace_once(
            text,
            "        return [self._passages2string(result) for result in results]\n",
            TITLE_METADATA_RETURN,
            "structured retriever title metadata",
        )
    text = replace_once(
        text,
        ORIGINAL_RIGHT_SIDE_LIMIT,
        TRACKED_RIGHT_SIDE_LIMIT,
        "trajectory truncation tracking",
    )
    text = replace_once(
        text,
        ORIGINAL_FINAL_OUTPUT,
        STRUCTURED_FINAL_OUTPUT,
        "structured protocol output",
    )

    validate_patched(text, target)
    target.write_text(text, encoding="utf-8")
    print(f"Applied strict action protocol patch: {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-r1-root", required=True)
    args = parser.parse_args()
    patch(Path(args.search_r1_root).resolve())
