from __future__ import annotations

import argparse
from pathlib import Path

MARKER = "# STACKPILOT_OBSERVATION_GEOMETRY_V1"
ACTION_MARKER = "# STACKPILOT_STRICT_ACTION_PROTOCOL_V2"

OLD_PROCESS = """    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        \"\"\"Process next observations from environment.\"\"\"
~~~~~~~~
        next_obs_ids = self.tokenizer(
            next_obs,~
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")~~~~~~~~~~~~
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids
""".replace("~", " ")

NEW_PROCESS = """    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        \"\"\"Apply the fixed total observation budget before batch padding.\"\"\"
        return tokenize_observation_batch(
            self.tokenizer,
            next_obs,
            int(self.config.max_obs_length),
        )
"""

OLD_RENDERER = """    def _passages2string(self, retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
~~~~~~~~~~~~
            content = doc_item['document']['contents']
            title = content.split("\\n")[0]
            text = "\\n".join(content.split("\\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\\n"

        return format_reference
""".replace("~", " ")

NEW_RENDERER = """    def _passages2string(self, retrieval_result):
        rendered = render_retrieval_observation(
            retrieval_result,
            self.tokenizer,
            int(self.config.max_obs_length),
        )
        self._stackpilot_last_observed_titles.append(
            list(rendered.observed_titles)
        )
        return rendered.full_text
"""

OLD_BATCH_SETUP = """        if do_search:
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

NEW_BATCH_SETUP = """        if do_search:
            search_title_batches = getattr(
                self, '_stackpilot_last_search_titles', None
            )
            search_observed_title_batches = getattr(
                self, '_stackpilot_last_observed_titles', None
            )
            if (
                not isinstance(search_title_batches, list)
                or len(search_title_batches) != len(search_results)
                or not isinstance(search_observed_title_batches, list)
                or len(search_observed_title_batches) != len(search_results)
            ):
                raise RuntimeError(
                    "retriever title metadata does not match search results"
                )
            search_title_batches = [
                list(titles) for titles in search_title_batches
            ]
            search_observed_title_batches = [
                list(titles) for titles in search_observed_title_batches
            ]
        else:
            search_title_batches = [[] for _ in search_results]
            search_observed_title_batches = [[] for _ in search_results]

"""

OLD_SEARCH_ACTION = """                    self._stackpilot_retrieved_titles[i].extend(
                        search_title_batches.pop(0)
                    )
                    next_obs.append(
                        f'\\n\\n<information>{search_result}</information>\\n\\n'
                    )
"""

NEW_SEARCH_ACTION = """                    self._stackpilot_retrieved_titles[i].extend(
                        search_title_batches.pop(0)
                    )
                    self._stackpilot_observed_titles[i].extend(
                        search_observed_title_batches.pop(0)
                    )
                    next_obs.append(search_result)
"""


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one {label}, found {count}")
    return text.replace(old, new, 1)


def validate_patched(text: str, target: Path) -> None:
    required = (
        MARKER,
        "render_retrieval_observation(",
        "tokenize_observation_batch(",
        "self._stackpilot_last_observed_titles = []",
        "self._stackpilot_last_observed_titles",
        "self._stackpilot_observed_titles",
        "'stackpilot_observed_titles'",
        "next_obs.append(search_result)",
        "assert len(search_observed_title_batches) == 0",
    )
    missing = [fragment for fragment in required if fragment not in text]
    forbidden = (
        "OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG",
        "f'\\n\\n<information>{search_result}</information>\\n\\n'",
    )
    remaining = [fragment for fragment in forbidden if fragment in text]
    if missing or remaining:
        raise RuntimeError(
            f"Incomplete observation geometry patch in {target}: "
            f"missing={missing}, legacy={remaining}"
        )


def patch(search_r1_root: Path) -> None:
    target = search_r1_root / "search_r1" / "llm_agent" / "generation.py"
    text = target.read_text(encoding="utf-8")
    if MARKER in text:
        validate_patched(text, target)
        print(f"Observation geometry patch already present: {target}")
        return
    if ACTION_MARKER not in text:
        raise RuntimeError(
            "Apply patch_searchr1_action_protocol.py before the observation "
            f"geometry patch: {target}"
        )

    text = replace_once(
        text,
        "from stackpilot.action_protocol import parse_action\n",
        "from stackpilot.action_protocol import parse_action\n"
        "from stackpilot.observation_geometry import (\n"
        "    render_retrieval_observation,\n"
        "    tokenize_observation_batch,\n"
        ")\n"
        f"{MARKER}\n",
        "observation imports",
    )
    text = replace_once(text, OLD_PROCESS, NEW_PROCESS, "observation tokenizer")
    text = replace_once(text, OLD_RENDERER, NEW_RENDERER, "passage renderer")
    text = replace_once(
        text,
        "        self._stackpilot_last_search_titles = []\n",
        "        self._stackpilot_last_search_titles = []\n"
        "        self._stackpilot_last_observed_titles = []\n",
        "observed title batch reset",
    )
    text = replace_once(
        text,
        "        self._stackpilot_retrieved_titles = [\n"
        "            [] for _ in range(protocol_batch_size)\n"
        "        ]\n",
        "        self._stackpilot_retrieved_titles = [\n"
        "            [] for _ in range(protocol_batch_size)\n"
        "        ]\n"
        "        self._stackpilot_observed_titles = [\n"
        "            [] for _ in range(protocol_batch_size)\n"
        "        ]\n",
        "observed title protocol state",
    )
    text = replace_once(
        text,
        OLD_BATCH_SETUP,
        NEW_BATCH_SETUP,
        "observed title metadata setup",
    )
    text = replace_once(
        text,
        OLD_SEARCH_ACTION,
        NEW_SEARCH_ACTION,
        "bounded search observation",
    )
    text = replace_once(
        text,
        "                    search_title_batches.pop(0)\n"
        "                    next_obs.append('')\n",
        "                    search_title_batches.pop(0)\n"
        "                    search_observed_title_batches.pop(0)\n"
        "                    next_obs.append('')\n",
        "forced-final observed title consumption",
    )
    text = replace_once(
        text,
        "        assert len(search_title_batches) == 0\n",
        "        assert len(search_title_batches) == 0\n"
        "        assert len(search_observed_title_batches) == 0\n",
        "observed title metadata exhaustion",
    )
    text = replace_once(
        text,
        "            'stackpilot_retrieved_titles': self._stackpilot_retrieved_titles,\n",
        "            'stackpilot_retrieved_titles': self._stackpilot_retrieved_titles,\n"
        "            'stackpilot_observed_titles': self._stackpilot_observed_titles,\n",
        "observed title protocol output",
    )

    validate_patched(text, target)
    target.write_text(text, encoding="utf-8")
    print(f"Applied fixed observation geometry patch: {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-r1-root", required=True)
    args = parser.parse_args()
    patch(Path(args.search_r1_root).resolve())
