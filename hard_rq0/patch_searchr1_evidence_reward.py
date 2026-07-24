from __future__ import annotations

import argparse
import os
from pathlib import Path

try:
    from hard_rq0.patch_searchr1_experiment_env import patch as patch_experiment_env
    from hard_rq0.patch_searchr1_reward_protocol import (
        MARKER as TERMINAL_REWARD_MARKER,
    )
    from hard_rq0.patch_searchr1_reward_protocol import (
        patch as patch_terminal_reward,
    )
except ModuleNotFoundError:  # direct `python hard_rq0/...py` execution
    from patch_searchr1_experiment_env import patch as patch_experiment_env
    from patch_searchr1_reward_protocol import MARKER as TERMINAL_REWARD_MARKER
    from patch_searchr1_reward_protocol import patch as patch_terminal_reward

MARKER = "# STACKPILOT_EVIDENCE_REWARD_V4"

HELPER = """# STACKPILOT_EVIDENCE_REWARD_V4
def _normalize_evidence_title(value):
    text = unicodedata.normalize("NFKC", str(value)).replace("_", " ").strip()
    text = text.strip("\\\"'").lower()
    return " ".join(text.split())


def _support_recall_from_titles(retrieved_titles, support_titles):
    retrieved = {
        _normalize_evidence_title(title)
        for title in (retrieved_titles or [])
        if _normalize_evidence_title(title)
    }
    gold = {
        _normalize_evidence_title(title)
        for title in (support_titles or [])
        if _normalize_evidence_title(title)
    }
    if not gold:
        return 0.0
    return len(gold & retrieved) / len(gold)


class RewardManager():
"""

BASE_SCORE = "            score = float(answer_score)\n"

EVIDENCE_SCORE = """            extra_info = data_item.non_tensor_batch.get('extra_info', {})
            support_titles = (
                extra_info.get('support_titles', [])
                if isinstance(extra_info, dict)
                else []
            )
            observed_titles = data_item.non_tensor_batch.get(
                'stackpilot_observed_titles'
            )
            if isinstance(observed_titles, np.ndarray):
                observed_titles = observed_titles.tolist()
            elif isinstance(observed_titles, tuple):
                observed_titles = list(observed_titles)
            if not isinstance(observed_titles, list):
                raise RuntimeError(
                    "evidence reward requires stackpilot_observed_titles list"
                )
            try:
                search_count = int(
                    data_item.non_tensor_batch['stackpilot_search_count']
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    "evidence reward requires integer stackpilot_search_count"
                ) from exc
            if search_count < 0:
                raise RuntimeError(
                    f"stackpilot_search_count must be non-negative: {search_count}"
                )
            evidence_score = (
                0.0
                if protocol_failure
                else _support_recall_from_titles(
                    observed_titles,
                    support_titles,
                )
            )
            answer_weight = float(os.environ.get("ANSWER_REWARD_WEIGHT", "1.0"))
            evidence_weight = float(os.environ.get("EVIDENCE_REWARD_WEIGHT", "0.5"))
            search_cost = float(os.environ.get("SEARCH_COST_WEIGHT", "0.02"))
            if trajectory_truncated:
                # The answer/search actions are not all present in PPO's token
                # sequence, so assigning either positive evidence or negative
                # search cost would credit tokens for an unrepresented path.
                score = 0.0
            else:
                score = (
                    answer_weight * float(answer_score)
                    + evidence_weight * evidence_score
                    - search_cost * search_count
                )
"""


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one {label}, found {count}")
    return text.replace(old, new, 1)


def validate_patched(text: str, target: Path) -> None:
    required = (
        MARKER,
        TERMINAL_REWARD_MARKER,
        "_support_recall_from_titles(",
        "'stackpilot_observed_titles'",
        "'stackpilot_search_count'",
        "'stackpilot_trajectory_truncated'",
        "_stackpilot_answer_em(terminal_answer, targets)",
        "if trajectory_truncated:",
        "EVIDENCE_REWARD_WEIGHT",
    )
    missing = [value for value in required if value not in text]
    forbidden = (
        "solution_str=sequences_str",
        "solution_str=response_str",
        "_support_recall_from_response",
        'r"<search>.*?</search>"',
    )
    remaining = [value for value in forbidden if value in text]
    if missing or remaining:
        raise RuntimeError(
            f"Incomplete structured evidence reward patch in {target}: "
            f"missing={missing}, forbidden={remaining}"
        )


def patch(search_r1_root: Path) -> None:
    reward_mode = os.environ.get("SEARCH_R1_REWARD_MODE", "").strip().lower()
    if reward_mode != "evidence":
        raise RuntimeError(
            "Evidence reward patch requires SEARCH_R1_REWARD_MODE=evidence; "
            f"got {reward_mode or '<unset>'!r}"
        )
    patch_experiment_env(search_r1_root)
    patch_terminal_reward(search_r1_root)
    target = search_r1_root / "verl" / "trainer" / "main_ppo.py"
    text = target.read_text(encoding="utf-8")
    if MARKER in text:
        validate_patched(text, target)
        print(f"Evidence reward patch already present: {target}")
        return

    if "import os\n" not in text:
        text = replace_once(
            text,
            "import re\nimport numpy as np\n",
            "import re\nimport numpy as np\nimport os\n",
            "reward imports",
        )
    if "import unicodedata\n" not in text:
        import_anchor = (
            "import os\n" if "import os\n" in text else "import numpy as np\n"
        )
        text = replace_once(
            text,
            import_anchor,
            import_anchor + "import unicodedata\n",
            "unicode import anchor",
        )

    text = replace_once(
        text,
        "\nclass RewardManager():\n",
        "\n" + HELPER,
        "RewardManager anchor",
    )
    text = replace_once(
        text,
        BASE_SCORE,
        EVIDENCE_SCORE,
        "structured evidence score",
    )
    validate_patched(text, target)
    target.write_text(text, encoding="utf-8")
    print(f"Applied structured evidence-aware reward patch: {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-r1-root", required=True)
    args = parser.parse_args()
    patch(Path(args.search_r1_root).resolve())
