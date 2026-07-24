from __future__ import annotations

import argparse
import re
from pathlib import Path

LEGACY_MARKER = "# STACKPILOT_TERMINAL_REWARD_V1"
MARKER = "# STACKPILOT_TERMINAL_REWARD_V2"
EVIDENCE_MARKER_RE = re.compile(r"# STACKPILOT_EVIDENCE_REWARD_V[34]")

PINNED_SCORE_BLOCK = """            score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, format_score=self.format_score)

            reward_tensor[i, valid_response_length - 1] = score
"""

ANSWER_HELPER = """
def _stackpilot_normalize_answer(value):
    # Keep this byte-for-byte equivalent in behavior to
    # stackpilot.common.normalize_text, which defines final-evaluation EM.
    text = str(value).lower()
    text = re.sub(r"\\b(a|an|the)\\b", " ", text)
    text = re.sub(r"[^a-z0-9\\s]", " ", text)
    return " ".join(text.split())


def _stackpilot_answer_em(prediction, targets):
    normalized_prediction = _stackpilot_normalize_answer(prediction)
    return float(
        any(
            normalized_prediction == _stackpilot_normalize_answer(target)
            for target in targets
        )
    )


class RewardManager():
"""

TERMINAL_SCORE_BLOCK = """            # STACKPILOT_TERMINAL_REWARD_V2
            if int(valid_response_length) <= 0:
                raise RuntimeError("terminal reward received an empty response")
            required_protocol_fields = (
                'stackpilot_terminal_answer',
                'stackpilot_protocol_failure',
                'stackpilot_trajectory_truncated',
            )
            missing_protocol_fields = [
                key
                for key in required_protocol_fields
                if key not in data_item.non_tensor_batch
            ]
            if missing_protocol_fields:
                raise RuntimeError(
                    "terminal reward is missing rollout protocol metadata: "
                    f"{missing_protocol_fields}"
                )
            terminal_answer = data_item.non_tensor_batch[
                'stackpilot_terminal_answer'
            ]
            if not isinstance(terminal_answer, str):
                raise RuntimeError(
                    "stackpilot_terminal_answer must be a string, got "
                    f"{type(terminal_answer).__name__}"
                )

            def _stackpilot_binary_protocol_field(name):
                try:
                    value = int(data_item.non_tensor_batch[name])
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(f"{name} must be 0 or 1") from exc
                if value not in (0, 1):
                    raise RuntimeError(
                        f"{name} must be 0 or 1, got {value!r}"
                    )
                return value

            protocol_failure = _stackpilot_binary_protocol_field(
                'stackpilot_protocol_failure'
            )
            trajectory_truncated = _stackpilot_binary_protocol_field(
                'stackpilot_trajectory_truncated'
            )
            targets = (
                ground_truth.get('target')
                if isinstance(ground_truth, dict)
                else None
            )
            if isinstance(targets, str):
                targets = [targets]
            elif isinstance(targets, np.ndarray):
                targets = targets.tolist()
            elif isinstance(targets, tuple):
                targets = list(targets)
            if not isinstance(targets, list):
                raise RuntimeError(
                    "reward ground_truth.target must be a string or list"
                )
            targets = [
                str(target).strip()
                for target in targets
                if target is not None and str(target).strip()
            ]
            if not targets:
                raise RuntimeError("reward ground_truth.target has no usable aliases")
            if protocol_failure or trajectory_truncated:
                answer_score = 0.0
            else:
                if not terminal_answer.strip():
                    raise RuntimeError(
                        "protocol-success rollout has an empty terminal answer"
                    )
                answer_score = _stackpilot_answer_em(terminal_answer, targets)
            score = float(answer_score)

            reward_tensor[i, valid_response_length - 1] = score
"""

BASE_SCORE = "            score = float(answer_score)\n"
REWARD_ASSIGNMENT = "            reward_tensor[i, valid_response_length - 1] = score\n"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one {label}, found {count}")
    return text.replace(old, new, 1)


def _remove_current_evidence_reward(text: str, target: Path) -> str:
    """Return the canonical answer-only V2 reward implementation."""

    if EVIDENCE_MARKER_RE.search(text) is None:
        return text

    helper_pattern = re.compile(
        r"\n# STACKPILOT_EVIDENCE_REWARD_V[34]\n"
        r"def _normalize_evidence_title\(value\):.*?"
        r"\n\nclass RewardManager\(\):\n",
        re.DOTALL,
    )
    text, helper_count = helper_pattern.subn("\nclass RewardManager():\n", text)
    if helper_count != 1:
        raise RuntimeError(
            f"Could not remove the current evidence helper from {target}; "
            f"found {helper_count}"
        )

    score_start_text = (
        "            extra_info = data_item.non_tensor_batch.get('extra_info', {})\n"
    )
    start = text.find(score_start_text)
    if start < 0:
        raise RuntimeError(
            f"Could not locate the current evidence score block in {target}"
        )
    end = text.find(REWARD_ASSIGNMENT, start)
    if end < 0:
        raise RuntimeError(
            f"Could not locate the current evidence reward assignment in {target}"
        )
    text = text[:start] + BASE_SCORE + "\n" + text[end:]
    if "unicodedata." not in text:
        text = text.replace("import unicodedata\n", "", 1)
    return text


def _remove_legacy_evidence_reward(text: str, target: Path) -> str:
    if not re.search(r"STACKPILOT_EVIDENCE_REWARD_V[12]", text):
        return text

    helper_pattern = re.compile(
        r"\n# STACKPILOT_EVIDENCE_REWARD_V[12]\n"
        r"def _normalize_evidence_title\(value\):.*?"
        r"\n\nclass RewardManager\(\):\n",
        re.DOTALL,
    )
    text, helper_count = helper_pattern.subn("\nclass RewardManager():\n", text)
    if helper_count != 1:
        raise RuntimeError(
            f"Could not migrate the legacy evidence helper in {target}; "
            f"found {helper_count}"
        )

    score_starts = (
        (
            "            if int(valid_response_length) <= 0:\n"
            '                raise RuntimeError("evidence reward received an empty response")'
        ),
        "            answer_score = compute_score_fn(\n",
    )
    start_positions = [
        position
        for start_text in score_starts
        if (position := text.find(start_text)) >= 0
    ]
    if not start_positions:
        raise RuntimeError(
            f"Could not locate the legacy evidence score block in {target}"
        )
    # Response-only V2 contains both anchors. They denote one score block, so
    # migration must begin at the earlier one rather than treating it as two
    # independent patches.
    start = min(start_positions)
    end = text.find(REWARD_ASSIGNMENT, start)
    if end < 0:
        raise RuntimeError(
            f"Could not locate the end of the legacy evidence score in {target}"
        )
    end += len(REWARD_ASSIGNMENT)
    text = text[:start] + PINNED_SCORE_BLOCK + text[end:]
    if "unicodedata." not in text:
        text = text.replace("import unicodedata\n", "", 1)
    return text


def _replace_legacy_terminal_reward(text: str, target: Path) -> str:
    start = text.find("            " + LEGACY_MARKER)
    if start < 0:
        raise RuntimeError(
            f"Could not locate the legacy terminal reward block in {target}"
        )
    end = text.find(REWARD_ASSIGNMENT, start)
    if end < 0:
        raise RuntimeError(
            f"Could not locate the legacy terminal reward assignment in {target}"
        )
    end += len(REWARD_ASSIGNMENT)
    return text[:start] + TERMINAL_SCORE_BLOCK + text[end:]


def _install_answer_helper(text: str, target: Path) -> str:
    if "def _stackpilot_answer_em(" in text:
        return text
    return replace_once(
        text,
        "\nclass RewardManager():\n",
        "\n" + ANSWER_HELPER,
        f"answer-normalization helper anchor in {target}",
    )


def validate_patched(text: str, target: Path) -> None:
    required = (
        MARKER,
        "def _stackpilot_answer_em(",
        "'stackpilot_terminal_answer'",
        "'stackpilot_protocol_failure'",
        "'stackpilot_trajectory_truncated'",
        "_stackpilot_answer_em(terminal_answer, targets)",
        "if protocol_failure or trajectory_truncated:",
        "score = float(answer_score)",
    )
    missing = [value for value in required if value not in text]
    forbidden = (
        LEGACY_MARKER,
        "solution_str=sequences_str",
        "solution_str=response_str",
        "qa_em.em_check(terminal_answer, targets)",
        "STACKPILOT_EVIDENCE_REWARD_V",
        "_support_recall_from_titles(",
    )
    remaining = [value for value in forbidden if value in text]
    if missing or remaining:
        raise RuntimeError(
            f"Incomplete terminal reward patch in {target}: "
            f"missing={missing}, forbidden={remaining}"
        )


def patch(search_r1_root: Path) -> None:
    target = search_r1_root / "verl" / "trainer" / "main_ppo.py"
    original = target.read_text(encoding="utf-8")
    text = _remove_current_evidence_reward(original, target)

    if MARKER in text:
        text = _install_answer_helper(text, target)
        validate_patched(text, target)
        if text != original:
            target.write_text(text, encoding="utf-8")
            print(f"Restored answer-only terminal reward protocol: {target}")
        else:
            print(f"Terminal reward protocol patch already present: {target}")
        return

    if LEGACY_MARKER in text:
        text = _replace_legacy_terminal_reward(text, target)
    else:
        text = _remove_legacy_evidence_reward(text, target)
        text = replace_once(
            text,
            PINNED_SCORE_BLOCK,
            TERMINAL_SCORE_BLOCK,
            "terminal reward score block",
        )
    text = _install_answer_helper(text, target)
    validate_patched(text, target)
    target.write_text(text, encoding="utf-8")
    print(f"Applied terminal reward protocol patch: {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-r1-root", required=True)
    args = parser.parse_args()
    patch(Path(args.search_r1_root).resolve())
