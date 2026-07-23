from __future__ import annotations

import argparse
from pathlib import Path

MARKER = "# STACKPILOT_EVIDENCE_REWARD_V1"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one {label}, found {count}")
    return text.replace(old, new, 1)


def patch(search_r1_root: Path) -> None:
    target = search_r1_root / "verl" / "trainer" / "main_ppo.py"
    text = target.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"Evidence reward patch already present: {target}")
        return

    text = replace_once(
        text,
        "import re\nimport numpy as np\n",
        "import re\nimport numpy as np\nimport os\nimport unicodedata\n",
        "reward imports",
    )
    helper_anchor = "\nclass RewardManager():\n"
    helper = '''
# STACKPILOT_EVIDENCE_REWARD_V1
def _normalize_evidence_title(value):
    text = unicodedata.normalize("NFKC", str(value)).replace("_", " ").strip()
    text = text.strip("\\\"'").lower()
    return " ".join(text.split())


def _support_recall_from_sequence(sequence, support_titles):
    blocks = re.findall(r"<information>(.*?)</information>", sequence, re.IGNORECASE | re.DOTALL)
    normalized_blocks = [_normalize_evidence_title(block) for block in blocks]
    gold = {
        _normalize_evidence_title(title)
        for title in (support_titles or [])
        if _normalize_evidence_title(title)
    }
    if not gold:
        return 0.0
    found = {
        title
        for title in gold
        if any(title in block for block in normalized_blocks)
    }
    return len(found) / len(gold)


class RewardManager():
'''
    text = replace_once(text, helper_anchor, "\n" + helper, "RewardManager anchor")

    score_anchor = '''            score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, format_score=self.format_score)

            reward_tensor[i, valid_response_length - 1] = score
'''
    score_replacement = '''            answer_score = compute_score_fn(
                solution_str=sequences_str,
                ground_truth=ground_truth,
                format_score=self.format_score,
            )
            extra_info = data_item.non_tensor_batch.get('extra_info', {})
            support_titles = extra_info.get('support_titles', []) if isinstance(extra_info, dict) else []
            evidence_score = _support_recall_from_sequence(sequences_str, support_titles)
            search_count = len(re.findall(r"<search>", sequences_str, re.IGNORECASE))
            answer_weight = float(os.environ.get("ANSWER_REWARD_WEIGHT", "1.0"))
            evidence_weight = float(os.environ.get("EVIDENCE_REWARD_WEIGHT", "0.5"))
            search_cost = float(os.environ.get("SEARCH_COST_WEIGHT", "0.02"))
            score = (
                answer_weight * float(answer_score)
                + evidence_weight * evidence_score
                - search_cost * search_count
            )

            reward_tensor[i, valid_response_length - 1] = score
'''
    text = replace_once(text, score_anchor, score_replacement, "reward score block")
    target.write_text(text, encoding="utf-8")
    print(f"Applied evidence-aware reward patch: {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-r1-root", required=True)
    args = parser.parse_args()
    patch(Path(args.search_r1_root).resolve())
