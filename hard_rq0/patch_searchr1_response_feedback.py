from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

MARKER = "# STACKPILOT_RESPONSE_FEEDBACK_ROLLOUT_V1"

IMPORT_ANCHOR = """from stackpilot.behavior_quotient_runtime import (
    append_behavior_telemetry,
    compute_behavior_advantages,
    select_behavior_rows,
)
"""
IMPORT_REPLACEMENT = IMPORT_ANCHOR + """# STACKPILOT_RESPONSE_FEEDBACK_ROLLOUT_V1
from stackpilot.response_feedback_runtime import run_grouped_feedback_rollouts
"""

TRAIN_CALL_ANCHOR = """                        final_gen_batch_output = generation_manager.run_llm_loop(
                            gen_batch=gen_batch,
                            initial_input_ids=first_input_ids,
                        )
"""
TRAIN_CALL_REPLACEMENT = """                        final_gen_batch_output = run_grouped_feedback_rollouts(
                            generation_manager=generation_manager,
                            gen_batch=gen_batch,
                            initial_input_ids=first_input_ids,
                            mode=os.environ.get(
                                'STACKPILOT_RF_ROLLOUT_MODE', 'iid'
                            ),
                            first_count=int(os.environ.get(
                                'STACKPILOT_RF_FIRST_COUNT', '4'
                            )),
                            maximum_titles=int(os.environ.get(
                                'STACKPILOT_RF_MAX_TITLES', '24'
                            )),
                            maximum_chars=int(os.environ.get(
                                'STACKPILOT_RF_MAX_CHARS', '1800'
                            )),
                            prompt_token_budget=int(os.environ.get(
                                'STACKPILOT_RF_PROMPT_TOKEN_BUDGET', '2048'
                            )),
                        )
"""

VALIDATION_CALL_ANCHOR = """                        final_gen_batch_output = generation_manager.run_llm_loop(
                            gen_batch=test_gen_batch,
                            initial_input_ids=first_input_ids,
                        )
"""
VALIDATION_CALL_REPLACEMENT = """                        final_gen_batch_output = run_grouped_feedback_rollouts(
                            generation_manager=generation_manager,
                            gen_batch=test_gen_batch,
                            initial_input_ids=first_input_ids,
                            mode=os.environ.get(
                                'STACKPILOT_RF_VALIDATION_MODE', 'iid'
                            ),
                            first_count=int(os.environ.get(
                                'STACKPILOT_RF_FIRST_COUNT', '4'
                            )),
                            maximum_titles=int(os.environ.get(
                                'STACKPILOT_RF_MAX_TITLES', '24'
                            )),
                            maximum_chars=int(os.environ.get(
                                'STACKPILOT_RF_MAX_CHARS', '1800'
                            )),
                            prompt_token_budget=int(os.environ.get(
                                'STACKPILOT_RF_PROMPT_TOKEN_BUDGET', '2048'
                            )),
                        )
"""


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one {label}, found {count}")
    return text.replace(old, new, 1)


def replace_rollout_call(
    text: str,
    *,
    gen_batch_name: str,
    exact_anchor: str,
    replacement: str,
    label: str,
) -> str:
    """Replace a rollout call without depending on incidental whitespace."""

    exact_count = text.count(exact_anchor)
    if exact_count == 1:
        return text.replace(exact_anchor, replacement, 1)
    if exact_count > 1:
        raise RuntimeError(f"Expected one {label}, found {exact_count}")
    pattern = re.compile(
        r"(?m)^[ \t]*final_gen_batch_output[ \t]*=[ \t]*"
        r"generation_manager\.run_llm_loop\([ \t\r\n]*"
        + rf"gen_batch[ \t]*=[ \t]*{re.escape(gen_batch_name)}[ \t]*,[ \t\r\n]*"
        + r"initial_input_ids[ \t]*=[ \t]*first_input_ids[ \t]*,[ \t\r\n]*"
        + r"\)[ \t]*\n"
    )
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one semantic {label}, found {len(matches)}"
        )
    match = matches[0]
    return text[: match.start()] + replacement + text[match.end() :]


def validate(text: str, target: Path) -> None:
    required = (
        MARKER,
        "run_grouped_feedback_rollouts(",
        "STACKPILOT_RF_ROLLOUT_MODE",
        "STACKPILOT_RF_VALIDATION_MODE",
        "STACKPILOT_RF_FIRST_COUNT",
        "STACKPILOT_RF_MAX_TITLES",
    )
    missing = [value for value in required if value not in text]
    if missing:
        raise RuntimeError(
            f"Incomplete response-feedback trainer patch in {target}: {missing}"
        )


def patch(search_r1_root: Path) -> None:
    target = search_r1_root / "verl" / "trainer" / "ppo" / "ray_trainer.py"
    text = target.read_text(encoding="utf-8")
    if MARKER in text:
        validate(text, target)
        print(f"Response-feedback rollout patch already present: {target}")
    else:
        if "# STACKPILOT_BEHAVIOR_QUOTIENT_TRAINER_V1" not in text:
            raise RuntimeError(
                "Apply patch_searchr1_behavior_quotient.py before response-feedback"
            )
        text = replace_once(text, IMPORT_ANCHOR, IMPORT_REPLACEMENT, "runtime import")
        text = replace_rollout_call(
            text,
            gen_batch_name="gen_batch",
            exact_anchor=TRAIN_CALL_ANCHOR,
            replacement=TRAIN_CALL_REPLACEMENT,
            label="training rollout",
        )
        text = replace_rollout_call(
            text,
            gen_batch_name="test_gen_batch",
            exact_anchor=VALIDATION_CALL_ANCHOR,
            replacement=VALIDATION_CALL_REPLACEMENT,
            label="validation rollout",
        )
        validate(text, target)
        target.write_text(text, encoding="utf-8")
        print(f"Applied response-feedback rollout patch: {target}")
    if os.environ.get("STACKPILOT_QUERY_CREDIT_PATCH", "0") == "1":
        from hard_rq0.patch_searchr1_query_credit import patch as patch_query_credit

        patch_query_credit(search_r1_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-r1-root", required=True)
    args = parser.parse_args()
    patch(Path(args.search_r1_root).resolve())
