from __future__ import annotations

import re

ACTION_RE = re.compile(r"<(search|answer)>(.*?)</\1>", re.IGNORECASE | re.DOTALL)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
CONTROL_TAG_RE = re.compile(
    r"</?(?:search|answer|think|information)>",
    re.IGNORECASE,
)


def parse_action(text: str) -> tuple[str, str]:
    """Parse one protocol action, allowing only optional ``<think>`` blocks.

    Search-R1 rollouts and offline evaluation both call this function. Keeping
    the parser here prevents training and evaluation from assigning different
    meanings to the same model output.
    """

    if not isinstance(text, str):
        raise TypeError(f"action text must be str, got {type(text).__name__}")

    matches = list(ACTION_RE.finditer(text))
    if len(matches) != 1:
        return "invalid", ""

    match = matches[0]
    think_matches = list(THINK_RE.finditer(text))
    # A complete think block is the only text allowed outside the action, but
    # it must not be usable to hide an unmatched/nested protocol tag from the
    # single-action check below.
    if any(CONTROL_TAG_RE.search(think.group(1)) for think in think_matches):
        return "invalid", ""
    think_spans = [(think.start(), think.end()) for think in think_matches]
    if any(
        think_start <= match.start() and match.end() <= think_end
        for think_start, think_end in think_spans
    ):
        return "invalid", ""

    outside = text[: match.start()] + text[match.end() :]
    if THINK_RE.sub("", outside).strip():
        return "invalid", ""

    value = match.group(2).strip()
    if not value or CONTROL_TAG_RE.search(value):
        return "invalid", ""
    return match.group(1).lower(), value
