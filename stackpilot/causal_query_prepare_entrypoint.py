from __future__ import annotations

from typing import Any

from stackpilot import causal_query_prepare as implementation

_ORIGINAL_BUILD = implementation.build_candidate_states


def build_candidate_states(
    rows: list[tuple[dict[str, Any], Any]],
    *,
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Keep only states with meaningful remaining evidence headroom.

    Query mediation is undefined as a useful search problem after all annotated
    support has already been observed. Filtering before balanced sampling keeps
    the reported prevalence conditional on an explicitly unresolved cohort.
    """

    candidates = _ORIGINAL_BUILD(rows, cfg=cfg)
    maximum = float(cfg["source"].get("maximum_prefix_support_recall", 0.5))
    if not 0.0 <= maximum < 1.0:
        raise ValueError(
            "source.maximum_prefix_support_recall must be in [0, 1); "
            f"got {maximum}"
        )
    selected = []
    for row in candidates:
        prior_turns = row.get("prior_turns") or []
        if not prior_turns:
            continue
        prefix_recall = float(prior_turns[-1].get("support_recall", 0.0))
        if prefix_recall <= maximum:
            copy = dict(row)
            copy["prefix_support_recall"] = prefix_recall
            copy["maximum_prefix_support_recall"] = maximum
            selected.append(copy)
    return selected


implementation.build_candidate_states = build_candidate_states


if __name__ == "__main__":
    implementation.main()
