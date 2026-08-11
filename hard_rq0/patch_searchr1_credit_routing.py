from __future__ import annotations

import argparse
from pathlib import Path

GENERATION_MARKER = "# STACKPILOT_CREDIT_ROUTING_GENERATION_V1"
TRAINER_MARKER = "# STACKPILOT_CREDIT_ROUTING_TRAINER_V1"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one {label}, found {count}")
    return text.replace(old, new, 1)


def patch_generation(root: Path) -> None:
    target = root / "search_r1" / "llm_agent" / "generation.py"
    text = target.read_text(encoding="utf-8")
    if GENERATION_MARKER in text:
        required = (
            "self._stackpilot_search_action_utilities",
            "self._stackpilot_last_action_utilities",
            "stackpilot_search_action_utilities",
            "search_action_utility_batches",
        )
        missing = [value for value in required if value not in text]
        if missing:
            raise RuntimeError(f"Incomplete credit-routing generation patch: {missing}")
        print(f"Credit-routing generation patch already present: {target}")
        return
    if "# STACKPILOT_BEHAVIOR_QUOTIENT_GENERATION_V1" not in text:
        raise RuntimeError("Apply behavior-quotient generation patch before credit routing")
    if "# STACKPILOT_OBSERVATION_GEOMETRY_V1" not in text:
        raise RuntimeError("Apply observation-geometry patch before credit routing")

    state_anchor = """        self._stackpilot_search_title_batches = [
            [] for _ in range(protocol_batch_size)
        ]

"""
    state_replacement = state_anchor + f"""        {GENERATION_MARKER}
        self._stackpilot_search_action_utilities = [
            [] for _ in range(protocol_batch_size)
        ]

"""
    text = replace_once(text, state_anchor, state_replacement, "credit-routing protocol state")

    reset_anchor = """        self._stackpilot_last_search_titles = []
        self._stackpilot_last_observed_titles = []
"""
    reset_replacement = reset_anchor + "        self._stackpilot_last_action_utilities = []\n"
    text = replace_once(text, reset_anchor, reset_replacement, "retrieval utility reset")

    metadata_anchor = """            self._stackpilot_last_search_titles.append(titles)
        return [self._passages2string(result) for result in results]
"""
    metadata_replacement = """            action_values = []
            for item in retrieval_result:
                if isinstance(item, dict) and item.get('stackpilot_action_utility') is not None:
                    action_values.append(float(item['stackpilot_action_utility']))
            if action_values and max(action_values) - min(action_values) > 1e-6:
                raise RuntimeError(
                    'credit-routing proxy returned inconsistent action utility '
                    f'at result={result_index}: {action_values}'
                )
            self._stackpilot_last_action_utilities.append(
                float(action_values[0]) if action_values else 0.0
            )
            self._stackpilot_last_search_titles.append(titles)
        return [self._passages2string(result) for result in results]
"""
    text = replace_once(text, metadata_anchor, metadata_replacement, "retrieval utility metadata")

    observed_getter = """            search_observed_title_batches = getattr(
                self, '_stackpilot_last_observed_titles', None
            )
"""
    observed_getter_replacement = observed_getter + """            search_action_utility_batches = getattr(
                self, '_stackpilot_last_action_utilities', None
            )
"""
    text = replace_once(text, observed_getter, observed_getter_replacement, "utility batch getter")

    condition_anchor = """                or not isinstance(search_observed_title_batches, list)
                or len(search_observed_title_batches) != len(search_results)
"""
    condition_replacement = condition_anchor + """                or not isinstance(search_action_utility_batches, list)
                or len(search_action_utility_batches) != len(search_results)
"""
    text = replace_once(text, condition_anchor, condition_replacement, "utility metadata validation")

    conversion_anchor = """            search_observed_title_batches = [
                list(titles) for titles in search_observed_title_batches
            ]
"""
    conversion_replacement = conversion_anchor + """            search_action_utility_batches = [
                float(value) for value in search_action_utility_batches
            ]
"""
    text = replace_once(text, conversion_anchor, conversion_replacement, "utility metadata copy")

    else_anchor = """            search_observed_title_batches = [[] for _ in search_results]
"""
    else_replacement = else_anchor + "            search_action_utility_batches = [0.0 for _ in search_results]\n"
    text = replace_once(text, else_anchor, else_replacement, "utility placeholder batches")

    current_title_anchor = """                    current_search_titles = list(search_title_batches.pop(0))
"""
    current_title_replacement = current_title_anchor + """                    current_search_utility = float(
                        search_action_utility_batches.pop(0)
                    )
"""
    text = replace_once(text, current_title_anchor, current_title_replacement, "search utility consumption")

    query_metadata_anchor = """                    self._stackpilot_search_title_batches[i].append(
                        current_search_titles
                    )
"""
    query_metadata_replacement = query_metadata_anchor + """                    self._stackpilot_search_action_utilities[i].append(
                        current_search_utility
                    )
"""
    text = replace_once(text, query_metadata_anchor, query_metadata_replacement, "trajectory utility metadata")

    forced_final_anchor = """                    search_observed_title_batches.pop(0)
                    next_obs.append('')
"""
    forced_final_replacement = """                    search_observed_title_batches.pop(0)
                    search_action_utility_batches.pop(0)
                    next_obs.append('')
"""
    text = replace_once(text, forced_final_anchor, forced_final_replacement, "forced-final utility consumption")

    assertion_anchor = """        assert len(search_observed_title_batches) == 0
"""
    assertion_replacement = assertion_anchor + "        assert len(search_action_utility_batches) == 0\n"
    text = replace_once(text, assertion_anchor, assertion_replacement, "utility metadata exhaustion")

    output_anchor = """            'stackpilot_search_title_batches': self._stackpilot_search_title_batches,
"""
    output_replacement = output_anchor + """            'stackpilot_search_action_utilities': self._stackpilot_search_action_utilities,
"""
    text = replace_once(text, output_anchor, output_replacement, "utility protocol output")
    target.write_text(text, encoding="utf-8")
    print(f"Applied credit-routing generation patch: {target}")


def patch_trainer(root: Path) -> None:
    target = root / "verl" / "trainer" / "ppo" / "ray_trainer.py"
    text = target.read_text(encoding="utf-8")
    if TRAINER_MARKER in text:
        required = (
            "apply_action_utility_shaping(",
            "STACKPILOT_CR_ACTION_ROUTE",
            "stackpilot_search_action_utilities",
        )
        missing = [value for value in required if value not in text]
        if missing:
            raise RuntimeError(f"Incomplete credit-routing trainer patch: {missing}")
        print(f"Credit-routing trainer patch already present: {target}")
        return
    if "# STACKPILOT_BEHAVIOR_QUOTIENT_TRAINER_V1" not in text:
        raise RuntimeError("Apply behavior-quotient trainer patch before credit routing")

    import_anchor = """from stackpilot.behavior_quotient_runtime import (
    append_behavior_telemetry,
    compute_behavior_advantages,
    select_behavior_rows,
)
"""
    import_replacement = import_anchor + f"""{TRAINER_MARKER}
from stackpilot.credit_routing_runtime import apply_action_utility_shaping
"""
    text = replace_once(text, import_anchor, import_replacement, "credit-routing runtime import")

    advantage_anchor = """        query_batches = data.non_tensor_batch.get('stackpilot_search_queries')
"""
    advantage_replacement = """        utility_batches = data.non_tensor_batch.get(
            'stackpilot_search_action_utilities'
        )
        token_level_rewards, cr_metrics, cr_group_rows = (
            apply_action_utility_shaping(
                token_level_rewards=token_level_rewards,
                eos_mask=response_mask,
                index=index,
                utility_batches=(
                    list(utility_batches) if utility_batches is not None else None
                ),
                route_mode=os.environ.get(
                    'STACKPILOT_CR_ACTION_ROUTE', 'off'
                ),
                trajectory_aggregation=os.environ.get(
                    'STACKPILOT_CR_TRAJECTORY_AGGREGATION', 'mean'
                ),
                coefficient=float(os.environ.get(
                    'STACKPILOT_CR_ACTION_COEFFICIENT', '0.0'
                )),
                clip=float(os.environ.get(
                    'STACKPILOT_CR_ACTION_CLIP', '2.0'
                )),
            )
        )
        query_batches = data.non_tensor_batch.get('stackpilot_search_queries')
"""
    text = replace_once(text, advantage_anchor, advantage_replacement, "action utility shaping")

    merge_anchor = """        signature_array = np.empty(len(bq_signatures), dtype=object)
"""
    merge_replacement = """        bq_metrics.update(cr_metrics)
        for bq_row in bq_rows:
            uid = str(bq_row.get('uid', ''))
            if uid in cr_group_rows:
                bq_row.update(cr_group_rows[uid])
        signature_array = np.empty(len(bq_signatures), dtype=object)
"""
    text = replace_once(text, merge_anchor, merge_replacement, "credit-routing telemetry merge")
    target.write_text(text, encoding="utf-8")
    print(f"Applied credit-routing trainer patch: {target}")


def patch(root: Path) -> None:
    patch_generation(root)
    patch_trainer(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-r1-root", required=True)
    args = parser.parse_args()
    patch(Path(args.search_r1_root).resolve())
