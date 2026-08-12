from __future__ import annotations

import argparse
from pathlib import Path

GENERATION_MARKER = "# STACKPILOT_BEHAVIOR_QUOTIENT_GENERATION_V1"
TRAINER_MARKER = "# STACKPILOT_BEHAVIOR_QUOTIENT_TRAINER_V1"
OBSERVATION_MARKER = "# STACKPILOT_OBSERVATION_GEOMETRY_V1"

GENERATION_STATE_ANCHOR = """        self._stackpilot_retrieved_titles = [
            [] for _ in range(protocol_batch_size)
        ]

"""
GENERATION_OBSERVATION_STATE_ANCHOR = """        self._stackpilot_retrieved_titles = [
            [] for _ in range(protocol_batch_size)
        ]
        self._stackpilot_observed_titles = [
            [] for _ in range(protocol_batch_size)
        ]

"""
GENERATION_BEHAVIOR_STATE = """        # STACKPILOT_BEHAVIOR_QUOTIENT_GENERATION_V1
        self._stackpilot_search_queries = [
            [] for _ in range(protocol_batch_size)
        ]
        self._stackpilot_search_title_batches = [
            [] for _ in range(protocol_batch_size)
        ]

"""
GENERATION_STATE_REPLACEMENT = GENERATION_STATE_ANCHOR + GENERATION_BEHAVIOR_STATE
GENERATION_OBSERVATION_STATE_REPLACEMENT = (
    GENERATION_OBSERVATION_STATE_ANCHOR + GENERATION_BEHAVIOR_STATE
)

GENERATION_SEARCH_ANCHOR = """                    self._stackpilot_executed_search_counts[i] += 1
                    self._stackpilot_retrieved_titles[i].extend(
                        search_title_batches.pop(0)
                    )
"""
GENERATION_SEARCH_REPLACEMENT = """                    current_search_titles = list(search_title_batches.pop(0))
                    self._stackpilot_executed_search_counts[i] += 1
                    self._stackpilot_search_queries[i].append(contents[i])
                    self._stackpilot_search_title_batches[i].append(
                        current_search_titles
                    )
                    self._stackpilot_retrieved_titles[i].extend(
                        current_search_titles
                    )
"""

GENERATION_OUTPUT_ANCHOR = """            'stackpilot_retrieved_titles': self._stackpilot_retrieved_titles,
"""
GENERATION_OUTPUT_REPLACEMENT = GENERATION_OUTPUT_ANCHOR + """            'stackpilot_search_queries': self._stackpilot_search_queries,
            'stackpilot_search_title_batches': self._stackpilot_search_title_batches,
"""

TRAINER_IMPORT_ANCHOR = """from search_r1.llm_agent.generation import LLMGenerationManager, GenerationConfig
"""
TRAINER_IMPORT_REPLACEMENT = TRAINER_IMPORT_ANCHOR + """# STACKPILOT_BEHAVIOR_QUOTIENT_TRAINER_V1
from stackpilot.behavior_quotient_runtime import (
    append_behavior_telemetry,
    compute_behavior_advantages,
    select_behavior_rows,
)
"""

TRAINER_ADVANTAGE_ANCHOR = """        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
"""
TRAINER_ADVANTAGE_REPLACEMENT = """        query_batches = data.non_tensor_batch.get('stackpilot_search_queries')
        title_batches = data.non_tensor_batch.get(
            'stackpilot_search_title_batches'
        )
        if query_batches is None or title_batches is None:
            raise RuntimeError(
                'behavior-quotient GRPO requires structured search-query and '
                'ranked-title metadata from the rollout worker'
            )
        (
            advantages, returns, bq_metrics, bq_rows, bq_signatures,
            bq_selected_mask,
        ) = (
            compute_behavior_advantages(
                token_level_rewards=token_level_rewards,
                eos_mask=response_mask,
                index=index,
                query_batches=list(query_batches),
                title_batches=list(title_batches),
                advantage_mode=os.environ.get(
                    'STACKPILOT_BQ_ADVANTAGE_MODE', 'surface'
                ),
                selection_mode=os.environ.get(
                    'STACKPILOT_BQ_SELECTION_MODE', 'all'
                ),
                update_per_prompt=int(
                    os.environ.get('STACKPILOT_BQ_UPDATE_PER_PROMPT', '0')
                ),
                signature_mode=os.environ.get(
                    'STACKPILOT_BQ_SIGNATURE_MODE', 'trajectory-ranked'
                ),
                seed=int(os.environ.get('STACKPILOT_BQ_SELECTION_SEED', '0')),
            )
        )
        signature_array = np.empty(len(bq_signatures), dtype=object)
        signature_array[:] = bq_signatures
        data.non_tensor_batch['stackpilot_behavior_signature'] = signature_array
        data.batch['stackpilot_bq_selected_mask'] = bq_selected_mask
        data.meta_info['stackpilot_bq_metrics'] = bq_metrics
        data.meta_info['stackpilot_bq_rows'] = bq_rows
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
"""

TRAINER_CALL_ANCHOR = """                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)

"""
TRAINER_CALL_REPLACEMENT = TRAINER_CALL_ANCHOR + """                        bq_metrics = batch.meta_info.pop(
                            'stackpilot_bq_metrics', None
                        )
                        if isinstance(bq_metrics, dict):
                            metrics.update(bq_metrics)
                        bq_rows = batch.meta_info.pop('stackpilot_bq_rows', None)
                        telemetry_path = os.environ.get(
                            'STACKPILOT_BQ_TELEMETRY_PATH', ''
                        ).strip()
                        if telemetry_path and isinstance(bq_rows, list):
                            append_behavior_telemetry(
                                telemetry_path,
                                global_step=self.global_steps,
                                rows=bq_rows,
                                metadata={
                                    'experiment_id': os.environ.get(
                                        'STACKPILOT_EXPERIMENT_ID', 'unknown'
                                    ),
                                    'variant': os.environ.get(
                                        'STACKPILOT_EXPERIMENT_VARIANT', 'unknown'
                                    ),
                                    'backend': os.environ.get(
                                        'STACKPILOT_BQ_BACKEND', 'unknown'
                                    ),
                                    'seed': int(
                                        os.environ.get('STACKPILOT_BQ_RUN_SEED', '0')
                                    ),
                                    'rollout_mode': os.environ.get(
                                        'STACKPILOT_RF_ROLLOUT_MODE', 'iid'
                                    ),
                                },
                            )
                        actor_batch = select_behavior_rows(batch)
                        metrics['behavior_quotient/actor_rows'] = float(
                            len(actor_batch)
                        )
                        metrics['behavior_quotient/generated_rows'] = float(
                            len(batch)
                        )

"""

TRAINER_ACTOR_MASK_ANCHOR = """                            if self.config.do_search and self.config.actor_rollout_ref.actor.state_masking:
                                batch, metrics = self._create_loss_mask(batch, metrics)
                            actor_output = self.actor_rollout_wg.update_actor(batch)
"""
TRAINER_ACTOR_MASK_REPLACEMENT = """                            if self.config.do_search and self.config.actor_rollout_ref.actor.state_masking:
                                actor_batch, metrics = self._create_loss_mask(
                                    actor_batch, metrics
                                )
                            actor_output = self.actor_rollout_wg.update_actor(
                                actor_batch
                            )
"""


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
            "self._stackpilot_search_queries",
            "self._stackpilot_search_title_batches",
            "'stackpilot_search_queries'",
            "'stackpilot_search_title_batches'",
        )
        missing = [value for value in required if value not in text]
        if missing:
            raise RuntimeError(f"Incomplete behavior-quotient generation patch: {missing}")
        print(f"Behavior-quotient generation patch already present: {target}")
        return
    if "# STACKPILOT_STRICT_ACTION_PROTOCOL_V2" not in text:
        raise RuntimeError(
            "Apply patch_searchr1_action_protocol.py before behavior-quotient telemetry"
        )
    if OBSERVATION_MARKER in text:
        text = replace_once(
            text,
            GENERATION_OBSERVATION_STATE_ANCHOR,
            GENERATION_OBSERVATION_STATE_REPLACEMENT,
            "observation-aware protocol state",
        )
    else:
        text = replace_once(
            text,
            GENERATION_STATE_ANCHOR,
            GENERATION_STATE_REPLACEMENT,
            "protocol state",
        )
    text = replace_once(text, GENERATION_SEARCH_ANCHOR, GENERATION_SEARCH_REPLACEMENT, "search metadata")
    text = replace_once(text, GENERATION_OUTPUT_ANCHOR, GENERATION_OUTPUT_REPLACEMENT, "protocol output")
    target.write_text(text, encoding="utf-8")
    print(f"Applied behavior-quotient generation patch: {target}")


def patch_trainer(root: Path) -> None:
    target = root / "verl" / "trainer" / "ppo" / "ray_trainer.py"
    text = target.read_text(encoding="utf-8")
    if TRAINER_MARKER in text:
        required = (
            "compute_behavior_advantages(",
            "append_behavior_telemetry(",
            "stackpilot_behavior_signature",
            "STACKPILOT_BQ_ADVANTAGE_MODE",
            "stackpilot_bq_selected_mask",
            "select_behavior_rows",
        )
        missing = [value for value in required if value not in text]
        if missing:
            raise RuntimeError(f"Incomplete behavior-quotient trainer patch: {missing}")
        print(f"Behavior-quotient trainer patch already present: {target}")
        return
    text = replace_once(text, TRAINER_IMPORT_ANCHOR, TRAINER_IMPORT_REPLACEMENT, "trainer import")
    text = replace_once(text, TRAINER_ADVANTAGE_ANCHOR, TRAINER_ADVANTAGE_REPLACEMENT, "GRPO advantage")
    text = replace_once(text, TRAINER_CALL_ANCHOR, TRAINER_CALL_REPLACEMENT, "advantage call")
    text = replace_once(
        text,
        TRAINER_ACTOR_MASK_ANCHOR,
        TRAINER_ACTOR_MASK_REPLACEMENT,
        "selected actor batch",
    )
    target.write_text(text, encoding="utf-8")
    print(f"Applied behavior-quotient trainer patch: {target}")


def patch(search_r1_root: Path) -> None:
    patch_generation(search_r1_root)
    patch_trainer(search_r1_root)
    from hard_rq0.patch_searchr1_response_feedback import patch as patch_fedback

    patch_feedback(search_r1_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-r1-root", required=True)
    args = parser.parse_args()
    patch(Path(args.search_r1_root).resolve())
