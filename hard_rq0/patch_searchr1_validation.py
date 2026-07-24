from __future__ import annotations

import argparse
import re
from pathlib import Path

LEGACY_MARKER = "# STACKPILOT_EXHAUSTIVE_SEARCH_VALIDATION_V2"
MARKER = "# STACKPILOT_EXHAUSTIVE_SEARCH_VALIDATION_V3"
OLD_IMPORT = "from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto"
NEW_IMPORT = "from verl.protocol import unpad_dataproto"
OLD_HELPER_ANCHOR = """\
class RayPPOTrainer(object):
"""
NEW_HELPER = """\
def _stackpilot_pad_validation_batch(data, size_divisor):
    # STACKPILOT_EXHAUSTIVE_SEARCH_VALIDATION_V3
    # Upstream appends data[:pad_size] only once.  If the final batch is
    # smaller than pad_size (for example one row on seven workers), that still
    # is not divisible.  Repeat deterministic prefixes until the exact padding
    # size has been added; unpadding then restores the original rows and order.
    if size_divisor < 1:
        raise ValueError(f"validation padding divisor must be positive: {size_divisor}")
    batch_size = len(data)
    if batch_size < 1:
        raise RuntimeError("cannot pad an empty validation batch")
    pad_size = (-batch_size) % size_divisor
    if pad_size == 0:
        return data, 0
    padding = []
    remaining = pad_size
    while remaining:
        take = min(remaining, batch_size)
        padding.append(data[:take])
        remaining -= take
    padded = DataProto.concat([data, *padding])
    if len(padded) % size_divisor != 0:
        raise RuntimeError(
            f"validation padding failed: {len(padded)} rows for divisor {size_divisor}"
        )
    return padded, pad_size


class RayPPOTrainer(object):
"""
OLD = """\
        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=self.config.data.val_batch_size,
                                         shuffle=False,
                                         drop_last=True,
                                         collate_fn=collate_fn)
"""
NEW = """\
        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=self.config.data.val_batch_size,
                                         shuffle=False,
                                         # Validation pads the final search batch
                                         # to the rollout world size, then unpads it.
                                         drop_last=False,
                                         collate_fn=collate_fn)
"""
OLD_NONSEARCH_PAD = """\
                test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
"""
NEW_NONSEARCH_PAD = """\
                test_gen_batch_padded, pad_size = _stackpilot_pad_validation_batch(
                    test_gen_batch, self.actor_rollout_wg.world_size
                )
"""
OLD_VALIDATION_START = """\
        if not self.config.do_search:
            for test_data in self.val_dataloader:
                test_batch = DataProto.from_single_dict(test_data)
"""
NEW_VALIDATION_START = """\
        validation_seen = 0

        if not self.config.do_search:
            for test_data in self.val_dataloader:
                validation_seen += len(test_data['input_ids'])
                test_batch = DataProto.from_single_dict(test_data)
"""
OLD_SEARCH_LOOP_START = """\
        else:
            for batch_dict in self.val_dataloader:
                timing_raw = {}
"""
NEW_SEARCH_LOOP_START = """\
        else:
            for batch_dict in self.val_dataloader:
                validation_seen += len(batch_dict['input_ids'])
                timing_raw = {}
"""
OLD_SEARCH_START = """\
                    first_input_ids = test_gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone()
"""
LEGACY_SEARCH_START = """\
                    # STACKPILOT_EXHAUSTIVE_SEARCH_VALIDATION_V2
                    test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(
                        test_gen_batch, self.actor_rollout_wg.world_size
                    )
                    first_input_ids = test_gen_batch_padded.batch['input_ids'][
                        :, -gen_config.max_start_length:
                    ].clone()
"""
NEW_SEARCH_START = """\
                    test_gen_batch_padded, pad_size = _stackpilot_pad_validation_batch(
                        test_gen_batch, self.actor_rollout_wg.world_size
                    )
                    first_input_ids = test_gen_batch_padded.batch['input_ids'][
                        :, -gen_config.max_start_length:
                    ].clone()
"""
OLD_SEARCH_GENERATION = (
    """\
                        final_gen_batch_output = generation_manager.run_llm_loop(
                            gen_batch=test_gen_batch,
                            initial_input_ids=first_input_ids,
                        )
"""
    + "                    \n"
    + """\
                    test_batch = test_batch.union(final_gen_batch_output)
"""
)
NEW_SEARCH_GENERATION = """\
                        final_gen_batch_output_padded = generation_manager.run_llm_loop(
                            gen_batch=test_gen_batch_padded,
                            initial_input_ids=first_input_ids,
                        )
                    final_gen_batch_output = unpad_dataproto(
                        final_gen_batch_output_padded, pad_size=pad_size
                    )

                    test_batch = test_batch.union(final_gen_batch_output)
"""
OLD_VALIDATION_COVERAGE = """\
        reward_tensor = torch.cat([rw.sum(-1) for rw in reward_tensor_lst], dim=0).cpu()  # (batch_size,)
        # reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
"""
NEW_VALIDATION_COVERAGE = """\
        expected_validation = len(self.val_dataset)
        if validation_seen != expected_validation:
            raise RuntimeError(
                "Validation coverage mismatch: DataLoader yielded "
                f"{validation_seen} of {expected_validation} dataset rows"
            )
        reward_tensor = torch.cat([rw.sum(-1) for rw in reward_tensor_lst], dim=0).cpu()  # (batch_size,)
        # reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        scored_validation = int(reward_tensor.shape[0])
        labeled_validation = int(data_sources.shape[0])
        if (
            scored_validation != expected_validation
            or labeled_validation != expected_validation
        ):
            raise RuntimeError(
                "Validation scoring coverage mismatch: expected "
                f"{expected_validation}, scored {scored_validation}, "
                f"labeled {labeled_validation}"
            )
"""
GENERATION_META_MARKER = "# STACKPILOT_VALIDATION_META_INFO_V1"
OLD_ACTIVE_ROLLOUT = """\
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })
"""
NEW_ACTIVE_ROLLOUT = f"""\
            {GENERATION_META_MARKER}
            rollings_active = DataProto.from_dict(
                {{k: v[active_mask] for k, v in rollings.batch.items()}},
                # Validation's do_sample=False must survive the active-row
                # filtering on every agent turn. Training carries its normal
                # (empty/default) generation metadata through the same path.
                meta_info=dict(rollings.meta_info),
            )
"""
OLD_PADDED_ACTIVE_ROLLOUT = """\
        padded_active_batch = DataProto.from_dict(padded_batch)
"""
NEW_PADDED_ACTIVE_ROLLOUT = """\
        padded_active_batch = DataProto.from_dict(
            padded_batch,
            # Active-row GPU padding must not silently restore stochastic
            # sampling during validation.
            meta_info=dict(active_batch.meta_info),
        )
"""


def replace_verified(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    occurrences = text.count(old)
    if occurrences != 1:
        raise RuntimeError(
            f"Pinned Search-R1 {label} block was not found exactly once; "
            f"found {occurrences}. Refusing an unverified patch."
        )
    return text.replace(old, new, 1)


def patch_generation_metadata(search_r1_root: Path) -> Path:
    path = search_r1_root / "search_r1" / "llm_agent" / "generation.py"
    text = path.read_text(encoding="utf-8")
    if GENERATION_META_MARKER in text:
        if (
            text.count(NEW_ACTIVE_ROLLOUT) != 2
            or text.count(NEW_PADDED_ACTIVE_ROLLOUT) != 1
        ):
            raise RuntimeError(
                f"Search-R1 validation metadata marker is incomplete in {path}"
            )
        return path

    active_pattern = re.compile(re.escape(OLD_ACTIVE_ROLLOUT.rstrip("\n")) + r"[ \t]*\n")
    active_occurrences = len(active_pattern.findall(text))
    if active_occurrences != 2:
        raise RuntimeError(
            "Pinned Search-R1 active-rollout metadata block was not found "
            f"exactly twice; found {active_occurrences}. Refusing an "
            f"unverified patch in {path}."
        )
    padded_occurrences = text.count(OLD_PADDED_ACTIVE_ROLLOUT)
    if padded_occurrences != 1:
        raise RuntimeError(
            "Pinned Search-R1 padded-rollout metadata block was not found "
            f"exactly once; found {padded_occurrences}. Refusing an "
            f"unverified patch in {path}."
        )
    text = active_pattern.sub(NEW_ACTIVE_ROLLOUT, text)
    text = text.replace(
        OLD_PADDED_ACTIVE_ROLLOUT,
        NEW_PADDED_ACTIVE_ROLLOUT,
        1,
    )
    path.write_text(text, encoding="utf-8")
    return path


def patch(search_r1_root: Path) -> Path:
    path = search_r1_root / "verl" / "trainer" / "ppo" / "ray_trainer.py"
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        required = (
            NEW_IMPORT,
            NEW_HELPER,
            NEW,
            NEW_NONSEARCH_PAD,
            NEW_VALIDATION_START,
            NEW_SEARCH_LOOP_START,
            NEW_SEARCH_START,
            NEW_SEARCH_GENERATION,
            NEW_VALIDATION_COVERAGE,
        )
        if any(block not in text for block in required):
            raise RuntimeError(
                f"Search-R1 exhaustive-validation marker is incomplete in {path}"
            )
    else:
        legacy = LEGACY_MARKER in text
        if legacy and any(
            block not in text
            for block in (NEW, LEGACY_SEARCH_START, NEW_SEARCH_GENERATION)
        ):
            raise RuntimeError(
                f"Search-R1 legacy exhaustive-validation marker is incomplete in {path}"
            )
        text = replace_verified(
            text,
            OLD_IMPORT,
            NEW_IMPORT,
            "validation protocol import",
        )
        text = replace_verified(
            text,
            OLD_HELPER_ANCHOR,
            NEW_HELPER,
            "validation padding helper",
        )
        text = replace_verified(text, OLD, NEW, "validation DataLoader")
        text = replace_verified(
            text,
            OLD_NONSEARCH_PAD,
            NEW_NONSEARCH_PAD,
            "non-search validation padding",
        )
        text = replace_verified(
            text,
            OLD_VALIDATION_START,
            NEW_VALIDATION_START,
            "non-search validation coverage",
        )
        text = replace_verified(
            text,
            OLD_SEARCH_LOOP_START,
            NEW_SEARCH_LOOP_START,
            "search validation coverage",
        )
        text = replace_verified(
            text,
            LEGACY_SEARCH_START if legacy else OLD_SEARCH_START,
            NEW_SEARCH_START,
            "search-validation input",
        )
        text = replace_verified(
            text,
            OLD_SEARCH_GENERATION,
            NEW_SEARCH_GENERATION,
            "search-validation generation",
        )
        text = replace_verified(
            text,
            OLD_VALIDATION_COVERAGE,
            NEW_VALIDATION_COVERAGE,
            "validation completion invariant",
        )
        if LEGACY_MARKER in text:
            raise RuntimeError(
                f"Search-R1 legacy exhaustive-validation marker remains in {path}"
            )
        path.write_text(text, encoding="utf-8")
    patch_generation_metadata(search_r1_root)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-r1-root", required=True)
    args = parser.parse_args()
    path = patch(Path(args.search_r1_root))
    print(f"Search-R1 exhaustive-validation patch ready: {path}")


if __name__ == "__main__":
    main()
