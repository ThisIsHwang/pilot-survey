from __future__ import annotations

import argparse
from pathlib import Path

MARKER = "# STACKPILOT_EXHAUSTIVE_SEARCH_VALIDATION_V2"
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
OLD_SEARCH_START = """\
                    first_input_ids = test_gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone()
"""
NEW_SEARCH_START = """\
                    # STACKPILOT_EXHAUSTIVE_SEARCH_VALIDATION_V2
                    test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(
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


def patch(search_r1_root: Path) -> Path:
    path = search_r1_root / "verl" / "trainer" / "ppo" / "ray_trainer.py"
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        if NEW not in text or NEW_SEARCH_GENERATION not in text:
            raise RuntimeError(
                f"Search-R1 exhaustive-validation marker is incomplete in {path}"
            )
        return path
    text = replace_verified(text, OLD, NEW, "validation DataLoader")
    text = replace_verified(
        text,
        OLD_SEARCH_START,
        NEW_SEARCH_START,
        "search-validation input",
    )
    text = replace_verified(
        text,
        OLD_SEARCH_GENERATION,
        NEW_SEARCH_GENERATION,
        "search-validation generation",
    )
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-r1-root", required=True)
    args = parser.parse_args()
    path = patch(Path(args.search_r1_root))
    print(f"Search-R1 exhaustive-validation patch ready: {path}")


if __name__ == "__main__":
    main()
