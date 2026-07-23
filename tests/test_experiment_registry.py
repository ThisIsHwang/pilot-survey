from __future__ import annotations

import unittest

from stackpilot.experiment_registry import (
    experiment_by_id,
    load_registry,
    make_run_id,
)


class ExperimentRegistryTest(unittest.TestCase):
    def test_registry_contains_numbered_followup_experiments(self) -> None:
        registry = load_registry()
        identifiers = [entry["id"] for entry in registry["experiments"]]
        self.assertEqual(
            identifiers,
            ["EXP-001", "EXP-002", "EXP-003", "EXP-004", "EXP-005", "EXP-006"],
        )

    def test_run_id_is_stable_and_seed_is_zero_padded(self) -> None:
        self.assertEqual(
            make_run_id("EXP-003", seed=13, profile="pilot", variant="mixed blind"),
            "EXP-003__seed-013__profile-pilot__variant-mixed-blind",
        )

    def test_parent_chain_is_registered(self) -> None:
        registry = load_registry()
        self.assertEqual(experiment_by_id(registry, "EXP-003")["parent"], "EXP-002")
        self.assertEqual(experiment_by_id(registry, "EXP-006")["parent"], "EXP-003")

    def test_unknown_experiment_is_rejected(self) -> None:
        with self.assertRaises(KeyError):
            experiment_by_id(load_registry(), "EXP-999")


if __name__ == "__main__":
    unittest.main()
