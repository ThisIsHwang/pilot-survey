from __future__ import annotations

import unittest

from stackpilot.experiment_registry import experiment_by_id, load_registry, make_run_id


class ExperimentRegistryTest(unittest.TestCase):
    def test_registry_contains_numbered_followup_experiments(self) -> None:
        registry = load_registry()
        identifiers = [entry["id"] for entry in registry["experiments"]]
        self.assertEqual(identifiers, ["EXP-001", "EXP-002", "EXP-003", "EXP-004", "EXP-005", "EXP-006", "EXP-009", "EXP-010", "EXP-011", "EXP-012", "EXP-013", "EXP-015", "EXP-016", "EXP-017", "EXP-018", "EXP-019"])

    def test_run_id_is_stable_and_seed_is_zero_padded(self) -> None:
        self.assertEqual(make_run_id("EXP-003", seed=13, profile="pilot", variant="mixed blind"), "EXP-003__seed-013__profile-pilot__variant-mixed-blind")

    def test_parent_chain_is_registered(self) -> None:
        registry = load_registry()
        expected = {"EXP-003": "EXP-002", "EXP-006": "EXP-003", "EXP-009": "EXP-002", "EXP-010": "EXP-009", "EXP-011": "EXP-009", "EXP-012": "EXP-010", "EXP-013": "EXP-012", "EXP-015": "EXP-013", "EXP-016": "EXP-015", "EXP-017": "EXP-016", "EXP-018": "EXP-016", "EXP-019": "EXP-016"}
        for experiment_id, parent in expected.items():
            self.assertEqual(experiment_by_id(registry, experiment_id)["parent"], parent)

    def test_unknown_experiment_is_rejected(self) -> None:
        with self.assertRaises(KeyError):
            experiment_by_id(load_registry(), "EXP-999")


if __name__ == "__main__":
    unittest.main()
