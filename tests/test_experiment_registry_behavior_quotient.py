from __future__ import annotations

import unittest

from stackpilot.experiment_registry import experiment_by_id, load_registry


class BehaviorQuotientRegistryTest(unittest.TestCase):
    def test_behavior_quotient_experiments_are_registered(self) -> None:
        registry = load_registry()
        identifiers = {entry["id"] for entry in registry["experiments"]}
        expected = {"EXP-024", "EXP-025", "EXP-026", "EXP-027", "EXP-028"}
        self.assertTrue(expected.issubset(identifiers))
        self.assertEqual(experiment_by_id(registry, "EXP-024")["parent"], "EXP-020")
        self.assertEqual(experiment_by_id(registry, "EXP-025")["parent"], "EXP-024")
        self.assertEqual(experiment_by_id(registry, "EXP-026")["parent"], "EXP-023")
        self.assertEqual(experiment_by_id(registry, "EXP-027")["parent"], "EXP-024")
        self.assertEqual(experiment_by_id(registry, "EXP-028")["parent"], "EXP-027")


if __name__ == "__main__":
    unittest.main()
