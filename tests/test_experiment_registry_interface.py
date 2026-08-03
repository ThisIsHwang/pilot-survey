from __future__ import annotations

import unittest

from stackpilot.experiment_registry import experiment_by_id, load_registry


class InterfaceExperimentRegistryTest(unittest.TestCase):
    def test_interface_experiments_are_registered(self) -> None:
        registry = load_registry()
        identifiers = {entry["id"] for entry in registry["experiments"]}
        self.assertTrue({"EXP-020", "EXP-021", "EXP-022", "EXP-023"}.issubset(identifiers))
        self.assertEqual(experiment_by_id(registry, "EXP-020")["parent"], "EXP-019")
        self.assertEqual(experiment_by_id(registry, "EXP-021")["parent"], "EXP-020")
        self.assertEqual(experiment_by_id(registry, "EXP-022")["parent"], "EXP-020")
        self.assertEqual(experiment_by_id(registry, "EXP-023")["parent"], "EXP-020")


if __name__ == "__main__":
    unittest.main()
