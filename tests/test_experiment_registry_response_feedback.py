from __future__ import annotations

import unittest

from stackpilot.experiment_registry import experiment_by_id, load_registry


class ResponseFeedbackRegistryTest(unittest.TestCase):
    def test_response_feedback_experiments_are_registered(self) -> None:
        registry = load_registry()
        identifiers = {entry["id"] for entry in registry["experiments"]}
        expected = {"EXP-029", "EXP-030", "EXP-031", "EXP-032", "EXP-033", "EXP-034"}
        self.assertTrue(expected.issubset(identifiers))
        self.assertEqual(experiment_by_id(registry, "EXP-029")["parent"], "EXP-024")
        self.assertEqual(experiment_by_id(registry, "EXP-030")["parent"], "EXP-029")
        self.assertEqual(experiment_by_id(registry, "EXP-031")["parent"], "EXP-030")
        self.assertEqual(experiment_by_id(registry, "EXP-032")["parent"], "EXP-022")
        self.assertEqual(experiment_by_id(registry, "EXP-033")["parent"], "EXP-030")
        self.assertEqual(experiment_by_id(registry, "EXP-034")["parent"], "EXP-021")


if __name__ == "__main__":
    unittest.main()
