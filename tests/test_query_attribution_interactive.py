from __future__ import annotations

import unittest

from stackpilot.query_attribution_interactive_job import clean_query, recall


class QueryAttributionInteractiveTests(unittest.TestCase):
    def test_query_cleanup(self) -> None:
        self.assertEqual(clean_query('<search> Gold A creator </search>'), 'Gold A creator')
        self.assertEqual(clean_query('"Gold A creator"\nextra'), 'Gold A creator')

    def test_recall(self) -> None:
        self.assertEqual(recall(["Gold A", "Gold B"], ["gold a"]), 0.5)


if __name__ == "__main__":
    unittest.main()
