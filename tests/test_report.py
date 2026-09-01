from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import event_store  # noqa: E402
import report  # noqa: E402


class ReportTests(unittest.TestCase):
    def test_evaluation_evidence_counts_separate_migrated_and_live(self):
        counts = report.evaluation_evidence_counts(event_store.load_events())
        self.assertEqual(counts["migrated_archived_evidence"], 60)
        self.assertGreater(counts["live_runtime_evidence"], 0)

    def test_current_champion_reports_its_live_evidence_class(self):
        rendered = report.build()
        self.assertIn("| Evidence class | live runtime evidence |", rendered)


if __name__ == "__main__":
    unittest.main()
