from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import build_dispatcher  # noqa: E402
import event_store as ledger  # noqa: E402
import search_stats  # noqa: E402


def entry(candidate, score, per_case, decision=None, dtype="float32"):
    value = {
        "candidate": candidate,
        "score": score,
        "dtype": dtype,
        "gpu": "NVIDIA A100-SXM4-80GB",
        "env_unchanged": True,
        "per_case": per_case,
    }
    if decision is not None:
        value["decision"] = decision
    return value


class TimingStatisticsTests(unittest.TestCase):
    def test_paired_comparison_uses_round_ratios(self):
        baseline = {
            "median_ms": 12.0,
            "wall_ms": 12.5,
            "round_medians_ms": [10.0, 12.0, 14.0],
        }
        candidate = {
            "median_ms": 10.0,
            "wall_ms": 10.0,
            "round_medians_ms": [8.0, 10.0, 12.0],
        }
        result = search_stats.compare_timings(baseline, candidate, seed=7)
        expected = search_stats.geomean([1.25, 1.2, 14.0 / 12.0])
        self.assertAlmostEqual(result["speedup"], 1.2)
        self.assertAlmostEqual(result["paired_speedup"], expected)
        self.assertLessEqual(result["paired_speedup_ci95"][0], expected)
        self.assertGreaterEqual(result["paired_speedup_ci95"][1], expected)

    def test_bootstrap_is_deterministic(self):
        values = [1.02, 1.04, 1.06]
        self.assertEqual(
            search_stats.bootstrap_geomean_ci(values, seed=123),
            search_stats.bootstrap_geomean_ci(values, seed=123),
        )


class LedgerPortfolioTests(unittest.TestCase):
    def test_uncertain_full_run_cannot_replace_champion(self):
        cases = {str(i): {"status": "ok", "speedup": 1.1} for i in range(1, 14)}
        promoted = entry("promoted.py", 1.15, cases, decision="promote")
        promoted["incumbent"] = "champion.py"
        rows = [
            entry("champion.py", 1.1, cases),
            entry("noisy.py", 1.2, cases, decision="uncertain"),
            promoted,
        ]
        best = ledger.champion(ledger.FULL_CASES, entries=rows, dtype="float32")
        self.assertEqual(best["candidate"], "promoted.py")

    def test_paired_promotion_beats_a_higher_historical_point_estimate(self):
        cases = {str(i): {"status": "ok", "speedup": 1.1} for i in range(1, 14)}
        promoted = entry("new.py", 1.18, cases, decision="promote")
        promoted["incumbent"] = "old.py"
        rows = [entry("old.py", 1.20, cases), promoted]
        best = ledger.champion(ledger.FULL_CASES, entries=rows, dtype="float32")
        self.assertEqual(best["candidate"], "new.py")

    def test_shape_archive_keeps_global_loser_and_latency_only_case(self):
        rows = [
            entry("global.py", 1.4, {
                "1": {"status": "ok", "speedup": 1.4},
                "14": {"status": "ok_no_baseline", "speedup": None,
                       "candidate_median_ms": 20.0},
            }),
            entry("specialist.py", 1.0, {
                "1": {"status": "ok", "speedup": 1.6,
                      "vs_incumbent": {"paired_ci95": [1.05, 1.08]}},
                "14": {"status": "ok_no_baseline", "speedup": None,
                       "candidate_median_ms": 15.0,
                       "vs_incumbent": {"paired_ci95": [1.2, 1.3]}},
            }, decision="screen_promote"),
        ]
        elites = ledger.per_shape_elites(rows, top_k=2)
        self.assertEqual(elites["1"][0]["candidate"], "specialist.py")
        self.assertEqual(elites["14"][0]["candidate"], "specialist.py")

    def test_subset_decision_never_becomes_global_promotion(self):
        report = {
            "cases": [{"case": 1}, {"case": 2}],
            "failed_cases": [],
            "env_unchanged": True,
            "incumbent_comparison": {"decision": "promote", "incumbent": "old.py"},
        }
        incumbent = {"candidate": "old.py"}
        self.assertEqual(
            ledger.decision_from_report(report, incumbent), "screen_promote")


class DispatcherTests(unittest.TestCase):
    def test_generated_dispatcher_is_valid_python(self):
        source = build_dispatcher.render("float32")
        compile(source, "generated_dispatcher.py", "exec")
        self.assertIn("CONFIG_TO_CANDIDATE", source)
        self.assertIn("load_from_baseline", source)


if __name__ == "__main__":
    unittest.main()
