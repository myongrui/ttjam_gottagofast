from __future__ import annotations

import hashlib
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import event_store  # noqa: E402
import migrate_history  # noqa: E402
import controller  # noqa: E402


class CanonicalMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.events = event_store.load_events()
        cls.migrated_events = [
            event for event in cls.events
            if event["source"].get("kind") == "migration"
        ]
        cls.rows = event_store.load(cls.migrated_events)

    def test_all_sixty_evaluations_are_reconstructed(self):
        self.assertEqual(len(self.rows), 60)
        self.assertEqual(len({row["evaluation_id"] for row in self.rows}), 60)
        self.assertEqual(len(event_store.events_of("decision", self.migrated_events)), 60)
        self.assertEqual(len(event_store.events_of("comparison", self.migrated_events)), 48)

    def test_v038_is_the_a100_80gb_champion(self):
        champion = event_store.champion(event_store.FULL_CASES, entries=self.rows,
                                       dtype="float32", gpu="NVIDIA A100-SXM4-80GB")
        self.assertEqual(champion["candidate"],
                         "v038_torchscript_dispatched_transformer_candidate.py")
        self.assertEqual(champion["score"], 2.2969)

    def test_hardware_scoped_specialist_portfolio_is_preserved(self):
        elites = event_store.per_shape_elites(
            self.rows, dtype="float32", top_k=1, proven_only=True,
            gpu="NVIDIA A100-SXM4-80GB")
        selected = {case: rows[0]["candidate"] for case, rows in elites.items()}
        self.assertEqual(selected, {
            "1": "v023_hoisted_attention_mask_candidate.py",
            "2": "v038_torchscript_dispatched_transformer_candidate.py",
            "3": "v038_torchscript_dispatched_transformer_candidate.py",
            "4": "v038_torchscript_dispatched_transformer_candidate.py",
            "5": "v038_torchscript_dispatched_transformer_candidate.py",
            "6": "v038_torchscript_dispatched_transformer_candidate.py",
            "7": "v038_torchscript_dispatched_transformer_candidate.py",
            "8": "v007_fuse_every_residual_add.py",
            "9": "v024_centralized_attention_mask_dispatch.py",
            "10": "v038_torchscript_dispatched_transformer_candidate.py",
            "11": "v038_torchscript_dispatched_transformer_candidate.py",
            "12": "v038_torchscript_dispatched_transformer_candidate.py",
            "13": "v041_single_token_attention_specialization.py",
        })
        self.assertTrue(all(row["gpu"] == "NVIDIA A100-SXM4-80GB"
                            for rows in elites.values() for row in rows))

    def test_a100_40gb_evidence_cannot_enter_80gb_selection(self):
        beam = event_store.global_beam(self.rows, dtype="float32",
                                      gpu="NVIDIA A100-SXM4-80GB")
        self.assertTrue(beam)
        self.assertTrue(all(row["gpu"] == "NVIDIA A100-SXM4-80GB" for row in beam))
        eighty = event_store.per_shape_elites(self.rows, dtype="float32", top_k=1,
                                               proven_only=True,
                                               gpu="NVIDIA A100-SXM4-80GB")
        forty = event_store.per_shape_elites(self.rows, dtype="float32", top_k=1,
                                              proven_only=True,
                                              gpu="NVIDIA A100-SXM4-40GB")
        self.assertEqual(eighty["3"][0]["candidate"],
                         "v038_torchscript_dispatched_transformer_candidate.py")
        self.assertEqual(forty["3"][0]["candidate"],
                         "v047_homogeneous_coordinate_ffn_bias.py")

    def test_candidates_through_v048_and_duplicate_v013_are_preserved(self):
        candidates = event_store.events_of("candidate", self.migrated_events)
        top_level = [event for event in candidates
                     if event["data"]["path"].count("/") == 1]
        self.assertEqual(len(top_level), 50)
        self.assertIn("v048", {event["data"]["candidate_id"] for event in top_level})
        self.assertEqual(sum(event["data"]["candidate_id"] == "v013" for event in top_level), 2)

    def test_immutable_artifact_hashes_still_match(self):
        for event in event_store.events_of("artifact", self.migrated_events):
            path = ROOT / event["data"]["path"]
            self.assertTrue(path.exists(), path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, event["data"]["sha256"], path)

    def test_migration_can_be_reconstructed_deterministically(self):
        rebuilt = migrate_history.build_events()
        migrate_history.validate_migration(rebuilt)
        self.assertEqual(len([event for event in rebuilt
                              if event["event_type"] == "evaluation"]), 60)

    def test_stale_or_unconfirmed_contract_hash_is_rejected(self):
        with self.assertRaises(SystemExit):
            controller.require_contract_hash("not-the-current-contract")

    def test_iteration_refuses_to_overwrite_raw_evidence(self):
        protected = ROOT / "archive" / "legacy" / "results" / "STATUS.md"
        completed = subprocess.run([
            str(ROOT / "tools" / "iterate.sh"), "candidates/v038_torchscript_dispatched_transformer_candidate.py",
            "1", "float32", "", str(protected.relative_to(ROOT)),
        ], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 6)
        self.assertIn("REFUSING_TO_OVERWRITE_IMMUTABLE_REPORT", completed.stdout)


if __name__ == "__main__":
    unittest.main()
