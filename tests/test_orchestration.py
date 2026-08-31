from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import event_store  # noqa: E402
import orchestration  # noqa: E402


def packet():
    contract_hash = event_store.contract_hash()
    return {
        "schema_version": 1,
        "plan": {
            "plan_id": "plan-test",
            "objective": "Find a falsifiable optimization direction",
            "lead_role": "lead",
            "contract_hash": contract_hash,
            "workstreams": [
                {"workstream_id": "ws-research", "role": "researcher",
                 "question": "Which mechanism applies?", "acceptance_criteria": ["primary source"]},
                {"workstream_id": "ws-skeptic", "role": "skeptic",
                 "question": "What invalidates it?", "acceptance_criteria": ["independent review"]},
            ],
        },
        "sources": [
            {"source_id": "src-1", "workstream_id": "ws-research", "agent_role": "researcher",
             "connector": "exa", "url": "https://example.com/source", "title": "Source",
             "retrieved_at": "2026-08-31T00:00:00Z", "quality": "primary"},
        ],
        "findings": [
            {"finding_id": "finding-1", "workstream_id": "ws-research", "agent_role": "researcher",
             "claim": "A mechanism may apply.", "source_ids": ["src-1"]},
        ],
        "critiques": [
            {"critique_id": "critique-1", "workstream_id": "ws-skeptic", "agent_role": "skeptic",
             "finding_ids": ["finding-1"], "verdict": "needs_more_evidence",
             "rationale": "No benchmark-specific measurement.", "source_ids": []},
        ],
        "synthesis": {
            "synthesis_id": "synthesis-1", "agent_role": "synthesizer",
            "hypothesis": "A falsifiable hypothesis.", "finding_ids": ["finding-1"],
            "critique_ids": ["critique-1"], "source_ids": ["src-1"],
            "uncertainties": ["No live benchmark evidence"],
        },
    }


class ResearchOrchestrationTests(unittest.TestCase):
    def build(self, value=None):
        return orchestration.build_events(
            value or packet(), existing_events=[],
            confirmed_contract_hash=event_store.contract_hash())

    def test_packet_records_typed_exa_provenance(self):
        events = self.build()
        self.assertEqual([event["event_type"] for event in events], [
            "research_plan", "research_delegation", "research_delegation", "research_source",
            "research_finding", "research_critique", "research_synthesis",
        ])
        self.assertTrue(all(event["data"]["evidence_label"] == "research_evidence"
                            for event in events))
        source = next(event for event in events if event["event_type"] == "research_source")
        self.assertEqual(source["source"]["connector"], "exa")
        self.assertEqual(source["source"]["url"], "https://example.com/source")

    def test_contract_selects_native_codex_exa_and_runpod(self):
        contract = event_store.contract()
        self.assertEqual(contract["orchestration"]["implementation"], "codex_native_subagents")
        self.assertEqual(contract["orchestration"]["search_connector"], "exa")
        self.assertEqual(contract["execution"]["provider"], "runpod")
        self.assertTrue(contract["execution"]["paid_gpu_requires_separate_authorization"])

    def test_non_exa_connector_is_rejected(self):
        value = packet()
        value["sources"][0]["connector"] = "generic_web"
        with self.assertRaisesRegex(ValueError, "Exa is the only approved"):
            self.build(value)

    def test_role_separation_is_enforced(self):
        value = packet()
        value["critiques"][0]["agent_role"] = "researcher"
        with self.assertRaisesRegex(ValueError, "skeptic workstream"):
            self.build(value)

    def test_findings_cannot_claim_another_workstreams_sources(self):
        value = packet()
        value["sources"].append({
            "source_id": "src-skeptic", "workstream_id": "ws-skeptic", "agent_role": "skeptic",
            "connector": "exa", "url": "https://example.com/counter", "title": "Counterevidence",
            "retrieved_at": "2026-08-31T00:01:00Z", "quality": "paper",
        })
        value["findings"][0]["source_ids"] = ["src-skeptic"]
        with self.assertRaisesRegex(ValueError, "researcher workstream"):
            self.build(value)

    def test_contract_confirmation_is_required(self):
        with self.assertRaisesRegex(ValueError, "unconfirmed or stale"):
            orchestration.build_events(packet(), existing_events=[], confirmed_contract_hash="stale")

    def test_research_packet_cannot_contain_benchmark_evidence(self):
        value = packet()
        value["synthesis"]["score"] = 9.9
        with self.assertRaisesRegex(ValueError, "benchmark-only fields"):
            self.build(value)

    def test_direct_append_cannot_bypass_research_schema(self):
        unknown = event_store.make_event(
            "research_unknown", {"evidence_label": "research_evidence"}, 1)
        with self.assertRaisesRegex(ValueError, "unsupported research event type"):
            event_store.validate_event(unknown, 1)

        incomplete = event_store.make_event(
            "research_source", {"evidence_label": "research_evidence"}, 1)
        with self.assertRaisesRegex(ValueError, "missing fields"):
            event_store.validate_event(incomplete, 1)

    def test_research_events_do_not_enter_benchmark_projection(self):
        historical = event_store.load_events()
        before = event_store.load(historical)
        research = self.build()
        after = event_store.load(historical + research)
        self.assertEqual(after, before)
        self.assertEqual(len(after), 60)

    def test_duplicate_urls_and_unknown_sources_are_rejected(self):
        duplicate = packet()
        duplicate["sources"].append(copy.deepcopy(duplicate["sources"][0]))
        duplicate["sources"][1]["source_id"] = "src-2"
        with self.assertRaisesRegex(ValueError, "duplicate source URLs"):
            self.build(duplicate)

        unknown = packet()
        unknown["findings"][0]["source_ids"] = ["src-missing"]
        with self.assertRaisesRegex(ValueError, "unknown identifiers"):
            self.build(unknown)


if __name__ == "__main__":
    unittest.main()
