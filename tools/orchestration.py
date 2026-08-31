#!/usr/bin/env python3
"""Validate Codex/Exa research packets and append typed research events."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set
from urllib.parse import urlparse

import event_store

PACKET_SCHEMA_VERSION = 1
WORKSTREAM_ROLES = {"researcher", "skeptic"}
SOURCE_QUALITIES = {"primary", "official", "paper", "practitioner", "secondary"}
CRITIQUE_VERDICTS = {"supported", "mixed", "rejected", "needs_more_evidence"}
RESEARCH_EVENT_TYPES = {
    "research_plan", "research_delegation", "research_source", "research_finding",
    "research_critique", "research_synthesis",
}
FORBIDDEN_BENCHMARK_FIELDS = {
    "score", "geomean", "per_case", "failed_cases", "env_unchanged", "evaluation_id",
    "comparison_id", "decision_id", "outcome", "promotion_scope", "paired_speedup",
    "paired_ci95", "candidate_median_ms",
}


def _require_fields(value: Dict[str, Any], fields: Iterable[str], context: str) -> None:
    missing = set(fields) - set(value)
    if missing:
        raise ValueError(f"{context} missing fields: {sorted(missing)}")


def _require_nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _require_string_list(value: Any, context: str, allow_empty: bool = False) -> List[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError(f"{context} must be a {'possibly empty ' if allow_empty else 'non-empty '}list")
    if not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{context} must contain non-empty strings")
    return value


def _require_unique(values: Iterable[str], context: str) -> Set[str]:
    items = list(values)
    if len(items) != len(set(items)):
        raise ValueError(f"duplicate {context}")
    return set(items)


def _require_references(values: Iterable[str], known: Set[str], context: str) -> None:
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"{context} references unknown identifiers: {sorted(unknown)}")


def _validate_timestamp(value: str) -> None:
    try:
        dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("source retrieved_at must be an ISO-8601 timestamp") from error


def _validate_url(value: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("research source URL must be absolute HTTP(S)")


def _reject_benchmark_fields(value: Any, context: str) -> None:
    if isinstance(value, dict):
        forbidden = FORBIDDEN_BENCHMARK_FIELDS & set(value)
        if forbidden:
            raise ValueError(f"{context} contains benchmark-only fields: {sorted(forbidden)}")
        for key, nested in value.items():
            _reject_benchmark_fields(nested, f"{context}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_benchmark_fields(nested, f"{context}[{index}]")


def validate_research_event(event: Dict[str, Any]) -> None:
    """Validate the invariant shared by every persisted research event."""
    event_type = event.get("event_type", "")
    if not event_type.startswith("research_"):
        return
    if event_type not in RESEARCH_EVENT_TYPES:
        raise ValueError(f"unsupported research event type {event_type}")
    data = event["data"]
    if data.get("evidence_label") != "research_evidence":
        raise ValueError("research events must be labelled research_evidence")
    _reject_benchmark_fields(data, event["event_type"])
    required = {
        "research_plan": {"plan_id", "objective", "lead_role", "contract_hash"},
        "research_delegation": {"plan_id", "workstream_id", "role", "question",
                                "acceptance_criteria", "delegated_by_role"},
        "research_source": {"plan_id", "source_id", "workstream_id", "agent_role", "connector",
                            "url", "title", "retrieved_at", "quality"},
        "research_finding": {"plan_id", "finding_id", "workstream_id", "agent_role", "claim",
                             "source_ids"},
        "research_critique": {"plan_id", "critique_id", "workstream_id", "agent_role",
                              "finding_ids", "verdict", "rationale", "source_ids"},
        "research_synthesis": {"plan_id", "synthesis_id", "agent_role", "hypothesis",
                               "finding_ids", "critique_ids", "source_ids", "uncertainties"},
    }
    _require_fields(data, required[event_type] | {"evidence_label"}, event_type)
    expected_roles = {
        "research_plan": ("lead_role", "lead"),
        "research_finding": ("agent_role", "researcher"),
        "research_critique": ("agent_role", "skeptic"),
        "research_synthesis": ("agent_role", "synthesizer"),
    }
    if event_type in expected_roles:
        field, role = expected_roles[event_type]
        if data[field] != role:
            raise ValueError(f"{event_type}.{field} must be {role}")
    if event_type == "research_delegation":
        if data["role"] not in WORKSTREAM_ROLES or data["delegated_by_role"] != "lead":
            raise ValueError("research delegations require a lead and a researcher or skeptic target")
    if event_type == "research_source":
        if data["agent_role"] not in WORKSTREAM_ROLES or data["connector"] != "exa":
            raise ValueError("research sources require a researcher or skeptic using Exa")
        if event["source"].get("connector") != "exa" or event["source"].get("url") != data["url"]:
            raise ValueError("research source provenance must preserve its Exa connector and URL")


def _config() -> Dict[str, Any]:
    return event_store.contract()["orchestration"]


def build_events(packet: Dict[str, Any], existing_events: Optional[List[Dict[str, Any]]] = None,
                 confirmed_contract_hash: Optional[str] = None) -> List[Dict[str, Any]]:
    """Validate a complete packet and turn it into canonical events without writing."""
    if not isinstance(packet, dict):
        raise ValueError("research packet must be an object")
    _require_fields(packet, {"schema_version", "plan", "sources", "findings", "critiques", "synthesis"},
                    "research packet")
    if packet["schema_version"] != PACKET_SCHEMA_VERSION:
        raise ValueError(f"unsupported research packet schema {packet['schema_version']}")
    _reject_benchmark_fields(packet, "research packet")

    current_hash = event_store.contract_hash()
    if confirmed_contract_hash != current_hash:
        raise ValueError("research contract is unconfirmed or stale")
    config = _config()
    if config["implementation"] != "codex_native_subagents" or config["search_connector"] != "exa":
        raise ValueError("unsupported research orchestration contract")

    plan = packet["plan"]
    _require_fields(plan, {"plan_id", "objective", "lead_role", "contract_hash", "workstreams"}, "plan")
    plan_id = _require_nonempty_string(plan["plan_id"], "plan.plan_id")
    _require_nonempty_string(plan["objective"], "plan.objective")
    if plan["lead_role"] != "lead":
        raise ValueError("plan.lead_role must be lead")
    if plan["contract_hash"] != current_hash:
        raise ValueError("packet plan uses a stale research contract")
    workstreams = plan["workstreams"]
    if not isinstance(workstreams, list) or not workstreams:
        raise ValueError("plan.workstreams must be a non-empty list")
    if len(workstreams) > config["max_parallel_subagents"]:
        raise ValueError("plan exceeds the approved parallel subagent limit")

    workstream_ids = _require_unique(
        (_require_nonempty_string(item.get("workstream_id"), "workstream.workstream_id")
         for item in workstreams), "workstream identifiers")
    workstream_roles: Dict[str, str] = {}
    for item in workstreams:
        _require_fields(item, {"workstream_id", "role", "question", "acceptance_criteria"}, "workstream")
        if item["role"] not in WORKSTREAM_ROLES:
            raise ValueError("workstream role must be researcher or skeptic")
        _require_nonempty_string(item["question"], "workstream.question")
        _require_string_list(item["acceptance_criteria"], "workstream.acceptance_criteria")
        workstream_roles[item["workstream_id"]] = item["role"]

    sources = packet["sources"]
    if not isinstance(sources, list) or not sources:
        raise ValueError("sources must be a non-empty list")
    source_ids = _require_unique(
        (_require_nonempty_string(item.get("source_id"), "source.source_id") for item in sources),
        "source identifiers")
    _require_unique((_require_nonempty_string(item.get("url"), "source.url") for item in sources),
                    "source URLs")
    source_workstreams: Dict[str, str] = {}
    for item in sources:
        _require_fields(item, {"source_id", "workstream_id", "agent_role", "connector", "url", "title",
                               "retrieved_at", "quality"}, "source")
        _require_references([item["workstream_id"]], workstream_ids, "source")
        if item["agent_role"] != workstream_roles[item["workstream_id"]]:
            raise ValueError("source agent_role must match its delegated workstream role")
        if item["connector"] != "exa":
            raise ValueError("Exa is the only approved research connector")
        _validate_url(item["url"])
        _require_nonempty_string(item["title"], "source.title")
        _validate_timestamp(item["retrieved_at"])
        if item["quality"] not in SOURCE_QUALITIES:
            raise ValueError(f"unsupported source quality {item['quality']}")
        source_workstreams[item["source_id"]] = item["workstream_id"]

    findings = packet["findings"]
    if not isinstance(findings, list) or not findings:
        raise ValueError("findings must be a non-empty list")
    finding_ids = _require_unique(
        (_require_nonempty_string(item.get("finding_id"), "finding.finding_id") for item in findings),
        "finding identifiers")
    for item in findings:
        _require_fields(item, {"finding_id", "workstream_id", "agent_role", "claim", "source_ids"}, "finding")
        _require_references([item["workstream_id"]], workstream_ids, "finding")
        if item["agent_role"] != "researcher" or workstream_roles[item["workstream_id"]] != "researcher":
            raise ValueError("findings must come from a researcher workstream")
        _require_nonempty_string(item["claim"], "finding.claim")
        cited_sources = _require_string_list(item["source_ids"], "finding.source_ids")
        _require_references(cited_sources, source_ids, "finding")
        if any(source_workstreams[source_id] != item["workstream_id"] for source_id in cited_sources):
            raise ValueError("finding sources must belong to its researcher workstream")

    critiques = packet["critiques"]
    if not isinstance(critiques, list) or not critiques:
        raise ValueError("critiques must be a non-empty list")
    critique_ids = _require_unique(
        (_require_nonempty_string(item.get("critique_id"), "critique.critique_id") for item in critiques),
        "critique identifiers")
    for item in critiques:
        _require_fields(item, {"critique_id", "workstream_id", "agent_role", "finding_ids", "verdict",
                               "rationale", "source_ids"}, "critique")
        _require_references([item["workstream_id"]], workstream_ids, "critique")
        if item["agent_role"] != "skeptic" or workstream_roles[item["workstream_id"]] != "skeptic":
            raise ValueError("critiques must come from a skeptic workstream")
        _require_references(_require_string_list(item["finding_ids"], "critique.finding_ids"),
                            finding_ids, "critique")
        if item["verdict"] not in CRITIQUE_VERDICTS:
            raise ValueError(f"unsupported critique verdict {item['verdict']}")
        _require_nonempty_string(item["rationale"], "critique.rationale")
        cited_sources = _require_string_list(item["source_ids"], "critique.source_ids", allow_empty=True)
        _require_references(cited_sources, source_ids, "critique")
        if any(source_workstreams[source_id] != item["workstream_id"] for source_id in cited_sources):
            raise ValueError("critique sources must belong to its skeptic workstream")

    synthesis = packet["synthesis"]
    _require_fields(synthesis, {"synthesis_id", "agent_role", "hypothesis", "finding_ids", "critique_ids",
                                "source_ids", "uncertainties"}, "synthesis")
    _require_nonempty_string(synthesis["synthesis_id"], "synthesis.synthesis_id")
    if synthesis["agent_role"] != "synthesizer":
        raise ValueError("synthesis must come from the synthesizer role")
    _require_nonempty_string(synthesis["hypothesis"], "synthesis.hypothesis")
    _require_references(_require_string_list(synthesis["finding_ids"], "synthesis.finding_ids"),
                        finding_ids, "synthesis")
    _require_references(_require_string_list(synthesis["critique_ids"], "synthesis.critique_ids"),
                        critique_ids, "synthesis")
    _require_references(_require_string_list(synthesis["source_ids"], "synthesis.source_ids"),
                        source_ids, "synthesis")
    _require_string_list(synthesis["uncertainties"], "synthesis.uncertainties", allow_empty=True)

    existing = existing_events if existing_events is not None else event_store.load_events()
    existing_ids = {event["data"].get(key) for event in existing for key in
                    ("plan_id", "workstream_id", "source_id", "finding_id", "critique_id", "synthesis_id")}
    packet_ids = {plan_id, *workstream_ids, *source_ids, *finding_ids, *critique_ids, synthesis["synthesis_id"]}
    collisions = packet_ids & existing_ids
    if collisions:
        raise ValueError(f"research identifiers already exist: {sorted(collisions)}")

    sequence = len(existing) + 1
    additions: List[Dict[str, Any]] = []

    def add(event_type: str, data: Dict[str, Any], source: Optional[Dict[str, Any]] = None) -> None:
        enriched = {**data, "evidence_label": "research_evidence"}
        event = event_store.make_event(event_type, enriched, sequence + len(additions), source=source or {
            "kind": "codex_native_subagent",
        })
        validate_research_event(event)
        additions.append(event)

    add("research_plan", {key: value for key, value in plan.items() if key != "workstreams"})
    for workstream in workstreams:
        add("research_delegation", {**workstream, "plan_id": plan_id, "delegated_by_role": "lead"})
    for source in sources:
        add("research_source", {**source, "plan_id": plan_id}, {
            "kind": "codex_native_subagent", "connector": "exa", "url": source["url"],
        })
    for finding in findings:
        add("research_finding", {**finding, "plan_id": plan_id})
    for critique in critiques:
        add("research_critique", {**critique, "plan_id": plan_id})
    add("research_synthesis", {**synthesis, "plan_id": plan_id})
    return additions


def ingest_packet(path: Path, confirmed_contract_hash: str) -> List[Dict[str, Any]]:
    packet = json.loads(path.read_text())
    existing = event_store.load_events()
    additions = build_events(packet, existing, confirmed_contract_hash)
    event_store.append_events(additions)
    return additions
