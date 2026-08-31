#!/usr/bin/env python3
"""One-time, deterministic migration of legacy autoresearch evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import event_store

ROOT = Path(__file__).resolve().parent.parent
LEGACY_LEDGER_CANDIDATES = (
    ROOT / "archive" / "legacy" / "results" / "ledger.jsonl",
    ROOT / "results" / "ledger.jsonl",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, text=True,
                          capture_output=True).stdout


def first_commit(path: Path) -> Optional[Dict[str, str]]:
    relative = str(path.relative_to(ROOT))
    output = git("log", "--follow", "--format=%H%x09%aI%x09%s", "--", relative).splitlines()
    if not output and relative.startswith("archive/legacy/"):
        original = relative.removeprefix("archive/legacy/")
        output = git("log", "--follow", "--format=%H%x09%aI%x09%s", "--", original).splitlines()
    if not output:
        return None
    commit, authored_at, subject = output[-1].split("\t", 2)
    return {"commit": commit, "authored_at": authored_at, "subject": subject}


def ledger_path() -> Path:
    for path in LEGACY_LEDGER_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("legacy ledger not found")


def read_ledger(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                row["_legacy_line"] = line_number
                rows.append(row)
    return rows


def artifact_role(path: Path) -> str:
    relative = str(path.relative_to(ROOT))
    if relative.startswith("candidates/"):
        return "candidate_source"
    if path.suffix == ".json" and path.name.startswith("v"):
        return "raw_benchmark_report"
    if "tuning" in path.name:
        return "raw_tuning_artifact"
    if path.suffix == ".log":
        return "raw_runtime_log"
    return "historical_artifact"


def artifact_event(path: Path, sequence: int) -> Dict[str, Any]:
    digest = sha256(path)
    provenance = first_commit(path)
    data: Dict[str, Any] = {
        "artifact_id": f"artifact-{digest[:16]}", "path": str(path.relative_to(ROOT)),
        "role": artifact_role(path), "sha256": digest,
        "size_bytes": path.stat().st_size, "immutable": True,
        "git_provenance": provenance,
    }
    if path.suffix == ".json":
        if not path.stat().st_size:
            data["content_status"] = "empty_legacy_artifact"
        else:
            try:
                json.loads(path.read_text())
                data["content_status"] = "valid_json"
            except json.JSONDecodeError:
                data["content_status"] = "invalid_legacy_json"
    timestamp = provenance["authored_at"] if provenance else None
    return event_store.make_event("artifact", data, sequence, timestamp,
                                  {"kind": "migration", "path": data["path"]})


def raw_match(row: Dict[str, Any], artifacts: Dict[str, Dict[str, Any]]) -> Optional[str]:
    path = ROOT / "results" / f"{Path(row['candidate']).stem}.json"
    relative = str(path.relative_to(ROOT))
    artifact = artifacts.get(relative)
    if not artifact or not path.stat().st_size:
        return None
    try:
        report = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    actual = (report.get("timestamp"), report.get("candidate"), report.get("gpu"),
              report.get("dtype"), len(report.get("cases", [])))
    expected = (row.get("timestamp"), row.get("candidate"), row.get("gpu"),
                row.get("dtype"), len(row.get("per_case", {})))
    return artifact["data"]["artifact_id"] if actual == expected else None


def git_events(sequence: int) -> List[Dict[str, Any]]:
    output = git("log", "--all", "--reverse", "--format=%H%x09%P%x09%aI%x09%s")
    events = []
    for line in output.splitlines():
        commit, parents, authored_at, subject = line.split("\t", 3)
        events.append(event_store.make_event("git_provenance", {
            "commit": commit, "parents": parents.split() if parents else [],
            "authored_at": authored_at, "subject": subject,
        }, sequence + len(events), authored_at, {"kind": "git"}))
    return events


def champion_milestones(rows: List[Dict[str, Any]], sequence: int) -> List[Dict[str, Any]]:
    milestones, scoped = [], {}
    for index, row in enumerate(rows, 1):
        if row.get("score", 0) <= 0 or event_store.case_set(row) != event_store.FULL_CASES:
            continue
        key = (row.get("gpu"), row.get("dtype"))
        current = scoped.get(key)
        decision = row.get("decision") or "legacy"
        promoted = False
        if decision == "legacy" and (current is None or current.get("decision") == "legacy"):
            promoted = current is None or row["score"] > current["score"]
        elif decision == "bootstrap" and current is None:
            promoted = True
        elif decision == "promote" and (current is None or row.get("incumbent") == current["candidate"]):
            promoted = True
        if not promoted:
            continue
        previous = current["candidate"] if current else None
        scoped[key] = {**row, "decision": decision}
        milestones.append(event_store.make_event("milestone_review", {
            "milestone_id": f"milestone-{row.get('gpu')}-{index}",
            "review_kind": "hardware_scoped_champion_transition",
            "evaluation_id": f"legacy-eval-{index:03d}",
            "hardware_id": event_store.hardware_id(row["gpu"]), "dtype": row["dtype"],
            "previous_champion": previous, "champion": row["candidate"],
            "score": row["score"], "evidence_label": "migrated_archived_evidence",
        }, sequence + len(milestones), row["timestamp"],
        {"kind": "migration", "legacy_ledger_line": row["_legacy_line"]}))
    return milestones


def build_events() -> List[Dict[str, Any]]:
    ledger = ledger_path()
    rows = read_ledger(ledger)
    events: List[Dict[str, Any]] = []

    def add(event_type: str, data: Dict[str, Any], timestamp: Optional[str] = None,
            source: Optional[Dict[str, Any]] = None) -> None:
        events.append(event_store.make_event(event_type, data, len(events) + 1,
                                             timestamp, source))

    add("migration", {"migration_id": "legacy-autoresearch-v1",
        "legacy_evaluation_count": len(rows), "contract_hash": event_store.contract_hash(),
        "evidence_label": "migrated_archived_evidence"},
        source={"kind": "migration", "path": str(ledger.relative_to(ROOT))})

    for gpu in sorted({row["gpu"] for row in rows}):
        memory_match = re.search(r"-(40|80)GB$", gpu)
        add("hardware", {"hardware_id": event_store.hardware_id(gpu),
            "accelerator": gpu, "memory_gb": int(memory_match.group(1)) if memory_match else None,
            "scope_key": f"{gpu}|{event_store.DEFAULT_DTYPE}"},
            source={"kind": "migration", "path": str(ledger.relative_to(ROOT))})

    events.extend(git_events(len(events) + 1))

    archived = list((ROOT / "archive" / "legacy").rglob("*"))
    artifact_paths = sorted(
        list((ROOT / "candidates").rglob("*.py"))
        + [path for path in (ROOT / "results").iterdir()
           if path.is_file() and path.name not in {"ledger.jsonl", "RESULTS.md", "STATUS.md"}]
        + [path for path in archived if path.is_file()],
        key=lambda path: str(path.relative_to(ROOT)),
    )
    artifacts: Dict[str, Dict[str, Any]] = {}
    for path in artifact_paths:
        event = artifact_event(path, len(events) + 1)
        events.append(event)
        artifacts[event["data"]["path"]] = event

    for path in sorted((ROOT / "candidates").rglob("*.py")):
        relative = str(path.relative_to(ROOT))
        artifact = artifacts[relative]
        provenance = artifact["data"].get("git_provenance")
        add("candidate", {
            "candidate_id": event_store.candidate_id(path.name),
            "candidate_key": path.name, "path": relative,
            "artifact_id": artifact["data"]["artifact_id"],
            "sha256": artifact["data"]["sha256"], "git_provenance": provenance,
            "historical_id_preserved": True,
        }, provenance.get("authored_at") if provenance else None,
        {"kind": "migration", "path": relative})

    for index, original in enumerate(rows, 1):
        row = dict(original)
        line_number = row.pop("_legacy_line")
        comparison = row.pop("incumbent_comparison", None)
        outcome = row.pop("decision", None) or "legacy"
        evaluation_id = f"legacy-eval-{index:03d}"
        row.update({
            "evaluation_id": evaluation_id,
            "candidate_id": event_store.candidate_id(row["candidate"]),
            "hardware_id": event_store.hardware_id(row["gpu"]),
            "raw_artifact_id": raw_match(original, artifacts),
            "raw_artifact_status": "matched" if raw_match(original, artifacts)
                                   else "not_preserved_or_overwritten_by_legacy_workflow",
            "evidence_label": "migrated_archived_evidence",
        })
        source = {"kind": "migration", "path": str(ledger.relative_to(ROOT)),
                  "legacy_ledger_line": line_number}
        add("evaluation", row, row["timestamp"], source)
        if comparison:
            add("comparison", {"comparison_id": f"legacy-comparison-{index:03d}",
                "evaluation_id": evaluation_id, "candidate": row["candidate"],
                "incumbent": row.get("incumbent"), "hardware_id": row["hardware_id"],
                "metrics": comparison}, row["timestamp"], source)
        add("decision", {"decision_id": f"legacy-decision-{index:03d}",
            "evaluation_id": evaluation_id, "outcome": outcome,
            "promotion_scope": {"hardware_id": row["hardware_id"], "dtype": row["dtype"],
                                "cases": sorted(row.get("per_case", {}), key=int)},
            "deterministic": True}, row["timestamp"], source)

    events.extend(champion_milestones(rows, len(events) + 1))
    for expected, event in enumerate(events, 1):
        event_store.validate_event(event, expected)
    return events


def validate_migration(events: List[Dict[str, Any]]) -> None:
    evaluations = [event for event in events if event["event_type"] == "evaluation"]
    if len(evaluations) != 60:
        raise ValueError(f"expected 60 evaluations, reconstructed {len(evaluations)}")
    candidate_paths = {event["data"]["path"] for event in events if event["event_type"] == "candidate"}
    disk_paths = {str(path.relative_to(ROOT)) for path in (ROOT / "candidates").rglob("*.py")}
    if candidate_paths != disk_paths:
        raise ValueError("candidate inventory mismatch")
    for event in events:
        if event["event_type"] == "artifact":
            path = ROOT / event["data"]["path"]
            if sha256(path) != event["data"]["sha256"]:
                raise ValueError(f"artifact hash mismatch: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=event_store.EVENTS)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    events = build_events()
    validate_migration(events)
    if args.check:
        print(f"migration valid: {len(events)} events, 60 evaluations")
        return 0
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite append-only store: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(event_store.canonical_json(event) + "\n" for event in events))
    print(f"wrote {len(events)} canonical events to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
