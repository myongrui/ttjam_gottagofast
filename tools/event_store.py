#!/usr/bin/env python3
"""Canonical append-only autoresearch history and deterministic projections."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parent.parent
EVENTS = ROOT / "research" / "events.jsonl"
CONTRACT = ROOT / "research.yaml"
FULL_CASES = "1,2,3,4,5,6,7,8,9,10,11,12,13"
DEFAULT_DTYPE = "float32"
BEAM_SIZE = 3
SCHEMA_VERSION = 1


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def contract() -> Dict[str, Any]:
    return json.loads(CONTRACT.read_text())


def contract_hash() -> str:
    return hashlib.sha256(CONTRACT.read_bytes()).hexdigest()


def target_gpu() -> str:
    return contract()["target"]["hardware"]["accelerator"]


def target_dtype() -> str:
    return contract()["target"]["dtype"]


def hardware_id(gpu: str) -> str:
    match = re.search(r"-(40|80)GB$", gpu)
    memory = match.group(1) if match else "unknown"
    slug = re.sub(r"[^a-z0-9]+", "-", gpu.lower()).strip("-")
    return f"hw-{slug}-{memory}gb"


def candidate_id(filename: str) -> str:
    match = re.match(r"(v\d+)", Path(filename).name)
    return match.group(1) if match else Path(filename).stem


def make_event(event_type: str, data: Dict[str, Any], sequence: int,
               timestamp: Optional[str] = None, source: Optional[Dict[str, Any]] = None,
               event_id: Optional[str] = None) -> Dict[str, Any]:
    body = {"event_type": event_type, "data": data, "source": source or {"kind": "runtime"}}
    digest = hashlib.sha256(canonical_json(body).encode()).hexdigest()[:16]
    return {
        "schema_version": SCHEMA_VERSION,
        "sequence": sequence,
        "event_id": event_id or f"evt-{sequence:05d}-{digest}",
        "timestamp": timestamp or utc_now(),
        **body,
    }


def validate_event(event: Dict[str, Any], expected_sequence: Optional[int] = None) -> None:
    required = {"schema_version", "sequence", "event_id", "timestamp", "event_type", "data", "source"}
    missing = required - set(event)
    if missing:
        raise ValueError(f"event missing fields: {sorted(missing)}")
    if event["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version {event['schema_version']}")
    if expected_sequence is not None and event["sequence"] != expected_sequence:
        raise ValueError(f"expected sequence {expected_sequence}, got {event['sequence']}")
    if not isinstance(event["data"], dict) or not isinstance(event["source"], dict):
        raise ValueError("event data and source must be objects")


def load_events(path: Path = EVENTS) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    events: List[Dict[str, Any]] = []
    seen = set()
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            event = json.loads(line)
            validate_event(event, len(events) + 1)
            if event["event_id"] in seen:
                raise ValueError(f"duplicate event_id at line {line_number}: {event['event_id']}")
            seen.add(event["event_id"])
            events.append(event)
    return events


def append_events(new_events: Iterable[Dict[str, Any]], path: Path = EVENTS) -> None:
    materialized = list(new_events)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_APPEND | os.O_CREAT | os.O_RDWR
    descriptor = os.open(path, flags, 0o644)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        existing = load_events(path)
        for offset, event in enumerate(materialized, 1):
            validate_event(event, len(existing) + offset)
        blob = "".join(canonical_json(event) + "\n" for event in materialized).encode()
        written = 0
        while written < len(blob):
            written += os.write(descriptor, blob[written:])
        os.fsync(descriptor)
    except BaseException:
        raise
    finally:
        os.close(descriptor)


def events_of(event_type: str, events: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    return [event for event in (events if events is not None else load_events())
            if event["event_type"] == event_type]


def load(events: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Project canonical events into the former evaluation-row shape."""
    rows = events if events is not None else load_events()
    comparisons = {e["data"]["evaluation_id"]: e["data"]
                   for e in rows if e["event_type"] == "comparison"}
    decisions = {e["data"]["evaluation_id"]: e["data"]
                 for e in rows if e["event_type"] == "decision"}
    projected = []
    for event in rows:
        if event["event_type"] != "evaluation":
            continue
        row = dict(event["data"])
        evaluation_id = row["evaluation_id"]
        comparison = comparisons.get(evaluation_id)
        decision = decisions.get(evaluation_id)
        row["incumbent_comparison"] = (comparison.get("metrics") if comparison else None)
        row["decision"] = decision.get("outcome") if decision else None
        projected.append(row)
    return projected


def case_set(entry: Dict[str, Any]) -> str:
    return ",".join(sorted(entry.get("per_case", {}).keys(), key=int))


def globally_eligible(entry: Dict[str, Any]) -> bool:
    return entry.get("decision") in (None, "legacy", "promote", "bootstrap")


def _scope_gpu(gpu: Optional[str]) -> str:
    return gpu or target_gpu()


def champion(cases: Optional[str] = None, entries: Optional[List[Dict[str, Any]]] = None,
             dtype: Optional[str] = None, gpu: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Follow promotion lineage within exactly one hardware scope."""
    best = None
    scoped_gpu = _scope_gpu(gpu)
    for entry in entries if entries is not None else load():
        if entry.get("score", 0) <= 0 or entry.get("gpu") != scoped_gpu:
            continue
        if cases is not None and case_set(entry) != cases:
            continue
        if dtype is not None and entry.get("dtype") != dtype:
            continue
        decision = entry.get("decision")
        if decision in (None, "legacy"):
            if best is None or (best.get("decision") in (None, "legacy")
                                and entry["score"] > best["score"]):
                best = entry
        elif decision == "bootstrap" and best is None:
            best = entry
        elif decision == "promote" and (best is None or entry.get("incumbent") == best.get("candidate")):
            best = entry
    return best


def compare_on(cases: str, dtype: Optional[str] = None,
               gpu: Optional[str] = None) -> List[Dict[str, Any]]:
    scoped_gpu = _scope_gpu(gpu)
    rows = [entry for entry in load()
            if case_set(entry) == cases and entry.get("score", 0) > 0
            and entry.get("gpu") == scoped_gpu
            and (dtype is None or entry.get("dtype") == dtype)]
    return sorted(rows, key=lambda entry: -entry["score"])


def global_beam(entries: Optional[List[Dict[str, Any]]] = None, size: int = BEAM_SIZE,
                cases: str = FULL_CASES, dtype: Optional[str] = None,
                gpu: Optional[str] = None) -> List[Dict[str, Any]]:
    rows = entries if entries is not None else load()
    scoped_gpu = _scope_gpu(gpu)
    scoped_dtype = dtype or target_dtype()
    current = champion(cases, entries=rows, dtype=scoped_dtype, gpu=scoped_gpu)
    ranked = sorted((entry for entry in rows if entry.get("score", 0) > 0
                     and case_set(entry) == cases and entry.get("dtype") == scoped_dtype
                     and entry.get("gpu") == scoped_gpu and globally_eligible(entry)),
                    key=lambda entry: -entry["score"])
    output, seen = [], set()
    if current:
        output.append(current); seen.add(current["candidate"])
    for entry in ranked:
        if entry["candidate"] in seen:
            continue
        output.append(entry); seen.add(entry["candidate"])
        if len(output) == size:
            break
    return output


def per_shape_elites(entries: Optional[List[Dict[str, Any]]] = None,
                     dtype: Optional[str] = None, top_k: int = 1,
                     proven_only: bool = False, gpu: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
    rows = entries if entries is not None else load()
    scoped_gpu, scoped_dtype = _scope_gpu(gpu), dtype or target_dtype()
    by_shape: Dict[str, List[Dict[str, Any]]] = {}
    for entry in rows:
        if (entry.get("gpu") != scoped_gpu or entry.get("dtype") != scoped_dtype
                or not entry.get("env_unchanged", False)):
            continue
        for case, result in entry.get("per_case", {}).items():
            speedup, candidate_ms = result.get("speedup"), result.get("candidate_median_ms")
            if result.get("status") not in ("ok", "ok_no_baseline"):
                continue
            if (not speedup or speedup <= 0) and (not candidate_ms or candidate_ms <= 0):
                continue
            direct_ci = (result.get("vs_incumbent") or {}).get("paired_ci95") or []
            inherited = entry.get("decision") in (None, "legacy", "bootstrap")
            direct_proven = len(direct_ci) == 2 and direct_ci[0] > 1.02
            proven = inherited or direct_proven
            if proven_only and not proven:
                continue
            by_shape.setdefault(case, []).append({
                "candidate": entry["candidate"], "speedup": speedup,
                "candidate_median_ms": candidate_ms,
                "paired_speedup": result.get("paired_speedup"),
                "paired_ci95": result.get("paired_ci95"), "proven": proven,
                "evidence": "legacy" if inherited else "proven" if direct_proven else "provisional",
                "timestamp": entry.get("timestamp", ""), "gpu": scoped_gpu,
                "evaluation_id": entry.get("evaluation_id"),
            })
    elites: Dict[str, List[Dict[str, Any]]] = {}
    for case, candidates in by_shape.items():
        candidates = ([row for row in candidates if row["speedup"]]
                      if any(row["speedup"] for row in candidates) else candidates)
        candidates.sort(key=(lambda row: -row["speedup"]) if candidates[0]["speedup"]
                        else (lambda row: row["candidate_median_ms"]))
        unique, seen = [], set()
        for row in candidates:
            if row["candidate"] in seen:
                continue
            unique.append(row); seen.add(row["candidate"])
            if len(unique) == top_k:
                break
        elites[case] = unique
    return elites


def decision_from_report(report: Dict[str, Any], incumbent_before: Optional[Dict[str, Any]]) -> str:
    if report.get("failed_cases") or not report.get("env_unchanged", False):
        return "invalid"
    report_cases = ",".join(str(case["case"]) for case in report.get("cases", []))
    comparison = report.get("incumbent_comparison")
    if not comparison:
        return "bootstrap" if report_cases == FULL_CASES and incumbent_before is None else "unpaired"
    paired_decision = comparison.get("decision", "uncertain")
    if report_cases != FULL_CASES:
        return f"screen_{paired_decision}"
    if incumbent_before is None:
        return "unpaired"
    if comparison.get("incumbent") != incumbent_before.get("candidate"):
        return "stale_incumbent"
    return paired_decision


def _per_case(report: Dict[str, Any]) -> Dict[str, Any]:
    output = {}
    for case in report["cases"]:
        output[str(case["case"])] = {
            "status": case["status"],
            "speedup": round(case["speedup"], 4) if case.get("speedup") else None,
            "candidate_median_ms": (round(case["candidate"]["median_ms"], 6)
                                    if case.get("candidate") else None),
            "paired_speedup": round(case["paired_speedup"], 4) if case.get("paired_speedup") else None,
            "paired_ci95": ([round(value, 4) for value in case["paired_speedup_ci95"]]
                            if case.get("paired_speedup_ci95") else None),
            "vs_incumbent": ({"paired_speedup": round(case["vs_incumbent"]["paired_speedup"], 4),
                              "paired_ci95": [round(value, 4) for value in case["vs_incumbent"]["paired_speedup_ci95"]]}
                             if case.get("vs_incumbent") else None),
            "error": case.get("error", "")[:200],
        }
    return output


def record_report(path: str, hypothesis: str = "") -> str:
    report_path = Path(path).resolve()
    report = json.loads(report_path.read_text())
    gpu, dtype = report["gpu"], report["dtype"]
    before = champion(FULL_CASES, dtype=dtype, gpu=gpu)
    decision = decision_from_report(report, before)
    existing = load_events()
    sequence = len(existing) + 1
    evaluation_id = f"eval-{hashlib.sha256(report_path.read_bytes()).hexdigest()[:16]}-{sequence}"
    relative = str(report_path.relative_to(ROOT))
    artifact_id = f"artifact-{hashlib.sha256(report_path.read_bytes()).hexdigest()[:16]}"
    artifact = make_event("artifact", {"artifact_id": artifact_id, "path": relative,
                          "role": "raw_benchmark_report", "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
                          "size_bytes": report_path.stat().st_size, "immutable": True}, sequence)
    data = {"evaluation_id": evaluation_id, "candidate": report["candidate"],
            "candidate_id": candidate_id(report["candidate"]), "hypothesis": hypothesis or report.get("candidate_doc", "").split("\n")[0],
            "incumbent": report.get("incumbent"), "dtype": dtype, "gpu": gpu,
            "hardware_id": hardware_id(gpu), "timestamp": report["timestamp"],
            "score": round(report["score"], 4), "geomean": round(report["geomean_speedup"], 4),
            "n_scored": report["n_scored"], "failed_cases": report["failed_cases"],
            "env_unchanged": report["env_unchanged"], "per_case": _per_case(report),
            "raw_artifact_id": artifact_id, "evidence_label": "live_runtime_evidence"}
    evaluation = make_event("evaluation", data, sequence + 1, report["timestamp"])
    additions = [artifact, evaluation]
    if report.get("incumbent_comparison"):
        additions.append(make_event("comparison", {"comparison_id": f"cmp-{evaluation_id}",
                    "evaluation_id": evaluation_id, "candidate": report["candidate"],
                    "incumbent": report.get("incumbent"), "hardware_id": hardware_id(gpu),
                    "metrics": report["incumbent_comparison"]}, sequence + len(additions)))
    additions.append(make_event("decision", {"decision_id": f"decision-{evaluation_id}",
                "evaluation_id": evaluation_id, "outcome": decision, "promotion_scope": {
                    "hardware_id": hardware_id(gpu), "dtype": dtype, "cases": list(_per_case(report))},
                "deterministic": True}, sequence + len(additions)))
    append_events(additions)
    return decision


def summary() -> str:
    rows = load()
    champ = champion(FULL_CASES, dtype=target_dtype(), gpu=target_gpu())
    lines = [f"{len(rows)} evaluation(s) recorded", f"scope: {target_gpu()}, {target_dtype()}"]
    if champ:
        lines.append(f"champion: {champ['candidate']} score={champ['score']:.4f}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    sub.add_parser("validate")
    sub.add_parser("contract-hash")
    champion_parser = sub.add_parser("champion")
    champion_parser.add_argument("--dtype", default=None)
    champion_parser.add_argument("--gpu", default=None)
    add_parser = sub.add_parser("add")
    add_parser.add_argument("report")
    add_parser.add_argument("hypothesis", nargs="?", default="")
    args = parser.parse_args()
    if args.command == "status":
        print(summary())
    elif args.command == "validate":
        events = load_events(); print(f"valid: {len(events)} canonical events")
    elif args.command == "contract-hash":
        print(contract_hash())
    elif args.command == "champion":
        best = champion(FULL_CASES, dtype=args.dtype or target_dtype(), gpu=args.gpu or target_gpu())
        if not best:
            return 1
        print(best["candidate"])
    elif args.command == "add":
        print(record_report(args.report, args.hypothesis))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
