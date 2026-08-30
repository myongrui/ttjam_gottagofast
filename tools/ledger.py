#!/usr/bin/env python3
"""Append-only record of every candidate the loop has evaluated.

The ledger is what stops the loop from being a random walk. Each iteration reads
it before proposing anything, so a hypothesis that has already been tried and
lost is not tried again, and a regression is visible as a regression rather than
being silently re-explored.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, "results", "ledger.jsonl")
FULL_CASES = "1,2,3,4,5,6,7,8,9,10,11,12,13"
DEFAULT_DTYPE = "float32"
BEAM_SIZE = 3


def globally_eligible(entry: Dict[str, Any]) -> bool:
    """Whether a full run has enough evidence to become a global parent."""
    decision = entry.get("decision")
    # Pre-portfolio entries are retained as legacy evidence. New entries need a
    # paired promotion decision, except for the first valid bootstrap run.
    return decision is None or decision in ("legacy", "promote", "bootstrap")


def record(entry: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps(entry) + "\n")


def load() -> List[Dict[str, Any]]:
    if not os.path.exists(LEDGER):
        return []
    out = []
    with open(LEDGER) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def case_set(entry: Dict[str, Any]) -> str:
    return ",".join(sorted(entry.get("per_case", {}).keys(), key=int))


def champion(cases: Optional[str] = None, entries: Optional[List[Dict[str, Any]]] = None,
             dtype: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Follow the statistically proven promotion lineage.

    Legacy point estimates seed the lineage. Once paired decisions are present,
    a challenger that proved it beat the current incumbent becomes champion even
    if baseline drift made its historical baseline ratio look slightly lower.
    """
    best = None
    for e in entries if entries is not None else load():
        if e.get("score", 0) <= 0:
            continue
        if cases is not None and case_set(e) != cases:
            continue
        if dtype is not None and e.get("dtype") != dtype:
            continue
        decision = e.get("decision")
        if decision is None or decision == "legacy":
            if best is None or (
                    best.get("decision") in (None, "legacy")
                    and e["score"] > best["score"]):
                best = e
        elif decision == "bootstrap" and best is None:
            best = e
        elif decision == "promote" and (
                best is None or e.get("incumbent") == best.get("candidate")):
            best = e
    return best


def compare_on(cases: str, dtype: Optional[str] = None,
               gpu: Optional[str] = None) -> List[Dict[str, Any]]:
    """Compatible entries measured on one case set, best first."""
    rows = [
        e for e in load()
        if case_set(e) == cases
        and e.get("score", 0) > 0
        and (dtype is None or e.get("dtype") == dtype)
        and (gpu is None or e.get("gpu") == gpu)
    ]
    return sorted(rows, key=lambda e: -e["score"])


def global_beam(entries: Optional[List[Dict[str, Any]]] = None,
                size: int = BEAM_SIZE, cases: str = FULL_CASES,
                dtype: str = DEFAULT_DTYPE) -> List[Dict[str, Any]]:
    """Top distinct full-sweep candidates retained as mutation parents."""
    rows = entries if entries is not None else load()
    current = champion(cases, entries=rows, dtype=dtype)
    ranked = sorted(
        (e for e in rows
         if e.get("score", 0) > 0
         and case_set(e) == cases
         and e.get("dtype") == dtype
         and globally_eligible(e)),
        key=lambda e: -e["score"],
    )
    out = []
    seen = set()
    if current is not None:
        out.append(current)
        seen.add(current["candidate"])
    for entry in ranked:
        if entry["candidate"] in seen:
            continue
        out.append(entry)
        seen.add(entry["candidate"])
        if len(out) == size:
            break
    return out


def per_shape_elites(entries: Optional[List[Dict[str, Any]]] = None,
                     dtype: str = DEFAULT_DTYPE,
                     top_k: int = 1,
                     proven_only: bool = False) -> Dict[str, List[Dict[str, Any]]]:
    """Best valid candidates for every shape, regardless of global score.

    Exact-shape dispatch makes a locally strong implementation useful even when
    its global geometric mean loses. Keeping a short list rather than only the
    winner also preserves parent diversity and softens point-estimate noise.
    """
    rows = entries if entries is not None else load()
    by_shape: Dict[str, List[Dict[str, Any]]] = {}
    for entry in rows:
        if entry.get("dtype") != dtype or not entry.get("env_unchanged", False):
            continue
        for case, result in entry.get("per_case", {}).items():
            speedup = result.get("speedup")
            candidate_ms = result.get("candidate_median_ms")
            status = result.get("status")
            if status not in ("ok", "ok_no_baseline"):
                continue
            if (not speedup or speedup <= 0) and (not candidate_ms or candidate_ms <= 0):
                continue
            vs_incumbent = result.get("vs_incumbent") or {}
            direct_ci = vs_incumbent.get("paired_ci95") or []
            legacy_or_bootstrap = (
                entry.get("decision") is None
                or entry.get("decision") in ("legacy", "bootstrap")
            )
            direct_proven = len(direct_ci) == 2 and direct_ci[0] > 1.02
            proven = legacy_or_bootstrap or direct_proven
            if proven_only and not proven:
                continue
            by_shape.setdefault(case, []).append({
                "candidate": entry["candidate"],
                "speedup": speedup,
                "candidate_median_ms": candidate_ms,
                "paired_speedup": result.get("paired_speedup"),
                "paired_ci95": result.get("paired_ci95"),
                "proven": proven,
                "evidence": ("legacy" if legacy_or_bootstrap else
                             "proven" if direct_proven else "provisional"),
                "timestamp": entry.get("timestamp", ""),
            })

    elites: Dict[str, List[Dict[str, Any]]] = {}
    for case, candidates in by_shape.items():
        if any(row["speedup"] for row in candidates):
            candidates = [row for row in candidates if row["speedup"]]
            candidates.sort(key=lambda row: -row["speedup"])
        else:
            candidates.sort(key=lambda row: row["candidate_median_ms"])
        unique = []
        seen = set()
        for row in candidates:
            if row["candidate"] in seen:
                continue
            unique.append(row)
            seen.add(row["candidate"])
            if len(unique) == top_k:
                break
        elites[case] = unique
    return elites


def decision_from_report(report: Dict[str, Any], incumbent_before: Optional[Dict[str, Any]]) -> str:
    """Classify a run without treating a subset win as global promotion."""
    if report.get("failed_cases") or not report.get("env_unchanged", False):
        return "invalid"
    report_cases = ",".join(str(c["case"]) for c in report.get("cases", []))
    comparison = report.get("incumbent_comparison")
    if not comparison:
        if report_cases == FULL_CASES and incumbent_before is None:
            return "bootstrap"
        return "unpaired"
    paired_decision = comparison.get("decision", "uncertain")
    if report_cases != FULL_CASES:
        return f"screen_{paired_decision}"
    if incumbent_before is None:
        return "unpaired"
    if comparison.get("incumbent") != incumbent_before.get("candidate"):
        return "stale_incumbent"
    return paired_decision


def from_report(path: str, hypothesis: str = "",
                incumbent_before: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    with open(path) as f:
        r = json.load(f)
    per_case = {}
    for c in r["cases"]:
        per_case[str(c["case"])] = {
            "status": c["status"],
            "speedup": round(c["speedup"], 4) if c.get("speedup") else None,
            "candidate_median_ms": (round(c["candidate"]["median_ms"], 6)
                                    if c.get("candidate") else None),
            "paired_speedup": (round(c["paired_speedup"], 4)
                               if c.get("paired_speedup") else None),
            "paired_ci95": ([round(v, 4) for v in c["paired_speedup_ci95"]]
                            if c.get("paired_speedup_ci95") else None),
            "vs_incumbent": ({
                "paired_speedup": round(c["vs_incumbent"]["paired_speedup"], 4),
                "paired_ci95": [round(v, 4) for v in
                                c["vs_incumbent"]["paired_speedup_ci95"]],
            } if c.get("vs_incumbent") else None),
            "error": c.get("error", "")[:200],
        }
    entry = {
        "candidate": r["candidate"],
        "hypothesis": hypothesis or r.get("candidate_doc", "").split("\n")[0],
        "incumbent": r.get("incumbent"),
        "dtype": r["dtype"],
        "gpu": r["gpu"],
        "timestamp": r["timestamp"],
        "score": round(r["score"], 4),
        "geomean": round(r["geomean_speedup"], 4),
        "n_scored": r["n_scored"],
        "failed_cases": r["failed_cases"],
        "env_unchanged": r["env_unchanged"],
        "per_case": per_case,
        "incumbent_comparison": r.get("incumbent_comparison"),
    }
    entry["decision"] = decision_from_report(r, incumbent_before)
    return entry


def summary() -> str:
    entries = load()
    if not entries:
        return "ledger empty"
    lines = [f"{len(entries)} attempt(s) recorded", ""]
    full = champion(FULL_CASES, entries=entries, dtype=DEFAULT_DTYPE)
    if full:
        lines.append(f"CHAMPION (full sweep): {full['candidate']}  "
                     f"score={full['score']}  ({full['hypothesis'][:60]})")
        lines.append("")
    lines.append(f"{'candidate':<34} {'score':>7}  {'cases':>6}  {'decision':<18} failed")
    for e in entries:
        lines.append(f"{e['candidate']:<34} {e['score']:>7.4f}  "
                     f"{e['n_scored']:>6}  {e.get('decision', 'legacy'):<18} "
                     f"{e['failed_cases']}")
    # Scores are only comparable within one case set, so show those groupings.
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for e in entries:
        key = (e.get("dtype", "unknown"), e.get("gpu", "unknown"), case_set(e))
        groups.setdefault(key, []).append(e)
    for (dtype, gpu, cs), rows in sorted(
            groups.items(), key=lambda kv: -len(kv[0][2])):
        if len(rows) < 2:
            continue
        lines += ["", f"comparable on cases [{cs}] ({dtype}, {gpu}):"]
        for e in sorted(rows, key=lambda e: -e["score"]):
            lines.append(f"  {e['candidate']:<32} {e['score']:>7.4f}")

    beam = global_beam(entries)
    lines += ["", f"global beam (top {BEAM_SIZE}, full float32 sweep):"]
    if beam:
        for e in beam:
            lines.append(f"  {e['candidate']:<32} {e['score']:>7.4f}")
    else:
        lines.append("  empty")

    elites = per_shape_elites(entries, top_k=2)
    lines += ["", "per-shape elite archive (float32; top two parents):"]
    for case in sorted(elites, key=int):
        rendered = ", ".join(
            (f"{row['candidate']} {row['speedup']:.4f}x"
             if row["speedup"] else
             f"{row['candidate']} {row['candidate_median_ms']:.4f}ms")
            + f" [{row['evidence']}]"
            for row in elites[case]
        )
        lines.append(f"  case {int(case):>2}: {rendered}")
    return "\n".join(lines)


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "status"
    if command == "add":
        before = champion(FULL_CASES, dtype=DEFAULT_DTYPE)
        record(from_report(
            sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "", before))
        print("recorded")
        print(summary())
    elif command in ("status", "summary"):
        print(summary())
    elif command == "verdict":
        with open(sys.argv[2]) as f:
            report = json.load(f)
        before = champion(FULL_CASES, dtype=DEFAULT_DTYPE)
        print(decision_from_report(report, before))
    elif command == "elites":
        dtype = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_DTYPE
        print(json.dumps(per_shape_elites(dtype=dtype, top_k=2), indent=2))
    elif command == "champion":
        dtype = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_DTYPE
        best = champion(FULL_CASES, dtype=dtype)
        if best:
            print(best["candidate"])
        else:
            raise SystemExit(1)
    else:
        raise SystemExit(
            "usage: ledger.py [status|add REPORT [HYPOTHESIS]|verdict REPORT|"
            "elites [DTYPE]|champion [DTYPE]]")
