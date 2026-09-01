#!/usr/bin/env python3
"""Deterministic routing over canonical autoresearch events.

Decides *what kind of move to make next*, which the portfolio search does not:
that policy selects which candidate wins, but is silent on whether the next
attempt should tune an existing structure or replace it. Everything here is
arithmetic over hardware-scoped evaluations and the roofline.

Three rules, each implementing a specific criticism of the selection-only loop:

  ROUTE   Autotuning moves a kernel toward the roofline it is already on; it
          cannot move the roofline. Gap-to-SOL therefore decides tune vs
          restructure, and gives a real stopping criterion instead of a guess.

  WIDTH   A beam of 3 over 4 candidates is "keep everything" wearing a costume.
          Beam and elite widths scale with the evidence actually available.

  BIAS    Screening on a hypothesis-chosen profile selects for candidates that
          suit that profile. We periodically force a neutral profile, and we
          measure how often a screen win fails to survive the full sweep.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import event_store
import roofline
import tuner
from race import PROFILES

ROOT = Path(__file__).resolve().parent.parent

# A shape within this factor of speed-of-light has no tuning headroom left.
SOL_DONE = 1.5
# Beyond this, the constraint is latency/structure, not parameters.
SOL_STRUCTURAL = 4.0
# Force a neutral screening profile every N cycles to bound selection bias.
NEUTRAL_EVERY = 3
# A tuning gain smaller than the measured identity-control noise is not a gain.
TUNE_NOISE_MARGIN = 1.02


@dataclass
class Decision:
    mode: str                       # TUNE | STRUCTURAL | STOP
    reason: str
    profile: str = "general"
    parent: Optional[str] = None
    target_cases: List[str] = field(default_factory=list)
    guidance: str = ""
    beam_size: int = 1
    elite_k: int = 1

    def as_dict(self) -> Dict[str, Any]:
        return {"mode": self.mode, "reason": self.reason, "profile": self.profile,
                "parent": self.parent, "target_cases": self.target_cases,
                "beam_size": self.beam_size, "elite_k": self.elite_k}


# --------------------------------------------------------------------------
# WIDTH -- scale search width to the evidence available
# --------------------------------------------------------------------------

def adaptive_widths(n_eligible: int) -> tuple[int, int]:
    """Beam and elite widths as a function of how many candidates exist.

    With two or three candidates a width-3 beam retains everything, which is not
    diversity -- it is an absence of selection pressure that costs real pod time
    on parents that have already lost. Width grows only as the pool does.
    """
    if n_eligible <= 2:
        beam = 1
    elif n_eligible <= 5:
        beam = 2
    else:
        beam = 3
    elite_k = 1 if n_eligible < 6 else 2
    return beam, elite_k


# --------------------------------------------------------------------------
# BIAS -- measure and bound screen-to-full divergence
# --------------------------------------------------------------------------

def screen_bias(entries: Optional[List[Dict[str, Any]]] = None,
                gpu: Optional[str] = None, dtype: Optional[str] = None) -> Dict[str, Any]:
    """How often a screen win fails to survive the full sweep.

    Screening is what makes the loop affordable, so the goal is not to remove it
    but to keep it honest. A candidate that screens `promote` and then fails the
    13-case sweep is evidence the screen profile flattered it.
    """
    scoped_gpu = gpu or event_store.target_gpu()
    scoped_dtype = dtype or event_store.target_dtype()
    rows = [entry for entry in (entries if entries is not None else event_store.load())
            if entry.get("gpu") == scoped_gpu and entry.get("dtype") == scoped_dtype]
    by_candidate: Dict[str, Dict[str, str]] = {}
    for e in rows:
        decision = e.get("decision") or ""
        slot = by_candidate.setdefault(e["candidate"], {})
        if decision.startswith("screen_"):
            slot["screen"] = decision[len("screen_"):]
        elif event_store.case_set(e) == event_store.FULL_CASES:
            slot["full"] = decision
    confirmed = contradicted = 0
    offenders = []
    for name, slot in by_candidate.items():
        if slot.get("screen") == "promote" and "full" in slot:
            if slot["full"] == "promote":
                confirmed += 1
            else:
                contradicted += 1
                offenders.append(name)
    total = confirmed + contradicted
    return {"confirmed": confirmed, "contradicted": contradicted,
            "rate": (contradicted / total) if total else 0.0,
            "offenders": offenders}


def choose_profile(target_cases: List[str], cycle: int) -> str:
    """Smallest profile covering the targeted cases, with periodic neutrality."""
    if cycle % NEUTRAL_EVERY == 0:
        return "general"
    if not target_cases:
        return "general"
    wanted = set(target_cases)
    best, best_cover = "general", -1.0
    for name, cases in PROFILES.items():
        members = set(cases.split(","))
        overlap = len(members & wanted)
        if not overlap:
            continue
        # Prefer high overlap, then fewer cases (cheaper to run).
        cover = overlap - 0.01 * len(members)
        if cover > best_cover:
            best, best_cover = name, cover
    return best


# --------------------------------------------------------------------------
# ROUTE -- tune, restructure, or stop
# --------------------------------------------------------------------------

def tuning_state(candidate: str) -> Dict[str, Any]:
    """Has this candidate's configuration space been swept, and did it help?

    This is the TUNE trigger rather than gap-to-SOL. Whether a candidate has
    unswept launch parameters is a fact readable from its source; the roofline
    gap is a modelled estimate, and the current model is optimistic enough that
    the TUNE band is unreachable. Prefer the fact.
    """
    path = ROOT / "candidates" / candidate
    if not path.exists():
        return {"tunable": False, "reason": "candidate file missing"}
    sites = tuner.find_sites(path.read_text())
    if not sites["kernels"] or not sites["num_warps"]:
        return {"tunable": False, "sites": sites,
                "reason": "no triton kernel with a launch parameter to sweep"}
    artifact_events = event_store.events_of("artifact")
    artifact = next((event for event in reversed(artifact_events)
                     if event["data"].get("path") ==
                     f"results/tuning_{path.stem}_best.json"), None)
    if artifact is None:
        return {"tunable": True, "swept": False, "sites": sites,
                "reason": f"{sites['kernels']} kernel(s), {sites['num_warps']} "
                          f"launch site(s), never swept"}
    record = ROOT / artifact["data"]["path"]
    best = json.loads(record.read_text())
    gain = best.get("global_geomean", 0.0)
    return {"tunable": True, "swept": True, "sites": sites,
            "global_geomean": gain,
            "reason": f"already swept (best geomean {gain:.4f})"}


def plateau_length(entries: Optional[List[Dict[str, Any]]] = None,
                   gpu: Optional[str] = None, dtype: Optional[str] = None) -> int:
    """Consecutive full-sweep runs since the last promotion."""
    scoped_gpu = gpu or event_store.target_gpu()
    scoped_dtype = dtype or event_store.target_dtype()
    rows = [entry for entry in (entries if entries is not None else event_store.load())
            if entry.get("gpu") == scoped_gpu and entry.get("dtype") == scoped_dtype]
    n = 0
    for e in reversed(rows):
        if event_store.case_set(e) != event_store.FULL_CASES:
            continue
        if e.get("decision") in ("promote", "bootstrap"):
            break
        n += 1
    return n


def champion_report(champ: Optional[Dict[str, Any]]) -> Optional[Path]:
    """Locate the incumbent's full-sweep report by content, not by filename.

    Reports written by hand carry arbitrary names, so matching on the embedded
    candidate field and the full case set is the only reliable lookup.
    """
    if not champ:
        return None
    want = champ["candidate"]
    newest, newest_mtime = None, -1.0
    for path in (ROOT / "results").rglob("*.json"):
        try:
            with path.open() as f:
                r = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        # results/ also holds tuning manifests, which are JSON lists.
        if not isinstance(r, dict) or r.get("candidate") != want:
            continue
        if r.get("gpu") != champ.get("gpu") or r.get("dtype") != champ.get("dtype"):
            continue
        cases = ",".join(str(c["case"]) for c in r.get("cases", []))
        if cases != event_store.FULL_CASES:
            continue
        mtime = path.stat().st_mtime
        if mtime > newest_mtime:
            newest, newest_mtime = path, mtime
    return newest


def decide(cycle: int, dtype: str = "float32",
           plateau_limit: int = 3) -> Decision:
    entries = event_store.load()
    champ = event_store.champion(event_store.FULL_CASES, dtype=dtype,
                                 gpu=event_store.target_gpu())
    eligible = [e for e in entries
                if e.get("score", 0) > 0 and e.get("gpu") == event_store.target_gpu()
                and e.get("dtype") == dtype
                and event_store.globally_eligible(e)]
    beam, elite_k = adaptive_widths(len(eligible))

    if champ is None:
        return Decision("STRUCTURAL", "no incumbent yet; establish one",
                        profile="general", beam_size=beam, elite_k=elite_k,
                        guidance="No champion exists. Propose a straightforward, "
                                 "correct optimization to establish a baseline.")

    # Tuning is mechanical and costs no tokens, so exhaust it before paying a
    # model to invent a new structure -- but only when there is something to
    # tune. A pure-PyTorch champion has no launch parameters at all.
    tune = tuning_state(champ["candidate"])
    if tune.get("tunable") and not tune.get("swept"):
        return Decision(
            "TUNE", f"champion has unswept launch parameters: {tune['reason']}",
            profile="launch", parent=champ["candidate"],
            beam_size=beam, elite_k=elite_k,
            guidance="Mechanical sweep of num_warps/num_stages; no model call.")

    report = champion_report(champ)
    if report is None:
        return Decision("STRUCTURAL", "no full report for the incumbent",
                        profile="general", parent=champ["candidate"],
                        beam_size=beam, elite_k=elite_k)

    gaps = roofline.gaps(report)
    done = [c for c, v in gaps.items() if v[2] < SOL_DONE]
    tunable = {c: v for c, v in gaps.items() if SOL_DONE <= v[2] < SOL_STRUCTURAL}
    structural = {c: v for c, v in gaps.items() if v[2] >= SOL_STRUCTURAL}
    stalled = plateau_length(entries, gpu=event_store.target_gpu(), dtype=dtype)

    if len(done) == len(gaps):
        return Decision("STOP", "every shape is within 1.5x of speed-of-light",
                        parent=champ["candidate"], beam_size=beam, elite_k=elite_k)

    if structural:
        # Target the worst offenders; those have the most headroom on the table.
        worst = sorted(structural.items(), key=lambda kv: -kv[1][2])[:3]
        targets = [c for c, _ in worst]
        detail = ", ".join(f"case {c} {v[2]:.0f}x off SOL ({v[3]}-bound)"
                           for c, v in worst)
        reason = f"{len(structural)}/{len(gaps)} shapes >{SOL_STRUCTURAL}x off SOL"
        if stalled >= plateau_limit:
            reason += f"; {stalled} runs without promotion"
        return Decision(
            "STRUCTURAL", reason,
            profile=choose_profile(targets, cycle), parent=champ["candidate"],
            target_cases=targets, beam_size=beam, elite_k=elite_k,
            guidance=(
                f"These shapes are latency-bound, not parameter-bound: {detail}. "
                "Neither bandwidth nor FLOPs is the constraint, so tile sizes and "
                "block dimensions will not help. Propose a STRUCTURAL change: "
                "eliminate kernel launches, fuse adjacent operations, remove "
                "host-device synchronization, or replace the algorithm."))

    worst = sorted(tunable.items(), key=lambda kv: -kv[1][2])[:3]
    targets = [c for c, _ in worst]
    return Decision(
        "TUNE", f"{len(tunable)} shapes within {SOL_STRUCTURAL}x of SOL",
        profile=choose_profile(targets, cycle), parent=champ["candidate"],
        target_cases=targets, beam_size=beam, elite_k=elite_k,
        guidance=("These shapes are close to their roofline; the structure is "
                  "right and the parameters are not. Propose a TUNING change: "
                  "block sizes, num_warps, num_stages, vectorization, or "
                  "occupancy -- not a new algorithm."))


if __name__ == "__main__":
    import sys
    d = decide(cycle=int(sys.argv[1]) if len(sys.argv) > 1 else 1)
    print(json.dumps(d.as_dict(), indent=2))
    print("\nguidance:", d.guidance)
    print("\nscreen bias:", json.dumps(screen_bias(), indent=2))
