#!/usr/bin/env python3
"""Regenerate human-readable views from the canonical event store."""
from __future__ import annotations

import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import build_dispatcher  # noqa: E402
import event_store  # noqa: E402
import shapes as shapes_mod  # noqa: E402

OUT = ROOT / "results" / "RESULTS.md"
STATUS = ROOT / "results" / "STATUS.md"


def geomean(values: List[float]) -> float:
    values = [v for v in values if v and v > 0]
    return math.exp(sum(map(math.log, values)) / len(values)) if values else 0.0


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip() or "?"
    except Exception:
        return "?"


def evaluation_evidence_counts(events: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for event in events:
        if event["event_type"] != "evaluation":
            continue
        label = event["data"].get("evidence_label", "unlabelled")
        counts[label] = counts.get(label, 0) + 1
    return counts


def build(dtype: str | None = None) -> str:
    dtype = dtype or event_store.target_dtype()
    gpu = event_store.target_gpu()
    champ = event_store.champion(event_store.FULL_CASES, dtype=dtype, gpu=gpu)
    if not champ:
        return ("# Current best submission\n\nNo full-sweep champion yet. "
                "Use the repo-local autoresearch skill to begin a confirmed run.\n")

    elites = event_store.per_shape_elites(dtype=dtype, top_k=1, proven_only=True, gpu=gpu)
    try:
        fallback, _mapping, notes = build_dispatcher.choose_mapping(dtype)
    except Exception as e:                      # composer unavailable
        fallback, notes = champ["candidate"], []
        elites = elites or {}
        notes = [(s["case"], fallback) for s in shapes_mod.as_dicts()]
        composer_note = f"_dispatcher composer unavailable: {e}_\n"
    else:
        composer_note = ""

    champ_cases = champ.get("per_case", {})
    rows, composed, specialists = [], [], []
    for shape in shapes_mod.as_dicts():
        case = str(shape["case"])
        chosen = dict(notes).get(shape["case"], fallback)
        elite = (elites.get(case) or [{}])[0]
        gen_speed = (champ_cases.get(case) or {}).get("speedup")
        if chosen == fallback:
            speed, evidence = gen_speed, "generalist"
        else:
            speed = elite.get("speedup") or gen_speed
            evidence = elite.get("evidence", "provisional")
            specialists.append((shape["case"], chosen, speed, gen_speed, evidence))
        if speed:
            composed.append(speed)
        rows.append((shape, chosen, speed, evidence))

    gen_only = [v.get("speedup") for v in champ_cases.values() if v.get("speedup")]
    gen_geo, comp_geo = geomean(gen_only), geomean(composed)

    L = []
    L.append("# Current best submission\n")
    L.append(f"_Regenerated {time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())} "
             f"from commit `{git_sha()}`. Do not edit — `tools/report.py` "
             f"overwrites this file._\n")
    L.append(composer_note)
    L.append("| | |")
    L.append("|---|---|")
    L.append(f"| **Composition** | 1 generalist + {len(specialists)} specialist(s) |")
    L.append(f"| **Generalist (fallback)** | `{fallback}` |")
    L.append(f"| **Generalist alone, 13-case geomean** | **{gen_geo:.4f}x** |")
    L.append(f"| **Composed, per-shape best** | **{comp_geo:.4f}x** |")
    L.append(f"| Dispatcher gain over generalist | {(comp_geo/gen_geo - 1) * 100:+.1f}% |"
             if gen_geo else "| Dispatcher gain | n/a |")
    L.append(f"| Measured on | {champ.get('gpu', '?')}, {dtype} |")
    L.append(f"| Champion decision | `{champ.get('decision', '?')}` "
             f"over {champ.get('n_scored', '?')} cases |")
    L.append(f"| Evaluations recorded | {len(event_store.load())} |")
    evidence_class = champ.get("evidence_label", "unlabelled").replace("_", " ")
    L.append(f"| Evidence class | {evidence_class} |\n")

    L.append("## Per-shape selection\n")
    L.append("| case | batch | seq | d_model | heads | implementation | speedup | evidence |")
    L.append("|---:|---:|---:|---:|---:|---|---:|---|")
    for shape, chosen, speed, evidence in rows:
        mark = "" if chosen == fallback else " **·**"
        L.append(f"| {shape['case']} | {shape['batch_size']} | {shape['seq_len']} "
                 f"| {shape['d_model']} | {shape['num_heads']} "
                 f"| `{chosen}`{mark} | "
                 f"{('%.4fx' % speed) if speed else '—'} | {evidence} |")
    L.append("\n**·** = specialist, i.e. a shape where a non-champion "
             "implementation is measurably faster.\n")

    if specialists:
        L.append("## Specialists\n")
        L.append("Retained because they beat the generalist on their own shape, "
                 "even where they lose overall.\n")
        for case, name, sp, gen, ev in specialists:
            delta = f"{(sp/gen - 1) * 100:+.1f}%" if (sp and gen) else "n/a"
            L.append(f"- **case {case}** — `{name}` at {sp:.4f}x vs generalist "
                     f"{gen:.4f}x ({delta}), evidence: {ev}"
                     if sp and gen else f"- **case {case}** — `{name}` ({ev})")
    else:
        L.append("## Specialists\n\nNone yet — the generalist is fastest on every "
                 "shape, so the dispatcher would add nothing.\n")

    L.append("\n## How to reproduce\n")
    L.append("```bash")
    L.append("python3 tools/controller.py contract  # show and confirm exact hash")
    L.append("# After start/resume confirmation and separate paid-GPU authorization:")
    L.append(f"python3 tools/controller.py evaluate candidates/{fallback} "
             f"--cases {event_store.FULL_CASES} --contract-hash <confirmed-hash> "
             "--paid-gpu-authorized")
    L.append("```")
    return "\n".join(L) + "\n"


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build())
    events = event_store.load_events()
    evaluations = event_store.load(events)
    champion = event_store.champion(event_store.FULL_CASES, entries=evaluations,
                                    dtype=event_store.target_dtype(), gpu=event_store.target_gpu())
    sessions = [event for event in events if event["event_type"] == "research_session"]
    state = sessions[-1]["data"]["state"] if sessions else "never started after migration"
    counts = {kind: sum(event["event_type"] == kind for event in events) for kind in
              ("candidate", "evaluation", "comparison", "decision", "milestone_review")}
    evidence_counts = evaluation_evidence_counts(events)
    migrated_evaluations = evidence_counts.get("migrated_archived_evidence", 0)
    live_evaluations = evidence_counts.get("live_runtime_evidence", 0)
    STATUS.write_text(
        "# Autoresearch status\n\n"
        "_Generated from `research/events.jsonl`; do not edit as state._\n\n"
        f"- Contract: `{event_store.contract_hash()}`\n"
        f"- Session: **{state}**\n"
        f"- Scope: **{event_store.target_gpu()}**, **{event_store.target_dtype()}**\n"
        f"- Champion: **{champion['candidate']}**, **{champion['score']:.4f}x**\n"
        f"- Canonical events: {len(events)}\n"
        f"- Evaluations: {counts['evaluation']} total; {migrated_evaluations} migrated; "
        f"{live_evaluations} live runtime\n"
        f"- Comparisons: {counts['comparison']}; decisions: {counts['decision']}; "
        f"milestone reviews: {counts['milestone_review']}\n\n"
        "Evidence labels: historical measurements are **migrated archived evidence**; "
        "new paid-GPU measurements are **live runtime evidence**; repository checks are "
        "**static validation** or **local tests**.\n")
    print(f"wrote {OUT.relative_to(ROOT)} and {STATUS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
