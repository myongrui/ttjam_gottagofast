#!/usr/bin/env python3
"""Stage a small portfolio of challengers against the current champion.

All challengers first run on a hypothesis-relevant screen. Their screen runs
remain in the append-only ledger and can update the per-shape elite archive even
when they do not advance. The strongest valid screen result is then promoted to
the expensive 13-case paired sweep.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import ledger  # noqa: E402


PROFILES = {
    # Fixed launch cost and short-sequence behavior.
    "launch": "1,2,12",
    # Attention backend, head-count, and long-sequence behavior.
    "attention": "9,11,13",
    # Large batch and wide GEMM/FFN throughput.
    "throughput": "5,6,8",
    "heads": "9,10,11",
    "long-seq": "12,13",
    # One representative from every materially different regime.
    "general": "2,6,8,11,13",
}


def report_path(candidate: str) -> Path:
    return ROOT / "results" / f"{Path(candidate).stem}.json"


def run_iteration(candidate: str, cases: str, dtype: str,
                  incumbent: str) -> Dict[str, Any]:
    command = [str(ROOT / "tools" / "iterate.sh"), candidate, cases, dtype]
    if incumbent:
        command.append(incumbent)
    completed = subprocess.run(command, cwd=ROOT)
    if completed.returncode:
        raise RuntimeError(
            f"iteration failed for {candidate} with exit {completed.returncode}")
    with report_path(candidate).open() as f:
        return json.load(f)


def screen_score(report: Dict[str, Any]) -> float:
    comparison = report.get("incumbent_comparison") or {}
    if report.get("failed_cases") or not report.get("env_unchanged", False):
        return float("-inf")
    return float(comparison.get("paired_geomean", float("-inf")))


def resolve_cases(profile: str) -> str:
    if profile in PROFILES:
        return PROFILES[profile]
    if profile.startswith("case:"):
        case = int(profile.split(":", 1)[1])
        if case < 1 or case > 14:
            raise ValueError("case profile must be in 1..14")
        return str(case)
    raise ValueError(
        f"unknown profile {profile!r}; choose {', '.join(PROFILES)} or case:N")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Screen a challenger portfolio, then full-sweep its winner")
    parser.add_argument("candidates", nargs="+", help="candidate Python files")
    parser.add_argument("--profile", default="general",
                        help="launch, attention, throughput, heads, long-seq, "
                             "general, or case:N")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--incumbent", default="",
                        help="freeze a specific incumbent; default ledger champion")
    parser.add_argument("--screen-only", action="store_true")
    parser.add_argument("--advance-floor", type=float, default=0.99,
                        help="minimum paired screen ratio for a sole challenger")
    args = parser.parse_args()

    cases = resolve_cases(args.profile)
    incumbent = args.incumbent
    if not incumbent:
        best = ledger.champion(ledger.FULL_CASES, dtype=args.dtype)
        if not best:
            raise SystemExit("no full-sweep incumbent in the ledger")
        incumbent = str(Path("candidates") / best["candidate"])

    frozen_name = os.path.basename(incumbent)
    print(f"screen profile={args.profile} cases={cases} incumbent={frozen_name}")
    screened: List[tuple[str, Dict[str, Any]]] = []
    for candidate in args.candidates:
        if os.path.basename(candidate) == frozen_name:
            print(f"skip incumbent {candidate}")
            continue
        report = run_iteration(candidate, cases, args.dtype, incumbent)
        screened.append((candidate, report))

    ranked = sorted(screened, key=lambda row: -screen_score(row[1]))
    print("\nscreen ranking:")
    for candidate, report in ranked:
        score = screen_score(report)
        rendered = "invalid" if score == float("-inf") else f"{score:.4f}x"
        print(f"  {Path(candidate).name:<34} {rendered}")

    if args.screen_only or not ranked:
        return 0
    finalist, finalist_report = ranked[0]
    finalist_score = screen_score(finalist_report)
    if finalist_score == float("-inf"):
        print("no valid challenger advances")
        return 0
    if finalist_score < args.advance_floor:
        print(f"no advance: {finalist_score:.4f}x is below "
              f"the {args.advance_floor:.4f}x screen floor")
        return 0

    print(f"\nfull-sweep finalist: {Path(finalist).name}")
    full = run_iteration(finalist, ledger.FULL_CASES, args.dtype, incumbent)
    comparison = full.get("incumbent_comparison") or {}
    ratio = comparison.get("paired_geomean")
    ci = comparison.get("paired_geomean_ci95")
    decision = comparison.get("decision", "unpaired")
    if ratio is not None and ci:
        print(f"paired result: {ratio:.4f}x, 95% CI "
              f"[{ci[0]:.4f}, {ci[1]:.4f}], {decision}")
    else:
        print(f"paired result unavailable: {decision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
