#!/usr/bin/env python3
"""Local scheduled health check. Reads local files; no Claude session involved.

A cloud routine cannot see this machine, so the scheduled check that actually
inspects live state has to run here. Everything diagnostic is deterministic; the
model is consulted only when the loop has genuinely stalled and a new structural
idea is the missing ingredient.

Install via launchd (see tools/com.techjam.healthcheck.plist).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import autoloop  # noqa: E402  (reuse load_dotenv / call_model / screen / slug)
import ledger  # noqa: E402
import routing  # noqa: E402

STATUS = ROOT / "results" / "STATUS.md"
STALE_HOURS = 12


def latest_activity() -> tuple[str, float]:
    """Most recently modified result file and its age in hours."""
    newest, mtime = None, 0.0
    for p in (ROOT / "results").glob("*"):
        m = p.stat().st_mtime
        if m > mtime:
            newest, mtime = p, m
    if newest is None:
        return "(no results)", float("inf")
    return newest.name, (time.time() - mtime) / 3600


def assess() -> dict:
    entries = ledger.load()
    champ = ledger.champion(ledger.FULL_CASES, dtype="float32")
    stalled = routing.plateau_length(entries)
    newest, age_h = latest_activity()
    decision = routing.decide(cycle=1)
    bias = routing.screen_bias(entries)

    if not entries:
        state = "NOT_RUNNING"
    elif age_h > STALE_HOURS:
        state = "IDLE"
    elif stalled >= 3:
        state = "STALLED"
    else:
        state = "PROGRESSING"

    return {"state": state, "champion": champ["candidate"] if champ else None,
            "score": champ["score"] if champ else 0.0, "plateau": stalled,
            "attempts": len(entries), "newest": newest,
            "age_hours": round(age_h, 1), "route": decision, "bias": bias}


def propose(a: dict, model: str) -> str | None:
    """Ask for one candidate. Only called when the loop is actually stuck."""
    prompt = autoloop.build_prompt(a["route"]) + (
        f"\n\nThe loop has stalled: {a['plateau']} full sweeps without a "
        f"promotion. The obvious moves have been tried — see the list above. "
        f"Propose something structurally different, not a variation.")
    try:
        code = autoloop.extract_code(autoloop.call_model(prompt, model))
    except Exception as e:
        return f"proposal failed: {type(e).__name__}: {e}"
    problems = autoloop.screen(code)
    if problems:
        return f"proposal rejected by screen: {'; '.join(problems)}"
    name = f"v{autoloop.next_version():03d}_{autoloop.slug(code)}.py"
    (ROOT / "candidates" / name).write_text(code)
    return f"proposed {name}"


def append_status(a: dict, note: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%MZ")
    line = (f"\n## Health check {stamp}\n\n"
            f"- state: **{a['state']}**\n"
            f"- champion: {a['champion']} @ {a['score']:.4f} "
            f"({a['attempts']} attempts, plateau {a['plateau']})\n"
            f"- last activity: {a['newest']}, {a['age_hours']}h ago\n"
            f"- route: {a['route'].mode} — {a['route'].reason}\n"
            f"- screen contradiction rate: {a['bias']['rate']:.0%}\n"
            f"- action: {note}\n")
    with STATUS.open("a") as f:
        f.write(line)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--propose", action="store_true",
                   help="ask the model for a candidate when STALLED (costs tokens)")
    p.add_argument("--model", default=None)
    args = p.parse_args()
    autoloop.load_dotenv()
    model = args.model or os.environ.get("OPENAI_MODEL")

    a = assess()
    note = "none needed"
    if a["state"] == "STALLED" and args.propose and model:
        note = propose(a, model)
    elif a["state"] == "STALLED":
        note = "stalled; --propose not enabled"
    elif a["state"] in ("IDLE", "NOT_RUNNING"):
        note = f"loop is {a['state'].lower()}; start the pod and run autoloop"

    # Keep the reader-facing summary current even when no loop is running.
    try:
        subprocess.run([sys.executable, str(ROOT / "tools" / "report.py")],
                       cwd=ROOT, capture_output=True, timeout=120)
    except Exception:
        pass

    STATUS.parent.mkdir(parents=True, exist_ok=True)
    if not STATUS.exists():
        STATUS.write_text("# TechJam loop status\n")
    append_status(a, note)
    print(json.dumps({k: v for k, v in a.items() if k != "route"}, indent=2))
    print("action:", note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
