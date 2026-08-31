#!/usr/bin/env python3
"""Codex-operated autoresearch lifecycle; contains no nested model client."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import event_store
import routing

ROOT = Path(__file__).resolve().parent.parent


def require_contract_hash(value: str) -> None:
    expected = event_store.contract_hash()
    if value != expected:
        raise SystemExit("research contract is unconfirmed or stale; run `python3 tools/controller.py contract`, "
                         "show it to the user, and pass the displayed hash only after confirmation")


def lifecycle_state() -> str:
    events = event_store.events_of("research_session")
    return events[-1]["data"]["state"] if events else "never_started"


def record_session(state: str, reason: str = "") -> None:
    sequence = len(event_store.load_events()) + 1
    event = event_store.make_event("research_session", {
        "state": state, "reason": reason, "contract_hash": event_store.contract_hash(),
        "operator": "codex", "evidence_label": "live_runtime_evidence",
    }, sequence)
    event_store.append_events([event])


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("contract")
    sub.add_parser("status")
    start = sub.add_parser("start"); start.add_argument("--contract-hash", required=True)
    resume = sub.add_parser("resume"); resume.add_argument("--contract-hash", required=True)
    stop = sub.add_parser("stop"); stop.add_argument("--reason", default="user requested stop")
    sub.add_parser("review")
    route = sub.add_parser("route"); route.add_argument("--cycle", type=int, default=1)
    race = sub.add_parser("race")
    race.add_argument("candidates", nargs="+")
    race.add_argument("--profile", default="general")
    race.add_argument("--dtype", default=None)
    race.add_argument("--contract-hash", required=True)
    race.add_argument("--paid-gpu-authorized", action="store_true", required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("candidate")
    evaluate.add_argument("--cases", default=event_store.FULL_CASES)
    evaluate.add_argument("--dtype", default=None)
    evaluate.add_argument("--incumbent", default="")
    evaluate.add_argument("--contract-hash", required=True)
    evaluate.add_argument("--paid-gpu-authorized", action="store_true", required=True)
    args = parser.parse_args()

    if args.command == "contract":
        print(event_store.CONTRACT.read_text().rstrip())
        print(f"\ncontract_sha256: {event_store.contract_hash()}")
        print("No research execution is authorized until the user confirms this exact contract.")
    elif args.command == "status":
        print(f"session: {lifecycle_state()}")
        print(event_store.summary())
    elif args.command in ("start", "resume"):
        require_contract_hash(args.contract_hash)
        record_session("running", args.command)
        print(f"research {args.command}ed under contract {args.contract_hash}")
    elif args.command == "stop":
        record_session("stopped", args.reason)
        print("research stopped; this records lifecycle state and does not terminate infrastructure")
    elif args.command == "review":
        subprocess.run([sys.executable, str(ROOT / "tools" / "report.py")], cwd=ROOT, check=True)
    elif args.command == "route":
        decision = routing.decide(args.cycle, dtype=event_store.target_dtype())
        print(json.dumps({**decision.as_dict(), "guidance": decision.guidance}, indent=2))
    elif args.command == "race":
        require_contract_hash(args.contract_hash)
        if lifecycle_state() != "running":
            raise SystemExit("research session is not running; confirm the contract and start or resume first")
        command = [sys.executable, str(ROOT / "tools" / "race.py"), *args.candidates,
                   "--profile", args.profile, "--dtype", args.dtype or event_store.target_dtype()]
        return subprocess.run(command, cwd=ROOT).returncode
    elif args.command == "evaluate":
        require_contract_hash(args.contract_hash)
        if lifecycle_state() != "running":
            raise SystemExit("research session is not running; confirm the contract and start or resume first")
        command = [str(ROOT / "tools" / "iterate.sh"), args.candidate, args.cases,
                   args.dtype or event_store.target_dtype(), args.incumbent]
        return subprocess.run(command, cwd=ROOT).returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
