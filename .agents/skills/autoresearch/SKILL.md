---
name: autoresearch
description: Start, resume, review, or stop this repository's transformer autoresearch workflow with Codex as lead researcher and deterministic evaluation controls.
---

# Autoresearch

Use this skill as the standard entrypoint for research in this repository. Codex owns evidence review, hypotheses, candidate edits, interpretation, and direction selection. Deterministic repository code owns correctness, benchmarking, promotion, timeouts, recovery, and persistence.

## Contract confirmation gate

Before `start`, `resume`, or any candidate execution:

1. Run `python3 tools/controller.py contract` and show the complete `research.yaml` contract plus its SHA-256 hash to the user.
2. Ask the user to confirm that exact contract. Do not treat a prior session's approval as confirmation if the hash changed.
3. Only after confirmation, run `python3 tools/controller.py start --contract-hash <hash>` or `resume`, then pass the same hash to `controller.py race`.

Review and stop operations do not execute candidates. `stop` records lifecycle state; infrastructure shutdown remains a separate explicitly authorized operation.

## Modes

- **Start:** Confirm the contract, inspect `LOOP.md` and generated status, record the running session, ask the controller for a deterministic route, then let Codex propose and edit one candidate.
- **Resume:** Confirm the current contract again, inspect all events since the last session event, and continue from the hardware-scoped champion and portfolio.
- **Review:** Run `python3 tools/controller.py review`, inspect `research/events.jsonl` and referenced immutable evidence, and label conclusions as static validation, local tests, migrated archived evidence, or live runtime evidence.
- **Stop:** Run `python3 tools/controller.py stop --reason '<reason>'`. Do not stop or terminate paid infrastructure unless the user separately authorizes it.

## Research cycle

1. Read `research.yaml`, `results/STATUS.md`, `results/RESULTS.md`, and the relevant canonical events and evidence artifacts.
2. Run `python3 tools/controller.py route --cycle N`; deterministic routing provides constraints, while Codex chooses the hypothesis and direction.
3. State one falsifiable hypothesis, expected affected cases, parent, and mechanism. Create exactly one new `candidates/vNNN_<slug>.py`; never modify historical candidates.
4. Perform static validation and label it as such. Static checks are not benchmark evidence.
5. With a confirmed contract and separately authorized paid GPU use, run the candidate through `controller.py race --paid-gpu-authorized`. Add that flag only after the user explicitly authorizes the paid run. Do not bypass `tools/harness.py` or the promotion rule.
6. Interpret the canonical evaluation, comparison, and decision events. Use only evidence matching the contract's hardware, dtype, and case scope for ranking or dispatch.
7. Regenerate views with `python3 tools/controller.py review`; never edit generated result or status views as state.

`research/events.jsonl` is the sole canonical history. Candidate sources and raw benchmark artifacts are immutable evidence. `archive/legacy/` is audit-only and must never be used as active state.
