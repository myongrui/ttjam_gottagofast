# Autoresearch status

_Generated from `research/events.jsonl`; do not edit as state._

- Contract: `a65dca4e8cec81a7aeb8466dcba664a289cd018524f6c3d10819ec0c02e7af8b`
- Session: **stopped**
- Scope: **NVIDIA A100-SXM4-80GB**, **float32**
- Champion: **v059_paper_fusion_portfolio.py**, **3.1165x**
- Canonical events: 550
- Evaluations: 81 total; 60 migrated; 21 live runtime
- Comparisons: 69; decisions: 81; milestone reviews: 5

Evidence labels: historical measurements are **migrated archived evidence**; new paid-GPU measurements are **live runtime evidence**; repository checks are **static validation** or **local tests**.

## Nightly health check — 2026-09-01

Loop healthy, no candidate proposed. Champion is `v059_paper_fusion_portfolio.py`
at 3.1165x (13-case geomean; composed dispatcher with 2 specialists reaches
3.1316x), up sharply from the 2.2969x champion seen at the previous check.
Last 3 full-sweep runs: v058 (uncertain, 3.0675x) → v059 (**promote**, 3.1165x)
→ v060 (uncertain, 3.1187x, paired CI [0.997, 1.013] straddles 1.0 vs v059) —
a promotion inside the 3-run window, so the no-promotion stall condition is
not met. No correctness failures in recent evaluations (v059/v060 both
`failed_cases: []`); the most recent hard-invalid candidate was v044 on
2026-08-31. The research session has been `stopped` since 02:17 UTC today,
but that reflects the operator-driven start/resume/contract-confirmation
flow in LOOP.md, not a stall.

## Nightly health check — 2026-09-01 (re-check, 20:04 UTC)

Re-run of today's check, ~2h after the 18:03 UTC entry above. Repo HEAD is
unchanged (still `931330a`), canonical event count unchanged at 550, no new
candidates or evaluations since the previous check. State is identical to
the entry above: champion still `v059_paper_fusion_portfolio.py` at 3.1165x,
no stall, no candidate proposed.
