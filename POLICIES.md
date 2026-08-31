# Search policies — active, superseded, and parked

How the loop decides *what to try next*. One policy is active; the rest are
parked with the reasoning that put them here, so switching is a decision rather
than a rewrite.

---

## ACTIVE — P2: portfolio search (beam + per-shape archive + racing)

Implemented in `tools/event_store.py` (`global_beam`, `per_shape_elites`,
`decision_from_report`), `tools/race.py`, `tools/build_dispatcher.py`.

Global beam of 3 promoted full-sweep parents; per-shape archive of the 2 fastest
valid candidates per case; hypothesis-relevant screening profile before any
13-case sweep; promotion requires the lower bound of the paired 95% bootstrap
interval to clear the incumbent by more than the 2% noise margin; challenger,
baseline and incumbent are timed in rotating order **in one process**; proven
specialists are composed into a dispatcher that selects at construction time.

* **Strength** — the promotion rule is statistical rather than a point-estimate
  comparison, which is the single most important property given the measured
  noise floor. Screening cuts pod cost. Specialists survive global losses.
  Construction-time dispatch adds no per-forward overhead, which matters when
  everything is launch-bound.
* **Weakness** — it decides *which candidate wins*, not *what to try next*. That
  gap is P1's job and remains unfilled.

---

## SUPERSEDED — P0: greedy hill climb (champion / challenger)

One incumbent; promote on a strictly better score. Was active until the
portfolio search replaced it.

* **Why it was replaced** — it compared **point-estimate medians with no
  confidence interval**. With an identity control measuring 0.980–1.002, that
  rule promotes noise as a matter of routine. It also compared numbers taken
  from separate processes, inheriting clock, thermal and allocator drift; kept
  only one incumbent, maximising local-optimum exposure; discarded specialists
  outright; and always paid for a full sweep.
* **What it got right, and P2 kept** — a hard correctness gate, append-only
  canonical events including failures and their reasons, and refusing to rank scores
  across different case sets.

---

## ACTIVE — P1: speed-of-light gated mode switching

Route each iteration to *tuning* or *structural change* by measured gap to the
hardware roofline. Implemented in `tools/routing.py` (`decide`), driven by `tools/roofline.py`.
Runs when Codex calls `tools/controller.py route`; it is arithmetic over
hardware-scoped canonical events and the roofline, with no model in the path.

```
TUNE --plateau (K configs, no gain > 2% noise margin)--> gap to SOL?
        gap < 1.5x  -> DONE        (at the hardware limit; stop spending)
        gap 1.5-4x  -> TUNE        (right structure, wrong parameters)
        gap > 4x    -> STRUCTURAL  (latency-bound; no tile size fixes this)
STRUCTURAL --wins--> TUNE (retune the new structure)
           --loses--> record refutation, try another family; DONE after M
```

* **Rationale** — autotuning moves a kernel toward the roofline it is already
  on; it cannot move the roofline. Near SOL, tuning is finished *by definition*,
  giving the loop a real stopping criterion instead of a guess. It is
  **orthogonal to P2**: P2 selects, P1 proposes.
* **Current behaviour** — reports `STRUCTURAL` for all 13 shapes (gaps
  10x–1000x). It does not yet *discriminate* between shapes, but it already does
  useful work: it steers generation away from tuning, names the worst offenders
  with their bound type, and picks the screening profile. It begins routing
  properly once some shapes approach SOL.
* **Known weakness** — the roofline is a hand-rolled analytical model and
  optimistic: it assumes perfect fusion and ignores the ~4 µs launch floor, so
  case 2's 1095x is not credible (~30–70x is the honest figure). Gating on a
  wrong model routes wrongly. Thresholds are unvalidated. The upgrade is to drive
  it from Nsight Compute counters — DRAM throughput, achieved occupancy, SM
  efficiency — rather than arithmetic.
* **Already earned its keep once** — it exposed a contradiction (profile says
  launch-bound, yet CUDA graphs lost) that located the host-device sync in
  `v001.FusedAttention`. That is diagnostic value no scalar score can provide.

---

---

## Criticisms of P2, and what was done about them

| Criticism | Fix | Where |
|---|---|---|
| Beam of 3 / elites of 2 over four candidates retains everything — selection pressure in name only | Widths scale with the eligible pool: beam 1 → 2 → 3 at 2 / 5 / 6+ candidates | `routing.adaptive_widths` |
| Screening on a hypothesis-chosen profile selects for candidates suited to that profile | A neutral `general` profile is forced every third cycle, and screen-`promote`-then-full-fail is counted and surfaced | `routing.choose_profile`, `routing.screen_bias` |
| Solves selection, not proposal — silent on tune vs restructure | P1 gives Codex deterministic routing evidence while Codex chooses the hypothesis | `controller.py route` → `routing.decide` |
| Mechanical tuning needs no research-model call | `TUNE` remains a deterministic config sweep whose raw outputs become referenced evidence | `routing.tuning_state` → `tools/tuner.py` |

## Assessment

Across the 60 migrated evaluations, the A100-80GB full-sweep lineage ends at
`v038` with 2.2969x and later compatible candidates do not clear the unchanged
promotion gate. Routing therefore calls for another structural hypothesis;
search policy is not evidence that a specific untested mechanism will win.
