# Optimization loop

A closed generate → verify → measure → record cycle for the transformer kernel
work, scored by measured speedup over the official baseline.

## The score

```
score = geometric mean of per-shape speedup, over shapes the baseline can run
        ... gated to 0 if ANY shape fails correctness or integrity
```

Geometric mean because speedups are ratios: it treats 2× on one shape and 0.5×
on another as a wash, which an arithmetic mean does not. The gate is hard on
purpose. Speed on top of a wrong answer is not a partial result, it is nothing.
Case 14 has no runnable baseline, so its archive is ranked by candidate latency
rather than a fabricated speedup and is reported separately from the geomean.

## Why the verifier is the design

Published results on LLM-driven kernel generation are unambiguous: measurement
exploitation, not incompetence, is the dominant failure mode.
[KernelBench-Verified](https://arxiv.org/abs/2607.16241) found frontier models
fall from 1.43× to **0.88×** geomean once the harness is hardened — most of the
reported gain was artifact. [Robust agentic CUDA benchmarking](https://arxiv.org/pdf/2509.14279)
catalogues three families, each of which this benchmark is exposed to:

| Family | The exposure here | Defence in `tools/harness.py` |
|---|---|---|
| **Concurrency** | Official timing uses `torch.cuda.Event` on the current stream; work issued on another stream is never timed | Every candidate is timed twice — events *and* a fully synchronized wall clock. `wall/event > 1.6` fails the run |
| **State caching** | The official timing loop reuses **one fixed input** for all repeats, so memoizing it looks like a huge win | Correctness runs on 5 fresh input distributions; `check_output_depends_on_input` asserts the output moves when the input does, and is stable when it repeats |
| **Environment** | Global `torch` flags (TF32, matmul precision) change what "baseline" means | `env_snapshot()` before and after; any drift zeroes the score |

Correctness itself is checked over five distributions (normal, ×8, ×0.05, and
two padding ratios) rather than one, because a single distribution invites
hardcoded bypasses that skip the computation for the values actually tested.

## Established findings

These are measured on this pod, not assumed. They constrain every candidate.

1. **Target float32, not bf16.** The gate is `abs ≤ 0.002 OR rel ≤ 2%`. Outputs
   reach magnitude ~4–8, where one bf16 ULP is ~0.03. Elements that land near
   zero through cancellation carry that ~0.03 absolute error, failing `abs`
   while `rel` is meaningless for them. In bf16 only a *bit-identical*
   implementation passes, which forbids optimization. v001 fails every shape in
   bf16 and passes every shape in fp32 — same code. fp32 keeps TF32 tensor
   cores, so this costs no hardware capability. The benchmark's own default is
   `float32`.
2. **Case 14 has no reference.** Its baseline needs a `[32,16,100000,100000]`
   fp32 score matrix — **19,073 GiB**. The baseline cannot run it on any
   hardware. The harness builds the real case-14 implementation and establishes
   correctness on a reduced-sequence proxy before timing the full shape; case 14
   itself yields a feasibility-and-runtime result only. It is also the
   one shape where a FlashAttention-style kernel wins by default rather than by
   margin.
3. **Noise floor is ~2%.** The identity candidate (v000) scores 0.993 geomean,
   worst case 0.980 on batch=1. Treat sub-2% deltas on tiny shapes as noise.
4. **Scores compare only within one case set, dtype, and GPU.** A 3-case score
   and a 13-case score are different quantities, as are bf16 and fp32 runs;
   `tools/ledger.py` groups only compatible measurements. Screen on the cases
   affected by the hypothesis, then use the full sweep for global promotion.
5. **Most shapes are launch-bound.** 11 of 14 have `d_model=128, ffn=128,
   head_dim=32`. v001's largest win is on case 2 (batch=1, 1.61×) — the
   smallest, most launch-dominated shape. Fusion and launch elimination beat
   tiling cleverness here.

## Search policy

The loop is a small portfolio search, not a `(1+1)` hill climber:

- A **global beam** retains the three best statistically promoted full-sweep
  candidates as mutation parents.
- A **per-shape archive** retains the two fastest valid candidates for every
  official case. A global loser can therefore remain useful as a specialist.
- A race screens one or more challengers on a hypothesis-relevant case family,
  then sends the strongest valid challenger to the 13-case sweep. Every screen
  result remains in the archive.
- New runs time the challenger, official baseline, and incumbent in rotating
  order in the same process. Promotion requires the lower bound of the paired
  95% bootstrap interval to exceed the incumbent by more than the 2% noise
  margin. Subset wins never promote the global champion.
- `tools/build_dispatcher.py` turns the best **proven** per-shape implementations
  into one candidate. A new specialist is proven only when its direct paired
  interval clears the 2% margin; faster point estimates remain provisional in
  the archive. Dispatch occurs at model construction from the static config, so
  it adds no per-forward shape-check overhead.

This is deliberately smaller than a learned cost model or MCTS. With only a few
measurements there is not enough data to fit a useful surrogate; the beam and
archive capture diversity without spending the pod budget on search machinery.

## Iteration protocol

Each race:

1. **Read** `results/ledger.jsonl` (`python3 tools/ledger.py`). Identify the
   beam, shape elites, and every hypothesis already refuted.
2. **Choose a parent** from the global beam or the elite for the shapes being
   targeted. Do not force every mutation to descend from the global champion.
3. **Propose one change** with a stated mechanism — why it should be faster and
   which shapes it should move. One variable at a time remains the default so a
   result teaches something.
4. **Write one or more challengers** as `candidates/vNNN_<slug>.py`.
5. **Race them** with `tools/race.py`. The relevant profile screens every
   challenger; the best valid screen advances to the full sweep.
6. **Record and classify** — automatic. A full paired result is `promote`,
   `reject`, or `uncertain`; failures are `invalid`, and subset evidence is
   prefixed with `screen_`.
7. **Compose specialists periodically** with `tools/build_dispatcher.py`, then
   race the generated dispatcher like any other challenger.

Screening profiles:

| Profile | Cases | Intended mechanism |
|---|---:|---|
| `launch` | 1,2,12 | launch elimination, graph replay, small-shape fusion |
| `attention` | 9,11,13 | attention backend, heads, long-sequence kernels |
| `throughput` | 5,6,8 | large-batch or wide GEMM/FFN throughput |
| `heads` | 9,10,11 | head-count-sensitive layouts and kernels |
| `long-seq` | 12,13 | sequence-length crossover behavior |
| `general` | 2,6,8,11,13 | one representative from each major regime |

Use `case:N` for an exact-shape specialist, including case 14.

## Candidate contract

```python
"""vNNN -- one-line hypothesis, then the mechanism."""

def build_model(config, bench):          # required
    return MyModel(config)

def load_from_baseline(model, baseline): # optional; needed when weights do not
    ...                                  # map 1:1 (e.g. fused QKV)
```

Without `load_from_baseline` the harness uses `bench.copy_model_weights(strict=True)`.

## Hypothesis backlog

Ordered by expected payoff per unit effort, informed by the shape table.

1. ~~`torch.compile` + CUDA graphs (`mode="reduce-overhead"`)~~ — **REFUTED,
   v003.** Correct but *slower*: 1.375 vs v001's 1.579 geomean on the same
   cases 1,2,12, a 13% regression. Inductor's fusion did not recover what its
   guard and dispatch overhead cost at these sizes. A narrower retry
   (`mode="default"`, or compiling only the FFN subgraph) is still open, but the
   blunt version is dead.
2. **Remove the host-device sync in v001** — `bool(valid_token_mask.all())` in
   `FusedAttention.forward` forces a device-to-host synchronization on **every
   attention call** (4 per forward). It stalls the pipeline every layer and makes
   CUDA-graph capture impossible, which likely explains both the latency-bound
   profile and v003's loss. Decide the mask path once at build time instead.
   Re-test CUDA graphs afterwards: v003's refutation may not survive this fix.
3. **Fused LayerNorm + residual in Triton** — the highest-value hand-written
   kernel, and the one Inductor is least likely to match.
4. **Shape dispatch** — the rules explicitly permit different implementations
   per shape. Small shapes want graphs; case 6 (batch 10000) and case 13
   (seq 1024) want tiling.
5. **FFN epilogue fusion** — GEMM + bias + GELU in one kernel.
6. **Case 14 specialist** — chunked FlashAttention; the only shape where the
   baseline cannot compete at all.

Search policy itself is a separate decision — see `POLICIES.md`.

## Unattended operation

`tools/autoloop.py` runs the loop with no Claude Code session, so it does not
consume a Claude usage window. OpenAI performs the proposal step; `race.py` does
the measuring, so the selection policy is unchanged.

```bash
export OPENAI_API_KEY=sk-...          # never commit; keep out of pod env
export OPENAI_MODEL=<model you have>  # printed at startup
python3 tools/autoloop.py --dry-run                      # generate + screen only
caffeinate -i python3 tools/autoloop.py --max-iters 20 --max-hours 6
```

Generated code is executed on the pod minutes later, so it passes a static
screen first (`FORBIDDEN` in `autoloop.py`) covering the three documented
reward-hacking families plus file/process I/O. The screen is a cheap filter, not
a sandbox — the harness's runtime integrity checks remain the real defence.

Stops on whichever comes first: iteration count, wall-clock, consecutive
non-improvements, or an unreachable pod. It does **not** start the pod; a dead
pod ends the run rather than recording candidates as failures.


### Pod lifecycle

Starting is automatic. `ensure_pod()` runs before every evaluation, not just at
launch: it probes ssh, starts the pod when it is down, and waits for the boot.
Pods on this account have stopped mid-session four times across both community
and secure clouds, and recovering costs one start where aborting costs the rest
of the run. `--no-start-pod` opts out.

Stopping is **not** symmetric and is therefore still opt-in.

### Why `--stop-pod` can cost you the GPU

A stopped Runpod pod stays bound to its original host, and its GPU is released
to other users. Restarting only succeeds if that specific machine still has a
free card. On a scarce GPU it does not:

```
HTTP 400: There are not enough free GPUs on the host machine to start this pod.
```

This happened twice on A100 80GB even while the pool showed HIGH availability —
capacity existed, just not on the pinned host. So `--stop-pod` trades money for
resumability. For a long session, prefer leaving the pod running; for an
overnight unattended run, keep `--stop-pod` and accept that resuming may mean
creating a fresh pod. Everything needed lives in git, so a rebuild is one
`podsync` plus a repoint of `RUNPOD_POD_ID`/`POD_SSH` in `.env`.


## Running it

```bash
python3 tools/ledger.py status

# Race one or several candidates. The current full-sweep champion is selected
# automatically and frozen for the race.
python3 tools/race.py --profile launch \
  candidates/v004_fused_norm.py candidates/v005_ffn_compile.py

# A manual paired run remains available. Fourth argument optionally fixes the
# incumbent; otherwise iterate.sh selects the current champion.
tools/iterate.sh candidates/v004_fused_norm.py 1,2,12 float32

# Materialize the current per-shape archive as a normal candidate, then race it.
python3 tools/build_dispatcher.py --out candidates/v010_portfolio.py
python3 tools/race.py --profile general candidates/v010_portfolio.py
```

## Human collaboration

The three-minute partner handoff, parallel-work split, and final shape-dispatch
submission workflow live in `READMEFORJJ.md`. This file stays focused on the
model-facing optimization policy and measured findings.

Pod control is via the RunPod MCP tools (`start-pod` / `stop-pod`, id
`132yirw3l7rq7s`). It is **community cloud and has already stopped once
mid-session** — `iterate.sh` exits 3 on an unreachable pod rather than recording
a false zero. `/workspace` survives a stop; it does not survive a terminate.
