# Transformer GPU Kernel Optimization — TechJam

Optimizing the runtime of a fixed Transformer forward pass on a specific GPU,
without changing the layer's mathematical contract.

**Measured result: 3.13× geometric-mean speedup** over the official PyTorch
baseline across the 13 runnable official test shapes, on an
NVIDIA A100-SXM4-80GB in float32, with every shape passing the required
accuracy gate (relative error < 0.02, absolute error < 0.002).

---

## Project overview

The task provides `torch_transformer_benchmark.py`: a reference Transformer
(pre-norm blocks — LayerNorm → multi-head self-attention → residual →
LayerNorm → FFN → residual, then a final LayerNorm) and a
`UserOptimizedTransformer` hook to replace with a faster implementation that
still matches the reference numerically.

Two observations shaped the whole solution:

1. **The official shapes are small.** Eleven of the fourteen cases use
   `d_model=128`, `ffn_dim=128`, `head_dim=32`, `4` layers. At that size the
   GPU is nowhere near compute-bound — the forward pass is dominated by
   **kernel-launch and host-submission overhead**, not by FLOPs. The wins come
   from *issuing fewer, larger operations*, not from a faster matmul.
2. **The rules permit per-shape implementations.** So the deliverable is not
   one kernel but a **portfolio**: several implementations plus a
   construction-time dispatcher that picks one per shape. Dispatch happens when
   the model is built, so it costs nothing per forward pass.

### The submitted implementation

`final_submission.py` is the deliverable (identical to
`candidates/v060_self_contained_portfolio.py`). It selects one of three paths
from the shape signature at construction time:

| Path | Used for | What it does |
|---|---|---|
| **Regular CUDA-graph replay** | cases 1, 4, 12, 13 | Captures the entire multi-layer forward into a single `torch.cuda.CUDAGraph` and replays it, collapsing hundreds of launches into one. |
| **Homogeneous-coordinate FFN + graph replay** | cases 2, 3 | Same capture, plus the FFN output bias is folded into two extra weight columns (a homogeneous-coordinate trick) so the bias broadcast disappears entirely. The bias is split into a high/low fp16 pair to preserve fp32 accuracy. |
| **TorchScript generalist** | everything else, including unseen shapes | Fused QKV projection into one GEMM, `scaled_dot_product_attention` (Flash / mem-efficient backends), residual folded into the GEMM epilogue via `torch.addmm`, whole model `torch.jit.script`-ed. Safe fallback for any shape. |

Underneath all three, the same optimizations apply: one fused QKV GEMM instead
of three, `F.scaled_dot_product_attention` with `is_causal=True` instead of a
materialized `[B, H, S, S]` score matrix, residual additions fused into GEMM
epilogues, and no host synchronization inside the forward pass.

### How it was built: a verifier-first autoresearch loop

The optimizations above were not hand-picked. The repository is a closed
**generate → verify → measure → record** cycle that produced and scored 60+
candidate implementations (`candidates/v000_*.py` … `v060_*.py`), keeping
append-only evidence for every one, including the failures.

The autoresearch loop must be executed through the **Codex app**. Its repo-local
`autoresearch` skill makes Codex the lead researcher: Codex reviews the evidence,
proposes and edits candidates, interprets results, and chooses the next direction,
while deterministic repository code handles validation, benchmarking, promotion,
timeouts, recovery, and persistence.

The loop is designed around a specific published risk: in LLM-driven kernel
generation, the dominant failure mode is **measurement exploitation, not
incompetence** — [KernelBench-Verified](https://arxiv.org/abs/2607.16241)
reports frontier models falling from 1.43× to 0.88× geomean once the harness is
hardened. `tools/harness.py` therefore treats correctness and integrity as a
hard gate rather than a scoring term:

- **Correctness over five input distributions** (normal, ×8, ×0.05, and two
  padding ratios), not one — a single distribution invites hardcoded bypasses.
- **Anti-concurrency check.** Every candidate is timed twice, with CUDA events
  *and* a fully synchronized wall clock. `wall/event > 1.6` fails the run,
  catching work hidden on an untimed side stream.
- **Anti-caching check.** The official timing loop reuses one fixed input, so
  memoization would look like a huge win; the harness asserts the output
  actually moves when the input moves.
- **Environment integrity.** `torch` global flags (TF32, matmul precision) are
  snapshotted before and after; any drift zeroes the score.
- **Statistical promotion.** A candidate is promoted only if the lower bound of
  a paired 95% bootstrap confidence interval clears the incumbent by more than
  the 2% measured noise margin. Baseline, incumbent and challenger are timed in
  rotating order *within one process*. A point-estimate comparison would promote
  noise routinely — the identity control measures 0.980–1.002×.

`research/events.jsonl` is the single canonical history (append-only); candidate
files and raw JSON reports are immutable evidence; `results/RESULTS.md` and
`results/STATUS.md` are regenerated views, never hand-edited.

### Repository map

```
final_submission.py              the deliverable — drop-in optimized Transformer
torch_transformer_benchmark.py   official benchmark, unmodified
candidates/                      every implementation tried, v000 → v060
results/                         RESULTS.md, STATUS.md, raw evaluation JSON
research/events.jsonl            canonical append-only history
tools/harness.py                 correctness + integrity + timing (the verifier)
tools/event_store.py             promotion rules, hardware-scoped ledger
tools/race.py                    screening and full-sweep control
tools/build_dispatcher.py        composes proven specialists into a portfolio
tools/controller.py              research lifecycle; contains no model client
tools/pod{run,put,get,sync}      remote GPU helpers (RunPod)
tests/                           unit tests for the ledger, promotion, dispatcher
LOOP.md                          the optimization loop and its measured findings
POLICIES.md                      search policies: active, superseded, parked
research.yaml                    the research contract (hashed and confirmed)
```

---

## Setup and installation

### Requirements

- **A CUDA GPU.** Results were measured on an **NVIDIA A100-SXM4-80GB**. The
  code runs on other CUDA GPUs, but the per-shape dispatch table was tuned for
  the A100 and the speedups will differ.
- **Python 3.10+**
- **PyTorch 2.8.0+cu128** (the measured configuration). PyTorch ≥ 2.2 should
  work; CUDA-graph capture and `scaled_dot_product_attention` are both required.
- ~4 GB free GPU memory for cases 1–13. Case 6 (`batch=10000`) is the largest
  runnable case; case 14 needs far more (see Limitations).

### Install

```bash
git clone <this-repo>
cd ttjggf

python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128

python3 -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
```

No other third-party dependencies — the harness, ledger and tooling use only the
standard library plus `torch`.

### Optional: remote GPU

If you develop on a machine without an A100, `tools/pod*` drive a RunPod pod over
its SSH proxy. Copy `.env.example` to `.env` and fill in `RUNPOD_API_KEY` and
`RUNPOD_POD_ID`; `.env` is gitignored. This is only needed to reproduce on rented
hardware — everything runs locally on any CUDA GPU.

---

## Steps to reproduce the results

### 1. Sanity check — the official benchmark, unmodified

```bash
python3 torch_transformer_benchmark.py \
  --batch-size 64 --seq-len 128 --d-model 128 --heads 4 \
  --ffn-dim 128 --layers 4 --causal --dtype float32
```

Out of the box `UserOptimizedTransformer` just calls the baseline, so this
reports ~1.0× and confirms the environment works.

### 2. Score the submission across all official shapes

This is the canonical reproduction. The harness constructs the baseline from the
official benchmark module, builds the candidate through its `build_model` /
`load_from_baseline` hooks, copies identical weights, then runs the correctness
gate, the integrity checks and the paired timing:

```bash
python3 tools/harness.py \
  --candidate final_submission.py \
  --bench ./torch_transformer_benchmark.py \
  --cases 1,2,3,4,5,6,7,8,9,10,11,12,13 \
  --dtype float32 \
  --repeats 30 \
  --out results/evaluations/reproduce.json
```

**Use `--dtype float32`.** This is not a convenience default — see Limitations.

The JSON report records, per case: the shape, correctness across all five input
distributions with max absolute and relative error, baseline and candidate
timings per round, the wall/event ratio, and the resulting speedup. A case with
`"status": "integrity_failed"` scores nothing regardless of its timing.

### 3. Run a single shape

```bash
python3 tools/harness.py --candidate final_submission.py \
  --bench ./torch_transformer_benchmark.py --cases 2 --dtype float32
```

### 4. Inspect the recorded evidence

```bash
cat results/RESULTS.md          # per-shape champion table (generated)
cat results/STATUS.md           # loop status and evidence classes (generated)
python3 tools/report.py         # regenerate both from research/events.jsonl
python3 -m unittest discover tests -v
```

### 5. Run the full research loop in Codex (optional, uses paid GPU)

The autoresearch loop is run from the **Codex app**, not by invoking the
controller commands manually:

1. Open this repository as a project in Codex.
2. Ask Codex to use the repo-local `autoresearch` skill to **start**, **resume**,
   **review**, or **stop** the research loop. For example: `Use autoresearch to
   resume the transformer optimization loop.`
3. Before starting, resuming, or evaluating a candidate, Codex will display the
   complete `research.yaml` contract and its SHA-256 hash. Confirm that exact
   contract when prompted.
4. If the next experiment requires RunPod, separately authorize the paid GPU run
   and any required infrastructure action when Codex asks.

Codex then reviews the evidence, selects a hypothesis, creates one candidate,
and interprets the result. The repository's deterministic controller and harness
perform validation, benchmarking, promotion, recovery, and persistence behind
the workflow. They still require the confirmed contract hash and explicit paid
GPU authorization, so no experiment starts accidentally.

### Measured results

NVIDIA A100-SXM4-80GB, PyTorch 2.8.0+cu128, float32, causal.
All cases use `num_layers=4` and `ffn_dim=d_model`.

| Case | batch | seq | d_model | heads | Speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 128 | 128 | 4 | 1.89× |
| 2 | 1 | 128 | 128 | 4 | **8.51×** |
| 3 | 4 | 128 | 128 | 4 | 7.89× |
| 4 | 16 | 128 | 128 | 4 | 4.66× |
| 5 | 128 | 128 | 128 | 4 | 2.11× |
| 6 | 10000 | 128 | 128 | 4 | 2.19× |
| 7 | 64 | 128 | 32 | 4 | 3.27× |
| 8 | 64 | 128 | 1024 | 4 | 1.14× |
| 9 | 64 | 128 | 128 | 1 | 1.90× |
| 10 | 64 | 128 | 128 | 2 | 2.12× |
| 11 | 64 | 128 | 128 | 16 | 2.88× |
| 12 | 64 | 32 | 128 | 4 | 4.99× |
| 13 | 64 | 1024 | 128 | 4 | 4.76× |
| 14 | 32 | 100000 | 1024 | 16 | no runnable baseline |

**Geometric mean over cases 1–13: 3.13×** (3.1165× for the single generalist
implementation; 3.1316× once proven per-shape specialists are composed in).

The shape of that table is the finding. The largest wins are exactly the smallest
shapes — case 2 (`batch=1`) at 8.5× — because those are pure launch overhead, and
CUDA-graph replay removes almost all of it. The smallest win is case 8
(`d_model=1024`), the only shape large enough to be genuinely compute-bound,
where there is little overhead left to remove and cuBLAS was already near the
roofline. `assets/experiment-speedups.html` plots the speedup trajectory across
the candidate history.

---

## Limitations and what I would improve with more time

### Limitations

**Case 14 was never measured against a reference.** The largest official shape
(`batch=32, seq=100000, d_model=1024, heads=16`) needs a
`[32, 16, 100000, 100000]` fp32 score matrix in the baseline — petabytes. The
baseline cannot run it, so there is no ground truth to compare against and no
honest speedup to report. It is archived by candidate latency only, and excluded
from the geomean rather than filled in with a fabricated number. The submission
falls back to its TorchScript generalist there, which does run (SDPA never
materializes the score matrix) but is unverified.

**float32 only.** The accuracy gate is `abs ≤ 0.002 OR rel ≤ 0.02`. Outputs reach
magnitude ~4–8, where one bf16 ULP is ~0.03. Elements that land near zero through
cancellation carry that ~0.03 absolute error and fail the `abs` test, while `rel`
is meaningless for them. In bf16 only a *bit-identical* implementation passes —
which forbids optimization outright. The same code fails every shape in bf16 and
passes every shape in fp32. This costs no hardware capability (fp32 still uses
TF32 tensor cores) and matches the benchmark's own default, but it does mean the
headline reduced-precision lever is unavailable under this gate.

**The dispatch table is hardware- and shape-specific.** The CUDA-graph key sets in
`final_submission.py` are literal tuples of the official shapes measured on an
A100-80GB. An unlisted shape silently takes the TorchScript generalist — correct,
and still meaningfully faster than baseline, but not the best path. Evidence is
scoped by GPU in the ledger precisely so A100-40GB measurements cannot leak into
A100-80GB selection, but that also means nothing here transfers to an H100 or a
consumer card without re-measuring.

**CUDA-graph replay carries real constraints.** Capture requires static shapes,
static input addresses, and no host synchronization or dynamic control flow in
the captured region, and it holds its memory pool for the process lifetime. Fine
for a fixed benchmark; it would need care in a serving system with variable batch
sizes.

**No hand-written kernel shipped.** Everything is PyTorch-level: fusion, dispatch,
graph capture, and the homogeneous-coordinate bias trick. Triton and raw CUDA were
in scope and repeatedly proposed, but no hand-written kernel beat the promotion
gate within the time available.

**The search stalled before it finished.** The event log records several runs of
five consecutive non-improvements. The loop is good at *deciding which candidate
wins* — that part is statistically sound — but weak at *choosing what to try
next*, the open gap documented in `POLICIES.md`. Late candidates cluster around
variations of the same CUDA-graph idea.

### What I would improve given more time

1. **An agent-agnostic autoresearch harness.** The current loop depends on the
   Codex app and its repo-local skill. With more time, I would build a standard
   harness and model-provider interface so any AI coding agent could inspect the
   evidence, propose a candidate, and run the same deterministic verify → measure
   → record cycle—not only a Codex agent.
2. **A fused LayerNorm + residual Triton kernel.** The highest-value remaining
   hand-written kernel and the one Inductor is least likely to match. At these
   sizes the normalization epilogue is a pure bandwidth cost sitting between two
   GEMMs.
3. **A real case-14 specialist.** Chunked FlashAttention with an independently
   derived reference — a blockwise CPU or reduced-shape computation — so the
   largest shape gets verified rather than assumed. It is the one case where the
   baseline cannot compete at all, so it is also the largest untapped headroom.
4. **Close the P1 gap: choose experiments by measured gap-to-roofline.**
   `tools/roofline.py` already estimates the speed-of-light bound per shape.
   Routing each iteration to *tuning* versus *structural change* based on that gap
   would stop the search spending five iterations on variants of an idea already
   within noise of its ceiling.
5. **Shape-generalized dispatch.** Replace the literal tuple key sets with measured
   *regions* of shape space, so a shape the benchmark did not publish still lands
   on the right path instead of defaulting to the generalist.
6. **Broaden the hardware evidence.** Re-run the full sweep on at least one other
   GPU class to separate a genuine algorithmic win from an A100 artifact. The
   ledger is already hardware-scoped and would support this unchanged.
7. **A hardened accuracy story for reduced precision.** Selective fp16/bf16 —
   keeping softmax and normalization accumulations in fp32 while the GEMMs run
   reduced — might clear the absolute-error gate where a wholesale cast cannot.
   That was never tested and is the most likely large win left on the table.

