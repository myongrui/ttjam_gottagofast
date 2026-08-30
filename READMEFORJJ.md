# JJ: 3-minute project handoff

## What we are doing

We are optimizing one Transformer layer implementation for an NVIDIA A100.
The score is the geometric mean of speedup across the published shapes. One
incorrect shape makes the score zero, so correctness and benchmark integrity
come before speed.

Current verified champion: `candidates/v001_sdpa_fused_qkv.py`, **1.6982×**
geomean over cases 1–13 in float32. It combines fused Q/K/V projection with
PyTorch scaled-dot-product attention.

Always confirm the latest state before starting:

```bash
python3 tools/ledger.py status
```

The ledger is the source of truth. A candidate file that is not in the ledger
has not been measured yet.

## The central idea

Different shapes need different implementations:

| Shape regime | Main bottleneck | Good directions |
|---|---|---|
| Small / launch-bound | CUDA launch and memory overhead | Fuse LayerNorm, residuals, masking, bias and GELU |
| Large batch / wide model | GPU throughput and bandwidth | Tensor-core-friendly tiling, GEMM and FFN fusion |
| Long sequence | Attention memory traffic | FlashAttention-style blocked attention |
| Case 14 (`seq=100000`) | Basic feasibility | Never materialize the quadratic attention matrix |

We develop several experimental candidate files, but the judges should receive
**one final candidate**. That candidate chooses the best implementation from the
static model configuration inside `build_model(config, bench)`.

```text
judge shape → build_model(config) → select specialist → run that model
```

Selection uses only legitimate static properties such as batch size, sequence
length, dimensions, heads and causal mode. Never select using input values,
seeds, cached outputs or knowledge of a particular test.

`tools/build_dispatcher.py` creates the final combined candidate. It performs
selection once when the model is built, so there is no shape-dispatch overhead
on every forward pass.

## How we work in parallel

Parallelize coding and analysis, but run GPU benchmarks one at a time. Two
simultaneous pod runs contend for the GPU and invalidate timing.

Suggested split:

- Person A: small/launch-bound cases — profiles `launch` and `heads`.
- Person B: large and exceptional cases — profiles `throughput`, `attention`,
  `long-seq` and `case:14`.
- One person acts as integration owner and runs the shared pod/ledger commands.

Create separate candidate files with owner prefixes, for example:

```text
candidates/a_v005_fused_norm.py
candidates/jj_v006_case14_flash.py
```

Do not concurrently edit these shared control files:

- `tools/harness.py`
- `tools/ledger.py`
- `tools/iterate.sh`
- `results/ledger.jsonl`

## Candidate workflow

1. Read `python3 tools/ledger.py status` and the findings in `LOOP.md`.
2. Choose one mechanism and state which shapes it should improve.
3. Copy the champion or relevant shape elite into a new candidate file.
4. Change one main idea so the measurement teaches us something.
5. Hand the candidate to the integration owner for a race.
6. Keep the result even if it loses globally; it may still be a shape elite.

Run a hypothesis-specific race:

```bash
python3 tools/race.py --profile launch candidates/a_v005_fused_norm.py
python3 tools/race.py --profile case:14 candidates/jj_v006_case14_flash.py
```

Useful profiles:

| Profile | Cases |
|---|---|
| `launch` | 1, 2, 12 |
| `throughput` | 5, 6, 8 |
| `attention` | 9, 11, 13 |
| `heads` | 9, 10, 11 |
| `long-seq` | 12, 13 |
| `general` | 2, 6, 8, 11, 13 |

The race screens the relevant shapes first and advances the strongest valid
challenger to the full sweep. Promotion requires a paired improvement beyond
the measured 2% noise margin.

## Building the final submission

After different shape specialists become proven elites:

```bash
python3 tools/build_dispatcher.py --out candidates/v010_portfolio.py
python3 tools/race.py --profile general candidates/v010_portfolio.py
```

The dispatcher uses the global champion as the fallback for unknown shapes.
Right now every proven float32 elite is still `v001`, so generating it today
would route every published shape to `v001`. Regenerate it after a specialist
is proven.

## Non-negotiable safety rules

- Use float32; bf16 failed the numerical tolerance for optimized attention.
- Never cache outputs across calls.
- Never create side CUDA streams to escape event timing.
- Never change global PyTorch/TF32 settings.
- Never modify or introspect the benchmark from a candidate.
- Case 14 is correctness-checked through a reduced-sequence proxy because the
  reference attention matrix would require roughly 19 TiB.

## Quick handoff template

```text
Candidate: candidates/jj_v006_case14_flash.py
Hypothesis: blocked attention avoids the quadratic score matrix.
Expected wins: case 14; possibly case 13.
Run: python3 tools/race.py --profile case:14 candidates/jj_v006_case14_flash.py
Notes/risks: <anything the integration owner should know>
```

For the detailed model-facing search policy and established findings, read
`LOOP.md`.
