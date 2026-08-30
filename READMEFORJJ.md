## What we are doing

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
length, dimensions, heads and causal mode. 

`tools/build_dispatcher.py` creates the final combined candidate. It performs
selection once when the model is built, so there is no shape-dispatch overhead
on every forward pass.

## How we work in parallel

Do not concurrently edit these shared control files:

- `tools/harness.py`
- `tools/ledger.py`
- `tools/iterate.sh`
- `results/ledger.jsonl`

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

For the detailed model-facing search policy and established findings, read
`LOOP.md`.
