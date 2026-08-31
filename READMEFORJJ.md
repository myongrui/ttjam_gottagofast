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

## Canonical autoresearch implementation

### Model layer — invent the structure
- New fusion
- New Triton kernel
- Different attention implementation
- Removed synchronization

### Mechanical tuner — optimize that structure
- Sweep BLOCK_M/N/K
- Sweep num_warps
- Sweep num_stages
- Sweep vector widths and launch configurations
- Benchmark valid combinations on relevant shapes
- Select the best configuration per shape
- Freeze the resulting shape-to-configuration table

### Portfolio policy — decide whether it wins
- Run the tuned candidate against the incumbent
- Apply correctness and confidence-interval gates
- Preserve global winners and shape specialists

## How we work in parallel

Do not concurrently edit these shared control files:

- `tools/harness.py`
- `tools/event_store.py`
- `tools/iterate.sh`
- `research/events.jsonl`

## Building the final submission

After different shape specialists become proven elites:

```bash
python3 tools/controller.py contract
# After the user confirms that exact contract:
python3 tools/build_dispatcher.py --out candidates/v049_portfolio.py
python3 tools/controller.py race --contract-hash <confirmed-hash> \
  --paid-gpu-authorized --profile general candidates/v049_portfolio.py
```

The dispatcher uses the global champion as the fallback for unknown shapes.
For the approved A100-80GB float32 scope, migrated archived evidence keeps
`v038_torchscript_dispatched_transformer_candidate.py` as the 2.2969x fallback
and selects `v024_centralized_attention_mask_dispatch.py` as the proven case-9
specialist after applying the unchanged 2% dispatcher margin.

For the detailed model-facing search policy and established findings, read
`LOOP.md`.
