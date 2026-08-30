#!/usr/bin/env python3
"""Speed-of-light bounds per shape, and the gap the current champion leaves.

This is the loop's mode-switch signal. Autotuning moves a kernel toward the
roofline it is already on; it cannot move the roofline. So when a shape is close
to SOL, tuning is finished and the only remaining move is structural. When a
shape is far from BOTH rooflines it is latency bound -- neither more bandwidth
nor more FLOPs is the constraint, so the fix is fusing kernels away or removing
launches, which tuning parameters cannot do either.

A100-SXM4-40GB: 1555 GB/s, 19.5 TFLOP/s fp32, 156 TFLOP/s TF32 tensor core
(TF32 is what matters here: allow_tf32 defaults to True in the benchmark).
"""
import json
import sys

BW = 1555e9
TF32 = 156e12
BYTES = 4  # fp32


def bounds(s):
    B, S, D, H, F, L = (s["batch_size"], s["seq_len"], s["d_model"],
                        s["num_heads"], s["ffn_dim"], s["num_layers"])
    T = B * S
    hd = D // H

    # FLOPs per layer: QKV + out projections, causal attention (half the
    # scores), and the two FFN matmuls.
    proj = 4 * 2 * T * D * D
    attn = 2 * B * H * S * S * hd          # QK^T and PV, halved for causal
    ffn = 2 * 2 * T * D * F
    flops = L * (proj + attn + ffn)

    # Memory: the optimistic bound is one read of the input activation and one
    # write of the output per layer, plus weights once. Real implementations
    # move more; this is deliberately the floor, so the gap is not flattered.
    weights = BYTES * (4 * D * D + 2 * D * F)
    act = BYTES * T * D
    mem = L * (2 * act + weights)

    t_mem = mem / BW
    t_flop = flops / TF32
    return t_mem * 1e3, t_flop * 1e3, max(t_mem, t_flop) * 1e3


def gaps(report_path):
    """case id -> (actual_ms, sol_ms, gap, bound) for one full-sweep report."""
    with open(report_path) as f:
        report = json.load(f)
    out = {}
    for c in report["cases"]:
        o = c.get("candidate")
        if not o:
            continue
        t_mem, t_flop, sol = bounds(c["shape"])
        out[str(c["case"])] = (o["median_ms"], sol, o["median_ms"] / sol,
                               "memory" if t_mem > t_flop else "compute")
    return out


def main(path):
    print(f"{'case':>4} {'actual ms':>10} {'SOL ms':>9} {'gap':>7} {'bound':>9}  verdict")
    for case, (actual, sol, gap, bound) in gaps(path).items():
        if gap < 1.5:
            verdict = "at SOL - done"
        elif gap < 4:
            verdict = f"tune ({bound}-bound)"
        else:
            verdict = "LATENCY-BOUND - restructure"
        print(f"{case:>4} {actual:>10.3f} {sol:>9.3f} {gap:>6.1f}x {bound:>9}  {verdict}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/v001_fp32_full.json")
