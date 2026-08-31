#!/usr/bin/env python3
"""Pod-side sweep executor. Measures many kernel configurations in one process.

Reuses harness.py for model construction, correctness and timing, so a tuned
configuration faces exactly the same gate as any other candidate: a config that
is fast but wrong scores nothing. Running every variant in a single process also
keeps the comparison paired -- same clocks, same allocator state, same device.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import traceback

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness  # noqa: E402
import shapes as shapes_mod  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--variants", required=True,
                   help="json file: [{name, path, config}, ...]")
    p.add_argument("--bench", default="/workspace/techjam/torch_transformer_benchmark.py")
    p.add_argument("--cases", required=True)
    p.add_argument("--dtype", default="float32")
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--warmup", type=int, default=15)
    p.add_argument("--rtol", type=float, default=0.02)
    p.add_argument("--atol", type=float, default=0.002)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = torch.device("cuda")
    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]
    bench = harness.load_module(args.bench, "bench")

    with open(args.variants) as f:
        variants = json.load(f)
    only = {int(c) for c in args.cases.split(",") if c.strip()}
    todo = shapes_mod.as_dicts(only)

    # Build every variant model for a shape from ONE baseline, then time them
    # all in a single rotating-order rotation. Measuring configurations against
    # each other in one paired pass is strictly better than separate runs: the
    # differences between block sizes here are small enough that cross-run clock
    # drift would otherwise swamp them.
    loaded = []
    for v in variants:
        try:
            loaded.append((v, harness.load_module(v["path"], f"tuned_{v['name']}")))
        except Exception as e:
            print(f"{v['name']:<28} IMPORT FAILED {e}", flush=True)

    per_shape: dict = {}
    for shape in todo:
        cfg = harness.make_config(bench, shape)
        case = str(shape["case"])
        per_shape[case] = {}
        try:
            baseline = bench.BaselineTransformer(cfg).to(
                device=device, dtype=dtype).eval()
        except Exception as e:
            print(f"case {case}: baseline build failed: {e}", flush=True)
            continue

        models = {"baseline": baseline}
        for v, mod in loaded:
            try:
                m = harness.build_model_from_baseline(
                    bench, mod, cfg, baseline, device, dtype)
                corr = harness.check_correctness(
                    bench, baseline, m, cfg, device, dtype, args.rtol, args.atol,
                    variants=harness.VARIANTS[:3])
                if corr["passed"]:
                    models[v["name"]] = m
                    per_shape[case][v["name"]] = {"correct": True}
                else:
                    per_shape[case][v["name"]] = {
                        "correct": False, "error": "failed correctness"}
                    print(f"{v['name']:<28} case {case:>2}  FAIL correctness", flush=True)
            except torch.cuda.OutOfMemoryError:
                per_shape[case][v["name"]] = {"correct": False, "error": "oom"}
            except Exception as e:
                per_shape[case][v["name"]] = {
                    "correct": False, "error": f"{type(e).__name__}: {e}"[:160]}
                print(f"{v['name']:<28} case {case:>2}  ERROR {e}"[:110], flush=True)

        if len(models) < 2:
            print(f"case {case}: no correct variant to time", flush=True)
            continue

        x, mask = bench.generate_random_case(
            config=cfg, device=device, dtype=dtype, seed=1234 + 100000,
            padding_ratio=0.0, input_scale=1.0)
        timings = harness.time_models(bench, models, x, mask, device,
                                      args.warmup, args.repeats, args.rounds)
        base_ms = timings["baseline"]["median_ms"]
        for name, t in timings.items():
            if name == "baseline":
                continue
            per_shape[case][name].update(
                {"ms": t["median_ms"], "speedup": base_ms / t["median_ms"]})
            print(f"{name:<28} case {case:>2}  ok    "
                  f"{base_ms / t['median_ms']:.4f}x", flush=True)
        del x, mask, models
        torch.cuda.empty_cache()

    # Re-key by variant so the report shape stays {name, config, cases{}}.
    results = []
    for v, _ in loaded:
        results.append({
            "name": v["name"], "config": v["config"],
            "cases": {c: per_shape.get(c, {}).get(v["name"], {})
                      for c in per_shape},
        })

    with open(args.out, "w") as f:
        json.dump({"dtype": args.dtype, "gpu": torch.cuda.get_device_name(device),
                   "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "results": results}, f, indent=2)
    print("WROTE", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
