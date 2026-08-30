#!/usr/bin/env python3
"""Evaluation harness for the transformer-kernel optimization loop.

Scores a candidate against the official baseline across the 14 published shapes.
Correctness is a gate, not a scoring term: a candidate that fails any runnable
shape scores nothing regardless of how fast it is.

The integrity checks are load-bearing. The literature on LLM-driven kernel
generation finds measurement exploits to be the dominant failure mode -- see
KernelBench-Verified (arXiv 2607.16241), where frontier models drop from 1.43x
to 0.88x geomean once evaluation is hardened, and arXiv 2509.14279, which
catalogues three families: concurrency exploits (work on a side stream escapes
torch.cuda.Event timing on the current stream -- precisely what the official
benchmark measures with), state caching (the official timing loop reuses ONE
fixed input for all repeats, so memoizing it looks like a huge win), and
environment manipulation (mutating torch global flags or the baseline itself).
Each check below maps to one of those families.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import statistics
import sys
import time
import traceback
from typing import Any, Dict, List

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import shapes as shapes_mod  # noqa: E402
from search_stats import bootstrap_geomean_ci, compare_timings, geomean  # noqa: E402


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: @dataclass resolves cls.__module__ via sys.modules,
    # and a manually-loaded module that is absent from it raises AttributeError.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# environment snapshot (guards the "environment manipulation" family)
# --------------------------------------------------------------------------

def env_snapshot() -> Dict[str, Any]:
    return {
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "grad_enabled": torch.is_grad_enabled(),
        "default_dtype": str(torch.get_default_dtype()),
        "num_threads": torch.get_num_threads(),
    }


def state_fingerprint(model: torch.nn.Module) -> str:
    """Cheap order-stable fingerprint of every parameter, to detect mutation."""
    acc = []
    for name, p in sorted(model.state_dict().items()):
        t = p.detach().float()
        acc.append(f"{name}:{t.sum().item():.6e}:{t.abs().max().item():.6e}")
    return "|".join(acc)


# --------------------------------------------------------------------------
# model construction
# --------------------------------------------------------------------------

def make_config(bench, shape: Dict[str, int], causal: bool = True):
    return bench.TransformerConfig(
        batch_size=shape["batch_size"],
        seq_len=shape["seq_len"],
        d_model=shape["d_model"],
        num_heads=shape["num_heads"],
        ffn_dim=shape["ffn_dim"],
        num_layers=shape["num_layers"],
        causal=causal,
    )


def build_model_from_baseline(bench, candidate, config, baseline, device, dtype):
    """Build one candidate and give it the baseline's exact weights."""
    model = candidate.build_model(config, bench)
    model = model.to(device=device, dtype=dtype).eval()
    if hasattr(candidate, "load_from_baseline"):
        candidate.load_from_baseline(model, baseline)
    else:
        bench.copy_model_weights(baseline, model, strict=True)
    return model


def build_models(bench, candidate, incumbent, config, device, dtype):
    """Baseline, challenger, and optional incumbent with identical weights."""
    baseline = bench.BaselineTransformer(config).to(device=device, dtype=dtype).eval()
    model = build_model_from_baseline(
        bench, candidate, config, baseline, device, dtype)
    incumbent_model = None
    if incumbent is not None:
        incumbent_model = build_model_from_baseline(
            bench, incumbent, config, baseline, device, dtype)
    return baseline, model, incumbent_model


# --------------------------------------------------------------------------
# correctness
# --------------------------------------------------------------------------

# A deliberately varied input suite. A single distribution invites hardcoded
# bypasses; scale sweeps in particular stress softmax stability, and the padding
# variants exercise the masking path that the fixed benchmark input never does.
VARIANTS = [
    {"name": "normal",       "seed": 11, "input_scale": 1.0,  "padding_ratio": 0.0},
    {"name": "large_scale",  "seed": 22, "input_scale": 8.0,  "padding_ratio": 0.0},
    {"name": "small_scale",  "seed": 33, "input_scale": 0.05, "padding_ratio": 0.0},
    {"name": "padded",       "seed": 44, "input_scale": 1.0,  "padding_ratio": 0.3},
    {"name": "padded_heavy", "seed": 55, "input_scale": 2.0,  "padding_ratio": 0.7},
]


def check_correctness(bench, baseline, model, config, device, dtype, rtol, atol,
                      variants=VARIANTS) -> Dict[str, Any]:
    results = []
    ok = True
    for v in variants:
        x, mask = bench.generate_random_case(
            config=config, device=device, dtype=dtype, seed=v["seed"],
            padding_ratio=v["padding_ratio"], input_scale=v["input_scale"],
        )
        with torch.inference_mode():
            ref = baseline(x, mask)
            got = model(x, mask)
        res = bench.compare_outputs(ref, got, rtol=rtol, atol=atol)
        passed = bool(getattr(res, "passed", False))
        results.append({
            "variant": v["name"],
            "passed": passed,
            "max_abs": float(getattr(res, "max_abs_error", float("nan"))),
            "max_rel": float(getattr(res, "max_relative_error", float("nan"))),
        })
        ok = ok and passed
        del x, mask, ref, got
        torch.cuda.empty_cache()
    return {"passed": ok, "variants": results}


def check_output_depends_on_input(bench, model, config, device, dtype) -> bool:
    """Guards the 'state caching' family: a memoizing candidate that replays a
    cached result will not react to a changed input."""
    x1, m1 = bench.generate_random_case(
        config=config, device=device, dtype=dtype, seed=101,
        padding_ratio=0.0, input_scale=1.0)
    x2, m2 = bench.generate_random_case(
        config=config, device=device, dtype=dtype, seed=202,
        padding_ratio=0.0, input_scale=1.0)
    with torch.inference_mode():
        o1 = model(x1, m1)
        o2 = model(x2, m2)
        o1b = model(x1, m1)
    differs = not torch.allclose(o1.float(), o2.float(), rtol=1e-3, atol=1e-3)
    repeatable = torch.allclose(o1.float(), o1b.float(), rtol=1e-3, atol=1e-3)
    del x1, x2, m1, m2, o1, o2, o1b
    torch.cuda.empty_cache()
    return bool(differs and repeatable)


# --------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------

def wall_time_model(model, x, mask, device, repeats):
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(repeats):
            model(x, mask)
    torch.cuda.synchronize(device)
    return (time.perf_counter() - t0) * 1e3 / repeats


def time_models(bench, models, x, mask, device, warmup, repeats, rounds):
    """Time several models in paired, rotating-order rounds.

    The event timing reproduces the official script exactly so our numbers are
    comparable to the submitted ones. Rotating the order makes every model
    occupy every timing position instead of always measuring the challenger on
    a colder or hotter GPU. Round medians are retained for paired comparisons.

    A synchronized wall-clock pass additionally catches work pushed onto a side
    stream, which can escape CUDA events recorded on the current stream.
    """
    if rounds < 1:
        raise ValueError("rounds must be positive")
    names = list(models)
    samples: Dict[str, List[float]] = {name: [] for name in names}
    round_medians: Dict[str, List[float]] = {name: [] for name in names}

    for model in models.values():
        bench.warmup_model(model, x, mask, warmup, device)

    for round_index in range(rounds):
        shift = round_index % len(names)
        order = names[shift:] + names[:shift]
        if (round_index // len(names)) % 2:
            order.reverse()
        for name in order:
            batch = bench.benchmark_once(
                models[name], x, mask, repeats, device)
            samples[name].extend(batch)
            round_medians[name].append(statistics.median(batch))

    # Wall time is an integrity cross-check rather than the promotion signal.
    # Reverse the last event order so the same model is not always measured last.
    wall_ms = {}
    for name in reversed(order):
        wall_ms[name] = wall_time_model(
            models[name], x, mask, device, repeats)

    return {
        name: {
            "median_ms": statistics.median(samples[name]),
            "mean_ms": statistics.fmean(samples[name]),
            "min_ms": min(samples[name]),
            "wall_ms": wall_ms[name],
            "n": len(samples[name]),
            "round_medians_ms": round_medians[name],
        }
        for name in names
    }


def evaluate_shape(bench, candidate, incumbent, shape, device, dtype,
                   args) -> Dict[str, Any]:
    out: Dict[str, Any] = {"case": shape["case"], "shape": shape}
    config = make_config(bench, shape)

    # Pre-screen: the baseline materializes a [B,H,S,S] fp32 score matrix. When
    # that exceeds device memory the baseline cannot run at all, and there is no
    # reference to compare against on this shape.
    score_gib = shapes_mod.baseline_score_bytes(shape) / 2**30
    total_gib = torch.cuda.get_device_properties(device).total_memory / 2**30
    out["baseline_score_gib"] = round(score_gib, 3)
    baseline_feasible = score_gib < 0.5 * total_gib

    try:
        baseline, model, incumbent_model = build_models(
            bench, candidate, incumbent, config, device, dtype)
    except Exception as e:
        out["status"] = "build_error"
        out["error"] = f"{type(e).__name__}: {e}"
        return out

    try:
        if baseline_feasible:
            corr = check_correctness(bench, baseline, model, config, device, dtype,
                                     args.rtol, args.atol)
            out["correctness"] = corr
            if not corr["passed"]:
                out["status"] = "incorrect"
                return out
            out["input_dependence_ok"] = check_output_depends_on_input(
                bench, model, config, device, dtype)
            if not out["input_dependence_ok"]:
                out["status"] = "integrity_failed"
                out["error"] = "output did not respond to a changed input (caching?)"
                return out
        else:
            # Build both implementations with the real config (so exact-shape
            # dispatch selects the same code path), then exercise them with a
            # shorter sequence for which the explicit baseline is feasible.
            proxy_shape = dict(shape)
            proxy_shape["seq_len"] = min(shape["seq_len"], args.proxy_seq_len)
            proxy_config = make_config(bench, proxy_shape)
            corr = check_correctness(
                bench, baseline, model, proxy_config, device, dtype,
                args.rtol, args.atol)
            out["correctness"] = corr
            out["correctness"]["proxy_seq_len"] = proxy_shape["seq_len"]
            if not corr["passed"]:
                out["status"] = "incorrect"
                return out
            out["input_dependence_ok"] = check_output_depends_on_input(
                bench, model, proxy_config, device, dtype)
            if not out["input_dependence_ok"]:
                out["status"] = "integrity_failed"
                out["error"] = "proxy output did not respond to changed input"
                return out

        x, mask = bench.generate_random_case(
            config=config, device=device, dtype=dtype, seed=args.seed + 100000,
            padding_ratio=args.padding_ratio, input_scale=1.0)

        models = {"candidate": model}
        if baseline_feasible:
            models["baseline"] = baseline
        if incumbent_model is not None:
            models["incumbent"] = incumbent_model
        timings = time_models(bench, models, x, mask, device, args.warmup,
                              args.repeats, args.rounds)
        cand_t = timings["candidate"]
        out["candidate"] = cand_t
        out["peak_mem_gib"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 3)

        if baseline_feasible:
            base_t = timings["baseline"]
            out["baseline"] = base_t
            comparison = compare_timings(
                base_t, cand_t, seed=args.seed + shape["case"])
            out.update(comparison)
            # Concurrency-exploit signal: event-measured time far below the
            # synchronized wall time means work escaped the timed stream.
            ratio = cand_t["wall_ms"] / max(cand_t["median_ms"], 1e-9)
            out["wall_to_event_ratio"] = round(ratio, 3)
            out["stream_ok"] = ratio < args.max_wall_ratio
            out["status"] = "ok" if out["stream_ok"] else "integrity_failed"
            if not out["stream_ok"]:
                out["error"] = (f"wall/event ratio {ratio:.2f} exceeds "
                                f"{args.max_wall_ratio}: work may be off the timed stream")
        else:
            out["status"] = "ok_no_baseline"

        if incumbent_model is not None:
            incumbent_t = timings["incumbent"]
            out["incumbent"] = incumbent_t
            out["vs_incumbent"] = compare_timings(
                incumbent_t, cand_t, seed=args.seed + 1000 + shape["case"])
    except torch.cuda.OutOfMemoryError as e:
        out["status"] = "oom"
        out["error"] = str(e).split("\n")[0]
    except Exception as e:
        out["status"] = "runtime_error"
        out["error"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()[-1500:]
    finally:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", required=True)
    p.add_argument("--incumbent", default="",
                   help="optional champion candidate for direct paired timing")
    p.add_argument("--bench", default="/workspace/techjam/torch_transformer_benchmark.py")
    p.add_argument("--cases", default="", help="comma-separated case ids; default all")
    p.add_argument("--dtype", default="bfloat16", choices=("float16", "bfloat16", "float32"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--rtol", type=float, default=0.02)
    p.add_argument("--atol", type=float, default=0.002)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--padding-ratio", type=float, default=0.0)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--max-wall-ratio", type=float, default=1.6)
    p.add_argument("--promotion-margin", type=float, default=0.02,
                   help="minimum relative gain required for promotion")
    p.add_argument("--proxy-seq-len", type=int, default=1024,
                   help="correctness sequence length when the baseline is infeasible")
    p.add_argument("--out", default="")
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]

    bench = load_module(args.bench, "bench")
    candidate = load_module(args.candidate, "candidate")
    incumbent = (load_module(args.incumbent, "incumbent_candidate")
                 if args.incumbent else None)

    env_before = env_snapshot()
    only = {int(c) for c in args.cases.split(",") if c.strip()} or None
    todo = shapes_mod.as_dicts(only)

    report: Dict[str, Any] = {
        "candidate": os.path.basename(args.candidate),
        "candidate_doc": (candidate.__doc__ or "").strip(),
        "incumbent": os.path.basename(args.incumbent) if args.incumbent else None,
        "dtype": args.dtype,
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cases": [],
    }

    for shape in todo:
        r = evaluate_shape(
            bench, candidate, incumbent, shape, device, dtype, args)
        report["cases"].append(r)
        s = r.get("speedup")
        print(f"case {r['case']:>2}  {r['status']:<16} "
              f"speedup={'n/a' if s is None else format(s, '.3f')}  "
              f"{r.get('error','')}", flush=True)

    env_after = env_snapshot()
    report["env_before"] = env_before
    report["env_after"] = env_after
    report["env_unchanged"] = env_before == env_after

    speedups = [c["speedup"] for c in report["cases"]
                if c.get("status") == "ok" and c.get("speedup")]
    failed = [c["case"] for c in report["cases"]
              if c["status"] not in ("ok", "ok_no_baseline")]
    report["failed_cases"] = failed
    report["n_scored"] = len(speedups)
    report["geomean_speedup"] = (
        math.exp(statistics.fmean(math.log(s) for s in speedups)) if speedups else 0.0)
    # Official baseline score. Any failure or environment mutation
    # zeroes it -- speed is only meaningful on top of a correct, honest run.
    report["score"] = 0.0 if (failed or not report["env_unchanged"]) else report["geomean_speedup"]

    incumbent_cases = [
        c for c in report["cases"]
        if c.get("status") in ("ok", "ok_no_baseline")
        and c.get("vs_incumbent")
    ]
    if incumbent is not None and incumbent_cases:
        n_rounds = min(len(c["vs_incumbent"]["round_speedups"])
                       for c in incumbent_cases)
        round_geomeans = [
            geomean([c["vs_incumbent"]["round_speedups"][i]
                     for c in incumbent_cases])
            for i in range(n_rounds)
        ]
        paired_geomean = geomean(round_geomeans)
        ci95 = bootstrap_geomean_ci(
            round_geomeans, seed=args.seed + 2000)
        if failed or not report["env_unchanged"]:
            decision = "invalid"
        elif ci95[0] > 1.0 + args.promotion_margin:
            decision = "promote"
        elif ci95[1] < 1.0:
            decision = "reject"
        else:
            decision = "uncertain"
        report["incumbent_comparison"] = {
            "incumbent": os.path.basename(args.incumbent),
            "n_cases": len(incumbent_cases),
            "round_geomeans": round_geomeans,
            "paired_geomean": paired_geomean,
            "paired_geomean_ci95": ci95,
            "promotion_margin": args.promotion_margin,
            "decision": decision,
        }

    blob = json.dumps(report, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(blob)
    print("\n=== SUMMARY ===")
    print(f"scored {report['n_scored']} cases | geomean {report['geomean_speedup']:.4f}x "
          f"| failed {failed} | env_unchanged {report['env_unchanged']}")
    if report.get("incumbent_comparison"):
        comparison = report["incumbent_comparison"]
        ci = comparison["paired_geomean_ci95"]
        print(f"vs {comparison['incumbent']}: "
              f"{comparison['paired_geomean']:.4f}x paired | "
              f"95% CI [{ci[0]:.4f}, {ci[1]:.4f}] | "
              f"{comparison['decision']}")
    print(f"SCORE {report['score']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
