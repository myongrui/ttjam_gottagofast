#!/usr/bin/env python3
"""Mechanical tuner: sweep kernel launch configurations, freeze the best per shape.

The generator writes kernels; nobody ever swept their parameters. Every Triton
candidate in this repo hardcodes num_warps=4 and derives its block size, so a
kernel measured at 1.74x is an *untuned* data point, not evidence that its
structure is wrong. This closes that gap without spending a single API token.

    generate valid variants -> upload once -> measure all in one pod process
    -> pick the best config per shape -> freeze a shape->config table

Correctness is re-checked per (variant, shape): a bad block size reads out of
bounds, so a configuration can be fast and wrong. The harness gate applies
unchanged.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
REMOTE = "/workspace/techjam"

# Hardware-valid ranges for sm_80. num_warps beyond 16 exceeds the 1024-thread
# block limit for these kernels; num_stages above 4 exhausts shared memory on
# the small tiles these candidates use.
DEFAULT_SPACE = {"num_warps": [1, 2, 4, 8, 16], "num_stages": [1, 2, 3, 4]}


def find_sites(src: str) -> Dict[str, Any]:
    """What is actually tunable in this candidate."""
    return {
        "num_warps": len(re.findall(r"num_warps\s*=\s*\d+", src)),
        "num_stages": len(re.findall(r"num_stages\s*=\s*\d+", src)),
        "kernels": len(re.findall(r"@triton\.jit", src)),
    }


def make_variant(src: str, config: Dict[str, int]) -> str:
    """Rewrite launch parameters. num_stages is injected beside num_warps when
    the candidate does not already set it -- most generated kernels omit it."""
    out = re.sub(r"num_warps\s*=\s*\d+", f"num_warps={config['num_warps']}", src)
    if re.search(r"num_stages\s*=\s*\d+", out):
        out = re.sub(r"num_stages\s*=\s*\d+", f"num_stages={config['num_stages']}", out)
    else:
        out = re.sub(r"(num_warps=\d+)", rf"\1, num_stages={config['num_stages']}", out)
    return out


def build_variants(candidate: Path, space: Dict[str, List[int]],
                   outdir: Path) -> List[Dict[str, Any]]:
    src = candidate.read_text()
    sites = find_sites(src)
    if not sites["kernels"]:
        raise SystemExit(f"{candidate.name} defines no @triton.jit kernel — "
                         f"nothing to tune")
    if not sites["num_warps"]:
        raise SystemExit(f"{candidate.name} has no num_warps launch parameter")
    print(f"{candidate.name}: {sites['kernels']} kernel(s), "
          f"{sites['num_warps']} launch site(s)")

    outdir.mkdir(parents=True, exist_ok=True)
    for stale in outdir.glob("*.py"):
        stale.unlink()

    variants = []
    keys = sorted(space)
    for combo in itertools.product(*(space[k] for k in keys)):
        cfg = dict(zip(keys, combo))
        name = "_".join(f"{k[4:]}{v}" for k, v in sorted(cfg.items()))
        path = outdir / f"{candidate.stem}__{name}.py"
        path.write_text(make_variant(src, cfg))
        variants.append({"name": name, "config": cfg,
                         "path": f"{REMOTE}/{outdir.relative_to(ROOT)}/{path.name}"})
    print(f"generated {len(variants)} variants in {outdir.relative_to(ROOT)}")
    return variants


def pick_best(report: Dict[str, Any]) -> Dict[str, Any]:
    """Best correct configuration per shape, plus the best single global one."""
    per_shape: Dict[str, Any] = {}
    global_scores: Dict[str, List[float]] = {}
    for entry in report["results"]:
        for case, rec in entry["cases"].items():
            if not rec.get("correct") or not rec.get("speedup"):
                continue
            best = per_shape.get(case)
            if best is None or rec["speedup"] > best["speedup"]:
                per_shape[case] = {"config": entry["config"],
                                   "speedup": rec["speedup"], "ms": rec["ms"]}
            global_scores.setdefault(entry["name"], []).append(rec["speedup"])
    import math
    global_best, global_val = None, 0.0
    for name, scores in global_scores.items():
        g = math.exp(sum(map(math.log, scores)) / len(scores))
        if g > global_val:
            cfg = next(e["config"] for e in report["results"] if e["name"] == name)
            global_best, global_val = cfg, g
    return {"per_shape": per_shape, "global": global_best, "global_geomean": global_val}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("candidate")
    p.add_argument("--cases", default="1,2,12")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--warps", default="1,2,4,8,16")
    p.add_argument("--stages", default="1,2,3,4")
    p.add_argument("--dry-run", action="store_true",
                   help="generate variants without spending pod time")
    args = p.parse_args()

    candidate = Path(args.candidate)
    space = {"num_warps": [int(x) for x in args.warps.split(",")],
             "num_stages": [int(x) for x in args.stages.split(",")]}
    outdir = ROOT / "candidates" / "tuned" / candidate.stem
    variants = build_variants(candidate, space, outdir)

    manifest = ROOT / "results" / f"tuning_{candidate.stem}_variants.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(variants, indent=2))
    if args.dry_run:
        print("dry-run: not measuring")
        return 0

    files = [str(p.relative_to(ROOT)) for p in sorted(outdir.glob("*.py"))]
    subprocess.run([str(ROOT / "tools" / "podsync"), REMOTE,
                    "tools/tune_runner.py", "tools/harness.py", "tools/shapes.py",
                    str(manifest.relative_to(ROOT))] + files,
                   cwd=ROOT, check=True)

    remote_out = f"{REMOTE}/results/tuning_{candidate.stem}.json"
    subprocess.run([str(ROOT / "tools" / "podrun"),
                    f"cd {REMOTE} && python tools/tune_runner.py "
                    f"--variants {REMOTE}/results/tuning_{candidate.stem}_variants.json "
                    f"--cases {args.cases} --dtype {args.dtype} "
                    f"--out {remote_out} 2>&1 | tail -60"], cwd=ROOT, check=True)

    local_out = ROOT / "results" / f"tuning_{candidate.stem}.json"
    subprocess.run([str(ROOT / "tools" / "podget"), remote_out, str(local_out)],
                   cwd=ROOT, check=True)

    report = json.loads(local_out.read_text())
    best = pick_best(report)
    print("\n=== best configuration per shape ===")
    for case in sorted(best["per_shape"], key=int):
        b = best["per_shape"][case]
        print(f"  case {case:>2}: {b['config']}  {b['speedup']:.4f}x")
    print(f"\nbest single config: {best['global']}  "
          f"geomean {best['global_geomean']:.4f}x")

    table = ROOT / "results" / f"tuning_{candidate.stem}_best.json"
    table.write_text(json.dumps(best, indent=2))
    print(f"froze shape->config table into {table.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
