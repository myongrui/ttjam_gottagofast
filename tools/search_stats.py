"""Dependency-free statistics used by the GPU search harness."""
from __future__ import annotations

import math
import random
import statistics
from typing import Any, Dict, List


def percentile(values: List[float], q: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def geomean(values: List[float]) -> float:
    if not values or any(v <= 0 for v in values):
        raise ValueError("geomean requires positive values")
    return math.exp(statistics.fmean(math.log(v) for v in values))


def bootstrap_geomean_ci(values: List[float], confidence: float = 0.95,
                         draws: int = 4000, seed: int = 0) -> List[float]:
    """Deterministic percentile-bootstrap CI over independent timing rounds.

    Individual CUDA-event samples within one launch batch are correlated. The
    independently ordered round medians are therefore the resampling unit.
    With only one round there is no honest uncertainty estimate, so the point
    estimate is returned as both bounds and callers should treat it as weak.
    """
    if not values:
        raise ValueError("values must not be empty")
    if len(values) == 1:
        return [values[0], values[0]]
    logs = [math.log(v) for v in values]
    rng = random.Random(seed)
    means = []
    for _ in range(draws):
        means.append(statistics.fmean(rng.choice(logs) for _ in logs))
    alpha = (1.0 - confidence) / 2.0
    return [math.exp(percentile(means, alpha)),
            math.exp(percentile(means, 1.0 - alpha))]


def compare_timings(numerator: Dict[str, Any], denominator: Dict[str, Any],
                    seed: int = 0) -> Dict[str, Any]:
    """Latency ratio with paired-round uncertainty (higher is better)."""
    round_speedups = [
        a / b for a, b in zip(numerator["round_medians_ms"],
                             denominator["round_medians_ms"])
    ]
    paired = geomean(round_speedups)
    return {
        "speedup": numerator["median_ms"] / denominator["median_ms"],
        "speedup_wall": numerator["wall_ms"] / denominator["wall_ms"],
        "paired_speedup": paired,
        "paired_speedup_ci95": bootstrap_geomean_ci(
            round_speedups, seed=seed),
        "round_speedups": round_speedups,
    }
