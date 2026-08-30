#!/usr/bin/env python3
"""Generate a candidate that dispatches each official shape to its elite.

The generated file follows the normal candidate contract. Selection happens in
build_model(), where the complete static config is already known, so there is no
per-forward Python shape check and only the selected implementation is built.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import ledger  # noqa: E402
import shapes  # noqa: E402


def config_key(shape):
    return (
        shape["batch_size"], shape["d_model"], shape["num_heads"],
        shape["seq_len"], shape["num_layers"], shape["ffn_dim"], True,
    )


def choose_mapping(dtype: str):
    elites = ledger.per_shape_elites(dtype=dtype, top_k=1, proven_only=True)
    champion = ledger.champion(ledger.FULL_CASES, dtype=dtype)
    if not champion:
        raise RuntimeError(f"no full-sweep {dtype} champion is available")
    fallback = champion["candidate"]
    available = {path.name for path in (ROOT / "candidates").glob("*.py")}
    if fallback not in available:
        raise RuntimeError(f"champion source is missing: candidates/{fallback}")

    mapping = {}
    notes = []
    for shape in shapes.as_dicts():
        case = str(shape["case"])
        selected = elites.get(case, [{}])[0].get("candidate", fallback)
        if selected not in available:
            selected = fallback
        mapping[config_key(shape)] = selected
        notes.append((shape["case"], selected))
    return fallback, mapping, notes


def render(dtype: str) -> str:
    fallback, mapping, notes = choose_mapping(dtype)
    mapping_literal = "{\n" + "\n".join(
        f"    {key!r}: {candidate!r},"
        for key, candidate in mapping.items()
    ) + "\n}"
    note_text = ", ".join(f"{case}:{candidate}" for case, candidate in notes)
    return f'''"""Portfolio dispatcher generated from the {dtype} per-shape elite archive.

Mapping: {note_text}
"""
import importlib.util
import os
import sys

import torch.nn as nn


CONFIG_TO_CANDIDATE = {mapping_literal}
FALLBACK_CANDIDATE = {fallback!r}
_MODULES = {{}}


def _config_key(config):
    return (
        config.batch_size, config.d_model, config.num_heads, config.seq_len,
        config.num_layers, config.ffn_dim, bool(config.causal),
    )


def _load_candidate(filename):
    if filename in _MODULES:
        return _MODULES[filename]
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    name = "portfolio_" + os.path.splitext(filename)[0]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    _MODULES[filename] = module
    return module


class SelectedModel(nn.Module):
    def __init__(self, config, bench):
        super().__init__()
        filename = CONFIG_TO_CANDIDATE.get(_config_key(config), FALLBACK_CANDIDATE)
        self._implementation = _load_candidate(filename)
        self._bench = bench
        self.inner = self._implementation.build_model(config, bench)

    def forward(self, x, valid_token_mask=None):
        return self.inner(x, valid_token_mask)


def build_model(config, bench):
    return SelectedModel(config, bench)


def load_from_baseline(model, baseline):
    loader = getattr(model._implementation, "load_from_baseline", None)
    if loader is not None:
        loader(model.inner, baseline)
    else:
        model._bench.copy_model_weights(baseline, model.inner, strict=True)
'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", default=ledger.DEFAULT_DTYPE)
    parser.add_argument("--out", help="generated candidate path; default stdout")
    args = parser.parse_args()
    source = render(args.dtype)
    if args.out:
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(source)
        print(f"wrote {destination}")
    else:
        print(source, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
