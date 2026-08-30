"""v002 -- v001 plus torch.compile with CUDA graphs.

Hypothesis: these shapes are launch-bound, not FLOP-bound. Eleven of the
fourteen have d_model=128, ffn_dim=128, head_dim=32, and v001's largest win was
on the *smallest* shape (case 2, batch=1, 1.61x) -- the signature of fixed
per-launch overhead dominating real work.

Two mechanisms, both aimed at that overhead rather than at arithmetic:
  * Inductor fuses the LayerNorm / residual-add / GELU glue between the GEMMs.
    Those are pure bandwidth ops whose cost is launch + memory traffic.
  * mode="reduce-overhead" enables CUDA graphs, replaying the whole layer stack
    as one captured graph instead of re-issuing every launch. The shapes are
    static per test case, which is exactly the precondition graphs need.

Expected to move the small shapes (1-5, 7, 9-12) most and the large ones
(6, 8, 13) least, since those already amortize launches over real work.

Risk: CUDA graphs require stable input addresses. The harness reuses one input
tensor throughout timing, so capture should hold; if a shape fails, the fallback
is mode="default" for that shape.
"""
import torch

from importlib import util as _util
import os as _os

_v001_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                           "v001_sdpa_fused_qkv.py")
_spec = _util.spec_from_file_location("v001_base", _v001_path)
_v001 = _util.module_from_spec(_spec)
import sys as _sys
_sys.modules["v001_base"] = _v001
_spec.loader.exec_module(_v001)


def build_model(config, bench):
    model = _v001.Model(config)
    model._needs_compile = True
    return model


def load_from_baseline(model, baseline):
    # Reuse v001's weight mapping, then compile. Compiling after the weight copy
    # avoids a recapture, and compile() returns a wrapper whose parameters are
    # shared with the original module.
    _v001.load_from_baseline(model, baseline)
    compiled = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    # Swap the forward in place so the harness keeps its handle on `model`.
    model._compiled = compiled
    model.forward = compiled.__call__
