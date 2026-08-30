"""v003 -- v001 plus torch.compile with CUDA graphs (v002's hypothesis, fixed).

Same hypothesis as v002: these shapes are launch-bound, so fusing the
LayerNorm/residual/GELU glue and replaying the layer stack as a captured CUDA
graph should beat re-issuing every launch. v002 failed on an integration bug,
not on the idea -- assigning `model.forward = compiled.__call__` recurses,
because the compiled wrapper dispatches back through `model.forward`.

Here the compiled callable is held in a wrapper module that delegates to it,
so there is no cycle. Weights are copied into the inner module before compiling.
"""
import os as _os
import sys as _sys
from importlib import util as _util

import torch
import torch.nn as nn

_v001_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                           "v001_sdpa_fused_qkv.py")
_spec = _util.spec_from_file_location("v001_base", _v001_path)
_v001 = _util.module_from_spec(_spec)
_sys.modules["v001_base"] = _v001
_spec.loader.exec_module(_v001)


class CompiledModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.inner = _v001.Model(config)
        self._compiled = None

    def compile_now(self):
        self._compiled = torch.compile(self.inner, mode="reduce-overhead",
                                       fullgraph=False)

    def forward(self, x, valid_token_mask=None):
        fn = self._compiled if self._compiled is not None else self.inner
        return fn(x, valid_token_mask)


def build_model(config, bench):
    return CompiledModel(config)


def load_from_baseline(model, baseline):
    _v001.load_from_baseline(model.inner, baseline)
    model.compile_now()
