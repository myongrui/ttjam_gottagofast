"""Pruned graph coverage with three-slot output buffers.

Hypothesis: combining v054's five winning graph configurations with v052's
clone-free three-slot replay will lift the full-sweep lower confidence bound
above the 2% promotion margin while preserving correctness and input dependence.
"""
import importlib.util
import os
import sys

import torch.nn as nn


def _load_ring_parent():
    filename = "v052_ring_buffer_cuda_graphs.py"
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    name = "pruned_ring_parent_v052"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_RING = _load_ring_parent()
_V051 = _RING._PARENT
GRAPH_KEYS = _V051.GRAPH_KEYS | {
    (16, 128, 4, 128, 4, 128, True),
    (64, 128, 4, 32, 4, 128, True),
}


class SelectedModel(nn.Module):
    def __init__(self, config, bench):
        super().__init__()
        self.uses_graph = _V051._config_key(config) in GRAPH_KEYS
        self.inner = (
            _RING.RingGraphReplayModel(config)
            if self.uses_graph
            else _V051._V038.build_model(config, bench)
        )

    def forward(self, x, valid_token_mask=None):
        return self.inner(x, valid_token_mask)


def build_model(config, bench):
    return SelectedModel(config, bench)


def load_from_baseline(model, baseline):
    target = model.inner.inner if model.uses_graph else model.inner
    _V051._V038.load_from_baseline(target, baseline)
