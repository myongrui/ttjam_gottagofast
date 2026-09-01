"""Expand v051 graph replay to the remaining launch-heavy moderate shapes.

Hypothesis: cases 1, 4, 9, 10, 11, and 12 still spend enough time submitting
the same transformer launch sequence that v051's input-dependent manual CUDA
graph will beat its scripted fallback. Cases 5, 6, 8, and 13 retain v038 because
their larger throughput or sequence workloads make replay overhead less useful.
"""
import importlib.util
import os
import sys

import torch.nn as nn


def _load_parent():
    filename = "v051_manual_cuda_graph_dispatch.py"
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    name = "expanded_graph_parent_v051"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_PARENT = _load_parent()
GRAPH_KEYS = _PARENT.GRAPH_KEYS | {
    (64, 128, 4, 128, 4, 128, True),
    (16, 128, 4, 128, 4, 128, True),
    (64, 128, 1, 128, 4, 128, True),
    (64, 128, 2, 128, 4, 128, True),
    (64, 128, 16, 128, 4, 128, True),
    (64, 128, 4, 32, 4, 128, True),
}


class SelectedModel(nn.Module):
    def __init__(self, config, bench):
        super().__init__()
        self.uses_graph = _PARENT._config_key(config) in GRAPH_KEYS
        self.inner = (
            _PARENT.GraphReplayModel(config)
            if self.uses_graph
            else _PARENT._V038.build_model(config, bench)
        )

    def forward(self, x, valid_token_mask=None):
        return self.inner(x, valid_token_mask)


def build_model(config, bench):
    return SelectedModel(config, bench)


def load_from_baseline(model, baseline):
    target = model.inner.inner if model.uses_graph else model.inner
    _PARENT._V038.load_from_baseline(target, baseline)
