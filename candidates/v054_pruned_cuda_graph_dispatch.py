"""Prune expanded graph replay to its five winning static configurations.

Hypothesis: v053's graph paths for cases 1, 9, 10, and 11 lose more to buffer
copies and replay overhead than they save in launch submission. Keeping manual
graphs only for cases 2, 3, 4, 7, and 12 should preserve the large wins while
returning those four regressors to the scripted v038 path.
"""
import importlib.util
import os
import sys

import torch.nn as nn


def _load_parent():
    filename = "v051_manual_cuda_graph_dispatch.py"
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    name = "pruned_graph_parent_v051"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_PARENT = _load_parent()
GRAPH_KEYS = _PARENT.GRAPH_KEYS | {
    (16, 128, 4, 128, 4, 128, True),
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
