"""Portfolio dispatcher generated from the float32 per-shape elite archive.

Mapping: 1:v053_expanded_cuda_graph_dispatch.py, 2:v058_graph_homogeneous_ffn.py, 3:v058_graph_homogeneous_ffn.py, 4:v053_expanded_cuda_graph_dispatch.py, 5:v053_expanded_cuda_graph_dispatch.py, 6:v049_case3_homogeneous_ffn_dispatch.py, 7:v051_manual_cuda_graph_dispatch.py, 8:v053_expanded_cuda_graph_dispatch.py, 9:v054_pruned_cuda_graph_dispatch.py, 10:v054_pruned_cuda_graph_dispatch.py, 11:v054_pruned_cuda_graph_dispatch.py, 12:v053_expanded_cuda_graph_dispatch.py, 13:v053_expanded_cuda_graph_dispatch.py, 14:v053_expanded_cuda_graph_dispatch.py
"""
import importlib.util
import os
import sys

import torch.nn as nn


CONFIG_TO_CANDIDATE = {
    (64, 128, 4, 128, 4, 128, True): 'v053_expanded_cuda_graph_dispatch.py',
    (1, 128, 4, 128, 4, 128, True): 'v058_graph_homogeneous_ffn.py',
    (4, 128, 4, 128, 4, 128, True): 'v058_graph_homogeneous_ffn.py',
    (16, 128, 4, 128, 4, 128, True): 'v053_expanded_cuda_graph_dispatch.py',
    (128, 128, 4, 128, 4, 128, True): 'v053_expanded_cuda_graph_dispatch.py',
    (10000, 128, 4, 128, 4, 128, True): 'v049_case3_homogeneous_ffn_dispatch.py',
    (64, 32, 4, 128, 4, 32, True): 'v051_manual_cuda_graph_dispatch.py',
    (64, 1024, 4, 128, 4, 1024, True): 'v053_expanded_cuda_graph_dispatch.py',
    (64, 128, 1, 128, 4, 128, True): 'v054_pruned_cuda_graph_dispatch.py',
    (64, 128, 2, 128, 4, 128, True): 'v054_pruned_cuda_graph_dispatch.py',
    (64, 128, 16, 128, 4, 128, True): 'v054_pruned_cuda_graph_dispatch.py',
    (64, 128, 4, 32, 4, 128, True): 'v053_expanded_cuda_graph_dispatch.py',
    (64, 128, 4, 1024, 4, 128, True): 'v053_expanded_cuda_graph_dispatch.py',
    (32, 1024, 16, 100000, 2, 1024, True): 'v053_expanded_cuda_graph_dispatch.py',
}
FALLBACK_CANDIDATE = 'v053_expanded_cuda_graph_dispatch.py'
_MODULES = {}


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
