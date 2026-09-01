"""Case-3 homogeneous-coordinate FFN-bias specialist.

Hypothesis: on case 3 (batch 4, sequence 128, width 128, four heads), folding
the FFN output bias into two constant GELU channels removes one launch per
transformer layer and beats the v038 TorchScript generalist by more than the
2% paired-confidence promotion margin. Every other static configuration keeps
the v038 implementation, isolating the mechanism to the affected shape.
"""
import importlib.util
import os
import sys

import torch.nn as nn


CASE3_KEY = (4, 128, 4, 128, 4, 128, True)
_MODULES = {}


def _config_key(config):
    return (
        config.batch_size,
        config.d_model,
        config.num_heads,
        config.seq_len,
        config.num_layers,
        config.ffn_dim,
        bool(config.causal),
    )


def _load_candidate(filename):
    if filename in _MODULES:
        return _MODULES[filename]
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    name = "case3_dispatch_" + os.path.splitext(filename)[0]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    _MODULES[filename] = module
    return module


class SelectedModel(nn.Module):
    def __init__(self, config, bench):
        super().__init__()
        filename = (
            "v047_homogeneous_coordinate_ffn_bias.py"
            if _config_key(config) == CASE3_KEY
            else "v038_torchscript_dispatched_transformer_candidate.py"
        )
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
