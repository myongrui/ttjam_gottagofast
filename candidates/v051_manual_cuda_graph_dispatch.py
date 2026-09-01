"""Manual CUDA-graph replay for latency-bound cases 2, 3, and 7.

Hypothesis: these three shapes are dominated by repeated launch submission.
Capturing a synchronization-free transformer once and replaying it after copying
fresh input and mask tensors should eliminate most per-forward launches while
remaining input-dependent and correct. Other shapes retain the v038 parent.
"""
import importlib.util
import os
import sys

import torch
import torch.nn as nn


GRAPH_KEYS = {
    (1, 128, 4, 128, 4, 128, True),
    (4, 128, 4, 128, 4, 128, True),
    (64, 32, 4, 128, 4, 32, True),
}


def _load_candidate(filename, name):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_V038 = _load_candidate(
    "v038_torchscript_dispatched_transformer_candidate.py",
    "manual_graph_parent_v038",
)
_V044 = _load_candidate(
    "v044_synchronization_free_compiled_transformer.py",
    "manual_graph_inner_v044",
)


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


class GraphReplayModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.inner = _V044.Model(config)
        self._graph = None
        self._static_x = None
        self._static_mask = None
        self._static_output = None

    def _capture(self, x, valid_token_mask):
        self._static_x = torch.empty_like(x)
        self._static_mask = torch.empty_like(valid_token_mask)
        self._static_x.copy_(x)
        self._static_mask.copy_(valid_token_mask)

        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                self.inner(self._static_x, self._static_mask)
        torch.cuda.current_stream().wait_stream(warmup_stream)

        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._static_output = self.inner(
                self._static_x,
                self._static_mask,
            )

    def forward(self, x, valid_token_mask=None):
        if valid_token_mask is None:
            return self.inner(x, valid_token_mask)
        if self._graph is None:
            self._capture(x, valid_token_mask)
        self._static_x.copy_(x)
        self._static_mask.copy_(valid_token_mask)
        self._graph.replay()
        return self._static_output.clone()


class SelectedModel(nn.Module):
    def __init__(self, config, bench):
        super().__init__()
        self.uses_graph = _config_key(config) in GRAPH_KEYS
        self.inner = (
            GraphReplayModel(config)
            if self.uses_graph
            else _V038.build_model(config, bench)
        )

    def forward(self, x, valid_token_mask=None):
        return self.inner(x, valid_token_mask)


def build_model(config, bench):
    return SelectedModel(config, bench)


def load_from_baseline(model, baseline):
    target = model.inner.inner if model.uses_graph else model.inner
    _V038.load_from_baseline(target, baseline)
