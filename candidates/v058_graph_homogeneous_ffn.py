"""Graph-captured homogeneous-coordinate FFN bias folding.

Hypothesis: v047's exact two-channel homogeneous-coordinate construction removes
the FFN-output bias broadcast before residual addmm, but its earlier eager/
TorchScript evaluation still paid host submission overhead.  For cases 2, 3,
4, and 12—where v047 previously improved its parent—capture a synchronization-
free v047 model inside v053's CUDA-graph wrapper.  This keeps cuBLAS and exact
erf GELU while removing one device kernel per layer.  All other static shapes
use v053 unchanged.
"""
from typing import Optional

import importlib.util
import os
import sys

import torch
import torch.nn as nn


HOMOGENEOUS_GRAPH_KEYS = {
    (1, 128, 4, 128, 4, 128, True),
    (4, 128, 4, 128, 4, 128, True),
    (16, 128, 4, 128, 4, 128, True),
    (64, 128, 4, 32, 4, 128, True),
}


def _load_candidate(filename, name):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_V053 = _load_candidate(
    "v053_expanded_cuda_graph_dispatch.py",
    "graph_homogeneous_parent_v053",
)
_V047 = _load_candidate(
    "v047_homogeneous_coordinate_ffn_bias.py",
    "graph_homogeneous_inner_v047",
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


class SynchronizationFreeHomogeneousModel(_V047.Model):
    def _attention_mask(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if valid_token_mask is None:
            return None
        keep = valid_token_mask[:, None, None, :]
        if self.causal:
            sequence_length = x.shape[1]
            causal_keep = torch.ones(
                (sequence_length, sequence_length),
                device=x.device,
                dtype=torch.bool,
            ).tril()
            keep = keep & causal_keep
        return torch.zeros(
            keep.shape,
            device=x.device,
            dtype=x.dtype,
        ).masked_fill(~keep, float("-inf"))


class HomogeneousGraphReplayModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.inner = SynchronizationFreeHomogeneousModel(config)
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
        self.uses_homogeneous_graph = (
            _config_key(config) in HOMOGENEOUS_GRAPH_KEYS
        )
        self.inner = (
            HomogeneousGraphReplayModel(config)
            if self.uses_homogeneous_graph
            else _V053.build_model(config, bench)
        )

    def forward(self, x, valid_token_mask=None):
        return self.inner(x, valid_token_mask)


def build_model(config, bench):
    return SelectedModel(config, bench)


def load_from_baseline(model, baseline):
    if model.uses_homogeneous_graph:
        _V047.load_from_baseline(model.inner.inner, baseline)
    else:
        _V053.load_from_baseline(model.inner, baseline)
