"""Paper-equivalent centered-residual Pre-RMSNorm transformer.

Hypothesis: the fixed pre-LayerNorm transformer can use the equivalent
zero-mean-main-branch construction from the Pre-RMSNorm paper.  Center the input
once, fold each LayerNorm affine into the following QKV or FFN-input projection,
and center every attention/FFN output projection at weight-load time.  The
residual stream then stays in the zero-mean subspace, where same-epsilon RMSNorm
equals LayerNorm.  This removes the mean reduction from eight block norms while
preserving v053's shape-specific CUDA-graph coverage and the original function
over real arithmetic.
"""
from typing import Optional

import importlib.util
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


def _load_v053():
    filename = "v053_expanded_cuda_graph_dispatch.py"
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    name = "centered_prerms_parent_v053"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_V053 = _load_v053()
GRAPH_KEYS = set(_V053.GRAPH_KEYS)


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


class CenteredAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rms = nn.RMSNorm(
            d_model,
            eps=1e-5,
            elementwise_affine=False,
        )
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        causal: bool,
    ) -> torch.Tensor:
        b, s, _ = x.shape
        qkv = self.qkv(self.rms(x)).view(
            b, s, 3, self.num_heads, self.head_dim
        )
        q = qkv[:, :, 0].transpose(1, 2)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)
        context = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=causal and attn_mask is None,
        )
        return context.transpose(1, 2).reshape(b, s, self.d_model)


class CenteredBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.attention = CenteredAttention(d_model, num_heads)
        self.rms2 = nn.RMSNorm(
            d_model,
            eps=1e-5,
            elementwise_affine=False,
        )
        self.ffn_in = nn.Linear(d_model, ffn_dim, bias=True)
        self.ffn_out = nn.Linear(ffn_dim, d_model, bias=True)

    @staticmethod
    def _residual_projection(
        residual: torch.Tensor,
        projected_input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        b, s, d = residual.shape
        return torch.addmm(
            residual.reshape(b * s, d) + bias,
            projected_input.reshape(b * s, projected_input.shape[-1]),
            weight.t(),
        ).view(b, s, d)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        causal: bool,
    ) -> torch.Tensor:
        context = self.attention(x, attn_mask, causal)
        x = self._residual_projection(
            x,
            context,
            self.attention.out_proj.weight,
            self.attention.out_proj.bias,
        )
        hidden = F.gelu(
            self.ffn_in(self.rms2(x)),
            approximate="none",
        )
        return self._residual_projection(
            x,
            hidden,
            self.ffn_out.weight,
            self.ffn_out.bias,
        )


class CenteredPreRMSModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.causal = bool(config.causal)
        self.layers = nn.ModuleList(
            [
                CenteredBlock(
                    config.d_model,
                    config.num_heads,
                    config.ffn_dim,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_rms = nn.RMSNorm(
            config.d_model,
            eps=1e-5,
            elementwise_affine=True,
        )
        self.register_buffer("final_bias", torch.empty(config.d_model))

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

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # P x initializes the zero-mean residual stream.
        x = x - x.mean(dim=-1, keepdim=True)
        attn_mask = self._attention_mask(x, valid_token_mask)
        for layer in self.layers:
            x = layer(x, attn_mask, self.causal)
        x = self.final_rms(x) + self.final_bias
        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0.0)
        return x


class PreRMSGraphReplayModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.inner = CenteredPreRMSModel(config)
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
    def __init__(self, config):
        super().__init__()
        self.uses_graph = _config_key(config) in GRAPH_KEYS
        self.inner = (
            PreRMSGraphReplayModel(config)
            if self.uses_graph
            else torch.jit.script(CenteredPreRMSModel(config))
        )

    def forward(self, x, valid_token_mask=None):
        return self.inner(x, valid_token_mask)


def build_model(config, bench):
    return SelectedModel(config)


def _fold_norm_into_linear(
    dst_weight: torch.Tensor,
    dst_bias: torch.Tensor,
    src_linear: nn.Linear,
    src_norm: nn.LayerNorm,
) -> None:
    dst_weight.copy_(src_linear.weight * src_norm.weight.unsqueeze(0))
    dst_bias.copy_(
        src_linear.bias + torch.mv(src_linear.weight, src_norm.bias)
    )


def _center_output_linear(dst_weight, dst_bias, src_linear) -> None:
    # For row-vector activations, W' = P W and b' = P b.
    dst_weight.copy_(
        src_linear.weight - src_linear.weight.mean(dim=0, keepdim=True)
    )
    dst_bias.copy_(src_linear.bias - src_linear.bias.mean())


def _load_centered(model: nn.Module, baseline: nn.Module) -> None:
    with torch.no_grad():
        state = model.state_dict()
        for index, src in enumerate(baseline.layers):
            prefix = "layers." + str(index) + "."
            attention = src.attention

            packed_weight = torch.cat(
                (
                    attention.q_proj.weight,
                    attention.k_proj.weight,
                    attention.v_proj.weight,
                ),
                dim=0,
            )
            packed_bias = torch.cat(
                (
                    attention.q_proj.bias,
                    attention.k_proj.bias,
                    attention.v_proj.bias,
                ),
                dim=0,
            )
            packed = nn.Linear(
                src.norm1.normalized_shape[0],
                packed_weight.shape[0],
                bias=True,
                device=packed_weight.device,
                dtype=packed_weight.dtype,
            )
            packed.weight.copy_(packed_weight)
            packed.bias.copy_(packed_bias)
            _fold_norm_into_linear(
                state[prefix + "attention.qkv.weight"],
                state[prefix + "attention.qkv.bias"],
                packed,
                src.norm1,
            )
            _center_output_linear(
                state[prefix + "attention.out_proj.weight"],
                state[prefix + "attention.out_proj.bias"],
                attention.out_proj,
            )
            _fold_norm_into_linear(
                state[prefix + "ffn_in.weight"],
                state[prefix + "ffn_in.bias"],
                src.ffn_in,
                src.norm2,
            )
            _center_output_linear(
                state[prefix + "ffn_out.weight"],
                state[prefix + "ffn_out.bias"],
                src.ffn_out,
            )

        state["final_rms.weight"].copy_(baseline.final_norm.weight)
        state["final_bias"].copy_(baseline.final_norm.bias)


def load_from_baseline(model, baseline):
    target = model.inner.inner if model.uses_graph else model.inner
    _load_centered(target, baseline)
