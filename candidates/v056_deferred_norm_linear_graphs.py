"""Paper-backed deferred LayerNorm-linear fusion inside CUDA graphs.

Hypothesis: cases 2, 3, and 7 remain dominated by the device kernels and HBM
round trips inside v053's captured transformer.  At every pre-sublayer boundary,
replace affine LayerNorm followed by packed QKV or FFN-input projection with the
real-arithmetic identity from fused normalization-linear work:

    A = W * diag(gamma), r = A @ 1, c = b + W @ beta
    Linear(LayerNorm(x)) = (A @ x - mean(x) * r) * rsqrt(var(x)+eps) + c

One shape-specialized Triton launch therefore replaces LayerNorm plus GEMM and
never stores the normalized activation.  The reduced sequence is captured with
v053's input/mask copy, replay, and output-clone semantics.  Every non-target
configuration uses v053 unchanged.
"""
from typing import Optional

import importlib.util
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    _HAS_TRITON = False


TARGET_KEYS = {
    (1, 128, 4, 128, 4, 128, True),
    (4, 128, 4, 128, 4, 128, True),
    (64, 32, 4, 128, 4, 32, True),
}


def _load_v053():
    filename = "v053_expanded_cuda_graph_dispatch.py"
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    name = "deferred_norm_parent_v053"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_V053 = _load_v053()


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


if _HAS_TRITON:
    @triton.jit
    def _deferred_norm_linear_kernel(
        x_ptr,
        a_ptr,
        r_ptr,
        c_ptr,
        out_ptr,
        rows: tl.constexpr,
        in_features: tl.constexpr,
        out_features: tl.constexpr,
        block_m: tl.constexpr,
        block_d: tl.constexpr,
        block_n: tl.constexpr,
        eps: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        row_offsets = pid_m * block_m + tl.arange(0, block_m)
        d_offsets = tl.arange(0, block_d)
        n_offsets = pid_n * block_n + tl.arange(0, block_n)

        row_mask = row_offsets < rows
        d_mask = d_offsets < in_features
        n_mask = n_offsets < out_features

        x = tl.load(
            x_ptr + row_offsets[:, None] * in_features + d_offsets[None, :],
            mask=row_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        mean = tl.sum(x, axis=1) / in_features
        centered = x - mean[:, None]
        variance = tl.sum(centered * centered, axis=1) / in_features
        inv_std = tl.rsqrt(variance + eps)

        # A is stored [out_features, in_features]; load its transpose tile.
        a_t = tl.load(
            a_ptr + n_offsets[None, :] * in_features + d_offsets[:, None],
            mask=d_mask[:, None] & n_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        raw = tl.dot(x, a_t, input_precision="ieee")

        r = tl.load(r_ptr + n_offsets, mask=n_mask, other=0.0).to(tl.float32)
        c = tl.load(c_ptr + n_offsets, mask=n_mask, other=0.0).to(tl.float32)
        result = (raw - mean[:, None] * r[None, :]) * inv_std[:, None]
        result += c[None, :]

        tl.store(
            out_ptr + row_offsets[:, None] * out_features + n_offsets[None, :],
            result,
            mask=row_mask[:, None] & n_mask[None, :],
        )


class DeferredNormLinear(nn.Module):
    """Affine LayerNorm plus Linear represented by precomputed A, r, and c."""

    def __init__(self, in_features: int, out_features: int, eps: float = 1e-5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = float(eps)
        self.register_buffer("a", torch.empty(out_features, in_features))
        self.register_buffer("r", torch.empty(out_features))
        self.register_buffer("c", torch.empty(out_features))

    def load_equivalent(self, norm: nn.LayerNorm, linear: nn.Linear) -> None:
        with torch.no_grad():
            a = linear.weight * norm.weight.unsqueeze(0)
            self.a.copy_(a)
            self.r.copy_(a.sum(dim=1))
            self.c.copy_(linear.bias + torch.mv(linear.weight, norm.bias))

    def _reference(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        centered = x - mean
        inv_std = torch.rsqrt(
            centered.square().mean(dim=-1, keepdim=True) + self.eps
        )
        return (F.linear(x, self.a) - mean * self.r) * inv_std + self.c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not (
            _HAS_TRITON
            and x.is_cuda
            and x.dtype == torch.float32
            and x.is_contiguous()
            and self.in_features in (32, 128)
        ):
            return self._reference(x)

        rows = x.numel() // self.in_features
        out = torch.empty(
            (rows, self.out_features),
            device=x.device,
            dtype=x.dtype,
        )
        block_m = 16
        block_d = self.in_features
        block_n = 128
        grid = (
            triton.cdiv(rows, block_m),
            triton.cdiv(self.out_features, block_n),
        )
        _deferred_norm_linear_kernel[grid](
            x.reshape(rows, self.in_features),
            self.a,
            self.r,
            self.c,
            out,
            rows=rows,
            in_features=self.in_features,
            out_features=self.out_features,
            block_m=block_m,
            block_d=block_d,
            block_n=block_n,
            eps=self.eps,
            num_warps=4,
            num_stages=2,
        )
        return out.view(*x.shape[:-1], self.out_features)


class DeferredAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.norm_qkv = DeferredNormLinear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        causal: bool,
    ) -> torch.Tensor:
        b, s, _ = x.shape
        qkv = self.norm_qkv(x).view(
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


class DeferredBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.attention = DeferredAttention(d_model, num_heads)
        self.norm_ffn = DeferredNormLinear(d_model, ffn_dim)
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
        hidden = F.gelu(self.norm_ffn(x), approximate="none")
        return self._residual_projection(
            x,
            hidden,
            self.ffn_out.weight,
            self.ffn_out.bias,
        )


class DeferredModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.causal = bool(config.causal)
        self.layers = nn.ModuleList(
            [
                DeferredBlock(
                    config.d_model,
                    config.num_heads,
                    config.ffn_dim,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)

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
        attn_mask = self._attention_mask(x, valid_token_mask)
        for layer in self.layers:
            x = layer(x, attn_mask, self.causal)
        x = self.final_norm(x)
        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0.0)
        return x


class DeferredGraphReplayModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.inner = DeferredModel(config)
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
        self.uses_deferred = _config_key(config) in TARGET_KEYS
        self.inner = (
            DeferredGraphReplayModel(config)
            if self.uses_deferred
            else _V053.build_model(config, bench)
        )

    def forward(self, x, valid_token_mask=None):
        return self.inner(x, valid_token_mask)


def build_model(config, bench):
    return SelectedModel(config, bench)


def _load_deferred(model: DeferredModel, baseline: nn.Module) -> None:
    with torch.no_grad():
        for dst, src in zip(model.layers, baseline.layers):
            attention = src.attention
            qkv_weight = torch.cat(
                (
                    attention.q_proj.weight,
                    attention.k_proj.weight,
                    attention.v_proj.weight,
                ),
                dim=0,
            )
            qkv_bias = torch.cat(
                (
                    attention.q_proj.bias,
                    attention.k_proj.bias,
                    attention.v_proj.bias,
                ),
                dim=0,
            )
            packed_qkv = nn.Linear(
                src.norm1.normalized_shape[0],
                qkv_weight.shape[0],
                bias=True,
                device=qkv_weight.device,
                dtype=qkv_weight.dtype,
            )
            packed_qkv.weight.copy_(qkv_weight)
            packed_qkv.bias.copy_(qkv_bias)
            dst.attention.norm_qkv.load_equivalent(src.norm1, packed_qkv)
            dst.attention.out_proj.load_state_dict(
                attention.out_proj.state_dict()
            )
            dst.norm_ffn.load_equivalent(src.norm2, src.ffn_in)
            dst.ffn_out.load_state_dict(src.ffn_out.state_dict())
        model.final_norm.load_state_dict(baseline.final_norm.state_dict())


def load_from_baseline(model, baseline):
    if model.uses_deferred:
        _load_deferred(model.inner.inner, baseline)
    else:
        _V053.load_from_baseline(model.inner, baseline)
