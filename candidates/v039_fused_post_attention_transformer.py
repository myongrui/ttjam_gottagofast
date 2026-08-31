"""Fused post-attention transformer candidate.

Hypothesis: the latency-bound canonical shapes are dominated by launches between
attention and the FFN. A single Triton kernel consumes SDPA's head-major output
directly and fuses its layout conversion, attention output projection, residual
addition, second LayerNorm, FFN input projection, and exact GELU. This replaces
the transpose-copy and four adjacent kernels with one launch; larger shapes
retain the TorchScript champion because independent GEMMs are then preferable.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _post_attention_fused_kernel(
    context,
    residual,
    out_weight,
    out_bias,
    norm_weight,
    norm_bias,
    ffn_weight,
    ffn_bias,
    residual_out,
    hidden_out,
    n_rows: tl.constexpr,
    seq_len: tl.constexpr,
    context_stride_b: tl.constexpr,
    context_stride_h: tl.constexpr,
    context_stride_s: tl.constexpr,
    context_stride_d: tl.constexpr,
    d_model: tl.constexpr,
    head_dim: tl.constexpr,
    ffn_dim: tl.constexpr,
    eps: tl.constexpr,
    block_m: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * block_m + tl.arange(0, block_m)
    d = tl.arange(0, d_model)
    f = tl.arange(0, ffn_dim)
    row_mask = rows < n_rows

    batch_index = rows // seq_len
    sequence_index = rows - batch_index * seq_len
    head_index = d // head_dim
    head_offset = d - head_index * head_dim

    context_offsets = (
        batch_index[:, None] * context_stride_b
        + head_index[None, :] * context_stride_h
        + sequence_index[:, None] * context_stride_s
        + head_offset[None, :] * context_stride_d
    )
    context_values = tl.load(
        context + context_offsets,
        mask=row_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    weight_offsets = d[:, None] * d_model + d[None, :]
    output_weight_t = tl.trans(
        tl.load(out_weight + weight_offsets).to(tl.float32)
    )
    projected = tl.dot(context_values, output_weight_t)

    residual_offsets = rows[:, None] * d_model + d[None, :]
    residual_values = tl.load(
        residual + residual_offsets,
        mask=row_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    projected += residual_values
    projected += tl.load(out_bias + d)[None, :].to(tl.float32)

    tl.store(
        residual_out + residual_offsets,
        projected,
        mask=row_mask[:, None],
    )

    mean = tl.sum(projected, axis=1) / d_model
    centered = projected - mean[:, None]
    variance = tl.sum(centered * centered, axis=1) / d_model
    normalized = centered * tl.rsqrt(variance[:, None] + eps)
    normalized = (
        normalized
        * tl.load(norm_weight + d)[None, :].to(tl.float32)
        + tl.load(norm_bias + d)[None, :].to(tl.float32)
    )

    ffn_weight_offsets = f[:, None] * d_model + d[None, :]
    ffn_weight_t = tl.trans(
        tl.load(ffn_weight + ffn_weight_offsets).to(tl.float32)
    )
    pre_activation = tl.dot(normalized, ffn_weight_t)
    pre_activation += tl.load(ffn_bias + f)[None, :].to(tl.float32)

    hidden = 0.5 * pre_activation * (
        1.0 + tl.erf(pre_activation * 0.7071067811865475244)
    )
    hidden_offsets = rows[:, None] * ffn_dim + f[None, :]
    tl.store(
        hidden_out + hidden_offsets,
        hidden,
        mask=row_mask[:, None],
    )


class FusedAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def context(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        causal: bool,
    ) -> torch.Tensor:
        b, s, _ = x.shape
        qkv = self.qkv(x).view(
            b, s, 3, self.num_heads, self.head_dim
        )
        q = qkv[:, :, 0].transpose(1, 2)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=causal and attn_mask is None,
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        causal: bool,
    ) -> torch.Tensor:
        context = self.context(x, attn_mask, causal)
        b, _, s, _ = context.shape
        return context.transpose(1, 2).reshape(b, s, self.d_model)


class StandardBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = FusedAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_in = nn.Linear(d_model, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, d_model)

    @staticmethod
    def _residual_projection(
        residual: torch.Tensor,
        projected_input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        b, s, d = residual.shape
        c = residual.reshape(b * s, d) + bias
        return torch.addmm(
            c,
            projected_input.reshape(
                b * s, projected_input.shape[-1]
            ),
            weight.t(),
        ).view(b, s, d)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        causal: bool,
    ) -> torch.Tensor:
        context = self.attention(self.norm1(x), attn_mask, causal)
        x = self._residual_projection(
            x,
            context,
            self.attention.out_proj.weight,
            self.attention.out_proj.bias,
        )
        hidden = F.gelu(
            self.ffn_in(self.norm2(x)),
            approximate="none",
        )
        return self._residual_projection(
            x,
            hidden,
            self.ffn_out.weight,
            self.ffn_out.bias,
        )


class StandardModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.causal = bool(config.causal)
        self.layers = nn.ModuleList(
            [
                StandardBlock(
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
        if bool(valid_token_mask.all()):
            return None

        s = x.shape[1]
        keep = valid_token_mask[:, None, None, :]
        if self.causal:
            causal_keep = torch.ones(
                (s, s),
                device=x.device,
                dtype=torch.bool,
            ).tril()
            keep = keep & causal_keep

        mask = torch.zeros(
            keep.shape,
            device=x.device,
            dtype=x.dtype,
        )
        return mask.masked_fill(~keep, float("-inf"))

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
            x = x.masked_fill(
                ~valid_token_mask[..., None],
                0.0,
            )
        return x


class FusedBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.ffn_dim = ffn_dim
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = FusedAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_in = nn.Linear(d_model, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, d_model)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        causal: bool,
    ) -> torch.Tensor:
        context = self.attention.context(
            self.norm1(x),
            attn_mask,
            causal,
        )
        b, _, s, _ = context.shape
        n_rows = b * s

        residual_out = torch.empty_like(x)
        hidden = torch.empty(
            (b, s, self.ffn_dim),
            device=x.device,
            dtype=x.dtype,
        )

        block_m = 16
        grid = (triton.cdiv(n_rows, block_m),)
        _post_attention_fused_kernel[grid](
            context,
            x,
            self.attention.out_proj.weight,
            self.attention.out_proj.bias,
            self.norm2.weight,
            self.norm2.bias,
            self.ffn_in.weight,
            self.ffn_in.bias,
            residual_out,
            hidden,
            n_rows=n_rows,
            seq_len=s,
            context_stride_b=context.stride(0),
            context_stride_h=context.stride(1),
            context_stride_s=context.stride(2),
            context_stride_d=context.stride(3),
            d_model=self.d_model,
            head_dim=self.head_dim,
            ffn_dim=self.ffn_dim,
            eps=self.norm2.eps,
            block_m=block_m,
            num_warps=4,
        )

        c = (
            residual_out.reshape(n_rows, self.d_model)
            + self.ffn_out.bias
        )
        return torch.addmm(
            c,
            hidden.reshape(n_rows, self.ffn_dim),
            self.ffn_out.weight.t(),
        ).view(b, s, self.d_model)


class FusedModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.causal = bool(config.causal)
        self.layers = nn.ModuleList(
            [
                FusedBlock(
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
        if bool(valid_token_mask.all()):
            return None

        s = x.shape[1]
        keep = valid_token_mask[:, None, None, :]
        if self.causal:
            causal_keep = torch.ones(
                (s, s),
                device=x.device,
                dtype=torch.bool,
            ).tril()
            keep = keep & causal_keep

        mask = torch.zeros(
            keep.shape,
            device=x.device,
            dtype=x.dtype,
        )
        return mask.masked_fill(~keep, float("-inf"))

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
            x = x.masked_fill(
                ~valid_token_mask[..., None],
                0.0,
            )
        return x


class HybridModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fast = FusedModel(config)
        self.slow = torch.jit.script(StandardModel(config))

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.shape[0] * x.shape[1] <= 1024:
            return self.fast(x, valid_token_mask)
        return self.slow(x, valid_token_mask)


def build_model(config, bench) -> nn.Module:
    if (
        int(config.d_model) == 128
        and int(config.ffn_dim) == 128
        and int(config.d_model) // int(config.num_heads) == 32
    ):
        return HybridModel(config)
    return torch.jit.script(StandardModel(config))


def load_from_baseline(model, baseline) -> None:
    with torch.no_grad():
        state = model.state_dict()
        if "fast.final_norm.weight" in state:
            prefixes = ("fast.", "slow.")
        else:
            prefixes = ("",)

        for root in prefixes:
            for index, src in enumerate(baseline.layers):
                prefix = root + "layers." + str(index) + "."
                attention = src.attention

                state[prefix + "attention.qkv.weight"].copy_(
                    torch.cat(
                        (
                            attention.q_proj.weight,
                            attention.k_proj.weight,
                            attention.v_proj.weight,
                        ),
                        dim=0,
                    )
                )
                state[prefix + "attention.qkv.bias"].copy_(
                    torch.cat(
                        (
                            attention.q_proj.bias,
                            attention.k_proj.bias,
                            attention.v_proj.bias,
                        ),
                        dim=0,
                    )
                )
                state[prefix + "attention.out_proj.weight"].copy_(
                    attention.out_proj.weight
                )
                state[prefix + "attention.out_proj.bias"].copy_(
                    attention.out_proj.bias
                )
                state[prefix + "norm1.weight"].copy_(src.norm1.weight)
                state[prefix + "norm1.bias"].copy_(src.norm1.bias)
                state[prefix + "norm2.weight"].copy_(src.norm2.weight)
                state[prefix + "norm2.bias"].copy_(src.norm2.bias)
                state[prefix + "ffn_in.weight"].copy_(src.ffn_in.weight)
                state[prefix + "ffn_in.bias"].copy_(src.ffn_in.bias)
                state[prefix + "ffn_out.weight"].copy_(src.ffn_out.weight)
                state[prefix + "ffn_out.bias"].copy_(src.ffn_out.bias)

            state[root + "final_norm.weight"].copy_(
                baseline.final_norm.weight
            )
            state[root + "final_norm.bias"].copy_(
                baseline.final_norm.bias
            )
