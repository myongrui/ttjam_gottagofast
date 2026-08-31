"""Fused-final-normalization candidate.

Hypothesis: the final LayerNorm and padded-token zeroing are adjacent, independent
per token, and currently require two separate CUDA launches.  A Triton kernel
computes final LayerNorm and applies the validity mask in the same write epilogue,
eliminating the final masked_fill launch while preserving the deferred-padding
semantics for valid and invalid tokens.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _final_norm_mask_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    mask_ptr,
    out_ptr,
    n_rows: tl.constexpr,
    d_model: tl.constexpr,
    eps: tl.constexpr,
    has_mask: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, block)
    valid_col = cols < d_model
    offsets = row * d_model + cols

    x = tl.load(x_ptr + offsets, mask=valid_col, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / d_model
    centered = tl.where(valid_col, x - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / d_model
    inv_std = tl.rsqrt(var + eps)

    weight = tl.load(weight_ptr + cols, mask=valid_col, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=valid_col, other=0.0).to(tl.float32)
    y = centered * inv_std * weight + bias

    if has_mask:
        keep = tl.load(mask_ptr + row).to(tl.int1)
        y = tl.where(keep, y, 0.0)

    tl.store(out_ptr + offsets, y, mask=valid_col)


class FusedAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x, valid_token_mask=None, causal=False):
        b, s, _ = x.shape
        qkv = self.qkv(x).view(b, s, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        if valid_token_mask is not None and not bool(valid_token_mask.all()):
            keep = valid_token_mask[:, None, None, :]
            if causal:
                keep = keep & torch.ones(
                    s, s, device=x.device, dtype=torch.bool
                ).tril()
            attn_mask = torch.zeros(keep.shape, device=x.device, dtype=x.dtype)
            attn_mask = attn_mask.masked_fill(~keep, float("-inf"))
            ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            ctx = F.scaled_dot_product_attention(q, k, v, is_causal=causal)

        return ctx.transpose(1, 2).reshape(b, s, self.d_model)


class Block(nn.Module):
    def __init__(self, d_model, num_heads, ffn_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = FusedAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_in = nn.Linear(d_model, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, d_model)

    @staticmethod
    def _residual_projection(residual, projected_input, linear):
        b, s, d = residual.shape
        c = residual.reshape(b * s, d) + linear.bias
        return torch.addmm(
            c,
            projected_input.reshape(b * s, projected_input.shape[-1]),
            linear.weight.t(),
        ).view(b, s, d)

    def forward(self, x, valid_token_mask, causal):
        ctx = self.attention(self.norm1(x), valid_token_mask, causal)
        x = self._residual_projection(x, ctx, self.attention.out_proj)
        hidden = F.gelu(self.ffn_in(self.norm2(x)), approximate="none")
        return self._residual_projection(x, hidden, self.ffn_out)


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [
                Block(config.d_model, config.num_heads, config.ffn_dim)
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)

    def _fused_final_norm(self, x, valid_token_mask):
        d_model = x.shape[-1]
        if (
            not x.is_cuda
            or x.dtype != torch.float32
            or not x.is_contiguous()
            or d_model > 8192
        ):
            x = self.final_norm(x)
            if valid_token_mask is not None:
                x = x.masked_fill(~valid_token_mask[..., None], 0)
            return x

        block = triton.next_power_of_2(d_model)
        out = torch.empty_like(x)
        flat_mask = x if valid_token_mask is None else valid_token_mask.contiguous().view(-1)
        _final_norm_mask_kernel[(x.numel() // d_model,)](
            x,
            self.final_norm.weight,
            self.final_norm.bias,
            flat_mask,
            out,
            x.numel() // d_model,
            d_model,
            self.final_norm.eps,
            valid_token_mask is not None,
            block,
            num_warps=1, num_stages=4,
        )
        return out

    def forward(self, x, valid_token_mask=None):
        for layer in self.layers:
            x = layer(x, valid_token_mask, self.config.causal)
        return self._fused_final_norm(x, valid_token_mask)


def build_model(config, bench):
    return Model(config)


def load_from_baseline(model, baseline):
    with torch.no_grad():
        for dst, src in zip(model.layers, baseline.layers):
            a = src.attention
            dst.attention.qkv.weight.copy_(
                torch.cat(
                    [a.q_proj.weight, a.k_proj.weight, a.v_proj.weight],
                    dim=0,
                )
            )
            dst.attention.qkv.bias.copy_(
                torch.cat(
                    [a.q_proj.bias, a.k_proj.bias, a.v_proj.bias],
                    dim=0,
                )
            )
            dst.attention.out_proj.load_state_dict(a.out_proj.state_dict())
            dst.norm1.load_state_dict(src.norm1.state_dict())
            dst.norm2.load_state_dict(src.norm2.state_dict())
            dst.ffn_in.load_state_dict(src.ffn_in.state_dict())
            dst.ffn_out.load_state_dict(src.ffn_out.state_dict())
        model.final_norm.load_state_dict(baseline.final_norm.state_dict())
