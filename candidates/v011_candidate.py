"""
Hypothesis: pipeline adjacent residual-add and LayerNorm operations across
transformer block boundaries, and implement each residual-plus-normalization as
one Triton kernel.

Mechanism: after attention, the residual result is needed both by the FFN
residual path and by norm2; after the FFN, it is needed both as the next block's
residual state and as that next block's norm1 input.  The fused kernel writes
the residual state while simultaneously computing its LayerNorm output, removing
one standalone residual-add launch and one standalone LayerNorm launch at each
such boundary.  The first norm1 remains standalone; every later norm1 is fused
with the preceding FFN residual, and final_norm is fused with the final FFN
residual.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _residual_layernorm_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    valid_ptr,
    out_residual_ptr,
    out_norm_ptr,
    n_cols,
    eps,
    ZERO_INVALID: tl.constexpr,
    HAS_VALID_MASK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    offset = row * n_cols + cols

    x = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offset, mask=mask, other=0.0).to(tl.float32)

    valid = 1
    if HAS_VALID_MASK:
        valid = tl.load(valid_ptr + row).to(tl.int1)

    value = x + residual
    if ZERO_INVALID:
        value = tl.where(valid, value, 0.0)

    tl.store(out_residual_ptr + offset, value, mask=mask)

    mean = tl.sum(value, axis=0) / n_cols
    centered = value - mean
    variance = tl.sum(centered * centered, axis=0) / n_cols
    inv_std = tl.rsqrt(variance + eps)

    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    normed = centered * inv_std * weight + bias
    tl.store(out_norm_ptr + offset, normed, mask=mask)


def residual_layernorm(x, residual, norm, valid_token_mask=None, zero_invalid=False):
    b, s, d = x.shape
    out_residual = torch.empty_like(x)
    out_norm = torch.empty_like(x)
    block = triton.next_power_of_2(d)
    if block > 1024:
        raise RuntimeError("d_model exceeds supported fused LayerNorm width")

    valid_ptr = valid_token_mask.reshape(-1) if valid_token_mask is not None else x.reshape(-1)
    _residual_layernorm_kernel[(b * s,)](
        x,
        residual,
        norm.weight,
        norm.bias,
        valid_ptr,
        out_residual,
        out_norm,
        d,
        norm.eps,
        ZERO_INVALID=zero_invalid,
        HAS_VALID_MASK=valid_token_mask is not None,
        BLOCK=block,
        num_warps=4,
    )
    return out_residual, out_norm


class FusedAttention(nn.Module):
    def __init__(self, d_model, num_heads):
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

        if valid_token_mask is None:
            ctx = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        else:
            keep = valid_token_mask[:, None, None, :]
            if causal:
                causal_keep = torch.ones(
                    s, s, device=x.device, dtype=torch.bool
                ).tril()
                keep = keep & causal_keep
            ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=keep)

        ctx = ctx.transpose(1, 2).reshape(b, s, self.d_model)
        out = self.out_proj(ctx)
        if valid_token_mask is not None:
            out = out.masked_fill(~valid_token_mask[..., None], 0)
        return out


class Block(nn.Module):
    def __init__(self, d_model, num_heads, ffn_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = FusedAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_in = nn.Linear(d_model, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, d_model)


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            Block(config.d_model, config.num_heads, config.ffn_dim)
            for _ in range(config.num_layers)
        ])
        self.final_norm = nn.LayerNorm(config.d_model)

    def forward(self, x, valid_token_mask=None):
        if len(self.layers) == 0:
            x = self.final_norm(x)
            if valid_token_mask is not None:
                x = x.masked_fill(~valid_token_mask[..., None], 0)
            return x

        h = self.layers[0].norm1(x)

        for i, layer in enumerate(self.layers):
            attention_out = layer.attention(
                h, valid_token_mask, self.config.causal
            )

            x, h_ffn = residual_layernorm(
                x,
                attention_out,
                layer.norm2,
                valid_token_mask=valid_token_mask,
                zero_invalid=False,
            )

            ffn_out = layer.ffn_out(
                F.gelu(layer.ffn_in(h_ffn), approximate="none")
            )

            next_norm = (
                self.layers[i + 1].norm1
                if i + 1 < len(self.layers)
                else self.final_norm
            )
            x, h = residual_layernorm(
                x,
                ffn_out,
                next_norm,
                valid_token_mask=valid_token_mask,
                zero_invalid=True,
            )

        if valid_token_mask is not None:
            h = h.masked_fill(~valid_token_mask[..., None], 0)
        return h


def build_model(config, bench):
    return Model(config)


def load_from_baseline(model, baseline):
    with torch.no_grad():
        for dst, src in zip(model.layers, baseline.layers):
            attention = src.attention
            dst.attention.qkv.weight.copy_(torch.cat([
                attention.q_proj.weight,
                attention.k_proj.weight,
                attention.v_proj.weight,
            ], dim=0))
            dst.attention.qkv.bias.copy_(torch.cat([
                attention.q_proj.bias,
                attention.k_proj.bias,
                attention.v_proj.bias,
            ], dim=0))
            dst.attention.out_proj.load_state_dict(attention.out_proj.state_dict())
            dst.norm1.load_state_dict(src.norm1.state_dict())
            dst.norm2.load_state_dict(src.norm2.state_dict())
            dst.ffn_in.load_state_dict(src.ffn_in.state_dict())
            dst.ffn_out.load_state_dict(src.ffn_out.state_dict())
        model.final_norm.load_state_dict(baseline.final_norm.state_dict())
