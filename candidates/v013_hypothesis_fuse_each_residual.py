"""Hypothesis: fuse each residual addition with the LayerNorm that immediately
consumes it.

Mechanism: transformer residuals are normally produced by a standalone add
kernel and then reread by LayerNorm.  This implementation uses one Triton
kernel to compute and store the residual while simultaneously reducing it for
LayerNorm and writing the normalized activations.  The residual must still be
materialized because it participates in the next sublayer, but the separate
add launch and its intermediate read are eliminated.  Attention remains SDPA
with fused QKV so long-sequence causal cases retain the FlashAttention path.
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
    mask_ptr,
    out_residual_ptr,
    out_norm_ptr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    has_mask: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    offsets = row * n_cols + cols
    col_mask = cols < n_cols

    x = tl.load(x_ptr + offsets, mask=col_mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=col_mask, other=0.0).to(tl.float32)
    summed = x + residual

    if has_mask:
        valid = tl.load(mask_ptr + row)
        summed = tl.where(valid, summed, 0.0)

    mean = tl.sum(summed, axis=0) / n_cols
    centered = tl.where(col_mask, summed - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / n_cols
    inv_std = tl.rsqrt(var + eps)

    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    normalized = centered * inv_std * weight + bias

    tl.store(out_residual_ptr + offsets, summed, mask=col_mask)
    tl.store(out_norm_ptr + offsets, normalized, mask=col_mask)


def _residual_layernorm(x, residual, norm, valid_token_mask=None):
    if (
        not x.is_cuda
        or not x.is_contiguous()
        or not residual.is_contiguous()
        or x.shape[-1] != norm.normalized_shape[0]
    ):
        y = x + residual
        if valid_token_mask is not None:
            y = y.masked_fill(~valid_token_mask[..., None], 0)
        return y, F.layer_norm(y, norm.normalized_shape, norm.weight, norm.bias, norm.eps)

    b, s, d = x.shape
    y = torch.empty_like(x)
    out = torch.empty_like(x)
    flat_mask = x.reshape(-1)
    has_mask = valid_token_mask is not None
    if has_mask:
        flat_mask = valid_token_mask.reshape(-1)

    block = triton.next_power_of_2(d)
    _residual_layernorm_kernel[(b * s,)](
        x,
        residual,
        norm.weight,
        norm.bias,
        flat_mask,
        y,
        out,
        n_cols=d,
        eps=norm.eps,
        has_mask=has_mask,
        BLOCK=block,
        num_warps=4,
    )
    return y, out


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

        if valid_token_mask is not None and not bool(valid_token_mask.all()):
            keep = valid_token_mask[:, None, None, :]
            if causal:
                causal_keep = torch.ones(
                    s, s, device=x.device, dtype=torch.bool
                ).tril()
                keep = keep & causal_keep
            attn_mask = torch.zeros(keep.shape, device=x.device, dtype=x.dtype)
            attn_mask = attn_mask.masked_fill(~keep, float("-inf"))
            ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            ctx = F.scaled_dot_product_attention(q, k, v, is_causal=causal)

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

    def ffn(self, normalized):
        return self.ffn_out(F.gelu(self.ffn_in(normalized), approximate="none"))


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

    def forward(self, x, valid_token_mask=None):
        if len(self.layers) == 0:
            x = self.final_norm(x)
            if valid_token_mask is not None:
                x = x.masked_fill(~valid_token_mask[..., None], 0)
            return x

        normalized = self.layers[0].norm1(x)
        for i, layer in enumerate(self.layers):
            attention_out = layer.attention(
                normalized, valid_token_mask, self.config.causal
            )
            post_attention, normalized_ffn = _residual_layernorm(
                x, attention_out, layer.norm2, valid_token_mask
            )
            ffn_out = layer.ffn(normalized_ffn)

            if i + 1 < len(self.layers):
                x, normalized = _residual_layernorm(
                    post_attention,
                    ffn_out,
                    self.layers[i + 1].norm1,
                    valid_token_mask,
                )
            else:
                _, normalized = _residual_layernorm(
                    post_attention, ffn_out, self.final_norm, valid_token_mask
                )

        return normalized


def build_model(config, bench):
    return Model(config)


def load_from_baseline(model, baseline):
    with torch.no_grad():
        for dst, src in zip(model.layers, baseline.layers):
            attn = src.attention
            dst.attention.qkv.weight.copy_(
                torch.cat(
                    [attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight],
                    dim=0,
                )
            )
            dst.attention.qkv.bias.copy_(
                torch.cat(
                    [attn.q_proj.bias, attn.k_proj.bias, attn.v_proj.bias],
                    dim=0,
                )
            )
            dst.attention.out_proj.load_state_dict(attn.out_proj.state_dict())
            dst.norm1.load_state_dict(src.norm1.state_dict())
            dst.norm2.load_state_dict(src.norm2.state_dict())
            dst.ffn_in.load_state_dict(src.ffn_in.state_dict())
            dst.ffn_out.load_state_dict(src.ffn_out.state_dict())
        model.final_norm.load_state_dict(baseline.final_norm.state_dict())
