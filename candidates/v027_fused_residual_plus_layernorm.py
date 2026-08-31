"""Fused residual-plus-LayerNorm pipeline candidate.

Hypothesis: transformer residuals that immediately feed the next LayerNorm do
not need to be finalized by the preceding projection GEMM.  The attention and
FFN output projections run without bias/residual epilogues, then one Triton
kernel performs residual addition, projection bias addition, and LayerNorm
together while retaining the summed residual for the following block.  This
replaces separate residual materialization and LayerNorm work with a single
kernel at every inter-layer boundary while preserving the original pre-norm
transformer algebra.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _residual_layernorm_kernel(
    residual_ptr,
    update_ptr,
    bias_ptr,
    weight_ptr,
    norm_bias_ptr,
    out_ptr,
    summed_ptr,
    stride,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
    STORE_SUM: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < stride
    offsets = row * stride + cols

    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
    update = tl.load(update_ptr + offsets, mask=mask, other=0.0)
    proj_bias = tl.load(bias_ptr + cols, mask=mask, other=0.0)
    summed = residual + update + proj_bias

    mean = tl.sum(summed, axis=0) / stride
    centered = tl.where(mask, summed - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / stride
    normalized = centered * tl.rsqrt(var + eps)

    gamma = tl.load(weight_ptr + cols, mask=mask, other=0.0)
    beta = tl.load(norm_bias_ptr + cols, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, normalized * gamma + beta, mask=mask)

    if STORE_SUM:
        tl.store(summed_ptr + offsets, summed, mask=mask)


def _fused_residual_layernorm(residual, update, projection_bias, norm, keep_sum=True):
    d_model = residual.shape[-1]
    block = triton.next_power_of_2(d_model)
    normalized = torch.empty_like(residual)
    summed = torch.empty_like(residual) if keep_sum else normalized
    rows = residual.numel() // d_model

    _residual_layernorm_kernel[(rows,)](
        residual,
        update,
        projection_bias,
        norm.weight,
        norm.bias,
        normalized,
        summed,
        d_model,
        eps=norm.eps,
        BLOCK=block,
        STORE_SUM=keep_sum,
        num_warps=4,
    )
    return summed, normalized


class FusedAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x, attn_mask=None, causal=False):
        b, s, _ = x.shape
        qkv = self.qkv(x).view(b, s, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        ctx = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=causal and attn_mask is None,
        )
        ctx = ctx.transpose(1, 2).reshape(b, s, self.d_model)
        return F.linear(ctx, self.out_proj.weight, None)


class Block(nn.Module):
    def __init__(self, d_model, num_heads, ffn_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = FusedAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_in = nn.Linear(d_model, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, d_model)

    def attention_update(self, normalized_x, attn_mask, causal):
        return self.attention(normalized_x, attn_mask, causal)

    def ffn_update(self, normalized_x):
        hidden = F.gelu(self.ffn_in(normalized_x), approximate="none")
        return F.linear(hidden, self.ffn_out.weight, None)


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

    def _attention_mask(self, x, valid_token_mask):
        if valid_token_mask is None or bool(valid_token_mask.all()):
            return None

        s = x.shape[1]
        keep = valid_token_mask[:, None, None, :]
        if self.config.causal:
            keep = keep & torch.ones(
                s, s, device=x.device, dtype=torch.bool
            ).tril()

        mask = torch.zeros(keep.shape, device=x.device, dtype=x.dtype)
        return mask.masked_fill(~keep, float("-inf"))

    def forward(self, x, valid_token_mask=None):
        attn_mask = self._attention_mask(x, valid_token_mask)
        normalized = self.layers[0].norm1(x)

        for index, layer in enumerate(self.layers):
            attention_update = layer.attention_update(
                normalized, attn_mask, self.config.causal
            )
            after_attention, normalized_ffn = _fused_residual_layernorm(
                x,
                attention_update,
                layer.attention.out_proj.bias,
                layer.norm2,
                keep_sum=True,
            )

            ffn_update = layer.ffn_update(normalized_ffn)

            if index + 1 < len(self.layers):
                next_layer = self.layers[index + 1]
                x, normalized = _fused_residual_layernorm(
                    after_attention,
                    ffn_update,
                    layer.ffn_out.bias,
                    next_layer.norm1,
                    keep_sum=True,
                )
            else:
                _, x = _fused_residual_layernorm(
                    after_attention,
                    ffn_update,
                    layer.ffn_out.bias,
                    self.final_norm,
                    keep_sum=False,
                )

        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x


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
