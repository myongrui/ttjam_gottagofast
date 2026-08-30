"""Hypothesis: fuse each LayerNorm directly into the following projection, while
also folding the preceding residual add into that same kernel when possible.

Mechanism: the normal transformer path materializes normalized activations after
every residual/LN pair, then launches a separate GEMM for QKV or FFN input.
This candidate uses one Triton kernel that computes residual + LayerNorm and
immediately performs the small projection tile. For layers after the first, the
kernel consumes the prior residual and FFN output together, writes the new
residual once, and produces QKV without materializing the normalized tensor.
The attention-to-FFN transition uses the same residual+LN+FFN-input fusion.
This removes several latency-dominant launches and intermediate writes on the
small d_model=128 cases while retaining SDPA for feasible causal attention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _residual_norm_linear_kernel(
    x_ptr,
    add_ptr,
    gamma_ptr,
    beta_ptr,
    weight_ptr,
    bias_ptr,
    residual_out_ptr,
    out_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    BK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    HAS_ADD: tl.constexpr,
    WRITE_RESIDUAL: tl.constexpr,
    EPS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rows = pid_m * BM + tl.arange(0, BM)
    ks = tl.arange(0, BK)
    ns = pid_n * BN + tl.arange(0, BN)

    row_mask = rows < M
    k_mask = ks < K
    x_mask = row_mask[:, None] & k_mask[None, :]

    x = tl.load(x_ptr + rows[:, None] * K + ks[None, :], mask=x_mask, other=0.0)
    if HAS_ADD:
        a = tl.load(add_ptr + rows[:, None] * K + ks[None, :], mask=x_mask, other=0.0)
        x = x + a

    mean = tl.sum(x, axis=1) / K
    centered = x - mean[:, None]
    var = tl.sum(centered * centered, axis=1) / K
    inv_std = tl.rsqrt(var + EPS)

    gamma = tl.load(gamma_ptr + ks, mask=k_mask, other=0.0)
    beta = tl.load(beta_ptr + ks, mask=k_mask, other=0.0)
    normed = centered * inv_std[:, None] * gamma[None, :] + beta[None, :]

    w = tl.load(
        weight_ptr + ns[:, None] * K + ks[None, :],
        mask=(ns[:, None] < N) & k_mask[None, :],
        other=0.0,
    )
    result = tl.dot(normed, tl.trans(w), input_precision="tf32")
    b = tl.load(bias_ptr + ns, mask=ns < N, other=0.0)
    result += b[None, :]

    tl.store(
        out_ptr + rows[:, None] * N + ns[None, :],
        result,
        mask=row_mask[:, None] & (ns[None, :] < N),
    )

    if WRITE_RESIDUAL and pid_n == 0:
        tl.store(
            residual_out_ptr + rows[:, None] * K + ks[None, :],
            x,
            mask=x_mask,
        )


def _next_power_of_2(n):
    return 1 << (n - 1).bit_length()


def fused_norm_linear(x, add, norm, linear, write_residual):
    b, s, k = x.shape
    m = b * s
    n = linear.out_features
    bk = _next_power_of_2(k)
    bm = 8
    bn = 64 if n >= 64 else _next_power_of_2(n)

    x2 = x.contiguous().view(m, k)
    add2 = x2 if add is None else add.contiguous().view(m, k)
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    residual = torch.empty((m, k), device=x.device, dtype=x.dtype) if write_residual else x2

    grid = (triton.cdiv(m, bm), triton.cdiv(n, bn))
    _residual_norm_linear_kernel[grid](
        x2,
        add2,
        norm.weight,
        norm.bias,
        linear.weight,
        linear.bias,
        residual,
        out,
        M=m,
        K=k,
        N=n,
        BK=bk,
        BM=bm,
        BN=bn,
        HAS_ADD=add is not None,
        WRITE_RESIDUAL=write_residual,
        EPS=norm.eps,
        num_warps=4,
    )
    if write_residual:
        residual = residual.view(b, s, k)
    return residual, out.view(b, s, n)


class FusedAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def forward_qkv(self, qkv, valid_token_mask=None, causal=False):
        b, s, _ = qkv.shape
        qkv = qkv.view(b, s, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        if valid_token_mask is not None and not bool(valid_token_mask.all()):
            keep = valid_token_mask[:, None, None, :]
            if causal:
                keep = keep & torch.ones(
                    s, s, device=qkv.device, dtype=torch.bool
                ).tril()
            mask = torch.zeros(keep.shape, device=qkv.device, dtype=qkv.dtype)
            mask = mask.masked_fill(~keep, float("-inf"))
            ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
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
        first = self.layers[0]
        _, qkv = fused_norm_linear(
            x, None, first.norm1, first.attention.qkv, write_residual=False
        )

        for i, layer in enumerate(self.layers):
            attn = layer.attention.forward_qkv(
                qkv, valid_token_mask, self.config.causal
            )

            residual, h = fused_norm_linear(
                x, attn, layer.norm2, layer.ffn_in, write_residual=True
            )
            ffn = layer.ffn_out(F.gelu(h, approximate="none"))

            if i + 1 < len(self.layers):
                nxt = self.layers[i + 1]
                x, qkv = fused_norm_linear(
                    residual,
                    ffn,
                    nxt.norm1,
                    nxt.attention.qkv,
                    write_residual=True,
                )
            else:
                x = residual + ffn

            if valid_token_mask is not None:
                x = x.masked_fill(~valid_token_mask[..., None], 0)

        x = self.final_norm(x)
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
                torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight], dim=0)
            )
            dst.attention.qkv.bias.copy_(
                torch.cat([a.q_proj.bias, a.k_proj.bias, a.v_proj.bias], dim=0)
            )
            dst.attention.out_proj.load_state_dict(a.out_proj.state_dict())
            dst.norm1.load_state_dict(src.norm1.state_dict())
            dst.norm2.load_state_dict(src.norm2.state_dict())
            dst.ffn_in.load_state_dict(src.ffn_in.state_dict())
            dst.ffn_out.load_state_dict(src.ffn_out.state_dict())
        model.final_norm.load_state_dict(baseline.final_norm.state_dict())
