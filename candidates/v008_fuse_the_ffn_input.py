"""v008 -- fuse the FFN input projection's exact GELU epilogue.

Hypothesis: after v007 has removed residual/LayerNorm launches, the remaining
small-shape latency is dominated by the standalone exact-GELU launch between
ffn_in and ffn_out.  This candidate replaces `Linear(ffn_in) + GELU` with one
Triton GEMM epilogue kernel.  The kernel computes the biased first FFN
projection and exact erf-based GELU before writing the activation consumed by
ffn_out, eliminating one launch per transformer block while retaining fp32
accumulation and the baseline's exact GELU definition.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


if _HAS_TRITON:
    @triton.jit
    def _linear_gelu_kernel(
        x_ptr,
        w_ptr,
        b_ptr,
        y_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for k0 in range(0, K, BLOCK_K):
            x = tl.load(
                x_ptr + offs_m[:, None] * K + (k0 + offs_k)[None, :],
                mask=(offs_m[:, None] < M) & ((k0 + offs_k)[None, :] < K),
                other=0.0,
            )
            w = tl.load(
                w_ptr + offs_n[None, :] * K + (k0 + offs_k)[:, None],
                mask=(offs_n[None, :] < N) & ((k0 + offs_k)[:, None] < K),
                other=0.0,
            )
            acc += tl.dot(x, w)

        bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
        z = acc + bias[None, :]
        gelu = 0.5 * z * (1.0 + tl.erf(z * 0.7071067811865476))
        tl.store(
            y_ptr + offs_m[:, None] * N + offs_n[None, :],
            gelu,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )


class LinearExactGELU(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        if (
            not _HAS_TRITON
            or not x.is_cuda
            or x.dtype != torch.float32
            or not x.is_contiguous()
        ):
            return F.gelu(self.linear(x), approximate="none")

        m = x.numel() // x.shape[-1]
        k = x.shape[-1]
        n = self.linear.weight.shape[0]
        y = torch.empty((m, n), device=x.device, dtype=x.dtype)

        grid = (triton.cdiv(m, 32), triton.cdiv(n, 64))
        _linear_gelu_kernel[grid](
            x.reshape(m, k),
            self.linear.weight,
            self.linear.bias,
            y,
            M=m,
            N=n,
            K=k,
            BLOCK_M=32,
            BLOCK_N=64,
            BLOCK_K=32,
            num_warps=4,
        )
        return y.view(*x.shape[:-1], n)


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
                keep = keep & torch.ones(
                    s, s, device=x.device, dtype=torch.bool
                ).tril()
            mask = torch.zeros(keep.shape, device=x.device, dtype=x.dtype)
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
        self.ffn_in = LinearExactGELU(d_model, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, d_model)

    def forward(self, x, valid_token_mask, causal):
        x = x + self.attention(self.norm1(x), valid_token_mask, causal)
        x = x + self.ffn_out(self.ffn_in(self.norm2(x)))
        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x


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
        for layer in self.layers:
            x = layer(x, valid_token_mask, self.config.causal)
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
            dst.ffn_in.linear.load_state_dict(src.ffn_in.state_dict())
            dst.ffn_out.load_state_dict(src.ffn_out.state_dict())
        model.final_norm.load_state_dict(baseline.final_norm.state_dict())
