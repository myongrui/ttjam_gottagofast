"""Hypothesis: the attention output projection need not materialize the expensive
[B, S, H, Dh] transpose/repack between SDPA and out_proj.

Mechanism: SDPA returns head-major [B, H, S, Dh] storage.  The custom Triton
kernel reads that layout directly while performing the output-projection GEMM
and bias epilogue, writing [B, S, D].  This removes the intermediate transpose/
contiguous-copy launch before every attention output projection while retaining
the fused-QKV and Flash/SDPA attention structure.
"""
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
    def _head_major_outproj_kernel(
        ctx_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        S: tl.constexpr,
        H: tl.constexpr,
        HD: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = offs_m < M
        n_mask = offs_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K

            batch = offs_m // S
            token = offs_m - batch * S
            head = offs_k // HD
            dim = offs_k - head * HD

            ctx_offsets = (
                (batch[:, None] * H + head[None, :]) * S * HD
                + token[:, None] * HD
                + dim[None, :]
            )
            a = tl.load(
                ctx_ptr + ctx_offsets,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )

            weight_offsets = offs_n[None, :] * K + offs_k[:, None]
            b = tl.load(
                weight_ptr + weight_offsets,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            acc += tl.dot(a, b, input_precision="tf32")

        bias = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
        acc += bias[None, :]

        out_offsets = offs_m[:, None] * N + offs_n[None, :]
        tl.store(out_ptr + out_offsets, acc, mask=m_mask[:, None] & n_mask[None, :])


class HeadMajorOutputProjection(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.linear = nn.Linear(d_model, d_model, bias=True)

    def forward(self, ctx):
        b, h, s, hd = ctx.shape
        if (
            not _HAS_TRITON
            or not ctx.is_cuda
            or ctx.dtype != torch.float32
            or h != self.num_heads
            or hd != self.head_dim
        ):
            return self.linear(ctx.transpose(1, 2).reshape(b, s, self.d_model))

        m = b * s
        n = self.d_model
        block_m = 64 if m >= 4096 else 32
        block_n = 128 if n <= 128 else 64
        out = torch.empty((b, s, n), device=ctx.device, dtype=ctx.dtype)
        grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
        _head_major_outproj_kernel[grid](
            ctx,
            self.linear.weight,
            self.linear.bias,
            out,
            M=m,
            N=n,
            K=n,
            S=s,
            H=h,
            HD=hd,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=32,
            num_warps=4,
            num_stages=3,
        )
        return out


class FusedAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = HeadMajorOutputProjection(d_model, num_heads)

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

    def forward(self, x, valid_token_mask, causal):
        x = x + self.attention(self.norm1(x), valid_token_mask, causal)
        x = x + self.ffn_out(F.gelu(self.ffn_in(self.norm2(x)), approximate="none"))
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
            dst.attention.out_proj.linear.load_state_dict(attn.out_proj.state_dict())
            dst.norm1.load_state_dict(src.norm1.state_dict())
            dst.norm2.load_state_dict(src.norm2.state_dict())
            dst.ffn_in.load_state_dict(src.ffn_in.state_dict())
            dst.ffn_out.load_state_dict(src.ffn_out.state_dict())
        model.final_norm.load_state_dict(baseline.final_norm.state_dict())
