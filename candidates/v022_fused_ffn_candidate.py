"""Fused-FFN candidate.

Hypothesis: for the common d_model=ffn_dim=128 cases, the FFN is dominated by
kernel-launch overhead rather than GEMM throughput.  A Triton kernel computes
Linear -> exact GELU -> Linear -> residual addition as one fused operation,
removing the intermediate activation materialization and replacing the FFN's
multiple launches with one kernel while retaining the existing SDPA attention
and weight layout.  Other shapes use the exact eager fallback.
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
    def _fused_ffn_kernel(
        x_ptr,
        residual_ptr,
        w1_ptr,
        b1_ptr,
        w2_ptr,
        b2_ptr,
        out_ptr,
        n_rows: tl.constexpr,
        D: tl.constexpr,
        H: tl.constexpr,
        BM: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rows = pid * BM + tl.arange(0, BM)
        cols_d = tl.arange(0, D)
        cols_h = tl.arange(0, H)

        row_mask = rows < n_rows

        x = tl.load(
            x_ptr + rows[:, None] * D + cols_d[None, :],
            mask=row_mask[:, None],
            other=0.0,
        )
        residual = tl.load(
            residual_ptr + rows[:, None] * D + cols_d[None, :],
            mask=row_mask[:, None],
            other=0.0,
        )

        w1 = tl.load(w1_ptr + cols_h[None, :] * D + cols_d[:, None])
        hidden = tl.dot(x, w1)
        hidden += tl.load(b1_ptr + cols_h)[None, :]
        hidden = 0.5 * hidden * (
            1.0 + tl.erf(hidden * 0.7071067811865476)
        )

        w2 = tl.load(w2_ptr + cols_d[None, :] * H + cols_h[:, None])
        out = tl.dot(hidden, w2)
        out += tl.load(b2_ptr + cols_d)[None, :]
        out += residual

        tl.store(
            out_ptr + rows[:, None] * D + cols_d[None, :],
            out,
            mask=row_mask[:, None],
        )


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
            attn_mask = torch.zeros(keep.shape, device=x.device, dtype=x.dtype)
            attn_mask = attn_mask.masked_fill(~keep, float("-inf"))
            ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            ctx = F.scaled_dot_product_attention(q, k, v, is_causal=causal)

        return ctx.transpose(1, 2).reshape(b, s, self.d_model)


class FusedFFN(nn.Module):
    def __init__(self, d_model, ffn_dim):
        super().__init__()
        self.d_model = d_model
        self.ffn_dim = ffn_dim
        self.ffn_in = nn.Linear(d_model, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, d_model)

    def forward(self, x, residual):
        use_fused = (
            _HAS_TRITON
            and x.is_cuda
            and x.dtype == torch.float32
            and self.d_model == 128
            and self.ffn_dim == 128
            and not x.requires_grad
        )
        if not use_fused:
            hidden = F.gelu(self.ffn_in(x), approximate="none")
            b, s, d = residual.shape
            c = residual.reshape(b * s, d) + self.ffn_out.bias
            return torch.addmm(
                c,
                hidden.reshape(b * s, self.ffn_dim),
                self.ffn_out.weight.t(),
            ).view(b, s, d)

        b, s, d = x.shape
        out = torch.empty_like(x)
        n_rows = b * s
        _fused_ffn_kernel[(triton.cdiv(n_rows, 16),)](
            x,
            residual,
            self.ffn_in.weight,
            self.ffn_in.bias,
            self.ffn_out.weight,
            self.ffn_out.bias,
            out,
            n_rows=n_rows,
            D=128,
            H=128,
            BM=16,
            num_warps=4,
            num_stages=2,
        )
        return out


class Block(nn.Module):
    def __init__(self, d_model, num_heads, ffn_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = FusedAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FusedFFN(d_model, ffn_dim)

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
        return self.ffn(self.norm2(x), x)


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
            dst.ffn.ffn_in.load_state_dict(src.ffn_in.state_dict())
            dst.ffn.ffn_out.load_state_dict(src.ffn_out.state_dict())
        model.final_norm.load_state_dict(baseline.final_norm.state_dict())
