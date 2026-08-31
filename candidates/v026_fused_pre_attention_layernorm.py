"""Fused pre-attention LayerNorm and QKV projection candidate.

Hypothesis: every transformer block currently launches a LayerNorm kernel before
the fused QKV GEMM.  For the dominant d_model=128 shapes, a Triton kernel can
normalize each token and immediately multiply it by the concatenated QKV weight
matrix, eliminating one launch and the normalized activation materialization per
attention layer.  The remaining attention, residual, FFN, masking, and output
semantics are unchanged.
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
    def _norm_qkv_128_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        norm_weight_ptr,
        norm_bias_ptr,
        out_ptr,
        n_rows: tl.constexpr,
        eps: tl.constexpr,
        D: tl.constexpr,
        N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs_d = tl.arange(0, BLOCK_D)
        offs_n = tl.arange(0, BLOCK_N)

        x = tl.load(x_ptr + row * D + offs_d).to(tl.float32)
        mean = tl.sum(x, axis=0) / D
        centered = x - mean
        var = tl.sum(centered * centered, axis=0) / D
        inv_std = tl.rsqrt(var + eps)

        gamma = tl.load(norm_weight_ptr + offs_d).to(tl.float32)
        beta = tl.load(norm_bias_ptr + offs_d).to(tl.float32)
        normed = centered * inv_std * gamma + beta

        w = tl.load(
            weight_ptr + offs_n[None, :] * D + offs_d[:, None]
        ).to(tl.float32)
        projected = tl.dot(normed[None, :], w, input_precision="tf32")
        projected += tl.load(bias_ptr + offs_n).to(tl.float32)[None, :]

        tl.store(out_ptr + row * N + offs_n, projected[0, :])


class _NormQKV128(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, norm_weight, norm_bias, eps):
        rows = x.numel() // 128
        out = torch.empty((rows, 384), device=x.device, dtype=x.dtype)
        _norm_qkv_128_kernel[(rows,)](
            x.reshape(rows, 128),
            weight,
            bias,
            norm_weight,
            norm_bias,
            out,
            n_rows=rows,
            eps=eps,
            D=128,
            N=384,
            BLOCK_D=128,
            BLOCK_N=384,
            num_warps=4,
            num_stages=2,
        )
        return out.view(*x.shape[:-1], 384)

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise RuntimeError("Inference-only fused LayerNorm/QKV path does not support backward")


class FusedNormQKV(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)

    def forward(self, x):
        if (
            _HAS_TRITON
            and x.is_cuda
            and x.dtype == torch.float32
            and self.d_model == 128
            and x.is_contiguous()
        ):
            return _NormQKV128.apply(
                x,
                self.qkv.weight,
                self.qkv.bias,
                self.norm.weight,
                self.norm.bias,
                self.norm.eps,
            )
        return self.qkv(self.norm(x))


class FusedAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.pre_qkv = FusedNormQKV(d_model, num_heads)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x, attn_mask=None, causal=False):
        b, s, _ = x.shape
        qkv = self.pre_qkv(x).view(b, s, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        ctx = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=causal and attn_mask is None,
        )
        return ctx.transpose(1, 2).reshape(b, s, self.d_model)


class Block(nn.Module):
    def __init__(self, d_model, num_heads, ffn_dim):
        super().__init__()
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

    def forward(self, x, attn_mask, causal):
        ctx = self.attention(x, attn_mask, causal)
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
        for layer in self.layers:
            x = layer(x, attn_mask, self.config.causal)
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
            dst.attention.pre_qkv.norm.load_state_dict(src.norm1.state_dict())
            dst.attention.pre_qkv.qkv.weight.copy_(
                torch.cat(
                    [a.q_proj.weight, a.k_proj.weight, a.v_proj.weight],
                    dim=0,
                )
            )
            dst.attention.pre_qkv.qkv.bias.copy_(
                torch.cat(
                    [a.q_proj.bias, a.k_proj.bias, a.v_proj.bias],
                    dim=0,
                )
            )
            dst.attention.out_proj.load_state_dict(a.out_proj.state_dict())
            dst.norm2.load_state_dict(src.norm2.state_dict())
            dst.ffn_in.load_state_dict(src.ffn_in.state_dict())
            dst.ffn_out.load_state_dict(src.ffn_out.state_dict())
        model.final_norm.load_state_dict(baseline.final_norm.state_dict())
