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
    token_mask_ptr,
    out_ptr,
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    d_model: tl.constexpr,
    x_stride_b: tl.constexpr,
    x_stride_s: tl.constexpr,
    mask_stride_b: tl.constexpr,
    mask_stride_s: tl.constexpr,
    out_stride_b: tl.constexpr,
    out_stride_s: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // seq_len
    seq = row - batch * seq_len
    offsets = tl.arange(0, BLOCK)
    valid_feature = offsets < d_model

    x_offsets = batch * x_stride_b + seq * x_stride_s + offsets
    values = tl.load(x_ptr + x_offsets, mask=valid_feature, other=0.0).to(tl.float32)

    mean = tl.sum(values, axis=0) / d_model
    centered = tl.where(valid_feature, values - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / d_model
    normalized = centered * tl.rsqrt(variance + eps)

    scale = tl.load(weight_ptr + offsets, mask=valid_feature, other=0.0).to(tl.float32)
    shift = tl.load(bias_ptr + offsets, mask=valid_feature, other=0.0).to(tl.float32)
    result = normalized * scale + shift

    token_valid = tl.load(
        token_mask_ptr + batch * mask_stride_b + seq * mask_stride_s
    ).to(tl.int1)
    result = tl.where(token_valid, result, 0.0)

    out_offsets = batch * out_stride_b + seq * out_stride_s + offsets
    tl.store(out_ptr + out_offsets, result, mask=valid_feature)


class FusedAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
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

    def forward(self, x, attn_mask, causal):
        ctx = self.attention(self.norm1(x), attn_mask, causal)
        x = self._residual_projection(x, ctx, self.attention.out_proj)
        hidden = F.gelu(self.ffn_in(self.norm2(x)), approximate="none")
        return self._residual_projection(x, hidden, self.ffn_out)


class Model(nn.Module):
    """Hypothesis: fusing the final LayerNorm and padded-token zeroing into one
    Triton row kernel removes the separate output masking launch on every masked
    invocation.  The kernel performs the same FP32 LayerNorm reduction and writes
    zeros directly for invalid rows, preserving padded-output semantics while
    reducing latency-bound epilogue launch overhead.
    """

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

    def _final_norm_and_mask(self, x, valid_token_mask):
        if (
            valid_token_mask is None
            or not x.is_cuda
            or x.dtype != torch.float32
            or self.final_norm.weight.dtype != torch.float32
        ):
            x = self.final_norm(x)
            if valid_token_mask is not None:
                x = x.masked_fill(~valid_token_mask[..., None], 0)
            return x

        b, s, d = x.shape
        block = triton.next_power_of_2(d)
        if block > 8192:
            x = self.final_norm(x)
            return x.masked_fill(~valid_token_mask[..., None], 0)

        out = torch.empty_like(x)
        _final_norm_mask_kernel[(b * s,)](
            x,
            self.final_norm.weight,
            self.final_norm.bias,
            valid_token_mask,
            out,
            b,
            s,
            d,
            x.stride(0),
            x.stride(1),
            valid_token_mask.stride(0),
            valid_token_mask.stride(1),
            out.stride(0),
            out.stride(1),
            self.final_norm.eps,
            BLOCK=block,
            num_warps=4,
        )
        return out

    def forward(self, x, valid_token_mask=None):
        attn_mask = self._attention_mask(x, valid_token_mask)
        for layer in self.layers:
            x = layer(x, attn_mask, self.config.causal)
        return self._final_norm_and_mask(x, valid_token_mask)


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
