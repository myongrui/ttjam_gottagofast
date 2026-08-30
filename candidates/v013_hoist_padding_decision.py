"""v013 -- hoist the padding decision out of the per-layer attention call.

Hypothesis:
  v007 (current per-shape elite on cases 6/8/11/13, screen_promote at 2.09x
  geomean on the general profile) still loses to v001 on case 2 specifically
  (paired speedup 0.819 vs incumbent -- an 18% regression on the smallest,
  most launch-bound shape). v007 inherited v001's FusedAttention unchanged, and
  that code does two wasteful things on every one of the 4 attention calls per
  forward:

    1. `bool(valid_token_mask.all())` forces a device-to-host synchronization,
       repeated once per layer even though the mask and causal flag are the
       same for the whole forward pass. v005 already tried removing this by
       switching unconditionally to a boolean SDPA mask -- that was refuted
       (screen_reject, 1.4457 vs incumbent) because it gave up the flash
       fast path even on the common all-valid case. That is not this change:
       the is_causal-only fast path is preserved for the no-padding case, only
       the *decision* is hoisted.
    2. `out.masked_fill(~valid_token_mask[..., None], 0)` inside
       FusedAttention is dead work under v007: `fused_residual_layernorm`
       already re-zeroes invalid rows in the same kernel that consumes `out`
       (`residual = tl.where(row_valid, residual, 0.0)`), so this masked_fill
       changes nothing about the final state and is a pure extra launch.

Mechanism:
  Compute `has_padding = valid_token_mask is not None and not
  valid_token_mask.all().item()` exactly once in Model.forward, and build the
  combined causal+padding additive mask once (it depends only on the mask and
  the causal flag, not per-layer state) instead of rebuilding it inside every
  layer. Pass `has_padding` and the precomputed mask down as plain arguments.
  FusedAttention drops the per-call sync, the per-call mask construction, and
  the now-redundant output masked_fill. `fused_residual_layernorm` is only
  given a mask tensor when `has_padding` is true, so the all-valid case also
  skips its on-device masking branch. The Triton fused residual+LayerNorm
  kernel itself is unchanged from v007.

  This is one host sync per forward instead of up to num_layers, one mask
  build instead of num_layers, and one fewer elementwise kernel launch per
  attention call -- all removed launches, no removed correctness: whenever
  real padding is present, has_padding is still true and the exact same
  additive-mask SDPA path and row-zeroing run as before.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _residual_layernorm_kernel(
    x_ptr,
    y_ptr,
    weight_ptr,
    bias_ptr,
    mask_ptr,
    residual_ptr,
    norm_ptr,
    stride_xr,
    stride_yr,
    stride_rr,
    stride_nr,
    eps,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    active = cols < N

    x = tl.load(x_ptr + row * stride_xr + cols, mask=active, other=0.0)
    y = tl.load(y_ptr + row * stride_yr + cols, mask=active, other=0.0)
    residual = x + y

    if HAS_MASK:
        row_valid = tl.load(mask_ptr + row).to(tl.int1)
        residual = tl.where(row_valid, residual, 0.0)

    mean = tl.sum(residual, axis=0) / N
    centered = tl.where(active, residual - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / N
    inv_std = tl.rsqrt(var + eps)

    weight = tl.load(weight_ptr + cols, mask=active, other=0.0)
    bias = tl.load(bias_ptr + cols, mask=active, other=0.0)
    normalized = centered * inv_std * weight + bias

    if HAS_MASK:
        normalized = tl.where(row_valid, normalized, 0.0)

    tl.store(residual_ptr + row * stride_rr + cols, residual, mask=active)
    tl.store(norm_ptr + row * stride_nr + cols, normalized, mask=active)


def fused_residual_layernorm(x, y, norm, valid_token_mask=None):
    b, s, d = x.shape
    x2 = x.reshape(-1, d)
    y2 = y.reshape(-1, d)
    residual = torch.empty_like(x2)
    normalized = torch.empty_like(x2)

    block = triton.next_power_of_2(d)
    if valid_token_mask is None:
        mask_ptr = x2
        has_mask = False
    else:
        mask_ptr = valid_token_mask.reshape(-1)
        has_mask = True

    _residual_layernorm_kernel[(b * s,)](
        x2,
        y2,
        norm.weight,
        norm.bias,
        mask_ptr,
        residual,
        normalized,
        x2.stride(0),
        y2.stride(0),
        residual.stride(0),
        normalized.stride(0),
        norm.eps,
        N=d,
        BLOCK=block,
        HAS_MASK=has_mask,
        num_warps=4,
    )
    return residual.view(b, s, d), normalized.view(b, s, d)


class FusedAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x, attn_mask, has_padding: bool, causal: bool):
        b, s, _ = x.shape
        qkv = self.qkv(x).view(b, s, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        if has_padding:
            ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            ctx = F.scaled_dot_product_attention(q, k, v, is_causal=causal)

        ctx = ctx.transpose(1, 2).reshape(b, s, self.d_model)
        out = self.out_proj(ctx)
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
        self.layers = nn.ModuleList(
            [
                Block(config.d_model, config.num_heads, config.ffn_dim)
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)

    def forward(self, x, valid_token_mask=None):
        b, s, _ = x.shape
        causal = self.config.causal

        has_padding = valid_token_mask is not None and not bool(valid_token_mask.all())
        if has_padding:
            keep = valid_token_mask[:, None, None, :]
            if causal:
                causal_keep = torch.ones(s, s, device=x.device, dtype=torch.bool).tril()
                keep = keep & causal_keep
            attn_mask = torch.zeros(keep.shape, device=x.device, dtype=x.dtype)
            attn_mask = attn_mask.masked_fill(~keep, float("-inf"))
            row_mask = valid_token_mask
        else:
            attn_mask = None
            row_mask = None

        state = x
        norm1 = self.layers[0].norm1(state)

        for index, layer in enumerate(self.layers):
            attn = layer.attention(norm1, attn_mask, has_padding, causal)
            state, norm2 = fused_residual_layernorm(state, attn, layer.norm2, row_mask)

            ffn = layer.ffn_out(F.gelu(layer.ffn_in(norm2), approximate="none"))

            if index + 1 < len(self.layers):
                state, norm1 = fused_residual_layernorm(
                    state, ffn, self.layers[index + 1].norm1, row_mask
                )
            else:
                _, state = fused_residual_layernorm(state, ffn, self.final_norm, row_mask)

        return state


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
