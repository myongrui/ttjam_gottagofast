"""v006 -- compile only the FFN residual path.

Hypothesis: the target cases are dominated by launch/dispatch overhead rather
than GEMM throughput, and compiling the small FFN subgraph alone can fuse
LayerNorm-adjacent pointwise work, GELU epilogues, and the residual add without
paying the guard/dispatch cost of compiling attention and mask handling.

Mechanism: attention remains the proven v001 SDPA implementation, while each
block's norm2 -> linear -> GELU -> linear -> residual sequence is compiled with
Inductor default mode. This isolates fusion to the launch-heavy FFN region and
avoids the full-model CUDA-graph strategy that regressed in v003.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FusedAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
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


class FFNResidual(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int):
        super().__init__()
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_in = nn.Linear(d_model, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, d_model)

    def forward(self, x):
        return x + self.ffn_out(
            F.gelu(self.ffn_in(self.norm2(x)), approximate="none")
        )


class Block(nn.Module):
    def __init__(self, d_model, num_heads, ffn_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = FusedAttention(d_model, num_heads)
        self.ffn = FFNResidual(d_model, ffn_dim)

        # Keep the original module registered for parameter movement and loading,
        # but do not register a second alias of it through the compiler wrapper.
        object.__setattr__(
            self,
            "_compiled_ffn",
            torch.compile(self.ffn, mode="default", fullgraph=True),
        )

    def forward(self, x, valid_token_mask, causal):
        x = x + self.attention(self.norm1(x), valid_token_mask, causal)
        x = self._compiled_ffn(x)
        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x


class Model(nn.Module):
    """Transformer using SDPA attention and a compiled FFN-residual subgraph."""
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
            dst.ffn.norm2.load_state_dict(src.norm2.state_dict())
            dst.ffn.ffn_in.load_state_dict(src.ffn_in.state_dict())
            dst.ffn.ffn_out.load_state_dict(src.ffn_out.state_dict())
        model.final_norm.load_state_dict(baseline.final_norm.state_dict())
