"""Native TransformerEncoderLayer fast-path candidate.

Hypothesis: on unmasked, non-causal latency-bound cases, PyTorch's native fused
Transformer encoder operator can replace the Python-level sequence of LayerNorm,
QKV projection, SDPA, output projection, residual adds, and FFN calls with one
structural operator dispatch per layer.  Masked and causal inputs retain the
known-correct fused-QKV SDPA implementation, so arbitrary valid-token masks and
causal semantics remain exact.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


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

        if valid_token_mask is None:
            ctx = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        else:
            keep = valid_token_mask[:, None, None, :]
            if causal:
                causal_keep = torch.ones(
                    s, s, device=x.device, dtype=torch.bool
                ).tril()
                keep = keep & causal_keep
            attn_mask = torch.zeros(keep.shape, device=x.device, dtype=x.dtype)
            attn_mask = attn_mask.masked_fill(~keep, float("-inf"))
            ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        out = self.out_proj(ctx.transpose(1, 2).reshape(b, s, self.d_model))
        if valid_token_mask is not None:
            out = out.masked_fill(~valid_token_mask[..., None], 0)
        return out


class Block(nn.Module):
    def __init__(self, d_model, num_heads, ffn_dim):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = FusedAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_in = nn.Linear(d_model, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, d_model)

    def native_forward(self, x):
        return torch._transformer_encoder_layer_fwd(
            x,
            self.d_model,
            self.num_heads,
            self.attention.qkv.weight,
            self.attention.qkv.bias,
            self.attention.out_proj.weight,
            self.attention.out_proj.bias,
            True,
            True,
            self.norm1.eps,
            self.norm1.weight,
            self.norm1.bias,
            self.norm2.weight,
            self.norm2.bias,
            self.ffn_in.weight,
            self.ffn_in.bias,
            self.ffn_out.weight,
            self.ffn_out.bias,
            None,
            None,
        )

    def forward(self, x, valid_token_mask, causal):
        if valid_token_mask is None and not causal:
            return self.native_forward(x)

        x = x + self.attention(self.norm1(x), valid_token_mask, causal)
        x = x + self.ffn_out(F.gelu(self.ffn_in(self.norm2(x)), approximate="none"))
        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x


class Model(nn.Module):
    """Uses the native fused encoder path only where its no-mask semantics match."""
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
            dst.attention.out_proj.load_state_dict(attn.out_proj.state_dict())
            dst.norm1.load_state_dict(src.norm1.state_dict())
            dst.norm2.load_state_dict(src.norm2.state_dict())
            dst.ffn_in.load_state_dict(src.ffn_in.state_dict())
            dst.ffn_out.load_state_dict(src.ffn_out.state_dict())
        model.final_norm.load_state_dict(baseline.final_norm.state_dict())
