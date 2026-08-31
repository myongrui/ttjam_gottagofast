"""Centralized attention-mask dispatch candidate.

Hypothesis: deciding whether a valid-token mask is fully valid once per model
invocation, and constructing a padded attention mask once when needed, removes
three host-device synchronizations and three repeated mask-construction sequences
per transformer layer stack.  Each attention layer receives the already-selected
SDPA mask mode, preserving exact padded-key exclusion while reducing launch and
synchronization overhead on latency-bound shapes.
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
