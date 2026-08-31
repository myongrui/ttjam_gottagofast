"""Folded-attention-output candidate.

Hypothesis: the attention output projection is a removable launch-bound GEMM on the
target small-width cases.  Because attention is linear in V, this implementation
folds the baseline output-projection matrix and bias into the V projection during
weight loading: attention(Q, K, V) @ W_out.T + b_out becomes
attention(Q, K, V @ W_out.T + b_v @ W_out.T + b_out).  The attention result is
therefore already in model space, eliminating every attention output-projection
GEMM while retaining only the necessary residual-add kernel.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FoldedAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)

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


class Block(nn.Module):
    def __init__(self, d_model, num_heads, ffn_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = FoldedAttention(d_model, num_heads)
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

    def forward(self, x, valid_token_mask, causal):
        x = x + self.attention(self.norm1(x), valid_token_mask, causal)
        hidden = F.gelu(self.ffn_in(self.norm2(x)), approximate="none")
        x = self._residual_projection(x, hidden, self.ffn_out)

        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x


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
            wo = a.out_proj.weight.double()
            wv = a.v_proj.weight.double()
            bv = a.v_proj.bias.double()
            bo = a.out_proj.bias.double()

            folded_v_weight = (wo @ wv).to(dtype=dst.attention.qkv.weight.dtype)
            folded_v_bias = (bv @ wo.t() + bo).to(
                dtype=dst.attention.qkv.bias.dtype
            )

            dst.attention.qkv.weight.copy_(
                torch.cat(
                    [
                        a.q_proj.weight,
                        a.k_proj.weight,
                        folded_v_weight,
                    ],
                    dim=0,
                )
            )
            dst.attention.qkv.bias.copy_(
                torch.cat(
                    [
                        a.q_proj.bias,
                        a.k_proj.bias,
                        folded_v_bias,
                    ],
                    dim=0,
                )
            )

            dst.norm1.load_state_dict(src.norm1.state_dict())
            dst.norm2.load_state_dict(src.norm2.state_dict())
            dst.ffn_in.load_state_dict(src.ffn_in.state_dict())
            dst.ffn_out.load_state_dict(src.ffn_out.state_dict())

        model.final_norm.load_state_dict(baseline.final_norm.state_dict())
