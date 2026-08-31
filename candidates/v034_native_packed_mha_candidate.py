"""Native packed-MHA candidate.

Hypothesis: replacing the explicit QKV projection, tensor reshapes, SDPA call,
context reshape, and output projection with PyTorch's native packed
MultiheadAttention inference path eliminates several Python-visible operator
boundaries and intermediate tensor launches.  The native path performs packed
QKV projection, attention, and output projection as one structural attention
operation when masks permit it; existing fused residual-as-GEMM-C FFN paths are
retained for the remainder of each block.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NativeAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=0.0,
            bias=True,
            batch_first=True,
        )

    def forward(self, x, key_padding_mask=None, attn_mask=None):
        y, _ = self.mha(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            attn_mask=attn_mask,
        )
        return y


class Block(nn.Module):
    def __init__(self, d_model, num_heads, ffn_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = NativeAttention(d_model, num_heads)
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

    def forward(self, x, key_padding_mask, causal_mask):
        x = x + self.attention(self.norm1(x), key_padding_mask, causal_mask)
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

    def _mask_args(self, x, valid_token_mask):
        key_padding_mask = None
        if valid_token_mask is not None and not bool(valid_token_mask.all()):
            key_padding_mask = ~valid_token_mask

        causal_mask = None
        if self.config.causal:
            s = x.shape[1]
            causal_mask = torch.ones(
                (s, s), device=x.device, dtype=torch.bool
            ).triu_(1)
        return key_padding_mask, causal_mask

    def forward(self, x, valid_token_mask=None):
        key_padding_mask, causal_mask = self._mask_args(x, valid_token_mask)
        for layer in self.layers:
            x = layer(x, key_padding_mask, causal_mask)
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
            dst.attention.mha.in_proj_weight.copy_(
                torch.cat(
                    [a.q_proj.weight, a.k_proj.weight, a.v_proj.weight],
                    dim=0,
                )
            )
            dst.attention.mha.in_proj_bias.copy_(
                torch.cat(
                    [a.q_proj.bias, a.k_proj.bias, a.v_proj.bias],
                    dim=0,
                )
            )
            dst.attention.mha.out_proj.load_state_dict(a.out_proj.state_dict())
            dst.norm1.load_state_dict(src.norm1.state_dict())
            dst.norm2.load_state_dict(src.norm2.state_dict())
            dst.ffn_in.load_state_dict(src.ffn_in.state_dict())
            dst.ffn_out.load_state_dict(src.ffn_out.state_dict())
        model.final_norm.load_state_dict(baseline.final_norm.state_dict())
