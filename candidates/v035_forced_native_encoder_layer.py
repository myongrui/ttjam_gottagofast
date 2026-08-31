"""Forced native encoder-layer fast-path candidate.

Hypothesis: directly invoking PyTorch's fused native transformer encoder-layer
operator for fully-valid, non-causal inputs eliminates the separate LayerNorm,
QKV, attention-output, residual, FFN, and second-residual launch sequence for
each layer. Masked or causal inputs retain the proven SDPA implementation so
that padded-key exclusion and causal semantics remain exact.
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
        self.d_model = d_model
        self.num_heads = num_heads
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

    def _native_unmasked(self, x):
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

    def forward(self, x, attn_mask, causal, native_unmasked):
        if native_unmasked:
            return self._native_unmasked(x)

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
        native_unmasked = attn_mask is None and not self.config.causal

        for layer in self.layers:
            x = layer(x, attn_mask, self.config.causal, native_unmasked)

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
