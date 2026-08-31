"""Native encoder fast-path candidate.

Hypothesis: for fully unmasked, non-causal encoder workloads, replacing the
Python-level sequence of LayerNorm, QKV projection, SDPA, output projection,
residual, and FFN operations with aten._transformer_encoder_layer_fwd removes
multiple Python dispatches and kernel launches per layer.  Masked and causal
inputs retain the established SDPA implementation so padded-key and causal
semantics remain exact.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x, attn_mask, causal):
        b, s, _ = x.shape
        qkv = self.qkv(x).view(b, s, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            is_causal=causal and attn_mask is None,
        )
        return y.transpose(1, 2).reshape(b, s, self.d_model)


class Block(nn.Module):
    def __init__(self, d_model, num_heads, ffn_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = Attention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_in = nn.Linear(d_model, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, d_model)

    @staticmethod
    def residual_linear(residual, x, linear):
        b, s, d = residual.shape
        c = residual.reshape(b * s, d) + linear.bias
        return torch.addmm(
            c, x.reshape(b * s, x.shape[-1]), linear.weight.t()
        ).view(b, s, d)

    def forward(self, x, attn_mask, causal):
        a = self.attention(self.norm1(x), attn_mask, causal)
        x = self.residual_linear(x, a, self.attention.out_proj)
        h = F.gelu(self.ffn_in(self.norm2(x)), approximate="none")
        return self.residual_linear(x, h, self.ffn_out)


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            Block(config.d_model, config.num_heads, config.ffn_dim)
            for _ in range(config.num_layers)
        ])
        self.final_norm = nn.LayerNorm(config.d_model)

    def _mask(self, x, valid_token_mask):
        if valid_token_mask is None or bool(valid_token_mask.all()):
            return None
        s = x.shape[1]
        keep = valid_token_mask[:, None, None, :]
        if self.config.causal:
            keep = keep & torch.ones(
                (s, s), device=x.device, dtype=torch.bool
            ).tril()
        mask = torch.zeros(keep.shape, device=x.device, dtype=x.dtype)
        return mask.masked_fill(~keep, float("-inf"))

    @staticmethod
    def native_layer(x, layer):
        return torch._transformer_encoder_layer_fwd(
            x,
            layer.attention.d_model,
            layer.attention.num_heads,
            layer.attention.qkv.weight,
            layer.attention.qkv.bias,
            layer.attention.out_proj.weight,
            layer.attention.out_proj.bias,
            True,
            True,
            layer.norm1.eps,
            layer.norm1.weight,
            layer.norm1.bias,
            layer.norm2.weight,
            layer.norm2.bias,
            layer.ffn_in.weight,
            layer.ffn_in.bias,
            layer.ffn_out.weight,
            layer.ffn_out.bias,
            None,
            None,
        )

    def forward(self, x, valid_token_mask=None):
        # The fused native operator has no causal flag.  Restrict it to the
        # mathematically unmasked encoder case rather than materializing an
        # explicit causal score mask and losing the flash-attention route.
        if valid_token_mask is None and not self.config.causal:
            for layer in self.layers:
                x = self.native_layer(x, layer)
            return self.final_norm(x)

        attn_mask = self._mask(x, valid_token_mask)
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
            dst.attention.qkv.weight.copy_(torch.cat([
                a.q_proj.weight, a.k_proj.weight, a.v_proj.weight
            ], dim=0))
            dst.attention.qkv.bias.copy_(torch.cat([
                a.q_proj.bias, a.k_proj.bias, a.v_proj.bias
            ], dim=0))
            dst.attention.out_proj.load_state_dict(a.out_proj.state_dict())
            dst.norm1.load_state_dict(src.norm1.state_dict())
            dst.norm2.load_state_dict(src.norm2.state_dict())
            dst.ffn_in.load_state_dict(src.ffn_in.state_dict())
            dst.ffn_out.load_state_dict(src.ffn_out.state_dict())
        model.final_norm.load_state_dict(baseline.final_norm.state_dict())
