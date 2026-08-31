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

    def forward(self, x, key_padding_mask=None, attn_mask=None, causal=False):
        if not causal:
            output, _ = torch._native_multi_head_attention(
                x,
                x,
                x,
                self.d_model,
                self.num_heads,
                self.qkv.weight,
                self.qkv.bias,
                self.out_proj.weight,
                self.out_proj.bias,
                key_padding_mask,
                False,
                True,
                1 if key_padding_mask is not None else None,
            )
            return output

        b, s, _ = x.shape
        qkv = self.qkv(x).view(b, s, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        ctx = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=attn_mask is None,
        )
        return self.out_proj(ctx.transpose(1, 2).reshape(b, s, self.d_model))


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

    def forward(self, x, key_padding_mask, attn_mask, causal):
        normalized = self.norm1(x)
        if causal:
            ctx = self.attention(normalized, None, attn_mask, True)
            x = self._residual_projection(x, ctx, self.attention.out_proj)
        else:
            x = x + self.attention(normalized, key_padding_mask, None, False)
        hidden = F.gelu(self.ffn_in(self.norm2(x)), approximate="none")
        return self._residual_projection(x, hidden, self.ffn_out)


class Model(nn.Module):
    """Hypothesis: route non-causal self-attention through PyTorch's native MHA
    operator while deciding padding mode once per invocation.  The native operator
    structurally replaces separate fused-QKV, SDPA, reshape, and output-projection
    dispatches with its dedicated inference path; padded non-causal inputs pass a
    compact [batch, seq] key-padding mask instead of a broadcast score mask.
    Causal inputs retain SDPA because native MHA has no equivalent causal fast path.
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

    def _masks(self, x, valid_token_mask):
        if valid_token_mask is None or bool(valid_token_mask.all()):
            return None, None

        if not self.config.causal:
            key_padding_mask = torch.zeros_like(valid_token_mask, dtype=x.dtype)
            key_padding_mask.masked_fill_(~valid_token_mask, float("-inf"))
            return key_padding_mask, None

        s = x.shape[1]
        keep = valid_token_mask[:, None, None, :]
        keep = keep & torch.ones(s, s, device=x.device, dtype=torch.bool).tril()
        attn_mask = torch.zeros(keep.shape, device=x.device, dtype=x.dtype)
        attn_mask.masked_fill_(~keep, float("-inf"))
        return None, attn_mask

    def forward(self, x, valid_token_mask=None):
        key_padding_mask, attn_mask = self._masks(x, valid_token_mask)
        for layer in self.layers:
            x = layer(x, key_padding_mask, attn_mask, self.config.causal)
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
