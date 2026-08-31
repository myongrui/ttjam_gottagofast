"""Synchronization-free memory-efficient-attention transformer.

Hypothesis: routing masked attention directly through PyTorch's fused efficient
attention operator eliminates the host synchronization used to test whether a
CUDA validity mask is all true. A broadcast key-padding bias and the operator's
native causal flag also avoid materializing a dense causal-plus-padding mask,
while retaining TorchScript's elimination of Python dispatch between layers.
"""
from typing import Optional

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

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: Optional[torch.Tensor],
        causal: bool,
    ) -> torch.Tensor:
        b, s, _ = x.shape
        qkv = self.qkv(x).view(
            b, s, 3, self.num_heads, self.head_dim
        )
        q = qkv[:, :, 0].transpose(1, 2)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)

        if attn_bias is None:
            ctx = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                is_causal=causal,
            )
        else:
            result = torch.ops.aten._scaled_dot_product_efficient_attention(
                q,
                k,
                v,
                attn_bias,
                False,
                0.0,
                causal,
            )
            ctx = result[0]

        return ctx.transpose(1, 2).reshape(b, s, self.d_model)


class Block(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = FusedAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_in = nn.Linear(d_model, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, d_model)

    @staticmethod
    def _residual_projection(
        residual: torch.Tensor,
        projected_input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        b, s, d = residual.shape
        c = residual.reshape(b * s, d) + bias
        return torch.addmm(
            c,
            projected_input.reshape(
                b * s, projected_input.shape[-1]
            ),
            weight.t(),
        ).view(b, s, d)

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: Optional[torch.Tensor],
        causal: bool,
    ) -> torch.Tensor:
        ctx = self.attention(self.norm1(x), attn_bias, causal)
        x = self._residual_projection(
            x,
            ctx,
            self.attention.out_proj.weight,
            self.attention.out_proj.bias,
        )
        hidden = F.gelu(
            self.ffn_in(self.norm2(x)),
            approximate="none",
        )
        return self._residual_projection(
            x,
            hidden,
            self.ffn_out.weight,
            self.ffn_out.bias,
        )


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.causal = bool(config.causal)
        self.num_heads = int(config.num_heads)
        self.layers = nn.ModuleList(
            [
                Block(
                    config.d_model,
                    config.num_heads,
                    config.ffn_dim,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)

    def _attention_bias(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if valid_token_mask is None:
            return None

        b = x.shape[0]
        s = x.shape[1]
        padding_bias = torch.where(
            valid_token_mask,
            0.0,
            float("-inf"),
        ).to(dtype=x.dtype)

        return padding_bias[:, None, None, :].expand(
            b,
            self.num_heads,
            s,
            s,
        )

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_bias = self._attention_bias(x, valid_token_mask)

        for layer in self.layers:
            x = layer(x, attn_bias, self.causal)

        x = self.final_norm(x)
        if valid_token_mask is not None:
            x = x.masked_fill(
                ~valid_token_mask[..., None],
                0.0,
            )
        return x


def build_model(config, bench) -> nn.Module:
    return torch.jit.script(Model(config))


def load_from_baseline(model, baseline) -> None:
    with torch.no_grad():
        state = model.state_dict()
        for index, src in enumerate(baseline.layers):
            prefix = "layers." + str(index) + "."
            attention = src.attention

            state[prefix + "attention.qkv.weight"].copy_(
                torch.cat(
                    (
                        attention.q_proj.weight,
                        attention.k_proj.weight,
                        attention.v_proj.weight,
                    ),
                    dim=0,
                )
            )
            state[prefix + "attention.qkv.bias"].copy_(
                torch.cat(
                    (
                        attention.q_proj.bias,
                        attention.k_proj.bias,
                        attention.v_proj.bias,
                    ),
                    dim=0,
                )
            )
            state[prefix + "attention.out_proj.weight"].copy_(
                attention.out_proj.weight
            )
            state[prefix + "attention.out_proj.bias"].copy_(
                attention.out_proj.bias
            )
            state[prefix + "norm1.weight"].copy_(src.norm1.weight)
            state[prefix + "norm1.bias"].copy_(src.norm1.bias)
            state[prefix + "norm2.weight"].copy_(src.norm2.weight)
            state[prefix + "norm2.bias"].copy_(src.norm2.bias)
            state[prefix + "ffn_in.weight"].copy_(src.ffn_in.weight)
            state[prefix + "ffn_in.bias"].copy_(src.ffn_in.bias)
            state[prefix + "ffn_out.weight"].copy_(src.ffn_out.weight)
            state[prefix + "ffn_out.bias"].copy_(src.ffn_out.bias)

        state["final_norm.weight"].copy_(baseline.final_norm.weight)
        state["final_norm.bias"].copy_(baseline.final_norm.bias)
