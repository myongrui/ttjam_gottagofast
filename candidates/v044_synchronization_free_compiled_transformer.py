"""Synchronization-free compiled transformer candidate.

Hypothesis: removing the data-dependent host read of valid_token_mask allows
torch.compile's reduce-overhead mode to capture each mask specialization as one
uninterrupted graph. Inductor can then fuse mask construction and adjacent
pointwise residual, normalization, and GELU operations while CUDA graphs remove
repeated launch and Python-dispatch overhead on latency-bound short shapes.
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
        attn_mask: Optional[torch.Tensor],
        causal: bool,
    ) -> torch.Tensor:
        b, s, _ = x.shape
        qkv = self.qkv(x).view(
            b, s, 3, self.num_heads, self.head_dim
        )
        q = qkv[:, :, 0].transpose(1, 2)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)

        context = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=causal and attn_mask is None,
        )
        return context.transpose(1, 2).reshape(b, s, self.d_model)


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
        return torch.addmm(
            residual.reshape(b * s, d) + bias,
            projected_input.reshape(b * s, projected_input.shape[-1]),
            weight.t(),
        ).view(b, s, d)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        causal: bool,
    ) -> torch.Tensor:
        context = self.attention(self.norm1(x), attn_mask, causal)
        x = self._residual_projection(
            x,
            context,
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

    def _attention_mask(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if valid_token_mask is None:
            return None

        keep = valid_token_mask[:, None, None, :]
        if self.causal:
            sequence_length = x.shape[1]
            causal_keep = torch.ones(
                (sequence_length, sequence_length),
                device=x.device,
                dtype=torch.bool,
            ).tril()
            keep = keep & causal_keep

        return torch.zeros(
            keep.shape,
            device=x.device,
            dtype=x.dtype,
        ).masked_fill(~keep, float("-inf"))

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_mask = self._attention_mask(x, valid_token_mask)

        for layer in self.layers:
            x = layer(x, attn_mask, self.causal)

        x = self.final_norm(x)
        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0.0)
        return x


def build_model(config, bench) -> nn.Module:
    model = Model(config)
    return torch.compile(
        model,
        mode="reduce-overhead",
        fullgraph=True,
        dynamic=False,
    )


def load_from_baseline(model, baseline) -> None:
    target = model._orig_mod
    with torch.no_grad():
        for dst, src in zip(target.layers, baseline.layers):
            attention = src.attention

            dst.attention.qkv.weight.copy_(
                torch.cat(
                    (
                        attention.q_proj.weight,
                        attention.k_proj.weight,
                        attention.v_proj.weight,
                    ),
                    dim=0,
                )
            )
            dst.attention.qkv.bias.copy_(
                torch.cat(
                    (
                        attention.q_proj.bias,
                        attention.k_proj.bias,
                        attention.v_proj.bias,
                    ),
                    dim=0,
                )
            )
            dst.attention.out_proj.weight.copy_(
                attention.out_proj.weight
            )
            dst.attention.out_proj.bias.copy_(
                attention.out_proj.bias
            )
            dst.norm1.weight.copy_(src.norm1.weight)
            dst.norm1.bias.copy_(src.norm1.bias)
            dst.norm2.weight.copy_(src.norm2.weight)
            dst.norm2.bias.copy_(src.norm2.bias)
            dst.ffn_in.weight.copy_(src.ffn_in.weight)
            dst.ffn_in.bias.copy_(src.ffn_in.bias)
            dst.ffn_out.weight.copy_(src.ffn_out.weight)
            dst.ffn_out.bias.copy_(src.ffn_out.bias)

        target.final_norm.weight.copy_(baseline.final_norm.weight)
        target.final_norm.bias.copy_(baseline.final_norm.bias)
