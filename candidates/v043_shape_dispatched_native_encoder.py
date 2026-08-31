"""Shape-dispatched native-encoder transformer.

Hypothesis: latency-bound small-token shapes are faster when each transformer
block is submitted through PyTorch's native encoder-layer operator, which
collapses repeated Python/TorchScript operator dispatch and enables the native
fused MHA path. Larger shapes retain the established TorchScript SDPA path, so
the structural dispatch targets launch-bound cases without regressing long or
high-batch workloads.
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
        ctx = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=causal and attn_mask is None,
        )
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
        attn_mask: Optional[torch.Tensor],
        causal: bool,
    ) -> torch.Tensor:
        ctx = self.attention(self.norm1(x), attn_mask, causal)
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


class ScriptedCore(nn.Module):
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
        if bool(valid_token_mask.all()):
            return None

        s = x.shape[1]
        keep = valid_token_mask[:, None, None, :]
        if self.causal:
            causal_keep = torch.ones(
                (s, s),
                device=x.device,
                dtype=torch.bool,
            ).tril()
            keep = keep & causal_keep

        mask = torch.zeros(
            keep.shape,
            device=x.device,
            dtype=x.dtype,
        )
        return mask.masked_fill(~keep, float("-inf"))

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
            x = x.masked_fill(
                ~valid_token_mask[..., None],
                0.0,
            )
        return x


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = int(config.d_model)
        self.num_heads = int(config.num_heads)
        self.ffn_dim = int(config.ffn_dim)
        self.causal = bool(config.causal)
        self.small_token_limit = 512
        self.core = torch.jit.script(ScriptedCore(config))

    def _native_mask(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
    ):
        b, s, _ = x.shape

        if not self.causal:
            if valid_token_mask is None:
                return None, None
            mask = torch.zeros(
                valid_token_mask.shape,
                dtype=x.dtype,
                device=x.device,
            )
            mask = mask.masked_fill(
                ~valid_token_mask,
                float("-inf"),
            )
            return mask, 1

        positions = torch.arange(s, device=x.device)
        blocked = positions[None, :] > positions[:, None]
        blocked = blocked[None, None, :, :]

        if valid_token_mask is not None:
            blocked = blocked | (~valid_token_mask[:, None, None, :])

        blocked = blocked.expand(b, self.num_heads, s, s)
        mask = torch.zeros(
            blocked.shape,
            dtype=x.dtype,
            device=x.device,
        )
        mask = mask.masked_fill(blocked, float("-inf"))
        return mask, 2

    def _native_forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        mask, mask_type = self._native_mask(x, valid_token_mask)

        for layer in self.core.layers:
            x = torch._transformer_encoder_layer_fwd(
                x,
                self.d_model,
                self.num_heads,
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
                mask,
                mask_type,
            )

        x = self.core.final_norm(x)
        if valid_token_mask is not None:
            x = x.masked_fill(
                ~valid_token_mask[..., None],
                0.0,
            )
        return x

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        token_count = x.shape[0] * x.shape[1]
        if (
            token_count <= self.small_token_limit
            and self.d_model == 128
            and self.ffn_dim == 128
        ):
            return self._native_forward(x, valid_token_mask)
        return self.core(x, valid_token_mask)


def build_model(config, bench) -> nn.Module:
    return Model(config)


def load_from_baseline(model, baseline) -> None:
    with torch.no_grad():
        state = model.core.state_dict()
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
