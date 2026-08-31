"""TorchScript-dispatched native encoder-layer candidate.

Hypothesis: replacing each launch-heavy sequence of normalization, attention,
residual, GELU, and FFN operators with PyTorch's native fused encoder-layer
operator reduces framework dispatch and intermediate-kernel launches on short
sequences. Long sequences retain SDPA so the implementation remains feasible
when materializing an attention score matrix would be prohibitive.
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.in_proj_weight = nn.Parameter(
            torch.empty(3 * d_model, d_model)
        )
        self.in_proj_bias = nn.Parameter(torch.empty(3 * d_model))
        self.out_proj_weight = nn.Parameter(
            torch.empty(d_model, d_model)
        )
        self.out_proj_bias = nn.Parameter(torch.empty(d_model))

        self.norm1 = nn.LayerNorm(d_model)
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

    def _sdpa_forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        mask_type: Optional[int],
        causal: bool,
    ) -> torch.Tensor:
        b, s, _ = x.shape
        normalized = self.norm1(x)
        qkv = F.linear(
            normalized,
            self.in_proj_weight,
            self.in_proj_bias,
        ).view(b, s, 3, self.num_heads, self.head_dim)

        q = qkv[:, :, 0].transpose(1, 2)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)

        sdpa_mask = attn_mask
        if attn_mask is not None and mask_type == 1:
            sdpa_mask = attn_mask[:, None, None, :]

        context = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=sdpa_mask,
            is_causal=causal and sdpa_mask is None,
        )
        context = context.transpose(1, 2).reshape(
            b, s, self.d_model
        )

        x = self._residual_projection(
            x,
            context,
            self.out_proj_weight,
            self.out_proj_bias,
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

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        mask_type: Optional[int],
        causal: bool,
        use_native: bool,
    ) -> torch.Tensor:
        if use_native:
            return torch._transformer_encoder_layer_fwd(
                x,
                self.d_model,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.out_proj_weight,
                self.out_proj_bias,
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
                attn_mask,
                mask_type,
            )

        return self._sdpa_forward(
            x,
            attn_mask,
            mask_type,
            causal,
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

    def _attention_mask(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        use_native: bool,
    ) -> Tuple[Optional[torch.Tensor], Optional[int]]:
        all_valid = valid_token_mask is None
        if valid_token_mask is not None:
            all_valid = bool(valid_token_mask.all())

        if not use_native:
            if all_valid:
                return None, None

            keep = valid_token_mask[:, None, None, :]
            if self.causal:
                s = x.shape[1]
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
            ).masked_fill(~keep, float("-inf"))
            return mask, 2 if self.causal else 1

        s = x.shape[1]
        if self.causal and s > 1:
            bad = torch.ones(
                (s, s),
                device=x.device,
                dtype=torch.bool,
            ).triu(1)
            bad = bad[None, None, :, :]

            if not all_valid:
                bad = bad | (
                    ~valid_token_mask[:, None, None, :]
                )

            bad = bad.expand(
                x.shape[0],
                self.num_heads,
                s,
                s,
            )
            mask = torch.zeros(
                bad.shape,
                device=x.device,
                dtype=x.dtype,
            ).masked_fill(bad, float("-inf"))
            return mask, 2

        if not all_valid:
            bad = ~valid_token_mask
            mask = torch.zeros(
                bad.shape,
                device=x.device,
                dtype=x.dtype,
            ).masked_fill(bad, float("-inf"))
            return mask, 1

        return None, None

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        use_native = x.shape[1] <= 256
        attn_mask, mask_type = self._attention_mask(
            x,
            valid_token_mask,
            use_native,
        )

        for layer in self.layers:
            x = layer(
                x,
                attn_mask,
                mask_type,
                self.causal,
                use_native,
            )

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

            state[prefix + "in_proj_weight"].copy_(
                torch.cat(
                    (
                        attention.q_proj.weight,
                        attention.k_proj.weight,
                        attention.v_proj.weight,
                    ),
                    dim=0,
                )
            )
            state[prefix + "in_proj_bias"].copy_(
                torch.cat(
                    (
                        attention.q_proj.bias,
                        attention.k_proj.bias,
                        attention.v_proj.bias,
                    ),
                    dim=0,
                )
            )
            state[prefix + "out_proj_weight"].copy_(
                attention.out_proj.weight
            )
            state[prefix + "out_proj_bias"].copy_(
                attention.out_proj.bias
            )
            state[prefix + "norm1.weight"].copy_(src.norm1.weight)
            state[prefix + "norm1.bias"].copy_(src.norm1.bias)
            state[prefix + "norm2.weight"].copy_(src.norm2.weight)
            state[prefix + "norm2.bias"].copy_(src.norm2.bias)
            state[prefix + "ffn_in.weight"].copy_(src.ffn_in.weight)
            state[prefix + "ffn_in.bias"].copy_(src.ffn_in.bias)
            state[prefix + "ffn_out.weight"].copy_(
                src.ffn_out.weight
            )
            state[prefix + "ffn_out.bias"].copy_(src.ffn_out.bias)

        state["final_norm.weight"].copy_(baseline.final_norm.weight)
        state["final_norm.bias"].copy_(baseline.final_norm.bias)
