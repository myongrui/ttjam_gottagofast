"""Self-contained version of the measured v059 runtime portfolio.

Hypothesis: inlining v059's effective TorchScript/SDPA fallback, regular CUDA
graph replay, and homogeneous-coordinate FFN graph paths preserves correctness
and reproduces its measured cases 1-13 performance within noise, while removing
all runtime imports of sibling candidate files. Static configuration dispatch
selects the same effective implementation that v059 selected for every official
shape, and baseline weights are loaded with the exact parent transformations.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


REGULAR_GRAPH_KEYS = {
    (64, 128, 4, 128, 4, 128, True),
    (16, 128, 4, 128, 4, 128, True),
    (64, 32, 4, 128, 4, 32, True),
    (64, 128, 4, 32, 4, 128, True),
}

HOMOGENEOUS_GRAPH_KEYS = {
    (1, 128, 4, 128, 4, 128, True),
    (4, 128, 4, 128, 4, 128, True),
}


def _config_key(config):
    return (
        config.batch_size,
        config.d_model,
        config.num_heads,
        config.seq_len,
        config.num_layers,
        config.ffn_dim,
        bool(config.causal),
    )


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
        batch_size, sequence_length, _ = x.shape
        qkv = self.qkv(x).view(
            batch_size,
            sequence_length,
            3,
            self.num_heads,
            self.head_dim,
        )
        query = qkv[:, :, 0].transpose(1, 2)
        key = qkv[:, :, 1].transpose(1, 2)
        value = qkv[:, :, 2].transpose(1, 2)
        context = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            is_causal=causal and attn_mask is None,
        )
        return context.transpose(1, 2).reshape(
            batch_size,
            sequence_length,
            self.d_model,
        )


class StandardBlock(nn.Module):
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
        batch_size, sequence_length, d_model = residual.shape
        combined = residual.reshape(batch_size * sequence_length, d_model) + bias
        return torch.addmm(
            combined,
            projected_input.reshape(
                batch_size * sequence_length,
                projected_input.shape[-1],
            ),
            weight.t(),
        ).view(batch_size, sequence_length, d_model)

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


class ScriptedModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.causal = bool(config.causal)
        self.layers = nn.ModuleList(
            [
                StandardBlock(
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

        sequence_length = x.shape[1]
        keep = valid_token_mask[:, None, None, :]
        if self.causal:
            causal_keep = torch.ones(
                (sequence_length, sequence_length),
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
            x = x.masked_fill(~valid_token_mask[..., None], 0.0)
        return x


class SynchronizationFreeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.causal = bool(config.causal)
        self.layers = nn.ModuleList(
            [
                StandardBlock(
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


class HomogeneousBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = FusedAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_in = nn.Linear(d_model, ffn_dim + 2, bias=True)
        self.ffn_out = nn.Linear(ffn_dim + 2, d_model, bias=False)

    @staticmethod
    def _biased_residual_projection(
        residual: torch.Tensor,
        projected_input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length, d_model = residual.shape
        combined = residual.reshape(batch_size * sequence_length, d_model) + bias
        return torch.addmm(
            combined,
            projected_input.reshape(
                batch_size * sequence_length,
                projected_input.shape[-1],
            ),
            weight.t(),
        ).view(batch_size, sequence_length, d_model)

    @staticmethod
    def _homogeneous_residual_projection(
        residual: torch.Tensor,
        projected_input: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length, d_model = residual.shape
        return torch.addmm(
            residual.reshape(batch_size * sequence_length, d_model),
            projected_input.reshape(
                batch_size * sequence_length,
                projected_input.shape[-1],
            ),
            weight.t(),
        ).view(batch_size, sequence_length, d_model)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        causal: bool,
    ) -> torch.Tensor:
        context = self.attention(self.norm1(x), attn_mask, causal)
        x = self._biased_residual_projection(
            x,
            context,
            self.attention.out_proj.weight,
            self.attention.out_proj.bias,
        )
        hidden = F.gelu(
            self.ffn_in(self.norm2(x)),
            approximate="none",
        )
        return self._homogeneous_residual_projection(
            x,
            hidden,
            self.ffn_out.weight,
        )


class SynchronizationFreeHomogeneousModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.causal = bool(config.causal)
        self.layers = nn.ModuleList(
            [
                HomogeneousBlock(
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


class GraphReplayModel(nn.Module):
    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner
        self._graph = None
        self._static_x = None
        self._static_mask = None
        self._static_output = None

    def _capture(self, x, valid_token_mask):
        self._static_x = torch.empty_like(x)
        self._static_mask = torch.empty_like(valid_token_mask)
        self._static_x.copy_(x)
        self._static_mask.copy_(valid_token_mask)

        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                self.inner(self._static_x, self._static_mask)
        torch.cuda.current_stream().wait_stream(warmup_stream)

        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._static_output = self.inner(
                self._static_x,
                self._static_mask,
            )

    def forward(self, x, valid_token_mask=None):
        if valid_token_mask is None:
            return self.inner(x, valid_token_mask)
        if self._graph is None:
            self._capture(x, valid_token_mask)
        self._static_x.copy_(x)
        self._static_mask.copy_(valid_token_mask)
        self._graph.replay()
        return self._static_output.clone()


class SelectedModel(nn.Module):
    def __init__(self, config, bench):
        super().__init__()
        key = _config_key(config)
        self.uses_homogeneous_graph = key in HOMOGENEOUS_GRAPH_KEYS
        self.uses_regular_graph = key in REGULAR_GRAPH_KEYS
        if self.uses_homogeneous_graph:
            self.inner = GraphReplayModel(
                SynchronizationFreeHomogeneousModel(config)
            )
        elif self.uses_regular_graph:
            self.inner = GraphReplayModel(SynchronizationFreeModel(config))
        else:
            self.inner = torch.jit.script(ScriptedModel(config))

    def forward(self, x, valid_token_mask=None):
        return self.inner(x, valid_token_mask)


def _load_standard_weights(model, baseline) -> None:
    with torch.no_grad():
        state = model.state_dict()
        for index, source in enumerate(baseline.layers):
            prefix = "layers." + str(index) + "."
            attention = source.attention
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
            state[prefix + "norm1.weight"].copy_(source.norm1.weight)
            state[prefix + "norm1.bias"].copy_(source.norm1.bias)
            state[prefix + "norm2.weight"].copy_(source.norm2.weight)
            state[prefix + "norm2.bias"].copy_(source.norm2.bias)
            state[prefix + "ffn_in.weight"].copy_(source.ffn_in.weight)
            state[prefix + "ffn_in.bias"].copy_(source.ffn_in.bias)
            state[prefix + "ffn_out.weight"].copy_(source.ffn_out.weight)
            state[prefix + "ffn_out.bias"].copy_(source.ffn_out.bias)

        state["final_norm.weight"].copy_(baseline.final_norm.weight)
        state["final_norm.bias"].copy_(baseline.final_norm.bias)


def _load_homogeneous_weights(model, baseline) -> None:
    with torch.no_grad():
        state = model.state_dict()
        for index, source in enumerate(baseline.layers):
            prefix = "layers." + str(index) + "."
            attention = source.attention
            ffn_dim = source.ffn_in.weight.shape[0]
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
            state[prefix + "norm1.weight"].copy_(source.norm1.weight)
            state[prefix + "norm1.bias"].copy_(source.norm1.bias)
            state[prefix + "norm2.weight"].copy_(source.norm2.weight)
            state[prefix + "norm2.bias"].copy_(source.norm2.bias)

            ffn_in_weight = state[prefix + "ffn_in.weight"]
            ffn_in_bias = state[prefix + "ffn_in.bias"]
            ffn_in_weight[:ffn_dim].copy_(source.ffn_in.weight)
            ffn_in_bias[:ffn_dim].copy_(source.ffn_in.bias)
            ffn_in_weight[ffn_dim:].zero_()
            ffn_in_bias[ffn_dim:].fill_(8.0)

            output_weight = state[prefix + "ffn_out.weight"]
            output_weight[:, :ffn_dim].copy_(source.ffn_out.weight)
            bias = source.ffn_out.bias
            high = bias.to(torch.float16).to(bias.dtype)
            low = bias - high
            output_weight[:, ffn_dim].copy_(high * 0.125)
            output_weight[:, ffn_dim + 1].copy_(low * 0.125)

        state["final_norm.weight"].copy_(baseline.final_norm.weight)
        state["final_norm.bias"].copy_(baseline.final_norm.bias)


def build_model(config, bench):
    return SelectedModel(config, bench)


def load_from_baseline(model, baseline):
    target = model.inner.inner if (
        model.uses_homogeneous_graph or model.uses_regular_graph
    ) else model.inner
    if model.uses_homogeneous_graph:
        _load_homogeneous_weights(target, baseline)
    else:
        _load_standard_weights(target, baseline)
