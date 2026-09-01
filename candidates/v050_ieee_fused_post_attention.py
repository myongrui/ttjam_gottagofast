"""IEEE-accurate fused post-attention case-2 candidate.

Hypothesis: v039's fused post-attention kernel missed the float32 correctness
gate only because Triton's default TF32 dot products accumulated differently
from the baseline. Forcing IEEE input precision in its two fused dot products
should pass all five case-2 distributions while retaining enough launch
elimination to beat v038 by more than the 2% paired-confidence margin.
"""
import importlib.util
import os
import sys

import triton
import triton.language as tl


def _load_parent():
    filename = "v039_fused_post_attention_transformer.py"
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    name = "ieee_fused_parent_v039"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@triton.jit
def _post_attention_fused_ieee_kernel(
    context,
    residual,
    out_weight,
    out_bias,
    norm_weight,
    norm_bias,
    ffn_weight,
    ffn_bias,
    residual_out,
    hidden_out,
    n_rows: tl.constexpr,
    seq_len: tl.constexpr,
    context_stride_b: tl.constexpr,
    context_stride_h: tl.constexpr,
    context_stride_s: tl.constexpr,
    context_stride_d: tl.constexpr,
    d_model: tl.constexpr,
    head_dim: tl.constexpr,
    ffn_dim: tl.constexpr,
    eps: tl.constexpr,
    block_m: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * block_m + tl.arange(0, block_m)
    d = tl.arange(0, d_model)
    f = tl.arange(0, ffn_dim)
    row_mask = rows < n_rows

    batch_index = rows // seq_len
    sequence_index = rows - batch_index * seq_len
    head_index = d // head_dim
    head_offset = d - head_index * head_dim

    context_offsets = (
        batch_index[:, None] * context_stride_b
        + head_index[None, :] * context_stride_h
        + sequence_index[:, None] * context_stride_s
        + head_offset[None, :] * context_stride_d
    )
    context_values = tl.load(
        context + context_offsets,
        mask=row_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    weight_offsets = d[:, None] * d_model + d[None, :]
    output_weight_t = tl.trans(
        tl.load(out_weight + weight_offsets).to(tl.float32)
    )
    projected = tl.dot(
        context_values,
        output_weight_t,
        input_precision="ieee",
    )

    residual_offsets = rows[:, None] * d_model + d[None, :]
    residual_values = tl.load(
        residual + residual_offsets,
        mask=row_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    projected += residual_values
    projected += tl.load(out_bias + d)[None, :].to(tl.float32)

    tl.store(
        residual_out + residual_offsets,
        projected,
        mask=row_mask[:, None],
    )

    mean = tl.sum(projected, axis=1) / d_model
    centered = projected - mean[:, None]
    variance = tl.sum(centered * centered, axis=1) / d_model
    normalized = centered * tl.rsqrt(variance[:, None] + eps)
    normalized = (
        normalized
        * tl.load(norm_weight + d)[None, :].to(tl.float32)
        + tl.load(norm_bias + d)[None, :].to(tl.float32)
    )

    ffn_weight_offsets = f[:, None] * d_model + d[None, :]
    ffn_weight_t = tl.trans(
        tl.load(ffn_weight + ffn_weight_offsets).to(tl.float32)
    )
    pre_activation = tl.dot(
        normalized,
        ffn_weight_t,
        input_precision="ieee",
    )
    pre_activation += tl.load(ffn_bias + f)[None, :].to(tl.float32)

    hidden = 0.5 * pre_activation * (
        1.0 + tl.erf(pre_activation * 0.7071067811865475244)
    )
    hidden_offsets = rows[:, None] * ffn_dim + f[None, :]
    tl.store(
        hidden_out + hidden_offsets,
        hidden,
        mask=row_mask[:, None],
    )


_PARENT = _load_parent()
_PARENT._post_attention_fused_kernel = _post_attention_fused_ieee_kernel


def build_model(config, bench):
    return _PARENT.build_model(config, bench)


def load_from_baseline(model, baseline):
    _PARENT.load_from_baseline(model, baseline)
