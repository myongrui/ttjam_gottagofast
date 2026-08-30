"""The 14 official test shapes from the problem-statement appendix.

All cases are causal. Note the scale spread: most are tiny (d_model=128,
ffn=128, head_dim=32), which makes them launch-overhead and bandwidth bound
rather than FLOP bound. Case 14 is the outlier by four orders of magnitude.
"""

# (id, batch, d_model, heads, seq_len, layers, ffn_dim)
OFFICIAL = [
    (1,     64,  128,  4,    128, 4,  128),
    (2,      1,  128,  4,    128, 4,  128),
    (3,      4,  128,  4,    128, 4,  128),
    (4,     16,  128,  4,    128, 4,  128),
    (5,    128,  128,  4,    128, 4,  128),
    (6,  10000,  128,  4,    128, 4,  128),
    (7,     64,   32,  4,    128, 4,   32),
    (8,     64, 1024,  4,    128, 4, 1024),
    (9,     64,  128,  1,    128, 4,  128),
    (10,    64,  128,  2,    128, 4,  128),
    (11,    64,  128, 16,    128, 4,  128),
    (12,    64,  128,  4,     32, 4,  128),
    (13,    64,  128,  4,   1024, 4,  128),
    (14,    32, 1024, 16, 100000, 2, 1024),
]

FIELDS = ("case", "batch_size", "d_model", "num_heads", "seq_len", "num_layers", "ffn_dim")


def as_dicts(only=None):
    out = []
    for row in OFFICIAL:
        d = dict(zip(FIELDS, row))
        if only is None or d["case"] in only:
            out.append(d)
    return out


def activation_bytes(d, dtype_bytes=2):
    """Bytes for one [B, S, D] activation tensor -- the cheap OOM pre-screen."""
    return d["batch_size"] * d["seq_len"] * d["d_model"] * dtype_bytes


def baseline_score_bytes(d):
    """Bytes the baseline's materialized [B, H, S, S] fp32 score matrix needs.

    The baseline computes softmax in fp32 over an explicit S x S matrix per head,
    so this grows quadratically in seq_len and is the binding constraint.
    """
    return d["batch_size"] * d["num_heads"] * d["seq_len"] * d["seq_len"] * 4
