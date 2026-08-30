# autoloop stopped

- **reason**: 5 consecutive non-improvements -- the search has stalled and needs a structural idea
- **iterations run**: 5
- **best full-sweep score**: 1.6982
- **stopped at**: 2026-08-30 16:02:59Z
- **last routing mode**: STRUCTURAL — 13/13 shapes >4.0x off SOL
- **profile**: launch   **parent**: v001_sdpa_fused_qkv.py
- **beam/elite width**: 2/1
- **screen-to-full contradiction rate**: 0% (0 of 0)

## Recent attempts

| candidate | score | decision | hypothesis |
|---|---:|---|---|
| v007_fuse_every_residual_add.py | 2.0944 | screen_promote | v007 -- fuse every residual add with the LayerNorm that immediately co |
| v008_fuse_the_ffn_input.py | 0.0000 | invalid | v008 -- fuse the FFN input projection's exact GELU epilogue. |
| v009_native_transformerencoderlayer_fast_path.py | 1.4052 | screen_reject | Native TransformerEncoderLayer fast-path candidate. |
| v010_hypothesis_the_attention_output.py | 0.0000 | invalid | Hypothesis: the attention output projection need not materialize the e |
| v011_candidate.py | 1.2377 | screen_reject | Hypothesis: pipeline adjacent residual-add and LayerNorm operations ac |
| v012_hypothesis_fuse_each_layernorm.py | 0.0000 | invalid | Hypothesis: fuse each LayerNorm directly into the following projection |

## Resume

```
autoloop stopped: 5 consecutive non-improvements -- the search has stalled and needs a structural idea. Best 1.6982 after 5 iterations. Read results/STATUS.md, results/ledger.jsonl and LOOP.md, diagnose the stall, and propose the next candidate.
```

## 2026-08-30 nightly health check (Claude, no pod access)

Champion is still v001 at 1.6982, unchanged since the initial sweep. The
5-iteration stall autoloop hit (v008-v012, all screen_reject or invalid) is
real, but it is not the interesting finding: **v007 screen_promote'd at
2.0944 geomean on the general profile (2026-08-30T15:53:25) and was never
carried to a 13-case full sweep.** It sits in the per-shape archive as the
proven elite for cases 6/8/11/13 (2.07x/1.13x/2.74x/4.77x) but never
displaced v001 as global champion because autoloop restarted numbering at
iteration 1 (v008) right after a KeyboardInterrupt, without following up on
the pending promotion. Per POLICIES.md, subset wins never auto-promote, but
nothing in the ledger shows the required full sweep was ever attempted --
that's a process gap, not a search-quality one.

v007's one weakness: it loses to v001 on case 2, the smallest/most
launch-bound shape (paired speedup 0.819, an 18% regression). Root cause:
v007 kept v001's `FusedAttention` verbatim, including (a) the
`bool(valid_token_mask.all())` host sync repeated once per layer, and (b) an
`out.masked_fill` that is dead work under v007 because
`fused_residual_layernorm` already re-zeroes invalid rows in the same kernel
that consumes `out`. Both cost a kernel launch/sync per attention call on
every forward, which matters most exactly on case 2's tiny launch-bound
shape. v005 already tried removing the sync by always taking SDPA's boolean
mask path and was correctly refuted (1.4457, screen_reject) because that
gives up the flash fast path even when there's no padding -- so that
mechanism, in that form, is dead.

Proposed `candidates/v013_hoist_padding_decision.py`: takes v007's proven
Triton fused residual+LayerNorm unchanged, and hoists the padding decision
(`has_padding`, and the additive mask when it's true) to run once per
`Model.forward` instead of once per layer, preserving the `is_causal` flash
path for the common no-padding case (unlike v005) while cutting host syncs
from up to 4/forward to 1, and dropping the now-redundant `masked_fill`
inside `FusedAttention`. Suggest racing on the `general` profile
(2,6,8,11,13) to confirm it recovers case 2 without disturbing v007's wins
on 6/8/11/13, then a full 13-case sweep -- this is a live pending-promotion
question, not a proposal that needs the win re-derived from scratch.

No candidate was run; no benchmark was executed. Static review only.
