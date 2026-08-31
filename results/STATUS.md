# autoloop stopped

- **reason**: 5 consecutive non-improvements -- the search has stalled and needs a structural idea
- **iterations run**: 8
- **best full-sweep score**: 1.9141
- **stopped at**: 2026-08-31 02:53:15Z
- **last routing mode**: STRUCTURAL — 13/13 shapes >4.0x off SOL
- **profile**: launch   **parent**: v018_deferred_padding_mask_candidate.py
- **beam/elite width**: 2/1
- **screen-to-full contradiction rate**: 33% (1 of 3) — offenders: v007_fuse_every_residual_add.py

## Recent attempts

| candidate | score | decision | hypothesis |
|---|---:|---|---|
| v018_deferred_padding_mask_candidate.py | 1.9141 | promote | Deferred-padding-mask candidate. |
| v019_fused_final_normalization_candidate.py | 1.7406 | screen_reject | Fused-final-normalization candidate. |
| v020_boolean_mask_attention_candidate.py | 1.6500 | screen_reject | Boolean-mask attention candidate. |
| v021_short_sequence_mask_dispatch.py | 2.0761 | screen_reject | Short-sequence mask-dispatch candidate. |
| v022_fused_ffn_candidate.py | 0.0000 | invalid | Fused-FFN candidate. |
| v023_hoisted_attention_mask_candidate.py | 1.9455 | screen_promote | Hoisted-attention-mask candidate. |

## Resume

```
autoloop stopped: 5 consecutive non-improvements -- the search has stalled and needs a structural idea. Best 1.9141 after 8 iterations. Read results/STATUS.md, results/ledger.jsonl and LOOP.md, diagnose the stall, and propose the next candidate.
```
