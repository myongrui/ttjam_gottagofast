# autoloop stopped

- **reason**: 5 consecutive non-improvements -- the search has stalled and needs a structural idea
- **iterations run**: 5
- **best full-sweep score**: 2.0916
- **stopped at**: 2026-08-31 03:50:17Z
- **last routing mode**: STRUCTURAL — 13/13 shapes >4.0x off SOL
- **profile**: launch   **parent**: v024_centralized_attention_mask_dispatch.py
- **beam/elite width**: 3/2
- **screen-to-full contradiction rate**: 25% (1 of 4) — offenders: v007_fuse_every_residual_add.py

## Recent attempts

| candidate | score | decision | hypothesis |
|---|---:|---|---|
| v029_hypothesis_fusing_the_final.py | 1.9973 | screen_uncertain |  |
| v029_hypothesis_fusing_the_final.py | 2.0739 | uncertain |  |
| v030_native_multi_head_attention.py | 2.2870 | screen_uncertain | Native multi-head-attention fast-path candidate. |
| v030_native_multi_head_attention.py | 2.0716 | uncertain | Native multi-head-attention fast-path candidate. |
| v031_native_mha_unmasked_attention.py | 1.9509 | screen_reject | Native-MHA unmasked attention candidate. |
| v032_final_layernorm_plus_padding.py | 0.0000 | invalid |  |

## Resume

```
autoloop stopped: 5 consecutive non-improvements -- the search has stalled and needs a structural idea. Best 2.0916 after 5 iterations. Read results/STATUS.md, results/ledger.jsonl and LOOP.md, diagnose the stall, and propose the next candidate.
```
