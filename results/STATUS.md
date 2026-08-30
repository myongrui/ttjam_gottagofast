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
