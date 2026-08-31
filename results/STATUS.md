# autoloop stopped

- **reason**: 5 consecutive non-improvements -- the search has stalled and needs a structural idea
- **iterations run**: 6
- **best full-sweep score**: 2.2969
- **stopped at**: 2026-08-31 05:25:59Z
- **last routing mode**: STRUCTURAL — 13/13 shapes >4.0x off SOL
- **profile**: general   **parent**: v038_torchscript_dispatched_transformer_candidate.py
- **beam/elite width**: 3/2
- **screen-to-full contradiction rate**: 20% (1 of 5) — offenders: v007_fuse_every_residual_add.py

## Recent attempts

| candidate | score | decision | hypothesis |
|---|---:|---|---|
| v040_frozen_torchscript_transformer_candidate.py | 2.4975 | screen_uncertain | Frozen TorchScript transformer candidate. |
| v040_frozen_torchscript_transformer_candidate.py | 2.3261 | uncertain | Frozen TorchScript transformer candidate. |
| v041_single_token_attention_specialization.py | 2.5818 | screen_reject | Single-token attention specialization. |
| v041_single_token_attention_specialization.py | 2.2982 | uncertain | Single-token attention specialization. |
| v042_torchscript_dispatched_native_encoder.py | 0.0000 | invalid | TorchScript-dispatched native encoder-layer candidate. |
| v043_shape_dispatched_native_encoder.py | 2.3280 | screen_reject | Shape-dispatched native-encoder transformer. |

## Resume

```
autoloop stopped: 5 consecutive non-improvements -- the search has stalled and needs a structural idea. Best 2.2969 after 6 iterations. Read results/STATUS.md, results/ledger.jsonl and LOOP.md, diagnose the stall, and propose the next candidate.
```
