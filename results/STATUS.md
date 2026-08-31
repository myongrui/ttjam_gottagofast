# autoloop stopped

- **reason**: 5 consecutive non-improvements -- the search has stalled and needs a structural idea
- **iterations run**: 5
- **best full-sweep score**: 2.2969
- **stopped at**: 2026-08-31 12:59:38Z
- **last routing mode**: STRUCTURAL — 13/13 shapes >4.0x off SOL; 5 runs without promotion
- **profile**: launch   **parent**: v038_torchscript_dispatched_transformer_candidate.py
- **beam/elite width**: 3/2
- **screen-to-full contradiction rate**: 33% (2 of 6) — offenders: v007_fuse_every_residual_add.py, v047_homogeneous_coordinate_ffn_bias.py

## Recent attempts

| candidate | score | decision | hypothesis |
|---|---:|---|---|
| v046_synchronization_free_memory_efficient.py | 2.6044 | screen_reject | Synchronization-free memory-efficient-attention transformer. |
| v046_synchronization_free_memory_efficient.py | 2.4569 | uncertain | Synchronization-free memory-efficient-attention transformer. |
| v047_homogeneous_coordinate_ffn_bias.py | 2.6064 | screen_promote | Homogeneous-coordinate FFN bias candidate. |
| v047_homogeneous_coordinate_ffn_bias.py | 2.4653 | uncertain | Homogeneous-coordinate FFN bias candidate. |
| v048_build_time_attention_mask.py | 2.5383 | screen_uncertain | Build-time attention-mask specialization. |
| v048_build_time_attention_mask.py | 2.4554 | uncertain | Build-time attention-mask specialization. |

## Resume

```
autoloop stopped: 5 consecutive non-improvements -- the search has stalled and needs a structural idea. Best 2.2969 after 5 iterations. Read results/STATUS.md, results/ledger.jsonl and LOOP.md, diagnose the stall, and propose the next candidate.
```
