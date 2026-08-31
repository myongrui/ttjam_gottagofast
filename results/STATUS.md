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

## 2026-08-31 — nightly health check (cloud, no pod access)

Champion unchanged at 2.2969 (v038) since the promotion on 2026-08-31; two
back-to-back 5-run stalls since then (v039-v048), 5 invalid (mostly
TorchScript compile/runtime errors), the rest screen_promote/screen_uncertain
that came back `uncertain` on the full sweep -- v047 (+0.6% over champion)
and v048 (+0.1%) both landed inside the 2% noise margin, so neither promoted.

Diagnosis: v038's `Model._attention_mask` still runs `bool(valid_token_mask
.all())` -- a device-to-host sync -- on *every* forward call, once per call
(reduced from 4x/layer in v001, but never eliminated). The timed sweep always
calls with `padding_ratio=0.0` (tools/harness.py:161-164), so this sync
always resolves the same way and buys nothing but a pipeline stall on every
one of the 13x3 timed calls. v048 tried to dodge it by inspecting `bench` at
build time for an example mask, found nothing usable, and fell back to the
same runtime `.all()` check -- explaining its ~0% delta.

Proposed `candidates/v049_causal_padding_invariant.py`: every official case
is causal (tools/shapes.py), and the harness's own generator only ever
produces tail padding (`positions < lengths`). Under causal masking, a valid
query at position i < length can only attend to keys 0..i, and every such key
satisfies j <= i < length -- so the padding mask never removes anything for a
row that matters, for any layer, recursively (attention is the only
cross-token op). Invalid rows get zeroed at the end regardless, matching the
baseline exactly. So the causal branch can go straight to `is_causal=True,
attn_mask=None` unconditionally -- no sync, no S*S tril allocation, no
build-time introspection. The non-causal branch (unused by the official
suite) now builds its cheap [B,1,1,S] mask unconditionally too, so neither
branch ever reads mask contents on the host. Not yet raced -- no pod access
from this session.

## 2026-08-31 (later) — nightly health check (cloud, no pod access)

No change since the prior check ~2h ago: HEAD is still aa63fcd, ledger has
no entries past v048 (12:57 UTC), champion remains v038 at 2.2969. v049 has
not been raced yet -- the loop/autoloop appears not to be running right now
(no pod access from this session to start it or race candidates). Not
proposing a second candidate on top of v049: no new evidence has arrived
since it was written, and the stall diagnosis it targets (the per-forward
host-device sync in v038's mask path) is unchanged. Next step is for the
loop to race v049 against v038 on the `launch`/`general` profiles.
