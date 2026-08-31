# Current best submission

_Regenerated 2026-08-31 05:08:21Z from commit `277f7b7`. Do not edit — `tools/report.py` overwrites this file._


| | |
|---|---|
| **Composition** | 1 generalist + 1 specialist(s) |
| **Generalist (fallback)** | `v038_torchscript_dispatched_transformer_candidate.py` |
| **Generalist alone, 13-case geomean** | **2.2969x** |
| **Composed, per-shape best** | **2.3090x** |
| Dispatcher gain over generalist | +0.5% |
| Measured on | NVIDIA A100-SXM4-80GB, float32 |
| Champion decision | `promote` over 13 cases |
| Attempts recorded | 45 |

## Per-shape selection

| case | batch | seq | d_model | heads | implementation | speedup | evidence |
|---:|---:|---:|---:|---:|---|---:|---|
| 1 | 64 | 128 | 128 | 4 | `v038_torchscript_dispatched_transformer_candidate.py` | 1.9783x | generalist |
| 2 | 1 | 128 | 128 | 4 | `v038_torchscript_dispatched_transformer_candidate.py` | 2.9111x | generalist |
| 3 | 4 | 128 | 128 | 4 | `v038_torchscript_dispatched_transformer_candidate.py` | 2.5115x | generalist |
| 4 | 16 | 128 | 128 | 4 | `v038_torchscript_dispatched_transformer_candidate.py` | 2.5113x | generalist |
| 5 | 128 | 128 | 128 | 4 | `v038_torchscript_dispatched_transformer_candidate.py` | 2.1113x | generalist |
| 6 | 10000 | 128 | 128 | 4 | `v038_torchscript_dispatched_transformer_candidate.py` | 2.1738x | generalist |
| 7 | 64 | 128 | 32 | 4 | `v038_torchscript_dispatched_transformer_candidate.py` | 2.3359x | generalist |
| 8 | 64 | 128 | 1024 | 4 | `v038_torchscript_dispatched_transformer_candidate.py` | 1.1250x | generalist |
| 9 | 64 | 128 | 128 | 1 | `v024_centralized_attention_mask_dispatch.py` **·** | 1.7763x | proven |
| 10 | 64 | 128 | 128 | 2 | `v038_torchscript_dispatched_transformer_candidate.py` | 1.9638x | generalist |
| 11 | 64 | 128 | 128 | 16 | `v038_torchscript_dispatched_transformer_candidate.py` | 2.8514x | generalist |
| 12 | 64 | 32 | 128 | 4 | `v038_torchscript_dispatched_transformer_candidate.py` | 2.5213x | generalist |
| 13 | 64 | 1024 | 128 | 4 | `v038_torchscript_dispatched_transformer_candidate.py` | 4.8261x | generalist |
| 14 | 32 | 100000 | 1024 | 16 | `v038_torchscript_dispatched_transformer_candidate.py` | — | generalist |

**·** = specialist, i.e. a shape where a non-champion implementation is measurably faster.

## Specialists

Retained because they beat the generalist on their own shape, even where they lose overall.

- **case 9** — `v024_centralized_attention_mask_dispatch.py` at 1.7763x vs generalist 1.6592x (+7.1%), evidence: proven

## How to reproduce

```bash
tools/iterate.sh candidates/v038_torchscript_dispatched_transformer_candidate.py 1,2,3,4,5,6,7,8,9,10,11,12,13 float32
python3 tools/build_dispatcher.py   # compose the specialists
```
