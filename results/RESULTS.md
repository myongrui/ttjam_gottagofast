# Current best submission

_Regenerated 2026-08-31 04:22:03Z from commit `7e3cf5d`. Do not edit — `tools/report.py` overwrites this file._


| | |
|---|---|
| **Composition** | 1 generalist + 0 specialist(s) |
| **Generalist (fallback)** | `v024_centralized_attention_mask_dispatch.py` |
| **Generalist alone, 13-case geomean** | **2.0916x** |
| **Composed, per-shape best** | **2.0916x** |
| Dispatcher gain over generalist | +0.0% |
| Measured on | NVIDIA A100-SXM4-80GB, float32 |
| Champion decision | `promote` over 13 cases |
| Attempts recorded | 41 |

## Per-shape selection

| case | batch | seq | d_model | heads | implementation | speedup | evidence |
|---:|---:|---:|---:|---:|---|---:|---|
| 1 | 64 | 128 | 128 | 4 | `v024_centralized_attention_mask_dispatch.py` | 1.8906x | generalist |
| 2 | 1 | 128 | 128 | 4 | `v024_centralized_attention_mask_dispatch.py` | 2.0714x | generalist |
| 3 | 4 | 128 | 128 | 4 | `v024_centralized_attention_mask_dispatch.py` | 2.0431x | generalist |
| 4 | 16 | 128 | 128 | 4 | `v024_centralized_attention_mask_dispatch.py` | 2.0280x | generalist |
| 5 | 128 | 128 | 128 | 4 | `v024_centralized_attention_mask_dispatch.py` | 2.0168x | generalist |
| 6 | 10000 | 128 | 128 | 4 | `v024_centralized_attention_mask_dispatch.py` | 2.1284x | generalist |
| 7 | 64 | 128 | 32 | 4 | `v024_centralized_attention_mask_dispatch.py` | 2.0157x | generalist |
| 8 | 64 | 128 | 1024 | 4 | `v024_centralized_attention_mask_dispatch.py` | 1.1243x | generalist |
| 9 | 64 | 128 | 128 | 1 | `v024_centralized_attention_mask_dispatch.py` | 1.7763x | generalist |
| 10 | 64 | 128 | 128 | 2 | `v024_centralized_attention_mask_dispatch.py` | 1.9624x | generalist |
| 11 | 64 | 128 | 128 | 16 | `v024_centralized_attention_mask_dispatch.py` | 2.7649x | generalist |
| 12 | 64 | 32 | 128 | 4 | `v024_centralized_attention_mask_dispatch.py` | 2.0396x | generalist |
| 13 | 64 | 1024 | 128 | 4 | `v024_centralized_attention_mask_dispatch.py` | 4.7247x | generalist |
| 14 | 32 | 100000 | 1024 | 16 | `v024_centralized_attention_mask_dispatch.py` | — | generalist |

**·** = specialist, i.e. a shape where a non-champion implementation is measurably faster.

## Specialists

None yet — the generalist is fastest on every shape, so the dispatcher would add nothing.


## How to reproduce

```bash
tools/iterate.sh candidates/v024_centralized_attention_mask_dispatch.py 1,2,3,4,5,6,7,8,9,10,11,12,13 float32
python3 tools/build_dispatcher.py   # compose the specialists
```
