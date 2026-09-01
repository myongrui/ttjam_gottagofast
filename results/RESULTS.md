# Current best submission

_Regenerated 2026-08-31 18:44:40Z from commit `3535b1c`. Do not edit — `tools/report.py` overwrites this file._


| | |
|---|---|
| **Composition** | 1 generalist + 5 specialist(s) |
| **Generalist (fallback)** | `v053_expanded_cuda_graph_dispatch.py` |
| **Generalist alone, 13-case geomean** | **3.0561x** |
| **Composed, per-shape best** | **3.1192x** |
| Dispatcher gain over generalist | +2.1% |
| Measured on | NVIDIA A100-SXM4-80GB, float32 |
| Champion decision | `promote` over 13 cases |
| Evaluations recorded | 72 |
| Evidence class | live runtime evidence |

## Per-shape selection

| case | batch | seq | d_model | heads | implementation | speedup | evidence |
|---:|---:|---:|---:|---:|---|---:|---|
| 1 | 64 | 128 | 128 | 4 | `v053_expanded_cuda_graph_dispatch.py` | 1.9094x | generalist |
| 2 | 1 | 128 | 128 | 4 | `v053_expanded_cuda_graph_dispatch.py` | 8.3144x | generalist |
| 3 | 4 | 128 | 128 | 4 | `v053_expanded_cuda_graph_dispatch.py` | 7.6125x | generalist |
| 4 | 16 | 128 | 128 | 4 | `v053_expanded_cuda_graph_dispatch.py` | 4.6209x | generalist |
| 5 | 128 | 128 | 128 | 4 | `v053_expanded_cuda_graph_dispatch.py` | 2.1085x | generalist |
| 6 | 10000 | 128 | 128 | 4 | `v049_case3_homogeneous_ffn_dispatch.py` **·** | 2.1860x | proven |
| 7 | 64 | 128 | 32 | 4 | `v051_manual_cuda_graph_dispatch.py` **·** | 3.2671x | proven |
| 8 | 64 | 128 | 1024 | 4 | `v053_expanded_cuda_graph_dispatch.py` | 1.1365x | generalist |
| 9 | 64 | 128 | 128 | 1 | `v054_pruned_cuda_graph_dispatch.py` **·** | 1.8998x | proven |
| 10 | 64 | 128 | 128 | 2 | `v054_pruned_cuda_graph_dispatch.py` **·** | 2.0960x | proven |
| 11 | 64 | 128 | 128 | 16 | `v054_pruned_cuda_graph_dispatch.py` **·** | 2.8853x | proven |
| 12 | 64 | 32 | 128 | 4 | `v053_expanded_cuda_graph_dispatch.py` | 5.0508x | generalist |
| 13 | 64 | 1024 | 128 | 4 | `v053_expanded_cuda_graph_dispatch.py` | 4.7708x | generalist |
| 14 | 32 | 100000 | 1024 | 16 | `v053_expanded_cuda_graph_dispatch.py` | — | generalist |

**·** = specialist, i.e. a shape where a non-champion implementation is measurably faster.

## Specialists

Retained because they beat the generalist on their own shape, even where they lose overall.

- **case 6** — `v049_case3_homogeneous_ffn_dispatch.py` at 2.1860x vs generalist 2.1138x (+3.4%), evidence: proven
- **case 7** — `v051_manual_cuda_graph_dispatch.py` at 3.2671x vs generalist 3.2001x (+2.1%), evidence: proven
- **case 9** — `v054_pruned_cuda_graph_dispatch.py` at 1.8998x vs generalist 1.8152x (+4.7%), evidence: proven
- **case 10** — `v054_pruned_cuda_graph_dispatch.py` at 2.0960x vs generalist 2.0180x (+3.9%), evidence: proven
- **case 11** — `v054_pruned_cuda_graph_dispatch.py` at 2.8853x vs generalist 2.5393x (+13.6%), evidence: proven

## How to reproduce

```bash
python3 tools/controller.py contract  # show and confirm exact hash
# After start/resume confirmation and separate paid-GPU authorization:
python3 tools/controller.py evaluate candidates/v053_expanded_cuda_graph_dispatch.py --cases 1,2,3,4,5,6,7,8,9,10,11,12,13 --contract-hash <confirmed-hash> --paid-gpu-authorized
```
