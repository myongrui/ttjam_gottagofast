# Current best submission

_Regenerated 2026-09-01 02:17:25Z from commit `4b54b2d`. Do not edit — `tools/report.py` overwrites this file._


| | |
|---|---|
| **Composition** | 1 generalist + 2 specialist(s) |
| **Generalist (fallback)** | `v059_paper_fusion_portfolio.py` |
| **Generalist alone, 13-case geomean** | **3.1165x** |
| **Composed, per-shape best** | **3.1316x** |
| Dispatcher gain over generalist | +0.5% |
| Measured on | NVIDIA A100-SXM4-80GB, float32 |
| Champion decision | `promote` over 13 cases |
| Evaluations recorded | 81 |
| Evidence class | live runtime evidence |

## Per-shape selection

| case | batch | seq | d_model | heads | implementation | speedup | evidence |
|---:|---:|---:|---:|---:|---|---:|---|
| 1 | 64 | 128 | 128 | 4 | `v059_paper_fusion_portfolio.py` | 1.8927x | generalist |
| 2 | 1 | 128 | 128 | 4 | `v059_paper_fusion_portfolio.py` | 8.5056x | generalist |
| 3 | 4 | 128 | 128 | 4 | `v059_paper_fusion_portfolio.py` | 7.8887x | generalist |
| 4 | 16 | 128 | 128 | 4 | `v059_paper_fusion_portfolio.py` | 4.6558x | generalist |
| 5 | 128 | 128 | 128 | 4 | `v059_paper_fusion_portfolio.py` | 2.1080x | generalist |
| 6 | 10000 | 128 | 128 | 4 | `v049_case3_homogeneous_ffn_dispatch.py` **·** | 2.1860x | proven |
| 7 | 64 | 128 | 32 | 4 | `v051_manual_cuda_graph_dispatch.py` **·** | 3.2671x | proven |
| 8 | 64 | 128 | 1024 | 4 | `v059_paper_fusion_portfolio.py` | 1.1356x | generalist |
| 9 | 64 | 128 | 128 | 1 | `v059_paper_fusion_portfolio.py` | 1.9031x | generalist |
| 10 | 64 | 128 | 128 | 2 | `v059_paper_fusion_portfolio.py` | 2.1154x | generalist |
| 11 | 64 | 128 | 128 | 16 | `v059_paper_fusion_portfolio.py` | 2.8784x | generalist |
| 12 | 64 | 32 | 128 | 4 | `v059_paper_fusion_portfolio.py` | 4.9925x | generalist |
| 13 | 64 | 1024 | 128 | 4 | `v059_paper_fusion_portfolio.py` | 4.7646x | generalist |
| 14 | 32 | 100000 | 1024 | 16 | `v059_paper_fusion_portfolio.py` | — | generalist |

**·** = specialist, i.e. a shape where a non-champion implementation is measurably faster.

## Specialists

Retained because they beat the generalist on their own shape, even where they lose overall.

- **case 6** — `v049_case3_homogeneous_ffn_dispatch.py` at 2.1860x vs generalist 2.1104x (+3.6%), evidence: proven
- **case 7** — `v051_manual_cuda_graph_dispatch.py` at 3.2671x vs generalist 3.1769x (+2.8%), evidence: proven

## How to reproduce

```bash
python3 tools/controller.py contract  # show and confirm exact hash
# After start/resume confirmation and separate paid-GPU authorization:
python3 tools/controller.py evaluate candidates/v059_paper_fusion_portfolio.py --cases 1,2,3,4,5,6,7,8,9,10,11,12,13 --contract-hash <confirmed-hash> --paid-gpu-authorized
```
