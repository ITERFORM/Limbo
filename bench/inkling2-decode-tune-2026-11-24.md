# Inkling-2-Mini — decode kernel tuning (2026-11-24)

The wave after `bench/inkling2-budget-2026-11-23.md`: the budget put InkGpu at 43.0 tok/s, 44% of the
288 GB/s bound, and its own profile (`tools/ink_prof.cu`, the CUPTI timeline, 16 head steps at n_ctx
512, one stream) said where the time went: 2,221 kernels per step, 21.5 ms busy, 2.8 ms idle — and
inside the busy, the Q8_0 projections already ran at 220-240 GB/s while the quantized expert paths
ran at 45-130 GB/s. The wall was the decodes and the per-product tail launches, not launch overhead.

## What changed

| change | before | after |
|---|---|---|
| `decode8<NVFP4>` | 8 byte loads + 4 UE4M3 decodes + 8 constant int8 codebook lookups + 8 int→float per 8 elements | 2 word loads and the E2M1 magnitude from its own bits, `(2 + m1) << (e2 - 1)`; no table lookups, values bit-identical |
| the row's tail | a second launch per product (~336/step; 45 µs for half a super-block of work) | fused into the product (`GemvArgs::n_tail`), two ordered accumulators so the store reproduces the old main-then-tail float association exactly |
| IQ2_XXS / IQ2_XS / IQ3_XXS / IQ1_S loads | per-byte assembly of the quant words | word loads (uint16 pairs where rows are only 2-aligned; the NVFP4 layout is 4-aligned) |
| the loop controller's gain/forget | two GEMV launches each (192/step) | one dual product (`gemv_dual`, W2/y2, `blockIdx.z` picks; 96/step), the optional scales folded after exactly as `gemvw` does |

The dead code went with it: `gemv_tail_kernel`, `gemv_experts_tail_kernel`, their four dispatchers,
and the NVFP4 codebook table.

## Measured (the same ink_prof run)

| what | before | after |
|---|---:|---:|
| `gemv_kernel<40u>` (NVFP4 expert products, 15/step) | 221.8 µs/call | **65.4 µs/call** (3.4x) |
| their time per step | 3,327 µs | 981 µs |
| kernels per step | 2,221 | 2,173 |
| step wall under CUPTI | 24,298 µs (41.2 tok/s) | 20,809 µs (48.1 tok/s) |
| **end-to-end decode (`ink_budget`, clean)** | **43.0 tok/s** | **50.0 tok/s** (+16%) |
| share of the 288 GB/s bound | 44% | **51%** |

The tail fusion alone was -0.97 ms of busy; the NVFP4 decode change moved that kernel from 108 to
357 GB/s effective (8 experts x 1.42 MiB per call). The IQ2/IQ3 word loads were a smaller win (their
cost is the per-element LUT lookups, not the byte loads).

## Where the step's 20.8 ms sits after the wave

| line | µs/step | note |
|---|---:|---|
| Q8_0 projections (wq/wk/wv/wo) | 4,576 | 220-244 GB/s, near the bound |
| IQ2/IQ3/Q2_0 expert gate/up | ~3,400 | ~133 GB/s, LUT-bound decodes |
| the output head (IQ4_XS) | 1,150 | 204 GB/s |
| NVFP4 expert downs | 981 | 357 GB/s effective after the decode change |
| the depth mixes | 914 | 120 launches of 7.6 µs |
| F32 controller + bias table | ~1,850 | 2304x64 / 4608x64 / 1024x16 products |
| kernels under 5 µs | 3,087 | 1,456 launches/step |
| idle between kernels | 2,558 | launch gaps |

Gates: all 17 `cuda_*_test` green after every round — `cuda_gemv_test`'s row sums (the fused tail
exercised at n_in 640), `cuda_gemm_test`'s *F32 product equals gemv() bit for bit*, and
`cuda_ink_moe` / `cuda_ink_step` with their bit-identical rerun checks. The budget's internal checks
(the measured VRAM slope, the traffic account bounding the measured rate) pass; `ink_budget` re-run
carries the 50.0 tok/s number.
