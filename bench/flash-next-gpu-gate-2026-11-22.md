# qwen4exp on the GPU: the Phase-2 decode gate (2026-11-22)

The Flash-Next forward pass on the device (docs/flash-next.md §6), held to the CPU reference and the
oracle captures. Teacher-forced on `bench/oracle/qwen4exp-coder-{hi,prose,code}.jsonl` with
`tools/oracle_gate_gpu`, which writes the same JSONL record as `tools/oracle_gate` (the CPU reference's
gate), so the two diff directly.

## The bar and the result

The gate target was the CPU-reference level on the same captures (44/48 against llama.cpp, the four
misses at near-ties) and logits close to the reference.

| prompt | positions | agree | vs oracle misses (oracle gap) | CPU ref (arith) | CPU ref (fp32) |
|---|---:|---:|---|---|---|
| hi | 16 | **12** | 4 (0.43, 0.49, 1.01, 0.02) | 12, same gaps | 12, same gaps |
| prose | 16 | **16** | 0 | 16 | — |
| code | 16 | **16** | 0 | 16 | — |
| **all** | **48** | **44** | 4, all near-ties | 44 | — |

GPU vs the CPU reference's own exact-arithmetic record (`fp32-hi.jsonl`, the same fp32-activation GEMV
arithmetic the kernels use): **max |dlogit at the oracle's top-1| = 0.001, mean 0.000, zero top-1
disagreements, and the identical four misses at the identical oracle gaps.** Against the oracle-arith
record (the reference rounded the way llama.cpp rounds) the mean |d1| is 0.37 to 0.52 — the size of that
rounding's own noise (the reference vs llama.cpp is 0.42 mean).

## What the port is

`src/cuda/gpu_q4e.cu` (the step), `src/cuda/gpu.cu` (the qwen4exp recipe in the loader), the HC / PLE
kernels in `src/cuda/ops.cu`, gemv tails in `src/cuda/gemv.cu` (n_in 320 and 640), and the attention
scorers in `src/cuda/attn_decode.cu` banded over 8-head groups so the 24/2 = 12-head group runs (the
reduce-scatter is 4 positions x 8 heads per warp; rung 1's G = 8 path is bit-identical, its tests say
so). Placement per the plan (6.5): the 4.11 GB of dense weights in VRAM, the 23.4 GiB of routed experts
in the file's own mapping computed by the CPU lane (planes 3, 2 workers) beside the GPU, and the 26.8
GiB PLE table read 16 rows per token on the host. No graph capture (the host round trips inside the
pass), prefill falls back to per-step.

## Cost (this first, unoptimised port)

| prompt | prefill | decode (warm) |
|---|---:|---:|
| hi (2 + 16) | 19 s (cold) | 0.15 tok/s (cold), see below |
| prose (55 + 16) | 168 s | 1.43 tok/s |
| code (95 + 16) | 91 s | 1.50 tok/s |

Everything below the dense path runs on the CPU lane: one token reads ~960 MiB of expert rows through
two workers (~8 GB/s), and the first tokens stream the working set off the SATA SSD (the hi run paid
the cold cache). The plan's 50 / 120 tok/s bar needs the §6.8 sequence (verify windows, int8
hyper-connections, the resident expert cache + prefetch, the INT8 prefill, batched PLE reads); this
gate is the correctness floor those stand on.

## Bugs this gate caught

* The flash scorer's reduce-scatter is 8 heads per warp: templating the group to 12 without banding
  wrote 48 values into `float v[32]` and never scored heads 8..11 — flaky NaN depending on what the
  scratch held. Fixed by running the three phases per band of 8 heads (G = 8 is one band, unchanged).

## Reproducing

```
make cuda
build/bin/tools/oracle_gate_gpu -m .../Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf \
    --oracle bench/oracle/qwen4exp-coder-hi.jsonl --out hi.jsonl
# same for prose / code; the records diff against bench/flash-next-2026-11-22/arith-*.jsonl (and fp32-hi.jsonl)
```
