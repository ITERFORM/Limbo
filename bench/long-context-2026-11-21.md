# Long context on the GPU server: 262,144 positions, two homes, three K/V formats (2026-11-21)

Rung 1 (Qwen3.6-35B-A3B UD-IQ3_XXS, trained length 262,144) on the RTX 4060 Ti 16 GB, desktop at 580 to 680 MiB, 768 MiB
reserve. Before this work the server stopped at 8,192 positions (65,536 after `c6efe4f`); now `limbo_serve_gpu --ctx 262144`
answers a 222,811-token prompt.

## 1. What changed

| change | where | effect |
|---|---|---|
| `n_ctx` up to 262,144 | `src/cuda/gpu.cu` | the split flash-decode has no position cap |
| K/V format chosen at open: F16, Q8, Q4 | `Gpu::open(..., KvFormat)`, `src/cuda/gpu_kv.cu` | a 262k cache never exists in f16 first |
| `set_kv_format(f, 0)` re-makes an empty cache in any format | `src/cuda/gpu_kv.cu` | old planes freed before the new ones are made |
| homes in every format | `src/cuda/gpu_state.cu` | a second conversation in Q8 / Q4 |
| Q4 format: q4_0 numerics, K and V rotated by R = H D / 16 | `include/limbo/cuda/kv.hpp`, `src/cuda/attn_decode.cu`, `src/cuda/ops_rows.cu` | 576 B per position and layer (f16: 2,048, Q8: 1,088) |
| token embedding in mapped host memory | `src/cuda/gpu.cu` | 398 MiB of VRAM to the cache; one row per token crosses the bus |
| prefill buffers share one region (delta net, attention, routed experts are never live at once) | `src/cuda/gpu_prefill.cu` | 336 to 183 MiB at chunk 1,024 |
| decode attention partials for one row; the T-row pass has its own | `src/cuda/gpu_decode.cu`, `src/cuda/gpu_rows.cu` | 135 to 17 MB at 262k |
| MTP block K/V: 8,192 positions of its own | `src/cuda/gpu_rows.cu` | 512 to 16 MiB at 262k |
| `--kv auto` (default): most homes that fit, then the most exact format | `src/cuda/gpu_backend.cu` | 32k: 2 homes q8; 262k: 1 home q4 |
| prefill planes 2 by default in the server; chunk 512 past 131,072 | `src/cuda/gpu_backend.cu` | see sections 3 and 4 |
| `Gpu::reserve_scratch` before homes and slots are sized | `src/cuda/gpu.cu` | the free memory the server sees is true |

The Q4 rotation is the idea of rotating keys before quantising them (ik_llama.cpp's K-cache Hadamard; QuaRot), written here
from the description: a 256-point Walsh-Hadamard with fixed signs, in shared memory in the attention preps, its inverse in the
decode combine and in the prefill's window expansion (the tensor-core attention kernel is untouched).

## 2. Memory at 262,144 positions (one Q4 home)

| | MiB |
|---|---|
| weights (no token embedding, no MTP block) | 12,698 |
| Q4 K/V | 1,440 |
| recurrent state of the home | 63 |
| prefill buffers (chunk 512) | 92 |
| decode scratch, logits, partials | about 40 |
| checkpoint slots | 3 x 63 |

`auto: 1 home of 262144 positions, 2 slots each, in 1640 MiB above the reserve`. In f16 the same cache would be 5,120 MiB.

## 3. Quality: teacher-forced against the llama.cpp captures

`bench_prefill --kv F --oracle bench/oracle/rung1-gpu-{prose,code,tools,long-1024,long-2048,long-4096}.jsonl --teacher --ctx 4300`
(planes 3). Top-1 agreement over 288 positions and the mean |d logit| at the capture's top-1 per capture:

| K/V | agree | prose | code | tools | long:1024 | long:2048 | long:4096 |
|---|---|---|---|---|---|---|---|
| f16 | 281 | 0.20 | 0.24 | 0.35 | 0.78 | 0.79 | 0.92 |
| q8 | 280 | 0.24 | 0.16 | 0.56 | 1.18 | 1.07 | 0.78 |
| q4, K rotated | 275 | 0.43 | 0.36 | 0.64 | 0.83 | 1.49 | 1.33 |
| q4, K and V rotated (final) | 278 | 0.43 | 0.29 | 0.67 | 0.92 | 1.30 | 0.71 |

llama.cpp moves by 0.3 to 7.6 nats against itself between micro-batch sizes on the long captures (w1-prefill-tune, section 3).
`tests/cuda_decode_kv_test.cu`: 60 random tokens then 8 steps, against f16: Q8 max |d logit| 0.61, Q4 0.79, top-1 8 of 8; the
8-row window equals 8 steps bit for bit in Q4; the batched prefill into Q4 continues like the steps (0.55, 8 of 8).

## 4. Prefill (bench_prefill --bench, tok/s)

| tokens | planes 3 (exact) | planes 2 (server default) | planes 1 (llama.cpp's rounding) | planes 3, chunk 512 | llama.cpp |
|---|---|---|---|---|---|
| 512 | 2,273 | 2,696 | 2,965 | 2,239 | 2,261 |
| 2,048 | 2,337 | 2,845 | 3,526 | 2,183 | 2,078 |
| 8,192 | 2,205 | 2,613 | 3,250 | 2,038 | |

Planes 2 agrees with the captures as often as planes 3 (281 of 288, w1-prefill-tune section 3). Deep chunks (`--depth 16384`, q4):
a 1,024-token chunk at 16k takes 504 ms against 380 ms at the start; the prompt attention runs at about 23 TFLOPS there, and its
fp32-accurate split of Q and P costs nothing measurable (511 ms without it), so it stays.

## 5. Decode (bench_decode --ctx-sweep, synthetic cache, tok/s)

| context | f16 | q8 | q4 |
|---|---|---|---|
| 4,096 | 94.8 | 95.6 | 91.2 |
| 32,768 | 76.7 | 84.1 | 82.0 |
| 65,536 | | 72.1 | 73.3 |
| 131,072 | | 57.5 | 60.8 |
| 200,000 | | | 51.7 |
| 260,000 | | | 45.3 |

The q4 attention reads 1.5 GB per token at 260k at about 130 GB/s: it is ALU-bound on unpacking and fp32 FMAs (8 heads x 256
dimensions per position), not on bandwidth. Open: int8 dot products or the tensor cores for the GQA group.

## 6. The server, end to end (`scripts/needle_serve.py`)

Haystacks of the repo's own text with one needle, two turns per conversation (the follow-up must come from the cache):

| ctx | K/V (auto) | conversation | prompt | prefill | decode | follow-up | needle |
|---|---|---|---|---|---|---|---|
| 32,768 | q8, 2 homes | 0 (home 0) | 28,646 | 17.4 s (1,651 tok/s) | 83.3 | 28,639 cached, 0.17 s | both turns |
| 32,768 | q8, 2 homes | 1 (home 1) | 32,549 | 20.6 s (1,578 tok/s) | 83.3 | 32,542 cached, 0.18 s | both turns |
| 262,144 | q4, 1 home | 0 | 222,811 | 364 s (612 tok/s) | 51.8 | 222,804 cached, 0.52 s | both turns |

    flock build/gpu.lock scripts/needle_serve.py --ctx 32768 --tokens 29000
    flock build/gpu.lock scripts/needle_serve.py --ctx 262144 --tokens 255000 --one
