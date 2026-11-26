# W1 prefill tuning, rung 1 (2026-11-20)

Prefill agent's log. Rung 1 = Qwen3.6-35B-A3B UD-IQ3_XXS on the RTX 4060 Ti, `limbo::cuda::Gpu::prefill`. Every GPU
run is under `flock build/gpu.lock`, build root `build-prefill`, scratch `build/tmp/prefill`. Numbers are appended
milestone by milestone, so an interrupted run keeps what was measured.

## 1. GPU oracle captures (task 1)

`oracle/oracle_dump --gpu` (n_gpu_layers 99, flash attention on, KQV offload and op_offload at libllama's defaults,
ubatch min(512, n_ctx)), the llama.cpp build of `/home/alca/projects/llama.cpp` (`e49a6b44e`, the Makefile's LLAMA_DIR,
the one that made the existing CPU captures). `--truncate N` cuts a prompt to its first N llama.cpp tokens and names it
`<name>:N`. Default behaviour (CPU, no flash attention) is unchanged.

| file | prompt | tokens | greedy |
|---|---|---|---|
| `bench/oracle/rung1-gpu-prose.jsonl` | prose | 55 | 64 |
| `bench/oracle/rung1-gpu-code.jsonl` | code | 95 | 64 |
| `bench/oracle/rung1-gpu-tools.jsonl` | tools | 88 | 64 |
| `bench/oracle/rung1-gpu-long-1024.jsonl` | long:1024 | 1024 | 32 |
| `bench/oracle/rung1-gpu-long-2048.jsonl` | long:2048 | 2048 | 32 |
| `bench/oracle/rung1-gpu-long-4096.jsonl` | long:4096 | 4096 | 32 |

- Captured twice: all six files are byte-identical between the two runs (llama.cpp CUDA is deterministic here).
- The first 16 greedy tokens of prose, code and tools equal the existing CPU-arithmetic captures
  (`rung1-{prose,code,tools}.jsonl`), same prompt ids.
- Command: `flock build/gpu.lock build-prefill/bin/oracle/oracle_dump -m <rung1> -f bench/prompts/<p>.txt --n-predict 64 --gpu --out ...`
  and `... -f bench/prompts/long.txt --n-predict 32 --gpu --truncate {1024,2048,4096}`.

## 2. Baseline (before any change of this session), same box, back to back

`build-prefill/bin/tools/gpu_forward --no-cpu --names "" --gen 0 --ctx 4096 --parity 0 --planes P --chunk 1024 --bench-prefill 512,2048 --profile`

| planes | pp512 tok/s | pp2048 tok/s |
|---|---|---|
| 1 | 2843 | 3187 |
| 2 | 2287 | 2524 |
| 3 (exact) | 1881 | 2009 |

Per-kernel numbers (layer 1, `bench_prefill --kernels`, synthetic skewed routing, T = 512, ms per call):

| op | 1 plane | 2 planes | 3 planes |
|---|---|---|---|
| gdn qkv (Q6_K 8192x2048) | 0.540 | 0.711 | 1.167 |
| gdn gate z (Q6_K 4096x2048) | 0.282 | 0.458 | 0.613 |
| gdn out (Q6_K 2048x4096) | 0.343 | 0.393 | 0.611 |
| moe gate/up (IQ2_S) | 0.963 | 1.482 | 2.091 |
| moe down (IQ3_S) | 0.665 | 0.752 | 0.778 |

Measured limits on this card (microbenchmarks, `build/tmp/prefill/scratch/mmabench*.cu`): int8 `mma.m16n8k16` and
`m16n8k32` both 189 TOPS dense; fp16 -> fp32 accumulate 44 TOPS; fp16 -> fp16 accumulate 87 TOPS. So the int8 path is
the only fast one, and the dense Q6_K kernel ran at 17 to 23 percent of the int8 peak.

SASS of the dense Q6_K kernel (static counts of the K loop, per IMMA): 12.5 FP ops of epilogue (FADD magic, FMUL
scale product, FFMA) + 9.4 integer ops of weight decode + 2.3 LDS: about 27 issue slots per tensor instruction that
needs 8 clocks, so the kernel is issue bound, not tensor bound. The 3-plane MoE gate/up: 9.6 FP + 7.8 integer.

## 3. llama.cpp numerics (task 2)

`planes = 1` is exactly Q8_1 now: d = amax / 127 in fp32, q = roundf(x * (127 / amax)) (ties away from zero), one fp32
scale per 32 values (`act_quant`, `tests/cuda_gemm_test.cu` checks it bit for bit against the host, ties included).
The rounding variants I tried (half away vs to even, x*(1/d) vs x/d, an fp16 scale) change the first generated token's
logit at the capture's top-1 by 0.05 to 1.8 nats on the same prompt while only about 3e-5 of the quantised values
differ: the logits of this model are chaotic at that level (router near-ties), so no choice among faithful emulations
can be verified through the captures.

**The gate (greedy continuations identical to the GPU captures) cannot be met by any implementation, llama.cpp
included.** llama.cpp CUDA against its own capture, same prompts, only the micro-batch changed (`oracle_dump --gpu
--ubatch N`, N = 64, 128, 256, 2048 against 512; 24 comparisons):

| prompt | ub 64 | ub 128 | ub 256 | ub 2048 |
|---|---|---|---|---|
| prose (64 tokens) | identical | identical | identical | identical |
| code (64) | differs at 4 (gap 0.088 -> 0.048) | identical | identical | identical |
| tools (64) | differs at 6 (0.217 -> 0.090) | identical | identical | identical |
| long:1024 (32) | differs at 2 (0.267 -> 2.559) | differs at 2 (0.267 -> 5.154) | differs at 3 (0.618 -> 0.691) | differs at 2 (0.267 -> 0.837) |
| long:2048 (32) | differs at 2 (2.745 -> 3.853) | differs at 2 (2.745 -> 2.726) | differs at 2 (2.745 -> 0.802) | differs at 2 (2.745 -> 3.459) |
| long:4096 (32) | differs at 4 (7.628 -> 1.630) | identical | differs at 9 (2.759 -> 1.913) | differs at 9 (2.759 -> 2.334) |

(each cell: first differing greedy index, and the top-2 gap of the 512-micro-batch capture -> of the other run).
llama.cpp disagrees with itself at gaps of 0.3 to 7.6 nats on 1024+ token prompts.

Limbo against the six captures, final code, `bench_prefill --oracle <6 files> --teacher --ctx 4300` (free-running greedy
decode after the batched prefill; the decode path is Limbo's fp32 GEMV, not llama.cpp's q8_1 MMVQ). Per capture: first
differing index with Limbo's top-2 gap and the capture's:

| capture | 1 plane (Q8_1) | 2 planes | 3 planes (exact) |
|---|---|---|---|
| prose (64) | identical | identical | identical |
| code (64) | 4 (0.145 / 0.088) | 31 (0.070 / 0.251) | 26 (0.589 / 0.112) |
| tools (64) | identical | 59 (0.017 / 0.067) | 59 (0.089 / 0.067) |
| long:1024 (32) | 2 (0.493 / 0.267) | 3 (0.942 / 0.618) | 3 (0.662 / 0.618) |
| long:2048 (32) | 2 (2.201 / 2.745) | 2 (0.800 / 2.745) | 2 (2.092 / 2.745) |
| long:4096 (32) | 9 (0.795 / 2.759) | identical | 9 (4.385 / 2.759) |
| identical | 2 of 6 | 2 of 6 | 1 of 6 |
| teacher-forced top-1 agreement (of 288 positions) | 278 | 281 | 281 |

Teacher-forced (the capture's own tokens fed through Limbo's decode): top-1 agrees at 278 (1 plane), 281 (2 planes) and 281
(3 planes) of 288 positions. Mean |d logit| at the capture's top-1 over matched steps is 0.2 to 0.4 on the short
prompts and 0.7 to 4 on the long ones for every mode: the exact mode is no closer to llama.cpp than the Q8_1 mode,
because the distance is llama.cpp's own noise (its Q8_1 activations) and Limbo's decode path is fp32. The 1-plane mode
is the one whose arithmetic (activation rounding) is llama.cpp's; its first differences on `code` index 4 and the long
prompts sit at capture gaps of 0.09 to 2.8, inside the 0.3 to 7.6 nats llama.cpp itself moves by between micro-batches.

## 4. Changes, with measurements

All times from `bench_prefill --bench 512,2048 --planes P` (3 timed runs after a warm-up, chunk 1024 unless said) or the
per-kernel mode; "bit-identical" = `--golden-check`: the logits after 700 tokens (chunk 256) differ in 0 of 248,320
elements from the run before the change, at 1, 2 and 3 planes.

1. **MoE gate/up, one block less of shared memory.** The 3-plane gate/up kernel staged two buffers (56 KB, one block
   per SM). Single buffer for the 16-token tiles (3 blocks per SM) and for the 32-token tiles (2 per SM):
   kernel at T = 512 / 2048: 2.09 / 6.11 ms -> 1.63 / 4.58 ms per layer (3 planes), unchanged at 1 plane
   (0.96 -> 0.98 / 2.54 -> 2.46). Prefetching the next stages' weight rows into L2 on top: no gain (1.04 / 1.70).
2. **Delta-net scan: no register spills, two chains at once.** The old kernel used 255 registers and a 1,128-byte
   stack (182 STL / 180 LDL in the loop). The new one finishes token t and starts token t+1 in one loop over the 128
   state elements (the chain of token t+1 needs only the updated element i of token t), with the decay, gate and
   sigmoid precomputed for all tokens by `gdn_coef_rows` (they cost 5.3 of the 12.8 ms of the loop: 128 threads
   computing expf/log1pf redundantly on the critical path). 240 registers, no stack. Scan 17.0 -> 10.6 ms
   (512 tokens, 30 layers). **Bit-identical** to the old kernel and, in `tests/cuda_attn_test`, to the decode scan.
3. **Second stream** carries the delta net's alpha/beta/z products beside the qkv product and the scan, and the shared
   expert beside the router and the routed experts (events instead of a join per kernel). 3 planes pp512 247.7 ->
   240.0 ms, pp2048 954 -> 920 ms; 1 plane neutral (156.4 vs 161.1 ms, noise). Bit-identical (`--no-streams` gives the
   same logits).
4. **Dense Q6_K through the fragment-direct kernel** (the MoE kernel's structure: raw rows staged with `cp.async`,
   fragments decoded straight from shared memory) for 1 and 2 planes, tiles of 32 tokens: qkv 0.54 -> 0.52 ms
   (1 plane), 0.71 -> 0.67 (2 planes); shared expert (512 rows) gate/up 6.5 -> 3.6 ms in the whole pass. At 3 planes
   the old kernel stays (1.17 vs 1.25 ms).
5. **rmsnorm + quantiser in one kernel** and a **16-token x 4-row `gemm_f32`** (x tile as conflict-free float4 slots;
   same per-lane accumulation order, so still bit-identical to `gemv`).
6. `planes = 1` is Q8_1 (section 3); `Gpu::set_prefill_streams`.

Dropped after measuring: ablations of the dense kernel (no decode, no epilogue, no MMA, no X copy, all four) each moved
it by under 12 percent and the skeleton alone took 0.63 ms, so it is bound by its stage structure (load, decode, two
barriers, compute), not by one phase; a fragment kernel with 64-token tiles (3.0 ms at 3 planes: register spills);
BM = 16 tiles (0.83 / 1.0 / 1.2 ms).

## 5. Results against the targets and llama.cpp (task 4)

Same box, same day, quiet card, `bench_prefill --bench 512,2048 --reps 5` (mean of 5 timed runs after a warm-up; prompt =
`bench/prompts/long.txt` cut to length, head on the last token), chunk 1024 unless said. llama.cpp: the fork's
`llama-bench` (`0beb5b84b`) run once under the lock today: `-ngl 99 -fa 1 -p 512,2048 -n 0 -r 3`.

| mode | before (this session's first run) | now pp512 | now pp2048, chunk 1024 | now pp2048, chunk 2048 | against llama.cpp pp512 / pp2048 |
|---|---|---|---|---|---|
| llama.cpp CUDA | | 2179 +- 100 | 2223 +- 14 | | 1.00 / 1.00 |
| 1 plane (Q8_1, llama numerics) | 2843 / 3187 | **3355** | **3680** | **3904** | 1.54x / 1.66x (1.76x) |
| 2 planes | 2287 / 2524 | 2822 | 3053 | 3203 | 1.30x / 1.37x (1.44x) |
| 3 planes (exact, default) | 1881 / 2009 | **2305** | **2452** | **2546** | 1.06x / 1.10x (1.15x) |

Targets: exact >= 2600 at 512 and 2048: **not met** (2305 at 512 = 89 percent, 2452 at 2048 = 94 percent, 2546 with chunk
2048 = 98 percent). llama numerics >= 4200 (about 2x): **not met** (3355 at 512 = 80 percent, 3904 at 2048 with chunk
2048 = 93 percent); it is 1.5x to 1.8x llama.cpp. Chunk 2048 costs 0.35 GB more scratch: free VRAM at the peak 1.63 GB
against 1.98 GB (card with the engine loaded, ctx 2100); it is a measured option, the default stays 1024.

Where the time goes now (512 tokens, one run with every phase timed alone, ms):

| phase | 3 planes | 1 plane | | phase | 3 planes | 1 plane |
|---|---|---|---|---|---|---|
| moe gate/up | 63.4 | 36.6 | | gdn scan | 10.0 | 9.4 |
| moe down | 32.9 | 23.1 | | combine | 7.5 | 6.4 |
| gdn qkv | 32.8 | 16.4 | | shared expert gate/up + down | 10.4 | 5.6 |
| gdn z | 17.5 | 8.2 | | router (F32, exact) | 5.9 | 5.7 |
| gdn out | 18.8 | 9.4 | | norms + quantisers | ~7 | ~6 |
| attention projections | 18.3 | 9.3 | | everything else | ~14 | ~14 |
| total (phases sum) | 243 | 152 | | | | |

## 6. Parity (exact mode unchanged)

`gpu_forward --names prose,code,tools --cpu-load build/tmp/gpuperf/cpuref --prefill-check 0,1,2,3 --parity 32` (the
CPU reference dumps of the perf doc), logits after the prompt, max |dlogit| of the batched prefill against the CPU
reference, and the 32-token greedy continuation against the CPU's:

| prompt | perf doc, 3 planes | now, 3 planes | now, 2 planes | now, 1 plane (Q8_1) |
|---|---|---|---|---|
| prose (55) | 3.859e-02, identical | 3.859e-02, identical | 4.450e-01, identical | 6.097e-01, identical |
| code (95) | 7.968e-04, identical | 7.963e-04, identical | 1.376e-01, differs at 26 | 1.719, differs at 4 |
| tools (88) | 1.057e-02, identical | 1.057e-02, identical | 4.525e-01, identical | 1.247, identical |

Exact mode is where it was to the digit. Longer prompts, 3 planes against the GPU's own token-by-token path
(`--prefill-check 2,3 --parity 16`): long:256 0.486 identical, long:1024 0.223 identical, long:1536 4.325 differs at 6,
long:2048 3.048 identical (perf doc: 0.459, 0.858, 4.014 differs at 3, 2.459). Note: since the decode agent's GEMV changes the GPU
token-by-token path is 3.36e-01 from the CPU on `tools` (1.04e-02 before), so the batched exact prefill (1.057e-02 from the CPU) is now
closer to the CPU than the token-by-token path on that prompt.

Kernel level (`tests/cuda_gemm_test`, real rows, error = |gpu - cpu| / sum|w x|, bars 1e-2 / 5e-4 / 2e-5): all passed; the
Q6_K path now runs through the fragment kernel at every plane count and the numbers sit in the old range. `gemm_f32`
equals `gemv()` bit for bit (0 of 17,920). `tests/cuda_attn_test`: scan T = 1+0, 3+2, 100+29, 257+255 equal to the decode
kernel at 0 of 4.6M values; `--golden-check`: 0 of 248,320 logits differ after the scan, stream, norm+quant, `gemm_f32`
and shared-memory changes, at 1, 2 and 3 planes.

## 7. Commands

```
export TMPDIR=$PWD/build/tmp/prefill
nice -n 10 make -j6 BUILD=build-prefill cuda            # tests, gpu_forward, bench_prefill
L="flock -w 7200 build/gpu.lock"; B=build-prefill/bin/tools/bench_prefill
$L $B --planes 3 --chunk 1024 --bench 512,2048 --reps 5 --ctx 2100 [--profile] [--no-streams]
$L $B --planes P --oracle bench/oracle/rung1-gpu-{prose,code,tools,long-1024,long-2048,long-4096}.jsonl --teacher --ctx 4300
$L $B --planes P --golden-save F | --golden-check F --ctx 800       # bit-identity of a refactor
$L $B --kernels --kt 512,2048 --kplanes 1,2,3 [--only qkv|z|out|shexp|router|gateup|down|quant]
$L build-prefill/bin/oracle/oracle_dump -m <rung1> -f bench/prompts/long.txt --n-predict 32 --gpu --truncate 1024 --out ...
$L build-prefill/bin/oracle/oracle_dump ... --gpu --ubatch 128        # llama.cpp against itself
$L build-prefill/bin/tools/gpu_forward --names prose,code,tools --cpu-load build/tmp/gpuperf/cpuref --prefill-check 0,1,2,3 --parity 32 --gen 64 --ctx 512
$L /home/alca/projects/llama.cpp-adaptive-kv-streaming/build/bin/llama-bench -m <rung1> -ngl 99 -fa 1 -p 512,2048 -n 0 -r 3
$L nice -n 10 make -j6 BUILD=build-prefill test
```
Raw outputs: `build/tmp/prefill/final/` (bench, profiles, gate, parity, llama-bench, tests).

## 8. Open

1. Both throughput targets are short by 6 to 20 percent. What bounds the kernels (measured, section 2 and 4): the dense
   and MoE products are issue-slot bound, not tensor bound: per int8 `mma` (8 clocks) the dense Q6_K kernel issues about
   27 instructions (12.5 of epilogue: the magic-float conversion, the scale product and the accumulate are per
   16-wide weight group and per output element, none of it can be moved into the tensor core because int8 `mma` has no
   fp32 accumulate and fp16 -> fp32 `mma` runs at a quarter of the int8 rate on this card), 3 planes make it 7 per element
   and group. The MoE gate/up runs at 68 percent (1 plane) and 39 percent (3 planes) of its weight-read floor (172 MB at 270 GB/s per layer).
2. Ideas not done (the first one is an estimate from the value ranges, not a measurement): fold two planes into the
   accumulator with a chained `mma` (the exact-float conversion needs |sum| < 2^22, which 21-bit activations meet for
   Q6_K and IQ3_S only with a base-127 digit split and not at all for IQ2_S, IQ4_XS, Q8_0); K-permuted activations so that one thread owns whole IQ2_S grid
   entries (the index and table work is shared by two threads today, but removing the decode entirely moved the kernel
   8 percent only); 8-warp MoE tiles (measured: not faster for gate/up or down; 8 to 10 percent faster for the dense Q6_K
   kernel at 3 planes, which is what ships).
3. A latent out-of-bounds read: the fragment MoE kernel stages 16-byte windows around each super-block, so the last
   super-block of the last expert row of a stack reads up to 33 bytes past the tensor; `compute-sanitizer` flags it when
   the stack is its own allocation (the engine's arena hides it). Not fixed (it needs padding in the weight upload,
   `gpu.cu`).
4. `oracle_gate`-style parity at 1024+ tokens needs a different judge than greedy identity (section 3).
