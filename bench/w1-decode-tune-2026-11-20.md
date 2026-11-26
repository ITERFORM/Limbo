# W1 decode tuning, rung 1 (2026-11-20)

Qwen3.6-35B-A3B UD-IQ3_XXS on the RTX 4060 Ti, Limbo's own kernels. Every number is from a command in section 11, one GPU
job at a time under `build/gpu.lock`. The card is shared with the desktop (22 percent utilisation at idle) and its clocks
drift between runs: the same binary measures within about 3 percent of itself (94 to 100 tok/s at 512 tokens across runs
of the same code), so differences under 3 percent are not claims.

## Result in one table

| | start (HEAD a7328f8) | now | target | llama.cpp on this box |
|---|---|---|---|---|
| decode at 512 tokens | 83.1 tok/s | **100.4** (9.96 ms) | 120 | 76.0 plain (tg128, empty context); ~120 with its MTP drafting |
| decode at 4k | 56.5 | **98.6** | | |
| decode at 8k | 43.6 (7800 tokens) | **95.5** (8000) | 100 | 72.1 at 8k depth (PLAN.md) |
| decode at 16k | not possible (see section 1) | **88.8** | | |
| decode at 32k | not possible | **79.4** | 75 | |
| bytes per token / floor | | 2040 MB; 7.5 ms = 133 tok/s at the measured 273 GB/s read ceiling | | |
| verify window, cost(T)/cost(1), T = 2..8 | not available | 1.23, 1.30, 1.49, 1.81, 2.13, 2.24, 2.64 at 512 tokens | | |

Targets met: 32k (79.4 against 75). Not met: 120 at 512 tokens (100.4; 120 needs 92 percent of the read bandwidth over the whole
step, and 77 percent of the bytes are Q6_K read at 238 to 259 GB/s), 100 at 8k (95.5). Parity unchanged (section 8).

## 1. Measurements before any change (HEAD a7328f8 + `tools/bench_decode.cu`)

Device bandwidth (`bench_decode --devbw`): 1 GiB device-to-device `cudaMemcpy` 8.8 ms = 122 GB/s of bytes copied (244 GB/s
counting read plus write); a pure-read kernel (16-byte loads, grids from 68 to 544 blocks) reaches **262 to 266 GB/s in the
first session and 274 GB/s in the quiet final one** (92 to 95 percent of the 288 GB/s nominal). The read figure is the reference.

Bytes one decode token reads (`bench_decode --bytes`, 40 layers): **2040 MB**. 77 percent is Q6_K (delta-net qkv/z/out 825 MB,
attention 230 MB, head 417 MB, shared experts 104 MB), the routed experts are 360 MB (gate/up IQ2_S 217 MB, down IQ3_S
(37 layers) or IQ4_XS (3) 142 MB), the F32 router and alpha/beta 100 MB.

Decode at a context (batched prefill of `bench/prompts/long.txt` repeated, then 128 greedy tokens, one sync per token):

| context | tok/s | ms/token | attention |
|---|---|---|---|
| 512 | 83.09 | 12.04 | 1.25 ms |
| 4096 | 56.50 | 17.70 | 6.72 ms |
| 7800 | 43.56 | 22.96 | 12.82 ms |
| 16k, 32k | impossible: `Gpu::open` refuses `n_ctx > 8192` (gpu.cu is not mine) and the old attention kernel keeps its scores in `n_ctx` floats of dynamic shared memory (48 KB: 12k positions) | | |

The 91 tok/s of the earlier document was a 60-token context. Per-kernel timeline of one graph-replayed step at 512 tokens (CUPTI
through `bench_decode --timeline`, 753 kernels, 11.43 ms): Q6_K GEMV 7.70 ms (205 launches), IQ2_S gate/up 1.44 ms, IQ3_S down
0.97 ms, F32 GEMV 0.77 ms (70 launches, 11 us), rmsnorm 0.41 ms (81 launches, 5.1 us each), delta-net scan 0.34 ms, router top-k
0.33 ms (8.2 us each), attention 0.22 ms. A dependent kernel in a graph costs 0.67 to 1.04 us of launch gap (`--launch-gap`).

## 2. What was built

| file | what |
|---|---|
| `src/cuda/attn_decode.cu`, `dec_ops.hpp` | split flash-decode attention for T rows; `attn_prep` for T rows with device-resident position |
| `src/cuda/gemv_exp.cu` | IQ2_S gate/up with SwiGLU fused, IQ3_S down (the routed experts) |
| `src/cuda/gemv_q6k.cu` | Q6_K products for 1 to 8 columns, one or two matrices, optional SwiGLU on the input |
| `src/cuda/gemv.cu` | `gemv_cols` (1 to 8 columns, every format, bit-identical to `gemv`), Q6_K routed to the new kernel |
| `src/cuda/dec_small.cu` | RMS norm, F32 router/alpha/beta products, softmax + top-8, all for T rows |
| `src/cuda/gpu_rows.cu` | `Gpu::step_rows`, `commit`, `read_logits_rows`, `argmax_rows`: the T-row pass, one captured graph per T |
| `src/cuda/gpu_decode.cu` | `launch_step` with parallel graph branches and the new kernels |
| `tools/bench_decode.cu` | `--devbw --bytes --ctx-sweep --profile --timeline (CUPTI) --rows --gemv --launch-gap` |
| `tests/cuda_decode_attn_test.cu`, `cuda_decode_rows_test.cu`, `cuda_gemv_test.cu` (extended) | section 9 |

## 3. Attention (task 3)

Positions are cut into chunks of 256 at fixed absolute boundaries (donor idea: kindi 2.4). A block per (KV head, chunk, row)
scores its chunk for the 8 query heads that share the KV head, so each K row and each V column is read once for all 8 heads
(a warp scores 4 positions per pass with a butterfly reduce-scatter: 31 shuffles for 32 sums), and writes (max, sum,
numerator); a combine kernel merges the chunks in position order and applies the gate. The position and token live in device
memory and the grid is sized for `n_ctx`, so one graph replays at every length. Rows of a verify window are `blockIdx.z`;
the reductions are fixed, so a row's result does not depend on T or on the length beyond it.

`tests/cuda_decode_attn_test.cu` (double-precision reference, 1 to 32760 positions, windows of up to 8 rows, 32768-position cache):
worst |gpu - cpu| 2.4e-7 (bar 3e-6), every row of a window bit-identical to its single-row launch.

| context | before tok/s | after tok/s (this change alone) | attention ms before | after |
|---|---|---|---|---|
| 512 | 83.09 | 87.31 | 1.25 | 0.45 |
| 4096 | 56.50 | 84.22 | 6.72 | 0.51 |
| 7800 | 43.56 | 76.80 | 12.82 | 0.86 |

Measured with a scratch copy of `gpu.cu` whose cap is raised to 65536 (`build/tmp/decode/mklong.sh`; the repo's gpu.cu is
read-only to this work): **16k and 32k need that one-line change, `n_ctx > 8192` in `Gpu::open`**. Final decode: 8000 tokens
95.48, 16000 88.79, 32000 79.40 tok/s; at 32k the attention reads 677 MB of KV per token, 2.5 ms at the read ceiling.

## 4. GEMV (task 2)

The generic i-quant GEMV decoded with about 11 instructions per weight (SASS: 349 instructions per four super-blocks of 8 weights
per lane) and was issue-bound at 150 GB/s; the generic Q6_K needed about 19 and held the big products at 225 to 245 GB/s.

* **IQ2_S gate/up, IQ3_S down** (`gemv_exp.cu`): four weights are negated with an XOR mask plus a per-byte add from a 256-entry
  table (kindi 2.2), the activation row stays in registers while a warp walks 4 to 8 rows, 8 rows are reduced with one
  butterfly (9 shuffles instead of 40), SwiGLU is fused into the gate/up pass. Persistent blocks, `__launch_bounds__(256, 2)`.
* **Q6_K** (`gemv_q6k.cu`): a lane owns 16 weights (four quarters of four elements): three 32-bit loads, the four 6-bit
  vectors built with whole-word mask-and-or, `(v | 0x80) - 0x20` for four bytes at once, the per-16 scale multiplies the unit's
  partial dots. 1 to 8 columns share the decode; per column the arithmetic is the same whatever the column count
  (`tests/cuda_gemv_test.cu`: 2, 5 and 8 columns bit-identical to one, all nine formats).

Microbench (`bench_decode --gemv`, a graph of launches cycling matrices through 190 to 250 MB so nothing sits in the 32 MiB L2;
the first column is the old kernels, the baseline file of the first session):

| product | before GB/s | after GB/s |
|---|---|---|
| Q6_K 8192 x 2048 (delta-net qkv, attention q) | 229 | 260 |
| Q6_K 4096 x 2048 (delta-net z) | 249 | 248 |
| Q6_K 2048 x 4096 accumulate (out, o) | 227 | 238 |
| Q6_K head 248320 x 2048 | 245 | 258 |
| IQ2_S gate/up, 8 experts (before: gate and up, then SwiGLU: 33.4 us) | 154 | 210 (25.6 us, SwiGLU included) |
| IQ3_S down, 8 experts | 144 | 199 (18.1 us; 11.8 us with warmer clocks) |
| F32 router 256 x 2048 | 238 (10.5 us in the step) | 221 (9.5 us; 12 in the step, beside the shared expert) |

A graph of 1 to 3 micro-launches is not the step: in the step the products run beside other work, so the step profile (section 10) is
the arbiter. The Q6_K kernel is faster per launch, but the span of the step moved by less than the noise (10.36 vs 10.33 ms in an
alternating A/B of the same process).

## 5. Launch count, graph branches, what did not work (task 4)

* **Parallel graph branches** (one captured fork/join on a second stream): the delta net's z product beside its conv, norm,
  alpha/beta, scan chain; the shared expert (gate/up, SwiGLU, down, gate dot) beside the router, top-k and routed experts.
  87.3 to 89.6 tok/s at 512 (7800 tokens: 76.8 to 85.6). Kept.
* **Small kernels** (`dec_small.cu`): RMS norm 5.1 us to 3.7 us (81 launches), softmax + top-8 as one warp 8.2 us to 3.1 us,
  F32 router/alpha/beta products with a block per row 10.5 us to 8.5 to 12 us (beside the shared expert). SwiGLU fused into the
  gate/up pass (the `swiglu` kernel is gone for IQ2_S layers).
* **CUPTI timeline of the step at 512 tokens (final)**: 714 kernels, span 9.87 ms, some kernel running 9.83 ms, **idle between kernels
  33 us**. The GPU is never idle; the step is the sum of the critical path's kernel durations, each with its own ramp and tail
  (a 7 MB product at 248 GB/s instead of 265 is 2 us over). The remaining gap to the 7.5 ms floor is 2.4 ms spread over 700 kernels.
* **Tried and dropped**: prefetching the next product's weights into L2 from a side branch during the latency-bound chains (the
  delta net's `out` weights during the scan chain, the attention's `o` weights): the product it fed got 2x faster (19 us to 10.6 us)
  but the prefetch kernel took 29.6 us and the whole step got slower (94.9 to 93.3 and 95.4 to 91.7 tok/s in alternating runs):
  DRAM was not idle, the bytes just moved. Removed. A float4-per-lane Q6_K with all activations of all columns in registers
  (189 registers at 8 columns) cost 8 percent at T = 8 and was replaced by loading each column's activations when used (64 to 100
  registers).
* Not done: merging the delta net's qkv, z and alpha/beta products into one launch, fusing conv + head norm + scan + gate, the
  down product with the combine; each is worth 0.1 to 0.2 ms by estimate, none measured.

## 6. Multi-row decode for speculation (task 5)

```
bool Gpu::step_rows(const int32_t* tokens, int T, int pos0, bool write_back);   // T = 1..8, tokens[t] at position pos0 + t
bool Gpu::commit(int n);                       // n <= T of the last write_back = false window; commit(0) drops it
bool Gpu::read_logits_rows(float* out, int T); // row t's n_vocab logits at out + t * n_vocab(); synchronises
bool Gpu::argmax_rows(int32_t* out, int T);    // the greedy token of each row (lowest index on a tie); synchronises
```

Every weight is read once for all T rows (`gemv_cols`); the attention scores T queries causally inside the window; the delta-net
scan runs the T tokens in order with the state in registers; the routed experts run per (row, choice) pair through the same fast
kernels `step()` uses (an expert chosen by two rows is read twice, the second time from L2). With `write_back = false` the delta-net
state and the conv history do not move; the conv inputs, scan inputs and gates are kept per layer, and `commit(n)` runs the conv
history shift and the scan over the first n tokens with write-back (kindi 2.5). KV rows of rejected tokens are simply overwritten
when those positions are next written. One captured graph per (T, write_back) and per commit length n; a graph is dropped and
recaptured when the state agent's `select_home` changes the state pointers.

`tests/cuda_decode_rows_test.cu` (20 random prompt tokens, then windows of T = 1, 2, 3, 4, 8 against T sequential `step()` calls): the
logits of every row are **bit-identical** to the sequential steps (max|dlogit| 0), the greedy tokens equal, the state after the window
equal (next step 0), and for write_back = false with `commit(n)`, n = 0, 1, T/2, T, the next step agrees with n sequential
steps (0.00e+00); a window of 9 rows is refused.

Cost of a window (`bench_decode --rows`, graph replay without a sync between windows, context 512 then 4096):

| T | window ms | cost(T)/cost(1) | ms per token | write_back = false ms | commit(T) ms | T sequential steps ms |
|---|---|---|---|---|---|---|
| 1 | 9.81 | 1.00 | 9.81 | 10.56 | 0.66 | 10.73 |
| 2 | 12.09 | 1.23 | 6.04 | 12.15 | 0.62 | 21.30 |
| 3 | 12.75 | 1.30 | 4.25 | 12.33 | 0.85 | 29.45 |
| 4 | 14.59 | 1.49 | 3.65 | 14.20 | 1.07 | 41.68 |
| 5 | 17.79 | 1.81 | 3.56 | 18.11 | 0.80 | 51.87 |
| 6 | 20.89 | 2.13 | 3.48 | 20.93 | 0.11 | 58.99 |
| 7 | 22.02 | 2.24 | 3.15 | 23.52 | 0.54 | 73.23 |
| 8 | 25.88 | 2.64 | 3.24 | 26.59 | 1.46 | 82.03 |

At 4096 tokens the ratios are 1.00, 1.14, 1.18, 1.49, 1.69, 1.79, 2.17, 2.32 (T = 1 to 8; T = 1 is 11.2 ms there). The commit is
0.1 to 1.5 ms (noise of the differenced measurement). The growth with T is arithmetic, not bytes: the products decode a weight once
but multiply it once per column in fp32, and the experts are per pair; a tensor-core form would flatten it (open).

## 7. The T = 1 step and a verify window agree

Same kernels, same per-row statements: `step_rows` rows equal sequential steps bit for bit (above). The attention kernel is the
same in decode and verify (kindi 6: "the attention path must be the same").

## 8. Parity

Teacher-forced against the CPU reference exactly as `tools/gpu_forward` measures it (`--names P --parity 32 --gen 64 --ctx 512`, the
CPU reference run beside the GPU, one prompt per locked run, commit-clean tree = HEAD a47b3f1 plus this work's files):

| prompt | top-1 agreement | on generated positions | max\|dlogit\| worst / median | free-running vs CPU greedy | vs the oracle capture |
|---|---|---|---|---|---|
| prose (55 + 32) | 87/87 | 32/32 | 1.29e-3 / 3.6e-4 | first 32 identical | `rung1-prose.jsonl` 16 identical |
| code (95 + 32) | 127/127 | 32/32 | 2.12e-3 / 3.5e-4 | first 32 identical | identical |
| tools (88 + 32) | 120/120 | 32/32 | 2.10 / 1.9e-2 | first 32 identical | identical |

`tools` has one router near-tie: at position 53 layer 20's 8th and 9th router logits differ by 1.9e-4 on the GPU while the residual
entering it already differs from the CPU's by 4e-5, so a different expert is chosen (`--trace-pos 53`: layers 0 to 19 agree to 4e-5,
layer 20 "experts SET DIFFERS"). **The engine before this work does the same at the same position and layer** (gap 2.2e-4), on the
same CPU reference, so it is not new; arithmetic changes anywhere toggle such ties (the intermediate builds of this work flipped
and unflipped `tools` and `prose`), exactly the mechanism of the PoC document's `prose` case. The oracle capture of the 4974-token
`long` prompt (`rung1-long.jsonl`, 16 tokens) is reproduced identically at 5k context, and 64 greedy tokens of prose, code, tools
and `long:1024` equal those of the engine before this work (the llama.cpp-GPU captures `rung1-gpu-*.jsonl` differ from both at the
same indices, so they are not a usable reference for this engine).

## 9. Tests

`flock -w 7200 build/gpu.lock make BUILD=build-decode test` passes in the working tree (all suites, including
`cuda_decode_attn_test`, `cuda_decode_rows_test`, the extended `cuda_gemv_test`: Q6_K, every format's multi-column bit-identity,
IQ2_S gate/up + SwiGLU against the generic pair 1.9e-7, IQ3_S down against the CPU decoder 3.4e-8, rmsnorm, top-k with a tie, F32
products 1.4e-8). While the other agents' step.cpp edits did not compile, the same suites passed in a private tree of HEAD plus this
work's files (`build/tmp/decode/sync.sh`).

## 10. Profiles (events, eager, branches off, so every phase is its own; ms per step)

| phase | 512 tokens | 7800 tokens |
|---|---|---|
| norm (81) | 0.456 | 0.456 |
| delta net qkv product | 2.760 | 2.777 |
| conv + head norm | 0.214 | 0.215 |
| alpha/beta | 0.158 | 0.158 |
| scan | 0.389 | 0.390 |
| gate + out product | 1.048 | 1.050 |
| shared expert | 0.830 | 0.830 |
| router | 0.443 | 0.442 |
| top-k | 0.202 | 0.201 |
| experts gate/up (+SwiGLU) | 1.119 | 1.119 |
| experts down | 0.787 | 0.788 |
| combine | 0.136 | 0.138 |
| attention q/k/v | 0.776 | 0.774 |
| attention prep + split flash-decode | 0.314 | 0.764 |
| attention o | 0.330 | 0.326 |
| head | 1.556 | 1.560 |
| total (eager, events) | 11.57 | 12.04 |

(The z product runs beside the scan in the graph, so its row reads 0.044 and the qkv row holds both.) The graph replay is 9.96 and
10.4 ms. The step is 74 percent of the pure-read ceiling in bytes per second (2040 MB / 9.96 ms = 205 GB/s of 274).

## 11. Commands

```
export TMPDIR=$PWD/build/tmp/decode
nice -n 10 make -j6 BUILD=build-decode cuda tools            # needs the shared tree to compile (step.cpp is the ARCH agent's)
build/tmp/decode/sync.sh                                     # private tree: git archive HEAD + this work's files, immune to others' WIP
L="flock -w 7200 build/gpu.lock"; B=build-decode/bin/tools/bench_decode
$L $B --devbw;   $B --bytes                                  # bandwidth, bytes per token
$L $B --ctx-sweep 512,4096,7800 --gen 128 --profile          # tok/s against context, phase profile
$L $B --ctx-sweep 512 --gen 32 --timeline 30                 # CUPTI per-kernel durations of the graph replay
$L $B --rows 1,2,3,4,5,6,7,8 --ctx 512                       # cost of a verify window
$L $B --gemv q6k:8192:2048 --gemv iq2s:512:2048:egu ...      # GEMV microbench
build/tmp/decode/mklong.sh && $L build/tmp/decode/long/bench_decode_long --ctx-sweep 8000,16000,32000 --gen 128
$L build-decode/bin/tools/gpu_forward --names prose --parity 32 --gen 64 --ctx 512     # teacher-forced gate, per prompt
$L build-decode/bin/tests/cuda_decode_rows_test; $L build-decode/bin/tests/cuda_gemv_test; $L build-decode/bin/tests/cuda_decode_attn_test
```

Raw outputs: `build/tmp/decode/final/` (sweep, long, rows, timeline, gemv), `build/tmp/decode/tf_final1_*.txt` (teacher-forced).

## 12. Open

1. 120 tok/s at 512 tokens and 100 at 8k are not reached: 9.96 ms against a 7.5 ms floor, the 2.4 ms between them is the ramp and
   tail of 700 dependent kernels plus Q6_K at 238 to 259 GB/s. Next: merge the delta net's three products and the attention's
   q/k/v into one launch each, fuse conv + head norm + scan + gate and the down product with the combine, a Q6_K tile that keeps
   the DRAM queue full across the ramp.
2. `Gpu::open` refuses `n_ctx > 8192`: the 16k and 32k rows above were measured with a scratch copy of gpu.cu raised to 65536.
3. Verify windows cost 2.64x at T = 8: the products multiply each decoded weight once per column in fp32 and the experts run per
   pair. A tensor-core form (kindi 2.2 with the plane scheme of `gemm.hpp`) or expert grouping by union would flatten the curve,
   but would have to be the same code as T = 1 to keep rows bit-identical.
4. The vram accounting (`vram_bytes()`) does not include the decode scratch (17 MB of attention partials at 32k) and the T-row
   buffers (about 35 MB, allocated at the first `step_rows`).

## 13. GEMV for the Qwen-trio formats (extension (a))

`src/cuda/gemv.cu`: IQ3_XXS, IQ2_XS, IQ2_XXS, IQ1_S, IQ1_M, Q4_K (27B) and Q5_K, IQ4_NL, Q2_0, F16 (Flash-Next) join the nine rung-1 formats in
`gemv`, `gemv_experts`, `gemv_experts_dual`, `gemv_swiglu` and `embed_row(s)` (the same lane mapping: lane L owns 8 elements of a
256-element super-block; IQ4_NL is 8 blocks of 32 and Q2_0 4 blocks of 64 per super-block). The codebooks are the ARCH agent's
`quants::kIq2xxsGrid / kIq2xsGrid / kIq3xxsGrid / kIq1sGrid / kSignsIq2xs`, read from global memory. `gemv_cols` runs one launch per
column for these (each column is the single-column product, so the bits do not depend on the column count); they decode at the
generic kernel's rate (about 11 instructions per weight), not at the tuned Q6_K / IQ2_S / IQ3_S rate.

`tests/cuda_gemv_test.cu`, section "formats added for the 27B and Flash-Next models": the rung-1 file holds none of them, so 100 rows x
512 of random block bytes with valid half-float scales are decoded by `src/quants.cpp` and dotted in double, and by the GPU; max relative
error to sum|w*x|: Q4_K 2.4e-8, Q5_K 3.6e-8, IQ2_XXS 2.2e-8, IQ2_XS 1.9e-8, IQ3_XXS 2.2e-8, IQ1_S 2.8e-8, IQ1_M 2.0e-8, IQ4_NL 2.3e-8,
Q2_0 2.0e-8, F16 2.3e-8 (bar 1e-5), and three columns equal the single-column products bit for bit. Not done: expert-stack tests
of these formats (the products are the same kernel as the tested ones), tuned kernels for them.

## 14. MTP speculation on rung 1 (extension (b))

The model's own nextn block (blk.40: eh_proj 4096 to 2048, enorm, hnorm, one full-attention + MoE layer, shared_head_norm, the model's
head) is uploaded by `Gpu::mtp_load` (about 340 MB; its experts are Q2_K / Q3_K, the router and shared gate BF16 converted to F32 on the
host) and run by `Gpu::mtp_rows(tokens, T, q0, h_from_mtp)`: x = eh_proj([rms(embed(token)) ; rms(h)]) with h the previous position's
hidden state before the final norm (row t of the last `step_rows` window, or the block's own previous output for chained drafts),
then the layer with a K/V cache of its own, then the head. The math is llama.cpp's `graph_mtp` for qwen35moe (read for the math only).
`mtp_argmax` reads the greedy token of each row. A draft need not be exact (the verify window decides), so `mtp_set_draft_vocab(n)`
scores only the first n ids in the head (65,536 of 248,320: the head's 417 MB become 109 MB).

Loop (`bench_decode --spec NAMES [--k K | --kmax K] [--tail N] [--draft-vocab V]`, host code in `tools/bench_decode.cu`): the last row of
the previous window is refreshed with its next token (one block pass over the accepted rows, whose last row is the first draft), k-1
more drafts are chained, `[c, d1..dk]` is verified in one `step_rows(write_back = false)` window, the accepted prefix is committed
(`commit(m + 1)`), and the next token comes from the window's row m. k is chosen each cycle to maximise E(k)/cost(k) with the window
times measured at start and an EMA of the acceptance of each draft position. The block's K/V covers only the last `--tail` (100)
prompt tokens, which are run through windows instead of the batched prefill; the rest of the prompt never reaches the block.

Greedy output identical with speculation on and off: **every run emitted exactly the tokens of plain greedy decode** (prose, code, tools,
`long:1024`, 128 tokens each, plain decode started from the same state: batched prefill of all but the last 100 tokens, then
single steps), as the bit-identity of window rows and commit (section 6) requires.

Decode tok/s (128 generated tokens; plain against speculative in the same process; `build/tmp/decode/final/spec.txt`):

| prompt | plain | auto k, head over 65,536 ids | k = 2, 65,536 ids | k = 2, full head | auto k, 32,768 ids |
|---|---|---|---|---|---|
| prose (55 tokens) | 98.6 | **143.1** (x1.45, k 1: 82% of drafts accepted) | 143.1 (x1.43; 57%) | 127.4 (x1.28; 60%) | 140.1 (x1.41) |
| code (95) | 98.7 | **141.6** (x1.44; 80%) | 144.1 (x1.45) | 116.8 (x1.17) | 144.6 (x1.47) |
| tools (88) | 98.5 | **136.7** (x1.39; 74%) | 134.6 (x1.35) | 125.9 (x1.27) | 140.9 (x1.42) |
| long:1024 (1024) | 98.8 | **137.1** (x1.39; 75%) | 123.6 (x1.24; 43%) | 105.2 (x1.05) | 134.9 (x1.35) |

llama.cpp's MTP on this box (orchestrator's figure): about 120 tok/s from 76 plain. Here the baseline is already 98.6, and speculation
adds a factor of 1.35 to 1.45 to it: **137 to 143 tok/s**. A cycle at k = 1 costs the verify window (T = 2: 11.1 ms) plus the block pass
(0.7 ms with the 65,536-id head, 1.9 ms with the full one: the head product is 80 percent of it) plus the commit and three syncs, for
1.75 to 1.83 tokens. The full head costs 14 to 30 points of speed for a few points of acceptance; the draft vocabulary is the lossless
approximation of this section (drafts only), off by `mtp_set_draft_vocab(0)`.
