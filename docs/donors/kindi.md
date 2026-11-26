# Donor spec: kindi (clean room)

| | |
|---|---|
| Donor | `ahtx/kindi` 0.6.1, commit `7fececb` (2026-11-16), read at `/home/alca/projects/kindi` |
| Licence | **AGPL-3.0**: the code is out of bounds. Only ideas, file-format facts and measurements cross into Limbo |
| Method | read the code and docs and rewrite every finding here in our own words. No donor code, no kernel strings, no line-by-line pseudo-code. Paths in `path:line` form point into the donor tree so the claims can be checked. |
| Tags | **READ** = seen in code · **MEASURED** = the donor's own measurement (doc named) · **CLAIMED** = donor prose, unverified · **EST** = our estimate, not measured |

Short verdict. kindi gets its speed from discipline, not from one kernel. Every product and every recurrent step
gives a token the same bits whatever batch it rides in. The whole step is a replayed CUDA graph with routing,
position and expert slots resident on the device. Speculation is exact by construction. MoE placement is planned
from measured pick counts. Most of these ideas are format-independent, and they are the part Limbo needs. The
ternary/Kepler kernels that make the headline K420 numbers are mostly irrelevant on an Ada card.

---

## 1. What kindi is

**Runtime (READ).** Python host code, CuPy for memory and launches, and CUDA C compiled at start by NVRTC
(`kindi/engine.py:88-95`, `kindi/tc.py:1-7`). The DeepSeek engine is different: C (gcc, `-march=native`, OpenMP)
plus CUDA via nvcc with `-fmad=false`, loaded with `dlopen` (`kindi/dsv41.py:41-56`, `kindi/dsv41_cpu.c:1-20`).
An OpenAI-compatible server (`kindi/server.py`, `kindi/lightning_gen.py`, `kindi/nanbeige_gen.py`,
`kindi/dsv41_server.py`) runs **one request at a time** (README "Limits").

**Module map (READ).**

| area | files | job |
|---|---|---|
| Bonsai (qwen35 hybrid, ternary PTQ1_0) single card | `engine.py`, `kindi.cu`, `chunk1.*` | 2B/4B/9B decode + chunked prompt |
| "plan engine" (27B, 1-2 cards) | `tp.py`, `chunk.*`, `kindi.cu` (g5 kernels) | recorded launch plans, tensor parallel |
| tensor-core plans (sm_80+, sm_61 dp4a twin, sm_60 lane twin) | `spec.py`, `tc.cu`, `tcf.cu`, `tca.cu`, `tcp.cu`, `tcl.cu`, `spec.cu` | int8 mma decode/verify T=1..8, CUDA graphs |
| drafters | `spec.py` (2B draft model, n-gram), `mtp.py`, `dspine.py`/`dspine.cu`, `dsv41_dspark.py`, `lookup.py` | speculative decoding |
| llama-arch K/I-quant (Nanbeige 33B IQ3_XXS) | `nanbeige.py`, `nbq*.cu`, `nbq_cpu.c`, `nbq_grids.h` | dense IQ3_XXS/IQ2_S/IQ3_S/Q4_K/Q5_K |
| Nemotron-H MoE (Lightning 30B-A3B, Q2_0 ternary) | `lightning.py`, `lightning*.cu` | Mamba-2 + MoE + attention, host experts |
| DeepSeek-V4.1-Flash 552B | `dsv41*.py`, `dsv41_cpu.c`, `moe_cpu.c`, `dsv41_gpu*.cu(h)` | GPU dense + CPU experts + expert cards |
| Gemma-4 / Nemotron Bolt E4B | `gemma4*.py/.cu` | separate decoder (not studied further) |
| shared policy | `moe.py` (MoE placement), `cache.py` (prompt cache), `sched.py` (blocking sync), `hostram.py` | |
| CPU engine (Bonsai 2-9B, AVX2) | `cpu.py`, `kcpu.c` | one OpenMP region per token |

**Request flow (READ, `cache.py:851-987`).** Tokenize. Then `Runner.prepare` finds the longest shared prefix with
the engine's current tokens, restores the newest checkpoint at or before the first difference, or a parked
conversation, a saved opening or a saved turn. It reads the rest of the prompt in chunks that never cross a
checkpoint position, then steps the last token with the head. Decode uses one replayed plan or graph per token,
or a verify pass when a drafter proposes. After the reply, a background job saves the turn to disk.

**Models (README; CLAIMED/MEASURED in docs).** Ternary-Bonsai-2 2B/4B/9B/27B (qwen35 = Qwen3.5/3.8 hybrid
GDN+attention, PTQ1_0), AH-Nanbeige4.3-33B IQ3_XXS (llama arch), Nemotron 3.5 Lightning 30B-A3B (ternary Q2_0
MoE), DeepSeek-V4.1-Flash (UD-Q4_K_XL, experts MXFP4), Nemotron Bolt E4B. **Qwen3.8-Flash-Next is announced for
0.7.0** (`CHANGELOG.md:150`, `moe.py:1`). It is the operator's live MoE, so kindi is about to compete on Limbo's
main target.

**Architecture facts it implements (READ, interoperability).** qwen35 hybrid (`engine.py:318-390`). Layer `il` is
full attention when `(il+1) % full_attention_interval == 0` (interval 4), else Gated DeltaNet, and an FFN follows
every layer. Attention: the q projection emits `[q | gate]` per head, then per-head RMS norm on q and on k, partial
RoPE (64 of 256 dims, base 1e7 on the 2B), and a sigmoid gate on the attention output before o-proj. GDN:
fused `qkv` and `z` projections; `beta = sigmoid(x·Wβ)`; `g = softplus(x·Wα + dt_bias)·A`; a causal depthwise
conv of kernel 4 with SiLU over the qkv channels; L2 norm on q and k; the delta rule on a 128x128 fp32 state per
value head; a gated RMS norm with SiLU(z); then out-proj. The 27B has 64 layers, width 5120, FFN 17408,
24 heads / 4 KV heads, 16 k-groups, 48 value heads (`convert.py:24-29`). Its GDN state is **152 MB** in total
(`cache.py:671`).

---

## 2. The CUDA kernels

### 2.1 Design rules that recur everywhere (READ)
1. **Never unpack weights to floats in the decode path.** Either the weight byte is a table index (LUT GEMV),
   or the activations are quantized to int8 and laid out so the weight's natural bit fields meet them in one
   `dp4a`/`mma`. That second option is the general one.
2. **Activations to int8 in runs of 32** (scale = amax/127, round to nearest; scale kept as fp16 in the CPU
   paths) and an exact int32 dot per run, scaled in fp32. This matches llama.cpp's CUDA numerics, and it is what
   makes `kindi verify` token-exact (`nbq.cu:1-12`, `lightning.cu:3-8`).
3. **Batch invariance.** A token's result never depends on how many tokens share the launch. Reductions loop
   tokens outside the reduction, and split-K partials are added in a fixed order by a separate pass or by the last
   block (`tc.cu:1-4`, `tcl.cu:1-9`, `kindi.cu:2010-2013`, `moe_cpu.c:13-16`, `dsv41_cpu.c:16-19`).
4. **Explicit rounding.** `__fmaf_rn`/`__fmul_rn`. Their own `expf` is built from Cody-Waite range reduction and
   a degree-7 polynomial in Horner form, so every compiler gives the same bits (`kindi.cu:41-65`). Norm sums are in
   fp64 because llama.cpp's are; DeepSeek builds with `-fmad=false`.
5. **Device-resident step arguments.** Decode kernels read the token and the position from device memory, and a
   kernel advances the position inside the graph (`lightning.cu:1363`). So one CUDA graph replays every step
   (`nbq.cu:11`, `lightning.py:10-11`).

### 2.2 Decode GEMV families
- **Kepler ternary LUT, `k_g4` (READ `kindi.cu:119-223`, MEASURED `docs/HOW-IT-WORKS.md:5-46`).** The pack
  re-codes 4 trits per byte (81 values). Per 512-column "phase", 384 threads build two 81-entry tables per lane
  for each group of 4 inputs. The tables come from a 3-level ternary recursion: each level is three shifted copies
  of the level below, with no division. Entry e of lane l sits at word `e*64+l`, so a warp's lookups never
  conflict on banks. A `__byte_perm` merges the lane offset and the weight byte into the byte address in one
  instruction. Weights are interleaved 4 rows per 16 bytes, so one 128-bit load serves 4 rows (24 GB/s on the
  K420 versus 14 GB/s with 32-bit loads). 16 rows go to a warp. A multi-row butterfly reduces 16 row partials
  over 32 lanes in 16 shuffles, and the fp16 scale is applied after the first 3 butterfly steps (8 lanes = one
  128-column scale group), so it is applied once per 2 rows. The next phase's x is prefetched while this phase
  computes. Up to 3 matrices that share an input run in one launch. The bound on the K420 is the SM's load/store
  pipe (lookups + shuffles + loads), not DRAM. The kernel reaches ~70% of that floor.
- **Pascal 5-trit tables (READ `kindi.cu:1823-1935`, `2128-2140`).** On the native 5-trit bytes, 24 tables of
  243 entries plus 2 of 81 per 128-column group are built in two stages: first the 27 sums of 3 columns and the
  9 sums of 2, then each 243-entry value as one add. A thread holds 1/2/4/8 rows. The "lane" variant gives lane p
  ownership of byte position p of a 16-row tile, with its own 256-entry table at `entry*32+lane`, so there are no
  bank conflicts and one 16-shuffle reduction at the end. "a" variants skip the 5832-entry table and add the two
  partial tables at lookup time, which gives the same fp32 adds, so the results are bit-identical.
- **Tensor-core decode, `tc.cu` (READ `tc.cu:1-20`).** Activations are quantized per 128-column group. Trits
  enter `mma.m16n8k32` as unsigned digits 0/1/2, and the group's activation sum is subtracted
  (Σ(d−1)q = Σdq − Σq, exact). The tc layout is built on the card from the 5-trit layout at the same size: per
  16-row tile × 8 groups, 3584 bytes, arranged so that each lane holds exactly its mma A fragments. **Five-trit
  digit extraction in registers.** A byte encodes v as ceil(v·256/243), so digit j is the high byte of
  3·((b·3^j) mod 256). Two bytes are done at once in 16-bit lanes with mask 0x00FF00FF, and the digits are
  chained (next = 3·prev masked), so there are no 32-bit multiplies. The same kernel serves T = 1..8 tokens:
  tokens are the mma N dimension, so a verify pass reads the weights once. A "b2" copy (2 bits per digit, 1.29×
  the bytes) replaces decoding with one logic op. It wins only where decode arithmetic, not memory, is the bound:
  GA100 yes, RTX 3090 no (27B 86.1 → 77.8 tok/s, `tc.py:47-58`, MEASURED in comment). "HP" mode puts two int8
  rows per token (~14-bit activations) through the same weight traffic (`spec.py:62-67`).
- **dp4a twin for Pascal (`tcp.cu:1-30`).** Same names, layouts and numerics, bit-identical to the tensor-core
  products. Two threads per row, joined with one shuffle.
- **I-quants / K-quants, `nbq.cu` (READ `nbq.cu:1-12`, `66-70`).** At load the fp16 block scale moves to its
  own array, so the rest of a block is 4-byte aligned (IQ3_XXS 96 B, IQ3_S 108 B, IQ2_S 80 B). Q4_K and Q5_K
  stay as in the file. **Sign trick:** no codebook byte is 0, so negating the 4 bytes of a grid word where the
  4 sign bits say so is an XOR with a per-byte 0xFF mask plus a per-byte +1 in one 32-bit add. The mask comes
  from one multiply-and-mask of the 4 sign bits. The bytes then go straight into `dp4a`. A bank-free IQ3_XXS
  codebook replication helps only on sm_80; it was 14% slower on an RTX 3090 (`CHANGELOG.md:193`).
- **Kepler exact-float dot (`nbq_k35.cu:1-14`).** Kepler has no dp4a and a slow integer multiply. Weights
  (−63..63) are offset to 1..127 and moved by byte-permute into the mantissa of a float equal to 2^16·(192+s).
  An FFMA with integer-valued inputs then adds them exactly, because every partial sum stays below 2^24. A final
  affine fix-up recovers the int32 dot. The results are bit-identical to the dp4a cards.
- **Q2_0 ternary MoE decode, `lightning.cu` (READ `lightning.cu:1-17`, `151-195`).** The activation
  quantizer writes int8 runs in a **lane order** chosen so that `(codeword >> 2k) & 0x03030303` lines up with
  activation word k in one dp4a. The weights are never shuffled; the activations are permuted once per token.
  One launch covers all chosen experts of a token through per-expert pointer tables (`blockIdx.y` = slot; a
  pointer may point to device or mapped host memory). The shared expert is the extra last slot. 4 warps × 2 rows
  per block. Router + top-k run in one launch, and the top-k is done by the block that finishes last, found with
  an atomic counter (`lightning.cu:411-415`). One decode MoE layer is 4 launches (`docs/dev/LIGHTNING-CLEANROOM.md:55`).

### 2.3 Prompt (prefill) GEMMs
- **fp32 tile GEMM with a pluggable unpack stage (`chunk.cu:16-60`).** 128 tokens × 128 rows per block, 8×8
  per thread. The weight layout supplies an "unpack to fp32 tile" stage, and the GEMM core knows nothing about
  formats. It prefetches registers for the next step, plus the next group's raw bytes (`chunk.cu:146`). An int8
  dp4a variant exists for sm_61+ (`chunk.cu:774-780`).
- **int8 tensor-core GEMM for codebook quants (`nbq_mm.cu:1-9`).** 128 rows × 128 tokens. A step's weights are
  decoded once into shared memory as int8 plus run scales, then go through m16n8k32 tiles. **Magic accumulator:**
  the int32 accumulator starts at 0x4B400000, so its bits read as a float equal 12582912 + dot (valid while
  |dot| < 2^22), and one FADD replaces an int-to-float conversion. The shared row stride is padded by 16 bytes so
  fragment loads hit 32 banks. IQ2_S, with a scale per 16, takes two k16 steps. K-quant mins are subtracted as
  min·xs·Σx.
- **Grouped MoE GEMM (`lightning.cu:953-960`, `1314-1320`, `lightning_tc.cu:1-12`).** A one-block kernel
  builds per-expert token lists (counts, exclusive scan, list entry per (token, choice)). The GEMM runs one
  expert per `blockIdx.z` over its contiguous token list, and the combine adds each token's experts in its own
  choice order. The token order inside an expert's range changes no number. The tensor-core variant decodes codes
  into int8 digits in the inputs' lane order and uses the same magic accumulator. Turing gets m8n8k16 equivalents
  with the same numbers.
- **"No head while reading a prompt"**: logits only for the last prompt token, +26% prompt speed on the 2B
  (MEASURED `docs/HOW-IT-WORKS.md:82`).

### 2.4 Attention
- **Decode with an fp16 cache, GQA-aware (`kindi.cu:588-703`, HOW-IT-WORKS §3).** Each K/V row is read once for
  its G query heads, q stays in registers, and a warp scores 4 positions per pass with one butterfly. The value
  kernel splits positions into chunks and combines them in order.
- **Split flash-decode on a q8_0 cache (`tcf.cu:460-620`).** The grid is (KV head, position chunk of
  FD_CH = 512, or 128 on tested sm_80/86, query). A block scores its chunk for all G heads of the group, and the
  partials (max, sum, value vector) are merged **in chunk order** by a combine kernel. Scores on q8_0: the query
  is also quantized to q8_0, each 32-block is an exact integer dot (dp4a) times dq·dk, and the blocks are added in
  order. This is llama.cpp's flash-attention arithmetic for that cache (`tp.py:13-14`, `nbq_q8kv.cu:1-8`).
  **This is where kindi wins at depth:** 27B on an RTX 3080 writes 44.0 vs 23.5 tok/s at 90K tokens
  (MEASURED `docs/BENCHMARKS.md:511`).
- **Prompt attention on tensor cores (`tca.cu:1-14`, `nbq_fa.cu:1-6`).** FA2-style online softmax. K tiles of 32
  positions are double-buffered with cp.async. Scores are int8 mma over q8_0 blocks (an exact integer) times the
  scales; values are fp16 mma with fp32 accumulation; Q fragments stay in registers.

### 2.5 Recurrent layers
- **GDN decode (`kindi.cu:483-506`, `1209-1243`).** One block per value head, one thread per value column (128),
  the state in global fp32. Each thread walks its column twice: the retrieval dot (decayed state · k), then the
  update plus output accumulation. The fused variant also does the head's RMS norm. Per token this reads and
  writes the whole state; on the 27B that is 2×152 MB per token (EST from `cache.py:671`).
- **GDN in prompts (`chunk.cu:228-231`).** T tokens run in one launch, **sequentially in token order inside the
  kernel**, with the decode arithmetic. That is exact and batch-invariant, but it is not a chunkwise-parallel
  delta rule. Prefill of GDN-heavy models is where kindi does **not** pull ahead (27B read on a 3080 ≈ llama.cpp:
  1,093 vs 936 at 512, 855 vs 855 at 22K, MEASURED `docs/BENCHMARKS.md:499-502`).
- **Verify without write-back + commit (`spec.cu:1-5`, `spec.py:9-12`).** A verify pass computes outputs with
  state write-back off and keeps each layer's GDN inputs. Once n tokens are accepted, a commit plan advances the
  state by exactly n steps from the kept inputs. So the recurrent state is exact without checkpoint copies or
  rewinds.
- Mamba-2 (Lightning): causal conv kernel 4, a selective scan over T tokens, and a grouped RMS norm fused with
  quantization for out_proj (`lightning.cu:486-570`).

### 2.6 Glue fusion (READ)
Norm + residual + "pending" split-K partial sums are added in one kernel (`k_rmsfold`, `k_xrms`). The producers
of a product's input also quantize their own chunk for the tensor cores, with the quantizer's exact arithmetic
(`tcf.cu:1-6`). Gated SiLU + Hadamard fold run in one kernel (`k_smfold`). The attention stages of a verify pass
run all its queries in one launch, with `blockIdx.y` = query.

### 2.7 Head and sampling
Exact head = full GEMV + argmax on the GPU. Top-k without copying the vocabulary: a binary search, in 16 halvings
on the GPU, for the largest threshold in [max−band, max] that keeps ≥ k candidates; then only the candidates are
copied (`kindi.cu:754-756`). The special tokens are always scored (fast head, §4).

### 2.8 Launch-overhead strategy (READ, MEASURED in comment)
- **Plans.** One token's kernels are recorded once (function, grid, block, args) and replayed. Only "dynamic"
  arguments (token, position) change (`tp.py:16-18`).
- **CUDA graphs.** One graph per token count T = 1..8 (decode and verify). Per pass, only the nodes whose
  arguments depend on position or token are patched with `cuGraphExecKernelNodeSetParams`. Motivation, measured:
  **1301 launches per token at ~16 µs each** through ctypes made the engine launch-bound (`spec.py:14-16`).
  Lightning and Nanbeige replay one graph per step with routing on the card.
- **Host CPU cost.** 0.02-0.05 cores per stream, against ~0.77 for llama.cpp (MEASURED `docs/BENCHMARKS.md:515`).
  Blocking sync (the host thread sleeps on the driver) is on by default. It costs 1-3% decode on CUDA and frees a
  core (`sched.py:1-5`, `docs/CARDS.md` intro). On ROCm Strix Halo it woke late, so kindi spins there.

### 2.9 Per-architecture selection and tuning (READ)
Capability probes choose the paths: `tc.ga100`, `b2`, `pascal`, `lane`, `small_ampere` (≤48 SMs, the configs
measured on an RTX 3060 / 5060 Ti) (`tc.py:20-90`, `spec.py:70-76`). New defaults switch on only for **measured**
compute capabilities (`TESTED_CC = {8.0, 8.6}`); everything else keeps the older behaviour until measured. **Ada
(8.9, our 4060 Ti) is not in that set.** GEMV (threads, rows/thread, column splits) is timed once per tensor shape
at start, over the real tensors of that shape (one hot tensor stays in L2 and picks wrong configs), and cached.
Per-GPU pinned tables exist for AMD (`tp.py:73-140`). A kernel failure names its node, grid, block and argument
count (`CHANGELOG.md:13-14`).

### 2.10 Why it is faster: attribution
- **K420 (13-28×):** mostly llama.cpp's weakness, not kindi's strength. CUDA ≥11 dropped sm_30, so llama.cpp runs
  there via Vulkan at 1.65 tok/s, while kindi's LUT GEMV is designed around Kepler's load/store pipe
  (MEASURED `WHY.md`, `docs/BENCHMARKS.md:421-433`). **Not transferable to Ada.**
- **RTX 3080 Lightning (read 1.56-1.77×, write 1.5×):** graph-replayed decode with on-device routing, a
  4-launch MoE layer, lane-ordered dp4a/mma Q2_0 products, and grouped tensor-core prompt GEMMs
  (MEASURED `docs/BENCHMARKS.md:23-25`; attribution READ/CLAIMED).
- **RTX 3080 27B (write 1.21× at 1K → 1.87× at 90K):** decode on tensor cores plus split flash-decode on a q8_0
  cache. The win grows with depth (MEASURED `docs/BENCHMARKS.md:508-511`).
- **Speculation:** the 27B with the 2B drafter 99.9 vs 48.7 tok/s on agent prompts
  (`docs/BENCHMARKS.md:525`); with DSpine 115.5 tok/s at 32K (`docs/DSPINE.md:36`). These are the biggest
  multipliers, and they are exact.

---

## 3. The weight pack (`kindi convert`)
- **Bonsai (READ `convert.py:1-11`, `layout.py:1-11`).** PTQ1_0 is re-coded to 4 trits per byte, slot-major, with
  the scales transposed to [phase][row]. At load, `layout.repack` interleaves 4 rows per 16 bytes, groups by
  [16-row group][512-col phase][4-row chunk][lane][16 bytes] so that a warp's 2 KB per phase is contiguous (+6%
  GEMV bandwidth on a K1200 from DRAM page hits), and orders scales lane by lane for one coalesced 128-byte load.
  Pascal+ builds `g5n_*.bin` (5-trit lane layout) once into the pack: 5.6 GB, ~12 min on first start
  (`docs/CARDS.md`, GTX 1080). The tc layout is built **on the card** from the 5-trit layout at the same size.
- **Other formats.** Lightning and Nanbeige read the GGUF directly and repack at load (`lightning.py:1-11`,
  `nbq.cu:7-9`). The DSpark drafter is converted from safetensors: FP8 E4M3 with E8M0 scales per 32×32 → Q8_0
  rows; FP4 experts → MXFP4 blocks, exact (`dsv41_dspark.py:1-11`).
- **Load path.** Weights go to the card one tensor at a time, read 256 MB at a time with `POSIX_FADV_DONTNEED`
  after each piece, so a host with 2 GB free can load the 27B (`engine.py:96-108`, `CHANGELOG.md:206-207`). The
  embedding table lives in **mapped pinned host memory**: a token reads one row over PCIe (`tp.py:897`; Lightning
  2.9 KB/token, `lightning.py:336`). That frees VRAM for context at a negligible PCIe cost.
- **State fingerprint (READ `cache.py:102-140`).** Saved states are keyed by a hash of every source file that
  decides the arithmetic plus the engine configuration (chunking, int8 prompt path, tc, KV type, head mode). A state
  from another engine build is never resumed.

**Format facts (interoperability).**
- **PTQ1_0 (GGUF type 143):** 128 weights, 28 bytes = `qs[24]` (5 trits/byte; bytes 0-15 hold elements 16n+i for
  digit n=0..4, bytes 16-23 hold 80+8n+i) + `qh[2]` (4 trits/byte, elements 120+2n+i) + fp16 d. Weight = (t−1)·d,
  and digit n of a byte b is ((b·3^n mod 256)·3) >> 8 (`convert.py:39-52`). Prism stores a block Hadamard rotation
  folded into the weights, with metadata `prism.hadamard.{sign_values, sign_widths, weight_names,
  inverse_weight_names, tied_output}`. The engine applies the matching FWHT with signs to the activations
  (`convert.py:110-120`).
- **Q2_0 (type 42):** 64 weights, fp16 d then 16 bytes. Weight j is in byte j/4 at bits 2(j%4); value (code−1)·d
  (`docs/dev/LIGHTNING-CLEANROOM.md:20-23`).
- **IQ3_XXS (type 18):** 256 weights, 98 bytes = fp16 d + 64 grid-index bytes (each index → 4 magnitudes from a
  256-entry grid) + 4 u32, each holding 4×7 sign bits and a 4-bit run scale. The 8th sign bit of each group of 8
  is the even parity (`nbq_cpu.c:13-15`).
- **MXFP4:** 32 weights, 17 bytes = E8M0 scale (2^(e−127)) + 16 code bytes. Byte j holds weight j in the low
  nibble and weight j+16 in the high nibble; E2M1 magnitudes are {0, .5, 1, 1.5, 2, 3, 4, 6} with a sign bit
  (`moe_cpu.c:3-5`).

---

## 4. Exact vs fast output head
- **Exact (default):** the full GEMV and argmax. On the 2B the head is ~9 ms of a 46 ms K420 step
  (README "Exact and fast output heads").
- **Fast (READ `cluster.py:1-6`, `layout.py:51-90`, `engine.py:406-438`).** Offline spherical k-means (1,024
  clusters, 12 iterations, ~2 min) over the dequantized, normalized head rows in the folded input space. The head
  rows are stored in cluster order, each cluster padded to whole 16-row groups (padding = zero weights, token −1),
  and the embedding lookup follows the permutation. Per token: score the centroids, take the best M = 256
  clusters (~13% of rows), expand them to a group list, run the LUT GEMV over just those groups with a compact
  output plus a token map, then argmax. The cluster that holds the special tokens (`</think>`, EOS) is always
  forced in. Without that, the 9B never closed its thinking (`engine.py:154-155`).
- **Accuracy and cost (MEASURED README / `docs/CONFIGURATION.md`):** ~15% faster generation on the 2B; top-1
  equals the full head 99.4% of the time at 256 clusters, 98.1% at 128. **Not exact.** It can be switched at run
  time (`kindi head`). The 27B and Nanbeige have no fast head.

---

## 5. Prompt cache (READ `cache.py:1-17`, `497-640`, `650-1000`; MEASURED `docs/PROMPT-CACHE.md`)
1. **Conversation on the card.** K/V is position-indexed, so it stays valid for any prefix. A request that
   extends the current tokens continues at their end.
2. **Recurrent-state checkpoints.** The GDN (or Mamba) state goes to host RAM every `checkpoint_every` tokens
   (default 256, widened automatically so that `(park_max+1)·state·ctx` fits a budget of 20% of RAM), at the
   generation-prompt **boundary**, and at the end of each prompt. Prompt batches never cross a checkpoint
   position. **Dense opening checkpoints** every 256 tokens below min(4096, ctx/2) let a new agent session resume
   just before its first difference (agent openings differ only near their end: the date, the working folder)
   (`cache.py:696-700`).
3. **VRAM checkpoint slot (`KINDI_CKPT_VRAM`, `cache.py:497-506`).** A checkpoint is a device-to-device copy
   into one VRAM slot (~1 ms). Its host copy is made only when the server is idle, in 1 MB pieces on a background
   thread that stops at the next piece when a request arrives, so it never shares the link with decode readbacks.
   It is forced right away only if the checkpoint must be parked, saved, or displaced. The newest checkpoint
   restores with a device copy.
4. **Parking (a move).** A request that shares < 50% of the active conversation (or branches off it) moves the
   active one (tokens, checkpoints, end state and K/V) to host RAM, `park_max` deep. It comes back when its turn
   comes (`cache.py:883`). Cost per parked conversation: ~16 KB/token + ~42 MB per checkpoint (Nemotron,
   `docs/PROMPT-CACHE.md`). Eviction order under the budget: checkpoints held only by a parked conversation,
   then the oldest checkpoints of the active one, then whole parked conversations, oldest first
   (`docs/CONFIGURATION.md`).
5. **Saved openings (disk).** When two requests share ≥ `prefix_min` (512) tokens and part before the opening's
   end (+16), the full state at that checkpoint goes to disk under the engine fingerprint, LRU `prefix_max`
   deep. It survives restarts.
6. **Saved turns (disk, 0.6.0).** After each answer, a background thread writes the state at the **end of the
   messages, before the generation prompt**, because the next request re-renders the reply with different tokens
   (an empty think block, for example), so the post-reply state would never match (`cache.py:994-1000`). K/V rows
   go to disk in **1,024-token blocks shared by the turns of one conversation**, so a turn writes only its new rows
   (`cache.py:15`, `docs/CONFIGURATION.md` `turn_disk_mb`).
7. **Measured (GTX 1080, `docs/PROMPT-CACHE.md`):** Nemotron agent turn 2: 2.76 → 0.67 s; RAG question 3:
   10.71 → 0.42 s (25×). 27B agent turn 2: 15.33 → 2.52 s. The first request is up to 8% slower, because the
   checkpoint copy runs during the read. Exactness: Bonsai replies identical; Nemotron identical in 283/295 agent
   turns, the rest near-ties, because the cache reads in other batch sizes.
8. **Sessions and priority (`CHANGELOG.md` 0.6.0).** Sessions are named by `X-Kindi-Session`. A background
   request yields to a chat request **between prompt chunks** and continues from the cache afterwards with the
   same answer. A stream gets SSE progress comments every 10 s. `dry_run` gives exact token counts; a prompt that
   does not fit gets HTTP 400 `context_length_exceeded` at once.

---

## 6. Speculative decoding
- **Exactness by construction (READ `spec.py:1-12`).** Verify uses the same batch-invariant products as decode
  (§2.1.3), with GDN verify-without-write-back and a commit (§2.5). Greedy output with any drafter is therefore
  plain greedy output, token for token. Sampled requests use speculative sampling with a deterministic draft,
  where a rejected draft's residual sample replaces it (`dsv41.py:551-562`). One caveat: with the non-FD
  attention path, drafting differed at near-ties. The fix forces the split flash-decode attention whenever
  drafting is on (`docs/DSPINE.md` Exactness, `CHANGELOG.md:112-114`). **The attention path must be the same in
  decode and verify.**
- **Verify cost.** Codebook verify kernels decode each weight run once for N = 2..8 inputs (`nbq_mvn.cu:1-2`).
  The tensor-core form puts tokens on the mma N dimension, so "a pass with 2..8 tokens costs about one
  single-token pass" for the products (CLAIMED `nbq_tcv.cu:7-8`). Whole-pass costs measured on 2×3080 (33B,
  layer split): T=2..8 → 1.18, 1.47, 1.67, 1.94, 2.16, 2.6, 3.02 steps (`lookup.py:8-9`). Without tensor cores or
  dp4a batching, T tokens ≈ T steps (K1200: 1.80/2.61/3.42 for T=2/3/4), so speculation is off on Pascal and
  Maxwell (MEASURED `docs/MTP.md:30-37`).
- **Drafters.**
  - *2B draft model for the 27B* (same card, separate 73,728 context): k=2 below 1K context, k=3 from 1K; 99.9 vs
    48.7 tok/s greedy on agent prompts on a 3080 (`docs/BENCHMARKS.md:525`). On an RTX 3060: 60.6 / 73.5 /
    66.7 tok/s at 512 / 8K / 24K (`docs/MTP.md:47`).
  - *DSpine* (Tencent's Apache-2.0 weights for the **Qwen3.8-27B family**; READ `dspine.py:1-18`, MEASURED
    `docs/DSPINE.md`). A 5-layer block drafter. Its context is the target's own hidden states at layers
    1/16/31/46/61, copied by the target's plans into a feature buffer, projected per position, and held in a K/V
    ring of 8,192 positions per draft layer. A block is the current token + 7 masks; it attends to the ring and
    causally inside the block, with a low-rank "code bridge" message passed row to row between layers. The rows
    go through the **target's own** output norm and head, and a top-16 refinement chain picks each draft given the
    one before. The draft is one CUDA graph of ~160 kernels: int4 (group 64) linears with int8 activations on
    dp4a, the bridge in fp16; ~8 ms on a 3080. Memory: 1.4 GB VRAM + 0.75 GB mapped host (bridge codebook,
    head×write table). Result: **3.10 tokens per verify pass, 115.5 tok/s** at 32K on a 3080 at 190 W (code and
    math up to 4.6/pass, prose 2.3-2.8). The ring is not restored with a prompt-cache restore, which lowers
    acceptance but keeps the output exact.
  - *DSpark* (DeepSeek's own 3 MTP stages): drafts ≤5 per step, each with a confidence logit.
  - *Own MTP head* for the 2B (61 MB, trained by kindi), k=2 up to 8K context.
  - *Prompt lookup* (READ `lookup.py`): the longest earlier match of the last 2-4 tokens; copy what followed.
- **Draft-length policy (the reusable idea).**
  - Lookup: choose k to maximize (1 + a + … + a^k)/cost(k+1) with the measured check-cost table and a decayed
    acceptance estimate a (prior 60%); k = 0 when a draft is expected to lose time, except that a 4-token match
    forces a 1-token probe so acceptance is re-measured (`lookup.py:34-58`). Result: 2.03× on "copy a file with
    changes", 1.17× on tool calls, 1.0× on prose (MEASURED `docs/BENCHMARKS.md:325-326`).
  - DSpark: E(k) = 1 + Σ_j Π_{i≤j} p_i with p_i = sigmoid(a·conf_i + b) (calibrated). cost(k) = gpu_ms +
    per_position_ms·k + expert_ms·U(k+1), where **U(m) is the expert-union factor of m consecutive tokens**:
    table 1.0, 1.69, 2.28, 2.80, 3.28, 3.73, 4.15, 4.54 for m = 1..8 on DeepSeek (6 of 384), replaced at run time
    by an EMA (0.85/0.15) of the measured distinct CPU experts per verify of m positions. Pick the k with the best
    E(k)/cost(k). +1.9% over the fixed table (`dsv41.py:580-650`). Fallback heuristic: +1 after a full accept,
    halved after a reject (`dsv41.py:738-740`).
  - Auto mode (spec.py): per-context thresholds measured per card, and a guard that pauses speculation when it is
    slower than plain decoding, active only from 32K (below that, its pauses cost 9-12%) (`spec.py:52-61`).

---

## 7. MoE with experts in host memory
- **Placement policy, shared by every MoE engine (READ `moe.py`).** Per-layer pick counts are measured on text
  and shipped as JSON (`lightning-experts.json`, `tests/data/dsv41/hot-counts.txt`). `plan()` gives each layer at
  least n_used experts on the card (so one token's picks always fit). The remaining card bytes go **greedily, slot
  by slot, to the layer whose next-most-used expert has the largest share-of-that-layer's-picks per byte**, so
  layers with spread-out routing get more slots (`moe.py:140-189`). Host RAM keeps the most used of the rest; the
  rest goes to "disk" (mmap of the file). The resulting `Placement` reports the expected share of picks each tier
  serves. Example: 7 GB on the DeepSeek expert card holds 399 experts that take 36% of picks
  (`docs/DEEPSEEK.md:85`).
- **Lightning on 6-8 GB cards (GPU computes everything; READ `lightning.py:1-11`, `38-43`, `416-445`,
  `lightning.cu:1245-1310`).**
  - *Decode:* cold experts sit in pinned host memory. Each MoE layer has D device **slots** (default 10) that
    keep the last used host experts. A single-thread kernel maintains owner and age per slot on the GPU: hits
    first, so a miss can never evict a slot in use this step, then LRU victims; a duplicate pick in one step is
    one copy. A copy kernel then **pulls** each missed expert from mapped pinned memory with wide coalesced
    16-byte reads ("far faster over PCIe than the products' small reads") and writes per-slot pointer tables. The
    products run on device memory. Everything is inside the step graph, with no host round trip. Tuning (MEASURED
    in comment): 6 → 10 slots gave +5-6% on an RTX 2070 SUPER; more slots cost card-resident experts (85 → 81).
  - *Prefill:* the host experts of a MoE layer are copied **whole** into one of two staging buffers by a copy
    stream, two layers ahead, with events so the compute stream waits only for its own layer. The grouped GEMM
    then reads staging. With a 512-token chunk nearly every expert is hit, so moving the whole layer is right.
  - *Measured:* RTX 2070 SUPER (8 GB, PCIe 3 x16), 74-122 of 128 experts per layer on the card: write 116.4 vs
    113.5 llama.cpp `-ncmoe 12`, read 2,121 vs 1,137. GTX 1080 on PCIe 2.0: write 75 vs 4.7 (llama.cpp CPU
    experts on an AVX-less Xeon). RTX 2060 with 6 GB held back (60/128 on the card): read 1,415 vs 520, write
    52.7 vs 51.5 (`docs/BENCHMARKS.md:43-90`). **Write gains over llama.cpp's CPU-expert offload are small when
    the host CPU is decent; read gains are large (≈1.9-2.7×).**
- **DeepSeek-V4.1-Flash 552B (CPU computes experts; READ `dsv41_cpu.c:1-20`, `600-700`, `1888-1935`,
  `moe_cpu.c:1-33`).**
  - Card 0 runs dense + attention as one forward. **At each MoE layer it calls back to the host**, which computes
    the routed experts and returns. The card and the CPU **alternate**; they do not overlap within a token.
    Optional expert cards (any sm_50+ ≥4 GB, e.g. Quadro M4000) hold a static hot set and compute their (token,
    expert) pairs **concurrently** with the host's pairs. The per-pair values are then combined in slot order. A
    half-batch pipeline (card does one half's layer while the CPU does the other half's experts) was **measured
    as no gain** and is off (`dsv41_cpu.c:2051-2053`).
  - CPU expert kernel (AVX2): MXFP4 codes kept as-is. A 16-entry `pshufb` table maps each code to the
    **doubled** weight offset to be non-negative. `maddubs` (unsigned weights × signed int8 activations) then
    `madd` to int32, and a per-block correction of 12·Σq removes the offset. The result is exact. One fp32 FMA per
    block with both scales. Gate and up rows are interleaved in 4+4 groups, so one thread produces silu(gate)·up
    for its rows. **Union batching:** for a batch (prompt chunk or verify), form the union of experts, with a
    token list per expert. Each expert row group (21 KB gate/up, 5 KB down) is read from RAM once and stays in
    L1/L2 while all of its tokens use it, two tokens per pass (codes decoded once for both).
  - Threads: a persistent pool, one thread pinned per physical core, **spin barriers between steps; between
    calls a short spin, then sleep on a futex**, so other pools get the cores. The dense pool runs on the
    hyper-thread siblings of the expert cores, so the two pools never fight over a core while one spins.
  - NUMA: every matrix is split by rows across nodes in proportion to their memory bandwidth, on 2 MiB-aligned
    `MADV_HUGEPAGE` pages bound with `MPOL_PREFERRED` (not BIND, which can trigger the OOM killer on a shared box).
  - Measured: 18.7 tok/s writing (with DSpark), prompt read 95-100 tok/s, "the CPU expert work sets this speed".
    Card time per token went from 34.6 → 24.1 ms by merging steps and running kernels concurrently
    (`docs/DEEPSEEK.md:96-98`, `CHANGELOG.md:106-107`). Prompt chunk 512 by default; 1024 is 7% faster but needs
    +1 GB VRAM.
- **Dense layers in host memory (Nanbeige on a K6000; READ `nanbeige.py:34-35`, `384-500`, `nbq_cpu.c:1-17`).**
  Layers that do not fit keep their weights pinned on the host. Per step, two whole-layer device slots are filled
  by async copies while the card computes, **and the CPU computes 55% of each host layer's FFN rows** in the same
  step (AVX2 IQ3_XXS kernel). That is a row-level split between "move the bytes" and "compute where the bytes
  are" (3.56 vs 1.52 tok/s llama.cpp at 8K, MEASURED `docs/BENCHMARKS.md`). The CPU IQ3_XXS kernel: the codebook
  is expanded once into a **signed-word table** of 16 sign patterns × 256 grid entries × 4 int8 = 16 KB, which
  stays in L1. At load the CPU's rows are re-encoded as one uint16 index per 4 weights (grid | sign pattern, with
  the parity bit pre-applied, 4.19 bits/weight). A step does one table load for 4 signed weights, widens them to
  int16, and runs `madd_epi16` against int16 inputs into exact int32.

---

## 8. Tensor parallel, CPU engine, AMD (brief)
- **Two-card TP (READ `tp.py:1-18`).** Head-group / KV-head / FFN-block splits. The three row-parallel products
  exchange partial sums through **mapped pinned host memory**, with exchange epochs counted on the GPU. Both cards
  add [card0 + card1] in the same order, so the residual stream is bit-identical on both. Layer split (Nanbeige)
  sends only the hidden state, and PCIe x1 suffices.
- **CPU engine (READ `kcpu.c:1-4`, `79-135`).** The **whole token runs inside one OpenMP parallel region**, with
  barriers between stages and no per-op thread start-up. Ternary GEMV on AVX2: a pair of inputs has an 8-entry
  signed-sum table = **one ymm register**. A 4-bit code per row (code 8 = zero, masked) selects through
  `vpermps` for 8 rows at once. Up to 3 matrices share one table build. MEASURED (`docs/BENCHMARKS.md` 0.3.0):
  2B on a Ryzen 9 7940HS: 64.8 vs 23.6 tok/s write against llama.cpp.
- **AMD/ROCm.** The same sources run through hipRTC with shuffle shims. Configs are pinned per gfx target or
  timed at start, and THP is disabled in-process because khugepaged collapses evict GPU queues
  (`engine.py:67-77`). A barrier inside a condition broke the 9B on AMD (`CHANGELOG.md:215-216`).

---

## 9. Flaws and limits
- One request at a time, and the card holds one engine. No concurrent main + sidekick (README "Limits").
- Parking **moves** conversations to host RAM, which is the "context coffin" pattern; resuming a parked
  conversation is a copy back.
- Python + CuPy hosting. Graphs hide it in decode, but prompt chunking, cache logic and the DeepSeek callbacks
  still cross Python/ctypes. CUDA 10.2 / NVRTC is pinned for Kepler, so the code relies on PTX header rewriting
  to reach mma on sm_80+ (`tc.py:1-7`, `99-110`) and uses inline-PTX `dp4a`.
- Model coverage is hand-built per architecture: shapes are hard-coded per profile (`convert.py:24-29`,
  `dsv41_cpu.c:54-75`), and DSpine fits only the 27B shape.
- The GDN prefill is sequential inside a kernel: no chunkwise-parallel delta rule, so there is no read-speed edge
  on the hybrid 27B (§2.5).
- The DeepSeek card and CPU alternate and do not overlap; prefill is bound by CPU experts at ~100 tok/s.
- Host-expert decode slots are a small LRU (10/layer). There is no lookahead or prefetch from router predictions,
  and the placement is static, computed once at load from shipped counts.
- The fast head is not exact.
- Tuning beyond sm_80/86 is unmeasured: Ada, our card, runs the older defaults.
- AGPL: unusable as code for Limbo.

---

## 10. Conflicts with the other donors
1. **PCIe (6.6 GB/s measured on the desktop).** kindi uses PCIe for host-expert misses in decode, whole-layer
   staging in prefill, the embedding row, and checkpoint/turn copies. llama.cpp-adaptive-kv-streaming streams KV
   pages every step past its stage. Strata streams experts and lends prompt slots. Each assumes it owns the
   link. kindi's own answer for one consumer (checkpoint copies only when idle, preemptible in 1 MB pieces) has to
   become a **link budget** in Limbo's planner, shared by expert pulls, KV pages and background copies.
2. **VRAM.** kindi fills the card with experts from usage counts, then fits the context to what is left
   (`engine.py:205-212`). KV streaming wants a stage sized to the live tapes. Strata wants an adaptive expert cache
   plus a prompt lend region. A drafter (DSpine 1.4 GB, DFlash2 1.5 GB) needs VRAM too. Three greedy allocators
   cannot all win; one planner must price expert slots, KV pages and drafter bytes in the same units (expected
   ms saved per byte).
3. **Pinned host RAM (31 GB box, ~2 GB free while Strata runs).** kindi pins host experts, checkpoint buffers and
   mapped tables; KV streaming pins the host KV (≈7 GB at 262K on the 27B); Strata pins ~20 GB of experts. These
   are mutually exclusive at full size.
4. **CPU cores.** kindi's DeepSeek pool pins one thread per physical core and spins between steps. modchip's
   recipe is `-t N-1 --poll 0`. Strata has its own host threads. The desktop and the Mazemaker pods share the
   same 8 cores. kindi's spin-then-futex and sibling placement are compatible ideas, but a pool that spins inside
   a step will throttle a GPU-bound decode thread on 8 cores.
5. **Who owns the graph.** kindi builds its own per-architecture graph (plans/CUDA graphs per T). The kv-streaming
   fork and modchip are ggml patches (ggml owns the graph, and kernels are per-op). Strata is its own engine. Limbo
   rebuilt from scratch can own the graph, and kindi's "fixed plan per T, patch only dynamic args" model is the
   closest to what Limbo needs.
6. **Formats.** kindi repacks into kernel-specific layouts at load or in its pack. Strata packs experts into
   4 KiB-aligned streaming extents. modchip adds GGML types 44/45 and repacks for CPU SIMD. llama.cpp is native
   GGUF. Limbo needs one pack concept that carries both "aligned extent for streaming" and "device-kernel layout",
   or a cheap on-device relayout (kindi builds its tc layout on the card).
7. **Numerics and exactness.** kindi = llama.cpp-exact, batch-invariant, verify==decode. The kv-streaming fork
   follows llama.cpp but is not batch-invariant across its streaming and non-streaming paths (unverified). Strata
   has its own numerics. Mixing engines per request would break the operator's token-exact tape and spec gates.
   One arithmetic contract is needed.
8. **The tape.** kindi's park-on-<50% rule moves the main conversation when a side call arrives, which is exactly
   what Ariadne forbids (main + sidekick resident in place, append-only prefix, no KV shift). Take kindi's
   checkpoint/opening/turn mechanics; drop its parking.

---

## 11. What Limbo should take: ranked by expected payoff on this box
Box: RTX 4060 Ti 16 GB (Ada sm_89, 34 SMs, ~288 GB/s), PCIe 6.6 GB/s measured, Ryzen 7 3800X 8C/16T AVX2,
31 GB DDR4, SATA SSD. Bars: decode ≥ 40, prefill ≥ 500 tok/s.

1. **The exactness contract, from day one.** Batch-invariant products and recurrent steps (per-token
   reductions; fixed-order split-K merges); verify == decode by construction; GDN verify without write-back + commit
   of n accepted; one attention path for decode and verify; deterministic exp/rounding; an engine fingerprint over
   the arithmetic code keying every persisted state; a token-exact verify gate against a reference
   (llama.cpp CUDA on the same GGUF) with kindi's near-tie rule (first difference counts as a tie when the
   reference's top-2 gap < 0.01 nats, or another reference run gives our token, `CHANGELOG.md:194-196`).
   *Payoff:* it unlocks exact speculation, the largest multiplier measured on this box (1.54-2.10× in llama.cpp,
   per Limbo's own bench), and exact tape resume.
2. **Whole-step CUDA graphs with device-resident token, position, routing and expert slots**, one graph per token
   count T = 1..8, patching only dynamic nodes. *Payoff:* removes host latency from MoE decode, so routing never
   syncs to the host. Required to reach 40 tok/s on a MoE with offload.
3. **The speculation stack.** (a) A block drafter on target hidden states (DSpine targets the Qwen3.8-27B family,
   the operator's dense 27B; compare against DFlash2's measured 43.1 tok/s). (b) Verify products that decode each
   weight run once for N tokens (tokens on the mma N dimension). (c) Draft length from E(k)/cost(k), with cost
   including **the expert-union term for offloaded MoE**. (d) Cost-gated prompt lookup with a 1-token probe
   (agent edits: 2×). (e) A pause guard, measured per context band.
4. **The MoE offload mechanics.** A usage-count placement with share-per-byte greedy across layers (extend it to
   price KV pages and drafter bytes in the same currency); device-side LRU decode slots fed by in-graph pulls of
   missed experts; double-buffered whole-layer staging for prefill chunks; a CPU expert path with **union
   batching** (row group read once per batch, kept in L1/L2), persistent pinned pool, spin-then-futex. *Payoff:*
   this is the template for "compute where the bytes are (CPU, few tokens) vs move the bytes (GPU, many
   tokens)". The crossover point must be measured here (§12).
5. **CPU kernels for codebook formats on AVX2.** A signed-codebook table in L1 + `madd_epi16` (IQ3_XXS; the same
   idea applies to IQ1_M's grid, to be designed); a `pshufb` code→int8 LUT + `maddubs` with an offset correction
   (MXFP4/small codebooks); a pair-table-in-register `vpermps` GEMV (ternary). With runtime dispatch, an AVX-512
   path can be added later.
6. **Prompt-cache mechanics for hybrids.** A VRAM checkpoint slot (D2D ~1 ms) with idle-time, preemptible host
   copy in 1 MB pieces; a checkpoint at the message boundary before the generation prompt; dense checkpoints in
   the opening; disk openings and turns keyed by the fingerprint; K/V on disk in 1,024-token blocks shared across
   turns. All of this is compatible with the tape. Leave parking out.
7. **Attention.** Split flash-decode over position chunks with an in-order merge; q8_0 K/V with integer scores; read
   each KV row once per GQA group; tensor-core prompt attention with cp.async double buffering. *Payoff:* the
   long-context decode slope (kindi 27B: 63 → 44 tok/s from 1K to 90K versus llama.cpp 52 → 23.5). That
   matters most when the KV is partly host-resident, because then attention bytes are the cost.
8. **VRAM savers.** Embedding in mapped pinned host memory (one row per token); logits only for the last prompt
   token; tensor-at-a-time upload with `FADV_DONTNEED` (loads on a 2 GB-free host).
9. **Small tricks worth stealing.** The magic-float int32 accumulator (0x4B400000); lane-ordered activation
   quantization so packed 2-bit codes meet int8 activations without shuffles; the sign-negation trick for
   codebook bytes; top-k by threshold binary search; per-shape start-time autotune over the real tensors, cached;
   "new defaults only on measured compute capabilities".
10. **Fast head, as a drafter-side trick only.** The shortlist head is not exact, so it never goes on the main path.
    It can cheapen a drafter that uses the target's head (DSpine does).

**Leave:** the Python/CuPy/NVRTC stack and PTX rewriting; the Kepler and Pascal LUT GEMVs (the int8 tensor-core
path dominates on Ada; revisit only for a Vulkan Maxwell backend on the ThinkPad); two-card TP; parking-as-move;
DeepSeek-V4 specifics (engram, compressed KV, indexer); the Gemma/Bolt backend; the Hadamard fold unless Limbo runs
Prism Bonsai models (the operator has Bonsai-2-27B Q2_0, so keep it as an optional pre-op).

---

## 12. Open questions: measurements needed on this box
1. **int8 mma decode efficiency on sm_89** for IQ3_XXS and Q2_0/PTQ1_0 at batch 1: what fraction of 288 GB/s?
   kindi tuned only for 8.0 and 8.6. (EST ceiling for a 1.75-bpw 27B: ~46 tok/s plain.)
2. **Verify cost curve** T = 1..8 for the dense 27B IQ3_XXS and the A3B MoE on the 4060 Ti; it decides the draft
   lengths. kindi's 2×3080 curve does not transfer.
3. **The PCIe pull path.** On this slot, do in-kernel 16-byte pulls from mapped pinned memory beat copy-engine
   `cudaMemcpyAsync` for 1-20 MB experts, and how much do they slow a concurrent KV-page stream?
4. **CPU vs GPU crossover per expert.** AVX2 union-batched expert GEMV GB/s on dual-channel DDR4 with 6-7 threads,
   versus PCIe transfer plus GPU compute. At how many tokens per expert does "move the bytes" win? Measure it for
   IQ1_M (the coder's experts) and Q4_K.
5. **Expert-union factor** U(m), m = 1..8, for Qwen3.8-Flash-Next (10 of 512 per layer) on agent and code text.
   It feeds the draft-length cost model.
6. **DSpine vs DFlash2 vs MTP** on the operator's 27B (IQ3_XXS rather than ternary): accepted tokens per pass on
   Ariadne transcripts, VRAM cost, and decode tok/s against the 43.1 tok/s DFlash2 baseline.
7. **GDN checkpoint cost on the 27B.** 152 MB D2D into a slot (EST ~1 ms); a host copy at 6.6 GB/s (EST ~23 ms).
   Is the idle-time copy enough under Ariadne's tool-call cadence?
8. **Graph replay and node patching** with ~1,000+ nodes per step on Ada: µs per step for patching, and whether
   conditional nodes (CUDA 12.4+) can replace host-side plan selection.
9. **Cost-gated prompt lookup** on Ariadne's edit-heavy turns. The operator measured llama.cpp's ngram-mod as
   "dead" (76.9 vs 78.2 tok/s on the A3B); kindi measured 2× on file edits with cost gating. Which applies here?
10. **Blocking sync vs spin** on the desktop while a CPU expert pool is busy (kindi: 1-3% decode cost for a free
    core).
