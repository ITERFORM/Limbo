# Donor spec: llama-modchip v3.0 (clean-room)

Donor: `synchopate/llama-modchip` at `cfc906a` (2026-11-19), Apache-2.0, cloned read-only at
`/home/alca/projects/llama-modchip`. The engine is one patch, `patches/llama-modchip-v3.0.patch`
(12,959 lines, 51 files, applied to llama.cpp `de32b268c`), plus Python tooling. Everything below is
written in my own words. No donor code is reproduced. Format facts are stated exactly because Limbo
must read the same bytes.

Provenance notation: `P:n` means line n of the patch. Other paths are relative to the donor root.
Each claim is tagged:
- **READ**: I verified it in code.
- **MEASURED(doc)**: the donor's own measurement, named, on their hardware, not reproduced here.
- **CLAIMED**: asserted without evidence I could check.
- **DERIVED**: my own arithmetic on donor numbers.

---

## 1. What modchip is, and its module map

modchip is a CPU-first fork of llama.cpp. It runs large hybrid MoE checkpoints (Qwen3.8-Flash-Next
`qwen4exp`, DeepSeek-V4-Flash, Nemotron-3.5 30B-A3B) from system RAM with zero or few GPUs. Most of
the value is in the ggml CPU backend: SIMD dot kernels, the matmul and MoE scheduler, op threading,
and memory advice. The rest is scripts, a Python SSE gateway, a draftless drafter, and "MaxQ" model
checkpoints compressed offline. The kernels are written for Zen 5 AVX-512. **Our desktop has no
AVX-512** (Ryzen 7 3800X, Zen 2), so every item below states what survives on AVX2.

| Group | Files (patch lines) | What changes (READ) |
|---|---|---|
| Type registry | `ggml.h` P:3475-3528, `ggml.c` P:11704-11752, `ggml-common.h` P:3569-3594, `llama.h` P:11990, loader/quant P:12066-12140 | Adds FRACTAL_64=44 and VQ3_64=45 (ftype 30/31). Adds tensor flag 256 (`MODCHIP_CSA_SKIP`). Registers type traits. Loader zero-pads short GGUF arrays (P:12088). |
| Reference quant/dequant | `ggml-quants.c` P:11289-11641, `vq3_codebook.h` P:11753-11989 | Reference quantizers and dequantizers for FRACTAL_64 and VQ3_64. The 128x3 int8 VQ3 codebook. PDEP scale unpack and pshufb tables. |
| x86 dot kernels | `ggml-cpu/arch/x86/quants.c` P:3598-6829 | AVX-512 VNNI/BW paths for Q1_0, Q2_0, Q8_0, MXFP4, IQ2_XXS and IQ1_S. NTA prefetch added to many AVX2 kernels. "1xK" multi-column kernels (AVX-512 only). An AVX-512 `quantize_row_q8_K`. |
| Generic CPU kernels | `ggml-cpu/quants.c` P:9923-10632, `quants.h` P:10633-10722 | FRACTAL_64 and VQ3_64 dots (AVX-512 / AVX2 / NEON / scalar) and their 1xK variants. |
| x86 repack fixes | `arch/x86/repack.cpp` P:6830-6932 | GEMV output stride used `nr` where `bs` was correct (P:6839,6848,6857). Aligned `_mm_load_ps` on q8_K scale fields replaced with unaligned loads. |
| Repack policy | `ggml-cpu/repack.cpp` P:10723-10764 | An early `return` became `continue` in the repacked MUL_MAT_ID expert loop (P:10739-10740). CPU_REPACK is skipped for 3D tensors with at least 64 experts unless `LLAMA_REPACK_MOE=1` (P:10748-10760). |
| Graph executor | `ggml-cpu.c` P:6952-8596, `ggml-cpu-impl.h` P:6933-6950, `tiled.cpp` P:10765-10819 | Activation q8 cache and barrier elision. 1xK tiling inside `mul_mat` and `mul_mat_id`. MoE active-expert compaction and work stealing. Per-op thread counts. MoE top-k pruning and SwiGLU skip (opt-in, lossy). Dual-CCD split (opt-in). Core pinning (opt-in). An op profiler. |
| Ops | `ggml-cpu/ops.cpp` P:8597-9922 | Fused single-pass GATED_DELTA_NET. Static head partitioning. Flash-attention L3/NTA prefetch policy, forced-SWA hook (off by default), AVX-512 MQA8 MLA kernel, decode static partition. Flattened RMS norm. `nth_element` top-k. SSM conv restrict. DSV4 HC ops threading. |
| mmap | `llama-mmap.cpp` P:12033-12065 | `MADV_HUGEPAGE` on non-lazy file ranges only when they total at least 1 GiB, with boundaries rounded inward to 2 MiB (P:12041-12060). |
| Models | `deepseek4.cpp`, `nemotron-h*.cpp`, `qwen35moe.cpp`, `qwen4exp.cpp` P:12140-12621 | Per-layer expert width `n_ff_exp(il)`. Optional `ffn_down_exps.bias`. PLE key/value width = `ple_head_dim*ple_n_heads`. `LLAMA_MOE_TOP_K` override. A DSV4 rollback plane-index fix. |
| Speculative | `common/ngram-map.cpp` P:23-1467, `rc_predictor.h` P:1480-3132, `rc_stockfish.h`, `rc_trie_v2.h` | The upstream `ngram-simple` drafter is replaced wholesale by RouteCrack + Two-Zone + Stockfish-PVS. |
| Server | `server-context.cpp` P:12926-12959 | Checkpoint eviction never drops the oldest checkpoint (the system+tools prefix). It evicts the second-oldest instead. |
| Backends | cuda/vulkan/sycl/opencl `supports_op` P:10820-10838, P:11666-11703 | Types 44/45 are refused on every GPU backend. Metal has real shaders (P:10839-11269). |

---

## 2. CPU kernels: the main prize

### 2.1 Common structure (READ)
- **Weights times activations.** Activations are quantized once per matmul into a Q8 block format
  ("vec_dot_type"). Q8_0 is 32 elements with an fp16 scale. Q8_K is 256 elements with an f32 scale
  and int16 block sums per 16 elements. The integer dot runs per block and is scaled by
  `d_w * d_a` into an f32 accumulator.
- **The AVX2 integer idiom** is signed-by-signed bytes through the sign trick. Take `|w|` as
  unsigned, apply w's sign to the activation, use maddubs to get int16 pair sums, then madd (with
  ones, or with per-sub-block integer scale multipliers) to int32, then convert to float and FMA.
  Two independent accumulators per loop break the FMA dependency chain.
- **The AVX-512 VNNI idiom** avoids the sign trick. Bias the weight codes to unsigned (Q8: XOR 0x80;
  MXFP4: LUT values +16; Q2_0: raw codes 0..3; Q1_0: bits become {0,2}). Use VPDPBUSD into int32,
  then subtract a second VPDPBUSD of (bias constant x activation), which equals bias·Σa per block.
  This is exact in int32 with no int16 saturation, and it handles a weight of -128 correctly
  (P:3949-3973; tested with -128 in `test-quantize-fns.cpp`, P:12734+).
- **Bit expansion in Q1_0.** On AVX-512 the 64 weight bits are used directly as a mask register to
  produce 64 bytes of {0,2} in one masked broadcast (P:3726). On AVX2 the upstream path is kept:
  broadcast bytes, AND with a per-lane bit mask, then compare.
- **Invariant worth keeping.** On AVX2, psign/abs with -128 wraps. Activations must be quantized to
  ±127. Weight formats that can hold -128 need the bias-and-correct form or a widening path.

### 2.2 Per format

| Format | Layout (exact) | AVX2 path on Zen 2 | AVX-512 extra |
|---|---|---|---|
| **Q1_0** (41, upstream) | 128 weights per 18 B: fp16 d plus 16 B of bits. Bit j (LSB-first, byte j/8) set means +d, clear means -d. 1.125 bpw. | Upstream AVX2 kernel plus NTA prefetch 4 blocks ahead when `nt_stream` is active (P:3758-3764). | VNNI mask expansion (P:3709-3749). 1xK kernels. |
| **Q2_0 "signed-scale"** (42, upstream) | 64 weights per 18 B: fp16 d plus 16 B. Code j sits at byte j/4, bits 2·(j%4). Value = (code-1)·d, giving {-1,0,+1,+2}·d. **d may be negative**, which mirrors the range to {+1,0,-1,-2}·abs(d). Decoders must keep the sign (P:3784). | Upstream unpack: widen 4 bytes to dwords, OR the left-shifted copies, mask 2 bits. Two blocks per step on two FMA chains, plus prefetch (P:3842-3853). | VNNI path (P:3773-3833). 1xK. |
| **Q8_0** | Upstream. | AVX2 kernel plus NTA prefetch (P:3977-3983). | Unsigned-bias VNNI, exact at -128 (P:3949-3973). |
| **MXFP4** (39, upstream) | Upstream: E8M0 shared exponent per 32 elements, 4-bit FP4 codes looked up through a 16-entry table. | Upstream LUT path plus NTA on weights and T0 on activations, 6 blocks ahead (P:3911-3914). | LUT+16 bias with VNNI (P:3873-3902). |
| **FRACTAL_64** (44; was **42** in modchip up to v2.x) | 64 weights per 30 B: fp16 d; `scales[4]` holds 8 nibble sub-scales (byte b: low nibble = s_b, high nibble = s_{b+4}); `qs[16]`; `qh[8]`. The 3-bit index of weight j = 16g+k (g 0..3, k 0..15) has its low 2 bits at qs[k] bits 2g..2g+1 and its high bit at qh[k mod 8] bit 2g+(k≥8). Codebook {-14,-8,-4,-1,+1,+4,+8,+14}. Value = d·(4+s_{j/8})·cb[idx]. 3.75 bpw (P:3577-3584, P:11429-11455). | A real AVX2 kernel (P:10036-10100). Broadcast qs to both lanes. Per-dword variable right shifts by 0/2/4/6 select group g. The qh bit is placed at position 2 with per-qword shifts. A pshufb 8-entry signed LUT maps the index to the value. Then the sign trick, maddubs, and madd against sub-scale multipliers (4+s) built by pshufb-broadcasting the nibble scales into int16 lanes. The donor reports the variable-shift version +55% faster than scalar unpacking (MEASURED(doc), ARCHITECTURE.md:585-588, then called type 42). | Mask/permute unpack. 1xK. |
| **VQ3_64** (45; was **43** in modchip up to v2.x, the same id as upstream F8_E4M3) | 64 weights per 24 B: fp16 d plus `qs[22]`. For k = 0..21, the low 7 bits are codebook index c_k (0..127) and bit 7 is bit k of a 24-bit scale word S. Bits 22 and 23 of S are the bit 6 (index MSB) of qs[20] and qs[21]: the two 2-D tail codes give up their half-codebook bit to carry scale bits. Sub-scale m_s = ((S>>3s)&7)+2, in {2..9}, applies to weights 8s..8s+7. For k < 20, the triple (w_k, w_{k+22}, w_{k+44}) = cb[c_k]. For k = 20,21, the pair (w_k, w_{k+22}) = cb[c_k][0..1]. Value = cb·G·d·m, with G = 8.6373016238e-3. The codebook is 128x3 int8 (`vq3_codebook.h`). 3.0 bpw (P:3586-3593, P:11585-11615, P:11763). | **Scalar.** A 22-iteration scalar loop gathers codebook bytes into a 64-byte stack array, then runs an AVX2 dot (P:10392-10446). Gather-bound on Zen 2. | VBMI two-table byte permutes keep the 384-byte codebook in 6 zmm registers (P:10331-10376). This is the only fast path. |
| IQ2_XXS / IQ1_S / IQ1_M (upstream) | Upstream. | IQ1_S: maddubs(ones, sign(a,w)) replaces a helper (P:4267-4268). IQ2_XXS: NTA prefetch. IQ1_M: an empty `if` left over from a deleted prefetch (P:4284-4285). The AVX2 grid lookup is unchanged. | IQ2_XXS sign generation via PDEP plus SWAR parity, applied as a masked negate of the activations instead of a sign-table lookup (P:4146-4184). |
| K-quants Q2..Q6_K | Upstream. | NTA prefetch of super-block i+2 at offsets +64/+128/+192 (P:3991-4136). | Q5_K: one 512-bit maddubs over two 32-byte halves (P:4078-4092). |

Quantizer ideas (DERIVED from P:11320-11427 and P:11463-11583). These matter only if Limbo ever
writes these formats:
- **Fractal-64:** per 8-weight sub-block, a least-squares scale refinement (two rounds).
  `d = max_subscale/19`. Each sub-scale is searched ±2 around its estimate for minimum squared error.
- **VQ3_64:** scale `d = 0.88·max|w|/9`. Sub-scales m in {2..9}. Three alternations between
  nearest-codeword search (expanded L2 distance using precomputed squared codebook entries) and
  least-squares sub-scale refit. The tail codes are searched only in the half of the codebook that
  their stolen bit allows.

### 2.3 Multi-column "1xK" kernels (READ; AVX-512 only)
The kernel decodes a weight block **once** into registers and dots it against K activation columns
(K in {8,4,3,2}) with K accumulator sets (P:4305-4310 onward). Dispatch walks 128-row L2 tiles and
splits each column remainder into steps of 8/4/3/2 so that no 1-column leftover remains (rules: use
8 if rem≥8 and rem≠9; 4 if rem≥4 and rem≠5; 3 if rem is 3 or 5; otherwise 2; P:7262-7359). The same
appears inside `mul_mat_id` for an expert's token bucket (P:7523-7612). It targets prefill tiles and
speculative-verify batches. **On AVX2 every 1xK entry point falls back to K separate single-column
dots** (P:6814-6823, P:4467-4469). On `_WIN32` the 1x8 kernel is split into two 1x4 calls. So the
3800X gets none of this. The idea is sound and must be written fresh for AVX2. With 16 ymm
registers, K=4 fits and K=8 spills.

### 2.4 Activation quantization and the "q8 cache" (READ)
- Each matmul re-quantizes src1. modchip keeps an **8-slot by 64 KiB** quantized-activation cache in
  the threadpool (P:7010-7023). It is keyed by src1 data pointer, vec_dot_type, shape and strides.
  An entry is invalidated whenever a node writes a tensor whose base range overlaps the cached base
  (P:8029-8037, P:8143-8156). On a hit the quantize step and its barrier are skipped. This covers
  gate/up/QKV projections that share one normed input.
- The cache metadata is **replicated per thread**, not shared. Every thread walks the same node
  sequence, so all threads reach the same hit/miss decision without communicating. This is a
  lock-free pattern worth reusing: deterministic replicated schedule decisions in place of shared
  state.
- **MoE id reuse:** consecutive `MUL_MAT_ID` nodes with the same ids tensor (gate_exps, up_exps)
  reuse the expert to row grouping and skip the regroup barrier (P:8216-8250).
- **Hazard they hit:** the cache must not be filled when a repack "extra" buffer owns the node. The
  repack path quantizes into the normal work buffer, which poisons later readers. Their example is
  the MTP shared experts after a Q2_K routed MoE (P:8166-8172). Lesson for Limbo: a quantized
  activation has exactly one owner and buffer, declared by the plan, never inferred from pointers.
- Parallel quantization is split by rows when there are at least as many rows as threads, and by
  columns of one row otherwise (P:7411-7420).

### 2.5 Matmul and MoE scheduling (READ)
- **Decode chunking.** When `nr0 ≥ 8·nth`, rows are split statically by thread (P:7494-7505). Slice
  size is rounded to 16 rows when `nr0 ≥ 16·nth` (P:7858). Inner `blck_1` is capped at the column
  count of one src0 matrix, so 3D tensors do not hop across 2 MiB matrices (P:7238-7242).
- **MoE active-expert compaction.** The list of experts that received at least one row is built once
  (P:7646-7648, P:7744-7750). Iteration covers only those, never all `n_as` (512 on Flash-Next).
- **Small-batch MoE (2..7 tokens, at least 256 rows).** Work items are (active expert x row slice)
  with `n_sub` 4 or 8, slices aligned to 16 rows, handed out through **one global atomic counter**
  (P:7798-7828).
- **Barrier elision.** When the quantized activation is valid, the ids grouping is valid, there is
  one token, and the rows are few, the counter-reset barrier is skipped (P:7472-7480, P:7766-7768).
- **NT stream.** Weight NTA prefetch inside expert rows is enabled automatically for MoE with at
  least 64 experts and at most 8 tokens. It is on by default; `LLAMA_NT_STREAM=0` turns it off
  (P:7779-7786, P:7621-7629).
- **Per-op thread counts.** GDN uses min(threads, heads x seqs) (P:7896-7900). GET_ROWS goes
  parallel above 16k elements (P:7910-7918). SOLVE_TRI and DSV4 HC ops are threaded when there are
  at least 4 tokens.

### 2.6 Prefetch policy and cache isolation (READ + MEASURED(doc))
Streamed weights are read once per token, so they are prefetched with `_MM_HINT_NTA` 2-6 blocks
ahead to keep them out of L3. Activations are prefetched with T0. During decode, KV rows older than
`LLAMA_KV_L3_WINDOW` (default 2048 tokens from the end) are prefetched NTA, while the recent window
is prefetched T0 (P:9136-9160). With a sparse mask the prefetcher skips ahead to the next unmasked
row, with a scan budget of 128.

Donor PMU numbers on a 16C Zen 5 at 28k context: L3 misses -5.8%, L1d misses -7.7%, cycles per token
**-9.0%** (MEASURED(doc), ARCHITECTURE.md:240-246). Zen 2 has a different, non-inclusive L3 per
4-core CCX, so the gain must be re-measured here.

### 2.7 Repack fixes, and why zero-copy 3D MoE matters (READ)
- **CPU_REPACK** rewrites weights into interleaved 8x8 tiles in **anonymous memory**. For a MoE that
  means copying every expert, which costs multi-GiB of RssAnon on top of the page cache. v3.0 skips
  repack for 3D expert tensors (at least 64 experts). Cold experts stay **mmap-backed and zero-copy**
  and run through the unrepacked MUL_MAT_ID path above. RSS then counts only touched pages
  (P:10748-10760).
- **Real upstream bugs fixed (READ):**
  - The repacked MUL_MAT_ID loop returned on the first expert with an empty row range for this
    thread, which skipped every later expert (P:10739-10740).
  - Repacked GEMV wrote output with stride `nr` where it should use `bs` (P:6839-6857).
  - Aligned loads were used on q8_K scale fields that are not 16-byte aligned.
- Donor measurement with repack on, before v3.0: q4_0 8x8 VNNI tiles cut retired instructions by
  30% (MEASURED(doc), ARCHITECTURE.md:249-262, 306-309). That was AVX-512 VNNI on dense Nemotron
  tensors.

---

## 3. GATED_DELTA_NET and flash attention

### 3.1 GDN (READ, P:9314-9612)
The recurrent state is S_v x S_v per head, stored transposed so that each row j is contiguous. The
old code made four passes per token over the 64 KiB state (S_v=128): decay, dot with k, rank-1
update, dot with q. The new code makes **one pass per row j**:
1. Load the row and multiply by the decay (scalar `exp(g)`, or the per-key vector for KDA).
2. Store it and accumulate its dot with k.
3. Compute `δ_j = (v_j - row·k)·β`.
4. Add `δ_j·k`, store, and accumulate the dot with q, giving `out_j = scale·(row·q)`.

For token 0 the input state is read directly, so the initial copy is fused away. Rows within a token
are independent. There is an AVX2+FMA path (16-wide in two halves with two accumulators,
P:9493-9556) and a fully unrolled AVX-512 path for S_v=128 that keeps k, q and the gate in
registers.

Threading changed from dynamic chunks (barrier plus atomic) to a **static split of (head x seq)
across threads with no barrier or atomic** (P:9601-9612). This is the claimed "inner barrier
eliminated". Scratch per thread is padded to 16 floats. Memory traffic per head per token drops from
about 4 reads and 2 writes of the state to 1 read and 1 write (DERIVED).

### 3.2 Flash attention (READ)
- **Decode** (`neq1==1`) uses static partitioning with no chunk counter or barrier. **Prefill** keeps
  dynamic `4·nth` chunks to balance causal work (P:9190-9265).
- The split-KV path (single query, at least 512 KV, f16/f32) is now used only when there are fewer
  heads than threads (P:9180-9183). Otherwise heads are split statically.
- **MQA8 for MLA** (AVX-512 + F16C; DK=DV=512 with one KV head): each 16-element f16 K/V segment is
  converted once and FMA'd into 8 query heads' accumulators, with online softmax per head
  (P:8914-9100). Donor measured 2.0x on FLASH_ATTN_EXT, 415 to 208 ms (MEASURED(doc),
  ARCHITECTURE.md:603). The idea, **one KV read serving all query heads of its group**, is exactly
  what GQA on the CPU needs. modchip has it only for MLA on AVX-512. The upstream generic GQA path
  re-reads K/V for every query head.
- **Forced SWA, sinks, interleave:** `LLAMA_ATTN_WINDOW` (default 0 = off), 16 sinks, stride. It
  skips out-of-window KV rows in models trained with full attention. Lossy. Off by default
  (P:9108-9120).

---

## 4. Threading model

- **Non-convoy** (READ + MEASURED(doc)): use `-t N-1 -tb N-1 --poll 0 --poll-batch 0`, leaving one
  physical core for HTTP and the OS and disabling the busy-poll between graph launches. With N busy
  workers on N cores, any OS or HTTP thread deschedules one worker, and **every graph barrier waits
  for it**. Donor: +25.9% pp512, +31.3% tg128 (MEASURED(doc), README.md:104, FAQ.md:213; 16C Zen 5,
  32GB tier, no spec).
- **Barrier implementation.** Under OpenMP the normal path uses `omp barrier`. The custom
  atomic-counter barrier is used only when the dual-CCD mode is active (P:7027-7046).
- **Topology** (READ, `scripts/llama_modchip.py:400-440`, `_cpu_topology.py`): L3 domains are read
  from sysfs. On parts with more than 2 CCDs and more than 24 cores, decode threads are capped at
  about 4 per CCD and the taskset keeps an equal count per CCD. Donor: Threadripper 7985WX at 32
  threads gave 80.14 tok/s against 73.84 at 64 threads (MEASURED(doc)). Physical cores only (no SMT
  siblings). Optional 1:1 pinning via `MODCHIP_PIN_CORES`, only when the thread count is 16 or 32
  (P:8575-8595).
- **Dual-CCD "vertical split"** (`LLAMA_DUAL_CCD_SPLIT`, P:8456-8565): the graph is cut at a named
  layer. CCD0's threads run the first part while CCD1's threads spin, then they swap. CPU ids are
  hard-coded for a 16C/32T Zen 5. Within one graph there is **no concurrency**: the idle half
  spin-waits. Any benefit can only be L3 locality. Not a pipeline.
- **Static vs dynamic** (READ, across GDN, FA and matmul): decode ops (one token) use static
  per-thread partitions with no shared counter and no barrier. Dynamic chunking is kept for prefill,
  where the work is uneven.
- **Mono vs dual CCD** on 16C Zen 5 decode: 20.4 tok/s on 8 cores vs 25.3-27.3 on 16 cores pinned.
  One CCD of 8 cores already uses about 82% of the DDR5 bus (MEASURED(doc), ARCHITECTURE.md:328-338).
  Decode is bandwidth-bound and cores beyond the bandwidth knee buy little.

---

## 5. Memory

- **THP** (READ, P:12041-12060): `madvise(MADV_HUGEPAGE)` is applied only to the **non-lazy** file
  ranges of the model mmap, and only if they total at least 1 GiB. Interior range boundaries are
  rounded **inward** to 2 MiB, so a huge page never straddles into a lazy range. A huge-page fault
  there would pre-fault 2 MiB of a randomly accessed table such as the PLE.
- **Donor THP evidence** (MEASURED(doc), ARCHITECTURE.md:303-310): dTLB misses -98.3% (18.42M to
  0.31M per token) and page-walk time 14.0 to 0.8 ms per token. **Caveat (READ):** that measurement
  was taken with the **repacked anonymous** arena (AnonHugePages 19 GiB) and a `ggml_aligned_malloc`
  hugepage patch. **Neither is in the v3.0 patch.** In v3.0, experts stay file-mmapped, and file THP
  on ext4/btrfs needs read-only file THP and khugepaged collapse, which is slow and not guaranteed.
  On this desktop: THP is `madvise`, `FilePmdMapped: 0`, and models live on **btrfs on the SATA SSD
  `/dev/sdc2`** (READ, `/proc/meminfo`, findmnt, 2026-11-19).
- **Lazy PLE shard** (READ + MEASURED(doc)): Qwen3.8-Flash-Next carries a 320,001,536-row
  per-layer-embedding (PLE) n-gram table. MaxQ stores it in shard 2 as `[256, 320001536]` Q1_0,
  which is 36 B per row and 10.73 GiB in total (FAQ.md:201-206). Per token only a handful of rows
  are read, so the shard is **demand-paged** and never prefetched. Shard 1 (24.98 GiB trunk) stays
  hot. Result: VmHWM 26.91-27.44 GiB under a cgroup with `memory.max=28.5G`, swap 0, oom_kill 0, for
  a 35.71 GiB file. The gap is the cold PLE pages (MEASURED(doc), README.md:42, FAQ.md:206). The PLE
  key/value input width is `ple_head_dim*ple_n_heads` (P:12583-12588): the PCA basis is folded into
  the next linear layer.
- **Zero-copy experts:** see 2.7. `mmap` plus no repack means RSS grows only with experts actually
  touched.

---

## 6. /dev/shm hybrid state snapshot (READ + MEASURED(doc))

- **Mechanism:** this is stock llama-server slot save/restore (`POST /slots/0?action=save|restore`)
  with `--slot-save-path /dev/shm`, so the file sits on tmpfs. The snapshot holds the KV of the
  attention layers, all GDN/SSM recurrent states, and the context checkpoint. Restore is a memcpy
  from tmpfs (`scripts/prewarm_agent_prefix.py`).
- **Numbers:** Qwen3.8 slot 0 at 512 tokens is 233.14 MiB (36 GDN states of 48x128x128 fp32, 12
  attention layers of q8_0 KV, one checkpoint). Restore takes 38.49 ms on average (6.11 GiB/s),
  about 162x faster than the 6.18 s prefill. With `--ctx-checkpoints 0` it is ~19 ms for 120.6 MiB.
  Greedy continuation is bit-exact (SHA-256 compared) (MEASURED(doc), README.md:185). A 22,237-token
  agent prefix (Nemotron) is 85 MB and restores in 18.9 ms.
- **Patch** (P:12936-12956): when checkpoints exceed `n_ctx_checkpoints`, eviction removes the
  **second-oldest**, never the oldest (which is the system+tools prefix). This is a "pin the base
  prefix" rule.
- **Not in the engine:** HARP-Cache (`kv_cache/harp_cache.py`) is a Python gateway-side prefix
  tracker for telemetry; its own docstring says so. EAKV (`kv_cache/eakv_quantizer.py`) is a Python
  prototype. Neither is wired into the patch (READ: `grep` finds no EAKV in the patch).

---

## 7. Speculative decoding

- **Cost model on CPU MoE** (MEASURED(doc), ROUTECRACK_TECHNIQUES.md:9-20):
  `T_batch(D) ≈ T1·(1+0.30·D)`. Each drafted token routes to its own experts, so verify cost grows
  with draft length. Net gain is `A - 0.30·D`, where A is accepted and D drafted. Break-even per-step
  acceptance is `p > B_cpu - 1 = 0.58` with `B_cpu ≈ 1.58` for D=5 (ARCHITECTURE.md:351-354). With a
  recurrent state, each rejection also costs a state restore plus replay. Nemotron's state is
  18.4 MB per slot, about 0.28 ms per memcpy.
- **RouteCrack drafter** (READ, P:23+, `speculative/`; CLAIMED accuracies): draft-model-free. It
  stacks:
  - (a) a **Two-Zone in-context matcher** over a 16k-token hot tail and a 24k-token prefix zone,
    which copies continuations of earlier occurrences;
  - (b) an offline 8-order FNV-64 n-gram hash cascade (tables trained per model vocabulary on 3.7M
    Nemotron tokens);
  - (c) an online arena trie, PPM blending, skip-grams, and 300 rules plus an 11,382-entry macro
    codebook;
  - (d) a linear probe on the logits of about 48-200 "hot tokens";
  - (e) **Stockfish-PVS**: three 64 KiB Zobrist transposition tables (5/4/3-gram suffix keys) that
    learn **online from accept/reject feedback**, using a "history gravity" update
    (`h += bonus - h·|bonus|/512`) and refuted-move records.

  Per-depth certainty gates are strict: 84%/92%/96% for d=0/1/2. The horizon extends to 7 only on
  "singular" evidence (a 2-hit in-context run, a macro chain, or TT agreement).
- **Numbers:**
  - Strict gates: 85.9% token precision, +3.4% wall on the hybrid model from the cascade alone
    (ARCHITECTURE.md:362-368).
  - Live bursts: 60.4 tok/s peak at mean length 6.90 on code edits; 18.7-28.6 tok/s average on fresh
    text (ARCHITECTURE.md:431-442; MEASURED(doc), Zen 5 CPU only).
  - On Qwen3.8, **the model's own MTP head wins** in 6/6 cold pairs: 12.66 tok/s, 93% accept,
    1.99x over none on MaxQ-64GB, against 10.58-12.82 tok/s for PVS (README.md:177-184).
  - For Qwen3.8 the offline tables must be **disabled** (`RC_TABLES_DIR=/nonexistent`); they are
    Nemotron-vocab.
- **Where each wins:** MTP/DFlash when the checkpoint carries a head. Draftless in-context copying on
  edit and refactor loops, or for models without a head.

---

## 8. Hybrid GPU/CPU

- **Tensor-role split** (READ, docs/HARDWARE_OPTIMIZATIONS.md:42-52): `-ngl 99 --n-cpu-moe N` keeps
  the whole dense backbone, KV and SSM on the GPU and places N layers of routed experts in RAM. It
  beats layer-index splitting, which would push attention onto the CPU.
  - Donor community result: RX 9070 XT 16 GB plus a 7945HX3D on Nemotron Fractal-MXFP4 at 256k:
    75.1-80.8 tok/s with `--n-cpu-moe 30` (MEASURED(doc), unverified submission).
- **"Live-VRAM solver"** (READ, `windows/modchip_win.ps1:271-291`):
  `cpu_moe_layers = total - floor((free_vram - base - 550 MiB) / per_layer_MiB)`. Per-model
  constants are hard-coded; unknown files default to base 4200 MiB and 500 MiB per layer. It is
  arithmetic only, with no runtime feedback.
- **GPU kernels:** the patch adds **no CUDA or Vulkan compute** for any type. It only makes CUDA,
  Vulkan, SYCL and OpenCL refuse types 44/45 (P:10820-10838, P:11666-11703). Metal gets dequant and
  mul_mv shaders (P:10945-11269). The Vulkan SSM_SCAN and MXFP4 shaders cited in the docs are
  upstream.
- For DeepSeek-V4 on a 12 GB card, CPU alone is faster than partial offload (READ,
  docs/WINDOWS_PORT.md:56-66). Only 5-7 of 43 expert layers fit, and CPU-only types force a PCIe
  sync per layer.

---

## 9. MaxQ compression (offline; enough to know what Limbo must run)

The compression pipeline is **not in the repository**: only results and descriptions. CLAIMED:
- **ConeCollapse:** shrink each routed expert's intermediate width (Qwen3.8 interior layers 640 to
  512; DSV4 2048 to 1792) by calibration ridge regression. It solves a Woodbury "DualGram" system on
  NC=384 calibration activations per expert in FP64. A per-expert `ffn_down_exps.bias` compensates
  the mean of the removed part. Boundary layers stay full width.
- **PLE compression:** PCA-Wiener projection of PLE rows to D=256, then 1-bit RaBitQ stored as Q1_0
  (26.82 to 10.73 GiB). The basis is absorbed into `ple_key` and `ple_value`.
- **Sensitivity-allocated standard types** for experts: IQ2_XXS, IQ1_M, IQ1_S, MXFP4, signed-scale
  Q2_0.

**What Limbo must support to run MaxQ v3.0 files** (READ from the loader and model changes):
- Q1_0 and signed-scale Q2_0.
- IQ1_S, IQ1_M, IQ2_XXS, MXFP4.
- **Per-layer expert width** (`n_ff_exp` as a per-layer array; short arrays zero-padded, P:12088).
- **Optional per-expert down bias.**
- **PLE width from `ple_head_dim·ple_n_heads`.**
- Lazy PLE paging.

Types 44/45 are needed only for the Nemotron VQ3/Fractal "CPU-Turbo" files.

**Quality as published:**

| Model | Size | PPL | KLD |
|---|---|---|---|
| MaxQ-64GB | 53.99 GiB | 3.43 | 0.497 |
| MaxQ-32GB | 35.71 GiB | 5.81 | 1.007 |
| ISTA GSQ-RCO-Coder-IQ1_M (the coder the operator serves on Strata) | 54.40 GiB | 5.09 | 0.930 |

These were measured against the donor's own BF16 reference on their own 40x512 wiki-alt chunks
(MEASURED(doc), README.md:131-137). **MaxQ-32GB is worse than the IQ1_M coder on PPL.** Only the
64 GB build beats it, and it does not fit 31 GB of RAM plus 16 GB of VRAM without streaming.

---

## 10. Flaws and limits (honest reading)

1. **AVX-512 first.**
   - The multi-column kernels, VNNI dots, VBMI VQ3 decode, IQ2_XXS/IQ1_S fast paths, AVX-512 Q8_K
     quantizer, MQA8 FA and RMS-norm vectorization are all AVX-512.
   - **On Zen 2 what remains:** NTA prefetch, fused/static GDN, static decode partitions, MoE
     compaction and work stealing, the q8 cache, threading discipline, and the Fractal-64 AVX2
     kernel.
   - VQ3_64 on AVX2 is a scalar gather.
2. **Type-id history.** Before v3.0, Fractal-64 used id 42 (upstream Q2_0) and VQ3 used id 43
   (upstream F8_E4M3). **Ids 42/43 in a GGUF are ambiguous by producer.** The block size
   disambiguates: Q2_0 is 18 B per 64, Fractal 30 B per 64, VQ3 24 B per 64, F8_E4M3 1 B per
   element. Limbo must check tensor byte sizes against the type, never trust the id alone.
   ARCHITECTURE.md still says VQ3_0=43 (doc drift, ARCHITECTURE.md:619).
3. **Upstream drift.** One monolithic patch onto a pinned commit. Many hooks are name-string based
   (`ffn_moe_*`, `l_out-%d`, layer ids parsed from tensor names, P:8216-8226, P:8466-8475). The
   shared-memory and mmap'd config files are read inside hot paths (P:8786-8892). These patterns are
   brittle and must not be carried over.
4. **Lossy speedups sit beside exact ones.** MoE top-k pruning (`LLAMA_MOE_TOP_K`, an mmap'd config
   with L2-norm rescale), SwiGLU top-k row skip, and forced SWA are all opt-in but marketed with the
   exact gains. Top-2 collapses the model; the donor admits it (ARCHITECTURE.md:660-668).
   - The +38.5% decode on Qwen3-30B came from top-k 4 of 8 (ARCHITECTURE.md:489), which is lossy.
5. **Evidence quality.**
   - The quality gates ("100/100", "9.55/10") are self-scored on small task sets.
   - The leaderboard rows are community submissions with mixed models and settings.
   - Speed attributions mix several changes; the donor footnotes this (README.md:183).
   - The THP caveat in section 5.
   - The long-context decode collapse is attributed to DRAM bandwidth, but their own numbers say
     otherwise (DERIVED): Qwen3-30B-A3B goes from 32.5 to 5.9 tok/s between 311 and 13,443 tokens
     (benchmarks/KV_SCALING_BENCHMARK_REPORT.md), which adds ~138 ms per token. The KV they cite
     (372 MB per token) costs about 5 ms at 75 GB/s. The real cost is CPU attention compute plus
     **K/V re-read per query head** (8 query heads per KV head means about 8x traffic through the
     caches), on quantized KV.
6. **Dead or odd code:** an empty prefetch `if` in IQ1_M (P:4284); the dual-CCD "split" spins the
   idle half.
7. **Single request.** The engine runs one slot, and the model is chosen to fit RAM. Nothing in
   modchip addresses concurrent conversations.

---

## 11. Conflicts with the other donors

1. **CPU cores: compute vs feeding the GPU.**
   - modchip wants N-1 cores in GEMV with barriers.
   - Strata/kv-streaming need host threads that react fast: CUDA launch and sync, the io_uring pack
     reader, KV page staging, the HTTP server, the tokenizer, and the operator's desktop.
   - modchip's convoy finding applies twice over: one descheduled worker stalls a barrier, and one
     stalled launch thread idles the GPU.
   - **Resolution:** the planner reserves cores. On the 3800X: GPU submit/IO plus OS/HTTP get about
     2 logical cores. CPU expert compute runs on the remaining physical cores, never busy-polls
     (futex park), and uses static partitions with work stealing only inside a step.
2. **DDR bandwidth is one pool.** CPU GEMV over cold experts, PCIe DMA from pinned host memory
   (6.6 GB/s measured), SSD reads into the page cache, and the desktop all draw on dual-channel DDR4
   behind a single Zen 2 CCD link. modchip assumes it owns the bus.
3. **Zero-copy mmap vs pinned DMA source.**
   - modchip keeps experts as page-cache mmap. Strata page-locks a RAM tier for DMA. The same bytes
     cannot be both cheaply.
   - **Resolution:** one anonymous, 2 MiB-backed, page-locked RAM tier loaded from the pack. The CPU
     kernels and the GPU copy engine both read it.
   - The page cache and mmap remain only for the cold SSD tier and random-access tables (PLE),
     never with `MADV_HUGEPAGE`.
4. **Layout.** CPU repack (8x8 interleave) and GPU-friendly layouts (kindi and Strata packs)
   disagree. Two copies cost RAM we do not have.
   - **Resolution:** one canonical block layout per tensor class, chosen by the planner. CPU kernels
     must run on that layout; for experts that is GGUF block order, which modchip shows works well
     unrepacked.
5. **Who owns the graph.** modchip's wins come from graph-executor hacks keyed on pointers and names.
   In Limbo, the planner emits explicit facts:
   - groups that share an input (quantize once);
   - expert-id groups reused across gate, up and down;
   - per-op partition mode (static or dynamic);
   - barrier needs.
6. **KV placement.**
   - modchip computes attention on the CPU over RAM KV.
   - kv-streaming keeps KV pages in pinned host memory and **moves** them to the GPU each step;
     past the stage it pays the streaming cost (memory: 54.6 to 70.4 ms per decode step, 39k to 51k).
   - **Combined:** the CPU computes partial attention (max, sum, accumulator) over host-resident
     pages, the GPU computes the resident pages, and a log-sum-exp merge combines them. This needs a
     **GQA-sharing AVX2 flash-decode kernel**: one K/V read for all query heads of its group. That is
     the MQA8 idea generalized; modchip lacks it for GQA and on AVX2.
7. **Spec decoding policy.** The verify cost slope differs by where the work lands:
   - flat-ish for GPU-resident dense layers;
   - about +0.30·T1 per drafted token when cold experts are computed on the CPU (modchip's model);
   - plus the GDN rollback either way.

   Draft length must be a per-step planner decision, not a fixed n_max. MTP/DFlash (measured 1.54x
   to 2.10x here) come before any draftless drafter.
8. **Format ids.** Old modchip GGUFs reuse 42/43 with different meanings, and 44/45 have no GPU
   kernels anywhere. Kindi and Strata packs re-lay bytes. Limbo's loader must identify formats by
   (type, block bytes) and producer metadata.

---

## 12. What Limbo should take, ranked by payoff on this box

Box facts (READ, 2026-11-19):
- Ryzen 7 3800X, 8C/16T, AVX2, no AVX-512.
- **Two 16 MiB L3 domains: CCX0 = cpus 0-3,8-11; CCX1 = cpus 4-7,12-15.** L2 512 KiB and L1d 32 KiB
  per core.
- 31 GB DDR4. RTX 4060 Ti 16 GB. PCIe measured 6.6 GB/s.
- THP `madvise`. Models on btrfs/SATA.
- The ThinkPad (Maxwell-class, 4 GB, Vulkan) gets nothing GPU-side from modchip.

| Rank | Take | Why it pays here |
|---|---|---|
| 1 | **CPU path for cold experts during decode and verify (N=1..8).** Includes: active-expert compaction; (expert x 16-row slice) tasks from one atomic counter; quantize activations once per shared-input group; reuse the expert grouping across gate, up and down; NTA prefetch of weight rows; and **fresh AVX2 multi-column kernels (K=2/3/4)** for verify batches, for the formats in use (IQ1_M, IQ2_XXS, Q2_K/Q4_K, Q8_0, MXFP4, plus Q1_0/Q2_0 if MaxQ). | DERIVED: coder-IQ1_M experts are about 1.95 MiB each (23,981 MiB over 48 layers x 256). A token with all 10 experts cold per layer touches about 936 MiB. That is about 149 ms over PCIe at 6.6 GB/s, against about 25 ms from DDR4 at an assumed 38 GB/s, so **computing where the bytes are wins about 6x** for decode. With a 60-75% GPU hit rate the CPU share is 234-374 MiB per token, about 6-10 ms. |
| 2 | **Threading discipline.** No busy-poll. Reserve cores for GPU submit, IO and OS. Static per-thread partitions for single-token ops. Dynamic only for prefill. Physical cores only. Place CPU compute threads per **CCX** so each one's slice stays in its 16 MiB L3. | Donor: non-convoy alone gave +25.9% pp and +31.3% tg (doc). It also keeps the GPU fed. |
| 3 | **GQA-sharing CPU flash-decode over host-resident KV pages**, merged with GPU partials (from the MQA8 idea plus the analysis in section 10). | Replaces moving KV pages over PCIe with compute in DDR4. kv-streaming's measured cost of moving past the stage is the target. |
| 4 | **2 MiB pages for the RAM tier.** Anonymous `MADV_HUGEPAGE`, or hugetlb/memfd, then pin. Never huge pages on lazy or random tables. Boundaries rounded inward. | Donor: -98% dTLB and about 13 ms per token saved on a 19 GB arena (doc). Random expert jumps over 20+ GiB on 4 KiB pages exceed the 2k-entry L2 TLB reach of Zen 2 (8 MiB). |
| 5 | **Fused single-pass GDN state update** (one read and one write of the state per token, static head partition). Port the same idea to the CUDA GDN kernel. | Flash-Next has 36 GDN layers. The 64 KiB state per head pass is pure bandwidth. |
| 6 | **Format facts and loader rules from section 2.2 and section 9:** signed-scale Q2_0, Q1_0, per-layer expert width, expert down bias, PLE width, (type, bytes) disambiguation, zero-padded short arrays. | Needed to load MaxQ-class and future files. |
| 7 | **Restart snapshot:** device state to host to tmpfs file in about 40 ms per 230 MiB, and **pin the base-prefix checkpoint** so eviction never drops it. | Fits Ariadne's tape: the snapshot serves cold starts only. A side call must never move a conversation (tape rule). |
| 8 | **Online accept/reject-learned drafter** (Zobrist TT with history gravity) plus in-context two-zone copy, as a fallback for models without MTP. Use the break-even formula `p > B-1` and `gain = A - c·D`, with c computed per step by the planner. | Small gains (+3-5%) except on copy-heavy edits. Below MTP and DFlash. |

**Leave:**
- VQ3_64 and Fractal-64 as runtime targets (no AVX2 speed for VQ3, no GPU kernels, legacy id
  collisions). Revisit only if the operator wants those files.
- MoE top-k pruning, SwiGLU skip and forced SWA as defaults. They are lossy; at most labelled
  experiments.
- The dual-CCD split.
- Name and pointer-keyed graph hacks, and config mmaps read in hot loops.
- The HARP and EAKV Python prototypes.
- The CPU repack of experts, which doubles RAM and breaks the shared DMA source.
- Any AVX-512 path written without saying it never ran on either box.

---

## 13. Open questions that need measurements on this box

All of these need the engine stopped or an explicit go. The live Strata process holds about 21 GB of
RAM and the whole GPU.

1. **DDR4 sustained read bandwidth** on the 3800X with 6-7 threads, per CCX and both together. This
   is the Zen 2 single-CCD fabric limit and the denominator of rank 1.
2. **AVX2 GEMV throughput** in GB/s of weight bytes per core and at 6-7 threads, for each type in
   use (IQ1_M, IQ2_XXS, Q2_K, Q4_K, Q8_0, MXFP4, Q2_0). Measure at batch 1/2/3/4, with and without
   NTA prefetch. The IQ grid lookups may make them compute-bound well below DDR speed.
3. **Contention:** PCIe DMA throughput and CUDA launch latency while 6-7 CPU threads saturate DDR4
   with expert GEMV. Separately, the desktop's responsiveness under the same load.
4. **Does `MADV_HUGEPAGE` on a btrfs file mmap produce `FilePmdMapped` on kernel 7.2?** Compare dTLB
   misses for random 2 MiB expert accesses on 4 KiB vs 2 MiB backing (anonymous arena).
5. **Expert residency profile** of coder-IQ1_M under a GPU cache of N slots: the cold bytes per
   token that would land on the CPU, and the hit rate vs N.
6. **Verify-batch expert union growth** on Flash-Next (512 experts, top-10): distinct experts per
   layer for 1..8 tokens. This gives the planner's c in `gain = A - c·D` when cold experts run on
   the CPU.
7. **CPU GQA flash-decode** on q8_0 and q4_0 KV at 16k-64k context per layer group, compared with
   kv-streaming's cost of moving the same pages over PCIe.
8. **Thread-count knee** with the GPU active: CPU compute threads vs end-to-end tok/s.
9. **PLE on qwen4exp:** rows and bytes per token, and the random 4 KiB fault rate from the SATA SSD
   (mmap faults were measured at about 150 MB/s), to decide whether PLE must be fully RAM-resident.
