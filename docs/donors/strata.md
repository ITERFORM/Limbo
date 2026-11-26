# Donor spec: Strata (clean-room reading notes)

Reader: Strata fork, 2026-11-19. Donor: `/home/alca/projects/strata-origin-upstream` (Niko1221/Strata, MIT;
local HEAD d31fbc0, engine 0.1.61 live on :8080). Everything below is written in this reader's own words. No donor
code is reproduced; formulas are restatements of ideas. Format facts (file layouts, GGUF type ids, model geometry)
are interoperability facts and are stated as such.

Tags: **READ** = read in the donor source/docs (cited `path:line`, paths relative to the donor root unless absolute).
**MEASURED(src)** = a number someone measured, with its source. **LIVE** = derived by this reader from the running
engine's own log `strata-coder-iq1_m.log` (start at line 830, 2026-11-19 04:58:50, 305 requests until ~07:52).
**CLAIMED** = the donor says so, not verified here. **EST** = this reader's arithmetic.

Licence note for the planner: Strata is MIT, but parts of it are themselves ggml (MIT): the i-quant GPU dot products and
dequantizers are transcribed from ggml (`docs/DETAILS.md:1401-1403` READ), and the CPU i-quant expert path links
ggml-cpu. Limbo must not take either from Strata; the i-quant codebook grids are format data Limbo needs bit-exact and
must source under ggml's own MIT notice (or regenerate), not via Strata.

---

## 1. What Strata is

**Three products** (READ `AGENTS.md:9-24`): a desktop app (Tauri + Node host), a Python "Terminal" (`strata.py`,
`strata_cli/`, `strata_control/`), and the engine + server. Only the last matters for Limbo.

**Engine + server split** (READ `docs/BATCHING.md:755-772`, `docs/DETAILS.md:17-21`):
- `engine/strata` is one C++/CUDA binary (also HIP, SYCL, Metal builds). It works on **token ids only**.
- `serve/server.py` (Python) owns HTTP (OpenAI + Anthropic shapes), chat templates, tokenization (the pack carries
  `tokenizer/`), tool-call parsing, reasoning handling, a web UI, metrics. It drives the engine over a **line protocol
  on stdin/stdout**: `GEN`/`GENI` (ids, keys, optional image file), and for batch slots `BGEN`, `BSTOP`, `BYIELD`,
  with `BT`/`BDONE`/`BADM`/`YIELDED` coming back.
- Live on this box (LIVE, `/proc`): python server pid 2722475 is the parent of engine pid 2722504; engine RSS ~21 GB,
  VRAM 15.27 GiB of 16 GiB.

**Request flow** (READ `src/program/generate.cpp:1060-1100`, `docs/DETAILS.md:714-777`): server templates and
tokenizes -> `GEN` -> engine matches the prompt against its live session and up to 6 recurrent-state checkpoints ->
reads the unmatched tail through the batched prompt path (chunks) or, for short tails, through verify windows ->
decode loop (draft, verify window, commit) -> token ids stream back -> server detokenizes and parses.

**Architectures**:
- Qwen3.8-Flash-Next family (GGUF arch `qwen4exp`; variants ISTA-DASLab Q2_0/IQ2_XS/IQ3_XXS/IQ3_S, the Coder,
  Swift 1.5, Gyro trellis packs, Unsloth UD-Q4_K_XL, NVFP4) on the main engine.
- Qwen3.6-35B-A3B (`qwen35moe`) on a **separate engine path** `src/qwen35/` (`docs/QWEN36.md:7-9` READ).
- Cyber-Tiel-Coder-35B-A3B added as a family by the local commit d31fbc0 (READ git log).

**Model geometry the live engine runs** (interop facts, READ `include/strata/core/geometry.hpp:88-90`,
`docs/GDN_CONTRACT.md:5-10`, `docs/QSAROWS.md:33-35`, pack `index.txt`, `include/strata/kernels/fused_gr.hpp:1-16`,
`include/strata/core/layer.hpp:543-556`, `include/strata/core/mtp.hpp:1-20`):
- 48 layers. Every 4th layer (index % 4 == 3) is QSA sparse attention (12 layers), the other 36 are Gated DeltaNet.
- Residual = a stack of 4 hyper-connection streams x n_embd 2560 (10,240 values); each mixer reads the stack through
  a gated "read" (norm, 320-wide low-rank down, SiLU, up, per-stream injection) and writes back per stream. Those
  weights are BF16 (1.37 GiB of the dense file).
- GDN: head size 128, 16 key heads, 48 value heads (value head v uses key head v mod 16), 4-tap causal conv over
  10,240 channels; state 128 x 48 x 128 fp32 per layer (~3 MiB).
- QSA: 24 query heads x 256, 2 KV heads, a lightning-style indexer (4 heads x 128) that scores 4-cell blocks; each query
  attends to the top 2,048 cells (cap 2,051).
- MoE: 512 routed experts per layer (the Coder keeps 256), top-10, plus one shared expert; expert FF width 640; router
  BF16 2560 x n_expert.
- PLE ("n-gram table"): only layer 1. Each token gathers 16 rows of 160 values (IQ4_NL, 90 B per row) from a
  hash-addressed table of 320,001,536 rows = 28.8 GB that lives in GGUF shard 2. 1,440 B of table traffic per token.
- One MTP layer (QSA attention with its own K/V, 512 experts top-10 + shared), vocab 248,320.

**How much is Strata's own code**: practically the whole forward pass is native (GR read/write, GDN decode and prompt
kernels, QSA indexer/selection/attention, PLE block, MoE grouping, head, on-device sampler, MTP, DFlash2 drafter, KV
streaming). Borrowed: ggml's i-quant arithmetic (transcribed on GPU, linked on CPU) and llama.cpp's `mtmd` for vision
(separate helper process `engine/strata-vision`). READ `docs/DETAILS.md:1401-1403`.

---

## 2. The pack

**Directory** (interop facts, READ `/home/alca/projects/Strata-data/packs/coder-iq1_m/`):
- `dense.bin` (1,475,120,640 B): every non-expert tensor in one pool, 64-byte aligned.
- `index.txt`: header "strata pack index v3", then "align 64 pool <bytes> tensors <n>", then one line per tensor: name,
  storage kind/type codes, offset and byte length in the pool (twice: source and destination), the two dims, and
  further numeric fields. Tensors marked as served natively carry no pool bytes (they are read from the GGUF).
- `native_experts.txt`: header "strata native experts v3: layer gu_type d_type offset blob_bytes gate_off up_off
  down_off [shard]" with n_expert and the total; one line per layer giving the gate/up ggml type, the down type, the
  layer's offset in `experts.bin`, the per-expert blob size, and the absolute offsets of the three expert tensors in
  the GGUF shard.
- `experts.bin` (25,146,163,200 B for the Coder): **blob = gate rows + up rows + down rows of one expert, contiguous;
  blobs are layer-major (layer x n_expert + expert); blob size is per layer.** The uniform Q2_0 pack is 48 x 512
  blobs of exactly 1,382,400 B with no padding (READ `include/strata/core/expert_source.hpp:718-731`).
- `experts.bin.src.json` (source shard names and sizes, sha256 of `native_experts.txt`), `conversions.json`,
  `tokenizer/` (tokenizer.json, vocab.json, merges.txt, chat_template.jinja, token_type.json).

**The live Coder pack's expert geometry** (LIVE from `native_experts.txt`; ggml type ids 18 IQ3_XXS, 20 IQ4_NL,
21 IQ3_S, 22 IQ2_S, 23 IQ4_XS, 42 Q2_0). The name says IQ1_M; the experts are not.

| gate/up | down | layers | blob bytes |
|---|---|---:|---:|
| IQ2_S | Q2_0 | 5 | 1,510,400 |
| IQ3_XXS | Q2_0 | 3 | 1,715,200 |
| IQ3_S | Q2_0 | 1 | 1,868,800 |
| IQ2_S | IQ4_NL | 15 | 1,971,200 |
| IQ3_XXS | IQ4_NL | 14 | 2,176,000 |
| IQ3_S | IQ4_NL | 9 | 2,329,600 |
| IQ4_XS | IQ4_NL | 1 | 2,662,400 |

Mean blob 2,046,400 B; 12,288 experts; 23.4 GiB of experts. Other formats Strata reads: Q2_0 (18 B per 64 weights,
signed scale), the Gyro trellis types TQ2_T / TQK6 / TQK7 (ggml ids 144-146 of the agention fork: 128-weight blocks of
one fp16 scale plus 32 steps of K bits; a 15-bit running state is hashed into a 2,048-entry fp16 codebook, two points
per step = four weights; Hadamard-rotated basis; READ `include/strata/kernels/cpu/tq_cpu.hpp:1-12`,
`docs/GYRO.md:5-7`), PTQ1_0 (26 B of base-3 digits + fp16 scale per block; READ
`include/strata/artifact/gguf_reader.hpp:244`), K-quants, NVFP4.

**Native packs vs copied packs**: a native pack does not copy experts if the GGUF can be mapped in place; it records the
three tensors' offsets so a blob can be assembled from three slices (a "transient" blob) or copied piecewise straight to
a GPU slot (READ `expert_source.hpp:225-252`).

**Expert profile** (interop facts, READ `include/strata/core/expert_cache.hpp:40-72`): file magic `STRP`, then u32
version, n_layers, n_expert, slots, n_ranked; then n_ranked (u16 layer, u16 expert) pairs in descending routing
frequency; then an n_layers x n_expert int32 table (the tool's form holds frequencies; the engine's saved form holds
each pair's rank, -1 = unranked). Produced by `tools/make_profile.py` from a `--dump-routing` trace of real prompts;
`--expert-profile-save` writes what the adaptive tier learned (VRAM-resident first, then by decayed heat, then by the
old rank), atomically (tmp + rename). The live Coder profile ranks all 12,288 pairs (LIVE line 840).

---

## 3. Expert residency

**Three tiers** (READ `docs/HOW_IT_WORKS.md:8-31`, `docs/DETAILS.md:241-283`): VRAM expert cache (the hottest),
RAM (page-locked, GPU-mapped copy of what VRAM lacks), file (the pack or GGUF, mmap or direct reads).

**VRAM cache** (READ `expert_cache.hpp:93-186`):
- One arena; **slots sized per layer** (each slot keeps the size of the layer it was filled for). Sizing in bytes rather
  than max-blob slots fit 3,556 instead of 2,673 experts and gave +13% decode (MEASURED(paper §6 finding 8)).
- Filled at start from the profile, in rank order, and verified by reading slot 0 back (LIVE: 3,745 slots = 7.14 GiB,
  filled in 15.1 s at 507 MB/s, lines 855-862).
- `admit` never evicts; eviction policy is a measured question left to the adaptive tier. Lesson recorded: a global
  first-come admission filled the whole cache inside the first token's first ~26 layers (2.97% hit); per-layer slot
  quotas fixed it (READ `expert_cache.hpp:169-186`).
- Residency table (layer x expert -> slot or -1) lives on the device; the window graph decides hits on the GPU itself
  (LIVE line 865 "decided on the device").
- Optional elastic VRAM via CUDA VMM segments so the tail can be given back without moving any slot address (captured
  graphs hold addresses) (READ `expert_cache.hpp:119-126`).

**Adaptive tier** (READ `src/program/generate.cpp:1022-1060`, `:669-676`, `:2150-2185`): every 4 verify rounds, swap
up to 96 of the most-routed missing experts in for the least-routed resident ones of the same layer; usage counts decay
x0.7 per adaptation; a swap needs a 1.5x count advantage (hysteresis). In the resident-RAM mode a swap first copies the
evicted expert back into an exchange buffer so the CPU can still compute it while the swap is in flight, then commits it
into the RAM copy, so RAM always holds exactly the complement of VRAM and no swap reads the file. By default the next
window waits for the copies (lag 1); an overlap mode gates each layer's plan on that layer's copies having landed.
LIVE: 1,290,570 exchanges over 89,877 windows = **14.4 swaps per window ~= 29 MB ~= 4.4 ms of PCIe per window**
(EST from 2.05 MB/6.7 GB/s); the live config runs the blocking tier ("adaptive overlap: off", line 893).
Adaptive beats static by a wide margin: 50% -> ~72% served from VRAM (MEASURED(paper §6 finding 4)). Live hit rate
78.9% of routed entries (LIVE, 305 requests).

**RAM tier** (READ `expert_source.hpp:98-162`, `:1281-1296`; `docs/DETAILS.md:241-283`):
- Resident mode copies exactly the experts VRAM lacks, plus the experts of the cache slots the prompt path will lend
  (section 4), into **anonymous, page-locked, CUDA-registered** memory. Reading experts from a file mapping instead ran
  the CPU pool at ~19 GB/s vs 42.8 GB/s from anonymous memory (file pages get evicted and re-faulted) (MEASURED(donor,
  `expert_source.hpp:1283-1292`)).
- Budget clamp at start: available RAM (cgroup-aware MemAvailable) minus a headroom (4 GiB default) minus 256 MiB.
  Page-locking is **paced**: pages are taken in steps while available RAM stays above a reserve, because one huge
  pinned allocation can get the process OOM-killed when the kernel cannot reclaim clean cache fast enough (#1250).
  LIVE lines 867-878: 18.95 GiB pinned in 73 s; 1,418 of 1,418 lendable slots covered.
- Waits for RAM to come back after a previous engine exits (#4051).

**File tier** (READ `docs/DETAILS.md:284-310`, `expert_source.hpp:253-300`): Linux defaults compute page-cached
experts straight from the mapping (zero copy), prefetch with whole-blob `pread`s on 8 I/O threads, a decode read-ahead
gate that turns itself off when the page cache cannot hold the working set. Unused in steady state on this box (LIVE:
"0 blob reads from the file").

**Routing-aware look-ahead** (READ `expert_source.hpp:307-312`): while the pool works on layer l, a thread applies
layer l+1's router to layer l's MoE input (the residual barely changes between layers) and warms the predicted
misses. A prediction only moves bytes; the real router decides.

**PCIe share of misses** (READ `include/strata/core/pcie_share_rule.hpp:1-35`, `expert_source.hpp:407-470`):
- Default rule: share of misses sent over PCIe = link / (link + k x R), R = the CPU pool's measured rate in GB/s of
  expert bytes, k = 1.85 for a pool >= 30 GB/s (its reads compete with the DMA for RAM bandwidth) falling linearly to
  0.8 at <= 12 GB/s (an arithmetic-bound pool), rounded to 0.05, clamped to [0.10, 0.95].
  LIVE lines 833, 915: probe 6.7 GB/s; pool 75 us per missed expert = 27.4 GB/s -> **share 0.15**.
- Optional cost model: per layer, least squares with forgetting fits CPU time = a + c n_cpu + d n_pcie (d = DMA reads
  slowing the pool) and GPU time = g0 + g n_pcie, then picks the n_pcie minimizing max(CPU, GPU); every 64th layer
  moves one expert across to keep both fits informed.
- How the bytes go: the copy engine (DMA into staging, then the GPU computes them after its cached experts) paid; a
  copy kernel inside the graph sat on the critical path and lost above 20% share (MEASURED(paper §6 finding 9)).

**MTP-aware prefetch**: none found; the MTP layer's own 512 experts are all VRAM-resident (READ `mtp.hpp:18-19`).

---

## 4. Prefill

READ `include/strata/prefill/prefill.hpp:60-215`, `include/strata/prefill/share_rules.hpp:15-22`,
`docs/HOW_IT_WORKS.md:48-55`, `docs/DETAILS.md:1383-1385`.

- **Chunks**: up to 8,192 tokens by default (`--prefill auto`; 16K/32K allowed when RAM permits; LIVE lines 866-869
  kept 8,192 because a bigger chunk lends more slots than RAM can cover).
- **Streaming**: a chunk at or above a floor (~1,024, card-dependent) streams **every expert the GPU does not hold**,
  routed or not (a big chunk routes to nearly all of them), through a ring of staging slots, one whole blob each, sized
  as a byte budget. Smaller chunks stage only the routed experts. Next-layer experts stream while the current layer's
  attention runs.
- **Lend region**: the prompt's buffers and ring are carved out of the **tail slots of the expert cache**, lent for the
  prompt and refilled after it; a short prompt lends only what it needs. The experts living in lent slots must also sit
  in the RAM copy, otherwise they are re-read from the drive during every chunk and at refill. LIVE: the prompt path
  borrows 1,583 slots (2.98 GiB), all covered. MEASURED(operator memory 2026-10-21): full coverage 654 tok/s, 668 of
  1,845 covered 173 tok/s; the lend region costs ~0.225 slots per chunk token. A "prompt room" can additionally borrow
  the MTP layer's expert block (only decode reads it).
- **CPU share**: for chunks up to 3,072 tokens the idle pool computes part of the non-resident experts; the share is
  balanced from running means of CPU ms per expert vs GPU ms per streamed expert. MEASURED(donor, RTX PRO 5000):
  +151% prompt speed at 1K, +45% at 2K, nothing at 4K+; first-token KL 0.011, top-1 100%.
- **Prefill bound on this box** (EST): (12,288 - 3,745) non-resident experts x 2.05 MB = 17.5 GB per streamed chunk
  -> 2.6 s per 8,192 tokens at 6.7 GB/s -> ~3,100 tok/s link ceiling. LIVE aggregate on reads >= 1K tokens: **1,121
  tok/s** (983 below 16K context, 1,218 at 32-64K). So steady-state prefill is GPU-side bound, not link bound
  (consistent with MEASURED(paper §6 finding 11): bound by per-expert launches and fp16 dequant).
- **Flat with context**: sparse attention reads a bounded cell set, so prompt speed hardly drops with depth
  (MEASURED(paper): 539 tok/s at 4K, 496 at 262K, older engine).
- PLE rows for the chunk are read on the host (direct I/O, 256 in flight, 1M-row cache) into double-buffered pinned
  memory. LIVE line 839: direct reads measured 7,958 rows/s (~33 MB/s random 4 KiB) on this SSD; at 16 rows per token
  that is ~500 tokens/s if nothing is cached. That can bound a cold prompt here.
- **Auto chunk shrink**: an earlier RAM-driven auto shrink was refuted (435 vs 491 tok/s) and removed in the operator's
  fork (operator memory). Upstream has a host-memory planner that picks chunk and ring from page cache, RAM copy, drive
  class and link (READ `docs/DETAILS.md:304-310`).
- KVA (opt-in, off): approximate late-layer state for long cold prompts; quality losses measured, so off by default
  (READ `docs/KVA.md`). An approximation, not for Limbo's default.

---

## 5. Decode

READ `include/strata/core/verify.hpp:1-20`, `include/strata/core/session.hpp:1-14,127-145`,
`include/strata/kernels/s2_expert_grouped.hpp:1-20`, `include/strata/core/flag_order.hpp:1-8`, paper §5-6.

**The verify window.** T rows (last accepted token + drafts) go through all 48 layers in **one captured CUDA graph
per window size** (live sizes 1-6 and 12, captured at start in 8-32 ms each; LIVE lines 895-909). Single-token and
multi-token kernels do identical per-token arithmetic, so a draft is accepted exactly when greedy would emit it.
Cost model: dense weights read once per window, plus the union of the rows' missed experts (MEASURED(donor): 1.75x one
token's misses at T=2, 2.4x at 3, 3.05x at 4).

**Per-layer CPU/GPU hand-off inside the graph.** For each layer the graph publishes the normed MoE input rows and the
routing to mapped host memory ("doorbell"), computes the VRAM hits with **grouped launches** (gate+up together for all
hits, SwiGLU, activation quantize, down: four launches per layer instead of five per expert), and then a **spin kernel
waits on a mapped-memory flag** that the host raises when the CPU's rows are in, and on a second monotonically rising
flag that a copy-stream host callback raises when the PCIe share's DMA has landed (the flag must stay behind every queued
copy, or the GPU reads half-written staging). A stage whose experts are all resident runs a "zero-doorbell" graph with
only the PLE wait at layer 1. Why graphs at all: "the engine is launch-bound" — a 43-node layer graph replayed in 1.585
ms vs 2.393 ms as direct launches (MEASURED(donor, `session.hpp:132-137`)).

**CPU pool** (READ `include/strata/kernels/cpu/pool.hpp:1-27,133-150`): one worker per **physical** core (never SMT
siblings), the host thread pinned to its own core spinning on GPU flags (pinning it: 36.3 vs 26.9 GB/s); a layer is one
flat batch (a barrier by construction), claimed with a single 64-bit CAS word holding epoch | job count | index so a
late waker cannot claim from the next batch; completion = all done AND all parked. The activation is quantized once per
layer and shared by all jobs. LIVE: 7 workers on logical CPUs 1-7, host on 0.

**Where the time goes.** "The CPU work is 96% exposed because the residual chain is strictly serial" (MEASURED(donor,
`s2_expert_grouped.hpp:3-6`, early engine)). Paper Table 5 (RTX 5070 12 GB, Ryzen 5 7600 6C DDR5, 4K context):
window 34-49 ms = GPU 14.6-15.3 + CPU experts 13.4-27.1 + drafting ~2 + other ~4-5; 3.2-3.3 tokens per window;
"GPU and CPU each need about half of every round, and they wait for each other at every layer". i-quants are CPU
**arithmetic**-bound (~5 GB/s per core, 23-26 GB/s on 6 cores; decoding once for 3 tokens is 2.0-2.4x the
throughput), Q2_0 is **RAM**-bound (AVX-512 VNNI kernel ~42 GB/s). Overlapping two token groups inside a window lost 7%.

**Live numbers on this box** (LIVE, 305 requests): decode **40.1 tok/s** overall on generations >= 100 tokens
(43.4 below 16K context, 39.4 at >= 64K); 2.60 tokens per window, 3.42 verified rows per window, 64.8 ms per window
(1.35 ms per layer); MTP draft acceptance 0.66-0.69. Per generated token: 120 missed routed entries, of which 54.7 are
**skipped** by the route-tail rule, ~7 go over PCIe, ~58 are computed by the CPU.

**Route-tail skip** (READ LIVE line 892, default ON): a missed expert that every row of the window routes at rank 7 or
lower (of 10) is not computed at all. LIVE: 11.98 M experts skipped = 43% of missed entries. CLAIMED +10-20% decode on
12-16 GB cards; answers differ slightly. **It is an approximation on by default.**

**Drafting** (READ `mtp.hpp:1-20`, `include/strata/spec/draft_policy.hpp:1-25`,
`include/strata/spec/suffix_drafter.hpp:1-12`, `docs/DETAILS.md:334-352`):
- MTP head drafts up to `--spec` tokens (live 4, `--spec-min-p 0.5`: a draft joins the window only while every draft so
  far has p >= 0.5 under the draft layer). The draft layer's projections run as Q8_0, its attention is dense over its
  own window (32,768-cell ring), all 512 of its experts sit in VRAM (LIVE: 952 MiB), and its head is restricted to a
  vocabulary subset (LIVE: 106,299 tokens, 212.9 MiB). CLAIMED acceptance 0.89/0.86/0.85 at steps 1-3 (their probe).
- Suffix (prompt-lookup) drafter: trigram index of prompt + output (4 most recent positions per trigram, bounded ~20 B
  per token), extends matches backward, proposes the longest. Up to 12-token windows.
- A policy chooses MTP vs lookup per round by **expected committed tokens per measured millisecond**: MTP from an EMA of
  tokens committed per window size, lookup from acceptance rates bucketed by match length; costs are measured round
  times per window size and context bucket. A fixed cost table makes runs repeatable.
- A DFlash2 block-diffusion drafter is an alternative path (`src/core/dflash2.cpp`).
- The drafter's cells are the model's positions, and the PROMPT fills them: per prompt cell i the main model's final
  multi-stream residual R_i and the token at i+1 go through the layer's front (embedding and hidden fc, the attention
  hyper-connection, k/v) into its K/V, batched on the prompt path; cells below `prompt_len - window - 64` are skipped
  (READ `include/strata/core/mtp.hpp:141-146` `prefill()`, `:233-245` E-9 `Prefill::draft_kv`, `first_needed()`).
  A drafter whose K/V holds only the recent windows drafts from almost no context (Limbo measured ~50 % kept against
  56-82 % once its prompt cells were filled, 2026-11-25).
- During a long prompt the drafter's 708 MB expert block is lent to the prompt pass's stage: copied to pinned host
  memory once, written back in place after (`mtp.hpp:91-107`, "THE PROMPT ROOM").
- The live config's `--control-vector-scaled ...experimental-speed-projection` is a refusal-direction projection
  (h -= (h.v)v after layers 4-44), not a speed mechanism: it changes answers and costs 0.2-0.4 % a token
  (READ `docs/DETAILS.md:1429-1450`). Not taken.
- PLE table residency: the whole table is locked in RAM only when table + expert arena + headroom fit, else direct
  reads (READ `include/strata/core/ple_residency.hpp:14-30`); the Linux safety headroom is 4 GiB (`:163`). On this
  box (26.8 GB table, 31 GB RAM) Strata reads the table directly too.

**Commit**: the window appends K/V and indexer keys for all rows; GDN state is not touched during the window; commit of
n accepted rows replays the GDN conv + recurrence from stored inputs, restores the indexer tail from a snapshot and
re-appends accepted keys, resets PLE history; rejected K/V cells are simply overwritten later (READ `verify.hpp:13-18`).

---

## 6. KV, sessions, slots

**State of one conversation** (READ `session.hpp:1-10`, `generate.cpp:1061-1064`): 12 QSA states (positional K/V +
indexer pooled keys) and 36 GDN states (recurrent state + conv history), plus PLE history. K/V is **positional**: it
stays where it is; a checkpoint only needs the recurrent part.

**KV formats** (READ `docs/DETAILS.md:228-240`, `include/strata/kernels/kv_stream.hpp:30-60`): int8 codes with an fp16
scale per 64 values (default; optional Hadamard rotation), q4_0 with a Walsh-Hadamard rotation, K8V4 hybrid. Page = 4
cells = one indexer block; 4,224 B per page per layer in int8. **~13.7 KB of host RAM per context token** with
streaming (1.7 GB at 128K). The live engine keeps all 131,072 cells in VRAM (no streaming in the banner).

**KV streaming for sparse attention** (READ `kv_stream.hpp:1-20`, `include/strata/core/layer.hpp:233-240`):
- The authoritative K/V lives in pinned, device-mapped host memory in the identity layout; VRAM holds a pool of page
  slots and the page table becomes a residency map (block -> slot or -1).
- After the indexer's selection and before attention, a **device-side resolve inside the captured graph**: one
  1,024-thread block walks the selected blocks, stamps hits (epoch + CLOCK reference bit), claims misses, picks victims
  with a CLOCK second-chance sweep that never evicts a slot this call uses, re-points the table, and a copy kernel pulls
  the missed blocks over PCIe (zero-copy).
- **Writers always write the host copy and also the slot when resident**, so slots never go stale: nothing to invalidate
  after a rejected draft or a resume.
- The MTP drafter's K/V is a static ring (block mod slots) of its window, refilled from the host copy on resume.
- Opt-in prefetch of the next attention layer's K/V upload on a second stream during prefill: -5.3% prefill time at 32K
  (MEASURED(donor `docs/KV_PREFETCH.md`, 2x RTX 3080)).
- Elastic K/V (`--kv-grow`): VMM addresses for the whole context, physical chunks taken from the expert cache as the
  context grows (READ `layer.hpp:285-300`).

**Checkpoints and parking** (READ `generate.cpp:1061-1100`, `docs/PROMPT_CACHE_TAIL.md`,
`docs/MESSAGE_BOUNDARY_CACHE.md`, `docs/DETAILS.md:714-777`):
- Up to 6 checkpoints of the recurrent state (~118 MB host each): at the last turn boundary (`<|im_start|>` = 248045),
  at the end of a >= 2,048-token system prompt ("root"), every 16,384 fresh tokens, optional near-tail and
  message-boundary ones. Valid only while the K/V cells before their position are untouched.
- Parked conversations ("conversation cache"): a copy of a whole conversation's state into host RAM, sized from free
  RAM. LIVE line 890: **off on this box** ("not enough free RAM to park conversations").

**Batch slots** (READ `docs/BATCHING.md:1-30,143-191`): `--batch N` keeps N conversations; a window carries one row per
slot. A second request **STOPs** the solo stream and moves it into a slot (state copied in, 50-60 ms for a short
conversation, more as it grows); slots decode without MTP unless `--batch-mtp`; a lone slot is copied back to the solo
path. Each slot costs VRAM the expert cache loses (0.56 GiB at 32K int8). MEASURED(operator memory, own fork
2026-10-21): two slots decoding together gave **no aggregate gain on this box** (2 x ~21 = 38-41 vs ~43 solo); parking
beat `parallel 2` here.

**"TAPE"**: there is no tape concept in upstream. The operator's earlier fork used `STRATA_KV_TAPE_HOST` for a CPU
replay of the resolve that prefetches the next layer's page misses — a KV-prefetch mirror, not context management
(operator memory `invariant_strata_tape_env_is_kv_prefetch`). The operator's pointer-exchange session swap
(`feat/session-swap`: moves 3-18 ms instead of 160-231 ms at 14K) was never upstream and its worktree is gone.

---

## 7. Kernels Strata owns (strategies only)

READ `include/strata/kernels/*.hpp` headers, `docs/EXPERTDEC.md`, `docs/QSAROWS.md`, `docs/GDN_CONTRACT.md`.

- **Grouped expert GEMV for hits**: group by blob (one expert = one group, entries = the rows that picked it), one launch
  per projection for all groups; decode each weight block once and apply it to all entries of the group. Opt-in
  refinement stages the group's activations in shared memory (bank-conflict-free layout), stages the IQ3 codebooks in
  shared memory, several rows per lane: -20% kernel time at T=2-4, +1.1% end to end, bit-identical.
- **Multi-column MMVQ** for dense projections at T rows, with per-token arithmetic identical to the single-token path
  (the exactness contract).
- **Fused hyper-connection read**: 2 kernels instead of 6 per read (norm + low-rank down + SiLU fused; up + gate + mean
  fused), with the previous half's write folded in; 192 such reads/writes per token.
- **GDN**: fused decode step; prompt recurrence in several dispatch tiers (key-head kernel; chunked fp32; tensor-core
  fp16-operand scan on sm_80+), each tier judged against an fp64 oracle with explicit numeric tiers (bitwise / reordered
  fp32 / mixed precision). State is advanced only after a path is known to run (no replay possible once started).
- **QSA**: block scoring with queries held in registers and a shuffle reduction that keeps the reference pairing (bitwise
  equal, 2-3x less per-row cost), top-k selection, sparse decode attention over each row's selection, indexer append.
- **Prompt MoE**: fused int8 i-quant expert kernels, MMQ-style grouped GEMMs over staged blobs, "few-rows" fp16 path
  for experts with few tokens.
- **KV streaming resolve**, on-device **sampler**, **NaN guard** rows checked inside the window graph.
- **CPU**: runtime ISA dispatch AVX-512 / AVX2 / AVX / scalar; own Q2_0 kernels (AVX-512 VNNI: one unpack + one dot per
  64 weights; 7x faster than scalar on AVX2); trellis kernels that replace hash + codebook by a 32,768 x 4 int16
  state-to-weights table (256 KiB, one 8-byte load per four weights), four fixed-order float accumulators so every ISA
  and every token-group size gives the same bits; ggml-cpu's dots for i-quants.
- Discipline worth copying: every fast path has a parity bench against a reference (`*_parity`, `*_bench`), numeric
  tiers are named, and graph inputs live at fixed pinned addresses (a stack address baked into a graph is a classic
  wrong-but-plausible-token bug, READ `layer.hpp:263-276`).

---

## 8. Flaws and limits

1. **No tape.** One live session; positional K/V means another conversation overwrites the cells, so the main
   conversation must be parked (a copy into host RAM) or re-read. With RAM short, parking is off. **LIVE: 106 cold
   reads from position 0 of > 4,096 tokens = 3.36 M tokens in 2,714 s — 30% of the engine's busy time (9,125 s
   prefill+decode) in this 3-hour session.** Batch slots copy state in and out and cost VRAM.
2. **RAM is spent first and once.** The complement pin takes ~19 GiB at start, sized from what happened to be free at
   that moment (browser open = less coverage = slower prefill). Start takes ~1.5-2 min (73 s pinning, 15 s fill).
3. **Two processes, Python front, text protocol on ids.** Templates, tool parsing and HTTP live in Python; the engine
   restarts lose everything.
4. **Single-family geometry.** Geometry constants are baked in (n_embd 2560, FF 640, blob offsets fixed per artifact,
   READ `expert_source.hpp:18-21`); A3B models need a separate engine path. New architectures cost a port each.
5. **Approximations on by default**: route-tail skip (43% of misses skipped live), CPU/GPU rounding differences in the
   expert path, CPU prefill share (KL ~0.004-0.011), non-reproducible adaptive tier.
6. **PCIe and VRAM at the edge on 16 GB**: live start left 591 MiB free; two requests started below 512 MiB (LIVE warnings
   "only 437/198 MiB of VRAM free": the driver may page VRAM to RAM and decode collapses).
7. **Configuration sprawl**: hundreds of `STRATA_*` switches and per-card defaults; behaviour depends on env and card
   class. The planner is a pile of rules, not one model.
8. **Blocking adaptive tier** in resident mode (~4.4 ms of link per window, exposure unmeasured).
9. **Upstream churn**: engine 0.1.37 -> 0.1.61 in ~5 weeks; the operator's forks (tape-shared, session-swap) fell
   hundreds of commits behind and were abandoned.

---

## 9. Conflicts with the other donors

1. **PCIe link (6.7 GB/s here) is one budget, claimed four ways.** Strata uses it for the decode miss share, adaptive
   swaps (~29 MB per window), prefill expert streaming (saturating) and sparse-KV page misses. The adaptive-kv-streaming
   llama fork streams **dense** attention K/V beyond its stage on **every** decode step (cost grows with context). Run
   side by side they double-book the link; neither knows the other's traffic.
2. **Pinned RAM is one budget.** Strata pins the whole expert complement (~19 GiB) and then has nothing left for parking
   or host K/V; the llama fork pins host K/V sized to the full context (26.6 KB/token for the dense 27B, 7 GB at 262K)
   plus checkpoints and a prompt cache. On a 31 GB desktop both cannot hold their full plans.
3. **CPU cores are one budget.** Strata's pool owns 7 of 8 physical cores plus a spinning host core for the whole
   session. modchip-style CPU decode wants all cores for GEMV; kindi/llama-style CPU expert offload the same. Two engines
   each sizing a pool to "all cores" convoy each other and the desktop.
4. **Graph ownership.** Strata owns a hand-written forward pass with fixed geometry; llama.cpp/modchip own a ggml graph;
   kindi owns its own kernels. No shared op layer exists, which is why the old Limbo's "publish experts into llama's
   window" seam ran at 0.015 tok/s (operator memory `bug:limbo-stream-dequant-bound`). Combining means one engine owning
   the graph, not adapters between engines.
5. **Exactness policy.** Strata tolerates small deviations by default (skip, rounding, shares); kindi verifies
   token-exact greedy against llama.cpp; the llama fork is llama's numerics. Limbo needs one explicit policy (exact by
   default, approximations opt-in and measured by KL).

---

## 10. The combination question: decode miss on the CPU or over PCIe?

Inputs (this box): expert blob 1.51-2.66 MB, mean 2.05 MB (LIVE pack). PCIe H2D 6.7 GB/s (LIVE probe). CPU pool 75 us
per missed expert = 27.4 GB/s of expert bytes (LIVE line 915; the Coder's experts are i-quants, so this pool is
arithmetic-bound, consistent with the k-rule giving 1.7 here).

Per expert, one token:
- (a) DMA + GPU compute: 2.05 MB / 6.7 GB/s = **~305 us** (225-397 us by layer), on the layer's critical path unless
  the copy starts before the router decides (prediction).
- (b) CPU from the RAM tier: **~75 us**, shipping only a 2,560-float row each way.
=> **CPU is ~4x cheaper per missed expert for decode on this box.**

Per window (LIVE averages): 2.60 tokens, 3.42 rows, ~159 distinct non-skipped misses (~3.3 per layer).
- (a) all over PCIe: 159 x 305 us = **~48 ms of link time per 65 ms window**, serialized layer by layer -> decode
  falls to roughly 23-27 tok/s (EST).
- (b) all on the CPU: 159 x 75 us = **~12 ms of pool time per window** (0.25 ms per layer) beside ~1.35 ms of GPU work
  per layer -> roughly today's 40 tok/s.
- Live balance: ~85% CPU / ~15% DMA (~10 ms pool, ~5.6 ms link, concurrent). Without the route-tail skip the misses
  roughly double (~292 per window): (a) ~89 ms, (b) ~22 ms.
- Per generated token: (a) ~18.7 ms, (b) ~4.6 ms.

Crossover (prefill): CPU cost for an expert hit by n rows ~ 75 us + (n - 1) x c_e. The paper's 2.0-2.4x throughput at
3 rows puts c_e near 14 us on a Zen 4; on this Zen 2 AVX2 box assume 14-25 us. PCIe stays ~305 us (GPU compute per row
is ~1 us). **n* ~ 10-18 rows per expert.** An 8,192-token chunk gives a Coder expert ~320 rows on average (8,192 x 10 /
256), so PCIe wins for almost every expert in prefill; only the rarely-routed tail of a chunk belongs on the CPU. The
planner primitive should therefore be **per expert, by its row count in this step**, not Strata's per-chunk switch.

Must be measured before Limbo commits numbers (all on this box, engine stopped by the operator):
1. CPU marginal cost per extra row per format on Zen 2 AVX2 (IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS gate/up; IQ4_NL, Q2_0
   down) at 1, 2, 4, 8, 16 rows. The donor ships a bench for exactly this (`src/kernels/cpu/pool_rows_bench.cpp`).
2. Distinct misses per layer-window (Strata logs entries, not distinct experts): a `--dump-routing` trace.
3. The DMA-vs-pool interference coefficient d (PcieModel on) and DDR4 bandwidth alone vs during DMA.
4. How much of the CPU time is exposed per layer (`STRATA_DECODE_TIMING=1`).
5. Exposure of adaptive swaps (blocking vs overlap A/B).

---

## 11. What Limbo should take (ranked by payoff on this box) and what to leave

Box: RTX 4060 Ti 16 GB, PCIe 6.7 GB/s, Ryzen 7 3800X 8C/16T AVX2, 31 GB DDR4, SATA SSD. Bars: decode >= 40, prefill
>= 500 tok/s. Tape semantics are hard requirements.

**Take**
1. **Tape-native sessions (fixes flaw 1, the largest measured loss: 30% of busy time re-reading).** Hybrid models make
   this cheap: positional K/V plus a ~118 MB recurrent checkpoint. Give main and sidekick each their own K/V region and
   recurrent state from day one (pointer-selected, never copied); host-backed K/V with a device page cache for sparse
   attention so two long conversations fit; checkpoints at turn boundaries for in-place rewind.
2. **Three-tier expert residency as a planner output**: per-layer byte-sized VRAM slots, profile pre-fill, adaptive
   swaps with decayed counts + hysteresis + per-layer quotas, swaps overlapped with decode (gate per layer on landed
   copies), RAM copy = exact complement of VRAM in anonymous page-locked GPU-mapped memory, paced pinning, file tier
   only as overflow.
3. **Hybrid MoE decode**: GPU computes hits with grouped launches; the CPU pool computes most misses from RAM; a
   cost-balanced share goes to the copy engine as DMA; per-expert placement by row count (section 10). Whole verify
   window as one captured graph per size; mapped-memory doorbells and monotone flags; zero host launches per layer.
4. **Pool design**: one worker per physical core, host core reserved, single-CAS epoch-tagged claims, shared per-layer
   activation quantization, size thresholds so idle UI never wakes it. (Matches the lock-free/OpenMP rule.)
5. **Prefill streaming**: big chunks, byte-budget ring carved from the cache tail, RAM coverage of lent slots enforced by
   the planner, next-layer streaming under attention, CPU takes few-row experts.
6. **Spec decoding**: model-native MTP with exact verify windows, draft vocabulary subset, suffix drafter, and the
   expected-tokens-per-ms policy; captured graphs per window size.
7. **Sparse-attention KV streaming** with device-side CLOCK resolve and write-through host copy (no invalidation ever).
8. **Routing look-ahead** (next layer's router on this layer's input) to start DMA before the router runs — the paper's
   own "next step", untested.
9. RAM-truthfulness helpers: cgroup-aware availability, wait-for-RAM-to-return, PLE-style table residency decided
   against the whole budget.
10. Engineering discipline: parity benches per kernel, named numeric tiers, fixed-address graph inputs, NaN guard in the
    graph.

**Leave**
- The Python front and the stdin id protocol (Limbo answers HTTP in-process).
- Route-tail skip, KVA, CPU-share rounding, control vectors as defaults (opt-in, measured by KL, if at all).
- Batch slots that copy state; parking as a copy (replaced by item 1).
- Hard-coded geometry and per-family engine paths (Limbo needs one op layer with shapes from the GGUF).
- The env-var thicket and per-card rule tables (one measured planner instead).
- HIP/SYCL/Metal/Windows-specific machinery for now.
- ggml-derived kernels in any form (licence: write Limbo's own).

---

## 12. Open questions needing a measurement on this box

1. CPU per-row marginal cost per format on Zen 2 AVX2 (section 10, item 1) — sets the per-expert placement threshold.
2. Decode time split per window on this box (`STRATA_DECODE_TIMING=1`): GPU-reach wait vs CPU vs draft vs adapt wait.
3. Distinct misses per layer-window and how many the look-ahead predicts correctly (`STRATA_LOOKAHEAD_STATS=1`).
4. DMA/pool interference (d) and DDR4 bandwidth with and without concurrent DMA.
5. Adaptive swap exposure: blocking vs `STRATA_ADAPT_OVERLAP_RESIDENT=1`.
6. Route-tail skip: decode gain vs KL on the operator's own prompts.
7. PLE reads during cold prompts: rows/s at depth 256, row-cache hit rate; is layer 1 ever the prefill bound here?
8. VRAM headroom needed to avoid driver paging (live start had 591 MiB, requests saw 198-437 MiB).
9. What two tape slots cost in VRAM and RAM for this model at 64K and 128K, and the decode cost of a sidekick
   prefill running beside main decode.

All of these need the GPU or the full CPU; none was run by this reader (the live engine owns the box).
