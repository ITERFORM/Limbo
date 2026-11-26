# Donor spec: adaptive KV streaming (llama.cpp-adaptive-kv-streaming)

Clean-room reader spec for Limbo. Everything below is a description of the donor's ideas in my own
words. No code was copied into this file. The donor is MIT and Limbo links none of it.

- **LOCAL** = `~/projects/llama.cpp-adaptive-kv-streaming`, branch `feature/adaptive-kv-stream` at
  `493e34eef` (itsXactlY fork), plus 7 uncommitted files read with `git diff` only. Upstream base
  `f280b2698` (2026-08-24). It holds Raymond Huang's 61 v1 streaming commits (all dated 2026-09-06), the
  Prism ML / Bonsai line (merged at `d1d7e66b5`, `f46c1e43a`), and 26 itsXactlY server/tape commits.
- **V2** = `origin/feature/adaptive-kv-stream-v2` at `4804b4044` (Raymond, 2026-11-09). It has 1020
  commits that LOCAL lacks (307 of them Raymond's), merge-base `465e49b9ce`. V2 is a rewrite of v1, not
  an extension: none of the v1 hashes appear in it.
- Tags: **READ** (I read it in source, cited `path:line` at LOCAL unless marked V2), **MEASURED** (the
  source of the number is named), **CLAIMED** (donor docs, not reproduced on this box).

## 0. Box facts this spec uses (desktop)

| fact | value | tag |
|---|---|---|
| GPU | RTX 4060 Ti 16 GB, 288 GB/s VRAM (18 Gbps x 128 bit) | READ nvidia-smi |
| PCIe | link max **Gen3 x8** (`pcie.link.gen.max=3`, width 8; device itself is Gen4-capable) | READ 2026-11-19 |
| H2D | **6.6 GB/s** effective | MEASURED (memory `invariant_strata_prefill_speed_is_the_ram_tier`) |
| CPU | Ryzen 7 3800X, 8C/16T, Zen 2, AVX2, **no AVX-512**, L3 2 x 16 MiB, one NUMA node | READ |
| RAM | 31 GB DDR4 dual channel, speed unknown (dmidecode needs root) | READ |
| 27B geometry | 16 full-attention blocks of 64, 4 KV heads, head dim 256/256, GQA 6, 48 GDN blocks | CLAIMED V2 README "How it works §1" + memory |
| 27B KV at q8_0 K / q4_0 V | 4 x (272 + 144) = **1,664 B/token/layer**, **26,624 B/token** over 16 layers; page (256 tok) = 425,984 B/layer, 6.8 MB over all layers | derived; matches memory 26.6 KB |

## 1. What the fork adds over upstream llama.cpp

| area | where (LOCAL) | what it does |
|---|---|---|
| gates + pool geometry | `src/llama-kv-stream-config.cpp:15-30`, `:36-72` | refuses streaming unless qwen35 arch, default (target) context, single KV stream, FA on, KV offload on, stage >= one 256-token page; divides a pool into scratch pages and equal per-layer resident pages |
| host-side planners (pure functions) | `src/llama-kv-stream-plan.cpp:44` (region plan), `:169` (extent with shrink hysteresis), `:220` (page regions), `:282` (prefetch dispatch), `:355` (resident/ring partition adapt), `:486` (feedback deltas) | priority-ordered resident placement; deadline-ordered prefetch selection; feedback-driven boundary between resident pages and ring slots |
| reference softmax merge | `src/llama-kv-stream-softmax.cpp` | exact log-sum-exp merge of unnormalized block partials (max, normalizer, numerator) in double; the test oracle for split attention |
| KV cache integration | `src/llama-kv-cache.cpp:172-340` | when a stage is set, attention-layer K/V tensors are allocated from a special CUDA "KV stream host" buffer type found by proc-address lookup; `:1538` per-ubatch adapt; `:2195` publishes dirty row indices; `:588` sequence compaction; `:1249` per-sequence reserve |
| context wiring | `src/llama-context.cpp:493-533`, `:1520` | builds and validates the config; calls adapt before every ubatch with (active cells, query tokens) |
| CUDA runtime | `ggml/src/ggml-cuda/ggml-cuda.cu:1409-1930` | runtime object (device pool, ring, resident cache, dirty rows, generation counter); host buffer = pinned+mapped+write-combined (`:1524`); reconfigure/repartition/decode-layout entry points all do a full device sync; FA nodes on that buffer type are routed to the streamed FA |
| streamed attention | `ggml/src/ggml-cuda/fattn.cu:139-2270` | transfer ring, resident cache, graph-level prefetch planner (`:1511`, `:1590`), streamed FA (`:1726`) |
| staged KV writes | `ggml/src/ggml-cuda/set-rows.cu:411`, `ggml-cuda.cu:2724` | contiguous row ranges go through a device staging buffer then one D2H copy, plus a D2D copy into the resident mirror |
| span tuner | `ggml/src/ggml-cuda/kv-stream-span-tuner.h:13-14` | A/B of "bounded 32-page spans" vs "unbounded spans" on end-to-end decode latency (16 samples each, needs 0.5 % gain), then frozen until the next layout change |
| tape/server (itsXactlY) | commits in §3 | `-np 2 -kvu` streaming, per-slot attention extents, compaction, shared-prefix fork, prefill share, split prompt batches, shared checkpoint cap, splice endpoint |
| Prism ML / Bonsai line | `67c1af28b`..`d8f26eec7` | PQ2_0 and PTQ1_0 ternary types (PTQ1_0 = 1.75 bpw, lossless vs PQ2_0, `01fd9521c`), Hadamard weight-fold runtime with FWHT kernels (`e0c828b90`), DSpark log-SNR conditioning (`b8e0bf0b6`), GDN rows-indexed ring decode (CPU/Metal only, `bed6e9a6b`), six decode-graph fusions (-25 % ops/token, `00e818761`), **per-channel K mean-centering for Q4_0 K** (`7fb3d5bf7`, `36d1deef2`) |
| expert-stream seam (alca) | `src/llama-expert-stream.{h,cpp}`, `ab5eef7a2` + uncommitted | §6 |

K mean-centering deserves a separate note (READ `common/kv-mean-center.h`). A calibrated per-layer,
per-channel mean is subtracted from every K before it is quantized. For one query, q·(k - mu) differs
from q·k by the constant q·mu across all keys, and softmax ignores a constant shift. In exact arithmetic
the change is lossless, and it lowers the error of Q4_0 K. The loader refuses a bias file whose rotation
basis differs from the running K-rotation state.

## 2. Adaptive KV streaming mechanics (v1, LOCAL)

### 2.1 Where bytes live
- **Authoritative history is host memory.** All attention K/V (every layer and token) sits in one
  pinned, device-mapped, **write-combined** host allocation (READ `ggml-cuda.cu:1524`; Windows omits
  write-combined). The tensors' data pointers are host addresses, and the buffer reports itself as
  host memory.
- **One device pool**, sized by `--kv-stream-stage-mib`, holds three regions:
  1. scratch ring slots, each one page of one layer;
  2. the resident cache, which is per-layer runs of pages;
  3. an F16 conversion page, needed only for KV types without direct quantized FA (READ
     `ggml-cuda.cu:1564`).

  The pool does not hold compute buffers. In v1 they are sized separately; the phase arena (§5)
  changes that.
- **Page = 256 token positions of one layer.** The page holds the K plane (all KV heads, token-major),
  padded to 128 B, then the V plane (READ `fattn.cu:1149`). For the 27B at q8_0/q4_0 a page is 416 KiB.
  `--kv-stream-stage-mib 1024` gives 157 resident pages/layer = **40,192 tokens** (MEASURED, memory
  `invariant_kv_stream_stage_capacity_and_relayout`).
- **Residency is by cell index, as a prefix.** For each layer, pages 0..n-1 are resident and the rest
  (including the mutable tail page) stream. Each layer stores resident K and V as two contiguous
  token-major planes, so the resident prefix is one span the stock kernel can read (READ `fattn.cu:1830-1860`).
  A loaded bitmap and a dirty bitmap per (layer, page) track freshness.

### 2.2 Write path (new tokens)
- The CPU marks the destination cell indices of every ubatch as dirty rows (READ `llama-kv-cache.cpp:2195`).
- **Prefill (contiguous rows):** the device quantizes the rows into a device staging buffer, then issues
  one bulk D2H copy into the host history. If those rows fall inside resident pages, a D2D copy also
  goes into the resident mirror (READ `set-rows.cu:411-440`).
- **Decode (one or a few rows):** the kernel writes the quantized row straight into mapped host memory
  over PCIe. A second write goes to the resident mirror when the page is resident (READ `ggml-cuda.cu:2724-2765`).
- Unchanged history is never written back. The host copy is always current, and the device copy is a cache.

### 2.3 Read path (one attention layer)
- **Resident span:** one stock FA call over the whole resident prefix. This uses native quantized kernels
  when built with all FA quants, otherwise a per-page F16 conversion. The page tables are refreshed only
  where dirty.
- **Streamed pages:** each page is uploaded by a separate non-blocking copy stream into a ring slot.
  Pages in consecutive slots are attended as one span. A span is capped at 32 pages when the span tuner
  picks "bounded" (READ `fattn.cu:2010-2056`).
- **Split attention:** each span yields an unnormalized partial: per row a max logit, a normalizer and a
  numerator vector. The decode vector kernel uses up to 16 parts per chunk (`GGML_CUDA_KV_STREAM_PARTS`,
  `fattn.cu:864`). A running max-rescaled accumulator folds the partials in, and one normalization pass
  runs at the end (READ `fattn.cu:899-965`). Prefill uses an MMA partial kernel when a mask exists,
  head dim = 256 and GQA <= 8 (READ `fattn.cu:1880`). Query rows are processed in tiles of at most 256
  so the scratch stays bounded.
- **Exactness rule:** if the cache is fully resident, or the whole cache is a single page, v1 calls stock
  FA unchanged, so logits match non-streamed runs (READ `fattn.cu:2168`). Once real streaming begins,
  the merge order differs from stock and logits drift in the last bits. V2 fixes this (§5).
- Head dim is fixed at 256 (READ `fattn.cu:860`), and only Qwen3.5-family geometry qualifies.

### 2.4 Prefetch and overlap
- Before a graph runs, the runtime walks every FA node on a streamed buffer, in final execution order,
  and creates one request per non-resident page (READ `fattn.cu:1590-1684`, `ggml-cuda.cu:1895-1927`).
  The mutable tail page of a layer becomes eligible only when that layer's producer (its SET_ROWS) has
  been recorded on the compute stream (READ `fattn.cu:1511-1533`).
- Free ring slots are filled in deadline order. That order runs across layers, so the copy engine works
  ahead into later layers ("cross-layer prefetch"). Runs of up to 32 consecutive pages of one layer go
  as **two copies**: one K run and one V run (READ `fattn.cu:145`, `:1359-1440`). Each slot has a ready
  event and a consumed event, and a slot is rewritten only after its last reader.
- **Deadline accounting:** when a sampled page is consumed, a one-thread kernel reads a mapped ready flag
  and counts a miss if the copy had not finished (READ `fattn.cu:1246-1256`). The copy-engine busy ratio
  is extrapolated from one timed upload per evaluation (READ `fattn.cu:1258-1283`).
- The qwen35 layout helps hide copies: attention is every 4th block, so each streamed layer has three
  GDN blocks of GPU time ahead of it.

### 2.5 Adaptation
- **Partition (resident pages vs ring slots), one evaluation per ubatch** (READ `llama-kv-stream-plan.cpp:355-484`):
  - **Target:** the largest resident count that still leaves ring slots >= 1.10 x the streamed pages of
    one layer.
  - **Starved:** above the target, or misses > 1 % while the copy engine is below 80 % busy. After 3
    starved evaluations, one page per layer is demoted into ring slots.
  - **Overprovisioned:** misses <= 1 %, copy engine < 50 % busy and ring peak < 50 %. After 8 such
    evaluations, one page per layer is promoted back.
  - A 64-evaluation cooldown separates repartitions.
  - Thresholds: `:400-406`, `plan.h:171-178`.
- **Decode layout** (READ `llama-kv-cache.cpp:1551-1625`): applies when a ubatch has <= 32 query tokens
  (`MAX_DECODE_QUERY_TOKENS`), 3 such ubatches arrive in a row (`3c2e23017`), and active pages exceed
  resident pages.
  - The layout stops spreading streaming over all layers. Most layers hold all active pages and a
    minimal set of split layers streams the excess. The split layers are spread evenly through depth so
    the resident layers in between give the copy engine time (READ `fattn.cu:472-505`).
  - A wide ubatch leaves this layout immediately.
  - The extent used is the maximum over the last 16 ubatches, so slots that alternate do not flip the
    layout every time (READ `:1548`).
- **Cost of every layout or partition change:** `cudaDeviceSynchronize`, then **the whole resident cache
  is invalidated** and lazily reloaded from host. The reason is that per-layer planes move whenever page
  counts change (READ `fattn.cu:666-676`, `ggml-cuda.cu:1670-1712`). In decode layout the active page
  count grows every 256 tokens, so this happens once per page. That costs about 125 ms per 256 tokens,
  roughly 1 % (MEASURED, memory `invariant_kv_stream_stage_capacity_and_relayout`). Padding the layout
  made it worse.
- **CUDA graphs** are used only while every layer is fully resident and KV writes can be mirrored. Once
  pages stream, the graph is launched eagerly (READ `ggml-cuda.cu:3268-3300`, `:3320-3345`).

### 2.6 PCIe per decode step
The ring is scratch, so streamed pages are **re-uploaded on every step**; nothing is reused across steps.
Bytes per step = (attended cells - resident tokens) x 26,624 B for the 27B at q8_0/q4_0. Prefill
amortizes the same bytes over every query row of the ubatch.

**Measured on this box** (memory `invariant_kv_stream_stage_capacity_and_relayout`, `project_llama_8080_final_config_20261014`):
- Stage 1024: target-only decode 54.6 ms at 39k, 70.4 ms at 51k. The bandwidth model predicts about
  +44 ms for that 11k-token excess, more than the +15.8 ms measured. Either that timer missed copy
  waits or the extent differed, so re-measure it (§11).
- Live config (stage 1536 = ~60k resident across all slots, `-np 3`): 35.0 / 27.0 / 23.9 tok/s at
  2k / 39k / 56k.

## 3. Multi-slot unified KV and the tape

- **`-np 2 --kv-unified` streaming** (`002b6bce1`): the gate really needs one KV *stream*, and unified KV
  always has one, so the check was widened (READ `llama-context.cpp:517`). The commit itself says
  adaptation still assumes a single growing sequence.
- **Slot isolation** (`995e30959`, READ `llama-kv-cache.cpp:588-680`, `:1249`):
  - A ubatch attends only up to the highest cell of its own sequences.
  - In the unified stream, a sequence grows right after its own newest cell. A new sequence starts after
    an 8,192-cell reserve.
  - Removing a whole sequence compacts every remaining sequence that sits >= 4,096 cells above the lowest
    run that fits it. Rows are moved with host get/set, which also resets residency.
  - MEASURED (commit message, 27B, main slot at 39k): 30.4 tok/s with a clean pool, 21.2 tok/s with 25k
    foreign cells before the fix, 29.0 tok/s after it, byte-identical output. After a hot-swap erase the
    main slot's extent dropped from 67,840 to 20,224 cells, and speed went from 19.0 to 32.5 tok/s.
- **Tape behaviour Ariadne relies on** (memories `project_strata_goal_clone_fork_tape_for_big_moe`,
  `feedback_strata_tape_means_in_place_homes`):
  - The prefix is append-only and reused in place (99 %). Rewriting happens only at epoch boundaries.
  - Main is slot 1 and the sidekick is slot 0, under `-kvu --no-cache-idle-slots`. A second
    conversation never evicts the first.
  - **No KV shift.** qwen35 uses IMROPE (4 position components), so `can_shift` is false by construction
    (`bad7199cd`). K-shift, cache reuse by shifting and the splice endpoint (`a8ef03db0`) cannot run on
    this model.
  - A hot swap prefills the next epoch in the sidekick slot, promotes it, then erases the old slot.
- **Server pieces worth keeping as ideas** (all MEASURED in commit messages):
  - **Fork a shared prefix** (`8b825885e`): tag the existing cells for the second sequence instead of
    copying them, and restore recurrent state from the source's checkpoint at or below the shared
    prefix. A second agent's first turn read 1,505 instead of 5,611 tokens.
  - **Prefill share** (`0f5d1a15d`): after a mixed batch, run decode-only steps until prompt work is back
    to its share of wall time. A decoding slot beside a 67k prefill went from 0.88 to 12.0 tok/s, and the
    prefill itself sped up from 253 to 364 tok/s.
  - **Split prompt batches** across slots that are reading prompts (`e244c23ca`). The short prompt's
    wait fell from 9.43 s to 0.92 s.
  - **One checkpoint budget** shared across slots; idle LRU slots release theirs first (`83fb37973`).
- **Recurrent state and rollback memory** (MEASURED, memory `invariant_kvu_pool_shared_memory_budget`):

  | item | cost |
  |---|---|
  | GDN state | ~157 MB per slot |
  | checkpoints | default 32 x ~157 MB = 5 GB host |
  | RS rollback buffer (VRAM) | slots x (1 + n_max) x ~150 MB; `-np 3`: n_max 2/3/4 = 1347 / 1796 / 2244 MiB, n_max 4 OOMs |
  | prompt cache | `--cache-ram` default 8 GiB |
  | target KV | 26.6 KB/token, pinned, sized to the whole pool (7.0 GB at 262,144) |

  With `-kvu` the context is one pool shared by every slot. Nothing is sized slots x context.

## 4. Speculative decoding in the fork

- **Types (upstream catalog):** draft-simple, eagle3, MTP, DFlash, DSpark (DFlash plus a Markov head),
  ngram-simple, ngram-map-k, **ngram-map-k4v**, ngram-mod, ngram-cache (READ `common/common.h:173-184`).
- **DFlash2 is a block-diffusion drafter**, not a small transformer. It has 5 non-causal layers plus an
  output head, block size 8, and its own 248,320-token vocab. It consumes target hidden states captured
  at target layers [6, 20, 34, 48, 62], and its Q6_K weights are 1,506 MiB (MEASURED, memory
  `fact:limbo-spec-decode-wins`).
  - Its batch was capped to its noise blocks, which shrank the draft compute buffer from 554 to 73 MiB
    (`0974d6a21`).
  - Q2_K acceptance was 0.31-0.41 with mean accepted length 2.9-3.15. With q8_0 draft KV it rose to
    0.42-0.57 and 3.1-3.85 (MEASURED, memory `project_27b_decode_regression_is_config`).
- **Measured multipliers on this box** (memory `fact:limbo-spec-decode-wins`, `fact:resident-sweep-2048`):

  | model | drafter | base -> spec tok/s | gain |
  |---|---|---|---|
  | 27B IQ3_XXS | DFlash2, n-max 4 (the peak of the sweep) | 22.3 -> 43.1 | 1.93x |
  | 27B GSQ-RCO-mtp | MTP | 23.0 -> 44.8 | 1.95x |
  | 35B-A3B | MTP | 77.3 -> 123.5 | 1.60x |

  ngram-mod is useless without a draft model at 2k.
- **v1 does not stream draft contexts.** The config gate allows only the target context. MTP's one
  layer of KV costs ~4 KB/token at f16, and the DFlash2 cache is 66-69 MiB at a 98k pool.
- **The 32-row limit:**
  - A verify of more than 32 rows (for example ngram n-max 32 plus the sampled token = 33) counts as
    "wide". It leaves the decode layout, which means a sync plus a full resident reload. Keep
    draft max + 1 <= 32.
  - Verify cost grows with context: 55 ms at 2k, 79 ms at 39k for 3 tokens, while the draft stays flat
    at ~10 ms. Multi-row FA over q8_0/q4_0 with GQA 6 is compute-bound; the 2-row vector path is as slow
    as the 3-row MMA path (MEASURED, memory `invariant_kv_stream_stage_capacity_and_relayout`).
  - That works out to ~0.65 us per resident token per verify step, while reading those bytes at
    288 GB/s would take ~0.09 us. The kernel sits about 7x off the bandwidth roofline.

## 5. Raymond's V2: what is new, what is churn

**Worth taking (ideas):**

1. **Ordered-span attention that keeps stock order.**
   - An attention call reads at most three spans: resident, current ring and wrap-around. They are read
     in logical token order, and the running softmax state is carried from span to span ("split/resume")
     instead of merging independent partials (CLAIMED V2 README "How it works §1").
   - The point is that streaming no longer changes the reduction order. This fixes v1's drift.
2. **Phase arena.**
   - One device parent allocation is shared by:
     - prefill compute workspace;
     - decode workspace;
     - drafter scratch;
     - vision projector weights and encoder workspace;
     - resident KV plus the ring, which get whatever grant the active phase leaves.
   - A handoff drains work, invalidates stale executable bindings, releases changed leases, commits the
     new layout, then rebinds.
   - The point is that prefill-peak and decode-peak memory are never held at the same time (CLAIMED V2
     README §2; itsXactlY's analysis, READ LOCAL `docs/kv-stream-phase-arena-integration.md`).
   - The startup fitter measures both phase graphs and refuses an arena too small for either.
3. **MTP KV as a 17th attention layer** in the same pool, with its own host history.
   - Drafting leases its resident part plus the complete non-resident suffix in the ring. Catch-up and
     successive predictions reuse that lease and upload only new tail rows. The ring is released for
     the target verify right after the last MTP reader.
   - Rollback candidates are host-backed with bounded device staging: depth 3 = 18.7 MiB device plus
     449 MiB pinned host (CLAIMED). Compare v1's 1.3-2.2 GB of VRAM RS buffers above.
4. **DFlash2 with a bounded draft window** of 2,048 positions: an 18.3 MiB cache, depth-7 rollback
   staging 43.6 MiB, weights (~1,080 MiB) outside the arena (CLAIMED).
5. **K4V defined precisely.** This is upstream's `ngram-map-k4v` self-speculation (`common/common.h:181`):
   - a host-RAM map from n-gram keys to up to **four** alternative m-token continuations;
   - the index is ~1 MiB, rebuilt at generation boundaries; with images, only text after the last image
     is eligible;
   - in V2 it sits in front of DFlash2: a usable hit skips the neural draft, and every proposal is still
     verified.
   - K4V has **nothing to do with Q4 cache precision**. Measured gain is workload-dependent: +3.22 % in
     one 32k case, little elsewhere (CLAIMED V2 `docs/development/dflash2-k4v-swe.md`).
6. **Wide-prefill conversion reuse.** At >= 128 query rows over quantized KV, one 2,048-token F16 stage
   (8 MiB for the 27B) is converted once and reused across query tiles. Measured at 64k: +5.1 % for
   Qwen, +32 % for Gemma. The ring alternates halves for two-wave transfers, +1.3-1.8 % for Qwen
   (CLAIMED).
7. **Deadline pressure spreads streaming across all layers** instead of concentrating it. Extra ring slots
   then become prefetch headroom (READ V2 `src/llama-kv-stream-policy.h`).
8. Suffix rejection after a verify **rebases the populated prefix without moving device bytes**, and
   extends reservations only over newly committed host rows (READ V2 `src/llama-kv-stream-layer-lease.h`).

**V2's own numbers (CLAIMED):**
- RTX 5070 Ti (PCIe 5.0), IQ4_XS, Q8/Q4, 128k: stock+UVM 2.48 tok/s, V2 target-only 26.15, MTP3 59.7.
  MTP3 at 262,144: 22.3.
- RTX 3080 10 GB, Gen4 x16, measured **23.5 GB/s H2D**, IQ2_XS at 128k: 8.53 tok/s target-only,
  17.06 MTP3. The streamed KV payload reached ~17.6 GB/s, about 75 % of H2D.
- **This box has 6.6 GB/s, 3.6x less PCIe than that 3080.** None of these speeds transfer here.

**Churn and limits:**
- **"One serial request, one CUDA GPU, no parallel slots."** That conflicts directly with the tape.
  LOCAL's `-np 2 -kvu` work is not in V2.
- Very heavy qualification machinery: ~30 new kv-stream modules and ~35 test files, plus 3,654 changed
  lines in six core files.
- Gemma/vision/Windows qualification ledgers.
- NVML calibration notes (NVML over-read a 23.5 GB/s copy as ~41 GB/s; never trust NVML PCIe counters
  without known-byte copies).

## 6. The local expert-stream seam (old limbo window/publish path)

What it is (READ `src/llama-expert-stream.h:33-147`, uncommitted diff):
- Routed expert tensors are skipped at load time.
- The graph replaces them with views into a context-owned VRAM window, `[rows, cols, n_slot]` per role,
  plus a per-layer remap table from global expert id to window slot.
- Misses map to a dedicated all-zero slot, so the expert contributes exactly 0. That is Strata's skip
  semantics, and it changes outputs.
- An external runner (old limbo) dequantizes pack experts on the host to the window type and publishes
  them through the backend. That type is now F16 by default (`EXPERT_STREAM_WINDOW`), because a 4-bit
  block value needs ~11 significant bits and BF16 keeps 8.

What it proved:
- The pack bytes, the publish path and the router-id-driven fill are byte-correct.
- The loader really skips expert tensors (memory `bug:limbo-stream-dequant-bound`).

Where it stalled:

| # | problem | numbers |
|---|---|---|
| 1 | host dequant on one thread | ~14 MB/s, so **0.015 tok/s** windowed vs 78 tok/s resident (A3B) and 2.8 tok/s plain mmap |
| 2 | 12-slot window smaller than 10 experts x prompt tokens | refused experts silently contributed zero during prefill |
| 3 | CUDA graph re-capture on every token | 207 captures on a 48-layer streamed decode, 11.5 s/token vs 1.8 resident; uncommitted `GGML_CUDA_NO_GRAPHS` |
| 4 | Hadamard fold metadata vs measurement | applying the fold turned output into noise, so the fold is off by default (`EXPERT_STREAM_FOLD`) |
| 5 | slot floor and allocation failure | an 8-slot floor overrides the budget (`llama-expert-stream.cpp:99`), and a failed allocation aborts instead of refusing (`:146`) |

Lessons for Limbo:
- Move experts in their stored quantized format and compute on them directly. Never convert on the
  host per token.
- Never let a capacity miss silently change math.
- Keep the decode graph's addresses invariant so it can replay.

## 7. Flaws and limits

1. **Gating:**
   - qwen35 only; head dim 256 only; FA and KV offload required.
   - One CUDA device, one KV stream, target context only (v1).
   - KV type pairs: direct FA only for f16/bf16/q4_0/q4_1/q5_0/q5_1/q8_0 when built with all quants;
     otherwise an F16 conversion page (READ `fattn.cu:1016-1147`).
2. **Residency policy is "prefix by cell index."** It is not driven by attention mass or recency. The
   resident set is the oldest pages, and the newest (mutable) page streams unless its layer is fully
   resident.
3. **Contiguous per-layer planes** mean any change in page count forces a full device sync and a full
   resident invalidation (§2.5). This is the deepest structural flaw.
4. **No reuse across steps:** each step re-sends every non-resident page. Past the stage, decode is
   bound by PCIe on this box (§9).
5. **CUDA graphs are off while streaming.** Each decode then pays per-kernel launch overhead for
   thousands of nodes.
6. **v1 merge order drifts** once streaming starts (V2 fixes it).
7. **Multi-slot adaptation** assumes one growing sequence. It was only patched with "max of the last 16
   ubatches" and the 3-vote rule.
8. **Host memory is write-combined.** CPU reads of it are uncached, which is catastrophic for any
   CPU-side consumer (§8, §9).
9. **Verify attention is ~7x off roofline** (§4), and it is the largest context-dependent cost even
   while everything is resident.
10. **Pinned history cannot swap:** 7.0 GB at 262k for the 27B, on a 31 GB box.
11. **Configuration by environment variables:** `GGML_CUDA_KV_STREAM_PARTS`, `..._FIXED_RING_SLOTS`,
    `LLAMA_KV_STREAM_TRACE`, `GGML_CUDA_PREFER_KV_HOST`.

## 8. Conflicts with the other donors

1. **One PCIe link, two streamers.** Strata's expert path (decode misses, prefill lending from its RAM
   tier) and KV pages compete for the same Gen3 x8 link at 6.6 GB/s. DMA reads also consume DDR
   bandwidth that CPU-side expert or attention work wants. Neither donor knows about the other's
   traffic or deadlines.
2. **Pinned RAM.** Strata pins ~20 GiB of experts (RAM tier) on a 31 GB box. KV streaming pins
   26.6 KB/token (7 GB at 262k for the 27B), plus 5 GB of checkpoints and 8 GiB of prompt cache by
   default. Both cannot run at full size, and nobody arbitrates between them.
3. **VRAM.**
   - Competing for it: expert cache slots, resident KV pages, drafter weights and windows, rollback
     staging and phase workspaces.
   - Each donor sizes its own pool. Strata reserves VRAM at start, while kv-stream takes
     `--kv-stream-stage-mib`.
   - The memory measured that a 16k Strata batch slot costs ~20 % of the expert cache
     (`project_strata_tape_shared_standard`).
4. **CPU cores.** Several consumers want the same 8 cores:
   - modchip-style CPU expert GEMV and GDN, which wants N-1 threads with `--poll 0`;
   - any CPU attention lane;
   - io_uring/pread workers for SSD experts;
   - tokenizer/server threads;
   - the operator's desktop.

   The operator rule applies: never throttle main, and keep the desktop usable.
5. **Write-combined vs cached host memory.** WC helps or is neutral for H2D DMA and breaks CPU reads. A
   history that both the GPU and the CPU consume must be cacheable pinned memory (measure H2D cost).
6. **Graph ownership and kernels.**
   - kv-stream works by intercepting ggml FA nodes based on their buffer type and swapping tensor views
     inside the op. Kindi-style hand-written kernels and fused graphs have no such hook.
   - Limbo has to make spans, page tables and placement explicit inputs to its attention op.
   - Both seams (kv-stream and expert-stream) broke CUDA graph replay.
7. **Exactness contracts disagree:**
   - Kindi verifies token-exactness against llama.cpp.
   - v1 drifts once streaming starts.
   - Strata's zero-slot skip changes outputs.
   - V2 preserves stock order only for GPU spans.

   Limbo needs one stated contract (§9 last bullet).

## 9. The combination question: stream KV pages over PCIe, or attend on the CPU and merge?

Notation for the 27B at q8_0 K / q4_0 V:
- b = 26,624 B/token over 16 layers.
- C = context length, R = resident tokens, E = C - R (host-only tokens per step).
- n = query rows per step: 1 plain, 4 for an MTP3 or DFlash-n4 verify.

**Per-token lane costs per step** (each lane is a resource that can do this work):

| lane | cost per excess token | basis |
|---|---|---|
| PCIe stream | 26,624 B / 6.6 GB/s = **4.03 us** | MEASURED link rate |
| CPU, memory floor | 26,624 B / ~35 GB/s = **0.76 us** | ASSUMPTION: dual-channel DDR4, practical read |
| CPU, compute | 196,608 MAC per token per query row / ~0.5 TMAC/s = **0.39 us x n** | ASSUMPTION: Zen 2 8C, int8 maddubs, 50 % of peak |
| GPU attention, current kernels | ~**0.65 us** per token per step at n = 3 | MEASURED (§4) |

Each token-layer does 4 KV heads x 6 GQA heads x 256 x 2 (QK and PV) = 12,288 MAC per query row.
The CPU kernel's intensity is 7.4 MAC/B per query row, against a machine balance of ~14 MAC/B. So the
CPU lane is **memory-bound at n = 1 and compute-bound at n >= 2**.

**Hard floor, independent of placement:** every host-only byte crosses DDR once per step, whether the
CPU reads it or a DMA engine does. So t_step >= E x b / DDR.

**Worked numbers.** Assumptions:
- G0 = 64 ms of non-attention GPU work per step (verify 54 ms plus 10 ms of DFlash draft; memory timers).
- 3.0 accepted tokens per step.
- Per-layer CPU<->GPU handoff ~50 us x 16 layers, about 0.8 ms per step.
- The lanes run concurrently. The attention phase of a step is max(GPU resident and streamed attention,
  CPU lane), and the PCIe lane must finish within the step.

| context / resident | E (bytes) | (a) PCIe only | (b) CPU only | (c) three-way split | DDR floor |
|---|---|---|---|---|---|
| 40k / R >= 40,192 (stage 1024) | 0 | no streaming; step ~90 ms, ~33 tok/s | same | same | 0 |
| 100k / R 40k | 60k (1.59 GB) | 241 ms copy-bound, **~12 tok/s** | 64+94 = ~159 ms, **~19 tok/s** | x = 28k on PCIe (PCIe-bound): ~114 ms, **~26 tok/s** | 45 ms |
| 100k / R 60k (stage 1536) | 40k (1.06 GB) | 161 ms, **~18.6 tok/s** | 64+max(39, 62) = ~127 ms, **~23.6 tok/s** | x = 10.6k on PCIe: ~111 ms, **~27 tok/s** | 30 ms |
| 200k / R 60k | 140k (3.73 GB) | 565 ms, **~5.3 tok/s** | 64+218 = ~282 ms, **~10.6 tok/s** | x = 50.5k on PCIe (PCIe-bound): ~204 ms, **~14.7 tok/s** | 107 ms |

Same table with a GQA-aware GPU decode kernel that dequantizes each K/V block once for all 6 x n rows
(assumed 4x faster: 0.16 us/token):
- (c) at 100k/R 60k: ~91 ms, **~33 tok/s**.
- Attention stops dominating. The rest is G0, which is 37 ms of weights at 288 GB/s plus the draft.

**Answer:**
- **(b) CPU attention beats (a) PCIe streaming for decode on this box.** It is 1.3x faster at 100k and
  2x faster at 200k, because DDR outruns a Gen3 x8 link by ~5x. The gap closes as n grows, since the
  CPU lane turns compute-bound.
- **(c) is best:** GPU over resident pages, the copy engine streaming the share it can hide inside the
  step, the CPU on the remainder, and an LSE merge in fixed page order.
- **Prefill is the opposite case.** One upload serves 512-2,048 query rows. CPU compute scales with n, so
  a 512-row ubatch on the CPU would cost about 512 x 0.39 us per token. Prefill should stream, or for
  short chunks keep K/V in place.
- Crossover in query rows (CPU compute = PCIe time per page): n* ~ 4.03 / 0.39 ~ **10**. Below ~10 rows,
  attend host pages on the CPU. Above, move the bytes.
- **The 40 tok/s bar** for the dense 27B past ~60k needs all three:
  1. the three-way split;
  2. a GPU attention kernel near roofline;
  3. spec decoding at ~3 tokens per step.

  At 200k the DDR floor alone (>= 107 ms per step) caps it below ~28 tok/s on this box.
- **Exactness contract for Limbo:** treat the per-page partial (max, normalizer, numerator) as the
  canonical unit and merge partials in page order, whatever lane produced them. Results then depend
  only on page boundaries, not on placement. They still differ from a single-pass kernel, and CPU and
  GPU partials of one page are not bit-identical, so greedy token parity needs a tolerance or a pinned
  placement in tests.

## 10. What Limbo should take (ranked by payoff on this box) and what to leave

**Take (ideas, reimplemented):**
1. **Host-authoritative paged KV with a device page table.** Page = 256 tokens per layer. Device slots
   are reached through a block table, not contiguous per-layer planes. Promotion and demotion move
   single pages with no global sync and no invalidation, and decode-graph addresses stay fixed so CUDA
   graphs replay. This removes v1's worst flaw. Long context for dense models depends on it.
2. **The three-lane decode attention planner (§9)**, with lane rates measured at startup and refit
   online.
   - Lanes: resident GPU pages, PCIe ring pages with cross-layer deadline prefetch, and CPU attention
     over cacheable pinned host pages.
   - Merge: LSE in page order.
   - Payoff: ~1.5-2.8x over v1-style streaming at 100-200k on Gen3 x8.
3. **A GQA- and verify-aware GPU attention kernel.** It reads each quantized K/V block once and applies
   it to all heads x rows. It is today's largest context-dependent decode cost (~7x off roofline,
   §4), even with zero streaming.
4. **Speculative decoding as a first-class phase:**
   - MTP KV as an extra pool layer;
   - bounded drafter windows (DFlash2 2,048 positions);
   - host-backed rollback snapshots with small device staging (V2: 18.7 MiB device instead of v1's
     1.3-2.2 GB);
   - suffix-rejection rebase without moving bytes;
   - the CPU K4V history map in front of the neural drafter;
   - keep verify rows <= 32.

   Payoff: 1.6-2.1x, measured. KV traffic is per step, so speculation also amortizes streaming.
5. **Phase arena.** One device budget is leased per phase (prefill, decode, draft, vision) and the
   resident KV gets the remainder. On a card at the edge, the freed MiB become resident pages: 1 GiB is
   ~40k tokens of the 27B.
6. **Tape semantics from LOCAL** (§3):
   - per-sequence attention extents;
   - sequences grow contiguously with a reserve;
   - compaction on erase;
   - shared-prefix fork by tagging cells plus a recurrent checkpoint;
   - prefill share and split prompt batches;
   - one global checkpoint budget;
   - no position-shifting surgery on IMROPE models.
7. **The write path:**
   - publish only changed rows to the host, bulk D2H for prefill ranges, mirrored writes for resident
     pages;
   - use **cacheable** pinned memory rather than write-combined, so the CPU lane can read it.
8. **Wide-prefill conversion reuse** (one F16 stage reused across query tiles) and two-wave ring halves.
9. **K mean-centering for Q4_0 K.** It is exact under the softmax shift, and if quality holds (measure
   KLD) it buys capacity: q4_0/q4_0 is 1,152 B/token/layer against 1,664, about 1.44x more resident
   tokens and 1.44x fewer streamed bytes.
10. **The feedback controller shape:**
    - inputs: miss ratio, copy busy ratio, ring peak;
    - grow and shrink hysteresis with a cooldown;
    - spread under deadline pressure.

    It belongs inside the single Limbo planner rather than a per-donor loop.

**Leave:**
- Interception by buffer type inside a fused FA op.
- Device syncs and full invalidation on every layout change.
- Write-combined history.
- Eager launches while streaming.
- The qwen35 and head-dim-256 gates.
- V2's single-request restriction and its qualification bureaucracy.
- Environment-variable knobs.
- The host-dequant expert window and zero-slot misses (§6).
- NVML PCIe counters.

## 11. Open questions that need a measurement on this box

1. Effective H2D on the Gen3 x8 link for 416 KiB page copies and for 32-page runs, from
   **write-combined vs cacheable** pinned memory, while kernels run. Is 6.6 GB/s still the rate under
   concurrent compute?
2. Sustained DDR4 read bandwidth, all cores and fewer cores, alone and alongside a 6.6 GB/s DMA stream,
   and how much the desktop leaves. Also the DIMM speed.
3. CPU attention throughput (AVX2; int8 q8_0 K with q quantized to int8; q4_0 V) per page for n = 1..8,
   for 2/4/6/8 threads. Find the thread count that still leaves the desktop and the GPU feeder thread
   unaffected.
4. Round-trip latency of one per-layer CPU-lane handoff: q device-to-host via mapped memory, a host
   flag, the partial host-to-device. The target is <= 50 us, times 16 layers per step.
5. GPU attention cost per resident token for n = 1..8 with the current kernels (one point: 0.65 us at
   n = 3), and with a GQA-shared-dequant kernel.
6. Re-measure decode at 39k/51k/60k/100k with copy-engine timing included. The memory's 54.6 -> 70.4 ms
   point is below the bandwidth model's +44 ms.
7. Mean accepted length at 50-200k on real Ariadne traffic for MTP vs DFlash2 vs DFlash2+K4V. Do
   acceptance and the 32-row cap still favour n-max 3-4?
8. KLD/PPL of K q4_0 with mean-centering vs q8_0 at long context.
9. Per-token launch overhead of the 27B decode graph, eager vs replayed, to price the page-table design.
10. Pinned-memory budget when a Strata-style expert RAM tier and a long dense KV history must share
    the 31 GB box. Which model mix should the planner refuse?
