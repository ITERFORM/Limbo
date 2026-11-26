# Limbo — plan

Limbo is a from-scratch C++20/CUDA inference engine, written by us under our own licence. It combines
the strongest ideas of four engines: llama.cpp-adaptive-kv-streaming, llama-modchip, Strata and kindi.
It answers Ariadne over one OpenAI-compatible HTTP hop. No donor code is linked or copied
(`docs/CLEANROOM.md`). Each donor is read once into its own spec, `docs/donors/*.md`, and those
specs, with their `path:line` provenance, are the only input the implementation uses.

Status 2026-11-19: design. No engine code yet. This file is the plan of record. Every wave below
ends at a gate that is measured on the desktop. The next wave does not start until that gate is
green.

---

## 0. Focus (operator, 2026-11-20): the Qwen trio, as fast as the card allows

"Make qwen, qwen flash, qwen-27b fly hard and fast as the main goal, with all tweaks; everything as
inspiration, combined without copy-paste." Gemma is not a target. Two peer conversation homes are the
default whenever they fit (`d2eeb70`).

| model | arch | incumbent on this box | floor / ceiling that bounds the target | target |
|---|---|---|---|---|
| Qwen3.6-35B-A3B UD-IQ3_XXS (rung 1) | `qwen35moe` | llama.cpp: decode 76.0 plain, ~120 with MTP; prefill 2,261 / 2,078 (512 / 2048) | ~1.3 GB of weights read per token at 288 GB/s ≈ 144 tok/s without speculation | decode >= 150 with MTP (2x plain llama.cpp), as close to 240 (2x its MTP) as verify cost allows; prefill >= 4,200 in llama-numerics mode |
| Qwen3.8-27B dense, GSQ-RCO IQ3_XXS-mtp | `qwen35` | llama.cpp: 22-23 plain, 43-48 with MTP / DFlash2 | 9.7 GB per token at 288 GB/s ≈ 30 tok/s without speculation | decode >= 46 with MTP (2x plain llama.cpp), beat its 48 with speculation; prefill >= 2x llama.cpp's (to be measured) |
| Qwen3.8-Flash-Next coder IQ1_M (rung 3) | `qwen4exp` | Strata live: decode 40.1, prefill 1,121; 30% of busy time re-reading (no tape) | 23.4 GiB of experts on a 16 GB card: VRAM hits + the CPU lane + PCIe Gen3 x8 | decode > 40.1 without lossy skips, prefill >= 2,200 (2x Strata), zero cold re-reads |

One engine: the CPU works BESIDE the GPU in the same step wherever that pays (§3.1) — never a separate CPU or
GPU product. `src/step.cpp` is only the exact test oracle the GPU kernels are checked against.

Order: kernels, multi-row verify and MTP on rung 1 → the 27B on the same kernels (new formats, dense FFN).
In parallel, the hybrid lane: CPU expert kernels + pinned RAM tier + residency planner, then GPU hits and
CPU misses in the same decode step and row-rule streaming in prefill — proven first on rung 2 (Cyber-Tiel
Q4_K_M, 21 GiB, does not fit; llama.cpp --n-cpu-moe: 51.4 / 58.9 MTP / 1,169 prefill), then Flash-Next;
later the CPU attention lane over host K/V pages for long context.

## 1. Why the first Limbo failed (`~/projects/limbo`, kept untouched)

| what it did | what it cost |
|---|---|
| Bolted expert streaming onto libllama's graph through a window/publish seam | 0.015 tok/s windowed vs 78 resident on the A3B; dequant ran on one core |
| Observed llama's graph from outside | 207 CUDA graph captures, 169 s load, 11.5 s/token |
| Debugged correctness inside someone else's qwen4exp forward pass | 1786 ms/token unstreamed and 7123 ms/token streamed, while still wrong ("the capital of France is Berlin") |
| Seven separate budget modules (arena, host_ledger, pin_budget, vram_slots, device_plan, pcie_balance, host_budget) | modules working against each other inside one engine |
| Had no verify loop | the model files' own MTP/DFlash heads, worth 1.54-2.10x on this box, were unusable |
| Ran waves L0 through L10-19 at once: wire, packs, residency, io ring, spec, slots and a ThinkPad build before any fast path existed | "wants too much at the same time, and failing everywhere" |

The rebuild therefore owns its forward pass and its allocator. It proves one model exactly before it
gets fast, keeps one planner over every budget, and runs its waves one at a time.

## 2. Why the donors work against each other

Desktop: RTX 4060 Ti 16 GB (288 GB/s) on **PCIe Gen3 x8, 6.6 GB/s measured**; Ryzen 7 3800X, 8
cores / 16 threads, AVX2, no AVX-512, two 16 MiB L3 domains (cpus 0-3,8-11 and 4-7,12-15); 31 GB of
DDR4; models on a SATA SSD that also holds the swap.

| resource | who claims it, each as if it owned it alone |
|---|---|
| PCIe link | Strata: decode miss share (~5.6 ms per window), adaptive swaps (~29 MB per window), prefill streaming (saturates the link), sparse-KV page misses. KV fork: every host-only KV byte on every decode step (26.6 KB/token on the 27B). kindi: expert pulls, layer staging, checkpoint copies |
| pinned RAM | Strata ~19 GiB of expert complement. KV fork 7 GB of host KV at 262k, plus 5 GB of checkpoints and 8 GiB of prompt cache. The box has 31 GB |
| VRAM | Strata's 7.14 GiB expert cache plus all KV in VRAM leaves 591 MiB at start, and requests saw 198-437 MiB. KV fork stage, drafters (1.4-1.5 GB), GDN rollback (1.3-2.2 GB in v1) |
| CPU cores | Strata 7 workers plus a spinning host core; modchip all cores but one, busy; kindi one pinned thread per core. The desktop and the Mazemaker pods share the same 8 cores |
| the graph | ggml patches (KV fork, modchip) vs hand-written engines (Strata, kindi). The v1 seam between them ran at 0.015 tok/s |
| numerics | kindi: token-exact vs llama.cpp. Strata: approximations on by default (route-tail skip drops 43% of misses). KV v1: drifts once streaming starts |
| the tape | KV fork: main + sidekick slots, the good model. kindi: parks, which is a move. Strata: no tape, so 106 cold re-reads (3.36M tokens, 2,714 s) took **30% of its busy time** in 3 hours on this box |

## 3. The combination

### 3.1 One rule for experts and KV pages: compute where the bytes are, unless enough rows read them

Two specs, working on different objects, arrived at the same crossover:

| object (desktop) | CPU from pinned RAM | over PCIe to the GPU | crossover |
|---|---|---|---|
| missed expert, 2.05 MB (Flash-Next coder) | ~75 µs (live pool, 27.4 GB/s) | ~305 µs | **~10-18 rows/expert** (EST) — `donors/strata.md` §10 |
| host KV, per token per step (27B, q8_0/q4_0) | 0.76 µs memory + 0.39 µs × rows (ASSUMED DDR ~35 GB/s) | 4.03 µs | **~10 query rows** (EST) — `donors/kv-streaming.md` §9 |

So for every non-resident item (an expert, a KV page) in every step:
- **under ~10 rows** (decode, verify windows of 1-8 rows): the CPU computes it from pinned RAM and
  only activations or softmax partials cross the bus;
- **over ~10 rows** (prefill chunks: ~320 rows per expert in an 8K chunk): the copy engine moves the
  bytes to the GPU;
- **resident items** always run on the GPU, with a measured DMA share on top so the link is never
  idle while the CPU is the bottleneck.

The thresholds are estimates now. W0 measures them, and the planner refits them online. Strata
already does this split for experts (GPU hits, CPU misses, 15% over DMA). Nobody does it for KV, and
nobody does both under one budget. The three-way KV split is estimated at ~27 tok/s at 100k
context against ~18.6 for streaming alone (27B numbers, `donors/kv-streaming.md` §9).

### 3.2 One planner, one currency

The planner replaces v1's seven budget modules and Strata's pile of rules:
- **What it measures:** the box. W0 probes the lane rates: PCIe, DDR, CPU µs per format per row,
  GPU kernels, the handoff (doorbell) round trip, and the SSD.
- **What it reads:** the GGUF geometry, the session demand (Ariadne's main + sidekick and their
  contexts) and the drafter.
- **Residency:** VRAM holds the dense weights, hot experts, resident KV pages, recurrent states, the
  drafter, and the per-phase workspaces leased from one arena. Pinned RAM holds the exact
  complement of the experts, host KV pages and checkpoints. Files hold only the overflow and
  random-access tables such as PLE.
- **Per step:** the lane for every non-resident item (§3.1), the DMA share, draft length k, and
  chunk size.
- **Currency:** expected step-milliseconds saved per byte, for VRAM and pinned RAM; milliseconds per
  step for the PCIe, DDR and CPU lanes.
- **Online:** decayed routing counts with 1.5x hysteresis and per-layer quotas, swaps overlapped
  with decode, and a KV feedback loop (miss ratio, copy busy, cooldown).
- **Refusal:** a configuration that cannot hold the tape plus the desktop's headroom is refused at
  start, with the numbers. Pinning is paced and cgroup-aware. Pinned memory is **cacheable**, not
  write-combined, because the CPU lane reads it, and backed by 2 MiB anonymous pages. Random-access
  tables never get huge pages.

### 3.3 One arithmetic contract (from kindi, with KV and Strata refinements)

- Batch-invariant products: per-token reductions and fixed-order split-K. Verify is the same code
  as decode.
- Recurrent layers verify without writing back, then commit exactly the n accepted tokens.
- Attention: the per-page partial (max, normalizer, numerator) is the unit, merged by LSE **in page
  order** whichever lane made it.
- The oracle is the operator's llama.cpp builds on the same GGUF: greedy tokens plus top-2 gaps.
  The near-tie rule: the first difference is a tie when the reference's top-2 gap is under 0.01 nats.
- Approximations (route-tail skip, the fast head, pruning, CPU-share rounding) are opt-in only, each
  with a KL number. Nothing lossy is on by default.

### 3.4 Tape-native sessions (Ariadne's hard requirement; Strata's largest measured loss)

- Each conversation is a **home**: its own KV page table, recurrent state and checkpoints. Main and
  sidekick both stay live, and switching between them is a pointer select. There is no parking, no
  copy and no KV shift (IMROPE forbids shifting anyway).
- Recurrent checkpoints sit at the message boundary before the generation prompt, at the end of the
  system prompt, and every 256 tokens through the opening. Each goes into a VRAM slot, with a
  preemptible host copy made while idle.
- Shared prefixes fork by tagging pages plus one checkpoint. Saved openings and turns go to disk
  under an engine fingerprint. Restart snapshots go to the home directory, never `/tmp`.
- A sidekick's prefill never starves main's decode: prefill gets a bounded share of each step.

### 3.5 Speculation is a phase, not an add-on

- One captured graph per window size, T = 1..8. The products read each weight once for every row.
- The draft length maximises E(k)/cost(k), where cost includes the expert-union growth U(m) of
  offloaded experts and the KV lane cost.
- Every model on the ladder carries an MTP head. Prompt lookup is gated by cost and probes with one
  token. DFlash2 and DSpine come with the dense 27B, later.

### 3.6 Where each donor's edge lands

| from | idea | Limbo module |
|---|---|---|
| kindi | exactness contract, whole-step graphs with on-device token/position/routing, int8 tensor-core decode, split flash-decode on q8_0, codebook sign trick, magic-float accumulator, L1 signed-codebook AVX2 GEMV, union-batched CPU experts, checkpoint/opening/turn cache, E(k)/cost(k) with U(m) | `cuda/`, `cpu/`, `step/`, `session/`, `spec/` |
| Strata | three-tier expert residency, split decode with doorbells and monotone flags, DMA share rule, adaptive swaps, lend-region prefill streaming, routing look-ahead, CLOCK resolve for sparse KV, physical-core pool with single-CAS claims, parity benches per kernel | `plan/`, `mem/`, `step/`, `cpu/` |
| KV fork | host-authoritative paged KV, cross-layer deadline prefetch, tape slot semantics, prefix fork, prefill share, phase arena, host-held rollback, K mean-centering for 4-bit K | `kv/`, `session/`, `mem/` |
| modchip | non-convoy threading, CCX placement, fused one-pass GDN, MoE active-expert compaction with an atomic work counter, quantize-once groups, NTA prefetch, 2 MiB pages, the idea of a GQA-sharing CPU flash-decode, format facts (signed Q2_0, per-layer expert width) | `cpu/`, `mem/`, `gguf/` |

**Left out:** donor kernels in any form; Python fronts; parking as a move; env-var thickets; lossy
defaults; AVX-512-only designs presented as wins (AVX-512 never runs on either box); ggml-graph
interception; write-combined history; VQ3_64 and Fractal-64.

## 4. Shape of the engine

- **Threads:**
  - one GPU feeder thread, pinned to a reserved core;
  - one CPU pool, on physical cores only, CCX-aware, spin-then-futex;
  - io_uring for the file tier;
  - one HTTP thread.

  Hand-offs are lock-free (rings, atomics, futex park). No mutex.
- **Modules:**
  - `gguf/`: mmap reader that validates every tensor by (type, bytes);
  - `tok/`: byte-level BPE and per-family chat renderers, checked against golden fixtures from the
    real template;
  - `plan/`, `mem/` (VRAM arena with fixed addresses, pinned arena, file tier), `kv/`, `session/`;
  - `model/`: qwen35moe first, then qwen4exp;
  - `cuda/`, `cpu/` (AVX2 with runtime dispatch);
  - `step/`: graphs, doorbells, verify/commit;
  - `spec/`, `serve/` (OpenAI wire with tool calls, SSE, stop, usage, slot status for Ariadne).
- **Build:** GNU Make, the same as Ariadne. g++ with `-std=c++20 -Wall -Wextra -Werror`; nvcc for
  sm_89 first. Tests are standalone binaries, and every fast kernel has a parity bench against a
  plain reference.

## 5. The model ladder (all on disk) and the incumbents to beat

| rung | model | why this rung | incumbent on this box |
|---|---|---|---|
| 1 | Qwen3.6-35B-A3B UD-IQ3_XXS, `qwen35moe`, 13.1 GiB | fits the card with ~2 GiB left: "fits, but no room for context". KV ~10.9 KB/token at q8_0, so 262k needs host pages | llama.cpp fork: 78.2 tok/s, 120.7 with MTP (2k); 72.1 at 8k depth |
| 2 | Cyber-Tiel-Coder-35B-A3B Q4_K_M, same arch, 21 GiB | same model and code, **does not fit**: expert tiers and the CPU lane | llama.cpp `--n-cpu-moe 18`: decode 51.4, MTP 58.9, prefill 1,169 (ub 2048) |
| 3 | Qwen3.8-Flash-Next coder IQ1_M, `qwen4exp`, 54 GB | the daily driver: adds hyper-connections, PLE, QSA sparse attention, 512 experts | Strata live: decode 40.1, prefill 1,121, hit rate 78.9%, plus 30% of busy time spent on cold re-reads |
| later | Qwen3.8-27B dense (download, ~10.5 GB) | DFlash2 and DSpine drafters | llama.cpp fork: 22-23 tok/s, 43-48 with spec |

The operator's bars, decode ≥ 40 and prefill ≥ 500 tok/s, apply to every rung on top of beating
that rung's incumbent.

## 6. Waves

**W0 — foundations and the measurement session.** No GPU kernels yet.
- Repo, build and test harness. GGUF reader. Tokenizer and chat renderers.
- `limbo-probe`, which measures all the open questions of the four specs (§7).
- `oracle`, which records greedy tokens and top-2 gaps from the operator's llama.cpp builds for a
  fixed prompt set per rung (code, prose, tool calls, long context).

*Gate:* tokenization byte-exact against llama.cpp on a corpus that includes tool-call prompts; the
tensor tables of all three ladder files agree with llama.cpp's; probe numbers written to `bench/`.

**W1 — rung 1, resident and exact.**
- CUDA kernels for the rung's types, GDN (fused decode, chunked prefill), gated attention over
  **paged KV from day one**, grouped MoE, and router top-k on device.
- Whole-step graphs for T = 1..8, batch invariance, and homes for sessions.

*Gate:* token-exact against the oracle; decode ≥ 78 tok/s; prefill ≥ llama.cpp's on the same
prompt.

**W2 — speculation on rung 1.**
- MTP verify windows and commit, the cost-based draft policy, prompt lookup.

*Gate:* greedy output identical with speculation on and off; decode ≥ 121 tok/s at 2k.

**W3 — the tape, long context and the wire, on rung 1.**
- Main and sidekick homes, checkpoints, prefix fork, prefill share.
- Host-authoritative KV pages and the three-lane decode attention (GPU resident, copy engine, CPU
  GQA partials) with an LSE merge.
- Planner v1, and the OpenAI wire with tool calls.

*Gate:*
- Ariadne drives a real session end to end.
- A side call moves 0 bytes of main's state, and main's next turn reuses 100% of its prefix.
- Decode at 128k and 262k is within 10% of the lane model fitted in W0.

**W4 — rung 2: experts that do not fit.**
- Per-layer VRAM slots, profile pre-fill, the pinned complement, and split decode (GPU hits, CPU
  misses, DMA share).
- The per-expert row rule for prefill streaming, overlapped adaptive swaps, routing look-ahead, and
  AVX2 multi-column K-quant kernels.

*Gate:* token-exact; decode > 58.9 with MTP; prefill > 1,169.

**W5 — rung 3: the daily driver.**
- Hyper-connections, PLE (table residency set by the planner, rows by direct I/O with a row cache),
  QSA plus the indexer plus sparse-KV CLOCK, and the 512-expert kernels for IQ2_S, IQ3_XXS, IQ3_S,
  IQ4_XS, IQ4_NL and Q2_0.

*Gate*, measured against Strata on this box: decode ≥ 40.1 **without route-tail skip**, prefill
≥ 1,121, and zero cold re-reads caused by side calls.

Then `ariadne limbo` gets a `limbo` profile. Strata leaves :8080 only after a soak, and only on the
operator's decision.

**Not before W5:** a second architecture family, the ThinkPad backend (Vulkan or CUDA on Maxwell,
decided then), AVX-512 paths, vision, more formats (TQ trellis, PTQ1_0).

## 7. W0 measurement session (needs a quiet box: Strata stopped, operator's go)

Numbered by priority. The tag after each item names the spec sections that ask for it.

1. DDR4 sustained read at 1-8 threads, per CCX, alone and beside a 6.6 GB/s DMA.
   (modchip §13.1, kv §11.2, strata §12.4)
2. PCIe H2D from cacheable vs write-combined pinned memory, for 416 KiB pages and 2 MB experts,
   under GPU load; in-kernel mapped pulls vs the copy engine. (kv §11.1, kindi §12.3)
3. AVX2 µs per expert row for each ladder format (IQ3_XXS, Q4_K, Q5_K, Q6_K, Q8_0, IQ2_S, IQ3_S,
   IQ4_XS, IQ4_NL, Q2_0) at 1, 2, 3, 4, 8 and 16 rows with 2-7 threads. This sets the expert row
   rule. (strata §12.1, modchip §13.2, kindi §12.4)
4. CPU GQA flash-decode per 256-token q8_0 page for 1-8 rows. This sets the KV row rule.
   (kv §11.3, modchip §13.7)
5. Doorbell round trip GPU → CPU → GPU through mapped flags; target ≤ 50 µs per layer. (kv §11.4)
6. Interference: DMA against the CPU pool, launch latency with DDR saturated, and how the desktop
   feels. (modchip §13.3, strata §12.4)
7. Routing statistics per rung: hit rate against VRAM slots, distinct experts per verify window
   U(m). (kindi §12.5, strata §12.3, modchip §13.6)
8. dTLB with 2 MiB vs 4 KiB pages on the pinned tier. (modchip §13.4)
9. SSD random and sequential rates for packs and PLE. (strata §12.7, modchip §13.9)

The probes and the oracle captures are launches. Each one runs only after a yes, never beside the
live engine.

## 8. Rules carried in from the operator

- Lock-free and OpenMP, with SIMD dispatch (AVX-512, AVX2, scalar). Every AVX-512 claim says it
  never ran.
- No test launches on the workstation without a yes. Never use the shared `/tmp`. Kill only PIDs we
  started. Podman, never docker.
- Commits carry no AI trailers. Never `--no-verify`. Stage files by name.

## 9. Licence

Limbo's own licence is the operator's to name. Until then: all rights reserved, no donor code. Codebook
grids that must match the GGUF formats bit for bit come from ggml under its MIT notice, or are
regenerated, and are listed in `THIRD_PARTY_NOTICES`. kindi (AGPL-3.0) contributes ideas only, through
`donors/kindi.md`.
