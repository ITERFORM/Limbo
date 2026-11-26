# Inkling-2-Mini — the memory and bandwidth budget (2026-11-23)

`inkling2_loop`, rung 4, item 3 of docs/inkling2.md §7. The measuring device is `tools/ink_budget.cu`
(`make BUILD=build-ink2 build-ink2/bin/tools/ink_budget`), run on the desktop beside the compositor
(~761 MiB of the card in use, 14.98 GiB free of 15.65). Every number below is its output verbatim;
the tool exits non-zero when a check fails, and the run ended `all checks passed (0 failure(s))`.

## 1. Verdict

- **The file's routed experts weigh 6.75 GiB, not 2.76.** §7's figure came from `role_bytes`' expert
  role list, which names `ffn_gate_exps`/`ffn_up_exps` and silently misses inkling2's fused
  `ffn_gate_up_exps` (4,090.5 of the 6,915.4 MiB); the tool counts from raw tensors and the fixed
  `role_bytes` agrees (6.75 GiB). The looped core (layers 6–17) holds 3,564 MiB of that and the
  prelude/coda 3,351 MiB.
- **The bias table is no longer the memory problem.** The fused gather is 96 KiB of scratch
  (1,025 × 24 floats, rebuilt per visit); the dense `[q × kv]` form of §6 was the 26 GiB problem and
  is never materialised. What scales now is the K/V state: 192 KiB per position over 48 slots.
- **Residency is KV, and the sliding-window ring is worth 6x of context.** Measured: 8,326 MiB at
  n_ctx 256 → 11,348 MiB at 16,384, a fitted 191.9 KiB per position against the account's 192.0
  (the check). Against this box's free VRAM that is **~37,700 resident positions as InkGpu
  allocates today, ~221,000 with the ring** (40 of the 48 slots are sliding and only ever hold
  1,024 positions).
- **Decode runs at 44% of the bandwidth bound.** 2,780 MiB of reads per token at n_ctx 512 bounds
  the card at 98.8 tok/s; InkGpu measures **43.0 tok/s** (64 head steps, one stream) — 93% of
  llama.cpp's 46.19 tok/s incumbent at the same context and model. (Re-measured 2026-11-24 after
  the kernel tuning of `bench/inkling2-decode-tune-2026-11-24.md`: **50.0 tok/s, 51% of the bound**,
  above the incumbent.)
- **Rung 2 is a pull-count problem, not a bytes problem.** 878 MiB of expert rows per token (8 of
  128 per visit across 48 visits), 585 MiB of it the looped core's two extra passes; host-resident
  experts bound decode at 7.17 tok/s over the measured 6.6 GB/s PCIe, with 384 expert reads per
  token (288 on the looped core alone). PLAN.md §3.1 puts a decode row under the ~10-row
  crossover, so host experts compute on the CPU from pinned RAM or stay pinned in kindi's slot
  table; in-graph pulls only overlap what the table misses.

## 2. The file account

7.89 GiB of tensors (744). One expert at the looped core's layers is 2.320 MiB (gate/up + down),
8 of 128 used per visit.

| class | MiB | read per token |
|---|---:|---|
| routed experts, looped layers 6–17 | 3,564.01 | 3x |
| routed experts, prelude + coda | 3,351.38 | 1x |
| dense per-layer weights, looped core | 435.59 | 3x |
| dense per-layer weights, prelude + coda | 261.36 | 1x |
| token embeddings | 234.65 | 1 row |
| output head | 234.65 | 1x |
| norms / queries / biases (global) | 0.03 | 1x |

## 3. The state account (per slot, per position)

| what | size |
|---|---|
| f16 K/V | 4.0 KiB per position per slot; 48 slots → 192.0 KiB per position |
| short-conv histories | 3.66 MiB total (48 slots × 3 rows × 6,656 floats) |
| the fused bias table | 96 KiB of scratch (1,025 × 24 floats), rebuilt per visit |
| logits | 0.76 MiB (200,058 unpadded rows) |

## 4. The measured residency

`InkGpu::open`, the free-VRAM delta; the slope is a least-squares fit over the four points.

| n_ctx | resident | free before |
|---:|---:|---:|
| 256 | 8,326.00 MiB | 14.98 GiB |
| 1,024 | 8,468.00 MiB | 14.98 GiB |
| 4,096 | 9,044.00 MiB | 14.98 GiB |
| 16,384 | 11,348.00 MiB | 14.98 GiB |

Fit: **191.9 KiB per position** (account: 192.0), 8,276.84 MiB fixed at n_ctx 0 — the 8,083 MiB of
uploaded tensors plus the decoded F32 vectors, codebooks and scratch. `ok the measured
per-position slope matches the state account (48 slots x 4 KiB)`.

Resident context on this box: **~37,693 positions as allocated; ~221,043 with the sliding-window
ring.** Beyond the ring's bound even the 8 global slots (layers 5, 11×3, 17×3, 23) overflow the
card at 262k — 8 GiB of K/V state — which is where docs/paged-kv.md's host pages take over.

## 5. The per-token device reads of one decode step (MiB)

| at context | 512 | 4,096 | 32,768 | 262,144 |
|---|---:|---:|---:|---:|
| weights: dense (looped core 3x) | 1,568 | 1,568 | 1,568 | 1,568 |
| weights: 8 of 128 experts per visit | 878 | 878 | 878 | 878 |
| output head | 235 | 235 | 235 | 235 |
| K/V reads (sliding capped at the window) | 96 | 288 | 1,184 | 8,352 |
| conv histories + embedding row | 4 | 4 | 4 | 4 |
| **total per token** | **2,780** | **2,972** | **3,868** | **11,036** |
| **bound at 288 GB/s (tok/s)** | **98.8** | **92.4** | **71.0** | **24.9** |

Measured: InkGpu decode **43.0 tok/s** at n_ctx 512 (64 head steps, one stream) — 2,780 MiB/token
in the account, **44% of the 288 GB/s bound**. `ok the measured rate is inside the traffic
account's bound`. (Re-measured 2026-11-24 after the kernel tuning: **50.0 tok/s, 51% of the
bound** — `bench/inkling2-decode-tune-2026-11-24.md`.) The long-context column is the looped **global** layers' K/V reads (8 slots ×
full history), not the sliding ones: the window cap keeps 40 of the 48 slots at 4 MiB of reads per
visit at any length.

## 6. The rung-2 case (experts wherever they live)

| what | the number |
|---|---|
| expert rows read per token | 878 MiB (585 MiB of it the looped core's extra 2 passes) |
| host RAM over the measured 6.6 GB/s PCIe | 0.14 s per token → **7.17 tok/s** |
| one expert pull | 2.320 MiB, ~369 µs on the link |
| expert reads per token | 384 (48 visits × 8); 288 on the looped core (12 layers × 8 × 3 passes) |

A decode step is 1 row per expert product. PLAN.md §3.1's crossover (~10 rows) therefore puts
host-resident experts on the CPU from pinned RAM, not over the bus; and any expert the slot table
does not pin is pulled up to three times per token on the looped core, because each pass routes its
own 8. The loop is what turns a residency problem into a pull-count problem.

## 7. What the account decides

- The 40 sliding slots cost window-sized caches and nothing else — that is the ring, and it is the
  difference between 37k and 221k resident positions. The 8 global slots are the only true O(ctx)
  state and the only KV that must page to the host past ~221k.
- The dense bias table stays dead: 96 KiB of scratch forever, at any context length. No budget line
  for it grows.
- Short-context decode is at 44% of the bandwidth bound and 93% of the incumbent's rate; the gap to
  the bound is kernel efficiency across the 48-visit sequence, not memory capacity.
- Non-resident experts are budgeted in pulls (384 per token), not bytes: the slot table's pinned hot
  set and the CPU's pinned-RAM lane carry decode; the copy engine only pays above the crossover.
