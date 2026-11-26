# Paged, shared K/V: one pool, slots, a host tier (plan of record, 2026-11-21)

## The operator's ruling (2026-11-21, binding; it overrides anything below that says otherwise)

1. Contexts above about 64-90k are a luxury Ariadne does not need: its TAPE (retrieval-first context management) keeps prompts
   short. The **device pool is the design**. Nothing for the host tier or YaRN beyond the seams (a page-id range for host pages,
   a per-chunk owner flag).
2. Default is **one slot**. Ariadne runs `--slots 2`: slot 1 main, slot 0 sidekick, and the sidekick does nothing but prewarm
   (it reads the main's NEXT prompt prefix in the background). Prefix sharing therefore works in both directions, and the
   important one is **main <- sidekick**: when the main's request arrives with a prompt the sidekick already read, the main adopts
   the sidekick's pages and its checkpoint (zero copy: refcount; the recurrent snapshot is copied into the main's state) and
   reads only the rest.
3. `--slots N` set explicitly = parallel work (N independent requests, still one shared pool).
4. Gate: the sidekick prewarms a ~30k prompt, then the main asks with that prompt + a short tail: the main's log shows
   cached ~= 30k, its prefill reads only the tail, the pool holds about one copy.

The operator's requirements (2026-11-21):
- Limbo's default is **one slot**.
- For Ariadne: **main + sidekick share one K/V pool** (unified, as llama.cpp `-kvu`): one capacity for both, and the sidekick's
  prompt reuses the main's prefix pages instead of holding a second copy.
- **"Does not fit" is not an answer.** What the card cannot hold lives in host RAM (the adaptive-KV-streaming idea of the
  operator's fork, docs/donors/kv-streaming.md sections 2, 9 and 10), up to 1M positions (YaRN past the trained 262,144).

## 1. Pages

- A page holds 256 consecutive positions of one sequence (= `dec::kAttnChunk`, the split flash-decode chunk) for every
  full-attention layer. Page id p owns rows `[p*256, p*256 + 256) x n_head_kv` of each layer's pool plane.
- Inside a page the rows keep today's per-position layouts (`include/limbo/cuda/kv.hpp`): Q8/Q4 codes + scale planes;
  F16 K as rows, F16 V transposed per page (`[kv head][256 dims][256 positions]`).
- Two tiers. The device pool: Pd pages, sized at open from the VRAM budget. The host pool: Ph pages in pinned, mapped,
  **cacheable** memory (`cudaHostAlloc(..., cudaHostAllocMapped)`, never write-combined: the CPU reads it), sized by
  `--kv-host-gib` (default: what n_ctx needs, at most free RAM minus 8 GiB). Ids below Pd are device pages, the rest host pages.
- A slot (sequence) owns a page table: one int32 per chunk index (`n_ctx / 256` entries) in device memory with a host mirror,
  chunk -> page id, -1 = none.
- Refcounts per page (host side). A page with refcount > 1 is a shared prefix. Writing a position of a shared page copies the
  page first (copy on write).

## 2. Kernels through the page table

- Decode attention (`attn_part_*`, T = 1..8 rows): the block of chunk c reads page `pt[c]`; every address comes from the page's
  base row instead of the position. A host page is read through its mapped pointer, unless the CPU lane owns that chunk (then
  the GPU block exits and the CPU writes the chunk's partials).
- Preps (decode, rows, prefill): position pos lives in row `pt[pos / 256] * 256 + pos % 256`. Pages are allocated on the host
  before the launch (the page table in device memory is updated before a graph replays; graphs keep fixed addresses).
- Prompt attention (as built): an f16 pool is read in place by the tensor-core kernel through the page table (a key's row is
  `pt[pos / 256] * 256 + pos % 256`, a block of 64 keys is inside one page): one pass, the bits of the contiguous kernel at any length.
  A Q8 / Q4 pool goes through the window expansion (dequantise the pages into f16 windows of 4096 positions, as before the paging), so
  the kernel stays as it is there. (The first plan gathered the f16 pool into windows too; that costs a copy per layer and moves the last
  bits at every window boundary, 4096 positions.) Host pages would be read over the bus there: a chunk has hundreds of query rows, and
  moving bytes beats CPU compute above about 10 rows (donor spec section 9).
- The MTP block keeps its own small K/V. Snapshots hold recurrent state only. Both stay as they are.

## 3. Slots

- `--slots N`, default **1**. `--slots 2` = Ariadne: main (slot 1) + sidekick (slot 0) in one pool. A slot = its page table +
  its recurrent state (today's "home") + its captured graphs.
- llama-server's endpoints, because Ariadne speaks them (`src/context/endpoint.cpp` in the ariadne tree):
  `GET /props` (n_ctx, total_slots, model), `GET /slots` (`[{id, n_prompt_tokens, n_past, is_processing}]`),
  `POST /slots/{id}?action=erase` (the slot's pages go back), body field `id_slot` (route to that slot). `X-Limbo-Session` stays.
- Prefix sharing: a request for slot s whose prompt shares a prefix with slot t's tape up to position q takes t's latest
  checkpoint at c <= q, copies its recurrent snapshot into s's state, points s's page table at t's pages for the chunks below
  `c / 256` (refcount + 1), copies the page holding c when c is not page-aligned, and reads the prompt from c.
- Preemption, not batching: the server runs one request at a time. A prime (`max_tokens` 1, not streamed: Ariadne's prewarm) steps
  aside when another request waits: after any piece of its prompt (the backend synced, so the device is at the tape's end) it stops,
  keeps its tape, and answers 503 with `Retry-After: 1` ("preempted by a waiting request; the N tokens read so far stay cached on
  slot S"; Ariadne's `prime()` retries a 503). The retry, parked behind the waiting request, continues from the tape and reads only
  the rest. Any other request runs to the end. `scripts/prewarm_yield.py` gates it on the GPU.
- Pool accounting: pages are taken on demand and given back on erase or rewind (refcount - 1). When the device pool is full the
  coldest page (the lowest chunk of the biggest slot) moves to the host pool. Only when the host pool is full too does a request
  fail, with the numbers (400: "the pool holds N positions, the prompt needs M").

## 4. The CPU lane (host pages in decode)

- Per full-attention layer of a decode step or verify window: the GPU computes the partials of the device pages; the CPU those
  of the host pages it owns (the same (m, l, o) per chunk as the GPU's `po` / `pml`, into mapped buffers); the existing combine
  kernel merges every chunk in chunk order. Results then depend on page boundaries only, not on where a page sits.
- Handoff: after the layer's prep the rotated query (T x heads x 256 floats) goes to mapped memory; a host node in the captured
  graph (`cudaLaunchHostFunc`) runs the CPU partials on a pool of 8 workers (AVX2), and the combine waits for it. Measure the
  per-layer cost; past ~50 us replace it with a spin handoff on mapped flags.
- AVX2 kernels for Q4 and Q8 (F16 after): per (row, kv head, chunk): the scores of the query heads that share the kv head, the
  chunk's max and sum, the numerators; float accumulation in a fixed order.
- Planner: per step, split the host chunks between the CPU lane and the GPU reading them over the bus so that the layer's lanes
  finish together (rates measured at start). First version: all host chunks to the CPU.

## 5. YaRN past the trained length

- n_ctx above the model's `context_length`: YaRN rope scaling (factor n_ctx / trained, Qwen's recipe: 4 for 1,048,576) in every
  rope path (decode and rows preps, the prefill rope table, the MTP prep, the CPU reference). At n_ctx <= trained: off, bit-identical.

## 6. Gates

- Paging must not move a number: with every page on the device, greedy text and logits at a given K/V format equal the
  contiguous engine's bit for bit (`make test`, `scripts/spec_check.py --kv f16|q8|q4`, `scripts/needle_serve.py`).
- Two slots with a shared 30k prefix hold the prefix once (pool pages ~ prefix + both suffixes); the sidekick's request reads
  only its suffix; `/slots` reports both; erase frees.
- Host tier: CPU partials against GPU partials of the same page within 1e-5 relative; greedy may differ only at near-ties.
- Speed: fully resident decode at 32k / 128k / 262k unchanged (within 2 %); 262k at q8 with half the pages on the host, and
  1M with YaRN at q4: measured and reported (the DDR floor puts 1M near 8-10 tok/s on this box).
