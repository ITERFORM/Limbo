# Rung 2 offload: the design the donors already specify

`qwen35moe` on a 16 GB card. Measured from the file, not estimated:

```
rung 2 (Cyber-Tiel-Coder-35B-A3B Q4_K_M, 21.58 GB)
  routed experts   ffn_gate_exps 5870 + ffn_up_exps 5870 + ffn_down_exps 7252 = 18.54 GiB
  everything else  attn 1234 + ssm 274 + shexp 131 + router 82 + embd 515 + output 398 + nextn 9 ≈ 2.6 GiB
  types: gate/up Q4_K, down Q5_K; 256 experts x 8 used x 41 layers; 1.80 MiB per expert per layer
```

2.6 GiB must be resident; 18.5 GiB must not. Streaming the misses a token at a time is out — 8 experts
x 41 layers x 1.80 MiB = 590 MiB a token, which is 49 ms even at a generous 12 GB/s over Gen3 x8, i.e.
below the incumbent's 51.4 tok/s before any other cost. So the experts have to live somewhere and be
brought in on demand.

## What I tried and why it was wrong

Last attempt (`--cpu-moe`, built and reverted 2026-11-23): the last N layers keep their routed stacks in
the file's mapping and the **CPU lane computes the picks**. It got rung 2 past the VRAM refusal for the
first time and then took an illegal memory access, because it does a D2H of the router's ids, a
`cudaStreamSynchronize`, a lane submit/wait and an H2D — **a full host round trip inside the step**. That
is the one thing the donor design forbids, and this engine's step brackets its streams with `fork()` /
`join()` (`gpu_decode.cu:145`, `:185`), which is where a mid-step synchronisation stops being legal.

## What the donors actually specify

`docs/donors/kindi.md:382-400` — **Lightning on 6-8 GB cards, "GPU computes everything"**:

- cold experts sit in **mapped pinned host memory**;
- each MoE layer has D device **slots** (default 10) holding the last used host experts;
- a **single-thread kernel maintains owner and age per slot on the GPU**: hits first, so a miss can never
  evict a slot in use this step, then LRU victims; **a duplicate pick in one step is one copy**;
- a copy kernel **pulls** each missed expert from mapped pinned memory with wide coalesced **16-byte
  reads** and writes per-slot pointer tables;
- the products run on device memory; **everything is inside the step graph, with no host round trip**.

Tuning, measured: 6 → 10 slots gave +5-6% on an RTX 2070 SUPER; more slots cost card-resident experts
(85 → 81).

Its measured verdict on this box class (`kindi.md:397-400`): on an RTX 2060 with 6 GB, 60 of 128 experts
on the card, **read 1,415 against llama.cpp's 520 (≈2.7x) and write 52.7 against 51.5 (a wash)**. The
donor's own summary: *"write gains over llama.cpp's CPU-expert offload are small when the host CPU is
decent; read gains are large."*

**Prefill** (`kindi.md:394-396`): the host experts of a layer are copied **whole** into one of two
staging buffers by a copy stream, **two layers ahead**, with events so the compute stream waits only for
its own layer. With a 512-token chunk nearly every expert is hit, so moving the whole layer is right.

**The rest of the mechanism** (`kindi.md:523-525`): a usage-count placement with share-per-byte greedy
across layers, **device-side LRU decode slots fed by in-graph pulls of missed experts**,
**double-buffered whole-layer staging for prefill chunks**, a CPU expert path with **union batching**
(row group read once per batch, kept in L1/L2), a persistent pinned pool, spin-then-futex.

**And the cost model** (`kindi.md:521`): draft length from `E(k)/cost(k)` where the cost includes **the
expert-union term for offloaded MoE** — a window's drafts share experts, so the second and later picks of
the same expert in one step are free.

**The whole-step graph is not optional** (`kindi.md:515-517`): *"Whole-step CUDA graphs with
device-resident token, position, routing and expert slots, one graph per token count T = 1..8, patching
only dynamic nodes. Payoff: removes host latency from MoE decode, so routing never syncs to the host.
**Required to reach 40 tok/s on a MoE with offload.**"*

`docs/donors/strata.md:80-82` supplies the layout that makes it cheap: for the Coder, **one expert's
gate rows + up rows + down rows are contiguous and blobs are layer-major**, so an expert is one pull.
A GGUF keeps the three stacks in separate tensors, so rung 2 pays three pulls an expert instead of one —
worth measuring before optimising (PLAN §12).

`docs/donors/modchip.md:240-242` is the cold tier: experts stay **file-mmapped**, file THP, `madvise`.

## The design, therefore

1. **Drop the CPU lane for rung 2.** The lane remains right for `qwen4exp` where the dense half is the
   minority; here 18.5 of 21.58 GiB are experts and the measured answer is to pull them to the device,
   not to compute them beside it.
2. **Cold tier**: the three expert tensors stay file-mmapped, and their **mapped pinned** mirror is what
   the pull reads. `cudaHostRegister(..., cudaHostRegisterMapped)` on tensors this size is not an option
   (it pins 18.5 GiB); the donor's route is a pinned pool holding the working set, refilled from the
   mapping — i.e. the same shape as the q4e RAM ring, sized to the hot set rather than to the whole
   tensor.
3. **Device slot table per layer** + the single-thread owner/age kernel: hits first, then LRU, a
   duplicate pick in one step is one copy.
4. **Pull kernel**: 16-byte coalesced reads from mapped pinned into the slot.
5. **Products on device** against the slot table, exactly as `gemv_experts*` already does
   (`gpu_decode.cu:210-217`) — it already indexes `Wexp[ids[e]]`, so a slot table is a change of the
   base pointer and a pointer table, not a change of the kernel shape.
6. **No host round trip**: the ids stay device-resident and the slot maintenance runs in-graph.
7. **Prefill**: double-buffered whole-layer staging, two layers ahead, copy stream + events; this
   replaces the batched int8 path's assumption that the stacks are resident.
8. **Then** whole-step graphs for T = 1..8 — the donor calls this the requirement for 40 tok/s with
   offload, not an optimisation.

## What this corrects in PLAN.md

PLAN §3.1's row rule ("compute where the bytes are, unless enough rows read them") is right, and its
crossover must be measured — but the *tier below VRAM* for a MoE is **pinned host memory that the device
pulls from**, not the CPU lane. The CPU lane is the answer when the dense weights are the majority
(rung 3: 3.8 GiB dense against 23.4 GiB of experts, and 2.6 vs 18.5 here — the same ratio, so rung 2
should follow rung 3's *shape*... except that rung 3's shape is a VRAM slab plus lane and the donors
measured the pull winning for read at 2.7x). The honest reading: **the donors measured the pull; the
slab-plus-lane for rung 3 was chosen before they were consulted.** Both are in the tree; the union term
in the cost model is what decides per token, and it is not implemented in either.
