# qwen4exp decode on the 4060 Ti: where the 1.5 tok/s went, and what 50 tok/s needs (2026-11-22)

Follows `bench/flash-next-gpu-gate-2026-11-22.md`, which put the Phase-2 port on the device at ~1.5 tok/s.
This is the measurement that came next, the residency slab of docs/flash-next.md 6.3/6.4, and the arithmetic
of why the operator's 50 tok/s bar is not reachable one token at a time.

`LIMBO_Q4E_TIME=1` on `tools/oracle_gate_gpu` reproduces everything below.

## 1. The clock

| field | what it is |
|---|---|
| `dense` | the device time of the 48 dense halves (HC reads, GDN/QSA, the shared expert), two events per layer — the GPU's own kernel time |
| `moe` | the stream span from a layer's dense half to the next layer's, the lane's wait included |
| `sync` | issuing the router's host round trip: D2H of `mixed`, router GEMV, top-k, D2H of the ids, 48x per token |
| `syncwait` | the `cudaStreamSynchronize` after it |
| `launch` | the host time issuing the dense half of a layer |
| `lane` | the CPU lane's exposed time, submit -> wait, per layer |
| `ple` | the host's 16 PLE table rows and their upload |
| `majf` | major page faults of the whole process across the token |
| `resident` | cumulative share of routed lookups served from the VRAM slab |
| `lane from ram / file` | cumulative: where the lane's experts were read from |

## 2. What it found

### 2.1 A warm working set is lane-bound; the model in use is drive-bound

`hi` (18 tokens), no slab: **22 ms** of device time against **360 ms** of exposed CPU lane, 0 major faults.
The lane moved 20 MB a layer in 8 ms — 2.5 GB/s, against Strata's measured 27.4 GB/s on seven workers.

`code` (111 tokens), no slab: **13,000 to 41,600 major page faults a token** — 52 to 166 MB off the SATA drive
a token at ~30-45 MB/s of random 4 KiB — lane 3,200 ms, 0.367 tok/s gen. The gate record's "~8 GB/s through
two workers" only ever described a page-cache-warm working set. **The 1.5 tok/s was the drive, not the port's
arithmetic and not the GPU.**

### 2.2 The lane scales with cores

`hi`, gate re-run each time: 2 workers 394 ms/token (lane 361), 4 workers 178 ms, **6 workers 158 ms** (lane
126) with the gate byte-identical (12/16, max&#124;d1&#124; 1.5421). 2.87x. `planes=1` is a further 1.23x (131 ms,
13/16) but is a different arithmetic contract, so it stays opt-in. 6.6's "2 workers = 8 GB/s" was a roofline
assumption for the 91 %-resident state, not a property of the box. The default is now every physical core
`default_cpus` leaves the lane.

## 3. The residency slab

A per-layer VRAM slab, slots sized to that layer's blob (gate | up | down, contiguous), admitted by decayed
routing frequency (gamma 0.998/token). A resident expert is computed **on the device** by the same `gemv`
kernels the dense weights use — which dequantize to fp32 and FMA, so it is *more* exact than the lane's three
int8 planes, not less. The router's own ids still decide which experts run.
`ops::moe_scatter_add` puts the lane's rows back in place so one `moe_combine` sums both halves.
`LIMBO_Q4E_CACHE_GB=0` is the correctness floor.

| capture | slab | agree | max&#124;d1&#124; | prefill tok/s | gen tok/s | steady-state token |
|---|---|---:|---:|---:|---:|---:|
| hi | off | 12/16 | 1.5421 | 2.42 | 2.50 | 281.6 ms |
| hi | 10 GB | 12/16 | 1.5420 | 4.12 | **11.48** | **49-50 ms (20 tok/s)** |
| code | off | 16/16 | 2.7084 | 0.475 | 0.367 | 3,200 ms |
| code | 10 GB | 16/16 | 2.7084 | **1.455 (3.1x)** | **3.589 (9.8x)** | 175-357 ms |

The gate is the same misses at the same oracle gaps with the same top-1 tokens whether an expert was computed
on the device or on the lane.

### 3.1 Measured and rejected

* **The pinned RAM tier (6.4's second tier) at any size.** 1 GB, 3 GB, 6 GB and 12 GB all lose. At 1-3 GB the
  ring is too small to survive the re-route (resident 4-19 %); at 6-12 GB the pin evicts the page cache the
  26.8 GiB PLE table and the residual experts live on (majf 10,896 a token, PLE 624 ms, `code` 0.462 against
  2.882 tok/s). It is implemented and off by default (`LIMBO_Q4E_RAM_GB`).
* **Making the lane wait for the filler**, so the drive has exactly one reader. Worse: 0.462 against 1.006
  tok/s. Six workers faulting in parallel beat one filler filling serially under the interference.
* **Deferring promotion to an expert's second touch**, so the lane's own read has warmed the page cache first.
  Worse: 56 % resident against 73 %, and the lane's extra work costs more than the cheaper copy saves.
* **Pinning the router's ids readback** (`int32_t sel[32]` → `cudaMallocHost`). This one is a *win*, and it
  moved 27 ms of the token from `sync` to `syncwait`: a pageable D2H makes the driver stage the copy, and the
  clock had been attributing the host's wait to the copy instead of to the GPU. Same 40 bytes, honest number.
  The lane's job index is pinned for the same reason.

## 4. What 50 tok/s actually needs

The slab took `hi` from 281 to 49 ms. The token is now `dense` **19.7 ms** (the GPU's own kernel time for the
48 dense halves) + `lane` 6.5 + the MoE's device half and the head, and the host is blocked for a large part
of the remainder — `launch` and `syncwait` together exceed the measured device time, i.e. the GPU is idle
through part of every token because the residual chain serializes GPU, lane and host at every layer.

**But the binding constraint is arithmetic, and the plan already says it (6.6a): a single-token decode cannot
reach 50 tok/s.** 3.83 GiB of dense weights at 216 GB/s is 19.0 ms — 52 tok/s at 100 % of the roofline and
zero overhead. Measured, the dense half alone is 19.7 ms. There is no headroom in a one-token step.

**50+ therefore requires verify windows**: read the dense weights once for W tokens. That is 6.8 item 1, and
it needs three things this port does not have yet, in order:

1. **A T-row `step_q4e`.** The building blocks all exist for rung 1 (`step_rows`, `read_logits_rows`,
   `commit`, and the whole `_rows` op set: `rmsnorm_rows`, `conv_silu_rows`, `head_norm_rows`,
   `gdn_scan_rows`, `attn_prep_rows`, `attention_rows`, `router_topk_rows`, `moe_combine_rows`). What is
   missing is the q4e layer itself over T rows.
2. **A grouped expert GEMM for the q4e formats.** `gemm.cu`'s `moe_kernel` — the A-fragment path that decodes
   a weight tile straight into the mma fragments, one launch per projection for every hit of a layer — is
   instantiated for IQ2_S, IQ3_S and Q6_K only. The q4e experts are IQ2_S (20 layers), **IQ3_XXS (17)**,
   IQ3_S (10), **IQ4_XS (1)** for gate/up and **IQ4_NL (39)**, **Q2_0 (9)** for down: four formats to add.
   Without it a window's hits cost 4 launches per expert, and at W=3 the union is 21 experts a layer.
3. **A draft source.** No MTP head in this file; prompt-lookup / n-gram, which the repo already carries for
   rung 1 and the 27B (`bench/ngram-2026-11-22.md`).

Cheaper wins available without any of that, measured or bounded:

* **Fuse the dense half's small kernels.** `q4e_hc_read` issues 4 separate `rmsnorm` launches for the 4
  streams; one `rmsnorm_rows` call does all four. 6 launches a layer, twice a layer, 48 layers.
* **int8 hyper-connections** (6.8 item 2): the HC matrices are 1.2 GiB of the 3.83 GiB a token reads, for
  almost no arithmetic — 3.83 → 3.23 GiB, 19.7 → 16.6 ms of `dense`.
* **The router's round trip**: `sync` + `syncwait` are the host waiting for the GPU at every layer. A
  doorbell (spin on a mapped flag written after the top-k) removes the whole-stream sync, and is the
  prerequisite for putting the step in a captured graph, which is what removes the per-launch cost.

## 5. Reproducing

```
F=.../Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf
LIMBO_Q4E_TIME=1 build/bin/tools/oracle_gate_gpu -m $F --oracle bench/oracle/qwen4exp-coder-code.jsonl --out c.jsonl
LIMBO_Q4E_CACHE_GB=0 ...                        # the correctness floor
LIMBO_Q4E_CACHE_GB=10 ...                       # the slab (default: all free VRAM minus 1 GiB)
LIMBO_Q4E_RAM_GB=3 ...                          # 6.4's second tier, off by default, measured worse
LIMBO_Q4E_WORKERS=2|4|6   LIMBO_Q4E_PLANES=1|3  # the lane sweep
```
