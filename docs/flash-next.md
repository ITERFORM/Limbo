# Qwen3.8-Flash-Next (`qwen4exp`) on Limbo

The operator's main model is a 48-block hybrid (GDN + sparse full attention) mixture-of-experts with three
architectural novelties beside the Qwen3.5 family: **hyper-connections** (four parallel residual streams), a **block
indexer** that makes full attention sparse beyond 2,051 tokens, and **PLE** (a hashed n-gram embedding table of 26.8 GiB
read at layer 1). This file is the math of every component in our own words, what one token reads, how the CPU
reference runs it and where it is gated, and the plan for the GPU port.

Sources read for the math (read only, nothing copied): llama.cpp `src/models/qwen4exp.cpp` (the graph builder),
`src/llama-memory-hybrid-idx.cpp` (the indexer cache: pools, visibility), the ggml CPU op `lightning_indexer`; for ideas the
operator's Strata (`docs/DETAILS.md`, `src/core/expert_cache.cpp`, its coder run log). llama.cpp is the measuring device
(`oracle/`), never engine code (docs/CLEANROOM.md).

One correction to the task's vocabulary: there is **no compressed K/V cache**. `attention.compress_ratios` is the block
size (4) of the *indexer's pooled keys*, non-zero on the 12 full-attention layers (3, 7, ..., 47). K and V are stored
whole (f16 by default). What is compressed is the indexer's view of the past, so that a query can pick the blocks to attend to.

## 1. Geometry (read from the file by `src/model.cpp`)

| | |
|---|---|
| files | two shards, 54.39 GiB: shard 1 = 1,223 tensors, shard 2 = one tensor (`per_layer_token_embd`, 26.8 GiB IQ4_NL, `[160, 320,001,536]`) |
| blocks | 48 (no MTP block in this file); layer l is full attention iff (l+1) % 4 == 0 (12 layers), else GDN (36 layers) |
| width | embd 2560, vocab 248,320, ctx 262,144, RMS eps 1e-6 |
| attention | 24 q heads / 2 kv heads x 256; `attn_q` is `[2560, 12288]` = per head `[q(256) \| gate(256)]`; rope 64 of 256 dims, NEOX pairing, base 1e7 (IMRoPE sections 11/11/10/0, which for text positions is the plain rotation) |
| GDN | conv kernel 4, state 128, 16 key groups, 48 value heads, inner 6144, conv channels 2x2048 + 6144 = 10240 |
| MoE | 256 experts, 10 used, expert ff 640, one shared expert ff 640 behind a sigmoid gate; softmax router, no `expert_weights_scale` key (no scaling) |
| hyper-connections | 4 streams, low rank 320 |
| indexer | 4 heads x 128, top_k 2048, block (compress ratio) 4 on the 12 attention layers |
| PLE | layers [1]; n-gram 3, 8 heads per n-gram (16 rows of 160), conv kernel 4 (dilated by 3), multipliers `[23703573157769, 20109073645365, 8052911324071]`, 16 head ranges of ~20,000,0xx rows (primes: 20000003, 20000023, ... 20000171), EOS 248044 |

Weight formats in this file (type id in brackets): routed gate/up in IQ2_S (20 layers), IQ3_XXS (17), IQ3_S (10), IQ4_XS (1);
routed down in IQ4_NL (39) or **Q2_0** (9 layers: 1, 8, 10, 12, 13, 20, 21, 30, 36); dense in Q4_K / Q6_K / IQ4_XS / IQ4_NL / Q5_K;
the hyper-connection matrices, routers and indexer projections in **BF16**; norms and conv taps F32 (PLE conv taps F16).
`quants.cpp` decodes all of them (IQ4_NL and Q2_0 included, 0 differences against ggml's `to_float` on 5 sampled rows per tensor).

## 2. One token, in order

```
e   = token_embd[token]                                   (IQ4_XS row, 2560 floats)
R   = [e, e, e, e]                                        (4 streams)
for l in 0 .. 47:
    if l == 1: R += PLE(R, token window)                  (section 2.5)
    x, inj = HC_read(R; hc_attn_*)                        (2.1)
    out    = GDN(x)  or  QSA_attention(x)                 (2.2, 2.3)
    R_s   += 2 sigmoid(inj_s) * out                       (2.1)
    x, inj = HC_read(R; hc_ffn_*)
    out    = MoE(x) + sigmoid(g.x) * SharedExpert(x)      (2.4)
    R_s   += 2 sigmoid(inj_s) * out
x   = HC_read(R; output_hc_*)  (no inject)                (there is no output norm: the head mixer is it)
logits = output.weight (Q6_K, 2560 x 248320) @ x
```

### 2.1 Hyper-connections

The residual is a stack `R` of 4 vectors of 2560. Reading it for a module:

1. `x_s = rmsnorm(R_s) * gamma_s` per stream (`hc_*_norm`, 10240 floats = 4 x 2560; the file has the `1 + w` already folded in).
2. Flatten to `X` (10240). `z = silu((D X) / 4)` with `D` = `hc_*_down` (BF16, 320 x 10240); `gate = U z` with `U` = `hc_*_up`
   (BF16, 10240 x 320).
3. `mixed = (1/4) sum_s x_s * sigmoid(gate_s)` (elementwise; 2560 values): the module's input.
4. `inj_s = (1/4) (W_inj X)_s`, `W_inj` = `hc_*_inject` (BF16, 4 x 10240): one scalar per stream.

Writing the module's output `out` (2560) back: `R_s += 2 sigmoid(inj_s) * out` for every stream with the same `out`
(the "comb" matrix of the general mHC family is the identity here; the weight is centred on 1, so a zero injection is
a plain residual). The streams therefore differ only by their four weight sequences, and they meet again only in the
read gate. The same read, without inject, followed by nothing, produces the head's input. Per layer two reads: 4 x 6.55 MiB
of BF16 matrices = **1.2 GiB of the 3.8 GiB dense bytes of a token**, for almost no arithmetic: the hyper-connections
are the largest bandwidth item after the experts and the one most worth requantising on the GPU.

### 2.2 GDN (the 36 linear layers; qwen35moe's recurrence with one change)

`qkv = W_qkv x` (10240: q 2048 = 16 x 128, k 2048, v 6144 = 48 x 128), `z = W_gate x` (6144). A causal depthwise conv (kernel 4,
state = the last 3 inputs per channel) plus silu runs over the 10240 channels. q and k are L2-normalised per head of 128
(q additionally x 1/sqrt(128)); value head `h` reads key head `h mod 16` (tiled, not paired). Per value head, with
`beta = sigmoid(W_beta x)` and `g = softplus(W_alpha x + b_dt) * a` (`ssm_a` < 0 is stored, so `exp(g) < 1`):

```
S <- exp(g) S ;   d = beta (v - S^T k) ;   S <- S + k d^T ;   o = S^T q      (S is 128 x 128 f32 per value head)
```

then `out = W_out ( rmsnorm_head(o) * w_norm * sigmoid(z) )`. **The one difference from Qwen3.5 is that the output gate is
`sigmoid(z)`, not `silu(z)`.** State per GDN layer: 48 x 128 x 128 x 4 B = 3.0 MiB (108 MiB for 36 layers, read and written
every token) plus 3 x 10240 conv values.

### 2.3 Attention and the block indexer ("QSA")

Plain part (12 layers): `qfull = W_q x` (per head `[q | gate]`); `q` and `k` get a per-head RMS norm (weights `[256]`), rotary on the first
64 dims, `K, V` are cached in f16 (2 kv heads x 256 = 1 KiB each per position and layer), scores `q.k / 16`, softmax, `o = sum w v`,
`out = W_o (o * sigmoid(gate))`.

Sparse part, on the same input `x`, for a layer with block size `P = 4`:

* **Cache.** For every position the raw indexer key `k_raw = W_ik x` (128, f16) is stored. When a position completes a
  block of 4, the block's **pooled key** is made: mean of its 4 raw keys, RMS norm (`indexer.k_norm`), rotary on the first 64
  dims **at the block's first position**, stored f16.
* **Query.** `q_i = W_iq x` as 4 heads of 128, RMS norm per head (`indexer.q_norm`), rotary at the token's position.
* **Score** of a complete block `j`: `s_j = sum_h relu(q_{i,h} . pooled_j) / sqrt(128)`.
* **Selection.** A token at position `t` sees `nv = (t+1) / 4` complete blocks. If `nv <= top_k / 4 = 512` every block is
  chosen; otherwise the 512 best by `s_j`. It attends to the 4 positions of each chosen block **plus the incomplete tail**
  (positions `4 nv .. t`, at most 3 and always including itself). No mask other than this set.

So **until position 2,051 attention is plain causal attention over everything**, and beyond it every layer reads exactly
2,048 + (0..3) cells. That is what makes a 131k-token context affordable: per token, 12 layers x 2,051 cells x 2 KiB = 49 MiB of K/V
(f16) plus the pooled keys scored for the selection (64 B per context token and layer: 100 MiB at 128k), against 3.0 GiB
of K/V for full attention at that length. The indexer cache costs 320 B per position and layer (3.75 KiB per position for the
12 layers; 480 MiB at 131,072).

### 2.4 Mixture of experts

`p = softmax(W_r x)` over 256 (BF16 router), top-10 indices, weights `w_i = p_i / sum_top10 p` (denominator clamped at 6.1e-5), per expert
`down_i( silu(gate_i x) * up_i x )` with ff 640, summed with `w_i`; the shared expert (`ff` 640, SwiGLU) is multiplied by
`sigmoid(g . x)` (`g` = `ffn_gate_inp_shexp`, 2560 BF16) and added. An expert is `gate + up + down` rows of one layer: 1.44 to
2.54 MiB depending on the layer's formats (23.42 GiB over 12,288 experts, 1.95 MiB on average).

### 2.5 PLE: the hashed n-gram embedding (layer 1 only)

1. **Window.** The ids `(t_p, t_{p-1}, t_{p-2})`. A missing predecessor (before the sequence start) or an EOS (248044) predecessor cuts the window:
   everything behind the cut reads as EOS. The token's own EOS does not cut.
2. **Hash.** For `n` in {2, 3}: `m_n = (t_p M_0) xor (t_{p-1} M_1) [xor (t_{p-2} M_2)]`, unsigned 64-bit wrapping products.
   Head `h = (n-2) * 8 + g` (g < 8) reads row `m_n mod V_h + O_h` of the table (160 IQ4_NL values = 90 B), so each token reads **16 rows,
   1.4 KB, each in its own 4 KiB page** of a 26.8 GiB table; `emb` = the 16 rows concatenated (2560).
3. `key = K emb` (BF16, 2560 -> 4 x 2560), `value = V emb` (BF16, 2560 -> 2560).
4. Per stream: `kn_s = rmsnorm(key_s) * gk_s`, `qn_s = rmsnorm(R_s) * gq_s`, `s_s = kn_s . qn_s / sqrt(2560)`,
   `gate_s = sigmoid(sign(s_s) * sqrt(max(|s_s|, 1e-6)))`, `gated_s = gate_s * value`.
5. `nrm_s = rmsnorm(gated_s) * gc_s`; a depthwise causal conv over time per channel (10240 channels, kernel 4, **dilation 3**, taps
   `w[ch, 0..3]` for lags 9, 6, 3, 0; zeros before the start) and silu: `c`.
6. `R_s += gated_s + c_s`.

Per token: 63 MiB of dense weights (`ple_key` 50 MiB, `ple_value` 12.5 MiB) and the 1.4 KB of table; state: the last 9 `nrm` rows (360 KiB).
The hash needs only token ids, so every row a chunk of tokens will read is known before the chunk runs and can be prefetched.

## 3. What a decode step reads per token

Sizes are the file's own tensor sizes (`role_bytes`, over `Model::tensors()`).

| group | bytes per token | where it lives in the plan |
|---|---:|---|
| GDN projections (36 layers: `attn_qkv` 678, `attn_gate` 362, `ssm_out` 423, alpha/beta/conv 23 MiB) | 1,485 MiB | VRAM |
| attention (12 layers: q 250, k 12, v 12, o 144) + indexer projections 37.5 | 455 MiB | VRAM |
| hyper-connections (48 x 2 reads x 2 matrices x 6.25 MiB + inject + norms; head 12.5) | 1,224 MiB | VRAM (requantised, section 6) |
| routers 60 + shared experts 139 | 199 MiB | VRAM |
| `output.weight` (Q6_K) | 497 MiB | VRAM |
| PLE dense (`ple_key`, `ple_value`, conv, norms) | 63 MiB | VRAM |
| **dense total** | **3,923 MiB = 3.83 GiB** | |
| routed experts: 10 per layer x 48 x 1.95 MiB | 937 MiB | VRAM hits / RAM / PCIe |
| PLE table rows: 16 x 90 B in 16 pages | 1.4 KB (64 KiB of page traffic) | SSD / page cache |
| embedding row | 1.4 KB | RAM |
| GDN state read + write | 2 x 108 MiB | VRAM |
| K/V + indexer, per token at 2k / 32k / 128k context | 49 / 49 + 25 / 49 + 100 MiB | VRAM |

So a token reads **4.76 GiB** of weights (3.83 dense, 0.92 routed) before state and KV. At 288 GB/s the dense part alone is 14.3 ms;
the file's 54.4 GiB do not fit the 16 GB card, which is why the routed experts and the PLE table need a placement.

## 4. The CPU reference (`limbo::step::Ref`, `src/step.cpp`)

What was there and what this work added. The hyper-connection read/write, the PLE hash block and the 36 GDN / 12 attention layers with the
MoE + shared expert were already in `step.cpp` (recovered from an earlier wave in 6acd5b1) but never gated: the oracle captures for this
model date from today. Added now:

* **The indexer (QSA).** `Ref::qsa_select`: raw key cache, pooled blocks, scoring, top-k, tail (section 2.3). Before, the reference
  attended to everything and printed a warning beyond `top_k`. `LIMBO_QSA_TOP_K=N` is a test hook that shrinks the budget so a short prompt
  exercises the selection (llama.cpp takes the same value through `oracle_nodes --top-k`).
* **The two halves of a hyper-connection layer** are separately callable (`layer_hc_attn`, `layer_hc_ffn`, `forward_layer_ffn`) and
  `note_token` feeds the n-gram window, so `ref_forward --layer-gate` runs every layer from the oracle's own input on the 4 x 2560 stack,
  split attention/MoE; it also reads 10 experts per token now (it assumed 8).
* **RSS.** The weights are mapped, never copied (`RssAnon` is 135 MB over a run: the KV, the GDN state and scratch). Touched mapped
  pages count in a process's RSS and a token touches ~5 GiB, so over a prompt the resident set climbs towards the whole file.
  `Ref::set_rss_budget` (`oracle_gate --rss-budget-gb G`) returns the mapped pages to the page cache between tokens when they exceed G
  (`madvise(MADV_DONTNEED)`; the data stays valid and is faulted back from the page cache).
* **Tools.** `oracle_nodes --wide` (a `[n_embd, streams, T]` node is a row of 10240), `--top-k`, no weight repack (`use_extra_bufts = false`:
  llama.cpp's CPU repack **copies the model into RAM**, 14 GB in the first 11 seconds on this file; a capture on the operator's desktop must
  not do that; `oracle_noise` too), `routing_stats` (section 6).

Speed (6 threads, `nice -n 10`, beside the operator's engine): **3.5 s per token with a warm page cache, 5 to 10 s per token when the
touched experts and dense weights (about 5 GiB per token, 28 GiB over a 100-token prompt) do not stay in the page cache and come from the SATA
SSD.** The reference decodes each weight block to floats and sums sequentially: it is exact and slow by design, its job is to be the oracle.

## 5. Gates

Captures: llama.cpp on the CPU (no device, flash attention off), greedy, 16 positions per prompt
(`bench/oracle/qwen4exp-coder-{hi,prose,code}.jsonl`). Results and per-position lines are in `bench/flash-next-2026-11-22/`.

### 5.1 Teacher-forced top-1 (`oracle_gate`, Limbo's activations rounded the way the oracle's CPU path rounds them: `--oracle-arith`)

| prompt | prompt tokens | positions | agree | misses (oracle gap) | mean \|d1\| | max \|d1\| | mean \|dgap\| |
|---|---:|---:|---:|---|---:|---:|---:|
| hi | 2 | 16 | 12 | 4 (0.43, 0.49, 1.01, 1.25) | 0.32 | 0.82 | 0.46 |
| prose | 55 | 16 | 16 | 0 | 0.32 | 1.41 | 0.61 |
| code | 95 | 16 | 16 | 0 | 0.61 | 2.66 | 0.49 |
| **all** | | **48** | **44** | **4** | **0.42** | | **0.52** |

(`d1` = Limbo's logit at the oracle's top-1 minus the oracle's; `dgap` = the same for the top-1/top-2 margin, the quantity that decides a flip.)
With fp32 activations (the exact arithmetic) `hi` is also 12/16 (misses at gaps 0.43, 0.49, 1.01 and a 0.019 tie).
Rung 1 was 49/50 with mean \|d1\| 0.4; the logit error here is the same size, but 6 of the 48 positions have an oracle gap under 1.0
(rung 1's captures were more decisive), and **every miss sits inside the noise of the oracle itself**:

* Treating `dgap` as noise with the observed rms 0.75, the expected number of flips over these 48 gaps is 2.4; 4 were seen (p about 0.2).
  No miss has a gap above 1.7 sigma.
* llama.cpp against itself (`oracle_noise`: the same prompt decoded one token per call, no weight repack, vs the capture that decoded the prompt as
  one batch, with repack) on `hi`: mean \|d1\| 0.30, max 1.23, rms `dgap` 0.72, **and its own top-1 flips at 2 of 16 positions**, positions
  5 and 6, two of Limbo's four misses. Limbo against that decode-mode llama.cpp: rms `dgap` 0.54, the sign of the margin differs at 2 of 16.
* Cause of the noise: IQ2_S/IQ3 experts, Q2_0 down projections and BF16 hyper-connections read activations the oracle rounds to Q8_K / Q8_0 /
  BF16 per block; 48 layers x 3 such products each add about 0.5 % to the layer's update (next section) and the logits move by half a nat.

### 5.2 Layer by layer (`ref_forward --layer-gate`): no structural error anywhere

The 17-token sequence `hi + 15 greedy tokens`, captured as llama.cpp's `l_last-N` (the whole 4 x 2560 stack after each layer) and
`hc_combine-N` (after the attention half), every layer run by Limbo **from the oracle's own input** at every position, attention and MoE halves apart.
`rel_upd` is the relative error of what the layer or half adds to the stack.

| arithmetic | whole layer, mean rel_upd (worst layer) | attention half | MoE half | experts routed differently | head alone on the oracle's last hidden state |
|---|---:|---:|---:|---:|---|
| fp32 | 1.9 % (4.7 %) | 2.1 % | 1.3 % | 9 of 800 (layer, row) | max \|dlogit\| 0.28 |
| `--oracle-arith` | 0.47 % (1.3 %) | 0.15 % | 0.05 % | **0 of 800** | max \|dlogit\| 2e-5, rel L2 1e-6 |

Every one of the 48 layers is at the same level (GDN, attention, the PLE layer 1, the first and the last). A bug would be a spike in one layer
or one position; the error does not grow with the row (row 0 of a GDN layer is exact to 1e-4, the later rows are flat at the activation-rounding noise). With the
oracle's arithmetic the router picks the same 10 experts as llama.cpp in all 800 (layer, row) cases, so the routing that the next section
measures is the model's, not an artefact of this reference.

### 5.3 The block selection is reproduced

With the real `top_k` 2048 the selection is the identity below 2,052 tokens, which no oracle prompt reaches. Both sides were given `top_k` 8 instead
(`oracle_nodes --top-k 8`, `LIMBO_QSA_TOP_K=8`; blocks of 4, so 2 blocks + tail from row 11 on) on the same 17 tokens, layers 3..23 (6 attention layers):
the attention half's `rel_upd` is **0.11 to 0.35 %** (worst row 0.9 %) with the selection, **4.4 to 9.7 %** (worst row 11, the first row where the selection acts)
with plain causal attention as control. The indexer's keys, pooling, rotary at the block start, scoring, top-k and tail agree with llama.cpp. What is
not gated is any prompt beyond 2,052 tokens (a 2k+ oracle capture on the CPU reference would take hours).

### 5.4 Cost

Peak RSS of `oracle_gate` over 18 / 71 / 111 tokens: 8.9 / 14.8 / 15.9 GB, of which `RssAnon` is **135 MB**: the rest is mapped weight pages. With `--rss-budget-gb 3` the same `hi` run peaks at **5.3 GB** (bit-identical results: 12/16, same max \|d1\|); the floor is the ~5 GiB one token touches.
The cost is re-faulting those pages from the page cache each token (slower when the cache is also contended).
`make test` as far as the CPU is concerned: chat, gguf, kvpool, quants, types, tok and model (with both shards) pass; `cpu_lane_test`
fails 42 checks **identically in the main tree's binary** (Qwen3.6 files, CPU-lane SIMD tolerance; not touched here).

## 6. Phase 2: the GPU port on the 16 GB RTX 4060 Ti

### 6.1 The bar, and the box

The operator's corrected bar for Strata on this model: **~50 tok/s decode sustained, ~120 peak, 2,400+ tok/s prefill.**
Strata's own log (`strata-coder-iq1_m.log`, 2026-11-17..19, one run, short requests) shows 34.6 to 39.3 tok/s decode with MTP drafts accepted 62 %,
144 to 519 tok/s on 80 to 3,758 new prompt tokens; the figures above are the operator's, the roofline in 6.6 is checked against both.
Measured on this box (`bench/probe-2026-11-19.md`): DDR 47 GB/s read at 4 threads; PCIe H2D 6.6 GB/s through the copy engine (**1.85 GB/s** for a
mapped pull beside a busy GPU, so no zero-copy); SATA SSD (860 EVO) 499 MB/s sequential, **92 us / 10.8k IOPS** for a random 4 KiB read at queue depth 1;
VRAM 288 GB/s (we take 75 % sustained = 216 GB/s, what the 27B dense decode achieved: 21.6 of a 28 tok/s floor).

### 6.2 What Strata does (read, `docs/DETAILS.md`, `src/core/expert_cache.cpp`, its coder run log)

* **VRAM cache of whole experts** in slots (`open_sized`: per-layer sizes): 3,608 slots = 6.88 GiB, filled at start from a **ranked profile**
  (`expert-profile-coder.bin`, 12,288 pairs, built from the operator's prompts with `--dump-routing`); the rule is "ranked by routing frequency, **no eviction**".
  An adaptive tier (`--adapt-every/-swaps/-decay`) follows the conversation: decayed per-expert counts, a bounded number of swaps per window, the incoming expert copied into the
  slot of an unused one; `STRATA_PCIE_BALANCE` / `STRATA_ADAPT_FETCH` let the PCIe share take each layer's most-routed misses.
* **RAM tier:** 18.64 GiB of the experts the GPU does not hold, **page-locked** (80 s to pin; a "lend" tier for prompts), computed by a **CPU lane of 7 pool workers + the host
  thread** (AVX2, "assumed 28 GB/s of expert bytes"); the SSD is only a fallback. Prompts borrow 1,420 cache slots (2.67 GiB) + a 96-slot ring, chunks of 8192.
* **Measured decode traffic** (same log): VRAM hit rate **70 to 84 % (77 % typical)**, 0.8 to 2.0 % of the routed experts over PCIe, the remaining ~15 to 28 % computed on the CPU lane.
* **Approximations it makes:** `STRATA_ROUTE_TAIL_SKIP` skips a missed expert that every token of a verify window routes at rank 7 or lower ("answers differ slightly");
  the GPU computes cached experts "rounded differently from the CPU". Limbo must stay exact: no skipped experts, one arithmetic.
* **Tried and dropped by Strata:** `STRATA_FS_SLOTS` (reserved slots refilled with predicted experts): "the smaller cache cost more than the slots returned in every box measured".
* **PLE:** `--ple-io direct` measured 5,011 rows/s (21 MB/s of random 4 KiB reads) on this SATA SSD, `mmap` 7x slower than on NVMe; a keep-awake thread touches the SSD.
* **Footprint** at that run: "740 MiB of VRAM free with everything loaded" on the 16,380 MiB card = **~15.3 GiB in use** (cache 6.88 + dense 3.4 + MTP draft head 0.95 + KV for 131k tokens in int8 + verify buffers + the borrowed prompt ring), 18.64 GiB pinned RAM, 8 CPU threads.

### 6.3 Routing locality, measured (`tools/routing_stats`, `bench/flash-next-2026-11-22/routing-stats.txt`)

Router choices of every layer for four sequences, taken with `ffn_moe_topk` from the same llama.cpp CPU runs the layer gate trusts (it routes like Limbo's reference in 800 of 800 cases
with the oracle arithmetic, 5.2): a **3,110-token stretch of Limbo's own C++** (`step.cpp`; the training sequence), the oracle `code` (95 + 15 tokens), `prose` (55 + 15) and `hi` (2 + 15).
47 of 48 layers (llama.cpp crops the last layer to the output rows). Prefill order is token order, so these are decode-order statistics; a decode of the model's own text may differ.

**Skew.** Per layer the 16 most used experts of 256 take 44 % of the lookups (23..68 % by layer), the 64 most used 80 % (55..94 %).

**Static profile** (the N most-used pairs of the training stream stay resident; `self` = ranked on the test stream itself, the upper bound):

| slots | GiB | long code (self) | `code` (train on long / self) | `prose` (long / self) |
|---:|---:|---:|---:|---:|
| 3,608 | 6.8 | 84.6 % | 43.9 / 82.2 % | 23.5 / 86.1 % |
| 4,500 | 8.5 | 89.3 % | 50.5 / 88.5 % | 31.1 / 92.6 % |
| 6,000 | 11.3 | 94.5 % | 60.9 / 95.6 % | 44.6 / 98.8 % |
| 7,500 | 14.2 | 97.5 % | 70.9 / 99.1 % | 59.2 / 100 % |
| 9,000 | 17.0 | 99.2 % | 80.9 / 100 % | 73.3 / 100 % |

Strata's measured 77 % (70..84) at 3,608 slots is where the self-ranked 84.6 % lands once the profile is not tuned to the text. **A profile is domain-specific**: ranked on one C++ file it gives 44 % on other code and 24 % on prose at the same size.

**Adaptive caches** on the long stream (cold start; swaps = bytes that must cross PCIe per token):

| policy | slots (GiB) | hit | second half | MiB/token in |
|---|---:|---:|---:|---:|
| decayed frequency, gamma 0.998 | 3,608 (6.8) | 85.1 % | 84.1 % | 12.6 |
| decayed frequency, gamma 0.998 | **4,800 (9.1)** | 91.0 % | 90.1 % | 13.4 |
| decayed frequency, gamma 0.990 | 4,800 (9.1) | 92.0 % | 91.1 % | 43.1 |
| decayed frequency, gamma 0.998 | 6,000 (11.4) | 94.7 % | 94.1 % | 13.1 |
| global LRU | 6,000 (11.4) | 94.6 % | 94.4 % | 48.9 |
| decayed frequency, gamma 0.998 | 9,000 (17.0) | 98.9 % | 99.0 % | 8.5 |
| global LRU | 9,000 (17.0) | 98.5 % | 98.8 % | 13.4 |

**Reuse is near in time:** 38 % of re-uses of an expert come at the very next token, 60 % within 4, 80 % within 16, 94 % within 64 (long stream; the short ones are tighter).
A window of W tokens touches per layer 10 / 16.2 / 21.2 / 25.6 / 33 / 40 / 60 distinct experts for W = 1 / 2 / 3 / 4 / 6 / 8 / 16 (verify windows: ~2x the lookups of one token for 3 tokens, not 3x).

**Predictability** (`ref_forward --route-ahead`, 110 tokens x 46 layers, 5,060 token-layers): feeding layer m's router the stack as it entered layer m (before its attention half) already recalls **84.7 %** of its 10 experts;
as it entered layer m-1 **74.4 %**, m-2 **66.3 %**, m-3 60.3 %, m-4 55.5 %, m-8 45.8 % (chance 3.9 %). A layer takes ~0.5 ms, so one layer of lead is 3 MB of PCIe.

### 6.4 Residency: what it takes to reach 98 % in VRAM, and what Limbo can have

**Pure residency cannot reach 98 % inside 16 GB.** The smallest cache that hits 98 % on the long stream is **~9,000 slots = 17 GiB** (global LRU 98.5 %, decayed frequency 98.9 %; static 99.2 %);
the working set of 3,000 tokens of code spans most of the 23.4 GiB. 17 GiB of experts plus 5 to 6 GiB of dense weights, state and KV is a 24 GB card.

What the card can spare: 16,380 MiB, minus the desktop (~0.5 GiB), minus the context (0.35 GiB), dense weights **3.23 GiB** (hyper-connection matrices requantised to int8 with a scale per 32,
Strata's `STRATA_HC_REQ8`; bf16 would be 3.83), GDN state 0.12, K/V + indexer (32k context f16: 0.87 GiB; 128k int8: 2.0 GiB), decode workspace 0.1, prefetch staging ring 64 slots 0.16
= **4.8 GiB at 32k, 5.9 GiB at 128k, leaving 10.2 / 9.2 GiB of experts: ~5,300 / ~4,800 slots.** (Prompts borrow ~1.6 GiB of that for their chunk workspace and ring, as Strata does, and restore it from RAM afterwards.)

The policy (**decayed frequency + a recency horizon + route-ahead prefetch**; all of it exact, nothing skipped):

1. **Resident set by decayed frequency** (gamma ~0.998 per token, half-life ~350 tokens; a second, short constant of ~16 tokens for the recency horizon, since 80 % of re-uses come within 16 tokens): a missed expert is admitted
   when its score beats the weakest resident's; admission traffic 13 MiB/token at gamma 0.998 (0.7 GB/s at 50 tok/s = 10 % of the link). Seeded from a **learned profile that persists across sessions** (Strata saves one too), because a static profile from one text is worth 24..44 % on another, while the adaptive one
   converges within the stream. Slots are per-layer sized (1.44 to 2.54 MiB); the slab is one allocation (no fragmentation: the 5 size classes are reused in place).
   Expected: **91 to 92 % resident at ~4,800 slots (128k), ~93 % at ~5,300 (32k)** (the 4,800 row above, and interpolated towards the 6,000 row).
2. **Route-ahead prefetch over PCIe through the staging ring:** from layer m-1's stack, predict layer m's 10 experts (74 % recall; 84.7 % at half a layer's lead); the ones not resident are copied (copy engine, 6.6 GB/s) while layer m-1 computes, landing in the ring and, if the score justifies, promoted.
   At 91 to 93 % resident a layer has ~0.8 misses on average (1.6 MiB; 0.3 ms of PCIe), so the one-layer lead holds. Covered share of the misses ~70 %: **in VRAM when needed = 91.8 + 8.2 x 0.70 = 97.5 % (128k), 93 + 7 x 0.70 = 97.9 % (32k).**
   Strata tried reserved foresight slots and lost; the ring here is 64 slots (3 % of the cache) and the predictions need no slots of their own beyond a layer's lifetime, so the trade is different, but it must be measured, not assumed.
3. **The rest (2 to 2.5 % of lookups, ~11 experts or 22 MB per token) on a CPU lane of 2 workers** (`src/cpu_lane*.cpp` is the start: AVX2 kernels for the IQ/K formats), computed with the same quantised-activation arithmetic as the GPU path.

So **~97.5 to 98 % of the routed experts are in VRAM at the layer that needs them, 91 to 93 % of them resident**; 98 % flat needs the 32k-context budget (10.2 GiB) and the prefetch to cover 70 % of its misses. It is **not** a guarantee: it rests on one 3,110-token stream and a 5,060-sample predictor measurement, and on
Strata's own hit rates being consistent with them (84.6 % self-ranked at its 3,608 slots vs its measured 77 %). Phase 2's first measurement must repeat the routing capture on decode traces of the operator's real sessions.

Against Strata, by resource:

| | Strata (log) | Limbo plan |
|---|---|---|
| VRAM in use | ~15.3 GiB (cache 6.88, dense 3.4, MTP head 0.95, KV 131k int8, prompt ring borrowed 2.67) | 15.0 GiB at 32k (cache 10.2 + 4.8), no draft head. **At Strata's own 77 % in VRAM the cache is ~2,600 slots (5 GiB; static self-ranked 80.4 % at 3,000, 70.5 % at 2,000): ~10 GiB in all, 5 GiB less** |
| experts in VRAM at use | 77 % (+1..2 % PCIe) | 97.5 to 98 % (92 % resident) |
| CPU lane share | ~20 % of lookups, 7 workers + host | 2 to 2.5 %, 2 workers |
| pinned RAM | 18.64 GiB | ~0.6 GiB (ring 0.16, lane buffers 0.1, PLE row cache 0.25); the cold experts sit in the **page cache**, evictable |
| approximation | tail-skip, GPU/CPU rounding differ | none |

### 6.5 Placement

| tensor / state | tier | note |
|---|---|---|
| dense: GDN, attention, indexer, routers, shared experts, `output.weight`, PLE `key`/`value` | **VRAM**, 3.23 GiB | hyper-connection matrices as int8 + scale per 32 (`hc_*_down/up`, 1.2 GiB bf16 -> 0.6): the biggest dense item per token |
| routed experts, hot | **VRAM** slab, ~4,800 to ~5,300 slots | per-layer slot sizes, decayed-frequency admission |
| routed experts, cold | **RAM page cache** (the mapped GGUF, 14 GiB), then SSD | read through a pinned ring by the copy engine, or computed by the CPU lane in place |
| `token_embd` (322 MiB IQ4_XS) | RAM (mapped); one 1.4 KB row per token goes up | not worth VRAM |
| **PLE table (26.8 GiB IQ4_NL)** | **SSD (mapped, never in RAM)** + a RAM row cache | a token reads 16 rows of 90 B |
| K/V + indexer + GDN state | VRAM | int8 K/V above ~64k (halves the 3 GiB at 128k) |

**PLE.** The table cannot live in RAM (26.8 GiB of the box's 31) and does not need to: 16 rows x 90 B per token, and the hash is a function of the token ids, so every row a verify window or a prompt chunk will read is known **before it runs**.
The rows repeat: on the 3,110-token code stream 59 % of the row reads are repeats (66 % for the bigram heads, 53 % for the trigram heads, 20,288 distinct rows in 49,760 reads); on a 70-token prose prompt none are (a cold prompt reads every row).
Plan: a row cache in RAM keyed by row id (90 B per row: 1 GiB would hold 11 M rows, 256 MiB is plenty for a session), and **batched asynchronous reads** (io_uring, queue depth 16 to 32) of the missing rows of a chunk at its start; decode issues the window's 16 x W rows when the draft is made, a layer-0 time (~0.4 ms) before they are needed.
Needed: a decode token 16 rows (~1.5 ms at queue depth 1 and 10.8k IOPS, 0.2 ms at queue depth 16), a 2,400 tok/s prefill up to 38k rows/s for a prose-like prompt (a SATA SSD's queue-depth-32 4 KiB rate is ~90k IOPS on paper; **not measured on this box, only queue depth 1 is**): the first Phase-2 measurement for PLE, and the reason the cache and the batching exist.

### 6.6 Decode: roofline and what 50 / 120 tok/s need

Per verify window of W tokens (`bench/flash-next-2026-11-22/decode_roofline.py`, estimates): GPU bytes = dense (3.83 or 3.23 GiB) + state/KV 0.3 GiB + the experts it computes, at 216 GB/s; PCIe and the CPU lane run beside the GPU
(2 workers = 8 GB/s of expert bytes); +10 % for per-layer imbalance, +1.5 ms fixed. `U` = distinct experts per layer per window from 6.3 (W=3: 21.2, W=4: 25.6); `A` = tokens accepted per window.

| config | W=1, A=1 | W=3, A=2.2 | W=4, A=3 | W=4, A=4 |
|---|---:|---:|---:|---:|
| Strata-like: 77 % resident, no prefetch, 7-worker lane, bf16 HC | 36 | 68 | 88 | 117 |
| Limbo: 91 % resident + prefetch 68 %, 2-worker lane, bf16 HC | 34 | 64 | 82 | 109 |
| Limbo: the same, **int8 HC** | **39** | **70** | **90** | **120** |
| Limbo: 99 % resident (17 GiB of experts), int8 HC | 39 | 70 | 89 | 119 |

What the table says: **(a) plain decode cannot reach 50 tok/s whatever the hit rate**: dense + state alone are 3.2 to 3.8 GiB per token = 19 to 22 ms at 216 GB/s; the bar needs **multi-token verify windows**.
The file has no MTP head (`nextn.*` absent), so drafts come from **prompt-lookup / n-gram speculation** (Strata's `src/ngram` and `--lookup-chain` are the model) or a small drafter; the 120 tok/s peak is W=4 with ~4 accepted, which repetitive code gives
and prose does not. **(b) The hit rate does not move tok/s here, it moves the resources**: a miss is overlapped with the GPU on the CPU lane and PCIe, so the cache buys a 2-worker lane in place of 7 and no 18.6 GiB pinned copy, and removes the stall tail when a layer is unlucky.
**(c)** the roofline has no draft, verify or synchronisation cost, so it is a ceiling. Strata's log of 2026-11-19 (264 tokens in 7.2 s, 157 of 245 drafts accepted: ~4.3 tokens per 4-draft window, ~120 ms per window) sits at a third of
the 35 ms the table gives such a window; the operator's current 50 sustained is 70 % of the W=3 / A=2.2 row, his 120 peak is the W=4 / A=4 row itself (larger windows raise it: 12 tokens with ~8 accepted would be ~190 tok/s on paper). A Limbo
at 70 % of its roofline lands at **~45 to 50 with W=3 and ~85 at its peak window, so the 50 tok/s bar is met only with int8 hyper-connections, windows of 3 or more and per-layer kernels at >= 70 % of the roofline,
and the 120 peak needs windows of 4+ with ~4 accepted at nearly 100 % of it.** Below that is Strata's level, not beyond it.
**(d)** the 2 to 2.5 % CPU lane stays under 8 GB/s only while the window is <= 4 tokens; a 12-token verify window (Strata's maximum) needs the lane or the prefetch to carry ~5x the traffic.

### 6.7 Prefill: 2,400 tok/s

Work per prompt token: 6.0 G weights touched (counted from the file: hyper-connections 0.64 G, other dense 3.0 G, 10 of 256 routed experts 2.36 G; the output head's 0.64 G only for the last token) = **12.0 GFLOP**, so 2,400 tok/s is **~29 TFLOP/s**.
The 4060 Ti's INT8 tensor-core peak is ~175 TOPS and FP16 with FP32 accumulate ~44 TFLOPS (published Ada figures scaled to 34 SMs, **not measured here**): 29 is 65 % of the second and 16 % of the first.
**The bar needs the INT8 tensor-core path** (weights dequantised to int8 in shared memory per tile, activations quantised per token as the oracle's CPU path does, which is also the arithmetic the gate matched), at >= 16 % of peak (plan for 25 %), expert-sorted token batches (a chunk of 8,192 tokens sends 320 to each expert on average), and the GDN recurrence in chunks.
Experts per chunk: every one of the 256 is used in every layer, so **a chunk reads all 23.4 GiB once**; 9 to 10 GiB are resident, ~14 GiB stream through the ring at 6.6 GB/s = **2.3 s per chunk against 3.4 s of compute at 8,192 tokens** (4,096-token chunks would be PCIe-bound at ~1,800 tok/s): double-buffered per layer, ring of two layers' misses (~0.6 GiB), chunk >= 8,192.
The cold 14 GiB must come from the **page cache** (a 0.5 GB/s SSD would make a chunk take 28 s); the RAM must hold it plus the PLE row cache. Chunk workspace (stack 4 x 2560 x 8,192 x 4 B = 0.33 GB, expert intermediate, q/k/v) ~1 GiB, borrowed from the cache.
Attention is plain below 2,052 tokens and 2,048 cells per query beyond; the indexer's pooled keys for a chunk are one more small GEMM per attention layer.

### 6.8 What Limbo must do differently, in order

1. **Verify windows and a draft source** (n-gram / prompt-lookup; no draft head to carry): without them 50 tok/s is out of reach (6.6 a).
2. **int8 hyper-connection reads** fused with the norm and the sigmoid gate (Strata does it too): -0.6 GiB per token, the biggest dense saving.
3. **Residency by decayed frequency + recency + route-ahead prefetch** (6.4), persisted profile, exact CPU lane of 2 workers: the footprint column of the table in 6.4.
4. **INT8 tensor-core grouped GEMM** for prefill with expert-sorted batches and layer-ahead streaming, chunk >= 8,192 (6.7).
5. **PLE as batched async row reads + a row cache** (6.5), measured at queue depth 16 to 32 first.
6. QSA: pooled-key scoring on the GPU (`top_k` selection over <= 512 blocks per query), K/V int8 above 64k.
7. Keep the CPU reference as the oracle for each kernel (it already runs every layer; the gate in 5.2 is the template).

Risks: the locality numbers are from one stream (the policy's targets are estimates until traced on real decodes); the prefetch predictor recall is from 110 tokens; the INT8 prefill and SSD IOPS figures are unmeasured on this card and drive; and Strata's own best number (50 to 120) was
reached with years of kernel tuning that a from-scratch engine does not have on day one.

## 7. Reproducing (CPU only, no GPU; `CUDA_VISIBLE_DEVICES=` for the llama.cpp tools, threads <= 6)

```
F=.../Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf        # the first shard stands for both
make lib tools oracle
build/bin/tools/oracle_gate -m $F --oracle bench/oracle/qwen4exp-coder-code.jsonl --out out.jsonl -t 6 --oracle-arith [--rss-budget-gb 3]
# layer gate: the capture is a jsonl line {"prompt_ids":[...]} (prompt + the greedy tokens)
build/bin/oracle/oracle_nodes -m $F --oracle cap.jsonl --out l_last.bin --node l_last --wide -t 5
build/bin/oracle/oracle_nodes -m $F --oracle cap.jsonl --out hcc.bin --node hc_combine --wide -t 5
build/bin/tools/ref_forward -m $F --layer-gate l_last.bin --attn-residual hcc.bin --oracle-arith -t 6 --ctx 64
LIMBO_QSA_TOP_K=8 ...                   # with  oracle_nodes ... --top-k 8   to exercise the block selection on a short prompt
build/bin/tools/ref_forward -m $F --layer-gate l_last.bin --route-ahead         # router recall from an earlier stack
# routing locality: --node none keeps only the routers' choices
build/bin/oracle/oracle_nodes -m $F --oracle long.jsonl --out route.bin --node none -t 5
build/bin/tools/routing_stats -m $F route.bin [test.bin ...]
build/bin/tools/role_bytes $F      # bytes and element counts by role; one expert's size per layer
```

`bench/flash-next-2026-11-22/`: the gate runs (`arith-*.jsonl`, `fp32-hi.jsonl`), both layer gates, `routing-stats.txt`, `route-ahead-code.txt`, `decode_roofline.py`.
Not run here by design: `make test`'s CUDA suites (the card is the operator's); the CPU suites pass, `cpu_lane_test`'s 42 failures are in the main tree too.
