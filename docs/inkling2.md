# Inkling-2-Mini: `inkling2_loop`, the architecture, from the reference implementations

The model, the loop, and every mechanism, written down before a line of the step is written. Three
sources, in this order of authority, and nothing here is inferred from a tensor name alone:

- **`transformers/models/inkling/modeling_inkling.py`** (installed, 5.14.1) — the model's own definition.
  It covers the attention, the relative bias, the short convolutions, the MoE and the router. It does
  **not** cover the loop, the block-mixing residuals, the attention sinks, the L2 clips or SiTU: those
  tensors are in the file and the module has no code for them. Its port is partial.
- **`src/models/inkling2.cpp` and `src/models/inkling2-loop.cpp`** in llama.cpp master (the tree this
  box built as the oracle, `~/projects/llama-inkling`) — complete, and validated by the fact that it
  produces coherent output here (84·3/2 → 126, 46.19 tok/s decode on the 4060 Ti).
- **The file itself** — `~/models/inkling/Inkling-2-Mini-UD-IQ2_XXS.gguf`, 744 tensors, read with
  `role_bytes` and a header parser. Shapes and constants below are the file's, not the config's.

Both reference implementations are read as **facts about the architecture** (CLEANROOM.md §3). Limbo's
step is Limbo's own code.

## 1. The file

```
arch inkling2_loop        24 blocks, n_embd 2304, vocab 201024, ctx 1048576
attention                 24 heads / 8 kv, head_dim 128, NO RoPE (rope.dimension_count 0)
sliding_window_pattern    [T,T,T,T,T,F] x4 -- layers 0-4,6-10,12-16,18-22 window 1024; 5,11,17,23 global
experts                   128 routed, 8 per token, NO shared expert, ff 1152
loop                      start 6, end 18, count 3, gate_rank 64, varying false
constants                 rms_eps 1e-6, qk_clip 22.627416610717773, relative_extent 1024,
                          relative_length 16, relative_clip 8.0, log_scale_floor 8192,
                          log_scale_alpha 0.1, attn_res_block_size 6, sconv_kernel_size 4,
                          route_scale (expert_weights_scale) 8.0, situ_beta 4.0,
                          situ_linear_beta 25.0, final_logit_softcapping 30.0
```

## 2. The layer graph

`build_layer(il, ip, cur, blocks, messages, loop)` — `il` is the physical layer, `ip` the visit index,
and the two differ inside the loop. In order:

1. **Depth mix before attention** (only when the layer has `attn_res_query`, i.e. every layer but 0):
   `mixed = mix(blocks, cur, attn_res_query)`. `mix` is a small attention over the *depth*: the bank of
   earlier block outputs plus `cur`, RMS-normalized (eps 1e-6), scored against a learned query with
   `1/sqrt(n_embd)`, softmax, weighted sum. `attn_res_query` is **[n_embd, 1] on sliding layers and
   [n_embd, 3] on the global ones** — one or three queries. The result feeds the attention's input
   (the Q input on sliding layers; the Q, K and V inputs separately on global ones).
2. **Attention** (§3).
3. `cur = convolve(wo @ attn_out, attn_post_conv, nullptr, ...)` — a short conv on the attention output,
   with the residual added inside `convolve`.
4. **Residual stream structure**: `if (ip % attn_res_block_size == 0) blocks.push_back(inpSA) else
   cur = inpSA + cur`. Every 6th visit *starts a new stream* by banking the pre-attention state; the
   others add to it. This is what `blocks` collects and what `mix` reads.
5. **Depth mix before the FFN**: `cur = mix(blocks, cur, ffn_res_query)`.
6. **The loop control** (only inside the loop): `cur = control(cur, message(ctrl), loop, ctrl)`.
7. `cur = rms_norm(cur, ffn_norm)`.
8. **MoE** (§4).
9. `cur = convolve(cur, ffn_post_conv, nullptr, ...)`, then `cur = ffn_inp + cur`.

## 3. Attention

```
q = wq @ x                     [n_head*hd]        k = k_conv(wk @ x)     [n_head_kv*hd]
r = wr @ x                     [n_head*d_rel]     v = v_conv(wv @ x)     [n_head_kv*hd]
q = clip(q, attn_q_clip, qk_clip)      k = clip(k, nullptr, qk_clip)
r = clip(r, attn_rel_clip, relative_clip)
v = rms_norm_per_head(v, eps)          -- V ONLY, and with no gamma: there is no tensor for one
```

**No q/k norm at all.** `inkling2.cpp:335-345` clips the raw q and k and normalises only V. The Python's
`q_norm`/`k_norm` are `InklingRMSNorm(head_dim)` with a weight that the file carries no tensor for, and
the reference that actually runs does not apply them. The clips are what bound q and k instead, and
`qk_clip` is **16·sqrt(2)** exactly (22.627416610717773), which is the kind of number a norm bound is.

`clip(cur, scale, threshold)`: `cur *= scale` when a scale is given, then
`cur *= clamp(threshold / (sqrt(sum(cur²)) + 1e-6), 0, 1)` — an L2 clip, per head.

**Position** is the learned relative bias, not RoPE:

```
rel_logits = (r @ attn_rel_bias)                  r [q, n_head, d_rel=16], proj [16, extent]
bias[q, h, kv] = rel_logits[q, h, clamp(q_pos - kv_pos, 0, extent-1)]
bias = 0 where (q_pos - kv_pos) < 0 or >= extent
```

`extent` is `sliding_window_size` (1024) on sliding layers and `relative_extent` (1024) on global ones.

**Log scaling** (global layers only, `log_scaling_n_floor` present):

```
tau = 1 + alpha * log(max(1, (q_pos + 1) / floor))     alpha 0.1, floor 8192
q *= tau      and      bias *= tau                      -- both, not k
```

**Sinks**: `sinks = attn_sink_proj @ attn_input` → one logit per head per token. The softmax is
`exp(s - t) / (1 + sum(exp(s - t)))` — a sink column with logit `t`, implemented by subtracting `t`
from the logits and adding 1 to the denominator. llama.cpp's fallback path passes a zero sink.

**Scaling — OPEN, and it matters**: `modeling_inkling.py` sets `scaling = 1.0 / head_dim` with the
**Scaling — RESOLVED to `1/sqrt(head_dim)`**, and the resolution is the paragraph above rather than a
measurement. `modeling_inkling.py` writes `scaling = 1.0 / head_dim` with the comment *"q/k are
RMS-normalized per head, hence 1/d rather than 1/sqrt(d)"* — and its premise is false for the model that
runs: there is no q/k norm, so the `1/d` it justifies does not follow. `inkling2.cpp` passes
`1.0f / sqrtf(n_embd_head)`, which is what the arithmetic wants for a *clipped* q and k. Implement
llama.cpp's, and let the gate confirm it on the first token.

**Short convolutions** — four per layer, each depthwise (groups = channels), kernel 4, no bias, with a
residual and a scale:

```
out = cur + sum_{j=0..3} weight[j] * x[t - 3 + j]        (causal, state holds the last 3 inputs)
out *= 1 / sqrt((dynamic ? 2 : 1) * 4 + 1)
```

`k_conv` and `v_conv` run on the projected k and v; `attn_post_conv` on the attention output;
`ffn_post_conv` on the MoE output. On the **global** layers k and v are *dynamic*: `attn_k_conv_proj`
and `attn_v_conv_proj` add input-dependent taps per head (hence the `2*` in the scale).

The state is **one shared history per visit with offsets** (k at 0, v at `n_embd_k_gqa`, the attention
output at `2*n_embd_k_gqa`, the FFN output after that) rather than four buffers — functionally the same,
but it is what the reference's `build_rs` lays out, and it is one buffer to reset. A dynamic tap belongs
to a *head* and is added to every channel of that head.

**The head's output is truncated to the unpadded vocabulary**: `output.weight` has 201024 rows and the
graph keeps only `vocab_size_unpadded` = **200058** (`inkling2.cpp:442`). A gate that compares 201024
logits against a 200058-logit reference will read the tail as a mismatch.

The attention output is gated: `attn_out = attn_gate(x) * (attention result)`.

## 4. The MoE

```
router_logits = ffn_gate_inp @ x                      [128]
scores = sigmoid(router_logits)
choice = scores + exp_probs_b                         -- e_score_correction_bias, F32[128]
top_k  = topk(choice, 8)                              -- selection only
logits = router_logits[top_k]
w      = softmax(logsigmoid(logits)) * route_scale    -- softmax over the 8 of log-sigmoid, x8.0
per expert e in top_k:
    gate, up = (ffn_gate_up_exps[e] @ x).chunk(2)     -- FUSED [2304, 2304, 128]
    h = situ(gate) * up
    y += (ffn_down_exps[e] @ h) * w[e]
```

There is no shared expert (`n_shared_experts = 0`), so the `shared_gammas` machinery in the reference is
dead code for this model. `ffn_down_exps.scale` and `ffn_gate_up_exps.scale` are **per-expert
dequantisation scales**, not a maths step: they belong to the NVFP4 recipe and must be applied to the
dequantised weights. llama.cpp shipped without them on `ffn_gate_up_exps` and the model loaded but used
wrong expert scaling (see §7).

**SiTU** is the activation (`situ_beta` 4.0, `situ_linear_beta` 25.0). It is not in ggml as an op and not
in transformers' `ACT2FN` (which offers only `silu`); llama.cpp builds it from primitives
(`llama-graph.cpp:2603`) and this is its exact form, with `beta = situ_beta = 4.0` and
`lb = situ_linear_beta = 25.0`:

```
situ(gate, up) = (beta * tanh(gate/beta) * sigmoid(gate)) * (lb * tanh(up/lb))
```

That is the softcap the config calls `mlp_gate_softcap` 4.0 and `mlp_up_softcap` 25.0: a tanh saturation
on the gate's magnitude, a sigmoid gate, and a tanh saturation on the linear branch. Note it is NOT
SiLU with a cap — the gate term keeps a sigmoid, so the activation is bounded on both sides.

## 5. The loop

```cpp
for (il = 0; il < 6; ++il)  cur = build_layer(il, il, cur, blocks);     // prelude: 0..5
prelude_blocks = blocks;
for (pass = 0; pass < 3; ++pass) {
    if (pass > 0) {
        bank = blocks[prelude_blocks.size()+1 .. end];                  // the previous pass's blocks
        bank.push_back(bank_out);                                       // and its output
        blocks = prelude_blocks;                                        // reset the bank
    }
    for (ip = 6; ip < 18; ++ip) {
        il = pass == 0 ? ip : 24 + (pass-1)*12 + (ip - 6);              // the cache slot
        cur = pass == 0 ? build_layer(il, ip, cur, blocks)
                        : build_layer(il, ip, cur, blocks, mix(bank, bank_out, msg_query), &loop);
    }
    bank_out = cur;
}
for (il = 18; il < 24; ++il) cur = build_layer(il, il, cur, blocks);    // coda: 18..23
build_output(cur, blocks);                                              // output_res_query mixes the bank
```

- **The weights are shared, the state is not.** `layer_phys(il) = il < 24 ? il : 6 + (il - 24) % 12`, so
  the 36 visits load the 12 physical layers 6..17 three times; only the KV/conv cache slots differ
  (24 + 2*12 = 48 slots). Anything sized by *layer visits* must not allocate weights per visit.
- **The control gate** mixes a message from the previous pass into the current state:

```
gate   = ctrl_down @ concat(rms_norm(cur), rms_norm(message))     down [4608, 64]
gain   = tanh(ctrl_gain @ gate)                                   gain [64, 2304]
forget = tanh(ctrl_forget @ gate)                                 forget [64, 2304]
out    = cur - forget*cur + gain*message
```

Four controllers per looped layer: `q`, `k`, `v`, `ffn` (the messages for q/k/v come from the same
`mix(bank, bank_out, msg_query)`, selected per control index; `loop_msg_query` is the query).

## 6. What this costs, measured by other people

From the model's own discussion thread, because these are the traps that cost a day each:

- **Per-expert scales**: mainline's loader asked for 714 of the file's 734 tensors and missed the 20
  `ffn_gate_up_exps.scale`; the model *loads* and is *wrong*. Expected-tensor counts are the only thing
  that catches it.
- **The relative bias is materialised densely**: one `[q_len, kv_len]` table per head. At `-c 262144`
with FA and q8_0 KV that was ~26 GiB of compute buffer on a 3090, and the fix is a *banded* path
(llama.cpp PR #385). Limbo's own KV budget must account for it: the bias table is O(q x kv) per head
per layer, which is why this model's long context is not the same problem as a RoPE model's.

### The fix, and it is a kernel change: fuse the bias into the attention

`ljedrz/llama.cpp:inkling2-loop-scale-reshare` (fetched here as `~/projects/llama-inkling-fast`, tip
`c12f9b9`) solves it by never building the table. `ggml_flash_attn_ext_add_rel_bias` hangs two extra
sources on the flash-attention node and the kernel does the gather per key:

```
table    [n_dist, n_head, n_batch, 1] F32   -- one row of n_dist biases per (head, query)
rel_pos  [n_kv,   n_batch, 1,       1] I32  -- the clamped distance for each (key, query)
bias[kv, batch, h] = table[rel_pos[kv, batch], h, batch]
```

For one token that table is `n_dist x n_head x 1` = 1024 x 24 floats = 96 KiB per layer, against
`n_head x q_len x kv_len` for the dense form — the whole of the 26 GiB. The table is the *projected*
bias (`attn_rel_bias` applied to that token's `r`, which is `[n_rel, n_head*n_tokens]`), so the cost is
one `n_rel x n_dist` matmul per token per layer, not per key.

**Two details that are part of the design, not incidental:**

- **The zero-outside-extent is a padded row, not a mask.** `rel_bias` is padded to `n_dist + 1` rows
  with the last row all zero, and `rel_pos` clamps the distance into `[0, n_dist]`. A distance at or
  beyond the extent lands on the zero row, so the "bias is zero outside `0 <= d < extent`" rule needs no
  mask and no branch.
- **The sink rides the same row.** `sinks = attn_sink_proj @ attn_input` is one logit per (head, query),
  and it is subtracted from the whole table row once. Subtracting a per-(head, query) constant from
  every logit of that query is exactly `exp(s - t) / (1 + sum(exp(s - t)))`, so the sink costs nothing
  in the kernel — it is a subtraction on 1024 values, not a column.

**What this means for Limbo**: the rung-4 attention kernel takes `q, k, v`, the per-(head, query) bias
row, and a per-key distance index, and adds the gathered bias inside the softmax loop. The [q x kv]
matrix is never materialised anywhere, at any context length. That is a *different kernel signature*
from every attention in the tree today (qwen35moe/qwen4exp have no bias at all), so it is not a
variation of `dec::attn` — it is its own kernel, and it is the first thing rung 4's CUDA port needs.
- **Speed on consumer cards**, at short context, all layers resident: 11-17 tok/s (9070 XT), ~15 tok/s
  at 135k ctx (4090), ~25 tok/s (2x3090), 5 tok/s (2070 Super, experts on the CPU). This box: **46.19
  tok/s** tg128 at 512 ctx, IQ2_XXS, all layers on the 4060 Ti.
- **DSpark** (the drafter in `mtp.safetensors`, converted to `dspark-Inkling-2-Mini-Q8_0.gguf`, 0.40 GB)
  gives 0.93 draft acceptance and mean length 3.79 on structured text; it is mainline PR #32218, **not
  merged** — the oracle on this box cannot run it, only Unsloth's prebuilt can.

## 7. What Limbo needs, in order

1. **The step, CPU, teacher-forced.** One token at a time, all 36 visits, gated against the oracle
   (llama.cpp's `--logits` / teacher-forced comparison, the same shape as `tools/oracle_gate`). This is
   where the scaling question in §3 gets settled and where SiTU's exact form is fixed.
2. **Then the CUDA port**, mechanism by mechanism, in the order that the oracle can gate: relative bias
   and sinks (attention), the convs, the loop control, SiTU and the MoE, the depth mixing.
3. **Then the budget.** The bias table is the memory problem, not the KV cache. The loop means the
   expert weights are read 3x per token from wherever they live, which makes the rung-2 donor work
   (kindi's slot table and in-graph pulls) directly relevant: the experts' rows, read three times
   per token on the looped core (measured: 6.75 GiB resident, 878 MiB of rows per token —
   `bench/inkling2-budget-2026-11-23.md`; the 2.76 GiB this line first carried was `role_bytes`'
   fused-gate undercount).

The oracle is `~/projects/llama-inkling/build/bin/llama-cli` on `~/models/inkling/Inkling-2-Mini-UD-IQ2_XXS.gguf`,
and `llama-tokenize` from the same build is the tokenizer gate (18 of 20 corpus files exact).

**Status 2026-11-23: item 1 done.** `Ref::forward_inkling` runs the whole 36-visit graph CPU-side,
teacher-forced, and the gate passes at the W1 bar: 61 of 64 positions agree (1 at an oracle gap
< 0.1, 2 inside llama.cpp's own batch-vs-decode noise, max |Δlogit| 0.77); every one of the 48
visit slots' `l_out` matches `oracle_nodes` to a mean of 1.7% (max 5.8%). The gate caught two
step bugs — conv weights read byte-wise, and the sink/bias fold that belongs after the q.k scale
(the §3 scaling note was right; my first implementation folded pre-scale and the oracle's numbers
overruled it) — both fixed; `bench/inkling2-cpu-gate-2026-11-23.md` has the numbers, the traps
and the trace-node set the CUDA port gates on. Next: item 2, the CUDA port, mechanism by
mechanism in the order the oracle can gate (relative bias + sinks first, per §6).

**Status 2026-11-23: item 2 done.** The whole graph is on the device (`src/cuda/ink_gpu.cu`): one
token in, the 36 visits as a fixed sequence of kernel launches on one stream with no host round trip
inside the pass, the logits or the argmax out. The mechanisms went in the order the gate could hold
them — the relative-bias attention and the sinks, the short convolutions, the depth mix and the loop
controller, a GEMV decode for every weight type the file uses (IQ1_S / IQ2_XXS / IQ2_XS / Q2_0 /
NVFP4, per-expert scales), and the routed experts (`src/cuda/ink_moe.cu`: the router's top-n under
the tie rule, the fused gate/up rows under the per-expert scales through SiTU, the weighted down
accumulation in selection order). The gate is the CPU Ref (which carries the oracle's), teacher-forced
on the oracle's prompt, 32 positions (`tests/cuda_ink_step_test.cu`): every one of the 48 visit slots'
`l_out` within 0.0006 at position 0 (bar 0.05), the greedy token agrees at 32 of 32, and reset()
plus a second run is bit-identical at every position. The logits track the Ref to 1e-3 through
position 15, then diverge discontinuously (first crossing at 16, max |Δlogit| 1.35 over the run):
the two runs' MoE inputs differ by accumulation-order noise, and once it crosses an 8th/9th expert
margin the runs pick different experts — both valid under the model's tie rule — so the gate is the
split in the test (8 clean positions before the first crossing, 3.0 overall) plus the greedy token,
not a flat logit bar. The MoE's own gate (`tests/cuda_ink_moe_test.cu`) pins the routing and the
experts to a double-precision fold at 2e-6 in every weight type, real shapes and small, with the tie
rule and bit-identical reruns. Next: item 3, the budget.

**Status 2026-11-23: item 3 done.** The budget is measured, not estimated
(`bench/inkling2-budget-2026-11-23.md`, every number from `tools/ink_budget.cu`, checks green). The
account caught one wrong number of its own: the routed experts are **6.75 GiB**, not 2.76 —
`role_bytes`' expert role list missed the fused `ffn_gate_up_exps` (4,090.5 MiB; fixed). The bias
table this item worried about is **96 KiB of scratch** under the §6 fused gather, at any context
length; what scales is the K/V state, 192 KiB per position over the 48 slots (measured 191.9 KiB of
VRAM per position across four `InkGpu::open` points). Against this box's free VRAM that is **~37.7k
resident positions as InkGpu allocates today and ~221k with the sliding-window ring** — 40 of the
48 slots are sliding and only ever hold 1,024 positions; past the ring only the 8 global slots are
O(ctx), and they are where `paged-kv.md`'s host pages take over. Decode reads **2,780 MiB per
token** at 512 ctx (dense 1,568 + experts 878 + head 235 + K/V 96): the bound is 98.8 tok/s at
288 GB/s and InkGpu measures **43.0 tok/s** (64 head steps, one stream) — 44% of the bound, 93% of
llama.cpp's 46.19 tok/s incumbent at the same model and context. At 262k the global slots' K/V
reads (8,352 MiB/token) bound decode at 24.9 tok/s. Host-resident experts (the rung-2 case) read
878 MiB of rows per token — 585 MiB of it the loop's extra passes, 384 expert pulls per token, 288
on the looped core — which is 7.17 tok/s over the measured 6.6 GB/s PCIe: PLAN.md §3.1's crossover
puts a decode row on the CPU from pinned RAM, so kindi's slot table and in-graph pulls are
load-bearing exactly where the loop multiplies the pull count. The §7 list is complete.

**Status 2026-11-24: decode tuned.** The budget's profile showed the wall was the quantized decodes
and the per-product tail launches, not launch overhead (`bench/inkling2-decode-tune-2026-11-24.md`):
`decode8<NVFP4>` now decodes the E2M1 magnitude from its bits over two word loads instead of eight
byte loads plus codebook lookups (values bit-identical, that kernel 108 → 357 GB/s effective), the
row's tail is fused into the product with the exact old float association (no second launches; the
tail kernels and their dispatchers deleted), the IQ2/IQ3/IQ1 words load as words, and the loop
controller's gain/forget share one dual product. **InkGpu decode: 43.0 → 50.0 tok/s at n_ctx 512
(+16%), 51% of the 288 GB/s bound** — above llama.cpp's 46.19 tok/s incumbent at the same model and
context. All 17 `cuda_*_test` green (the exactness contracts among them: `cuda_gemv_test`'s row
sums, `cuda_gemm_test`'s bit-for-bit equality with `gemv()`, the ink gates' bit-identical reruns).

**Status 2026-11-24: serving.** `inkling2_loop` is served by both entries. `limbo_serve` drives it
through `Ref`; `limbo_serve_gpu` dispatches on the file's arch to `InkBackend`
(`src/cuda/ink_backend.cu`), `InkGpu` behind `serve::Backend`: prompt tokens without the head, the
device argmax at temperature 0, one pinned logits buffer otherwise. The response parser learned the
model's message family — `<|message_model|><|content_thinking|>` / `<|content_text|>` /
`<|content_invoke_tool_json|>` with the function-name label dropped and the `{"name","args"}` call
form — so `reasoning_content`, `content` and OpenAI `tool_calls` split correctly (gates in
`tests/chat_test.cpp`, including streaming convergence). Measured on the device entry: the split,
tool calls, SSE streaming, and tape reuse (`cached_tokens: 29` on an extending second turn).
One conversation per process: no extra slots or recurrent checkpoints (`InkGpu` has neither, so a
diverging prompt re-reads exactly).

**Status 2026-11-24: the sliding ring, and `--ctx max`.** The 40 sliding slots are rings of
window + 7 = 1,031 rows (position p at row p % 1031); only the 8 global slots hold n_ctx rows.
`ink::attention` takes the ring and launches only the window's ≤5 absolute chunks for a sliding
layer, so its partials and merge order are the full grid's: `cuda_ink_attn_test` gates the ring
bit-identical to the full cache past the wrap (pos0 up to 4,088, T 1 and 8) and refuses a ring shorter
than window + T − 1. `limbo_serve_gpu --ctx max` fits the context after the weights are resident
(free − `--vram-reserve-mib` − rings − 256 MiB scratch margin, over 32 KiB of global K/V per
position). Measured on this box (RTX 4060 Ti, IQ2_XXS, 546 MiB in use by the desktop before start):
**ctx 190,976** (was ~37.7k), 15.0 of 16.0 GB in use, backend open in 2.9 s. Over HTTP: decode
48.2 tok/s at 29 positions; a 2,247-token needle prompt retrieves its first-line fact in the
reasoning (only the global slots can reach it) at 42.1 tok/s prefill / 39.5 tok/s decode.
**Prefill is the open item:** InkGpu runs the prompt one token at a time, ~45 tok/s, so filling
the 190k context costs over an hour; llama.cpp's batched pp512 is 894 tok/s on the same card.

**Status 2026-11-24: step_rows and the decode graph.** `InkGpu::step_rows` reads the prompt in pieces
of up to 8 rows: the dense projections go through `gemv_cols` once per piece (gemv's bits for any
column count), everything stateful row by row, so the pass is the bits of 8 `step()` calls (gated
in `tests/cuda_ink_step_test.cu`, across the ring's wrap). The decode pass is a CUDA graph per head
flag (the position and the log-position factor read from device memory). Measured: decode 49.8 →
53.2 tok/s (`ink_prof`, n_ctx 512; idle between kernels 2,336 → 255 µs/token); over HTTP on the
2,247-token needle prefill 42.1 → 62.3 tok/s and decode at that depth 39.5 → 42.6 tok/s, the output
the same bytes. What is left is the arithmetic itself: the experts (878 MiB/token) still run per
row, and the expert GEMVs read at 157–175 GB/s. The next large step is a grouped expert GEMM for the
prompt (every expert read once per piece instead of once per token), which needs an NVFP4 GEMM.

**Status 2026-11-24: the prompt pass.** `InkGpu::prefill_rows` reads the prompt in pieces of 512 tokens on
the int8 tensor cores: the dense products through `gemm` over the activation planes (Q4_K/Q5_K with their
activation sums; the small BF16 matrices as F32 through `gemm_f32`), the router and the controller through
`gemm_f32` and a tiled short-K kernel, the experts grouped (`moe_build`, `moe_rows` over the fused gate/up with
SiTU after, `moe_down`; NVFP4 joined the int8 GEMM's formats, gated in `cuda_gemm_test` like the others),
the short convolutions as one parallel FIR over the piece, and the attention through `attention_rows`:
8 consecutive rows x 3 heads share every K/V tile. Not `step()`'s bits: the planes round the activations and a
router near-tie can then pick another expert for a token. Gated in `cuda_ink_step_test` against `step()` over
1,100 positions in uneven pieces: the greedy token agrees at every piece end (or is within 0.5 of a near-tie),
median |dlogit| ≤ 2 (measured 0.95 at 2 planes; the floor at 3 planes is 0.81, the routing flips),
max ≤ 8. Measured (`ink_prof --prefill`, RTX 4060 Ti): **881 tok/s** over 2,048 tokens (2 planes; 967 with
`LIMBO_INK_PLANES=1`, llama.cpp's own Q8 activations), 780 tok/s over 8,192; over HTTP 700.6 tok/s on a
2,247-token prompt and 607.9 tok/s on 12,047 (was 42-62). llama.cpp's pp512 on this card: 894.

**Status 2026-11-24: DSpark speculative decoding.** `src/cuda/ink_draft.cu` runs the model's own 3-layer block
drafter (`dspark-Inkling-2-Mini-Q8_0.gguf`, arch `dflash`, from the checkpoint's `mtp.safetensors`): per target
position a context K/V from a per-layer mix of the target's features (the inputs of visits 5 / 23 / 41 and the
final norm, captured by every pass with `InkGpu::set_capture`), a block pass over [anchor, mask x 7] through 3
layers (NeoX RoPE, the sliding layers' window 1,025, the middle layer over the last 32k positions), the base
logits through the target's head, and the Markov head's GRU chain (`joint_proj` over [state, w1[prev], hidden,
f_rnn], softcap after the bias). The check is `InkGpu::verify`: the exact batched pass (every row's greedy token
is `step()`'s), then `commit(n)` rolls the short-conv histories back to the last kept row. Drafts are cut at the
first one whose softmax probability under the drafter is below 0.7 (`LIMBO_INK_DRAFT_CONF`), k = 3 by default.
Gates: `cuda_ink_step_test` (verify + commit(2) of a 4-row window, then decoding bit-identical to a run that never
verified); over HTTP the speculative output is byte-identical to the plain greedy output (code and prose, 300-400
tokens). Measured (ctx 32k, 400 tokens, greedy): plain 44.3 / 44.6 tok/s, DSpark k 3 conf 0.7 **62.8 tok/s on
code (+42%), 49.3 on prose (+11%)**; 85% / 63% of the checked drafts kept, draft 5.0 ms and a 2-4 row verify
33-39 ms a cycle. The ceiling is the MoE: a verify row reads its own 8 experts (no overlap to share), ~7 ms a row.
