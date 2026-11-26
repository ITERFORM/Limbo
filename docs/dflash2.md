# DFlash2 block drafter

Written from reading the operator's llama.cpp fork (MIT; `src/models/dflash.cpp`, `common/speculative.cpp`, `conversion/qwen.py`) as a description
of the math, and from the checkpoint's own `config.json` and tensor table. No code is taken from it.
Checkpoint: `incoai/Qwen3.6-35B-A3B-DFlash2` (`config.json` + `model.safetensors`, 0.98 GiB BF16). Engine: `src/cuda/dflash.cu`,
`include/limbo/cuda/dflash.hpp`, hooks in `src/cuda/gpu_dflash.cu`, source in `GpuBackend::decode_spec`.

## What it is

A 6-layer Qwen3-style decoder (hidden 2048, 32 query heads over 8 KV heads, head dim 128, FFN 6144, RoPE theta 1e7, full 128-dim NEOX,
sliding window 2048 on every layer, attention NOT causal). It has no embedding table and no head: it uses the target's token embedding
for its inputs and the target's LM head for its outputs. One pass fills a whole block of 8 positions, `[anchor, MASK x 7]`, and yields 7
draft tokens for the positions after the anchor. There is one denoising step, no iteration.

## The context: target hidden states as K/V

The drafter does not run over the prompt. For every position the target has read it keeps a K/V entry per layer, made from the target's
hidden states:

1. Take the residual stream at 8 points of the target (the config's `target_layer_ids` 1, 6, 11, 16, 22, 27, 32, 37 are layer OUTPUTS,
   `hidden_states[id + 1]` of the reference; in the engine that is the INPUT of layer id + 1, i.e. 2, 7, 12, ..., 38 of the target's 40) and
   concatenate them: 8 x 2048 = 16384 values per position.
2. `g = RMSNorm_hidden_norm(fc @ concat)` with `fc` [2048, 16384]. One vector per position, shared by all drafter layers.
3. In layer l: `K = k_proj_l(g)`, then RMSNorm over each head with `k_norm_l`, then RoPE at the position; `V = v_proj_l(g)`.
   Both are stored (the engine keeps F16, 24 KiB a position over the 6 layers).

A position enters the cache when the target has processed it (target pass rows that are kept: all of a plain step or prefill row, the
accepted rows of a verify window). The anchor is the token the target has NOT yet read, so the context always ends right before it.
Only the last 2048 positions are ever attended, so a prompt needs only its last 2048 positions ingested.

## The block pass

Rows t = 0..7 at positions pos + t (pos = the anchor's position), inputs `embed(anchor)`, `embed(MASK) x 7` (MASK = 248077 in the
checkpoint's config). Per layer:

```
a   = RMSNorm_input(x)
dyn = attn_conv_proj(a)                      # 512 = [side 2][tap 2][group 128] per row
a   = conv(a, dyn, attn_conv_base, side 0)
q,k,v = q_proj/k_proj/v_proj(a);  q,k RMS-normed per head, RoPE at pos + t
o   = attention(q, [context K/V ; block k/v])   # block rows see all 8 block keys; context key p is seen by row t when p > pos + t - 2048
o   = o_proj(o);  o = conv(o, dyn, attn_conv_base, side 1)
x   = x + o
h   = RMSNorm_post(x)
dyn2 = mlp_conv_proj(h);  h = conv(h, dyn2, mlp_conv_base, side 0)
h   = down(silu(gate(h)) * up(h));  h = conv(h, dyn2, mlp_conv_base, side 1)
x   = x + h
```

`conv` is a short causal convolution along the block (kernel 2, so it reads the row itself and the row before; zero before row 0), applied
per channel with a coefficient that has a dynamic and a static part: channels are in groups of 16 (128 groups of the 2048), and
`w[row][tap][ch] = dyn[row][side][tap][ch / 16] + base[side][tap][ch]`; `out[row][ch] = w0 * in[row][ch] + w1 * in[row-1][ch]`.
`dyn` comes from the same normed input in both of a half-layer's conv calls (side 0 before the attention/FFN, side 1 after).

After the 6 layers: `hn = RMSNorm_norm(x)`; logits of rows 1..7 = target LM head over `hn`.

## The selector (turns 7 rows of logits into a chain)

Every row keeps its top-16 candidates by logit (`unary` = their logits). Two BF16 codebooks [vocab, 256] (`predecessor_codebook`,
`successor_codebook`) and a projection `hidden_projection` [256, 2048] give a pairwise score between a candidate at row i and a
candidate (the predecessor) at row i - 1:

```
gate_i        = hidden_projection @ hn_i                      (256)
score[i][k][j] = unary_i[j] + sum_d succ[cand_i[j]][d] * pred[prev_k][d] * gate_i[d]
```

where `prev_k` is the anchor token for row 1 (a single predecessor) and candidate k of row i - 1 afterwards. The walk is greedy:
row 1 takes `argmax_j score[1][0][j]`; row i takes `argmax_j score[i][k*][j]` with k* the candidate row i - 1 chose. The softmax of the
chosen row of scores gives a confidence per draft (the fork stops the block when it is below `p_min`; the engine only reports it).

## In the engine

* `Dflash` (src/cuda/dflash.cu) owns the weights (BF16 in one arena, norms/bases as F32; the codebooks stay in mapped host memory, a draft
  reads 32 rows of each), a per-slot ring of 2064 positions of K/V (48 MiB), the scratch, and the kernels. Matrix products reuse the
  engine's `gemv_cols` (BF16, C = 8 rows); the new kernels are the capture, the conv, the q/k norm + RoPE, a split attention over the ring,
  the two-pass top-16, the selector scores and the walk. `tests/cuda_dflash_test.cu` holds them to a plain CPU implementation on a
  synthetic checkpoint (2,300 positions, ring wrap, cut, share, gap).
* Capture: the target's passes copy the residual rows at the 8 layers into a BF16 stage (8 rows; a prefill chunk has its own). `Gpu::commit(n)`
  and every pass that keeps its rows then call `ingest`, which does `fc`, the norm, K/V of 6 layers into the slot's ring.
* The block is 8 rows, the verify window kMaxRows = 8: anchor + 7 drafts fit exactly, so `dec::kMaxRows` is unchanged.
* decode_spec picks the source per cycle by expected tokens per ms, with acceptance EMAs per draft position of its own.
