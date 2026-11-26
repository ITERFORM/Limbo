# Rung 1's graph, measured from the oracle (2026-11-19)

The oracle's own graph for `Qwen3.6-35B-A3B-UD-IQ3_XXS`, node by node, with shapes — taken from
`llama-eval-callback`'s per-node dump (`bench/raw/rung1-graph.txt`, one prompt token), which prints each
node's op, its input tensors with shapes, and the node's values. This is a *measurement of behaviour*,
not source: the reference implementation is used as an instrument, which is what PLAN.md says the oracle
is for. Everything below is what Limbo must reproduce, and every claim here is checkable against the
dump and against the per-node values in it.

## Geometry (from the file, see the geometry layer)

40 main blocks + 1 MTP block (`blk.40.nextn.*`, unused by a normal forward — the oracle logs those
tensors as unused unless the MTP path runs). embd 2048, 16 heads / 2 KV, `key_length` 256,
`value_length` 256, rope `dimension_count` 64 with `freq_base` 1e7 and `dimension_sections`
`[11, 11, 10, 0]` (which partitions the 32 rotated frequency *pairs*, so for text input it is the
standard partial RoPE over the first 64 of the 256 head dims). `full_attention_interval` 4 → full
attention at layers 3, 7, … 39; the other 30 layers are linear (GDN). 256 experts, 8 used, expert ff
512, shared expert ff 512.

## The pipeline

    x = token_embd[token]
    for each of the 40 layers:  x = x + attn(norm(x));  x = x + moe(norm_post(x))
    result_norm   = rms_norm(x) * output_norm.weight
    result_output = output.weight @ result_norm          -> 248320 logits

Two norms per layer, both RMS with the weight applied after: `attn_norm` before attention, and
`post_attention_norm` before the MoE. There is no separate FFN norm.

## Linear layer (30 of 40) — GDN

    attn_norm     = rms_norm(x) * blk.N.attn_norm.weight                    # 2048
    qkv           = blk.N.attn_qkv.weight @ attn_norm                       # 8192 = q 2048 | k 2048 | v 4096
    conv_input    = concat(conv_state[3, 8192], qkv[1, 8192])               # 4 rows: 3 history + the new one
    conv_out      = silu(ssm_conv(conv_input, blk.N.ssm_conv1d.weight))     # depthwise, kernel 4, 8192 wide
    q             = view(conv_out[0..2048))    -> [128, 16]                 # 16 key heads of 128
    k             = view(conv_out[2048..4096)) -> [128, 16]
    v             = view(conv_out[4096..8192)) -> [128, 32]                 # 32 value heads of 128
    q_hat         = SCALE(rms_norm(q))                                      # norm then a fixed scale
    k_hat         = SCALE(rms_norm(k))
    alpha         = blk.N.ssm_alpha.weight @ attn_norm                      # 32
    a_softplus    = softplus(alpha + blk.N.ssm_dt.bias)                     # 32
    gate          = a_softplus * blk.N.ssm_a                                # 32, one decay per value head
    beta_sigmoid  = sigmoid(blk.N.ssm_beta.weight @ attn_norm)              # 32
    out, new_state = GATED_DELTA_NET(q_hat, k_hat, v, gate, beta_sigmoid, state)   # 4096 and [128,128,32]
    o             = rms_norm(view(out)) * blk.N.ssm_norm.weight             # per value head, 128
    z             = silu(view(blk.N.attn_gate.weight @ attn_norm))          # 4096 -> [128, 32]
    gated         = o * z
    linear_attn_out = blk.N.ssm_out.weight @ gated                          # 2048
    attn_residual = x + linear_attn_out

The state is 524288 floats per layer = 32 heads × 128 × 128, and the conv state is 24576 = 3 × 8192.
`GATED_DELTA_NET` emits the output *and* the new state in one node (`{4096, 129}`).

**Still to pin**: the exact delta-rule recurrence inside `GATED_DELTA_NET`. The published Gated DeltaNet
form is `S = a·S + b·(v − a·Sᵀk)kᵀ`, `o = Sᵀq`, and the sign of `gate` (whether the decay is `exp(-gate)`
or the gate is already negative) is exactly the kind of detail to settle by measurement, not by
argument: the dump carries the values of `attn_output-0`, `new_state-0` and `cache_s_l0`, so the first
implementation is checked against them directly.

## Full-attention layer (10 of 40)

    attn_norm   = rms_norm(x) * blk.N.attn_norm.weight                      # 2048
    q_full      = blk.N.attn_q.weight @ attn_norm                           # 8192 = q 4096 | gate 4096
    q           = view(q_full[0..4096)) -> [256, 16]                        # 16 heads of 256
    q_normed    = rms_norm(q) * blk.N.attn_q_norm.weight                    # per head
    q_rope      = rope(q_normed, pos)                                       # partial, first 64 dims
    gate_sig    = sigmoid(view(q_full[4096..8192)))                         # 4096, the attention gate
    k           = view(blk.N.attn_k.weight @ attn_norm) -> [256, 2]
    k_normed    = rms_norm(k) * blk.N.attn_k_norm.weight
    k_rope      = rope(k_normed, pos)
    v           = view(blk.N.attn_v.weight @ attn_norm) -> [256, 2]
    cache_k[pos] = k_rope (f16);  cache_v[pos] = v (f16)
    attn_pregate = flash_attention(q_rope, cache_k, cache_v)                # GQA 16 query heads : 2 KV heads
    attn_gated  = attn_pregate * gate_sig
    attn_output = blk.N.attn_output.weight @ attn_gated                     # 4096 -> 2048
    attn_residual = x + attn_output

The q projection is **double width**: 8192 for 16 heads means 256 per head for q plus 256 per head for
the gate that multiplies the attention output. k and v are 2 heads of 256. The KV cache is f16.

## MoE (every layer)

    post        = rms_norm(attn_residual) * blk.N.post_attention_norm.weight
    logits      = blk.N.ffn_gate_inp.weight @ post                          # 256
    probs       = softmax(logits)
    topk        = argsort(probs)[0..8)                                      # 8 of 256
    w           = probs[topk]
    w_norm      = w / clamp(sum(w), min 6.1e-05)
    gate        = blk.N.ffn_gate_exps @ post                                # [512, 8]  per expert
    up          = blk.N.ffn_up_exps   @ post                                # [512, 8]
    h           = swiglu(gate, up)                                          # [512, 8]
    down        = blk.N.ffn_down_exps @ h                                   # [2048, 8]
    routed      = sum over the 8 experts of down * w_norm                  # 2048
    s_gate      = sigmoid(blk.N.ffn_gate_inp_shexp · post)                  # a scalar: the weight is 1-D
    shared      = blk.N.ffn_down_shexp @ swiglu(blk.N.ffn_gate_shexp @ post,
                                                blk.N.ffn_up_shexp   @ post)
    ffn_out     = routed + shared * s_gate
    l_out       = attn_residual + ffn_out

The shared expert's gate is a **dot product** with a 1-D weight (`ffn_gate_inp_shexp.weight`, 2048
values), then a sigmoid: one scalar per token.

## Expert tensor layouts

`ffn_gate_exps` and `ffn_up_exps` are `[2048, 512, 256]` (embd, ff, experts); `ffn_down_exps` is
`[512, 2048, 256]`. Per-expert payload on this rung is 1.08 MiB, which is the number W4's slot sizing
uses.

## How this is checked

1. `tools/quant_parity` already proves the weight decoding for every format this rung uses (max|diff| 0
   against ggml on 2696 tensors).
2. The dump in `bench/raw/rung1-graph.txt` carries per-node values for the prompt "hi": the reference
   forward pass compares its own activation for a named node against the dumped one, which localises a
   mistake to a layer instead of to a token.
3. `bench/oracle/rung1-*.jsonl` carries greedy tokens and top-2 gaps for four prompts, with the
   near-tie rule of PLAN.md §3.3, which is the end-to-end gate.
