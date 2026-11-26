# Rung 3: the file, walked by Limbo's own reader (2026-11-19)

`/home/alca/projects/Strata-data/models/coder-IQ1_M/` — two shards, 29,608,446,496 + 28,800,138,432 B
= 58.4 GB decimal (54.4 GiB), the same accounting the v1 walk closed
(memory `fact:limbo-coder-iq1-m-gguf-layout`, 2026-11-05).

| shard | arch | tensors | data offset | coverage |
|---|---|---|---|---|
| `-00001-of-00002` | `qwen4exp` | 1223 | 11,024,416 | **1.000000** |
| `-00002-of-00002` | (none) | 1 | 192 | **1.000000** |

Shard 2 is one tensor, `per_layer_token_embd.weight`, IQ4_NL, shape `[160, 320001536]`, 28,800,138,240 B
— which is the file size minus its 192-byte header, exactly. Shard 1's `split.count` 2 and
`split.tensors.count` 1224 agree with 1223 + 1. Both shards validate against the v1 byte-exact walk
independently of llama.cpp.

## What this file's recipe actually is

The name says IQ1_M. **There is not one IQ1_M tensor in it** (type 29: zero). The measured types:

| type | tensors | size |
|---|---|---|
| IQ4_NL | 86 + 1 (the PLE tensor in shard 2) | 8,816 + 27,466 MiB |
| IQ3_XXS | 34 | 5,206 MiB |
| IQ2_S | 40 | 5,125 MiB |
| IQ3_S | 20 | 3,438 MiB |
| Q6_K | 129 | 1,791 MiB |
| BF16 | 484 | 1,397 MiB |
| **Q2_0** | **9** | 1,013 MiB |
| IQ4_XS | 45 | 909 MiB |
| Q5_K / Q4_K | 35 / 47 | 331 / 189 MiB |
| F32 / F16 / Q8_0 | 292 / 1 / 1 | 9.6 / 0.08 / 1.66 MiB |

**Correction to the v1 note**, which said "routed tensors dtype 42 = Q2_0": the routed experts are
`ffn_gate_exps` and `ffn_up_exps` in **IQ3_XXS** for all 48 layers, and `ffn_down_exps` in **IQ4_NL**
except for **9 layers in Q2_0** (`blk.{1,8,10,12,13,20,21,30,36}`). So the mixed-precision recipe is
per-tensor and per-layer, and W5 needs IQ3_XXS + IQ4_NL + Q2_0 kernels for the routed path — not an
IQ1_M kernel. Q2_0 is the BitNet-style 2-bit type (64 elements / 18 bytes), and its kernel is a real
one to write, not a corner case.

BF16 carries the hyper-connection tensors (`hc_attn_down/up`, `hc_ffn_down/up`, `[320, 10240]` — that
is `hyper_connection.low_rank` 320 × `count` 4 × 2560) and the router (`ffn_gate_inp`, and
`ffn_gate_inp_shexp`, which is 1-D).

## Geometry, from the metadata

`qwen4exp`: 48 layers, `embedding_length` 2560, 24 heads / 2 KV heads, `context_length` 262144,
`full_attention_interval` **4** → 12 full-attention layers and 36 linear ones, which is exactly what the
roles show (12 × `attn_q/k/v/o` + `indexer.*`, 36 × `ssm_*` + `attn_qkv` + `attn_gate`). Expert side:
256 experts, 10 active, `expert_feed_forward_length` 640, shared expert 640. SSM/GDN:
`conv_kernel` 4, `state_size` 128, `group_count` 16, `time_step_rank` 48, `inner_size` 6144.
Sparse attention: `attention.indexer.head_count` 4, `key_length` 128, `top_k` 2048, plus 48
`compress_ratios`. Hyper-connections: count 4, low_rank 320. PLE: `ple.layers` (1 entry),
`ngram_size` 3, `heads_per_ngram` 8, with `ple_key/value/conv1d/norm_*` in shard 1 and the big table in
shard 2.

**No `nextn.*` role exists in this file** — unlike rung 1, which carries 4 `nextn.*` tensors (the MTP
head). W5's speculation cannot use a model-native MTP head on rung 3 as this file stands; it needs
prompt lookup, a separate drafter, or a differently-quantised file that carries the head.

## The 46 roles of `qwen4exp` (from the file, not from a list)

Per-layer (48): `ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`, `ffn_gate_shexp`, `ffn_up_shexp`,
`ffn_down_shexp`, `ffn_gate_inp`, `ffn_gate_inp_shexp`, `hc_attn_down`, `hc_attn_up`, `hc_attn_inject`,
`hc_attn_norm`, `hc_ffn_down`, `hc_ffn_up`, `hc_ffn_inject`, `hc_ffn_norm`.
Per full-attention layer (12): `attn_q`, `attn_k`, `attn_v`, `attn_output`, `attn_q_norm`,
`attn_k_norm`, `indexer.q_proj`, `indexer.k_proj`, `indexer.q_norm`, `indexer.k_norm`.
Per linear layer (36): `attn_qkv`, `attn_gate`, `ssm_a`, `ssm_alpha`, `ssm_beta`, `ssm_conv1d`,
`ssm_dt.bias`, `ssm_norm`, `ssm_out`.
Global: `token_embd`, `output`, `output_hc_down`, `output_hc_up`, `output_hc_norm`, `ple_conv1d`,
`ple_key`, `ple_value`, `ple_norm_conv`, `ple_norm_key`, `ple_norm_query`.

`gguf_info <file> --roles` prints this for any file, `--type N` lists the tensors of one type, and
`--tensor NAME` prints one tensor's shape and span — all three exist so the next wave does not have to
re-derive them by hand.
