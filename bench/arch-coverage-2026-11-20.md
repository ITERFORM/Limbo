# Architecture coverage on the CPU reference (2026-11-20)

Limbo's CPU reference (`limbo::step::Ref`, `src/step.cpp`) was qwen35moe only. This file records, milestone by
milestone, what it takes to run three more models (Qwen3.8-27B dense `qwen35`, Gemma-4 E2B and Gemma-4 26B-A4B
`gemma4`) and the numbers that prove each one against llama.cpp. Every number comes from a command run in this
session; oracle captures are CPU-only (`CUDA_VISIBLE_DEVICES=` so llama.cpp's CUDA backend finds no device and
`op_offload` has nothing to offload to — its log says `ggml_cuda_init: failed to initialize CUDA: no CUDA-capable
device is detected`), at nice 10 with 8 threads.

Files (HF cache, snapshot paths):

| tag | file | arch | size |
|---|---|---|---|
| q27 | `Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf` | qwen35 | 9.72 GiB, 866 tensors |
| ge2 | `gemma-4-E2B-it-Q4_K_M.gguf` | gemma4 | 2.88 GiB, 601 tensors |
| g26 | `gemma-4-26B-A4B-it-UD-IQ4_XS.gguf` | gemma4 | 12.65 GiB, 658 tensors |

## M1. The model map covers the three new files (`tests/model_test.cpp`)

`src/model.cpp` now reads, from the file's own metadata, what the new architectures need: the dense FFN width (one
value or one per layer), `head_count_kv` per layer, the sliding-window pattern (one bool per layer), the window,
the sliding layers' head width / rotary width / rope base, `shared_kv_layers` (and the layer each shared layer
reads, `Geometry::kv_source`), the per-layer-input width, the final logit soft cap. Its structural assertion changed
from "a role covers a union of {full, linear, MTP}" to "a role covers a union of whole classes of layers, a class
being (kind, sliding or global, owns its K/V or reads another's)", which is the same rule on rung 1 (no sliding
layers, no shared KV) and which lets `attn_v.weight` be absent on gemma's global layers without the map learning a
wrong structure.

`LIMBO_TEST_GGUF=<file> build-arch/bin/tests/model_test`, all four files (rung 1 included): `model: all passed`.

| file | layers | embd | heads / kv | notes the test pins |
|---|---:|---:|---|---|
| q27 (qwen35) | 65 = 64 + 1 MTP | 5120 | 24 / 4 | 16 attention + 48 GDN layers; dense ff 17408; GDN conv 4, state 128, 16 key groups, 48 value heads, inner 6144; q projection is `2 * 24 * 256` rows (query + gate) |
| ge2 (gemma4) | 35 | 1536 | 8 / 1 | sliding window 512 on 28 layers (head 256, rope base 1e4), global on 7 (head 512, rope base 1e6); the last 20 layers read the K/V of layer 13 (sliding) or 14 (global); ff 6144 on layers 0-14, 12288 on 15-34; per-layer input 256; softcap 30 |
| g26 (gemma4) | 30 | 2816 | 16 / 8 (2 on the 5 global layers) | sliding window 1024 on 25 layers; 128 experts, 8 used, ff 704 (fused `ffn_gate_up_exps`) beside a dense ff 2112 on every layer; global layers have no `attn_v`; no output matrix (tied to `token_embd`); softcap 30 |

## M2. Weight formats: 13 of 13 on the 27B, 5 of 5 on each gemma, 0 differences against ggml's `to_float`

The rung-1 decoder knew 9 formats. The 27B also carries IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S and IQ1_M (and BF16, Q2_K,
Q4_K, Q6_K, which were there). Added in `src/quants.cpp`: those five decoders, written from the block layouts in
`include/limbo/quants.hpp`, reading five more ggml grids (`ksigns_iq2xs`, `iq2xxs_grid`, `iq2xs_grid`,
`iq3xxs_grid`, `iq1s_grid`) from the new `src/ggml_grids_ext.cpp`, extracted verbatim from `ggml-common.h` under the
same MIT notice as `src/ggml_grids.cpp`.

`build-arch/bin/tools/quant_parity <file>` (5 rows sampled per tensor, every value compared):

| file | formats checked | differences |
|---|---|---:|
| q27 | F32 Q2_K Q4_K Q6_K IQ2_XXS IQ2_XS IQ3_XXS IQ1_S IQ3_S IQ2_S IQ4_XS IQ1_M BF16 | 0 |
| ge2 | F32 Q4_K Q5_K Q6_K BF16 | 0 |
| g26 | F32 Q8_0 IQ4_NL IQ3_S IQ4_XS | 0 |

`tests/quants_test` (block-walk checks for the five new formats): `quants: all passed`.

## M3. Tokenizer: the gemma4 family, byte-exact against `llama-tokenize` (20 of 20 files, four models)

The GGUF declares `tokenizer.ggml.model = "gemma4"` (not a SentencePiece `llama` model): a vocabulary of 262,144
tokens with 514,906 merges, `tokenizer.ggml.add_space_prefix = 0`, `add_bos_token = 1`. llama.cpp cuts it as BPE
over raw UTF-8 (`src/llama-vocab.cpp`, pre-type GEMMA4): the normaliser turns every space into U+2581, the
pre-split is only `[^\n]+|[\n]+`, merges run per run over whole characters with no byte alphabet, a run that is
only newlines is taken whole when the vocabulary has it, and a symbol with no token falls back to its bytes'
`<0xXX>` tokens. `src/tok.cpp` now implements that family next to the Qwen byte-level one, selected by
`tokenizer.ggml.model`.

One more oracle behaviour had to be mirrored: llama.cpp treats the tokens it recognises as end-of-generation by text
(`<eos>`, `<turn|>`, `<|tool_response>`, ...) as control tokens whatever type the file gave them. gemma types
`<eos>` as normal, so before the fix `"<eos>"` in a prompt was five characters instead of token 1; `</s>` stays an
ordinary token when the vocabulary also has `<|tool_response>`.

`scripts/tok_compare.sh <model>` (a scratch copy pointing at `build-arch/bin/tools/tok_dump`, same logic; the
script itself hard-codes `build/bin`), 10 files in the raw pass and the special-parsing pass each:

| model | files matched | tokens in the largest file |
|---|---:|---:|
| gemma-4-E2B (Q4_K_M) | 20 / 20 | 44,907 (`long.txt`) |
| gemma-4-26B-A4B (UD-IQ4_XS) | 20 / 20 | 44,907 |
| Qwen3.8-27B (`qwen35`, Qwen BPE) | 20 / 20 | 44,907 |
| Qwen3.6-35B-A3B (rung 1, after the change) | 20 / 20 | |
| Cyber-Tiel-Coder-35B-A3B (rung 2, after the change) | 20 / 20 | |

Four extra gemma files (chat markup `<bos><|turn>user\n...<turn|>`, `<|"|>`, `<eos>`, `</s>`, `<mask>`, runs of
newlines, CRLF, tabs, `<0x41>` written out, an already-escaped U+2581, emoji, CJK), both passes, E2B and 26B:
4 / 4 each. `tests/tok_test` with `LIMBO_TEST_GGUF` on each of the four files: `tok: all passed`, including new
unit tests of the raw-UTF-8 path on a built vocabulary.

## M4. The reference is a recipe now; rung 1 is unchanged

`limbo::step::Ref` reads a recipe from the file at `open()` (public as `Ref::config().layers`):

- per layer a **mixer** (`Gdn`, or `Attention` with: sliding window or global, heads / KV heads / head width per
  layer, rotary width and base per kind, optional per-pair frequency divisors from `rope_freqs`, a gated double-width
  q, V taken from K when the layer has no `attn_v`, an unweighted V norm, a score scale, and a K/V cache of its own or
  another layer's), an **FFN** (`Dense`, `MoeShared` = routed experts + a shared expert behind a sigmoid gate, or
  `DenseAndMoe` = a dense MLP in parallel with routed experts, each with its own norms and the router reading the
  residual stream; activation silu or gelu; fused gate+up experts; a per-expert down scale) and the **norm
  placement** (a norm on the mixer output and on the FFN output, or none);
- per model the embedding scale, per-layer token embeddings (the block after every FFN), the tied or separate head,
  the final logit soft cap.

Every tensor of the file is either resolved by the recipe or named in `Ref::unused_tensors()`; `open()` fails on any
other (the claim check, one level above `Model`'s role map). Not run, by design: the MTP block (`blk.40` on
rung 1, 15 tensors of `blk.64` on the 27B) and, on E2B, the K/V projections and K norm of the 20 layers that read
another layer's cache (60 tensors). The 26B and the rest have none.

Recipes as read from the files (`recipe_dump`, a scratch program over `Ref::config()`):

| model | mixers | FFN | norms / extras |
|---|---|---|---|
| qwen35moe 35B-A3B | 30 GDN + 10 attention (gated q, 16/2 x 256, rope 64 of 256) | 40 x routed 8/256, ff 512, + shared expert | norm before each half only |
| qwen35 27B | 48 GDN (16 key / 48 value heads x 128) + 16 attention (gated q, 24/4 x 256, rope 64) | 64 x dense SwiGLU, ff 17408 | as above |
| gemma4 E2B | 28 sliding (window 512, 8/1 x 256, rope 256, base 1e4) + 7 global (8/1 x 512, rope 512, base 1e6, `rope_freqs`); layers 15-34 read the K/V of layer 13 (sliding) or 14 (global) | 35 x dense GeGLU, ff 6144 (layers 0-14) / 12288 (15-34) | norm after the mixer and after the FFN; per-layer embeddings (256 wide) after every FFN; layer scalar; embedding x sqrt(1536); tied head; logits 30 tanh(x/30) |
| gemma4 26B-A4B | 25 sliding (window 1024, 16/8 x 256) + 5 global (16/2 x 512, V = K, `rope_freqs`) | 30 x (dense GeGLU ff 2112 + routed 8/128, ff 704, fused gate+up, per-expert down scale) | as E2B, plus 3 more norms around the two FFN branches; embedding x sqrt(2816); no per-layer embeddings |

**Rung 1 regression** (`oracle_gate` on `bench/oracle/rung1-*.jsonl`, default fp32 arithmetic, 8 threads, same
captures as `bench/w1-cpu-gate-2026-11-19.md`): hi 1/2 (the one miss at oracle gap 0.0637, Limbo top-2
599 (19.2427) / 1044 (19.2419), the numbers the old gate recorded), prose 16/16 (max |d1| 2.0104), code 16/16
(0.9503), tools 16/16 (1.0899): **49 of 50, max |d1| identical to the old gate's 2.010 / 0.950 / 1.090**.
`ref_forward --layer-gate` is unchanged in what it computes for rung 1 (rerun below under M7).

## M5. Gemma-4: state when the operator stopped it (2026-11-20, focus moved to Qwen)

Stopped on the operator's word ("NOT google/gemma"); nothing below is being worked on. What compiles and what was
measured, so that nobody has to guess:

| model | forward pass | gate against its CPU capture (`bench/oracle/gemma4-*-{hi,prose,code}.jsonl`, 16 positions each) |
|---|---|---|
| gemma-4-E2B-it Q4_K_M | runs; recipe complete (sliding + global layers, shared-KV tail, per-layer embeddings, layer scalar, soft cap) | hi 16/16, prose 16/16, code 16/16 (max abs logit difference 2.79, 4.55, 2.27) |
| gemma-4-26B-A4B UD-IQ4_XS | runs; dense + routed experts, fused gate/up, per-expert down scale | hi 16/16, prose 16/16, **code 8/16** (2 disagreements at oracle gap < 0.1, 6 at gap >= 0.1, max abs logit difference 25.6): **open, not diagnosed** |

Evidence that the recipe is structurally right, so the 26B `code` miss is something narrower:

- E2B `--layer-gate` (every layer fed the oracle's own input, `--oracle-arith`, 56 tokens): no spike in any of the 35
  layers, mean error of a layer's update 0.06 % to 0.7 %, head max |dlogit| 0.00004. A 700-token prompt (window 512)
  over layers 13-19: attention-half error 0.105 % beyond position 512 against 0.076 % before it, and with the window
  switched off (negative control) 8.7 %: the sliding-window mask is verified against the oracle.
- 26B `--layer-gate` (all 30 layers, 56 tokens): no spike, mean error of a layer's update 0.14 % to 0.5 %, 0 of 1,625
  (layer, row) pairs routed to a different expert set, head max |dlogit| 0.00005.
- llama.cpp against itself on E2B (prompt as one batch against one token per decode, CPU): mean |dlogit| 1.0 (prose) and
  0.73 (code), max 6.9 and 2.3. The gemma checkpoints amplify arithmetic differences far more than the Qwen ones
  (the 27B's gate differences are 0.1), so E2B's 2-5 nat differences are inside the oracle's own noise floor.
- One real bug was found and fixed on the way: a generic `<name>.scale` handler took gemma's `ffn_gate_inp.scale` (a
  vector, the router's per-channel input multiplier) for a scalar (all 26B FFN halves were 13-30 % off until then).

Not done: the 26B `code` prompt (could be the router's near ties, or a path the 56-token layer gate does not exercise),
the 26B window beyond 1024 positions, a BOS-prefixed capture (the captures follow `oracle_dump`: no BOS), CPU speed
measured on a quiet box. All of it compiles and the gemma paths cost nothing on the Qwen ones.

## M6. Qwen3.8-27B (`qwen35`, dense): the gate passes, 48 of 48

The 27B is rung 1's dense sibling: the same GDN + gated-attention stack with a dense SwiGLU in place of the MoE. The
recipe read from its file: 64 main layers (48 GDN with 16 key heads and 48 value heads of 128, so a value head shares
key head `h % 16`; 16 gated-attention layers of 24 heads / 4 KV heads x 256, rope 64 of 256), a dense FFN of 17,408 on
every layer, embd 5120, one MTP block (`blk.64`, 15 tensors, not run). Nothing new in the forward pass beyond what
rung 1 already did; the work was the formats (M2), the geometry read from the file instead of rung 1's constants
(`n_v_head` was `2 * n_head`, now the file's `ssm.time_step_rank`) and the tokenizer (M3).

- **Formats:** `quant_parity` 13 of 13 formats, 0 differences against ggml's `to_float` (M2).
- **Tokenizer:** `tok_compare` 20 of 20 (M3).
- **Gate:** teacher-forced against the llama.cpp CPU capture (`bench/oracle/qwen35-27b-{hi,prose,code}.jsonl`, captured with
  `CUDA_VISIBLE_DEVICES=` so nothing was offloaded), default fp32 arithmetic, 8 threads, `oracle_gate`:

| prompt | prompt tokens | positions | agree | disagree (gap < 0.1) | disagree (gap >= 0.1) | max abs d(oracle top-1 logit) | max abs d(gap) |
|---|---:|---:|---:|---:|---:|---:|---:|
| hi | 2 | 16 | 16 | 0 | 0 | 0.152 | 0.153 |
| prose | 55 | 16 | 16 | 0 | 0 | 0.539 | 0.609 |
| code | 95 | 16 | 16 | 0 | 0 | 0.767 | 1.282 |
| **all** | | **48** | **48** | **0** | **0** | mean 0.167 | |

The per-position lines are in `bench/arch-coverage-2026-11-20/q27-*.jsonl`. The logit differences (mean 0.17 nats)
are a third of rung 1's 0.4 because the capture is pure CPU (no op_offload noise) and the model is dense; three
positions of `hi` have an oracle gap below 0.1 (0.013 the smallest) and the reference gets all three right.

- **CPU speed** (8 threads, `nice -n 10`, on a box shared with other agents, so read it as a ceiling on the cost): decode
  0.058 to 0.068 tokens/s (14.7 to 17.2 s per token), prefill 0.043 to 0.053 tokens/s (it includes the first read of the 10 GB file). The reference decodes every weight into floats one block at a time and
  sums sequentially (that is what keeps it exact and what makes it slow): 27 G weights per token at about 1.6 G/s in
  all.
