# Inkling-2-Mini — the CPU reference against the oracle (2026-11-23)

`inkling2_loop`, rung 4. The reference is `limbo::step::Ref::forward_inkling` (`src/step.cpp`: the
recipe resolved in `43be96e`, the graph written this session). The oracle is the operator's
llama-inkling build (`~/projects/llama-inkling`, sources 10:41, build 10:49) — `oracle_dump` and
`oracle_nodes` linked against it through `make BUILD=build-inkoracle oracle
LLAMA_DIR=/home/alca/projects/llama-inkling`. The captures `bench/oracle/inkling2-{hi,prose,code,tools}.jsonl`
were taken CPU-only (`n_gpu_layers 0`, flash off, greedy), like every other rung's captures. The raw
outputs are in `build/tmp/inkgate/` (scratch, not committed except this file).

## 1. Verdict

**The gate passes at the W1 bar.** Teacher-forced on 64 positions (hi 16, prose 16, code 16,
tools 16), default fp32 arithmetic: **61 agree**, 1 disagrees at an oracle gap of **< 0.1** (the
rule of PLAN §3.3), and 2 disagree at oracle gaps 0.153 / 0.118 where **llama.cpp's own
batch-vs-one-token noise (oracle_noise) is 0.193 / 0.087 nats and flips the order by itself**
(noise gap −0.039 / −0.129 there; own noise over the code prompt: max 0.754, mean 0.141 nats).
Our worst |Δlogit| at the oracle's top-1 id on any prompt: 0.77.

- Per-slot `l_out` against `oracle_nodes` (hi, row 0, all 48 visit slots): relative error
  **max 5.8%, mean 1.7%** — arithmetic-noise class, no structural divergence anywhere in the looped
  core, the prelude or the coda.
- The reference step runs **0.33-0.41 tok/s prefill and decode at 8 threads** (48 visits per token,
  dequant on the fly, ~8.5 GB mapped). Exact, not fast — the GPU port is the fast path.

## 2. The gate

| prompt | prompt tokens | positions | agree | disagree (oracle gap < 0.1) | disagree (gap ≥ 0.1) | max \|Δlogit\| of oracle top-1 |
|---|---:|---:|---:|---:|---:|---:|
| hi | 2 | 16 | 16 | 0 | 0 | 0.069 |
| prose | 57 | 16 | 15 | 1 | 0 | 0.287 |
| code | 102 | 16 | 14 | 0 | 2 | 0.211 |
| tools | 151 | 16 | 16 | 0 | 0 | 0.765 |
| **all** | | **64** | **61** | **1** | **2** | |

The two code disagreements (gen 9 pos 110, gen 10 pos 111): Limbo's own top-2 gaps there are
**0.038 and 0.012** — coin flips in our distribution — with |d1| 0.15 / 0.11. The oracle's own
noise between batched prefill and one-token decode at exactly those positions is 0.193 / 0.087 and
its gap flips sign (−0.039 / −0.129), so llama.cpp does not agree with itself about the winner
there either.

## 3. What the gate caught, and the two fixes in the step

1. **The short-convolution weights were read byte-wise.** `ink_convolve` indexed the F32
   `{n_conv, channels}` weight through a `uint8_t*`: tap 3 of channel 0 (float 0.789 = bytes
   `00 00 4a 3f`) acted as 0x3f = 63. The measured `acc[ch]/x[ch]` factors (61, 62, 62, 63 for
   channels 0-3) were those bytes. Every conv output was garbage; V's per-head RMS norm hid it,
   K exposed it. Fixed: F32 weights read as floats, other types through the dequant path.
2. **The sink and the relative bias enter the softmax after the q.k scale.** The first
   implementation folded `bias − sink` into the pre-scale sum (`scale*(q.k + bias − sink)`, from
   reading the graph's plumbing as `mul_mat → add kq_b → soft_max_ext(scale)`). The oracle's own
   numbers say otherwise: with the post-scale fold (`scale*q.k + bias − sink`, sink column at
   logit 0 in the denominator) the oracle's `ffn_inp-0` is reproduced to a mean of 4.8e-4 over all
   2304 channels, while the pre-scale fold misses by 0.09 with completely different attention
   probabilities (the oracle's attention at pos 0 emits ~0.0085 of V on head 0 — the sink absorbs
   ~99%, which only the post-scale fold predicts). This is also what the Python reference does
   (`attn_weights = q·k^T * scaling; attn_weights += position_bias`) and what docs/inkling2.md §3
   already described. Lesson recorded: the oracle's numbers, not a reading of the graph plumbing,
   settle a fold order.

Everything else — the depth mixes (softmax over the bank per query), the L2 clips with
`1e-6`-shifted norms, the per-head V norm without a weight, the log-scaled τ on the global layers'
q and r, the fused gate/up per-expert NVFP4/Q2_0 scales, SiTU (`beta·tanh(g/β)·sigmoid(g) ·
lb·tanh(u/lb)`), the sigmoid+bias routing with the `1e-20` renormalisation and the ×8 weight scale,
the loop's bank reset and controller gates, and the 200058-row unpadded head — matched the oracle's
nodes on the first pass after the two fixes.

## 4. Trace nodes kept for the next wave

`Ref::trace()` for `ink.on` records, per visit slot and only at position 0: `kq` (the raw q·k of the
token's own key, per head), `Qcur`, `Kcur`, `Vcur`, `Rcur`, `attn_out`, `ffn_inp`,
`ffn_moe_logits`, `ffn_moe_topk`, `ffn_moe_weights_scaled`, plus `attn_out[slot]`/`l_out[slot]`
after every visit. The names match llama.cpp's own `cb()` names (`oracle_nodes` dumps the same
names), keyed by visit slot — the CUDA port's per-node gate reads them unchanged.

## 5. Commands

```
make BUILD=build-inkoracle oracle LLAMA_DIR=/home/alca/projects/llama-inkling
./build-inkoracle/bin/oracle/oracle_dump -m ~/models/inkling/Inkling-2-Mini-UD-IQ2_XXS.gguf \
    -f bench/prompts/$p.txt --n-predict 16 -t 8 --out bench/oracle/inkling2-$p.jsonl
./build/bin/tools/oracle_gate -m ~/models/inkling/Inkling-2-Mini-UD-IQ2_XXS.gguf \
    --oracle bench/oracle/inkling2-$p.jsonl --out build/tmp/inkgate/$p.jsonl -t 8
./build-inkoracle/bin/oracle/oracle_nodes -m ... --oracle bench/oracle/inkling2-hi.jsonl \
    --out build/tmp/inkgate/hi-lout.bin -t 4 --node l_out      # and Qcur/Kcur/Vcur/Rcur/ffn_moe_*
./build-inkoracle/bin/oracle/oracle_noise -m ... --oracle bench/oracle/inkling2-code.jsonl \
    --out build/tmp/inkgate/code-noise.jsonl -t 8
```
