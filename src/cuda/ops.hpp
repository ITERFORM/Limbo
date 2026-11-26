// Copyright (c) 2026 ITERFORM. All rights reserved.
#pragma once
// src/cuda/ops.hpp — the small device kernels of the forward pass (everything that is not a GEMV).
//
// Each launcher mirrors one step of the CPU reference (src/step.cpp) op for op: where the reference sums in
// double (the RMS norms) the kernel sums in double, where it works in float the kernel works in float, and
// the transcendental functions are the accurate ones (expf, log1pf, sin, cos), never the fast intrinsics.
// Only .cu translation units include this header.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace limbo::cuda::ops {

// y = x * inv(rms(x)) * w, the sum of squares in double; eps is the float the reference passes.
void rmsnorm(const float* x, const float* w, float* y, int n, float eps, cudaStream_t s);

// Depthwise causal conv over the last `taps` rows (taps-1 of history in `hist`, row-major [taps-1][width],
// then the new row `qkv`), followed by SiLU; shifts the history by one row. taps is 4 for rung 1.
void conv_silu(const float* qkv, float* hist, const float* w, float* out, int width, cudaStream_t s);

// L2 norm of every 128-wide head of q and k (2 * n_k_head heads, contiguous in `in`):
// x * (1/sqrt(sum_sq/hd + eps)) * scale, with the reference's float eps and scale.
void head_norm(const float* in, float* out, int n_heads, int hd, float eps, float scale, cudaStream_t s);

// The delta-rule recurrence for every value head (one block per head, one thread per value column).
// qk = [q_hat | k_hat], each n_k_head * hd; v = n_v_head * hd; state S = [n_v_head][hd][hd].
// Writes o[h*hd + j] = (S^T q)[j] * scale. Value head h reads key head h % n_k_head.
void gdn_scan(float* S, const float* qk, const float* v, const float* alpha, const float* beta, const float* dt_bias,
              const float* ssm_a, float* o, int n_v_head, int n_k_head, int hd, int dt_rank, float scale,
              cudaStream_t s);

// gated[h*hd + j] = (o * inv_rms(o_h) * sn[j]) * silu(z[h*hd + j]); eps is the reference's double.
// sigmoid_z: qwen4exp's GDN gates with sigmoid(z) instead of silu(z) (the one difference from qwen3.5).
void gdn_gate(const float* o, const float* z, const float* sn, float* gated, int n_heads, int hd, double eps,
              cudaStream_t s, bool sigmoid_z = false);

// Full-attention preparation, one launch: per-head RMS norm with weights, NEOX partial RoPE and the f16 KV
// cache write for the position in device memory, *pos (so a captured graph replays for any position). qfull is
// [n_head][q | gate] of hd each; q_out is [n_head][hd].
void attn_prep(const float* qfull, const float* kbuf, const float* vbuf, const float* qn, const float* kn,
               float* q_out, __half* kcache, __half* vcache, const int* pos, int vpitch, double eps, double rope_base,
               int rope_dim, int n_head, int n_head_kv, int hd, cudaStream_t s);

// GQA attention over cache positions [0, *pos], softmax in float, times the sigmoid of the gate half of qfull.
// The scores live in dynamic shared memory sized for max_ctx positions (a graph cannot resize it).
void attention(const float* q, const __half* kcache, const __half* vcache, const float* qfull, float* out,
               const int* pos, int max_ctx, int vpitch, int n_head, int n_head_kv, int hd, float scale, cudaStream_t s);

void swiglu(const float* g, const float* u, float* h, int n, cudaStream_t s);

// softmax over the router logits, then the top n_used in descending order with their weights
// p / max(sum of the chosen p, 6.1035e-5). Ties go to the lower expert id.
void router_topk(const float* logits, int n_expert, int n_used, int32_t* ids, float* wts, cudaStream_t s);

// sg = sigmoid(gw . post)
void shexp_gate(const float* gw, const float* post, float* sg, int n, cudaStream_t s);

// x[j] += (sum_i dn[i][j] * wts[i]) + shared[j] * sg
void moe_combine(float* x, const float* dn, const float* wts, const float* shared, const float* sg, int n,
                 int n_used, cudaStream_t s);

// dn[miss_i[m]][j] += packed[m][j], for m < n_miss. The routed experts a step splits between the device (a
// resident expert, already written into its own row of dn) and the CPU lane (whose jobs come back packed by
// job index): this puts the lane's rows back where the expert that produced them belongs, so one combine
// kernel sums both halves in the model's own order.
void moe_scatter_add(float* dn, const float* packed, const int32_t* miss_i, int n_miss, int n, cudaStream_t s);

// *tok = argmax of logits (ties: lowest index).
void argmax(const float* logits, int n, int32_t* tok, cudaStream_t s);

// *out[t] = argmax of row t of logits[T][n] (ties: lowest index), one block a row.
void argmax_rows(const float* logits, int n, int T, int32_t* out, cudaStream_t s);

// ---- batched forms (prompt prefill): the same arithmetic over T rows, one launch ---------------------------

// rmsnorm of `rows` rows of n floats (rows contiguous).
void rmsnorm_rows(const float* x, const float* w, float* y, int rows, int n, float eps, cudaStream_t s);

// conv_silu over T consecutive tokens: qkv[T][width] in, out[T][width]; `hist` is the 3-row history before token 0
// and the 3 last input rows after token T-1.
void conv_silu_rows(const float* qkv, float* hist, const float* w, float* out, int width, int T, cudaStream_t s,
                    bool write_back = true, int t0 = 0);

// head_norm for T rows: row t of `in` (stride in_stride) -> row t of `out` (stride out_stride).
void head_norm_rows(const float* in, int in_stride, float* out, int out_stride, int T, int n_heads, int hd, float eps,
                    float scale, cudaStream_t s);

// gdn_gate for `rows` (token, head) rows of 128 at once, one warp per row. sigmoid_z picks the gate's
// nonlinearity, exactly as gdn_gate's flag does: false (the default, and what rung 1 wants) is silu(z), true is
// sigmoid(z) -- qwen4exp's recipe. They are not close: silu(z) is z*sigmoid(z).
void gdn_gate_rows(const float* o, const float* z, const float* sn, float* gated, int rows, int hd, double eps,
                   cudaStream_t s, bool sigmoid_z = false);

// The scan's per-token coefficients, in place for T rows of dt_rank values (row stride ab_stride):
// alpha <- exp(softplus(alpha + dt_bias) * ssm_a), beta <- sigmoid(beta). Run before gdn_scan_rows.
void gdn_coef_rows(float* alpha, float* beta, int ab_stride, const float* dt_bias, const float* ssm_a, int T, int dt_rank,
                   cudaStream_t s);

// gdn_scan over T consecutive tokens (sequential in token order inside one kernel per head): row t of qk, v,
// alpha, beta (strides in floats; alpha and beta as gdn_coef_rows left them) -> row t of o. The state S ends after
// token T-1. Bit for bit the decode kernel's arithmetic. write_back false: o is still written for every row but S is
// left exactly as it was found -- a verify window's rows are candidates, and only commit may move the recurrence.
void gdn_scan_rows(float* S, const float* qk, int qk_stride, const float* v, int v_stride, const float* alpha,
                   const float* beta, int ab_stride, float* o, int o_stride, int T, int n_v_head, int n_k_head, int hd,
                   int dt_rank, float scale, cudaStream_t s, bool write_back = true, float* S_out = nullptr);

// KV cache layout (decode and prefill alike): K is [pos][kv head][hd] f16; V is stored transposed,
// [kv head][hd][vpitch] f16 with vpitch = n_ctx rounded up to 64, so a thread that owns one value dimension
// reads consecutive positions and the tensor-core kernel loads four positions of one dimension in one 8-byte load.
//
// attn_prep / attention for T tokens at positions pos0 .. pos0+T-1 (rows of the activation buffers are
// contiguous: qfull 2*n_head*hd, kbuf and vbuf n_head_kv*hd, q_out n_head*hd). attention_rows reads the cache
// for every position <= its own, so the chunk's keys must be written first (attn_prep_rows).
// rope_table fills cs[T][2][rope_dim/2] (cos row, then sin row, per token) for positions pos0 .. pos0+T-1; the
// attn_prep_rows of every full-attention layer of the chunk then read it instead of recomputing theta in double.
void rope_table(float* cs, int pos0, int T, double rope_base, int rope_dim, cudaStream_t s);
void attn_prep_rows(const float* qfull, const float* kbuf, const float* vbuf, const float* qn, const float* kn,
                    float* q_out, __half* kcache, __half* vcache, const float* rope_cs, int pos0, int T, int vpitch,
                    double eps, int rope_dim, int n_head, int n_head_kv, int hd, cudaStream_t s, const int* pt = nullptr);
// exact = false drops the low halves of Q and P (one f16 product per tile instead of two: llama.cpp's flash attention
// rounds both to f16 too); the prefill uses it in its 1-plane (llama numerics) mode and half the tensor time at depth.
void attention_rows(const float* q, const __half* kcache, const __half* vcache, const float* qfull, float* out,
                    int pos0, int T, int n_ctx, int vpitch, int n_head, int n_head_kv, int hd, float scale,
                    cudaStream_t s, bool exact = true, const int* pt = nullptr);

// The same attention over a window of keys (positions [kbase, kbase + n_rows), kbase and n_rows multiples of 64): several windows
// in ascending order, flags bit 0 on the first and bit 1 on the last, give the bits of one pass. `out` carries the numerator
// between windows and ml [T][n_head][2] the running (max, sum); flags = 3 needs neither ml nor a previous state.
void attention_rows_window(const float* q, const __half* kwin, const __half* vwin, const float* qfull, float* out, int pos0, int T,
                           int kbase, int n_rows, int vpitch, int n_head, int n_head_kv, int hd, float scale, int flags, float* ml,
                           cudaStream_t s, bool exact = true, const int* pt = nullptr);

// attn_prep_rows over the Q8 cache of include/limbo/cuda/kv.hpp: K and V of the chunk's positions are quantised (int8 + a half
// scale per 32 dimensions, a row per (position, kv head)); q_out as in attn_prep_rows.
void attn_prep_rows_q8(const float* qfull, const float* kbuf, const float* vbuf, const float* qn, const float* kn, float* q_out,
                       int8_t* kq, uint16_t* ks, int8_t* vq, uint16_t* vs, const float* rope_cs, int pos0, int T, double eps,
                       int rope_dim, int n_head, int n_head_kv, int hd, cudaStream_t s, const int* pt = nullptr);

// Q8 -> f16 for the window [k0, k0 + n_rows) of one layer's cache: K [row][kv head][256], V transposed [kv head][256][wp]; positions
// at or past valid_end become 0. n_rows a multiple of 32.
void kv_expand_window(const int8_t* kq, const uint16_t* ks, const int8_t* vq, const uint16_t* vs, __half* kw, __half* vw, int k0,
                      int n_rows, int valid_end, int n_head_kv, int wp, cudaStream_t s, const int* pt = nullptr);
// The same two over a Q4 cache: the prep rotates the query, the key and the value and writes 4-bit codes; the expansion's keys
// stay rotated (the rotated query scores them as the plain pair) and its values are rotated back.
void attn_prep_rows_q4(const float* qfull, const float* kbuf, const float* vbuf, const float* qn, const float* kn, float* q_out,
                       int8_t* kq, uint16_t* ks, int8_t* vq, uint16_t* vs, const float* rope_cs, int pos0, int T, double eps,
                       int rope_dim, int n_head, int n_head_kv, int hd, cudaStream_t s, const int* pt = nullptr);
void kv_expand_window_q4(const int8_t* kq, const uint16_t* ks, const int8_t* vq, const uint16_t* vs, __half* kw, __half* vw, int k0,
                         int n_rows, int valid_end, int n_head_kv, int wp, cudaStream_t s, const int* pt = nullptr);

// Paged K/V (docs/paged-kv.md): pt is the sequence's page table (int32 per 256-position chunk, device memory; null: the chunk is its
// own page, the contiguous layout); page p owns positions [p*256, p*256+256) of every plane and an f16 V page is
// [kv head][256 dims][256 positions]. The preps and the expansions above take it as their last argument. attention_rows with a table
// reads the f16 pool in place, one pass (vpitch is ignored, n_ctx is the positions the table addresses): the bits of the contiguous
// kernel. A Q8 / Q4 pool goes through the expansions into windows, as before.

// router_topk / shexp_gate for T tokens: logits[T][n_expert] -> ids[T][n_used], wts[T][n_used]; sg[T].
void router_topk_rows(const float* logits, int n_expert, int n_used, int T, int32_t* ids, float* wts, cudaStream_t s);
void shexp_gate_rows(const float* gw, const float* post, float* sg, int n, int T, cudaStream_t s);

// x[t][j] += (sum_i dn[pos_of[t*n_used+i]][j] * wts[t][i]) + shared[t][j] * sg[t]
void moe_combine_rows(float* x, const float* dn, const int32_t* pos_of, const float* wts, const float* shared,
                      const float* sg, int n, int n_used, int T, cudaStream_t s);

// ---- hyper-connections and the PLE block (qwen4exp; src/step.cpp hc_mix / hc_combine / ple_hash_block) ----

// y[i] = silu(x[i] * scale): the HC mixer's low-rank bottleneck (scale = 1 / hc_count).
void silu_scale(const float* x, float* y, int n, float scale, cudaStream_t s);

// The HC read's gate mix: mixed[i] = inv_hc * sum_s xn[s][i] * sigmoid(gate[s][i]), n = n_embd, the streams
// contiguous (xn and gate are [hc][n]).
void hc_gate_mix(const float* xn, const float* gate, float* mixed, int hc, int n, float inv_hc, cudaStream_t s);

// The HC write: st[s][i] += out[i] * 2 * sigmoid(inject[s] * inv_hc) (inject un-scaled, as the gemv left it).
void hc_combine(float* st, const float* out, const float* inject, int hc, int n, float inv_hc, cudaStream_t s);

// st[s][i] = x[i] for every stream: the token's embedding copied into the residual stack.
void stack_fill(float* st, const float* x, int hc, int n, cudaStream_t s);

// The PLE block's per-stream gate (step.cpp ple_hash_block): kn_s = rmsnorm(key_s, gk_s), qn_s = rmsnorm(st_s, gq_s);
// s_s = (kn_s . qn_s, summed in double) / sqrt(n); gate_s = sigmoid(sign(s_s) * sqrt(max(|s_s|, 1e-6)));
// gated[s][i] = gate_s * val[i].
void ple_gate(const float* key, const float* st, const float* val, const float* gk, const float* gq, float* gated,
              int hc, int n, float eps, cudaStream_t s);

// The PLE block's dilated depthwise conv over time, then SiLU, then the history shift. taps is {kernel, channels}
// (taps[c * kern + k], the kernel contiguous) and hist [hist_rows][W] with hist_rows = (kern-1) * dilation, newest
// last: tap k reads (kern-1-k) * dilation positions back (0: the current row). Afterward hist holds the last
// hist_rows rows of nrm, oldest first.
void ple_conv_silu(const float* nrm, float* hist, const float* taps, float* conv, int W, int kern, int dil,
                   cudaStream_t s);

// The same four over a window of T rows, row t contiguous ([T][...]), each row's arithmetic identical to the
// single-row call above -- which is what makes a window's T rows equal T sequential steps.
void hc_gate_mix_rows(const float* xn, const float* gate, float* mixed, int hc, int n, int T, float inv_hc,
                      cudaStream_t s);
void hc_combine_rows(float* st, const float* out, const float* inject, int hc, int n, int T, float inv_hc,
                     cudaStream_t s);
void ple_gate_rows(const float* key, const float* st, const float* val, const float* gk, const float* gq, float* gated,
                   int hc, int n, int T, float eps, cudaStream_t s);
void ple_conv_silu_rows(const float* nrm, float* hist, const float* taps, float* conv, int W, int kern, int dil,
                        int T, cudaStream_t s);

// st[i] += a[i] + b[i]: the PLE block's write into the residual stack.
void add_into(float* st, const float* a, const float* b, int n, cudaStream_t s);

}  // namespace limbo::cuda::ops
