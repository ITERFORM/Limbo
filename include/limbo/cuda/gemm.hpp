// Copyright (c) 2026 ITERFORM. All rights reserved.
#pragma once
// limbo/cuda/gemm.hpp — batched products for prompt prefill: Y[t][n] = sum_k W[n][k] x[t][k] over T rows at once.
//
// The weights stay in their GGUF block layout. Every format the rung-1 forward pass uses (Q8_0, Q6_K, IQ2_S,
// IQ3_S, IQ4_XS) is "integer code x group scale": a block decodes to small integers (|q| <= 127) plus one
// fp32 scale per 16 or 32 columns, so a weight tile is decoded once into shared memory as int8 and applied to
// every token row of the tile on the int8 tensor cores (mma.m16n8k16, int32 accumulate: exact).
//
// The activations are the only rounded quantity. Each row is cut into blocks of 32, a block's amplitude
// becomes an integer of R levels (R = 127, 16192 or 2,000,000) and that integer is split into `nt` balanced
// base-128 digits, one int8 plane each: nt = 1 is a plain Q8 activation (llama.cpp's own arithmetic),
// nt = 2 resolves the block amplitude to 1 part in 16192, nt = 3 to 1 in 2e6 (fp32-like). The planes are
// multiplied separately by the int8 tiles and recombined in fp32 per group with the weight and activation
// scales, so the products are exact in the integer domain and the only differences from the fp32 GEMV are
// the activation rounding and the summation order.
//
// Only .cu translation units include this header.

#include <cuda_runtime.h>

#include <cstdint>
#include <string>

namespace limbo::cuda {

// Device codebooks for the int8 path (IQ2_S, IQ3_S, IQ4_XS grids). Idempotent.
bool gemm_init(std::string* err);

// F16 (dense only), Q8_0, Q6_K, IQ2_S, IQ3_S, IQ4_XS and, from gemm_ext_*.cu, IQ3_XXS, IQ2_XXS, IQ2_XS, IQ1_S, IQ1_M, IQ4_NL, Q2_0, Q4_K, Q5_K, Q2_K, Q3_K (not BF16 / F32).
bool gemm_supported(uint32_t type);

// Quantises `rows` activation rows of K floats (K % 128 == 0) into nt int8 planes q[nt][rows][K] and the
// per-(row, 32-block) unit sc[rows][K/32]; plane 0 is the most significant digit.
bool act_quant(const float* x, int rows, int K, int nt, int8_t* q, float* sc, cudaStream_t s);
// The same, and sums[rows][K/16] = the sum of the dequantised activations (the integers times the block scale) of every
// 16 columns: the minimum term of the K-quants (Q4_K, Q5_K, Q2_K) multiplies it.
bool act_quant_sums(const float* x, int rows, int K, int nt, int8_t* q, float* sc, float* sums, cudaStream_t s);
// Whether products of this format need the sums (gemm / moe_gate_up / moe_down then refuse a null `xsum`).
bool gemm_needs_sums(uint32_t type);

// rmsnorm of `rows` rows of n floats (n % 256 == 0) and act_quant of its output in one pass: y = x * inv(rms(x)) * w
// exactly as ops::rmsnorm_rows computes it (sum of squares in double), and q / sc as act_quant(y, rows, n, nt) writes
// them. y is kept: the F32 products read it.
bool rmsnorm_quant(const float* x, const float* w, float* y, int rows, int n, float eps, int nt, int8_t* q, float* sc,
                   cudaStream_t s);

// y[t][n] = (accumulate ? y[t][n] : 0) + sum_k W[n][k] x[t][k] for t < T, n < n_out. x comes from act_quant
// (same T, n_in = K, nt); y is [T][n_out]. n_in % 128 == 0.
// Decode products of IQ3_S/IQ2_S/Q6_K/Q4_K/Q5_K on a frag_reserve'd stream go through the fragment kernel (1..16 rows, one cost);
// false: not taken, the caller runs the FP32 kernel.
bool frag_reserve(cudaStream_t s, std::string* err);
void frag_release(cudaStream_t s);
bool frag_takes(uint32_t type, int n_in, cudaStream_t s);
bool frag_cols(uint32_t type, const void* W, const float* x, int x_stride, float* y, int y_stride, int n_out, int n_in, int C,
               bool accumulate, cudaStream_t s);
bool frag_swiglu(uint32_t type, const void* W, const float* g, const float* u, float* y, int n_out, int n_in, bool accumulate,
                 cudaStream_t s);
bool gemm(uint32_t type, const void* W, const int8_t* xq, const float* xs, int nt, int T, float* y, int n_out,
          int n_in, bool accumulate, cudaStream_t s, const float* xsum = nullptr);

// The expert lists of one MoE layer. All arrays are device memory. An expert with up to kMoeBmBig pairs is cut into
// tiles of kMoeBmSmall (the usual case at a few hundred tokens); a busier one into tiles of kMoeBmBig, so its
// weights are decoded once per 32 pairs instead of once per 16 (the big tile's shared memory allows one block
// per SM, which only pays when an expert has that many pairs).
constexpr int kMoeBmSmall = 16;
constexpr int kMoeBmBig = 32;
struct MoeLists {
    int32_t* cnt = nullptr;        // [n_expert] pairs per expert
    int32_t* start = nullptr;      // [n_expert + 1] exclusive prefix sum of cnt
    int32_t* list_tok = nullptr;   // [T * n_used] token of each pair, grouped by expert
    int32_t* pos_of = nullptr;     // [T * n_used] position in the grouped order of (token, choice), at t * n_used + i
    int4* tiles_small = nullptr;   // [moe_max_tiles_small(n_expert)] (expert, first pair, pairs, 0)
    int4* tiles_big = nullptr;     // [moe_max_tiles_big(T * n_used, n_expert)]
    int32_t* n_tiles = nullptr;    // [2] tiles in each class
    // optional: expert -> device address of its blob; the W arguments are then byte offsets into the blob
    const uint8_t* const* eptr = nullptr;
    // rows of a busy expert's tiles: kMoeBmBig, or 64 for a long prompt (each decoded weight feeds twice the rows;
    // the IQ2_S / IQ3_S / Q6_K fragment kernel and the ext formats take it, moe_* refuse it otherwise)
    int big_bm = kMoeBmBig;
};
inline int moe_max_tiles_small(int n_expert) { return 2 * n_expert; }
inline int moe_max_tiles_big(int pairs, int n_expert) { return pairs / (kMoeBmBig + 1) + 1 + n_expert; }

// Whether moe_gate_up / moe_down take MoeLists::big_bm = 64 for this stack (K the product's input width).
bool moe_bm64_ok(uint32_t type, int K, uint64_t expert_bytes);

// Groups the (token, choice) pairs of ids[T][n_used] by expert and cuts them into tiles.
bool moe_build(const int32_t* ids, int T, int n_expert, int n_used, const MoeLists& L, cudaStream_t s);

// h[pair][n] = silu(gate) * up for every pair, n < n_ff: Wg and Wu are expert stacks (n_ff rows of K weights
// per expert, expert_bytes apart, the same type); the input of a pair is row list_tok[pair] of the quantised
// activations (T rows of K).
bool moe_gate_up(uint32_t type, const void* Wg, const void* Wu, uint64_t expert_bytes, const int8_t* xq,
                 const float* xs, int nt, int T, int n_pairs, const MoeLists& L, int n_expert, float* h, int n_ff,
                 int K, cudaStream_t s, const float* xsum = nullptr);

// dn[pair][n] = sum_k Wd[expert][n][k] hq[pair][k], n < n_out; hq is act_quant of h (n_pairs rows of K).
bool moe_down(uint32_t type, const void* Wd, uint64_t expert_bytes, const int8_t* hq, const float* hs, int nt,
              int n_pairs, const MoeLists& L, int n_expert, float* dn, int n_out, int K, cudaStream_t s,
              const float* hsum = nullptr);

// moe_down's product with the token rows of the lists as its input: y[pair][n] = sum_k W[expert][n][k] x[list_tok[pair]][k]
// for n < n_out, x the act_quant of T token rows. Inkling-2's fused gate/up (2 * n_ff rows per expert, its own
// activation applied by the caller) is this product with n_out = 2 * n_ff.
bool moe_rows(uint32_t type, const void* W, uint64_t expert_bytes, const int8_t* xq, const float* xs, int nt, int T,
              int n_pairs, const MoeLists& L, int n_expert, float* y, int n_out, int K, cudaStream_t s,
              const float* xsum = nullptr);

// F16 weights (a dense product only, no expert stacks): the same activation planes through the fp16 tensor cores, see
// gemm_f16.cu. gemm() routes an F16 `type` here.
bool gemm_f16(const void* W, const int8_t* xq, const float* xs, int nt, int T, float* y, int n_out, int n_in, bool accumulate,
              cudaStream_t s);

// F32 weights, the router's and the delta net's alpha/beta. Same arithmetic order as gemv() (lane L owns
// elements 8L..8L+7 of every 256 columns, a fixed shuffle tree), so a row of y equals the GEMV of that row.
// y[t][n] for n < n_out; x is [T][n_in] fp32, n_in % 256 == 0.
bool gemm_f32(const float* W, const float* x, float* y, int T, int n_out, int n_in, cudaStream_t s);

}  // namespace limbo::cuda
