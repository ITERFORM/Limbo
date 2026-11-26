// Copyright (c) 2026 ITERFORM. All rights reserved.
#pragma once
// limbo/cuda/gemv.hpp — device matrix-vector products over weights in their GGUF block layout.
//
// y = W x with W resident in VRAM exactly as the file stores it (no repack in the PoC), x and y fp32,
// fp32 accumulation. One warp owns one output row; each lane decodes 8 consecutive elements of a
// 256-element super-block, so a warp consumes one super-block (or eight Q8_0 blocks) per step and the
// partial sums meet in a fixed shuffle tree: the result is deterministic and independent of the grid.
//
// The block layouts and the decode math are the ones in limbo/quants.hpp; the kernels are written from
// those definitions, and tests/cuda_gemv_test.cu holds them to src/quants.cpp on real rows.
//
// Only .cu translation units include this header.

#include <cuda_runtime.h>

#include <cstdint>
#include <string>

namespace limbo::cuda {

// Copies the codebooks (src/ggml_grids.cpp, via limbo/quants.hpp) into device memory. Idempotent; must
// run once before any IQ2_S / IQ3_S / IQ4_XS product.
bool init_tables(std::string* err);

// F32, BF16, Q8_0, Q2_K, Q3_K, Q6_K, IQ2_S, IQ3_S, IQ4_XS.
bool gemv_supported(uint32_t type);

// The plain / dual GEMV reads the weight bytes through shared memory (cp.async, bit-identical results) unless switched off.
void gemv_set_staged(bool on);
// Two rows per warp (default): the x loads are shared and two decodes run side by side; same results.
void gemv_set_rows2(bool on);
// Multi-column products over long rows (K >= 8192) take x through shared memory (default on; same results).
void gemv_set_xs(bool on);

// y[r] = W[r, :] . x for r in [0, n_out). W is n_out rows of n_in elements in `type`'s block layout;
// n_in must be a whole number of the format's blocks (a multiple of 256 for the super-block formats), and
// the tail past the last super-block (n_in % 256) a multiple of 8 elements. With accumulate, y[r] += the
// product instead of replacing it.
bool gemv(uint32_t type, const void* W, const float* x, float* y, int n_out, int n_in, bool accumulate,
          cudaStream_t s);

// gemv_cols for a matrix of few rows (n_out <= 64) over a long input, F32 / BF16 / F16 only: one block a row, the row's
// super-blocks split over its warps. Same products, another summation order than gemv(): close, not bit-identical.
// false when the shape or type is not one it takes (the caller falls back to gemv_cols).
bool gemv_narrow(uint32_t type, const void* W, const float* x, int x_stride, float* y, int y_stride, int n_out, int n_in, int C,
                 bool accumulate, cudaStream_t s);

// C columns at once (1..8; up to 16 through frag_cols): y[c * y_stride + r] = W[r, :] . x[c * x_stride ..] for c < C. Each weight is read and decoded
// once for all the columns, and each column's result is bit-identical to gemv()'s whatever C is (n_in % 256 is the row's
// partial super-block, as gemv() takes it). With accumulate, y += the product.
bool gemv_cols(uint32_t type, const void* W, const float* x, int x_stride, float* y, int y_stride, int n_out, int n_in, int C,
               bool accumulate, cudaStream_t s);

// The MoE form: a stack of experts, each n_out rows of n_in elements, expert_bytes apart. For slot e in
// [0, n_sel): y[e * n_out + r] = Wexp[ids[e]][r, :] . (x + e * x_stride). x_stride 0 shares one input
// across the slots (gate, up), n_in gives each its own (down). ids is device memory, so the router's
// choice never visits the host.
// x_group > 1: slot e reads x row e / x_group (rows x_stride apart) -- T tokens' experts in one launch, every slot's
// arithmetic the single-token form's.
// ids[e] < 0 skips slot e: its output is left as it was.
bool gemv_experts(uint32_t type, const void* W, uint64_t expert_bytes, const int32_t* ids, int n_sel,
                  const float* x, int x_stride, float* y, int n_out, int n_in, cudaStream_t s, int x_group = 1);

// Fused forms for the decode step (one launch where it was two or three; the arithmetic is unchanged).
// y0 = W0 x and y1 = W1 x: two matrices of one type and shape over the same input (an expert's gate and up).
bool gemv_dual(uint32_t type, const void* W0, const void* W1, const float* x, float* y0, float* y1, int n_out,
               int n_in, cudaStream_t s);
// The MoE form of gemv_dual: both stacks, the same ids, one shared input x.
bool gemv_experts_dual(uint32_t type, const void* W0, const void* W1, uint64_t expert_bytes, const int32_t* ids,
                       int n_sel, const float* x, float* y0, float* y1, int n_out, int n_in, cudaStream_t s);
// y = W h with h[i] = silu(g[i]) * u[i] formed in shared memory, once per block (n_in <= 512).
bool gemv_swiglu(uint32_t type, const void* W, const float* g, const float* u, float* y, int n_out, int n_in,
                 bool accumulate, cudaStream_t s);
// Decode fast paths for the routed experts (src/cuda/gemv_exp.cu). Each returns false, having launched nothing, when the
// type or shape is not one it handles, and the caller then uses the generic products above.
// IQ2_S gate/up with 2048 columns, SwiGLU fused: h[e * n_out + r] = silu(Wg[ids[e]][r] . x) * (Wu[ids[e]][r] . x).
// With slots_per_row > 0 the slots are the rows of a verify window (row = slot / slots_per_row, ids and h indexed by slot,
// x row r at x + r * x_stride); with 0 every slot reads the one row x.
bool gemv_experts_gate_up(uint32_t type, const void* Wg, const void* Wu, uint64_t expert_bytes, const int32_t* ids, int n_sel,
                          const float* x, float* h, int n_out, int n_in, cudaStream_t s, int slots_per_row = 0, int x_stride = 0);
// IQ3_S down with 512 columns: y[e * n_out + r] = Wd[ids[e]][r] . h[e * n_in .. e * n_in + n_in).
bool gemv_experts_down_fast(uint32_t type, const void* W, uint64_t expert_bytes, const int32_t* ids, int n_sel, const float* h,
                            float* y, int n_out, int n_in, cudaStream_t s);
// Q6_K products (src/cuda/gemv_q6k.cu), used by gemv / gemv_cols / gemv_dual / gemv_swiglu for Q6_K weights whose n_in is a
// multiple of 512: C columns, one or two matrices (n_z), the input x or silu(x) * x2.
bool q6k_launch(const void* W, const void* W2, const float* x, const float* x2, int x_stride, float* y, float* y2, int y_stride, int n_out,
                int n_in, int C, bool accumulate, int n_z, cudaStream_t s);
// The codebooks of those kernels (called by init_tables).
bool exp_init(std::string* err);

// out[0..n_in) = row *token of the embedding table W (n_in elements per row, `type` layout).
bool embed_row(uint32_t type, const void* W, const int32_t* token, float* out, int n_in, cudaStream_t s);

// out[t * n_in ..] = row tokens[t] of the embedding table, for t < T (tokens in device memory).
bool embed_rows(uint32_t type, const void* W, const int32_t* tokens, int T, float* out, int n_in, cudaStream_t s);

}  // namespace limbo::cuda
