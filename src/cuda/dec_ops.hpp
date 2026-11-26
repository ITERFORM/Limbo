// Copyright (c) 2026 ITERFORM. All rights reserved.
#pragma once
// src/cuda/dec_ops.hpp — the decode kernels that take T query rows (T = 1 is the plain decode step, T = 2..8 a
// verify window): the same code runs both, so a row's result does not depend on how many rows share the launch.
// Owner: the decode agent. Only .cu translation units include this header.
//
// Row layout of every activation buffer: row t starts at t * (the single-token size), the same buffers as the
// T = 1 decode. The position of row t is *pos0 + t, read from device memory so one captured graph replays at
// every position.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace limbo::cuda::dec {

constexpr int kMaxRows = 8;       // the widest verify window
constexpr int kAttnChunk = 256;   // positions per flash-decode chunk (a chunk boundary is a fixed absolute position)

// Paged K/V (docs/paged-kv.md): every attention entry below takes `pt`, the page table of the sequence (int32 per 256-position chunk,
// device memory, read by the kernels: a captured graph replays against whatever it holds). Chunk c is page pt[c] of the pool; page p
// owns positions [p*256, p*256+256) of every plane, and an f16 V page is [kv head][256 dims][256 positions] (vpitch is then ignored).
// pt null: chunk c is page c and V rows are vpitch apart, the contiguous layout (the MTP block's own cache).

// Full-attention preparation for T rows, one launch: per-head RMS norm, NEOX partial RoPE and the f16 KV write of
// position *pos0 + t. qfull is [T][n_head][q | gate], kbuf and vbuf [T][n_head_kv * hd], q_out [T][n_head * hd].
void attn_prep(const float* qfull, const float* kbuf, const float* vbuf, const float* qn, const float* kn, float* q_out,
               __half* kcache, __half* vcache, const int* pos0, int T, int vpitch, double eps, double rope_base,
               int rope_dim, int n_head, int n_head_kv, int hd, cudaStream_t s, const int* pt = nullptr);

// Scratch of the split flash-decode: floats needed for up to max_rows rows at max_ctx positions.
size_t attn_scratch_floats(int max_ctx, int max_rows, int n_head);

// Split flash-decode for T rows: row t attends positions 0..*pos0 + t (causal inside a window; the keys of the
// window must already be in the cache). A (KV head, 256-position chunk, row) block scores its chunk for all the
// query heads of the group (each K row is read once for the group's 8 heads) and writes (max, sum, numerator);
// a combine kernel merges the chunks in position order and applies the sigmoid gate (the second half of qfull).
// out is [T][n_head * hd]. `scratch` is attn_scratch_floats() floats; the grid is sized for max_ctx positions so
// the same graph runs at every length. hd must be 256 and n_head / n_head_kv at most 8 (the group is padded to 8 slots).
bool attention(const float* q, const __half* kcache, const __half* vcache, const float* qfull, float* out, float* scratch,
                const int* pos0, int T, int max_ctx, int vpitch, int n_head, int n_head_kv, int hd, float scale,
                cudaStream_t s, const int* pt = nullptr);

// The same two stages over a Q8 cache (include/limbo/cuda/kv.hpp): K and V rows [pos][kv head][256] int8 with their scale planes
// [pos][kv head][8] half. attn_prep_q8 quantises the K and V of positions *pos0 + t; attention_q8 reads them (the combine is shared).
void attn_prep_q8(const float* qfull, const float* kbuf, const float* vbuf, const float* qn, const float* kn, float* q_out, int8_t* kq,
                  uint16_t* ks, int8_t* vq, uint16_t* vs, const int* pos0, int T, double eps, double rope_base, int rope_dim, int n_head,
                  int n_head_kv, int hd, cudaStream_t s, const int* pt = nullptr);
bool attention_q8(const float* q, const int8_t* kq, const uint16_t* ks, const int8_t* vq, const uint16_t* vs, const float* qfull, float* out,
                  float* scratch, const int* pos0, int T, int max_ctx, int n_head, int n_head_kv, int hd, float scale, cudaStream_t s,
                  const int* pt = nullptr);

// The same over a Q4 cache: K and V rows (both rotated) [pos][kv head][128] bytes of nibbles with Q8's scale planes. attn_prep_q4
// writes the rotated query (q_out) beside the quantised K and V; attention_q4 scores it against the rotated keys and rotates the
// merged output back.
void attn_prep_q4(const float* qfull, const float* kbuf, const float* vbuf, const float* qn, const float* kn, float* q_out, int8_t* kq,
                  uint16_t* ks, int8_t* vq, uint16_t* vs, const int* pos0, int T, double eps, double rope_base, int rope_dim, int n_head,
                  int n_head_kv, int hd, cudaStream_t s, const int* pt = nullptr);
bool attention_q4(const float* q, const int8_t* kq, const uint16_t* ks, const int8_t* vq, const uint16_t* vs, const float* qfull, float* out,
                  float* scratch, const int* pos0, int T, int max_ctx, int n_head, int n_head_kv, int hd, float scale, cudaStream_t s,
                  const int* pt = nullptr);

// ---- the small kernels of the decode critical path (src/cuda/dec_small.cu), T rows each --------------------------------

// y[t] = x[t] * inv(rms(x[t])) * w, sum of squares in double; n <= 16384, input rows contiguous, output rows y_stride apart
// (0: n). segs > 1: each row holds segs segments of n, normed apart with their own weights (w + k * n) in one launch.
bool rmsnorm(const float* x, const float* w, float* y, int n, float eps, int T, cudaStream_t s, int y_stride = 0,
             int x_stride = 0, int segs = 1);

// F32 weights, n_in <= 2048: y0[c * y_stride + r] = W0[r, :] . x[c * x_stride ..] (and y1 from W1 when n_z = 2), c < C.
bool f32_cols(const float* W0, const float* W1, const float* x, int x_stride, float* y0, float* y1, int y_stride, int n_out, int n_in, int C,
              int n_z, cudaStream_t s);

// Softmax over the router logits of each of T rows (n <= 256) and the top k by descending probability, ties to the lower
// expert id: ids[t][k], wts[t][k] = p / max(sum of the chosen p, 6.1035e-5).
bool topk(const float* logits, int n, int k, int T, int32_t* ids, float* wts, cudaStream_t s);

}  // namespace limbo::cuda::dec
