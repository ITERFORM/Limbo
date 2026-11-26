// Copyright (c) 2026 ITERFORM. All rights reserved.
#pragma once
// src/cuda/q4e_qsa.hpp — qwen4exp's sparse attention on the device (src/cuda/q4e_qsa.cu). Only .cu files include it.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace limbo::cuda::qsa {

constexpr float kEps = 1e-6f;   // the indexer norms' epsilon (the model's RMS eps)
constexpr int kPage = 256;      // positions per K/V page (dec's chunk)

// T rows at positions pos0..: kraw [T][128] = W_ik x, qraw [T][heads][128] = W_iq x. Each row's raw key goes to the ring
// ([16][128] f16, position & 15); the block it completes (position + 1 a multiple of kp) is pooled -- mean, rms *
// ikn, rope at the block's first position -- into pool[block]; its query is rms * iqn and roped at its position: qn.
bool index_rows(const float* kraw, const float* qraw, const float* ikn, const float* iqn, __half* ring, __half* pool,
                float* qn, const int* pos0, int T, int kp, int heads, int rope_dim, double base, cudaStream_t s);
// index_rows for a long run at host position pos0, in parallel (the prompt pass): the same pool, qn and ring.
bool index_rows_par(const float* kraw, const float* qraw, const float* ikn, const float* iqn, __half* ring, __half* pool,
                    float* qn, int pos0, int T, int kp, int heads, int rope_dim, double base, cudaStream_t s);
// The rows' block scores and their n_top best blocks, ascending: sel [T][n_top]. A row with no more than n_top
// complete blocks selects nothing (its attention is the plain causal one). scores [T][stride], stride >= nv_max.
bool select_rows(const float* qn, const __half* pool, float* scores, size_t stride, int32_t* sel, const int* pos0, int T,
                 int nv_max, int kp, int heads, int n_top, cudaStream_t s);
// The attention of rows that select: over the selected blocks' cells and the incomplete tail, the paged f16 cache
// (pt; vpitch for the unpaged layout), out = attention * sigmoid(gate) like dec::attention. kq non-null: the pool is Q8
// (kq/ks/vq/vs, kc/vc unused).
size_t attn_scratch_floats(int T, int n_head, int n_top, int kp);
bool attention_rows(const float* q, const __half* kc, const __half* vc, const float* qfull, float* out, float* scratch,
                    const int32_t* sel, const int* pos0, const int* pt, int vpitch, int T, int kp, int n_top, int n_head,
                    int n_kv, float scale, cudaStream_t s, const int8_t* kq = nullptr, const uint16_t* ks = nullptr,
                    const int8_t* vq = nullptr, const uint16_t* vs = nullptr);

}  // namespace limbo::cuda::qsa
