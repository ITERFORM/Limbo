// Copyright (c) 2026 ITERFORM. All rights reserved.
#pragma once
// src/cuda/ink_flow.hpp — the inkling2 depth mix and the loop controller (docs/inkling2.md §2/§5).
// Only .cu translation units include this header.

#include <cuda_runtime.h>

#include <cstddef>

namespace limbo::cuda::ink {

// The depth mix: `states` [n_s][n_embd] (the bank plus cur), `query` [q_count][n_embd] (1 or 4), out
// [q_count][n_embd]. Every state is RMS-normalised (eps 1e-6, no weight) before the scores; the score of
// state s for query q is dot(norm_s, query_q) / sqrt(n_embd); the softmax is over the states per query and
// the out is the weighted sum of the normed states. q_count <= 4, n_s <= 8, n_embd a multiple of 8.
bool mix(const float* states, const float* query, float* out, int n_s, int n_embd, int q_count, cudaStream_t s);

// The loop controller's bookends around its three GEMVs (F32, the caller's): gate_in writes
// [rms_norm(cur) | rms_norm(message)] (eps 1e-6, both, no weight) for ctrl_down's input; combine computes
// out = cur - tanh(forget)*cur + tanh(gain)*message from the two projected vectors.
bool control_gate(const float* cur, const float* message, float* gate_in, int n_embd, cudaStream_t s);
bool control_combine(const float* cur, const float* message, const float* gain, const float* forget,
                     float* out, int n_embd, cudaStream_t s);

// The same over `rows` rows (the prompt pass): row r's states / cur / message / out at r times their stride (in
// floats), gate_in packed [rows][2 * n_embd], gain / forget packed [rows][n_embd]. Each row's arithmetic is the
// single-row kernel's. The combine writes cur in place.
bool mix_rows(const float* states, size_t states_stride, const float* query, float* out, size_t out_stride, int n_s,
              int n_embd, int q_count, int rows, cudaStream_t s);
bool control_gate_rows(const float* cur, size_t cur_stride, const float* message, size_t msg_stride, float* gate_in,
                       int n_embd, int rows, cudaStream_t s);
bool control_combine_rows(float* cur, size_t cur_stride, const float* message, size_t msg_stride, const float* gain,
                          const float* forget, int n_embd, int rows, cudaStream_t s);

}  // namespace limbo::cuda::ink
