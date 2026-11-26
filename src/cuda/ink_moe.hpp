// Copyright (c) 2026 ITERFORM. All rights reserved.
#pragma once
// src/cuda/ink_moe.hpp — the inkling2 routed experts on the GPU (docs/inkling2.md §4). The arithmetic is
// Ref::ink_moe's (src/step.cpp) exactly: the router's sigmoid scores plus the selection bias pick the top
// n_used experts (ties to the lower id), the chosen sigmoid probabilities are renormalised with +1e-20 and
// scaled by route_scale, and each expert runs its fused gate/up rows under the per-expert dequantisation
// scale through SiTU before the down rows (their own per-expert scale) accumulate into the output weighted
// by the routing weights:
//
//   scores  = sigmoid(gate_inp . x) + exp_bias;        top n_used by score, ties to the lower id
//   w[e]    = sigmoid(logit_e) / (sum_chosen + 1e-20) * route_scale
//   g, u    = (Wg[ids[e]] . x) * scale[ids[e]]         the fused tensor: gate rows first, then up
//   h       = (beta * tanh(g/beta) * sigmoid(g)) * (lb * tanh(u/lb))        (SiTU; beta 4, lb 25)
//   out    += ((Wd[ids[e]] . h_e) * scale[ids[e]]) * w[e]
//
// The products are gemv_experts (limbo/cuda/gemv.hpp) over the weights in their GGUF block layout; the
// per-expert scales are F32[n_expert] indexed by expert id and may be null (a file whose layers carry no
// .scale tensors dequantises to unit scale). Every reduction is a fixed-order loop or a fixed shuffle tree,
// so the result does not depend on the launch geometry. Only .cu translation units include this header.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace limbo::cuda::ink {

// Device scratch the two product stages share, in floats: max(2 * n_sel * n_ff, n_sel * n_embd).
size_t moe_scratch_floats(int n_sel, int n_ff, int n_embd);

// The router and the selection. `router` is n_expert rows of n_embd elements of `router_type`; `bias` is the
// file's exp_probs_b (F32[n_expert], device memory). Writes the chosen expert ids by descending score to
// `ids` and their weights to `w` (n_used entries each). `logits` is n_expert floats of scratch (the router's
// raw logits, kept for the caller's tracing). n_expert <= 1024. False when the product refuses.
bool moe_route(const float* x, const void* router, uint32_t router_type, const float* bias, int n_expert,
               int n_used, int n_embd, float route_scale, int32_t* ids, float* w, float* logits, cudaStream_t s);

// The fused gate/up rows of the selected experts, per-expert scale on both halves, then SiTU into `h`
// [n_sel][n_ff]. `Wg` is one tensor of 2 * n_ff rows of n_embd per expert (gate rows first), expert_bytes
// apart; `raw` is moe_scratch_floats() floats of scratch. `scale` (F32[n_expert] by expert id) may be null.
bool moe_gate_up_situ(const void* Wg, uint32_t type, uint64_t expert_bytes, const int32_t* ids, int n_sel,
                      const float* x, const float* scale, int n_ff, int n_embd, float beta, float lb,
                      float* h, float* raw, cudaStream_t s);

// The down rows of the selected experts under their per-expert scale, weighted by `w` and summed into `out`
// [n_embd], the experts added in selection order. `Wd` is n_embd rows of n_ff per expert, expert_bytes
// apart; `raw` is moe_scratch_floats() floats of scratch. `scale` may be null.
bool moe_down_accum(const void* Wd, uint32_t type, uint64_t expert_bytes, const int32_t* ids, int n_sel,
                    const float* h, const float* scale, const float* w, int n_ff, int n_embd, float* out,
                    float* raw, cudaStream_t s);

// moe_gate_up_situ / moe_down_accum over `rows` tokens at once with every row's arithmetic theirs (bit-identical):
// ids / w [rows][n_sel], x [rows][n_embd], h [rows][n_sel][n_ff], out [rows][n_embd], raw rows times the scratch.
bool moe_gate_up_situ_rows(const void* Wg, uint32_t type, uint64_t expert_bytes, const int32_t* ids, int n_sel,
                           const float* x, int rows, const float* scale, int n_ff, int n_embd, float beta, float lb,
                           float* h, float* raw, cudaStream_t s);
bool moe_down_accum_rows(const void* Wd, uint32_t type, uint64_t expert_bytes, const int32_t* ids, int n_sel,
                         int rows, const float* h, const float* scale, const float* w, int n_ff, int n_embd, float* out,
                         float* raw, cudaStream_t s);

// The prompt pass over `rows` tokens (src/cuda/ink_gpu.cu prefill_rows): the router's selection per row
// (logits [rows][n_expert] in, ids / w [rows][n_used] out); SiTU over the grouped pairs of limbo::cuda::moe_build
// (raw [pair][2 * n_ff], the pair's expert written to pe); the per-row weighted sum of the grouped down products
// in selection order (dn [pair][n_embd], out [rows][n_embd]).
bool route_rows(const float* logits, const float* bias, int n_expert, int n_used, float route_scale, int32_t* ids,
                float* w, int rows, cudaStream_t s);
bool situ_pairs(const float* raw, const int32_t* ids, const int32_t* pos_of, int32_t* pe, const float* scale,
                int n_pairs, int n_ff, float beta, float lb, float* h, cudaStream_t s);
bool combine_rows(const float* dn, const int32_t* pos_of, const int32_t* ids, const float* scale, const float* w,
                  int n_used, int n_embd, int rows, float* out, cudaStream_t s);

}  // namespace limbo::cuda::ink
