// Copyright (c) 2026 ITERFORM. All rights reserved.
// src/cuda/ink_moe.cu — the inkling2 routed experts (docs/inkling2.md §4), the arithmetic of Ref::ink_moe
// (src/step.cpp). The products are limbo/cuda/gemv.hpp's grouped expert form over the weights in their GGUF
// block layout; the folds around them are small kernels whose reductions are all fixed-order: the router's
// selection ranks every expert against every other (the reference's comparator, ties to the lower id), the
// renormalisation sums the chosen probabilities in selection order, and the down accumulation adds the
// experts in selection order — so a result never depends on how the work is spread over the launch.
#include <cmath>

#include "ink_moe.hpp"
#include "limbo/cuda/gemv.hpp"

namespace limbo::cuda::ink {

namespace {

__device__ __forceinline__ float sigmoid_d(float x) { return 1.0f / (1.0f + expf(-x)); }

// The router: scores = sigmoid(logits) + bias, the top n_used by the reference's comparator (descending
// score, ties to the lower id), then w = prob / (sum of the chosen probs + 1e-20) * route_scale summed in
// selection order. One block: every expert's score is computed once into shared memory, then every expert
// ranks itself against all of them.
__global__ void route_kernel(const float* __restrict__ logits, const float* __restrict__ bias, int n_expert,
                             int n_used, float route_scale, int32_t* __restrict__ ids, float* __restrict__ w) {
    logits += size_t(blockIdx.x) * n_expert;   // one block per row
    ids += size_t(blockIdx.x) * n_used;
    w += size_t(blockIdx.x) * n_used;
    __shared__ float probs[1024];
    for (int e = threadIdx.x; e < n_expert; e += blockDim.x) probs[e] = sigmoid_d(logits[e]);
    __syncthreads();
    for (int e = threadIdx.x; e < n_expert; e += blockDim.x) {
        const float se = probs[e] + bias[e];
        int rank = 0;
        for (int j = 0; j < n_expert; ++j) {
            const float sj = probs[j] + bias[j];
            if (sj > se || (sj == se && j < e)) ++rank;
        }
        if (rank < n_used) ids[rank] = e;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float wsum = 0.0f;
        for (int i = 0; i < n_used; ++i) wsum += probs[ids[i]];
        wsum += 1e-20f;   // the reference's expert_weights_norm_eps
        for (int i = 0; i < n_used; ++i) w[i] = probs[ids[i]] / wsum * route_scale;
    }
}

// h[e][j] = situ(g, u) with g and u the expert's scaled gate/up products (the fused rows: gate first).
__global__ void situ_kernel(const float* __restrict__ raw, const int32_t* __restrict__ ids,
                            const float* __restrict__ scale, int n_sel, int n_ff, float beta, float lb,
                            float* __restrict__ h) {
    const size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (i >= size_t(n_sel) * size_t(n_ff)) return;
    const int e = int(i / size_t(n_ff)), j = int(i % size_t(n_ff));
    const float s = scale ? scale[ids[e]] : 1.0f;
    const float g = raw[size_t(e) * 2 * size_t(n_ff) + size_t(j)] * s;
    const float u = raw[size_t(e) * 2 * size_t(n_ff) + size_t(n_ff) + size_t(j)] * s;
    h[i] = (beta * tanhf(g / beta) * sigmoid_d(g)) * (lb * tanhf(u / lb));
}

// out[j] = sum_e (raw[e][j] * scale[ids[e]]) * w[e], the experts added in selection order. blockIdx.y is the row of
// the batched form (raw, ids, w and out of row t at t times their per-row sizes).
__global__ void accum_kernel(const float* __restrict__ raw, const int32_t* __restrict__ ids,
                             const float* __restrict__ scale, const float* __restrict__ w, int n_sel,
                             int n_embd, float* __restrict__ out) {
    const int j = int(blockIdx.x * blockDim.x + threadIdx.x);
    raw += size_t(blockIdx.y) * n_sel * n_embd;
    ids += size_t(blockIdx.y) * n_sel;
    w += size_t(blockIdx.y) * n_sel;
    out += size_t(blockIdx.y) * n_embd;
    if (j >= n_embd) return;
    float acc = 0.0f;
    for (int e = 0; e < n_sel; ++e) {
        const float s = scale ? scale[ids[e]] : 1.0f;
        acc += (raw[size_t(e) * n_embd + j] * s) * w[e];
    }
    out[j] = acc;
}

// pe[pos_of[i]] = ids[i]: the expert of every pair in the grouped order.
__global__ void pair_map_kernel(const int32_t* __restrict__ ids, const int32_t* __restrict__ pos_of, int n,
                                int32_t* __restrict__ pe) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) pe[pos_of[i]] = ids[i];
}

// situ_kernel over the grouped pairs: raw [pair][2 * n_ff] (gate first), the pair's expert from pe.
__global__ void situ_pairs_kernel(const float* __restrict__ raw, const int32_t* __restrict__ pe,
                                  const float* __restrict__ scale, int n_pairs, int n_ff, float beta, float lb,
                                  float* __restrict__ h) {
    const size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (i >= size_t(n_pairs) * size_t(n_ff)) return;
    const int p = int(i / size_t(n_ff)), j = int(i % size_t(n_ff));
    const float s = scale ? scale[pe[p]] : 1.0f;
    const float g = raw[size_t(p) * 2 * size_t(n_ff) + size_t(j)] * s;
    const float u = raw[size_t(p) * 2 * size_t(n_ff) + size_t(n_ff) + size_t(j)] * s;
    h[i] = (beta * tanhf(g / beta) * sigmoid_d(g)) * (lb * tanhf(u / lb));
}

// accum_kernel per row over the grouped down products: out[t][j] = sum_i (dn[pos_of[t, i]][j] * scale) * w[t, i],
// the experts added in selection order.
__global__ void combine_rows_kernel(const float* __restrict__ dn, const int32_t* __restrict__ pos_of,
                                    const int32_t* __restrict__ ids, const float* __restrict__ scale,
                                    const float* __restrict__ w, int n_used, int n_embd, float* __restrict__ out) {
    const int j = int(blockIdx.x * blockDim.x + threadIdx.x), t = blockIdx.y;
    if (j >= n_embd) return;
    float acc = 0.0f;
    for (int e = 0; e < n_used; ++e) {
        const int k = t * n_used + e;
        const float s = scale ? scale[ids[k]] : 1.0f;
        acc += (dn[size_t(pos_of[k]) * n_embd + j] * s) * w[k];
    }
    out[size_t(t) * n_embd + j] = acc;
}

}  // namespace

bool route_rows(const float* logits, const float* bias, int n_expert, int n_used, float route_scale, int32_t* ids,
                float* w, int rows, cudaStream_t s) {
    if (rows < 1 || n_expert <= 0 || n_expert > 1024 || n_used <= 0 || n_used > n_expert) return false;
    route_kernel<<<rows, 256, 0, s>>>(logits, bias, n_expert, n_used, route_scale, ids, w);
    return cudaGetLastError() == cudaSuccess;
}

bool situ_pairs(const float* raw, const int32_t* ids, const int32_t* pos_of, int32_t* pe, const float* scale,
                int n_pairs, int n_ff, float beta, float lb, float* h, cudaStream_t s) {
    if (n_pairs < 1 || n_ff < 1) return false;
    pair_map_kernel<<<(n_pairs + 255) / 256, 256, 0, s>>>(ids, pos_of, n_pairs, pe);
    const size_t n = size_t(n_pairs) * size_t(n_ff);
    situ_pairs_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, s>>>(raw, pe, scale, n_pairs, n_ff, beta, lb, h);
    return cudaGetLastError() == cudaSuccess;
}

bool combine_rows(const float* dn, const int32_t* pos_of, const int32_t* ids, const float* scale, const float* w,
                  int n_used, int n_embd, int rows, float* out, cudaStream_t s) {
    if (rows < 1) return false;
    combine_rows_kernel<<<dim3((n_embd + 255) / 256, rows), 256, 0, s>>>(dn, pos_of, ids, scale, w, n_used, n_embd, out);
    return cudaGetLastError() == cudaSuccess;
}

size_t moe_scratch_floats(int n_sel, int n_ff, int n_embd) {
    const size_t gu = size_t(2) * size_t(n_sel) * size_t(n_ff), dn = size_t(n_sel) * size_t(n_embd);
    return gu > dn ? gu : dn;
}

bool moe_route(const float* x, const void* router, uint32_t router_type, const float* bias, int n_expert,
               int n_used, int n_embd, float route_scale, int32_t* ids, float* w, float* logits, cudaStream_t s) {
    if (n_expert <= 0 || n_expert > 1024 || n_used <= 0 || n_used > n_expert || n_embd <= 0) return false;
    if (!gemv(router_type, router, x, logits, n_expert, n_embd, false, s)) return false;
    route_kernel<<<1, 256, 0, s>>>(logits, bias, n_expert, n_used, route_scale, ids, w);
    return cudaGetLastError() == cudaSuccess;
}

bool moe_gate_up_situ(const void* Wg, uint32_t type, uint64_t expert_bytes, const int32_t* ids, int n_sel,
                      const float* x, const float* scale, int n_ff, int n_embd, float beta, float lb,
                      float* h, float* raw, cudaStream_t s) {
    if (n_sel <= 0 || n_ff <= 0 || n_embd <= 0) return false;
    // One grouped product over the fused tensor (2 * n_ff rows per expert: gate first, up after), then SiTU.
    if (!gemv_experts(type, Wg, expert_bytes, ids, n_sel, x, 0, raw, 2 * n_ff, n_embd, s)) return false;
    const size_t n = size_t(n_sel) * size_t(n_ff);
    situ_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, s>>>(raw, ids, scale, n_sel, n_ff, beta, lb, h);
    return cudaGetLastError() == cudaSuccess;
}

bool moe_gate_up_situ_rows(const void* Wg, uint32_t type, uint64_t expert_bytes, const int32_t* ids, int n_sel,
                           const float* x, int rows, const float* scale, int n_ff, int n_embd, float beta, float lb,
                           float* h, float* raw, cudaStream_t s) {
    if (n_sel <= 0 || rows <= 0 || n_ff <= 0 || n_embd <= 0) return false;
    const int slots = n_sel * rows;
    if (!gemv_experts(type, Wg, expert_bytes, ids, slots, x, n_embd, raw, 2 * n_ff, n_embd, s, n_sel)) return false;
    const size_t n = size_t(slots) * size_t(n_ff);
    situ_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, s>>>(raw, ids, scale, slots, n_ff, beta, lb, h);
    return cudaGetLastError() == cudaSuccess;
}

bool moe_down_accum_rows(const void* Wd, uint32_t type, uint64_t expert_bytes, const int32_t* ids, int n_sel,
                         int rows, const float* h, const float* scale, const float* w, int n_ff, int n_embd, float* out,
                         float* raw, cudaStream_t s) {
    if (n_sel <= 0 || rows <= 0 || n_ff <= 0 || n_embd <= 0) return false;
    if (!gemv_experts(type, Wd, expert_bytes, ids, n_sel * rows, h, n_ff, raw, n_embd, n_ff, s)) return false;
    accum_kernel<<<dim3(static_cast<unsigned>((n_embd + 255) / 256), static_cast<unsigned>(rows)), 256, 0, s>>>(
        raw, ids, scale, w, n_sel, n_embd, out);
    return cudaGetLastError() == cudaSuccess;
}

bool moe_down_accum(const void* Wd, uint32_t type, uint64_t expert_bytes, const int32_t* ids, int n_sel,
                    const float* h, const float* scale, const float* w, int n_ff, int n_embd, float* out,
                    float* raw, cudaStream_t s) {
    if (n_sel <= 0 || n_ff <= 0 || n_embd <= 0) return false;
    // Each slot carries its own input row (h + e * n_ff), so one grouped product covers every expert.
    if (!gemv_experts(type, Wd, expert_bytes, ids, n_sel, h, n_ff, raw, n_embd, n_ff, s)) return false;
    accum_kernel<<<static_cast<unsigned>((n_embd + 255) / 256), 256, 0, s>>>(raw, ids, scale, w, n_sel, n_embd, out);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace limbo::cuda::ink
