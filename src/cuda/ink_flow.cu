// Copyright (c) 2026 ITERFORM. All rights reserved.
// src/cuda/ink_flow.cu — the inkling2 depth mix and the loop controller (docs/inkling2.md §2/§5).
// The block's 8 warps split the n_embd dims; every reduction is a fixed shuffle tree or a fixed-order shared
// pass, so a result does not depend on the launch geometry.
#include <cmath>

#include "ink_flow.hpp"

namespace limbo::cuda::ink {

namespace {

__device__ __forceinline__ float warp_sum_d(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// states [n_s][n_embd], query [q_count][n_embd], out [q_count][n_embd]. The arithmetic of each quantity is
// one fixed sequence: a warp's strided partial over its dims, the shuffle tree, then the 8 warp partials added
// in warp order by one thread. The states and queries only differ in being worked side by side (one barrier
// per phase instead of two per state and query); the floats are the same.
__global__ void ink_mix_k(const float* states, const float* query, float* out, int n_s, int n_embd,
                          int q_count, size_t states_stride, size_t out_stride) {
    states += size_t(blockIdx.x) * states_stride;   // one block per row
    out += size_t(blockIdx.x) * out_stride;
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int per = (n_embd + 7) / 8, d0 = w * per, d1 = min(d0 + per, n_embd);
    __shared__ float part[8 * 4 * 8];   // [state][query][warp] partials (the norms use query slot 0)
    __shared__ float inv8[8], sc[32], ps[9 * 4];

    // the per-state sums of squares
    for (int s = 0; s < n_s; ++s) {
        const float* st = states + size_t(s) * n_embd;
        float ss = 0.0f;
        for (int d = d0 + lane; d < d1; d += 32) ss += st[d] * st[d];
        const float ss2 = warp_sum_d(ss);
        if (lane == 0) part[(s * 4) * 8 + w] = ss2;
    }
    __syncthreads();
    if (int(threadIdx.x) < n_s) {
        const int s = threadIdx.x;
        float r = 0.0f;
        for (int i = 0; i < 8; ++i) r += part[(s * 4) * 8 + i];
        inv8[s] = rsqrtf(r / float(n_embd) + 1e-6f);
    }
    __syncthreads();

    // the scores, every (state, query) at once
    for (int s = 0; s < n_s; ++s) {
        const float* st = states + size_t(s) * n_embd;
        for (int q = 0; q < q_count; ++q) {
            const float* qv = query + size_t(q) * n_embd;
            float dot = 0.0f;
            for (int d = d0 + lane; d < d1; d += 32) dot += (st[d] * inv8[s]) * qv[d];
            const float dot2 = warp_sum_d(dot);
            if (lane == 0) part[(s * 4 + q) * 8 + w] = dot2;
        }
    }
    __syncthreads();
    if (int(threadIdx.x) < n_s * q_count) {
        const int s = threadIdx.x / q_count, q = threadIdx.x % q_count;
        float r = 0.0f;
        for (int i = 0; i < 8; ++i) r += part[(s * 4 + q) * 8 + i];
        sc[size_t(s) * q_count + q] = r * rsqrtf(float(n_embd));
    }
    __syncthreads();

    // the softmax per query, then the mixes
    if (int(threadIdx.x) < q_count) {
        const int q = threadIdx.x;
        float* pq = ps + q * 9;
        float m = -INFINITY;
        for (int s = 0; s < n_s; ++s) m = fmaxf(m, sc[size_t(s) * q_count + q]);
        float sum = 0.0f;
        for (int s = 0; s < n_s; ++s) {
            pq[s] = __expf(sc[size_t(s) * q_count + q] - m);
            sum += pq[s];
        }
        pq[8] = sum > 0.0f ? 1.0f / sum : 0.0f;
    }
    __syncthreads();
    for (int q = 0; q < q_count; ++q) {
        const float* pq = ps + q * 9;
        float* o = out + size_t(q) * n_embd;
        for (int d = d0 + lane; d < d1; d += 32) {
            float acc = 0.0f;
            for (int s = 0; s < n_s; ++s) acc += pq[s] * pq[8] * (states[size_t(s) * n_embd + d] * inv8[s]);
            o[d] = acc;
        }
    }
}

__global__ void ink_gate_k(const float* cur, const float* message, float* gate_in, int n_embd, size_t cur_stride,
                           size_t msg_stride) {
    cur += size_t(blockIdx.x) * cur_stride;   // one block per row
    message += size_t(blockIdx.x) * msg_stride;
    gate_in += size_t(blockIdx.x) * 2 * size_t(n_embd);
    __shared__ float sum8[8];
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int per = (n_embd + 7) / 8, d0 = w * per, d1 = min(d0 + per, n_embd);
    for (int half = 0; half < 2; ++half) {
        const float* x = half == 0 ? cur : message;
        float* y = gate_in + size_t(half) * n_embd;
        float ss = 0.0f;
        for (int d = d0 + lane; d < d1; d += 32) ss += x[d] * x[d];
        const float ss2 = warp_sum_d(ss);
        if (lane == 0) sum8[w] = ss2;
        __syncthreads();
        float inv = 1.0f;
        if (threadIdx.x == 0) {
            float r = 0.0f;
            for (int i = 0; i < 8; ++i) r += sum8[i];
            sum8[0] = rsqrtf(r / float(n_embd) + 1e-6f);
        }
        __syncthreads();
        inv = sum8[0];
        for (int d = d0 + lane; d < d1; d += 32) y[d] = x[d] * inv;
        __syncthreads();
    }
}

__global__ void ink_combine_k(const float* cur, const float* message, const float* gain, const float* forget,
                              float* out, int n_embd, size_t cur_stride, size_t msg_stride) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_embd) return;
    const size_t r = blockIdx.y;   // the row: cur/out and the message at their strides, gain/forget packed
    const float c = cur[r * cur_stride + i];
    out[r * cur_stride + i] = c - tanhf(forget[r * n_embd + i]) * c + tanhf(gain[r * n_embd + i]) * message[r * msg_stride + i];
}

}  // namespace

bool mix(const float* states, const float* query, float* out, int n_s, int n_embd, int q_count, cudaStream_t s) {
    if (n_s < 1 || n_s > 8 || q_count < 1 || q_count > 4 || n_embd < 8 || n_embd > 2048 * 2) return false;
    ink_mix_k<<<1, 256, 0, s>>>(states, query, out, n_s, n_embd, q_count, 0, 0);
    return cudaGetLastError() == cudaSuccess;
}

bool mix_rows(const float* states, size_t states_stride, const float* query, float* out, size_t out_stride, int n_s,
              int n_embd, int q_count, int rows, cudaStream_t s) {
    if (rows < 1 || n_s < 1 || n_s > 8 || q_count < 1 || q_count > 4 || n_embd < 8 || n_embd > 2048 * 2) return false;
    ink_mix_k<<<rows, 256, 0, s>>>(states, query, out, n_s, n_embd, q_count, states_stride, out_stride);
    return cudaGetLastError() == cudaSuccess;
}

bool control_gate_rows(const float* cur, size_t cur_stride, const float* message, size_t msg_stride, float* gate_in,
                       int n_embd, int rows, cudaStream_t s) {
    if (rows < 1 || n_embd < 8) return false;
    ink_gate_k<<<rows, 256, 0, s>>>(cur, message, gate_in, n_embd, cur_stride, msg_stride);
    return cudaGetLastError() == cudaSuccess;
}

bool control_combine_rows(float* cur, size_t cur_stride, const float* message, size_t msg_stride, const float* gain,
                          const float* forget, int n_embd, int rows, cudaStream_t s) {
    if (rows < 1) return false;
    ink_combine_k<<<dim3((n_embd + 255) / 256, rows), 256, 0, s>>>(cur, message, gain, forget, cur, n_embd, cur_stride,
                                                                   msg_stride);
    return cudaGetLastError() == cudaSuccess;
}

bool control_gate(const float* cur, const float* message, float* gate_in, int n_embd, cudaStream_t s) {
    if (n_embd < 8) return false;
    ink_gate_k<<<1, 256, 0, s>>>(cur, message, gate_in, n_embd, 0, 0);
    return cudaGetLastError() == cudaSuccess;
}

bool control_combine(const float* cur, const float* message, const float* gain, const float* forget,
                     float* out, int n_embd, cudaStream_t s) {
    const int block = 256;
    ink_combine_k<<<(n_embd + block - 1) / block, block, 0, s>>>(cur, message, gain, forget, out, n_embd, 0, 0);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace limbo::cuda::ink
