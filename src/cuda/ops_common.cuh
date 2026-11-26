// Copyright (c) 2026 ITERFORM. All rights reserved.
#pragma once
// src/cuda/ops_common.cuh — device helpers shared by ops.cu (decode) and ops_rows.cu (prefill).
#include <cuda_runtime.h>

namespace limbo::cuda::ops {
namespace {


__device__ __forceinline__ float silu_f(float x) { return x / (1.0f + expf(-x)); }
__device__ __forceinline__ float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float softplus_f(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

template <typename T>
__device__ __forceinline__ T warp_sum(T v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Block-wide sum; every thread gets the same value (the per-warp partials are added in the same order by all).
template <typename T>
__device__ __forceinline__ T block_sum(T v, T* sm) {
    v = warp_sum(v);
    if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = v;
    __syncthreads();
    T r = 0;
    for (int i = 0; i < static_cast<int>(blockDim.x >> 5); ++i) r += sm[i];
    __syncthreads();
    return r;
}

__device__ __forceinline__ float block_max(float v, float* sm) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = v;
    __syncthreads();
    float r = -FLT_MAX;
    for (int i = 0; i < static_cast<int>(blockDim.x >> 5); ++i) r = fmaxf(r, sm[i]);
    __syncthreads();
    return r;
}

// (value, index) ordering for argmax: the larger value wins, the lower index on a tie.
__device__ __forceinline__ bool better(float v1, int i1, float v2, int i2) { return v1 > v2 || (v1 == v2 && i1 < i2); }

__device__ __forceinline__ void block_argmax(float& v, int& idx, float* sv, int* si) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, v, o);
        const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
        if (better(ov, oi, v, idx)) {
            v = ov;
            idx = oi;
        }
    }
    if ((threadIdx.x & 31) == 0) {
        sv[threadIdx.x >> 5] = v;
        si[threadIdx.x >> 5] = idx;
    }
    __syncthreads();
    float bv = sv[0];
    int bi = si[0];
    for (int i = 1; i < static_cast<int>(blockDim.x >> 5); ++i)
        if (better(sv[i], si[i], bv, bi)) {
            bv = sv[i];
            bi = si[i];
        }
    __syncthreads();
    v = bv;
    idx = bi;
}

}  // namespace
}  // namespace limbo::cuda::ops
