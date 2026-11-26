// Copyright (c) 2026 ITERFORM. All rights reserved.
// src/cuda/gemm_f16.cu — the batched product for F16 weights, from the same activation planes as the int8 formats.
//
// F16 weights are not integers times a scale, so the int8 tensor cores cannot take them. They go through the fp16 tensor
// cores instead: the weights are exact in fp16, a plane digit (an int8, |d| <= 127) is exact in fp16, every product is
// exact in fp32 and the 32 products of an activation block sum exactly while they fit 24 bits; the planes are then
// recombined in fp32 per block exactly as the int8 path does, with the block's activation scale (the weight scale is 1).
// So the numerics have the same plane dial: 1 plane = Q8_1 activations, 3 planes = fp32-like. fp16 -> fp32 accumulate
// runs at a quarter of the int8 rate on this card (44 against 189 TOPS, MEASURED), which only matters for large F16
// tensors; the formats of the models in use keep theirs small.
//
// One block = 4 warps = 64 weight rows x 32 token rows, K in stages of 32 (one activation block).
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "limbo/cuda/gemm.hpp"

namespace limbo::cuda {

namespace {

constexpr int kRows = 64;   // weight rows per block
constexpr int kTok = 32;    // token rows per block
constexpr int kPitch = 40;  // halves per smem row: 32 + 8 so that ldmatrix is conflict free

__device__ __forceinline__ void ldsm4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void ldsm2(uint32_t& r0, uint32_t& r1, const void* p) {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(a));
}
__device__ __forceinline__ void mma_f16(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

template <int NT, bool ACC>
__global__ void __launch_bounds__(128) f16_kernel(const __half* W, const int8_t* xq, const float* xs, float* y, int T, int n_out,
                                                  int K) {
    __shared__ __align__(16) __half Ws[kRows * kPitch];
    __shared__ __align__(16) __half Xs[NT * kTok * kPitch];
    __shared__ float Sc[kTok];
    const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31, gq = lane >> 2, tq = lane & 3;
    const int n0 = blockIdx.x * kRows, t0 = blockIdx.y * kTok;
    float acc[4][4];
#pragma unroll
    for (int n = 0; n < 4; ++n)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[n][i] = 0.f;
    const int nstage = K / 32;
    const int64_t plane = static_cast<int64_t>(T) * K;
    for (int st = 0; st < nstage; ++st) {
        __syncthreads();
        // weights: 64 rows x 4 chunks of 8 halves
        for (int i = tid; i < kRows * 4; i += 128) {
            const int r = i >> 2, c = i & 3;
            const int row = min(n0 + r, n_out - 1);
            *reinterpret_cast<uint4*>(Ws + r * kPitch + c * 8) =
                *reinterpret_cast<const uint4*>(W + static_cast<int64_t>(row) * K + st * 32 + c * 8);
        }
        // activations: the int8 digits of every plane as halves
        for (int i = tid; i < NT * kTok * 4; i += 128) {
            const int p = i / (kTok * 4), rem = i % (kTok * 4);
            const int r = rem >> 2, c = rem & 3;
            const int t = min(t0 + r, T - 1);
            const uint2 v = *reinterpret_cast<const uint2*>(xq + p * plane + static_cast<int64_t>(t) * K + st * 32 + c * 8);
            __half h[8];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                h[j] = __float2half(static_cast<float>(static_cast<int8_t>((v.x >> (8 * j)) & 0xff)));
                h[4 + j] = __float2half(static_cast<float>(static_cast<int8_t>((v.y >> (8 * j)) & 0xff)));
            }
            *reinterpret_cast<uint4*>(Xs + (p * kTok + r) * kPitch + c * 8) = *reinterpret_cast<const uint4*>(h);
        }
        if (tid < kTok) Sc[tid] = xs[static_cast<int64_t>(min(t0 + tid, T - 1)) * (K / 32) + st];
        __syncthreads();
        uint32_t A[2][4];
#pragma unroll
        for (int ks = 0; ks < 2; ++ks)
            ldsm4(A[ks][0], A[ks][1], A[ks][2], A[ks][3],
                  Ws + (wid * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * kPitch + ks * 16 + (lane >> 4) * 8);
#pragma unroll
        for (int tn = 0; tn < 4; ++tn) {
            float d[NT][4];
#pragma unroll
            for (int p = 0; p < NT; ++p) {
#pragma unroll
                for (int i = 0; i < 4; ++i) d[p][i] = 0.f;
#pragma unroll
                for (int ks = 0; ks < 2; ++ks) {
                    uint32_t b0, b1;
                    ldsm2(b0, b1, Xs + (p * kTok + tn * 8 + (lane & 7)) * kPitch + ks * 16 + ((lane >> 3) & 1) * 8);
                    mma_f16(d[p], A[ks], b0, b1);
                }
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float v = d[0][i];
#pragma unroll
                for (int p = 1; p < NT; ++p) v = fmaf(v, 128.0f, d[p][i]);
                acc[tn][i] = fmaf(Sc[tn * 8 + 2 * tq + (i & 1)], v, acc[tn][i]);
            }
        }
    }
#pragma unroll
    for (int tn = 0; tn < 4; ++tn)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int t = t0 + tn * 8 + 2 * tq + (i & 1), n = n0 + wid * 16 + gq + (i >= 2 ? 8 : 0);
            if (t >= T || n >= n_out) continue;
            float* dst = y + static_cast<int64_t>(t) * n_out + n;
            if constexpr (ACC) *dst += acc[tn][i];
            else *dst = acc[tn][i];
        }
}

}  // namespace

bool gemm_f16(const void* W, const int8_t* xq, const float* xs, int nt, int T, float* y, int n_out, int n_in, bool accumulate,
              cudaStream_t s) {
    if (n_in % 32 != 0 || n_out <= 0 || T <= 0 || nt < 1 || nt > 3) return false;
    const dim3 grid(static_cast<unsigned>((n_out + kRows - 1) / kRows), static_cast<unsigned>((T + kTok - 1) / kTok));
    const __half* w = static_cast<const __half*>(W);
    if (accumulate) {
        if (nt == 1) f16_kernel<1, true><<<grid, 128, 0, s>>>(w, xq, xs, y, T, n_out, n_in);
        else if (nt == 2) f16_kernel<2, true><<<grid, 128, 0, s>>>(w, xq, xs, y, T, n_out, n_in);
        else f16_kernel<3, true><<<grid, 128, 0, s>>>(w, xq, xs, y, T, n_out, n_in);
    } else {
        if (nt == 1) f16_kernel<1, false><<<grid, 128, 0, s>>>(w, xq, xs, y, T, n_out, n_in);
        else if (nt == 2) f16_kernel<2, false><<<grid, 128, 0, s>>>(w, xq, xs, y, T, n_out, n_in);
        else f16_kernel<3, false><<<grid, 128, 0, s>>>(w, xq, xs, y, T, n_out, n_in);
    }
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace limbo::cuda
