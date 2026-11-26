// Copyright (c) 2026 ITERFORM. All rights reserved.
// src/cuda/ink_elem.cu — the inkling2 short convolutions and L2 clips (docs/inkling2.md §2/§3), one token
// row per launch (the decode step; verify windows run the same kernels once per row).
//
// conv: out = (cur + conv(x)) * 1/sqrt((dyn ? 2 : 1)*sconv + 1) with no activation, x the slot's history
// rows plus cur, and the state keeps the last sconv-1 rows. The weight is {sconv taps, n_ch} with the taps
// contiguous per channel; a dyn tensor of sconv taps per head adds tap lag of head h over that head's
// channels to x row sconv-1-lag. clip: an L2 clip per head over `row` elements after the optional
// per-channel scale. Both kernels use fixed reduction trees, so a channel's result does not depend on the
// launch geometry.
#include <cmath>

#include "ink_ops.hpp"

namespace limbo::cuda::ink {

namespace {

__device__ __forceinline__ float warp_sum_d(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// One channel per thread pair (two channels per warp): the taps sum, the dynamic taps, the state store.
__global__ void ink_conv_k(const float* cur, const float* dyn, const float* w, float* out, float* hist,
                           int offset, int n_ch, int n_heads, int n_sconv_channels, int sconv, float inv) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= n_ch) return;
    float sum = 0.0f;
    for (int j = 0; j < sconv; ++j) {
        // x row j: the history rows (oldest first), then cur as row sconv-1
        const float xv = j < sconv - 1 ? hist[size_t(j) * n_sconv_channels + offset + ch] : cur[ch];
        sum += w[size_t(ch) * sconv + j] * xv;
    }
    if (dyn) {
        const int per_head = n_ch / n_heads;
        const int hd = ch / per_head;
        for (int lag = 0; lag < sconv; ++lag) {
            // tap lag weights x row sconv-1-lag (the newest input at lag 0)
            const float tap = dyn[size_t(hd) * sconv + lag];
            const int xr = sconv - 1 - lag;
            const float xv = xr < sconv - 1 ? hist[size_t(xr) * n_sconv_channels + offset + ch] : cur[ch];
            sum += tap * xv;
        }
    }
    out[ch] = (cur[ch] + sum) * inv;

    // the state keeps the last sconv-1 rows: shift one row forward and append cur
    for (int j = 0; j < sconv - 1; ++j) {
        const float v = j < sconv - 2 ? hist[size_t(j + 1) * n_sconv_channels + offset + ch] : cur[ch];
        hist[size_t(j) * n_sconv_channels + offset + ch] = v;
    }
}

// The convolution over T consecutive rows of one slot. The convolution is over the inputs only, so every row's
// output is independent: one thread per (channel, row) builds the row's window from the history (rows before the
// piece) and the piece's own inputs, with exactly ink_conv_k's arithmetic. A second kernel then leaves the last
// sconv - 1 inputs in the history.
__global__ void ink_conv_rows_k(const float* cur, size_t cs, const float* dyn, size_t ds, const float* w, float* out,
                                size_t os, const float* hist, int offset, int n_ch, int n_heads, int n_sconv_channels,
                                int sconv, float inv) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (ch >= n_ch) return;
    // window row j (0..sconv-1) is input row t - (sconv - 1) + j: a history row when negative
    const auto xin = [&](int j) {
        const int r = t - (sconv - 1) + j;
        return r < 0 ? hist[size_t(sconv - 1 + r) * n_sconv_channels + offset + ch] : cur[size_t(r) * cs + ch];
    };
    const float c = cur[size_t(t) * cs + ch];
    float sum = 0.0f;
    for (int j = 0; j < sconv; ++j) sum += w[size_t(ch) * sconv + j] * (j < sconv - 1 ? xin(j) : c);
    if (dyn) {
        const float* dy = dyn + size_t(t) * ds;
        const int per_head = n_ch / n_heads;
        const int hd = ch / per_head;
        for (int lag = 0; lag < sconv; ++lag) {
            const float tap = dy[size_t(hd) * sconv + lag];
            const int xr = sconv - 1 - lag;
            sum += tap * (xr < sconv - 1 ? xin(xr) : c);
        }
    }
    out[size_t(t) * os + ch] = (c + sum) * inv;
}

__global__ void ink_conv_hist_k(const float* cur, size_t cs, float* hist, int offset, int n_ch, int n_sconv_channels,
                                int sconv, int T) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= n_ch) return;
    float v[8];
    for (int k = 0; k < sconv - 1; ++k) {
        const int r = T - (sconv - 1) + k;   // the input row history row k holds after the piece
        v[k] = r < 0 ? hist[size_t(T + k) * n_sconv_channels + offset + ch] : cur[size_t(r) * cs + ch];
    }
    for (int k = 0; k < sconv - 1; ++k) hist[size_t(k) * n_sconv_channels + offset + ch] = v[k];
}

// One block per (row, head); the fixed reduction tree over the row's elements is a warp shuffle plus a
// fixed shared pass, so the factor is bit-identical whatever the launch.
__global__ void ink_clip_k(float* v, const float* scale, int row, float threshold) {
    const int t = blockIdx.x, h = blockIdx.y;
    float* vh = v + (size_t(t) * gridDim.y + h) * row;
    const float* sc = scale ? scale + size_t(h) * row : nullptr;
    extern __shared__ float sm[];
    const int i0 = threadIdx.x;
    float ss = 0.0f;
    for (int i = i0; i < row; i += blockDim.x) {
        float x = vh[i];
        if (sc) x *= sc[i];
        vh[i] = x;
        ss += x * x;
    }
    ss = warp_sum_d(ss);
    if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x < 32) {
        float r = threadIdx.x < (blockDim.x >> 5) ? sm[threadIdx.x] : 0.0f;
        r = warp_sum_d(r);
        if (threadIdx.x == 0) {
            const float norm = sqrtf(r) + 1e-6f;
            sm[0] = fminf(fmaxf(threshold / norm, 0.0f), 1.0f);
        }
    }
    __syncthreads();
    const float factor = sm[0];
    for (int i = i0; i < row; i += blockDim.x) vh[i] *= factor;
}

}  // namespace

bool conv(int sconv, const float* cur, const float* dyn, const float* w, float* out, float* hist, int offset,
          int n_ch, int n_heads, int n_sconv_channels, int T, cudaStream_t s) {
    (void)T;
    if (sconv < 2 || n_ch <= 0 || (dyn && (n_ch % n_heads) != 0)) return false;
    const float inv = float(1.0 / std::sqrt(double((dyn ? 2 : 1) * sconv + 1)));
    const int block = 256;
    ink_conv_k<<<(n_ch + block - 1) / block, block, 0, s>>>(cur, dyn, w, out, hist, offset, n_ch, n_heads,
                                                            n_sconv_channels, sconv, inv);
    return cudaGetLastError() == cudaSuccess;
}

bool conv_rows(int sconv, const float* cur, size_t cs, const float* dyn, size_t ds, const float* w, float* out,
               size_t os, float* hist, int offset, int n_ch, int n_heads, int n_sconv_channels, int T, cudaStream_t s) {
    if (sconv < 2 || sconv > 9 || n_ch <= 0 || T < 1 || (dyn && (n_ch % n_heads) != 0)) return false;
    const float inv = float(1.0 / std::sqrt(double((dyn ? 2 : 1) * sconv + 1)));
    ink_conv_rows_k<<<dim3((n_ch + 127) / 128, T), 128, 0, s>>>(cur, cs, dyn, ds, w, out, os, hist, offset, n_ch,
                                                                n_heads, n_sconv_channels, sconv, inv);
    ink_conv_hist_k<<<(n_ch + 127) / 128, 128, 0, s>>>(cur, cs, hist, offset, n_ch, n_sconv_channels, sconv, T);
    return cudaGetLastError() == cudaSuccess;
}

bool clip(float* v, const float* scale, int n_head, int row, float threshold, int T, cudaStream_t s) {
    if (n_head <= 0 || row <= 0 || T <= 0) return false;
    const int block = 128;
    const int n_warp = (block + 31) / 32;
    ink_clip_k<<<dim3(T, n_head), block, n_warp * sizeof(float), s>>>(v, scale, row, threshold);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace limbo::cuda::ink
