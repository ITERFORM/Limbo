// Copyright (c) 2026 ITERFORM. All rights reserved.
// src/cuda/dec_small.cu — the small kernels on the decode critical path, written for latency: RMS norm, the F32 products of the
// router and the delta net's alpha/beta, and the router's softmax and top-8. A decode step runs 80 norms and 40 router chains,
// each a dependent kernel of a few microseconds, so a microsecond here is 0.1 to 0.3 ms per token.
//
// All of them take T rows (blockIdx), the same code for decode and verify windows.
#include <cfloat>
#include <climits>
#include <cmath>

#include "dec_ops.hpp"

namespace limbo::cuda::dec {

namespace {

template <typename T>
__device__ __forceinline__ T wsum(T v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ bool better_s(float v1, int i1, float v2, int i2) { return v1 > v2 || (v1 == v2 && i1 < i2); }

// y = x * inv(rms(x)) * w with the sum of squares in double, as the reference. grid T, block 256, n <= NI * 1024 floats per
// row: every thread owns NI float4 (elements 4t.. and 1024 + 4t.. for NI = 2) and its weight values are loaded before the reduction.
template <int NI>
__global__ void __launch_bounds__(256) rmsnorm_k(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y, int n,
                                                 float eps, int y_stride, int x_stride) {
    __shared__ double sm[8];
    const int t = threadIdx.x;
    // blockIdx.y: the segment of the row (a hyper-connection stack's stream), each with its own weights
    x += static_cast<int64_t>(blockIdx.x) * x_stride + static_cast<int64_t>(blockIdx.y) * n;
    y += static_cast<int64_t>(blockIdx.x) * y_stride + static_cast<int64_t>(blockIdx.y) * n;
    w += static_cast<int64_t>(blockIdx.y) * n;
    const int nv = n / 4;  // float4 per row (512 for n_embd 2048)
    float4 xv[NI], wv[NI];
#pragma unroll
    for (int i = 0; i < NI; ++i) {
        const int j = t + i * 256;
        xv[i] = j < nv ? *reinterpret_cast<const float4*>(x + 4 * j) : make_float4(0.f, 0.f, 0.f, 0.f);
        wv[i] = j < nv ? __ldg(reinterpret_cast<const float4*>(w + 4 * j)) : make_float4(0.f, 0.f, 0.f, 0.f);
    }
    double ss = 0;
#pragma unroll
    for (int i = 0; i < NI; ++i) {
        ss += static_cast<double>(xv[i].x) * xv[i].x;
        ss += static_cast<double>(xv[i].y) * xv[i].y;
        ss += static_cast<double>(xv[i].z) * xv[i].z;
        ss += static_cast<double>(xv[i].w) * xv[i].w;
    }
    ss = wsum(ss);
    if ((t & 31) == 0) sm[t >> 5] = ss;
    __syncthreads();
    double tot = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) tot += sm[i];
    const float inv = static_cast<float>(1.0 / sqrt(tot / n + static_cast<double>(eps)));
#pragma unroll
    for (int i = 0; i < NI; ++i) {
        const int j = t + i * 256;
        if (j < nv)
            *reinterpret_cast<float4*>(y + 4 * j) =
                make_float4(xv[i].x * inv * wv[i].x, xv[i].y * inv * wv[i].y, xv[i].z * inv * wv[i].z, xv[i].w * inv * wv[i].w);
    }
}

// F32 rows: y[c][r] = W[r, :] . x[c], one block per (row, matrix), 128 threads, each thread four float4 of the row and the
// same of every column; the per-column sums go through a fixed tree (warp shuffles, then the four warps in order).
// grid (n_out, n_z), n_in = 2048.
template <int C>
__global__ void __launch_bounds__(128) f32_cols_k(const float* __restrict__ W0, const float* __restrict__ W1, const float* __restrict__ x,
                                                 int x_stride, float* y0, float* y1, int y_stride, int n_in) {
    __shared__ float sm[C][4];
    const int t = threadIdx.x;
    const float* W = blockIdx.y ? W1 : W0;
    float* y = blockIdx.y ? y1 : y0;
    const int row = blockIdx.x;
    const float* wr = W + static_cast<int64_t>(row) * n_in;
    const int nv = n_in / 4;
    float4 wv[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) wv[i] = t + i * 128 < nv ? __ldg(reinterpret_cast<const float4*>(wr) + t + i * 128) : make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int c = 0; c < C; ++c) {
        const float4* xc = reinterpret_cast<const float4*>(x + static_cast<int64_t>(c) * x_stride);
        float a = 0.f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int j = t + i * 128;
            if (j < nv) {
                const float4 xv = __ldg(xc + j);
                a = fmaf(wv[i].x, xv.x, a);
                a = fmaf(wv[i].y, xv.y, a);
                a = fmaf(wv[i].z, xv.z, a);
                a = fmaf(wv[i].w, xv.w, a);
            }
        }
        a = wsum(a);
        if ((t & 31) == 0) sm[c][t >> 5] = a;
    }
    __syncthreads();
    if (t < C) {
        const float r = ((sm[t][0] + sm[t][1]) + sm[t][2]) + sm[t][3];
        y[static_cast<int64_t>(t) * y_stride + row] = r;
    }
}

// Softmax over n <= 256 router logits and the top k by (probability desc, index asc); weights p / max(sum of the chosen p,
// 6.1035e-5). One warp per row: a lane holds 8 of the 256 probabilities and the k rounds are warp argmaxes. grid T, block 32.
template <int P>
__global__ void __launch_bounds__(32) topk_k(const float* __restrict__ logits, int n, int k, int32_t* ids, float* wts) {
    const int lane = threadIdx.x;
    logits += static_cast<int64_t>(blockIdx.x) * n;
    ids += static_cast<int64_t>(blockIdx.x) * k;
    wts += static_cast<int64_t>(blockIdx.x) * k;
    float v[P];
    float m = -FLT_MAX;
#pragma unroll
    for (int i = 0; i < P; ++i) {
        const int idx = lane + 32 * i;
        v[i] = idx < n ? logits[idx] : -FLT_MAX;
        m = fmaxf(m, v[i]);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < P; ++i) {
        const int idx = lane + 32 * i;
        v[i] = idx < n ? expf(v[i] - m) : 0.f;
        s += v[i];
    }
    s = wsum(s);
    const float inv = s > 0.f ? 1.0f / s : 0.0f;
#pragma unroll
    for (int i = 0; i < P; ++i) v[i] *= inv;
    float chosen[16];
    int cid[16];
    for (int r = 0; r < k; ++r) {
        float bv = -2.0f;
        int bi = INT_MAX;
#pragma unroll
        for (int i = 0; i < P; ++i) {
            const int idx = lane + 32 * i;
            if (idx < n && better_s(v[i], idx, bv, bi)) {
                bv = v[i];
                bi = idx;
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (better_s(ov, oi, bv, bi)) {
                bv = ov;
                bi = oi;
            }
        }
        chosen[r] = bv;
        cid[r] = bi;
        if ((bi & 31) == lane) {
#pragma unroll
            for (int i = 0; i < P; ++i)
                if (lane + 32 * i == bi) v[i] = -1.0f;
        }
    }
    if (lane == 0) {
        float wsum_ = 0.f;
        for (int r = 0; r < k; ++r) wsum_ += chosen[r];
        wsum_ = fmaxf(wsum_, 6.103515625e-05f);
        for (int r = 0; r < k; ++r) {
            ids[r] = cid[r];
            wts[r] = chosen[r] / wsum_;
        }
    }
}

}  // namespace

bool rmsnorm(const float* x, const float* w, float* y, int n, float eps, int T, cudaStream_t s, int y_stride,
             int x_stride, int segs) {
    if (n % 4 != 0 || n > 16384 || T < 1 || segs < 1) return false;
    const int ys = y_stride > 0 ? y_stride : n;
    // x_stride 0 keeps the rows contiguous (every existing caller); a qwen4exp hyper-connection read passes the
    // stack's row pitch, because its 4 residual streams are interleaved inside each row.
    const int xs = x_stride > 0 ? x_stride : n;
    const dim3 g(static_cast<unsigned>(T), static_cast<unsigned>(segs));
    if (n <= 2048) rmsnorm_k<2><<<g, 256, 0, s>>>(x, w, y, n, eps, ys, xs);
    else if (n <= 4096) rmsnorm_k<4><<<g, 256, 0, s>>>(x, w, y, n, eps, ys, xs);
    else if (n <= 8192) rmsnorm_k<8><<<g, 256, 0, s>>>(x, w, y, n, eps, ys, xs);
    else rmsnorm_k<16><<<g, 256, 0, s>>>(x, w, y, n, eps, ys, xs);
    return true;
}

bool f32_cols(const float* W0, const float* W1, const float* x, int x_stride, float* y0, float* y1, int y_stride, int n_out, int n_in, int C,
              int n_z, cudaStream_t s) {
    if (n_in % 4 != 0 || n_in > 2048 || C < 1 || C > kMaxRows || n_z < 1 || n_z > 2) return false;
    const dim3 grid(static_cast<unsigned>(n_out), static_cast<unsigned>(n_z));
    switch (C) {
        case 1: f32_cols_k<1><<<grid, 128, 0, s>>>(W0, W1, x, x_stride, y0, y1, y_stride, n_in); break;
        case 2: f32_cols_k<2><<<grid, 128, 0, s>>>(W0, W1, x, x_stride, y0, y1, y_stride, n_in); break;
        case 3: f32_cols_k<3><<<grid, 128, 0, s>>>(W0, W1, x, x_stride, y0, y1, y_stride, n_in); break;
        case 4: f32_cols_k<4><<<grid, 128, 0, s>>>(W0, W1, x, x_stride, y0, y1, y_stride, n_in); break;
        case 5: f32_cols_k<5><<<grid, 128, 0, s>>>(W0, W1, x, x_stride, y0, y1, y_stride, n_in); break;
        case 6: f32_cols_k<6><<<grid, 128, 0, s>>>(W0, W1, x, x_stride, y0, y1, y_stride, n_in); break;
        case 7: f32_cols_k<7><<<grid, 128, 0, s>>>(W0, W1, x, x_stride, y0, y1, y_stride, n_in); break;
        default: f32_cols_k<8><<<grid, 128, 0, s>>>(W0, W1, x, x_stride, y0, y1, y_stride, n_in); break;
    }
    return cudaGetLastError() == cudaSuccess;
}

bool topk(const float* logits, int n, int k, int T, int32_t* ids, float* wts, cudaStream_t s) {
    if (n > 512 || k < 1 || k > 16 || T < 1) return false;
    if (n <= 256) topk_k<8><<<T, 32, 0, s>>>(logits, n, k, ids, wts);
    else topk_k<16><<<T, 32, 0, s>>>(logits, n, k, ids, wts);   // qwen4exp's MTP layer routes over 512
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace limbo::cuda::dec
