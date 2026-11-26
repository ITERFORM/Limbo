// Copyright (c) 2026 ITERFORM. All rights reserved.
// src/cuda/gemv_q6k.cu — Q6_K products at decode, written to be bandwidth-bound.
//
// Q6_K is 77 percent of the bytes a decode token reads (the delta net's three projections, the attention projections, the
// shared experts and the LM head). The generic GEMV decodes it with about 19 instructions per weight, which holds the big
// products at 225 to 245 GB/s of a 265 GB/s card and makes every extra input column cost a full pass of ALU work. Here a lane
// owns a unit of 16 weights (four quarters of a 32-element group, four consecutive elements each): it loads three 32-bit
// words (the low and high nibbles and the 2-bit fields), builds the four quarters' 6-bit values with four mask-and-or steps on
// whole words, takes 32 off all four bytes of a word at once, and the per-16-element scale multiplies the unit's four partial
// dots. A warp walks R rows with the same activations in registers, so the activation loads are shared, and the R rows' sums
// are reduced with one butterfly.
//
// C columns (1..8) share the weight decode; per column the arithmetic is the same whatever C is.
#include <cuda_fp16.h>

#include "dec_ops.hpp"
#include "limbo/cuda/gemv.hpp"
#include "limbo/types.hpp"

namespace limbo::cuda {

namespace {


__device__ __forceinline__ float h2f_q(const uint8_t* p) {
    return __half2float(__ushort_as_half(*reinterpret_cast<const unsigned short*>(p)));
}
// A 32-bit word at a 2-byte aligned address (Q6_K super-blocks are 210 bytes).
__device__ __forceinline__ uint32_t ld32_q(const uint8_t* p) {
    return static_cast<uint32_t>(*reinterpret_cast<const unsigned short*>(p)) |
           (static_cast<uint32_t>(*reinterpret_cast<const unsigned short*>(p + 2)) << 16);
}
__device__ __forceinline__ float dot4_q(uint32_t w, const float4& x, float acc) {
    acc = fmaf(static_cast<float>(static_cast<int8_t>(w)), x.x, acc);
    acc = fmaf(static_cast<float>(static_cast<int8_t>(w >> 8)), x.y, acc);
    acc = fmaf(static_cast<float>(static_cast<int8_t>(w >> 16)), x.z, acc);
    acc = fmaf(static_cast<float>(static_cast<int8_t>(w >> 24)), x.w, acc);
    return acc;
}
// Bytes 0..63 minus 32 as four signed bytes: (v | 0x80) - 0x20 cannot borrow across bytes, flipping bit 7 then adds 128 mod 256.
__device__ __forceinline__ uint32_t sub32(uint32_t v) { return ((v | 0x80808080u) - 0x20202020u) ^ 0x80808080u; }

__device__ __forceinline__ float silu_q(float x) { return x / (1.0f + expf(-x)); }

template <int N, int OFF>
__device__ __forceinline__ float rscatter_q(const float (&v)[N], int lane) {
    if constexpr (N == 1) {
        float r = v[0];
#pragma unroll
        for (int o = OFF; o > 0; o >>= 1) r += __shfl_xor_sync(0xffffffffu, r, o);
        return r;
    } else {
        constexpr int H = N / 2;
        float a[H];
        const bool hi = (lane & OFF) != 0;
#pragma unroll
        for (int i = 0; i < H; ++i) {
            const float keep = hi ? v[i + H] : v[i], send = hi ? v[i] : v[i + H];
            a[i] = keep + __shfl_xor_sync(0xffffffffu, send, OFF);
        }
        return rscatter_q<H, OFF / 2>(a, lane);
    }
}

// grid (ceil(n_out / (8 R)), n_z), block 256. Lane = (sbp: super-block parity of the pair, h: half, g8: group of 4 elements).
// x2 != nullptr: the input is silu(x) * x2 (the shared expert's down projection), formed per column in registers.
template <int R, int C>
__global__ void __launch_bounds__(256)
    q6k_kernel(const uint8_t* __restrict__ W, const uint8_t* __restrict__ W2, uint64_t row_bytes, const float* __restrict__ x,
               const float* __restrict__ x2, int x_stride, float* y, float* y2, int y_stride, int n_out, int n_sb, bool accumulate) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g8 = lane & 7, h = (lane >> 3) & 1, sbp = lane >> 4;
    const uint8_t* Wm = blockIdx.y ? W2 : W;
    float* ym = blockIdx.y ? y2 : y;
    const int row0 = (blockIdx.x * 8 + warp) * R;
    const uint8_t* rowp[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int rr = min(row0 + r, n_out - 1);
        rowp[r] = Wm + static_cast<uint64_t>(rr) * row_bytes;
    }
    const int sc_off = 192 + h * 8 + (g8 >> 2);
    float acc[R][C];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < C; ++c) acc[r][c] = 0.f;

#pragma unroll 2
    for (int it = 0; it < n_sb / 2; ++it) {
        const int sb = 2 * it + sbp;
        // activations of this unit: quarter qi covers elements h*128 + qi*32 + 4*g8 .. +3 of the super-block. One column
        // (the decode step) loads them once for the R rows; several columns (R = 1) load each column's when it is used,
        // which keeps the registers of the wide variants low without changing any column's arithmetic.
        auto load_x = [&](int c, float4 (&xv)[4]) {
#pragma unroll
            for (int qi = 0; qi < 4; ++qi) {
                const int64_t off = static_cast<int64_t>(c) * x_stride + sb * 256 + h * 128 + qi * 32 + 4 * g8;
                float4 v = __ldg(reinterpret_cast<const float4*>(x + off));
                if (x2) {
                    const float4 u = __ldg(reinterpret_cast<const float4*>(x2 + off));
                    v.x = silu_q(v.x) * u.x;
                    v.y = silu_q(v.y) * u.y;
                    v.z = silu_q(v.z) * u.z;
                    v.w = silu_q(v.w) * u.w;
                }
                xv[qi] = v;
            }
        };
        float4 x0[4];
        if constexpr (C == 1) load_x(0, x0);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint8_t* sbp_ = rowp[r] + static_cast<uint64_t>(sb) * 210;
            const uint32_t qlA = ld32_q(sbp_ + h * 64 + 4 * g8), qlB = ld32_q(sbp_ + h * 64 + 32 + 4 * g8), qhw = ld32_q(sbp_ + 128 + h * 32 + 4 * g8);
            const float d = h2f_q(sbp_ + 208);
            const int s0 = static_cast<int8_t>(sbp_[sc_off]), s1 = static_cast<int8_t>(sbp_[sc_off + 2]),
                      s2 = static_cast<int8_t>(sbp_[sc_off + 4]), s3 = static_cast<int8_t>(sbp_[sc_off + 6]);
            const uint32_t m6 = 0x30303030u, m4 = 0x0F0F0F0Fu;
            const uint32_t v0 = sub32((qlA & m4) | ((qhw << 4) & m6));
            const uint32_t v1 = sub32((qlB & m4) | ((qhw << 2) & m6));
            const uint32_t v2 = sub32(((qlA >> 4) & m4) | (qhw & m6));
            const uint32_t v3 = sub32(((qlB >> 4) & m4) | ((qhw >> 2) & m6));
            const float d0 = d * static_cast<float>(s0), d1 = d * static_cast<float>(s1), d2 = d * static_cast<float>(s2),
                        d3 = d * static_cast<float>(s3);
#pragma unroll 2
            for (int c = 0; c < C; ++c) {
                float4 xc[4];
                if constexpr (C == 1) {
#pragma unroll
                    for (int qi = 0; qi < 4; ++qi) xc[qi] = x0[qi];
                } else {
                    load_x(c, xc);
                }
                acc[r][c] = fmaf(d0, dot4_q(v0, xc[0], 0.f), acc[r][c]);
                acc[r][c] = fmaf(d1, dot4_q(v1, xc[1], 0.f), acc[r][c]);
                acc[r][c] = fmaf(d2, dot4_q(v2, xc[2], 0.f), acc[r][c]);
                acc[r][c] = fmaf(d3, dot4_q(v3, xc[3], 0.f), acc[r][c]);
            }
        }
    }
    // R rows' sums over the 32 lanes: after the scatter lane (idx * 32 / R) .. holds row idx (idx = lane / (32 / R))
#pragma unroll
    for (int c = 0; c < C; ++c) {
        float v[R];
#pragma unroll
        for (int r = 0; r < R; ++r) v[r] = acc[r][c];
        const float sum = rscatter_q<R, 16>(v, lane);
        constexpr int G = 32 / R;
        const int row = row0 + lane / G;
        if (lane % G == 0 && row < n_out) {
            float* yp = ym + static_cast<int64_t>(c) * y_stride + row;
            *yp = accumulate ? *yp + sum : sum;
        }
    }
}

template <int R, int C>
void launch_q6k(const uint8_t* W, const uint8_t* W2, uint64_t rb, const float* x, const float* x2, int xs, float* y, float* y2, int ys,
                int n_out, int n_sb, bool acc, int n_z, cudaStream_t s) {
    q6k_kernel<R, C><<<dim3(static_cast<unsigned>((n_out + 8 * R - 1) / (8 * R)), static_cast<unsigned>(n_z)), 256, 0, s>>>(
        W, W2, rb, x, x2, xs, y, y2, ys, n_out, n_sb, acc);
}

template <int C>
void pick_r(const uint8_t* W, const uint8_t* W2, uint64_t rb, const float* x, const float* x2, int xs, float* y, float* y2, int ys, int n_out,
            int n_sb, bool acc, int n_z, cudaStream_t s) {
    if constexpr (C == 1) {
        if (n_out * n_z >= 4096) launch_q6k<4, 1>(W, W2, rb, x, x2, xs, y, y2, ys, n_out, n_sb, acc, n_z, s);
        else if (n_out * n_z >= 1024) launch_q6k<2, 1>(W, W2, rb, x, x2, xs, y, y2, ys, n_out, n_sb, acc, n_z, s);
        else launch_q6k<1, 1>(W, W2, rb, x, x2, xs, y, y2, ys, n_out, n_sb, acc, n_z, s);
    } else {
        launch_q6k<1, C>(W, W2, rb, x, x2, xs, y, y2, ys, n_out, n_sb, acc, n_z, s);
    }
}

}  // namespace

bool q6k_launch(const void* W, const void* W2, const float* x, const float* x2, int x_stride, float* y, float* y2, int y_stride, int n_out,
                int n_in, int C, bool accumulate, int n_z, cudaStream_t s) {
    if (n_in % 512 != 0 || n_out <= 0 || C < 1 || C > 8 || n_z < 1 || n_z > 2) return false;
    const uint64_t rb = static_cast<uint64_t>(n_in / 256) * 210;
    const int nsb = n_in / 256;
    const uint8_t* w = static_cast<const uint8_t*>(W);
    const uint8_t* w2 = static_cast<const uint8_t*>(W2);
    switch (C) {
        case 1: pick_r<1>(w, w2, rb, x, x2, x_stride, y, y2, y_stride, n_out, nsb, accumulate, n_z, s); break;
        case 2: pick_r<2>(w, w2, rb, x, x2, x_stride, y, y2, y_stride, n_out, nsb, accumulate, n_z, s); break;
        case 3: pick_r<3>(w, w2, rb, x, x2, x_stride, y, y2, y_stride, n_out, nsb, accumulate, n_z, s); break;
        case 4: pick_r<4>(w, w2, rb, x, x2, x_stride, y, y2, y_stride, n_out, nsb, accumulate, n_z, s); break;
        case 5: pick_r<5>(w, w2, rb, x, x2, x_stride, y, y2, y_stride, n_out, nsb, accumulate, n_z, s); break;
        case 6: pick_r<6>(w, w2, rb, x, x2, x_stride, y, y2, y_stride, n_out, nsb, accumulate, n_z, s); break;
        case 7: pick_r<7>(w, w2, rb, x, x2, x_stride, y, y2, y_stride, n_out, nsb, accumulate, n_z, s); break;
        case 8: pick_r<8>(w, w2, rb, x, x2, x_stride, y, y2, y_stride, n_out, nsb, accumulate, n_z, s); break;
        default: return false;
    }
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace limbo::cuda
