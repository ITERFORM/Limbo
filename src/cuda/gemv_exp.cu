// Copyright (c) 2026 ITERFORM. All rights reserved.
// src/cuda/gemv_exp.cu — the routed experts' products at decode, written to be bandwidth-bound.
//
// The generic GEMV in gemv.cu decodes every weight with about 11 instructions and is issue-bound on the i-quants
// (150 GB/s of a 265 GB/s card). Here the work per weight is cut by doing the codebook arithmetic on four bytes at
// a time (docs/donors/kindi.md 2.2: the sign trick: a sign bit pattern is a per-byte XOR mask plus a per-byte +1,
// so a whole grid word is negated in two instructions, and the masks come from a 256-entry table), by keeping the
// activation row in registers while a warp walks several weight rows, and by reducing all the rows' partial sums
// with one butterfly (9 shuffles for 8 rows instead of 40).
//
//   gate/up + SwiGLU, IQ2_S weights, 2048 columns: h[e][r] = silu(Wg[ids[e]][r] . x) * (Wu[ids[e]][r] . x)
//   down, IQ3_S weights, 512 columns:              y[e][r] = Wd[ids[e]][r] . h[e]
//
// Both keep the mapping of gemv.cu (lane L owns elements 8L..8L+7 of a 256-element super-block: one codebook entry
// of IQ2_S or two of IQ3_S), so the products differ from the generic ones only in the order of the float additions.
#include <cuda_fp16.h>

#include "dec_ops.hpp"
#include "limbo/cuda/gemv.hpp"
#include "limbo/quants.hpp"
#include "limbo/types.hpp"

#ifndef EXP_GU_RPW
#define EXP_GU_RPW 4
#endif
#ifndef EXP_GU_BLK
#define EXP_GU_BLK 2
#endif
#ifndef EXP_DN_RPW
#define EXP_DN_RPW 8
#endif
#ifndef EXP_DN_BLK
#define EXP_DN_BLK 3
#endif

namespace limbo::cuda {

namespace {

constexpr uint32_t kIQ2_S = static_cast<uint32_t>(Type::IQ2_S);
constexpr uint32_t kIQ3_S = static_cast<uint32_t>(Type::IQ3_S);

__device__ uint64_t d_g2[1024];
__device__ uint32_t d_g3[512];
bool g_exp_ready = false;

// Four sign bits (the low nibble of s4) as bytes 0/1: bit i lands in byte i.
__device__ __forceinline__ uint32_t sign_bytes(uint32_t s4) { return ((s4 & 0xFu) * 0x00204081u) & 0x01010101u; }

__device__ __forceinline__ float h2f_u16(const uint8_t* p) {
    return __half2float(__ushort_as_half(*reinterpret_cast<const unsigned short*>(p)));
}

// acc + sum_j int8(w, j) * x_j
__device__ __forceinline__ float dot4(uint32_t w, const float4& x, float acc) {
    acc = fmaf(static_cast<float>(static_cast<int8_t>(w)), x.x, acc);
    acc = fmaf(static_cast<float>(static_cast<int8_t>(w >> 8)), x.y, acc);
    acc = fmaf(static_cast<float>(static_cast<int8_t>(w >> 16)), x.z, acc);
    acc = fmaf(static_cast<float>(static_cast<int8_t>(w >> 24)), x.w, acc);
    return acc;
}

// Per sign byte: {m_lo, m_hi, M_lo, M_hi}: m = bytes 0/1 of the sign bits (low and high nibble), M = m * 0xFF. No byte
// of the i-quant grids is 0 or above 127, so -w = (w ^ M) + m never carries into the next byte.
__device__ __forceinline__ void fill_sign_table(uint4* s_sg) {
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        const uint32_t lo = sign_bytes(static_cast<uint32_t>(i)), hi = sign_bytes(static_cast<uint32_t>(i) >> 4);
        s_sg[i] = make_uint4(lo, hi, lo * 0xFFu, hi * 0xFFu);
    }
}

// Butterfly reduce-scatter of N (2, 4 or 8) values per lane over the warp: after it, every lane holds the warp-wide
// sum of value (lane >> (5 - log2 N)). The stages halve the live values at offsets 16, 8, 4, ... and the offsets
// left over are plain all-reduce steps. N = 8: 9 shuffles, N = 4: 6, N = 2: 4.
template <int N, int OFF>
__device__ __forceinline__ float rscatter(const float (&v)[N], int lane) {
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
        return rscatter<H, OFF / 2>(a, lane);
    }
}

__device__ __forceinline__ float silu_e(float x) { return x / (1.0f + expf(-x)); }

// ---- IQ2_S gate/up with SwiGLU ---------------------------------------------------------------------------
// Persistent blocks of 256 threads (the tables of the block are built once, then it walks work items). An item is
// (slot, 8 * RPW consecutive output rows); each warp walks RPW rows of it, and per row it reads the gate row and the up
// row of the slot's expert, so SwiGLU needs no second pass. IQ2_S super-block: d @0, qs[32] @2, signs[32] @34,
// qh[8] @66, scales[8] @74 (82 bytes); 8 super-blocks per 2048-column row. The input row is the same for every slot, so
// a lane keeps its 64 activations in registers for the whole kernel.
template <int RPW>
__global__ void __launch_bounds__(256, EXP_GU_BLK)
    gate_up_iq2s_kernel(const uint8_t* __restrict__ Wg, const uint8_t* __restrict__ Wu, uint64_t expert_bytes,
                        const int32_t* __restrict__ ids, const float* __restrict__ x, float* __restrict__ h, int n_out,
                        int n_sel, int slots_per_row, int x_stride) {
    constexpr int NSB = 8, RB = NSB * 82;
    __shared__ uint64_t s_g[1024];
    __shared__ uint4 s_sg[256];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int ib32 = lane >> 2, l4 = lane & 3;
    const int sh_qh = 8 - 2 * l4, sh_nib = (l4 >> 1) * 4;

    float4 xr[NSB][2];
    int xrow = -1;  // the input row the registers hold; rows are slot / slots_per_row (a verify window has one per token)
    for (int i = threadIdx.x; i < 1024; i += blockDim.x) s_g[i] = d_g2[i];
    fill_sign_table(s_sg);
    __syncthreads();

    const int chunks = n_out / (8 * RPW);
    for (int item = blockIdx.x; item < chunks * n_sel; item += gridDim.x) {
        const int slot = item / chunks, chunk = item - slot * chunks;
        const uint64_t eoff = static_cast<uint64_t>(ids[slot]) * expert_bytes;
        const int row0 = (chunk * 8 + warp) * RPW;
        const int want = slots_per_row > 0 ? slot / slots_per_row : 0;
        if (want != xrow) {
            xrow = want;
            const float* xp = x + static_cast<int64_t>(want) * x_stride;
#pragma unroll
            for (int s = 0; s < NSB; ++s) {
                xr[s][0] = __ldg(reinterpret_cast<const float4*>(xp + s * 256 + lane * 8));
                xr[s][1] = __ldg(reinterpret_cast<const float4*>(xp + s * 256 + lane * 8 + 4));
            }
        }
        float acc[2 * RPW];
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
#pragma unroll
            for (int z = 0; z < 2; ++z) {
                const uint8_t* wr = (z ? Wu : Wg) + eoff + static_cast<uint64_t>(row0 + r) * RB;
                float a = 0.f;
#pragma unroll
                for (int s = 0; s < NSB; ++s) {
                    const uint8_t* sb = wr + s * 82;
                    const float d = h2f_u16(sb);
                    const uint32_t qs = sb[2 + lane], sg = sb[34 + lane], qh = sb[66 + ib32], sc = sb[74 + ib32];
                    const uint32_t idx = qs | ((qh << sh_qh) & 0x300u);
                    const uint2 g = reinterpret_cast<const uint2*>(s_g)[idx];
                    const uint4 m = s_sg[sg];
                    const uint32_t w0 = (g.x ^ m.z) + m.x, w1 = (g.y ^ m.w) + m.y;
                    float p = dot4(w0, xr[s][0], 0.f);
                    p = dot4(w1, xr[s][1], p);
                    const float dl = d * (0.5f + static_cast<float>((sc >> sh_nib) & 15u)) * 0.25f;
                    a = fmaf(dl, p, a);
                }
                acc[2 * r + z] = a;
            }
        }
        // acc = {g0, u0, g1, u1, ...}: after the scatter every lane holds the sum of value lane / (32 / (2 RPW))
        const float v = rscatter<2 * RPW, 16>(acc, lane);
        constexpr int G = 32 / (2 * RPW);  // lanes per value
        const float other = __shfl_down_sync(0xffffffffu, v, G);  // gate and up of a row sit in neighbouring lane groups
        if (lane % (2 * G) == 0) h[static_cast<int64_t>(slot) * n_out + row0 + lane / (2 * G)] = silu_e(v) * other;
    }
}

// ---- IQ3_S down -------------------------------------------------------------------------------------------
// Persistent blocks as above. IQ3_S super-block: d @0, qs[64] @2, qh[8] @66, signs[32] @74, scales[4] @106 (110 bytes);
// 2 super-blocks per 512-column row. h is [n_sel][512] (a row per slot), y is [n_sel][n_out].
template <int RPW>
__global__ void __launch_bounds__(256, EXP_DN_BLK)
    down_iq3s_kernel(const uint8_t* __restrict__ Wd, uint64_t expert_bytes, const int32_t* __restrict__ ids,
                     const float* __restrict__ h, float* __restrict__ y, int n_out, int n_sel) {
    constexpr int NSB = 2, RB = NSB * 110;
    __shared__ uint32_t s_g[512];
    __shared__ uint4 s_sg[256];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int l = lane & 3, qh_i = lane >> 2, sc_i = lane >> 3, sh_nib = ((lane >> 2) & 1) * 4;
    const int sh1 = 8 - 2 * l, sh2 = 7 - 2 * l;
    for (int i = threadIdx.x; i < 512; i += blockDim.x) s_g[i] = d_g3[i];
    fill_sign_table(s_sg);
    __syncthreads();

    const int chunks = n_out / (8 * RPW);
    for (int item = blockIdx.x; item < chunks * n_sel; item += gridDim.x) {
        const int slot = item / chunks, chunk = item - slot * chunks;
        const uint8_t* wbase = Wd + static_cast<uint64_t>(ids[slot]) * expert_bytes;
        const float* hx = h + static_cast<int64_t>(slot) * 512;
        float4 xr[NSB][2];
#pragma unroll
        for (int s = 0; s < NSB; ++s) {
            xr[s][0] = *reinterpret_cast<const float4*>(hx + s * 256 + lane * 8);
            xr[s][1] = *reinterpret_cast<const float4*>(hx + s * 256 + lane * 8 + 4);
        }
        const int row0 = (chunk * 8 + warp) * RPW;
        float acc[RPW];
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            const uint8_t* wr = wbase + static_cast<uint64_t>(row0 + r) * RB;
            float a = 0.f;
#pragma unroll
            for (int s = 0; s < NSB; ++s) {
                const uint8_t* sb = wr + s * 110;
                const float d = h2f_u16(sb);
                const uint32_t qs2 = *reinterpret_cast<const unsigned short*>(sb + 2 + 2 * lane);
                const uint32_t qh = sb[66 + qh_i], sg = sb[74 + lane], scb = sb[106 + sc_i];
                const uint32_t i1 = (qs2 & 0xffu) | ((qh << sh1) & 256u), i2 = (qs2 >> 8) | ((qh << sh2) & 256u);
                const uint32_t g1 = s_g[i1], g2 = s_g[i2];
                const uint4 m = s_sg[sg];
                const uint32_t w0 = (g1 ^ m.z) + m.x, w1 = (g2 ^ m.w) + m.y;
                float p = dot4(w0, xr[s][0], 0.f);
                p = dot4(w1, xr[s][1], p);
                const float db = d * (1.f + 2.f * static_cast<float>((scb >> sh_nib) & 15u));
                a = fmaf(db, p, a);
            }
            acc[r] = a;
        }
        const float v = rscatter<RPW, 16>(acc, lane);
        constexpr int G = 32 / RPW;  // lanes per row
        if (lane % G == 0) y[static_cast<int64_t>(slot) * n_out + row0 + lane / G] = v;
    }
}

}  // namespace

bool exp_init(std::string* err) {
    if (g_exp_ready) return true;
    cudaError_t e = cudaMemcpyToSymbol(d_g2, quants::kIq2sGrid, sizeof(uint64_t) * 1024);
    if (e == cudaSuccess) e = cudaMemcpyToSymbol(d_g3, quants::kIq3sGrid, sizeof(uint32_t) * 512);
    if (e != cudaSuccess) {
        if (err) *err = std::string("exp_init: ") + cudaGetErrorString(e);
        return false;
    }
    g_exp_ready = true;
    return true;
}

// Persistent grids: this many blocks of 256 threads per SM.
static int sm_count() {
    static int n = 0;
    if (!n) cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, 0);
    return n > 0 ? n : 34;
}

bool gemv_experts_gate_up(uint32_t type, const void* Wg, const void* Wu, uint64_t expert_bytes, const int32_t* ids, int n_sel,
                          const float* x, float* h, int n_out, int n_in, cudaStream_t s, int slots_per_row, int x_stride) {
    constexpr int RPW = EXP_GU_RPW;
    if (type != kIQ2_S || n_in != 2048 || n_out % (8 * RPW) != 0 || n_sel <= 0 || !g_exp_ready) return false;
    const int items = (n_out / (8 * RPW)) * n_sel;
    const int blocks = items < EXP_GU_BLK * sm_count() ? items : EXP_GU_BLK * sm_count();
    gate_up_iq2s_kernel<RPW><<<blocks, 256, 0, s>>>(static_cast<const uint8_t*>(Wg), static_cast<const uint8_t*>(Wu), expert_bytes,
                                                   ids, x, h, n_out, n_sel, slots_per_row, x_stride);
    return cudaGetLastError() == cudaSuccess;
}

bool gemv_experts_down_fast(uint32_t type, const void* W, uint64_t expert_bytes, const int32_t* ids, int n_sel, const float* h,
                            float* y, int n_out, int n_in, cudaStream_t s) {
    constexpr int RPW = EXP_DN_RPW;
    if (type != kIQ3_S || n_in != 512 || n_out % (8 * RPW) != 0 || n_sel <= 0 || !g_exp_ready) return false;
    const int items = (n_out / (8 * RPW)) * n_sel;
    const int blocks = items < EXP_DN_BLK * sm_count() ? items : EXP_DN_BLK * sm_count();
    down_iq3s_kernel<RPW><<<blocks, 256, 0, s>>>(static_cast<const uint8_t*>(W), expert_bytes, ids, h, y, n_out, n_sel);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace limbo::cuda
