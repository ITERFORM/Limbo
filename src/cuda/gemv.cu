// Copyright (c) 2026 ITERFORM. All rights reserved.
// src/cuda/gemv.cu — device GEMV for the rung-1 weight formats, in the GGUF block layout.
//
// Written from the layouts in include/limbo/quants.hpp and the decode math in src/quants.cpp (format facts,
// CLEANROOM.md §3); no ggml kernel was read. The mapping is the same for every format: lane L of a warp
// owns elements [8L, 8L+8) of the current 256-element super-block, so each lane's decode is a closed form
// of (format, L) with no loop-carried state, and a warp reads one super-block per step.
//
//   Q8_0    8 blocks of 34 B per super-block (lane = block*4 + quarter)
//   Q2_K, Q3_K, Q6_K, IQ2_S, IQ3_S, IQ4_XS   one 84/110/210/82/110/136 B block
//   F32, BF16                                 1024 / 512 B of plain values
#include "limbo/cuda/gemv.hpp"
#include "limbo/cuda/gemm.hpp"

#include <cuda_fp16.h>

#include "limbo/quants.hpp"
#include "limbo/types.hpp"

namespace limbo::cuda {

namespace {

constexpr uint32_t kF32 = static_cast<uint32_t>(Type::F32);
constexpr uint32_t kBF16 = static_cast<uint32_t>(Type::BF16);
constexpr uint32_t kQ8_0 = static_cast<uint32_t>(Type::Q8_0);
constexpr uint32_t kQ2_K = static_cast<uint32_t>(Type::Q2_K);
constexpr uint32_t kQ3_K = static_cast<uint32_t>(Type::Q3_K);
constexpr uint32_t kQ6_K = static_cast<uint32_t>(Type::Q6_K);
constexpr uint32_t kIQ2_S = static_cast<uint32_t>(Type::IQ2_S);
constexpr uint32_t kIQ3_S = static_cast<uint32_t>(Type::IQ3_S);
constexpr uint32_t kIQ4_XS = static_cast<uint32_t>(Type::IQ4_XS);
constexpr uint32_t kF16 = static_cast<uint32_t>(Type::F16);
constexpr uint32_t kQ4_K = static_cast<uint32_t>(Type::Q4_K);
constexpr uint32_t kQ5_K = static_cast<uint32_t>(Type::Q5_K);
constexpr uint32_t kIQ2_XXS = static_cast<uint32_t>(Type::IQ2_XXS);
constexpr uint32_t kIQ2_XS = static_cast<uint32_t>(Type::IQ2_XS);
constexpr uint32_t kIQ3_XXS = static_cast<uint32_t>(Type::IQ3_XXS);
constexpr uint32_t kIQ1_S = static_cast<uint32_t>(Type::IQ1_S);
constexpr uint32_t kIQ1_M = static_cast<uint32_t>(Type::IQ1_M);
constexpr uint32_t kIQ4_NL = static_cast<uint32_t>(Type::IQ4_NL);
constexpr uint32_t kQ2_0 = static_cast<uint32_t>(Type::Q2_0);
constexpr uint32_t kNVFP4 = static_cast<uint32_t>(Type::NVFP4);

// Bytes of one 256-element super-block.
__host__ __device__ constexpr int sb_bytes(uint32_t t) {
    switch (t) {
        case kF32: return 1024;
        case kBF16: return 512;
        case kQ8_0: return 8 * 34;
        case kQ2_K: return 84;
        case kQ3_K: return 110;
        case kQ6_K: return 210;
        case kIQ2_S: return 82;
        case kIQ3_S: return 110;
        case kIQ4_XS: return 136;
        case kF16: return 512;
        case kQ4_K: return 144;
        case kQ5_K: return 176;
        case kIQ2_XXS: return 66;
        case kIQ2_XS: return 74;
        case kIQ3_XXS: return 98;
        case kIQ1_S: return 50;
        case kIQ1_M: return 56;
        case kIQ4_NL: return 8 * 18;  // eight 32-element blocks
        case kQ2_0: return 4 * 18;    // four 64-element blocks
        case kNVFP4: return 4 * 36;   // four 64-element blocks
        default: return 0;
    }
}

// The codebooks, in device memory (filled by init_tables from limbo::quants).
__device__ uint64_t d_iq2s_grid[1024];
__device__ uint32_t d_iq3s_grid[512];
__device__ int8_t d_iq4nl[16];
// The grids of the formats added for the 27B and Flash-Next models; read from global memory (cached), not staged in shared.
__device__ uint64_t d_iq2xxs[256];
__device__ uint64_t d_iq2xs[512];
__device__ uint32_t d_iq3xxs[256];
__device__ uint64_t d_iq1s[2048];
__device__ uint8_t d_ksigns[128];
// NVFP4's halved UE4M3 scale decode (src/quants.cpp); the E2M1 codebook is decoded from its bits in
// decode8 below (src/ggml_grids_ext.cpp's doubled table, exact in float).

__device__ __forceinline__ float ue4m3_f(uint8_t x) {
    if (x == 0 || x == 0x7f) return 0.0f;
    const int ex = (x >> 3) & 0xf, man = x & 0x7;
    const float raw = ex == 0 ? ldexpf(float(man), -9) : ldexpf(1.0f + float(man) / 8.0f, ex - 7);
    return raw * 0.5f;
}

struct Tabs {
    const uint64_t* iq2s;
    const uint32_t* iq3s;
    const int8_t* nl;
};

__device__ __forceinline__ float h2f(const uint8_t* p) {
    return __half2float(__ushort_as_half(static_cast<unsigned short>(p[0] | (p[1] << 8))));
}

// float(byte k of w) without the int-to-float pipe (a quarter-rate unit): the byte goes under the mantissa of 2^23 (one PRMT) and
// the bias comes off (one exact FADD). Same value as static_cast<float>((w >> 8k) & 0xff).
template <int K>
__device__ __forceinline__ float ub2f(uint32_t w) {
    return __uint_as_float(__byte_perm(w, 0x4B000000u, (7 << 12) | (4 << 8) | (4 << 4) | K)) - 8388608.0f;
}
// float(int8 byte k of w): flipping the sign bit makes the byte v + 128, and the magic-number conversion is exact.
template <int K>
__device__ __forceinline__ float sb2f(uint32_t w) {
    return ub2f<K>(w ^ 0x80808080u) - 128.0f;
}
// v with its sign flipped when bit `bit` of m is set: the same bits as v * (bit ? -1 : 1).
template <int BIT>
__device__ __forceinline__ float flip(float v, uint32_t m) {
    return __uint_as_float(__float_as_uint(v) ^ ((m << (31 - BIT)) & 0x80000000u));
}

// Decodes the 8 elements [8*lane, 8*lane+8) of the super-block at `sb` into w. The arithmetic mirrors the
// order of operations in src/quants.cpp so the products agree to the last bit before accumulation.
template <uint32_t T>
__device__ __forceinline__ void decode8(const uint8_t* sb, int lane, float* w, const Tabs& tb) {
    if constexpr (T == kF32) {
        const float* p = reinterpret_cast<const float*>(sb) + lane * 8;
#pragma unroll
        for (int i = 0; i < 8; ++i) w[i] = p[i];
    } else if constexpr (T == kBF16) {
        const uint8_t* p = sb + lane * 16;
#pragma unroll
        for (int i = 0; i < 8; ++i) w[i] = __uint_as_float(static_cast<uint32_t>(p[2 * i] | (p[2 * i + 1] << 8)) << 16);
    } else if constexpr (T == kQ8_0) {
        const uint8_t* p = sb + (lane >> 2) * 34;
        const float d = h2f(p);
        const int8_t* q = reinterpret_cast<const int8_t*>(p + 2) + (lane & 3) * 8;
#pragma unroll
        for (int i = 0; i < 8; ++i) w[i] = d * static_cast<float>(q[i]);
    } else if constexpr (T == kQ2_K) {
        const int half = lane >> 4, j = (lane >> 2) & 3, l32 = (lane & 3) * 8;
        const uint8_t sc = sb[half * 8 + j * 2 + ((lane >> 1) & 1)];
        const float d = h2f(sb + 80), dmin = h2f(sb + 82);
        const float dl = d * static_cast<float>(sc & 15), ml = dmin * static_cast<float>(sc >> 4);
        const uint32_t* q = reinterpret_cast<const uint32_t*>(sb + 16 + half * 32 + l32);  // 84-byte blocks: 4-byte aligned
        const uint32_t c0 = (q[0] >> (2 * j)) & 0x03030303u, c1 = (q[1] >> (2 * j)) & 0x03030303u;
        w[0] = dl * ub2f<0>(c0) - ml;
        w[1] = dl * ub2f<1>(c0) - ml;
        w[2] = dl * ub2f<2>(c0) - ml;
        w[3] = dl * ub2f<3>(c0) - ml;
        w[4] = dl * ub2f<0>(c1) - ml;
        w[5] = dl * ub2f<1>(c1) - ml;
        w[6] = dl * ub2f<2>(c1) - ml;
        w[7] = dl * ub2f<3>(c1) - ml;
    } else if constexpr (T == kQ3_K) {
        const int half = lane >> 4, j = (lane >> 2) & 3, l32 = (lane & 3) * 8;
        const int is = half * 8 + j * 2 + ((lane >> 1) & 1);
        // The 16 six-bit scales sit in 12 bytes: low nibbles in bytes 0..7, the high two bits of all
        // sixteen in bytes 8..11. Scale index is = 4*g + i.
        const int g = is >> 2, i4 = is & 3;
        const uint8_t lo_byte = sb[96 + (g & 1) * 4 + i4];
        const int lo = (g >> 1) ? ((lo_byte >> 4) & 15) : (lo_byte & 15);
        const int hi = (sb[96 + 8 + i4] >> (2 * g)) & 3;
        const float d_all = h2f(sb + 108);
        const float dl = d_all * static_cast<float>((lo | (hi << 4)) - 32);
        const uint8_t m = static_cast<uint8_t>(1u << (half * 4 + j));
        const uint8_t* hm = sb + l32;
        const uint8_t* q = sb + 32 + half * 32 + l32;
        const int shift = 2 * j;
#pragma unroll
        for (int i = 0; i < 8; ++i)
            w[i] = dl * static_cast<float>(static_cast<int>((q[i] >> shift) & 3) - ((hm[i] & m) ? 0 : 4));
    } else if constexpr (T == kQ6_K) {
        const int half = lane >> 4, qi = (lane >> 2) & 3, l = (lane & 3) * 8;
        const int sc = static_cast<int8_t>(sb[192 + half * 8 + qi * 2 + ((lane >> 1) & 1)]);
        const float d = h2f(sb + 208);
        const float dsc = d * static_cast<float>(sc);
        const uint8_t* ql = sb + half * 64 + (qi & 1) * 32 + l;
        const uint8_t* qh = sb + 128 + half * 32 + l;
        const int nib = (qi >> 1) * 4;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int q = ((ql[i] >> nib) & 0xF) | (((qh[i] >> (2 * qi)) & 3) << 4);
            w[i] = dsc * static_cast<float>(q - 32);
        }
    } else if constexpr (T == kIQ2_S) {
        const int ib32 = lane >> 2, l4 = lane & 3;
        const float d = h2f(sb);
        const uint8_t scb = sb[74 + ib32];
        const int nib = (l4 >> 1) == 0 ? (scb & 0xf) : (scb >> 4);
        const float dl = d * (0.5f + static_cast<float>(nib)) * 0.25f;
        const unsigned idx = static_cast<unsigned>(sb[2 + ib32 * 4 + l4]) |
                             ((static_cast<unsigned>(sb[66 + ib32]) << (8 - 2 * l4)) & 0x300u);
        const uint64_t grid = tb.iq2s[idx];
        const uint8_t sg = sb[34 + ib32 * 4 + l4];
        const uint32_t g0 = static_cast<uint32_t>(grid), g1 = static_cast<uint32_t>(grid >> 32);
        w[0] = flip<0>(dl * ub2f<0>(g0), sg);
        w[1] = flip<1>(dl * ub2f<1>(g0), sg);
        w[2] = flip<2>(dl * ub2f<2>(g0), sg);
        w[3] = flip<3>(dl * ub2f<3>(g0), sg);
        w[4] = flip<4>(dl * ub2f<0>(g1), sg);
        w[5] = flip<5>(dl * ub2f<1>(g1), sg);
        w[6] = flip<6>(dl * ub2f<2>(g1), sg);
        w[7] = flip<7>(dl * ub2f<3>(g1), sg);
    } else if constexpr (T == kIQ3_S) {
        const int l = lane & 3, p = lane >> 3, pass = (lane >> 2) & 1;
        const float d = h2f(sb);
        const uint8_t scb = sb[106 + p];
        const int nib = pass ? (scb >> 4) : (scb & 0xf);
        const float db = d * (1.f + 2.f * static_cast<float>(nib));
        const unsigned qh = sb[66 + (lane >> 2)];
        const unsigned i1 = static_cast<unsigned>(sb[2 + 2 * lane]) | ((qh << (8 - 2 * l)) & 256u);
        const unsigned i2 = static_cast<unsigned>(sb[2 + 2 * lane + 1]) | ((qh << (7 - 2 * l)) & 256u);
        const uint32_t g1 = tb.iq3s[i1], g2 = tb.iq3s[i2];
        const uint8_t sg = sb[74 + lane];
        w[0] = flip<0>(db * ub2f<0>(g1), sg);
        w[1] = flip<1>(db * ub2f<1>(g1), sg);
        w[2] = flip<2>(db * ub2f<2>(g1), sg);
        w[3] = flip<3>(db * ub2f<3>(g1), sg);
        w[4] = flip<4>(db * ub2f<0>(g2), sg);
        w[5] = flip<5>(db * ub2f<1>(g2), sg);
        w[6] = flip<6>(db * ub2f<2>(g2), sg);
        w[7] = flip<7>(db * ub2f<3>(g2), sg);
    } else if constexpr (T == kIQ4_XS) {
        const int ib = lane >> 2, k = (lane & 3) * 8;
        const float d = h2f(sb);
        const int sh = sb[2] | (sb[3] << 8);
        const int sl = sb[4 + (ib >> 1)];
        const int ls = ((sl >> (4 * (ib & 1))) & 0xf) | (((sh >> (2 * ib)) & 3) << 4);
        const float dl = d * static_cast<float>(ls - 32);
        const uint8_t* qs = sb + 8 + ib * 16 + (k & 15);
        const bool hi = k >= 16;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int nb = hi ? (qs[i] >> 4) : (qs[i] & 0xf);
            w[i] = dl * static_cast<float>(tb.nl[nb]);
        }
    } else if constexpr (T == kF16) {
        const uint8_t* p = sb + lane * 16;
#pragma unroll
        for (int i = 0; i < 8; ++i) w[i] = h2f(p + 2 * i);
    } else if constexpr (T == kQ4_K || T == kQ5_K) {
        // 64-element groups j: elements 64j..64j+31 are the low nibbles of 32 bytes, 64j+32.. the high nibbles, each with its own
        // 6-bit scale and min (the 12 scale bytes pack scales in the low bits, mins in the high bits)
        const int j = lane >> 3, half = (lane >> 2) & 1, l0 = (lane & 3) * 8;
        const int is = 2 * j + half;
        const uint8_t* sc12 = sb + 4;
        uint32_t scv, mnv;
        if (is < 4) {
            scv = sc12[is] & 63;
            mnv = sc12[is + 4] & 63;
        } else {
            scv = (sc12[is + 4] & 0xF) | ((sc12[is - 4] >> 6) << 4);
            mnv = (sc12[is + 4] >> 4) | ((sc12[is] >> 6) << 4);
        }
        const float d = h2f(sb), dmin = h2f(sb + 2);
        const float dl = d * static_cast<float>(scv), ml = dmin * static_cast<float>(mnv);
        if constexpr (T == kQ4_K) {
            const uint8_t* q = sb + 16 + 32 * j + l0;
#pragma unroll
            for (int i = 0; i < 8; ++i) w[i] = dl * static_cast<float>(half ? (q[i] >> 4) : (q[i] & 0xF)) - ml;
        } else {
            const uint8_t* ql = sb + 48 + 32 * j + l0;
            const uint8_t* qh = sb + 16 + l0;
            const uint8_t u = static_cast<uint8_t>(1u << is);
#pragma unroll
            for (int i = 0; i < 8; ++i)
                w[i] = dl * static_cast<float>((half ? (ql[i] >> 4) : (ql[i] & 0xF)) + ((qh[i] & u) ? 16 : 0)) - ml;
        }
    } else if constexpr (T == kIQ2_XXS) {
        const int ib32 = lane >> 2, l = lane & 3;
        const float d = h2f(sb);
        const uint8_t* qs = sb + 2 + ib32 * 8;
        const uint32_t idx_word = *reinterpret_cast<const uint16_t*>(qs) |
                                  (static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(qs + 2)) << 16);
        const uint32_t sign_word = *reinterpret_cast<const uint16_t*>(qs + 4) |
                                   (static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(qs + 6)) << 16);
        const float db = d * (0.5f + static_cast<float>(sign_word >> 28)) * 0.25f;
        const uint64_t grid = d_iq2xxs[(idx_word >> (8 * l)) & 0xff];
        const uint8_t sg = d_ksigns[(sign_word >> (7 * l)) & 127];
        const uint32_t g0 = static_cast<uint32_t>(grid), g1 = static_cast<uint32_t>(grid >> 32);
        w[0] = flip<0>(db * ub2f<0>(g0), sg);
        w[1] = flip<1>(db * ub2f<1>(g0), sg);
        w[2] = flip<2>(db * ub2f<2>(g0), sg);
        w[3] = flip<3>(db * ub2f<3>(g0), sg);
        w[4] = flip<4>(db * ub2f<0>(g1), sg);
        w[5] = flip<5>(db * ub2f<1>(g1), sg);
        w[6] = flip<6>(db * ub2f<2>(g1), sg);
        w[7] = flip<7>(db * ub2f<3>(g1), sg);
    } else if constexpr (T == kIQ2_XS) {
        const int ib32 = lane >> 2, l = lane & 3;
        const float d = h2f(sb);
        const uint8_t* qp = sb + 2 + 2 * (4 * ib32 + l);
        const unsigned q = *reinterpret_cast<const uint16_t*>(qp);
        const uint8_t scb = sb[66 + ib32];
        const float db = d * (0.5f + static_cast<float>((l >> 1) ? (scb >> 4) : (scb & 0xf))) * 0.25f;
        const uint64_t grid = d_iq2xs[q & 511];
        const uint8_t sg = d_ksigns[q >> 9];
        const uint32_t g0 = static_cast<uint32_t>(grid), g1 = static_cast<uint32_t>(grid >> 32);
        w[0] = flip<0>(db * ub2f<0>(g0), sg);
        w[1] = flip<1>(db * ub2f<1>(g0), sg);
        w[2] = flip<2>(db * ub2f<2>(g0), sg);
        w[3] = flip<3>(db * ub2f<3>(g0), sg);
        w[4] = flip<4>(db * ub2f<0>(g1), sg);
        w[5] = flip<5>(db * ub2f<1>(g1), sg);
        w[6] = flip<6>(db * ub2f<2>(g1), sg);
        w[7] = flip<7>(db * ub2f<3>(g1), sg);
    } else if constexpr (T == kIQ3_XXS) {
        const int ib32 = lane >> 2, l = lane & 3;
        const float d = h2f(sb);
        const uint8_t* tail = sb + 2 + 64 + 4 * ib32;
        const uint32_t word = *reinterpret_cast<const uint16_t*>(tail) |
                              (static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(tail + 2)) << 16);
        const float db = d * (0.5f + static_cast<float>(word >> 28)) * 0.5f;
        const uint8_t sg = d_ksigns[(word >> (7 * l)) & 127];
        const uint32_t g1 = d_iq3xxs[sb[2 + ib32 * 8 + 2 * l]], g2 = d_iq3xxs[sb[2 + ib32 * 8 + 2 * l + 1]];
        w[0] = flip<0>(db * ub2f<0>(g1), sg);
        w[1] = flip<1>(db * ub2f<1>(g1), sg);
        w[2] = flip<2>(db * ub2f<2>(g1), sg);
        w[3] = flip<3>(db * ub2f<3>(g1), sg);
        w[4] = flip<4>(db * ub2f<0>(g2), sg);
        w[5] = flip<5>(db * ub2f<1>(g2), sg);
        w[6] = flip<6>(db * ub2f<2>(g2), sg);
        w[7] = flip<7>(db * ub2f<3>(g2), sg);
    } else if constexpr (T == kIQ1_S) {
        const int ib = lane >> 2, l = lane & 3;
        const float d = h2f(sb);
        const unsigned h = *reinterpret_cast<const uint16_t*>(sb + 34 + 2 * ib);
        const float dl = d * static_cast<float>(2 * ((h >> 12) & 7) + 1);
        const float delta = (h & 0x8000) ? -0.125f : 0.125f;
        const uint64_t grid = d_iq1s[static_cast<unsigned>(sb[2 + 4 * ib + l]) | (((h >> (3 * l)) & 7) << 8)];
        const uint32_t g0 = static_cast<uint32_t>(grid), g1 = static_cast<uint32_t>(grid >> 32);
        w[0] = dl * (sb2f<0>(g0) + delta);
        w[1] = dl * (sb2f<1>(g0) + delta);
        w[2] = dl * (sb2f<2>(g0) + delta);
        w[3] = dl * (sb2f<3>(g0) + delta);
        w[4] = dl * (sb2f<0>(g1) + delta);
        w[5] = dl * (sb2f<1>(g1) + delta);
        w[6] = dl * (sb2f<2>(g1) + delta);
        w[7] = dl * (sb2f<3>(g1) + delta);
    } else if constexpr (T == kIQ1_M) {
        const int ib = lane >> 2, l = lane & 3;
        unsigned sc[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) sc[i] = sb[48 + 2 * i] | (sb[49 + 2 * i] << 8);
        const unsigned sbits = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0u) | ((sc[2] >> 4) & 0x0f00u) | (sc[3] & 0xf000u);
        const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(sbits)));
        const unsigned scw = sc[ib >> 1];
        const float dl = d * static_cast<float>(2 * ((scw >> (6 * (ib & 1) + 3 * (l >> 1))) & 7) + 1);
        const unsigned qhb = sb[32 + 2 * ib + (l >> 1)];
        const unsigned idx = static_cast<unsigned>(sb[4 * ib + l]) | ((qhb << ((l & 1) ? 4 : 8)) & 0x700u);
        const float delta = (qhb & ((l & 1) ? 0x80u : 0x08u)) ? -0.125f : 0.125f;
        const uint64_t grid = d_iq1s[idx];
        const uint32_t g0 = static_cast<uint32_t>(grid), g1 = static_cast<uint32_t>(grid >> 32);
        w[0] = dl * (sb2f<0>(g0) + delta);
        w[1] = dl * (sb2f<1>(g0) + delta);
        w[2] = dl * (sb2f<2>(g0) + delta);
        w[3] = dl * (sb2f<3>(g0) + delta);
        w[4] = dl * (sb2f<0>(g1) + delta);
        w[5] = dl * (sb2f<1>(g1) + delta);
        w[6] = dl * (sb2f<2>(g1) + delta);
        w[7] = dl * (sb2f<3>(g1) + delta);
    } else if constexpr (T == kIQ4_NL) {
        const int b = lane >> 2, e0 = (lane & 3) * 8;
        const uint8_t* blk = sb + b * 18;
        const float d = h2f(blk);
        const uint8_t* qs = blk + 2 + (e0 & 15);
        const bool hi = e0 >= 16;
#pragma unroll
        for (int i = 0; i < 8; ++i) w[i] = d * static_cast<float>(tb.nl[hi ? (qs[i] >> 4) : (qs[i] & 0xf)]);
    } else if constexpr (T == kQ2_0) {
        const int b = lane >> 3, j0 = (lane & 7) * 8;
        const uint8_t* blk = sb + b * 18;
        const float d = h2f(blk);
        const uint8_t* qs = blk + 2 + (j0 >> 2);
#pragma unroll
        for (int i = 0; i < 8; ++i) w[i] = static_cast<float>(static_cast<int>((qs[i >> 2] >> (2 * (i & 3))) & 3) - 1) * d;
    } else if constexpr (T == kNVFP4) {
        // { u8 d[4]; u8 qs[32]; } per 64 elements: one UE4M3 scale per 16, nibbles columnar (src/quants.cpp).
        // A lane's 8 elements are one 16-element sub-block's bytes 0..7 -- low nibbles on even lanes, high
        // nibbles on odd ones. Two word loads replace the per-element byte loads and the E2M1 magnitude comes
        // from its own bits ((2 + m1) << (e2 - 1)), so the decode issues no table lookups; the values, and
        // every product, are the ones the codebook form decodes.
        const int b = lane >> 3, s = (lane >> 1) & 3;
        const uint8_t* p = sb + b * 36;
        const float d = ue4m3_f(p[s]);
        const uint32_t* qw = reinterpret_cast<const uint32_t*>(p + 4 + s * 8);
        const uint32_t q0 = qw[0], q1 = qw[1];
        const int sh = (lane & 1) ? 4 : 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t nib = ((i < 4 ? q0 : q1) >> (8 * (i & 3) + sh)) & 15u;
            const int m = int(nib & 7u), e2 = m >> 1, m1 = m & 1;
            const int mag = e2 ? ((2 + m1) << (e2 - 1)) : m1;
            const float v = __int2float_rn(mag);
            w[i] = d * __uint_as_float(__float_as_uint(v) ^ (nib & 8u ? 0x80000000u : 0u));
        }
    }
}

struct GemvArgs {
    const uint8_t* W;
    const uint8_t* W2;  // second matrix of a dual product (blockIdx.z picks), same type and shape
    float* y2;
    const float* x2;    // non-null: the input is silu(x) * x2, formed once per block in shared memory (n_in <= 512)
    uint64_t row_bytes;
    uint64_t expert_bytes;
    const int32_t* ids;  // null for a plain GEMV
    const float* x;
    int x_stride;
    float* y;
    int y_stride;
    int n_out;
    int n_sb;
    int n_tail;  // the row's partial super-block (n_in % 256), summed inside the product instead of a tail launch
    bool acc;
    int x_group = 1;  // the grouped expert form: slot s reads x row s / x_group (x_stride apart)
};

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float silu_g(float x) { return x / (1.0f + expf(-x)); }

template <uint32_t T>
__global__ void __launch_bounds__(256) gemv_kernel(GemvArgs a) {
    if (a.ids && a.ids[blockIdx.y] < 0) return;   // a slot the caller left out (q4e: a lane miss)
    __shared__ float s_h[512];
    __shared__ uint64_t s_iq2s[T == kIQ2_S ? 1024 : 1];
    __shared__ uint32_t s_iq3s[T == kIQ3_S ? 512 : 1];
    __shared__ int8_t s_nl[(T == kIQ4_XS || T == kIQ4_NL) ? 16 : 1];
    Tabs tb{s_iq2s, s_iq3s, s_nl};
    if constexpr (T == kIQ2_S) {
        for (int i = threadIdx.x; i < 1024; i += blockDim.x) s_iq2s[i] = d_iq2s_grid[i];
        __syncthreads();
    } else if constexpr (T == kIQ3_S) {
        for (int i = threadIdx.x; i < 512; i += blockDim.x) s_iq3s[i] = d_iq3s_grid[i];
        __syncthreads();
    } else if constexpr (T == kIQ4_XS || T == kIQ4_NL) {
        if (threadIdx.x < 16) s_nl[threadIdx.x] = d_iq4nl[threadIdx.x];
        __syncthreads();
    }

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + warp;
    const int slot = blockIdx.y;
    const float* x = a.x + static_cast<int64_t>(slot / a.x_group) * a.x_stride;
    if (a.x2) {
        const float* u = a.x2 + static_cast<int64_t>(slot) * a.x_stride;
        for (int i = threadIdx.x; i < a.n_sb * 256; i += blockDim.x) s_h[i] = silu_g(x[i]) * u[i];
        __syncthreads();
        x = s_h;
    }
    if (row >= a.n_out) return;
    const uint8_t* W = blockIdx.z ? a.W2 : a.W;
    if (a.ids) W += static_cast<uint64_t>(a.ids[slot]) * a.expert_bytes;
    const uint8_t* wrow = W + static_cast<uint64_t>(row) * a.row_bytes;

    float acc = 0.f, tail = 0.f;
#pragma unroll 4
    for (int s = 0; s < a.n_sb; ++s) {
        float w[8];
        decode8<T>(wrow + static_cast<uint64_t>(s) * sb_bytes(T), lane, w, tb);
        const float4* xv = reinterpret_cast<const float4*>(x + s * 256 + lane * 8);
        const float4 x0 = a.x2 ? xv[0] : __ldg(xv), x1 = a.x2 ? xv[1] : __ldg(xv + 1);
        acc += w[0] * x0.x;
        acc += w[1] * x0.y;
        acc += w[2] * x0.z;
        acc += w[3] * x0.w;
        acc += w[4] * x1.x;
        acc += w[5] * x1.y;
        acc += w[6] * x1.z;
        acc += w[7] * x1.w;
    }
    // The row's partial super-block, fused here instead of a second launch (the x2 forms take whole
    // super-blocks only). It sums apart so the write reproduces the old main-then-tail association exactly.
    if (a.n_tail > 0 && lane * 8 < a.n_tail) {
        float w[8];
        decode8<T>(wrow + static_cast<uint64_t>(a.n_sb) * sb_bytes(T), lane, w, tb);
        const float4* xv = reinterpret_cast<const float4*>(x + a.n_sb * 256 + lane * 8);
        const float4 x0 = __ldg(xv), x1 = __ldg(xv + 1);
        tail += w[0] * x0.x;
        tail += w[1] * x0.y;
        tail += w[2] * x0.z;
        tail += w[3] * x0.w;
        tail += w[4] * x1.x;
        tail += w[5] * x1.y;
        tail += w[6] * x1.z;
        tail += w[7] * x1.w;
    }
    acc = warp_sum(acc);
    tail = warp_sum(tail);
    if (lane == 0) {
        float* y = (blockIdx.z ? a.y2 : a.y) + static_cast<int64_t>(slot) * a.y_stride + row;
        if (a.acc) {
            float yv = *y;
            if (a.n_sb > 0) yv += acc;
            if (a.n_tail > 0) yv += tail;
            *y = yv;
        } else if (a.n_sb > 0) {
            *y = a.n_tail > 0 ? acc + tail : acc;
        } else {
            *y = tail;
        }
    }
}

// ---- staged GEMV: the same arithmetic as gemv_kernel, the weight bytes arrive through shared memory ----------------------
// A warp still owns a row, but instead of every lane fetching the scattered bytes its decode needs straight from global
// memory (ten or so byte loads per super-block), the warp copies G super-blocks of its row into shared memory with 4-byte
// cp.async (coalesced, NS chunks in flight) and decodes from there. decode8 and the accumulation order are unchanged, so every
// result is bit-identical to gemv_kernel's. Needs an even number of super-blocks per row (every super-block size is even, so a
// row then starts 4-byte aligned and a chunk is a whole number of words).
constexpr int kStageNS = 3;
template <uint32_t T>
__host__ __device__ constexpr int stage_g() { return sb_bytes(T) >= 176 ? 2 : 4; }

__device__ __forceinline__ void cp_async4(void* dst, const void* src) {
    const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(d), "l"(src));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

template <uint32_t T, int R, int CM>
__global__ void __launch_bounds__(256) gemv_staged_kernel(GemvArgs a) {
    if (CM == 1 && a.ids && a.ids[blockIdx.y] < 0) return;
    constexpr int SB = sb_bytes(T), G = stage_g<T>(), NS = R == 1 ? kStageNS : 2;
    // + 4: a row that starts 2 bytes past a word (an odd number of super-blocks of an even size before it) is copied
    // from the word below its start, and decoded from that offset
    __shared__ __align__(16) uint8_t s_w[8][R][NS][G * SB + 4];
    __shared__ uint64_t s_iq2s[T == kIQ2_S ? 1024 : 1];
    __shared__ uint32_t s_iq3s[T == kIQ3_S ? 512 : 1];
    __shared__ int8_t s_nl[(T == kIQ4_XS || T == kIQ4_NL) ? 16 : 1];
    Tabs tb{s_iq2s, s_iq3s, s_nl};
    if constexpr (T == kIQ2_S) {
        for (int i = threadIdx.x; i < 1024; i += blockDim.x) s_iq2s[i] = d_iq2s_grid[i];
        __syncthreads();
    } else if constexpr (T == kIQ3_S) {
        for (int i = threadIdx.x; i < 512; i += blockDim.x) s_iq3s[i] = d_iq3s_grid[i];
        __syncthreads();
    } else if constexpr (T == kIQ4_XS || T == kIQ4_NL) {
        if (threadIdx.x < 16) s_nl[threadIdx.x] = d_iq4nl[threadIdx.x];
        __syncthreads();
    }

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * 8 + warp) * R;  // R consecutive rows per warp, sharing the x loads
    if (row0 >= a.n_out) return;
    const int slot = blockIdx.y;   // the expert slot of the grouped form (CM == 1); 0 otherwise
    const uint8_t* W = blockIdx.z ? a.W2 : a.W;
    if (a.ids) W += static_cast<uint64_t>(a.ids[slot]) * a.expert_bytes;
    const uint8_t* wrow[R];
    int woff[R];   // 0 or 2: where the row starts inside its first copied word
#pragma unroll
    for (int r = 0; r < R; ++r) {
        wrow[r] = W + static_cast<uint64_t>(min(row0 + r, a.n_out - 1)) * a.row_bytes;
        woff[r] = static_cast<int>(reinterpret_cast<uintptr_t>(wrow[r]) & 3u);
    }
    const float* x = a.x + (a.ids ? static_cast<int64_t>(slot / a.x_group) * a.x_stride : 0);
    const int nit = (a.n_sb + G - 1) / G;

    auto issue = [&](int it) {
        if (it < nit) {
            const int g = min(G, a.n_sb - it * G);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                // G * SB is a multiple of 4, so every chunk of the row starts at the row's own offset in its word
                const uint8_t* src = wrow[r] + static_cast<uint64_t>(it) * G * SB - woff[r];
                uint8_t* dst = s_w[warp][r][it % NS];
                const int nw = (woff[r] + g * SB + 3) / 4;
                for (int i = lane; i < nw; i += 32) cp_async4(dst + 4 * i, src + 4 * i);
            }
        }
        cp_async_commit();
    };
#pragma unroll
    for (int i = 0; i < NS - 1; ++i) issue(i);

    constexpr int ncol = CM;  // exactly CM columns (ncols in the args is for the launcher)
    const int xs4 = a.x_stride >> 2;
    float acc[R][CM];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < CM; ++c) acc[r][c] = 0.f;
    for (int it = 0; it < nit; ++it) {
        issue(it + NS - 1);  // the buffer it held, (it - 1) % NS, was read before the previous iteration's syncwarp
        cp_async_wait<NS - 1>();
        __syncwarp();
        const int g = min(G, a.n_sb - it * G);
        const float4* xv = reinterpret_cast<const float4*>(x + static_cast<int64_t>(it) * G * 256 + lane * 8);
        // the whole chunk unrolled: the G super-blocks (and R rows) decode independently, so their loads and lookups overlap
        auto one = [&](int j) {
            float w[R][8];
#pragma unroll
            for (int r = 0; r < R; ++r) decode8<T>(s_w[warp][r][it % NS] + woff[r] + j * SB, lane, w[r], tb);
#pragma unroll
            for (int c = 0; c < CM; ++c)
                {  // one column's x at a time: few live registers, the decoded weights serve every column
                    const float4 x0 = __ldg(xv + c * xs4 + j * 64), x1 = __ldg(xv + c * xs4 + j * 64 + 1);
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        acc[r][c] += w[r][0] * x0.x;
                        acc[r][c] += w[r][1] * x0.y;
                        acc[r][c] += w[r][2] * x0.z;
                        acc[r][c] += w[r][3] * x0.w;
                        acc[r][c] += w[r][4] * x1.x;
                        acc[r][c] += w[r][5] * x1.y;
                        acc[r][c] += w[r][6] * x1.z;
                        acc[r][c] += w[r][7] * x1.w;
                    }
                }
        };
        if (g == G) {
#pragma unroll
            for (int j = 0; j < G; ++j) one(j);
        } else {
            for (int j = 0; j < g; ++j) one(j);
        }
        __syncwarp();
    }
    // The row's partial super-block fused here (the CM == 1 form gemv() runs; the multi-column forms take
    // whole super-blocks only), summed apart so the write reproduces the old main-then-tail association.
    float tacc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) tacc[r] = 0.f;
    if constexpr (CM == 1) {
        if (a.n_tail > 0) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
                if (row0 + r < a.n_out && lane * 8 < a.n_tail) {
                    float w[8];
                    decode8<T>(wrow[r] + static_cast<uint64_t>(a.n_sb) * SB, lane, w, tb);
                    const float4* xv = reinterpret_cast<const float4*>(x + a.n_sb * 256 + lane * 8);
                    const float4 x0 = __ldg(xv), x1 = __ldg(xv + 1);
                    tacc[r] += w[0] * x0.x;
                    tacc[r] += w[1] * x0.y;
                    tacc[r] += w[2] * x0.z;
                    tacc[r] += w[3] * x0.w;
                    tacc[r] += w[4] * x1.x;
                    tacc[r] += w[5] * x1.y;
                    tacc[r] += w[6] * x1.z;
                    tacc[r] += w[7] * x1.w;
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const float t = warp_sum(tacc[r]);
#pragma unroll
        for (int c = 0; c < CM; ++c) {
            const float v = warp_sum(acc[r][c]);
            if (lane == 0 && c < ncol && row0 + r < a.n_out) {
                float* y = (blockIdx.z ? a.y2 : a.y) + static_cast<int64_t>(a.ids ? slot : c) * a.y_stride + row0 + r;
                if (a.acc) {
                    float yv = *y;
                    yv += v;
                    if (a.n_tail > 0) yv += t;
                    *y = yv;
                } else {
                    *y = a.n_tail > 0 ? v + t : v;
                }
            }
        }
    }
}

// ---- staged multi-column GEMV with the x chunk in shared memory ---------------------------------------------------------
// The block's warps walk the super-blocks in step (2 per chunk), so one cp.async copy of the chunk's x for all CM columns serves
// every row of the block, where gemv_staged_kernel reads x through L1 and thrashes it once CM columns of a long row (K = 17408:
// 70 KB each) no longer fit. x sits permuted (unit 2l and 2l+1 of the 8 elements lane l owns go to units l and 32 + l), so a lane's
// two float4 reads are conflict-free. The arithmetic and its order are gemv_staged_kernel's: results are bit-identical.
template <uint32_t T, int R, int CM>
__global__ void __launch_bounds__(256) gemv_xs_kernel(GemvArgs a) {
    constexpr int SB = sb_bytes(T), G = 2, NS = 2;
    __shared__ __align__(16) uint8_t s_w[8][R][NS][G * SB];
    __shared__ __align__(16) float s_x[NS][CM][G * 256];
    __shared__ uint64_t s_iq2s[T == kIQ2_S ? 1024 : 1];
    __shared__ uint32_t s_iq3s[T == kIQ3_S ? 512 : 1];
    __shared__ int8_t s_nl[(T == kIQ4_XS || T == kIQ4_NL) ? 16 : 1];
    Tabs tb{s_iq2s, s_iq3s, s_nl};
    if constexpr (T == kIQ2_S) {
        for (int i = threadIdx.x; i < 1024; i += blockDim.x) s_iq2s[i] = d_iq2s_grid[i];
    } else if constexpr (T == kIQ3_S) {
        for (int i = threadIdx.x; i < 512; i += blockDim.x) s_iq3s[i] = d_iq3s_grid[i];
    } else if constexpr (T == kIQ4_XS || T == kIQ4_NL) {
        if (threadIdx.x < 16) s_nl[threadIdx.x] = d_iq4nl[threadIdx.x];
    }
    __syncthreads();

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * 8 + warp) * R;
    const bool live = row0 < a.n_out;  // a dead warp still takes part in the block's x copies and barriers
    const uint8_t* wrow[R];
#pragma unroll
    for (int r = 0; r < R; ++r) wrow[r] = a.W + static_cast<uint64_t>(min(row0 + r, a.n_out - 1)) * a.row_bytes;
    const int nit = a.n_sb / G;

    auto issue = [&](int it) {
        if (it < nit) {
            const int st = it % NS;
            if (live) {
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const uint8_t* src = wrow[r] + static_cast<uint64_t>(it) * G * SB;
                    uint8_t* dst = s_w[warp][r][st];
                    for (int i = lane; i < G * SB / 4; i += 32) cp_async4(dst + 4 * i, src + 4 * i);
                }
            }
            for (int u = threadIdx.x; u < CM * 128; u += 256) {  // 16-byte units of the chunk's x: column, super-block, unit
                const int c = u >> 7, q = u & 127, sb = q >> 6, e = q & 63;
                const float* src = a.x + static_cast<int64_t>(c) * a.x_stride + static_cast<int64_t>(it) * G * 256 + q * 4;
                float* dst = &s_x[st][c][sb * 256 + ((e & 1) * 32 + (e >> 1)) * 4];
                const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(d), "l"(src));
            }
        }
        cp_async_commit();
    };
    issue(0);
    float acc[R][CM];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < CM; ++c) acc[r][c] = 0.f;
    for (int it = 0; it < nit; ++it) {
        issue(it + 1);
        cp_async_wait<1>();
        __syncthreads();
        const int st = it % NS;
#pragma unroll
        for (int j = 0; j < G; ++j) {
            float w[R][8];
#pragma unroll
            for (int r = 0; r < R; ++r) decode8<T>(s_w[warp][r][st] + j * SB, lane, w[r], tb);
#pragma unroll
            for (int c = 0; c < CM; ++c) {
                const float4 x0 = *reinterpret_cast<const float4*>(&s_x[st][c][j * 256 + lane * 4]);
                const float4 x1 = *reinterpret_cast<const float4*>(&s_x[st][c][j * 256 + 128 + lane * 4]);
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    acc[r][c] += w[r][0] * x0.x;
                    acc[r][c] += w[r][1] * x0.y;
                    acc[r][c] += w[r][2] * x0.z;
                    acc[r][c] += w[r][3] * x0.w;
                    acc[r][c] += w[r][4] * x1.x;
                    acc[r][c] += w[r][5] * x1.y;
                    acc[r][c] += w[r][6] * x1.z;
                    acc[r][c] += w[r][7] * x1.w;
                }
            }
        }
        __syncthreads();  // the chunk is free for the copy two iterations on
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < CM; ++c) {
            const float v = warp_sum(acc[r][c]);
            if (lane == 0 && live && row0 + r < a.n_out) {
                float* y = a.y + static_cast<int64_t>(c) * a.y_stride + row0 + r;
                *y = a.acc ? *y + v : v;
            }
        }
}

template <uint32_t T>
__global__ void embed_kernel(const uint8_t* W, uint64_t row_bytes, const int32_t* token, float* out) {
    __shared__ uint64_t s_iq2s[T == kIQ2_S ? 1024 : 1];
    __shared__ uint32_t s_iq3s[T == kIQ3_S ? 512 : 1];
    __shared__ int8_t s_nl[(T == kIQ4_XS || T == kIQ4_NL) ? 16 : 1];
    Tabs tb{s_iq2s, s_iq3s, s_nl};
    if constexpr (T == kIQ2_S) {
        for (int i = threadIdx.x; i < 1024; i += blockDim.x) s_iq2s[i] = d_iq2s_grid[i];
        __syncthreads();
    } else if constexpr (T == kIQ3_S) {
        for (int i = threadIdx.x; i < 512; i += blockDim.x) s_iq3s[i] = d_iq3s_grid[i];
        __syncthreads();
    } else if constexpr (T == kIQ4_XS || T == kIQ4_NL) {
        if (threadIdx.x < 16) s_nl[threadIdx.x] = d_iq4nl[threadIdx.x];
        __syncthreads();
    }
    const int lane = threadIdx.x;
    const uint8_t* row = W + static_cast<uint64_t>(*token) * row_bytes;
    float w[8];
    decode8<T>(row + static_cast<uint64_t>(blockIdx.x) * sb_bytes(T), lane, w, tb);
    float* o = out + blockIdx.x * 256 + lane * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) o[i] = w[i];
}

// The batched form: blockIdx.y picks the token.
template <uint32_t T>
__global__ void embed_rows_kernel(const uint8_t* W, uint64_t row_bytes, const int32_t* tokens, float* out, int n_in) {
    __shared__ uint64_t s_iq2s[T == kIQ2_S ? 1024 : 1];
    __shared__ uint32_t s_iq3s[T == kIQ3_S ? 512 : 1];
    __shared__ int8_t s_nl[(T == kIQ4_XS || T == kIQ4_NL) ? 16 : 1];
    Tabs tb{s_iq2s, s_iq3s, s_nl};
    if constexpr (T == kIQ2_S) {
        for (int i = threadIdx.x; i < 1024; i += blockDim.x) s_iq2s[i] = d_iq2s_grid[i];
        __syncthreads();
    } else if constexpr (T == kIQ3_S) {
        for (int i = threadIdx.x; i < 512; i += blockDim.x) s_iq3s[i] = d_iq3s_grid[i];
        __syncthreads();
    } else if constexpr (T == kIQ4_XS || T == kIQ4_NL) {
        if (threadIdx.x < 16) s_nl[threadIdx.x] = d_iq4nl[threadIdx.x];
        __syncthreads();
    }
    const int lane = threadIdx.x;
    const uint8_t* row = W + static_cast<uint64_t>(tokens[blockIdx.y]) * row_bytes;
    float w[8];
    decode8<T>(row + static_cast<uint64_t>(blockIdx.x) * sb_bytes(T), lane, w, tb);
    float* o = out + static_cast<int64_t>(blockIdx.y) * n_in + blockIdx.x * 256 + lane * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) o[i] = w[i];
}

// C columns at once: y[c][r] = W[r, :] . x[c] for c < C, one launch, each weight decoded once. Per column the
// arithmetic is exactly gemv_kernel's (the same eight products per lane and super-block in the same order, the same
// shuffle tree), so a column's result does not depend on C and equals the single-column product bit for bit.
template <uint32_t T, int C>
__global__ void __launch_bounds__(256) gemv_cols_kernel(const uint8_t* __restrict__ W, uint64_t row_bytes, const float* __restrict__ x,
                                                       int x_stride, float* y, int y_stride, int n_out, int n_sb, int n_tail, bool accumulate) {
    __shared__ uint64_t s_iq2s[T == kIQ2_S ? 1024 : 1];
    __shared__ uint32_t s_iq3s[T == kIQ3_S ? 512 : 1];
    __shared__ int8_t s_nl[(T == kIQ4_XS || T == kIQ4_NL) ? 16 : 1];
    Tabs tb{s_iq2s, s_iq3s, s_nl};
    if constexpr (T == kIQ2_S) {
        for (int i = threadIdx.x; i < 1024; i += blockDim.x) s_iq2s[i] = d_iq2s_grid[i];
        __syncthreads();
    } else if constexpr (T == kIQ3_S) {
        for (int i = threadIdx.x; i < 512; i += blockDim.x) s_iq3s[i] = d_iq3s_grid[i];
        __syncthreads();
    } else if constexpr (T == kIQ4_XS || T == kIQ4_NL) {
        if (threadIdx.x < 16) s_nl[threadIdx.x] = d_iq4nl[threadIdx.x];
        __syncthreads();
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + warp;
    if (row >= n_out) return;
    const uint8_t* wrow = W + static_cast<uint64_t>(row) * row_bytes;
    float acc[C];
#pragma unroll
    for (int c = 0; c < C; ++c) acc[c] = 0.f;
#pragma unroll 2
    for (int s = 0; s < n_sb; ++s) {
        float w[8];
        decode8<T>(wrow + static_cast<uint64_t>(s) * sb_bytes(T), lane, w, tb);
#pragma unroll
        for (int c = 0; c < C; ++c) {
            const float4* xv = reinterpret_cast<const float4*>(x + static_cast<int64_t>(c) * x_stride + s * 256 + lane * 8);
            const float4 x0 = __ldg(xv), x1 = __ldg(xv + 1);
            acc[c] += w[0] * x0.x;
            acc[c] += w[1] * x0.y;
            acc[c] += w[2] * x0.z;
            acc[c] += w[3] * x0.w;
            acc[c] += w[4] * x1.x;
            acc[c] += w[5] * x1.y;
            acc[c] += w[6] * x1.z;
            acc[c] += w[7] * x1.w;
        }
    }
    // the row's partial super-block, summed apart as gemv_kernel does (so the bits stay the single column's)
    float tl[C];
#pragma unroll
    for (int c = 0; c < C; ++c) tl[c] = 0.f;
    if (n_tail > 0 && lane * 8 < n_tail) {
        float w[8];
        decode8<T>(wrow + static_cast<uint64_t>(n_sb) * sb_bytes(T), lane, w, tb);
#pragma unroll
        for (int c = 0; c < C; ++c) {
            const float4* xv = reinterpret_cast<const float4*>(x + static_cast<int64_t>(c) * x_stride + n_sb * 256 + lane * 8);
            const float4 x0 = __ldg(xv), x1 = __ldg(xv + 1);
            tl[c] += w[0] * x0.x;
            tl[c] += w[1] * x0.y;
            tl[c] += w[2] * x0.z;
            tl[c] += w[3] * x0.w;
            tl[c] += w[4] * x1.x;
            tl[c] += w[5] * x1.y;
            tl[c] += w[6] * x1.z;
            tl[c] += w[7] * x1.w;
        }
    }
#pragma unroll
    for (int c = 0; c < C; ++c) {
        const float r = warp_sum(acc[c]), t = warp_sum(tl[c]);
        if (lane == 0) {
            float* yp = y + static_cast<int64_t>(c) * y_stride + row;
            if (accumulate) {
                float yv = *yp;
                if (n_sb > 0) yv += r;
                if (n_tail > 0) yv += t;
                *yp = yv;
            } else if (n_sb > 0) {
                *yp = n_tail > 0 ? r + t : r;
            } else {
                *yp = t;
            }
        }
    }
}

template <uint32_t T, int C>
void launch_cols(const uint8_t* W, uint64_t rb, const float* x, int xs, float* y, int ys, int n_out, int n_sb, int n_tail, bool acc,
                 cudaStream_t s) {
    gemv_cols_kernel<T, C><<<static_cast<unsigned>((n_out + 7) / 8), 256, 0, s>>>(W, rb, x, xs, y, ys, n_out, n_sb, n_tail, acc);
}

template <uint32_t T>
bool dispatch_cols(int C, const uint8_t* W, uint64_t rb, const float* x, int xs, float* y, int ys, int n_out, int n_sb, int n_tail, bool acc,
                   cudaStream_t s) {
    switch (C) {
        case 1: launch_cols<T, 1>(W, rb, x, xs, y, ys, n_out, n_sb, n_tail, acc, s); break;
        case 2: launch_cols<T, 2>(W, rb, x, xs, y, ys, n_out, n_sb, n_tail, acc, s); break;
        case 3: launch_cols<T, 3>(W, rb, x, xs, y, ys, n_out, n_sb, n_tail, acc, s); break;
        case 4: launch_cols<T, 4>(W, rb, x, xs, y, ys, n_out, n_sb, n_tail, acc, s); break;
        case 5: launch_cols<T, 5>(W, rb, x, xs, y, ys, n_out, n_sb, n_tail, acc, s); break;
        case 6: launch_cols<T, 6>(W, rb, x, xs, y, ys, n_out, n_sb, n_tail, acc, s); break;
        case 7: launch_cols<T, 7>(W, rb, x, xs, y, ys, n_out, n_sb, n_tail, acc, s); break;
        case 8: launch_cols<T, 8>(W, rb, x, xs, y, ys, n_out, n_sb, n_tail, acc, s); break;
        default: return false;
    }
    return cudaGetLastError() == cudaSuccess;
}

bool g_tables_ready = false;

bool g_staged = true;
bool g_rows2 = true;
bool g_xs = true;
// 4 rows a warp (x loads shared by more rows) from g_r4_cols columns on, for matrices of g_r4_rows rows or more
int env_int(const char* k, int d) {
    const char* v = std::getenv(k);
    return v && *v ? std::atoi(v) : d;
}
int g_r4_cols = env_int("LIMBO_GEMV_R4_COLS", 2);
int g_r4_rows = env_int("LIMBO_GEMV_R4_ROWS", 2048);

template <uint32_t T>
constexpr bool staged_type() {
    return T != kF32 && T != kBF16 && T != kF16 && T != kQ6_K;
}

template <uint32_t T>
bool launch_gemv(const GemvArgs& a, int n_slots, int n_z, cudaStream_t s) {
    if constexpr (staged_type<T>()) {
        // any super-block count and the grouped expert form too: a row that starts mid-word is copied from the word below
        // (G * SB is a multiple of 4, so only the row start can be misaligned, and super-blocks are an even size)
        if (g_staged && !a.x2 && a.n_sb > 0 && (a.ids || n_slots == 1)) {
            if (g_rows2) {
                const dim3 grid(static_cast<unsigned>((a.n_out + 15) / 16), static_cast<unsigned>(n_slots),
                                static_cast<unsigned>(n_z));
                gemv_staged_kernel<T, 2, 1><<<grid, 256, 0, s>>>(a);
            } else {
                const dim3 grid(static_cast<unsigned>((a.n_out + 7) / 8), static_cast<unsigned>(n_slots),
                                static_cast<unsigned>(n_z));
                gemv_staged_kernel<T, 1, 1><<<grid, 256, 0, s>>>(a);
            }
            return cudaGetLastError() == cudaSuccess;
        }
    }
    const dim3 grid(static_cast<unsigned>((a.n_out + 7) / 8), static_cast<unsigned>(n_slots), static_cast<unsigned>(n_z));
    gemv_kernel<T><<<grid, 256, 0, s>>>(a);
    return cudaGetLastError() == cudaSuccess;
}

// C = 2..8 columns through the staged kernel (the weights are read once for all of them; each column's bits are the single-column ones).
template <uint32_t T>
bool launch_staged_cols(GemvArgs a, int C, cudaStream_t s) {
    if constexpr (staged_type<T>()) {
        if (g_r4_cols > 0 && C >= g_r4_cols && a.n_out >= g_r4_rows) {   // 4 rows a warp
            const dim3 grid4(static_cast<unsigned>((a.n_out + 31) / 32), 1, 1);
            if constexpr (sb_bytes(T) <= 176) {
                if (g_xs && a.n_sb >= 32 && C <= 4 && a.n_sb % 2 == 0) {
                    switch (C) {
                        case 2: gemv_xs_kernel<T, 4, 2><<<grid4, 256, 0, s>>>(a); break;
                        case 3: gemv_xs_kernel<T, 4, 3><<<grid4, 256, 0, s>>>(a); break;
                        default: gemv_xs_kernel<T, 4, 4><<<grid4, 256, 0, s>>>(a); break;
                    }
                    return cudaGetLastError() == cudaSuccess;
                }
            }
            switch (C) {
                case 2: gemv_staged_kernel<T, 4, 2><<<grid4, 256, 0, s>>>(a); break;
                case 3: gemv_staged_kernel<T, 4, 3><<<grid4, 256, 0, s>>>(a); break;
                case 4: gemv_staged_kernel<T, 4, 4><<<grid4, 256, 0, s>>>(a); break;
                case 5: gemv_staged_kernel<T, 4, 5><<<grid4, 256, 0, s>>>(a); break;
                case 6: gemv_staged_kernel<T, 4, 6><<<grid4, 256, 0, s>>>(a); break;
                case 7: gemv_staged_kernel<T, 4, 7><<<grid4, 256, 0, s>>>(a); break;
                case 8: gemv_staged_kernel<T, 4, 8><<<grid4, 256, 0, s>>>(a); break;
                default: return false;
            }
            return cudaGetLastError() == cudaSuccess;
        }
        const dim3 grid(static_cast<unsigned>((a.n_out + 15) / 16), 1, 1);
        if (g_xs && a.n_sb >= 32 && C <= 4 && a.n_sb % 2 == 0) {  // long rows: x through shared memory
            switch (C) {
                case 2: gemv_xs_kernel<T, 2, 2><<<grid, 256, 0, s>>>(a); break;
                case 3: gemv_xs_kernel<T, 2, 3><<<grid, 256, 0, s>>>(a); break;
                default: gemv_xs_kernel<T, 2, 4><<<grid, 256, 0, s>>>(a); break;
            }
            return cudaGetLastError() == cudaSuccess;
        }
        switch (C) {
            case 2: gemv_staged_kernel<T, 2, 2><<<grid, 256, 0, s>>>(a); break;
            case 3: gemv_staged_kernel<T, 2, 3><<<grid, 256, 0, s>>>(a); break;
            case 4: gemv_staged_kernel<T, 2, 4><<<grid, 256, 0, s>>>(a); break;
            case 5: gemv_staged_kernel<T, 2, 5><<<grid, 256, 0, s>>>(a); break;
            case 6: gemv_staged_kernel<T, 2, 6><<<grid, 256, 0, s>>>(a); break;
            case 7: gemv_staged_kernel<T, 2, 7><<<grid, 256, 0, s>>>(a); break;
            case 8: gemv_staged_kernel<T, 2, 8><<<grid, 256, 0, s>>>(a); break;
            default: return false;
        }
        return cudaGetLastError() == cudaSuccess;
    }
    return false;
}

bool staged_cols(uint32_t type, const GemvArgs& a, int C, cudaStream_t s) {
    switch (type) {
        case kQ8_0: return launch_staged_cols<kQ8_0>(a, C, s);
        case kQ2_K: return launch_staged_cols<kQ2_K>(a, C, s);
        case kQ3_K: return launch_staged_cols<kQ3_K>(a, C, s);
        case kIQ2_S: return launch_staged_cols<kIQ2_S>(a, C, s);
        case kIQ3_S: return launch_staged_cols<kIQ3_S>(a, C, s);
        case kIQ4_XS: return launch_staged_cols<kIQ4_XS>(a, C, s);
        case kQ4_K: return launch_staged_cols<kQ4_K>(a, C, s);
        case kQ5_K: return launch_staged_cols<kQ5_K>(a, C, s);
        case kIQ2_XXS: return launch_staged_cols<kIQ2_XXS>(a, C, s);
        case kIQ2_XS: return launch_staged_cols<kIQ2_XS>(a, C, s);
        case kIQ3_XXS: return launch_staged_cols<kIQ3_XXS>(a, C, s);
        case kIQ1_S: return launch_staged_cols<kIQ1_S>(a, C, s);
        case kIQ1_M: return launch_staged_cols<kIQ1_M>(a, C, s);
        case kIQ4_NL: return launch_staged_cols<kIQ4_NL>(a, C, s);
        case kQ2_0: return launch_staged_cols<kQ2_0>(a, C, s);
        case kNVFP4: return launch_staged_cols<kNVFP4>(a, C, s);
        default: return false;
    }
}

bool dispatch(uint32_t type, const GemvArgs& a, int n_slots, cudaStream_t s, int n_z = 1) {
    switch (type) {
        case kF32: return launch_gemv<kF32>(a, n_slots, n_z, s);
        case kBF16: return launch_gemv<kBF16>(a, n_slots, n_z, s);
        case kQ8_0: return launch_gemv<kQ8_0>(a, n_slots, n_z, s);
        case kQ2_K: return launch_gemv<kQ2_K>(a, n_slots, n_z, s);
        case kQ3_K: return launch_gemv<kQ3_K>(a, n_slots, n_z, s);
        case kQ6_K: return launch_gemv<kQ6_K>(a, n_slots, n_z, s);
        case kIQ2_S: return launch_gemv<kIQ2_S>(a, n_slots, n_z, s);
        case kIQ3_S: return launch_gemv<kIQ3_S>(a, n_slots, n_z, s);
        case kIQ4_XS: return launch_gemv<kIQ4_XS>(a, n_slots, n_z, s);
        case kF16: return launch_gemv<kF16>(a, n_slots, n_z, s);
        case kQ4_K: return launch_gemv<kQ4_K>(a, n_slots, n_z, s);
        case kQ5_K: return launch_gemv<kQ5_K>(a, n_slots, n_z, s);
        case kIQ2_XXS: return launch_gemv<kIQ2_XXS>(a, n_slots, n_z, s);
        case kIQ2_XS: return launch_gemv<kIQ2_XS>(a, n_slots, n_z, s);
        case kIQ3_XXS: return launch_gemv<kIQ3_XXS>(a, n_slots, n_z, s);
        case kIQ1_S: return launch_gemv<kIQ1_S>(a, n_slots, n_z, s);
        case kIQ1_M: return launch_gemv<kIQ1_M>(a, n_slots, n_z, s);
        case kIQ4_NL: return launch_gemv<kIQ4_NL>(a, n_slots, n_z, s);
        case kQ2_0: return launch_gemv<kQ2_0>(a, n_slots, n_z, s);
        case kNVFP4: return launch_gemv<kNVFP4>(a, n_slots, n_z, s);
        default: return false;
    }
}

// The row stride is the file's own row bytes: for n_in a multiple of 256 this equals (n_in / 256) * sb_bytes,
// and for a row with a tail (n_in % 256, qwen4exp's hc_up 320 and down projections 640) it carries the tail too.
uint64_t row_bytes_of(uint32_t type, int n_in) { return type_bytes(type, static_cast<uint64_t>(n_in)); }

}  // namespace

bool init_tables(std::string* err) {
    if (g_tables_ready) return true;
    cudaError_t e = cudaMemcpyToSymbol(d_iq2s_grid, quants::kIq2sGrid, sizeof(uint64_t) * 1024);
    if (e == cudaSuccess) e = cudaMemcpyToSymbol(d_iq3s_grid, quants::kIq3sGrid, sizeof(uint32_t) * 512);
    if (e == cudaSuccess) e = cudaMemcpyToSymbol(d_iq4nl, quants::kIq4NlValues, 16);
    if (e != cudaSuccess) {
        if (err) *err = std::string("init_tables: ") + cudaGetErrorString(e);
        return false;
    }
    if (e == cudaSuccess) e = cudaMemcpyToSymbol(d_iq2xxs, quants::kIq2xxsGrid, sizeof(uint64_t) * 256);
    if (e == cudaSuccess) e = cudaMemcpyToSymbol(d_iq2xs, quants::kIq2xsGrid, sizeof(uint64_t) * 512);
    if (e == cudaSuccess) e = cudaMemcpyToSymbol(d_iq3xxs, quants::kIq3xxsGrid, sizeof(uint32_t) * 256);
    if (e == cudaSuccess) e = cudaMemcpyToSymbol(d_iq1s, quants::kIq1sGrid, sizeof(uint64_t) * 2048);
    if (e == cudaSuccess) e = cudaMemcpyToSymbol(d_ksigns, quants::kSignsIq2xs, 128);
    if (e != cudaSuccess) {
        if (err) *err = std::string("init_tables: ") + cudaGetErrorString(e);
        return false;
    }
    if (!exp_init(err)) return false;
    g_tables_ready = true;
    return true;
}

bool gemv_supported(uint32_t type) { return sb_bytes(type) != 0; }

void gemv_set_staged(bool on) { g_staged = on; }
void gemv_set_rows2(bool on) { g_rows2 = on; }
void gemv_set_xs(bool on) { g_xs = on; }

namespace {

// A matrix of a few rows over a long input (qwen4exp's hc inject: 4 rows of 10240). gemv_kernel gives each row one warp,
// which walks the whole row alone; here one block takes the row, its 8 warps split the super-blocks, every column
// shares each decoded weight, and the warps meet in a fixed order. Plain value formats only (no grid tables).
template <uint32_t T, int C>
__global__ void __launch_bounds__(256) gemv_narrow_kernel(const uint8_t* __restrict__ W, uint64_t row_bytes, const float* __restrict__ x,
                                                          int x_stride, float* y, int y_stride, int n_sb, int n_tail, bool acc) {
    __shared__ float sm[C][8];
    const Tabs tb{nullptr, nullptr, nullptr};
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x;
    const uint8_t* wrow = W + static_cast<uint64_t>(row) * row_bytes;
    float sum[C];
#pragma unroll
    for (int c = 0; c < C; ++c) sum[c] = 0.f;
    const int n_all = n_sb + (n_tail > 0 ? 1 : 0);
    for (int sb = warp; sb < n_all; sb += 8) {
        if (sb == n_sb && lane * 8 >= n_tail) break;   // the partial super-block: its first n_tail elements
        float w[8];
        decode8<T>(wrow + static_cast<uint64_t>(sb) * sb_bytes(T), lane, w, tb);
#pragma unroll
        for (int c = 0; c < C; ++c) {
            const float4* xv = reinterpret_cast<const float4*>(x + static_cast<int64_t>(c) * x_stride + sb * 256 + lane * 8);
            const float4 x0 = __ldg(xv), x1 = __ldg(xv + 1);
            sum[c] += w[0] * x0.x + w[1] * x0.y + w[2] * x0.z + w[3] * x0.w + w[4] * x1.x + w[5] * x1.y + w[6] * x1.z + w[7] * x1.w;
        }
    }
#pragma unroll
    for (int c = 0; c < C; ++c) {
        float v = sum[c];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) sm[c][warp] = v;
    }
    __syncthreads();
    if (threadIdx.x < C) {
        const int c = threadIdx.x;
        float v = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) v += sm[c][i];
        float* out = y + static_cast<int64_t>(c) * y_stride + row;
        *out = acc ? *out + v : v;
    }
}

template <uint32_t T>
bool launch_narrow(int C, const uint8_t* W, uint64_t rb, const float* x, int x_stride, float* y, int y_stride, int n_out, int n_sb,
                   int n_tail, bool acc, cudaStream_t s) {
    switch (C) {
#define LIMBO_NARROW(K) \
    case K: gemv_narrow_kernel<T, K><<<n_out, 256, 0, s>>>(W, rb, x, x_stride, y, y_stride, n_sb, n_tail, acc); break;
        LIMBO_NARROW(1) LIMBO_NARROW(2) LIMBO_NARROW(3) LIMBO_NARROW(4) LIMBO_NARROW(5) LIMBO_NARROW(6) LIMBO_NARROW(7) LIMBO_NARROW(8)
#undef LIMBO_NARROW
        default: return false;
    }
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace

bool gemv_narrow(uint32_t type, const void* W, const float* x, int x_stride, float* y, int y_stride, int n_out, int n_in, int C,
                 bool accumulate, cudaStream_t s) {
    const int n_tail = n_in % 256;
    if (n_out <= 0 || n_out > 64 || C < 1 || C > 8 || n_tail % 8 != 0 || (C > 1 && x_stride % 4 != 0)) return false;
    if (type != kBF16 && type != kF32 && type != kF16) return false;
    const auto* w = static_cast<const uint8_t*>(W);
    const uint64_t rb = row_bytes_of(type, n_in);
    const int nsb = n_in / 256;
    if (type == kBF16) return launch_narrow<kBF16>(C, w, rb, x, x_stride, y, y_stride, n_out, nsb, n_tail, accumulate, s);
    if (type == kF16) return launch_narrow<kF16>(C, w, rb, x, x_stride, y, y_stride, n_out, nsb, n_tail, accumulate, s);
    return launch_narrow<kF32>(C, w, rb, x, x_stride, y, y_stride, n_out, nsb, n_tail, accumulate, s);
}

bool gemv(uint32_t type, const void* W, const float* x, float* y, int n_out, int n_in, bool accumulate,
          cudaStream_t s) {
    // the fragment kernel when it takes the product (gemm.cu): a step's row has a window column's bits
    if (n_in > 0 && n_out > 0 && frag_cols(type, W, x, n_in, y, n_out, n_out, n_in, 1, accumulate, s)) return true;
    if (!gemv_supported(type) || n_in <= 0 || n_out <= 0) return false;
    // A row is a whole number of the format's blocks. The elements past the last full super-block
    // (n_in % 256; qwen4exp's hc_up 320 and the down projections 640) are one partial super-block of
    // 8 elements per lane, so the tail must be a multiple of 8.
    const int n_tail = n_in % 256;
    if (type_bytes(type, static_cast<uint64_t>(n_in)) == 0 || n_tail % 8 != 0) return false;
    if (type == kQ6_K && n_tail == 0 &&
        q6k_launch(W, nullptr, x, nullptr, 0, y, nullptr, 0, n_out, n_in, 1, accumulate, 1, s)) return true;
    GemvArgs a{};
    a.W = static_cast<const uint8_t*>(W);
    a.row_bytes = row_bytes_of(type, n_in);
    a.x = x;
    a.y = y;
    a.n_out = n_out;
    a.n_sb = n_in / 256;
    a.n_tail = n_tail;
    a.acc = accumulate;
    // The row's partial super-block is summed inside the product (GemvArgs::n_tail), never a second launch.
    return dispatch(type, a, 1, s);
}

bool gemv_cols(uint32_t type, const void* W, const float* x, int x_stride, float* y, int y_stride, int n_out, int n_in, int C,
               bool accumulate, cudaStream_t s) {
    const int n_tail = n_in % 256;   // a row's partial super-block, as gemv() takes it
    // up to 16 columns through the fragment kernel when it takes the product, else up to 8 FP32
    if (C >= 1 && C <= 16 && n_tail == 0 && frag_cols(type, W, x, x_stride, y, y_stride, n_out, n_in, C, accumulate, s)) return true;
    if (!gemv_supported(type) || type_bytes(type, uint64_t(n_in)) == 0 || n_tail % 8 != 0 || n_out <= 0 || C < 1 || C > 8) return false;
    if (C > 1 && x_stride % 4 != 0) {   // the column kernels load x as float4
        for (int c = 0; c < C; ++c)
            if (!gemv(type, W, x + static_cast<int64_t>(c) * x_stride, y + static_cast<int64_t>(c) * y_stride, n_out, n_in, accumulate, s)) return false;
        return true;
    }
    if (n_tail == 0 && type == kQ6_K && q6k_launch(W, nullptr, x, nullptr, x_stride, y, nullptr, y_stride, n_out, n_in, C, accumulate, 1, s)) return true;
    const uint8_t* w = static_cast<const uint8_t*>(W);
    const uint64_t rb = row_bytes_of(type, n_in);
    const int nsb = n_in / 256;
    if (C >= 2 && g_staged && n_tail == 0 && nsb % 2 == 0 && x_stride % 4 == 0) {
        GemvArgs a{};
        a.W = w;
        a.row_bytes = rb;
        a.x = x;
        a.x_stride = x_stride;
        a.y = y;
        a.y_stride = y_stride;
        a.n_out = n_out;
        a.n_sb = nsb;
        a.acc = accumulate;
        if (staged_cols(type, a, C, s)) return true;
        cudaGetLastError();
    }
    switch (type) {
        case kF32: return dispatch_cols<kF32>(C, w, rb, x, x_stride, y, y_stride, n_out, nsb, n_tail, accumulate, s);
        case kBF16: return dispatch_cols<kBF16>(C, w, rb, x, x_stride, y, y_stride, n_out, nsb, n_tail, accumulate, s);
        case kQ8_0: return dispatch_cols<kQ8_0>(C, w, rb, x, x_stride, y, y_stride, n_out, nsb, n_tail, accumulate, s);
        case kQ2_K: return dispatch_cols<kQ2_K>(C, w, rb, x, x_stride, y, y_stride, n_out, nsb, n_tail, accumulate, s);
        case kQ3_K: return dispatch_cols<kQ3_K>(C, w, rb, x, x_stride, y, y_stride, n_out, nsb, n_tail, accumulate, s);
        case kQ6_K: return dispatch_cols<kQ6_K>(C, w, rb, x, x_stride, y, y_stride, n_out, nsb, n_tail, accumulate, s);
        case kIQ2_S: return dispatch_cols<kIQ2_S>(C, w, rb, x, x_stride, y, y_stride, n_out, nsb, n_tail, accumulate, s);
        case kIQ3_S: return dispatch_cols<kIQ3_S>(C, w, rb, x, x_stride, y, y_stride, n_out, nsb, n_tail, accumulate, s);
        case kIQ4_XS: return dispatch_cols<kIQ4_XS>(C, w, rb, x, x_stride, y, y_stride, n_out, nsb, n_tail, accumulate, s);
        case kF16: return dispatch_cols<kF16>(C, w, rb, x, x_stride, y, y_stride, n_out, nsb, n_tail, accumulate, s);
        case kIQ3_XXS: return dispatch_cols<kIQ3_XXS>(C, w, rb, x, x_stride, y, y_stride, n_out, nsb, n_tail, accumulate, s);
        case kQ2_0: return dispatch_cols<kQ2_0>(C, w, rb, x, x_stride, y, y_stride, n_out, nsb, n_tail, accumulate, s);
        case kIQ4_NL: return dispatch_cols<kIQ4_NL>(C, w, rb, x, x_stride, y, y_stride, n_out, nsb, n_tail, accumulate, s);
        case kQ5_K: return dispatch_cols<kQ5_K>(C, w, rb, x, x_stride, y, y_stride, n_out, nsb, n_tail, accumulate, s);
        case kQ4_K: return dispatch_cols<kQ4_K>(C, w, rb, x, x_stride, y, y_stride, n_out, nsb, n_tail, accumulate, s);
        default: break;
    }
    // the remaining formats: one launch per column (each column is the single-column product, so the bits do not depend on C)
    for (int c = 0; c < C; ++c)
        if (!gemv(type, W, x + static_cast<int64_t>(c) * x_stride, y + static_cast<int64_t>(c) * y_stride, n_out, n_in, accumulate, s)) return false;
    return true;
}

bool gemv_experts(uint32_t type, const void* W, uint64_t expert_bytes, const int32_t* ids, int n_sel,
                  const float* x, int x_stride, float* y, int n_out, int n_in, cudaStream_t s, int x_group) {
    if (!gemv_supported(type) || n_out <= 0 || n_sel <= 0 || x_group < 1) return false;
    // A row is a whole number of the format's blocks plus (n_in % 256) elements of a partial one — the same tail
    // gemv() takes. qwen4exp's routed down projection is n_in = n_ff_exp = 640, so without this the whole
    // grouped form is unavailable to it and the experts fall back to a launch each.
    const int n_tail = n_in % 256;
    if (type_bytes(type, static_cast<uint64_t>(n_in)) == 0 || n_tail % 8 != 0) return false;
    GemvArgs a{};
    a.W = static_cast<const uint8_t*>(W);
    a.row_bytes = row_bytes_of(type, n_in);
    a.expert_bytes = expert_bytes;
    a.ids = ids;
    a.x = x;
    a.x_stride = x_stride;
    a.y = y;
    a.y_stride = n_out;
    a.n_out = n_out;
    a.n_sb = n_in / 256;
    a.n_tail = n_tail;
    a.acc = false;
    a.x_group = x_group;
    // The row's partial super-block is summed inside the product (GemvArgs::n_tail), never a second launch.
    return dispatch(type, a, n_sel, s);
}

bool embed_row(uint32_t type, const void* W, const int32_t* token, float* out, int n_in, cudaStream_t s) {
    if (!gemv_supported(type) || n_in % 256 != 0) return false;
    const uint64_t rb = row_bytes_of(type, n_in);
    const uint8_t* w = static_cast<const uint8_t*>(W);
    const unsigned g = static_cast<unsigned>(n_in / 256);
    switch (type) {
        case kF32: embed_kernel<kF32><<<g, 32, 0, s>>>(w, rb, token, out); break;
        case kBF16: embed_kernel<kBF16><<<g, 32, 0, s>>>(w, rb, token, out); break;
        case kQ8_0: embed_kernel<kQ8_0><<<g, 32, 0, s>>>(w, rb, token, out); break;
        case kQ2_K: embed_kernel<kQ2_K><<<g, 32, 0, s>>>(w, rb, token, out); break;
        case kQ3_K: embed_kernel<kQ3_K><<<g, 32, 0, s>>>(w, rb, token, out); break;
        case kQ6_K: embed_kernel<kQ6_K><<<g, 32, 0, s>>>(w, rb, token, out); break;
        case kIQ2_S: embed_kernel<kIQ2_S><<<g, 32, 0, s>>>(w, rb, token, out); break;
        case kIQ3_S: embed_kernel<kIQ3_S><<<g, 32, 0, s>>>(w, rb, token, out); break;
        case kIQ4_XS: embed_kernel<kIQ4_XS><<<g, 32, 0, s>>>(w, rb, token, out); break;
        case kF16: embed_kernel<kF16><<<g, 32, 0, s>>>(w, rb, token, out); break;
        case kQ4_K: embed_kernel<kQ4_K><<<g, 32, 0, s>>>(w, rb, token, out); break;
        case kQ5_K: embed_kernel<kQ5_K><<<g, 32, 0, s>>>(w, rb, token, out); break;
        case kIQ2_XXS: embed_kernel<kIQ2_XXS><<<g, 32, 0, s>>>(w, rb, token, out); break;
        case kIQ2_XS: embed_kernel<kIQ2_XS><<<g, 32, 0, s>>>(w, rb, token, out); break;
        case kIQ3_XXS: embed_kernel<kIQ3_XXS><<<g, 32, 0, s>>>(w, rb, token, out); break;
        case kIQ1_S: embed_kernel<kIQ1_S><<<g, 32, 0, s>>>(w, rb, token, out); break;
        case kIQ1_M: embed_kernel<kIQ1_M><<<g, 32, 0, s>>>(w, rb, token, out); break;
        case kIQ4_NL: embed_kernel<kIQ4_NL><<<g, 32, 0, s>>>(w, rb, token, out); break;
        case kQ2_0: embed_kernel<kQ2_0><<<g, 32, 0, s>>>(w, rb, token, out); break;
        case kNVFP4: embed_kernel<kNVFP4><<<g, 32, 0, s>>>(w, rb, token, out); break;
        default: return false;
    }
    return cudaGetLastError() == cudaSuccess;
}

bool embed_rows(uint32_t type, const void* W, const int32_t* tokens, int T, float* out, int n_in, cudaStream_t s) {
    if (!gemv_supported(type) || n_in % 256 != 0 || T <= 0) return false;
    const uint64_t rb = row_bytes_of(type, n_in);
    const uint8_t* w = static_cast<const uint8_t*>(W);
    const dim3 g(static_cast<unsigned>(n_in / 256), static_cast<unsigned>(T));
    switch (type) {
        case kF32: embed_rows_kernel<kF32><<<g, 32, 0, s>>>(w, rb, tokens, out, n_in); break;
        case kBF16: embed_rows_kernel<kBF16><<<g, 32, 0, s>>>(w, rb, tokens, out, n_in); break;
        case kQ8_0: embed_rows_kernel<kQ8_0><<<g, 32, 0, s>>>(w, rb, tokens, out, n_in); break;
        case kQ2_K: embed_rows_kernel<kQ2_K><<<g, 32, 0, s>>>(w, rb, tokens, out, n_in); break;
        case kQ3_K: embed_rows_kernel<kQ3_K><<<g, 32, 0, s>>>(w, rb, tokens, out, n_in); break;
        case kQ6_K: embed_rows_kernel<kQ6_K><<<g, 32, 0, s>>>(w, rb, tokens, out, n_in); break;
        case kIQ2_S: embed_rows_kernel<kIQ2_S><<<g, 32, 0, s>>>(w, rb, tokens, out, n_in); break;
        case kIQ3_S: embed_rows_kernel<kIQ3_S><<<g, 32, 0, s>>>(w, rb, tokens, out, n_in); break;
        case kIQ4_XS: embed_rows_kernel<kIQ4_XS><<<g, 32, 0, s>>>(w, rb, tokens, out, n_in); break;
        case kF16: embed_rows_kernel<kF16><<<g, 32, 0, s>>>(w, rb, tokens, out, n_in); break;
        case kQ4_K: embed_rows_kernel<kQ4_K><<<g, 32, 0, s>>>(w, rb, tokens, out, n_in); break;
        case kQ5_K: embed_rows_kernel<kQ5_K><<<g, 32, 0, s>>>(w, rb, tokens, out, n_in); break;
        case kIQ2_XXS: embed_rows_kernel<kIQ2_XXS><<<g, 32, 0, s>>>(w, rb, tokens, out, n_in); break;
        case kIQ2_XS: embed_rows_kernel<kIQ2_XS><<<g, 32, 0, s>>>(w, rb, tokens, out, n_in); break;
        case kIQ3_XXS: embed_rows_kernel<kIQ3_XXS><<<g, 32, 0, s>>>(w, rb, tokens, out, n_in); break;
        case kIQ1_S: embed_rows_kernel<kIQ1_S><<<g, 32, 0, s>>>(w, rb, tokens, out, n_in); break;
        case kIQ1_M: embed_rows_kernel<kIQ1_M><<<g, 32, 0, s>>>(w, rb, tokens, out, n_in); break;
        case kIQ4_NL: embed_rows_kernel<kIQ4_NL><<<g, 32, 0, s>>>(w, rb, tokens, out, n_in); break;
        case kQ2_0: embed_rows_kernel<kQ2_0><<<g, 32, 0, s>>>(w, rb, tokens, out, n_in); break;
        case kNVFP4: embed_rows_kernel<kNVFP4><<<g, 32, 0, s>>>(w, rb, tokens, out, n_in); break;
        default: return false;
    }
    return cudaGetLastError() == cudaSuccess;
}

bool gemv_dual(uint32_t type, const void* W0, const void* W1, const float* x, float* y0, float* y1, int n_out,
               int n_in, cudaStream_t s) {
    const int n_tail = n_in % 256;
    if (n_out > 0 && frag_takes(type, n_in, s))
        return frag_cols(type, W0, x, n_in, y0, n_out, n_out, n_in, 1, false, s) && frag_cols(type, W1, x, n_in, y1, n_out, n_out, n_in, 1, false, s);
    if (!gemv_supported(type) || n_tail % 8 != 0 || n_out <= 0) return false;
    if (type == kQ6_K && n_tail == 0 && q6k_launch(W0, W1, x, nullptr, 0, y0, y1, 0, n_out, n_in, 1, false, 2, s)) return true;
    GemvArgs a{};
    a.W = static_cast<const uint8_t*>(W0);
    a.W2 = static_cast<const uint8_t*>(W1);
    a.row_bytes = row_bytes_of(type, n_in);
    a.x = x;
    a.y = y0;
    a.y2 = y1;
    a.n_out = n_out;
    a.n_sb = n_in / 256;
    a.n_tail = n_tail;
    return dispatch(type, a, 1, s, 2);
}

bool gemv_experts_dual(uint32_t type, const void* W0, const void* W1, uint64_t expert_bytes, const int32_t* ids,
                       int n_sel, const float* x, float* y0, float* y1, int n_out, int n_in, cudaStream_t s) {
    if (!gemv_supported(type) || n_in % 256 != 0 || n_out <= 0 || n_sel <= 0) return false;
    GemvArgs a{};
    a.W = static_cast<const uint8_t*>(W0);
    a.W2 = static_cast<const uint8_t*>(W1);
    a.row_bytes = row_bytes_of(type, n_in);
    a.expert_bytes = expert_bytes;
    a.ids = ids;
    a.x = x;
    a.y = y0;
    a.y2 = y1;
    a.y_stride = n_out;
    a.n_out = n_out;
    a.n_sb = n_in / 256;
    return dispatch(type, a, n_sel, s, 2);
}

bool gemv_swiglu(uint32_t type, const void* W, const float* g, const float* u, float* y, int n_out, int n_in,
                 bool accumulate, cudaStream_t s) {
    if (n_out > 0 && frag_swiglu(type, W, g, u, y, n_out, n_in, accumulate, s)) return true;
    if (!gemv_supported(type) || n_in % 256 != 0 || n_in > 512 || n_out <= 0) return false;
    if (type == kQ6_K && q6k_launch(W, nullptr, g, u, 0, y, nullptr, 0, n_out, n_in, 1, accumulate, 1, s)) return true;
    GemvArgs a{};
    a.W = static_cast<const uint8_t*>(W);
    a.row_bytes = row_bytes_of(type, n_in);
    a.x = g;
    a.x2 = u;
    a.y = y;
    a.n_out = n_out;
    a.n_sb = n_in / 256;
    a.acc = accumulate;
    return dispatch(type, a, 1, s);
}

}  // namespace limbo::cuda
