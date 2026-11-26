// Copyright (c) 2026 ITERFORM. All rights reserved.
#pragma once
// src/cuda/gemm_mmq.cuh — the shared-memory int8 tensor-core GEMM kernel (mmq_kernel) and the weight decoders it
// uses, shared by gemm.cu (Q8_0, Q6_K, IQ2_S, IQ3_S, IQ4_XS and the fragment-direct kernels) and the gemm_ext_*.cu
// translation units (the other formats, one file per group so that the instantiations compile in parallel).
//
// Every weight format decodes to int8 codes times a floating scale per group of 16 or 32 columns (and, for the
// K-quants with a minimum, minus a per-group minimum times the sum of the group's activations): decode8q turns the
// 8 columns [8*lane, 8*lane+8) of a 256-column super-block into 8 packed int8, a scale and a minimum.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>

#include "limbo/cuda/gemm.hpp"
#include "limbo/types.hpp"

namespace limbo::cuda::mmq {

constexpr uint32_t kQ8_0 = static_cast<uint32_t>(Type::Q8_0);
constexpr uint32_t kQ6_K = static_cast<uint32_t>(Type::Q6_K);
constexpr uint32_t kIQ2_S = static_cast<uint32_t>(Type::IQ2_S);
constexpr uint32_t kIQ3_S = static_cast<uint32_t>(Type::IQ3_S);
constexpr uint32_t kIQ4_XS = static_cast<uint32_t>(Type::IQ4_XS);
constexpr uint32_t kIQ3_XXS = static_cast<uint32_t>(Type::IQ3_XXS);
constexpr uint32_t kIQ2_XXS = static_cast<uint32_t>(Type::IQ2_XXS);
constexpr uint32_t kIQ2_XS = static_cast<uint32_t>(Type::IQ2_XS);
constexpr uint32_t kIQ1_S = static_cast<uint32_t>(Type::IQ1_S);
constexpr uint32_t kIQ1_M = static_cast<uint32_t>(Type::IQ1_M);
constexpr uint32_t kIQ4_NL = static_cast<uint32_t>(Type::IQ4_NL);
constexpr uint32_t kQ2_0 = static_cast<uint32_t>(Type::Q2_0);
constexpr uint32_t kQ4_K = static_cast<uint32_t>(Type::Q4_K);
constexpr uint32_t kQ5_K = static_cast<uint32_t>(Type::Q5_K);
constexpr uint32_t kQ2_K = static_cast<uint32_t>(Type::Q2_K);
constexpr uint32_t kQ3_K = static_cast<uint32_t>(Type::Q3_K);
constexpr uint32_t kNVFP4 = static_cast<uint32_t>(Type::NVFP4);

constexpr int kBN = 128;         // weight rows per block (in shared memory)
constexpr int kBK = 128;         // columns per stage
constexpr int kWP = kBK + 16;    // pitch in bytes of a Ws / Xs row: 16 more than a multiple of 128 so ldmatrix is conflict free
constexpr int kSP = 9;           // pitch in floats of a row of Wsc (8 groups)
constexpr int kXSP = 5;          // pitch in floats of a row of Xsc (4 blocks of 32)
constexpr int kMagic = 0x4B400000;
constexpr float kMagicF = 12582912.0f;  // the float the bits 0x4B400000 read as

// Bytes of one 256-column super-block of a row (the 32- and 64-column block formats count 8 and 4 blocks).
__host__ __device__ constexpr int sb_bytes(uint32_t t) {
    return t == kQ8_0 ? 8 * 34 : t == kQ6_K ? 210 : t == kIQ2_S ? 82 : t == kIQ3_S ? 110 : t == kIQ4_XS ? 136
         : t == kIQ3_XXS ? 98 : t == kIQ2_XXS ? 66 : t == kIQ2_XS ? 74 : t == kIQ1_S ? 50 : t == kIQ1_M ? 56
         : t == kIQ4_NL ? 8 * 18 : t == kQ2_0 ? 4 * 18 : t == kQ4_K ? 144 : t == kQ5_K ? 176 : t == kQ2_K ? 84 : t == kQ3_K ? 110
         : t == kNVFP4 ? 4 * 36 : 0;
}
// Columns per weight scale.
template <uint32_t TY>
__host__ __device__ constexpr int group_size() {
    return (TY == kQ6_K || TY == kIQ2_S || TY == kIQ2_XS || TY == kIQ1_M || TY == kQ2_K || TY == kQ3_K || TY == kNVFP4) ? 16
                                                                                                                       : 32;
}
// Formats whose weight is d*sc*q - dmin*m: the product needs the sum of the group's activations (act_quant_sums).
template <uint32_t TY>
__host__ __device__ constexpr bool has_min() {
    return TY == kQ4_K || TY == kQ5_K || TY == kQ2_K;
}

struct Tabs {
    const uint64_t* t2s;
    const uint32_t* t3s;
    const int8_t* tnl;
    const uint32_t* sel2;  // IQ2_S grid as byte-permute selectors, see init
    const uint64_t* g2xxs;  // IQ2_XXS grid, 256 entries of 8 bytes
    const uint64_t* g2xs;   // IQ2_XS grid, 512
    const uint32_t* g3xxs;  // IQ3_XXS grid, 256 entries of 4 bytes
    const uint64_t* g1s;    // IQ1_S / IQ1_M grid, 2048 entries of 8 signed bytes
    const uint8_t* signs;   // the 7-bit sign table of IQ2_XXS, IQ2_XS and IQ3_XXS: 128 bytes
};


__device__ __forceinline__ float h2f(const uint8_t* p) {
    return __half2float(__ushort_as_half(*reinterpret_cast<const unsigned short*>(p)));
}
// Four sign bits (the low nibble of `s4`) as bytes 0/1: bit i lands in byte i.
__device__ __forceinline__ uint32_t sign_bytes(uint32_t s4) { return ((s4 & 0xFu) * 0x00204081u) & 0x01010101u; }
// Negates the bytes of g where m has a 1 (m = sign_bytes): -x = (x ^ 0xFF) + 1 per byte. No byte of the i-quant
// grids is 0 or above 127, so the +1 never carries into the next byte.
__device__ __forceinline__ uint32_t negate_where(uint32_t g, uint32_t m) { return (g ^ (m * 0xFFu)) + m; }
__device__ __forceinline__ uint32_t ld32_2(const uint8_t* p) {
    return static_cast<uint32_t>(*reinterpret_cast<const unsigned short*>(p)) |
           (static_cast<uint32_t>(*reinterpret_cast<const unsigned short*>(p + 2)) << 16);
}

// The 8 elements [8*lane, 8*lane+8) of the super-block at sb as 8 packed int8 (lo = elements 0..3, hi = 4..7)
// and the one fp32 scale they share (weight = scale * q). The same mapping, field by field, as decode8 in
// gemv.cu, so a stage of a block is lanes 16h..16h+15. The signs and the six-bit offsets are applied four bytes
// at a time.
__device__ __forceinline__ unsigned ld16(const uint8_t* p) { return *reinterpret_cast<const unsigned short*>(p); }

// IQ1: 8 grid bytes in {-1, 0, 1} and a delta of +-1/8 -> the integers 8*g +- 1 (the scale carries the 1/8).
__device__ __forceinline__ uint32_t iq1_codes(uint32_t g, bool neg) {
    const uint32_t t = (g << 3) & 0xF8F8F8F8u;
    return neg ? __vsub4(t, 0x01010101u) : __vadd4(t, 0x01010101u);
}
// The four 2-bit codes of a byte as four bytes 0..3.
__device__ __forceinline__ uint32_t spread2(uint32_t b) { return (b | (b << 6) | (b << 12) | (b << 18)) & 0x03030303u; }
// scale and minimum j (0..7) of a K-quant's 12 packed 6-bit values.
__device__ __forceinline__ void k4_scale_min(int j, const uint8_t* q, int& sc, int& mn) {
    if (j < 4) {
        sc = q[j] & 63;
        mn = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        mn = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// NVFP4: ggml's UE4M3 sub-block scale, HALVED to match the doubled E2M1 codebook; 0x00 and 0x7F are zero
// (src/quants.cpp ue4m3_to_f32, value for value).
__device__ __forceinline__ float ue4m3_mmq(uint8_t x) {
    if (x == 0 || x == 0x7f) return 0.0f;
    const int ex = (x >> 3) & 0xf, man = x & 0x7;
    const float raw = ex == 0 ? ldexpf(float(man), -9) : ldexpf(1.0f + float(man) / 8.0f, ex - 7);
    return raw * 0.5f;
}
// The doubled E2M1 code of a nibble as an int8 byte: magnitudes 0 1 2 3 4 6 8 12, bit 3 the sign.
__device__ __forceinline__ uint32_t fp4_code(uint32_t n) {
    const uint32_t m = n & 7u;
    const uint32_t v = m < 4u ? m : (0x0C080604u >> (8u * (m - 4u))) & 0xffu;
    return (n & 8u) ? ((0x100u - v) & 0xffu) : v;
}

template <uint32_t TY>
__device__ __forceinline__ void decode8q(const uint8_t* sb, int lane, uint32_t& lo, uint32_t& hi, float& scale, float& mn,
                                         const Tabs& tb) {
    mn = 0.f;
    if constexpr (TY == kQ8_0) {
        const uint8_t* p = sb + (lane >> 2) * 34;
        scale = h2f(p);
        const uint8_t* q = p + 2 + (lane & 3) * 8;
        lo = ld32_2(q);
        hi = ld32_2(q + 4);
    } else if constexpr (TY == kQ6_K) {
        const int half = lane >> 4, qi = (lane >> 2) & 3, l = (lane & 3) * 8;
        const int sc = static_cast<int8_t>(sb[192 + half * 8 + qi * 2 + ((lane >> 1) & 1)]);
        scale = h2f(sb + 208) * static_cast<float>(sc);
        const uint8_t* ql = sb + half * 64 + (qi & 1) * 32 + l;
        const uint8_t* qh = sb + 128 + half * 32 + l;
        const int nib = (qi >> 1) * 4, hs = 2 * qi;
        const uint32_t l0 = ld32_2(ql), l1 = ld32_2(ql + 4), h0 = ld32_2(qh), h1 = ld32_2(qh + 4);
        const uint32_t v0 = ((l0 >> nib) & 0x0F0F0F0Fu) | (((h0 >> hs) & 0x03030303u) << 4);
        const uint32_t v1 = ((l1 >> nib) & 0x0F0F0F0Fu) | (((h1 >> hs) & 0x03030303u) << 4);
        // v - 32 per byte: (v | 0x80) - 0x20 = v + 96 cannot borrow; flipping bit 7 then adds 128 mod 256
        lo = ((v0 | 0x80808080u) - 0x20202020u) ^ 0x80808080u;
        hi = ((v1 | 0x80808080u) - 0x20202020u) ^ 0x80808080u;
    } else if constexpr (TY == kIQ2_S) {
        const int ib32 = lane >> 2, l4 = lane & 3;
        const float d = h2f(sb);
        const uint8_t scb = sb[74 + ib32];
        const int nib = (l4 >> 1) == 0 ? (scb & 0xf) : (scb >> 4);
        scale = d * (0.5f + static_cast<float>(nib)) * 0.25f;
        const unsigned idx = static_cast<unsigned>(sb[2 + ib32 * 4 + l4]) |
                             ((static_cast<unsigned>(sb[66 + ib32]) << (8 - 2 * l4)) & 0x300u);
        const uint2 g = reinterpret_cast<const uint2*>(tb.t2s)[idx];
        const unsigned sg = sb[34 + ib32 * 4 + l4];
        lo = negate_where(g.x, sign_bytes(sg));
        hi = negate_where(g.y, sign_bytes(sg >> 4));
    } else if constexpr (TY == kIQ3_S) {
        const int l = lane & 3, p = lane >> 3, pass = (lane >> 2) & 1;
        const float d = h2f(sb);
        const uint8_t scb = sb[106 + p];
        const int nib = pass ? (scb >> 4) : (scb & 0xf);
        scale = d * (1.f + 2.f * static_cast<float>(nib));
        const unsigned qh = sb[66 + (lane >> 2)];
        const unsigned i1 = static_cast<unsigned>(sb[2 + 2 * lane]) | ((qh << (8 - 2 * l)) & 256u);
        const unsigned i2 = static_cast<unsigned>(sb[2 + 2 * lane + 1]) | ((qh << (7 - 2 * l)) & 256u);
        const uint32_t g1 = tb.t3s[i1], g2 = tb.t3s[i2];
        const unsigned sg = sb[74 + lane];
        lo = negate_where(g1, sign_bytes(sg));
        hi = negate_where(g2, sign_bytes(sg >> 4));
    } else if constexpr (TY == kIQ4_XS) {
        const int ib = lane >> 2, k = (lane & 3) * 8;
        const float d = h2f(sb);
        const int sh = sb[2] | (sb[3] << 8);
        const int sl = sb[4 + (ib >> 1)];
        const int ls = ((sl >> (4 * (ib & 1))) & 0xf) | (((sh >> (2 * ib)) & 3) << 4);
        scale = d * static_cast<float>(ls - 32);
        const uint8_t* qs = sb + 8 + ib * 16 + (k & 15);
        const uint32_t w0 = ld32_2(qs), w1 = ld32_2(qs + 4);
        const int sh4 = k >= 16 ? 4 : 0;
        uint32_t o0 = 0, o1 = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            o0 |= (static_cast<uint32_t>(tb.tnl[(w0 >> (8 * i + sh4)) & 0xf]) & 0xffu) << (8 * i);
            o1 |= (static_cast<uint32_t>(tb.tnl[(w1 >> (8 * i + sh4)) & 0xf]) & 0xffu) << (8 * i);
        }
        lo = o0;
        hi = o1;
    } else if constexpr (TY == kIQ3_XXS) {
        const int ib32 = lane >> 2, l = lane & 3;
        const float d = h2f(sb);
        const uint32_t word = ld32_2(sb + 66 + 4 * ib32);
        scale = d * (0.5f + static_cast<float>(word >> 28)) * 0.5f;
        const unsigned sg = tb.signs[(word >> (7 * l)) & 127];
        const uint32_t g1 = tb.g3xxs[sb[2 + 8 * ib32 + 2 * l]], g2 = tb.g3xxs[sb[2 + 8 * ib32 + 2 * l + 1]];
        lo = negate_where(g1, sign_bytes(sg));
        hi = negate_where(g2, sign_bytes(sg >> 4));
    } else if constexpr (TY == kIQ2_XXS) {
        const int ib32 = lane >> 2, l = lane & 3;
        const float d = h2f(sb);
        const uint32_t idxw = ld32_2(sb + 2 + 8 * ib32), sgnw = ld32_2(sb + 6 + 8 * ib32);
        scale = d * (0.5f + static_cast<float>(sgnw >> 28)) * 0.25f;
        const uint2 g = reinterpret_cast<const uint2*>(tb.g2xxs)[(idxw >> (8 * l)) & 0xff];
        const unsigned sg = tb.signs[(sgnw >> (7 * l)) & 127];
        lo = negate_where(g.x, sign_bytes(sg));
        hi = negate_where(g.y, sign_bytes(sg >> 4));
    } else if constexpr (TY == kIQ2_XS) {
        const int ib32 = lane >> 2, l = lane & 3;
        const float d = h2f(sb);
        const unsigned q = ld16(sb + 2 + 2 * lane);
        const uint8_t scb = sb[66 + ib32];
        const int nib = (l >> 1) ? (scb >> 4) : (scb & 0xf);
        scale = d * (0.5f + static_cast<float>(nib)) * 0.25f;
        const uint2 g = reinterpret_cast<const uint2*>(tb.g2xs)[q & 511];
        const unsigned sg = tb.signs[q >> 9];
        lo = negate_where(g.x, sign_bytes(sg));
        hi = negate_where(g.y, sign_bytes(sg >> 4));
    } else if constexpr (TY == kIQ1_S) {
        const int ib = lane >> 2, l = lane & 3;
        const float d = h2f(sb);
        const unsigned h = ld16(sb + 34 + 2 * ib);
        scale = d * static_cast<float>(2 * ((h >> 12) & 7) + 1) * 0.125f;
        const unsigned idx = static_cast<unsigned>(sb[2 + 4 * ib + l]) | (((h >> (3 * l)) & 7u) << 8);
        const uint2 g = __ldg(reinterpret_cast<const uint2*>(tb.g1s) + idx);
        const bool neg = (h & 0x8000u) != 0;
        lo = iq1_codes(g.x, neg);
        hi = iq1_codes(g.y, neg);
    } else if constexpr (TY == kIQ1_M) {
        const int ib = lane >> 2, l = lane & 3;
        const unsigned s0 = ld16(sb + 48), s1 = ld16(sb + 50), s2 = ld16(sb + 52), s3 = ld16(sb + 54);
        const unsigned sbits = (s0 >> 12) | ((s1 >> 8) & 0x00f0u) | ((s2 >> 4) & 0x0f00u) | (s3 & 0xf000u);
        const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(sbits)));
        const unsigned scw = ld16(sb + 48 + 2 * (ib >> 1));
        scale = d * static_cast<float>(2 * ((scw >> (6 * (ib & 1) + 3 * (l >> 1))) & 7) + 1) * 0.125f;
        const unsigned qh = sb[32 + 2 * ib + (l >> 1)];
        const unsigned idx = static_cast<unsigned>(sb[4 * ib + l]) | ((qh << ((l & 1) ? 4 : 8)) & 0x700u);
        const uint2 g = __ldg(reinterpret_cast<const uint2*>(tb.g1s) + idx);
        const bool neg = (qh & ((l & 1) ? 0x80u : 0x08u)) != 0;
        lo = iq1_codes(g.x, neg);
        hi = iq1_codes(g.y, neg);
    } else if constexpr (TY == kIQ4_NL) {
        const int ib = lane >> 2, l = lane & 3;
        const uint8_t* p = sb + 18 * ib;
        scale = h2f(p);
        const uint8_t* qs = p + 2 + (l & 1) * 8;
        const uint32_t w0 = ld32_2(qs), w1 = ld32_2(qs + 4);
        const int sh4 = (l >> 1) * 4;
        uint32_t o0 = 0, o1 = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            o0 |= (static_cast<uint32_t>(tb.tnl[(w0 >> (8 * i + sh4)) & 0xf]) & 0xffu) << (8 * i);
            o1 |= (static_cast<uint32_t>(tb.tnl[(w1 >> (8 * i + sh4)) & 0xf]) & 0xffu) << (8 * i);
        }
        lo = o0;
        hi = o1;
    } else if constexpr (TY == kNVFP4) {
        // 64-element blocks of { 4 UE4M3 scales, 32 bytes }: sub-block s holds 16 elements, the low nibbles of
        // its 8 bytes the first 8, the high nibbles the next 8 (src/quants.cpp deq_nvfp4).
        const int ib = lane >> 3, l8 = lane & 7, sub = l8 >> 1, sh = (l8 & 1) * 4;
        const uint8_t* p = sb + 36 * ib;
        scale = ue4m3_mmq(p[sub]);
        const uint8_t* qs = p + 4 + 8 * sub;
        uint32_t o0 = 0, o1 = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            o0 |= fp4_code(static_cast<uint32_t>(qs[i]) >> sh) << (8 * i);
            o1 |= fp4_code(static_cast<uint32_t>(qs[4 + i]) >> sh) << (8 * i);
        }
        lo = o0;
        hi = o1;
    } else if constexpr (TY == kQ2_0) {
        const int ib = lane >> 3, l8 = lane & 7;
        const uint8_t* p = sb + 18 * ib;
        scale = h2f(p);
        lo = __vsub4(spread2(p[2 + 2 * l8]), 0x01010101u);  // codes 0..3 -> -1, 0, 1, 2
        hi = __vsub4(spread2(p[3 + 2 * l8]), 0x01010101u);
    } else if constexpr (TY == kQ4_K || TY == kQ5_K) {
        const int c = lane >> 3, g = (lane >> 2) & 1, l = lane & 3;
        const float d = h2f(sb), dmin = h2f(sb + 2);
        int sc, m;
        k4_scale_min(2 * c + g, sb + 4, sc, m);
        scale = d * static_cast<float>(sc);
        mn = dmin * static_cast<float>(m);
        if constexpr (TY == kQ4_K) {
            const uint8_t* q = sb + 16 + 32 * c + 8 * l;
            lo = (ld32_2(q) >> (4 * g)) & 0x0F0F0F0Fu;
            hi = (ld32_2(q + 4) >> (4 * g)) & 0x0F0F0F0Fu;
        } else {
            const uint8_t* q = sb + 48 + 32 * c + 8 * l;
            const uint8_t* qh = sb + 16 + 8 * l;
            const int hs = 2 * c + g;
            lo = ((ld32_2(q) >> (4 * g)) & 0x0F0F0F0Fu) | (((ld32_2(qh) >> hs) & 0x01010101u) << 4);
            hi = ((ld32_2(q + 4) >> (4 * g)) & 0x0F0F0F0Fu) | (((ld32_2(qh + 4) >> hs) & 0x01010101u) << 4);
        }
    } else if constexpr (TY == kQ2_K) {
        const int half = lane >> 4, j = (lane >> 2) & 3, w0 = (lane & 3) * 8;
        const float d = h2f(sb + 80), dmin = h2f(sb + 82);
        const unsigned scb = sb[half * 8 + 2 * j + (w0 >> 4)];
        scale = d * static_cast<float>(scb & 15);
        mn = dmin * static_cast<float>(scb >> 4);
        const uint8_t* q = sb + 16 + 32 * half + w0;
        lo = (ld32_2(q) >> (2 * j)) & 0x03030303u;
        hi = (ld32_2(q + 4) >> (2 * j)) & 0x03030303u;
    } else if constexpr (TY == kQ3_K) {
        // 3-bit codes: two low bits from qs, the high bit from the hmask (set = the code as is, clear = 4 less), minus 4 -> -4..3
        const int half = lane >> 4, j = (lane >> 2) & 3, l32 = (lane & 3) * 8;
        const int is = half * 8 + j * 2 + ((lane >> 1) & 1);
        const int g = is >> 2, i4 = is & 3;
        const unsigned lob = sb[96 + (g & 1) * 4 + i4];
        const int slo = (g >> 1) ? static_cast<int>((lob >> 4) & 15) : static_cast<int>(lob & 15);
        const int shi = (sb[104 + i4] >> (2 * g)) & 3;
        scale = h2f(sb + 108) * static_cast<float>((slo | (shi << 4)) - 32);
        const uint8_t* q = sb + 32 + half * 32 + l32;
        const uint8_t* hm = sb + l32;
        const int hb = half * 4 + j;
        lo = __vsub4(((ld32_2(q) >> (2 * j)) & 0x03030303u) | (((ld32_2(hm) >> hb) & 0x01010101u) << 2), 0x04040404u);
        hi = __vsub4(((ld32_2(q + 4) >> (2 * j)) & 0x03030303u) | (((ld32_2(hm + 4) >> hb) & 0x01010101u) << 2), 0x04040404u);
    }
}

__device__ __forceinline__ void ldsm_x2(uint32_t& r0, uint32_t& r1, const void* p) {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(a));
}
__device__ __forceinline__ void ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(a));
}
// d = A(16x16 int8) * B(16x8 int8) + c, int32.
__device__ __forceinline__ void mma_s8(int* d, uint32_t a0, uint32_t a1, uint32_t b0, int c0, int c1, int c2, int c3) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(c0), "r"(c1), "r"(c2), "r"(c3));
}

__device__ __forceinline__ float silu_f(float x) { return x / (1.0f + expf(-x)); }

struct Args {
    const uint8_t* W0;
    const uint8_t* W1;           // the up stack in the fused gate/up kernel
    uint64_t row_bytes;
    uint64_t expert_bytes;
    const int8_t* xq;            // [NT][xrows][K]
    const float* xs;             // [xrows][K/32]
    const float* xsum;           // [xrows][K/16] sum of the dequantised activations per 16 columns (formats with a minimum)
    uint64_t xterm_stride;       // bytes between planes of xq = xrows * K
    const int32_t* xrow_list;    // pair -> row of xq (null: the row is the pair index)
    const int4* tiles;           // MoE tiles (expert, first pair, pairs, 0); null: dense, tiles of BM rows
    const int32_t* n_tiles;
    float* y;
    int ldy;
    int n_out;
    int K;
    int T;                       // dense: rows
    Tabs tb;
    const uint8_t* const* eptr;  // MoE: expert -> its blob (W0 / W1 are then byte offsets into it); null: stacks
    int ksplit;                  // dense: K slices (blockIdx.z), 0/1 none
    float* part;                 // ksplit > 1: slice z's sums at part + z * T * ldy
};

enum Mode { kStore = 0, kAcc = 1, kGateUp = 2 };

constexpr int kXMP = 9;  // pitch in floats of a row of the activation sums (8 granules of 16 columns)

template <uint32_t TY, int NT>
__host__ __device__ constexpr size_t smem_bytes(int BM) {
    return size_t(kBN) * kWP + size_t(NT) * BM * kWP + size_t(kBN) * kSP * 4 + size_t(BM) * kXSP * 4 +
           (has_min<TY>() ? size_t(kBN) * kSP * 4 + size_t(BM) * kXMP * 4 : 0);
}

template <uint32_t TY, int NT, int BM, int MODE>
__global__ void __launch_bounds__(128) mmq_kernel(const Args a) {
    extern __shared__ __align__(16) uint8_t smem[];
    int8_t* Ws = reinterpret_cast<int8_t*>(smem);
    int8_t* Xs = Ws + kBN * kWP;
    float* Wsc = reinterpret_cast<float*>(Xs + NT * BM * kWP);
    float* Xsc = Wsc + kBN * kSP;
    float* Wmn = Xsc + BM * kXSP;                // minimums per weight group (formats with a minimum)
    float* Xsm = Wmn + (has_min<TY>() ? kBN * kSP : 0);  // activation sums per 16 columns

    constexpr int GS = group_size<TY>();
    constexpr int KS = GS / 16;
    constexpr int TN = BM / 8;
    constexpr int SBB = sb_bytes(TY);

    const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
    const int gq = lane >> 2, tq = lane & 3;

    // The codebooks live in shared memory: a lane's lookup is a gather, and a gather from global memory
    // costs a cache line per lane.
    __shared__ __align__(16) uint64_t s_t2s[TY == kIQ2_S ? 1024 : 1];
    __shared__ uint32_t s_t3s[TY == kIQ3_S ? 512 : 1];
    __shared__ int8_t s_tnl[(TY == kIQ4_XS || TY == kIQ4_NL) ? 16 : 1];
    constexpr bool kSigns = TY == kIQ3_XXS || TY == kIQ2_XXS || TY == kIQ2_XS;
    __shared__ __align__(16) uint64_t s_g2xxs[TY == kIQ2_XXS ? 256 : 1];
    __shared__ __align__(16) uint64_t s_g2xs[TY == kIQ2_XS ? 512 : 1];
    __shared__ uint32_t s_g3xxs[TY == kIQ3_XXS ? 256 : 1];
    __shared__ uint8_t s_sgn[kSigns ? 128 : 1];
    Tabs tb{s_t2s, s_t3s, s_tnl, nullptr, s_g2xxs, s_g2xs, s_g3xxs, a.tb.g1s, s_sgn};
    if constexpr (kSigns) {
        if (tid < 128) s_sgn[tid] = a.tb.signs[tid];
    }
    if constexpr (TY == kIQ2_XXS) {
        for (int i = tid; i < 256; i += 128) s_g2xxs[i] = a.tb.g2xxs[i];
    } else if constexpr (TY == kIQ2_XS) {
        for (int i = tid; i < 512; i += 128) s_g2xs[i] = a.tb.g2xs[i];
    } else if constexpr (TY == kIQ3_XXS) {
        for (int i = tid; i < 256; i += 128) s_g3xxs[i] = a.tb.g3xxs[i];
    }
    if constexpr (TY == kIQ2_S) {
        for (int i = tid; i < 1024; i += 128) s_t2s[i] = a.tb.t2s[i];
    } else if constexpr (TY == kIQ3_S) {
        for (int i = tid; i < 512; i += 128) s_t3s[i] = a.tb.t3s[i];
    } else if constexpr (TY == kIQ4_XS || TY == kIQ4_NL) {
        if (tid < 16) s_tnl[tid] = a.tb.tnl[tid];
    }

    int e = 0, off, cnt;
    if (a.tiles) {
        if (static_cast<int>(blockIdx.y) >= *a.n_tiles) return;
        const int4 t = a.tiles[blockIdx.y];
        e = t.x;
        off = t.y;
        cnt = t.z;
    } else {
        off = blockIdx.y * BM;
        cnt = min(BM, a.T - off);
    }
    const uint8_t* eb_ = a.eptr ? a.eptr[e] : nullptr;
    const uint8_t* wb0 = eb_ ? eb_ + reinterpret_cast<uintptr_t>(a.W0) : a.W0 + static_cast<uint64_t>(e) * a.expert_bytes;
    const uint8_t* wb1 = MODE == kGateUp ? (eb_ ? eb_ + reinterpret_cast<uintptr_t>(a.W1)
                                                : a.W1 + static_cast<uint64_t>(e) * a.expert_bytes)
                                         : nullptr;
    const int n0 = blockIdx.x * (MODE == kGateUp ? 64 : 128);

    float acc[2][TN][4];
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int n = 0; n < TN; ++n)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[j][n][i] = 0.f;

    // The rows of xq this tile reads, resolved once.
    __shared__ int s_xrow[BM];
    for (int i = tid; i < BM; i += 128) {
        const int p = off + (i < cnt ? i : 0);
        s_xrow[i] = a.xrow_list ? a.xrow_list[p] : p;
    }
    __syncthreads();

    const int nstage = a.K / kBK;
    for (int stage = 0; stage < nstage; ++stage) {
        __syncthreads();
        // ---- weights: 128 rows x 16 groups of 8 columns -> int8 + scale; thread = (group g16, rows r0, r0+8, ...)
#pragma unroll 4
        for (int it = 0; it < 16; ++it) {
            const int r = it * 8 + (tid >> 4), g16 = tid & 15;
            const uint8_t* wb;
            int grow;
            if constexpr (MODE == kGateUp) {
                wb = ((r >> 4) & 1) ? wb1 : wb0;
                grow = n0 + (r >> 5) * 16 + (r & 15);
            } else {
                wb = wb0;
                grow = n0 + r;
            }
            if (grow >= a.n_out) grow = a.n_out - 1;
            const uint8_t* sbp = wb + static_cast<uint64_t>(grow) * a.row_bytes + static_cast<uint64_t>(stage >> 1) * SBB;
            uint32_t lo, hi;
            float sc, mn;
            decode8q<TY>(sbp, (stage & 1) * 16 + g16, lo, hi, sc, mn, tb);
            *reinterpret_cast<uint2*>(Ws + r * kWP + g16 * 8) = make_uint2(lo, hi);
            if ((g16 & (GS / 8 - 1)) == 0) {
                Wsc[r * kSP + g16 / (GS / 8)] = sc;
                if constexpr (has_min<TY>()) Wmn[r * kSP + g16 / (GS / 8)] = mn;
            }
        }
        // ---- activations: NT planes x BM rows x 128 bytes
#pragma unroll
        for (int it = 0; it < NT * BM * 8 / 128; ++it) {
            const int item = it * 128 + tid;
            const int t = item / (BM * 8), rem = item % (BM * 8);
            const int row = rem >> 3, c = rem & 7;
            const int8_t* src = a.xq + t * a.xterm_stride + static_cast<uint64_t>(s_xrow[row]) * a.K + stage * kBK + c * 16;
            *reinterpret_cast<uint4*>(Xs + (t * BM + row) * kWP + c * 16) = *reinterpret_cast<const uint4*>(src);
        }
        for (int i = tid; i < BM * 4; i += 128) {
            const int row = i >> 2, c = i & 3;
            Xsc[row * kXSP + c] = a.xs[static_cast<uint64_t>(s_xrow[row]) * (a.K / 32) + stage * 4 + c];
        }
        if constexpr (has_min<TY>()) {
            for (int i = tid; i < BM * 8; i += 128) {
                const int row = i >> 3, c = i & 7;
                Xsm[row * kXMP + c] = a.xsum[static_cast<uint64_t>(s_xrow[row]) * (a.K / 16) + stage * 8 + c];
            }
        }
        __syncthreads();

        // ---- tensor-core products
#pragma unroll
        for (int kk = 0; kk < kBK / 16; kk += KS) {
            uint32_t A[KS][2][2];
            uint32_t B[KS][NT][TN];
#pragma unroll
            for (int s = 0; s < KS; ++s) {
#pragma unroll
                for (int j = 0; j < 2; ++j)
                    ldsm_x2(A[s][j][0], A[s][j][1], Ws + (wid * 32 + j * 16 + (lane & 15)) * kWP + (kk + s) * 16);
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    if constexpr (TN >= 4) {
#pragma unroll
                        for (int p = 0; p < TN / 4; ++p)
                            ldsm_x4(B[s][t][4 * p], B[s][t][4 * p + 1], B[s][t][4 * p + 2], B[s][t][4 * p + 3],
                                    Xs + (t * BM + (4 * p + (lane >> 3)) * 8 + (lane & 7)) * kWP + (kk + s) * 16);
                    } else {
#pragma unroll
                        for (int p = 0; p < TN / 2; ++p)
                            ldsm_x2(B[s][t][2 * p], B[s][t][2 * p + 1],
                                    Xs + (t * BM + (2 * p + ((lane >> 3) & 1)) * 8 + (lane & 7)) * kWP + (kk + s) * 16);
                    }
                }
            }
            const int gi = kk / KS, xb = kk / 2;
            float sxr[TN][2];
#pragma unroll
            for (int tn = 0; tn < TN; ++tn) {
                sxr[tn][0] = Xsc[(tn * 8 + 2 * tq) * kXSP + xb];
                sxr[tn][1] = Xsc[(tn * 8 + 2 * tq + 1) * kXSP + xb];
            }
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int ra = wid * 32 + j * 16 + gq;
                const float sw0 = Wsc[ra * kSP + gi], sw1 = Wsc[(ra + 8) * kSP + gi];
                float mw0 = 0.f, mw1 = 0.f;
                if constexpr (has_min<TY>()) {
                    mw0 = Wmn[ra * kSP + gi];
                    mw1 = Wmn[(ra + 8) * kSP + gi];
                }
#pragma unroll
                for (int tn = 0; tn < TN; ++tn) {
                    int d[NT][4];
#pragma unroll
                    for (int t = 0; t < NT; ++t) {
                        mma_s8(d[t], A[0][j][0], A[0][j][1], B[0][t][tn], kMagic, kMagic, kMagic, kMagic);
#pragma unroll
                        for (int s = 1; s < KS; ++s)
                            mma_s8(d[t], A[s][j][0], A[s][j][1], B[s][t][tn], d[t][0], d[t][1], d[t][2], d[t][3]);
                    }
                    const float sx0 = sxr[tn][0], sx1 = sxr[tn][1];
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        float v = __int_as_float(d[0][i]) - kMagicF;
#pragma unroll
                        for (int t = 1; t < NT; ++t) v = fmaf(v, 128.0f, __int_as_float(d[t][i]) - kMagicF);
                        const float sw = i < 2 ? sw0 : sw1;
                        const float sx = (i & 1) ? sx1 : sx0;
                        acc[j][tn][i] = fmaf(sw * sx, v, acc[j][tn][i]);
                        if constexpr (has_min<TY>()) {
                            // minus the group's minimum times the sum of its (dequantised) activations
                            const int tok = tn * 8 + 2 * tq + (i & 1);
                            float S = Xsm[tok * kXMP + gi * (GS / 16)];
                            if constexpr (GS == 32) S += Xsm[tok * kXMP + gi * 2 + 1];
                            acc[j][tn][i] = fmaf(-(i < 2 ? mw0 : mw1), S, acc[j][tn][i]);
                        }
                    }
                }
            }
        }
    }

    // ---- results
#pragma unroll
    for (int tn = 0; tn < TN; ++tn) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int tok = tn * 8 + 2 * tq + (i & 1);
            if (tok >= cnt) continue;
            const int rr = gq + (i >= 2 ? 8 : 0);
            float* yrow = a.y + static_cast<int64_t>(off + tok) * a.ldy;
            if constexpr (MODE == kGateUp) {
                const int n = n0 + wid * 16 + rr;
                if (n < a.n_out) yrow[n] = silu_f(acc[0][tn][i]) * acc[1][tn][i];
            } else {
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const int n = n0 + wid * 32 + j * 16 + rr;
                    if (n >= a.n_out) continue;
                    if constexpr (MODE == kAcc) yrow[n] += acc[j][tn][i];
                    else yrow[n] = acc[j][tn][i];
                }
            }
        }
    }
}

template <uint32_t TY, int NT, int BM, int MODE>
bool launch_mmq(const Args& a, dim3 grid, cudaStream_t s) {
    constexpr size_t smem = smem_bytes<TY, NT>(BM);
    static bool configured = false;
    if (!configured) {
        if (cudaFuncSetAttribute(mmq_kernel<TY, NT, BM, MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(smem)) != cudaSuccess)
            return false;
        configured = true;
    }
    mmq_kernel<TY, NT, BM, MODE><<<grid, 128, smem, s>>>(a);
    return cudaGetLastError() == cudaSuccess;
}

// One launch of the shared-memory kernel for the (planes, token rows, mode) combinations the products use: dense tiles of
// 64 token rows (32 at three planes), MoE tiles of 16 and 32; store, accumulate and the fused gate/up.
template <uint32_t TY>
bool run_mmq(const Args& a, int nt, int bm, int mode, dim3 grid, cudaStream_t s) {
    const int key = nt * 1000 + bm * 10 + mode;
    switch (key) {
        case 1640: return launch_mmq<TY, 1, 64, kStore>(a, grid, s);
        case 1641: return launch_mmq<TY, 1, 64, kAcc>(a, grid, s);
        case 2640: return launch_mmq<TY, 2, 64, kStore>(a, grid, s);
        case 2641: return launch_mmq<TY, 2, 64, kAcc>(a, grid, s);
        case 3320: return launch_mmq<TY, 3, 32, kStore>(a, grid, s);
        case 3321: return launch_mmq<TY, 3, 32, kAcc>(a, grid, s);
        case 1160: return launch_mmq<TY, 1, 16, kStore>(a, grid, s);
        case 2160: return launch_mmq<TY, 2, 16, kStore>(a, grid, s);
        case 3160: return launch_mmq<TY, 3, 16, kStore>(a, grid, s);
        case 1162: return launch_mmq<TY, 1, 16, kGateUp>(a, grid, s);
        case 2162: return launch_mmq<TY, 2, 16, kGateUp>(a, grid, s);
        case 3162: return launch_mmq<TY, 3, 16, kGateUp>(a, grid, s);
        case 1320: return launch_mmq<TY, 1, 32, kStore>(a, grid, s);
        case 2320: return launch_mmq<TY, 2, 32, kStore>(a, grid, s);
        case 1322: return launch_mmq<TY, 1, 32, kGateUp>(a, grid, s);
        case 2322: return launch_mmq<TY, 2, 32, kGateUp>(a, grid, s);
        case 3322: return launch_mmq<TY, 3, 32, kGateUp>(a, grid, s);
        case 1642: return launch_mmq<TY, 1, 64, kGateUp>(a, grid, s);
        case 2642: return launch_mmq<TY, 2, 64, kGateUp>(a, grid, s);
        default: return false;
    }
}

// Which gemm_ext_*.cu owns a format (0: none, it is handled in gemm.cu or not at all).
__host__ __device__ constexpr int ext_group(uint32_t t) {
    return (t == kIQ3_XXS || t == kIQ2_XXS || t == kIQ2_XS) ? 1
           : (t == kIQ1_S || t == kIQ1_M || t == kIQ4_NL || t == kQ2_0) ? 2
           : (t == kQ4_K || t == kQ5_K || t == kQ2_K || t == kQ3_K || t == kNVFP4) ? 3 : 0;
}

}  // namespace limbo::cuda::mmq

namespace limbo::cuda {
// gemm_ext_a.cu, gemm_ext_b.cu, gemm_ext_c.cu: device tables of their formats and the launch of their kernels.
bool ext_init_a(std::string* err);
bool ext_init_b(std::string* err);
bool ext_init_c(std::string* err);
bool ext_launch_a(uint32_t type, const mmq::Args& a, int nt, int bm, int mode, dim3 grid, cudaStream_t s);
bool ext_launch_b(uint32_t type, const mmq::Args& a, int nt, int bm, int mode, dim3 grid, cudaStream_t s);
bool ext_launch_c(uint32_t type, const mmq::Args& a, int nt, int bm, int mode, dim3 grid, cudaStream_t s);
}  // namespace limbo::cuda
