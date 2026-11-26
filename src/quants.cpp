// Copyright (c) 2026 ITERFORM. All rights reserved.
// src/quants.cpp — GGUF block formats to float, one function per format.
//
// Written from the layouts in include/limbo/quants.hpp and the bit-level meaning of each format (an
// interoperability fact, CLEANROOM.md §3). The codebooks come from ggml under its MIT notice
// (THIRD_PARTY_NOTICES). Every format here is checked against ggml's own to_float on real tensors of
// all three ladder files by tools/quant_parity — a decoder nobody has compared to the reference is a
// decoder nobody should trust.
#include "limbo/quants.hpp"

#include "limbo/types.hpp"

#include <algorithm>
#include <cmath>

namespace limbo::quants {

namespace {

inline uint16_t rd16(const uint8_t* p) { return uint16_t(p[0] | (p[1] << 8)); }
inline float scale_at(const uint8_t* p) { return f16_to_f32(rd16(p)); }

// ---- plain -------------------------------------------------------------------------------------
void deq_f32(const uint8_t* src, float* dst, uint64_t n) {
    for (uint64_t i = 0; i < n; ++i) std::memcpy(&dst[i], src + 4 * i, 4);
}

void deq_f16(const uint8_t* src, float* dst, uint64_t n) {
    for (uint64_t i = 0; i < n; ++i) dst[i] = f16_to_f32(rd16(src + 2 * i));
}

void deq_bf16(const uint8_t* src, float* dst, uint64_t n) {
    for (uint64_t i = 0; i < n; ++i) {
        const uint32_t bits = uint32_t(rd16(src + 2 * i)) << 16;
        std::memcpy(&dst[i], &bits, 4);
    }
}

// ---- Q8_0 --------------------------------------------------------------------------------------
void deq_q8_0(const uint8_t* src, float* dst, uint64_t n) {
    for (uint64_t i = 0; i < n; i += 32, src += 34) {
        const float d = scale_at(src);
        const int8_t* q = reinterpret_cast<const int8_t*>(src + 2);
        for (int j = 0; j < 32; ++j) dst[i + j] = d * float(q[j]);
    }
}

// ---- Q2_K --------------------------------------------------------------------------------------
void deq_q2_K(const uint8_t* src, float* dst, uint64_t n) {
    for (uint64_t b = 0; b < n; b += 256, src += 84) {
        const uint8_t* scales = src;
        const uint8_t* q = src + 16;
        const float d = scale_at(src + 80);
        const float dmin = scale_at(src + 82);
        uint64_t out = b;
        int is = 0;
        for (int half = 0; half < 2; ++half) {
            for (int j = 0, shift = 0; j < 4; ++j, shift += 2) {
                const uint8_t sc0 = scales[is++];
                const float dl0 = d * float(sc0 & 0xF), ml0 = dmin * float(sc0 >> 4);
                for (int l = 0; l < 16; ++l) dst[out + l] = dl0 * float((q[l] >> shift) & 3) - ml0;
                out += 16;
                const uint8_t sc1 = scales[is++];
                const float dl1 = d * float(sc1 & 0xF), ml1 = dmin * float(sc1 >> 4);
                for (int l = 0; l < 16; ++l) dst[out + l] = dl1 * float((q[l + 16] >> shift) & 3) - ml1;
                out += 16;
            }
            q += 32;
        }
    }
}

// ---- Q3_K --------------------------------------------------------------------------------------
// The 16 six-bit scales arrive as 12 bytes: the low four bits live in bytes 0..7 and the high two
// bits of all sixteen are packed into bytes 8..11. This unpacks them the way the layout says.
void q3_k_scales(const uint8_t* packed, int8_t* out) {
    uint32_t aux[4];
    std::memcpy(aux, packed, 12);
    const uint32_t tmp = aux[2];
    const uint32_t lo = 0x0f0f0f0fu, hi = 0x03030303u;
    uint32_t a0 = (aux[0] & lo) | (((tmp >> 0) & hi) << 4);
    uint32_t a1 = (aux[1] & lo) | (((tmp >> 2) & hi) << 4);
    uint32_t a2 = ((aux[0] >> 4) & lo) | (((tmp >> 4) & hi) << 4);
    uint32_t a3 = ((aux[1] >> 4) & lo) | (((tmp >> 6) & hi) << 4);
    const uint32_t ordered[4] = {a0, a1, a2, a3};
    std::memcpy(out, ordered, 16);
}

void deq_q3_K(const uint8_t* src, float* dst, uint64_t n) {
    for (uint64_t b = 0; b < n; b += 256, src += 110) {
        const uint8_t* hm = src;
        const uint8_t* q = src + 32;
        const float d_all = scale_at(src + 108);
        int8_t scales[16];
        q3_k_scales(src + 96, scales);
        uint64_t out = b;
        int is = 0;
        uint8_t m = 1;
        for (int half = 0; half < 2; ++half) {
            for (int j = 0, shift = 0; j < 4; ++j, shift += 2, m <<= 1) {
                const float dl0 = d_all * float(scales[is++] - 32);
                for (int l = 0; l < 16; ++l)
                    dst[out + l] = dl0 * float(int(((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4)));
                out += 16;
                const float dl1 = d_all * float(scales[is++] - 32);
                for (int l = 0; l < 16; ++l)
                    dst[out + l] = dl1 * float(int(((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4)));
                out += 16;
            }
            q += 32;
        }
    }
}

// ---- Q4_K / Q5_K -------------------------------------------------------------------------------
// The 6-bit scales and mins share 12 bytes: scales in the low six bits of bytes 0..7 and of the low
// nibbles of bytes 8..11, mins in the high six bits.
void k4_scale_min(int j, const uint8_t* q, uint8_t* sc, uint8_t* mn) {
    if (j < 4) {
        *sc = q[j] & 63;
        *mn = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *mn = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

void deq_q4_K(const uint8_t* src, float* dst, uint64_t n) {
    for (uint64_t b = 0; b < n; b += 256, src += 144) {
        const float d = scale_at(src);
        const float dmin = scale_at(src + 2);
        const uint8_t* scales = src + 4;
        const uint8_t* q = src + 16;
        int is = 0;
        uint64_t out = b;
        // 64 elements per step: the same 32 quant bytes, low nibbles then high nibbles, each half with
        // its own scale and min.
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc, mn;
            k4_scale_min(is + 0, scales, &sc, &mn);
            const float d1 = d * float(sc), m1 = dmin * float(mn);
            k4_scale_min(is + 1, scales, &sc, &mn);
            const float d2 = d * float(sc), m2 = dmin * float(mn);
            for (int l = 0; l < 32; ++l) dst[out + l] = d1 * float(q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) dst[out + 32 + l] = d2 * float(q[l] >> 4) - m2;
            out += 64;
            q += 32;
            is += 2;
        }
    }
}

void deq_q5_K(const uint8_t* src, float* dst, uint64_t n) {
    for (uint64_t b = 0; b < n; b += 256, src += 176) {
        const float d = scale_at(src);
        const float dmin = scale_at(src + 2);
        const uint8_t* scales = src + 4;
        const uint8_t* qh = src + 16;
        const uint8_t* ql = src + 48;
        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        uint64_t out = b;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc, mn;
            k4_scale_min(is + 0, scales, &sc, &mn);
            const float d1 = d * float(sc), m1 = dmin * float(mn);
            k4_scale_min(is + 1, scales, &sc, &mn);
            const float d2 = d * float(sc), m2 = dmin * float(mn);
            for (int l = 0; l < 32; ++l) dst[out + l] = d1 * float((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) dst[out + 32 + l] = d2 * float((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
            out += 64;
            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

// ---- Q6_K --------------------------------------------------------------------------------------
void deq_q6_K(const uint8_t* src, float* dst, uint64_t n) {
    for (uint64_t b = 0; b < n; b += 256, src += 210) {
        const uint8_t* ql = src;
        const uint8_t* qh = src + 128;
        const int8_t* sc = reinterpret_cast<const int8_t*>(src + 192);
        const float d = scale_at(src + 208);
        uint64_t out = b;
        for (int half = 0; half < 2; ++half) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int q1 = int((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int q2 = int((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int q3 = int((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int q4 = int((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                dst[out + l + 0] = d * float(sc[is + 0]) * float(q1);
                dst[out + l + 32] = d * float(sc[is + 2]) * float(q2);
                dst[out + l + 64] = d * float(sc[is + 4]) * float(q3);
                dst[out + l + 96] = d * float(sc[is + 6]) * float(q4);
            }
            out += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

// ---- IQ2_S -------------------------------------------------------------------------------------
// qs carries two things: bytes 0..31 are 2-bit grid indices and bytes 32..63 are sign bytes. The
// third index bit comes from qh, two bits per 32-lane group.
void deq_iq2_s(const uint8_t* src, float* dst, uint64_t n) {
    for (uint64_t b = 0; b < n; b += 256, src += 82) {
        const float d = scale_at(src);
        const uint8_t* qs = src + 2;
        const uint8_t* qh = qs + 64;
        const uint8_t* signs = qs + 32;
        const uint8_t* scales = qh + 8;
        uint64_t out = b;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const float dl0 = d * (0.5f + float(scales[ib32] & 0xf)) * 0.25f;
            const float dl1 = d * (0.5f + float(scales[ib32] >> 4)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const float dl = (l / 2 == 0) ? dl0 : dl1;
                const unsigned idx = unsigned(qs[l]) | ((unsigned(qh[ib32]) << (8 - 2 * l)) & 0x300u);
                const uint64_t grid = kIq2sGrid[idx];
                for (int j = 0; j < 8; ++j) {
                    const uint8_t byte = uint8_t(grid >> (8 * j));
                    dst[out + j] = dl * float(byte) * ((signs[l] & kIq2SignMask[j]) ? -1.f : 1.f);
                }
                out += 8;
            }
            qs += 4;
            signs += 4;
        }
    }
}

// ---- IQ3_S -------------------------------------------------------------------------------------
void deq_iq3_s(const uint8_t* src, float* dst, uint64_t n) {
    for (uint64_t b = 0; b < n; b += 256, src += 110) {
        const float d = scale_at(src);
        const uint8_t* qs = src + 2;
        const uint8_t* qh = qs + 64;
        const uint8_t* signs = qh + 8;
        const uint8_t* scales = signs + 32;
        uint64_t out = b;
        for (int ib32 = 0; ib32 < 8; ib32 += 2) {
            const float db1 = d * (1.f + 2.f * float(scales[ib32 / 2] & 0xf));
            const float db2 = d * (1.f + 2.f * float(scales[ib32 / 2] >> 4));
            for (int pass = 0; pass < 2; ++pass) {
                const float db = pass == 0 ? db1 : db2;
                for (int l = 0; l < 4; ++l) {
                    const unsigned i1 = unsigned(qs[2 * l + 0]) | ((unsigned(qh[pass]) << (8 - 2 * l)) & 256u);
                    const unsigned i2 = unsigned(qs[2 * l + 1]) | ((unsigned(qh[pass]) << (7 - 2 * l)) & 256u);
                    const uint32_t g1 = kIq3sGrid[i1], g2 = kIq3sGrid[i2];
                    for (int j = 0; j < 4; ++j) {
                        const uint8_t b1 = uint8_t(g1 >> (8 * j)), b2 = uint8_t(g2 >> (8 * j));
                        dst[out + j + 0] = db * float(b1) * ((signs[l] & kIq2SignMask[j + 0]) ? -1.f : 1.f);
                        dst[out + j + 4] = db * float(b2) * ((signs[l] & kIq2SignMask[j + 4]) ? -1.f : 1.f);
                    }
                    out += 8;
                }
                qs += 8;
                signs += 4;
            }
            qh += 2;
        }
    }
}

// ---- IQ2_XXS / IQ2_XS / IQ3_XXS ----------------------------------------------------------------
// The "xxs" family keeps a 7-bit index into a table of 128 sign patterns (the 8th sign is the parity of
// the other seven, which is why the table has 128 entries) next to the grid index.
inline uint32_t rd32(const uint8_t* p) { return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24); }
inline float signed_byte(uint64_t grid, int j, uint8_t signs) {
    return float(uint8_t(grid >> (8 * j))) * ((signs & kIq2SignMask[j]) ? -1.f : 1.f);
}

void deq_iq2_xxs(const uint8_t* src, float* dst, uint64_t n) {
    for (uint64_t b = 0; b < n; b += 256, src += 66) {
        const float d = scale_at(src);
        const uint8_t* qs = src + 2;
        uint64_t out = b;
        for (int ib32 = 0; ib32 < 8; ++ib32, qs += 8) {
            const uint32_t idx_word = rd32(qs), sign_word = rd32(qs + 4);
            const float db = d * (0.5f + float(sign_word >> 28)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const uint64_t grid = kIq2xxsGrid[(idx_word >> (8 * l)) & 0xff];
                const uint8_t signs = kSignsIq2xs[(sign_word >> (7 * l)) & 127];
                for (int j = 0; j < 8; ++j) dst[out + j] = db * signed_byte(grid, j, signs);
                out += 8;
            }
        }
    }
}

void deq_iq2_xs(const uint8_t* src, float* dst, uint64_t n) {
    for (uint64_t b = 0; b < n; b += 256, src += 74) {
        const float d = scale_at(src);
        const uint8_t* qs = src + 2;
        const uint8_t* scales = src + 66;
        uint64_t out = b;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const float db[2] = {d * (0.5f + float(scales[ib32] & 0xf)) * 0.25f,
                                 d * (0.5f + float(scales[ib32] >> 4)) * 0.25f};
            for (int l = 0; l < 4; ++l) {
                const unsigned q = rd16(qs + 2 * (4 * ib32 + l));
                const uint64_t grid = kIq2xsGrid[q & 511];
                const uint8_t signs = kSignsIq2xs[q >> 9];
                for (int j = 0; j < 8; ++j) dst[out + j] = db[l / 2] * signed_byte(grid, j, signs);
                out += 8;
            }
        }
    }
}

void deq_iq3_xxs(const uint8_t* src, float* dst, uint64_t n) {
    for (uint64_t b = 0; b < n; b += 256, src += 98) {
        const float d = scale_at(src);
        const uint8_t* qs = src + 2;
        const uint8_t* tail = qs + 64;   // one scale-and-signs word per 32 lanes
        uint64_t out = b;
        for (int ib32 = 0; ib32 < 8; ++ib32, qs += 8) {
            const uint32_t word = rd32(tail + 4 * ib32);
            const float db = d * (0.5f + float(word >> 28)) * 0.5f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t signs = kSignsIq2xs[(word >> (7 * l)) & 127];
                const uint32_t g1 = kIq3xxsGrid[qs[2 * l + 0]], g2 = kIq3xxsGrid[qs[2 * l + 1]];
                for (int j = 0; j < 4; ++j) {
                    dst[out + j + 0] = db * float(uint8_t(g1 >> (8 * j))) * ((signs & kIq2SignMask[j + 0]) ? -1.f : 1.f);
                    dst[out + j + 4] = db * float(uint8_t(g2 >> (8 * j))) * ((signs & kIq2SignMask[j + 4]) ? -1.f : 1.f);
                }
                out += 8;
            }
        }
    }
}

// ---- IQ1_S / IQ1_M -----------------------------------------------------------------------------
// A grid entry is 8 signed bytes in {-1, 0, 1}; the value is (grid + delta) with delta = +-1/8.
constexpr float kIq1Delta = 0.125f;
inline float grid_i8(uint64_t grid, int j) { return float(int8_t(uint8_t(grid >> (8 * j)))); }

void deq_iq1_s(const uint8_t* src, float* dst, uint64_t n) {
    for (uint64_t b = 0; b < n; b += 256, src += 50) {
        const float d = scale_at(src);
        const uint8_t* qs = src + 2;
        const uint8_t* qh = src + 34;
        uint64_t out = b;
        for (int ib = 0; ib < 8; ++ib, qs += 4) {
            const unsigned h = rd16(qh + 2 * ib);
            const float dl = d * float(2 * ((h >> 12) & 7) + 1);
            const float delta = (h & 0x8000) ? -kIq1Delta : kIq1Delta;
            for (int l = 0; l < 4; ++l) {
                const uint64_t grid = kIq1sGrid[unsigned(qs[l]) | (((h >> (3 * l)) & 7) << 8)];
                for (int j = 0; j < 8; ++j) dst[out + j] = dl * (grid_i8(grid, j) + delta);
                out += 8;
            }
        }
    }
}

void deq_iq1_m(const uint8_t* src, float* dst, uint64_t n) {
    for (uint64_t b = 0; b < n; b += 256, src += 56) {
        const uint8_t* qs = src;
        const uint8_t* qh = src + 32;
        const uint16_t sc[4] = {uint16_t(rd16(src + 48)), uint16_t(rd16(src + 50)), uint16_t(rd16(src + 52)),
                                uint16_t(rd16(src + 54))};
        // the super-scale is an f16 whose four nibbles ride in the top nibble of each scale word
        const uint16_t sbits = uint16_t((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
        const float d = f16_to_f32(sbits);
        uint64_t out = b;
        for (int ib = 0; ib < 8; ++ib, qs += 4, qh += 2) {
            const float dl[2] = {d * float(2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 7) + 1),
                                 d * float(2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 7) + 1)};
            const unsigned idx[4] = {unsigned(qs[0]) | ((unsigned(qh[0]) << 8) & 0x700u),
                                     unsigned(qs[1]) | ((unsigned(qh[0]) << 4) & 0x700u),
                                     unsigned(qs[2]) | ((unsigned(qh[1]) << 8) & 0x700u),
                                     unsigned(qs[3]) | ((unsigned(qh[1]) << 4) & 0x700u)};
            const float delta[4] = {(qh[0] & 0x08) ? -kIq1Delta : kIq1Delta, (qh[0] & 0x80) ? -kIq1Delta : kIq1Delta,
                                    (qh[1] & 0x08) ? -kIq1Delta : kIq1Delta, (qh[1] & 0x80) ? -kIq1Delta : kIq1Delta};
            for (int l = 0; l < 4; ++l) {
                const uint64_t grid = kIq1sGrid[idx[l]];
                for (int j = 0; j < 8; ++j) dst[out + j] = dl[l / 2] * (grid_i8(grid, j) + delta[l]);
                out += 8;
            }
        }
    }
}

// ---- IQ4_XS / IQ4_NL ---------------------------------------------------------------------------
void deq_iq4_xs(const uint8_t* src, float* dst, uint64_t n) {
    for (uint64_t b = 0; b < n; b += 256, src += 136) {
        const float d = scale_at(src);
        const uint16_t scales_h = rd16(src + 2);
        const uint8_t* scales_l = src + 4;
        const uint8_t* qs = src + 8;
        uint64_t out = b;
        for (int ib = 0; ib < 8; ++ib) {
            const int ls = int((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) | (int((scales_h >> (2 * ib)) & 3) << 4);
            const float dl = d * float(ls - 32);
            for (int j = 0; j < 16; ++j) {
                dst[out + j + 0] = dl * float(kIq4NlValues[qs[j] & 0xf]);
                dst[out + j + 16] = dl * float(kIq4NlValues[qs[j] >> 4]);
            }
            out += 32;
            qs += 16;
        }
    }
}

void deq_iq4_nl(const uint8_t* src, float* dst, uint64_t n) {
    for (uint64_t b = 0; b < n; b += 32, src += 18) {
        const float d = scale_at(src);
        const uint8_t* qs = src + 2;
        for (int j = 0; j < 16; ++j) {
            dst[b + j + 0] = d * float(kIq4NlValues[qs[j] & 0xf]);
            dst[b + j + 16] = d * float(kIq4NlValues[qs[j] >> 4]);
        }
    }
}

// ---- Q2_0 --------------------------------------------------------------------------------------
void deq_q2_0(const uint8_t* src, float* dst, uint64_t n) {
    for (uint64_t b = 0; b < n; b += 64, src += 18) {
        const float d = scale_at(src);
        const uint8_t* qs = src + 2;
        for (int j = 0; j < 64; ++j) {
            const int q = (qs[j / 4] >> (2 * (j % 4))) & 3;
            dst[b + j] = float(q - 1) * d;  // 0,1,2,3 -> -1, 0, +1, +2
        }
    }
}

// ---- NVFP4 -------------------------------------------------------------------------------------
// ggml_ue4m3_to_fp32 (ggml-impl.h): unsigned, 4 exponent bits with bias 7, 3 mantissa bits, and the
// result is HALVED to match the doubled codebook. 0x00 and 0x7F decode to zero -- ggml's own quirk,
// copied deliberately: the parity gate compares value for value, so a "better" NaN would fail it.
inline float ue4m3_to_f32(uint8_t x) {
    if (x == 0 || x == 0x7f) return 0.0f;
    const int exp = (x >> 3) & 0xf, man = x & 0x7;
    const float raw = exp == 0 ? std::ldexp(float(man), -9) : std::ldexp(1.0f + float(man) / 8.0f, exp - 7);
    return raw * 0.5f;
}

void deq_nvfp4(const uint8_t* src, float* dst, uint64_t n) {
    for (uint64_t b = 0; b < n; b += 64, src += 36) {
        for (int s = 0; s < 4; ++s) {
            const float d = ue4m3_to_f32(src[s]);  // one UE4M3 scale per 16 elements
            const uint8_t* qs = src + 4 + 8 * s;
            for (int j = 0; j < 8; ++j) {
                dst[b + 16 * s + j] = d * float(kFp4Values[qs[j] & 0xf]);
                dst[b + 16 * s + j + 8] = d * float(kFp4Values[qs[j] >> 4]);
            }
        }
    }
}

}  // namespace

bool supported(uint32_t type) {
    switch (static_cast<Type>(type)) {
        case Type::F32:
        case Type::F16:
        case Type::BF16:
        case Type::Q8_0:
        case Type::Q2_K:
        case Type::Q3_K:
        case Type::Q4_K:
        case Type::Q5_K:
        case Type::Q6_K:
        case Type::IQ2_XXS:
        case Type::IQ2_XS:
        case Type::IQ2_S:
        case Type::IQ3_XXS:
        case Type::IQ3_S:
        case Type::IQ1_S:
        case Type::IQ1_M:
        case Type::IQ4_XS:
        case Type::IQ4_NL:
        case Type::Q2_0:
        case Type::NVFP4:
            return true;
        default:
            return false;
    }
}

bool dequant_row(uint32_t type, const uint8_t* src, float* dst, uint64_t n) {
    switch (static_cast<Type>(type)) {
        case Type::F32: deq_f32(src, dst, n); return true;
        case Type::F16: deq_f16(src, dst, n); return true;
        case Type::BF16: deq_bf16(src, dst, n); return true;
        case Type::Q8_0: deq_q8_0(src, dst, n); return true;
        case Type::Q2_K: deq_q2_K(src, dst, n); return true;
        case Type::Q3_K: deq_q3_K(src, dst, n); return true;
        case Type::Q4_K: deq_q4_K(src, dst, n); return true;
        case Type::Q5_K: deq_q5_K(src, dst, n); return true;
        case Type::Q6_K: deq_q6_K(src, dst, n); return true;
        case Type::IQ2_XXS: deq_iq2_xxs(src, dst, n); return true;
        case Type::IQ2_XS: deq_iq2_xs(src, dst, n); return true;
        case Type::IQ2_S: deq_iq2_s(src, dst, n); return true;
        case Type::IQ3_XXS: deq_iq3_xxs(src, dst, n); return true;
        case Type::IQ3_S: deq_iq3_s(src, dst, n); return true;
        case Type::IQ1_S: deq_iq1_s(src, dst, n); return true;
        case Type::IQ1_M: deq_iq1_m(src, dst, n); return true;
        case Type::IQ4_XS: deq_iq4_xs(src, dst, n); return true;
        case Type::IQ4_NL: deq_iq4_nl(src, dst, n); return true;
        case Type::Q2_0: deq_q2_0(src, dst, n); return true;
        case Type::NVFP4: deq_nvfp4(src, dst, n); return true;
        default: return false;
    }
}

uint16_t f32_to_f16(float f) {
    uint32_t b;
    std::memcpy(&b, &f, 4);
    const uint32_t sign = (b >> 16) & 0x8000u;
    const int32_t exp = int32_t((b >> 23) & 0xff) - 127 + 15;
    uint32_t man = b & 0x7fffffu;
    if (((b >> 23) & 0xff) == 0xff) return uint16_t(sign | 0x7c00u | (man ? 0x200u : 0u));  // inf / nan
    if (exp >= 31) return uint16_t(sign | 0x7c00u);                                          // overflow
    if (exp <= 0) {                                                                          // subnormal or zero
        if (exp < -10) return uint16_t(sign);
        man |= 0x800000u;
        const uint32_t shift = uint32_t(14 - exp);
        uint32_t h = man >> shift;
        const uint32_t rem = man & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1))) ++h;
        return uint16_t(sign | h);
    }
    uint32_t h = (uint32_t(exp) << 10) | (man >> 13);
    const uint32_t rem = man & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1))) ++h;  // a carry into the exponent is the right result
    return uint16_t(sign | h);
}

void quantize_q8_0(const float* src, uint8_t* dst, uint64_t n) {
    for (uint64_t b = 0; b < n / 32; ++b) {
        const float* x = src + b * 32;
        float amax = 0;
        for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[i]));
        const uint16_t dh = f32_to_f16(amax / 127.0f);
        const float d = f16_to_f32(dh);
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        uint8_t* o = dst + b * 34;
        std::memcpy(o, &dh, 2);
        for (int i = 0; i < 32; ++i) o[2 + i] = uint8_t(int8_t(std::lround(x[i] * id)));
    }
}

}  // namespace limbo::quants
