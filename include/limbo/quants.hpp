// Copyright (c) 2026 ITERFORM. All rights reserved.
#pragma once
// limbo/quants.hpp — weight decoding: GGUF block formats to float.
//
// The block layouts and the bit-level meaning of each format are interoperability facts
// (CLEANROOM.md §3) and are written down here exactly, format by format. The codebook grids are
// ggml's, copied verbatim under the MIT notice (CLEANROOM.md §4, THIRD_PARTY_NOTICES). The decoding
// code is Limbo's own.
//
//   F32  1 elem  4 B     plain float
//   F16  1 elem  2 B     IEEE half
//   BF16 1 elem  2 B     bfloat16 (high half of an IEEE float)
//   Q8_0    32/34    { f16 d; i8 qs[32]; }                       v = d*q
//   Q2_K   256/84    { u8 scales[16]; u8 qs[64]; f16 d, dmin; }   v = d*(sc&15)*q2 - dmin*(sc>>4)
//                    two 128-element halves, 2 bits per element, shift 0/2/4/6 per 16-lane group
//   Q3_K   256/110   { u8 hmask[32]; u8 qs[64]; u8 scales[12]; f16 d; }
//                    2 low bits from qs, the third bit from hmask (absent bit reads as -4),
//                    scales are 16 six-bit values packed into 12 bytes, minus 32
//   Q4_K   256/144   { f16 d, dmin; u8 scales[12]; u8 qs[128]; }  v = d*sc*q4 - dmin*m
//   Q5_K   256/176   as Q4_K plus { u8 qh[32]; } for the fifth bit
//   Q6_K   256/210   { u8 ql[128]; u8 qh[64]; i8 scales[16]; f16 d; }  v = d*scale*(q6-32)
//   IQ2_S  256/82    { f16 d; u8 qs[64]; u8 qh[8]; u8 scales[8]; }
//                    qs[0..31] are 2-bit grid indices, qs[32..63] are sign bytes; qh holds the third
//                    index bit; scales hold a nibble per 16 lanes, 2 per byte;
//                    v = d*(0.5+nibble)*0.25 * grid8 * sign
//   IQ3_S  256/110   { f16 d; u8 qs[64]; u8 qh[8]; u8 signs[32]; u8 scales[4]; }
//                    3-bit grid indices with the high bit in qh, 4-bit scales in nibbles,
//                    v = d*(1+2*nibble) * grid4 * sign
//   IQ2_XXS 256/66  { f16 d; u16 qs[32]; }    per 32 lanes two u32 words: word 0 = 4 grid-index bytes, word 1 =
//                    four 7-bit sign indices (low 28 bits) and a 4-bit scale (top): v = d*(0.5+s)*0.25*grid8*sign
//   IQ2_XS  256/74  { f16 d; u16 qs[32]; u8 scales[8]; }  each u16 = 9-bit grid index + 7-bit sign index;
//                    a scale nibble per 16 lanes: v = d*(0.5+nibble)*0.25*grid8*sign
//   IQ3_XXS 256/98  { f16 d; u8 qs[96]; }     qs[0..63] grid indices (4 lanes each), qs[64..95] one u32 per 32
//                    lanes: four 7-bit sign indices and a 4-bit scale: v = d*(0.5+s)*0.5*grid4*sign
//   IQ1_S   256/50  { f16 d; u8 qs[32]; u16 qh[8]; }  per 32 lanes: 3 bits of scale, a sign bit for the +-1/8
//                    delta and three index bits per 8 lanes: v = d*(2*s+1)*(grid_i8 + delta)
//   IQ1_M   256/56  { u8 qs[32]; u8 qh[16]; u16 scales[4]; }  as IQ1_S with a delta sign and 3 index bits per
//                    8 lanes in qh and two 3-bit scales per 32 lanes; the f16 super-scale is spread over the
//                    top nibbles of the four scale words
//   IQ4_XS 256/136   { f16 d; u16 scales_h; u8 scales_l[4]; u8 qs[128]; }
//                    6-bit scale per 32 lanes: v = d*(ls-32) * iq4nl[nibble]
//   IQ4_NL  32/18    { f16 d; u8 qs[16]; }                        v = d * iq4nl[nibble]
//   Q2_0    64/18    { f16 d; u8 qs[16]; }                        code 0,1,2,3 -> -1,0,+1,+2 times d
//   NVFP4   64/36    { u8 d[4]; u8 qs[32]; }  4 UE4M3 sub-block scales (one per 16 elements) then 32
//                    bytes of packed 4-bit E2M1: v = ue4m3(d[s]) * fp4[nibble]. The scale is stored
//                    HALVED because the codebook holds the DOUBLED E2M1 magnitudes, and 0x00/0x7F
//                    decode to zero -- both are ggml's behaviour, quirks included.
//
// dequant_row decodes `n` elements from a block-aligned source. It returns false for a type it does
// not implement, rather than returning numbers nobody should trust.

#include <cstdint>
#include <cstring>

namespace limbo::quants {

// The ggml codebook grids (src/ggml_grids.cpp, generated from ggml under its MIT notice).
extern const uint8_t kIq2SignMask[8];
extern const uint64_t kIq2sGrid[1024];
extern const uint32_t kIq3sGrid[512];
extern const int8_t kIq4NlValues[16];
// The grids of IQ2_XXS, IQ2_XS, IQ3_XXS and IQ1_S/IQ1_M and the 7-bit sign table they share
// (src/ggml_grids_ext.cpp, the same provenance).
extern const uint8_t kSignsIq2xs[128];
extern const uint64_t kIq2xxsGrid[256];
extern const uint64_t kIq2xsGrid[512];
extern const uint32_t kIq3xxsGrid[256];
extern const uint64_t kIq1sGrid[2048];

// The doubled E2M1 codebook NVFP4 (and ggml's MXFP4) index through (src/ggml_grids_ext.cpp).
extern const int8_t kFp4Values[16];

// IEEE half to float, for the formats whose scale is an f16.
inline float f16_to_f32(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000) << 16;
    const uint32_t exp = (h >> 10) & 0x1f;
    const uint32_t man = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            int e = -1;
            uint32_t m = man;
            do {
                ++e;
                m <<= 1;
            } while ((m & 0x400) == 0);
            bits = sign | (uint32_t(127 - 15 - e) << 23) | ((m & 0x3ff) << 13);
        }
    } else if (exp == 0x1f) {
        bits = sign | 0x7f800000u | (man << 13);
    } else {
        bits = sign | ((exp + 112) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// Q8_0 quantiser (a drafter's weights are made at load, not read from a file): n elements, a multiple of 32, to n / 32 blocks of
// { f16 d; i8 qs[32] }, d = amax / 127 rounded to f16, qs = round(x / d) (ties away from zero). 34 * n / 32 bytes at dst.
void quantize_q8_0(const float* src, uint8_t* dst, uint64_t n);
// float to IEEE half, round to nearest even
uint16_t f32_to_f16(float f);

// True when dequant_row implements this type id.
bool supported(uint32_t type);

// Decodes n elements of `type` from src into dst. false when the type is not implemented.
bool dequant_row(uint32_t type, const uint8_t* src, float* dst, uint64_t n);

}  // namespace limbo::quants
