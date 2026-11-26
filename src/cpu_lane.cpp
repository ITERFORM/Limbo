// Copyright (c) 2026 ITERFORM. All rights reserved.
// src/cpu_lane.cpp — the CPU lane's kernels: block decode, int8 activations, union-batched dots.
//
// Written from the block layouts in include/limbo/quants.hpp (interoperability facts, CLEANROOM.md §3) and
// from the shape of the three donor specs, not from any donor's source:
//   - docs/donors/kindi.md §7: union batching (a weight row read once for every token that picked the expert),
//     the sign-trick int8 idiom, one signed table per codebook format;
//   - docs/donors/modchip.md §2.1/§2.3: signed-by-signed through |w| and sign(a, w), maddubs then madd with the
//     per-sub-block integer scales, activations quantized once to +-127 (psign of -128 wraps), K=2..4 columns
//     fit the 16 ymm registers, 8 do not (so the columns are walked in groups of 4);
//   - docs/donors/strata.md §5/§7: the activation is quantized once per layer and shared by every expert.
//
// Shape of a kernel. A weight block (256 elements, or 32/64 for IQ4_NL/Q2_0) is decoded ONCE into a small
// L1-resident `Block`: int8 weights, one int16 scale vector per 32-lane chunk (the per-16 sub-block scales
// broadcast the way madd wants them), the per-16 minimum terms where the format has them, and the block's
// float scales. Then every activation column (a row of the expert's tokens, times the activation planes) is
// dotted against the decoded block. Three variants of that dot:
//   U  unsigned weights (K-quants; Q3_K and Q6_K are shifted to unsigned and the shift becomes a minimum
//      term): maddubs(w, a), madd by the sub-block scales, minus dmin * sum(m_j * bsum_j)
//   S  signed weights (the codebook formats): maddubs(|w|, sign(a, w)), madd by the scales
//   F  signed weights with one float scale per 32 lanes (IQ4_NL, Q2_0)
// The scalar ISA is the plain definition: quants::dequant_row for the weights, float dot against the
// dequantized activation columns. It exists as the fallback and as the cross-check of the int8 path.
//
// AVX-512: not implemented, never ran on either box (PLAN.md §2). Isa::Avx512 resolves to the AVX2 kernels.
#include "limbo/cpu_lane.hpp"

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "limbo/quants.hpp"
#include "limbo/types.hpp"

namespace limbo::cpu {

#define LIMBO_AVX2 __attribute__((target("avx2,fma")))
#define LIMBO_AI inline __attribute__((always_inline))

Isa cpu_isa() {
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512vl"))
        return Isa::Avx512;
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) return Isa::Avx2;
    return Isa::Scalar;
}

Isa detect_isa() {
    // No AVX-512 kernel exists, so the best level this build can run is AVX2 even on an AVX-512 machine.
    return cpu_isa() == Isa::Scalar ? Isa::Scalar : Isa::Avx2;
}

const char* isa_name(Isa isa) {
    switch (isa) {
        case Isa::Scalar: return "scalar";
        case Isa::Avx2: return "avx2";
        case Isa::Avx512: return "avx512(reserved: no kernels, never ran)";
    }
    return "?";
}

namespace {

using quants::f16_to_f32;

inline uint16_t rd16(const uint8_t* p) { return uint16_t(p[0] | (p[1] << 8)); }
inline uint32_t rd32(const uint8_t* p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}
inline float h2f(const uint8_t* p) { return f16_to_f32(rd16(p)); }

// ---- tables --------------------------------------------------------------------------------------------
struct Tables {
    uint64_t sgn[256];   // a sign byte -> 0xFF in every byte whose bit is set
    uint32_t q20[256];   // a Q2_0 byte -> four int8 values (code - 1)
    Tables() {
        for (int s = 0; s < 256; ++s) {
            uint64_t m = 0;
            for (int j = 0; j < 8; ++j)
                if (s & (1 << j)) m |= uint64_t(0xFF) << (8 * j);
            sgn[s] = m;
        }
        for (int b = 0; b < 256; ++b) {
            uint32_t v = 0;
            for (int j = 0; j < 4; ++j) {
                const int c = ((b >> (2 * j)) & 3) - 1;
                v |= uint32_t(uint8_t(int8_t(c))) << (8 * j);
            }
            q20[b] = v;
        }
    }
};
const Tables& tab() {
    static const Tables t;
    return t;
}

enum class Var : uint8_t { U, S, F };

// One decoded weight block.
struct alignas(32) Block {
    int8_t w[256];        // U: unsigned 0..127 stored as int8; S, F: signed
    int8_t wa[256];       // |w| (S, F)
    int16_t sv[8][16];    // per 32-lane chunk: the per-16 integer scales, 8 lanes each (madd pairs lanes 2l, 2l+1)
    int16_t mv[16];       // U: per-16 minimum terms
    float d, dmin;        // the block's float scales (U, S)
    float fd[8];          // F: one float scale per 32-lane chunk
};

LIMBO_AVX2 LIMBO_AI void put_chunk(Block& o, int k, __m256i v) {
    _mm256_store_si256(reinterpret_cast<__m256i*>(o.w + 32 * k), v);
}
// Chunk k's scales when the format has one scale per 32 lanes.
LIMBO_AVX2 LIMBO_AI void put_sv32(Block& o, int k, int sc) {
    _mm256_store_si256(reinterpret_cast<__m256i*>(o.sv[k]), _mm256_set1_epi16(int16_t(sc)));
}
// ... and when it has one per 16 lanes (lo for lanes 0..15 of the chunk, hi for lanes 16..31).
LIMBO_AVX2 LIMBO_AI void put_sv16(Block& o, int k, int lo, int hi) {
    _mm256_store_si256(reinterpret_cast<__m256i*>(o.sv[k]),
                       _mm256_set_m128i(_mm_set1_epi16(int16_t(hi)), _mm_set1_epi16(int16_t(lo))));
}

// The grid magnitudes g with the chunk's 32 sign bits s4 applied: byte k takes bit k % 8 of sign byte k / 8.
LIMBO_AVX2 LIMBO_AI __m256i signed_gv(__m256i g, uint32_t s4) {
    const __m256i rep = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
                                         3, 3, 3);
    const __m256i bits = _mm256_set1_epi64x(int64_t(0x8040201008040201ull));
    const __m256i m = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32(int(s4)), rep), bits), bits);
    return _mm256_sub_epi8(_mm256_xor_si256(g, m), m);
}

// bit `bit` of every byte of v, moved to bit 4 (the fifth bit of Q5_K); two bits at `shift`, moved to bits 4..5 (Q6_K)
LIMBO_AVX2 LIMBO_AI __m256i bit_to_4(__m256i v, int bit) {
    return _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(v, bit), _mm256_set1_epi8(1)), 4);
}
LIMBO_AVX2 LIMBO_AI __m256i two_to_4(__m256i v, int shift) {
    return _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(v, shift), _mm256_set1_epi8(3)), 4);
}

// ---- decoders ------------------------------------------------------------------------------------------
// Each policy: the type id, elements and bytes of a block, the dot variant, and decode().

// The 12-byte scale/min packing of Q4_K and Q5_K.
LIMBO_AI void k4_scales(const uint8_t* q, int* sc, int* mn) {
    for (int j = 0; j < 8; ++j) {
        if (j < 4) {
            sc[j] = q[j] & 63;
            mn[j] = q[j + 4] & 63;
        } else {
            sc[j] = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
            mn[j] = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        }
    }
}

struct DQ4K {
    static constexpr uint32_t type = 12;
    static constexpr int elems = 256, bytes = 144;
    static constexpr Var var = Var::U;
    static constexpr bool minv = true;
    LIMBO_AVX2 static LIMBO_AI void decode(const Tables&, const uint8_t* b, Block& o) {
        o.d = h2f(b);
        o.dmin = h2f(b + 2);
        int sc[8], mn[8];
        k4_scales(b + 4, sc, mn);
        const uint8_t* qs = b + 16;
        const __m256i m4 = _mm256_set1_epi8(0x0F);
        for (int g = 0; g < 4; ++g) {
            const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qs + 32 * g));
            put_chunk(o, 2 * g, _mm256_and_si256(v, m4));
            put_chunk(o, 2 * g + 1, _mm256_and_si256(_mm256_srli_epi16(v, 4), m4));
        }
        for (int k = 0; k < 8; ++k) put_sv32(o, k, sc[k]);
        for (int j = 0; j < 16; ++j) o.mv[j] = int16_t(mn[j / 2]);
    }
};

struct DQ5K {
    static constexpr uint32_t type = 13;
    static constexpr int elems = 256, bytes = 176;
    static constexpr Var var = Var::U;
    static constexpr bool minv = true;
    LIMBO_AVX2 static LIMBO_AI void decode(const Tables&, const uint8_t* b, Block& o) {
        o.d = h2f(b);
        o.dmin = h2f(b + 2);
        int sc[8], mn[8];
        k4_scales(b + 4, sc, mn);
        const __m256i qh = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + 16));
        const uint8_t* ql = b + 48;
        const __m256i m4 = _mm256_set1_epi8(0x0F);
        // group g: low nibbles take bit 2g of qh as their fifth bit, high nibbles bit 2g+1
        const __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql));
        const __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql + 32));
        const __m256i v2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql + 64));
        const __m256i v3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql + 96));
        put_chunk(o, 0, _mm256_or_si256(_mm256_and_si256(v0, m4), bit_to_4(qh, 0)));
        put_chunk(o, 1, _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(v0, 4), m4), bit_to_4(qh, 1)));
        put_chunk(o, 2, _mm256_or_si256(_mm256_and_si256(v1, m4), bit_to_4(qh, 2)));
        put_chunk(o, 3, _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(v1, 4), m4), bit_to_4(qh, 3)));
        put_chunk(o, 4, _mm256_or_si256(_mm256_and_si256(v2, m4), bit_to_4(qh, 4)));
        put_chunk(o, 5, _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(v2, 4), m4), bit_to_4(qh, 5)));
        put_chunk(o, 6, _mm256_or_si256(_mm256_and_si256(v3, m4), bit_to_4(qh, 6)));
        put_chunk(o, 7, _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(v3, 4), m4), bit_to_4(qh, 7)));
        for (int k = 0; k < 8; ++k) put_sv32(o, k, sc[k]);
        for (int j = 0; j < 16; ++j) o.mv[j] = int16_t(mn[j / 2]);
    }
};

struct DQ6K {
    static constexpr uint32_t type = 14;
    static constexpr int elems = 256, bytes = 210;
    static constexpr Var var = Var::U;
    static constexpr bool minv = true;
    LIMBO_AVX2 static LIMBO_AI void decode(const Tables&, const uint8_t* b, Block& o) {
        o.d = h2f(b + 208);
        o.dmin = o.d;   // v = d * sc * (q - 32): the -32 is a minimum term of 32 * sc
        const int8_t* sc = reinterpret_cast<const int8_t*>(b + 192);
        const __m256i m4 = _mm256_set1_epi8(0x0F);
        for (int h = 0; h < 2; ++h) {
            const uint8_t* ql = b + 64 * h;
            const __m256i A = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql));
            const __m256i B = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql + 32));
            const __m256i C = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + 128 + 32 * h));
            put_chunk(o, 4 * h + 0, _mm256_or_si256(_mm256_and_si256(A, m4), two_to_4(C, 0)));
            put_chunk(o, 4 * h + 1, _mm256_or_si256(_mm256_and_si256(B, m4), two_to_4(C, 2)));
            put_chunk(o, 4 * h + 2, _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(A, 4), m4), two_to_4(C, 4)));
            put_chunk(o, 4 * h + 3, _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(B, 4), m4), two_to_4(C, 6)));
        }
        for (int k = 0; k < 8; ++k) put_sv16(o, k, sc[2 * k], sc[2 * k + 1]);
        for (int j = 0; j < 16; ++j) o.mv[j] = int16_t(32 * int(sc[j]));
    }
};

struct DQ3K {
    static constexpr uint32_t type = 11;
    static constexpr int elems = 256, bytes = 110;
    static constexpr Var var = Var::U;
    static constexpr bool minv = true;
    LIMBO_AVX2 static LIMBO_AI void decode(const Tables&, const uint8_t* b, Block& o) {
        o.d = h2f(b + 108);
        o.dmin = o.d;   // v = d * sc * (u - 4), u = low two bits | (high bit set ? 4 : 0)
        // The sixteen 6-bit scales: low nibbles in bytes 0..7, the high two bits of all of them in bytes 8..11.
        const uint8_t* p = b + 96;
        int sc[16];
        for (int j = 0; j < 16; ++j) {
            const int lo = (j < 8) ? (p[j] & 0xF) : (p[j - 8] >> 4);
            const int hi = (p[8 + (j & 3)] >> (2 * (j >> 2))) & 3;
            sc[j] = (lo | (hi << 4)) - 32;
        }
        const __m256i hm = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
        const __m256i m3 = _mm256_set1_epi8(3), four = _mm256_set1_epi8(4), zero = _mm256_setzero_si256();
        for (int h = 0; h < 2; ++h) {
            const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + 32 + 32 * h));
            for (int j = 0; j < 4; ++j) {
                const __m256i lo2 = _mm256_and_si256(_mm256_srli_epi16(v, 2 * j), m3);
                const __m256i bit = _mm256_and_si256(hm, _mm256_set1_epi8(char(1 << (4 * h + j))));
                const __m256i set = _mm256_andnot_si256(_mm256_cmpeq_epi8(bit, zero), four);   // 4 where the bit is set
                put_chunk(o, 4 * h + j, _mm256_or_si256(lo2, set));
            }
        }
        for (int k = 0; k < 8; ++k) put_sv16(o, k, sc[2 * k], sc[2 * k + 1]);
        for (int j = 0; j < 16; ++j) o.mv[j] = int16_t(4 * sc[j]);
    }
};

struct DQ2K {
    static constexpr uint32_t type = 10;
    static constexpr int elems = 256, bytes = 84;
    static constexpr Var var = Var::U;
    static constexpr bool minv = true;
    LIMBO_AVX2 static LIMBO_AI void decode(const Tables&, const uint8_t* b, Block& o) {
        o.d = h2f(b + 80);
        o.dmin = h2f(b + 82);
        const __m256i m3 = _mm256_set1_epi8(3);
        for (int h = 0; h < 2; ++h) {
            const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + 16 + 32 * h));
            put_chunk(o, 4 * h + 0, _mm256_and_si256(v, m3));
            put_chunk(o, 4 * h + 1, _mm256_and_si256(_mm256_srli_epi16(v, 2), m3));
            put_chunk(o, 4 * h + 2, _mm256_and_si256(_mm256_srli_epi16(v, 4), m3));
            put_chunk(o, 4 * h + 3, _mm256_and_si256(_mm256_srli_epi16(v, 6), m3));
        }
        for (int k = 0; k < 8; ++k) put_sv16(o, k, b[2 * k] & 0xF, b[2 * k + 1] & 0xF);
        for (int j = 0; j < 16; ++j) o.mv[j] = int16_t(b[j] >> 4);
    }
};

// Four 8-byte grid entries into one vector by loads and lane inserts: never stored and reloaded as a whole (four
// 8-byte stores read back by one 32-byte load fail store forwarding, which made every chunk wait on the stores).
LIMBO_AVX2 LIMBO_AI __m256i gv64(const uint64_t* g, unsigned i0, unsigned i1, unsigned i2, unsigned i3) {
    const __m128d lo = _mm_loadh_pd(_mm_castsi128_pd(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(g + i0))),
                                    reinterpret_cast<const double*>(g + i1));
    const __m128d hi = _mm_loadh_pd(_mm_castsi128_pd(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(g + i2))),
                                    reinterpret_cast<const double*>(g + i3));
    return _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_castpd_si128(lo)), _mm_castpd_si128(hi), 1);
}

// Eight 4-byte grid entries by loads and inserts (the gather's alternative).
LIMBO_AVX2 LIMBO_AI __m256i gv32(const uint32_t* g, const unsigned* i) {
    __m128i lo = _mm_cvtsi32_si128(int(g[i[0]]));
    lo = _mm_insert_epi32(lo, int(g[i[1]]), 1);
    lo = _mm_insert_epi32(lo, int(g[i[2]]), 2);
    lo = _mm_insert_epi32(lo, int(g[i[3]]), 3);
    __m128i hi = _mm_cvtsi32_si128(int(g[i[4]]));
    hi = _mm_insert_epi32(hi, int(g[i[5]]), 1);
    hi = _mm_insert_epi32(hi, int(g[i[6]]), 2);
    hi = _mm_insert_epi32(hi, int(g[i[7]]), 3);
    return _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
}

struct DIQ2S {
    static constexpr uint32_t type = 22;
    static constexpr int elems = 256, bytes = 82;
    static LIMBO_AI float scale(const uint8_t* b) { return h2f(b) * 0.125f; }
    // chunk c: its four 8-lane grid magnitudes as one vector, its four sign bytes, the integer scales of its two halves
    LIMBO_AVX2 static LIMBO_AI __m256i gv(const uint8_t* b, int c) {
        const uint8_t* qs = b + 2 + 4 * c;
        const unsigned h = b[66 + c];
        return gv64(quants::kIq2sGrid, unsigned(qs[0]) | ((h << 8) & 0x300u), unsigned(qs[1]) | ((h << 6) & 0x300u),
                    unsigned(qs[2]) | ((h << 4) & 0x300u), unsigned(qs[3]) | ((h << 2) & 0x300u));
    }
    static LIMBO_AI uint32_t signs(const uint8_t* b, int c) { return rd32(b + 34 + 4 * c); }
    static LIMBO_AI void scales(const uint8_t* b, int c, int& lo, int& hi) {
        lo = 2 * (b[74 + c] & 0xF) + 1;
        hi = 2 * (b[74 + c] >> 4) + 1;
    }
    static constexpr Var var = Var::S;
    static constexpr bool minv = false;
    LIMBO_AVX2 static LIMBO_AI void decode(const Tables&, const uint8_t* b, Block& o) {
        o.d = h2f(b) * 0.125f;   // v = d * (0.5 + nib) * 0.25 * grid = (d / 8) * (2 nib + 1) * grid
        for (int c = 0; c < 8; ++c) {
            int lo, hi;
            scales(b, c, lo, hi);
            put_chunk(o, c, signed_gv(gv(b, c), signs(b, c)));
            put_sv16(o, c, lo, hi);
        }
    }
};

struct DIQ3S {
    static constexpr uint32_t type = 21;
    static constexpr int elems = 256, bytes = 110;
    static LIMBO_AI float scale(const uint8_t* b) { return h2f(b); }
    // chunk c: grid index j = qs[8c + j] | bit j of qh[c] as bit 8, the eight 4-lane magnitudes by one gather
    LIMBO_AVX2 static LIMBO_AI __m256i gv(const uint8_t* b, int c) {
        const uint8_t* q = b + 2 + 8 * c;
        const unsigned h = b[66 + c];
        const unsigned i[8] = {q[0] | ((h << 8) & 256u), q[1] | ((h << 7) & 256u), q[2] | ((h << 6) & 256u), q[3] | ((h << 5) & 256u),
                               q[4] | ((h << 4) & 256u), q[5] | ((h << 3) & 256u), q[6] | ((h << 2) & 256u), q[7] | ((h << 1) & 256u)};
        return gv32(quants::kIq3sGrid, i);
    }
    static LIMBO_AI uint32_t signs(const uint8_t* b, int c) { return rd32(b + 74 + 4 * c); }
    static LIMBO_AI void scales(const uint8_t* b, int c, int& lo, int& hi) { lo = hi = 1 + 2 * ((b[106 + c / 2] >> (4 * (c & 1))) & 0xF); }
    static constexpr Var var = Var::S;
    static constexpr bool minv = false;
    LIMBO_AVX2 static LIMBO_AI void decode(const Tables&, const uint8_t* b, Block& o) {
        o.d = h2f(b);
        for (int c = 0; c < 8; ++c) {
            int lo, hi;
            scales(b, c, lo, hi);
            put_chunk(o, c, signed_gv(gv(b, c), signs(b, c)));
            put_sv32(o, c, lo);
        }
    }
};

struct DIQ3XXS {
    static constexpr uint32_t type = 18;
    static constexpr int elems = 256, bytes = 98;
    static LIMBO_AI float scale(const uint8_t* b) { return h2f(b) * 0.25f; }
    LIMBO_AVX2 static LIMBO_AI __m256i gv(const uint8_t* b, int c) {
        const uint8_t* q = b + 2 + 8 * c;
        const unsigned i[8] = {q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7]};
        return gv32(quants::kIq3xxsGrid, i);
    }
    static LIMBO_AI uint32_t signs(const uint8_t* b, int c) {
        const uint32_t w = rd32(b + 66 + 4 * c);
        return uint32_t(quants::kSignsIq2xs[w & 127]) | (uint32_t(quants::kSignsIq2xs[(w >> 7) & 127]) << 8) |
               (uint32_t(quants::kSignsIq2xs[(w >> 14) & 127]) << 16) | (uint32_t(quants::kSignsIq2xs[(w >> 21) & 127]) << 24);
    }
    static LIMBO_AI void scales(const uint8_t* b, int c, int& lo, int& hi) { lo = hi = 2 * int(rd32(b + 66 + 4 * c) >> 28) + 1; }
    static constexpr Var var = Var::S;
    static constexpr bool minv = false;
    LIMBO_AVX2 static LIMBO_AI void decode(const Tables&, const uint8_t* b, Block& o) {
        o.d = h2f(b) * 0.25f;   // v = d * (0.5 + s) * 0.5 * grid = (d / 4) * (2 s + 1) * grid
        for (int c = 0; c < 8; ++c) {
            int lo, hi;
            scales(b, c, lo, hi);
            put_chunk(o, c, signed_gv(gv(b, c), signs(b, c)));
            put_sv32(o, c, lo);
        }
    }
};

LIMBO_AVX2 LIMBO_AI __m256i nl_chunk(const uint8_t* qs, __m128i lut) {
    const __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs));
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const __m128i lo = _mm_shuffle_epi8(lut, _mm_and_si128(v, m4));
    const __m128i hi = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(v, 4), m4));
    return _mm256_set_m128i(hi, lo);
}

struct DIQ4XS {
    static constexpr uint32_t type = 23;
    static constexpr int elems = 256, bytes = 136;
    static constexpr Var var = Var::S;
    static constexpr bool minv = false;
    LIMBO_AVX2 static LIMBO_AI void decode(const Tables&, const uint8_t* b, Block& o) {
        o.d = h2f(b);
        const unsigned sh = rd16(b + 2);
        const uint8_t* sl = b + 4;
        const uint8_t* qs = b + 8;
        const __m128i lut = _mm_loadu_si128(reinterpret_cast<const __m128i*>(quants::kIq4NlValues));
        for (int c = 0; c < 8; ++c) {
            const int ls = int((sl[c / 2] >> (4 * (c % 2))) & 0xF) | (int((sh >> (2 * c)) & 3) << 4);
            put_chunk(o, c, nl_chunk(qs + 16 * c, lut));
            put_sv32(o, c, ls - 32);
        }
    }
};

struct DIQ4NL {
    static constexpr uint32_t type = 20;
    static constexpr int elems = 32, bytes = 18;
    static constexpr Var var = Var::F;
    static constexpr bool minv = false;
    LIMBO_AVX2 static LIMBO_AI void decode(const Tables&, const uint8_t* b, Block& o) {
        const __m128i lut = _mm_loadu_si128(reinterpret_cast<const __m128i*>(quants::kIq4NlValues));
        o.fd[0] = h2f(b);
        put_chunk(o, 0, nl_chunk(b + 2, lut));
    }
};

// 32 lanes of Q2_0 codes: byte idx[k] of q, its two bits for lane k % 4, minus 1
LIMBO_AVX2 LIMBO_AI __m256i q20_chunk(__m256i q, __m256i idx) {
    const __m256i lo_bit = _mm256_set1_epi32(0x40100401), hi_bit = _mm256_set1_epi32(int(0x80200802u));
    const __m256i w = _mm256_shuffle_epi8(q, idx);
    const __m256i lo = _mm256_cmpeq_epi8(_mm256_and_si256(w, lo_bit), lo_bit);
    const __m256i hi = _mm256_cmpeq_epi8(_mm256_and_si256(w, hi_bit), hi_bit);
    return _mm256_sub_epi8(_mm256_sub_epi8(_mm256_sub_epi8(_mm256_set1_epi8(-1), lo), hi), hi);
}

struct DQ20 {
    static constexpr uint32_t type = 42;
    static constexpr int elems = 64, bytes = 18;
    static constexpr Var var = Var::F;
    static constexpr bool minv = false;
    LIMBO_AVX2 static LIMBO_AI void decode(const Tables&, const uint8_t* b, Block& o) {
        const float d = h2f(b);   // may be negative: the sign stays in the scale
        o.fd[0] = o.fd[1] = d;
        // element 4i + j is bits 2j..2j+1 of byte i, minus 1: each byte spread over 4 lanes, its two bits tested
        const __m256i q = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b + 2)));
        put_chunk(o, 0, q20_chunk(q, _mm256_setr_epi8(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6,
                                               6, 7, 7, 7, 7)));
        put_chunk(o, 1, q20_chunk(q, _mm256_setr_epi8(8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13,
                                               13, 13, 14, 14, 14, 14, 15, 15, 15, 15)));
    }
};

// ---- the AVX2 dots --------------------------------------------------------------------------------------
LIMBO_AVX2 LIMBO_AI float hsum256(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

// Columns cols[0..NC) of one decoded block, accumulating into acc[0..NC).
template <int NC, bool MIN>
LIMBO_AVX2 LIMBO_AI void dot_u(const Block& B, const ActQ& a, const int* cols, int bi, int elems, __m256* acc) {
    const int K = a.K;
    const int8_t* q[NC];
    float sc[NC], scm[NC];
    const int16_t* bsp[NC];
    for (int c = 0; c < NC; ++c) {
        const size_t col = size_t(cols[c]);
        q[c] = a.q + col * size_t(K) + size_t(bi) * size_t(elems);
        const float da = a.d[col * size_t(K / a.blk) + size_t(bi)];
        sc[c] = B.d * da;
        scm[c] = B.dmin * da;
        bsp[c] = a.bs + col * size_t(K / 16) + size_t(bi) * 16;
    }
    __m256i x[NC];
    for (int c = 0; c < NC; ++c) x[c] = _mm256_setzero_si256();
    for (int k = 0; k < 8; ++k) {
        const __m256i w = _mm256_load_si256(reinterpret_cast<const __m256i*>(B.w + 32 * k));
        const __m256i s = _mm256_load_si256(reinterpret_cast<const __m256i*>(B.sv[k]));
        for (int c = 0; c < NC; ++c) {
            const __m256i av = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q[c] + 32 * k));
            x[c] = _mm256_add_epi32(x[c], _mm256_madd_epi16(_mm256_maddubs_epi16(w, av), s));
        }
    }
    __m256i mvv = _mm256_setzero_si256();
    if (MIN) mvv = _mm256_load_si256(reinterpret_cast<const __m256i*>(B.mv));
    for (int c = 0; c < NC; ++c) {
        acc[c] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(x[c]), _mm256_set1_ps(sc[c]), acc[c]);
        if (MIN) {
            const __m256i bsv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bsp[c]));
            const __m256i y = _mm256_madd_epi16(bsv, mvv);
            acc[c] = _mm256_fnmadd_ps(_mm256_cvtepi32_ps(y), _mm256_set1_ps(scm[c]), acc[c]);
        }
    }
}

template <int NC>
LIMBO_AVX2 LIMBO_AI void dot_s(const Block& B, const ActQ& a, const int* cols, int bi, int elems, __m256* acc) {
    const int K = a.K;
    const int8_t* q[NC];
    float sc[NC];
    for (int c = 0; c < NC; ++c) {
        const size_t col = size_t(cols[c]);
        q[c] = a.q + col * size_t(K) + size_t(bi) * size_t(elems);
        sc[c] = B.d * a.d[col * size_t(K / a.blk) + size_t(bi)];
    }
    __m256i x[NC];
    for (int c = 0; c < NC; ++c) x[c] = _mm256_setzero_si256();
    for (int k = 0; k < 8; ++k) {
        const __m256i w = _mm256_load_si256(reinterpret_cast<const __m256i*>(B.w + 32 * k));
        const __m256i wa = _mm256_abs_epi8(w);
        const __m256i s = _mm256_load_si256(reinterpret_cast<const __m256i*>(B.sv[k]));
        for (int c = 0; c < NC; ++c) {
            const __m256i av = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q[c] + 32 * k));
            x[c] = _mm256_add_epi32(x[c], _mm256_madd_epi16(_mm256_maddubs_epi16(wa, _mm256_sign_epi8(av, w)), s));
        }
    }
    for (int c = 0; c < NC; ++c) acc[c] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(x[c]), _mm256_set1_ps(sc[c]), acc[c]);
}

template <int NC>
LIMBO_AVX2 LIMBO_AI void dot_f(const Block& B, const ActQ& a, const int* cols, int bi, int elems, __m256* acc) {
    const int K = a.K;
    const int nch = elems / 32;
    const __m256i ones = _mm256_set1_epi16(1);
    for (int k = 0; k < nch; ++k) {
        const __m256i w = _mm256_load_si256(reinterpret_cast<const __m256i*>(B.w + 32 * k));
        const __m256i wa = _mm256_abs_epi8(w);
        for (int c = 0; c < NC; ++c) {
            const size_t col = size_t(cols[c]);
            const size_t chunk = size_t(bi) * size_t(nch) + size_t(k);
            const __m256i av = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a.q + col * size_t(K) + chunk * 32));
            const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(wa, _mm256_sign_epi8(av, w)), ones);
            const float s = B.fd[k] * a.d[col * size_t(K / 32) + chunk];
            acc[c] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(s), acc[c]);
        }
    }
}

template <class D, int NC>
LIMBO_AVX2 LIMBO_AI void dot_block(const Block& B, const ActQ& a, const int* cols, int bi, __m256* acc) {
    if constexpr (D::var == Var::U) dot_u<NC, D::minv>(B, a, cols, bi, D::elems, acc);
    else if constexpr (D::var == Var::S) dot_s<NC>(B, a, cols, bi, D::elems, acc);
    else dot_f<NC>(B, a, cols, bi, D::elems, acc);
}

template <class D>
LIMBO_AVX2 void gemv_avx2(const uint8_t* W, uint64_t row_bytes, int nrows, const ActQ& a, const int* cols, int ncols,
                          float* y, int ldy) {
    const Tables& T = tab();
    const int nb = a.K / D::elems;
    alignas(32) Block B;
    alignas(32) __m256 acc[kMaxCols];
    for (int r = 0; r < nrows; ++r) {
        const uint8_t* rp = W + uint64_t(r) * row_bytes;
        for (int c = 0; c < ncols; ++c) acc[c] = _mm256_setzero_ps();
        for (int bi = 0; bi < nb; ++bi) {
            D::decode(T, rp + size_t(bi) * D::bytes, B);
            int c = 0;
            for (; c + 4 <= ncols; c += 4) dot_block<D, 4>(B, a, cols + c, bi, acc + c);
            if (c + 2 <= ncols) {
                dot_block<D, 2>(B, a, cols + c, bi, acc + c);
                c += 2;
            }
            if (c < ncols) dot_block<D, 1>(B, a, cols + c, bi, acc + c);
        }
        for (int c = 0; c < ncols; ++c) y[size_t(c) * size_t(ldy) + size_t(r)] = hsum256(acc[c]);
    }
}

// The codebook formats (IQ2_S, IQ3_S, IQ3_XXS) for up to 4 columns, decode fused into the dot: the grid magnitudes
// are unsigned, so the signs go onto the activation (psign) and maddubs takes the magnitudes as they are -- no signed
// weights built, stored and taken apart again (abs + sign) per block.
template <class D, int NC>
LIMBO_AVX2 LIMBO_AI void cb_row(const uint8_t* rp, const ActQ& a, const int* cols, __m256* acc) {
    const int K = a.K, nb = K / 256;
    const __m256i rep = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
                                         3, 3, 3);
    const __m256i bits = _mm256_set1_epi64x(int64_t(0x8040201008040201ull));
    const __m256i one = _mm256_set1_epi8(1);
    const int8_t* q[NC];
    const float* ds[NC];
    for (int c = 0; c < NC; ++c) {
        q[c] = a.q + size_t(cols[c]) * size_t(K);
        ds[c] = a.d + size_t(cols[c]) * size_t(K / a.blk);
    }
    for (int bi = 0; bi < nb; ++bi) {
        const uint8_t* b = rp + size_t(bi) * D::bytes;
        __m256i x[NC];
        for (int c = 0; c < NC; ++c) x[c] = _mm256_setzero_si256();
#pragma GCC unroll 8
        for (int k = 0; k < 8; ++k) {
            int lo, hi;
            D::scales(b, k, lo, hi);
            const __m256i gv = D::gv(b, k);
            const __m256i neg =
                _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32(int(D::signs(b, k))), rep), bits), bits);
            const __m256i sg = _mm256_or_si256(neg, one);   // -1 where the bit is set, +1 elsewhere
            const __m256i sv = _mm256_set_m128i(_mm_set1_epi16(int16_t(hi)), _mm_set1_epi16(int16_t(lo)));
            for (int c = 0; c < NC; ++c) {
                const __m256i av = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q[c] + bi * 256 + 32 * k));
                x[c] = _mm256_add_epi32(x[c], _mm256_madd_epi16(_mm256_maddubs_epi16(gv, _mm256_sign_epi8(av, sg)), sv));
            }
        }
        const float d = D::scale(b);
        for (int c = 0; c < NC; ++c)
            acc[c] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(x[c]), _mm256_set1_ps(d * ds[c][bi]), acc[c]);
    }
}

template <class D>
LIMBO_AVX2 void gemv_cb(const uint8_t* W, uint64_t row_bytes, int nrows, const ActQ& a, const int* cols, int ncols, float* y,
                        int ldy);

// ---- the scalar ISA: the plain definition ---------------------------------------------------------------
void gemv_scalar(uint32_t type, int blk_elems, const uint8_t* W, uint64_t row_bytes, int nrows, const ActQ& a,
                 const int* cols, int ncols, float* y, int ldy) {
    (void)blk_elems;
    std::vector<float> wf(size_t(a.K));
    for (int r = 0; r < nrows; ++r) {
        quants::dequant_row(type, W + uint64_t(r) * row_bytes, wf.data(), uint64_t(a.K));
        for (int c = 0; c < ncols; ++c) {
            const size_t col = size_t(cols[c]);
            const int8_t* q = a.q + col * size_t(a.K);
            const float* d = a.d + col * size_t(a.K / a.blk);
            float acc = 0.0f;
            for (int b = 0; b < a.K / a.blk; ++b) {
                float s = 0.0f;
                for (int i = 0; i < a.blk; ++i) s += wf[size_t(b * a.blk + i)] * float(q[b * a.blk + i]);
                acc += s * d[b];
            }
            y[size_t(c) * size_t(ldy) + size_t(r)] = acc;
        }
    }
}

template <class D>
LIMBO_AVX2 void gemv_cb(const uint8_t* W, uint64_t row_bytes, int nrows, const ActQ& a, const int* cols, int ncols, float* y,
                        int ldy) {
    static const bool generic = std::getenv("LIMBO_LANE_GENERIC") != nullptr;
    if (ncols > 4 || a.blk != 256 || generic) {
        gemv_avx2<D>(W, row_bytes, nrows, a, cols, ncols, y, ldy);
        return;
    }
    for (int r = 0; r < nrows; ++r) {
        const uint8_t* rp = W + uint64_t(r) * row_bytes;
        __m256 acc[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
        switch (ncols) {
            case 1: cb_row<D, 1>(rp, a, cols, acc); break;
            case 2: cb_row<D, 2>(rp, a, cols, acc); break;
            case 3: cb_row<D, 3>(rp, a, cols, acc); break;
            default: cb_row<D, 4>(rp, a, cols, acc); break;
        }
        for (int c = 0; c < ncols; ++c) y[size_t(c) * size_t(ldy) + size_t(r)] = hsum256(acc[c]);
    }
}

using GemvFn = void (*)(const uint8_t*, uint64_t, int, const ActQ&, const int*, int, float*, int);

GemvFn avx2_fn(uint32_t type) {
    switch (static_cast<Type>(type)) {
        case Type::Q2_K: return gemv_avx2<DQ2K>;
        case Type::Q3_K: return gemv_avx2<DQ3K>;
        case Type::Q4_K: return gemv_avx2<DQ4K>;
        case Type::Q5_K: return gemv_avx2<DQ5K>;
        case Type::Q6_K: return gemv_avx2<DQ6K>;
        case Type::IQ2_S: return gemv_cb<DIQ2S>;
        case Type::IQ3_S: return gemv_cb<DIQ3S>;
        case Type::IQ3_XXS: return gemv_cb<DIQ3XXS>;
        case Type::IQ4_XS: return gemv_avx2<DIQ4XS>;
        case Type::IQ4_NL: return gemv_avx2<DIQ4NL>;
        case Type::Q2_0: return gemv_avx2<DQ20>;
        default: return nullptr;
    }
}

}  // namespace

bool lane_supports(uint32_t type) { return avx2_fn(type) != nullptr; }

int weight_block_elems(uint32_t type) {
    switch (static_cast<Type>(type)) {
        case Type::IQ4_NL: return 32;
        case Type::Q2_0: return 64;
        default: return lane_supports(type) ? 256 : 0;
    }
}

int act_block(uint32_t type) {
    const int e = weight_block_elems(type);
    return e == 0 ? 0 : (e == 256 ? 256 : 32);
}

bool gemv_cols(Isa isa, uint32_t type, const uint8_t* W, uint64_t row_bytes, int nrows, const ActQ& a,
               const int* cols, int ncols, float* y, int ldy) {
    if (!lane_supports(type) || ncols < 1 || ncols > kMaxCols) return false;
    const int be = weight_block_elems(type);
    if (a.blk != act_block(type) || a.K % be != 0) return false;
    if (isa == Isa::Scalar) {
        gemv_scalar(type, be, W, row_bytes, nrows, a, cols, ncols, y, ldy);
        return true;
    }
    avx2_fn(type)(W, row_bytes, nrows, a, cols, ncols, y, ldy);   // Avx512: reserved, runs the AVX2 kernels
    return true;
}

// ---- activations ----------------------------------------------------------------------------------------
size_t actq_bytes(int K, int ncols) {
    auto up = [](size_t n) { return (n + 63) & ~size_t(63); };
    return up(size_t(ncols) * size_t(K)) + up(size_t(ncols) * size_t(K / 32) * 4) + up(size_t(ncols) * size_t(K / 16) * 2) + 64;
}

void actq_bind(ActQ* a, void* mem, int K, int blk, int ncols) {
    auto up = [](size_t n) { return (n + 63) & ~size_t(63); };
    uint8_t* p = static_cast<uint8_t*>(mem);
    a->K = K;
    a->blk = blk;
    a->ncols = ncols;
    a->q = reinterpret_cast<int8_t*>(p);
    p += up(size_t(ncols) * size_t(K));
    a->d = reinterpret_cast<float*>(p);
    p += up(size_t(ncols) * size_t(K / 32) * 4);
    a->bs = reinterpret_cast<int16_t*>(p);
}

namespace {

// One plane of one row: quantizes `in` (K floats) block by block, leaves the residual in `res` (may alias in).
void quant_plane_scalar(const float* in, float* res, int K, int blk, int8_t* q, float* d, int16_t* bs) {
    for (int b = 0; b < K / blk; ++b) {
        float amax = 0.0f;
        for (int i = 0; i < blk; ++i) amax = std::max(amax, std::fabs(in[b * blk + i]));
        const float dd = amax / 127.0f;
        const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
        d[b] = dd;
        for (int i = 0; i < blk; ++i) {
            const float x = in[b * blk + i];
            const long v = std::lrintf(x * inv);
            q[b * blk + i] = int8_t(v);
            res[b * blk + i] = x - dd * float(v);
        }
    }
    for (int j = 0; j < K / 16; ++j) {
        int s = 0;
        for (int i = 0; i < 16; ++i) s += q[16 * j + i];
        bs[j] = int16_t(s);
    }
}

LIMBO_AVX2 void quant_plane_avx2(const float* in, float* res, int K, int blk, int8_t* q, float* d, int16_t* bs) {
    const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    const __m256i bias = _mm256_set1_epi8(char(0x80));
    for (int b = 0; b < K / blk; ++b) {
        const float* x = in + b * blk;
        __m256 m = _mm256_setzero_ps();
        for (int i = 0; i < blk; i += 8) m = _mm256_max_ps(m, _mm256_and_ps(_mm256_loadu_ps(x + i), absmask));
        __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
        m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
        m4 = _mm_max_ss(m4, _mm_shuffle_ps(m4, m4, 1));
        const float amax = _mm_cvtss_f32(m4);
        const float dd = amax / 127.0f;
        const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
        d[b] = dd;
        const __m256 vinv = _mm256_set1_ps(inv), vd = _mm256_set1_ps(dd);
        for (int i = 0; i < blk; i += 32) {
            __m256i v[4];
            for (int t = 0; t < 4; ++t) {
                const __m256 xv = _mm256_loadu_ps(x + i + 8 * t);
                v[t] = _mm256_cvtps_epi32(_mm256_mul_ps(xv, vinv));
                _mm256_storeu_ps(res + b * blk + i + 8 * t, _mm256_fnmadd_ps(_mm256_cvtepi32_ps(v[t]), vd, xv));
            }
            const __m256i p01 = _mm256_packs_epi32(v[0], v[1]), p23 = _mm256_packs_epi32(v[2], v[3]);
            const __m256i p = _mm256_permutevar8x32_epi32(_mm256_packs_epi16(p01, p23), perm);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(q + b * blk + i), p);
            // sums of 16: bias to unsigned, sad against zero gives four sums of 8
            alignas(32) uint64_t s[4];
            _mm256_store_si256(reinterpret_cast<__m256i*>(s), _mm256_sad_epu8(_mm256_xor_si256(p, bias), _mm256_setzero_si256()));
            bs[(b * blk + i) / 16] = int16_t(int(s[0] + s[1]) - 2048);
            bs[(b * blk + i) / 16 + 1] = int16_t(int(s[2] + s[3]) - 2048);
        }
    }
}

void quant_row(Isa isa, const float* x, int row, int planes, ActQ* a) {
    const int K = a->K;
    alignas(32) float res[8192];
    const float* in = x;
    for (int p = 0; p < planes; ++p) {
        const size_t col = size_t(row) * size_t(planes) + size_t(p);
        int8_t* q = a->q + col * size_t(K);
        float* d = a->d + col * size_t(K / a->blk);
        int16_t* bs = a->bs + col * size_t(K / 16);
        if (isa == Isa::Scalar) quant_plane_scalar(in, res, K, a->blk, q, d, bs);
        else quant_plane_avx2(in, res, K, a->blk, q, d, bs);
        in = res;
    }
}

}  // namespace

void actq_quantize_row(Isa isa, const float* x, int row, int planes, ActQ* a) {
    if (a->K > 8192) std::abort();   // the residual scratch is on the stack
    quant_row(isa, x, row, planes, a);
}

void actq_quantize(Isa isa, const float* x, int nrows, int planes, ActQ* a) {
    for (int r = 0; r < nrows; ++r) actq_quantize_row(isa, x + size_t(r) * size_t(a->K), r, planes, a);
}

// ---- the reference --------------------------------------------------------------------------------------
void expert_ffn_reference(const ExpertRef& w, const float* x, float* out) {
    const int e = w.n_embd, nf = w.n_ff;
    std::vector<float> row(static_cast<size_t>(std::max(e, nf)));
    std::vector<double> h(static_cast<size_t>(nf), 0.0);
    const uint64_t grow = type_bytes(w.gate_type, uint64_t(e)), urow = type_bytes(w.up_type, uint64_t(e));
    const uint64_t drow = type_bytes(w.down_type, uint64_t(nf));
    for (int j = 0; j < nf; ++j) {
        quants::dequant_row(w.gate_type, w.gate + uint64_t(j) * grow, row.data(), uint64_t(e));
        double g = 0.0;
        for (int i = 0; i < e; ++i) g += double(row[size_t(i)]) * double(x[i]);
        quants::dequant_row(w.up_type, w.up + uint64_t(j) * urow, row.data(), uint64_t(e));
        double u = 0.0;
        for (int i = 0; i < e; ++i) u += double(row[size_t(i)]) * double(x[i]);
        h[size_t(j)] = g / (1.0 + std::exp(-g)) * u;
    }
    for (int j = 0; j < e; ++j) {
        quants::dequant_row(w.down_type, w.down + uint64_t(j) * drow, row.data(), uint64_t(nf));
        double s = 0.0;
        for (int i = 0; i < nf; ++i) s += double(row[size_t(i)]) * h[size_t(i)];
        out[j] = float(s);
    }
}

}  // namespace limbo::cpu
