// Copyright (c) 2026 ITERFORM. All rights reserved.
// tests/quants_test.cpp — the weight decoder's own invariants, without the oracle.
//
// The oracle comparison lives in tools/quant_parity (it links ggml). What belongs here is what can be
// checked from the format definitions alone: the block geometry in the type table must match what each
// decoder walks, the half conversion must be right, and the two formats whose code-to-value mapping is
// not the obvious one (Q2_0's -1/0/+1/+2, IQ4_NL's table) must produce exactly the documented values.
#include "limbo/check.hpp"
#include "limbo/quants.hpp"
#include "limbo/types.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

void put16(uint8_t* p, uint16_t v) {
    p[0] = uint8_t(v & 0xff);
    p[1] = uint8_t(v >> 8);
}

// f16 bit patterns for small values, computed here rather than assumed.
uint16_t f16_from_float(float x) {
    uint32_t bits;
    std::memcpy(&bits, &x, 4);
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = int32_t((bits >> 23) & 0xff) - 127 + 15;
    uint32_t man = (bits >> 13) & 0x3ffu;
    if (exp <= 0) return uint16_t(sign);
    if (exp >= 31) return uint16_t(sign | 0x7c00u);
    return uint16_t(sign | (uint32_t(exp) << 10) | man);
}

}  // namespace

int main() {
    // ---- the half conversion -----------------------------------------------------------------
    CHECK(limbo::quants::f16_to_f32(0x0000) == 0.0f, "f16 +0");
    CHECK(limbo::quants::f16_to_f32(0x8000) == 0.0f && std::signbit(limbo::quants::f16_to_f32(0x8000)), "f16 -0");
    CHECK(limbo::quants::f16_to_f32(0x3C00) == 1.0f, "f16 1.0");
    CHECK(limbo::quants::f16_to_f32(0xC000) == -2.0f, "f16 -2.0");
    CHECK(limbo::quants::f16_to_f32(0x3800) == 0.5f, "f16 0.5");
    CHECK(limbo::quants::f16_to_f32(0x7C00) > 1e38f, "f16 inf");
    for (float x : {1.0f, -2.0f, 0.5f, 0.25f, 1024.0f, -0.125f})
        CHECK(limbo::quants::f16_to_f32(f16_from_float(x)) == x, "f16 round trip");

    // ---- the block geometry each decoder walks must be the geometry the table declares ---------
    // (format, elements per block) as the decoders are written.
    struct Row {
        uint32_t type;
        uint32_t elems;
        const char* name;
    };
    const Row kRows[] = {
        {0, 1, "F32"},      {1, 1, "F16"},      {30, 1, "BF16"},   {8, 32, "Q8_0"},   {10, 256, "Q2_K"},
        {11, 256, "Q3_K"},  {12, 256, "Q4_K"},  {13, 256, "Q5_K"}, {14, 256, "Q6_K"}, {22, 256, "IQ2_S"},
        {21, 256, "IQ3_S"}, {23, 256, "IQ4_XS"}, {20, 32, "IQ4_NL"}, {42, 64, "Q2_0"},
        {16, 256, "IQ2_XXS"}, {17, 256, "IQ2_XS"}, {18, 256, "IQ3_XXS"}, {19, 256, "IQ1_S"}, {29, 256, "IQ1_M"},
        {40, 64, "NVFP4"},
    };
    for (const Row& r : kRows) {
        const limbo::TypeInfo* ti = limbo::type_info(r.type);
        CHECK(ti != nullptr, r.name);
        if (!ti) continue;
        CHECK(ti->block_elems == r.elems, "the decoder's block size is the table's block size");
        CHECK(limbo::quants::supported(r.type), "the format is implemented");
    }
    CHECK(!limbo::quants::supported(4), "an unimplemented type is refused rather than decoded");

    // ---- Q2_0: codes 0,1,2,3 mean -1, 0, +1, +2 ----------------------------------------------
    {
        uint8_t blk[18] = {0};
        put16(blk, f16_from_float(2.0f));
        // 64 elements, 4 codes per byte: 0,1,2,3 repeated
        for (int i = 0; i < 16; ++i) blk[2 + i] = uint8_t(0 | (1 << 2) | (2 << 4) | (3 << 6));
        std::vector<float> out(64);
        CHECK(limbo::quants::dequant_row(42, blk, out.data(), 64), "Q2_0 decodes");
        bool ok = true;
        for (int j = 0; j < 64; ++j) {
            const float want = float((j % 4) - 1) * 2.0f;
            if (out[size_t(j)] != want) ok = false;
        }
        CHECK(ok, "Q2_0 maps code 0,1,2,3 to -1,0,+1,+2 times d");
    }

    // ---- Q8_0: d times each int8 --------------------------------------------------------------
    {
        uint8_t blk[34] = {0};
        put16(blk, f16_from_float(0.5f));
        for (int j = 0; j < 32; ++j) reinterpret_cast<int8_t*>(blk + 2)[j] = int8_t(j - 16);
        std::vector<float> out(32);
        CHECK(limbo::quants::dequant_row(8, blk, out.data(), 32), "Q8_0 decodes");
        bool ok = true;
        for (int j = 0; j < 32; ++j)
            if (out[size_t(j)] != 0.5f * float(j - 16)) ok = false;
        CHECK(ok, "Q8_0 is d times the quants");
    }

    // ---- IQ4_NL: the documented 16-value table ----------------------------------------------
    {
        uint8_t blk[18] = {0};
        put16(blk, f16_from_float(1.0f));
        for (int j = 0; j < 16; ++j) blk[2 + j] = uint8_t(j | (j << 4));
        std::vector<float> out(32);
        CHECK(limbo::quants::dequant_row(20, blk, out.data(), 32), "IQ4_NL decodes");
        bool ok = true;
        for (int j = 0; j < 16; ++j) {
            if (out[size_t(j)] != float(limbo::quants::kIq4NlValues[j])) ok = false;
            if (out[size_t(j + 16)] != float(limbo::quants::kIq4NlValues[j])) ok = false;
        }
        CHECK(ok, "IQ4_NL reads its nibbles through the value table");
        CHECK(limbo::quants::kIq4NlValues[0] == -127 && limbo::quants::kIq4NlValues[15] == 113,
              "the IQ4_NL table is the one ggml ships");
    }

    // ---- NVFP4: a halved UE4M3 scale times the doubled E2M1 codebook ---------------------------
    // Block: d[4] (one UE4M3 per 16 elements) then qs[32] (two nibbles per byte, low nibbles first).
    {
        uint8_t blk[36] = {0};
        blk[0] = 0x38;  // exp 7, man 0 -> 1.0, halved -> 0.5
        blk[1] = 0x7f;  // ggml decodes this as 0
        for (int j = 0; j < 8; ++j) blk[4 + j] = 0x21;   // sub-block 0: low 1 (1.0), high 2 (2.0)
        for (int j = 8; j < 16; ++j) blk[4 + j] = 0xff;  // sub-block 1: -12.0 twice, but d is 0
        std::vector<float> out(64);
        CHECK(limbo::quants::dequant_row(40, blk, out.data(), 64), "NVFP4 decodes");
        bool ok = true;
        for (int j = 0; j < 8; ++j) {
            if (out[size_t(j)] != 0.5f * 1.0f) ok = false;      // d[0]=0.5, fp4[1]=1
            if (out[size_t(j + 8)] != 0.5f * 2.0f) ok = false;  // fp4[2]=2
        }
        for (int j = 16; j < 64; ++j)
            if (out[size_t(j)] != 0.0f) ok = false;  // d[1]=0x7f and d[2..3]=0
        CHECK(ok, "NVFP4 is ue4m3(d) times fp4[nibble], halved scale and all");
        CHECK(limbo::quants::kFp4Values[0] == 0 && limbo::quants::kFp4Values[7] == 12 &&
                  limbo::quants::kFp4Values[15] == -12,
              "the FP4 codebook is the doubled one ggml ships");
    }

    // ---- the grid formats: a block of zeros reads grid entry 0 with the smallest scale ---------------
    // (the real comparison against ggml's own decode is tools/quant_parity; this pins the block walk).
    {
        using limbo::quants::kIq1sGrid;
        using limbo::quants::kIq2xsGrid;
        using limbo::quants::kIq2xxsGrid;
        using limbo::quants::kIq3xxsGrid;
        std::vector<float> out(256);
        auto all_equal = [&](auto want) {
            for (int j = 0; j < 256; ++j)
                if (out[size_t(j)] != want(j)) return false;
            return true;
        };
        {
            uint8_t blk[66] = {0};
            put16(blk, f16_from_float(1.0f));
            CHECK(limbo::quants::dequant_row(16, blk, out.data(), 256), "IQ2_XXS decodes");
            CHECK(all_equal([&](int j) { return 0.125f * float(uint8_t(kIq2xxsGrid[0] >> (8 * (j % 8)))); }),
                  "IQ2_XXS: index 0, scale nibble 0 is d * 0.5 * 0.25 times grid 0");
        }
        {
            uint8_t blk[74] = {0};
            put16(blk, f16_from_float(1.0f));
            CHECK(limbo::quants::dequant_row(17, blk, out.data(), 256), "IQ2_XS decodes");
            CHECK(all_equal([&](int j) { return 0.125f * float(uint8_t(kIq2xsGrid[0] >> (8 * (j % 8)))); }),
                  "IQ2_XS: index 0, scale nibble 0 is d * 0.5 * 0.25 times grid 0");
        }
        {
            uint8_t blk[98] = {0};
            put16(blk, f16_from_float(1.0f));
            CHECK(limbo::quants::dequant_row(18, blk, out.data(), 256), "IQ3_XXS decodes");
            CHECK(all_equal([&](int j) { return 0.25f * float(uint8_t(kIq3xxsGrid[0] >> (8 * (j % 4)))); }),
                  "IQ3_XXS: index 0, scale nibble 0 is d * 0.5 * 0.5 times grid 0");
        }
        {
            uint8_t blk[50] = {0};
            put16(blk, f16_from_float(1.0f));
            CHECK(limbo::quants::dequant_row(19, blk, out.data(), 256), "IQ1_S decodes");
            CHECK(all_equal([&](int j) { return float(int8_t(uint8_t(kIq1sGrid[0] >> (8 * (j % 8))))) + 0.125f; }),
                  "IQ1_S: index 0, scale 1, positive delta is grid 0 + 1/8");
            blk[34 + 1] = 0x80;   // qh[0] bit 15: the delta flips to -1/8
            CHECK(limbo::quants::dequant_row(19, blk, out.data(), 256), "IQ1_S decodes with a negative delta");
            bool ok = true;
            for (int j = 0; j < 32; ++j)
                if (out[size_t(j)] != float(int8_t(uint8_t(kIq1sGrid[0] >> (8 * (j % 8))))) - 0.125f) ok = false;
            CHECK(ok, "IQ1_S: bit 15 of qh selects the negative delta for its 32 lanes");
        }
        {
            uint8_t blk[56] = {0};
            // the f16 super-scale 1.0 = 0x3C00, one nibble per scale word's top nibble, low nibble first
            put16(blk + 48 + 4, 0xC000);
            put16(blk + 48 + 6, 0x3000);
            CHECK(limbo::quants::dequant_row(29, blk, out.data(), 256), "IQ1_M decodes");
            CHECK(all_equal([&](int j) { return float(int8_t(uint8_t(kIq1sGrid[0] >> (8 * (j % 8))))) + 0.125f; }),
                  "IQ1_M: the super-scale is spread over the scale words' top nibbles");
        }
    }

    // ---- the Q8_0 quantiser and the f32 -> f16 it rounds its scale with ------------------------------
    {
        const struct { float v; uint16_t h; } known[] = {{0.0f, 0x0000}, {1.0f, 0x3c00}, {-2.5f, 0xc100}, {65504.0f, 0x7bff}, {6.103515625e-5f, 0x0400},
                                                         {1e-7f, 0x0002}, {1.0f + 1.0f / 2048, 0x3c00}, {1.0f + 3.0f / 2048, 0x3c02}, {0.333251953125f, 0x3555}};
        bool h_ok = true;
        for (const auto& k : known)
            if (limbo::quants::f32_to_f16(k.v) != k.h) h_ok = false;
        CHECK(h_ok, "f32_to_f16 rounds to nearest even (normals, subnormals, ties)");
        CHECK(limbo::quants::f32_to_f16(1e6f) == 0x7c00 && limbo::quants::f32_to_f16(-1e6f) == 0xfc00, "f32_to_f16 overflows to infinity");
        std::vector<float> src(96);
        for (size_t j = 0; j < src.size(); ++j) src[j] = std::sin(float(j) * 0.7f) * (j < 32 ? 1.0f : j < 64 ? 40.0f : 0.001f);
        std::vector<uint8_t> q(3 * 34);
        std::vector<float> back(96);
        limbo::quants::quantize_q8_0(src.data(), q.data(), 96);
        CHECK(limbo::quants::dequant_row(8, q.data(), back.data(), 96), "the quantised blocks decode");
        double worst = 0;
        for (size_t j = 0; j < 96; ++j) {
            const float amax = j < 32 ? 1.0f : j < 64 ? 40.0f : 0.001f;
            worst = std::max(worst, std::fabs(double(back[j]) - double(src[j])) / double(amax));
        }
        CHECK(worst < 0.5 / 127 + 2e-3, "Q8_0 round trip is within half a step of the block's amax");
        float zeros[32] = {};
        uint8_t zq[34];
        limbo::quants::quantize_q8_0(zeros, zq, 32);
        float zb[32];
        limbo::quants::dequant_row(8, zq, zb, 32);
        bool z_ok = true;
        for (float v : zb) z_ok = z_ok && v == 0.0f;
        CHECK(z_ok, "an all-zero block stays zero");
    }
    return limbo::test::finish("quants");
}
