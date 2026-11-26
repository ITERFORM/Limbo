// Copyright (c) 2026 ITERFORM. All rights reserved.
// tests/types_test.cpp — format facts: the type table must carry ggml's block geometry.
//
// The numbers here are the ggml type-size table (interoperability, not expression — CLEANROOM.md §3).
// A wrong block size does not crash anything: it shows up as tensor spans that do not fit the file,
// which is what tests/gguf_test.cpp checks against the real ladder files.
#include "limbo/check.hpp"
#include "limbo/types.hpp"

#include <cstdio>
#include <string_view>

using limbo::type_bytes;
using limbo::type_info;

namespace {

struct Row {
    uint32_t id;
    uint32_t elems;
    uint32_t bytes;
    const char* name;
};

// GGUF type id -> (elements per block, bytes per block).
const Row kTable[] = {
    {0, 1, 4, "F32"},         {1, 1, 2, "F16"},         {2, 32, 18, "Q4_0"},     {3, 32, 20, "Q4_1"},
    {6, 32, 22, "Q5_0"},      {7, 32, 24, "Q5_1"},      {8, 32, 34, "Q8_0"},     {9, 32, 36, "Q8_1"},
    {10, 256, 84, "Q2_K"},    {11, 256, 110, "Q3_K"},   {12, 256, 144, "Q4_K"},  {13, 256, 176, "Q5_K"},
    {14, 256, 210, "Q6_K"},   {15, 256, 292, "Q8_K"},   {16, 256, 66, "IQ2_XXS"}, {17, 256, 74, "IQ2_XS"},
    {18, 256, 98, "IQ3_XXS"}, {19, 256, 50, "IQ1_S"},   {20, 32, 18, "IQ4_NL"},  {21, 256, 110, "IQ3_S"},
    {22, 256, 82, "IQ2_S"},   {23, 256, 136, "IQ4_XS"}, {24, 1, 1, "I8"},        {25, 1, 2, "I16"},
    {26, 1, 4, "I32"},        {27, 1, 8, "I64"},        {28, 1, 8, "F64"},       {29, 256, 56, "IQ1_M"},
    {30, 1, 2, "BF16"},       {34, 256, 54, "TQ1_0"},   {35, 256, 66, "TQ2_0"},  {39, 32, 17, "MXFP4"},
    {40, 64, 36, "NVFP4"},    {41, 128, 18, "Q1_0"},    {42, 64, 18, "Q2_0"},
};

}  // namespace

int main() {
    for (const Row& r : kTable) {
        const limbo::TypeInfo* t = type_info(r.id);
        CHECK(t != nullptr, r.name);
        if (!t) continue;
        if (t->block_elems != r.elems || t->block_bytes != r.bytes) {
            std::printf("FAIL %s: ggml says %u elems / %u bytes, the table says %u / %u\n", r.name, r.elems,
                        r.bytes, t->block_elems, t->block_bytes);
            ++limbo::test::failures();
        } else {
            std::printf("ok   %-8s %3u elems / %3u bytes\n", r.name, r.elems, r.bytes);
        }
        if (t->name != std::string_view(r.name)) {
            std::printf("FAIL %s: the name in the table is \"%.*s\"\n", r.name, (int)t->name.size(), t->name.data());
            ++limbo::test::failures();
        }
    }

    // Ids with no entry (gaps in the ggml enum, and ids from nowhere) must return nullptr.
    for (uint32_t id : {4u, 5u, 31u, 32u, 33u, 36u, 37u, 38u, 43u, 999u, 0xffffffffu})
        CHECK(type_info(id) == nullptr, "id with no entry is unknown");

    // Byte spans: whole blocks only, never a partial block.
    CHECK(type_bytes(0, 8) == 32, "F32 x8 = 32 bytes");
    CHECK(type_bytes(2, 32) == 18, "Q4_0 x32 = one block");
    CHECK(type_bytes(2, 64) == 36, "Q4_0 x64 = two blocks");
    CHECK(type_bytes(2, 31) == 0, "Q4_0 x31 is not a whole number of blocks");
    CHECK(type_bytes(12, 256) == 144, "Q4_K x256 = one block");
    CHECK(type_bytes(12, 384) == 0, "Q4_K x384 is not a whole number of blocks");
    CHECK(type_bytes(8, 0) == 0, "zero elements is not a span");
    CHECK(type_bytes(999, 32) == 0, "an unknown type has no span");

    return limbo::test::finish("types");
}
