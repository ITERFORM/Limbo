// Copyright (c) 2026 ITERFORM. All rights reserved.
#pragma once
// limbo/types.hpp — GGUF tensor types: ids, names and block geometry (format facts).

#include <cstdint>
#include <string_view>

namespace limbo {

// GGUF/ggml type ids as they appear in files. Ids are interoperability facts.
enum class Type : uint32_t {
    F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q5_1 = 7, Q8_0 = 8, Q8_1 = 9,
    Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14, Q8_K = 15,
    IQ2_XXS = 16, IQ2_XS = 17, IQ3_XXS = 18, IQ1_S = 19, IQ4_NL = 20, IQ3_S = 21, IQ2_S = 22,
    IQ4_XS = 23, I8 = 24, I16 = 25, I32 = 26, I64 = 27, F64 = 28, IQ1_M = 29, BF16 = 30,
    TQ1_0 = 34, TQ2_0 = 35, MXFP4 = 39, NVFP4 = 40, Q1_0 = 41, Q2_0 = 42,
};

struct TypeInfo {
    Type type;
    std::string_view name;
    uint32_t block_elems;   // elements per block (1 for plain types)
    uint32_t block_bytes;   // bytes per block
};

// nullptr for an id Limbo does not know. Ids 42/43 meant other formats in old modchip
// files, so a caller must also check that a tensor's byte span matches (docs/donors/modchip.md §10.2).
const TypeInfo* type_info(uint32_t id);

// Bytes for n elements of a type; 0 if n is not a whole number of blocks or the type is unknown.
uint64_t type_bytes(uint32_t id, uint64_t n);

}  // namespace limbo
