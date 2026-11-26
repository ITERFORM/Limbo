// Copyright (c) 2026 ITERFORM. All rights reserved.
// src/types.cpp — the type table.
#include "limbo/types.hpp"

namespace limbo {

namespace {
constexpr TypeInfo kTypes[] = {
    {Type::F32, "F32", 1, 4},          {Type::F16, "F16", 1, 2},
    {Type::Q4_0, "Q4_0", 32, 18},      {Type::Q4_1, "Q4_1", 32, 20},
    {Type::Q5_0, "Q5_0", 32, 22},      {Type::Q5_1, "Q5_1", 32, 24},
    {Type::Q8_0, "Q8_0", 32, 34},      {Type::Q8_1, "Q8_1", 32, 36},
    {Type::Q2_K, "Q2_K", 256, 84},     {Type::Q3_K, "Q3_K", 256, 110},
    {Type::Q4_K, "Q4_K", 256, 144},    {Type::Q5_K, "Q5_K", 256, 176},
    {Type::Q6_K, "Q6_K", 256, 210},    {Type::Q8_K, "Q8_K", 256, 292},
    {Type::IQ2_XXS, "IQ2_XXS", 256, 66}, {Type::IQ2_XS, "IQ2_XS", 256, 74},
    {Type::IQ3_XXS, "IQ3_XXS", 256, 98}, {Type::IQ1_S, "IQ1_S", 256, 50},
    {Type::IQ4_NL, "IQ4_NL", 32, 18},  {Type::IQ3_S, "IQ3_S", 256, 110},
    {Type::IQ2_S, "IQ2_S", 256, 82},   {Type::IQ4_XS, "IQ4_XS", 256, 136},
    {Type::I8, "I8", 1, 1},            {Type::I16, "I16", 1, 2},
    {Type::I32, "I32", 1, 4},          {Type::I64, "I64", 1, 8},
    {Type::F64, "F64", 1, 8},          {Type::IQ1_M, "IQ1_M", 256, 56},
    {Type::BF16, "BF16", 1, 2},        {Type::TQ1_0, "TQ1_0", 256, 54},
    {Type::TQ2_0, "TQ2_0", 256, 66},   {Type::MXFP4, "MXFP4", 32, 17},
    {Type::NVFP4, "NVFP4", 64, 36},    {Type::Q1_0, "Q1_0", 128, 18},
    {Type::Q2_0, "Q2_0", 64, 18},
};
}  // namespace

const TypeInfo* type_info(uint32_t id) {
    for (const auto& t : kTypes)
        if (static_cast<uint32_t>(t.type) == id) return &t;
    return nullptr;
}

uint64_t type_bytes(uint32_t id, uint64_t n) {
    const TypeInfo* t = type_info(id);
    if (!t || n % t->block_elems != 0) return 0;
    return n / t->block_elems * t->block_bytes;
}

}  // namespace limbo
