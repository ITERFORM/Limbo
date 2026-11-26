// Copyright (c) 2026 ITERFORM. All rights reserved.
// src/cuda/gemm_ext_a.cu — the batched int8 products for IQ3_XXS, IQ2_XXS and IQ2_XS (see gemm_mmq.cuh): the device
// copies of their codebooks and the instantiations of the shared-memory kernel.
#include <vector>

#include "gemm_mmq.cuh"
#include "limbo/quants.hpp"

namespace limbo::cuda {

namespace {
using namespace mmq;
uint64_t* g_xxs = nullptr;   // IQ2_XXS grid
uint64_t* g_xs = nullptr;    // IQ2_XS grid
uint32_t* g_3xxs = nullptr;  // IQ3_XXS grid
uint8_t* g_sgn = nullptr;    // the 7-bit sign table
}  // namespace

bool ext_init_a(std::string* err) {
    if (g_xxs) return true;
    cudaError_t e = cudaMalloc(reinterpret_cast<void**>(&g_xxs), sizeof(uint64_t) * 256);
    if (e == cudaSuccess) e = cudaMalloc(reinterpret_cast<void**>(&g_xs), sizeof(uint64_t) * 512);
    if (e == cudaSuccess) e = cudaMalloc(reinterpret_cast<void**>(&g_3xxs), sizeof(uint32_t) * 256);
    if (e == cudaSuccess) e = cudaMalloc(reinterpret_cast<void**>(&g_sgn), 128);
    if (e == cudaSuccess) e = cudaMemcpy(g_xxs, quants::kIq2xxsGrid, sizeof(uint64_t) * 256, cudaMemcpyHostToDevice);
    if (e == cudaSuccess) e = cudaMemcpy(g_xs, quants::kIq2xsGrid, sizeof(uint64_t) * 512, cudaMemcpyHostToDevice);
    if (e == cudaSuccess) e = cudaMemcpy(g_3xxs, quants::kIq3xxsGrid, sizeof(uint32_t) * 256, cudaMemcpyHostToDevice);
    if (e == cudaSuccess) e = cudaMemcpy(g_sgn, quants::kSignsIq2xs, 128, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        if (err) *err = std::string("gemm_ext_a init: ") + cudaGetErrorString(e);
        g_xxs = nullptr;
        return false;
    }
    return true;
}

bool ext_launch_a(uint32_t type, const Args& a, int nt, int bm, int mode, dim3 grid, cudaStream_t s) {
    if (!g_xxs) return false;
    Args b = a;
    b.tb.g2xxs = g_xxs;
    b.tb.g2xs = g_xs;
    b.tb.g3xxs = g_3xxs;
    b.tb.signs = g_sgn;
    switch (type) {
        case mmq::kIQ3_XXS: return mmq::run_mmq<mmq::kIQ3_XXS>(b, nt, bm, mode, grid, s);
        case mmq::kIQ2_XXS: return mmq::run_mmq<mmq::kIQ2_XXS>(b, nt, bm, mode, grid, s);
        case mmq::kIQ2_XS: return mmq::run_mmq<mmq::kIQ2_XS>(b, nt, bm, mode, grid, s);
        default: return false;
    }
}

}  // namespace limbo::cuda
