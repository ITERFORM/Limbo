// Copyright (c) 2026 ITERFORM. All rights reserved.
// src/cuda/gemm_ext_b.cu — the batched int8 products for IQ1_S, IQ1_M, IQ4_NL and Q2_0 (see gemm_mmq.cuh).
#include <vector>

#include "gemm_mmq.cuh"
#include "limbo/quants.hpp"

namespace limbo::cuda {

namespace {
using namespace mmq;
uint64_t* g_g1s = nullptr;  // the IQ1 grid
int8_t* g_nl = nullptr;     // the IQ4_NL values
}  // namespace

bool ext_init_b(std::string* err) {
    if (g_g1s) return true;
    cudaError_t e = cudaMalloc(reinterpret_cast<void**>(&g_g1s), sizeof(uint64_t) * 2048);
    if (e == cudaSuccess) e = cudaMalloc(reinterpret_cast<void**>(&g_nl), 16);
    if (e == cudaSuccess) e = cudaMemcpy(g_g1s, quants::kIq1sGrid, sizeof(uint64_t) * 2048, cudaMemcpyHostToDevice);
    if (e == cudaSuccess) e = cudaMemcpy(g_nl, quants::kIq4NlValues, 16, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        if (err) *err = std::string("gemm_ext_b init: ") + cudaGetErrorString(e);
        g_g1s = nullptr;
        return false;
    }
    return true;
}

bool ext_launch_b(uint32_t type, const Args& a, int nt, int bm, int mode, dim3 grid, cudaStream_t s) {
    if (!g_g1s) return false;
    Args b = a;
    b.tb.g1s = g_g1s;
    b.tb.tnl = g_nl;
    switch (type) {
        case mmq::kIQ1_S: return mmq::run_mmq<mmq::kIQ1_S>(b, nt, bm, mode, grid, s);
        case mmq::kIQ1_M: return mmq::run_mmq<mmq::kIQ1_M>(b, nt, bm, mode, grid, s);
        case mmq::kIQ4_NL: return mmq::run_mmq<mmq::kIQ4_NL>(b, nt, bm, mode, grid, s);
        case mmq::kQ2_0: return mmq::run_mmq<mmq::kQ2_0>(b, nt, bm, mode, grid, s);
        default: return false;
    }
}

}  // namespace limbo::cuda
