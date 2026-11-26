// Copyright (c) 2026 ITERFORM. All rights reserved.
// src/cuda/gemm_ext_c.cu — the batched int8 products for the K-quants with a minimum: Q4_K, Q5_K and Q2_K, and Q3_K (no minimum; see
// gemm_mmq.cuh). The minimum term needs the sums of the activations (act_quant_sums). And NVFP4 (Inkling-2's experts):
// no codebook in memory, its E2M1 codes are decoded from their bits.
#include "gemm_mmq.cuh"

namespace limbo::cuda {

bool ext_init_c(std::string*) { return true; }

bool ext_launch_c(uint32_t type, const mmq::Args& a, int nt, int bm, int mode, dim3 grid, cudaStream_t s) {
    switch (type) {
        case mmq::kQ4_K: return mmq::run_mmq<mmq::kQ4_K>(a, nt, bm, mode, grid, s);
        case mmq::kQ5_K: return mmq::run_mmq<mmq::kQ5_K>(a, nt, bm, mode, grid, s);
        case mmq::kQ2_K: return mmq::run_mmq<mmq::kQ2_K>(a, nt, bm, mode, grid, s);
        case mmq::kQ3_K: return mmq::run_mmq<mmq::kQ3_K>(a, nt, bm, mode, grid, s);
        case mmq::kNVFP4: return mmq::run_mmq<mmq::kNVFP4>(a, nt, bm, mode, grid, s);
        default: return false;
    }
}

}  // namespace limbo::cuda
