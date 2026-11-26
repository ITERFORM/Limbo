// Copyright (c) 2026 ITERFORM. All rights reserved.
// src/cuda/gpu_q4e_pf.cu — qwen4exp's prompt pass: a piece of up to kQ4ePfMax tokens, layer-major. For each layer
// the whole expert stack is staged into VRAM once (the slab's residents device to device, the rest from the file's
// mapping), then the piece runs through the layer in micro-batches of kQ4eMb tokens in token order (the delta-net
// recurrence, the convolutions, the K/V and the indexer state advance exactly as the single-row step advances them).
// The products are int8 tensor-core GEMMs over the activation planes (limbo/cuda/gemm.hpp), the BF16 mixers a tiled
// fp32 GEMM, the routed experts grouped (moe_build / moe_gate_up / moe_down). Not the decode step's bits: the planes
// round the activations, as the other prompt passes do.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#include "dec_ops.hpp"
#include "gpu_impl.hpp"
#include "limbo/cuda/gemm.hpp"
#include "limbo/cuda/gemv.hpp"
#include "limbo/cuda/gpu.hpp"
#include "limbo/types.hpp"
#include "ops.hpp"
#include "q4e_qsa.hpp"

namespace limbo::cuda {

void q4e_unpin_this_thread();   // gpu_q4e.cu

using namespace impl;

namespace {

// y[t][n] = W[n] . x[t], W BF16 [n_out][n_in]: a block is 64 outputs x 64 rows, k in slabs of 32 through shared memory,
// each thread 4 x 4 outputs.
__global__ void __launch_bounds__(256) bf16_gemm_k(const uint16_t* __restrict__ W, const float* __restrict__ x, int xs,
                                                   float* __restrict__ y, int ys, int n_out, int n_in, int T) {
    __shared__ float sw[32][65], sx[32][65];
    const int n0 = blockIdx.x * 64, t0 = blockIdx.y * 64, tid = threadIdx.x;
    const int tn = (tid & 15) * 4, tt = (tid >> 4) * 4;
    float acc[4][4] = {};
    for (int k0 = 0; k0 < n_in; k0 += 32) {
        for (int i = tid; i < 64 * 32; i += 256) {
            const int r = i / 32, c = i % 32, n = n0 + r, k = k0 + c, t = t0 + r;
            sw[c][r] = n < n_out && k < n_in ? __uint_as_float(uint32_t(W[size_t(n) * n_in + k]) << 16) : 0.f;
            sx[c][r] = t < T && k < n_in ? x[size_t(t) * xs + k] : 0.f;
        }
        __syncthreads();
#pragma unroll 8
        for (int c = 0; c < 32; ++c) {
            float a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                a[i] = sw[c][tn + i];
                b[i] = sx[c][tt + i];
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[j][i] = fmaf(a[i], b[j], acc[j][i]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int t = t0 + tt + j;
        if (t >= T) continue;
#pragma unroll
        for (int i = 0; i < 4; ++i)
            if (n0 + tn + i < n_out) y[size_t(t) * ys + n0 + tn + i] = acc[j][i];
    }
}

// The same product on the tensor cores: W's BF16 rows against x split into two BF16 terms (hi + lo, ~16 mantissa
// bits), fp32 accumulation. A block is 64 outputs x 64 rows, four warps of 32 x 32; K in slabs of 32. With
// splits > 1 block z sums its share of K into part[z] (reduced by splitk_sum_k).
template <bool LO>
__global__ void __launch_bounds__(128) bf16_tc_gemm_k(const __nv_bfloat16* __restrict__ W, const float* __restrict__ x, int xs,
                                                      float* __restrict__ y, int ys, int n_out, int n_in, int T, int k_per) {
    using namespace nvcuda;
    constexpr int BK = 32, LD = BK + 8;
    __shared__ __align__(32) __nv_bfloat16 sw[64][LD], sxh[64][LD], sxl[64][LD];
    __shared__ __align__(32) float so[4][32 * 32];
    const int n0 = blockIdx.x * 64, t0 = blockIdx.y * 64, warp = threadIdx.x >> 5, wn = warp & 1, wt = warp >> 1;
    const int k_lo = blockIdx.z * k_per, k_hi = min(n_in, k_lo + k_per);
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.0f);
    for (int k0 = k_lo; k0 < k_hi; k0 += BK) {
        for (int i = threadIdx.x; i < 64 * 4; i += 128) {
            const int r = i >> 2, c = (i & 3) * 8, n = n0 + r;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (n < n_out) v = *reinterpret_cast<const uint4*>(W + size_t(n) * n_in + k0 + c);
            *reinterpret_cast<uint4*>(&sw[r][c]) = v;
        }
        for (int i = threadIdx.x; i < 64 * 8; i += 128) {
            const int r = i >> 3, c = (i & 7) * 4, t = t0 + r;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (t < T) v = *reinterpret_cast<const float4*>(x + size_t(t) * xs + k0 + c);
            const float f[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const __nv_bfloat16 h = __float2bfloat16_rn(f[u]);
                sxh[r][c + u] = h;
                if constexpr (LO) sxl[r][c + u] = __float2bfloat16_rn(f[u] - __bfloat162float(h));
            }
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bh[2], bl[2];
#pragma unroll
            for (int i = 0; i < 2; ++i) wmma::load_matrix_sync(a[i], &sw[wn * 32 + i * 16][kk], LD);
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                wmma::load_matrix_sync(bh[j], &sxh[wt * 32 + j * 16][kk], LD);
                if constexpr (LO) wmma::load_matrix_sync(bl[j], &sxl[wt * 32 + j * 16][kk], LD);
            }
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    if constexpr (LO) wmma::mma_sync(acc[i][j], a[i], bl[j], acc[i][j]);
                    wmma::mma_sync(acc[i][j], a[i], bh[j], acc[i][j]);
                }
        }
        __syncthreads();
    }
    // the warp's 32 x 32 through shared memory: so[warp][t * 32 + n]
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
            wmma::store_matrix_sync(&so[warp][(j * 16) * 32 + i * 16], acc[i][j], 32, wmma::mem_col_major);
    __syncwarp();
    float* out = y + size_t(blockIdx.z) * size_t(T) * size_t(ys);
    const int lane = threadIdx.x & 31;
    for (int r = 0; r < 32; ++r) {
        const int t = t0 + wt * 32 + r, n = n0 + wn * 32 + lane;
        if (t < T && n < n_out) out[size_t(t) * ys + n] = so[warp][r * 32 + lane];
    }
}

__global__ void splitk_sum_k(const float* __restrict__ part, float* __restrict__ y, int ys, int n_out, int T, int splits) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (n >= n_out) return;
    float a = 0.f;
    for (int z = 0; z < splits; ++z) a += part[(size_t(z) * T + t) * n_out + n];
    y[size_t(t) * ys + n] = a;
}

// activation planes of the prompt pass's int8 products: 2 (default) or 1 (Q8_1, llama.cpp's numerics)
int q4e_pf_planes() {
    static const int n = [] {
        const char* e = std::getenv("LIMBO_Q4E_PF_PLANES");
        return e && std::atoi(e) == 1 ? 1 : 2;
    }();
    return n;
}

__global__ void stack_fill_rows_k(float* st, const float* x, int hc, int n, int T) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i >= hc * n) return;
    st[size_t(t) * hc * n + i] = x[size_t(t) * n + i % n];
}

__global__ void count_add_k(const int32_t* src, int32_t* dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

bool q4e_count_add(const int32_t* src, int32_t* dst, int n, cudaStream_t s) {
    count_add_k<<<(n + 255) / 256, 256, 0, s>>>(src, dst, n);
    return cudaGetLastError() == cudaSuccess;
}

__global__ void iota_k(int* v, int base, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] = base + i;
}

}  // namespace

// ---- workspace ---------------------------------------------------------------------------------------------
// The prompt pass's workspace, made at open before the expert slab takes the card (the stage is lent from the slab).
bool Gpu::Impl::ensure_q4e_pf() {
    if (qpf.ready) return true;
    if (qpf.failed) return false;
    const Cfg& cc = c;
    const Q4e& qq = q;
    const size_t P = kQ4ePfMax, B = kQ4eMb, e = size_t(cc.n_embd), hc = size_t(qq.hc), pitch = hc * e;
    const size_t qw = size_t(cc.n_head) * cc.hd, kvw = size_t(cc.n_head_kv) * cc.hd;
    const size_t vn = size_t(cc.n_v_head) * cc.gdn_head, qn = size_t(cc.n_k_head) * cc.gdn_head;
    const size_t nu = size_t(cc.n_used), pairs = B * nu, nff = size_t(cc.n_ff_exp), ns = size_t(cc.n_ff_shexp);
    const size_t ew = size_t(qq.n_heads) * size_t(qq.np);
    auto f = [&](size_t n) { return dalloc<float>(n); };
    qpf.st = f(P * pitch);
    qpf.posv = dalloc<int32_t>(P);
    qpf.tok = dalloc<int32_t>(P);
    qpf.xn = f(B * pitch);
    qpf.lo = f(B * size_t(qq.rank));
    qpf.gate = f(B * pitch);
    qpf.mixed = f(B * e);
    qpf.inj = f(B * hc);
    qpf.out = f(B * e);
    qpf.x = f(B * e);
    qpf.emb = f(P * ew);
    qpf.splitk = f(size_t(kSplitK) * B * 512);
    if (qq.idx_kpool > 0) {
        const int kp = qq.idx_kpool, n_top = std::max(1, qq.idx_top_k / kp);
        const size_t nb = size_t(n_ctx) / size_t(kp) + 1;
        qpf.ikraw = f(B * 128);
        qpf.iqraw = f(B * size_t(qq.idx_heads) * 128);
        qpf.iqn = f(B * size_t(qq.idx_heads) * 128);
        qpf.qscores = f(size_t(kQsaG) * nb);
        qpf.qsel = dalloc<int32_t>(size_t(kQsaG) * size_t(n_top));
        qpf.qscr = f(qsa::attn_scratch_floats(kQsaG, cc.n_head, n_top, kp));
        if (!qpf.qscr || !qpf.qsel || !qpf.qscores || !ensure_qsa()) {
            qpf.failed = true;
            return false;
        }
    }
    qpf.pkey = f(B * pitch);
    qpf.pval = f(B * e);
    qpf.pgated = f(B * pitch);
    qpf.pnrm = f(B * pitch);
    qpf.pconv = f(B * pitch);
    qpf.qfull = f(B * 2 * qw);
    qpf.kbuf = f(B * kvw);
    qpf.vbuf = f(B * kvw);
    qpf.qbuf = f(B * qw);
    qpf.attn = f(B * qw);
    qpf.qkv = f(B * size_t(cc.qkv_w));
    qpf.conv = f(B * size_t(cc.qkv_w));
    qpf.qk = f(B * 2 * qn);
    qpf.al = f(B * size_t(cc.dt_rank));
    qpf.be = f(B * size_t(cc.dt_rank));
    qpf.o = f(B * vn);
    qpf.z = f(B * vn);
    qpf.gated = f(B * vn);
    qpf.rlog = f(B * size_t(cc.n_expert));
    qpf.wts = f(pairs);
    qpf.ids = dalloc<int32_t>(pairs);
    qpf.h = f(pairs * nff);
    qpf.dn = f(pairs * e);
    qpf.gs = f(B * ns);
    qpf.us = f(B * ns);
    qpf.hs = f(B * ns);
    qpf.shared = f(B * e);
    qpf.sg = f(B);
    const size_t kmax = std::max({e, qw, vn, nff, ew});
    qpf.xq = dalloc<int8_t>(3 * std::max(B * kmax, pairs * nff));
    qpf.xs = f(std::max(B * kmax, pairs * nff) / 32 + 64);
    qpf.xsum = f(std::max(B * kmax, pairs * nff) / 16 + 64);
    qpf.lists.cnt = dalloc<int32_t>(size_t(cc.n_expert));
    qpf.lists.start = dalloc<int32_t>(size_t(cc.n_expert) + 1);
    qpf.lists.list_tok = dalloc<int32_t>(pairs);
    qpf.lists.pos_of = dalloc<int32_t>(pairs);
    qpf.lists.tiles_small = dalloc<int4>(size_t(moe_max_tiles_small(cc.n_expert)));
    qpf.lists.tiles_big = dalloc<int4>(size_t(moe_max_tiles_big(int(pairs), cc.n_expert)));
    qpf.lists.n_tiles = dalloc<int32_t>(2);
    qpf.cnt_d = dalloc<int32_t>(size_t(cc.n_layer_main) * size_t(cc.n_expert));
    if (!qpf.cnt_d || cudaMemset(qpf.cnt_d, 0, sizeof(int32_t) * size_t(cc.n_layer_main) * size_t(cc.n_expert)) != cudaSuccess) {
        qpf.failed = true;
        return false;
    }
    for (int b = 0; b < 2; ++b) {
        qpf.eptr_d[b] = reinterpret_cast<const uint8_t**>(dalloc<uint64_t>(size_t(cc.n_expert)));
        if (cudaMallocHost(reinterpret_cast<void**>(&qpf.eptr_h[b]), sizeof(void*) * size_t(cc.n_expert)) != cudaSuccess ||
            cudaEventCreateWithFlags(&qpf.staged[b], cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&qpf.done[b], cudaEventDisableTiming) != cudaSuccess) {
            qpf.failed = true;
            return false;
        }
    }
    if (cudaMallocHost(reinterpret_cast<void**>(&qpf.cnt_h), sizeof(int32_t) * size_t(cc.n_layer_main) * size_t(cc.n_expert)) !=
            cudaSuccess ||
        cudaMallocHost(reinterpret_cast<void**>(&qpf.emb_h[0]), sizeof(float) * P * ew) != cudaSuccess ||
        cudaMallocHost(reinterpret_cast<void**>(&qpf.emb_h[1]), sizeof(float) * P * ew) != cudaSuccess) {
        qpf.failed = true;
        return false;
    }
    std::string gerr;
    if (!qpf.st || !qpf.eptr_d[1] || !qpf.xq || !qpf.dn || !gemm_init(&gerr)) {
        qpf.failed = true;
        return false;
    }
    qpf.ready = true;
    return true;
}

// y [T][n_out] = W x[t]: int8 GEMM for the quantised formats, the tiled kernel for BF16, gemm_f32 for F32.
bool Gpu::Impl::q4e_mm(const W& w, const float* x, int xs, float* y, int ys, int n_out, int n_in, int T) {
    cudaStream_t s = stream;
    if (w.type == uint32_t(Type::BF16) && n_in % 32 == 0 && xs % 4 == 0 && !std::getenv("LIMBO_Q4E_BF16_SIMT")) {
        const int splits = n_out <= 512 && n_in >= 2048 ? kSplitK : 1;
        const int k_per = (n_in / 32 + splits - 1) / splits * 32;
        const dim3 g(unsigned((n_out + 63) / 64), unsigned((T + 63) / 64), unsigned(splits));
        // the activation in BF16 (the checkpoint's own numerics); LIMBO_Q4E_BF16_SPLIT=1 adds the low half (~16 bits)
        static const bool lo = std::getenv("LIMBO_Q4E_BF16_SPLIT") != nullptr;
        auto go = [&](float* out, int ld) {
            if (lo)
                bf16_tc_gemm_k<true><<<g, 128, 0, s>>>(reinterpret_cast<const __nv_bfloat16*>(w.p), x, xs, out, ld, n_out, n_in, T, k_per);
            else
                bf16_tc_gemm_k<false><<<g, 128, 0, s>>>(reinterpret_cast<const __nv_bfloat16*>(w.p), x, xs, out, ld, n_out, n_in, T, k_per);
        };
        if (splits == 1) {
            go(y, ys);
        } else {
            go(qpf.splitk, n_out);
            splitk_sum_k<<<dim3(unsigned((n_out + 255) / 256), unsigned(T)), 256, 0, s>>>(qpf.splitk, y, ys, n_out, T, splits);
        }
        return cudaGetLastError() == cudaSuccess;
    }
    if (w.type == uint32_t(Type::BF16)) {
        bf16_gemm_k<<<dim3(unsigned((n_out + 63) / 64), unsigned((T + 63) / 64)), 256, 0, s>>>(
            reinterpret_cast<const uint16_t*>(w.p), x, xs, y, ys, n_out, n_in, T);
        return cudaGetLastError() == cudaSuccess;
    }
    if (w.type == uint32_t(Type::F32) && n_in % 256 == 0 && xs == n_in && ys == n_out)
        return gemm_f32(reinterpret_cast<const float*>(w.p), x, y, T, n_out, n_in, s);
    if (gemm_supported(w.type) && n_in % 128 == 0 && xs == n_in && ys == n_out) {
        const int nt = q4e_pf_planes();
        if (!act_quant_sums(x, T, n_in, nt, qpf.xq, qpf.xs, qpf.xsum, s)) return false;
        return gemm(w.type, w.p, qpf.xq, qpf.xs, nt, T, y, n_out, n_in, false, s, qpf.xsum);
    }
    for (int t = 0; t < T; ++t)
        if (!limbo::cuda::gemv(w.type, w.p, x + size_t(t) * xs, y + size_t(t) * ys, n_out, n_in, false, s)) return false;
    return true;
}

namespace {
struct HcClock {
    bool on = std::getenv("LIMBO_Q4E_PF_LOG") != nullptr;
    std::vector<cudaEvent_t> ev;
    void mark(cudaStream_t s) {
        if (!on) return;
        cudaEvent_t e;
        cudaEventCreate(&e);
        cudaEventRecord(e, s);
        ev.push_back(e);
    }
    void report() {
        if (!on || ev.size() < 6) return;
        cudaEventSynchronize(ev.back());
        double acc[5] = {};
        for (size_t i = 0; i + 5 < ev.size() + 0 && i + 5 <= ev.size() - 1; i += 6)
            for (int k = 0; k < 5; ++k) {
                float a = 0;
                cudaEventElapsedTime(&a, ev[i + size_t(k)], ev[i + size_t(k) + 1]);
                acc[k] += a;
            }
        std::fprintf(stderr, "q4e prefill hc: norms %.0f, down %.0f, silu+up %.0f, gate_mix %.0f, inject %.0f ms\n", acc[0],
                     acc[1], acc[2], acc[3], acc[4]);
        for (cudaEvent_t e : ev) cudaEventDestroy(e);
        ev.clear();
    }
};
HcClock g_hc_clock;
// the delta-net mixer's steps: qkv | conv + norms | alpha, beta | scan | gate | out
struct GdnClock {
    bool on = std::getenv("LIMBO_Q4E_PF_LOG") != nullptr;
    std::vector<cudaEvent_t> ev;
    void mark(cudaStream_t s) {
        if (!on) return;
        cudaEvent_t e;
        cudaEventCreate(&e);
        cudaEventRecord(e, s);
        ev.push_back(e);
    }
    void report() {
        if (!on || ev.size() < 7) return;
        cudaEventSynchronize(ev.back());
        double acc[6] = {};
        for (size_t i = 0; i + 6 < ev.size() + 1 && i + 6 <= ev.size() - 1; i += 7)
            for (int k = 0; k < 6; ++k) {
                float a = 0;
                cudaEventElapsedTime(&a, ev[i + size_t(k)], ev[i + size_t(k) + 1]);
                acc[k] += a;
            }
        std::fprintf(stderr, "q4e prefill gdn: qkv %.0f, conv+norm %.0f, alpha/beta %.0f, scan %.0f, gate %.0f, out %.0f ms\n",
                     acc[0], acc[1], acc[2], acc[3], acc[4], acc[5]);
        for (cudaEvent_t e : ev) cudaEventDestroy(e);
        ev.clear();
    }
};
GdnClock g_gdn_clock;
}  // namespace

bool Gpu::Impl::q4e_hc_read_pf(const Layer::Hc& m, const float* st, int T) {
    HcClock& hk = g_hc_clock;
    const Cfg& cc = c;
    const int e = cc.n_embd, hc = q.hc, rank = q.rank, pitch = hc * e;
    const float inv_hc = 1.0f / float(hc), eps = float(cc.eps);
    cudaStream_t s = stream;
    hk.mark(s);
    if (!dec::rmsnorm(st, m.norm, qpf.xn, e, eps, T, s, pitch, pitch, hc)) return false;
    hk.mark(s);
    if (!q4e_mm(m.down, qpf.xn, pitch, qpf.lo, rank, rank, pitch, T)) return false;
    hk.mark(s);
    ops::silu_scale(qpf.lo, qpf.lo, T * rank, inv_hc, s);
    if (!q4e_mm(m.up, qpf.lo, rank, qpf.gate, pitch, pitch, rank, T)) return false;
    hk.mark(s);
    ops::hc_gate_mix_rows(qpf.xn, qpf.gate, qpf.mixed, hc, e, T, inv_hc, s);
    hk.mark(s);
    if (m.inject.p && !q4e_mm(m.inject, qpf.xn, pitch, qpf.inj, hc, hc, pitch, T)) return false;
    hk.mark(s);
    return true;
}

// Layer l's experts for buffer b, on the copy stream: a VRAM resident is used where it sits, every other expert is
// copied into the stage -- from the pinned RAM tier (DMA), or from the file's mapping when no tier holds it. The
// pointer table goes up with them; `staged[b]` marks the lot, and the copy waits for the compute that last read b.
bool Gpu::Impl::q4e_stage_layer(int l, int b) {
    const Cfg& cc = c;
    Q4e& qq = q;
    Layer& y = L[size_t(l)];
    const uint64_t ge = y.ge_bytes, de = y.de_bytes, blob = 2 * ge + de;
    const Q4e::Slab& sb = qq.slab[size_t(l)];
    cudaStream_t cs = qq.cstream;
    if (cudaEventSynchronize(qpf.staged[b]) != cudaSuccess) return false;   // eptr_h[b]'s last upload has been read
    if (cudaStreamWaitEvent(cs, qpf.done[b], 0) != cudaSuccess) return false;
    if (qq.lend_single && cudaStreamWaitEvent(cs, qpf.done[b ^ 1], 0) != cudaSuccess) return false;
    const uint8_t** tab = qpf.eptr_h[b];
    // packed into the lent slab ranges of this buffer (q4e_lend), blob by blob
    const std::vector<std::pair<uint8_t*, uint64_t>>& rng = qq.lend_rng[size_t(b)];
    size_t ri = 0;
    uint64_t off = 0;
    for (int ex = 0; ex < cc.n_expert; ++ex) {
        const int32_t slot = sb.base ? sb.slot_of[size_t(ex)] : -1;
        if (slot >= 0) {
            tab[ex] = sb.base + uint64_t(slot) * sb.blob;
            continue;
        }
        while (ri < rng.size() && off + blob > rng[ri].second) {
            ++ri;
            off = 0;
        }
        if (ri >= rng.size()) return false;
        uint8_t* dst = rng[ri].first + off;
        off += blob;
        tab[ex] = dst;
        const int32_t rt = sb.ram ? sb.ram_slot_of[size_t(ex)] : -1;
        if (rt >= 0 && sb.ram_ready[size_t(rt)]) {
            if (cudaMemcpyAsync(dst, sb.ram_ptr[size_t(rt)], blob, cudaMemcpyHostToDevice, cs) != cudaSuccess) return false;
            continue;
        }
        if (cudaMemcpyAsync(dst, y.geh.p + uint64_t(ex) * ge, ge, cudaMemcpyHostToDevice, cs) != cudaSuccess ||
            cudaMemcpyAsync(dst + ge, y.ueh.p + uint64_t(ex) * ge, ge, cudaMemcpyHostToDevice, cs) != cudaSuccess ||
            cudaMemcpyAsync(dst + 2 * ge, y.deh.p + uint64_t(ex) * de, de, cudaMemcpyHostToDevice, cs) != cudaSuccess)
            return false;
    }
    if (cudaMemcpyAsync(qpf.eptr_d[b], tab, sizeof(void*) * size_t(cc.n_expert), cudaMemcpyHostToDevice, cs) != cudaSuccess)
        return false;
    return cudaEventRecord(qpf.staged[b], cs) == cudaSuccess;
}

namespace {
struct MoeClock {
    bool on = std::getenv("LIMBO_Q4E_PF_LOG") != nullptr;
    std::vector<cudaEvent_t> ev;
    std::vector<int> stage;
    void mark(int k, cudaStream_t s) {
        if (!on) return;
        cudaEvent_t e;
        cudaEventCreate(&e);
        cudaEventRecord(e, s);
        ev.push_back(e);
        stage.push_back(k);
    }
    void report() {
        if (!on || ev.empty()) return;
        cudaEventSynchronize(ev.back());
        double acc[8] = {};
        for (size_t i = 0; i + 1 < ev.size(); ++i) {
            if (stage[i + 1] == 0) continue;
            float a = 0;
            cudaEventElapsedTime(&a, ev[i], ev[i + 1]);
            acc[stage[i + 1]] += a;
        }
        std::fprintf(stderr, "q4e prefill moe: router+topk+build %.0f, quant %.0f, gate_up %.0f, quant h %.0f, down %.0f, shared %.0f, combine %.0f ms\n",
                     acc[1], acc[2], acc[3], acc[4], acc[5], acc[6], acc[7]);
        for (cudaEvent_t e : ev) cudaEventDestroy(e);
        ev.clear();
        stage.clear();
    }
};
MoeClock g_moe_clock;
}  // namespace

bool Gpu::Impl::q4e_moe_pf(int l, int T, int b) {
    MoeClock& mc = g_moe_clock;
    const Cfg& cc = c;
    Layer& y = L[size_t(l)];
    cudaStream_t s = stream;
    const int e = cc.n_embd, ns = cc.n_ff_shexp, nu = cc.n_used, nff = cc.n_ff_exp, pairs = T * nu;
    const uint64_t ge = y.ge_bytes, blob = 2 * ge + y.de_bytes;
    const int nt = q4e_pf_planes();
    auto at = [](uint64_t o) { return reinterpret_cast<const void*>(uintptr_t(o)); };
    qpf.lists.eptr = qpf.eptr_d[b];
    static const bool bm64_env = !std::getenv("LIMBO_Q4E_PF_BM32");
    qpf.lists.big_bm = bm64_env && T >= 256 && moe_bm64_ok(y.geh.type, e, blob) && moe_bm64_ok(y.deh.type, nff, blob) ? 64 : kMoeBmBig;
    mc.mark(0, s);
    if (!q4e_mm(y.router, qpf.mixed, e, qpf.rlog, cc.n_expert, cc.n_expert, e, T)) return false;
    if (!dec::topk(qpf.rlog, cc.n_expert, nu, T, qpf.ids, qpf.wts, s)) return false;
    if (!moe_build(qpf.ids, T, cc.n_expert, nu, qpf.lists, s)) return false;
    // the routing counts, accumulated per layer on the device side of the copy: the heat the residency is sorted by
    if (!q4e_count_add(qpf.lists.cnt, qpf.cnt_d + size_t(l) * size_t(cc.n_expert), cc.n_expert, s)) return false;
    mc.mark(1, s);
    if (!act_quant_sums(qpf.mixed, T, e, nt, qpf.xq, qpf.xs, qpf.xsum, s)) return false;
    mc.mark(2, s);
    if (!moe_gate_up(y.geh.type, at(0), at(ge), blob, qpf.xq, qpf.xs, nt, T, pairs, qpf.lists, cc.n_expert, qpf.h, nff, e,
                     s, qpf.xsum))
        return false;
    mc.mark(3, s);
    if (!act_quant_sums(qpf.h, pairs, nff, nt, qpf.xq, qpf.xs, qpf.xsum, s)) return false;
    mc.mark(4, s);
    if (!moe_down(y.deh.type, at(2 * ge), blob, qpf.xq, qpf.xs, nt, pairs, qpf.lists, cc.n_expert, qpf.dn, e, nff, s,
                  qpf.xsum))
        return false;
    mc.mark(5, s);
    if (!q4e_mm(y.sgate, qpf.mixed, e, qpf.gs, ns, ns, e, T) || !q4e_mm(y.sup, qpf.mixed, e, qpf.us, ns, ns, e, T))
        return false;
    ops::swiglu(qpf.gs, qpf.us, qpf.hs, T * ns, s);
    if (!q4e_mm(y.sdown, qpf.hs, ns, qpf.shared, e, e, ns, T)) return false;
    ops::shexp_gate_rows(y.shgate, qpf.mixed, qpf.sg, e, T, s);
    mc.mark(6, s);
    if (cudaMemsetAsync(qpf.out, 0, sizeof(float) * size_t(T) * e, s) != cudaSuccess) return false;
    ops::moe_combine_rows(qpf.out, qpf.dn, qpf.lists.pos_of, qpf.wts, qpf.shared, qpf.sg, e, nu, T, s);
    mc.mark(7, s);
    return cudaGetLastError() == cudaSuccess;
}

// The PLE rows of tokens[0..n) at pos0.. into emb_h[buf], on background threads (tok_hist must hold the positions).
void Gpu::Impl::q4e_ple_start(const int32_t* tokens, int n, int pos0, int buf) {
    Q4ePf::PleJob& j = qpf.ple_job;
    j.join();
    j.pos0 = pos0;
    j.n = n;
    j.buf = buf;
    const int ew = q.n_heads * q.np;
    float* dst = qpf.emb_h[buf];
    std::vector<int32_t> tok(tokens, tokens + n);
    auto shared = std::make_shared<std::vector<int32_t>>(std::move(tok));
    constexpr int kTh = 32;
    for (int k = 0; k < kTh; ++k)
        j.th.emplace_back([this, shared, n, pos0, ew, k, dst]() {
            q4e_unpin_this_thread();
            for (int t = k; t < n; t += kTh) q4e_ple_rows((*shared)[size_t(t)], pos0 + t, dst + size_t(t) * ew, false);
        });
}

// The piece tokens[0..n) at positions pos0.., layer-major; the state ends as n single steps leave it. `next` (optional):
// the piece that follows, whose PLE rows are fetched while this one computes.
bool Gpu::Impl::prefill_q4e(const int32_t* tokens, int n, int pos0, const int32_t* next, int next_n) {
    Impl& I = *this;
    const Cfg& cc = c;
    Q4e& qq = q;
    cudaStream_t s = stream;
    if (n < 1 || n > kQ4ePfMax || !ensure_q4e_pf() || !ensure_q4e_rows() || !ensure_dec_scratch()) return false;
    if (pos0 < 0 || pos0 + n > n_ctx || !pages_for(pos0, pos0 + n)) return false;
    const int e = cc.n_embd, hc = qq.hc, pitch = hc * e;
    const int qw = cc.n_head * cc.hd, kvw = cc.n_head_kv * cc.hd;
    const int qn = cc.n_k_head * cc.gdn_head, qkn_w = 2 * qn, vn = cc.n_v_head * cc.gdn_head;
    const float gdn_scale = float(1.0 / std::sqrt(double(cc.gdn_head))), gdn_eps = float(cc.eps / cc.gdn_head);
    const float attn_scale = float(1.0 / std::sqrt(double(cc.hd))), inv_hc = 1.0f / float(hc), eps = float(cc.eps);
    for (int t = 0; t < n; ++t) qq.tok_hist[size_t(pos0 + t)] = tokens[t];
    if (cudaMemcpyAsync(qpf.tok, tokens, sizeof(int32_t) * size_t(n), cudaMemcpyHostToDevice, s) != cudaSuccess) return false;
    iota_k<<<(n + 255) / 256, 256, 0, s>>>(qpf.posv, pos0, n);

    // the residual stack of every token
    for (int b0 = 0; b0 < n; b0 += kQ4eMb) {
        const int T = std::min(kQ4eMb, n - b0);
        if (!embed_rows(I.embd.type, I.embd.p, qpf.tok + b0, T, qpf.x, e, s)) return false;
        stack_fill_rows_k<<<dim3(unsigned((pitch + 255) / 256), unsigned(T)), 256, 0, s>>>(qpf.st + size_t(b0) * pitch,
                                                                                          qpf.x, hc, e, T);
    }
    const int ew = qq.n_heads * qq.np;
    const int* pt = I.pt();
    const int nl = cc.n_layer_main, nmb = (n + kQ4eMb - 1) / kQ4eMb;
    int ple_layer = -1;
    for (int l = 0; l < nl && ple_layer < 0; ++l)
        if (L[size_t(l)].ple_key.p) ple_layer = l;
    // the piece's PLE rows: fetched already (the previous piece started them), or started now, on host threads while
    // the GPU runs the layers before the block
    int ple_buf = 0;
    if (ple_layer >= 0) {
        Q4ePf::PleJob& j = qpf.ple_job;
        if (j.pos0 == pos0 && j.n == n && !j.th.empty()) {
            ple_buf = j.buf;
        } else {
            ple_buf = j.th.empty() ? 0 : j.buf ^ 1;
            j.join();
            q4e_ple_start(tokens, n, pos0, ple_buf);
        }
    }
    q4e_pr_drain();

    const bool plog = std::getenv("LIMBO_Q4E_PF_LOG") != nullptr;
    std::vector<cudaEvent_t> evs;
    auto ev_at = [&](size_t i) -> cudaEvent_t { return evs[i]; };
    const size_t kSt = 0, kC = size_t(2 * nl), kSec = size_t(4 * nl);
    if (plog) {
        evs.resize(kSec + size_t(nl) * size_t(nmb) * 6);
        for (cudaEvent_t& v : evs) cudaEventCreate(&v);
    }
    std::vector<cudaEvent_t> fev;
    double f_idx = 0, f_dense = 0, f_sparse = 0;
    auto fmark = [&](int k) {
        if (!plog) return;
        cudaEvent_t e2;
        cudaEventCreate(&e2);
        cudaEventRecord(e2, s);
        fev.push_back(e2);
        (void)k;
    };
    auto mark = [&](int l, int mb, int k) {
        if (plog) cudaEventRecord(ev_at(kSec + (size_t(l) * size_t(nmb) + size_t(mb)) * 6 + size_t(k)), s);
    };
    const auto wall0 = std::chrono::steady_clock::now();
    double ms_ple_wait = 0;

    if (plog) cudaEventRecord(ev_at(kSt), qq.cstream);
    if (!q4e_stage_layer(0, 0)) return false;
    if (plog) cudaEventRecord(ev_at(kSt + 1), qq.cstream);
    for (int l = 0; l < nl; ++l) {
        Layer& y = L[size_t(l)];
        const int b = l & 1;
        if (cudaStreamWaitEvent(s, qpf.staged[b], 0) != cudaSuccess) return false;
        if (l == ple_layer) {
            const auto tp = std::chrono::steady_clock::now();
            qpf.ple_job.join();
            qpf.ple_job.pos0 = -1;
            ms_ple_wait = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tp).count();
            if (cudaMemcpyAsync(qpf.emb, qpf.emb_h[ple_buf], sizeof(float) * size_t(n) * size_t(ew), cudaMemcpyHostToDevice,
                                s) != cudaSuccess)
                return false;
            // the next piece's rows while this one computes: its positions' tokens go into the history first. The
            // buffer is free once the copy above has run (a host wait at that piece's own PLE layer orders it).
            if (next && next_n > 0) {
                for (int t = 0; t < next_n; ++t) qq.tok_hist[size_t(pos0 + n + t)] = next[t];
                if (cudaStreamSynchronize(s) != cudaSuccess) return false;
                q4e_ple_start(next, next_n, pos0 + n, ple_buf ^ 1);
            }
        }
        if (plog) cudaEventRecord(ev_at(kC + 2 * size_t(l)), s);
        for (int mb = 0, b0 = 0; b0 < n; ++mb, b0 += kQ4eMb) {
            const int T = std::min(kQ4eMb, n - b0);
            float* st = qpf.st + size_t(b0) * pitch;
            const int* pv = qpf.posv + b0;
            mark(l, mb, 0);
            if (l == ple_layer) {
                const float* emb = qpf.emb + size_t(b0) * ew;
                if (!q4e_mm(y.ple_key, emb, ew, qpf.pkey, pitch, pitch, ew, T) ||
                    !q4e_mm(y.ple_value, emb, ew, qpf.pval, e, e, ew, T))
                    return false;
                ops::ple_gate_rows(qpf.pkey, st, qpf.pval, y.ple_gk, y.ple_gq, qpf.pgated, hc, e, T, eps, s);
                if (!dec::rmsnorm(qpf.pgated, y.ple_gc, qpf.pnrm, e, eps, T, s, pitch, pitch, hc)) return false;
                ops::ple_conv_silu_rows(qpf.pnrm, qq.phist, y.ple_conv, qpf.pconv, pitch, qq.kern, qq.ngram, T, s);
                ops::add_into(st, qpf.pgated, qpf.pconv, T * pitch, s);
            }

            // the attention half
            mark(l, mb, 1);
            if (!q4e_hc_read_pf(y.hca, st, T)) return false;
            mark(l, mb, 2);
            if (y.full) {
                if (!q4e_mm(y.q, qpf.mixed, e, qpf.qfull, 2 * qw, 2 * qw, e, T) ||
                    !q4e_mm(y.k, qpf.mixed, e, qpf.kbuf, kvw, kvw, e, T))
                    return false;
                const float* vsrc = qpf.vbuf;
                if (y.v.p) {
                    if (!q4e_mm(y.v, qpf.mixed, e, qpf.vbuf, kvw, kvw, e, T)) return false;
                } else {
                    vsrc = qpf.kbuf;
                }
                I.q4e_prep(l, qpf.qfull, qpf.kbuf, vsrc, y.qn, y.kn, qpf.qbuf, pv, T);
                const bool q8 = I.kv.fmt == KvFormat::Q8;
                const size_t L = size_t(l);
                // rows [0, nd) attend every earlier position (the tensor-core kernel); a row with more complete
                // blocks than the selection keeps attends its selected blocks and its tail, kQsaG rows at a time
                fmark(0);
                int nd = T;
                const int kp = qq.idx_kpool, n_top = kp > 0 ? std::max(1, qq.idx_top_k / kp) : 0;
                if (y.iq.p && kp > 0) {
                    const int heads = qq.idx_heads;
                    if (!q4e_mm(y.ik, qpf.mixed, e, qpf.ikraw, 128, 128, e, T) ||
                        !q4e_mm(y.iq, qpf.mixed, e, qpf.iqraw, heads * 128, heads * 128, e, T) ||
                        !qsa::index_rows_par(qpf.ikraw, qpf.iqraw, y.ikn, y.iqn, qsa.ring[size_t(l)], qsa.pool[size_t(l)],
                                             qpf.iqn, pos0 + b0, T, kp, heads, cc.rope_dim, cc.rope_base, s))
                        return false;
                    const int p_first = (n_top + 1) * kp - 1;   // the first position with nv = (p + 1) / kp > n_top
                    nd = std::clamp(p_first - (pos0 + b0), 0, T);
                }
                fmark(1);
                if (nd > 0 && !q8) {
                    ops::attention_rows(qpf.qbuf, I.kc[L], I.vc[L], qpf.qfull, qpf.attn, pos0 + b0, nd, n_ctx, 0, cc.n_head,
                                        cc.n_head_kv, cc.hd, attn_scale, s, true, pt);
                } else if (nd > 0) {
                    // a Q8 pool: f16 windows of kQ4ePfWin positions, the attention continuing its softmax across them
                    constexpr int kWin = 4096;
                    if (!qpf.kwin && (!(qpf.kwin = dalloc<float>(size_t(kWin) * kvw / 2 + 64)) ||
                                      !(qpf.vwin = dalloc<float>(size_t(kWin) * kvw / 2 + 64)) ||
                                      !(qpf.aml = dalloc<float>(size_t(kQ4eMb) * cc.n_head * 2 + 64))))
                        return false;
                    const int total = pos0 + b0 + nd;
                    for (int k0 = 0; k0 < total; k0 += kWin) {
                        const int rows = std::min(kWin, (total - k0 + 63) / 64 * 64);
                        ops::kv_expand_window(I.kv.kq[L], I.kv.ks[L], I.kv.vq[L], I.kv.vs[L], reinterpret_cast<__half*>(qpf.kwin),
                                              reinterpret_cast<__half*>(qpf.vwin), k0, rows, total, cc.n_head_kv, kWin, s, pt);
                        const int flags = (k0 == 0 ? 1 : 0) | (k0 + kWin >= total ? 2 : 0);
                        ops::attention_rows_window(qpf.qbuf, reinterpret_cast<const __half*>(qpf.kwin), reinterpret_cast<const __half*>(qpf.vwin),
                                                   qpf.qfull, qpf.attn, pos0 + b0, nd, k0, rows, kWin, cc.n_head, cc.n_head_kv, cc.hd,
                                                   attn_scale, flags, qpf.aml, s, true);
                    }
                }
                fmark(2);
                for (int g0 = nd; g0 < T; g0 += kQsaG) {
                    const int G = std::min(kQsaG, T - g0), heads = qq.idx_heads;
                    const int nv_max = (pos0 + b0 + g0 + G) / kp;
                    if (!qsa::select_rows(qpf.iqn + size_t(g0) * heads * 128, qsa.pool[size_t(l)], qpf.qscores, qsa.stride,
                                          qpf.qsel, pv + g0, G, nv_max, kp, heads, n_top, s))
                        return false;
                    if (!qsa::attention_rows(qpf.qbuf + size_t(g0) * qw, q8 ? nullptr : I.kc[L], q8 ? nullptr : I.vc[L],
                                             qpf.qfull + size_t(g0) * 2 * qw, qpf.attn + size_t(g0) * qw, qpf.qscr, qpf.qsel,
                                             pv + g0, pt, 0, G, kp, n_top, cc.n_head, cc.n_head_kv, attn_scale, s,
                                             q8 ? I.kv.kq[L] : nullptr, q8 ? I.kv.ks[L] : nullptr, q8 ? I.kv.vq[L] : nullptr,
                                             q8 ? I.kv.vs[L] : nullptr))
                        return false;
                }
                fmark(3);
                if (!q4e_mm(y.o, qpf.attn, qw, qpf.out, e, e, qw, T)) return false;
            } else {
                GdnClock& gk = g_gdn_clock;
                gk.mark(s);
                if (!q4e_mm(y.qkv, qpf.mixed, e, qpf.qkv, cc.qkv_w, cc.qkv_w, e, T)) return false;
                gk.mark(s);
                ops::conv_silu_rows(qpf.qkv, I.hist[size_t(l)], y.conv, qpf.conv, cc.qkv_w, T, s, true);
                ops::head_norm_rows(qpf.conv, cc.qkv_w, qpf.qk, qkn_w, T, 2 * cc.n_k_head, cc.gdn_head, gdn_eps, gdn_scale,
                                    s);
                gk.mark(s);
                if (!q4e_mm(y.alpha, qpf.mixed, e, qpf.al, cc.dt_rank, cc.dt_rank, e, T) ||
                    !q4e_mm(y.beta, qpf.mixed, e, qpf.be, cc.dt_rank, cc.dt_rank, e, T))
                    return false;
                ops::gdn_coef_rows(qpf.al, qpf.be, cc.dt_rank, y.dtb, y.ssm_a, T, cc.dt_rank, s);
                gk.mark(s);
                ops::gdn_scan_rows(I.S[size_t(l)], qpf.qk, qkn_w, qpf.conv + 2 * qn, cc.qkv_w, qpf.al, qpf.be, cc.dt_rank,
                                   qpf.o, vn, T, cc.n_v_head, cc.n_k_head, cc.gdn_head, cc.dt_rank, gdn_scale, s, true);
                gk.mark(s);
                if (!q4e_mm(y.gate, qpf.mixed, e, qpf.z, vn, vn, e, T)) return false;
                ops::gdn_gate_rows(qpf.o, qpf.z, y.ssm_norm, qpf.gated, T * cc.n_v_head, cc.gdn_head, cc.eps, s, true);
                gk.mark(s);
                if (!q4e_mm(y.out, qpf.gated, vn, qpf.out, e, e, vn, T)) return false;
                gk.mark(s);
            }
            ops::hc_combine_rows(st, qpf.out, qpf.inj, hc, e, T, inv_hc, s);

            // the FFN half
            mark(l, mb, 3);
            if (!q4e_hc_read_pf(y.hcf, st, T)) return false;
            mark(l, mb, 4);
            if (!q4e_moe_pf(l, T, b)) return false;
            ops::hc_combine_rows(st, qpf.out, qpf.inj, hc, e, T, inv_hc, s);
            mark(l, mb, 5);
        }
        if (plog) cudaEventRecord(ev_at(kC + 2 * size_t(l) + 1), s);
        if (cudaEventRecord(qpf.done[b], s) != cudaSuccess) return false;
        if (l + 1 < nl) {
            if (plog) {   // the copy's own time: the wait for the buffer's last reader first
                cudaStreamWaitEvent(qq.cstream, qpf.done[b ^ 1], 0);
                cudaEventRecord(ev_at(kSt + 2 * size_t(l + 1)), qq.cstream);
            }
            if (!q4e_stage_layer(l + 1, b ^ 1)) return false;
            if (plog) cudaEventRecord(ev_at(kSt + 2 * size_t(l + 1) + 1), qq.cstream);
        }
    }
    if (plog) {
        cudaStreamSynchronize(s);
        double st_ms = 0, c_ms = 0, sec[5] = {0, 0, 0, 0, 0}, mix_full = 0;
        for (int l = 0; l < nl; ++l) {
            float a = 0;
            cudaEventElapsedTime(&a, ev_at(kSt + 2 * size_t(l)), ev_at(kSt + 2 * size_t(l) + 1));
            st_ms += a;
            cudaEventElapsedTime(&a, ev_at(kC + 2 * size_t(l)), ev_at(kC + 2 * size_t(l) + 1));
            c_ms += a;
            for (int mb = 0; mb < nmb; ++mb)
                for (int k = 0; k < 5; ++k) {
                    const size_t i = kSec + (size_t(l) * size_t(nmb) + size_t(mb)) * 6 + size_t(k);
                    cudaEventElapsedTime(&a, ev_at(i), ev_at(i + 1));
                    sec[k] += a;
                    if (k == 2 && L[size_t(l)].full) mix_full += a;
                }
        }
        std::fprintf(stderr,
                     "q4e prefill: %d tokens at %d, wall %.0f ms: stage %.0f ms (copy stream), compute %.0f ms [ple %.0f "
                     "(host wait %.0f), hc-attn %.0f, mixer %.0f (full %.0f), hc-ffn %.0f, moe %.0f]\n",
                     n, pos0, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wall0).count(),
                     st_ms, c_ms, sec[0], ms_ple_wait, sec[1], sec[2], mix_full, sec[3], sec[4]);
        for (size_t i = 0; i + 3 < fev.size(); i += 4) {
            float a = 0;
            cudaEventElapsedTime(&a, fev[i], fev[i + 1]);
            f_idx += a;
            cudaEventElapsedTime(&a, fev[i + 1], fev[i + 2]);
            f_dense += a;
            cudaEventElapsedTime(&a, fev[i + 2], fev[i + 3]);
            f_sparse += a;
        }
        g_moe_clock.report();
        g_hc_clock.report();
        g_gdn_clock.report();
        std::fprintf(stderr, "q4e prefill: PLE row cache %llu hits, %llu reads\n",
                     (unsigned long long)qq.ple_hits.load(), (unsigned long long)qq.ple_miss.load());
        std::fprintf(stderr, "q4e prefill: full attention: index %.0f ms, dense %.0f ms, sparse %.0f ms\n", f_idx, f_dense, f_sparse);
        for (cudaEvent_t& v : fev) cudaEventDestroy(v);
        for (cudaEvent_t& v : evs) cudaEventDestroy(v);
    }
    qrw.pending = 0;
    qpf.counted = true;
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace limbo::cuda
