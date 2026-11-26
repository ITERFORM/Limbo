// Copyright (c) 2026 ITERFORM. All rights reserved.
// tests/cuda_gemm_test.cu — the batched int8 tensor-core products against the CPU decoder, on real rows.
//
// For each format the prefill path uses (Q8_0, Q6_K, IQ2_S, IQ3_S, IQ4_XS) the test takes real rows of the
// rung-1 file, runs Y = X W^T for 70 token rows (one full tile of 64 and a ragged one of 6) with the
// activation quantiser at 1, 2 and 3 planes, and compares every element with src/quants.cpp's decoder and a
// double-precision dot. The error is |gpu - cpu| / sum_i |w_i x_i| per element; the bars are 1e-2, 5e-4 and
// 2e-5 for 1, 2 and 3 planes (the activation rounding is the only error: R = 127, 16192, 2e6 levels per
// block of 32). Then the MoE path on one real layer (grouped lists, fused gate/up with SwiGLU, down) and the
// F32 product, which must equal gemv() bit for bit.
//
// The model is LIMBO_TEST_GGUF or the rung-1 file. No GPU, or under 1 GB free beside the desktop: SKIP, exit 0.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "limbo/check.hpp"
#include "limbo/cuda/gemm.hpp"
#include "limbo/cuda/gemv.hpp"
#include "limbo/gguf.hpp"
#include "limbo/quants.hpp"
#include "limbo/types.hpp"

namespace {

using limbo::Type;
namespace cu = limbo::cuda;

constexpr const char* kRung1 =
    "/home/alca/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-MTP-GGUF/snapshots/"
    "5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf";

#define CUDA_OK(call)                                                                                  \
    do {                                                                                               \
        cudaError_t e_ = (call);                                                                       \
        if (e_ != cudaSuccess) {                                                                       \
            std::printf("FAIL %s: %s (%s:%d)\n", #call, cudaGetErrorString(e_), __FILE__, __LINE__);    \
            ++limbo::test::failures();                                                                 \
            return;                                                                                    \
        }                                                                                              \
    } while (0)

std::vector<float> make_x(int n, uint32_t seed) {
    std::vector<float> x(static_cast<size_t>(n));
    uint32_t s = seed * 2654435761u + 12345u;
    for (int i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        const float u = static_cast<float>(s >> 8) / 16777216.0f;
        s = s * 1664525u + 1013904223u;
        const float v = static_cast<float>(s >> 8) / 16777216.0f;
        float g = (u + v - 1.0f) * 2.4f;
        if (i % 97 == 0) g *= 8.0f;  // outliers: the block amplitude is what the quantiser sees
        x[static_cast<size_t>(i)] = g;
    }
    return x;
}

struct Err {
    double max_rel = 0, sum_rel = 0;
    long n = 0;
    void add(double gpu, double cpu, double l1) {
        const double rel = l1 > 0 ? std::fabs(gpu - cpu) / l1 : std::fabs(gpu - cpu);
        max_rel = std::max(max_rel, rel);
        sum_rel += rel;
        ++n;
    }
};

struct Found {
    const limbo::gguf::Tensor* t = nullptr;
};

Found find_tensor(const std::vector<std::unique_ptr<limbo::gguf::File>>& files, uint32_t type) {
    for (int pass = 0; pass < 2; ++pass)
        for (const auto& f : files)
            for (const auto& t : f->tensors()) {
                if (t.type != type || t.ne[0] % 256 != 0 || t.ne[3] != 1) continue;
                if (pass == 0 ? (t.ne[2] == 1 && t.ne[1] >= 256) : (t.ne[2] > 8 && t.ne[1] * t.ne[2] >= 256)) return {&t};
            }
    return {};
}

const double kBar[4] = {0, 1e-2, 5e-4, 2e-5};

void run_dense(uint32_t type, const char* name, const std::string& what_rows, const std::vector<uint8_t>& host, int n_in);

void test_dense(const std::vector<std::unique_ptr<limbo::gguf::File>>& files, uint32_t type, const char* name) {
    const Found fd = find_tensor(files, type);
    if (!fd.t) {
        std::printf("--   %-7s no tensor of this type in the model: not tested\n", name);
        return;
    }
    const limbo::gguf::Tensor& t = *fd.t;
    const int n_in = static_cast<int>(t.ne[0]);
    const int total = static_cast<int>(t.ne[1] * t.ne[2]);
    const int n_out = 256;
    const uint64_t rb = limbo::type_bytes(type, static_cast<uint64_t>(n_in));
    std::vector<uint8_t> host(static_cast<size_t>(n_out) * rb);
    for (int k = 0; k < n_out; ++k)
        std::memcpy(host.data() + static_cast<size_t>(k) * rb,
                    t.data + static_cast<uint64_t>(static_cast<int64_t>(k) * (total - 1) / (n_out - 1)) * rb, rb);
    run_dense(type, name, std::string(t.name), host, n_in);
}

// 256 weight rows of `n_in` columns (host holds their GGUF bytes) times 70 token rows, at 1, 2 and 3 planes, against the CPU
// decoder and a double-precision dot.
void run_dense(uint32_t type, const char* name, const std::string& what_rows, const std::vector<uint8_t>& host, int n_in) {
    const int n_out = 256, T = 70;
    const uint64_t rb = limbo::type_bytes(type, static_cast<uint64_t>(n_in));
    std::vector<float> wf(static_cast<size_t>(n_out) * static_cast<size_t>(n_in));
    for (int k = 0; k < n_out; ++k)
        limbo::quants::dequant_row(type, host.data() + static_cast<size_t>(k) * rb, wf.data() + static_cast<size_t>(k) * static_cast<size_t>(n_in),
                                   static_cast<uint64_t>(n_in));
    std::vector<float> x(static_cast<size_t>(T) * static_cast<size_t>(n_in));
    for (int i = 0; i < T; ++i) {
        const std::vector<float> xi = make_x(n_in, type * 131u + static_cast<uint32_t>(i));
        std::memcpy(x.data() + static_cast<size_t>(i) * static_cast<size_t>(n_in), xi.data(), sizeof(float) * static_cast<size_t>(n_in));
    }
    std::vector<double> cpu(static_cast<size_t>(T) * n_out), l1(static_cast<size_t>(T) * n_out);
    for (int i = 0; i < T; ++i)
        for (int r = 0; r < n_out; ++r) {
            double d = 0, a = 0;
            for (int k = 0; k < n_in; ++k) {
                const double p = double(wf[static_cast<size_t>(r) * static_cast<size_t>(n_in) + static_cast<size_t>(k)]) *
                                 double(x[static_cast<size_t>(i) * static_cast<size_t>(n_in) + static_cast<size_t>(k)]);
                d += p;
                a += std::fabs(p);
            }
            cpu[static_cast<size_t>(i) * n_out + static_cast<size_t>(r)] = d;
            l1[static_cast<size_t>(i) * n_out + static_cast<size_t>(r)] = a;
        }

    void* dW = nullptr;
    float *dx = nullptr, *dy = nullptr, *dsc = nullptr;
    int8_t* dq = nullptr;
    CUDA_OK(cudaMalloc(&dW, host.size()));
    CUDA_OK(cudaMalloc(&dx, sizeof(float) * x.size()));
    CUDA_OK(cudaMalloc(&dy, sizeof(float) * static_cast<size_t>(T) * n_out));
    CUDA_OK(cudaMalloc(&dq, 3 * static_cast<size_t>(T) * static_cast<size_t>(n_in)));
    CUDA_OK(cudaMalloc(&dsc, sizeof(float) * static_cast<size_t>(T) * static_cast<size_t>(n_in) / 32));
    CUDA_OK(cudaMemcpy(dW, host.data(), host.size(), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dx, x.data(), sizeof(float) * x.size(), cudaMemcpyHostToDevice));
    std::printf("     %-7s %-30s n_in %4d, %d weight rows x %d tokens\n", name, what_rows.c_str(), n_in, n_out, T);
    float* dsum = nullptr;
    CUDA_OK(cudaMalloc(&dsum, sizeof(float) * static_cast<size_t>(T) * static_cast<size_t>(n_in) / 16));
    for (int nt = 1; nt <= 3; ++nt) {
        CHECK(cu::act_quant_sums(dx, T, n_in, nt, dq, dsc, dsum, nullptr), "act_quant launches");
        CHECK(cu::gemm(type, dW, dq, dsc, nt, T, dy, n_out, n_in, false, nullptr, dsum), "gemm launches");
        CUDA_OK(cudaDeviceSynchronize());
        std::vector<float> y(static_cast<size_t>(T) * n_out);
        CUDA_OK(cudaMemcpy(y.data(), dy, sizeof(float) * y.size(), cudaMemcpyDeviceToHost));
        Err e;
        for (size_t i = 0; i < y.size(); ++i) e.add(y[i], cpu[i], l1[i]);
        // accumulate form
        std::vector<float> y0(y.size());
        for (size_t i = 0; i < y0.size(); ++i) y0[i] = 0.25f * static_cast<float>(i % 11) - 1.0f;
        CUDA_OK(cudaMemcpy(dy, y0.data(), sizeof(float) * y0.size(), cudaMemcpyHostToDevice));
        CHECK(cu::gemm(type, dW, dq, dsc, nt, T, dy, n_out, n_in, true, nullptr, dsum), "gemm accumulate launches");
        CUDA_OK(cudaDeviceSynchronize());
        CUDA_OK(cudaMemcpy(y.data(), dy, sizeof(float) * y.size(), cudaMemcpyDeviceToHost));
        Err ea;
        for (size_t i = 0; i < y.size(); ++i) ea.add(y[i], cpu[i] + double(y0[i]), l1[i] + std::fabs(double(y0[i])));
        std::printf("             %d plane%s  max rel err %.3e  mean %.3e   accumulate max %.3e\n", nt, nt == 1 ? " " : "s",
                    e.max_rel, e.sum_rel / double(e.n), ea.max_rel);
        char what[96];
        std::snprintf(what, sizeof what, "%s: %d plane(s) within %.0e of sum|w*x|", name, nt, kBar[nt]);
        CHECK(e.max_rel <= kBar[nt], what);
        std::snprintf(what, sizeof what, "%s: %d plane(s) accumulate within %.0e", name, nt, kBar[nt]);
        CHECK(ea.max_rel <= kBar[nt], what);
    }
    cudaFree(dW);
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dq);
    cudaFree(dsc);
    cudaFree(dsum);
}

// ---- the 1-plane quantiser is Q8_1 -----------------------------------------------------------------------
// One plane is llama.cpp's activation format: per block of 32, d = amax / 127 and q = round(x * (127 / amax)) with
// ties away from zero, the scale kept in fp32. Checked bit for bit against the same arithmetic on the host, on random
// rows (with outliers), on a block built to hit exact .5 ties, and on an all-zero block.
void test_q8_1() {
    constexpr int T = 6, K = 256, nb = K / 32;
    std::vector<float> x(static_cast<size_t>(T) * K);
    for (int i = 0; i < T; ++i) {
        const std::vector<float> xi = make_x(K, 4001u + static_cast<uint32_t>(i));
        std::memcpy(x.data() + static_cast<size_t>(i) * K, xi.data(), sizeof(float) * K);
    }
    // row 4, block 0: amax 127 -> d = 1, so x = k + 0.5 sits exactly on a tie; row 5, block 1: all zero
    const float ties[8] = {127.f, 0.5f, 1.5f, -0.5f, -2.5f, 2.5f, 63.5f, -126.5f};
    for (int j = 0; j < 8; ++j) x[4 * K + static_cast<size_t>(j)] = ties[j];
    for (int j = 0; j < 32; ++j) x[5 * K + 32 + static_cast<size_t>(j)] = 0.f;
    float *dx, *dsc;
    int8_t* dq;
    CUDA_OK(cudaMalloc(&dx, sizeof(float) * x.size()));
    CUDA_OK(cudaMalloc(&dq, x.size()));
    CUDA_OK(cudaMalloc(&dsc, sizeof(float) * T * nb));
    CUDA_OK(cudaMemcpy(dx, x.data(), sizeof(float) * x.size(), cudaMemcpyHostToDevice));
    CHECK(cu::act_quant(dx, T, K, 1, dq, dsc, nullptr), "act_quant (1 plane) launches");
    CUDA_OK(cudaDeviceSynchronize());
    std::vector<int8_t> q(x.size());
    std::vector<float> sc(static_cast<size_t>(T) * nb);
    CUDA_OK(cudaMemcpy(q.data(), dq, q.size(), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(sc.data(), dsc, sizeof(float) * sc.size(), cudaMemcpyDeviceToHost));
    long bad_q = 0, bad_s = 0;
    for (int r = 0; r < T; ++r)
        for (int b = 0; b < nb; ++b) {
            float amax = 0.f;
            for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(x[static_cast<size_t>(r) * K + static_cast<size_t>(b) * 32 + static_cast<size_t>(j)]));
            const float id = amax > 0.f ? 127.0f / amax : 0.f, d = amax > 0.f ? amax / 127.0f : 0.f;
            bad_s += std::memcmp(&d, &sc[static_cast<size_t>(r) * nb + static_cast<size_t>(b)], 4) != 0;
            for (int j = 0; j < 32; ++j) {
                const size_t i = static_cast<size_t>(r) * K + static_cast<size_t>(b) * 32 + static_cast<size_t>(j);
                const int want = static_cast<int>(std::round(x[i] * id));  // std::round: ties away from zero
                bad_q += want != q[i];
            }
        }
    std::printf("     Q8_1    %ld of %zu values and %ld of %zu scales differ from the host\n", bad_q, x.size(), bad_s, sc.size());
    CHECK(bad_q == 0 && bad_s == 0, "1 plane = Q8_1: d = amax/127, q = round(x*127/amax), ties away from zero");
    CHECK(q[4 * K + 1] == 1 && q[4 * K + 2] == 2 && q[4 * K + 3] == -1 && q[4 * K + 4] == -3, "ties go away from zero");
    cudaFree(dx);
    cudaFree(dq);
    cudaFree(dsc);
}

// ---- one real MoE layer --------------------------------------------------------------------------------
struct MoeCase {
    uint32_t gtype = 0, utype = 0, dtype = 0;
    const uint8_t *gdata = nullptr, *udata = nullptr, *ddata = nullptr;
    uint64_t gbytes = 0, ubytes = 0, dbytes = 0;
    int K = 0, n_ff = 0, n_exp = 0, n_embd = 0;
    std::string label;
};

void run_moe(const MoeCase& m) {
    const int K = m.K, n_ff = m.n_ff, n_exp = m.n_exp, n_embd = m.n_embd, n_used = 8, T = 40;
    const uint64_t rb_g = limbo::type_bytes(m.gtype, static_cast<uint64_t>(K));
    const uint64_t rb_d = limbo::type_bytes(m.dtype, static_cast<uint64_t>(n_ff));
    const uint64_t eb_g = rb_g * static_cast<uint64_t>(n_ff), eb_d = rb_d * static_cast<uint64_t>(n_embd);

    // ids: expert 5 in every token (a hot expert spanning several tiles), the rest pseudo-random and distinct.
    std::vector<int32_t> ids(static_cast<size_t>(T) * n_used);
    uint32_t s = 7;
    for (int t = 0; t < T; ++t) {
        std::vector<int> pick{5};
        while (static_cast<int>(pick.size()) < n_used) {
            s = s * 1664525u + 1013904223u;
            const int e = static_cast<int>((s >> 8) % static_cast<uint32_t>(n_exp));
            if (std::find(pick.begin(), pick.end(), e) == pick.end()) pick.push_back(e);
        }
        for (int i = 0; i < n_used; ++i) ids[static_cast<size_t>(t) * n_used + static_cast<size_t>(i)] = pick[static_cast<size_t>(i)];
    }
    std::vector<float> x(static_cast<size_t>(T) * K);
    for (int t = 0; t < T; ++t) {
        const std::vector<float> xi = make_x(K, 900u + static_cast<uint32_t>(t));
        std::memcpy(x.data() + static_cast<size_t>(t) * K, xi.data(), sizeof(float) * static_cast<size_t>(K));
    }

    void *dg = nullptr, *du = nullptr, *dd = nullptr;
    float *dx = nullptr, *dxs = nullptr, *dh = nullptr, *dhs = nullptr, *ddn = nullptr, *dxsum = nullptr, *dhsum = nullptr;
    int8_t *dxq = nullptr, *dhq = nullptr;
    int32_t* dids = nullptr;
    cu::MoeLists L;
    const int n_pairs = T * n_used;
    CUDA_OK(cudaMalloc(&dg, m.gbytes));
    CUDA_OK(cudaMalloc(&du, m.ubytes));
    CUDA_OK(cudaMalloc(&dd, m.dbytes));
    CUDA_OK(cudaMalloc(&dx, sizeof(float) * x.size()));
    CUDA_OK(cudaMalloc(&dxq, 3 * x.size()));
    CUDA_OK(cudaMalloc(&dxs, sizeof(float) * x.size() / 32));
    CUDA_OK(cudaMalloc(&dh, sizeof(float) * static_cast<size_t>(n_pairs) * n_ff));
    CUDA_OK(cudaMalloc(&dhq, 3 * static_cast<size_t>(n_pairs) * n_ff));
    CUDA_OK(cudaMalloc(&dhs, sizeof(float) * static_cast<size_t>(n_pairs) * n_ff / 32));
    CUDA_OK(cudaMalloc(&ddn, sizeof(float) * static_cast<size_t>(n_pairs) * n_embd));
    CUDA_OK(cudaMalloc(&dxsum, sizeof(float) * x.size() / 16));
    CUDA_OK(cudaMalloc(&dhsum, sizeof(float) * static_cast<size_t>(n_pairs) * n_ff / 16));
    CUDA_OK(cudaMalloc(&dids, sizeof(int32_t) * ids.size()));
    CUDA_OK(cudaMalloc(&L.cnt, sizeof(int32_t) * static_cast<size_t>(n_exp)));
    CUDA_OK(cudaMalloc(&L.start, sizeof(int32_t) * static_cast<size_t>(n_exp + 1)));
    CUDA_OK(cudaMalloc(&L.list_tok, sizeof(int32_t) * static_cast<size_t>(n_pairs)));
    CUDA_OK(cudaMalloc(&L.pos_of, sizeof(int32_t) * static_cast<size_t>(n_pairs)));
    CUDA_OK(cudaMalloc(&L.tiles_small, sizeof(int4) * static_cast<size_t>(cu::moe_max_tiles_small(n_exp))));
    CUDA_OK(cudaMalloc(&L.tiles_big, sizeof(int4) * static_cast<size_t>(cu::moe_max_tiles_big(n_pairs, n_exp))));
    CUDA_OK(cudaMalloc(&L.n_tiles, 2 * sizeof(int32_t)));
    CUDA_OK(cudaMemcpy(dg, m.gdata, m.gbytes, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(du, m.udata, m.ubytes, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dd, m.ddata, m.dbytes, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dx, x.data(), sizeof(float) * x.size(), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dids, ids.data(), sizeof(int32_t) * ids.size(), cudaMemcpyHostToDevice));

    for (int nt = 1; nt <= 3; ++nt) {
        CHECK(cu::moe_build(dids, T, n_exp, n_used, L, nullptr), "moe_build launches");
        CHECK(cu::act_quant_sums(dx, T, K, nt, dxq, dxs, dxsum, nullptr), "act_quant (post) launches");
        CHECK(cu::moe_gate_up(m.gtype, dg, du, eb_g, dxq, dxs, nt, T, n_pairs, L, n_exp, dh, n_ff, K, nullptr, dxsum),
              "moe_gate_up launches");
        CHECK(cu::act_quant_sums(dh, n_pairs, n_ff, nt, dhq, dhs, dhsum, nullptr), "act_quant (h) launches");
        CHECK(cu::moe_down(m.dtype, dd, eb_d, dhq, dhs, nt, n_pairs, L, n_exp, ddn, n_embd, n_ff, nullptr, dhsum),
              "moe_down launches");
        CUDA_OK(cudaDeviceSynchronize());
        std::vector<int32_t> list_tok(static_cast<size_t>(n_pairs)), pos_of(static_cast<size_t>(n_pairs)), start(static_cast<size_t>(n_exp + 1));
        std::vector<float> h(static_cast<size_t>(n_pairs) * n_ff), dn(static_cast<size_t>(n_pairs) * n_embd);
        CUDA_OK(cudaMemcpy(list_tok.data(), L.list_tok, sizeof(int32_t) * list_tok.size(), cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(pos_of.data(), L.pos_of, sizeof(int32_t) * pos_of.size(), cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(start.data(), L.start, sizeof(int32_t) * start.size(), cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(h.data(), dh, sizeof(float) * h.size(), cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(dn.data(), ddn, sizeof(float) * dn.size(), cudaMemcpyDeviceToHost));

        bool lists_ok = start[static_cast<size_t>(n_exp)] == n_pairs;
        for (int p = 0; p < n_pairs && lists_ok; ++p) {
            const int pos = pos_of[static_cast<size_t>(p)], e = ids[static_cast<size_t>(p)];
            lists_ok = pos >= start[static_cast<size_t>(e)] && pos < start[static_cast<size_t>(e) + 1] &&
                       list_tok[static_cast<size_t>(pos)] == p / n_used;
        }
        CHECK(lists_ok, "moe_build: every (token, choice) pair sits in its expert's range with its token");

        Err eh, ed;
        std::vector<float> w(static_cast<size_t>(K)), g(static_cast<size_t>(K)), u(static_cast<size_t>(K)), wd(static_cast<size_t>(n_ff));
        // gate/up/down rows sampled over pairs, experts and rows
        for (int p = 0; p < n_pairs; p += 9) {
            const int t = p / n_used, e = ids[static_cast<size_t>(p)], pos = pos_of[static_cast<size_t>(p)];
            for (int n = (p % 5); n < n_ff; n += 61) {
                limbo::quants::dequant_row(m.gtype, m.gdata + static_cast<uint64_t>(e) * eb_g + static_cast<uint64_t>(n) * rb_g, g.data(), static_cast<uint64_t>(K));
                limbo::quants::dequant_row(m.utype, m.udata + static_cast<uint64_t>(e) * eb_g + static_cast<uint64_t>(n) * rb_g, u.data(), static_cast<uint64_t>(K));
                double dgv = 0, duv = 0, lg = 0, lu = 0;
                for (int k = 0; k < K; ++k) {
                    const double xv = x[static_cast<size_t>(t) * K + static_cast<size_t>(k)];
                    dgv += double(g[static_cast<size_t>(k)]) * xv;
                    duv += double(u[static_cast<size_t>(k)]) * xv;
                    lg += std::fabs(double(g[static_cast<size_t>(k)]) * xv);
                    lu += std::fabs(double(u[static_cast<size_t>(k)]) * xv);
                }
                const double silu = dgv / (1.0 + std::exp(-dgv));
                const double ref = silu * duv;
                const double scale = 1.1 * lg * std::fabs(duv) + lu * std::fabs(silu);
                eh.add(h[static_cast<size_t>(pos) * n_ff + static_cast<size_t>(n)], ref, scale);
            }
            // down: the GPU's own h row as the input, so only the down product is under test
            for (int n = (p % 7); n < n_embd; n += 97) {
                limbo::quants::dequant_row(m.dtype, m.ddata + static_cast<uint64_t>(e) * eb_d + static_cast<uint64_t>(n) * rb_d, wd.data(), static_cast<uint64_t>(n_ff));
                double d = 0, a = 0;
                for (int k = 0; k < n_ff; ++k) {
                    const double pr = double(wd[static_cast<size_t>(k)]) * double(h[static_cast<size_t>(pos) * n_ff + static_cast<size_t>(k)]);
                    d += pr;
                    a += std::fabs(pr);
                }
                ed.add(dn[static_cast<size_t>(pos) * n_embd + static_cast<size_t>(n)], d, a);
            }
        }
        std::printf("     %s, %d plane%s: gate/up+SwiGLU max rel %.3e (%ld samples)   down max rel %.3e (%ld samples)\n",
                    m.label.c_str(), nt, nt == 1 ? " " : "s", eh.max_rel, eh.n, ed.max_rel, ed.n);
        char what[96];
        std::snprintf(what, sizeof what, "MoE gate/up, %d plane(s), within %.0e", nt, kBar[nt]);
        CHECK(eh.max_rel <= kBar[nt], what);
        std::snprintf(what, sizeof what, "MoE down, %d plane(s), within %.0e", nt, kBar[nt]);
        CHECK(ed.max_rel <= kBar[nt], what);
    }
    cudaFree(dg);
    cudaFree(du);
    cudaFree(dd);
    cudaFree(dx);
    cudaFree(dxq);
    cudaFree(dxs);
    cudaFree(dh);
    cudaFree(dhq);
    cudaFree(dhs);
    cudaFree(ddn);
    cudaFree(dxsum);
    cudaFree(dhsum);
    cudaFree(dids);
    cudaFree(L.cnt);
    cudaFree(L.start);
    cudaFree(L.list_tok);
    cudaFree(L.pos_of);
    cudaFree(L.tiles_small);
    cudaFree(L.tiles_big);
    cudaFree(L.n_tiles);
}

void test_moe(const std::vector<std::unique_ptr<limbo::gguf::File>>& files) {
    const limbo::gguf::Tensor *tg = nullptr, *tu = nullptr, *td = nullptr;
    for (const auto& f : files) {
        if (!tg) tg = f->tensor("blk.0.ffn_gate_exps.weight");
        if (!tu) tu = f->tensor("blk.0.ffn_up_exps.weight");
        if (!td) td = f->tensor("blk.0.ffn_down_exps.weight");
    }
    if (!tg || !tu || !td || tg->type != tu->type || !cu::gemm_supported(tg->type) || !cu::gemm_supported(td->type)) {
        std::printf("--   no supported MoE layer 0 in the model: MoE path not tested\n");
        return;
    }
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    if (free_b < tg->nbytes * 2 + td->nbytes + (1ull << 30)) {
        std::printf("SKIP MoE path: %.0f MB free\n", double(free_b) / 1e6);
        return;
    }
    MoeCase m;
    m.gtype = tg->type;
    m.utype = tu->type;
    m.dtype = td->type;
    m.gdata = tg->data;
    m.udata = tu->data;
    m.ddata = td->data;
    m.gbytes = tg->nbytes;
    m.ubytes = tu->nbytes;
    m.dbytes = td->nbytes;
    m.K = static_cast<int>(tg->ne[0]);
    m.n_ff = static_cast<int>(tg->ne[1]);
    m.n_exp = static_cast<int>(tg->ne[2]);
    m.n_embd = static_cast<int>(td->ne[1]);
    m.label = "MoE layer 0";
    run_moe(m);
}


// ---- formats the model file at hand may not hold: random valid blocks ------------------------------------------------
// Random bytes are valid codes in every format below (indices, signs, scale nibbles); only the f16 scale fields are set
// to sane values (random bits would give NaN or huge scales). The reference is the CPU decoder, as for real rows.
uint16_t f16_bits(float v) { return __half_as_ushort(__float2half(v)); }

std::vector<uint8_t> synth_rows(uint32_t type, int n_rows, int n_in, uint32_t seed) {
    const limbo::TypeInfo* ti = limbo::type_info(type);
    const size_t rb = static_cast<size_t>(limbo::type_bytes(type, static_cast<uint64_t>(n_in)));
    std::vector<uint8_t> b(rb * static_cast<size_t>(n_rows));
    uint32_t st = seed * 2654435761u + 99u;
    auto rnd = [&]() {
        st = st * 1664525u + 1013904223u;
        return st >> 8;
    };
    for (uint8_t& v : b) v = static_cast<uint8_t>(rnd());
    if (type == static_cast<uint32_t>(Type::F16)) {  // plain halves of a few hundredths, both signs
        for (size_t i = 0; i + 1 < b.size(); i += 2) {
            const uint16_t h = f16_bits((float(rnd() & 0xffff) / 65536.0f - 0.5f) * 0.2f);
            b[i] = static_cast<uint8_t>(h & 0xff);
            b[i + 1] = static_cast<uint8_t>(h >> 8);
        }
        return b;
    }
    auto setf = [&](uint8_t* at, float v) {
        const uint16_t h = f16_bits(v);
        at[0] = static_cast<uint8_t>(h & 0xff);
        at[1] = static_cast<uint8_t>(h >> 8);
    };
    auto sane = [&](float lo, float hi) { return lo + (hi - lo) * (float(rnd() & 0xffff) / 65536.0f); };
    const size_t nblk = b.size() / ti->block_bytes;
    for (size_t i = 0; i < nblk; ++i) {
        uint8_t* p = b.data() + i * ti->block_bytes;
        switch (static_cast<Type>(type)) {
            case Type::IQ3_XXS: case Type::IQ2_XXS: case Type::IQ2_XS: case Type::IQ1_S: case Type::IQ4_NL: case Type::Q2_0:
                setf(p, sane(0.004f, 0.03f));
                break;
            case Type::NVFP4:  // four UE4M3 sub-block scales of about 0.004 .. 0.03 (exponent 1..2)
                for (int k = 0; k < 4; ++k) p[k] = static_cast<uint8_t>(((1 + (rnd() & 1)) << 3) | (rnd() & 7));
                break;
            case Type::Q4_K: case Type::Q5_K:
                setf(p, sane(0.0005f, 0.004f));
                setf(p + 2, sane(0.0005f, 0.004f));
                break;
            case Type::Q2_K:
                setf(p + 80, sane(0.004f, 0.03f));
                setf(p + 82, sane(0.004f, 0.03f));
                break;
            case Type::IQ1_M: {
                const uint16_t h = f16_bits(sane(0.004f, 0.03f));
                for (int k = 0; k < 4; ++k) {  // the four nibbles of the f16 ride in the top nibbles of the scale words
                    const uint16_t w = static_cast<uint16_t>((uint16_t(p[48 + 2 * k]) | (uint16_t(p[49 + 2 * k]) << 8)) & 0x0fff) |
                                       static_cast<uint16_t>(((h >> (4 * k)) & 0xf) << 12);
                    p[48 + 2 * k] = static_cast<uint8_t>(w & 0xff);
                    p[49 + 2 * k] = static_cast<uint8_t>(w >> 8);
                }
                break;
            }
            default: break;
        }
    }
    return b;
}

void test_dense_synth(uint32_t type, const char* name) {
    const int n_in = 512, n_out = 256;
    run_dense(type, name, "random valid blocks", synth_rows(type, n_out, n_in, type), n_in);
}

void test_moe_synth(uint32_t type, const char* name) {
    constexpr int K = 512, n_ff = 256, n_exp = 16, n_embd = 256;
    const uint64_t rb_g = limbo::type_bytes(type, K), rb_d = limbo::type_bytes(type, n_ff);
    const std::vector<uint8_t> g = synth_rows(type, n_ff * n_exp, K, type + 1), u = synth_rows(type, n_ff * n_exp, K, type + 2),
                               d = synth_rows(type, n_embd * n_exp, n_ff, type + 3);
    MoeCase m;
    m.gtype = m.utype = m.dtype = type;
    m.gdata = g.data();
    m.udata = u.data();
    m.ddata = d.data();
    m.gbytes = m.ubytes = rb_g * n_ff * n_exp;
    m.dbytes = rb_d * n_embd * n_exp;
    m.K = K;
    m.n_ff = n_ff;
    m.n_exp = n_exp;
    m.n_embd = n_embd;
    m.label = std::string("MoE ") + name + " (random blocks)";
    run_moe(m);
}

// ---- F32: must equal gemv() bit for bit ------------------------------------------------------------------
void test_f32(const std::vector<std::unique_ptr<limbo::gguf::File>>& files) {
    const limbo::gguf::Tensor* t = nullptr;
    for (const auto& f : files)
        if (!t) t = f->tensor("blk.0.ffn_gate_inp.weight");
    if (!t || t->type != static_cast<uint32_t>(Type::F32)) {
        std::printf("--   no F32 router in the model: F32 product not tested\n");
        return;
    }
    const int n_in = static_cast<int>(t->ne[0]), n_out = static_cast<int>(t->ne[1]), T = 70;
    std::vector<float> x(static_cast<size_t>(T) * n_in);
    for (int i = 0; i < T; ++i) {
        const std::vector<float> xi = make_x(n_in, 4000u + static_cast<uint32_t>(i));
        std::memcpy(x.data() + static_cast<size_t>(i) * n_in, xi.data(), sizeof(float) * static_cast<size_t>(n_in));
    }
    void* dW = nullptr;
    float *dx = nullptr, *dy = nullptr, *dref = nullptr;
    CUDA_OK(cudaMalloc(&dW, t->nbytes));
    CUDA_OK(cudaMalloc(&dx, sizeof(float) * x.size()));
    CUDA_OK(cudaMalloc(&dy, sizeof(float) * static_cast<size_t>(T) * n_out));
    CUDA_OK(cudaMalloc(&dref, sizeof(float) * static_cast<size_t>(T) * n_out));
    CUDA_OK(cudaMemcpy(dW, t->data, t->nbytes, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dx, x.data(), sizeof(float) * x.size(), cudaMemcpyHostToDevice));
    CHECK(cu::gemm_f32(static_cast<const float*>(dW), dx, dy, T, n_out, n_in, nullptr), "gemm_f32 launches");
    for (int i = 0; i < T; ++i)
        CHECK(cu::gemv(static_cast<uint32_t>(Type::F32), dW, dx + static_cast<size_t>(i) * n_in, dref + static_cast<size_t>(i) * n_out, n_out, n_in, false, nullptr),
              "gemv reference launches");
    CUDA_OK(cudaDeviceSynchronize());
    std::vector<float> y(static_cast<size_t>(T) * n_out), r(y.size());
    CUDA_OK(cudaMemcpy(y.data(), dy, sizeof(float) * y.size(), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(r.data(), dref, sizeof(float) * r.size(), cudaMemcpyDeviceToHost));
    size_t diff = 0;
    for (size_t i = 0; i < y.size(); ++i) diff += std::memcmp(&y[i], &r[i], 4) != 0;
    std::printf("     F32 router [%d x %d], %d tokens: %zu of %zu elements differ from gemv()\n", n_out, n_in, T, diff, y.size());
    CHECK(diff == 0, "F32 product equals gemv() bit for bit");
    cudaFree(dW);
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dref);
}

}  // namespace

int main() {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        std::printf("SKIP cuda_gemm_test: no CUDA device\n");
        return 0;
    }
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    if (free_b < (1ull << 30)) {
        std::printf("SKIP cuda_gemm_test: %.0f MB free of %.0f MB, need 1 GB\n", double(free_b) / 1e6, double(total_b) / 1e6);
        return 0;
    }
    const char* env = std::getenv("LIMBO_TEST_GGUF");
    const std::string path = env && *env ? env : kRung1;
    std::vector<std::unique_ptr<limbo::gguf::File>> files;
    std::string err;
    auto f = limbo::gguf::File::open(path, &err);
    if (!f) {
        std::printf("SKIP cuda_gemm_test: cannot open %s: %s\n", path.c_str(), err.c_str());
        return 0;
    }
    files.push_back(std::move(f));

    CHECK(cu::gemm_init(&err), "codebooks reach the device");
    struct Fmt {
        uint32_t type;
        const char* name;
    };
    const Fmt fmts[] = {{static_cast<uint32_t>(Type::Q8_0), "Q8_0"},     {static_cast<uint32_t>(Type::Q6_K), "Q6_K"},
                        {static_cast<uint32_t>(Type::IQ2_S), "IQ2_S"},   {static_cast<uint32_t>(Type::IQ3_S), "IQ3_S"},
                        {static_cast<uint32_t>(Type::IQ4_XS), "IQ4_XS"}};
    std::printf("-- the 1-plane quantiser against Q8_1\n");
    test_q8_1();
    std::printf("-- dense products, real rows, GPU int8 tensor cores vs CPU decode + double dot\n");
    for (const Fmt& fm : fmts) test_dense(files, fm.type, fm.name);
    std::printf("-- one real MoE layer: lists, fused gate/up + SwiGLU, down\n");
    test_moe(files);
    const Fmt ext[] = {{static_cast<uint32_t>(Type::IQ3_XXS), "IQ3_XXS"}, {static_cast<uint32_t>(Type::IQ2_XXS), "IQ2_XXS"},
                       {static_cast<uint32_t>(Type::IQ2_XS), "IQ2_XS"},   {static_cast<uint32_t>(Type::IQ1_S), "IQ1_S"},
                       {static_cast<uint32_t>(Type::IQ1_M), "IQ1_M"},     {static_cast<uint32_t>(Type::IQ4_NL), "IQ4_NL"},
                       {static_cast<uint32_t>(Type::Q2_0), "Q2_0"},       {static_cast<uint32_t>(Type::Q4_K), "Q4_K"},
                       {static_cast<uint32_t>(Type::Q5_K), "Q5_K"},       {static_cast<uint32_t>(Type::Q2_K), "Q2_K"},
                       {static_cast<uint32_t>(Type::NVFP4), "NVFP4"}};
    std::printf("-- more formats (random valid blocks): dense products and MoE tiles against the CPU decoder\n");
    for (const Fmt& fm : ext) {
        test_dense_synth(fm.type, fm.name);
        test_moe_synth(fm.type, fm.name);
    }
    test_dense_synth(static_cast<uint32_t>(Type::F16), "F16");  // dense only: no expert stacks in F16
    std::printf("-- F32 products\n");
    test_f32(files);
    return limbo::test::finish("cuda_gemm_test");
}
