// Copyright (c) 2026 ITERFORM. All rights reserved.
// tests/cuda_gemv_test.cu — the device GEMV against the CPU decoder, on real rows of the rung-1 file.
//
// For each of the nine formats rung 1 carries (F32, BF16, Q8_0, Q2_K, Q3_K, Q6_K, IQ2_S, IQ3_S, IQ4_XS)
// the test takes real rows of a real tensor, runs y = W x on the GPU with W in its GGUF block layout, and
// compares with src/quants.cpp's decoder plus a double-precision dot. The bar is the PoC's:
//
//     |y_gpu - y_cpu| <= 1e-5 * sum_i |w_i * x_i|     per row
//
// then the same for the MoE form (ids from device memory, shared and per-expert inputs), the
// accumulate form, and the embedding row. The model is LIMBO_TEST_GGUF (':' or ';' separated), or the
// rung-1 file in the Hugging Face cache. No GPU, or under 1 GB free beside the desktop: SKIP, exit 0.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "limbo/check.hpp"
#include "../src/cuda/dec_ops.hpp"
#include "limbo/cuda/gemv.hpp"
#include "limbo/cuda/gemm.hpp"
#include "limbo/gguf.hpp"
#include "limbo/quants.hpp"
#include "limbo/types.hpp"

namespace {

using limbo::Type;
namespace cu = limbo::cuda;

constexpr const char* kRung1 =
    "/home/alca/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-MTP-GGUF/snapshots/"
    "5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf";

std::vector<std::string> model_paths() {
    std::vector<std::string> out;
    const char* env = std::getenv("LIMBO_TEST_GGUF");
    std::string s = env && *env ? env : "";
    size_t at = 0;
    while (at < s.size()) {
        size_t p = s.find_first_of(":;", at);
        if (p == std::string::npos) p = s.size();
        if (p > at) out.push_back(s.substr(at, p - at));
        at = p + 1;
    }
    if (out.empty()) out.push_back(kRung1);
    return out;
}

#define CUDA_OK(call)                                                                                  \
    do {                                                                                               \
        cudaError_t e_ = (call);                                                                       \
        if (e_ != cudaSuccess) {                                                                       \
            std::printf("FAIL %s: %s (%s:%d)\n", #call, cudaGetErrorString(e_), __FILE__, __LINE__);    \
            ++limbo::test::failures();                                                                 \
            return;                                                                                    \
        }                                                                                              \
    } while (0)

// A deterministic input with a realistic spread: mostly unit-scale, a few large outliers.
std::vector<float> make_x(int n, uint32_t seed) {
    std::vector<float> x(static_cast<size_t>(n));
    uint32_t s = seed * 2654435761u + 12345u;
    for (int i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        const float u = static_cast<float>(s >> 8) / 16777216.0f;  // [0,1)
        s = s * 1664525u + 1013904223u;
        const float v = static_cast<float>(s >> 8) / 16777216.0f;
        float g = (u + v - 1.0f) * 2.4f;  // triangular, sd ~ 1
        if (i % 97 == 0) g *= 8.0f;
        x[static_cast<size_t>(i)] = g;
    }
    return x;
}

struct Err {
    double max_rel = 0, sum_rel = 0;
    int rows = 0;
    void add(double gpu, double cpu, double l1) {
        const double rel = l1 > 0 ? std::fabs(gpu - cpu) / l1 : std::fabs(gpu - cpu);
        max_rel = std::max(max_rel, rel);
        sum_rel += rel;
        ++rows;
    }
};

struct Found {
    const limbo::gguf::Tensor* t = nullptr;
    const limbo::gguf::File* f = nullptr;
};

// First tensor of `type` with the shape the test wants: a 2-D matrix (rows >= 64), or an expert stack. The
// plain test falls back to an expert stack read as one tall matrix when the format only occurs in stacks
// (Q2_K, Q3_K, IQ2_S, IQ3_S and IQ4_XS do, in this file): its rows are real rows all the same.
Found find_tensor(const std::vector<std::unique_ptr<limbo::gguf::File>>& files, uint32_t type, bool experts,
                  bool allow_stack = false) {
    for (int pass = 0; pass < 2; ++pass) {
        if (pass == 1 && !allow_stack) break;
        for (const auto& f : files)
            for (const auto& t : f->tensors()) {
                if (t.type != type || t.ne[0] % 256 != 0) continue;
                if (experts ? (t.ne[2] > 8 && t.ne[3] == 1)
                            : (pass == 0 ? (t.ne[2] == 1 && t.ne[3] == 1 && t.ne[1] >= 64) : (t.ne[2] > 8 && t.ne[3] == 1)))
                    return {&t, f.get()};
            }
    }
    return {};
}

double cpu_row_dot(uint32_t type, const uint8_t* row, const float* x, int n, double* l1) {
    std::vector<float> w(static_cast<size_t>(n));
    limbo::quants::dequant_row(type, row, w.data(), static_cast<uint64_t>(n));
    double dot = 0, a = 0;
    for (int i = 0; i < n; ++i) {
        const double p = double(w[static_cast<size_t>(i)]) * double(x[i]);
        dot += p;
        a += std::fabs(p);
    }
    *l1 = a;
    return dot;
}

// ---- plain GEMV -------------------------------------------------------------------------------------
void test_gemv(const std::vector<std::unique_ptr<limbo::gguf::File>>& files, uint32_t type, const char* name) {
    const Found fd = find_tensor(files, type, false, true);
    if (!fd.t) {
        std::printf("--   %-7s no tensor of this type in the model: not tested\n", name);
        return;
    }
    const limbo::gguf::Tensor& t = *fd.t;
    const int n_in = static_cast<int>(t.ne[0]);
    const int total = static_cast<int>(t.ne[1] * t.ne[2] * t.ne[3]);
    const int n_rows = std::min(total, 512);
    const uint64_t rb = limbo::type_bytes(type, static_cast<uint64_t>(n_in));

    // Rows spread over the whole tensor, the last one included.
    std::vector<int> rows(static_cast<size_t>(n_rows));
    for (int k = 0; k < n_rows; ++k)
        rows[static_cast<size_t>(k)] = n_rows == 1 ? 0 : static_cast<int>(static_cast<int64_t>(k) * (total - 1) / (n_rows - 1));
    std::vector<uint8_t> host(static_cast<size_t>(n_rows) * rb);
    for (int k = 0; k < n_rows; ++k)
        std::memcpy(host.data() + static_cast<size_t>(k) * rb, t.data + static_cast<uint64_t>(rows[static_cast<size_t>(k)]) * rb, rb);
    const std::vector<float> x = make_x(n_in, type);

    void* dW = nullptr;
    float *dx = nullptr, *dy = nullptr;
    CUDA_OK(cudaMalloc(&dW, host.size()));
    CUDA_OK(cudaMalloc(&dx, sizeof(float) * static_cast<size_t>(n_in)));
    CUDA_OK(cudaMalloc(&dy, sizeof(float) * static_cast<size_t>(n_rows)));
    CUDA_OK(cudaMemcpy(dW, host.data(), host.size(), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dx, x.data(), sizeof(float) * x.size(), cudaMemcpyHostToDevice));
    CHECK(cu::gemv(type, dW, dx, dy, n_rows, n_in, false, nullptr), "gemv launches");
    CUDA_OK(cudaDeviceSynchronize());
    std::vector<float> y(static_cast<size_t>(n_rows));
    CUDA_OK(cudaMemcpy(y.data(), dy, sizeof(float) * y.size(), cudaMemcpyDeviceToHost));

    Err err;
    std::vector<double> cpu(static_cast<size_t>(n_rows)), l1s(static_cast<size_t>(n_rows));
    for (int k = 0; k < n_rows; ++k) {
        cpu[static_cast<size_t>(k)] = cpu_row_dot(type, host.data() + static_cast<size_t>(k) * rb, x.data(), n_in, &l1s[static_cast<size_t>(k)]);
        err.add(y[static_cast<size_t>(k)], cpu[static_cast<size_t>(k)], l1s[static_cast<size_t>(k)]);
    }

    // The accumulate form: y starts at a known value and the product is added to it.
    std::vector<float> y0(static_cast<size_t>(n_rows));
    for (int k = 0; k < n_rows; ++k) y0[static_cast<size_t>(k)] = 0.5f * static_cast<float>(k % 7) - 1.0f;
    CUDA_OK(cudaMemcpy(dy, y0.data(), sizeof(float) * y0.size(), cudaMemcpyHostToDevice));
    CHECK(cu::gemv(type, dW, dx, dy, n_rows, n_in, true, nullptr), "gemv accumulate launches");
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(y.data(), dy, sizeof(float) * y.size(), cudaMemcpyDeviceToHost));
    Err acc_err;
    for (int k = 0; k < n_rows; ++k)
        acc_err.add(y[static_cast<size_t>(k)], cpu[static_cast<size_t>(k)] + double(y0[static_cast<size_t>(k)]),
                    l1s[static_cast<size_t>(k)] + std::fabs(double(y0[static_cast<size_t>(k)])));

    cudaFree(dW);
    cudaFree(dx);
    cudaFree(dy);
    std::printf("     %-7s %-34s n_in %5d  rows %3d  max rel err %.3e  mean %.3e  accumulate max %.3e\n", name,
                std::string(t.name).c_str(), n_in, err.rows, err.max_rel, err.sum_rel / err.rows, acc_err.max_rel);
    char what[96];
    std::snprintf(what, sizeof what, "%s: gemv within 1e-5 of the row's sum|w*x|", name);
    CHECK(err.max_rel <= 1e-5, what);
    std::snprintf(what, sizeof what, "%s: accumulate form within 1e-5", name);
    CHECK(acc_err.max_rel <= 1e-5, what);
}

// ---- the MoE form -----------------------------------------------------------------------------------
void test_experts(const std::vector<std::unique_ptr<limbo::gguf::File>>& files, uint32_t type, const char* name) {
    const Found fd = find_tensor(files, type, true);
    if (!fd.t) {
        std::printf("--   %-7s no expert stack of this type in the model: MoE form not tested\n", name);
        return;
    }
    const limbo::gguf::Tensor& t = *fd.t;
    const int n_in = static_cast<int>(t.ne[0]), n_out = static_cast<int>(t.ne[1]), n_exp = static_cast<int>(t.ne[2]);
    const uint64_t rb = limbo::type_bytes(type, static_cast<uint64_t>(n_in));
    const uint64_t ebytes = rb * static_cast<uint64_t>(n_out);
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    if (free_b < t.nbytes + (1ull << 30)) {
        std::printf("SKIP %-7s MoE form: %.0f MB free, the stack needs %.0f MB + 1 GB\n", name, double(free_b) / 1e6,
                    double(t.nbytes) / 1e6);
        return;
    }
    const int n_sel = 8;
    // Ids that include the first and last expert and an out-of-order spread; a repeated id would be legal too.
    const int32_t ids_h[8] = {n_exp - 1, 0, 17, 101, 3, n_exp / 2, n_exp - 2, 64};
    const std::vector<float> x = make_x(n_in * n_sel, type + 100);

    void* dW = nullptr;
    int32_t* dids = nullptr;
    float *dx = nullptr, *dy = nullptr;
    CUDA_OK(cudaMalloc(&dW, t.nbytes));
    CUDA_OK(cudaMalloc(&dids, sizeof ids_h));
    CUDA_OK(cudaMalloc(&dx, sizeof(float) * x.size()));
    CUDA_OK(cudaMalloc(&dy, sizeof(float) * static_cast<size_t>(n_sel) * static_cast<size_t>(n_out)));
    CUDA_OK(cudaMemcpy(dW, t.data, t.nbytes, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dids, ids_h, sizeof ids_h, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dx, x.data(), sizeof(float) * x.size(), cudaMemcpyHostToDevice));

    Err shared_err, own_err;
    for (int mode = 0; mode < 2; ++mode) {  // 0: one input shared by all slots, 1: an input per slot
        CHECK(cu::gemv_experts(type, dW, ebytes, dids, n_sel, dx, mode == 0 ? 0 : n_in, dy, n_out, n_in, nullptr),
              "gemv_experts launches");
        CUDA_OK(cudaDeviceSynchronize());
        std::vector<float> y(static_cast<size_t>(n_sel) * static_cast<size_t>(n_out));
        CUDA_OK(cudaMemcpy(y.data(), dy, sizeof(float) * y.size(), cudaMemcpyDeviceToHost));
        for (int e = 0; e < n_sel; ++e) {
            const float* xe = x.data() + (mode == 0 ? 0 : static_cast<size_t>(e) * static_cast<size_t>(n_in));
            const int step = std::max(1, n_out / 48);  // 48 rows per expert, the last one included
            for (int r = 0; r < n_out; r += step) {
                const int rr = (r + step >= n_out) ? n_out - 1 : r;
                const uint8_t* row = t.data + static_cast<uint64_t>(ids_h[e]) * ebytes + static_cast<uint64_t>(rr) * rb;
                double l1;
                const double cpu = cpu_row_dot(type, row, xe, n_in, &l1);
                (mode == 0 ? shared_err : own_err)
                    .add(y[static_cast<size_t>(e) * static_cast<size_t>(n_out) + static_cast<size_t>(rr)], cpu, l1);
            }
        }
    }
    cudaFree(dW);
    cudaFree(dids);
    cudaFree(dx);
    cudaFree(dy);
    std::printf("     %-7s %-34s [%d,%d,%d]  8 experts  shared-x max %.3e (%d rows)  per-expert-x max %.3e (%d rows)\n",
                name, std::string(t.name).c_str(), n_in, n_out, n_exp, shared_err.max_rel, shared_err.rows,
                own_err.max_rel, own_err.rows);
    char what[96];
    std::snprintf(what, sizeof what, "%s: MoE form, shared input, within 1e-5", name);
    CHECK(shared_err.max_rel <= 1e-5, what);
    std::snprintf(what, sizeof what, "%s: MoE form, per-expert input, within 1e-5", name);
    CHECK(own_err.max_rel <= 1e-5, what);
}


// ---- the fused decode forms: dual products and the SwiGLU-input product ---------------------------------------
void test_fused(const std::vector<std::unique_ptr<limbo::gguf::File>>& files) {
    const limbo::gguf::Tensor *tq = nullptr, *tsd = nullptr, *tg = nullptr, *tu = nullptr;
    for (const auto& f : files) {
        if (!tq) tq = f->tensor("blk.0.attn_qkv.weight");
        if (!tsd) tsd = f->tensor("blk.0.ffn_down_shexp.weight");
        if (!tg) tg = f->tensor("blk.0.ffn_gate_exps.weight");
        if (!tu) tu = f->tensor("blk.0.ffn_up_exps.weight");
    }
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    // plain dual: two halves of one matrix as the two operands, against two gemv() calls, bit for bit
    if (tq && cu::gemv_supported(tq->type) && free_b > (2ull << 30)) {
        const int n_in = static_cast<int>(tq->ne[0]), n_out = 512;
        const uint64_t rb = limbo::type_bytes(tq->type, static_cast<uint64_t>(n_in));
        const std::vector<float> x = make_x(n_in, 77);
        void* dW = nullptr;
        float *dx = nullptr, *dy = nullptr;
        CUDA_OK(cudaMalloc(&dW, 2 * rb * n_out));
        CUDA_OK(cudaMalloc(&dx, sizeof(float) * x.size()));
        CUDA_OK(cudaMalloc(&dy, sizeof(float) * 4 * n_out));
        CUDA_OK(cudaMemcpy(dW, tq->data, 2 * rb * n_out, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(dx, x.data(), sizeof(float) * x.size(), cudaMemcpyHostToDevice));
        const uint8_t* w0 = static_cast<const uint8_t*>(dW);
        const uint8_t* w1 = w0 + rb * n_out;
        CHECK(cu::gemv_dual(tq->type, w0, w1, dx, dy, dy + n_out, n_out, n_in, nullptr), "gemv_dual launches");
        CHECK(cu::gemv(tq->type, w0, dx, dy + 2 * n_out, n_out, n_in, false, nullptr), "gemv (reference, first) launches");
        CHECK(cu::gemv(tq->type, w1, dx, dy + 3 * n_out, n_out, n_in, false, nullptr), "gemv (reference, second) launches");
        CUDA_OK(cudaDeviceSynchronize());
        std::vector<float> y(4 * static_cast<size_t>(n_out));
        CUDA_OK(cudaMemcpy(y.data(), dy, sizeof(float) * y.size(), cudaMemcpyDeviceToHost));
        const bool same = std::memcmp(y.data(), y.data() + 2 * n_out, sizeof(float) * 2 * n_out) == 0;
        std::printf("     dual product %s [%d x %d]: %s\n", std::string(tq->name).c_str(), n_out, n_in, same ? "bit-identical to two gemv() calls" : "DIFFERS");
        CHECK(same, "gemv_dual equals two gemv() calls bit for bit");
        cudaFree(dW);
        cudaFree(dx);
        cudaFree(dy);
    }
    // SwiGLU input: y = W silu(g) * u, against the CPU decoder and a double dot
    if (tsd && cu::gemv_supported(tsd->type) && tsd->ne[0] <= 512 && free_b > (2ull << 30)) {
        const int n_in = static_cast<int>(tsd->ne[0]), n_out = static_cast<int>(tsd->ne[1]);
        const uint64_t rb = limbo::type_bytes(tsd->type, static_cast<uint64_t>(n_in));
        const std::vector<float> g = make_x(n_in, 5), u = make_x(n_in, 6);
        std::vector<float> h(g.size());
        for (size_t i = 0; i < g.size(); ++i) h[i] = g[i] / (1.0f + std::exp(-g[i])) * u[i];
        void* dW = nullptr;
        float *dg = nullptr, *du = nullptr, *dy = nullptr;
        CUDA_OK(cudaMalloc(&dW, tsd->nbytes));
        CUDA_OK(cudaMalloc(&dg, sizeof(float) * g.size()));
        CUDA_OK(cudaMalloc(&du, sizeof(float) * u.size()));
        CUDA_OK(cudaMalloc(&dy, sizeof(float) * static_cast<size_t>(n_out)));
        CUDA_OK(cudaMemcpy(dW, tsd->data, tsd->nbytes, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(dg, g.data(), sizeof(float) * g.size(), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(du, u.data(), sizeof(float) * u.size(), cudaMemcpyHostToDevice));
        CHECK(cu::gemv_swiglu(tsd->type, dW, dg, du, dy, n_out, n_in, false, nullptr), "gemv_swiglu launches");
        CUDA_OK(cudaDeviceSynchronize());
        std::vector<float> y(static_cast<size_t>(n_out));
        CUDA_OK(cudaMemcpy(y.data(), dy, sizeof(float) * y.size(), cudaMemcpyDeviceToHost));
        Err e;
        for (int r = 0; r < n_out; r += 7) {
            double l1;
            const double cpu = cpu_row_dot(tsd->type, tsd->data + static_cast<uint64_t>(r) * rb, h.data(), n_in, &l1);
            e.add(y[static_cast<size_t>(r)], cpu, l1);
        }
        std::printf("     gemv_swiglu %s [%d x %d]: max rel err %.3e (%d rows)\n", std::string(tsd->name).c_str(), n_out, n_in, e.max_rel, e.rows);
        CHECK(e.max_rel <= 1e-5, "gemv_swiglu within 1e-5 of the CPU decoder");
        cudaFree(dW);
        cudaFree(dg);
        cudaFree(du);
        cudaFree(dy);
    }
    // MoE dual: the gate and up stacks of layer 0 against gemv_experts() on each
    if (tg && tu && tg->type == tu->type && cu::gemv_supported(tg->type) && free_b > tg->nbytes * 2 + (1ull << 30)) {
        const int n_in = static_cast<int>(tg->ne[0]), n_out = static_cast<int>(tg->ne[1]), n_exp = static_cast<int>(tg->ne[2]);
        const uint64_t eb = limbo::type_bytes(tg->type, static_cast<uint64_t>(n_in)) * static_cast<uint64_t>(n_out);
        const int32_t ids_h[8] = {n_exp - 1, 0, 17, 101, 3, n_exp / 2, n_exp - 2, 64};
        const std::vector<float> x = make_x(n_in, 31);
        void *dg = nullptr, *du = nullptr;
        int32_t* dids = nullptr;
        float *dx = nullptr, *dy = nullptr;
        CUDA_OK(cudaMalloc(&dg, tg->nbytes));
        CUDA_OK(cudaMalloc(&du, tu->nbytes));
        CUDA_OK(cudaMalloc(&dids, sizeof ids_h));
        CUDA_OK(cudaMalloc(&dx, sizeof(float) * x.size()));
        CUDA_OK(cudaMalloc(&dy, sizeof(float) * 4 * 8 * static_cast<size_t>(n_out)));
        CUDA_OK(cudaMemcpy(dg, tg->data, tg->nbytes, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(du, tu->data, tu->nbytes, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(dids, ids_h, sizeof ids_h, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(dx, x.data(), sizeof(float) * x.size(), cudaMemcpyHostToDevice));
        const size_t blk = 8 * static_cast<size_t>(n_out);
        CHECK(cu::gemv_experts_dual(tg->type, dg, du, eb, dids, 8, dx, dy, dy + blk, n_out, n_in, nullptr), "gemv_experts_dual launches");
        CHECK(cu::gemv_experts(tg->type, dg, eb, dids, 8, dx, 0, dy + 2 * blk, n_out, n_in, nullptr), "gemv_experts (reference, gate) launches");
        CHECK(cu::gemv_experts(tu->type, du, eb, dids, 8, dx, 0, dy + 3 * blk, n_out, n_in, nullptr), "gemv_experts (reference, up) launches");
        CUDA_OK(cudaDeviceSynchronize());
        std::vector<float> y(4 * blk);
        CUDA_OK(cudaMemcpy(y.data(), dy, sizeof(float) * y.size(), cudaMemcpyDeviceToHost));
        const bool same = std::memcmp(y.data(), y.data() + 2 * blk, sizeof(float) * 2 * blk) == 0;
        std::printf("     expert dual product %s [%d,%d,%d], 8 experts: %s\n", std::string(tg->name).c_str(), n_in, n_out, n_exp, same ? "bit-identical to two gemv_experts() calls" : "DIFFERS");
        CHECK(same, "gemv_experts_dual equals two gemv_experts() calls bit for bit");
        cudaFree(dg);
        cudaFree(du);
        cudaFree(dids);
        cudaFree(dx);
        cudaFree(dy);
    }
}

// ---- the embedding row ------------------------------------------------------------------------------
void test_embed(const std::vector<std::unique_ptr<limbo::gguf::File>>& files) {
    const limbo::gguf::Tensor* t = nullptr;
    for (const auto& f : files) {
        t = f->tensor("token_embd.weight");
        if (t) break;
    }
    if (!t || !cu::gemv_supported(t->type)) {
        std::printf("--   no supported token_embd.weight: embedding row not tested\n");
        return;
    }
    const int n = static_cast<int>(t->ne[0]);
    const uint64_t rb = limbo::type_bytes(t->type, static_cast<uint64_t>(n));
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    if (free_b < (1ull << 30)) return;
    const int ids[3] = {0, 5834, static_cast<int>(t->ne[1]) - 1};
    // Only the three rows go to the device: the kernel indexes by token, so copy a table of 3 rows and use ids 0..2.
    std::vector<uint8_t> host(3 * rb);
    for (int k = 0; k < 3; ++k) std::memcpy(host.data() + static_cast<size_t>(k) * rb, t->data + static_cast<uint64_t>(ids[k]) * rb, rb);
    void* dW = nullptr;
    int32_t* dtok = nullptr;
    float* dout = nullptr;
    CUDA_OK(cudaMalloc(&dW, host.size()));
    CUDA_OK(cudaMalloc(&dtok, 4));
    CUDA_OK(cudaMalloc(&dout, sizeof(float) * static_cast<size_t>(n)));
    CUDA_OK(cudaMemcpy(dW, host.data(), host.size(), cudaMemcpyHostToDevice));
    double worst = 0;
    for (int k = 0; k < 3; ++k) {
        const int32_t tok = k;
        CUDA_OK(cudaMemcpy(dtok, &tok, 4, cudaMemcpyHostToDevice));
        CHECK(cu::embed_row(t->type, dW, dtok, dout, n, nullptr), "embed_row launches");
        CUDA_OK(cudaDeviceSynchronize());
        std::vector<float> g(static_cast<size_t>(n)), c(static_cast<size_t>(n));
        CUDA_OK(cudaMemcpy(g.data(), dout, sizeof(float) * g.size(), cudaMemcpyDeviceToHost));
        limbo::quants::dequant_row(t->type, host.data() + static_cast<size_t>(k) * rb, c.data(), static_cast<uint64_t>(n));
        double scale = 0;
        for (float v : c) scale = std::max(scale, double(std::fabs(v)));
        for (int i = 0; i < n; ++i) worst = std::max(worst, std::fabs(double(g[static_cast<size_t>(i)]) - double(c[static_cast<size_t>(i)])) / std::max(scale, 1e-30));
    }
    cudaFree(dW);
    cudaFree(dtok);
    cudaFree(dout);
    std::printf("     token_embd row decode (3 rows): max |gpu - cpu| / max|w| = %.3e\n", worst);
    CHECK(worst <= 1e-6, "embedding row decode matches the CPU decoder");
}


// ---- the decode kernels of src/cuda/gemv_q6k.cu, gemv_exp.cu, dec_small.cu -----------------------------------
const limbo::gguf::Tensor* tensor_named(const std::vector<std::unique_ptr<limbo::gguf::File>>& files, const std::string& name) {
    for (const auto& f : files)
        for (const auto& t : f->tensors())
            if (std::string(t.name) == name) return &t;
    return nullptr;
}

// C columns at once equal the single-column product bit for bit (the weights are decoded once for all of them).
void test_cols(const std::vector<std::unique_ptr<limbo::gguf::File>>& files, uint32_t type, const char* name) {
    const Found fd = find_tensor(files, type, false, true);
    if (!fd.t) return;
    const limbo::gguf::Tensor& t = *fd.t;
    const int n_in = static_cast<int>(t.ne[0]);
    const int total = static_cast<int>(t.ne[1] * t.ne[2] * t.ne[3]);
    const int n_rows = std::min(total, 600);  // not a multiple of the row blocks of any kernel
    const uint64_t rb = limbo::type_bytes(type, static_cast<uint64_t>(n_in));
    constexpr int C = 8;
    std::vector<float> x(static_cast<size_t>(C) * static_cast<size_t>(n_in));
    for (int c = 0; c < C; ++c) {
        const std::vector<float> xc = make_x(n_in, 100u + static_cast<uint32_t>(c));
        std::copy(xc.begin(), xc.end(), x.begin() + static_cast<size_t>(c) * static_cast<size_t>(n_in));
    }
    void* dW = nullptr;
    float *dx = nullptr, *dy = nullptr, *d1 = nullptr;
    CUDA_OK(cudaMalloc(&dW, static_cast<size_t>(n_rows) * rb));
    CUDA_OK(cudaMalloc(&dx, sizeof(float) * x.size()));
    CUDA_OK(cudaMalloc(&dy, sizeof(float) * static_cast<size_t>(C) * static_cast<size_t>(n_rows)));
    CUDA_OK(cudaMalloc(&d1, sizeof(float) * static_cast<size_t>(n_rows)));
    CUDA_OK(cudaMemcpy(dW, t.data, static_cast<size_t>(n_rows) * rb, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dx, x.data(), sizeof(float) * x.size(), cudaMemcpyHostToDevice));
    bool same = true, ok = true;
    for (int cols : {2, 5, 8}) {
        ok = ok && cu::gemv_cols(type, dW, dx, n_in, dy, n_rows, n_rows, n_in, cols, false, nullptr);
        CUDA_OK(cudaDeviceSynchronize());
        std::vector<float> yy(static_cast<size_t>(cols) * static_cast<size_t>(n_rows)), y1(static_cast<size_t>(n_rows));
        CUDA_OK(cudaMemcpy(yy.data(), dy, sizeof(float) * yy.size(), cudaMemcpyDeviceToHost));
        for (int c = 0; c < cols; ++c) {
            ok = ok && cu::gemv(type, dW, dx + static_cast<size_t>(c) * static_cast<size_t>(n_in), d1, n_rows, n_in, false, nullptr);
            CUDA_OK(cudaDeviceSynchronize());
            CUDA_OK(cudaMemcpy(y1.data(), d1, sizeof(float) * y1.size(), cudaMemcpyDeviceToHost));
            same = same && std::memcmp(y1.data(), &yy[static_cast<size_t>(c) * static_cast<size_t>(n_rows)], sizeof(float) * y1.size()) == 0;
        }
    }
    cudaFree(dW);
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(d1);
    char what[120];
    std::snprintf(what, sizeof what, "%s: gemv_cols with 2, 5 and 8 columns equals the single-column gemv bit for bit", name);
    CHECK(ok && same, what);
}

// fragment-kernel windows: within the activation rounding of the CPU dot, the same bits at every column count
void check_windows(uint32_t type, const char* name, const uint8_t* rows, uint64_t rb, int n_rows, int n_in) {
    constexpr int C = 16;
    std::vector<float> x(static_cast<size_t>(C) * static_cast<size_t>(n_in));
    for (int c = 0; c < C; ++c) {
        const std::vector<float> xc = make_x(n_in, 300u + static_cast<uint32_t>(c));
        std::copy(xc.begin(), xc.end(), x.begin() + static_cast<size_t>(c) * static_cast<size_t>(n_in));
    }
    std::vector<double> ref(static_cast<size_t>(C) * n_rows), l1(static_cast<size_t>(C) * n_rows);
    for (int c = 0; c < C; ++c)
        for (int r = 0; r < n_rows; ++r)
            ref[static_cast<size_t>(c) * n_rows + r] =
                cpu_row_dot(type, rows + static_cast<size_t>(r) * rb, &x[static_cast<size_t>(c) * n_in], n_in, &l1[static_cast<size_t>(c) * n_rows + r]);
    void* dW = nullptr;
    float *dx = nullptr, *dy = nullptr;
    CUDA_OK(cudaMalloc(&dW, static_cast<size_t>(n_rows) * rb));
    CUDA_OK(cudaMalloc(&dx, sizeof(float) * x.size()));
    CUDA_OK(cudaMalloc(&dy, sizeof(float) * static_cast<size_t>(C) * n_rows));
    CUDA_OK(cudaMemcpy(dW, rows, static_cast<size_t>(n_rows) * rb, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dx, x.data(), sizeof(float) * x.size(), cudaMemcpyHostToDevice));
    double max_err = 0;
    bool ok = true, same = true;
    for (const bool accum : {false, true}) {
        std::vector<float> full;
        for (const int cols : {16, 1, 2, 3, 5, 8}) {
            std::vector<float> yy(static_cast<size_t>(C) * n_rows, 0.75f);
            CUDA_OK(cudaMemcpy(dy, yy.data(), sizeof(float) * yy.size(), cudaMemcpyHostToDevice));
            const bool launched = cu::gemv_cols(type, dW, dx, n_in, dy, n_rows, n_rows, n_in, cols, accum, nullptr);
            ok = ok && launched;
            CUDA_OK(cudaDeviceSynchronize());
            CUDA_OK(cudaMemcpy(yy.data(), dy, sizeof(float) * yy.size(), cudaMemcpyDeviceToHost));
            for (int c = 0; c < cols; ++c)
                for (int r = 0; r < n_rows; ++r) {
                    const size_t i = static_cast<size_t>(c) * n_rows + r;
                    const double got = double(yy[i]) - (accum ? 0.75 : 0.0);
                    max_err = std::max(max_err, std::fabs(got - ref[i]) / std::max(l1[i], 1.0));
                }
            if (cols == 16) full = yy;
            else same = same && std::memcmp(full.data(), yy.data(), sizeof(float) * static_cast<size_t>(cols) * n_rows) == 0;
        }
    }
    cudaFree(dW);
    cudaFree(dx);
    cudaFree(dy);
    std::printf("     %-6s window rows %4d x %5d: max err %.3e of the row's sum|w*x| (1..16 columns)\n", name, n_rows, n_in, max_err);
    char what[160];
    std::snprintf(what, sizeof what, "%s %d x %d: every window column within 2e-4 of the row's sum|w*x|", name, n_rows, n_in);
    CHECK(ok && max_err <= 2e-4, what);
    std::snprintf(what, sizeof what, "%s %d x %d: a column's bits do not depend on the window's size", name, n_rows, n_in);
    CHECK(same, what);
}

void test_frag_windows(const std::vector<std::unique_ptr<limbo::gguf::File>>& files) {
    std::string err;
    CHECK(cu::gemm_init(&err), "the int8 tables reach the device");
    CHECK(cu::frag_reserve(nullptr, &err), "the default stream gets window scratch");
    const uint32_t types[] = {static_cast<uint32_t>(Type::IQ3_S),   static_cast<uint32_t>(Type::Q6_K),    static_cast<uint32_t>(Type::IQ2_S),
                              static_cast<uint32_t>(Type::IQ3_XXS), static_cast<uint32_t>(Type::IQ2_XXS), static_cast<uint32_t>(Type::IQ2_XS),
                              static_cast<uint32_t>(Type::Q2_K),    static_cast<uint32_t>(Type::Q3_K),    static_cast<uint32_t>(Type::Q8_0),
                              static_cast<uint32_t>(Type::IQ4_XS)};
    const char* names[] = {"IQ3_S", "Q6_K", "IQ2_S", "IQ3_XXS", "IQ2_XXS", "IQ2_XS", "Q2_K", "Q3_K", "Q8_0", "IQ4_XS"};
    for (int ti = 0; ti < 10; ++ti) {
        const Found fd = find_tensor(files, types[ti], false, true);
        if (!fd.t) continue;
        const limbo::gguf::Tensor& t = *fd.t;
        const int n_in = static_cast<int>(t.ne[0]);
        const int total = static_cast<int>(t.ne[1] * t.ne[2] * t.ne[3]);
        const uint64_t rb = limbo::type_bytes(types[ti], static_cast<uint64_t>(n_in));
        for (const int n_rows : {std::min(total, 600), std::min(total, 4000)})
            check_windows(types[ti], names[ti], static_cast<const uint8_t*>(t.data), rb, n_rows, n_in);
    }
    // synthetic blocks, 5,120 wide (K cut into slices): the K-quants with a minimum and the i-quants the models lack
    for (const Type kt : {Type::Q4_K, Type::Q5_K, Type::IQ3_XXS, Type::IQ2_XXS, Type::IQ2_XS}) {
        const uint32_t type = static_cast<uint32_t>(kt);
        constexpr int NIN = 5120, NROWS = 4000;
        const uint64_t rb = limbo::type_bytes(type, NIN);
        std::vector<uint8_t> host(static_cast<size_t>(NROWS) * rb);
        uint32_t st = 0x9e3779b9u ^ type;
        for (uint8_t& b : host) {
            st = st * 1664525u + 1013904223u;
            b = static_cast<uint8_t>(st >> 24);
        }
        const uint64_t sbb = limbo::type_bytes(type, 256);
        const bool kq = kt == Type::Q4_K || kt == Type::Q5_K;
        for (size_t b = 0; b < host.size() / sbb; ++b) {   // d 0.125 (and the K-quants' dmin 0.03125); every other byte decodes
            uint8_t* p = host.data() + b * sbb;
            p[0] = 0x00;
            p[1] = 0x30;
            if (kq) {
                p[2] = 0x00;
                p[3] = 0x28;
            }
        }
        const char* nm = kt == Type::Q4_K ? "Q4_K" : kt == Type::Q5_K ? "Q5_K" : kt == Type::IQ3_XXS ? "IQ3_XXS" : kt == Type::IQ2_XXS ? "IQ2_XXS" : "IQ2_XS";
        check_windows(type, nm, host.data(), rb, 600, NIN);
        check_windows(type, nm, host.data(), rb, NROWS, NIN);
    }
    cu::frag_release(nullptr);
}

void test_decode_fast(const std::vector<std::unique_ptr<limbo::gguf::File>>& files) {
    // IQ2_S gate/up with SwiGLU against the generic pair, and IQ3_S down against the generic form, with the slots of a
    // verify window (two rows, each with its own input and its own experts)
    const Found g = find_tensor(files, static_cast<uint32_t>(Type::IQ2_S), true);
    const Found d = find_tensor(files, static_cast<uint32_t>(Type::IQ3_S), true);
    if (g.t) {
        const std::string gname(g.t->name);
        const size_t at = gname.find("gate");
        const limbo::gguf::Tensor* u = at == std::string::npos ? nullptr : tensor_named(files, gname.substr(0, at) + "up" + gname.substr(at + 4));
        if (u && u->type == g.t->type) {
            const int n_in = static_cast<int>(g.t->ne[0]), n_out = static_cast<int>(g.t->ne[1]), n_exp = static_cast<int>(g.t->ne[2]);
            const uint64_t eb = limbo::type_bytes(g.t->type, static_cast<uint64_t>(n_in)) * static_cast<uint64_t>(n_out);
            constexpr int ROWS = 2, K = 8, N = ROWS * K;
            std::vector<int32_t> ids(N);
            for (int i = 0; i < N; ++i) ids[static_cast<size_t>(i)] = (i * 29 + 7 * (i / K)) % n_exp;
            std::vector<float> x(static_cast<size_t>(ROWS) * static_cast<size_t>(n_in));
            for (int r = 0; r < ROWS; ++r) {
                const std::vector<float> xr = make_x(n_in, 7u + static_cast<uint32_t>(r));
                std::copy(xr.begin(), xr.end(), x.begin() + static_cast<size_t>(r) * static_cast<size_t>(n_in));
            }
            void *dg = nullptr, *du = nullptr;
            float *dx = nullptr, *dh = nullptr, *dgg = nullptr, *duu = nullptr;
            int32_t* did = nullptr;
            const size_t bytes = eb * static_cast<size_t>(n_exp);
            CUDA_OK(cudaMalloc(&dg, bytes));
            CUDA_OK(cudaMalloc(&du, bytes));
            CUDA_OK(cudaMalloc(&dx, sizeof(float) * x.size()));
            CUDA_OK(cudaMalloc(&dh, sizeof(float) * static_cast<size_t>(N) * static_cast<size_t>(n_out)));
            CUDA_OK(cudaMalloc(&dgg, sizeof(float) * static_cast<size_t>(K) * static_cast<size_t>(n_out)));
            CUDA_OK(cudaMalloc(&duu, sizeof(float) * static_cast<size_t>(K) * static_cast<size_t>(n_out)));
            CUDA_OK(cudaMalloc(&did, sizeof(int32_t) * N));
            CUDA_OK(cudaMemcpy(dg, g.t->data, bytes, cudaMemcpyHostToDevice));
            CUDA_OK(cudaMemcpy(du, u->data, bytes, cudaMemcpyHostToDevice));
            CUDA_OK(cudaMemcpy(dx, x.data(), sizeof(float) * x.size(), cudaMemcpyHostToDevice));
            CUDA_OK(cudaMemcpy(did, ids.data(), sizeof(int32_t) * N, cudaMemcpyHostToDevice));
            const bool launched = cu::gemv_experts_gate_up(g.t->type, dg, du, eb, did, N, dx, dh, n_out, n_in, nullptr, K, n_in);
            CHECK(launched, "gemv_experts_gate_up launches (IQ2_S)");
            CUDA_OK(cudaDeviceSynchronize());
            std::vector<float> h(static_cast<size_t>(N) * static_cast<size_t>(n_out));
            CUDA_OK(cudaMemcpy(h.data(), dh, sizeof(float) * h.size(), cudaMemcpyDeviceToHost));
            double worst = 0;
            for (int r = 0; r < ROWS; ++r) {
                CHECK(cu::gemv_experts_dual(g.t->type, dg, du, eb, did + r * K, K, dx + static_cast<size_t>(r) * static_cast<size_t>(n_in), dgg, duu,
                                            n_out, n_in, nullptr),
                      "gemv_experts_dual (reference) launches");
                CUDA_OK(cudaDeviceSynchronize());
                std::vector<float> gg(static_cast<size_t>(K) * static_cast<size_t>(n_out)), uu(gg.size());
                CUDA_OK(cudaMemcpy(gg.data(), dgg, sizeof(float) * gg.size(), cudaMemcpyDeviceToHost));
                CUDA_OK(cudaMemcpy(uu.data(), duu, sizeof(float) * uu.size(), cudaMemcpyDeviceToHost));
                for (size_t i = 0; i < gg.size(); ++i) {
                    const double ref = double(gg[i]) / (1.0 + std::exp(-double(gg[i]))) * double(uu[i]);
                    const double got = h[static_cast<size_t>(r) * gg.size() + i];
                    worst = std::max(worst, std::fabs(got - ref) / (1.0 + std::fabs(ref)));
                }
            }
            std::printf("     IQ2_S gate/up + SwiGLU, %d slots of %d rows: max |h - h_ref| / (1 + |h_ref|) %.3e\n", N, ROWS, worst);
            CHECK(worst <= 2e-6, "IQ2_S gate/up with SwiGLU fused equals the generic pair within 2e-6");
            cudaFree(dg);
            cudaFree(du);
            cudaFree(dx);
            cudaFree(dh);
            cudaFree(dgg);
            cudaFree(duu);
            cudaFree(did);
        }
    }
    if (d.t) {
        const int n_in = static_cast<int>(d.t->ne[0]), n_out = static_cast<int>(d.t->ne[1]), n_exp = static_cast<int>(d.t->ne[2]);
        const uint64_t eb = limbo::type_bytes(d.t->type, static_cast<uint64_t>(n_in)) * static_cast<uint64_t>(n_out);
        constexpr int N = 16;
        std::vector<int32_t> ids(N);
        for (int i = 0; i < N; ++i) ids[static_cast<size_t>(i)] = (i * 31 + 3) % n_exp;
        std::vector<float> h(static_cast<size_t>(N) * static_cast<size_t>(n_in));
        for (int i = 0; i < N; ++i) {
            const std::vector<float> xr = make_x(n_in, 50u + static_cast<uint32_t>(i));
            std::copy(xr.begin(), xr.end(), h.begin() + static_cast<size_t>(i) * static_cast<size_t>(n_in));
        }
        void* dw = nullptr;
        float *dh = nullptr, *dy = nullptr, *dr = nullptr;
        int32_t* did = nullptr;
        const size_t bytes = eb * static_cast<size_t>(n_exp);
        CUDA_OK(cudaMalloc(&dw, bytes));
        CUDA_OK(cudaMalloc(&dh, sizeof(float) * h.size()));
        CUDA_OK(cudaMalloc(&dy, sizeof(float) * static_cast<size_t>(N) * static_cast<size_t>(n_out)));
        CUDA_OK(cudaMalloc(&dr, sizeof(float) * static_cast<size_t>(N) * static_cast<size_t>(n_out)));
        CUDA_OK(cudaMalloc(&did, sizeof(int32_t) * N));
        CUDA_OK(cudaMemcpy(dw, d.t->data, bytes, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(dh, h.data(), sizeof(float) * h.size(), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(did, ids.data(), sizeof(int32_t) * N, cudaMemcpyHostToDevice));
        CHECK(cu::gemv_experts_down_fast(d.t->type, dw, eb, did, N, dh, dy, n_out, n_in, nullptr), "gemv_experts_down_fast launches (IQ3_S)");
        CHECK(cu::gemv_experts(d.t->type, dw, eb, did, N, dh, n_in, dr, n_out, n_in, nullptr), "gemv_experts (reference) launches");
        CUDA_OK(cudaDeviceSynchronize());
        std::vector<float> y(static_cast<size_t>(N) * static_cast<size_t>(n_out)), r(y.size());
        CUDA_OK(cudaMemcpy(y.data(), dy, sizeof(float) * y.size(), cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(r.data(), dr, sizeof(float) * r.size(), cudaMemcpyDeviceToHost));
        // against the CPU decoder on a sample of rows of every slot
        Err err;
        for (int e = 0; e < N; ++e)
            for (int row = 0; row < n_out; row += 37) {
                double l1 = 0;
                const double cpu = cpu_row_dot(d.t->type, static_cast<const uint8_t*>(d.t->data) + static_cast<uint64_t>(ids[static_cast<size_t>(e)]) * eb +
                                                              static_cast<uint64_t>(row) * limbo::type_bytes(d.t->type, static_cast<uint64_t>(n_in)),
                                               &h[static_cast<size_t>(e) * static_cast<size_t>(n_in)], n_in, &l1);
                err.add(y[static_cast<size_t>(e) * static_cast<size_t>(n_out) + static_cast<size_t>(row)], cpu, l1);
            }
        double vs_ref = 0;
        for (size_t i = 0; i < y.size(); ++i) vs_ref = std::max(vs_ref, std::fabs(double(y[i]) - double(r[i])));
        std::printf("     IQ3_S down, %d slots: max rel err vs the CPU decoder %.3e (%d rows), max |y - y_generic| %.3e\n", N, err.max_rel, err.rows, vs_ref);
        CHECK(err.max_rel <= 1e-5, "IQ3_S down (fast kernel) within 1e-5 of the row's sum|w*x|");
        cudaFree(dw);
        cudaFree(dh);
        cudaFree(dy);
        cudaFree(dr);
        cudaFree(did);
    }
}

void test_dec_small() {
    namespace dc = limbo::cuda::dec;
    constexpr int T = 3, N = 2048, NE = 256, K = 8;
    uint32_t s = 31337;
    auto rnd = [&]() {
        s = s * 1664525u + 1013904223u;
        return static_cast<float>(s >> 8) / 16777216.0f * 2.f - 1.f;
    };
    std::vector<float> x(static_cast<size_t>(T) * N), w(N), lg(static_cast<size_t>(T) * NE);
    for (float& v : x) v = rnd() * 3.f;
    for (float& v : w) v = rnd();
    for (float& v : lg) v = rnd() * 4.f;
    lg[NE + 5] = lg[NE + 9];  // an exact tie in row 1: the lower expert id must win
    float *dx = nullptr, *dw = nullptr, *dy = nullptr, *dl = nullptr, *dwt = nullptr;
    int32_t* did = nullptr;
    CUDA_OK(cudaMalloc(&dx, sizeof(float) * x.size()));
    CUDA_OK(cudaMalloc(&dw, sizeof(float) * w.size()));
    CUDA_OK(cudaMalloc(&dy, sizeof(float) * x.size()));
    CUDA_OK(cudaMalloc(&dl, sizeof(float) * lg.size()));
    CUDA_OK(cudaMalloc(&dwt, sizeof(float) * T * K));
    CUDA_OK(cudaMalloc(&did, sizeof(int32_t) * T * K));
    CUDA_OK(cudaMemcpy(dx, x.data(), sizeof(float) * x.size(), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dw, w.data(), sizeof(float) * w.size(), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dl, lg.data(), sizeof(float) * lg.size(), cudaMemcpyHostToDevice));
    CHECK(dc::rmsnorm(dx, dw, dy, N, 1e-6f, T, nullptr), "rmsnorm launches");
    CHECK(dc::topk(dl, NE, K, T, did, dwt, nullptr), "topk launches");
    CUDA_OK(cudaDeviceSynchronize());
    std::vector<float> y(x.size()), wt(static_cast<size_t>(T) * K);
    std::vector<int32_t> id(static_cast<size_t>(T) * K);
    CUDA_OK(cudaMemcpy(y.data(), dy, sizeof(float) * y.size(), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(wt.data(), dwt, sizeof(float) * wt.size(), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(id.data(), did, sizeof(int32_t) * id.size(), cudaMemcpyDeviceToHost));
    double nerr = 0;
    for (int t = 0; t < T; ++t) {
        double ss = 0;
        for (int i = 0; i < N; ++i) ss += double(x[static_cast<size_t>(t) * N + static_cast<size_t>(i)]) * double(x[static_cast<size_t>(t) * N + static_cast<size_t>(i)]);
        const float inv = static_cast<float>(1.0 / std::sqrt(ss / N + 1e-6));
        for (int i = 0; i < N; ++i) {
            const float ref = x[static_cast<size_t>(t) * N + static_cast<size_t>(i)] * inv * w[static_cast<size_t>(i)];
            nerr = std::max(nerr, std::fabs(double(y[static_cast<size_t>(t) * N + static_cast<size_t>(i)]) - double(ref)));
        }
    }
    std::printf("     rmsnorm, %d rows: max |gpu - cpu| %.3e\n", T, nerr);
    CHECK(nerr <= 1e-6, "rmsnorm matches the reference (double sum of squares)");
    bool ids_ok = true;
    double werr = 0;
    for (int t = 0; t < T; ++t) {
        std::vector<double> p(NE);
        double m = -1e300, sum = 0;
        for (int i = 0; i < NE; ++i) m = std::max(m, double(lg[static_cast<size_t>(t) * NE + static_cast<size_t>(i)]));
        for (int i = 0; i < NE; ++i) {
            p[static_cast<size_t>(i)] = std::exp(double(lg[static_cast<size_t>(t) * NE + static_cast<size_t>(i)]) - m);
            sum += p[static_cast<size_t>(i)];
        }
        std::vector<int> order(NE);
        for (int i = 0; i < NE; ++i) order[static_cast<size_t>(i)] = i;
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return p[static_cast<size_t>(a)] > p[static_cast<size_t>(b)]; });
        double chosen = 0;
        for (int k = 0; k < K; ++k) chosen += p[static_cast<size_t>(order[static_cast<size_t>(k)])];
        for (int k = 0; k < K; ++k) {
            ids_ok = ids_ok && id[static_cast<size_t>(t) * K + static_cast<size_t>(k)] == order[static_cast<size_t>(k)];
            werr = std::max(werr, std::fabs(double(wt[static_cast<size_t>(t) * K + static_cast<size_t>(k)]) - p[static_cast<size_t>(order[static_cast<size_t>(k)])] / chosen));
        }
    }
    std::printf("     topk, %d rows of %d logits (a tie in row 1): ids %s, max weight error %.3e\n", T, NE, ids_ok ? "equal" : "DIFFER", werr);
    CHECK(ids_ok, "topk picks the reference experts in descending order (ties to the lower id)");
    CHECK(werr <= 1e-6, "topk weights match the reference");
    cudaFree(dx);
    cudaFree(dw);
    cudaFree(dy);
    cudaFree(dl);
    cudaFree(dwt);
    cudaFree(did);

    // F32 products of the router and alpha/beta: columns and the second matrix
    constexpr int NO = 64, C = 3;
    std::vector<float> W0(static_cast<size_t>(NO) * N), W1(W0.size()), xx(static_cast<size_t>(C) * N);
    for (float& v : W0) v = rnd();
    for (float& v : W1) v = rnd();
    for (float& v : xx) v = rnd() * 2.f;
    float *a0, *a1, *ax, *y0, *y1;
    CUDA_OK(cudaMalloc(&a0, sizeof(float) * W0.size()));
    CUDA_OK(cudaMalloc(&a1, sizeof(float) * W1.size()));
    CUDA_OK(cudaMalloc(&ax, sizeof(float) * xx.size()));
    CUDA_OK(cudaMalloc(&y0, sizeof(float) * NO * C));
    CUDA_OK(cudaMalloc(&y1, sizeof(float) * NO * C));
    CUDA_OK(cudaMemcpy(a0, W0.data(), sizeof(float) * W0.size(), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(a1, W1.data(), sizeof(float) * W1.size(), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(ax, xx.data(), sizeof(float) * xx.size(), cudaMemcpyHostToDevice));
    CHECK(dc::f32_cols(a0, a1, ax, N, y0, y1, NO, NO, N, C, 2, nullptr), "f32_cols launches");
    CUDA_OK(cudaDeviceSynchronize());
    std::vector<float> r0(static_cast<size_t>(NO) * C), r1(r0.size());
    CUDA_OK(cudaMemcpy(r0.data(), y0, sizeof(float) * r0.size(), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(r1.data(), y1, sizeof(float) * r1.size(), cudaMemcpyDeviceToHost));
    Err e32;
    for (int c = 0; c < C; ++c)
        for (int r = 0; r < NO; ++r) {
            double d0 = 0, d1 = 0, l0 = 0, l1 = 0;
            for (int i = 0; i < N; ++i) {
                const double p0 = double(W0[static_cast<size_t>(r) * N + static_cast<size_t>(i)]) * double(xx[static_cast<size_t>(c) * N + static_cast<size_t>(i)]);
                const double p1 = double(W1[static_cast<size_t>(r) * N + static_cast<size_t>(i)]) * double(xx[static_cast<size_t>(c) * N + static_cast<size_t>(i)]);
                d0 += p0;
                d1 += p1;
                l0 += std::fabs(p0);
                l1 += std::fabs(p1);
            }
            e32.add(r0[static_cast<size_t>(c) * NO + static_cast<size_t>(r)], d0, l0);
            e32.add(r1[static_cast<size_t>(c) * NO + static_cast<size_t>(r)], d1, l1);
        }
    // column independence: one column alone gives the same bits
    CHECK(dc::f32_cols(a0, nullptr, ax + N, N, y0, nullptr, NO, NO, N, 1, 1, nullptr), "f32_cols (one column) launches");
    CUDA_OK(cudaDeviceSynchronize());
    std::vector<float> one(NO);
    CUDA_OK(cudaMemcpy(one.data(), y0, sizeof(float) * NO, cudaMemcpyDeviceToHost));
    CHECK(std::memcmp(one.data(), &r0[static_cast<size_t>(NO)], sizeof(float) * NO) == 0, "f32_cols: a column alone equals the same column of three");
    std::printf("     f32_cols, %d rows x %d columns x 2 matrices: max rel err %.3e\n", NO, C, e32.max_rel);
    CHECK(e32.max_rel <= 1e-5, "f32_cols within 1e-5 of the row's sum|w*x|");
    cudaFree(a0);
    cudaFree(a1);
    cudaFree(ax);
    cudaFree(y0);
    cudaFree(y1);
}

// ---- the formats added for the 27B and Flash-Next models: synthetic blocks against the CPU decoder -------------------
// The rung-1 file holds none of them, so the rows are random block bytes with valid half-float scales; the reference is
// src/quants.cpp's decoder and a double dot, the bar the same 1e-5 of sum|w*x| as the rung-1 formats.
void test_ext_formats() {
    struct E {
        Type t;
        const char* name;
    };
    const E es[] = {{Type::Q4_K, "Q4_K"},     {Type::Q5_K, "Q5_K"},   {Type::IQ2_XXS, "IQ2_XXS"}, {Type::IQ2_XS, "IQ2_XS"},
                    {Type::IQ3_XXS, "IQ3_XXS"}, {Type::IQ1_S, "IQ1_S"}, {Type::IQ1_M, "IQ1_M"},     {Type::IQ4_NL, "IQ4_NL"},
                    {Type::Q2_0, "Q2_0"},     {Type::F16, "F16"}};
    constexpr int NIN = 512, NROWS = 100, C = 3;
    for (const E& e : es) {
        const uint32_t type = static_cast<uint32_t>(e.t);
        const limbo::TypeInfo* ti = limbo::type_info(type);
        const uint64_t rb = limbo::type_bytes(type, NIN);
        std::vector<uint8_t> host(static_cast<size_t>(NROWS) * rb);
        uint32_t st = 0x9e3779b9u ^ type;
        for (uint8_t& b : host) {
            st = st * 1664525u + 1013904223u;
            b = static_cast<uint8_t>(st >> 24);
        }
        const size_t nblk = host.size() / ti->block_bytes;
        auto put16 = [&](uint8_t* p, uint16_t v) {
            p[0] = static_cast<uint8_t>(v & 0xff);
            p[1] = static_cast<uint8_t>(v >> 8);
        };
        for (size_t b = 0; b < nblk; ++b) {
            uint8_t* p = host.data() + b * ti->block_bytes;
            switch (e.t) {
                case Type::Q4_K:
                case Type::Q5_K:
                    put16(p, 0x3000);
                    put16(p + 2, 0x2800);
                    break;
                case Type::IQ1_M: {  // the f16 scale rides in the top nibbles of the four scale words
                    const uint16_t sbits = 0x3000;
                    for (int i = 0; i < 4; ++i) {
                        const unsigned w = p[48 + 2 * i] | (p[49 + 2 * i] << 8);
                        put16(p + 48 + 2 * i, static_cast<uint16_t>((w & 0x0fffu) | (((sbits >> (4 * i)) & 0xfu) << 12)));
                    }
                    break;
                }
                case Type::F16: {  // a block is one element
                    st = st * 1664525u + 1013904223u;
                    const __half h = __float2half(static_cast<float>(st >> 8) / 16777216.0f * 2.f - 1.f);
                    put16(p, __half_as_ushort(h));
                    break;
                }
                default: put16(p, 0x3000); break;
            }
        }
        const std::vector<float> x = make_x(NIN, 300u + type);
        std::vector<float> xs(static_cast<size_t>(C) * NIN);
        for (int c = 0; c < C; ++c) {
            const std::vector<float> xc = make_x(NIN, 400u + type + static_cast<uint32_t>(c));
            std::copy(xc.begin(), xc.end(), xs.begin() + static_cast<size_t>(c) * NIN);
        }
        void* dW = nullptr;
        float *dx = nullptr, *dy = nullptr, *dyc = nullptr;
        CUDA_OK(cudaMalloc(&dW, host.size()));
        CUDA_OK(cudaMalloc(&dx, sizeof(float) * xs.size()));
        CUDA_OK(cudaMalloc(&dy, sizeof(float) * NROWS));
        CUDA_OK(cudaMalloc(&dyc, sizeof(float) * NROWS * C));
        CUDA_OK(cudaMemcpy(dW, host.data(), host.size(), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(dx, x.data(), sizeof(float) * NIN, cudaMemcpyHostToDevice));
        char what[120];
        std::snprintf(what, sizeof what, "%s: supported by the GPU kernels", e.name);
        CHECK(cu::gemv_supported(type), what);
        CHECK(cu::gemv(type, dW, dx, dy, NROWS, NIN, false, nullptr), "gemv launches");
        CUDA_OK(cudaDeviceSynchronize());
        std::vector<float> y(NROWS);
        CUDA_OK(cudaMemcpy(y.data(), dy, sizeof(float) * NROWS, cudaMemcpyDeviceToHost));
        Err err;
        for (int r = 0; r < NROWS; ++r) {
            double l1 = 0;
            const double cpu = cpu_row_dot(type, host.data() + static_cast<size_t>(r) * rb, x.data(), NIN, &l1);
            err.add(y[static_cast<size_t>(r)], cpu, l1);
        }
        // columns: the multi-column call equals the single-column product bit for bit
        CUDA_OK(cudaMemcpy(dx, xs.data(), sizeof(float) * xs.size(), cudaMemcpyHostToDevice));
        CHECK(cu::gemv_cols(type, dW, dx, NIN, dyc, NROWS, NROWS, NIN, C, false, nullptr), "gemv_cols launches");
        bool same = true;
        std::vector<float> yc(static_cast<size_t>(NROWS) * C), y1(NROWS);
        CUDA_OK(cudaDeviceSynchronize());
        CUDA_OK(cudaMemcpy(yc.data(), dyc, sizeof(float) * yc.size(), cudaMemcpyDeviceToHost));
        for (int c = 0; c < C; ++c) {
            CHECK(cu::gemv(type, dW, dx + static_cast<size_t>(c) * NIN, dy, NROWS, NIN, false, nullptr), "gemv (column) launches");
            CUDA_OK(cudaDeviceSynchronize());
            CUDA_OK(cudaMemcpy(y1.data(), dy, sizeof(float) * NROWS, cudaMemcpyDeviceToHost));
            same = same && std::memcmp(y1.data(), &yc[static_cast<size_t>(c) * NROWS], sizeof(float) * NROWS) == 0;
        }
        std::printf("     %-8s %d rows x %d: max rel err vs the CPU decoder %.3e\n", e.name, NROWS, NIN, err.max_rel);
        std::snprintf(what, sizeof what, "%s: gemv within 1e-5 of the row's sum|w*x|", e.name);
        CHECK(err.max_rel <= 1e-5, what);
        std::snprintf(what, sizeof what, "%s: three columns equal the single-column products bit for bit", e.name);
        CHECK(same, what);
        cudaFree(dW);
        cudaFree(dx);
        cudaFree(dy);
        cudaFree(dyc);
    }
}

// ---- rows with a tail: n_in not a multiple of 256 (qwen4exp's hc_up 320, down projections 640) --------
void test_tail() {
    struct E {
        Type t;
        int n_in;
        const char* name;
    };
    const E es[] = {
        {Type::BF16, 64, "BF16"},        // tail only (n_in < 256)
        {Type::BF16, 320, "BF16"},       // one super-block + 64 (hc_up)
        {Type::BF16, 640, "BF16"},       // two super-blocks + 128
        {Type::IQ4_NL, 640, "IQ4_NL"},   // routed / shared down
        {Type::Q2_0, 640, "Q2_0"},       // routed down on the Q2_0 layers
        {Type::Q8_0, 640, "Q8_0"},
    };
    constexpr int NROWS = 100;
    for (const E& e : es) {
        const uint32_t type = static_cast<uint32_t>(e.t);
        const limbo::TypeInfo* ti = limbo::type_info(type);
        const int n_in = e.n_in;
        const uint64_t rb = limbo::type_bytes(type, static_cast<uint64_t>(n_in));
        std::vector<uint8_t> host(static_cast<size_t>(NROWS) * rb);
        uint32_t st = 0x51ed270bu ^ type ^ static_cast<uint32_t>(n_in);
        for (uint8_t& b : host) {
            st = st * 1664525u + 1013904223u;
            b = static_cast<uint8_t>(st >> 24);
        }
        auto put16 = [&](uint8_t* p, uint16_t v) {
            p[0] = static_cast<uint8_t>(v & 0xff);
            p[1] = static_cast<uint8_t>(v >> 8);
        };
        // every word the decoder reads as a half (block scales and plain elements) must be finite
        const size_t nblk = host.size() / ti->block_bytes;
        for (size_t b = 0; b < nblk; ++b) {
            uint8_t* p = host.data() + b * ti->block_bytes;
            if (ti->block_elems == 1) {
                st = st * 1664525u + 1013904223u;
                const __half h = __float2half(static_cast<float>(st >> 8) / 16777216.0f * 2.f - 1.f);
                put16(p, __half_as_ushort(h));
            } else {
                put16(p, 0x3000);
            }
        }
        const std::vector<float> x = make_x(n_in, 700u + type + static_cast<uint32_t>(n_in));
        void* dW = nullptr;
        float *dx = nullptr, *dy = nullptr;
        CUDA_OK(cudaMalloc(&dW, host.size()));
        CUDA_OK(cudaMalloc(&dx, sizeof(float) * static_cast<size_t>(n_in)));
        CUDA_OK(cudaMalloc(&dy, sizeof(float) * NROWS));
        CUDA_OK(cudaMemcpy(dW, host.data(), host.size(), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(dx, x.data(), sizeof(float) * static_cast<size_t>(n_in), cudaMemcpyHostToDevice));
        CHECK(cu::gemv(type, dW, dx, dy, NROWS, n_in, false, nullptr), "tail gemv launches");
        CUDA_OK(cudaDeviceSynchronize());
        std::vector<float> y(NROWS);
        CUDA_OK(cudaMemcpy(y.data(), dy, sizeof(float) * NROWS, cudaMemcpyDeviceToHost));
        Err err;
        for (int r = 0; r < NROWS; ++r) {
            double l1 = 0;
            const double cpu = cpu_row_dot(type, host.data() + static_cast<size_t>(r) * rb, x.data(), n_in, &l1);
            err.add(y[static_cast<size_t>(r)], cpu, l1);
        }
        // accumulate onto a seeded y, as the decode step's fused add does
        std::vector<float> seed(NROWS, 2.5f), sums(NROWS);
        CUDA_OK(cudaMemcpy(dy, seed.data(), sizeof(float) * NROWS, cudaMemcpyHostToDevice));
        CHECK(cu::gemv(type, dW, dx, dy, NROWS, n_in, true, nullptr), "tail gemv accumulate launches");
        CUDA_OK(cudaDeviceSynchronize());
        CUDA_OK(cudaMemcpy(sums.data(), dy, sizeof(float) * NROWS, cudaMemcpyDeviceToHost));
        double acc_err = 0;
        for (int r = 0; r < NROWS; ++r) {
            double l1 = 0;
            const double cpu = cpu_row_dot(type, host.data() + static_cast<size_t>(r) * rb, x.data(), n_in, &l1);
            acc_err = std::max(acc_err, std::fabs(double(sums[static_cast<size_t>(r)]) - (double(seed[static_cast<size_t>(r)]) + cpu)) / std::max(l1, 1.0));
        }
        std::printf("     %-8s tail n_in %3d: max rel err %.3e  accumulate %.3e\n", e.name, n_in, err.max_rel, acc_err);
        char what[120];
        std::snprintf(what, sizeof what, "%s n_in %d: gemv within 1e-5 of the row's sum|w*x|", e.name, n_in);
        CHECK(err.max_rel <= 1e-5 && acc_err <= 1e-5, what);
        // several columns over the same tail rows: each column the single-column product bit for bit, plain and accumulated
        {
            constexpr int C = 8;
            std::vector<float> xs(static_cast<size_t>(C) * static_cast<size_t>(n_in));
            for (int c = 0; c < C; ++c) {
                const std::vector<float> xc = make_x(n_in, 900u + type + static_cast<uint32_t>(c));
                std::copy(xc.begin(), xc.end(), xs.begin() + static_cast<size_t>(c) * static_cast<size_t>(n_in));
            }
            float *dxs = nullptr, *dys = nullptr, *d1 = nullptr;
            CUDA_OK(cudaMalloc(&dxs, sizeof(float) * xs.size()));
            CUDA_OK(cudaMalloc(&dys, sizeof(float) * C * NROWS));
            CUDA_OK(cudaMalloc(&d1, sizeof(float) * NROWS));
            CUDA_OK(cudaMemcpy(dxs, xs.data(), sizeof(float) * xs.size(), cudaMemcpyHostToDevice));
            bool ok = true, same = true;
            for (const bool accum : {false, true})
                for (int cols : {2, 5, 8}) {
                    std::vector<float> seeds(static_cast<size_t>(C) * NROWS, 1.25f);
                    CUDA_OK(cudaMemcpy(dys, seeds.data(), sizeof(float) * seeds.size(), cudaMemcpyHostToDevice));
                    ok = ok && cu::gemv_cols(type, dW, dxs, n_in, dys, NROWS, NROWS, n_in, cols, accum, nullptr);
                    CUDA_OK(cudaDeviceSynchronize());
                    std::vector<float> yy(static_cast<size_t>(cols) * NROWS), y1(NROWS);
                    CUDA_OK(cudaMemcpy(yy.data(), dys, sizeof(float) * yy.size(), cudaMemcpyDeviceToHost));
                    for (int c = 0; c < cols; ++c) {
                        CUDA_OK(cudaMemcpy(d1, seeds.data(), sizeof(float) * NROWS, cudaMemcpyHostToDevice));
                        ok = ok && cu::gemv(type, dW, dxs + static_cast<size_t>(c) * static_cast<size_t>(n_in), d1, NROWS, n_in, accum, nullptr);
                        CUDA_OK(cudaDeviceSynchronize());
                        CUDA_OK(cudaMemcpy(y1.data(), d1, sizeof(float) * NROWS, cudaMemcpyDeviceToHost));
                        same = same && std::memcmp(y1.data(), &yy[static_cast<size_t>(c) * NROWS], sizeof(float) * NROWS) == 0;
                    }
                }
            std::snprintf(what, sizeof what, "%s n_in %d: gemv_cols with 2, 5 and 8 columns equals gemv bit for bit", e.name, n_in);
            CHECK(ok && same, what);
            cudaFree(dxs);
            cudaFree(dys);
            cudaFree(d1);
        }
        cudaFree(dW);
        cudaFree(dx);
        cudaFree(dy);
    }
}

}  // namespace

// gemv_narrow: a few rows over a long input (qwen4exp's hc inject 4 x 10240), 1..8 columns, plain and accumulate. Another
// summation order than gemv(), so the bar is closeness: |d| <= 1e-5 of the row's L1 mass.
void test_narrow() {
    struct E {
        Type t;
        int n_out, n_in;
    };
    const E es[] = {{Type::BF16, 4, 10240}, {Type::BF16, 4, 10240 + 64}, {Type::BF16, 7, 1000 * 8}, {Type::F32, 4, 4096}, {Type::F16, 3, 2560}};
    for (const E& e : es) {
        const uint32_t type = static_cast<uint32_t>(e.t);
        const uint64_t rb = limbo::type_bytes(type, static_cast<uint64_t>(e.n_in));
        std::vector<uint8_t> host(static_cast<size_t>(e.n_out) * rb);
        uint32_t st = 0x9e3779b9u ^ type ^ static_cast<uint32_t>(e.n_in);
        const int esz = e.t == Type::F32 ? 4 : 2;
        for (size_t i = 0; i < host.size() / size_t(esz); ++i) {
            st = st * 1664525u + 1013904223u;
            const float v = static_cast<float>(st >> 8) / 16777216.0f * 2.f - 1.f;
            uint8_t* p = host.data() + i * size_t(esz);
            if (e.t == Type::F32) std::memcpy(p, &v, 4);
            else if (e.t == Type::F16) {
                const uint16_t h = __half_as_ushort(__float2half(v));
                std::memcpy(p, &h, 2);
            } else {
                uint32_t u;
                std::memcpy(&u, &v, 4);
                const uint16_t h = static_cast<uint16_t>(u >> 16);
                std::memcpy(p, &h, 2);
            }
        }
        constexpr int CMAX = 8;
        const int xs = (e.n_in + 3) / 4 * 4 + 4;
        std::vector<float> x(static_cast<size_t>(CMAX) * xs);
        for (int c = 0; c < CMAX; ++c) {
            const std::vector<float> xc = make_x(e.n_in, 900u + type + static_cast<uint32_t>(c));
            std::copy(xc.begin(), xc.end(), x.begin() + static_cast<size_t>(c) * xs);
        }
        void* dW = nullptr;
        float *dx = nullptr, *dy = nullptr, *dr = nullptr;
        CUDA_OK(cudaMalloc(&dW, host.size()));
        CUDA_OK(cudaMalloc(&dx, sizeof(float) * x.size()));
        CUDA_OK(cudaMalloc(&dy, sizeof(float) * CMAX * 64));
        CUDA_OK(cudaMalloc(&dr, sizeof(float) * 64));
        CUDA_OK(cudaMemcpy(dW, host.data(), host.size(), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(dx, x.data(), sizeof(float) * x.size(), cudaMemcpyHostToDevice));
        double worst = 0;
        bool ok = true;
        for (int C : {1, 3, 5, 8})
            for (bool accum : {false, true}) {
                std::vector<float> seed(static_cast<size_t>(CMAX) * 64, 0.25f);
                CUDA_OK(cudaMemcpy(dy, seed.data(), sizeof(float) * seed.size(), cudaMemcpyHostToDevice));
                ok = ok && cu::gemv_narrow(type, dW, dx, xs, dy, 64, e.n_out, e.n_in, C, accum, nullptr);
                std::vector<float> got(seed.size());
                CUDA_OK(cudaDeviceSynchronize());
                CUDA_OK(cudaMemcpy(got.data(), dy, sizeof(float) * got.size(), cudaMemcpyDeviceToHost));
                for (int c = 0; c < C; ++c) {
                    ok = ok && cu::gemv(type, dW, dx + static_cast<size_t>(c) * xs, dr, e.n_out, e.n_in, false, nullptr);
                    std::vector<float> ref(static_cast<size_t>(e.n_out));
                    CUDA_OK(cudaMemcpy(ref.data(), dr, sizeof(float) * ref.size(), cudaMemcpyDeviceToHost));
                    for (int r = 0; r < e.n_out; ++r) {
                        double l1 = 0;
                        for (int k = 0; k < e.n_in; ++k) l1 += std::fabs(double(x[static_cast<size_t>(c) * xs + k]));
                        const double want = double(ref[size_t(r)]) + (accum ? 0.25 : 0.0);
                        worst = std::max(worst, std::fabs(double(got[size_t(c) * 64 + r]) - want) / std::max(l1, 1.0));
                    }
                }
            }
        char what[160];
        std::snprintf(what, sizeof what, "narrow %s %d x %d, 1/3/5/8 columns, plain and accumulate: max |d| / L1 %.2e <= 1e-5",
                      std::string(limbo::type_info(type)->name).c_str(), e.n_out, e.n_in, worst);
        CHECK(ok && worst <= 1e-5, what);
        CHECK(!cu::gemv_narrow(static_cast<uint32_t>(Type::Q8_0), dW, dx, xs, dy, 64, 4, 1024, 1, false, nullptr),
              "narrow refuses a block-scaled format (the caller falls back)");
        cudaFree(dW);
        cudaFree(dx);
        cudaFree(dy);
        cudaFree(dr);
    }
}

int main() {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        std::printf("SKIP cuda_gemv_test: no CUDA device\n");
        return 0;
    }
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    if (free_b < (1ull << 30)) {
        std::printf("SKIP cuda_gemv_test: %.0f MB free of %.0f MB, need 1 GB (the desktop holds the rest)\n",
                    double(free_b) / 1e6, double(total_b) / 1e6);
        return 0;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::printf("     %s, sm_%d%d, %.0f MB free of %.0f MB\n", prop.name, prop.major, prop.minor, double(free_b) / 1e6,
                double(total_b) / 1e6);

    std::vector<std::unique_ptr<limbo::gguf::File>> files;
    for (const std::string& p : model_paths()) {
        std::string err;
        auto f = limbo::gguf::File::open(p, &err);
        if (!f) {
            std::printf("SKIP cuda_gemv_test: cannot open %s: %s\n", p.c_str(), err.c_str());
            continue;
        }
        files.push_back(std::move(f));
    }
    if (files.empty()) return 0;

    std::string err;
    CHECK(cu::init_tables(&err), "codebook tables reach the device");

    struct Fmt {
        uint32_t type;
        const char* name;
    };
    const Fmt fmts[] = {
        {static_cast<uint32_t>(Type::F32), "F32"},     {static_cast<uint32_t>(Type::BF16), "BF16"},
        {static_cast<uint32_t>(Type::Q8_0), "Q8_0"},   {static_cast<uint32_t>(Type::Q2_K), "Q2_K"},
        {static_cast<uint32_t>(Type::Q3_K), "Q3_K"},   {static_cast<uint32_t>(Type::Q6_K), "Q6_K"},
        {static_cast<uint32_t>(Type::IQ2_S), "IQ2_S"}, {static_cast<uint32_t>(Type::IQ3_S), "IQ3_S"},
        {static_cast<uint32_t>(Type::IQ4_XS), "IQ4_XS"},
    };
    std::printf("-- plain GEMV, real rows, GPU vs CPU decode + double dot (bar 1e-5 of sum|w*x|)\n");
    for (const Fmt& f : fmts) test_gemv(files, f.type, f.name);
    std::printf("-- MoE form, 8 experts chosen by device-side ids\n");
    for (const Fmt& f : fmts) test_experts(files, f.type, f.name);
    std::printf("-- fused decode forms\n");
    test_fused(files);
    std::printf("-- embedding row\n");
    test_embed(files);
    std::printf("-- several columns at once, one launch (bit-identical to one column)\n");
    for (const Fmt& f : fmts) test_cols(files, f.type, f.name);
    std::printf("-- decode products through the fragment kernel (1..16 columns, int8 activation planes)\n");
    test_frag_windows(files);
    std::printf("-- the decode kernels: expert fast paths, norm, top-k, F32 products\n");
    test_decode_fast(files);
    test_dec_small();
    std::printf("-- formats added for the 27B and Flash-Next models (synthetic blocks against the CPU decoder)\n");
    test_ext_formats();
    std::printf("-- rows with a tail (n_in %% 256 != 0): qwen4exp's hc_up 320 and the down projections 640\n");
    test_tail();
    std::printf("-- few rows over a long input (gemv_narrow): qwen4exp's hc inject\n");
    test_narrow();
    return limbo::test::finish("cuda_gemv_test");
}
