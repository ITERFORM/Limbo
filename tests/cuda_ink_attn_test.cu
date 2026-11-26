// Copyright (c) 2026 ITERFORM. All rights reserved.
// tests/cuda_ink_attn_test.cu — the inkling2 relative-bias attention (src/cuda/ink_attn.cu) against a
// double-precision reference, at positions around every boundary it has (the chunk edge 255/256, the bias
// table's extent 1023/1024/1025, the window edge), for a single row and for verify windows of up to 8 rows.
//
//   1. accuracy: max |gpu - cpu| over every output element, bar 3e-6 (outputs are weighted means of N(0,1)-ish
//      values), at pos0 0, 3, 255, 256, 257, 1000, 2048 and 3000, with window 0 and window 1024;
//   2. batch invariance: row t of a window is bit-identical to the single-row launch at position pos0 + t;
//   3. causality: the cache rows after the window hold arbitrary finite values and must not move the result;
//   4. the sliding ring: a window-1024 slot kept as a ring of 1031 rows (position p at row p % 1031, what
//      InkGpu allocates for a sliding layer) is bit-identical to the full cache, past the ring's wrap, and a
//      ring too small for the window plus the verify rows is refused.
//
// The reference is the graph's own fold (docs/inkling2.md §3): logit_j = scale*q.k_j + table[d_j][h] with
// d_j = min(q_pos - kv_pos_j, extent), p = softmax over the visible keys with the sink column at logit 0 in
// the denominator (the table already carries -sink[h]). K and V are f16 caches [pos][kv head * hd] like the
// CPU step's (src/step.cpp). No GPU: SKIP, exit 0.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../src/cuda/ink_ops.hpp"
#include "limbo/check.hpp"

namespace {

#define CUDA_OK(call)                                                                                \
    do {                                                                                             \
        cudaError_t e_ = (call);                                                                     \
        if (e_ != cudaSuccess) {                                                                     \
            std::printf("FAIL %s: %s (%s:%d)\n", #call, cudaGetErrorString(e_), __FILE__, __LINE__);  \
            ++limbo::test::failures();                                                               \
            return;                                                                                  \
        }                                                                                            \
    } while (0)

constexpr int NH = 24, NKV = 8, HD = 128, EXTENT = 1024, N_CTX = 4096;

float frand(uint32_t& s) {
    s = s * 1664525u + 1013904223u;
    return static_cast<float>(s >> 8) / 16777216.0f * 2.f - 1.f;
}

// The reference: out[t][h][d] over the visible keys of q_pos + t.
void ref_attn(const std::vector<float>& q, const std::vector<float>& kv_cache, const std::vector<double>& table,
              std::vector<float>& out, int T, int pos0, int n_head, int n_head_kv, double scale, int extent,
              int window) {
    const int kvw = n_head_kv * HD, G = n_head / n_head_kv;
    out.assign(size_t(T) * n_head * HD, 0.0f);
    for (int t = 0; t < T; ++t) {
        const int q_pos = pos0 + t;
        int t0 = 0;
        if (window > 0) t0 = std::max(0, q_pos - window + 1);
        for (int h = 0; h < n_head; ++h) {
            const int kv = h / G;
            const float* qh = q.data() + (size_t(t) * n_head + h) * HD;
            double m = 0.0, denom = 0.0;   // the sink column at logit 0 joins the max and the denominator
            std::vector<double> w(size_t(q_pos - t0 + 1));
            for (int j = t0; j <= q_pos; ++j) {
                const float* kr = kv_cache.data() + size_t(j) * kvw + size_t(kv) * HD;
                double qk = 0;
                for (int d = 0; d < HD; ++d) qk += double(qh[d]) * double(kr[d]);
                const int dist = std::min(q_pos - j, extent);
                const double l = scale * qk + table[size_t(dist) * n_head + h];
                w[size_t(j - t0)] = l;
                m = std::max(m, l);
            }
            denom = std::exp(-m);
            for (int j = t0; j <= q_pos; ++j) {
                const double e = std::exp(w[size_t(j - t0)] - m);
                w[size_t(j - t0)] = e;
                denom += e;
            }
            float* oh = out.data() + (size_t(t) * n_head + h) * HD;
            for (int j = t0; j <= q_pos; ++j) {
                const float* vr = kv_cache.data() + size_t(j) * kvw + size_t(kv) * HD;
                const double p = w[size_t(j - t0)] / denom;
                for (int d = 0; d < HD; ++d) oh[d] += float(p * double(vr[d]));
            }
        }
    }
}

}  // namespace

int main() {
    int dev = 0;
    if (cudaGetDeviceCount(&dev) != cudaSuccess || dev == 0) {
        std::printf("cuda_ink_attn_test: SKIP (no GPU)\n");
        return limbo::test::finish("cuda_ink_attn");
    }

    uint32_t seed = 1357924680u;
    const int kvw = NKV * HD;
    std::vector<float> q(size_t(8) * NH * HD), kv_cache(size_t(N_CTX) * kvw);
    std::vector<double> table(size_t(EXTENT + 1) * NH);
    for (auto& x : q) x = frand(seed) * 1.5f;
    for (auto& x : kv_cache) x = frand(seed) * 1.5f;
    // the table is a per-(distance, head) bias of a relative-bias model: smooth in d, plus the -sink[h] fold
    for (int d = 0; d <= EXTENT; ++d)
        for (int h = 0; h < NH; ++h)
            table[size_t(d) * NH + h] = (d == EXTENT ? 0.0 : 3.0 * std::sin(0.013 * d + 0.7 * h) - 2.0 * std::cos(0.031 * d * (h % 5 + 1)))
                                        - (2.0 + 4.0 * std::sin(0.9 * h));

    float *d_q = nullptr, *d_table = nullptr, *d_out = nullptr, *d_scratch = nullptr;
    __half* d_kv = nullptr;
    int* d_pos0 = nullptr;
    const size_t n_ctx_kv = size_t(N_CTX) * kvw;
    CUDA_OK(cudaMalloc(&d_q, size_t(8) * NH * HD * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_kv, n_ctx_kv * sizeof(__half)));
    CUDA_OK(cudaMalloc(&d_table, table.size() * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_out, size_t(8) * NH * HD * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_pos0, sizeof(int)));
    const size_t scr = limbo::cuda::ink::attention_scratch_floats(N_CTX, 8, NH, HD);
    CUDA_OK(cudaMalloc(&d_scratch, scr * sizeof(float)));

    // the f16 cache the kernels read: the same values the reference sees, rounded like the CPU step's cache
    std::vector<__half> kv16(n_ctx_kv);
    for (size_t i = 0; i < n_ctx_kv; ++i) kv16[i] = __float2half(kv_cache[i]);
    CUDA_OK(cudaMemcpy(d_kv, kv16.data(), n_ctx_kv * sizeof(__half), cudaMemcpyHostToDevice));
    std::vector<float> table32(table.size());
    for (size_t i = 0; i < table.size(); ++i) table32[i] = float(table[i]);
    CUDA_OK(cudaMemcpy(d_table, table32.data(), table.size() * sizeof(float), cudaMemcpyHostToDevice));

    // the reference reads the f16 cache values too (the rounded ones), so the comparison isolates the kernel
    std::vector<float> kv_ref(n_ctx_kv);
    for (size_t i = 0; i < n_ctx_kv; ++i) kv_ref[i] = __half2float(kv16[i]);

    const int cases[] = {0, 3, 255, 256, 257, 1000, 2048, 3000};
    const int windows[] = {0, 1024};
    const float scale = float(1.0 / std::sqrt(double(HD)));

    for (int w : windows) {
        for (int pos0 : cases) {
            for (int T : {1, 8}) {
                if (pos0 + T - 1 >= N_CTX) continue;
                std::vector<float> qh(size_t(T) * NH * HD);
                for (auto& x : qh) x = frand(seed) * 1.5f;
                CUDA_OK(cudaMemcpy(d_q, qh.data(), qh.size() * sizeof(float), cudaMemcpyHostToDevice));
                CUDA_OK(cudaMemcpy(d_pos0, &pos0, sizeof(int), cudaMemcpyHostToDevice));

                std::vector<float> ref;
                ref_attn(qh, kv_ref, table, ref, T, pos0, NH, NKV, scale, EXTENT, w);
                CUDA_OK(cudaMemset(d_out, 0, size_t(T) * NH * HD * sizeof(float)));
                if (!limbo::cuda::ink::attention(d_q, d_kv, d_kv, d_table, d_out, d_scratch, d_pos0, T, N_CTX, 0,
                                                 NH, NKV, HD, scale, EXTENT, w, nullptr)) {
                    std::printf("FAIL attention() refused pos0 %d T %d window %d\n", pos0, T, w);
                    ++limbo::test::failures();
                    continue;
                }
                CUDA_OK(cudaDeviceSynchronize());
                std::vector<float> got(size_t(T) * NH * HD);
                CUDA_OK(cudaMemcpy(got.data(), d_out, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
                double err = 0;
                for (size_t i = 0; i < got.size(); ++i) err = std::max(err, std::fabs(double(got[i]) - double(ref[i])));
                char what[160];
                std::snprintf(what, sizeof what, "accuracy pos0 %d T %d window %d (max|d| %.2e)", pos0, T, w, err);
                CHECK(err <= 3e-6, what);

                // batch invariance: each row alone is bit-identical to the window's row
                if (T == 8) {
                    for (int t = 0; t < 8; ++t) {
                        const int p1 = pos0 + t;
                        CUDA_OK(cudaMemcpy(d_q, qh.data() + size_t(t) * NH * HD, size_t(NH) * HD * sizeof(float),
                                           cudaMemcpyHostToDevice));
                        CUDA_OK(cudaMemcpy(d_pos0, &p1, sizeof(int), cudaMemcpyHostToDevice));
                        CUDA_OK(cudaMemset(d_out, 0, size_t(NH) * HD * sizeof(float)));
                        if (!limbo::cuda::ink::attention(d_q, d_kv, d_kv, d_table, d_out, d_scratch, d_pos0, 1,
                                                         N_CTX, 0, NH, NKV, HD, scale, EXTENT, w, nullptr)) {
                            std::printf("FAIL attention() refused single row\n");
                            ++limbo::test::failures();
                            break;
                        }
                        CUDA_OK(cudaDeviceSynchronize());
                        std::vector<float> one(size_t(NH) * HD);
                        CUDA_OK(cudaMemcpy(one.data(), d_out, one.size() * sizeof(float), cudaMemcpyDeviceToHost));
                        bool same = true;
                        for (int i = 0; i < NH * HD; ++i)
                            if (one[size_t(i)] != got[size_t(t) * NH * HD + size_t(i)]) same = false;
                        char what2[160];
                        std::snprintf(what2, sizeof what2, "batch invariance pos0 %d row %d window %d", pos0, t, w);
                        CHECK(same, what2);
                    }
                }
            }
        }
    }

    // causality: rows past the query hold garbage and must not move the result
    {
        const int pos0 = 1000, T = 4;
        std::vector<float> qh(size_t(T) * NH * HD);
        for (auto& x : qh) x = frand(seed) * 1.5f;
        CUDA_OK(cudaMemcpy(d_q, qh.data(), qh.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_pos0, &pos0, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemset(d_out, 0, size_t(T) * NH * HD * sizeof(float)));
        if (limbo::cuda::ink::attention(d_q, d_kv, d_kv, d_table, d_out, d_scratch, d_pos0, T, N_CTX, 0, NH, NKV,
                                        HD, scale, EXTENT, 0, nullptr)) {
            CUDA_OK(cudaDeviceSynchronize());
            std::vector<float> clean(size_t(T) * NH * HD);
            CUDA_OK(cudaMemcpy(clean.data(), d_out, clean.size() * sizeof(float), cudaMemcpyDeviceToHost));
            for (size_t i = size_t(pos0 + T) * kvw; i < n_ctx_kv; ++i) kv16[i] = __float2half(frand(seed) * 9.0f);
            CUDA_OK(cudaMemcpy(d_kv, kv16.data(), n_ctx_kv * sizeof(__half), cudaMemcpyHostToDevice));
            CUDA_OK(cudaMemset(d_out, 0, size_t(T) * NH * HD * sizeof(float)));
            limbo::cuda::ink::attention(d_q, d_kv, d_kv, d_table, d_out, d_scratch, d_pos0, T, N_CTX, 0, NH, NKV,
                                        HD, scale, EXTENT, 0, nullptr);
            CUDA_OK(cudaDeviceSynchronize());
            std::vector<float> dirty(size_t(T) * NH * HD);
            CUDA_OK(cudaMemcpy(dirty.data(), d_out, dirty.size() * sizeof(float), cudaMemcpyDeviceToHost));
            bool same = std::memcmp(clean.data(), dirty.data(), clean.size() * sizeof(float)) == 0;
            CHECK(same, "causality: rows past the query must not move the result");
        }
    }

    // the sliding ring: the same window-1024 rows, rebuilt as a ring of RING rows, must give the same bits
    {
        constexpr int W = 1024, RING = W + limbo::cuda::ink::kMaxRows - 1;
        // restore the clean cache the window cases ran on (the causality case dirtied its tail)
        for (size_t i = 0; i < n_ctx_kv; ++i) kv16[i] = __float2half(kv_cache[i]);
        CUDA_OK(cudaMemcpy(d_kv, kv16.data(), n_ctx_kv * sizeof(__half), cudaMemcpyHostToDevice));
        __half* d_ring = nullptr;
        CUDA_OK(cudaMalloc(&d_ring, size_t(RING) * kvw * sizeof(__half)));
        std::vector<__half> ring16(size_t(RING) * kvw);
        for (int pos0 : {0, 1000, 1030, 1031, 2048, 3000, 4088}) {
            for (int T : {1, 8}) {
                if (pos0 + T - 1 >= N_CTX) continue;
                const int last = pos0 + T - 1;
                // the ring holds the RING positions ending at the last stored row; older rows are stale garbage
                for (size_t i = 0; i < ring16.size(); ++i) ring16[i] = __float2half(frand(seed) * 9.0f);
                for (int p = std::max(0, last - RING + 1); p <= last; ++p)
                    std::memcpy(ring16.data() + size_t(p % RING) * kvw, kv16.data() + size_t(p) * kvw,
                                size_t(kvw) * sizeof(__half));
                CUDA_OK(cudaMemcpy(d_ring, ring16.data(), ring16.size() * sizeof(__half), cudaMemcpyHostToDevice));
                std::vector<float> qh(size_t(T) * NH * HD);
                for (auto& x : qh) x = frand(seed) * 1.5f;
                CUDA_OK(cudaMemcpy(d_q, qh.data(), qh.size() * sizeof(float), cudaMemcpyHostToDevice));
                CUDA_OK(cudaMemcpy(d_pos0, &pos0, sizeof(int), cudaMemcpyHostToDevice));
                std::vector<float> full(size_t(T) * NH * HD), ring(size_t(T) * NH * HD);
                bool ran = limbo::cuda::ink::attention(d_q, d_kv, d_kv, d_table, d_out, d_scratch, d_pos0, T, N_CTX,
                                                       0, NH, NKV, HD, scale, EXTENT, W, nullptr);
                CUDA_OK(cudaDeviceSynchronize());
                CUDA_OK(cudaMemcpy(full.data(), d_out, full.size() * sizeof(float), cudaMemcpyDeviceToHost));
                ran = limbo::cuda::ink::attention(d_q, d_ring, d_ring, d_table, d_out, d_scratch, d_pos0, T, N_CTX,
                                                  RING, NH, NKV, HD, scale, EXTENT, W, nullptr) && ran;
                CUDA_OK(cudaDeviceSynchronize());
                CUDA_OK(cudaMemcpy(ring.data(), d_out, ring.size() * sizeof(float), cudaMemcpyDeviceToHost));
                char what[160];
                std::snprintf(what, sizeof what, "ring %d == full cache, pos0 %d T %d window %d", RING, pos0, T, W);
                CHECK(ran && std::memcmp(full.data(), ring.data(), full.size() * sizeof(float)) == 0, what);
            }
        }
        CHECK(!limbo::cuda::ink::attention(d_q, d_ring, d_ring, d_table, d_out, d_scratch, d_pos0, 8, N_CTX, W + 6,
                                           NH, NKV, HD, scale, EXTENT, W, nullptr),
              "a ring shorter than window + T - 1 is refused");
        CHECK(!limbo::cuda::ink::attention(d_q, d_ring, d_ring, d_table, d_out, d_scratch, d_pos0, 1, N_CTX, RING,
                                           NH, NKV, HD, scale, EXTENT, 0, nullptr),
              "a ring without a window is refused");
        cudaFree(d_ring);
    }

    cudaFree(d_q);
    cudaFree(d_kv);
    cudaFree(d_table);
    cudaFree(d_out);
    cudaFree(d_scratch);
    cudaFree(d_pos0);
    return limbo::test::finish("cuda_ink_attn");
}
