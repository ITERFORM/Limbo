// Copyright (c) 2026 ITERFORM. All rights reserved.
// tests/cuda_ink_flow_test.cu — the inkling2 depth mix and loop controller (src/cuda/ink_flow.cu) against a
// double-precision reference (docs/inkling2.md §2/§5).
//
//   1. mix: for n_s 1..5 states and q_count 1 and 4, every out element matches to 2e-6;
//   2. control: the gate buffer and the combined output match to 2e-6;
//   3. determinism: a second identical launch is bit-identical.
// No GPU: SKIP, exit 0.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../src/cuda/ink_flow.hpp"
#include "limbo/check.hpp"

namespace {

#define CUDA_OK(call)                                                                                \
    do {                                                                                             \
        cudaError_t e_ = (call);                                                                     \
        if (e_ != cudaSuccess) {                                                                     \
            std::printf("FAIL %s: %s (%s:%d)\n", #call, cudaGetErrorString(e_), __FILE__, __LINE__);  \
            ++limbo::test::failures();                                                               \
            return limbo::test::finish("cuda_ink_flow");                                             \
        }                                                                                            \
    } while (0)

constexpr int E = 2304;

float frand(uint32_t& s) {
    s = s * 1664525u + 1013904223u;
    return static_cast<float>(s >> 8) / 16777216.0f * 2.f - 1.f;
}

}  // namespace

int main() {
    int dev = 0;
    if (cudaGetDeviceCount(&dev) != cudaSuccess || dev == 0) {
        std::printf("cuda_ink_flow_test: SKIP (no GPU)\n");
        return limbo::test::finish("cuda_ink_flow");
    }
    uint32_t seed = 1122334455u;

    for (int n_s = 1; n_s <= 5; ++n_s) {
        for (int q_count : {1, 4}) {
            std::vector<float> st(size_t(n_s) * E), qv(size_t(q_count) * E);
            for (auto& x : st) x = frand(seed) * 1.3f;
            for (auto& x : qv) x = frand(seed) * 1.3f;
            // the reference
            std::vector<double> invs(n_s), scores(size_t(n_s) * q_count);
            for (int s = 0; s < n_s; ++s) {
                double ss = 0;
                for (int i = 0; i < E; ++i) ss += double(st[size_t(s) * E + i]) * double(st[size_t(s) * E + i]);
                invs[size_t(s)] = 1.0 / std::sqrt(ss / E + 1e-6);
            }
            for (int q = 0; q < q_count; ++q)
                for (int s = 0; s < n_s; ++s) {
                    double dot = 0;
                    for (int i = 0; i < E; ++i)
                        dot += double(st[size_t(s) * E + i]) * invs[size_t(s)] * double(qv[size_t(q) * E + i]);
                    scores[size_t(s) * q_count + q] = dot / std::sqrt(double(E));
                }
            std::vector<double> ref(size_t(q_count) * E);
            for (int q = 0; q < q_count; ++q) {
                double m = -1e300, sum = 0;
                std::vector<double> p(n_s);
                for (int s = 0; s < n_s; ++s) m = std::max(m, scores[size_t(s) * q_count + q]);
                for (int s = 0; s < n_s; ++s) { p[size_t(s)] = std::exp(scores[size_t(s) * q_count + q] - m); sum += p[size_t(s)]; }
                for (int s = 0; s < n_s; ++s)
                    for (int i = 0; i < E; ++i)
                        ref[size_t(q) * E + i] += p[size_t(s)] / sum * double(st[size_t(s) * E + i]) * invs[size_t(s)];
            }
            float *d_st = nullptr, *d_q = nullptr, *d_out = nullptr;
            CUDA_OK(cudaMalloc(&d_st, st.size() * sizeof(float)));
            CUDA_OK(cudaMalloc(&d_q, qv.size() * sizeof(float)));
            CUDA_OK(cudaMalloc(&d_out, size_t(q_count) * E * sizeof(float)));
            CUDA_OK(cudaMemcpy(d_st, st.data(), st.size() * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_OK(cudaMemcpy(d_q, qv.data(), qv.size() * sizeof(float), cudaMemcpyHostToDevice));
            if (!limbo::cuda::ink::mix(d_st, d_q, d_out, n_s, E, q_count, nullptr)) {
                std::printf("FAIL mix() refused n_s %d q %d\n", n_s, q_count);
                ++limbo::test::failures();
            } else {
                CUDA_OK(cudaDeviceSynchronize());
                std::vector<float> got(size_t(q_count) * E);
                CUDA_OK(cudaMemcpy(got.data(), d_out, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
                double err = 0;
                for (size_t i = 0; i < got.size(); ++i) err = std::max(err, std::fabs(double(got[i]) - ref[i]));
                char what[128];
                std::snprintf(what, sizeof what, "mix n_s %d q_count %d (max|d| %.2e)", n_s, q_count, err);
                CHECK(err <= 2e-6, what);
                // determinism
                std::vector<float> first = got;
                CUDA_OK(cudaMemset(d_out, 0, got.size() * sizeof(float)));
                limbo::cuda::ink::mix(d_st, d_q, d_out, n_s, E, q_count, nullptr);
                CUDA_OK(cudaDeviceSynchronize());
                CUDA_OK(cudaMemcpy(got.data(), d_out, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
                CHECK(std::memcmp(first.data(), got.data(), got.size() * sizeof(float)) == 0, "mix determinism");
            }
            cudaFree(d_st);
            cudaFree(d_q);
            cudaFree(d_out);
        }
    }

    // ---- control ----
    {
        std::vector<float> cur(E), msg(E), gain(E), forget(E);
        for (auto& x : cur) x = frand(seed) * 1.1f;
        for (auto& x : msg) x = frand(seed) * 1.1f;
        for (auto& x : gain) x = frand(seed) * 1.1f;
        for (auto& x : forget) x = frand(seed) * 1.1f;
        std::vector<double> refg(size_t(2) * E), refo(E);
        for (int half = 0; half < 2; ++half) {
            const std::vector<float>& x = half == 0 ? cur : msg;
            double ss = 0;
            for (int i = 0; i < E; ++i) ss += double(x[size_t(i)]) * double(x[size_t(i)]);
            const double inv = 1.0 / std::sqrt(ss / E + 1e-6);
            for (int i = 0; i < E; ++i) refg[size_t(half) * E + i] = double(x[size_t(i)]) * inv;
        }
        for (int i = 0; i < E; ++i)
            refo[size_t(i)] = double(cur[size_t(i)]) - std::tanh(double(forget[size_t(i)])) * double(cur[size_t(i)])
                              + std::tanh(double(gain[size_t(i)])) * double(msg[size_t(i)]);
        float *d_cur = nullptr, *d_msg = nullptr, *d_gain = nullptr, *d_forget = nullptr, *d_out = nullptr,
              *d_gate = nullptr;
        CUDA_OK(cudaMalloc(&d_cur, E * sizeof(float)));
        CUDA_OK(cudaMalloc(&d_msg, E * sizeof(float)));
        CUDA_OK(cudaMalloc(&d_gain, E * sizeof(float)));
        CUDA_OK(cudaMalloc(&d_forget, E * sizeof(float)));
        CUDA_OK(cudaMalloc(&d_out, E * sizeof(float)));
        CUDA_OK(cudaMalloc(&d_gate, size_t(2) * E * sizeof(float)));
        CUDA_OK(cudaMemcpy(d_cur, cur.data(), E * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_msg, msg.data(), E * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_gain, gain.data(), E * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_forget, forget.data(), E * sizeof(float), cudaMemcpyHostToDevice));
        if (!limbo::cuda::ink::control_gate(d_cur, d_msg, d_gate, E, nullptr) ||
            !limbo::cuda::ink::control_combine(d_cur, d_msg, d_gain, d_forget, d_out, E, nullptr)) {
            std::printf("FAIL control() refused\n");
            ++limbo::test::failures();
        } else {
            CUDA_OK(cudaDeviceSynchronize());
            std::vector<float> gotg(size_t(2) * E), goto_(E);
            CUDA_OK(cudaMemcpy(gotg.data(), d_gate, gotg.size() * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_OK(cudaMemcpy(goto_.data(), d_out, E * sizeof(float), cudaMemcpyDeviceToHost));
            double errg = 0, erro = 0;
            for (size_t i = 0; i < gotg.size(); ++i) errg = std::max(errg, std::fabs(double(gotg[i]) - refg[i]));
            for (int i = 0; i < E; ++i) erro = std::max(erro, std::fabs(double(goto_[size_t(i)]) - refo[size_t(i)]));
            char what[128];
            std::snprintf(what, sizeof what, "control gate buffer (max|d| %.2e)", errg);
            CHECK(errg <= 2e-6, what);
            std::snprintf(what, sizeof what, "control combine (max|d| %.2e)", erro);
            CHECK(erro <= 2e-6, what);
        }
        cudaFree(d_cur);
        cudaFree(d_msg);
        cudaFree(d_gain);
        cudaFree(d_forget);
        cudaFree(d_out);
        cudaFree(d_gate);
    }

    return limbo::test::finish("cuda_ink_flow");
}
