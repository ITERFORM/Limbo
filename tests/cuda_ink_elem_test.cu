// Copyright (c) 2026 ITERFORM. All rights reserved.
// tests/cuda_ink_elem_test.cu — the inkling2 short convolutions and L2 clips (src/cuda/ink_elem.cu) against a
// double-precision reference, including the recurrent state's shift over several consecutive tokens.
//
//   1. conv (static and with dynamic taps): the output and the shifted history after each of 64 tokens match
//      the reference to 2e-6, with the same state-evolution structure as the CPU step's (docs/inkling2.md §2);
//   2. clip: with and without the per-channel scale, the factor and every element match to 2e-6;
//   3. determinism: a second launch over the same inputs is bit-identical.
// No GPU: SKIP, exit 0.
#include <cuda_runtime.h>

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
            return limbo::test::finish("cuda_ink_elem");                                             \
        }                                                                                            \
    } while (0)

constexpr int SCONV = 4, NCH = 1024, NHEAD = 8, NSC = 2048, OFFSET = 512;

float frand(uint32_t& s) {
    s = s * 1664525u + 1013904223u;
    return static_cast<float>(s >> 8) / 16777216.0f * 2.f - 1.f;
}

// The reference recurrence over the same history layout the kernel sees.
void ref_conv(const std::vector<double>& cur, const std::vector<double>& dyn, const std::vector<double>& w,
              std::vector<double>& out, std::vector<double>& hist, int n_ch, int n_heads, int nsc, int sconv,
              bool use_dyn) {
    const double inv = 1.0 / std::sqrt(double((use_dyn ? 2 : 1) * sconv + 1));
    out.assign(size_t(n_ch), 0.0);
    for (int ch = 0; ch < n_ch; ++ch) {
        double sum = 0;
        for (int j = 0; j < sconv; ++j) {
            const double xv = j < sconv - 1 ? hist[size_t(j) * nsc + OFFSET + ch] : cur[size_t(ch)];
            sum += w[size_t(ch) * sconv + j] * xv;
        }
        if (use_dyn) {
            const int per_head = n_ch / n_heads;
            const int hd = ch / per_head;
            for (int lag = 0; lag < sconv; ++lag) {
                const int xr = sconv - 1 - lag;
                const double xv = xr < sconv - 1 ? hist[size_t(xr) * nsc + OFFSET + ch] : cur[size_t(ch)];
                sum += dyn[size_t(hd) * sconv + lag] * xv;
            }
        }
        out[size_t(ch)] = (cur[size_t(ch)] + sum) * inv;
    }
    for (int ch = 0; ch < n_ch; ++ch)
        for (int j = 0; j < sconv - 1; ++j) {
            const double v = j < sconv - 2 ? hist[size_t(j + 1) * nsc + OFFSET + ch] : cur[size_t(ch)];
            hist[size_t(j) * nsc + OFFSET + ch] = v;
        }
}

}  // namespace

int main() {
    int dev = 0;
    if (cudaGetDeviceCount(&dev) != cudaSuccess || dev == 0) {
        std::printf("cuda_ink_elem_test: SKIP (no GPU)\n");
        return limbo::test::finish("cuda_ink_elem");
    }

    uint32_t seed = 2468013579u;

    // ---- conv, static and dynamic, over 64 consecutive tokens ----
    for (int use_dyn = 0; use_dyn <= 1; ++use_dyn) {
        std::vector<double> w(size_t(NCH) * SCONV), hist(size_t(SCONV - 1) * NSC, 0.0), dyn(size_t(NHEAD) * SCONV);
        std::vector<float> wf(w.size()), histf(hist.size(), 0.0f), dynf(dyn.size());
        for (size_t i = 0; i < w.size(); ++i) { w[i] = frand(seed) * 0.8; wf[i] = float(w[i]); }
        for (size_t i = 0; i < dyn.size(); ++i) { dyn[i] = frand(seed) * 0.5; dynf[i] = float(dyn[i]); }

        float *d_cur = nullptr, *d_w = nullptr, *d_hist = nullptr, *d_out = nullptr, *d_dyn = nullptr;
        CUDA_OK(cudaMalloc(&d_cur, NCH * sizeof(float)));
        CUDA_OK(cudaMalloc(&d_w, w.size() * sizeof(float)));
        CUDA_OK(cudaMalloc(&d_hist, hist.size() * sizeof(float)));
        CUDA_OK(cudaMalloc(&d_out, NCH * sizeof(float)));
        CUDA_OK(cudaMalloc(&d_dyn, dyn.size() * sizeof(float)));
        CUDA_OK(cudaMemcpy(d_w, wf.data(), w.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_dyn, dynf.data(), dyn.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_hist, histf.data(), hist.size() * sizeof(float), cudaMemcpyHostToDevice));

        double err_out = 0, err_state = 0;
        std::vector<float> got(NCH);
        {
            // determinism first, from the pristine zero state: two runs over the same input are bit-identical
            std::vector<float> curf(NCH), first(NCH), zeros(hist.size(), 0.0f);
            for (auto& x : curf) x = frand(seed) * 1.2f;
            CUDA_OK(cudaMemcpy(d_cur, curf.data(), NCH * sizeof(float), cudaMemcpyHostToDevice));
            for (int run = 0; run < 2; ++run) {
                CUDA_OK(cudaMemcpy(d_hist, zeros.data(), hist.size() * sizeof(float), cudaMemcpyHostToDevice));
                CUDA_OK(cudaMemset(d_out, 0, NCH * sizeof(float)));
                limbo::cuda::ink::conv(SCONV, d_cur, use_dyn ? d_dyn : nullptr, d_w, d_out, d_hist, OFFSET,
                                       NCH, NHEAD, NSC, 1, nullptr);
                CUDA_OK(cudaDeviceSynchronize());
                CUDA_OK(cudaMemcpy(got.data(), d_out, NCH * sizeof(float), cudaMemcpyDeviceToHost));
                if (run == 0) first = got;
            }
            CHECK(std::memcmp(first.data(), got.data(), NCH * sizeof(float)) == 0, "conv determinism");
            CUDA_OK(cudaMemcpy(d_hist, zeros.data(), hist.size() * sizeof(float), cudaMemcpyHostToDevice));
        }
        for (int step = 0; step < 64; ++step) {
            std::vector<double> cur(NCH);
            std::vector<float> curf(NCH);
            for (int i = 0; i < NCH; ++i) { cur[size_t(i)] = frand(seed) * 1.2; curf[size_t(i)] = float(cur[size_t(i)]); }
            CUDA_OK(cudaMemcpy(d_cur, curf.data(), NCH * sizeof(float), cudaMemcpyHostToDevice));

            std::vector<double> ref;
            ref_conv(cur, dyn, w, ref, hist, NCH, NHEAD, NSC, SCONV, use_dyn != 0);
            if (!limbo::cuda::ink::conv(SCONV, d_cur, use_dyn ? d_dyn : nullptr, d_w, d_out, d_hist, OFFSET,
                                        NCH, NHEAD, NSC, 1, nullptr)) {
                std::printf("FAIL conv() refused step %d\n", step);
                ++limbo::test::failures();
                break;
            }
            CUDA_OK(cudaDeviceSynchronize());
            CUDA_OK(cudaMemcpy(got.data(), d_out, NCH * sizeof(float), cudaMemcpyDeviceToHost));
            for (int i = 0; i < NCH; ++i)
                err_out = std::max(err_out, std::fabs(double(got[size_t(i)]) - ref[size_t(i)]));

            // the state after the token: the shifted history
            std::vector<float> ghist(hist.size());
            CUDA_OK(cudaMemcpy(ghist.data(), d_hist, hist.size() * sizeof(float), cudaMemcpyDeviceToHost));
            for (int j = 0; j < SCONV - 1; ++j)
                for (int ch = 0; ch < NCH; ++ch)
                    err_state = std::max(err_state, std::fabs(double(ghist[size_t(j) * NSC + OFFSET + ch])
                                                       - hist[size_t(j) * NSC + OFFSET + ch]));
        }
        char what[128];
        std::snprintf(what, sizeof what, "conv %s outputs over 64 tokens (max|d| %.2e)",
                      use_dyn ? "dynamic" : "static", err_out);
        CHECK(err_out <= 2e-6, what);
        std::snprintf(what, sizeof what, "conv %s state shift (max|d| %.2e)", use_dyn ? "dynamic" : "static",
                      err_state);
        CHECK(err_state <= 2e-6, what);

        cudaFree(d_cur);
        cudaFree(d_w);
        cudaFree(d_hist);
        cudaFree(d_out);
        cudaFree(d_dyn);
    }

    // ---- clip, with and without the per-channel scale ----
    for (int with_scale = 0; with_scale <= 1; ++with_scale) {
        constexpr int ROW = 128, NH = 24;
        std::vector<float> v(size_t(NH) * ROW), sc(size_t(NH) * ROW);
        for (auto& x : v) x = frand(seed) * 2.0f;
        for (auto& x : sc) x = frand(seed) * 1.5f;
        float *d_v = nullptr, *d_sc = nullptr;
        CUDA_OK(cudaMalloc(&d_v, v.size() * sizeof(float)));
        CUDA_OK(cudaMalloc(&d_sc, sc.size() * sizeof(float)));
        CUDA_OK(cudaMemcpy(d_v, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_sc, sc.data(), sc.size() * sizeof(float), cudaMemcpyHostToDevice));

        std::vector<double> ref(v.size());
        for (int h = 0; h < NH; ++h) {
            double ss = 0;
            for (int i = 0; i < ROW; ++i) {
                double x = v[size_t(h) * ROW + i];
                if (with_scale) x *= sc[size_t(h) * ROW + i];
                ref[size_t(h) * ROW + i] = x;
                ss += x * x;
            }
            const double factor = std::min(std::max(8.0 / (std::sqrt(ss) + 1e-6), 0.0), 1.0);
            for (int i = 0; i < ROW; ++i) ref[size_t(h) * ROW + i] *= factor;
        }
        if (!limbo::cuda::ink::clip(d_v, with_scale ? d_sc : nullptr, NH, ROW, 8.0f, 1, nullptr)) {
            std::printf("FAIL clip() refused\n");
            ++limbo::test::failures();
        } else {
            CUDA_OK(cudaDeviceSynchronize());
            std::vector<float> got(v.size());
            CUDA_OK(cudaMemcpy(got.data(), d_v, v.size() * sizeof(float), cudaMemcpyDeviceToHost));
            double err = 0;
            for (size_t i = 0; i < v.size(); ++i) err = std::max(err, std::fabs(double(got[i]) - ref[i]));
            char what[128];
            std::snprintf(what, sizeof what, "clip with scale %d (max|d| %.2e)", with_scale, err);
            CHECK(err <= 2e-6, what);
        }
        cudaFree(d_v);
        cudaFree(d_sc);
    }

    return limbo::test::finish("cuda_ink_elem");
}
