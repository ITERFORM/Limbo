// Copyright (c) 2026 ITERFORM. All rights reserved.
// tests/cuda_decode_attn_test.cu — the decode attention (split flash-decode, src/cuda/attn_decode.cu) against a
// double-precision reference, from one position to 32k, for a single row and for verify windows of up to 8 rows.
//
//   1. accuracy: max |gpu - cpu| over every output element, bar 3e-6 (outputs are weighted means of N(0,1)-ish values),
//      at pos0 0, 3, 255, 256, 257, 1000, 4095, 16383 and 32760 (a window of 8 rows ending at the last position);
//   2. batch invariance: row t of a window is bit-identical to the single-row launch at position pos0 + t;
//   3. causality: the rows after the window hold arbitrary finite values and must not move the result.
//
// Pools: f16, Q8 and Q4 (a host quantiser; the Q4 reference rotates the output back). Layouts as the engine's (src/cuda/dec_ops.hpp): K [pos][kv head][256] f16, V [kv head][256][vpitch] f16,
// q [T][16][256], qfull [T][16][512] (q | gate). The device scratch is sized for 32768 positions and 8 rows.
// No GPU: SKIP, exit 0.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "../src/cuda/dec_ops.hpp"
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

int NH = 16;   // 16 on 2 KV heads: a group of 8 (rung 1); 24: a group of 12 (qwen4exp)
constexpr int NKV = 2, HD = 256, N_CTX = 32768;

// kv_rot_sign of include/limbo/cuda/kv.hpp (host copy) and R^T x = D H x / 16, the rotation the Q4 combine undoes
float rot_sign(int d) {
    uint32_t h = static_cast<uint32_t>(d) * 2654435761u;
    h ^= h >> 16;
    h *= 2246822519u;
    h ^= h >> 13;
    return (h & 1u) ? -1.0f : 1.0f;
}
void unrotate256(std::vector<double>& x) {
    std::vector<double> y(x.size());
    for (int h = 1; h < 256; h <<= 1) {
        for (int d = 0; d < 256; ++d) y[size_t(d)] = (d & h) ? x[size_t(d ^ h)] - x[size_t(d)] : x[size_t(d)] + x[size_t(d ^ h)];
        x = y;
    }
    for (int d = 0; d < 256; ++d) x[size_t(d)] *= 0.0625 * double(rot_sign(d));
}

float frand(uint32_t& s) {
    s = s * 1664525u + 1013904223u;
    return static_cast<float>(s >> 8) / 16777216.0f * 2.f - 1.f;
}

struct Case {
    int pos0, T;
};

void run(int fmt) {   // 0 f16, 1 Q8, 2 Q4 (its codes stand for rotated values: the reference rotates the output back)
    const bool q8 = fmt == 1, q4 = fmt == 2, quant = q8 || q4;
    const char* fname = q8 ? "Q8" : q4 ? "Q4" : "f16";
    const int vpitch = (N_CTX + 63) / 64 * 64;
    uint32_t seed = 987654321u;
    std::vector<__half> kc(static_cast<size_t>(N_CTX) * NKV * HD), vt(static_cast<size_t>(NKV) * HD * vpitch);
    for (auto& k : kc) k = __float2half(frand(seed) * 1.0f);
    for (auto& v : vt) v = __float2half(frand(seed) * 1.7f);
    std::vector<float> kf(kc.size()), vf(vt.size());
    for (size_t i = 0; i < kc.size(); ++i) kf[i] = __half2float(kc[i]);
    for (size_t i = 0; i < vt.size(); ++i) vf[i] = __half2float(vt[i]);

    // Q8: planes [pos][kv head][256] int8 and [pos][kv head][8] half scales for K and V (V gathered from the transposed layout),
    // quantised on the host the way kv_quant_lane does (amax / 127, round to nearest even); the reference reads the dequantised values.
    std::vector<int8_t> kqh, vqh;
    std::vector<uint16_t> ksh, vsh;
    if (quant) {
        const size_t rows = static_cast<size_t>(N_CTX) * NKV;
        const size_t rb = q4 ? HD / 2 : HD;   // bytes of a pool row
        kqh.resize(rows * rb);
        vqh.resize(rows * rb);
        ksh.resize(rows * (HD / 32));
        vsh.resize(rows * (HD / 32));
        for (int p = 0; p < N_CTX; ++p)
            for (int kh = 0; kh < NKV; ++kh)
                for (int b = 0; b < HD / 32; ++b) {
                    for (int which = 0; which < 2; ++which) {
                        float x[32], amax = 0;
                        for (int i = 0; i < 32; ++i) {
                            const int d = b * 32 + i;
                            x[i] = which == 0 ? kf[(static_cast<size_t>(p) * NKV + kh) * HD + d] : vf[(static_cast<size_t>(kh) * HD + d) * vpitch + p];
                            amax = std::max(amax, std::fabs(x[i]));
                        }
                        float dd = amax / 127.0f, inv = dd > 0.f ? 127.0f / amax : 0.f;
                        if (q4) {   // q4_0: the block's value of largest magnitude (the larger on a tie) over -8
                            float m = x[0];
                            for (int i = 1; i < 32; ++i)
                                if (std::fabs(x[i]) > std::fabs(m) || (std::fabs(x[i]) == std::fabs(m) && x[i] > m)) m = x[i];
                            dd = m / -8.0f;
                            inv = dd != 0.f ? 1.0f / dd : 0.f;
                        }
                        const uint16_t dh = __half_as_ushort(__float2half_rn(dd));
                        const float dq = __half2float(__ushort_as_half(dh));
                        const size_t row = static_cast<size_t>(p) * NKV + kh;
                        (which == 0 ? ksh : vsh)[row * (HD / 32) + b] = dh;
                        for (int i = 0; i < 32; ++i) {
                            int qv = static_cast<int>(std::nearbyint(x[i] * inv));
                            if (q4) {
                                qv = std::min(15, std::max(0, qv + 8)) - 8;
                                uint8_t* by = reinterpret_cast<uint8_t*>((which == 0 ? kqh : vqh).data()) + row * rb + (b * 32 + i) / 2;
                                if (i % 2 == 0) *by = 0;
                                *by = static_cast<uint8_t>(*by | ((qv + 8) << (4 * (i % 2))));
                            } else {
                                (which == 0 ? kqh : vqh)[row * HD + b * 32 + i] = static_cast<int8_t>(qv);
                            }
                            const float back = static_cast<float>(qv) * dq;
                            if (which == 0) kf[(static_cast<size_t>(p) * NKV + kh) * HD + b * 32 + i] = back;
                            else vf[(static_cast<size_t>(kh) * HD + b * 32 + i) * vpitch + p] = back;
                        }
                    }
                }
    }
    __half *dk = nullptr, *dv = nullptr;
    int8_t *dkq = nullptr, *dvq = nullptr;
    uint16_t *dks = nullptr, *dvs = nullptr;
    float *dq, *dqf, *dout, *dscr;
    int* dpos;
    if (quant) {
        CUDA_OK(cudaMalloc(&dkq, kqh.size()));
        CUDA_OK(cudaMalloc(&dvq, vqh.size()));
        CUDA_OK(cudaMalloc(&dks, ksh.size() * 2));
        CUDA_OK(cudaMalloc(&dvs, vsh.size() * 2));
        CUDA_OK(cudaMemcpy(dkq, kqh.data(), kqh.size(), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(dvq, vqh.data(), vqh.size(), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(dks, ksh.data(), ksh.size() * 2, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(dvs, vsh.data(), vsh.size() * 2, cudaMemcpyHostToDevice));
    } else {
        CUDA_OK(cudaMalloc(&dk, kc.size() * 2));
        CUDA_OK(cudaMalloc(&dv, vt.size() * 2));
        CUDA_OK(cudaMemcpy(dk, kc.data(), kc.size() * 2, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(dv, vt.data(), vt.size() * 2, cudaMemcpyHostToDevice));
    }
    CUDA_OK(cudaMalloc(&dq, size_t(8) * NH * HD * 4));
    CUDA_OK(cudaMalloc(&dqf, size_t(8) * NH * 2 * HD * 4));
    CUDA_OK(cudaMalloc(&dout, size_t(8) * NH * HD * 4));
    const size_t scr = limbo::cuda::dec::attn_scratch_floats(N_CTX, 8, NH);
    CUDA_OK(cudaMalloc(&dscr, scr * 4));
    CUDA_OK(cudaMalloc(&dpos, 64));

    const Case cases[] = {{0, 1}, {3, 1}, {255, 1}, {256, 1}, {257, 3}, {1000, 8}, {4095, 2}, {16383, 1}, {32760, 8}};
    const float scale = 0.5f;  // score std about 1.6: a real softmax, not a flat one
    double worst = 0;
    bool all_inv = true;
    for (const Case& cs : cases) {
        const int T = cs.T, pos0 = cs.pos0;
        std::vector<float> q(size_t(T) * NH * HD), qfull(size_t(T) * NH * 2 * HD);
        for (int t = 0; t < T; ++t)
            for (int h = 0; h < NH; ++h)
                for (int d = 0; d < HD; ++d) {
                    const float x = frand(seed) * 0.6f;  // scores of a few units: a sharp but not one-hot softmax
                    q[(size_t(t) * NH + h) * HD + d] = x;
                    qfull[(size_t(t) * NH + h) * 2 * HD + d] = x;
                    qfull[(size_t(t) * NH + h) * 2 * HD + HD + d] = frand(seed) * 2.f;
                }
        CUDA_OK(cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(dqf, qfull.data(), qfull.size() * 4, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(dpos, &pos0, 4, cudaMemcpyHostToDevice));
        auto launch = [&](const float* qq, const float* qf, int TT) {
            return q8   ? limbo::cuda::dec::attention_q8(qq, dkq, dks, dvq, dvs, qf, dout, dscr, dpos, TT, N_CTX, NH, NKV, HD, scale, nullptr)
                   : q4 ? limbo::cuda::dec::attention_q4(qq, dkq, dks, dvq, dvs, qf, dout, dscr, dpos, TT, N_CTX, NH, NKV, HD, scale, nullptr)
                        : limbo::cuda::dec::attention(qq, dk, dv, qf, dout, dscr, dpos, TT, N_CTX, vpitch, NH, NKV, HD, scale, nullptr);
        };
        if (!launch(dq, dqf, T)) {
            std::printf("FAIL attention launch (pos0 %d T %d)\n", pos0, T);
            ++limbo::test::failures();
            return;
        }
        CUDA_OK(cudaDeviceSynchronize());
        std::vector<float> got(size_t(T) * NH * HD);
        CUDA_OK(cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost));

        // double-precision reference, one thread per (row, head)
        std::vector<double> errs(size_t(T) * NH, 0.0), mxs(size_t(T) * NH, 0.0);
        auto ref_head = [&](int t, int h) {
            const int pos = pos0 + t;
            const int kh = h / (NH / NKV);
            const float* qh = &q[(size_t(t) * NH + h) * HD];
            std::vector<double> sc(size_t(pos + 1), 0.0);
            double m = -1e300;
            for (int p = 0; p <= pos; ++p) {
                const float* k = &kf[(size_t(p) * NKV + kh) * HD];
                double a = 0;
                for (int d = 0; d < HD; ++d) a += double(qh[d]) * double(k[d]);
                sc[size_t(p)] = a * double(scale);
                m = std::max(m, sc[size_t(p)]);
            }
            double l = 0;
            for (int p = 0; p <= pos; ++p) {
                sc[size_t(p)] = std::exp(sc[size_t(p)] - m);
                l += sc[size_t(p)];
            }
            double e = 0, mx1 = 0;
            std::vector<double> ov(HD);
            for (int d = 0; d < HD; ++d) {
                const float* v = &vf[(size_t(kh) * HD + d) * vpitch];
                double o = 0;
                for (int p = 0; p <= pos; ++p) o += sc[size_t(p)] * double(v[p]);
                ov[size_t(d)] = o / l;
            }
            if (q4) unrotate256(ov);
            for (int d = 0; d < HD; ++d) {
                const double g = 1.0 / (1.0 + std::exp(-double(qfull[(size_t(t) * NH + h) * 2 * HD + HD + d])));
                const double ref = ov[size_t(d)] * g;
                e = std::max(e, std::fabs(double(got[(size_t(t) * NH + h) * HD + d]) - ref));
                mx1 = std::max(mx1, std::fabs(ref));
            }
            errs[size_t(t) * NH + h] = e;
            mxs[size_t(t) * NH + h] = mx1;
        };
        {
            std::vector<std::thread> th;
            for (int t = 0; t < T; ++t)
                for (int h = 0; h < NH; ++h) th.emplace_back(ref_head, t, h);
            for (auto& x : th) x.join();
        }
        const double err = *std::max_element(errs.begin(), errs.end()), mx = *std::max_element(mxs.begin(), mxs.end());
        worst = std::max(worst, err);
        char what[160];
        std::snprintf(what, sizeof what, "%s attention G%d pos0 %5d, %d row(s): max|gpu - cpu| %.2e (max|out| %.2f) <= 3e-6", fname, NH / NKV, pos0, T, err, mx);
        CHECK(err <= 3e-6, what);

        // the same rows one at a time: bit-identical
        bool inv = true;
        for (int t = 0; t < T && T > 1; ++t) {
            const int p1 = pos0 + t;
            CUDA_OK(cudaMemcpy(dpos, &p1, 4, cudaMemcpyHostToDevice));
            if (!launch(dq + size_t(t) * NH * HD, dqf + size_t(t) * NH * 2 * HD, 1)) {
                std::printf("FAIL single-row launch\n");
                ++limbo::test::failures();
                return;
            }
            std::vector<float> one(size_t(NH) * HD);
            CUDA_OK(cudaMemcpy(one.data(), dout, one.size() * 4, cudaMemcpyDeviceToHost));
            if (std::memcmp(one.data(), &got[size_t(t) * NH * HD], one.size() * 4) != 0) inv = false;
        }
        if (T > 1) {
            all_inv = all_inv && inv;
            std::snprintf(what, sizeof what, "%s attention G%d pos0 %5d: each of the %d rows bit-identical to its single-row launch", fname, NH / NKV, pos0, T);
            CHECK(inv, what);
        }
    }
    std::printf("worst |gpu - cpu| over all cases: %.3e; row/window bit-identity: %s\n", worst, all_inv ? "yes" : "NO");
    if (dk) cudaFree(dk);
    if (dv) cudaFree(dv);
    if (dkq) cudaFree(dkq);
    if (dvq) cudaFree(dvq);
    if (dks) cudaFree(dks);
    if (dvs) cudaFree(dvs);
    cudaFree(dq);
    cudaFree(dqf);
    cudaFree(dout);
    cudaFree(dscr);
    cudaFree(dpos);
}

}  // namespace

int main() {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        std::printf("SKIP cuda_decode_attn_test: no CUDA device\n");
        return 0;
    }
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    if (free_b < (size_t(1) << 30)) {
        std::printf("SKIP cuda_decode_attn_test: only %.2f GB of VRAM free\n", double(free_b) / 1e9);
        return 0;
    }
    run(0);
    run(1);
    run(2);
    NH = 24;   // qwen4exp's group of 12: f16 as before, Q8 through the scalar scorer's second band (Q4 has no group-12 scorer)
    run(0);
    run(1);
    NH = 12;   // the 27Bs' group of 6
    run(0);
    run(1);
    run(2);
    return limbo::test::finish("cuda_decode_attn_test");
}
