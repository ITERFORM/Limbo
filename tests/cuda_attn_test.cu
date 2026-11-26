// Copyright (c) 2026 ITERFORM. All rights reserved.
// tests/cuda_attn_test.cu — the tensor-core prompt attention (ops::attention_rows) against a double-precision
// reference, on random queries and a random f16 KV cache, for chunks at the start of a sequence and after a prefix.
//
// Layouts as the engine's: K [pos][kv head][256] f16, V transposed [kv head][256][vpitch] f16, q [T][16][256]
// fp32, qfull [T][16][512] (q | gate). Bar: max |gpu - cpu| <= 2e-6 * max|cpu| (the products are fp32-accurate:
// the activations go through the tensor cores as high + low halves). With exact = false (the 1-plane mode) Q and P are rounded
// to f16 once, like llama.cpp's flash attention: bar 2e-3. Then the batched delta-net scan
// (ops::gdn_coef_rows + ops::gdn_scan_rows over T tokens) against the decode kernel run token by token: the state
// and every output must be equal bit for bit. No GPU: SKIP, exit 0.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../src/cuda/ops.hpp"
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

float frand(uint32_t& s) {
    s = s * 1664525u + 1013904223u;
    return static_cast<float>(s >> 8) / 16777216.0f * 2.f - 1.f;
}

void run_case(int pos0, int T, int n_ctx, bool exact = true) {
    constexpr int NH = 16, NKV = 2, HD = 256;
    const int vpitch = (n_ctx + 63) / 64 * 64;
    const int len = pos0 + T;
    uint32_t seed = 12345u + static_cast<uint32_t>(pos0 * 7 + T);
    std::vector<float> q(static_cast<size_t>(T) * NH * HD), qfull(static_cast<size_t>(T) * NH * 2 * HD);
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < NH; ++h)
            for (int d = 0; d < HD; ++d) {
                const float x = frand(seed) * 3.f;  // scores of a few units, like a real head
                q[(static_cast<size_t>(t) * NH + h) * HD + d] = x;
                qfull[(static_cast<size_t>(t) * NH + h) * 2 * HD + d] = x;
                qfull[(static_cast<size_t>(t) * NH + h) * 2 * HD + HD + d] = frand(seed) * 2.f;
            }
    std::vector<__half> kc(static_cast<size_t>(n_ctx) * NKV * HD, __float2half(0.f)), vt(static_cast<size_t>(NKV) * HD * vpitch, __float2half(0.f));
    std::vector<float> kf(kc.size()), vf(vt.size());
    for (int p = 0; p < len; ++p)
        for (int kh = 0; kh < NKV; ++kh)
            for (int d = 0; d < HD; ++d) {
                const __half k = __float2half(frand(seed)), v = __float2half(frand(seed));
                kc[(static_cast<size_t>(p) * NKV + kh) * HD + d] = k;
                vt[(static_cast<size_t>(kh) * HD + d) * vpitch + p] = v;
            }
    // positions past the end hold arbitrary finite values: the causal mask must keep them out
    for (int p = len; p < n_ctx; ++p)
        for (int kh = 0; kh < NKV; ++kh)
            for (int d = 0; d < HD; ++d) {
                kc[(static_cast<size_t>(p) * NKV + kh) * HD + d] = __float2half(frand(seed) * 4.f);
                vt[(static_cast<size_t>(kh) * HD + d) * vpitch + p] = __float2half(frand(seed) * 4.f);
            }

    float *dq, *dqf, *dout;
    __half *dk, *dv;
    CUDA_OK(cudaMalloc(&dq, q.size() * 4));
    CUDA_OK(cudaMalloc(&dqf, qfull.size() * 4));
    CUDA_OK(cudaMalloc(&dout, q.size() * 4));
    CUDA_OK(cudaMalloc(&dk, kc.size() * 2));
    CUDA_OK(cudaMalloc(&dv, vt.size() * 2));
    CUDA_OK(cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dqf, qfull.data(), qfull.size() * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dk, kc.data(), kc.size() * 2, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dv, vt.data(), vt.size() * 2, cudaMemcpyHostToDevice));
    limbo::cuda::ops::attention_rows(dq, dk, dv, dqf, dout, pos0, T, n_ctx, vpitch, NH, NKV, HD, 1.0f / 16.0f, nullptr, exact);
    CUDA_OK(cudaDeviceSynchronize());
    std::vector<float> out(q.size());
    CUDA_OK(cudaMemcpy(out.data(), dout, out.size() * 4, cudaMemcpyDeviceToHost));

    double worst = 0, scale_max = 0;
    std::vector<double> sc(static_cast<size_t>(len));
    for (int t = 0; t < T; t += (T > 40 ? 7 : 1))
        for (int h = 0; h < NH; h += 3) {
            const int kh = h / (NH / NKV), pos = pos0 + t;
            double m = -1e300;
            for (int p = 0; p <= pos; ++p) {
                double s = 0;
                for (int d = 0; d < HD; ++d)
                    s += double(q[(static_cast<size_t>(t) * NH + h) * HD + d]) * double(__half2float(kc[(static_cast<size_t>(p) * NKV + kh) * HD + d]));
                sc[static_cast<size_t>(p)] = s / 16.0;
                m = std::max(m, sc[static_cast<size_t>(p)]);
            }
            double sum = 0;
            for (int p = 0; p <= pos; ++p) {
                sc[static_cast<size_t>(p)] = std::exp(sc[static_cast<size_t>(p)] - m);
                sum += sc[static_cast<size_t>(p)];
            }
            for (int d = 0; d < HD; ++d) {
                double o = 0;
                for (int p = 0; p <= pos; ++p) o += sc[static_cast<size_t>(p)] / sum * double(__half2float(vt[(static_cast<size_t>(kh) * HD + d) * vpitch + p]));
                const double gate = 1.0 / (1.0 + std::exp(-double(qfull[(static_cast<size_t>(t) * NH + h) * 2 * HD + HD + d])));
                const double ref = o * gate;
                worst = std::max(worst, std::fabs(ref - double(out[(static_cast<size_t>(t) * NH + h) * HD + d])));
                scale_max = std::max(scale_max, std::fabs(ref));
            }
        }
    std::printf("     pos0 %4d, T %3d, ctx %d: max |gpu - cpu| = %.3e (max |out| %.3e, ratio %.2e)\n", pos0, T, n_ctx, worst, scale_max, worst / scale_max);
    char what[96];
    std::snprintf(what, sizeof what, "attention%s pos0 %d T %d within %s absolute", exact ? "" : " (f16 Q and P)", pos0, T, exact ? "2e-6" : "2e-3");
    CHECK(worst <= (exact ? 2e-6 : 2e-3), what);
    cudaFree(dq);
    cudaFree(dqf);
    cudaFree(dout);
    cudaFree(dk);
    cudaFree(dv);
}

// Windows of keys: attention_rows_window over three windows of the cache (pointers offset into the same arrays) equals the
// single pass bit for bit; and the Q8 cache read (kv_expand_window + windows) equals the f16 attention over the dequantised
// values (fp16(q * d), what the expansion stores) to the same bar as the f16 path.
void run_window_case(int pos0, int T, int n_ctx, int win, bool q8) {
    constexpr int NH = 16, NKV = 2, HD = 256;
    const int vpitch = (n_ctx + 63) / 64 * 64, len = pos0 + T;
    uint32_t seed = 4242u + static_cast<uint32_t>(pos0 * 3 + T + (q8 ? 1 : 0));
    std::vector<float> q(static_cast<size_t>(T) * NH * HD), qfull(static_cast<size_t>(T) * NH * 2 * HD);
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < NH; ++h)
            for (int d = 0; d < HD; ++d) {
                const float x = frand(seed) * 3.f;
                q[(static_cast<size_t>(t) * NH + h) * HD + d] = x;
                qfull[(static_cast<size_t>(t) * NH + h) * 2 * HD + d] = x;
                qfull[(static_cast<size_t>(t) * NH + h) * 2 * HD + HD + d] = frand(seed) * 2.f;
            }
    // keys and values as floats, [pos][kv head][256]
    std::vector<float> kf(static_cast<size_t>(n_ctx) * NKV * HD, 0.f), vf(kf.size(), 0.f);
    for (int p = 0; p < len; ++p)
        for (size_t i = 0; i < static_cast<size_t>(NKV) * HD; ++i) {
            kf[static_cast<size_t>(p) * NKV * HD + i] = frand(seed);
            vf[static_cast<size_t>(p) * NKV * HD + i] = frand(seed);
        }
    std::vector<__half> kc(kf.size()), vt(static_cast<size_t>(NKV) * HD * vpitch, __float2half(0.f));
    std::vector<int8_t> kq(kf.size()), vq(kf.size());
    std::vector<uint16_t> ks(kf.size() / 32), vs(kf.size() / 32);
    auto quant = [](const float* x, int8_t* qo, uint16_t* so) {  // kv_quant_lane's arithmetic on the host
        float a = 0.f;
        for (int i = 0; i < 32; ++i) a = std::max(a, std::fabs(x[i]));
        const float d = a / 127.0f, inv = d > 0.f ? 127.0f / a : 0.f;
        *so = __half_as_ushort(__float2half_rn(d));
        for (int i = 0; i < 32; ++i) qo[i] = static_cast<int8_t>(std::nearbyint(x[i] * inv));
    };
    if (q8) {
        for (size_t b = 0; b < kf.size() / 32; ++b) {
            quant(kf.data() + b * 32, kq.data() + b * 32, ks.data() + b);
            quant(vf.data() + b * 32, vq.data() + b * 32, vs.data() + b);
        }
        for (size_t i = 0; i < kf.size(); ++i) {  // what the expansion stores: fp16(q * d)
            kf[i] = __half2float(__float2half_rn(static_cast<float>(kq[i]) * __half2float(__ushort_as_half(ks[i / 32]))));
            vf[i] = __half2float(__float2half_rn(static_cast<float>(vq[i]) * __half2float(__ushort_as_half(vs[i / 32]))));
        }
    }
    for (size_t i = 0; i < kf.size(); ++i) kc[i] = __float2half(kf[i]);
    for (int p = 0; p < n_ctx; ++p)
        for (int kh = 0; kh < NKV; ++kh)
            for (int d = 0; d < HD; ++d) vt[(static_cast<size_t>(kh) * HD + d) * vpitch + p] = __float2half(vf[(static_cast<size_t>(p) * NKV + kh) * HD + d]);
    float *dq, *dqf, *dout, *dout2, *dml;
    __half *dk, *dv, *dkw = nullptr, *dvw = nullptr;
    int8_t *dkq = nullptr, *dvq = nullptr;
    uint16_t *dks = nullptr, *dvs = nullptr;
    CUDA_OK(cudaMalloc(&dq, q.size() * 4));
    CUDA_OK(cudaMalloc(&dqf, qfull.size() * 4));
    CUDA_OK(cudaMalloc(&dout, q.size() * 4));
    CUDA_OK(cudaMalloc(&dout2, q.size() * 4));
    CUDA_OK(cudaMalloc(&dml, static_cast<size_t>(T) * NH * 2 * 4));
    CUDA_OK(cudaMalloc(&dk, kc.size() * 2));
    CUDA_OK(cudaMalloc(&dv, vt.size() * 2));
    CUDA_OK(cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dqf, qfull.data(), qfull.size() * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dk, kc.data(), kc.size() * 2, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dv, vt.data(), vt.size() * 2, cudaMemcpyHostToDevice));
    namespace ops = limbo::cuda::ops;
    ops::attention_rows(dq, dk, dv, dqf, dout, pos0, T, n_ctx, vpitch, NH, NKV, HD, 1.0f / 16.0f, nullptr);
    if (!q8) {
        // windows into the same arrays
        for (int k0 = 0; k0 < len; k0 += win) {
            const int rows = std::min(win, (len - k0 + 63) / 64 * 64);
            const int flags = (k0 == 0 ? 1 : 0) | (k0 + win >= len ? 2 : 0);
            ops::attention_rows_window(dq, dk + static_cast<size_t>(k0) * NKV * HD, dv + k0, dqf, dout2, pos0, T, k0, rows, vpitch, NH, NKV, HD, 1.0f / 16.0f,
                                       flags, dml, nullptr);
        }
    } else {
        const int wp = win;
        CUDA_OK(cudaMalloc(&dkw, static_cast<size_t>(win) * NKV * HD * 2));
        CUDA_OK(cudaMalloc(&dvw, static_cast<size_t>(win) * NKV * HD * 2));
        CUDA_OK(cudaMalloc(&dkq, kq.size()));
        CUDA_OK(cudaMalloc(&dvq, vq.size()));
        CUDA_OK(cudaMalloc(&dks, ks.size() * 2));
        CUDA_OK(cudaMalloc(&dvs, vs.size() * 2));
        CUDA_OK(cudaMemcpy(dkq, kq.data(), kq.size(), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(dvq, vq.data(), vq.size(), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(dks, ks.data(), ks.size() * 2, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(dvs, vs.data(), vs.size() * 2, cudaMemcpyHostToDevice));
        // garbage in the window scratch past the filled positions: the expansion must zero it
        CUDA_OK(cudaMemset(dkw, 0xff, static_cast<size_t>(win) * NKV * HD * 2));
        CUDA_OK(cudaMemset(dvw, 0xff, static_cast<size_t>(win) * NKV * HD * 2));
        for (int k0 = 0; k0 < len; k0 += win) {
            const int rows = std::min(win, (len - k0 + 63) / 64 * 64);
            ops::kv_expand_window(dkq, dks, dvq, dvs, dkw, dvw, k0, rows, len, NKV, wp, nullptr);
            const int flags = (k0 == 0 ? 1 : 0) | (k0 + win >= len ? 2 : 0);
            ops::attention_rows_window(dq, dkw, dvw, dqf, dout2, pos0, T, k0, rows, wp, NH, NKV, HD, 1.0f / 16.0f, flags, dml, nullptr);
        }
    }
    CUDA_OK(cudaDeviceSynchronize());
    std::vector<float> o1(q.size()), o2(q.size());
    CUDA_OK(cudaMemcpy(o1.data(), dout, o1.size() * 4, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(o2.data(), dout2, o2.size() * 4, cudaMemcpyDeviceToHost));
    // the online softmax is the same loop, but the running sum is added up in another order across a window boundary (the quad's
    // partial sums are folded into one lane there), so the windows agree with one pass to fp32 rounding, not bit for bit
    size_t differ = 0;
    double dmax = 0;
    for (size_t i = 0; i < o1.size(); ++i) {
        differ += std::memcmp(&o1[i], &o2[i], 4) != 0;
        dmax = std::max(dmax, std::fabs(double(o1[i]) - double(o2[i])));
    }
    char what[200];
    std::snprintf(what, sizeof what, "%s windows of %d, pos0 %d T %d: max |difference to one pass| %.3e (%zu of %zu outputs differ in the last bits)",
                  q8 ? "Q8 cache read in" : "key", win, pos0, T, dmax, differ, o1.size());
    std::printf("%s %s\n", dmax <= 2e-6 ? "ok  " : "FAIL", what);
    if (dmax > 2e-6) ++limbo::test::failures();
    if (q8) {  // and the f16 attention over those values is the reference the main cases check against a double-precision dot
        double worst = 0, mx = 0;
        std::vector<double> sc(static_cast<size_t>(len));
        for (int t = 0; t < T; t += (T > 40 ? 11 : 1))
            for (int h = 0; h < NH; h += 5) {
                const int kh = h / (NH / NKV), pos = pos0 + t;
                double m = -1e300;
                for (int p = 0; p <= pos; ++p) {
                    double sdot = 0;
                    for (int d = 0; d < HD; ++d) sdot += double(q[(static_cast<size_t>(t) * NH + h) * HD + d]) * double(kf[(static_cast<size_t>(p) * NKV + kh) * HD + d]);
                    sc[static_cast<size_t>(p)] = sdot / 16.0;
                    m = std::max(m, sc[static_cast<size_t>(p)]);
                }
                double sum = 0;
                for (int p = 0; p <= pos; ++p) {
                    sc[static_cast<size_t>(p)] = std::exp(sc[static_cast<size_t>(p)] - m);
                    sum += sc[static_cast<size_t>(p)];
                }
                for (int d = 0; d < HD; ++d) {
                    double o = 0;
                    for (int p = 0; p <= pos; ++p) o += sc[static_cast<size_t>(p)] / sum * double(vf[(static_cast<size_t>(p) * NKV + kh) * HD + d]);
                    const double gate = 1.0 / (1.0 + std::exp(-double(qfull[(static_cast<size_t>(t) * NH + h) * 2 * HD + HD + d])));
                    worst = std::max(worst, std::fabs(o * gate - double(o2[(static_cast<size_t>(t) * NH + h) * HD + d])));
                    mx = std::max(mx, std::fabs(o * gate));
                }
            }
        std::printf("Q8 cache, pos0 %d, T %d: max |gpu - cpu| = %.3e (max |out| %.3e)\n", pos0, T, worst, mx);
        CHECK(worst <= 2e-6, "Q8 cache read within 2e-6 absolute of the double-precision attention over the dequantised values");
    }
    cudaFree(dq); cudaFree(dqf); cudaFree(dout); cudaFree(dout2); cudaFree(dml); cudaFree(dk); cudaFree(dv);
    cudaFree(dkw); cudaFree(dvw); cudaFree(dkq); cudaFree(dvq); cudaFree(dks); cudaFree(dvs);
}

// The batched scan is the decode scan, scheduled differently: S and o equal bit for bit, over a chunk and a second
// chunk that continues from the first one's state.
void run_scan_case(int T1, int T2) {
    constexpr int NV = 32, NK = 16, HD = 128, RK = 32;
    const int T = T1 + T2, qk_stride = 2 * NK * HD, v_stride = NV * HD, ab_stride = RK;
    uint32_t seed = 777u + static_cast<uint32_t>(T);
    auto fill = [&](std::vector<float>& v, float amp, float off) {
        for (float& x : v) x = frand(seed) * amp + off;
    };
    std::vector<float> qk(static_cast<size_t>(T) * qk_stride), v(static_cast<size_t>(T) * v_stride), al(static_cast<size_t>(T) * ab_stride),
        be(static_cast<size_t>(T) * ab_stride), dtb(RK), sa(RK), S0(static_cast<size_t>(NV) * HD * HD);
    fill(qk, 0.3f, 0.f);
    fill(v, 1.f, 0.f);
    fill(al, 2.f, 0.f);
    fill(be, 2.f, 0.f);
    fill(dtb, 1.f, 0.5f);
    fill(S0, 0.2f, 0.f);
    for (float& x : sa) x = -std::fabs(frand(seed)) * 3.f - 0.1f;  // ssm_a is negative
    const float scale = 1.0f / std::sqrt(128.0f);
    float *dqk, *dv, *dal, *dbe, *dal2, *dbe2, *ddtb, *dsa, *dS1, *dS2, *do1, *do2;
    CUDA_OK(cudaMalloc(&dqk, qk.size() * 4));
    CUDA_OK(cudaMalloc(&dv, v.size() * 4));
    CUDA_OK(cudaMalloc(&dal, al.size() * 4));
    CUDA_OK(cudaMalloc(&dbe, be.size() * 4));
    CUDA_OK(cudaMalloc(&dal2, al.size() * 4));
    CUDA_OK(cudaMalloc(&dbe2, be.size() * 4));
    CUDA_OK(cudaMalloc(&ddtb, dtb.size() * 4));
    CUDA_OK(cudaMalloc(&dsa, sa.size() * 4));
    CUDA_OK(cudaMalloc(&dS1, S0.size() * 4));
    CUDA_OK(cudaMalloc(&dS2, S0.size() * 4));
    CUDA_OK(cudaMalloc(&do1, static_cast<size_t>(T) * v_stride * 4));
    CUDA_OK(cudaMalloc(&do2, static_cast<size_t>(T) * v_stride * 4));
    CUDA_OK(cudaMemcpy(dqk, qk.data(), qk.size() * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dv, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dal, al.data(), al.size() * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dbe, be.data(), be.size() * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dal2, al.data(), al.size() * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dbe2, be.data(), be.size() * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(ddtb, dtb.data(), dtb.size() * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dsa, sa.data(), sa.size() * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dS1, S0.data(), S0.size() * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dS2, S0.data(), S0.size() * 4, cudaMemcpyHostToDevice));
    namespace ops = limbo::cuda::ops;
    for (int t = 0; t < T; ++t)  // decode: one token per launch
        ops::gdn_scan(dS1, dqk + static_cast<size_t>(t) * qk_stride, dv + static_cast<size_t>(t) * v_stride,
                      dal + static_cast<size_t>(t) * ab_stride, dbe + static_cast<size_t>(t) * ab_stride, ddtb, dsa,
                      do1 + static_cast<size_t>(t) * v_stride, NV, NK, HD, RK, scale, nullptr);
    // batched: two chunks, the second continues from the first one's state
    const int Ts[2] = {T1, T2};
    int at = 0;
    for (int c = 0; c < 2; ++c) {
        if (Ts[c] == 0) continue;
        ops::gdn_coef_rows(dal2 + static_cast<size_t>(at) * ab_stride, dbe2 + static_cast<size_t>(at) * ab_stride, ab_stride, ddtb, dsa,
                           Ts[c], RK, nullptr);
        ops::gdn_scan_rows(dS2, dqk + static_cast<size_t>(at) * qk_stride, qk_stride, dv + static_cast<size_t>(at) * v_stride, v_stride,
                           dal2 + static_cast<size_t>(at) * ab_stride, dbe2 + static_cast<size_t>(at) * ab_stride, ab_stride,
                           do2 + static_cast<size_t>(at) * v_stride, v_stride, Ts[c], NV, NK, HD, RK, scale, nullptr);
        at += Ts[c];
    }
    CUDA_OK(cudaDeviceSynchronize());
    std::vector<float> S1(S0.size()), S2(S0.size()), o1(static_cast<size_t>(T) * v_stride), o2(o1.size());
    CUDA_OK(cudaMemcpy(S1.data(), dS1, S1.size() * 4, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(S2.data(), dS2, S2.size() * 4, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(o1.data(), do1, o1.size() * 4, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(o2.data(), do2, o2.size() * 4, cudaMemcpyDeviceToHost));
    size_t bad_o = 0, bad_s = 0;
    for (size_t i = 0; i < o1.size(); ++i) bad_o += std::memcmp(&o1[i], &o2[i], 4) != 0;
    for (size_t i = 0; i < S1.size(); ++i) bad_s += std::memcmp(&S1[i], &S2[i], 4) != 0;
    std::printf("%s scan T = %d + %d: %zu of %zu outputs and %zu of %zu state values differ from the decode kernel\n",
                bad_o + bad_s == 0 ? "ok  " : "FAIL", T1, T2, bad_o, o1.size(), bad_s, S1.size());
    if (bad_o + bad_s != 0) ++limbo::test::failures();
    for (float* p : {dqk, dv, dal, dbe, dal2, dbe2, ddtb, dsa, dS1, dS2, do1, do2}) cudaFree(p);
}

}  // namespace

int main() {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        std::printf("SKIP cuda_attn_test: no CUDA device\n");
        return 0;
    }
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    if (free_b < (1ull << 29)) {
        std::printf("SKIP cuda_attn_test: %.0f MB free\n", double(free_b) / 1e6);
        return 0;
    }
    run_case(0, 100, 256);
    run_case(0, 64, 128);
    run_case(37, 130, 512);
    run_case(300, 70, 512);
    run_case(1000, 33, 1200);
    run_case(300, 70, 512, false);
    run_case(1000, 33, 1200, false);
    run_window_case(0, 150, 700, 256, false);
    run_window_case(300, 90, 700, 128, false);
    run_window_case(100, 130, 600, 128, true);
    run_window_case(500, 70, 640, 192, true);
    run_scan_case(1, 0);
    run_scan_case(3, 2);
    run_scan_case(100, 29);
    run_scan_case(257, 255);
    return limbo::test::finish("cuda_attn_test");
}
