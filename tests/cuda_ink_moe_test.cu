// Copyright (c) 2026 ITERFORM. All rights reserved.
// tests/cuda_ink_moe_test.cu — the inkling2 routed experts (src/cuda/ink_moe.cu) against a double-precision
// CPU model of the same math: limbo/quants.hpp's decoder plus the fold of src/step.cpp's ink_moe (sigmoid
// scores + selection bias, top-8 with ties to the lower id, the +1e-20 renormalisation times route_scale,
// per-expert dequantisation scales, SiTU, and the weighted down accumulation in selection order).
//
//   1. route: the chosen ids and the weights against the double fold, at n_expert 128 (the model's count),
//      plus the tie rule on crafted equal scores (equal router rows and equal bias must pick the lower ids);
//   2. the pipeline (route -> gate/up + SiTU -> down accumulation) in the file's own expert weight types
//      (IQ1_S / IQ2_XXS / IQ2_S gate-up against Q2_0 / NVFP4 down, per-expert scales on both), small
//      shapes and one at the model's real shape (2304 / 1152 / 128), ids exact and every float within 2e-6;
//   3. determinism: a second identical run is bit-identical everywhere.
//
// Synthetic weights are random block bytes with valid scales (the cuda_gemv_test.cu idiom): 0x3000 (0.125)
// for the f16 block scales, UE4M3 code 0x28 (0.125) for NVFP4's sub-block scales. No GPU: SKIP, exit 0.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../src/cuda/ink_moe.hpp"
#include "limbo/check.hpp"
#include "limbo/cuda/gemv.hpp"
#include "limbo/quants.hpp"
#include "limbo/types.hpp"

namespace {

using limbo::Type;
namespace cu = limbo::cuda;
namespace ink = limbo::cuda::ink;

constexpr float kBeta = 4.0f, kLb = 25.0f, kRouteScale = 8.0f;

#define CUDA_OK(call)                                                                                  \
    do {                                                                                               \
        cudaError_t e_ = (call);                                                                       \
        if (e_ != cudaSuccess) {                                                                       \
            std::printf("FAIL %s: %s (%s:%d)\n", #call, cudaGetErrorString(e_), __FILE__, __LINE__);    \
            ++limbo::test::failures();                                                                 \
            return;                                                                                    \
        }                                                                                              \
    } while (0)

struct Rng {
    uint32_t s;
    explicit Rng(uint32_t seed) : s(seed) {}
    uint32_t next() { return s = s * 1664525u + 1013904223u; }
    float uni() { return static_cast<float>(next() >> 8) / 16777216.0f; }   // [0, 1)
    float sym() { return uni() * 2.0f - 1.0f; }
};

std::vector<float> make_x(int n, uint32_t seed, float scale) {
    std::vector<float> x(static_cast<size_t>(n));
    Rng r(seed);
    for (int i = 0; i < n; ++i) x[static_cast<size_t>(i)] = r.sym() * scale;
    return x;
}

// Random block bytes with valid scales, normalised the way a quantizer would: the payload is random in every
// format, then each block's scale is set so the decoded values stay within 0.3 (the i-quant grids hold raw
// bytes, so a fixed scale would put IQ2_S values an order of magnitude above Q2_0's). f16-scaled formats
// store the rounded 0.3/max scale as the block's f16 d; NVFP4 stores the nearest UE4M3 code in its four
// sub-block scales (0x40 decodes to exactly 1, the unit the search starts from).
float ue4m3(uint8_t x) {
    if (x == 0 || x == 0x7f) return 0.0f;
    const int exp = (x >> 3) & 0xf, man = x & 7;
    const float raw = exp == 0 ? std::ldexp(float(man), -9) : std::ldexp(1.0f + float(man) / 8.0f, exp - 7);
    return raw * 0.5f;
}

std::vector<uint8_t> make_stack(uint32_t type, uint64_t rows, uint64_t row_bytes, uint32_t seed) {
    const limbo::TypeInfo* ti = limbo::type_info(type);
    std::vector<uint8_t> host(static_cast<size_t>(rows) * row_bytes);
    Rng r(seed);
    for (uint8_t& b : host) b = static_cast<uint8_t>(r.next() >> 24);
    std::vector<float> dec(ti->block_elems);
    const size_t nblk = host.size() / ti->block_bytes;
    for (size_t b = 0; b < nblk; ++b) {
        uint8_t* p = host.data() + b * ti->block_bytes;
        if (static_cast<limbo::Type>(type) == Type::NVFP4) {
            p[0] = p[1] = p[2] = p[3] = 0x40;   // unit sub-scales
            limbo::quants::dequant_row(type, p, dec.data(), ti->block_elems);
            float mx = 0;
            for (float v : dec) mx = std::max(mx, std::fabs(v));
            const float target = mx > 0 ? 0.3f / mx : 1.0f;
            uint8_t best = 0x40;
            float bestd = 1e30f;
            for (int c = 1; c < 127; ++c) {
                const float d = std::fabs(ue4m3(static_cast<uint8_t>(c)) - target);
                if (d < bestd) {
                    bestd = d;
                    best = static_cast<uint8_t>(c);
                }
            }
            p[0] = p[1] = p[2] = p[3] = best;
        } else {
            p[0] = 0x00;
            p[1] = 0x3c;   // d = 1
            limbo::quants::dequant_row(type, p, dec.data(), ti->block_elems);
            float mx = 0;
            for (float v : dec) mx = std::max(mx, std::fabs(v));
            const float d = mx > 0 ? 0.3f / mx : 1.0f;
            const __half h = __float2half(d);
            const uint16_t bits = __half_as_ushort(h);
            p[0] = static_cast<uint8_t>(bits & 0xff);
            p[1] = static_cast<uint8_t>(bits >> 8);
        }
    }
    return host;
}

// The CPU model: every product dequantised with limbo/quants.hpp and dotted in double, the folds in double.
double cpu_dot(uint32_t type, const uint8_t* row, const double* x, int n) {
    std::vector<float> buf(static_cast<size_t>(n));
    if (!limbo::quants::dequant_row(type, row, buf.data(), static_cast<uint64_t>(n))) {
        std::printf("FAIL dequant_row does not implement type %u\n", type);
        ++limbo::test::failures();
        return 0;
    }
    double acc = 0;
    for (int i = 0; i < n; ++i) acc += double(buf[static_cast<size_t>(i)]) * x[i];
    return acc;
}

struct CpuOut {
    std::vector<double> logits;   // the router's
    std::vector<int32_t> ids;
    std::vector<double> w, h, out;
};

// The router's fold: sigmoid logits + bias, top n_used by the reference's comparator (ties to the lower id),
// the weights renormalised with +1e-20 and scaled by route_scale.
void cpu_route(uint32_t router_type, const std::vector<uint8_t>& router, const std::vector<float>& bias,
               const std::vector<float>& x, int n_expert, int n_used, int n_embd, CpuOut& o) {
    std::vector<double> xd(x.begin(), x.end());
    o.logits.resize(static_cast<size_t>(n_expert));
    std::vector<double> probs(static_cast<size_t>(n_expert));
    for (int i = 0; i < n_expert; ++i) {
        const double l = cpu_dot(router_type, router.data() + static_cast<size_t>(i) * limbo::type_bytes(router_type, n_embd),
                                 xd.data(), n_embd);
        o.logits[static_cast<size_t>(i)] = l;
        probs[static_cast<size_t>(i)] = 1.0 / (1.0 + std::exp(-l));
    }
    std::vector<int> idx(static_cast<size_t>(n_expert));
    for (int i = 0; i < n_expert; ++i) idx[static_cast<size_t>(i)] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        const double sa = probs[static_cast<size_t>(a)] + double(bias[static_cast<size_t>(a)]);
        const double sb = probs[static_cast<size_t>(b)] + double(bias[static_cast<size_t>(b)]);
        return sa > sb || (sa == sb && a < b);
    });
    double wsum = 0;
    for (int i = 0; i < n_used; ++i) wsum += probs[static_cast<size_t>(idx[static_cast<size_t>(i)])];
    wsum += 1e-20;
    for (int i = 0; i < n_used; ++i) {
        const int ex = idx[static_cast<size_t>(i)];
        o.ids.push_back(ex);
        o.w.push_back(probs[static_cast<size_t>(ex)] / wsum * double(kRouteScale));
    }
}

// The experts: gate/up under the per-expert scale, SiTU, then the weighted down rows in selection order.
void cpu_experts(const CpuOut& route, uint32_t gu_type, const std::vector<uint8_t>& gu, uint64_t gu_row,
                 uint64_t gu_exp, const std::vector<float>& gu_scale, uint32_t dn_type, const std::vector<uint8_t>& dn,
                 uint64_t dn_row, uint64_t dn_exp, const std::vector<float>& dn_scale, const std::vector<float>& x,
                 int n_embd, int n_ff, CpuOut& o) {
    std::vector<double> xd(x.begin(), x.end());
    const int n_used = int(route.ids.size());
    o.h.assign(static_cast<size_t>(n_used) * n_ff, 0.0);
    o.out.assign(static_cast<size_t>(n_embd), 0.0);
    for (int i = 0; i < n_used; ++i) {
        const int ex = route.ids[static_cast<size_t>(i)];
        const uint8_t* gbase = gu.data() + static_cast<uint64_t>(ex) * gu_exp;
        const uint8_t* dbase = dn.data() + static_cast<uint64_t>(ex) * dn_exp;
        const double gs = gu_scale.empty() ? 1.0 : double(gu_scale[static_cast<size_t>(ex)]);
        const double ds = dn_scale.empty() ? 1.0 : double(dn_scale[static_cast<size_t>(ex)]);
        for (int j = 0; j < n_ff; ++j) {
            const double g = cpu_dot(gu_type, gbase + static_cast<uint64_t>(j) * gu_row, xd.data(), n_embd) * gs;
            const double u = cpu_dot(gu_type, gbase + static_cast<uint64_t>(n_ff + j) * gu_row, xd.data(), n_embd) * gs;
            o.h[static_cast<size_t>(i) * n_ff + j] =
                (double(kBeta) * std::tanh(g / double(kBeta)) * (1.0 / (1.0 + std::exp(-g)))) *
                (double(kLb) * std::tanh(u / double(kLb)));
        }
        for (int j = 0; j < n_embd; ++j) {
            const double d = cpu_dot(dn_type, dbase + static_cast<uint64_t>(j) * dn_row,
                                     o.h.data() + static_cast<size_t>(i) * n_ff, n_ff) * ds;
            o.out[static_cast<size_t>(j)] += d * route.w[static_cast<size_t>(i)];
        }
    }
}

CpuOut cpu_pipeline(uint32_t router_type, const std::vector<uint8_t>& router, const std::vector<float>& bias,
                    uint32_t gu_type, const std::vector<uint8_t>& gu, uint64_t gu_row, uint64_t gu_exp,
                    const std::vector<float>& gu_scale, uint32_t dn_type, const std::vector<uint8_t>& dn,
                    uint64_t dn_row, uint64_t dn_exp, const std::vector<float>& dn_scale, const std::vector<float>& x,
                    int n_expert, int n_used, int n_embd, int n_ff) {
    CpuOut o;
    cpu_route(router_type, router, bias, x, n_expert, n_used, n_embd, o);
    cpu_experts(o, gu_type, gu, gu_row, gu_exp, gu_scale, dn_type, dn, dn_row, dn_exp, dn_scale, x, n_embd, n_ff, o);
    return o;
}

double max_diff(const std::vector<float>& a, const std::vector<double>& b) {
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(double(a[i]) - b[i]));
    return m;
}

double max_abs(const std::vector<double>& a) {
    double m = 0;
    for (double v : a) m = std::max(m, std::fabs(v));
    return m;
}

// ---- the router's fold and the tie rule -------------------------------------------------------------------
void test_route() {
    constexpr int NE = 128, NU = 8, E = 512;
    std::vector<float> x = make_x(E, 11, 0.02f);
    // A router of random F32 rows and a wide selection bias, plus the tie case: experts 5, 9 and 20 share
    // one row and one bias and must come out in id order.
    std::vector<uint8_t> router(static_cast<size_t>(NE) * E * sizeof(float));
    float* rp = reinterpret_cast<float*>(router.data());
    std::vector<float> bias(NE);
    Rng r(22);
    for (int i = 0; i < NE * E; ++i) rp[i] = r.sym() * 0.1f;
    for (float& v : bias) v = r.sym();
    std::memcpy(rp + 9 * E, rp + 5 * E, sizeof(float) * E);
    std::memcpy(rp + 20 * E, rp + 5 * E, sizeof(float) * E);
    bias[5] = bias[9] = bias[20] = 8.0f;

    CpuOut ref;
    cpu_route(static_cast<uint32_t>(Type::F32), router, bias, x, NE, NU, E, ref);

    float *d_x = nullptr, *d_router = nullptr, *d_bias = nullptr, *d_w = nullptr, *d_logits = nullptr;
    int32_t* d_ids = nullptr;
    CUDA_OK(cudaMalloc(&d_x, sizeof(float) * x.size()));
    CUDA_OK(cudaMalloc(&d_router, router.size()));
    CUDA_OK(cudaMalloc(&d_bias, sizeof(float) * bias.size()));
    CUDA_OK(cudaMalloc(&d_w, sizeof(float) * NU));
    CUDA_OK(cudaMalloc(&d_logits, sizeof(float) * NE));
    CUDA_OK(cudaMalloc(&d_ids, sizeof(int32_t) * NU));
    CUDA_OK(cudaMemcpy(d_x, x.data(), sizeof(float) * x.size(), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_router, router.data(), router.size(), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_bias, bias.data(), sizeof(float) * bias.size(), cudaMemcpyHostToDevice));

    CHECK(ink::moe_route(d_x, d_router, static_cast<uint32_t>(Type::F32), d_bias, NE, NU, E, kRouteScale, d_ids,
                         d_w, d_logits, nullptr),
          "route: launches");
    CUDA_OK(cudaDeviceSynchronize());
    std::vector<int32_t> ids(NU);
    std::vector<float> w(NU), logits(NE);
    CUDA_OK(cudaMemcpy(ids.data(), d_ids, sizeof(int32_t) * NU, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(w.data(), d_w, sizeof(float) * NU, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(logits.data(), d_logits, sizeof(float) * NE, cudaMemcpyDeviceToHost));

    CHECK(std::memcmp(ids.data(), ref.ids.data(), sizeof(int32_t) * NU) == 0, "route: the chosen ids match the double fold");
    CHECK(ids[0] == 5 && ids[1] == 9 && ids[2] == 20, "route: tied scores take the lower ids in order (5, 9, 20)");
    const double dw = max_diff(w, ref.w);
    std::printf("     route: max |dw| %.3e\n", dw);
    CHECK(dw <= 2e-6, "route: the weights within 2e-6");
    // the raw logits are the plain products; they leave the router stage for the caller's tracing
    double dl = 0;
    for (int i = 0; i < NE; ++i) dl = std::max(dl, std::fabs(double(logits[static_cast<size_t>(i)]) - ref.logits[static_cast<size_t>(i)]));
    std::printf("     route: max |dlogit| %.3e\n", dl);
    CHECK(dl <= 2e-6, "route: the router logits within 2e-6");

    // determinism: the same call again is bit-identical (the outputs seeded with NaN first)
    const std::vector<float> junk(NU, std::nanf(""));
    CUDA_OK(cudaMemcpy(d_w, junk.data(), sizeof(float) * NU, cudaMemcpyHostToDevice));
    CHECK(ink::moe_route(d_x, d_router, static_cast<uint32_t>(Type::F32), d_bias, NE, NU, E, kRouteScale, d_ids,
                         d_w, d_logits, nullptr),
          "route (second run): launches");
    CUDA_OK(cudaDeviceSynchronize());
    std::vector<int32_t> ids2(NU);
    std::vector<float> w2(NU);
    CUDA_OK(cudaMemcpy(ids2.data(), d_ids, sizeof(int32_t) * NU, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(w2.data(), d_w, sizeof(float) * NU, cudaMemcpyDeviceToHost));
    CHECK(std::memcmp(ids.data(), ids2.data(), sizeof(int32_t) * NU) == 0 &&
              std::memcmp(w.data(), w2.data(), sizeof(float) * NU) == 0,
          "route: a second run is bit-identical");

    cudaFree(d_x);
    cudaFree(d_router);
    cudaFree(d_bias);
    cudaFree(d_w);
    cudaFree(d_logits);
    cudaFree(d_ids);
}

// ---- the expert pipeline in the file's own types ---------------------------------------------------------
struct Pair {
    Type gu, dn;
    int n_embd, n_ff, n_expert;
    float xs;   // the input scale: the fold's outputs stay O(1) for the 2e-6 bar to mean something
    const char* name;
};

void test_pair(const Pair& p) {
    const uint32_t gu_type = static_cast<uint32_t>(p.gu), dn_type = static_cast<uint32_t>(p.dn);
    const uint32_t router_type = static_cast<uint32_t>(Type::F32);
    const int NE = p.n_expert, NU = 8, E = p.n_embd, F = p.n_ff;
    const uint64_t gu_row = limbo::type_bytes(gu_type, E), gu_exp = uint64_t(2 * F) * gu_row;
    const uint64_t dn_row = limbo::type_bytes(dn_type, F), dn_exp = uint64_t(E) * dn_row;
    const uint64_t router_row = limbo::type_bytes(router_type, E);
    const uint32_t seed = 1000u + static_cast<uint32_t>(p.gu) * 7u + static_cast<uint32_t>(p.dn);

    std::vector<float> x = make_x(E, seed + 1, p.xs);
    std::vector<uint8_t> router(NE * router_row);
    {
        Rng r(seed + 3);
        float* rp = reinterpret_cast<float*>(router.data());
        for (int i = 0; i < NE * E; ++i) rp[i] = r.sym() * 0.1f;
    }
    const std::vector<uint8_t> gu = make_stack(gu_type, uint64_t(NE) * 2 * F, gu_row, seed + 4);
    const std::vector<uint8_t> dn = make_stack(dn_type, uint64_t(NE) * E, dn_row, seed + 5);
    std::vector<float> bias(NE), gu_scale(NE), dn_scale(NE);
    {
        Rng r(seed + 6);
        for (float& v : bias) v = r.sym();
        for (float& v : gu_scale) v = 0.5f + r.uni();
        for (float& v : dn_scale) v = 0.5f + r.uni();
    }

    const CpuOut ref = cpu_pipeline(router_type, router, bias, gu_type, gu, gu_row, gu_exp, gu_scale, dn_type, dn,
                                    dn_row, dn_exp, dn_scale, x, NE, NU, E, F);

    float *d_x = nullptr, *d_router = nullptr, *d_bias = nullptr, *d_gu_scale = nullptr, *d_dn_scale = nullptr;
    float *d_w = nullptr, *d_logits = nullptr, *d_h = nullptr, *d_raw = nullptr, *d_out = nullptr;
    void *d_gu = nullptr, *d_dn = nullptr;
    int32_t* d_ids = nullptr;
    CUDA_OK(cudaMalloc(&d_x, sizeof(float) * x.size()));
    CUDA_OK(cudaMalloc(&d_router, router.size()));
    CUDA_OK(cudaMalloc(&d_bias, sizeof(float) * bias.size()));
    CUDA_OK(cudaMalloc(&d_gu, gu.size()));
    CUDA_OK(cudaMalloc(&d_dn, dn.size()));
    CUDA_OK(cudaMalloc(&d_gu_scale, sizeof(float) * gu_scale.size()));
    CUDA_OK(cudaMalloc(&d_dn_scale, sizeof(float) * dn_scale.size()));
    CUDA_OK(cudaMalloc(&d_ids, sizeof(int32_t) * NU));
    CUDA_OK(cudaMalloc(&d_w, sizeof(float) * NU));
    CUDA_OK(cudaMalloc(&d_logits, sizeof(float) * NE));
    CUDA_OK(cudaMalloc(&d_h, sizeof(float) * NU * F));
    CUDA_OK(cudaMalloc(&d_out, sizeof(float) * E));
    CUDA_OK(cudaMalloc(&d_raw, sizeof(float) * ink::moe_scratch_floats(NU, F, E)));
    CUDA_OK(cudaMemcpy(d_x, x.data(), sizeof(float) * x.size(), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_router, router.data(), router.size(), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_bias, bias.data(), sizeof(float) * bias.size(), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_gu, gu.data(), gu.size(), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_dn, dn.data(), dn.size(), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_gu_scale, gu_scale.data(), sizeof(float) * gu_scale.size(), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_dn_scale, dn_scale.data(), sizeof(float) * dn_scale.size(), cudaMemcpyHostToDevice));

    char what[160];
    std::snprintf(what, sizeof what, "%s: route launches", p.name);
    CHECK(ink::moe_route(d_x, d_router, router_type, d_bias, NE, NU, E, kRouteScale, d_ids, d_w, d_logits, nullptr), what);
    std::snprintf(what, sizeof what, "%s: gate/up + SiTU launches", p.name);
    CHECK(ink::moe_gate_up_situ(d_gu, gu_type, gu_exp, d_ids, NU, d_x, d_gu_scale, F, E, kBeta, kLb, d_h, d_raw, nullptr), what);
    std::snprintf(what, sizeof what, "%s: down accumulation launches", p.name);
    CHECK(ink::moe_down_accum(d_dn, dn_type, dn_exp, d_ids, NU, d_h, d_dn_scale, d_w, F, E, d_out, d_raw, nullptr), what);
    CUDA_OK(cudaDeviceSynchronize());

    std::vector<int32_t> ids(NU);
    std::vector<float> w(NU), h(static_cast<size_t>(NU) * F), out(E);
    CUDA_OK(cudaMemcpy(ids.data(), d_ids, sizeof(int32_t) * NU, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(w.data(), d_w, sizeof(float) * NU, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(h.data(), d_h, sizeof(float) * h.size(), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(out.data(), d_out, sizeof(float) * E, cudaMemcpyDeviceToHost));

    std::snprintf(what, sizeof what, "%s: the chosen ids match the double fold", p.name);
    CHECK(std::memcmp(ids.data(), ref.ids.data(), sizeof(int32_t) * NU) == 0, what);
    const double dw = max_diff(w, ref.w), dh = max_diff(h, ref.h), dout = max_diff(out, ref.out);
    std::printf("     %-18s max |dw| %.3e  max |dh| %.3e  max |dout| %.3e  (|h| %.2e  |out| %.2e)\n", p.name, dw,
                dh, dout, max_abs(ref.h), max_abs(ref.out));
    std::snprintf(what, sizeof what, "%s: the routing weights within 2e-6", p.name);
    CHECK(dw <= 2e-6, what);
    std::snprintf(what, sizeof what, "%s: SiTU's h within 2e-6", p.name);
    CHECK(dh <= 2e-6, what);
    std::snprintf(what, sizeof what, "%s: the accumulated output within 2e-6", p.name);
    CHECK(dout <= 2e-6, what);

    // determinism: the same three stages again, outputs seeded with NaN first, are bit-identical
    {
        const std::vector<float> junkh(h.size(), std::nanf("")), junko(E, std::nanf(""));
        CUDA_OK(cudaMemcpy(d_h, junkh.data(), sizeof(float) * h.size(), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_out, junko.data(), sizeof(float) * E, cudaMemcpyHostToDevice));
    }
    CHECK(ink::moe_route(d_x, d_router, router_type, d_bias, NE, NU, E, kRouteScale, d_ids, d_w, d_logits, nullptr), "rerun: route launches");
    CHECK(ink::moe_gate_up_situ(d_gu, gu_type, gu_exp, d_ids, NU, d_x, d_gu_scale, F, E, kBeta, kLb, d_h, d_raw, nullptr), "rerun: gate/up launches");
    CHECK(ink::moe_down_accum(d_dn, dn_type, dn_exp, d_ids, NU, d_h, d_dn_scale, d_w, F, E, d_out, d_raw, nullptr), "rerun: down launches");
    CUDA_OK(cudaDeviceSynchronize());
    std::vector<int32_t> ids2(NU);
    std::vector<float> w2(NU), h2(h.size()), out2(E);
    CUDA_OK(cudaMemcpy(ids2.data(), d_ids, sizeof(int32_t) * NU, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(w2.data(), d_w, sizeof(float) * NU, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(h2.data(), d_h, sizeof(float) * h.size(), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(out2.data(), d_out, sizeof(float) * E, cudaMemcpyDeviceToHost));
    std::snprintf(what, sizeof what, "%s: a second run is bit-identical", p.name);
    CHECK(std::memcmp(ids.data(), ids2.data(), sizeof(int32_t) * NU) == 0 &&
              std::memcmp(w.data(), w2.data(), sizeof(float) * NU) == 0 &&
              std::memcmp(h.data(), h2.data(), sizeof(float) * h.size()) == 0 &&
              std::memcmp(out.data(), out2.data(), sizeof(float) * E) == 0,
          what);

    cudaFree(d_x);
    cudaFree(d_router);
    cudaFree(d_bias);
    cudaFree(d_gu);
    cudaFree(d_dn);
    cudaFree(d_gu_scale);
    cudaFree(d_dn_scale);
    cudaFree(d_ids);
    cudaFree(d_w);
    cudaFree(d_logits);
    cudaFree(d_h);
    cudaFree(d_raw);
    cudaFree(d_out);
}

}  // namespace

int main() {
    int dev = 0;
    if (cudaGetDeviceCount(&dev) != cudaSuccess || dev == 0) {
        std::printf("cuda_ink_moe_test: SKIP (no GPU)\n");
        return limbo::test::finish("cuda_ink_moe");
    }
    std::string err;
    if (!cu::init_tables(&err)) {
        std::printf("FAIL init_tables: %s\n", err.c_str());
        return limbo::test::finish("cuda_ink_moe");
    }

    test_route();

    const Pair pairs[] = {
        {Type::IQ1_S, Type::Q2_0, 512, 320, 32, 0.05f, "IQ1_S/Q2_0 small"},
        {Type::IQ2_S, Type::NVFP4, 512, 320, 32, 0.05f, "IQ2_S/NVFP4 small"},
        {Type::IQ2_XXS, Type::Q2_0, 512, 320, 32, 0.05f, "IQ2_XXS/Q2_0 small"},
        {Type::IQ1_S, Type::Q2_0, 2304, 1152, 128, 0.02f, "IQ1_S/Q2_0 real"},
    };
    for (const Pair& p : pairs) test_pair(p);

    return limbo::test::finish("cuda_ink_moe");
}
