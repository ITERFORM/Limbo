// Copyright (c) 2026 ITERFORM. All rights reserved.
// tests/cuda_decode_rows_test.cu — Gpu::step_rows (a verify window of T = 1..8 tokens) and Gpu::commit against T sequential
// step() calls, on rung 1 with 20 pseudo-random prompt tokens ahead of the window.
//
//   1. write_back = true: the logits of every row equal the sequential steps' within 1e-3, the greedy tokens are the same,
//      and the state afterwards is the sequential state (the logits of the next step agree within 1e-3);
//   2. write_back = false then commit(n) for n = 0, 1, T/2 and T: the state is the sequential state after n steps (the
//      logits of the next token agree within 1e-3), and for n = 0 it is the state before the window.
//
// The model is LIMBO_TEST_GGUF or the rung-1 file. SKIP (exit 0) when there is no GPU or under 15.5 GB free.
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
#include "limbo/cuda/gpu.hpp"
#include "limbo/model.hpp"

namespace {

constexpr const char* kRung1 =
    "/home/alca/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-MTP-GGUF/snapshots/"
    "5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf";

int top1(const std::vector<float>& l) { return int(std::max_element(l.begin(), l.end()) - l.begin()); }
double max_diff(const float* a, const float* b, size_t n) {
    double m = 0;
    for (size_t i = 0; i < n; ++i) m = std::max(m, std::fabs(double(a[i]) - double(b[i])));
    return m;
}

}  // namespace

int main() {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        std::printf("SKIP cuda_decode_rows_test: no CUDA device\n");
        return 0;
    }
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    if (free_b < static_cast<size_t>(15.5e9)) {
        std::printf("SKIP cuda_decode_rows_test: %.2f GB free of %.2f GB, need 15.5 GB\n", double(free_b) / 1e9, double(total_b) / 1e9);
        return 0;
    }
    const char* env = std::getenv("LIMBO_TEST_GGUF");
    const std::string path = env && *env && !std::strchr(env, ':') ? env : kRung1;
    std::string err;
    auto model = limbo::model::Model::open({path}, &err);
    if (!model) {
        std::printf("SKIP cuda_decode_rows_test: cannot open %s: %s\n", path.c_str(), err.c_str());
        return 0;
    }
    auto gpu = limbo::cuda::Gpu::open(*model, 128, static_cast<uint64_t>(14.6e9), &err);
    CHECK(gpu != nullptr, "the engine opens");
    if (!gpu) {
        std::printf("     %s\n", err.c_str());
        return limbo::test::finish("cuda_decode_rows_test");
    }
    const size_t nv = static_cast<size_t>(gpu->n_vocab());
    constexpr int P = 20;
    std::vector<int32_t> ids(P + 16 + 2);
    uint32_t s = 4242;
    for (auto& v : ids) {
        s = s * 1664525u + 1013904223u;
        v = 1000 + static_cast<int32_t>((s >> 8) % 200000u);
    }
    auto prefix = [&]() {
        gpu->reset();
        for (int i = 0; i < P; ++i)
            if (!gpu->step(ids[size_t(i)], i, false)) return false;
        return true;
    };
    // n sequential steps from the prefix, the logits of each in ref[0..n), then one more step (the "next" token)
    auto sequential = [&](int n, std::vector<std::vector<float>>& ref, std::vector<float>& next) {
        ref.assign(size_t(n), std::vector<float>(nv));
        if (!prefix()) return false;
        for (int t = 0; t < n; ++t)
            if (!gpu->step(ids[size_t(P + t)], P + t, true) || !gpu->read_logits(ref[size_t(t)].data())) return false;
        next.assign(nv, 0.f);
        return gpu->step(ids[size_t(P + n)], P + n, true) && gpu->read_logits(next.data());
    };

    const int Ts[] = {1, 2, 3, 4, 8};
    for (int T : Ts) {
        std::vector<std::vector<float>> ref;
        std::vector<float> ref_next;
        if (!sequential(T, ref, ref_next)) {
            CHECK(false, "sequential steps launch");
            return limbo::test::finish("cuda_decode_rows_test");
        }
        // 1. write_back = true
        {
            if (!prefix() || !gpu->step_rows(ids.data() + P, T, P, true)) {
                CHECK(false, "step_rows launches");
                return limbo::test::finish("cuda_decode_rows_test");
            }
            std::vector<float> got(size_t(T) * nv);
            CHECK(gpu->read_logits_rows(got.data(), T), "row logits copy");
            std::vector<int32_t> am(static_cast<size_t>(T), 0);
            CHECK(gpu->argmax_rows(am.data(), T), "row argmax");
            double worst = 0;
            bool same_top = true;
            size_t bits = 0;
            for (int t = 0; t < T; ++t) {
                worst = std::max(worst, max_diff(&got[size_t(t) * nv], ref[size_t(t)].data(), nv));
                bits += std::memcmp(&got[size_t(t) * nv], ref[size_t(t)].data(), nv * 4) == 0;
                same_top = same_top && am[size_t(t)] == top1(ref[size_t(t)]);
            }
            char what[160];
            std::snprintf(what, sizeof what, "T=%d write_back: %d rows, max|dlogit| %.2e (%zu rows bit-identical) <= 1e-3", T, T, worst, bits);
            CHECK(worst <= 1e-3, what);
            std::snprintf(what, sizeof what, "T=%d write_back: the greedy token of every row is the sequential one", T);
            CHECK(same_top, what);
            std::vector<float> nx(nv);
            const bool ok = gpu->step(ids[size_t(P + T)], P + T, true) && gpu->read_logits(nx.data());
            const double d = ok ? max_diff(nx.data(), ref_next.data(), nv) : 1e9;
            std::snprintf(what, sizeof what, "T=%d write_back: the state after the window equals the sequential state (next step %.2e)", T, d);
            CHECK(ok && d <= 1e-3, what);
        }
        // 2. verify without write-back, then commit(n)
        std::vector<int> ns = {0, 1, T / 2, T};
        std::sort(ns.begin(), ns.end());
        ns.erase(std::unique(ns.begin(), ns.end()), ns.end());
        for (int n : ns) {
            std::vector<std::vector<float>> r2;
            std::vector<float> want;
            if (!sequential(n, r2, want)) {
                CHECK(false, "sequential steps launch");
                return limbo::test::finish("cuda_decode_rows_test");
            }
            if (!prefix() || !gpu->step_rows(ids.data() + P, T, P, false) || !gpu->commit(n)) {
                CHECK(false, "step_rows(write_back = false) / commit launch");
                return limbo::test::finish("cuda_decode_rows_test");
            }
            std::vector<float> nx(nv);
            const bool ok = gpu->step(ids[size_t(P + n)], P + n, true) && gpu->read_logits(nx.data());
            const double d = ok ? max_diff(nx.data(), want.data(), nv) : 1e9;
            char what[160];
            std::snprintf(what, sizeof what, "T=%d verify without write-back, commit(%d): the next step agrees with %d sequential steps (%.2e)", T, n, n, d);
            CHECK(ok && d <= 1e-3, what);
        }
    }
    CHECK(!gpu->step_rows(ids.data(), 9, 0, true), "a window of 9 rows is refused");
    return limbo::test::finish("cuda_decode_rows_test");
}
