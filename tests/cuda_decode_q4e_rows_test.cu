// Copyright (c) 2026 ITERFORM. All rights reserved.
// tests/cuda_decode_q4e_rows_test.cu — Gpu::q4e_step_rows (a qwen4exp verify window of T = 1..8 rows) and
// Gpu::q4e_commit against T sequential step() calls, on a pseudo-random prompt ahead of the window.
//
// This is the gate docs/flash-next.md 6.6a needs BEFORE the window is wired into dispatch or captured into a
// graph: a window whose write_back is wrong fails silently, because the numbers it produces are plausible and
// only the state it leaves behind is wrong.
//
//   1. write_back = true: the logits of every row equal the sequential steps' within 1e-3, the greedy token of
//      every row is the sequential one, and the state afterwards is the sequential state (the logits of the next
//      step agree within 1e-3);
//   2. write_back = false then q4e_commit(n) for n = 0, 1, T/2 and T: the state is the sequential state after n
//      steps (the logits of the next token agree within 1e-3), and for n = 0 it is the state before the window.
//
// The model must be a qwen4exp (qwen3.8-flash-next) file: LIMBO_TEST_GGUF, a ':'-separated shard list, else the
// coder IQ1_M pair. SKIP (exit 0) when there is no GPU, under 15.5 GB free, the file is missing, or the file is
// not a qwen4exp model — the window's workspace is q4e's and does not exist for the other recipes.
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

constexpr const char* kQ4eDefault =
    "/home/alca/projects/Strata-data/models/coder-IQ1_M/"
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf:"
    "/home/alca/projects/Strata-data/models/coder-IQ1_M/"
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf";

int top1(const std::vector<float>& l) { return int(std::max_element(l.begin(), l.end()) - l.begin()); }
double max_diff(const float* a, const float* b, size_t n) {
    double m = 0;
    for (size_t i = 0; i < n; ++i) m = std::max(m, std::fabs(double(a[i]) - double(b[i])));
    return m;
}

// A ':'-separated list of paths, as the loader takes a model.
std::vector<std::string> split(const char* s) {
    std::vector<std::string> out;
    std::string cur;
    for (const char* p = s;; ++p) {
        if (*p == ':' || *p == '\0') {
            if (!cur.empty()) out.push_back(cur);
            cur.clear();
            if (*p == '\0') break;
        } else {
            cur.push_back(*p);
        }
    }
    return out;
}

}  // namespace

int main() {
    // the window against the sequential steps is an arithmetic identity only with the fp32-like lane (3 planes); the
    // engine's default 1 plane computes a miss in int8 and a hit in fp32, so a hit/miss flip between the runs moves logits
    if (!std::getenv("LIMBO_Q4E_PLANES")) setenv("LIMBO_Q4E_PLANES", "3", 1);
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        std::printf("SKIP cuda_decode_q4e_rows_test: no CUDA device\n");
        return 0;
    }
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    if (free_b < static_cast<size_t>(15.5e9)) {
        std::printf("SKIP cuda_decode_q4e_rows_test: %.2f GB free of %.2f GB, need 15.5 GB\n", double(free_b) / 1e9,
                    double(total_b) / 1e9);
        return 0;
    }
    const char* env = std::getenv("LIMBO_TEST_GGUF");
    const std::vector<std::string> paths = split(env && *env ? env : kQ4eDefault);
    std::string err;
    auto model = limbo::model::Model::open(paths, &err);
    if (!model) {
        std::printf("SKIP cuda_decode_q4e_rows_test: cannot open %s: %s\n", paths[0].c_str(), err.c_str());
        return 0;
    }
    // LIMBO_TEST_KV=q8: the same run over a Q8 K/V pool
    const char* tkv = std::getenv("LIMBO_TEST_KV");
    const limbo::cuda::KvFormat kv_fmt = tkv && std::string(tkv) == "q8" ? limbo::cuda::KvFormat::Q8 : limbo::cuda::KvFormat::F16;
    auto gpu = limbo::cuda::Gpu::open(*model, 128, static_cast<uint64_t>(14.6e9), &err, kv_fmt);
    CHECK(gpu != nullptr, "the engine opens");
    if (!gpu) {
        std::printf("     %s\n", err.c_str());
        return limbo::test::finish("cuda_decode_q4e_rows_test");
    }
    const size_t nv = static_cast<size_t>(gpu->n_vocab());
    constexpr int P = 8;
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

    // A model that is not qwen4exp has no q4e window: its workspace is never carved, so the first call refuses.
    // That is a SKIP, not a failure — the file, not the code, is what this test needs.
    if (!prefix() || !gpu->q4e_step_rows(ids.data() + P, 1, P, false)) {
        std::printf("SKIP cuda_decode_q4e_rows_test: %s is not a qwen4exp model (no q4e window workspace)\n",
                    paths[0].c_str());
        return 0;
    }

    const int Ts[] = {1, 2, 4};
    for (int T : Ts) {
        std::vector<std::vector<float>> ref;
        std::vector<float> ref_next;
        if (!sequential(T, ref, ref_next)) {
            CHECK(false, "sequential steps launch");
            return limbo::test::finish("cuda_decode_q4e_rows_test");
        }
        // 1. write_back = true: the window IS the accepted sequence, so it must move the state itself.
        {
            if (!prefix() || !gpu->q4e_step_rows(ids.data() + P, T, P, true)) {
                CHECK(false, "q4e_step_rows launches");
                return limbo::test::finish("cuda_decode_q4e_rows_test");
            }
            std::vector<float> got(size_t(T) * nv);
            CHECK(gpu->q4e_read_logits_rows(got.data(), T), "row logits copy");
            double worst = 0;
            bool same_top = true;
            size_t bits = 0;
            for (int t = 0; t < T; ++t) {
                worst = std::max(worst, max_diff(&got[size_t(t) * nv], ref[size_t(t)].data(), nv));
                bits += std::memcmp(&got[size_t(t) * nv], ref[size_t(t)].data(), nv * 4) == 0;
                same_top = same_top && top1(std::vector<float>(&got[size_t(t) * nv], &got[size_t(t) * nv] + nv)) ==
                                           top1(ref[size_t(t)]);
            }
            char what[192];
            std::snprintf(what, sizeof what, "T=%d write_back: %d rows, max|dlogit| %.2e (%zu rows bit-identical) <= 1e-3",
                          T, T, worst, bits);
            CHECK(worst <= 1e-3, what);
            std::snprintf(what, sizeof what, "T=%d write_back: the greedy token of every row is the sequential one", T);
            CHECK(same_top, what);
            std::vector<float> nx(nv);
            const bool ok = gpu->step(ids[size_t(P + T)], P + T, true) && gpu->read_logits(nx.data());
            const double d = ok ? max_diff(nx.data(), ref_next.data(), nv) : 1e9;
            std::snprintf(what, sizeof what,
                          "T=%d write_back: the state after the window equals the sequential state (next step %.2e)", T,
                          d);
            CHECK(ok && d <= 1e-3, what);
        }
        // 2. verify without write-back, then q4e_commit(n): the window must have moved NOTHING on its own.
        std::vector<int> ns = {0, 1, T / 2, T};
        std::sort(ns.begin(), ns.end());
        ns.erase(std::unique(ns.begin(), ns.end()), ns.end());
        for (int n : ns) {
            std::vector<std::vector<float>> r2;
            std::vector<float> want;
            if (!sequential(n, r2, want)) {
                CHECK(false, "sequential steps launch");
                return limbo::test::finish("cuda_decode_q4e_rows_test");
            }
            if (!prefix() || !gpu->q4e_step_rows(ids.data() + P, T, P, false) || !gpu->q4e_commit(n)) {
                CHECK(false, "q4e_step_rows(write_back = false) / q4e_commit launch");
                return limbo::test::finish("cuda_decode_q4e_rows_test");
            }
            std::vector<float> nx(nv);
            const bool ok = gpu->step(ids[size_t(P + n)], P + n, true) && gpu->read_logits(nx.data());
            const double d = ok ? max_diff(nx.data(), want.data(), nv) : 1e9;
            char what[192];
            std::snprintf(what, sizeof what,
                          "T=%d verify without write-back, commit(%d): the next step agrees with %d sequential steps (%.2e)",
                          T, n, n, d);
            CHECK(ok && d <= 1e-3, what);
        }
    }
    CHECK(!gpu->q4e_step_rows(ids.data(), 9, 0, true), "a window of 9 rows is refused");
    return limbo::test::finish("cuda_decode_q4e_rows_test");
}
