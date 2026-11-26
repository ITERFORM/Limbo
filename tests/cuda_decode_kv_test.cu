// Copyright (c) 2026 ITERFORM. All rights reserved.
// tests/cuda_decode_kv_test.cu — the Q8 and Q4 K/V caches (include/limbo/cuda/kv.hpp, Gpu::set_kv_format) end to end on rung 1.
//
//   1. the switch: 60 prompt tokens go through the f16 cache, set_kv_format(Q8) quantises them, 8 more steps run on Q8; the logits are
//      compared with the same 8 steps on f16 (the difference is the cache's quantisation: printed, bounded) and the cache is 53 percent
//      of the f16 bytes;
//   2. Q8 decode is self-consistent: an 8-row window (step_rows) equals 8 sequential steps bit for bit, both reading and writing Q8;
//   3. Q4 (rotated keys): set_kv_format(Q4, 0) re-makes the cache empty, the same prefix and 8 steps run on Q4 and are compared with
//      f16 (bounded; a wrong rotation pairing would scramble every score); the window equals the steps bit for bit; the batched
//      prefill (Q4 written by the prompt pass, read through its f16 windows) continues into the same 8 steps.
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

int top1(const float* l, size_t n) { return static_cast<int>(std::max_element(l, l + n) - l); }
double max_diff(const float* a, const float* b, size_t n) {
    double m = 0;
    for (size_t i = 0; i < n; ++i) m = std::max(m, std::fabs(double(a[i]) - double(b[i])));
    return m;
}

}  // namespace

int main() {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        std::printf("SKIP cuda_decode_kv_test: no CUDA device\n");
        return 0;
    }
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    if (free_b < static_cast<size_t>(15.5e9)) {
        std::printf("SKIP cuda_decode_kv_test: %.2f GB free of %.2f GB, need 15.5 GB\n", double(free_b) / 1e9, double(total_b) / 1e9);
        return 0;
    }
    const char* env = std::getenv("LIMBO_TEST_GGUF");
    const std::string path = env && *env && !std::strchr(env, ':') ? env : kRung1;
    std::string err;
    auto model = limbo::model::Model::open({path}, &err);
    if (!model) {
        std::printf("SKIP cuda_decode_kv_test: cannot open %s: %s\n", path.c_str(), err.c_str());
        return 0;
    }
    auto gpu = limbo::cuda::Gpu::open(*model, 192, static_cast<uint64_t>(14.6e9), &err);
    CHECK(gpu != nullptr, "the engine opens");
    if (!gpu) {
        std::printf("     %s\n", err.c_str());
        return limbo::test::finish("cuda_decode_kv_test");
    }
    const size_t nv = static_cast<size_t>(gpu->n_vocab());
    constexpr int P = 60, W = 8;
    std::vector<int32_t> ids(P + W + 2);
    uint32_t s = 777;
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
    auto run8 = [&](std::vector<float>& out) {
        out.assign(size_t(W) * nv, 0.f);
        for (int t = 0; t < W; ++t)
            if (!gpu->step(ids[size_t(P + t)], P + t, true) || !gpu->read_logits(&out[size_t(t) * nv])) return false;
        return true;
    };

    const uint64_t f16_bytes = gpu->kv_cache_bytes();
    std::vector<float> ref, q8a;
    CHECK(prefix() && run8(ref), "f16: prefix and 8 steps");
    CHECK(prefix() && gpu->set_kv_format(limbo::cuda::KvFormat::Q8, P), "the prefix is quantised into a Q8 cache (set_kv_format)");
    CHECK(gpu->kv_format() == limbo::cuda::KvFormat::Q8, "the format is Q8");
    const double ratio = double(gpu->kv_cache_bytes()) / double(f16_bytes);
    std::printf("     cache bytes Q8 / f16 = %.3f\n", ratio);
    CHECK(ratio > 0.52 && ratio < 0.54, "the Q8 cache is 53 percent of the f16 bytes");
    CHECK(run8(q8a), "Q8: 8 steps");
    double worst = 0, mean = 0;
    int agree = 0;
    for (int t = 0; t < W; ++t) {
        const double d = max_diff(&ref[size_t(t) * nv], &q8a[size_t(t) * nv], nv);
        worst = std::max(worst, d);
        mean += d / W;
        agree += top1(&ref[size_t(t) * nv], nv) == top1(&q8a[size_t(t) * nv], nv);
    }
    std::printf("     Q8 against f16 over %d steps: max|dlogit| %.3e, mean of the per-step max %.3e, top-1 equal in %d of %d\n", W, worst, mean, agree, W);
    CHECK(worst <= 1.0, "Q8 cache: logits within 1.0 of the f16 cache's (quantisation, random tokens)");

    // Q8 self-consistency: the window equals the sequential steps (the engine stays on Q8)
    std::vector<float> seq, win(size_t(W) * nv);
    CHECK(prefix() && run8(seq), "Q8: prefix and 8 sequential steps");
    CHECK(prefix() && gpu->step_rows(ids.data() + P, W, P, true) && gpu->read_logits_rows(win.data(), W), "Q8: prefix and an 8-row window");
    CHECK(std::memcmp(seq.data(), win.data(), seq.size() * 4) == 0, "Q8: the window's logits equal 8 sequential steps bit for bit");
    CHECK(!gpu->set_kv_format(limbo::cuda::KvFormat::F16), "going back to f16 with the cache filled is refused");

    // ---- Q4 ----
    CHECK(gpu->set_kv_format(limbo::cuda::KvFormat::Q4, 0) && gpu->kv_format() == limbo::cuda::KvFormat::Q4,
          "set_kv_format(Q4, 0) re-makes the cache empty in Q4");
    const double r4 = double(gpu->kv_cache_bytes()) / double(f16_bytes);
    std::printf("     cache bytes Q4 / f16 = %.3f\n", r4);
    CHECK(r4 > 0.27 && r4 < 0.29, "the Q4 cache is 28 percent of the f16 bytes");
    std::vector<float> q4a;
    CHECK(prefix() && run8(q4a), "Q4: prefix and 8 steps");
    double worst4 = 0, mean4 = 0;
    int agree4 = 0;
    for (int t = 0; t < W; ++t) {
        const double d = max_diff(&ref[size_t(t) * nv], &q4a[size_t(t) * nv], nv);
        worst4 = std::max(worst4, d);
        mean4 += d / W;
        agree4 += top1(&ref[size_t(t) * nv], nv) == top1(&q4a[size_t(t) * nv], nv);
    }
    std::printf("     Q4 against f16 over %d steps: max|dlogit| %.3e, mean of the per-step max %.3e, top-1 equal in %d of %d\n", W, worst4, mean4,
                agree4, W);
    CHECK(worst4 <= 3.0 && agree4 >= W - 2, "Q4 cache: logits within 3.0 of the f16 cache's, top-1 equal in all but two steps");
    std::vector<float> seq4, win4(size_t(W) * nv);
    CHECK(prefix() && run8(seq4), "Q4: prefix and 8 sequential steps");
    CHECK(prefix() && gpu->step_rows(ids.data() + P, W, P, true) && gpu->read_logits_rows(win4.data(), W), "Q4: prefix and an 8-row window");
    CHECK(std::memcmp(seq4.data(), win4.data(), seq4.size() * 4) == 0, "Q4: the window's logits equal 8 sequential steps bit for bit");
    std::vector<float> pf4;
    gpu->reset();
    CHECK(gpu->prefill(ids.data(), P, 0, false) && run8(pf4), "Q4: the batched prefill of the prefix and 8 steps");
    double worst_pf = 0;
    int agree_pf = 0;
    for (int t = 0; t < W; ++t) {
        worst_pf = std::max(worst_pf, max_diff(&seq4[size_t(t) * nv], &pf4[size_t(t) * nv], nv));
        agree_pf += top1(&seq4[size_t(t) * nv], nv) == top1(&pf4[size_t(t) * nv], nv);
    }
    std::printf("     Q4 batched prefill against the token-by-token prefix: max|dlogit| %.3e, top-1 equal in %d of %d\n", worst_pf, agree_pf, W);
    CHECK(worst_pf <= 3.0 && agree_pf >= W - 2, "Q4: the batched prefill's cache continues like the steps' (within 3.0, top-1 in all but two)");
    CHECK(gpu->set_kv_format(limbo::cuda::KvFormat::F16, 0) && gpu->kv_format() == limbo::cuda::KvFormat::F16, "set_kv_format(F16, 0) goes back");
    std::vector<float> back;
    CHECK(prefix() && run8(back) && std::memcmp(back.data(), ref.data(), ref.size() * 4) == 0, "f16 again: the logits are the first run's bit for bit");
    return limbo::test::finish("cuda_decode_kv_test");
}
