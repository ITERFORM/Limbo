// Copyright (c) 2026 ITERFORM. All rights reserved.
// tests/cuda_gpu_test.cu — limbo::cuda::Gpu end to end on rung 1: the captured decode graph and the batched prefill.
//
//   1. step() replayed from a CUDA graph gives bit-identical logits to step() launched kernel by kernel.
//   2. prefill() with planes = 0 is the token-by-token path (bit-identical logits).
//   3. prefill() with 3 activation planes, in chunks of 32 and split over two calls (pos0 > 0), stays within
//      0.1 of the token-by-token logits (router near-ties can move individual experts, so this is not a
//      summation-order bound) and picks the same greedy token whenever the reference's top-2 gap exceeds 0.5.
//      The 2-plane numbers are printed, not asserted.
//
// The prompt is 64 pseudo-random token ids. The model is LIMBO_TEST_GGUF or the rung-1 file. SKIP (exit 0) when
// there is no GPU or under 15.5 GB free: the engine takes ~13.9 GB and the card is shared with the desktop.
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

struct Top {
    int i1 = 0;
    float gap = 0;
};
Top top_of(const std::vector<float>& l) {
    int i1 = 0, i2 = -1;
    for (size_t v = 1; v < l.size(); ++v) {
        if (l[v] > l[static_cast<size_t>(i1)]) {
            i2 = i1;
            i1 = static_cast<int>(v);
        } else if (i2 < 0 || l[v] > l[static_cast<size_t>(i2)]) {
            i2 = static_cast<int>(v);
        }
    }
    return {i1, l[static_cast<size_t>(i1)] - l[static_cast<size_t>(i2)]};
}
double max_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(double(a[i]) - double(b[i])));
    return m;
}

}  // namespace

int main() {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        std::printf("SKIP cuda_gpu_test: no CUDA device\n");
        return 0;
    }
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    if (free_b < static_cast<size_t>(15.5e9)) {
        std::printf("SKIP cuda_gpu_test: %.2f GB free of %.2f GB, need 15.5 GB (the desktop or another engine holds the rest)\n",
                    double(free_b) / 1e9, double(total_b) / 1e9);
        return 0;
    }
    const char* env = std::getenv("LIMBO_TEST_GGUF");
    const std::string path = env && *env && !std::strchr(env, ':') ? env : kRung1;
    std::string err;
    auto model = limbo::model::Model::open({path}, &err);
    if (!model) {
        std::printf("SKIP cuda_gpu_test: cannot open %s: %s\n", path.c_str(), err.c_str());
        return 0;
    }
    auto gpu = limbo::cuda::Gpu::open(*model, 256, static_cast<uint64_t>(14.6e9), &err);
    CHECK(gpu != nullptr, "the engine opens");
    if (!gpu) {
        std::printf("     %s\n", err.c_str());
        return limbo::test::finish("cuda_gpu_test");
    }
    const int nv = gpu->n_vocab();
    constexpr int N = 64;
    std::vector<int32_t> ids(N);
    uint32_t s = 99;
    for (int i = 0; i < N; ++i) {
        s = s * 1664525u + 1013904223u;
        ids[static_cast<size_t>(i)] = 1000 + static_cast<int32_t>((s >> 8) % 200000u);
    }
    std::vector<float> ref(static_cast<size_t>(nv)), got(static_cast<size_t>(nv));

    // 1. graph against kernel by kernel: the logits of the last 6 positions
    std::vector<std::vector<float>> eager(6, std::vector<float>(static_cast<size_t>(nv))), graph = eager;
    for (int mode = 0; mode < 2; ++mode) {
        gpu->set_graphs(mode == 1);
        gpu->reset();
        for (int i = 0; i < N; ++i) {
            const bool want = i >= N - 6;
            if (!gpu->step(ids[static_cast<size_t>(i)], i, want)) {
                CHECK(false, "step launches");
                return limbo::test::finish("cuda_gpu_test");
            }
            if (want) {
                auto& dst = (mode == 0 ? eager : graph)[static_cast<size_t>(i - (N - 6))];
                if (!gpu->read_logits(dst.data())) {
                    CHECK(false, "logits copy");
                    return limbo::test::finish("cuda_gpu_test");
                }
            }
        }
    }
    size_t differing = 0;
    for (size_t k = 0; k < 6; ++k) differing += std::memcmp(eager[k].data(), graph[k].data(), eager[k].size() * 4) != 0;
    std::printf("     CUDA graph vs kernel-by-kernel: %zu of 6 positions' logits differ\n", differing);
    CHECK(differing == 0, "the graph replay gives bit-identical logits");
    ref = eager[5];
    const Top rt = top_of(ref);

    // 2. planes = 0 is the token-by-token path
    gpu->set_graphs(true);
    gpu->set_prefill(0, 32);
    gpu->reset();
    CHECK(gpu->prefill(ids.data(), N, 0, true), "prefill (planes 0) launches");
    CHECK(gpu->read_logits(got.data()), "logits copy");
    CHECK(std::memcmp(ref.data(), got.data(), ref.size() * 4) == 0, "prefill with planes 0 equals the token-by-token logits bit for bit");

    // 3. the batched path
    for (int planes = 1; planes <= 3; ++planes) {
        gpu->set_prefill(planes, 32);
        gpu->reset();
        const bool ok = gpu->prefill(ids.data(), 40, 0, false) && gpu->prefill(ids.data() + 40, N - 40, 40, true);
        CHECK(ok, "batched prefill launches (two calls, chunks of 32)");
        CHECK(gpu->read_logits(got.data()), "logits copy");
        const double d = max_diff(ref, got);
        const Top bt = top_of(got);
        std::printf("     %d plane%s: max|dlogit| %.3e vs token by token, top-1 %s (reference gap %.3f)\n", planes, planes == 1 ? " " : "s", d,
                    bt.i1 == rt.i1 ? "equal" : "DIFFERENT", rt.gap);
        if (planes == 3) {
            CHECK(d <= 0.1, "3 planes: within 0.1 of the token-by-token logits");
            CHECK(rt.gap < 0.5f || bt.i1 == rt.i1, "3 planes: same greedy token when the reference gap exceeds 0.5");
        }
    }
    return limbo::test::finish("cuda_gpu_test");
}
