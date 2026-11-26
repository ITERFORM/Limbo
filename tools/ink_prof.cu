// Copyright (c) 2026 ITERFORM. All rights reserved.
// tools/ink_prof.cu — the per-kernel timeline of one InkGpu decode step (src/cuda/ink_gpu.cu), through CUPTI
// loaded at run time (the tools/bench_decode.cu idiom: the tool needs no link flag). For each kernel name and
// grid it prints the launches per step, the mean duration and the time per step, plus where the step's wall
// time goes: busy kernels vs the idle gaps between them (launch overhead, dependency waits).
//
//   ink_prof [file.gguf] [--steps N]      (default LIMBO_TEST_INK_GGUF or the Inkling-2-Mini UD-IQ2_XXS)
//   ink_prof [file.gguf] --prefill N      the prompt pass instead: N tokens in InkGpu::prefill_rows pieces, the
//                                         per-step figures then per prompt token
//
// The run: open at n_ctx 512, 8 warmup steps, then N profiled head steps on one stream with no per-step sync
// (the way a decode loop runs), one sync at the end. No GPU or no model: SKIP, exit 0.
#include <cuda_runtime.h>
#include <cupti.h>
#include <cxxabi.h>
#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "../src/cuda/ink_gpu.hpp"
#include "limbo/model.hpp"

namespace {

namespace cu = limbo::cuda;

constexpr const char* kModel = "/home/alca/models/inkling/Inkling-2-Mini-UD-IQ2_XXS.gguf";

// ---- the CUPTI timeline (the tools/bench_decode.cu machinery) ------------------------------------------
struct KRec {
    std::string name;
    uint64_t start, end;
    int grid, block;
};
std::vector<KRec>* g_krecs = nullptr;
struct Cupti {
    void* h = nullptr;
    CUptiResult (*regcb)(CUpti_BuffersCallbackRequestFunc, CUpti_BuffersCallbackCompleteFunc) = nullptr;
    CUptiResult (*enable)(CUpti_ActivityKind) = nullptr;
    CUptiResult (*disable)(CUpti_ActivityKind) = nullptr;
    CUptiResult (*flush)(uint32_t) = nullptr;
    CUptiResult (*next)(uint8_t*, size_t, CUpti_Activity**) = nullptr;
} g_cupti;

void CUPTIAPI cupti_req(uint8_t** buf, size_t* size, size_t* max_records) {
    *size = 16u << 20;
    *buf = static_cast<uint8_t*>(std::aligned_alloc(8, *size));
    *max_records = 0;
}
void CUPTIAPI cupti_done(CUcontext, uint32_t, uint8_t* buf, size_t, size_t valid) {
    CUpti_Activity* rec = nullptr;
    while (g_cupti.next(buf, valid, &rec) == CUPTI_SUCCESS) {
        if (rec->kind != CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL && rec->kind != CUPTI_ACTIVITY_KIND_KERNEL)
            continue;
        const CUpti_ActivityKernel13* k = reinterpret_cast<const CUpti_ActivityKernel13*>(rec);
        KRec r;
        int st = 0;
        char* dm = abi::__cxa_demangle(k->name, nullptr, nullptr, &st);
        r.name = st == 0 && dm ? dm : k->name;
        std::free(dm);
        r.start = k->start;
        r.end = k->end;
        r.grid = int(k->gridX * k->gridY * k->gridZ);
        r.block = int(k->blockX * k->blockY * k->blockZ);
        g_krecs->push_back(r);
    }
    std::free(buf);
}

bool cupti_open() {
    g_cupti.h = dlopen("libcupti.so", RTLD_NOW);
    if (!g_cupti.h) g_cupti.h = dlopen("/opt/cuda/targets/x86_64-linux/lib/libcupti.so", RTLD_NOW);
    if (!g_cupti.h) {
        std::fprintf(stderr, "ink_prof: cannot load libcupti.so: %s\n", dlerror());
        return false;
    }
    g_cupti.regcb = reinterpret_cast<decltype(g_cupti.regcb)>(dlsym(g_cupti.h, "cuptiActivityRegisterCallbacks"));
    g_cupti.enable = reinterpret_cast<decltype(g_cupti.enable)>(dlsym(g_cupti.h, "cuptiActivityEnable"));
    g_cupti.disable = reinterpret_cast<decltype(g_cupti.disable)>(dlsym(g_cupti.h, "cuptiActivityDisable"));
    g_cupti.flush = reinterpret_cast<decltype(g_cupti.flush)>(dlsym(g_cupti.h, "cuptiActivityFlushAll"));
    g_cupti.next = reinterpret_cast<decltype(g_cupti.next)>(dlsym(g_cupti.h, "cuptiActivityGetNextRecord"));
    return g_cupti.regcb && g_cupti.enable && g_cupti.disable && g_cupti.flush && g_cupti.next;
}

}  // namespace

int main(int argc, char** argv) {
    const char* env = std::getenv("LIMBO_TEST_INK_GGUF");
    std::string path = argc > 1 && std::strncmp(argv[1], "--", 2) != 0 ? argv[1] : (env && *env ? env : kModel);
    int steps = 16, prefill = 0, depth = 0, ctx_arg = 0;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--steps") && i + 1 < argc) steps = std::atoi(argv[++i]);
        if (!std::strcmp(argv[i], "--prefill") && i + 1 < argc) prefill = std::atoi(argv[++i]);
        if (!std::strcmp(argv[i], "--depth") && i + 1 < argc) depth = std::atoi(argv[++i]);
        if (!std::strcmp(argv[i], "--ctx") && i + 1 < argc) ctx_arg = std::atoi(argv[++i]);
    }
    if (prefill > 0) steps = prefill;
    const int ctx = ctx_arg > 0 ? ctx_arg : prefill > 0 ? depth + prefill + cu::InkGpu::kPiece + 256 : std::max(512, depth + steps + 64);

    int dev = 0;
    if (cudaGetDeviceCount(&dev) != cudaSuccess || dev == 0) {
        std::printf("ink_prof: SKIP (no GPU)\n");
        return 0;
    }
    {
        std::FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) {
            std::printf("ink_prof: SKIP (no model at %s)\n", path.c_str());
            return 0;
        }
        std::fclose(f);
    }
    if (!cupti_open()) return 1;

    std::string err;
    std::unique_ptr<limbo::model::Model> m = limbo::model::Model::open({path}, &err);
    if (!m) {
        std::printf("ink_prof: %s\n", err.c_str());
        return 1;
    }
    std::unique_ptr<cu::InkGpu> gpu = cu::InkGpu::open(*m, ctx, 0, &err);
    if (!gpu) {
        std::printf("ink_prof: %s\n", err.c_str());
        return 1;
    }
    std::printf("ink_prof: %s, %d slots, %d %s profiled at n_ctx %d\n", path.c_str(), gpu->n_slots(), steps,
                prefill > 0 ? "prompt tokens" : "steps", ctx);

    std::vector<int32_t> toks(size_t(std::max(steps, 1)) + cu::InkGpu::kPiece + size_t(depth) + 16);
    for (size_t i = 0; i < toks.size(); ++i) toks[i] = 1000 + int32_t((i * 7919) % 150000);
    // --depth: the context is filled first (prompt pieces), the profiled work then runs at that depth
    int p0 = 0;
    for (; p0 < depth;) {
        const int T = std::min(cu::InkGpu::kPiece, depth - p0);
        gpu->prefill_rows(toks.data() + p0, T, p0, false);
        p0 += T;
    }
    if (prefill > 0) {
        gpu->prefill_rows(toks.data() + p0, cu::InkGpu::kPiece, p0, false);   // warm-up piece
        p0 += cu::InkGpu::kPiece;
    } else {
        for (int i = 0; i < 8; ++i) gpu->step(1910 + i % 32, p0 + i, true);
        p0 += 8;
    }
    gpu->sync();

    std::vector<KRec> recs;
    g_krecs = &recs;
    g_cupti.regcb(cupti_req, cupti_done);
    g_cupti.enable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    const auto t0 = std::chrono::steady_clock::now();
    if (prefill > 0) {
        for (int p = 0; p < prefill;) {
            const int T = std::min(cu::InkGpu::kPiece, prefill - p);
            gpu->prefill_rows(toks.data() + p0 + p, T, p0 + p, p + T == prefill);
            p += T;
        }
    } else {
        for (int i = 0; i < steps; ++i) gpu->step(1910 + (i * 7919) % 32, p0 + i, true);
    }
    gpu->sync();
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    g_cupti.flush(1);
    g_cupti.disable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    g_krecs = nullptr;

    std::sort(recs.begin(), recs.end(), [](const KRec& a, const KRec& b) { return a.start < b.start; });
    struct Agg {
        double us = 0;
        int n = 0;
        int grid = 0, block = 0;
    };
    std::map<std::string, Agg> agg;
    double busy = 0;
    double tiny_us = 0;
    int tiny_n = 0;
    for (const KRec& r : recs) {
        Agg& a = agg[r.name + "  [grid " + std::to_string(r.grid) + "]"];
        const double us = double(r.end - r.start) / 1e3;
        a.us += us;
        a.n++;
        a.grid = r.grid;
        a.block = r.block;
        busy += us;
        if (us < 5.0) {
            tiny_us += us;
            ++tiny_n;
        }
    }
    const double span = recs.empty() ? 0 : double(recs.back().end - recs.front().start) / 1e3;
    double covered = 0;
    {
        uint64_t cur_s = 0, cur_e = 0;
        bool have = false;
        for (const KRec& r : recs) {
            if (!have) {
                cur_s = r.start;
                cur_e = r.end;
                have = true;
            } else if (r.start <= cur_e) {
                cur_e = std::max(cur_e, r.end);
            } else {
                covered += double(cur_e - cur_s) / 1e3;
                cur_s = r.start;
                cur_e = r.end;
            }
        }
        if (have) covered += double(cur_e - cur_s) / 1e3;
    }
    std::printf(
        "timeline: %zu kernels in %d steps (%.1f per step)\n"
        "  wall %.1f us/step (%.1f tok/s), kernel span %.1f us/step, busy %.1f us/step, "
        "idle between kernels %.1f us/step\n"
        "  kernels under 5 us: %.0f per step, %.1f us/step (%.0f%% of busy)\n",
        recs.size(), steps, double(recs.size()) / steps, wall / steps * 1e6, double(steps) / wall,
        span / steps, busy / steps, (span - covered) / steps, double(tiny_n) / steps, tiny_us / steps,
        busy > 0 ? 100.0 * tiny_us / busy : 0.0);
    std::vector<std::pair<std::string, Agg>> v(agg.begin(), agg.end());
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.second.us > b.second.us; });
    for (int i = 0; i < 30 && i < int(v.size()); ++i) {
        std::string nm = v[size_t(i)].first;
        if (nm.size() > 140) nm = nm.substr(0, 140);
        const Agg& a = v[size_t(i)].second;
        std::printf("   %8.1f us/step  %5.1f/step x %6.2f us  block %4d  %s\n", a.us / steps, double(a.n) / steps,
                    a.us / a.n, a.block, nm.c_str());
    }
    return 0;
}
