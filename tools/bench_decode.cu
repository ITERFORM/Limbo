// Copyright (c) 2026 ITERFORM. All rights reserved.
// tools/bench_decode.cu — the decode measurement session: device bandwidth, bytes per token, per-phase profile,
// tokens per second against context length, and the cost of a multi-row (verify) window.
//
//   bench_decode --devbw                                   device-to-device copy and pure-read bandwidth
//   bench_decode -m model.gguf --bytes                     weight bytes one decode token reads, by role
//   bench_decode -m model.gguf --ctx-sweep 512,4096 --gen 128 [--profile] [--no-graphs] [--max-ctx N]
//                [--kv f16|q8|q4|q8-synth|q4-synth]
//   bench_decode -m model.gguf --rows 1,2,3,4,5,6,7,8 [--ctx 512]       cost of step_rows(T) against T sequential steps
//
// Every GPU mode opens the engine once; the card is shared, so run under build/gpu.lock. The context is filled with
// a batched prefill of bench/prompts/long.txt (repeated to length); then --gen tokens are decoded greedily, one sync
// per token (the loop a server runs), and again without a sync per token (the device time of the steps).
#include <cuda_runtime.h>
#include <cupti.h>
#include <cxxabi.h>
#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "limbo/cuda/gemv.hpp"
#include "limbo/cuda/gpu.hpp"
#include "limbo/gguf.hpp"
#include "limbo/model.hpp"
#include "limbo/tok.hpp"
#include "limbo/types.hpp"

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

const char* kRung1 =
    "/home/alca/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-MTP-GGUF/snapshots/"
    "5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf";

std::vector<int> split_ints(const std::string& s) {
    std::vector<int> out;
    size_t at = 0;
    while (at <= s.size()) {
        size_t p = s.find(',', at);
        if (p == std::string::npos) p = s.size();
        if (p > at) out.push_back(std::atoi(s.substr(at, p - at).c_str()));
        at = p + 1;
    }
    return out;
}

#define CK(call)                                                                                      \
    do {                                                                                              \
        cudaError_t e_ = (call);                                                                      \
        if (e_ != cudaSuccess) {                                                                      \
            std::fprintf(stderr, "bench_decode: %s failed: %s (%s:%d)\n", #call, cudaGetErrorString(e_), __FILE__, __LINE__); \
            std::exit(1);                                                                             \
        }                                                                                             \
    } while (0)

// ---- device bandwidth -------------------------------------------------------------------------------------

// Pure read: every thread streams 16-byte loads with `U` independent loads in flight and folds them into one
// xor, so the compiler cannot drop the loads and nothing is written.
template <int U>
__global__ void read_kernel(const uint4* p, size_t n16, uint32_t* out) {
    uint32_t acc = 0;
    const size_t stride = size_t(gridDim.x) * blockDim.x * U;
    for (size_t i = size_t(blockIdx.x) * blockDim.x * U + threadIdx.x; i < n16; i += stride) {
        uint4 v[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const size_t j = i + size_t(u) * blockDim.x;
            v[u] = j < n16 ? __ldcs(p + j) : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int u = 0; u < U; ++u) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
    }
    if (acc == 0x9e3779b9u) out[0] = acc;
}

void devbw() {
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    int mem_clock_khz = 0, bus = 0;
    cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, 0);
    cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, 0);
    std::printf("device: %s sm_%d%d, %d SMs, L2 %d MiB, mem clock %d MHz x %d bit bus (nominal %.1f GB/s at DDR x2)\n", prop.name,
                prop.major, prop.minor, prop.multiProcessorCount, prop.l2CacheSize >> 20, mem_clock_khz / 1000, bus,
                2.0 * mem_clock_khz * 1e3 * bus / 8 / 1e9);
    size_t freeb = 0, totb = 0;
    CK(cudaMemGetInfo(&freeb, &totb));
    const size_t bytes = size_t(1) << 30;  // 1 GiB: far above the 32 MiB L2
    if (freeb < 3 * bytes) {
        std::fprintf(stderr, "bench_decode: only %.2f GB free\n", double(freeb) / 1e9);
        return;
    }
    char *a = nullptr, *b = nullptr;
    uint32_t* sink = nullptr;
    CK(cudaMalloc(&a, bytes));
    CK(cudaMalloc(&b, bytes));
    CK(cudaMalloc(&sink, 64));
    CK(cudaMemset(a, 1, bytes));
    CK(cudaMemset(b, 2, bytes));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    auto timeit = [&](auto&& fn, int reps) {
        fn();
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(e0));
        for (int i = 0; i < reps; ++i) fn();
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        return double(ms) / reps;
    };
    double ms = timeit([&] { CK(cudaMemcpyAsync(b, a, bytes, cudaMemcpyDeviceToDevice)); }, 20);
    std::printf("D2D cudaMemcpy 1 GiB: %.3f ms = %.1f GB/s counting the bytes copied, %.1f GB/s counting read+write\n", ms,
                double(bytes) / ms / 1e6, 2.0 * double(bytes) / ms / 1e6);
    const size_t n16 = bytes / 16;
    const int sms = prop.multiProcessorCount;
    double best = 0;
    for (int per_sm : {2, 4, 6, 8, 12, 16}) {
        for (int bs : {256, 512}) {
            const int grid = sms * per_sm * 256 / bs;
            const double t4 = timeit([&] { read_kernel<4><<<grid, bs>>>(reinterpret_cast<const uint4*>(a), n16, sink); }, 20);
            const double t1 = timeit([&] { read_kernel<1><<<grid, bs>>>(reinterpret_cast<const uint4*>(a), n16, sink); }, 20);
            const double g4 = double(bytes) / t4 / 1e6, g1 = double(bytes) / t1 / 1e6;
            best = std::max({best, g4, g1});
            std::printf("  read kernel, %2d x %3d threads/SM (grid %4d, block %d): U=1 %.1f GB/s, U=4 %.1f GB/s\n", per_sm, 256,
                        grid, bs, g1, g4);
        }
    }
    std::printf("pure-read kernel best: %.1f GB/s\n", best);
    cudaFree(a);
    cudaFree(b);
    cudaFree(sink);
}

// ---- per-kernel timeline through CUPTI (loaded at run time: the tool needs no link flag) ---------------------------

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
        if (rec->kind != CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL && rec->kind != CUPTI_ACTIVITY_KIND_KERNEL) continue;
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
        std::fprintf(stderr, "bench_decode: cannot load libcupti.so: %s\n", dlerror());
        return false;
    }
    g_cupti.regcb = reinterpret_cast<decltype(g_cupti.regcb)>(dlsym(g_cupti.h, "cuptiActivityRegisterCallbacks"));
    g_cupti.enable = reinterpret_cast<decltype(g_cupti.enable)>(dlsym(g_cupti.h, "cuptiActivityEnable"));
    g_cupti.disable = reinterpret_cast<decltype(g_cupti.disable)>(dlsym(g_cupti.h, "cuptiActivityDisable"));
    g_cupti.flush = reinterpret_cast<decltype(g_cupti.flush)>(dlsym(g_cupti.h, "cuptiActivityFlushAll"));
    g_cupti.next = reinterpret_cast<decltype(g_cupti.next)>(dlsym(g_cupti.h, "cuptiActivityGetNextRecord"));
    return g_cupti.regcb && g_cupti.enable && g_cupti.disable && g_cupti.flush && g_cupti.next;
}

// Runs `steps` decode steps under CUPTI (one step per call of `step`) and prints, per kernel name, the launches per
// step, the mean duration, the time per step, and the idle gaps between dependent kernels.
template <typename F>
void timeline(F&& step, int steps, int top) {
    if (!cupti_open()) return;
    std::vector<KRec> recs;
    g_krecs = &recs;
    g_cupti.regcb(cupti_req, cupti_done);
    g_cupti.enable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    for (int i = 0; i < steps; ++i) step();
    CK(cudaDeviceSynchronize());
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
    for (const KRec& r : recs) {
        Agg& a = agg[r.name + "  [grid " + std::to_string(r.grid) + "]"];
        a.us += double(r.end - r.start) / 1e3;
        a.n++;
        a.grid = r.grid;
        a.block = r.block;
        busy += double(r.end - r.start) / 1e3;
    }
    const double span = recs.empty() ? 0 : double(recs.back().end - recs.front().start) / 1e3;
    // time during which at least one kernel ran, and the idle remainder (launch gaps, dependency waits)
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
    std::printf("timeline: %zu kernels in %d steps (%.1f per step); span %.1f us per step, sum of kernel times %.1f us per step "
                "(concurrent branches overlap); some kernel running for %.1f us per step, idle between kernels %.1f us per step\n",
                recs.size(), steps, double(recs.size()) / steps, span / steps, busy / steps, covered / steps, (span - covered) / steps);
    std::vector<std::pair<std::string, Agg>> v(agg.begin(), agg.end());
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.second.us > b.second.us; });
    // the kernels by their own duration: how much of the step is many small launches
    {
        const double edge[] = {3, 6, 12, 25, 50, 1e30};
        double us[6] = {}, n[6] = {};
        for (const KRec& r : recs) {
            const double d = double(r.end - r.start) / 1e3;
            int b = 0;
            while (d >= edge[b]) ++b;
            us[b] += d;
            n[b] += 1;
        }
        const char* lab[] = {"<3us", "3-6us", "6-12us", "12-25us", "25-50us", ">=50us"};
        std::printf("   by duration:");
        for (int b = 0; b < 6; ++b) std::printf("  %s %.0f x = %.0f us", lab[b], n[b] / steps, us[b] / steps);
        std::printf("  (per step)\n");
    }
    if (const char* t = std::getenv("LIMBO_TIMELINE_TOP")) top = std::atoi(t);
    for (int i = 0; i < top && i < int(v.size()); ++i) {
        std::string nm = v[size_t(i)].first;
        if (nm.size() > 150) nm = nm.substr(0, 150);
        const Agg& a = v[size_t(i)].second;
        std::printf("   %8.1f us/step  %5.1f/step x %6.2f us  block %4d  %s\n", a.us / steps, double(a.n) / steps,
                    a.us / a.n, a.block, nm.c_str());
    }
}

// ---- launch overhead inside a graph ----------------------------------------------------------------------

__global__ void tiny_kernel(float* p) {
    if (threadIdx.x == 0 && blockIdx.x == 0) p[0] += 1.0f;
}
__global__ void tiny_dep_kernel(const float* in, float* out, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) out[i] = in[i] * 1.0001f + 1.0f;
}

void launch_gap() {
    float *a, *b;
    CK(cudaMalloc(&a, 1 << 20));
    CK(cudaMalloc(&b, 1 << 20));
    CK(cudaMemset(a, 0, 1 << 20));
    CK(cudaMemset(b, 0, 1 << 20));
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    auto run = [&](const char* name, auto&& body) {
        const int n = 1000;
        cudaGraph_t g;
        cudaGraphExec_t ge;
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
        for (int i = 0; i < n; ++i) body(i);
        CK(cudaStreamEndCapture(s, &g));
        CK(cudaGraphInstantiate(&ge, g, 0));
        for (int r = 0; r < 3; ++r) CK(cudaGraphLaunch(ge, s));
        CK(cudaStreamSynchronize(s));
        CK(cudaEventRecord(e0, s));
        for (int r = 0; r < 10; ++r) CK(cudaGraphLaunch(ge, s));
        CK(cudaEventRecord(e1, s));
        CK(cudaEventSynchronize(e1));
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        std::printf("  %-52s %6.2f us per kernel in a graph of %d\n", name, double(ms) * 1e3 / (10.0 * n), n);
        cudaGraphExecDestroy(ge);
        cudaGraphDestroy(g);
    };
    std::printf("launch gaps (a chain of dependent kernels, one stream):\n");
    run("1 block x 32 threads, no memory", [&](int) { tiny_kernel<<<1, 32, 0, s>>>(a); });
    run("34 blocks x 256 threads, copy 4 KB", [&](int i) { tiny_dep_kernel<<<34, 256, 0, s>>>(i & 1 ? b : a, i & 1 ? a : b, 1024); });
    run("136 blocks x 256 threads, copy 16 KB", [&](int i) { tiny_dep_kernel<<<136, 256, 0, s>>>(i & 1 ? b : a, i & 1 ? a : b, 4096); });
    run("512 blocks x 256 threads, copy 128 KB", [&](int i) { tiny_dep_kernel<<<512, 256, 0, s>>>(i & 1 ? b : a, i & 1 ? a : b, 32768); });
    cudaFree(a);
    cudaFree(b);
    cudaStreamDestroy(s);
}

// ---- GEMV microbenchmark ----------------------------------------------------------------------------------

__global__ void fill_kernel(uint8_t* p, size_t n, uint32_t seed) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        uint32_t h = uint32_t(i) * 2654435761u ^ seed;
        h ^= h >> 15;
        h *= 2246822519u;
        h ^= h >> 13;
        p[i] = uint8_t(h);
    }
}
__global__ void fill_f32_kernel(float* p, size_t n, uint32_t seed) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        uint32_t h = uint32_t(i) * 2654435761u ^ seed;
        h ^= h >> 15;
        h *= 2246822519u;
        h ^= h >> 13;
        p[i] = (float(h & 0xffff) - 32768.f) / 32768.f;
    }
}
// Random block bytes give NaN scales sometimes; the timing does not depend on the values (correctness is
// tests/cuda_gemv_test.cu's job).

uint32_t type_from_name(const std::string& n) {
    if (n == "f32") return uint32_t(limbo::Type::F32);
    if (n == "q8") return uint32_t(limbo::Type::Q8_0);
    if (n == "q6k") return uint32_t(limbo::Type::Q6_K);
    if (n == "iq2s") return uint32_t(limbo::Type::IQ2_S);
    if (n == "iq3s") return uint32_t(limbo::Type::IQ3_S);
    if (n == "iq4xs") return uint32_t(limbo::Type::IQ4_XS);
    if (n == "iq4nl") return uint32_t(limbo::Type::IQ4_NL);
    if (n == "q20") return uint32_t(limbo::Type::Q2_0);
    if (n == "iq3xxs") return uint32_t(limbo::Type::IQ3_XXS);
    return 0xffffffffu;
}

// spec = type:n_out:n_in[:mode] with mode dense (default), accum, edual (gate+up over 8 slots), eone (8 slots).
void gemv_bench(const std::string& spec) {
    std::vector<std::string> f;
    for (size_t at = 0; at <= spec.size();) {
        size_t p = spec.find(':', at);
        if (p == std::string::npos) p = spec.size();
        f.push_back(spec.substr(at, p - at));
        at = p + 1;
    }
    if (f.size() < 3) {
        std::fprintf(stderr, "bad --gemv spec '%s' (type:n_out:n_in[:mode])\n", spec.c_str());
        return;
    }
    const uint32_t type = type_from_name(f[0]);
    const int n_out = std::atoi(f[1].c_str()), n_in = std::atoi(f[2].c_str());
    const std::string mode = f.size() > 3 ? f[3] : "dense";
    std::string err;
    if (!limbo::cuda::init_tables(&err)) {
        std::fprintf(stderr, "init_tables: %s\n", err.c_str());
        return;
    }
    const uint64_t row_bytes = limbo::type_bytes(type, uint64_t(n_in));
    const uint64_t mat_bytes = row_bytes * uint64_t(n_out);
    const bool experts = mode == "edual" || mode == "eone" || mode == "egu" || mode == "edown";
    const int n_exp = 256, n_sel = 8;
    const uint64_t pool_target = uint64_t(192) << 20;
    // experts: two stacks of 256 experts (gate, up) are already far above the 32 MiB L2
    const int pool = experts ? 2 : int(std::max<uint64_t>(4, (pool_target + mat_bytes - 1) / mat_bytes));
    const uint64_t alloc = experts ? mat_bytes * n_exp : mat_bytes;
    std::vector<uint8_t*> mats(static_cast<size_t>(pool));
    for (int i = 0; i < pool; ++i) {
        CK(cudaMalloc(&mats[size_t(i)], alloc + 256));
        fill_kernel<<<512, 256>>>(mats[size_t(i)], alloc + 256, 0x1234u + uint32_t(i));
    }
    float *x = nullptr, *y0 = nullptr, *y1 = nullptr;
    CK(cudaMalloc(&x, size_t(n_in) * 8 * 4 + 4096));
    CK(cudaMalloc(&y0, size_t(n_out) * 8 * 4 + 4096));
    CK(cudaMalloc(&y1, size_t(n_out) * 8 * 4 + 4096));
    fill_f32_kernel<<<64, 256>>>(x, size_t(n_in) * 8, 77u);
    const int launches = pool > 24 ? std::min(pool, 300) : 24;  // enough launches to cycle every matrix: nothing stays in the 32 MiB L2
    std::vector<int32_t> hid(size_t(launches) * n_sel);
    for (size_t i = 0; i < hid.size(); ++i) hid[i] = int32_t((i * 37 + (i / n_sel) * 101) % n_exp);
    int32_t* ids = nullptr;
    CK(cudaMalloc(&ids, hid.size() * 4));
    CK(cudaMemcpy(ids, hid.data(), hid.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaDeviceSynchronize());
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    auto one = [&](int k) {
        const uint8_t* W = mats[size_t(k % pool)];
        const uint8_t* W2 = mats[size_t((k + 1) % pool)];
        bool ok = true;
        if (mode == "dense") ok = limbo::cuda::gemv(type, W, x, y0, n_out, n_in, false, s);
        else if (mode == "accum") ok = limbo::cuda::gemv(type, W, x, y0, n_out, n_in, true, s);
        else if (mode == "edual")
            ok = limbo::cuda::gemv_experts_dual(type, W, W2, mat_bytes, ids + (k % launches) * n_sel, n_sel, x, y0, y1, n_out, n_in, s);
        else if (mode == "egu") {
            ok = limbo::cuda::gemv_experts_gate_up(type, W, W2, mat_bytes, ids + (k % launches) * n_sel, n_sel, x, y0, n_out, n_in, s);
        } else if (mode == "edown") {
            ok = limbo::cuda::gemv_experts_down_fast(type, W, mat_bytes, ids + (k % launches) * n_sel, n_sel, x, y0, n_out, n_in, s);
        } else if (mode == "eone")
            ok = limbo::cuda::gemv_experts(type, W, mat_bytes, ids + (k % launches) * n_sel, n_sel, x, n_in, y0, n_out, n_in, s);
        if (!ok) {
            std::fprintf(stderr, "gemv launch failed\n");
            std::exit(1);
        }
    };
    for (int k = 0; k < 4; ++k) one(k);
    CK(cudaStreamSynchronize(s));
    cudaGraph_t g;
    cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    for (int k = 0; k < launches; ++k) one(k);
    CK(cudaStreamEndCapture(s, &g));
    CK(cudaGraphInstantiate(&ge, g, 0));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    for (int r = 0; r < 3; ++r) CK(cudaGraphLaunch(ge, s));
    CK(cudaStreamSynchronize(s));
    const int reps = 20;
    CK(cudaEventRecord(e0, s));
    for (int r = 0; r < reps; ++r) CK(cudaGraphLaunch(ge, s));
    CK(cudaEventRecord(e1, s));
    CK(cudaEventSynchronize(e1));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    const double us = double(ms) * 1e3 / (double(reps) * launches);
    const double bytes = experts ? double(mat_bytes) * n_sel * ((mode == "edual" || mode == "egu") ? 2 : 1) : double(mat_bytes);
    std::printf("gemv %-6s %6d x %5d %-6s: %9.2f MB/launch, %8.2f us/launch (graph of %d, %d matrices cycled) = %6.1f GB/s\n", f[0].c_str(),
                n_out, n_in, mode.c_str(), bytes / 1e6, us, launches, pool, bytes / (us * 1e3));
    for (uint8_t* m : mats) cudaFree(m);
    cudaFree(x);
    cudaFree(y0);
    cudaFree(y1);
    cudaFree(ids);
    cudaGraphExecDestroy(ge);
    cudaGraphDestroy(g);
    cudaStreamDestroy(s);
}

// ---- weight bytes per decode token ------------------------------------------------------------------------

void bytes_per_token(const limbo::model::Model& m) {
    const limbo::model::Geometry& g = m.geometry();
    struct Row {
        std::string role;
        std::string type;
        std::map<std::string, int> types;  // type name -> number of layers
        uint64_t bytes = 0;
        int count = 0;
    };
    std::map<std::string, Row> rows;
    auto add = [&](int layer, const char* role, double frac, const char* tag) {
        const limbo::gguf::Tensor* t = m.tensor(layer, role);
        if (!t) return;
        Row& r = rows[std::string(tag)];
        r.role = tag;
        const limbo::TypeInfo* ti = limbo::type_info(t->type);
        r.type = ti ? std::string(ti->name) : "?";
        ++r.types[r.type];
        r.bytes += static_cast<uint64_t>(double(t->nbytes) * frac);
        ++r.count;
    };
    for (int l = 0; l < g.n_layer_main; ++l) {
        if (g.is_full_attention(l)) {
            add(l, "attn_q.weight", 1, "attn q");
            add(l, "attn_k.weight", 1, "attn k");
            add(l, "attn_v.weight", 1, "attn v");
            add(l, "attn_output.weight", 1, "attn o");
        } else {
            add(l, "attn_qkv.weight", 1, "gdn qkv");
            add(l, "attn_gate.weight", 1, "gdn z");
            add(l, "ssm_out.weight", 1, "gdn out");
            add(l, "ssm_alpha.weight", 1, "gdn alpha");
            add(l, "ssm_beta.weight", 1, "gdn beta");
        }
        add(l, "ffn_gate_inp.weight", 1, "router");
        const double f = double(g.n_expert_used) / double(g.n_expert);
        add(l, "ffn_gate_exps.weight", f, "experts gate (8 of 256)");
        add(l, "ffn_up_exps.weight", f, "experts up (8 of 256)");
        add(l, "ffn_down_exps.weight", f, "experts down (8 of 256)");
        add(l, "ffn_gate_shexp.weight", 1, "shared gate");
        add(l, "ffn_up_shexp.weight", 1, "shared up");
        add(l, "ffn_down_shexp.weight", 1, "shared down");
    }
    add(-1, "output.weight", 1, "head");
    uint64_t tot = 0;
    std::printf("weight bytes read by one decode token (qwen35moe, %d layers):\n", g.n_layer_main);
    for (auto& kv : rows) {
        std::string ty;
        for (auto& t : kv.second.types) ty += t.first + ":" + std::to_string(t.second) + " ";
        std::printf("  %-26s x%2d  %9.2f MB  %s\n", kv.first.c_str(), kv.second.count, double(kv.second.bytes) / 1e6, ty.c_str());
        tot += kv.second.bytes;
    }
    std::printf("  total %.1f MB (%.3f GiB)\n", double(tot) / 1e6, double(tot) / double(1ull << 30));
}

std::vector<int32_t> long_prompt_ids(const limbo::tok::Tokenizer& tok, const std::string& path, int n) {
    std::ifstream in(path, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    std::vector<int32_t> base;
    limbo::tok::Options o;
    o.parse_special = true;
    tok.encode(ss.str(), base, o);
    std::vector<int32_t> ids(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) ids[static_cast<size_t>(i)] = base[static_cast<size_t>(i) % base.size()];
    return ids;
}

// The same "@@ name" prompt files as tools/gpu_forward.
std::string read_prompt_file(const std::string& path, const std::string& want) {
    std::ifstream in(path, std::ios::binary);
    std::string line, cur_name, cur_text, found;
    bool have = false, got = false;
    while (std::getline(in, line)) {
        if (line.rfind("@@ ", 0) == 0) {
            if (have && cur_name == want) {
                found = cur_text;
                got = true;
            }
            cur_name = line.substr(3);
            cur_text.clear();
            have = true;
            continue;
        }
        if (have) {
            cur_text += line;
            cur_text += '\n';
        }
    }
    if (have && cur_name == want) {
        found = cur_text;
        got = true;
    }
    return got ? found : std::string();
}

// Greedy speculative decoding with the model's own MTP block against plain greedy decode of the same prompt. The prompt but
// its last `tail` tokens goes through the batched prefill; the tail goes through single steps (plain) or step_rows windows
// (spec), which leave identical states, so both must emit the same tokens.
//   cycle: refresh the block over the rows the last window accepted (the last row's draft is the first draft), chain k - 1
//   more drafts through the block, verify [c, d1..dk] in one write_back = false window, commit the accepted prefix.
void run_spec(limbo::cuda::Gpu& gpu, const limbo::tok::Tokenizer& tok, const std::string& prompt_dir, const std::string& name, int n_gen,
              int k_fixed, int k_max, int tail_arg) {
    std::string text = read_prompt_file(prompt_dir + "/" + name + ".txt", name);
    std::vector<int32_t> ids;
    limbo::tok::Options o;
    o.parse_special = true;
    if (name.rfind("long:", 0) == 0) {
        ids = long_prompt_ids(tok, prompt_dir + "/long.txt", std::atoi(name.c_str() + 5));
    } else {
        if (text.empty()) {
            std::fprintf(stderr, "bench_decode: no prompt '%s'\\n", name.c_str());
            return;
        }
        tok.encode(text, ids, o);
    }
    const int n = static_cast<int>(ids.size());
    const int tail = std::min(tail_arg, n - 1);
    const int nv = gpu.n_vocab();
    (void)nv;
    auto prefill_head = [&]() {
        gpu.reset();
        gpu.sync();
        if (n - tail > 0 && !gpu.prefill(ids.data(), n - tail, 0, false)) std::exit(1);
    };

    // ---- plain greedy decode
    prefill_head();
    for (int q = n - tail; q < n; ++q)
        if (!gpu.step(ids[size_t(q)], q, q == n - 1)) std::exit(1);
    std::vector<int32_t> plain;
    plain.push_back(gpu.argmax());
    const auto t0 = Clock::now();
    for (int i = 1; i < n_gen; ++i) {
        if (!gpu.step(plain.back(), n - 1 + i, true)) std::exit(1);
        plain.push_back(gpu.argmax());
    }
    const double plain_ms = ms_since(t0);

    // ---- calibrate the window and the block (a few timed passes each) for the draft-length policy
    prefill_head();
    for (int q = n - tail; q < n; q += 8) {
        const int T = std::min(8, n - q);
        if (!gpu.step_rows(ids.data() + q, T, q, true)) std::exit(1);
    }
    std::vector<double> t_win(9, 0.0);
    {
        std::vector<int32_t> w(8, ids[0]);
        const int pos = n;
        for (int T = 1; T <= 8; ++T) {
            gpu.sync();
            const auto tc = Clock::now();
            for (int r = 0; r < 3; ++r) {
                if (!gpu.step_rows(w.data(), T, pos, false)) std::exit(1);
                if (!gpu.commit(0)) std::exit(1);
            }
            gpu.sync();
            t_win[size_t(T)] = ms_since(tc) / 3;
        }
    }
    double t_chain = 0;
    {
        int32_t tk = ids[0], out1 = 0;
        gpu.sync();
        const auto tc = Clock::now();
        for (int r = 0; r < 4; ++r) {
            if (!gpu.mtp_rows(&tk, 1, n + 8, false)) std::exit(1);
            if (!gpu.mtp_argmax(&out1, 1)) std::exit(1);
        }
        t_chain = ms_since(tc) / 4;
    }

    // ---- speculative decode
    struct Acc {
        double a = 0.65;  // acceptance of the j-th draft given the earlier ones, an EMA
    };
    std::vector<Acc> acc(16);
    long cycles = 0, drafted = 0, accepted_total = 0;
    std::vector<long> k_hist(16, 0);
    auto run_once = [&](int kfix) {
        prefill_head();
        int P = n - tail;
        std::vector<int32_t> out;
        // the tail in windows; the block follows every window but the last
        int last_T = 0;
        std::vector<int32_t> prev;  // tokens of the last window
        for (int q = n - tail; q < n; q += 8) {
            const int T = std::min(8, n - q);
            if (!gpu.step_rows(ids.data() + q, T, q, true)) std::exit(1);
            last_T = T;
            prev.assign(ids.begin() + q, ids.begin() + q + T);
            P = q + T;
            if (q + T < n && !gpu.mtp_rows(ids.data() + q + 1, T, q, false)) std::exit(1);
        }
        std::vector<int32_t> am(8);
        if (!gpu.argmax_rows(am.data(), last_T)) std::exit(1);
        int32_t c = am[size_t(last_T - 1)];
        int R = last_T;  // rows of the last window the block has not seen with their next token
        const auto ts = Clock::now();
        while (int(out.size()) < n_gen) {
            // refresh: the first R rows of the last window: row j has hidden state of position P - R + j and next token prev[j + 1] (the last: c)
            std::vector<int32_t> nxt(static_cast<size_t>(R), 0);
            for (int j = 0; j + 1 < R; ++j) nxt[size_t(j)] = prev[size_t(j) + 1];
            nxt[size_t(R - 1)] = c;
            if (!gpu.mtp_rows(nxt.data(), R, P - R, false)) std::exit(1);
            std::vector<int32_t> rowarg(static_cast<size_t>(R), 0);
            if (!gpu.mtp_argmax(rowarg.data(), R)) std::exit(1);
            // the draft length for this cycle
            int k = kfix;
            if (k <= 0) {
                double best = -1;
                for (int kk = 1; kk <= k_max; ++kk) {
                    double e = 1, pr = 1;
                    for (int j = 1; j <= kk; ++j) {
                        pr *= acc[size_t(j)].a;
                        e += pr;
                    }
                    const double cost = t_chain * kk + t_win[size_t(kk + 1)] + 0.3;
                    if (e / cost > best) {
                        best = e / cost;
                        k = kk;
                    }
                }
            }
            std::vector<int32_t> d(size_t(k) + 1);  // d[1..k]
            d[1] = rowarg[size_t(R - 1)];
            for (int j = 2; j <= k; ++j) {
                if (!gpu.mtp_rows(&d[size_t(j - 1)], 1, P + j - 2, true)) std::exit(1);
                if (!gpu.mtp_argmax(&d[size_t(j)], 1)) std::exit(1);
            }
            std::vector<int32_t> w(size_t(k) + 1);
            w[0] = c;
            for (int j = 1; j <= k; ++j) w[size_t(j)] = d[size_t(j)];
            if (!gpu.step_rows(w.data(), k + 1, P, false)) std::exit(1);
            if (!gpu.argmax_rows(am.data(), k + 1)) std::exit(1);
            int m = 0;
            while (m < k && am[size_t(m)] == d[size_t(m + 1)]) ++m;
            if (!gpu.commit(m + 1)) std::exit(1);
            for (int j = 0; j <= m; ++j) out.push_back(w[size_t(j)]);
            for (int j = 1; j <= k; ++j) {  // the EMA of the acceptance of draft j given the earlier ones
                if (j <= m + 1) acc[size_t(j)].a = 0.9 * acc[size_t(j)].a + 0.1 * (j <= m ? 1.0 : 0.0);
            }
            ++cycles;
            drafted += k;
            accepted_total += m;
            ++k_hist[size_t(k)];
            c = am[size_t(m)];
            prev = w;
            R = m + 1;
            P += m + 1;
        }
        out.resize(size_t(n_gen));
        return std::make_pair(out, ms_since(ts));
    };
    const int kuse = k_fixed;
    // a warm-up run (graph capture, caches), then the timed one
    run_once(kuse);
    cycles = drafted = accepted_total = 0;
    std::fill(k_hist.begin(), k_hist.end(), 0);
    auto res = run_once(kuse);
    const std::vector<int32_t>& spec = res.first;
    size_t first = spec.size();
    for (size_t i = 0; i < spec.size() && i < plain.size(); ++i)
        if (spec[i] != plain[i]) {
            first = i;
            break;
        }
    // plain emits n_gen tokens in (n_gen - 1) timed steps (the first token comes from the prompt's last position)
    const double plain_tps = double(n_gen - 1) / (plain_ms / 1e3), spec_tps = double(n_gen) / (res.second / 1e3);
    std::printf("spec '%s' (%d prompt tokens, k %s): plain %.2f tok/s, speculative %.2f tok/s (x%.2f); tokens %s; %.2f tokens per cycle, %ld cycles, "
                "%.0f%% of %ld drafts accepted; k used:",
                name.c_str(), n, kuse > 0 ? std::to_string(kuse).c_str() : "auto", plain_tps, spec_tps, spec_tps / plain_tps,
                first == spec.size() ? "IDENTICAL" : ("DIFFER at " + std::to_string(first)).c_str(), double(n_gen) / double(cycles), cycles,
                drafted ? 100.0 * double(accepted_total) / double(drafted) : 0.0, drafted);
    for (int kk = 1; kk <= k_max; ++kk)
        if (k_hist[size_t(kk)]) std::printf(" %d:%ld", kk, k_hist[size_t(kk)]);
    std::printf("\n   window ms:");
    for (int T = 1; T <= 8; ++T) std::printf(" T%d %.2f", T, t_win[size_t(T)]);
    std::printf("; block step %.2f ms\n", t_chain);
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path = kRung1, prompt_dir = "bench/prompts", sweep, rows_arg;
    int n_gen = 128, max_ctx = 0, rows_ctx = 512;
    std::string kv_mode = "f16", spec_names;
    int spec_k = 0, spec_kmax = 4, spec_tail = 16, spec_dv = 0;
    std::vector<std::string> gemv_specs;
    int timeline_steps = 0;
    bool do_launch = false, do_devbw = false, do_bytes = false, profile = false, graphs = true;
    double min_free_gb = 14.0;
    for (int i = 1; i < argc; ++i) {
        auto is = [&](const char* a) { return std::strcmp(argv[i], a) == 0; };
        auto val = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
        if (is("-m") || is("--model")) model_path = val();
        else if (is("--devbw")) do_devbw = true;
        else if (is("--launch-gap")) do_launch = true;
        else if (is("--timeline")) timeline_steps = std::atoi(val());
        else if (is("--bytes")) do_bytes = true;
        else if (is("--ctx-sweep")) sweep = val();
        else if (is("--gen")) n_gen = std::atoi(val());
        else if (is("--max-ctx")) max_ctx = std::atoi(val());
        else if (is("--profile")) profile = true;
        else if (is("--no-graphs")) graphs = false;
        else if (is("--gemv")) gemv_specs.push_back(val());
        else if (is("--spec")) spec_names = val();
        else if (is("--kv")) kv_mode = val();
        else if (is("--k")) spec_k = std::atoi(val());
        else if (is("--kmax")) spec_kmax = std::atoi(val());
        else if (is("--tail")) spec_tail = std::atoi(val());
        else if (is("--draft-vocab")) spec_dv = std::atoi(val());
        else if (is("--rows")) rows_arg = val();
        else if (is("--ctx")) rows_ctx = std::atoi(val());
        else if (is("--prompt-dir")) prompt_dir = val();
        else if (is("--min-free-gb")) min_free_gb = std::atof(val());
        else {
            std::fprintf(stderr, "usage: bench_decode [--devbw] [-m model] [--bytes] [--ctx-sweep N,N] [--gen N] [--profile] [--rows T,T] [--ctx N]\n");
            return 2;
        }
    }
    if (do_devbw) {
        devbw();
        return 0;
    }
    if (do_launch) {
        launch_gap();
        return 0;
    }
    if (!gemv_specs.empty()) {
        for (const std::string& sp : gemv_specs) gemv_bench(sp);
        return 0;
    }
    std::string err;
    auto model = limbo::model::Model::open({model_path}, &err);
    if (!model) {
        std::fprintf(stderr, "bench_decode: %s\n", err.c_str());
        return 1;
    }
    if (do_bytes) bytes_per_token(*model);
    if (sweep.empty() && rows_arg.empty() && spec_names.empty()) return 0;

    limbo::tok::Vocab vocab;
    if (!limbo::tok::Vocab::load(*model->shards()[0], vocab, &err)) {
        std::fprintf(stderr, "bench_decode: %s\n", err.c_str());
        return 1;
    }
    const limbo::tok::Tokenizer tok(vocab);
    const std::vector<int> ctxs = split_ints(sweep);
    const std::vector<int> rowsv = split_ints(rows_arg);
    int need = max_ctx;
    for (int c : ctxs) need = std::max(need, c + 2 * n_gen + 40);
    need = std::max(need, rows_ctx + 2 * n_gen + 40);
    if (!spec_names.empty()) need = std::max(need, 2048 + 4 * n_gen + 64);
    // the cache is made in the asked format (a 262144-position f16 cache does not fit; q8 / q4 prefill straight into theirs)
    const auto open_fmt = kv_mode.rfind("q4", 0) == 0 ? limbo::cuda::KvFormat::Q4
                          : kv_mode.rfind("q8", 0) == 0 ? limbo::cuda::KvFormat::Q8 : limbo::cuda::KvFormat::F16;
    auto gpu = limbo::cuda::Gpu::open(*model, need, static_cast<uint64_t>(min_free_gb * 1e9), &err, open_fmt);
    if (!gpu) {
        std::fprintf(stderr, "bench_decode: %s\n", err.c_str());
        return 3;
    }
    gpu->set_graphs(graphs);
    std::printf("engine: n_ctx %d, weights %.3f GB, vram %.3f GB, free after open %.2f GB\n", gpu->n_ctx(),
                double(gpu->weight_bytes()) / 1e9, double(gpu->vram_bytes()) / 1e9, double(gpu->free_after_bytes()) / 1e9);

    if (!spec_names.empty()) {
        if (!gpu->mtp_load(*model)) {
            std::fprintf(stderr, "bench_decode: the file has no MTP block (or it does not fit)\n");
            return 1;
        }
        std::printf("MTP block loaded: engine vram now %.3f GB\n", double(gpu->vram_bytes()) / 1e9);
        std::vector<std::string> names;
        for (size_t at = 0; at <= spec_names.size();) {
            size_t p = spec_names.find(',', at);
            if (p == std::string::npos) p = spec_names.size();
            if (p > at) names.push_back(spec_names.substr(at, p - at));
            at = p + 1;
        }
        if (spec_dv > 0 && !gpu->mtp_set_draft_vocab(spec_dv)) return 1;
        for (const std::string& nm : names) run_spec(*gpu, tok, prompt_dir, nm, n_gen, spec_k, spec_kmax, spec_tail);
        return 0;
    }
    for (int ctx : ctxs) {
        const bool synth = kv_mode == "q8-synth" || kv_mode == "q4-synth";
        const auto synth_fmt = kv_mode == "q4-synth" ? limbo::cuda::KvFormat::Q4 : limbo::cuda::KvFormat::Q8;
        const std::vector<int32_t> ids = synth ? std::vector<int32_t>() : long_prompt_ids(tok, prompt_dir + "/long.txt", ctx);
        gpu->reset();
        gpu->sync();
        const auto t_pre = Clock::now();
        std::vector<int32_t> gen;
        if (synth) {
            // no prefill: a pseudo-random Q8 / Q4 cache of ctx positions (decode time does not depend on the values)
            if (gpu->kv_format() != synth_fmt && !gpu->set_kv_format(synth_fmt, 0)) {
                std::fprintf(stderr, "bench_decode: cannot switch to the %s cache\n", kv_mode.c_str());
                return 1;
            }
            if (!gpu->kv_fill_synthetic(ctx)) return 1;
            gen.push_back(1000);
        } else {
            if (!gpu->prefill(ids.data(), ctx, 0, true)) {
                std::fprintf(stderr, "bench_decode: prefill of %d failed\n", ctx);
                return 1;
            }
            gen.push_back(gpu->argmax());
            if (kv_mode == "q8" && !gpu->set_kv_format(limbo::cuda::KvFormat::Q8, ctx)) {
                std::fprintf(stderr, "bench_decode: cannot switch to the Q8 cache\n");
                return 1;
            }
        }
        const double pre_ms = ms_since(t_pre);
        // a few warm-up steps (the graph is captured at the second), not timed
        int pos = ctx;
        for (int k = 0; k < 3; ++k) {
            if (!gpu->step(gen.back(), pos++, true)) return 1;
            gen.push_back(gpu->argmax());
        }
        const auto t_dec = Clock::now();
        for (int k = 0; k < n_gen; ++k) {
            if (!gpu->step(gen.back(), pos++, true)) {
                std::fprintf(stderr, "bench_decode: step failed at %d\n", pos);
                return 1;
            }
            gen.push_back(gpu->argmax());
        }
        const double dec_ms = ms_since(t_dec);
        // the same steps without a sync per token: the device time of the loop
        const auto t_pipe = Clock::now();
        for (int k = 0; k < n_gen; ++k) {
            if (!gpu->step(gen[size_t(k % int(gen.size()))], pos++, true)) {
                std::fprintf(stderr, "bench_decode: step failed at %d\n", pos);
                return 1;
            }
        }
        gpu->sync();
        const double pipe_ms = ms_since(t_pipe);
        std::printf("ctx %6d: prefill %.0f ms (%.0f tok/s); decode %d tokens: %.2f tok/s (%.3f ms/token, greedy, a sync per token); "
                    "pipelined %.2f tok/s (%.3f ms/token)\n",
                    ctx, pre_ms, double(ctx) / (pre_ms / 1e3), n_gen, double(n_gen) / (dec_ms / 1e3), dec_ms / n_gen,
                    double(n_gen) / (pipe_ms / 1e3), pipe_ms / n_gen);
        std::printf("   ids:");
        for (size_t i = 0; i < std::min<size_t>(gen.size(), 20); ++i) std::printf(" %d", gen[i]);
        std::printf("\n   allids:");
        for (size_t i = 0; i < std::min<size_t>(gen.size(), size_t(n_gen) + 4); ++i) std::printf(" %d", gen[i]);
        std::printf("\n   kv %s: cache %.3f GB, engine vram %.3f GB\n", kv_mode.c_str(), double(gpu->kv_cache_bytes()) / 1e9, double(gpu->vram_bytes()) / 1e9);
        if (timeline_steps > 0) {
            timeline([&] {
                const int p1 = pos++;
                gpu->step(gen[size_t(p1 % int(gen.size()))], p1, true);
            }, timeline_steps, 40);
        }
        if (profile) {
            std::vector<std::pair<std::string, double>> prof;
            gpu->set_decode_profile(&prof);
            const int n_prof = 16;
            for (int k = 0; k < n_prof; ++k) {
                gpu->step(gen.back(), pos++, true);
                gpu->sync();
            }
            gpu->set_decode_profile(nullptr);
            double tot = 0;
            for (auto& kv : prof) tot += kv.second;
            std::printf("   profile at ctx %d (events, eager, %d steps, %.3f ms per step):\n", ctx, n_prof, tot / n_prof);
            for (auto& kv : prof)
                std::printf("      %-24s %8.3f ms/step  %5.1f%%\n", kv.first.c_str(), kv.second / n_prof, 100.0 * kv.second / tot);
        }
        std::fflush(stdout);
    }
    if (!rowsv.empty()) {
        const int ctx = rows_ctx;
        // the timeline walks fresh windows of the text (realistic misses), so the prompt runs on past them
        const std::vector<int32_t> ids = long_prompt_ids(tok, prompt_dir + "/long.txt", ctx + 64 + 8 * (timeline_steps + 8));
        gpu->reset();
        gpu->sync();
        if (!gpu->prefill(ids.data(), ctx, 0, true)) {
            std::fprintf(stderr, "bench_decode: prefill failed\n");
            return 1;
        }
        gpu->sync();
        std::printf("verify windows at context %d (graph replay, no sync between windows; the state advances by T per window)\n", ctx);
        std::printf("  T   window ms   ratio to T=1   ms per token   |  no write-back ms   commit(T) ms   | T sequential steps ms\n");
        double base = 0;
        // q4e has its own window pass (Gpu::q4e_step_rows / q4e_commit)
        auto rows = [&](const int32_t* tk, int T, int p0, bool wb) {
            return gpu->is_q4e() ? gpu->q4e_step_rows(tk, T, p0, wb) : gpu->step_rows(tk, T, p0, wb);
        };
        auto commit = [&](int n) { return gpu->is_q4e() ? gpu->q4e_commit(n) : gpu->commit(n); };
        for (int T : rowsv) {
            if (T < 1 || T > 8) continue;
            const int K = 24;
            int pos = ctx;
            std::vector<int32_t> w(ids.begin() + ctx, ids.begin() + ctx + T);
            auto time_ms = [&](auto&& fn) {
                for (int i = 0; i < 3; ++i) fn();
                gpu->sync();
                const auto t0 = Clock::now();
                for (int i = 0; i < K; ++i) fn();
                gpu->sync();
                return ms_since(t0) / K;
            };
            const double wb = time_ms([&] {
                if (!rows(w.data(), T, pos, true)) std::exit(1);
                pos += T;
                if (pos + 8 > gpu->n_ctx() - 8) pos = ctx;
            });
            pos = ctx;
            const double nwb = time_ms([&] {
                if (!rows(w.data(), T, pos, false)) std::exit(1);
            });
            const double both = time_ms([&] {
                if (!rows(w.data(), T, pos, false) || !commit(T)) std::exit(1);
                pos += T;
                if (pos + 8 > gpu->n_ctx() - 8) pos = ctx;
            });
            pos = ctx;
            const double seq = time_ms([&] {
                for (int t = 0; t < T; ++t) {
                    if (!gpu->step(w[size_t(t)], pos + t, true)) std::exit(1);
                }
                pos += T;
                if (pos + 8 > gpu->n_ctx() - 8) pos = ctx;
            });
            if (timeline_steps > 0) {
                // consecutive windows of new text, written back: the routing (and the lane's share) of a real decode
                std::printf("  T=%d windows over fresh text, written back:\n", T);
                gpu->reset();
                gpu->sync();
                if (!gpu->prefill(ids.data(), ctx, 0, true)) std::exit(1);
                int tp = ctx;
                for (int k = 0; k < 4; ++k, tp += T) rows(ids.data() + tp, T, tp, true);   // warm
                gpu->sync();
                timeline([&] {
                    rows(ids.data() + tp, T, tp, true);
                    tp += T;
                }, timeline_steps, 40);
            }
            if (T == 1) base = wb;
            std::printf("  %d   %9.3f   %12.2f   %12.3f   |  %16.3f   %12.3f   | %10.3f\n", T, wb, base > 0 ? wb / base : 1.0, wb / T, nwb,
                        both - nwb, seq);
        }
    }
    return 0;
}
