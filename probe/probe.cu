// Copyright (c) 2026 ITERFORM. All rights reserved.
// probe/probe.cu — limbo-probe: the W0 measurement session of PLAN.md §7.
//
// Every number the planner is built on comes from here, on this desktop, measured rather than taken
// from a spec. Items, as numbered in PLAN.md §7:
//
//   1. DDR4 sustained read, 1..8 threads, per CCX, alone and beside a PCIe DMA.
//   2. PCIe H2D from cacheable vs write-combined pinned memory, 416 KiB and 2 MiB chunks, copy engine
//      vs in-kernel mapped pulls, alone and under GPU load.
//   4. CPU GQA flash-decode per 256-token q8_0 page, 1..8 rows.
//   5. Doorbell round trip GPU -> CPU -> GPU through mapped flags.
//   6. Interference: DDR beside DMA, launch latency with the DDR path saturated.
//   8. dTLB: 4 KiB vs 2 MiB pages, random page walk.
//   9. SSD random and sequential rates.
//
// Deferred, with the reason: item 3 (AVX2 µs per expert row for each ladder format) belongs with the
// kernels that will be verified in W4 — the i-quant layouts need the ggml codebook grids, which enter
// with a notice, and a row rule measured on an unverified dequant would be a number nobody can trust.
// Item 7 (routing statistics) needs a forward pass, so it lands in W1 with the reference.
//
//   make probe && build/bin/tools/limbo-probe > bench/probe-<date>.txt
//
// Output is one fact per line: "<item> <name> <value> <unit>", so the report is generated from the run,
// not written by hand.
#include <cuda_runtime.h>
#include <immintrin.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/statfs.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

void pin_to_cpu(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    sched_setaffinity(0, sizeof set, &set);
}

double median(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, size_t(p * double(v.size())))];
}

float f16_to_f32(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000) << 16;
    const uint32_t exp = (h >> 10) & 0x1f;
    const uint32_t man = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            int e = -1;
            uint32_t m = man;
            do {
                ++e;
                m <<= 1;
            } while ((m & 0x400) == 0);
            bits = sign | (uint32_t(127 - 15 - e) << 23) | ((m & 0x3ff) << 13);
        }
    } else if (exp == 0x1f) {
        bits = sign | 0x7f800000u | (man << 13);
    } else {
        bits = sign | ((exp + 112) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

void say(const char* item, const std::string& name, double value, const char* unit) {
    std::printf("%-4s %-46s %10.3f %s\n", item, name.c_str(), value, unit);
    std::fflush(stdout);
}

// ---------------------------------------------------------------------------------------------
// item 1 / 6: DDR4 sustained read
// ---------------------------------------------------------------------------------------------
constexpr size_t kDdrBytes = 256ull << 20;

// Buffers are kept between runs: a measurement that spends half its time memsetting is not measuring
// the memory path.
std::map<int, std::vector<void*>>& ddr_cache() {
    static std::map<int, std::vector<void*>> c;
    return c;
}

double ddr_read_gbs(int n_threads, int first_cpu, bool pin_each) {
    auto& cache = ddr_cache()[n_threads];
    const size_t per = kDdrBytes / size_t(n_threads);
    if (cache.empty()) {
        for (int t = 0; t < n_threads; ++t) {
            void* p = nullptr;
            if (posix_memalign(&p, 4096, per) != 0) return 0;
            std::memset(p, t + 1, per);
            cache.push_back(p);
        }
    }
    std::vector<double> acc(size_t(n_threads), 0);
    std::vector<std::thread> pool;
    const auto t0 = Clock::now();
    for (int t = 0; t < n_threads; ++t) {
        pool.emplace_back([&, t] {
            if (pin_each) pin_to_cpu(first_cpu + t);
            const float* p = static_cast<const float*>(cache[size_t(t)]);
            const size_t n = per / sizeof(float);
            __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
            size_t i = 0;
            for (; i + 16 <= n; i += 16) {
                s0 = _mm256_add_ps(s0, _mm256_load_ps(p + i));
                s1 = _mm256_add_ps(s1, _mm256_load_ps(p + i + 8));
            }
            const __m256 s = _mm256_add_ps(s0, s1);
            alignas(32) float lane[8];
            _mm256_store_ps(lane, s);
            double sum = 0;
            for (float f : lane) sum += f;
            acc[size_t(t)] = sum;
        });
    }
    for (auto& th : pool) th.join();
    const double ms = ms_since(t0);
    double sink = 0;
    for (double a : acc) sink += a;
    static volatile double keep_alive;
    keep_alive = sink;  // the accumulation is the measurement, so it must not be optimised away
    return double(kDdrBytes) / (ms / 1000.0) / 1e9;
}

// ---------------------------------------------------------------------------------------------
// item 8: dTLB, 4 KiB vs 2 MiB pages
// ---------------------------------------------------------------------------------------------
// The metric is nanoseconds per random page touch, with the same region and the same number of
// touches for both page sizes: a GB/s figure over 64 bytes per page would compare 65,536 touches
// against 128 and measure nothing. Reported with the page count, so the result is interpretable.
//
// The region must be large enough that the touched footprint leaves the caches: one cache line per
// page over 256 MiB is 4 MiB of lines, which fits in one core's L2 (4 MiB here) and measures the L2,
// not the dTLB. 2 GiB and 2M touches give a 128 MiB footprint, past the 32 MiB L3.
//
// A random touch walk still mixes the TLB and the caches. What isolates the page size is a
// dependent-load pointer chase: every hop is a load whose address is the value of the previous load,
// so nothing can be prefetched and the hop cost is the memory system's latency for that page size —
// page walk included. Same region, same hop count, only the page size changes.
unsigned long long& huge_page_kb() {
    static unsigned long long v = 0;
    return v;
}

double tlb_chase_ns(size_t bytes, bool huge, size_t hops, size_t* out_pages) {
    const size_t page = huge ? (2ull << 20) : 4096ull;
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
    if (p == MAP_FAILED) return 0;
    madvise(p, bytes, huge ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
    std::memset(p, 0, bytes);
    const size_t n_pages = bytes / page;
    *out_pages = n_pages;

    // One node per hop, each in its own page where there are enough pages, at a random 64-byte-aligned
    // offset inside it, chained into a cycle.
    std::vector<uintptr_t> addr(hops);
    uint64_t seed = 0x243f6a8885a308d3ull;
    auto next_rand = [&] {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        return seed;
    };
    for (size_t i = 0; i < hops; ++i) {
        const size_t pg = size_t(next_rand() >> 17) % n_pages;
        const size_t off = (size_t(next_rand() >> 17) % (page / 64)) * 64;
        addr[i] = reinterpret_cast<uintptr_t>(static_cast<uint8_t*>(p) + pg * page + off);
    }
    for (size_t i = 0; i < hops; ++i)
        *reinterpret_cast<uintptr_t*>(addr[i]) = addr[(i + 1) % hops];

    uintptr_t cur = addr[0];
    const auto t0 = Clock::now();
    for (size_t i = 0; i < hops; ++i) cur = *reinterpret_cast<const uintptr_t*>(cur);
    const double ms = ms_since(t0);
    volatile uintptr_t sink = cur;
    (void)sink;
    // Without this the 2 MiB run may be a 4 KiB run wearing a label: MADV_HUGEPAGE is a request, and
    // with MAP_POPULATE the faults happen before khugepaged can collapse anything. Measured, not
    // assumed: read AnonHugePages for the mapping out of /proc/self/smaps.
    if (huge) {
        FILE* f = std::fopen("/proc/self/smaps", "r");
        if (f) {
            char line[512];
            unsigned long long lo = 0, hi = 0, anon_huge = 0, found = 0;
            while (std::fgets(line, sizeof line, f)) {
                unsigned long long a = 0, b = 0;
                if (std::sscanf(line, "%llx-%llx", &a, &b) == 2) {
                    lo = a;
                    hi = b;
                    continue;
                }
                if (std::sscanf(line, "AnonHugePages: %llu", &anon_huge) == 1) {
                    if (reinterpret_cast<uintptr_t>(p) >= lo && reinterpret_cast<uintptr_t>(p) < hi) found = anon_huge;
                }
            }
            std::fclose(f);
            huge_page_kb() = found;
        }
    }
    munmap(p, bytes);
    return ms * 1e6 / double(hops);
}

double tlb_ns_per_touch(size_t bytes, bool huge, size_t touches, size_t* out_pages) {
    const size_t page = huge ? (2ull << 20) : 4096ull;
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
    if (p == MAP_FAILED) return 0;
    madvise(p, bytes, huge ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
    std::memset(p, 1, bytes);
    const size_t n_pages = bytes / page;
    *out_pages = n_pages;
    std::vector<size_t> order(touches);
    uint64_t seed = 0x9e3779b97f4a7c15ull;
    for (size_t i = 0; i < touches; ++i) {
        seed = seed * 6364136223846793005ull + 1442695040888963407ull;
        order[i] = size_t(seed >> 17) % n_pages;
    }
    volatile uint8_t sink = 0;
    const auto t0 = Clock::now();
    for (size_t i = 0; i < touches; ++i) sink = uint8_t(sink + static_cast<const uint8_t*>(p)[order[i] * page]);
    const double ms = ms_since(t0);
    (void)sink;
    munmap(p, bytes);
    return ms * 1e6 / double(touches);
}

// Reads the transparent-hugepage policy so the 2 MiB number can be interpreted: if it is "never",
// the 2 MiB run is really a 4 KiB run and the comparison means nothing.
std::string thp_policy() {
    FILE* f = std::fopen("/sys/kernel/mm/transparent_hugepage/enabled", "r");
    if (!f) return "unknown";
    char buf[256] = {0};
    if (std::fgets(buf, sizeof buf, f)) {}
    std::fclose(f);
    std::string s(buf);
    while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
    return s;
}

// ---------------------------------------------------------------------------------------------
// item 4: CPU GQA flash-decode over a 256-token q8_0 page (one KV head, 4 query heads, head_dim 128)
// ---------------------------------------------------------------------------------------------
struct Q8Page {
    std::vector<uint8_t> k, v;
    int n_tokens = 256, head_dim = 128;
};

void q8_dequant(const uint8_t* blk, float* out, int n) {
    const uint8_t* p = blk;
    for (int i = 0; i < n; i += 32, p += 34) {
        const float d = f16_to_f32(uint16_t(p[0] | (p[1] << 8)));
        const int8_t* q = reinterpret_cast<const int8_t*>(p + 2);
        for (int j = 0; j < 32; ++j) out[i + j] = d * float(q[j]);
    }
}

double gqa_page_us(const Q8Page& page, int rows) {
    const int hd = page.head_dim;
    std::vector<float> kf(size_t(page.n_tokens) * hd), vf(size_t(page.n_tokens) * hd);
    std::vector<float> q(size_t(rows) * hd, 0.01f), out(size_t(rows) * hd), sc(size_t(page.n_tokens));
    std::vector<double> times;
    const int reps = 200;
    for (int r = 0; r < reps; ++r) {
        const auto t0 = Clock::now();
        q8_dequant(page.k.data(), kf.data(), int(kf.size()));
        q8_dequant(page.v.data(), vf.data(), int(vf.size()));
        for (int row = 0; row < rows; ++row) {
            const float* qr = q.data() + size_t(row) * hd;
            float m = -1e30f;
            for (int t = 0; t < page.n_tokens; ++t) {
                const float* kt = kf.data() + size_t(t) * hd;
                __m256 acc = _mm256_setzero_ps();
                for (int d = 0; d + 8 <= hd; d += 8)
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(qr + d), _mm256_loadu_ps(kt + d), acc);
                alignas(32) float lane[8];
                _mm256_store_ps(lane, acc);
                const float s =
                    (((lane[0] + lane[1]) + (lane[2] + lane[3])) + ((lane[4] + lane[5]) + (lane[6] + lane[7]))) *
                    0.08838834764831845f;  // 1/sqrt(128)
                sc[size_t(t)] = s;
                m = std::max(m, s);
            }
            float z = 0;
            for (int t = 0; t < page.n_tokens; ++t) {
                sc[size_t(t)] = std::exp(sc[size_t(t)] - m);
                z += sc[size_t(t)];
            }
            float* orow = out.data() + size_t(row) * hd;
            for (int d = 0; d < hd; ++d) orow[d] = 0;
            for (int t = 0; t < page.n_tokens; ++t) {
                const float p = sc[size_t(t)] / z;
                const float* vt = vf.data() + size_t(t) * hd;
                for (int d = 0; d + 8 <= hd; d += 8)
                    _mm256_storeu_ps(orow + d, _mm256_fmadd_ps(_mm256_set1_ps(p), _mm256_loadu_ps(vt + d),
                                                              _mm256_loadu_ps(orow + d)));
            }
        }
        times.push_back(std::chrono::duration<double, std::micro>(Clock::now() - t0).count());
    }
    return median(times);
}

// ---------------------------------------------------------------------------------------------
// CUDA: item 2 (PCIe), item 5 (doorbell), item 6 (DMA beside the CPU)
// ---------------------------------------------------------------------------------------------
__global__ void pull_kernel(const float* __restrict__ src, float* __restrict__ sink, size_t n) {
    const size_t stride = size_t(gridDim.x) * blockDim.x;
    float acc = 0;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) acc += src[i];
    if (acc == -1.0f) *sink = acc;
}

__global__ void burn_kernel(float* sink, int iters) {
    float a = 1.0f + threadIdx.x * 1e-9f;
    for (int i = 0; i < iters; ++i) a = fmaf(a, 1.0000001f, 1e-12f);
    if (a == -1.0f) *sink = a;
}

__global__ void doorbell_kernel(volatile unsigned int* flag, volatile unsigned int* reply) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    unsigned int seen = 0;
    for (;;) {
        const unsigned int v = *flag;
        if (v == 0xFFFFFFFFu) return;
        if (v != seen) {
            seen = v;
            *reply = seen;
        }
    }
}

double pcie_h2d_gbs(void* host, size_t chunk, int chunks, bool mapped_pull, cudaStream_t stream, float* dev) {
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    if (mapped_pull) {
        void* dev_ptr = nullptr;
        if (cudaHostGetDevicePointer(&dev_ptr, host, 0) != cudaSuccess) return 0;
        const size_t n = chunk / sizeof(float);
        cudaEventRecord(a, stream);
        for (int i = 0; i < chunks; ++i)
            pull_kernel<<<2048, 256, 0, stream>>>(static_cast<const float*>(dev_ptr), dev, n);
        cudaEventRecord(b, stream);
    } else {
        cudaEventRecord(a, stream);
        for (int i = 0; i < chunks; ++i)
            cudaMemcpyAsync(dev + size_t(i % 8) * (chunk / sizeof(float)), host, chunk, cudaMemcpyHostToDevice, stream);
        cudaEventRecord(b, stream);
    }
    cudaEventSynchronize(b);
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return double(chunk) * double(chunks) / (double(ms) / 1000.0) / 1e9;
}

}  // namespace

int main(int argc, char** argv) {
    bool quick = false;
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--quick") == 0) quick = true;

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::printf("# limbo-probe — PLAN.md §7 measurements\n");
    std::printf("# device: %s, sm_%d.%d, %d SMs, %.1f GiB VRAM\n", prop.name, prop.major, prop.minor,
                prop.multiProcessorCount, double(prop.totalGlobalMem) / 1073741824.0);
    std::printf("# host: %d cpus, DDR buffers %zu MiB, transparent_hugepage=%s\n\n",
                int(std::thread::hardware_concurrency()), kDdrBytes >> 20, thp_policy().c_str());

    // ---- item 1: DDR alone -------------------------------------------------------------------
    for (int t : {1, 2, 4, 8})
        say("1", "ddr read, " + std::to_string(t) + " threads, pinned", ddr_read_gbs(t, 0, true), "GB/s");
    say("1", "ddr read, 4 threads on cpus 0-3 (one CCX)", ddr_read_gbs(4, 0, true), "GB/s");
    say("1", "ddr read, 4 threads on cpus 4-7 (other CCX)", ddr_read_gbs(4, 4, true), "GB/s");
    say("1", "ddr read, 8 threads, unpinned", ddr_read_gbs(8, 0, false), "GB/s");

    // ---- item 8: dTLB ------------------------------------------------------------------------
    const size_t tlb_bytes = 4ull << 30;
    const size_t tlb_hops = 1ull << 20;
    size_t pages4k = 0, pages2m = 0;
    const double ns4k = tlb_chase_ns(tlb_bytes, false, tlb_hops, &pages4k);
    const double ns2m = tlb_chase_ns(tlb_bytes, true, tlb_hops, &pages2m);
    say("8", "dependent-load chase, 4 KiB pages (" + std::to_string(pages4k) + " pages)", ns4k, "ns/hop");
    say("8", "dependent-load chase, 2 MiB pages (" + std::to_string(pages2m) + " pages)", ns2m, "ns/hop");
    if (ns2m > 0) say("8", "4 KiB / 2 MiB cost ratio", ns4k / ns2m, "x");
    say("8", "AnonHugePages actually backing the 2 MiB run", double(huge_page_kb()) / 1024.0, "MiB");

    cudaSetDevice(0);
    float* dev = nullptr;
    cudaMalloc(&dev, 64ull << 20);
    cudaStream_t stream = nullptr, load_stream = nullptr;
    cudaStreamCreate(&stream);
    cudaStreamCreate(&load_stream);

    // ---- item 2: PCIe H2D -------------------------------------------------------------------
    const int chunks_416k = quick ? 16 : 256;
    const int chunks_2m = quick ? 8 : 128;
    const char* names[3] = {"cacheable pinned", "write-combined pinned", "mapped, in-kernel pull"};
    for (int mode = 0; mode < 3; ++mode) {
        void* host = nullptr;
        const unsigned flags =
            mode == 1 ? cudaHostAllocWriteCombined : (mode == 2 ? cudaHostAllocMapped : cudaHostAllocDefault);
        if (cudaHostAlloc(&host, 2ull << 20, flags) != cudaSuccess) {
            say("2", std::string("H2D ") + names[mode] + ": allocation failed", 0, "GB/s");
            continue;
        }
        std::memset(host, 7, 2ull << 20);
        for (int sz = 0; sz < 2; ++sz) {
            const size_t chunk = sz == 0 ? (416ull << 10) : (2ull << 20);
            const int n = sz == 0 ? chunks_416k : chunks_2m;
            say("2", std::string("H2D ") + names[mode] + ", " + (sz == 0 ? "416 KiB" : "2 MiB"),
                pcie_h2d_gbs(host, chunk, n, mode == 2, stream, dev), "GB/s");
        }
        burn_kernel<<<1024, 256, 0, load_stream>>>(dev, 1 << 22);
        say("2", std::string("H2D ") + names[mode] + ", 2 MiB, beside a busy GPU",
            pcie_h2d_gbs(host, 2ull << 20, chunks_2m, mode == 2, stream, dev), "GB/s");
        cudaStreamSynchronize(load_stream);
        cudaFreeHost(host);
    }

    // ---- item 5: doorbell --------------------------------------------------------------------
    {
        unsigned int* flag = nullptr;
        unsigned int* reply = nullptr;
        cudaHostAlloc(&flag, 4096, cudaHostAllocMapped);
        cudaHostAlloc(&reply, 4096, cudaHostAllocMapped);
        *flag = 0;
        *reply = 0;
        unsigned int* dflag = nullptr;
        unsigned int* dreply = nullptr;
        cudaHostGetDevicePointer(&dflag, flag, 0);
        cudaHostGetDevicePointer(&dreply, reply, 0);
        doorbell_kernel<<<1, 32, 0, stream>>>(dflag, dreply);
        std::vector<double> trips;
        unsigned int v = 0;
        for (int i = 0; i < 2000; ++i) {
            const auto t0 = Clock::now();
            ++v;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *flag = v;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            while (*reply != v) _mm_pause();
            trips.push_back(std::chrono::duration<double, std::micro>(Clock::now() - t0).count());
        }
        *flag = 0xFFFFFFFFu;
        cudaStreamSynchronize(stream);
        say("5", "doorbell round trip, median", median(trips), "us");
        say("5", "doorbell round trip, p99", percentile(trips, 0.99), "us");
        cudaFreeHost(flag);
        cudaFreeHost(reply);
    }

    // ---- item 6: interference ----------------------------------------------------------------
    {
        void* host = nullptr;
        cudaHostAlloc(&host, 8ull << 20, cudaHostAllocDefault);
        std::memset(host, 3, 8ull << 20);
        std::atomic<bool> run{true};
        std::thread dma([&] {
            for (int i = 0; run.load(); ++i)
                cudaMemcpyAsync(dev + size_t(i % 8) * (2ull << 20), host, 8ull << 20, cudaMemcpyHostToDevice, load_stream);
            cudaStreamSynchronize(load_stream);
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        say("6", "ddr read, 8 threads, beside an 8 MiB DMA loop", ddr_read_gbs(8, 0, true), "GB/s");
        run = false;
        dma.join();
        cudaFreeHost(host);
    }
    {
        std::atomic<bool> run{true};
        std::thread cpu([&] {
            while (run.load()) ddr_read_gbs(8, 0, true);
        });
        std::vector<double> lat;
        for (int i = 0; i < 300; ++i) {
            const auto t0 = Clock::now();
            burn_kernel<<<64, 256, 0, stream>>>(dev, 1024);
            cudaStreamSynchronize(stream);
            lat.push_back(std::chrono::duration<double, std::micro>(Clock::now() - t0).count());
        }
        run = false;
        cpu.join();
        say("6", "empty-kernel launch+sync, DDR saturated, median", median(lat), "us");
        say("6", "empty-kernel launch+sync, DDR saturated, p99", percentile(lat, 0.99), "us");
    }
    {
        std::vector<double> lat;
        for (int i = 0; i < 300; ++i) {
            const auto t0 = Clock::now();
            burn_kernel<<<64, 256, 0, stream>>>(dev, 1024);
            cudaStreamSynchronize(stream);
            lat.push_back(std::chrono::duration<double, std::micro>(Clock::now() - t0).count());
        }
        say("6", "empty-kernel launch+sync, idle box, median", median(lat), "us");
    }

    // ---- item 4: GQA flash-decode over a q8_0 page -------------------------------------------
    {
        Q8Page page;
        page.k.resize(size_t(256) * 128 / 32 * 34);
        page.v.resize(page.k.size());
        uint64_t seed = 12345;
        for (auto& b : page.k) {
            seed = seed * 6364136223846793005ull + 1442695040888963407ull;
            b = uint8_t(seed >> 33);
        }
        for (auto& b : page.v) {
            seed = seed * 6364136223846793005ull + 1442695040888963407ull;
            b = uint8_t(seed >> 33);
        }
        for (int rows : {1, 2, 3, 4, 8})
            say("4", "q8_0 page decode, 256 tokens, " + std::to_string(rows) + " row(s)", gqa_page_us(page, rows), "us");
    }

    // ---- item 9: SSD ------------------------------------------------------------------------
    {
        const char* path = "build/tmp/probe-ssd.bin";
        const size_t total = quick ? (64ull << 20) : (512ull << 20);
        // Which filesystem this is matters: a filesystem that ignores O_DIRECT (or a network mount)
        // would let the page cache answer, and 4 KiB "random reads" would report cache latency.
        struct statfs sfs{};
        if (statfs("build/tmp", &sfs) == 0)
            say("9", "filesystem under build/tmp (f_type)", double(uint64_t(sfs.f_type)), "hex");
        if (FILE* m = std::fopen("/proc/mounts", "r")) {
            char line[512];
            while (std::fgets(line, sizeof line, m)) {
                if (std::strstr(line, " / ") || std::strstr(line, " /home ")) {
                    std::string s(line);
                    while (!s.empty() && s.back() == '\n') s.pop_back();
                    std::printf("9    mount: %s\n", s.c_str());
                }
            }
            std::fclose(m);
        }
        const int fd = open(path, O_RDWR | O_CREAT | O_TRUNC | O_DIRECT, 0644);
        if (fd < 0) {
            say("9", std::string("cannot open ") + path + " with O_DIRECT", 0, "-");
        } else {
            void* buf = nullptr;
            posix_memalign(&buf, 4096, 4ull << 20);
            // Incompressible: a repeated byte lets btrfs compress the whole file away, and the read
            // then comes out of RAM as fast as the CPU can decompress it (it reported 1278 MB/s on a
            // SATA 860 EVO, whose link tops out near 550 MB/s).
            {
                uint32_t* w = static_cast<uint32_t*>(buf);
                uint64_t s = 0x243f6a8885a308d3ull;
                for (size_t i = 0; i < (4ull << 20) / 4; ++i) {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    w[i] = uint32_t(s);
                }
            }
            for (size_t off = 0; off < total; off += (4ull << 20))
                if (pwrite(fd, buf, 4ull << 20, off_t(off)) != ssize_t(4ull << 20)) break;
            fsync(fd);
            // Drop this file's pages so the reads below come off the device, not out of RAM: without
            // this the numbers are the page cache's, which is exactly what they were the first run.
            posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
            {
                const auto t0 = Clock::now();
                size_t got = 0;
                for (size_t off = 0; off < total; off += (4ull << 20)) {
                    const ssize_t r = pread(fd, buf, 4ull << 20, off_t(off));
                    if (r <= 0) break;
                    got += size_t(r);
                }
                const double ms = ms_since(t0);
                say("9", "sequential read, 4 MiB blocks", double(got) / (ms / 1000.0) / 1e6, "MB/s");
            }
            for (int blk_kb : {4, 64}) {
                const size_t blk = size_t(blk_kb) << 10;
                const int n = quick ? 200 : 800;
                std::vector<double> us;
                uint64_t s = 999;
                posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
                for (int i = 0; i < n; ++i) {
                    s = s * 6364136223846793005ull + 1442695040888963407ull;
                    const off_t off = off_t((s % (total / blk)) * blk);
                    const auto t0 = Clock::now();
                    pread(fd, buf, blk, off);
                    us.push_back(std::chrono::duration<double, std::micro>(Clock::now() - t0).count());
                }
                say("9", "random read " + std::to_string(blk_kb) + " KiB, median", median(us), "us");
                say("9", "random read " + std::to_string(blk_kb) + " KiB, IOPS", 1e6 / median(us), "iops");
            }
            close(fd);
            unlink(path);
            std::free(buf);
        }
    }

    std::printf("\n# deferred: item 3 (per-format expert rows) lands in W4 with the verified kernels;\n");
    std::printf("#           item 7 (routing statistics) lands in W1 with the reference forward pass.\n");

    cudaFree(dev);
    cudaStreamDestroy(stream);
    cudaStreamDestroy(load_stream);
    return 0;
}
