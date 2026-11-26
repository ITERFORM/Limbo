// Copyright (c) 2026 ITERFORM. All rights reserved.
// src/cuda/gpu_q4e.cu — qwen4exp's (Qwen3.8-Flash-Next) decode step on the device. Owner: the q4e agent.
//
// One token at a time, in the order of src/step.cpp (forward_hc / ple_hash_block / hc_mix / layer_linear /
// layer_attention / moe): the residual is `hc` parallel streams; layer 1 opens with the PLE hashed n-gram
// block; every block reads the streams through a low-rank hyper-connection mixer, runs its GDN or
// sparse-attention mixer, writes back with 2 * sigmoid(inject), then does the same for the MoE. The head's own
// mixer replaces the output norm.
//
// Placement (docs/flash-next.md 6.5): the dense weights are in VRAM; the 23.4 GiB of routed experts stay in
// the file's own mapping and their SwiGLU FFNs run on the CPU lane (include/limbo/cpu_lane.hpp, planes = 3,
// two workers) beside the GPU — only the FFN input and the experts' outputs cross the bus. The PLE table
// (26.8 GiB) is read row by row on the host (16 rows per token) and uploaded as one 2560-float vector.
//
// The host's part runs on doorbells: a layer's split rings a mapped flag, the step thread (the feeder) admits, runs
// the lane with its workers and raises done; the device waits for it in a spin kernel. No stream sync inside a step.

#include <algorithm>
#include <array>
#include <future>
#include <chrono>
#include <cfloat>
#include <fstream>
#include <sched.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <unistd.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/resource.h>
#include <vector>

#include "gpu_impl.hpp"
#include "limbo/cuda/gemm.hpp"
#include "limbo/cuda/gemv.hpp"
#include "limbo/quants.hpp"
#include "limbo/types.hpp"
#include "dec_ops.hpp"
#include "q4e_qsa.hpp"
#include "ops.hpp"

namespace limbo::cuda {

namespace {

constexpr uint32_t F32 = static_cast<uint32_t>(Type::F32);

// LIMBO_Q4E_TIME=1: where a decode token's milliseconds go (the `STRATA_DECODE_TIMING` of
// docs/donors/strata.md 12.2). One line per token on stderr, and a cumulative summary at the end:
//
//   dense  the device time of the 48 dense halves (HC reads, GDN/QSA, the MoE's shared expert), between a
//          pair of events per layer. This is the GPU's own busy time, not the host's launch time.
//   sync   the host round trip the router costs: the D2H of `mixed`, the router GEMV, top-k, the D2H of the
//          ids, and the stream sync that waits for them — 48 of them per token.
//   lane   the CPU lane's exposed time: submit -> wait, per layer. The shared expert runs inside it.
//   ple    the host's 16 PLE table rows and their upload.
//   majf   major page faults of the whole process across the token. The expert rows come from the file's own
//          mapping, so a major fault is a 4 KiB read the page cache could not serve — an SSD read. This is
//          the number the residency policy of docs/flash-next.md 6.4 has to drive to zero.
//
// Off unless the variable is set; with it unset the step pays one predictable branch per token.
struct Q4eClock {
    bool on = false;
    bool init = false;
    cudaEvent_t in[64] = {}, out[64] = {};
    double dense = 0, sync = 0, sync_wait = 0, launch = 0, lane = 0, ple = 0;
    double moe = 0;   // out[l] -> in[l+1]: the MoE's own stream span, the lane's wait included
    // the device's own view: in[0] -> fin, and the part of it spent standing in the doorbell waits for the lane
    cudaEvent_t w0[128] = {}, w1[128] = {}, fin = nullptr;
    int nw = 0;
    bool has_fin = false;
    long maj = 0, minf = 0;
    uint64_t tokens = 0, last_hits = 0, last_lookups = 0;
    static Q4eClock& get();
    void begin_token();
    void end_token(int layers, double wall_ms, const char* what, int pos, uint64_t hits, uint64_t lookups,
                   uint64_t from_ram, uint64_t from_file);
};

Q4eClock& Q4eClock::get() {
    static Q4eClock c;
    if (!c.init) {
        c.init = true;
        c.on = std::getenv("LIMBO_Q4E_TIME") != nullptr;
        if (c.on) {
            for (int i = 0; i < 64; ++i) {
                (void)cudaEventCreate(&c.in[i]);
                (void)cudaEventCreate(&c.out[i]);
            }
            for (int i = 0; i < 128; ++i) {
                (void)cudaEventCreate(&c.w0[i]);
                (void)cudaEventCreate(&c.w1[i]);
            }
            (void)cudaEventCreate(&c.fin);
        }
    }
    return c;
}

void Q4eClock::begin_token() {
    dense = sync = sync_wait = launch = lane = ple = moe = 0;
    nw = 0;
    has_fin = false;
}

void Q4eClock::end_token(int layers, double wall_ms, const char* what, int pos, uint64_t hits, uint64_t lookups,
                         uint64_t from_ram, uint64_t from_file) {
    for (int l = 0; l < layers && l < 64; ++l) {
        float ms = 0;
        if (cudaEventElapsedTime(&ms, in[l], out[l]) == cudaSuccess) dense += double(ms);
        if (l + 1 < layers && cudaEventElapsedTime(&ms, out[l], in[l + 1]) == cudaSuccess) moe += double(ms);
    }
    double gpu = 0, stall = 0;
    float ms = 0;
    if (has_fin && cudaEventElapsedTime(&ms, in[0], fin) == cudaSuccess) gpu = double(ms);
    for (int i = 0; i < nw; ++i)
        if (cudaEventElapsedTime(&ms, w0[i], w1[i]) == cudaSuccess) stall += double(ms);
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    const long maj0 = ru.ru_majflt - maj, min0 = ru.ru_minflt - minf;
    maj = ru.ru_majflt;
    minf = ru.ru_minflt;
    ++tokens;
    std::fprintf(stderr,
                 "[q4e %-4s pos %5d] %7.1f ms = dense %6.1f + sync %5.1f + lane %6.1f + ple %4.1f + host %5.1f"
                 "  | moe %6.1f launch %5.1f syncwait %5.1f | majf %5ld minf %6ld | resident %5.1f%%"
                 " | lane from ram %llu file %llu | gpu %6.1f stall %6.1f\n",
                 what, pos, wall_ms, dense, sync, lane, ple, wall_ms - dense - sync - lane - ple, moe, launch,
                 sync_wait, maj0, min0,
                 lookups > last_lookups ? 100.0 * double(hits - last_hits) / double(lookups - last_lookups) : 0.0,
                 static_cast<unsigned long long>(from_ram), static_cast<unsigned long long>(from_file), gpu, stall);
    std::fflush(stderr);
    last_hits = hits;
    last_lookups = lookups;
}

// The residency split after the top-k and the doorbell: each selection's slot, read through the mapped table (never
// cached: the host edits it between layers), into the hits' launch ids and the host's mapped copy (ids, then slots);
// the FFN input into the lane's mapped tile; then bell = seq, after every write of the block is out.
__global__ void q4e_bell_kernel(const int32_t* __restrict__ ids, int n, const int32_t* table, int32_t* d_sel, int32_t* h_ids,
                                int32_t* h_slot, const float4* __restrict__ x, float4* xh, int nx4, uint32_t* bell,
                                const uint32_t* __restrict__ d_seq) {
    const int p = threadIdx.x;
    if (p < n) {
        const int32_t e = ids[p];
        const int32_t t = __ldcv(table + e);
        d_sel[p] = t;
        h_ids[p] = e;
        h_slot[p] = t;
    }
    for (int i = p; i < nx4; i += blockDim.x) xh[i] = x[i];
    __threadfence_system();
    __syncthreads();
    if (p == 0) {
        __threadfence_system();
        *reinterpret_cast<volatile uint32_t*>(bell) = *d_seq;
    }
}

// The prefetch's prediction into the host's mapped table, then its flag = seq (the bell's ordering).
__global__ void q4e_pred_kernel(const int32_t* __restrict__ top, int32_t* h_ids, uint32_t* flag, const uint32_t* __restrict__ d_seq) {
    if (threadIdx.x < 16) h_ids[threadIdx.x] = top[threadIdx.x];
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        *reinterpret_cast<volatile uint32_t*>(flag) = *d_seq;
    }
}

// The device side of a doorbell: one thread polls the mapped flag until it reads this call's seq (or the abort flag
// does, or ~10 s pass: err = seq, the host fails the step), then the block copies nw words of what the host wrote
// (zeros when it never came).
__global__ void q4e_wait_kernel(const uint32_t* flag, const uint32_t* abort_f, uint32_t* err_f, const uint32_t* __restrict__ d_seq,
                                const uint32_t* src, uint32_t* dst, int nw) {
    __shared__ int ok;
    if (threadIdx.x == 0) {
        const uint32_t want = *d_seq;
        const long long t0 = clock64();
        int r = 1;
        while (__ldcv(flag) != want) {
            if (__ldcv(abort_f) == want) {
                r = 0;
                break;
            }
            if (clock64() - t0 > 25000000000ll) {
                *reinterpret_cast<volatile uint32_t*>(err_f) = want;
                r = 0;
                break;
            }
            __nanosleep(100);
        }
        ok = r;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < nw; i += blockDim.x) dst[i] = ok ? __ldcv(src + i) : 0u;
}

// dn[idx[m]] += packed[m] for m < *count, packed read through (it may be the lane's mapped outputs).
__global__ void q4e_scatter_kernel(float* dn, const float* packed, const int32_t* __restrict__ idx,
                                   const int32_t* __restrict__ count, int n) {
    const int m = blockIdx.y;
    if (m >= *count) return;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) dn[static_cast<int64_t>(idx[m]) * n + j] += __ldcv(packed + static_cast<int64_t>(m) * n + j);
}

}  // namespace

// The PLE window and hash (step.cpp ple_hash_block): the token and its predecessors, a missing or EOS
// predecessor cutting the window, then per n-gram one row per head of the table, dequantized into `emb`.
bool pread_full(int fd, void* dst, uint64_t n, uint64_t off);

namespace {
inline uint64_t ple_hash(uint64_t r) {
    r ^= r >> 33;
    r *= 0xff51afd7ed558ccdull;
    r ^= r >> 33;
    return r;
}
}  // namespace

void Gpu::Impl::q4e_ple_rows(int32_t token, int pos, float* emb, bool ahead) {
    Q4e& qq = q;
    const int np = qq.np, ngram = qq.ngram;
    const int per = qq.n_heads / (ngram - 1);
    int64_t ctx[8];
    ctx[0] = token;
    bool cut = false;
    for (int s = 1; s < ngram; ++s) {
        const int64_t t = pos - s >= 0 ? int64_t(qq.tok_hist[static_cast<size_t>(pos - s)]) : -1;
        cut = cut || t < 0 || t == qq.eos;
        ctx[s] = cut ? qq.eos : t;
    }
    const int nh = std::min(64, (ngram - 1) * per);
    const bool cache = qq.ple_cache && qq.row_bytes <= sizeof(Q4e::PleSlot::row);
    // every row's address first: the cached ones are taken now, the others get a readahead each so the drive serves
    // them together instead of one synchronous fault after the other
    uint64_t rows[64];
    bool hit[64] = {};
    for (int n = 2; n <= ngram; ++n) {
        uint64_t mixed = static_cast<uint64_t>(ctx[0]) * qq.mult[0];
        for (int j = 1; j < n; ++j) mixed ^= static_cast<uint64_t>(ctx[j]) * qq.mult[static_cast<size_t>(j)];
        for (int g = 0; g < per; ++g) {
            const int h = (n - 2) * per + g;
            if (h >= 64) return;
            rows[h] = mixed % qq.vocab[static_cast<size_t>(h)] + qq.off[static_cast<size_t>(h)];
            if (cache) {
                const uint64_t want = rows[h] << 1, b = ple_hash(rows[h]);
                for (int p = 0; p < 4 && !hit[h]; ++p) {
                    Q4e::PleSlot& sl = qq.ple_cache[(b + uint64_t(p)) & qq.ple_mask];
                    const uint64_t t1 = sl.tag.load(std::memory_order_acquire);
                    if (t1 != want) continue;
                    alignas(16) uint8_t tmp[96];
                    std::memcpy(tmp, sl.row, size_t(qq.row_bytes));
                    std::atomic_thread_fence(std::memory_order_acquire);
                    if (sl.tag.load(std::memory_order_relaxed) != t1) break;
                    quants::dequant_row(qq.table_type, tmp, emb + static_cast<size_t>(h) * np, static_cast<uint64_t>(np));
                    hit[h] = true;
                }
                if (hit[h]) continue;
            }
            if (!ahead) continue;
            if (qq.table_fd >= 0) {
                (void)::posix_fadvise(qq.table_fd, off_t(qq.table_off + rows[h] * qq.row_bytes), off_t(qq.row_bytes),
                                      POSIX_FADV_WILLNEED);
                continue;
            }
            const uintptr_t a = reinterpret_cast<uintptr_t>(qq.table + rows[h] * qq.row_bytes);
            const uintptr_t lo = a & ~uintptr_t(4095), hi = (a + qq.row_bytes + 4095) & ~uintptr_t(4095);
            (void)::madvise(reinterpret_cast<void*>(lo), size_t(hi - lo), MADV_WILLNEED);
        }
    }
    int misses = 0;
    for (int h = 0; h < nh; ++h) {
        if (hit[h]) continue;
        ++misses;
        // read, not faulted: a fault maps the page into this process (and keeps it there), a pread leaves it in the
        // page cache only
        alignas(16) uint8_t rd[128];
        const uint8_t* src = qq.table + rows[h] * qq.row_bytes;
        if (qq.table_fd >= 0 && qq.row_bytes <= sizeof rd &&
            pread_full(qq.table_fd, rd, qq.row_bytes, qq.table_off + rows[h] * qq.row_bytes))
            src = rd;
        quants::dequant_row(qq.table_type, src, emb + static_cast<size_t>(h) * np, static_cast<uint64_t>(np));
        if (!cache) continue;
        // into an empty slot of the four, else over one of them (picked by the hash) unless its writer is mid-copy.
        // Tags: ~0 empty, row << 1 valid, odd otherwise = a writer holds it; a reader only ever accepts row << 1.
        const uint64_t want = rows[h] << 1, b = ple_hash(rows[h]);
        bool put = false;
        for (int p = 0; p < 4 && !put; ++p) {
            Q4e::PleSlot& sl = qq.ple_cache[(b + uint64_t(p)) & qq.ple_mask];
            uint64_t t = ~uint64_t(0);
            if (sl.tag.load(std::memory_order_relaxed) != t) continue;
            if (!sl.tag.compare_exchange_strong(t, want | 1u, std::memory_order_acquire)) continue;
            std::memcpy(sl.row, src, size_t(qq.row_bytes));
            sl.tag.store(want, std::memory_order_release);
            put = true;
        }
        if (!put) {
            Q4e::PleSlot& sl = qq.ple_cache[(b + (b >> 62)) & qq.ple_mask];
            uint64_t t = sl.tag.load(std::memory_order_relaxed);
            if (t != ~uint64_t(0) && !(t & 1u) && sl.tag.compare_exchange_strong(t, want | 1u, std::memory_order_acquire)) {
                std::memcpy(sl.row, src, size_t(qq.row_bytes));
                sl.tag.store(want, std::memory_order_release);
            }
        }
    }
    if (cache) {
        qq.ple_hits.fetch_add(uint64_t(nh - misses), std::memory_order_relaxed);
        qq.ple_miss.fetch_add(uint64_t(misses), std::memory_order_relaxed);
    }
}

// ---- the resident expert cache (docs/flash-next.md 6.3 / 6.4) -----------------------------------------
//
// The 23.4 GiB of routed experts cannot be resident on this card, but 91 to 93 % of the *lookups* can: the
// routing is that skewed (6.3 — the 16 most used experts of 256 take 44 % of the lookups, the 64 most used
// 80 %). So one slab per layer, its slots sized to that layer's blob (gate | up | down, contiguous), filled
// by decayed routing frequency. A resident expert is computed on the device by the same fp32 gemv kernels
// the dense weights use; a non-resident one still goes to the CPU lane. Nothing is approximated: the
// router's own ids decide which experts run, and both paths multiply the file's own dequantized weights into
// the same fp32 FFN input. What changes is where the bytes come from — and on a decode whose expert working
// set does not fit the page cache that is the difference between RAM and the SATA drive (measured on the
// `code` capture: 13k to 30k major page faults per token, 1.7 to 5.0 s of exposed lane time, all of it the
// drive; on a cache-resident working set the same lane is arithmetic-bound at 126 ms).
//
// `budget` 0 (or an allocation the shared card refuses) leaves every expert on the lane: that is the
// correctness floor, and the gate is run in both states.

// n bytes at file offset off into dst; false on a short read or an error
bool pread_full(int fd, void* dst, uint64_t n, uint64_t off) {
    uint8_t* d = static_cast<uint8_t*>(dst);
    while (n > 0) {
        const ssize_t r = ::pread(fd, d, size_t(std::min<uint64_t>(n, 1ull << 30)), off_t(off));
        if (r <= 0) return false;
        d += r;
        off += uint64_t(r);
        n -= uint64_t(r);
    }
    return true;
}

// A helper thread started by the (pinned) step thread inherits its core: give it every cpu back.
void q4e_unpin_this_thread() {
    cpu_set_t set;
    CPU_ZERO(&set);
    const int nc = int(std::thread::hardware_concurrency());
    for (int c = 0; c < nc && c < CPU_SETSIZE; ++c) CPU_SET(c, &set);
    (void)sched_setaffinity(0, sizeof(set), &set);
}

// MemAvailable, in bytes: the RAM the box can hand out without pushing anything into swap.
uint64_t avail_ram() {
    std::ifstream mi("/proc/meminfo");
    std::string key;
    while (mi >> key) {
        uint64_t v = 0;
        std::string unit;
        if (!(mi >> v >> unit)) break;
        if (key == "MemAvailable:") return v * 1024;
    }
    return 0;
}

// Ask the kernel for an expert's bytes in one request instead of letting the access path fault them one page at a
// time. A non-resident expert is 2.18 MiB, 557 pages: read by touch alone that is 557 synchronous 4 KiB faults on
// the calling thread, and the clock's own numbers say what they cost -- syncwait tracks majf at roughly 45 us a
// fault, so a token with 2,700 of them spends 120 ms of a 200 ms token inside cudaStreamSynchronize, waiting for a
// host thread that is faulting. MADV_WILLNEED makes it one readahead per expert, off the fault path; the copies
// and the lane read afterwards find the pages present or in flight. The range is page-rounded because madvise
// wants an aligned address, and the tensors inside a GGUF are not page-aligned.
void touch_ahead(const uint8_t* p, uint64_t n) {
    if (!p || n == 0) return;
    constexpr uintptr_t kPage = 4096;
    const uintptr_t lo = reinterpret_cast<uintptr_t>(p) & ~(kPage - 1);
    const uintptr_t hi = (reinterpret_cast<uintptr_t>(p) + n + kPage - 1) & ~(kPage - 1);
    (void)::madvise(reinterpret_cast<void*>(lo), static_cast<size_t>(hi - lo), MADV_WILLNEED);
}

bool Gpu::Impl::q4e_cache_open(uint64_t vram_budget, uint64_t ram_budget, std::string* err) {
    Q4e& qq = q;
    if (const char* g = std::getenv("LIMBO_Q4E_GROUPED")) qq.grouped_probe = std::atoi(g) != 0;
    const int nl = c.n_layer_main, ne = c.n_expert;
    qq.slab.assign(size_t(nl), Q4e::Slab());
    qq.ram_release.assign(size_t(nl), std::vector<int32_t>());
    qq.eheat.assign(size_t(nl) * size_t(ne), 0.0f);
    const size_t npair = size_t(kQ4ePairs);
    if (cudaHostAlloc(reinterpret_cast<void**>(&qq.slot_map), sizeof(int32_t) * size_t(nl) * size_t(ne), cudaHostAllocMapped) !=
            cudaSuccess ||
        cudaHostGetDevicePointer(reinterpret_cast<void**>(&qq.slot_map_d), qq.slot_map, 0) != cudaSuccess ||
        cudaHostAlloc(reinterpret_cast<void**>(&qq.split_h), sizeof(int32_t) * 2 * npair, cudaHostAllocMapped) != cudaSuccess ||
        cudaHostGetDevicePointer(reinterpret_cast<void**>(&qq.split_d), qq.split_h, 0) != cudaSuccess ||
        !(qq.d_sel = dalloc<int32_t>(npair)) ||
        cudaHostAlloc(reinterpret_cast<void**>(&qq.flag_h), sizeof(uint32_t) * Q4e::kFlags, cudaHostAllocMapped) != cudaSuccess ||
        cudaHostGetDevicePointer(reinterpret_cast<void**>(&qq.flag_d), qq.flag_h, 0) != cudaSuccess ||
        cudaHostAlloc(reinterpret_cast<void**>(&qq.ctl_h[0]), 2 * sizeof(Q4eCtl), cudaHostAllocMapped) != cudaSuccess ||
        cudaHostGetDevicePointer(reinterpret_cast<void**>(&qq.ctl_d[0]), qq.ctl_h[0], 0) != cudaSuccess ||
        !(qq.ctl_dev[0] = reinterpret_cast<Q4eCtl*>(dalloc<int32_t>(2 * sizeof(Q4eCtl) / sizeof(int32_t)))) ||
        !(qq.d_seq = reinterpret_cast<uint32_t*>(dalloc<int32_t>(1)))) {
        if (err) *err = "gpu: cannot allocate the residency table";
        return false;
    }
    std::fill(qq.slot_map, qq.slot_map + size_t(nl) * size_t(ne), -1);
    std::fill(qq.flag_h, qq.flag_h + Q4e::kFlags, 0u);
    qq.ctl_h[1] = qq.ctl_h[0] + 1;
    qq.ctl_d[1] = qq.ctl_d[0] + 1;
    qq.ctl_dev[1] = qq.ctl_dev[0] + 1;
    if (const char* d = std::getenv("LIMBO_Q4E_SPLIT_MIN")) qq.split_min = std::max(0, std::atoi(d));
    for (int b = 0; b < 2 && qq.n_heads > 0; ++b)
        if (cudaHostAlloc(reinterpret_cast<void**>(&qq.emb_h[b]), sizeof(float) * size_t(kQ4eRows) * size_t(qq.n_heads) * size_t(qq.np),
                          cudaHostAllocMapped) !=
                cudaSuccess ||
            cudaHostGetDevicePointer(reinterpret_cast<void**>(&qq.emb_hd[b]), qq.emb_h[b], 0) != cudaSuccess) {
            if (err) *err = "gpu: cannot allocate the PLE rows' mapped buffers";
            return false;
        }

    for (int l = 0; l < nl; ++l) {
        Q4e::Slab& s = qq.slab[size_t(l)];
        s.ge = L[size_t(l)].ge_bytes;
        s.de = L[size_t(l)].de_bytes;
        s.blob = 2 * s.ge + s.de;
        s.slot_of = qq.slot_map + size_t(l) * size_t(ne);
        s.ram_slot_of.assign(size_t(ne), -1);
    }
    uint64_t blob_sum = 0;
    for (const Q4e::Slab& s : qq.slab) blob_sum += s.blob;
    if (blob_sum == 0 || ne == 0) return true;

    // An equal byte budget per layer. Blobs run 1.44 to 2.54 MiB, so a layer of small experts holds more of
    // them for the same bytes — which is the point of sizing slots per layer rather than in experts.
    auto carve = [&](uint64_t want, bool vram) -> uint64_t {
        if (want == 0) return 0;
        const uint64_t per_layer = want / uint64_t(nl);
        uint64_t total = 0;
        for (int l = 0; l < nl; ++l) {
            const Q4e::Slab& s = qq.slab[size_t(l)];
            uint64_t n = s.blob ? per_layer / s.blob : 0;
            if (n > uint64_t(ne)) n = uint64_t(ne);
            if (vram) qq.slab[size_t(l)].n_slots = int(n);
            else qq.slab[size_t(l)].ram_slots = int(n);
            total += n * s.blob;
        }
        return total;
    };

    // Tier 1, VRAM. Take what the card gives: the desktop and another engine share it, and a smaller cache is
    // a slower decode rather than a broken one.
    if (vram_budget > 0) {
        uint8_t* base = nullptr;
        uint64_t got = 0;
        for (uint64_t want = vram_budget; want >= (64ull << 20); want /= 2) {
            carve(want, true);
            uint64_t need = 0;
            for (const Q4e::Slab& s : qq.slab) need += uint64_t(s.n_slots) * s.blob;
            if (need == 0) break;
            if (cudaMalloc(reinterpret_cast<void**>(&base), static_cast<size_t>(need) + 256) == cudaSuccess) {   // + the stage kernels' overread
                got = need;
                break;
            }
            (void)cudaGetLastError();
        }
        if (base) {
            owned.push_back(base);
            owned_bytes += got;
            qq.slab_bytes = got;
            uint64_t off = 0;
            for (int l = 0; l < nl; ++l) {
                Q4e::Slab& s = qq.slab[size_t(l)];
                s.base = base + off;
                s.expert_of.assign(size_t(s.n_slots), -1);
                off += uint64_t(s.n_slots) * s.blob;
            }
        } else {
            for (int l = 0; l < nl; ++l) qq.slab[size_t(l)].n_slots = 0;
        }
    }

    // The prompt pass's stage is lent from the slab (q4e_lend): the fewest tail slots a layer such that every layer's stage
    // (its non-resident experts and its own lent ones) packs, blob by blob, into the lent ranges: of its parity when the
    // stage is double-buffered (the next layer's lands while this one computes), of every layer when it is single.
    auto lend_fit = [&](bool dbl) {
        for (int k = 1; k <= 64; ++k) {
            bool ok = true;
            for (int l = 0; l < nl && ok; ++l) {
                const Q4e::Slab& a = qq.slab[size_t(l)];
                uint64_t cap = 0;
                for (int m = dbl ? (l & 1) : 0; m < nl; m += dbl ? 2 : 1) cap += uint64_t(k) * qq.slab[size_t(m)].blob / a.blob;
                ok = a.n_slots >= k && cap >= uint64_t(ne - a.n_slots + k);
            }
            if (ok) return k;
        }
        return 0;
    };
    const int k_dbl = qq.slab_bytes > 0 ? lend_fit(true) : 0, k_one = qq.slab_bytes > 0 ? lend_fit(false) : 0;
    auto ram_need = [&](int k) {
        uint64_t n = 0;
        for (const Q4e::Slab& s : qq.slab) n += uint64_t(std::min(ne, ne - s.n_slots + k)) * s.blob;
        return n;
    };

    // Tier 2, pinned RAM: by default exactly the complement of VRAM (every expert lives in one of the two, so
    // neither the decode lane nor the prompt pass's stage ever reads the drive), capped by what the box can pin
    // with a floor left for the page cache (the PLE rows, the desktop). LIMBO_Q4E_RAM_GB sets it, 0 disables it.
    // what stays free for the desktop and the page cache: the donor's Linux safety headroom (docs/donors/strata.md, PLE
    // table residency: 4 GiB, 8 on Windows); LIMBO_Q4E_RAM_HEADROOM_GB sets it
    uint64_t kRamFloor = 4ull << 30;
    if (const char* h = std::getenv("LIMBO_Q4E_RAM_HEADROOM_GB")) kRamFloor = uint64_t(std::atof(h) * double(1ull << 30));
    if (ram_budget == UINT64_MAX) {
        // a previous engine's pinned memory comes back over seconds after it exits: wait while it still grows
        for (int i = 0; i < 60; ++i) {
            const uint64_t a0 = avail_ram();
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            if (avail_ram() < a0 + (256ull << 20)) break;
        }
        const uint64_t av = avail_ram();
        ram_budget = av > kRamFloor ? av - kRamFloor : 0;
    }
    // double-buffered when RAM holds the complement and the lent experts of k_dbl slots a layer, else single (k_one):
    // a lent expert RAM cannot hold would have to come back from the drive
    qq.lend_single = k_dbl == 0 || ram_need(k_dbl) > ram_budget;
    qq.lend_k = qq.lend_single ? k_one : k_dbl;
    for (int l = 0; l < nl && qq.lend_k > 0; ++l) {
        const Q4e::Slab& a = qq.slab[size_t(l)];
        const std::pair<uint8_t*, uint64_t> r(a.base + uint64_t(a.n_slots - qq.lend_k) * a.blob, uint64_t(qq.lend_k) * a.blob);
        if (qq.lend_single) {
            qq.lend_rng[0].push_back(r);
            qq.lend_rng[1].push_back(r);
        } else {
            qq.lend_rng[l & 1].push_back(r);
        }
    }
    ram_budget = std::min(ram_budget, ram_need(qq.lend_k));
    if (ram_budget > 0) {
        uint64_t need = 0;
        for (const Q4e::Slab& s : qq.slab) need += uint64_t(std::min(ne, ne - s.n_slots + qq.lend_k)) * s.blob;
        if (ram_budget >= need) {
            for (Q4e::Slab& s : qq.slab) s.ram_slots = std::min(ne, ne - s.n_slots + qq.lend_k);   // + the lent slots' experts
        } else {
            carve(ram_budget, false);
        }
        uint64_t off = 0;
        for (int l = 0; l < nl; ++l) {
            Q4e::Slab& s = qq.slab[size_t(l)];
            const uint64_t want = uint64_t(s.ram_slots + 1) * s.blob;   // + the spare
            if (s.ram_slots == 0) continue;
            void* p = nullptr;
            // paced: one layer at a time against the floor, so the kernel can reclaim clean cache as it goes
            if (avail_ram() < kRamFloor * 3 / 4 + want || cudaHostAlloc(&p, static_cast<size_t>(want), cudaHostAllocDefault) != cudaSuccess) {
                (void)cudaGetLastError();
                s.ram_slots = 0;
                continue;
            }
            s.ram = static_cast<uint8_t*>(p);
            const int nt = s.ram_slots + 1;
            s.ram_ptr.resize(size_t(nt));
            for (int t = 0; t < nt; ++t) s.ram_ptr[size_t(t)] = s.ram + uint64_t(t) * s.blob;
            s.ram_spare = s.ram_slots;
            s.ram_expert_of.assign(size_t(nt), -1);
            s.ram_ready.assign(size_t(nt), 0);
            s.ram_pending.assign(size_t(nt), 0);
            off += want;
        }
        qq.ram_bytes = off;
        if (off > 0) {
            qq.fill.assign(size_t(Q4e::kFillRing), Q4e::FillReq());
            qq.filler = std::thread(&Gpu::Impl::q4e_fill_loop, this);
        }
    }
    for (Q4e::Slab& s : qq.slab) s.stamp.assign(size_t(s.n_slots), 0);
    if (cudaStreamCreateWithFlags(&qq.cstream, cudaStreamNonBlocking) != cudaSuccess) {
        if (err) *err = "gpu: cannot create the expert copy stream";
        return false;
    }
    {
        if (const char* d = std::getenv("LIMBO_Q4E_DMA_MIN")) qq.dma_min = std::atoi(d);
        if (const char* d = std::getenv("LIMBO_Q4E_PCIE_SHARE")) qq.pcie_share = float(std::atof(d));
        if (const char* d = std::getenv("LIMBO_Q4E_TAIL_SKIP")) qq.tail_skip = std::max(0, std::atoi(d));
        for (const Q4e::Slab& s : qq.slab) qq.dma_blob = std::max(qq.dma_blob, (s.blob + 255) & ~uint64_t(255));
        if (const char* d = std::getenv("LIMBO_Q4E_PREFETCH")) qq.pf_n = std::clamp(std::atoi(d), 0, int(Q4e::kPf));
        if ((qq.dma_min > 0 || qq.pf_n > 0) && qq.dma_blob > 0) {
            const int32_t iota[Q4e::kDmaShare] = {0, 1, 2, 3};
            if (cudaStreamCreateWithFlags(&qq.dma_stream, cudaStreamNonBlocking) != cudaSuccess ||
                cudaEventCreateWithFlags(&qq.dma_ev, cudaEventDisableTiming) != cudaSuccess ||
                !(qq.dma_buf = dalloc<uint8_t>(size_t(qq.dma_blob) * Q4e::kDmaShare)) ||
                !(qq.dn_dma = dalloc<float>(size_t(Q4e::kDma) * size_t(c.n_embd))) || !(qq.dev_iota = dalloc<int32_t>(Q4e::kDmaShare)) ||
                !(qq.dma_g = dalloc<float>(size_t(Q4e::kDma) * size_t(c.n_ff_exp))) ||
                !(qq.dma_u = dalloc<float>(size_t(Q4e::kDma) * size_t(c.n_ff_exp))) ||
                !(qq.dma_h = dalloc<float>(size_t(Q4e::kDma) * size_t(c.n_ff_exp))) ||
                cudaMemcpy(qq.dev_iota, iota, sizeof iota, cudaMemcpyHostToDevice) != cudaSuccess) {
                (void)cudaGetLastError();
                qq.dma_min = 0;
                qq.pf_n = 0;
            }
            if (qq.pf_n > 0 &&
                (cudaStreamCreateWithFlags(&qq.pf_stream, cudaStreamNonBlocking) != cudaSuccess ||
                 cudaEventCreateWithFlags(&qq.pf_ev[0], cudaEventDisableTiming) != cudaSuccess ||
                 cudaEventCreateWithFlags(&qq.pf_ev[1], cudaEventDisableTiming) != cudaSuccess ||
                 !(qq.pf_buf = dalloc<uint8_t>(size_t(qq.dma_blob) * 2 * Q4e::kPf)) ||
                 cudaHostAlloc(reinterpret_cast<void**>(&qq.pred_h), sizeof(int32_t) * 16 * size_t(nl), cudaHostAllocMapped) != cudaSuccess ||
                 cudaHostGetDevicePointer(reinterpret_cast<void**>(&qq.pred_d), qq.pred_h, 0) != cudaSuccess ||
                 cudaHostAlloc(reinterpret_cast<void**>(&qq.pf_slot_h), sizeof(int32_t) * Q4e::kPf, cudaHostAllocMapped) != cudaSuccess ||
                 cudaHostGetDevicePointer(reinterpret_cast<void**>(&qq.pf_slot_d), qq.pf_slot_h, 0) != cudaSuccess ||
                 !(qq.pred_top = dalloc<int32_t>(16)) || !(qq.pred_rlog = dalloc<float>(size_t(c.n_expert))) ||
                 !(qq.pred_w = dalloc<float>(16)))) {
                (void)cudaGetLastError();
                qq.pf_n = 0;
            }
        } else {
            qq.dma_min = 0;
        }
    }
    qq.pr_ev.assign(size_t(nl), nullptr);
    for (cudaEvent_t& ev : qq.pr_ev)
        if (cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) != cudaSuccess) {
            if (err) *err = "gpu: cannot create the promotion events";
            return false;
        }
    return q4e_cache_warm(err);
}

// Before a prompt pass: every expert in the lent tail slots goes down to a free RAM slot and the slots are free, the
// stage's room. Idempotent (decode promotions may have refilled them); the rebalance after the pass refills them.
bool Gpu::Impl::q4e_lend() {
    Q4e& qq = q;
    if (qq.lend_k <= 0 || !qq.cstream) return false;
    q4e_pr_drain();
    std::vector<std::pair<int, int>> landed;
    int dropped = 0;
    for (int l = 0; l < int(qq.slab.size()); ++l) {
        Q4e::Slab& s = qq.slab[size_t(l)];
        for (int t = s.n_slots - qq.lend_k; t < s.n_slots; ++t) {
            const int32_t v = s.expert_of[size_t(t)];
            if (v < 0) continue;
            int rt = -1;
            if (s.ram_used < s.ram_slots) {
                rt = s.ram_used++;
            } else {
                for (int u = 0; u < s.ram_slots && rt < 0; ++u)
                    if (s.ram_expert_of[size_t(u)] < 0 && !s.ram_pending[size_t(u)]) rt = u;
            }
            s.slot_of[size_t(v)] = -1;
            s.expert_of[size_t(t)] = -1;
            if (rt < 0) {   // no RAM room: it goes back to the drive (the filler fetches it when routed)
                ++dropped;
                continue;
            }
            if (cudaMemcpyAsync(s.ram_ptr[size_t(rt)], s.base + uint64_t(t) * s.blob, s.blob, cudaMemcpyDeviceToHost, qq.cstream) != cudaSuccess)
                return false;
            s.ram_expert_of[size_t(rt)] = v;
            s.ram_slot_of[size_t(v)] = rt;
            s.ram_ready[size_t(rt)] = 0;
            landed.emplace_back(l, rt);
        }
    }
    if (cudaStreamSynchronize(qq.cstream) != cudaSuccess) return false;
    for (const auto& [l, rt] : landed) qq.slab[size_t(l)].ram_ready[size_t(rt)] = 1;
    if (std::getenv("LIMBO_Q4E_PF_LOG"))
        std::fprintf(stderr, "q4e lend: %zu experts to RAM, %d to the drive, %d slots a layer (%s stage)\n", landed.size(), dropped, qq.lend_k,
                     qq.lend_single ? "single" : "double");
    return true;
}

// Both tiers filled at open, straight from the mapping (VRAM through the layer's pinned spare): the first
// experts of each layer to VRAM, the next ones to RAM. Which ones does not matter for the result, only for the
// first tokens' hit rate; routing heat sorts them from there. Eight threads, a layer at a time each.
void Gpu::Impl::Q4e::profile_save() const {
    if (profile_path.empty() || lt.empty()) return;
    const std::string tmp = profile_path + ".tmp";
    if (std::FILE* f = std::fopen(tmp.c_str(), "wb")) {
        const uint32_t hdr[3] = {0x4c513445u, uint32_t(slab.size()), uint32_t(lt.size() / std::max<size_t>(1, slab.size()))};
        const bool ok = std::fwrite(hdr, sizeof hdr, 1, f) == 1 && std::fwrite(lt.data(), sizeof(float), lt.size(), f) == lt.size();
        std::fclose(f);
        if (ok) (void)std::rename(tmp.c_str(), profile_path.c_str());
    }
}

bool Gpu::Impl::q4e_cache_warm(std::string* err) {
    Q4e& qq = q;
    const int nl = c.n_layer_main, ne = c.n_expert;
    // the routing profile of earlier runs, when there is one: each layer's experts by count, VRAM takes the head
    qq.lt.assign(size_t(nl) * size_t(ne), 0.0f);
    bool have_profile = false;
    if (!qq.profile_path.empty())
        if (std::FILE* f = std::fopen(qq.profile_path.c_str(), "rb")) {
            uint32_t hdr[3] = {};
            have_profile = std::fread(hdr, sizeof hdr, 1, f) == 1 && hdr[0] == 0x4c513445u && int(hdr[1]) == nl && int(hdr[2]) == ne &&
                           std::fread(qq.lt.data(), sizeof(float), qq.lt.size(), f) == qq.lt.size();
            std::fclose(f);
            if (!have_profile) std::fill(qq.lt.begin(), qq.lt.end(), 0.0f);
        }
    std::vector<std::vector<int32_t>> xorder(static_cast<size_t>(nl));
    for (int l = 0; l < nl; ++l) {
        std::vector<int32_t>& o = xorder[size_t(l)];
        o.resize(size_t(ne));
        for (int e = 0; e < ne; ++e) o[size_t(e)] = e;
        if (!have_profile) continue;
        const float* cnt = qq.lt.data() + size_t(l) * size_t(ne);
        std::stable_sort(o.begin(), o.end(), [&](int32_t a, int32_t b) { return cnt[a] > cnt[b]; });
        // the prior: the profile's shares, worth 64 tokens of routing
        double tot = 0;
        for (int e = 0; e < ne; ++e) tot += cnt[e];
        if (tot > 0)
            for (int e = 0; e < ne; ++e) qq.eheat[size_t(l) * size_t(ne) + size_t(e)] = float(64.0 * c.n_used * cnt[e] / tot);
    }
    std::atomic<int> next{0};
    std::atomic<bool> bad{false};
    const auto t0 = std::chrono::steady_clock::now();
    auto work = [&]() {
        cudaStream_t cs = nullptr;
        if (cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking) != cudaSuccess) {
            bad = true;
            return;
        }
        uint8_t* bounce = nullptr;
        uint64_t bounce_n = 0;
        cudaEvent_t hev[2] = {nullptr, nullptr};
        if (cudaEventCreateWithFlags(&hev[0], cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&hev[1], cudaEventDisableTiming) != cudaSuccess) {
            bad = true;
            return;
        }
        for (int l; (l = next.fetch_add(1)) < nl && !bad;) {
            const Layer& y = L[size_t(l)];
            Q4e::Slab& s = qq.slab[size_t(l)];
            // by pread from the files (the mapping is not touched, so the 23 GiB do not join this process's working
            // set); the layer's pages leave the page cache once it is copied
            int fg = -1, fu = -1, fdn = -1;
            uint64_t og = 0, ou = 0, od = 0;
            const bool by_file = q.file_of(y.geh.p, &fg, &og) && q.file_of(y.ueh.p, &fu, &ou) && q.file_of(y.deh.p, &fdn, &od) &&
                                 fg >= 0 && fu >= 0 && fdn >= 0;
            auto read = [&](int ex, uint8_t* dst) {
                if (by_file) {
                    const uint64_t e64 = uint64_t(ex);
                    if (!pread_full(fg, dst, y.ge_bytes, og + e64 * y.ge_bytes) ||
                        !pread_full(fu, dst + s.ge, y.ge_bytes, ou + e64 * y.ge_bytes) ||
                        !pread_full(fdn, dst + 2 * s.ge, y.de_bytes, od + e64 * y.de_bytes))
                        bad = true;
                    return;
                }
                std::memcpy(dst, y.geh.p + uint64_t(ex) * y.ge_bytes, size_t(y.ge_bytes));
                std::memcpy(dst + s.ge, y.ueh.p + uint64_t(ex) * y.ge_bytes, size_t(y.ge_bytes));
                std::memcpy(dst + 2 * s.ge, y.deh.p + uint64_t(ex) * y.de_bytes, size_t(y.de_bytes));
            };
            int ex = 0;
            const std::vector<int32_t>& ord = xorder[size_t(l)];
            if (s.n_slots > 0) {
                if (bounce_n < s.blob) {
                    if (bounce) cudaFreeHost(bounce);
                    bounce_n = s.blob;
                    if (cudaHostAlloc(reinterpret_cast<void**>(&bounce), size_t(2 * bounce_n), cudaHostAllocDefault) != cudaSuccess) {
                        bad = true;
                        break;
                    }
                }
                for (int t = 0; t < s.n_slots && ex < ne; ++t, ++ex) {
                    const int32_t xe = ord[size_t(ex)];
                    uint8_t* b = bounce + uint64_t(t & 1) * bounce_n;
                    if (t >= 2 && cudaEventSynchronize(hev[t & 1]) != cudaSuccess) bad = true;   // the half's last copy landed
                    read(xe, b);
                    if (cudaMemcpyAsync(s.base + uint64_t(t) * s.blob, b, s.blob, cudaMemcpyHostToDevice, cs) != cudaSuccess ||
                        cudaEventRecord(hev[t & 1], cs) != cudaSuccess)
                        bad = true;
                    s.slot_of[size_t(xe)] = t;
                    s.expert_of[size_t(t)] = xe;
                }
                s.used = s.n_slots;
                if (cudaStreamSynchronize(cs) != cudaSuccess) bad = true;
            }
            for (int t = 0; t < s.ram_slots && ex < ne; ++t, ++ex) {
                const int32_t xe = ord[size_t(ex)];
                read(xe, s.ram_ptr[size_t(t)]);
                s.ram_slot_of[size_t(xe)] = t;
                s.ram_expert_of[size_t(t)] = xe;
                s.ram_ready[size_t(t)] = 1;
                s.ram_used = t + 1;
            }
            if (by_file) {
                (void)::posix_fadvise(fg, off_t(og), off_t(uint64_t(ne) * y.ge_bytes), POSIX_FADV_DONTNEED);
                (void)::posix_fadvise(fu, off_t(ou), off_t(uint64_t(ne) * y.ge_bytes), POSIX_FADV_DONTNEED);
                (void)::posix_fadvise(fdn, off_t(od), off_t(uint64_t(ne) * y.de_bytes), POSIX_FADV_DONTNEED);
            }
        }
        if (bounce) cudaFreeHost(bounce);
        cudaEventDestroy(hev[0]);
        cudaEventDestroy(hev[1]);
        cudaStreamDestroy(cs);
    };
    std::vector<std::thread> th;
    for (int i = 0; i < 8; ++i) th.emplace_back(work);
    for (std::thread& t : th) t.join();
    if (bad) {
        if (err) *err = "gpu: warming the expert cache failed";
        return false;
    }
    std::atomic_thread_fence(std::memory_order_release);
    uint64_t in_vram = 0, in_ram = 0;
    for (const Q4e::Slab& s : qq.slab) {
        in_vram += uint64_t(s.used);
        in_ram += uint64_t(s.ram_used);
    }
    std::fprintf(stderr, "q4e experts: %llu in VRAM (%.2f GiB), %llu in pinned RAM (%.2f GiB), %llu on the drive; warm %.1f s\n",
                 (unsigned long long)in_vram, double(qq.slab_bytes) / double(1ull << 30), (unsigned long long)in_ram,
                 double(qq.ram_bytes) / double(1ull << 30),
                 (unsigned long long)(uint64_t(nl) * uint64_t(ne) - in_vram - in_ram),
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return true;
}

// A promotion's copies have landed: e computes from VRAM slot vs, its victim v lives in the spare's buffer, and
// the RAM slot e came from is the new spare.
void Gpu::Impl::q4e_pr_finish(int l) {
    Q4e::Slab& s = q.slab[size_t(l)];
    if (!s.pr_on) return;
    const int32_t e = s.pr_e, v = s.pr_v, vs = s.pr_vs, rt = s.pr_rt;
    s.slot_of[size_t(e)] = vs;
    s.expert_of[size_t(vs)] = e;
    s.ram_slot_of[size_t(e)] = -1;
    s.ram_pending[size_t(rt)] = 0;
    if (v >= 0) {
        const int sp = s.ram_spare;
        std::swap(s.ram_ptr[size_t(sp)], s.ram_ptr[size_t(rt)]);   // rt's index now names the buffer holding v
        s.ram_expert_of[size_t(rt)] = v;
        s.ram_slot_of[size_t(v)] = rt;
        s.ram_ready[size_t(rt)] = 1;
    } else {
        s.ram_expert_of[size_t(rt)] = -1;
        s.ram_ready[size_t(rt)] = 0;
    }
    s.pr_on = false;
}

// One VRAM slot vs takes expert e out of the RAM tier; vs's expert (if any) goes down into the layer's spare buffer,
// which becomes e's old RAM slot. On the copy stream; sync_book: the tables change now (the caller syncs the
// stream before anything reads them), else at q4e_pr_finish.
bool Gpu::Impl::q4e_swap(int l, int32_t e, int vs, bool sync_book) {
    Q4e::Slab& s = q.slab[size_t(l)];
    const int32_t rt = s.ram_slot_of[size_t(e)], v = s.expert_of[size_t(vs)];
    if (rt < 0 || !s.ram_ready[size_t(rt)] || v < -1) return false;
    uint8_t* dst = s.base + uint64_t(vs) * s.blob;
    if (v >= 0 && cudaMemcpyAsync(s.ram_ptr[size_t(s.ram_spare)], dst, s.blob, cudaMemcpyDeviceToHost, q.cstream) != cudaSuccess)
        return false;
    if (cudaMemcpyAsync(dst, s.ram_ptr[size_t(rt)], s.blob, cudaMemcpyHostToDevice, q.cstream) != cudaSuccess) return false;
    if (v >= 0) s.slot_of[size_t(v)] = -1;
    s.pr_on = true;
    s.pr_e = e;
    s.pr_v = v;
    s.pr_vs = vs;
    s.pr_rt = rt;
    s.ram_pending[size_t(rt)] = 1;
    if (sync_book) q4e_pr_finish(l);
    else s.expert_of[size_t(vs)] = -2;
    return true;
}

// After a prompt pass: its routing counts into the heat, then each layer's VRAM re-sorted -- the hottest experts the
// RAM tier holds swap with the coldest residents while they are hotter. The decode that follows starts from the
// prompt's own routing instead of the open-time order (which left the slab at ~42 % of lookups).
void Gpu::Impl::q4e_rebalance() {
    Q4e& qq = q;
    if (!qpf.counted || !qpf.cnt_d || !qq.cstream) return;
    qpf.counted = false;
    const int nl = c.n_layer_main, ne = c.n_expert;
    if (cudaStreamSynchronize(stream) != cudaSuccess) return;
    q4e_pr_drain();
    if (cudaMemcpy(qpf.cnt_h, qpf.cnt_d, sizeof(int32_t) * size_t(nl) * size_t(ne), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemset(qpf.cnt_d, 0, sizeof(int32_t) * size_t(nl) * size_t(ne)) != cudaSuccess)
        return;
    const auto t0 = std::chrono::steady_clock::now();
    int swaps = 0;
    for (int l = 0; l < nl; ++l) {
        float* heat = qq.eheat.data() + size_t(l) * size_t(ne);
        // the prompt is a prior worth 64 decode tokens of routing: a reply routes unlike its prompt (code in, prose
        // out), and raw prompt counts (hundreds an expert) would hold VRAM against the reply's own experts
        int64_t tot = 0;
        for (int e = 0; e < ne; ++e) tot += qpf.cnt_h[size_t(l) * size_t(ne) + size_t(e)];
        const float w = tot > 0 ? 64.0f * float(c.n_used) / float(tot) : 0.0f;
        for (int e = 0; e < ne; ++e) heat[e] += w * float(qpf.cnt_h[size_t(l) * size_t(ne) + size_t(e)]);
        if (!qq.lt.empty())
            for (int e = 0; e < ne; ++e) qq.lt[size_t(l) * size_t(ne) + size_t(e)] += float(qpf.cnt_h[size_t(l) * size_t(ne) + size_t(e)]);
        Q4e::Slab& s = qq.slab[size_t(l)];
        if (!s.base || !s.ram || s.n_slots == 0) continue;
        std::vector<int32_t> win, los;
        for (int e = 0; e < ne; ++e) {
            const int32_t rt = s.ram_slot_of[size_t(e)];
            if (rt >= 0 && s.ram_ready[size_t(rt)] && !s.ram_pending[size_t(rt)]) win.push_back(e);
        }
        std::sort(win.begin(), win.end(), [&](int32_t a, int32_t b) { return heat[a] > heat[b]; });
        for (int t = 0; t < s.n_slots; ++t)
            if (s.expert_of[size_t(t)] >= -1) los.push_back(t);
        auto hv = [&](int t) { const int32_t x = s.expert_of[size_t(t)]; return x < 0 ? -1.0f : heat[x]; };
        std::sort(los.begin(), los.end(), [&](int a, int b) { return hv(a) < hv(b); });
        for (size_t i = 0; i < win.size() && i < los.size(); ++i) {
            if (!(heat[win[i]] > hv(los[i]))) break;
            if (!q4e_swap(l, win[i], los[i], true)) break;
            ++swaps;
        }
    }
    (void)cudaStreamSynchronize(qq.cstream);
    qq.profile_save();
    if (std::getenv("LIMBO_Q4E_PF_LOG"))
        std::fprintf(stderr, "q4e rebalance: %d swaps in %.0f ms\n", swaps,
                     std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
}

void Gpu::Impl::q4e_pr_drain() {
    Q4e& qq = q;
    if (!qq.cstream) return;
    (void)cudaStreamSynchronize(qq.cstream);
    for (int l = 0; l < int(qq.slab.size()); ++l) q4e_pr_finish(l);
}

void Gpu::Impl::q4e_cache_close() {
    Q4e& qq = q;
    qq.profile_save();
    qq.fill_stop.store(true, std::memory_order_release);
    if (qq.filler.joinable()) qq.filler.join();
    if (qq.cstream) (void)cudaStreamSynchronize(qq.cstream);
    for (cudaEvent_t ev : qq.pr_ev)
        if (ev) (void)cudaEventDestroy(ev);
    qq.pr_ev.clear();
    if (qq.cstream) (void)cudaStreamDestroy(qq.cstream);
    qq.cstream = nullptr;
    for (Q4e::Slab& s : qq.slab)
        if (s.ram) {
            (void)cudaFreeHost(s.ram);
            s.ram = nullptr;
        }
    for (Q4e::Shard& sh : qq.shards)
        if (sh.fd >= 0) {
            ::close(sh.fd);
            sh.fd = -1;
        }
}

// Tier 3 fills tier 2 off the step's critical path. One thread, one request ring: the step never waits for the
// drive, it only ever reads a slot this thread has already published.
void Gpu::Impl::q4e_fill_loop() {
    Q4e& qq = q;
    // Unpinned at a lower priority: the scheduler places it beside the lane instead of a fixed cpu it may then wait on.
    q4e_unpin_this_thread();
    (void)nice(10);
    while (!qq.fill_stop.load(std::memory_order_acquire)) {
        const uint32_t h = qq.fill_head.load(std::memory_order_relaxed);
        if (h == qq.fill_tail.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::microseconds(250));
            continue;
        }
        const Q4e::FillReq r = qq.fill[h % Q4e::kFillRing];
        qq.fill_head.store(h + 1, std::memory_order_release);
        const Layer& y = L[size_t(r.l)];
        Q4e::Slab& s = qq.slab[size_t(r.l)];
        uint8_t* dst = s.ram_ptr[size_t(r.slot)];
        const uint64_t ex = uint64_t(static_cast<uint32_t>(r.e));
        int fg = -1, fu = -1, fdn = -1;
        uint64_t og = 0, ou = 0, od = 0;
        if (qq.file_of(y.geh.p, &fg, &og) && qq.file_of(y.ueh.p, &fu, &ou) && qq.file_of(y.deh.p, &fdn, &od) && fg >= 0 &&
            fu >= 0 && fdn >= 0 && pread_full(fg, dst, y.ge_bytes, og + ex * y.ge_bytes) &&
            pread_full(fu, dst + s.ge, y.ge_bytes, ou + ex * y.ge_bytes) &&
            pread_full(fdn, dst + 2 * s.ge, y.de_bytes, od + ex * y.de_bytes)) {
        } else {
            std::memcpy(dst, y.geh.p + ex * y.ge_bytes, static_cast<size_t>(y.ge_bytes));
            std::memcpy(dst + s.ge, y.ueh.p + ex * y.ge_bytes, static_cast<size_t>(y.ge_bytes));
            std::memcpy(dst + 2 * s.ge, y.deh.p + ex * y.de_bytes, static_cast<size_t>(y.de_bytes));
        }
        std::atomic_thread_fence(std::memory_order_release);
        // Published after the bytes. `ram_slot_of` was set by the step when it queued this fill, so the step's
        // acquire on this flag is what makes the copies above visible to it.
        s.ram_ready[size_t(r.slot)] = 1;
        // blob bytes at fill_mbps: blob / mbps microseconds. The lane waits for these fills, so the default is
        // unpaced — the ceiling exists for a box where the drive must be shared with something else.
        if (qq.fill_mbps)
            std::this_thread::sleep_for(std::chrono::microseconds(int64_t(s.blob) / int64_t(qq.fill_mbps)));
    }
}

void Gpu::Impl::q4e_cache_decay() {
    for (float& h : q.eheat) h *= q.decay;
    q.pr_budget = 48;   // promotions a token (one in flight a layer): each is ~2 blobs over the bus, off the step's stream
    ++q.epoch;
}

// Layer l's split of n selections (router ids on the device) and its doorbell, with nx floats of FFN input to xh_d.
// pair0: the first (row, used) pair of the selections, where their split goes (a window group's own range).
bool Gpu::Impl::q4e_bell(int idx, int l, const int32_t* ids, int n, const float* x, float* xh_d, int nx, int pair0) {
    static_assert(kQ4ePairs == int(sizeof(Q4eCtl::miss_idx) / sizeof(int32_t)), "ctl holds a window's pairs");
    Q4e& qq = q;
    if (n <= 0 || pair0 + n > kQ4ePairs || nx % 4 != 0) return false;
    q4e_bell_kernel<<<1, kQ4ePairs, 0, stream>>>(ids, n, qq.slot_map_d + size_t(l) * size_t(c.n_expert), qq.d_sel + pair0,
                                                 qq.split_d + pair0, qq.split_d + kQ4ePairs + pair0, reinterpret_cast<const float4*>(x),
                                                 reinterpret_cast<float4*>(xh_d), nx / 4, qq.flag_d + Q4e::kBell + idx, qq.d_seq);
    return cudaGetLastError() == cudaSuccess;
}

// The device waits for flag (one of flag_h) to read this call's seq, then copies nw words from src_d to dst.
bool Gpu::Impl::q4e_wait(int flag, const void* src_d, void* dst, int nw) {
    Q4e& qq = q;
    q4e_wait_kernel<<<1, 256, 0, stream>>>(qq.flag_d + flag, qq.flag_d + Q4e::kAbort, qq.flag_d + Q4e::kErr, qq.d_seq,
                                           static_cast<const uint32_t*>(src_d),
                                           static_cast<uint32_t*>(dst), nw);
    return cudaGetLastError() == cudaSuccess;
}

// The host side: spin on layer l's bell (the feeder owns its core); false after ~10 s without one.
bool Gpu::Impl::q4e_bell_wait(int idx) {
    Q4e& qq = q;
    const uint32_t want = qq.seq;
    uint32_t* f = qq.flag_h + Q4e::kBell + idx;
    for (uint64_t i = 0; __atomic_load_n(f, __ATOMIC_ACQUIRE) != want; ++i) {
        __builtin_ia32_pause();
        if ((i & 0xfffff) != 0xfffff) continue;
        const cudaError_t r = cudaStreamQuery(stream);
        if (r == cudaErrorNotReady) {
            if (cudaPeekAtLastError() == cudaErrorNotReady) (void)cudaGetLastError();
        } else if (__atomic_load_n(f, __ATOMIC_ACQUIRE) != want) {
            std::fprintf(stderr, "q4e: bell %d never rang (seq %u)\n", idx, want);
            return false;
        }
    }
    return true;
}

void Gpu::Impl::q4e_done(int idx) { __atomic_store_n(q.flag_h + Q4e::kDone + idx, q.seq, __ATOMIC_RELEASE); }

// Every wait of the running call lets go (its layers compute from zeroed control data; the caller fails the step).
void Gpu::Impl::q4e_abort() { __atomic_store_n(q.flag_h + Q4e::kAbort, q.seq, __ATOMIC_RELEASE); }

// After the last layer is served: the call's device work done, and no wait of it gave up.
bool Gpu::Impl::q4e_end() {
    if (cudaStreamSynchronize(stream) != cudaSuccess) return false;
    if (__atomic_load_n(q.flag_h + Q4e::kErr, __ATOMIC_ACQUIRE) != q.seq) return true;
    std::fprintf(stderr, "q4e: a doorbell wait timed out (seq %u): the step's results are void\n", q.seq);
    return false;
}

namespace {
// Lazy module loading puts a context-wide barrier behind a kernel's first launch: launched while a doorbell wait spins, it
// holds every stream until the wait gives up. Every module loads at start instead (~16 MB of VRAM, measured).
[[maybe_unused]] const int q4e_eager = ::setenv("CUDA_MODULE_LOADING", "EAGER", 1);
}  // namespace

// After the split has landed: every slot it hit is this epoch's, so no victim.
void Gpu::Impl::q4e_split_book(int l, int n, int pair0) {
    Q4e::Slab& s = q.slab[size_t(l)];
    const int32_t* slot = q.split_h + kQ4ePairs + pair0;
    for (int p = 0; p < n; ++p)
        if (slot[p] >= 0) s.stamp[size_t(slot[p])] = q.epoch;
}

// At the end of a layer's serve, once the lane and the PCIe share are done with the RAM buffers: a promotion whose copies
// have landed is booked (its expert's RAM buffer becomes the spare, and the next split sees the slot).
void Gpu::Impl::q4e_pr_poll(int l) {
    if (!q.slab[size_t(l)].pr_on) return;
    const cudaError_t r = cudaEventQuery(q.pr_ev[size_t(l)]);
    if (r == cudaSuccess) q4e_pr_finish(l);
    else if (r == cudaErrorNotReady && cudaPeekAtLastError() == cudaErrorNotReady) (void)cudaGetLastError();
}

// One layer's admission. Every selected expert's heat is raised by this lookup; a hit is a VRAM-resident expert.
// A miss is read from the RAM tier when it holds the expert, else from the file's mapping (and queued to the
// filler). With a RAM tier the two tiers hold each other's complement, and a promotion SWAPS the hottest RAM-held
// miss with the coldest VRAM resident on the copy stream -- one in flight a layer, a few a token, off the step's
// stream; the slot turns into a hit once its copies have landed. Without one, a miss is promoted straight from
// the mapping on the step's stream (the correctness floor). A slot read this epoch (this token, or this window)
// is never a victim.
void Gpu::Impl::q4e_cache_admit(int l, const int32_t* sel, const int32_t* dslot, int n_used, bool* is_hit, Q4e::Blob* src,
                                 bool may_promote) {
    Q4e& qq = q;
    const int ne = c.n_expert;
    Q4e::Slab& s = qq.slab[size_t(l)];
    const Layer& y = L[size_t(l)];
    float* heat = qq.eheat.data() + size_t(l) * size_t(ne);

    for (int i = 0; i < n_used; ++i) heat[static_cast<uint32_t>(sel[i])] += 1.0f;
    if (!qq.lt.empty())
        for (int i = 0; i < n_used; ++i) qq.lt[size_t(l) * size_t(ne) + static_cast<uint32_t>(sel[i])] += 1.0f;
    for (int i = 0; i < n_used; ++i) is_hit[i] = dslot[i] >= 0;   // the device's split decided; the host follows it

    int32_t held_ram[32];
    int nheld_ram = 0;
    int best = -1;
    float best_h = -1.0f;
    for (int i = 0; i < n_used; ++i) {
        if (is_hit[i]) continue;
        const uint32_t e = static_cast<uint32_t>(sel[i]);
        const int32_t t = s.ram ? s.ram_slot_of[e] : -1;
        const uint8_t* r = nullptr;
        if (t >= 0) {
            std::atomic_thread_fence(std::memory_order_acquire);
            if (s.ram_ready[size_t(t)]) r = s.ram_ptr[size_t(t)];
        }
        if (r) {
            src[i] = {r, r + s.ge, r + 2 * s.ge};
            ++qq.miss_ram;
            held_ram[nheld_ram++] = t;
            if (!s.ram_pending[size_t(t)] && heat[e] > best_h) {
                best = i;
                best_h = heat[e];
            }
        } else {
            const uint64_t ex = e;
            src[i] = {y.geh.p + ex * y.ge_bytes, y.ueh.p + ex * y.ge_bytes, y.deh.p + ex * y.de_bytes};
            touch_ahead(src[i].g, y.ge_bytes);
            touch_ahead(src[i].u, y.ge_bytes);
            touch_ahead(src[i].d, y.de_bytes);
            ++qq.miss_file;
        }
    }

    // the coldest VRAM resident this epoch has not read (-1: none)
    auto coldest_slot = [&]() -> int {
        int vs = -1;
        float cold = FLT_MAX;
        for (int t = 0; t < s.n_slots; ++t) {
            if (s.stamp[size_t(t)] == qq.epoch) continue;
            const int32_t re = s.expert_of[size_t(t)];
            if (re == -1) return t;
            if (re < 0) continue;   // a promotion's target
            if (heat[static_cast<uint32_t>(re)] < cold) {
                cold = heat[static_cast<uint32_t>(re)];
                vs = t;
            }
        }
        return vs;
    };

    if (!may_promote) {
    } else if (s.ram && s.n_slots > 0) {
        if (best >= 0 && !s.pr_on && qq.pr_budget > 0) {
            const int32_t e = sel[best];
            const int32_t rt = s.ram_slot_of[static_cast<uint32_t>(e)];
            int vs = s.used < s.n_slots ? s.used++ : coldest_slot();
            int32_t v = vs >= 0 ? s.expert_of[size_t(vs)] : -1;
            (void)rt;
            if (v >= 0 && !(heat[static_cast<uint32_t>(e)] > heat[static_cast<uint32_t>(v)] + 0.5f)) vs = -1;
            if (vs >= 0 && q4e_swap(l, e, vs, false)) {
                (void)cudaEventRecord(qq.pr_ev[size_t(l)], qq.cstream);
                --qq.pr_budget;
                ++qq.promotions;
            }
        }
    } else {
        // No RAM tier: the copy comes off the file's mapping on the step's stream (pageable: the driver stages it
        // and blocks this thread on whatever the page cache cannot serve).
        for (int i = 0; s.n_slots > 0 && i < n_used; ++i) {
            if (is_hit[i]) continue;
            const int victim = s.used < s.n_slots ? s.used++ : coldest_slot();
            if (victim < 0) continue;
            const int32_t old = s.expert_of[size_t(victim)];
            if (old >= 0) s.slot_of[static_cast<uint32_t>(old)] = -1;
            s.expert_of[size_t(victim)] = sel[i];
            s.slot_of[static_cast<uint32_t>(sel[i])] = victim;
            s.stamp[size_t(victim)] = qq.epoch;
            const uint64_t ex = static_cast<uint32_t>(sel[i]);
            uint8_t* dst = s.base + uint64_t(victim) * s.blob;
            (void)cudaMemcpyAsync(dst, y.geh.p + ex * y.ge_bytes, y.ge_bytes, cudaMemcpyHostToDevice, stream);
            (void)cudaMemcpyAsync(dst + s.ge, y.ueh.p + ex * y.ge_bytes, y.ge_bytes, cudaMemcpyHostToDevice, stream);
            (void)cudaMemcpyAsync(dst + 2 * s.ge, y.deh.p + ex * y.de_bytes, y.de_bytes, cudaMemcpyHostToDevice, stream);
        }
    }

    // Every miss that neither tier holds is queued to the filler. The step does not wait for it.
    for (int i = 0; s.ram && i < n_used; ++i) {
        const uint32_t e = static_cast<uint32_t>(sel[i]);
        if (is_hit[i] || s.ram_slot_of[e] >= 0 || (s.pr_on && s.pr_v == int32_t(e))) continue;
        const int32_t t = q4e_ram_fill(l, static_cast<int32_t>(e), held_ram, nheld_ram);
        if (t >= 0) held_ram[nheld_ram++] = t;
    }
    qq.cache_lookups += uint64_t(n_used);
}

// Queues one expert into the RAM tier: a free slot, or the coldest resident that is not in `held`, and never
// one whose fill has not landed — the lane must not read bytes the filler is about to overwrite. Returns the
// slot it claimed, or -1 when the tier holds nothing evictable or the filler's ring is full.
int32_t Gpu::Impl::q4e_ram_fill(int l, int32_t e, const int32_t* held, int nheld) {
    Q4e& qq = q;
    Q4e::Slab& s = qq.slab[size_t(l)];
    if (s.ram_slots == 0) return -1;
    const uint32_t room = uint32_t(Q4e::kFillRing) -
                          (qq.fill_tail.load(std::memory_order_relaxed) - qq.fill_head.load(std::memory_order_acquire));
    if (room == 0) return -1;
    float* heat = qq.eheat.data() + size_t(l) * size_t(c.n_expert);
    int victim = -1;
    if (s.ram_used < s.ram_slots) {
        victim = s.ram_used++;
    } else {
        float coldest = FLT_MAX;
        for (int t = 0; t < s.ram_slots; ++t) {
            if (s.ram_pending[size_t(t)]) continue;
            bool taken = false;
            for (int h = 0; h < nheld; ++h)
                if (held[h] == t) taken = true;
            if (taken) continue;
            const int32_t re = s.ram_expert_of[size_t(t)];
            if (re < 0) {
                victim = t;
                break;
            }
            if (!s.ram_ready[size_t(t)]) continue;   // its fill is in flight
            if (heat[static_cast<uint32_t>(re)] < coldest) {
                coldest = heat[static_cast<uint32_t>(re)];
                victim = t;
            }
        }
        if (victim < 0) return -1;
        const int32_t old = s.ram_expert_of[size_t(victim)];
        if (old >= 0) s.ram_slot_of[static_cast<uint32_t>(old)] = -1;
        s.ram_ready[size_t(victim)] = 0;
    }
    s.ram_expert_of[size_t(victim)] = e;
    s.ram_slot_of[static_cast<uint32_t>(e)] = victim;
    const uint32_t tail = qq.fill_tail.load(std::memory_order_relaxed);
    qq.fill[tail % Q4e::kFillRing] = Q4e::FillReq{l, e, victim};
    qq.fill_tail.store(tail + 1, std::memory_order_release);
    return victim;
}

// The hyper-connection read (step.cpp hc_mix): per-stream rmsnorm, the low-rank down -> silu(x/hc) -> up
// path, the sigmoid gate mix, and (when the module has one) the inject row. Writes `mixed` and `inject`.
bool Gpu::Impl::q4e_hc_read(const Layer::Hc& m, const float* st) {
    const Cfg& cc = c;
    const int e = cc.n_embd, hc = q.hc, rank = q.rank;
    const float inv_hc = 1.0f / float(hc);
    const float eps = static_cast<float>(cc.eps);
    cudaStream_t s = stream;
    if (!dec::rmsnorm(st, m.norm, q.xn, e, eps, 1, s, 0, 0, hc)) return false;   // the hc streams in one launch
    if (!gemv(m.down, q.xn, q.lo, rank, hc * e)) return false;
    ops::silu_scale(q.lo, q.lo, rank, inv_hc, s);
    if (!gemv(m.up, q.lo, q.gate, hc * e, rank)) return false;
    ops::hc_gate_mix(q.xn, q.gate, q.mixed, hc, e, inv_hc, s);
    // the inject's few rows (gemv_narrow: one block a row instead of one warp)
    if (m.inject.p && !limbo::cuda::gemv_narrow(m.inject.type, m.inject.p, q.xn, 0, q.inj, 0, hc, hc * e, 1, false, s) &&
        !gemv(m.inject, q.xn, q.inj, hc, hc * e))
        return false;
    return true;
}

// The GDN mixer on `mixed` (step.cpp layer_linear): conv(qkv), q/k head norms, the delta-rule scan over the
// state, the sigmoid output gate (qwen4exp's one difference from qwen3.5), ssm_out into `out`.
bool Gpu::Impl::q4e_gdn(int l, const float* mixed, float* out) {
    const Cfg& cc = c;
    Layer& y = L[static_cast<size_t>(l)];
    cudaStream_t s = stream;
    const int e = cc.n_embd, qn = cc.n_k_head * cc.gdn_head, vn = cc.n_v_head * cc.gdn_head;
    const float gdn_scale = static_cast<float>(1.0 / std::sqrt(double(cc.gdn_head)));
    const float gdn_eps = static_cast<float>(cc.eps / cc.gdn_head);
    if (!gemv(y.qkv, mixed, this->qkv, cc.qkv_w, e)) return false;
    ops::conv_silu(this->qkv, hist[static_cast<size_t>(l)], y.conv, conv, cc.qkv_w, s);
    ops::head_norm(conv, qk, 2 * cc.n_k_head, cc.gdn_head, gdn_eps, gdn_scale, s);
    if (y.alpha.type == F32 && y.beta.type == F32 && e <= 2048 && cc.dt_rank <= 64) {
        if (!dec::f32_cols(reinterpret_cast<const float*>(y.alpha.p), reinterpret_cast<const float*>(y.beta.p), mixed, e,
                           alpha, beta, 64, cc.dt_rank, e, 1, 2, s))
            return false;
    } else if (y.alpha.type == y.beta.type) {
        if (!gemv_dual(y.alpha.type, y.alpha.p, y.beta.p, mixed, alpha, beta, cc.dt_rank, e, s)) return false;
    } else {
        if (!gemv(y.alpha, mixed, alpha, cc.dt_rank, e) || !gemv(y.beta, mixed, beta, cc.dt_rank, e)) return false;
    }
    ops::gdn_scan(S[static_cast<size_t>(l)], qk, conv + 2 * qn, alpha, beta, y.dtb, y.ssm_a, o, cc.n_v_head,
                  cc.n_k_head, cc.gdn_head, cc.dt_rank, gdn_scale, s);
    if (!gemv(y.gate, mixed, z, vn, e)) return false;
    ops::gdn_gate(o, z, y.ssm_norm, gated, cc.n_v_head, cc.gdn_head, cc.eps, s, true);
    return gemv(y.out, gated, out, e, vn);
}

// The sparse-attention mixer on `mixed` (step.cpp layer_attention): q with its gate half, the f16 K/V cache,
// the split flash-decode over the visible positions, W_o. The QSA block selection is not ported (the step
// refuses past its budget); below it the visible set is the plain causal one.
bool Gpu::Impl::q4e_attn(int l, const float* mixed, float* out) {
    const Cfg& cc = c;
    Layer& y = L[static_cast<size_t>(l)];
    const int e = cc.n_embd, qw = cc.n_head * cc.hd, kvw = cc.n_head_kv * cc.hd;
    const float attn_scale = static_cast<float>(1.0 / std::sqrt(double(cc.hd)));
    if (!gemv(y.q, mixed, qfull, 2 * qw, e) || !gemv(y.k, mixed, kbuf, kvw, e)) return false;
    // v may be the raw k projection (before k's norm and rope): the prep writes it to the cache unchanged.
    const float* vsrc = vbuf;
    if (y.v.p) {
        if (!gemv(y.v, mixed, vbuf, kvw, e)) return false;
    } else {
        vsrc = kbuf;
    }
    (void)attn_scale;
    q4e_prep(l, qfull, kbuf, vsrc, y.qn, y.kn, qbuf, d_args + 1, 1);
    bool sparse = false;
    if (y.iq.p && !q4e_qsa(l, mixed, e, 1, qsa_pos, d_args + 1, qbuf, qfull, attn, &sparse)) return false;
    if (!sparse && !q4e_attend(l, qbuf, qfull, attn, attn_scr, d_args + 1, 1)) return false;
    return gemv(y.o, attn, out, e, qw);
}

// The MoE on `mixed` (step.cpp moe + the shared expert): the router and the top-10 stay on the device, the
// shared expert beside them, and the 10 routed SwiGLUs run on the CPU lane over the file's own expert rows.
// Writes out = sum_i w_i expert_i + sigmoid(g . mixed) * shared.
// Defined below, beside ensure_q4e_rows: one batched launch where the input width is a whole number of 256-blocks,
// one plain gemv a row otherwise. q4e_moe_rows (the window's MoE) needs it before its definition.
static bool q4e_gemv_rows(uint32_t type, const void* W, const float* x, int x_stride, float* y, int y_stride,
                          int n_out, int n_in, int T, cudaStream_t s);

// The device half of one layer's MoE, enqueued without waiting for anything: the router, the top-k, the split and
// its doorbell; the resident experts and the shared expert; the wait for the feeder (q4e_moe_serve); the lane's and
// the PCIe share's products into `dn`; the combine. Writes out = sum_i w_i expert_i + sigmoid(g . mixed) * shared.
// LIMBO_Q4E_PREDICT=1 (a measurement, single steps only): layer l+1's router applied to layer l's MoE input, its top 16
// against what l+1 then picks. The figure that matters for a prefetch over the idle PCIe link: of the first k predicted
// experts that are NOT resident, how many l+1 really selects (each one a miss the lane would not have to read).
struct Q4ePredict {
    bool on = false, init = false;
    int32_t *pred = nullptr, *act = nullptr;
    float *rlog = nullptr, *w = nullptr;
    int nl = 0;
    double steps = 0, overlap = 0, misses = 0, useful[4] = {}, taken[4] = {};
    static Q4ePredict& get() {
        static Q4ePredict p;
        if (!p.init) {
            p.init = true;
            p.on = std::getenv("LIMBO_Q4E_PREDICT") != nullptr;
        }
        return p;
    }
    bool ensure(int n_layer, int n_expert) {
        if (pred) return true;
        nl = n_layer;
        return cudaMalloc(&pred, sizeof(int32_t) * size_t(n_layer) * 16) == cudaSuccess &&
               cudaMalloc(&act, sizeof(int32_t) * size_t(n_layer) * 16) == cudaSuccess &&
               cudaMalloc(&rlog, sizeof(float) * size_t(n_expert)) == cudaSuccess && cudaMalloc(&w, sizeof(float) * 16) == cudaSuccess;
    }
};

bool Gpu::Impl::q4e_moe(int l) {
    const Cfg& cc = c;
    Q4e& qq = q;
    Layer& y = L[static_cast<size_t>(l)];
    cudaStream_t s = stream;
    const int e = cc.n_embd, ns = cc.n_ff_shexp, nu = cc.n_used, nff = cc.n_ff_exp;

    Q4eClock& ck = Q4eClock::get();
    if (ck.on && l < 64) (void)cudaEventRecord(ck.out[static_cast<size_t>(l)], s);
    if (!gemv(y.router, qq.mixed, rlog, cc.n_expert, e)) return false;
    if (!dec::topk(rlog, cc.n_expert, nu, 1, ids, wts, s)) return false;
    if (!q4e_bell(2 * l, l, ids, nu, qq.mixed, qq.host_x_d, e, 0)) return false;
    // the next layer's likely experts, for the feeder's prefetch (behind the bell: the lane starts first)
    if (qq.pf_n > 0 && l + 1 < cc.n_layer_main) {
        if (!gemv(L[size_t(l) + 1].router, qq.mixed, qq.pred_rlog, cc.n_expert, e) ||
            !dec::topk(qq.pred_rlog, cc.n_expert, 16, 1, qq.pred_top, qq.pred_w, s))
            return false;
        q4e_pred_kernel<<<1, 32, 0, s>>>(qq.pred_top, qq.pred_d + size_t(l + 1) * 16, qq.flag_d + Q4e::kPred + l + 1, qq.d_seq);
    }
    if (Q4ePredict& ps = Q4ePredict::get(); ps.on && ps.ensure(cc.n_layer_main, cc.n_expert)) {
        (void)cudaMemcpyAsync(ps.act + size_t(l) * 16, ids, sizeof(int32_t) * nu, cudaMemcpyDeviceToDevice, s);
        if (l + 1 < cc.n_layer_main && (!gemv(L[size_t(l) + 1].router, qq.mixed, ps.rlog, cc.n_expert, e) ||
                                        !dec::topk(ps.rlog, cc.n_expert, 16, 1, ps.pred + size_t(l + 1) * 16, ps.w, s)))
            return false;
    }

    // The resident experts: gemv_experts over all nu selections with the split's slots as ids (a miss's -1 skips
    // its slot), each product straight into its own row of `dn`.
    const Q4e::Slab& sb = qq.slab[static_cast<size_t>(l)];
    if (cudaMemsetAsync(dn, 0, sizeof(float) * e * nu, s) != cudaSuccess) return false;
    if (sb.base) {
        if (!limbo::cuda::gemv_experts(y.geh.type, sb.base, sb.blob, qq.d_sel, nu, qq.mixed, 0, gbuf, nff, e, s) ||
            !limbo::cuda::gemv_experts(y.ueh.type, sb.base + sb.ge, sb.blob, qq.d_sel, nu, qq.mixed, 0, ubuf, nff, e, s))
            return false;
        ops::swiglu(gbuf, ubuf, hbuf, nu * nff, s);
        if (!limbo::cuda::gemv_experts(y.deh.type, sb.base + 2 * sb.ge, sb.blob, qq.d_sel, nu, hbuf, nff, dn, e, nff, s)) return false;
    }
    // The shared expert, beside the lane: SwiGLU down through the sigmoid gate of ffn_gate_inp_shexp.
    if (y.sgate.type == y.sup.type) {
        if (!gemv_dual(y.sgate.type, y.sgate.p, y.sup.p, qq.mixed, gs, us, ns, e, s)) return false;
    } else if (!gemv(y.sgate, qq.mixed, gs, ns, e) || !gemv(y.sup, qq.mixed, us, ns, e)) {
        return false;
    }
    ops::swiglu(gs, us, hs, ns, s);
    if (!gemv(y.sdown, hs, shared, e, ns)) return false;
    ops::shexp_gate(y.shgate, qq.mixed, sg, e, s);

    const int wi = ck.on && ck.nw < 128 ? ck.nw++ : -1;
    if (wi >= 0) (void)cudaEventRecord(ck.w0[wi], s);
    if (!q4e_wait(Q4e::kDone + 2 * l, qq.ctl_d[0], qq.ctl_dev[0], Q4eCtl::kHead + nu)) return false;
    if (wi >= 0) (void)cudaEventRecord(ck.w1[wi], s);
    const dim3 g1(static_cast<unsigned>((e + 255) / 256), Q4e::kDma), g2(static_cast<unsigned>((e + 255) / 256), static_cast<unsigned>(nu));
    q4e_scatter_kernel<<<g1, 256, 0, s>>>(dn, qq.dn_dma, qq.ctl_dev[0]->dma_idx, &qq.ctl_dev[0]->n_dma, e);
    q4e_scatter_kernel<<<g2, 256, 0, s>>>(dn, qq.host_dn_d, qq.ctl_dev[0]->miss_idx, &qq.ctl_dev[0]->n_miss, e);
    if (cudaMemsetAsync(qq.out, 0, sizeof(float) * e, s) != cudaSuccess) return false;
    ops::moe_combine(qq.out, dn, wts, shared, sg, e, nu, s);
    return cudaGetLastError() == cudaSuccess;
}

// The prefetch for layer l1, issued while layer l1 - 1's lane runs: its prediction (raised right behind the bell; skipped
// when it is not there within 300 us), then the first pf_n predicted experts that are neither resident nor promoting,
// RAM tier to the staging set of l1's parity on their own stream. Only the spare buffer of a RAM tier is ever written,
// and a staged expert is only used while it is still a miss, so its source bytes cannot move under the copy.
void Gpu::Impl::q4e_prefetch(int l1) {
    Q4e& qq = q;
    const uint32_t want = qq.seq;
    const uint32_t* f = qq.flag_h + Q4e::kPred + l1;
    const auto t0 = std::chrono::steady_clock::now();
    while (__atomic_load_n(f, __ATOMIC_ACQUIRE) != want)
        if (std::chrono::steady_clock::now() - t0 > std::chrono::microseconds(300)) return;
    Q4e::Slab& sb = qq.slab[size_t(l1)];
    if (!sb.ram || !sb.slot_of) return;
    const int set = l1 & 1;
    uint8_t* pb = qq.pf_buf + uint64_t(set) * Q4e::kPf * qq.dma_blob;
    const int32_t* P = qq.pred_h + size_t(l1) * 16;
    int n = 0;
    for (int i = 0; i < 16 && n < qq.pf_n; ++i) {
        const int32_t x = P[i];
        if (x < 0 || x >= c.n_expert || sb.slot_of[x] >= 0) continue;
        const int32_t rt = sb.ram_slot_of[size_t(x)];
        if (rt < 0 || !sb.ram_ready[size_t(rt)] || sb.ram_pending[size_t(rt)]) continue;
        if (cudaMemcpyAsync(pb + uint64_t(n) * qq.dma_blob, sb.ram_ptr[size_t(rt)], sb.blob, cudaMemcpyHostToDevice, qq.pf_stream) !=
            cudaSuccess)
            break;
        qq.pf_ex[set][n++] = x;
    }
    if (cudaEventRecord(qq.pf_ev[set], qq.pf_stream) != cudaSuccess) n = 0;
    qq.pf_cnt[set] = n;
    qq.pf_layer[set] = l1;
    qq.pf_seq[set] = want;
    qq.pf_issued += uint64_t(n);
}

// The host half of layer l's MoE (the feeder): wait for its bell, admit, start the lane on the misses (and the PCIe
// share on its stream), hand the device the control block, raise done once every product is in.
bool Gpu::Impl::q4e_moe_serve(int l) {
    const Cfg& cc = c;
    Q4e& qq = q;
    Layer& y = L[static_cast<size_t>(l)];
    const int e = cc.n_embd, nu = cc.n_used, nff = cc.n_ff_exp;
    Q4eClock& ck = Q4eClock::get();
    const auto t_wait = std::chrono::steady_clock::now();
    if (!q4e_bell_wait(2 * l)) return false;
    const auto t_bell = std::chrono::steady_clock::now();
    if (ck.on) ck.sync_wait += std::chrono::duration<double, std::milli>(t_bell - t_wait).count();

    const int32_t *sel = qq.split_h, *dslot = qq.split_h + kQ4ePairs;
    q4e_split_book(l, nu);
    bool is_hit[32];
    Q4e::Blob src[32];
    q4e_cache_admit(l, sel, dslot, nu, is_hit, src);
    int miss_of[32], n_miss = 0, n_hit = 0;
    for (int i = 0; i < nu; ++i) {
        if (is_hit[i]) ++n_hit;
        else if (qq.tail_skip > 0 && i >= qq.tail_skip) ++qq.skipped;   // top-k order: i is the rank
        else miss_of[n_miss++] = i;
    }
    qq.cache_hits += uint64_t(n_hit);
    // the misses the prefetch staged for this layer (this step's set): computed on the device, not read by the lane
    int st_of[Q4e::kPf] = {}, st_slot[Q4e::kPf] = {}, n_st = 0;
    const int pset = l & 1;
    if (qq.pf_n > 0 && qq.pf_layer[pset] == l && qq.pf_seq[pset] == qq.seq) {
        for (int m = 0; m < n_miss && n_st < Q4e::kPf;) {
            const int i = miss_of[m];
            int k = 0;
            while (k < qq.pf_cnt[pset] && qq.pf_ex[pset][k] != sel[i]) ++k;
            if (k == qq.pf_cnt[pset]) {
                ++m;
                continue;
            }
            st_of[n_st] = i;
            st_slot[n_st++] = k;
            for (int j = m; j + 1 < n_miss; ++j) miss_of[j] = miss_of[j + 1];
            --n_miss;
        }
        qq.pf_used += uint64_t(n_st);
    }
    // the PCIe share (docs/donors/strata.md): pcie_share of the misses, RAM-held ones, go over the bus and are computed on the device
    const Q4e::Slab& sb = qq.slab[static_cast<size_t>(l)];
    int dma_of[Q4e::kDmaShare] = {}, n_dma = 0;
    if (qq.dma_min > 0 && n_miss >= qq.dma_min && sb.ram) {
        const int want = std::min({Q4e::kDmaShare, Q4e::kDma - n_st, int(qq.pcie_share * float(n_miss) + 0.5f)});
        for (int m = n_miss - 1; m >= 0 && n_dma < want; --m) {
            const int i = miss_of[m];
            const int32_t rt = sb.ram_slot_of[static_cast<uint32_t>(sel[i])];
            if (rt < 0 || !sb.ram_ready[size_t(rt)] || src[i].g != sb.ram_ptr[size_t(rt)]) continue;
            dma_of[n_dma++] = i;
            for (int k = m; k + 1 < n_miss; ++k) miss_of[k] = miss_of[k + 1];
            --n_miss;
        }
    }

    cpu::ExpertJob jobs[32];
    for (int m = 0; m < n_miss; ++m) {
        const int i = miss_of[m];
        cpu::ExpertJob& job = jobs[m];
        job.w.gate = src[i].g;
        job.w.up = src[i].u;
        job.w.down = src[i].d;
        job.w.gate_type = y.geh.type;
        job.w.up_type = y.ueh.type;
        job.w.down_type = y.deh.type;
        job.w.n_embd = e;
        job.w.n_ff = nff;
        job.nrows = 1;
        job.act_row = nullptr;
        job.out = qq.host_dn + static_cast<size_t>(m) * e;
    }
    std::string lerr;
    const auto t_lane = std::chrono::steady_clock::now();
    if (n_miss > 0 && !qq.lane->submit(qq.host_x, 1, jobs, n_miss, &lerr)) return false;
    if (ck.on) ck.sync += std::chrono::duration<double, std::milli>(t_lane - t_bell).count();

    // The device's misses on their own stream, beside the lane, into dn_dma: first the staged ones (once their copy is
    // in), then the PCIe share's (the blob over the bus first). The device's mixed stays put until done. Launched after
    // the lane's submit, so they cost the lane nothing.
    cudaStream_t ds = qq.dma_stream;
    if (n_st > 0) {
        for (int k = 0; k < n_st; ++k) qq.pf_slot_h[k] = st_slot[k];
        const uint8_t* pb = qq.pf_buf + uint64_t(pset) * Q4e::kPf * qq.dma_blob;
        if (cudaStreamWaitEvent(ds, qq.pf_ev[pset], 0) != cudaSuccess ||
            !limbo::cuda::gemv_experts(y.geh.type, pb, qq.dma_blob, qq.pf_slot_d, n_st, qq.mixed, 0, qq.dma_g, nff, e, ds) ||
            !limbo::cuda::gemv_experts(y.ueh.type, pb + sb.ge, qq.dma_blob, qq.pf_slot_d, n_st, qq.mixed, 0, qq.dma_u, nff, e, ds))
            return false;
        ops::swiglu(qq.dma_g, qq.dma_u, qq.dma_h, n_st * nff, ds);
        if (!limbo::cuda::gemv_experts(y.deh.type, pb + 2 * sb.ge, qq.dma_blob, qq.pf_slot_d, n_st, qq.dma_h, nff, qq.dn_dma, e, nff, ds))
            return false;
    }
    if (n_dma > 0) {
        float *g = qq.dma_g + size_t(n_st) * nff, *u = qq.dma_u + size_t(n_st) * nff, *h = qq.dma_h + size_t(n_st) * nff;
        for (int k = 0; k < n_dma; ++k)
            if (cudaMemcpyAsync(qq.dma_buf + uint64_t(k) * qq.dma_blob, src[dma_of[k]].g, sb.blob, cudaMemcpyHostToDevice, ds) != cudaSuccess)
                return false;
        if (!limbo::cuda::gemv_experts(y.geh.type, qq.dma_buf, qq.dma_blob, qq.dev_iota, n_dma, qq.mixed, 0, g, nff, e, ds) ||
            !limbo::cuda::gemv_experts(y.ueh.type, qq.dma_buf + sb.ge, qq.dma_blob, qq.dev_iota, n_dma, qq.mixed, 0, u, nff, e, ds))
            return false;
        ops::swiglu(g, u, h, n_dma * nff, ds);
        if (!limbo::cuda::gemv_experts(y.deh.type, qq.dma_buf + 2 * sb.ge, qq.dma_blob, qq.dev_iota, n_dma, h, nff,
                                       qq.dn_dma + size_t(n_st) * e, e, nff, ds))
            return false;
        qq.dma_n += uint64_t(n_dma);
    }
    const int n_dev = n_st + n_dma;
    if (n_dev > 0 && cudaEventRecord(qq.dma_ev, ds) != cudaSuccess) return false;
    // the next layer's likely misses over the bus while this one finishes
    if (qq.pf_n > 0 && l + 1 < cc.n_layer_main) q4e_prefetch(l + 1);
    Q4eCtl& ctl = *qq.ctl_h[0];
    ctl.n_miss = n_miss;
    ctl.n_dma = n_dev;
    for (int k = 0; k < n_st; ++k) ctl.dma_idx[k] = st_of[k];
    for (int k = 0; k < n_dma; ++k) ctl.dma_idx[n_st + k] = dma_of[k];
    for (int m = 0; m < n_miss; ++m) ctl.miss_idx[m] = miss_of[m];
    if (n_miss > 0) qq.lane->wait();
    const auto t_done = std::chrono::steady_clock::now();
    if (ck.on) ck.lane += std::chrono::duration<double, std::milli>(t_done - t_lane).count();
    if (n_dev > 0 && cudaEventSynchronize(qq.dma_ev) != cudaSuccess) return false;
    q4e_done(2 * l);
    q4e_pr_poll(l);
    if (ck.on) ck.sync += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_done).count();
    return true;
}

// The verify window, rows r0 .. r0 + n - 1 of it (group g; a window of split_min rows or more runs as two groups, so the
// CPU lane of one group runs while the device computes the other). The dense half of layer l and the MoE's front: the
// router, the top-k, the residency split and its doorbell, the resident experts and the shared expert. Every op is the
// _rows sibling of the single step's, on the group's rows; the second group reads the first's K/V, conv inputs and
// delta-net state (Stmp when the window does not write back).
bool Gpu::Impl::q4e_rows_front(int l, int r0, int n, int g, int T, int pos0, bool write_back, bool split, int pb) {
    Impl& I = *this;
    const Cfg& cc = c;
    Q4e& qq = q;
    cudaStream_t s = stream;
    const int e = cc.n_embd, hc = qq.hc, pitch = hc * e, nu = cc.n_used, nff = cc.n_ff_exp, ns = cc.n_ff_shexp, ne = cc.n_expert;
    const float inv_hc = 1.0f / float(hc), eps = static_cast<float>(cc.eps);
    const int qw = cc.n_head * cc.hd, kvw = cc.n_head_kv * cc.hd;
    const int qn = cc.n_k_head * cc.gdn_head, qkn_w = 2 * qn, vn = cc.n_v_head * cc.gdn_head;
    const float gdn_scale = static_cast<float>(1.0 / std::sqrt(double(cc.gdn_head)));
    const float gdn_eps = static_cast<float>(cc.eps / cc.gdn_head);
    const float attn_scale = static_cast<float>(1.0 / std::sqrt(double(cc.hd)));
    Layer& y = I.L[static_cast<size_t>(l)];
    Q4eClock& ck = Q4eClock::get();
    if (ck.on && g == 0 && l < 64) (void)cudaEventRecord(ck.in[static_cast<size_t>(l)], s);
    float* st = qrw.st + size_t(r0) * pitch;
    const size_t R = size_t(r0);
    // The PLE block of its layer opens layer 1: the stack grows by gated + silu(dilated conv). The conv rolls the
    // window's own history, so the second group continues from the first.
    if (y.ple_key.p) {
        const int ew = qq.n_heads * qq.np;
        if (g == 0 && !I.q4e_wait(Q4e::kPle + pb, qq.emb_hd[pb], qrw.emb, T * ew)) return false;
        if (!q4e_gemv_rows(y.ple_key.type, y.ple_key.p, qrw.emb + R * ew, ew, qrw.pkey + R * pitch, pitch, pitch, ew, n, s) ||
            !q4e_gemv_rows(y.ple_value.type, y.ple_value.p, qrw.emb + R * ew, ew, qrw.pval + R * e, e, e, ew, n, s))
            return false;
        ops::ple_gate_rows(qrw.pkey + R * pitch, st, qrw.pval + R * e, y.ple_gk, y.ple_gq, qrw.pgated + R * pitch, hc, e, n, eps, s);
        if (!dec::rmsnorm(qrw.pgated + R * pitch, y.ple_gc, qrw.pnrm + R * pitch, e, eps, n, s, pitch, pitch, hc)) return false;
        ops::ple_conv_silu_rows(qrw.pnrm + R * pitch, qrw.phist, y.ple_conv, qrw.pconv + R * pitch, pitch, qq.kern, qq.ngram, n, s);
        ops::add_into(st, qrw.pgated + R * pitch, qrw.pconv + R * pitch, n * pitch, s);
    }

    // The attention half: HC read, mixer, HC write.
    if (!I.q4e_hc_read_rows(y.hca, st, n, r0)) return false;
    const float* mixed = qrw.mixed + R * e;
    const int* pd = qrw.d_args + 2 * g + 1;
    if (y.full) {
        float *qfull = qrw.qfull + R * 2 * qw, *kbuf = qrw.kbuf + R * kvw, *vbuf = qrw.vbuf + R * kvw, *qbuf = qrw.qbuf + R * qw,
              *attn = qrw.attn + R * qw;
        if (!q4e_gemv_rows(y.q.type, y.q.p, mixed, e, qfull, 2 * qw, 2 * qw, e, n, s) ||
            !q4e_gemv_rows(y.k.type, y.k.p, mixed, e, kbuf, kvw, kvw, e, n, s))
            return false;
        // v may be the raw k projection (before k's norm and rope): the prep writes it to the cache unchanged.
        const float* vsrc = vbuf;
        if (y.v.p) {
            if (!q4e_gemv_rows(y.v.type, y.v.p, mixed, e, vbuf, kvw, kvw, e, n, s)) return false;
        } else {
            vsrc = kbuf;
        }
        I.q4e_prep(l, qfull, kbuf, vsrc, y.qn, y.kn, qbuf, pd, n);
        bool sparse = false;
        if (y.iq.p && !I.q4e_qsa(l, mixed, e, n, pos0 + r0, pd, qbuf, qfull, attn, &sparse)) return false;
        if (!sparse && !I.q4e_attend(l, qbuf, qfull, attn, qrw.attn_scr, pd, n)) return false;
        if (!q4e_gemv_rows(y.o.type, y.o.p, attn, qw, qrw.out + R * e, e, e, qw, n, s)) return false;
    } else {
        // This layer's own conv input (commit replays layer by layer); the second group's conv reads the first's rows.
        float* qkvbuf = qrw.qkvk + size_t(l) * size_t(kQ4eRows) * size_t(cc.qkv_w);
        if (!q4e_gemv_rows(y.qkv.type, y.qkv.p, mixed, e, qkvbuf + R * cc.qkv_w, cc.qkv_w, cc.qkv_w, e, n, s)) return false;
        ops::conv_silu_rows(qkvbuf, I.hist[size_t(l)], y.conv, qrw.conv, cc.qkv_w, r0 + n, s, write_back && r0 + n == T, r0);
        ops::head_norm_rows(qrw.conv + R * cc.qkv_w, cc.qkv_w, qrw.qk + R * qkn_w, qkn_w, n, 2 * cc.n_k_head, cc.gdn_head, gdn_eps,
                            gdn_scale, s);
        // This layer's own alpha/beta slot (commit reads it back, coef'd).
        const int ab_stride = 2 * cc.dt_rank;
        float* al = qrw.ab + size_t(l) * size_t(kQ4eRows) * ab_stride + R * ab_stride;
        float* be = al + cc.dt_rank;
        if (y.alpha.type == F32 && y.beta.type == F32 && e <= 2048 && cc.dt_rank <= 64) {
            if (!dec::f32_cols(reinterpret_cast<const float*>(y.alpha.p), reinterpret_cast<const float*>(y.beta.p), mixed, e, al, be,
                               ab_stride, cc.dt_rank, e, n, 2, s))
                return false;
        } else if (!q4e_gemv_rows(y.alpha.type, y.alpha.p, mixed, e, al, ab_stride, cc.dt_rank, e, n, s) ||
                   !q4e_gemv_rows(y.beta.type, y.beta.p, mixed, e, be, ab_stride, cc.dt_rank, e, n, s)) {
            return false;
        }
        ops::gdn_coef_rows(al, be, ab_stride, y.dtb, y.ssm_a, n, cc.dt_rank, s);
        // a split window that does not write back: the first group leaves its state in Stmp, the second starts there
        float* S_in = split && r0 > 0 && !write_back ? qrw.Stmp : I.S[size_t(l)];
        float* S_out = split && r0 == 0 && !write_back ? qrw.Stmp : nullptr;
        ops::gdn_scan_rows(S_in, qrw.qk + R * qkn_w, qkn_w, qrw.conv + R * cc.qkv_w + 2 * qn, cc.qkv_w, al, be, ab_stride,
                           qrw.o + R * vn, vn, n, cc.n_v_head, cc.n_k_head, cc.gdn_head, cc.dt_rank, gdn_scale, s, write_back, S_out);
        if (!q4e_gemv_rows(y.gate.type, y.gate.p, mixed, e, qrw.z + R * vn, vn, vn, e, n, s)) return false;
        // sigmoid_z = true (q4e's gate), not the silu(z) of gdn_gate_rows' default
        ops::gdn_gate_rows(qrw.o + R * vn, qrw.z + R * vn, y.ssm_norm, qrw.gated + R * vn, n * cc.n_v_head, cc.gdn_head, cc.eps, s, true);
        if (!q4e_gemv_rows(y.out.type, y.out.p, qrw.gated + R * vn, vn, qrw.out + R * e, e, e, vn, n, s)) return false;
    }
    ops::hc_combine_rows(st, qrw.out + R * e, qrw.inj + R * hc, hc, e, n, inv_hc, s);

    // The FFN half's read, then the MoE's front.
    if (!I.q4e_hc_read_rows(y.hcf, st, n, r0)) return false;
    if (ck.on && g == 0 && l < 64) (void)cudaEventRecord(ck.out[static_cast<size_t>(l)], s);
    const int np = n * nu;
    const size_t P0 = R * nu;
    if (!q4e_gemv_rows(y.router.type, y.router.p, mixed, e, qrw.rlog + R * ne, ne, ne, e, n, s) ||
        !dec::topk(qrw.rlog + R * ne, ne, nu, n, qrw.ids + P0, qrw.wts + P0, s) ||
        !q4e_bell(2 * l + g, l, qrw.ids + P0, np, mixed, qrw.post_d + R * e, n * e, int(P0)))
        return false;
    // the resident experts: pair p = t * nu + i reads its row (x_group nu), a miss's -1 skips it, products land in its row of dn
    const Q4e::Slab& sb = qq.slab[size_t(l)];
    if (cudaMemsetAsync(qrw.dn + P0 * e, 0, sizeof(float) * size_t(np) * size_t(e), s) != cudaSuccess) return false;
    if (sb.base) {
        float *gb = qrw.gbuf + P0 * nff, *ub = qrw.ubuf + P0 * nff, *hb = qrw.hbuf + P0 * nff;
        if (!limbo::cuda::gemv_experts(y.geh.type, sb.base, sb.blob, qq.d_sel + P0, np, mixed, e, gb, nff, e, s, nu) ||
            !limbo::cuda::gemv_experts(y.ueh.type, sb.base + sb.ge, sb.blob, qq.d_sel + P0, np, mixed, e, ub, nff, e, s, nu))
            return false;
        ops::swiglu(gb, ub, hb, np * nff, s);
        if (!limbo::cuda::gemv_experts(y.deh.type, sb.base + 2 * sb.ge, sb.blob, qq.d_sel + P0, np, hb, nff, qrw.dn + P0 * e, e, nff, s))
            return false;
    }
    // the shared expert, beside the lane
    if (!q4e_gemv_rows(y.sgate.type, y.sgate.p, mixed, e, qrw.gs + R * ns, ns, ns, e, n, s) ||
        !q4e_gemv_rows(y.sup.type, y.sup.p, mixed, e, qrw.us + R * ns, ns, ns, e, n, s))
        return false;
    ops::swiglu(qrw.gs + R * ns, qrw.us + R * ns, qrw.hs + R * ns, n * ns, s);
    if (!q4e_gemv_rows(y.sdown.type, y.sdown.p, qrw.hs + R * ns, ns, qrw.shared + R * e, e, e, ns, n, s)) return false;
    ops::shexp_gate_rows(y.shgate, mixed, qrw.sg + R, e, n, s);
    return cudaGetLastError() == cudaSuccess;
}

// The MoE's back for the group: the wait for its lane products (q4e_rows_finish), them into the pairs' rows, the
// combine and the HC write.
bool Gpu::Impl::q4e_rows_back(int l, int r0, int n, int g) {
    const Cfg& cc = c;
    Q4e& qq = q;
    cudaStream_t s = stream;
    const int e = cc.n_embd, hc = qq.hc, nu = cc.n_used;
    const size_t R = size_t(r0), P0 = R * nu;
    Q4eClock& ck = Q4eClock::get();
    const int wi = ck.on && ck.nw < 128 ? ck.nw++ : -1;
    if (wi >= 0) (void)cudaEventRecord(ck.w0[wi], s);
    if (!q4e_wait(Q4e::kDone + 2 * l + g, qq.ctl_d[g], qq.ctl_dev[g], Q4eCtl::kHead + n * nu)) return false;
    if (wi >= 0) (void)cudaEventRecord(ck.w1[wi], s);
    const dim3 gs(static_cast<unsigned>((e + 255) / 256), static_cast<unsigned>(n * nu));
    q4e_scatter_kernel<<<gs, 256, 0, s>>>(qrw.dn, qrw.dn_hd + P0 * e, qq.ctl_dev[g]->miss_idx, &qq.ctl_dev[g]->n_miss, e);
    if (cudaMemsetAsync(qrw.out + R * e, 0, sizeof(float) * size_t(n) * size_t(e), s) != cudaSuccess) return false;
    ops::moe_combine_rows(qrw.out + R * e, qrw.dn, qrw.d_pair + P0, qrw.wts + P0, qrw.shared + R * e, qrw.sg + R, e, nu, n, s);
    ops::hc_combine_rows(qrw.st + R * size_t(hc) * e, qrw.out + R * e, qrw.inj + R * hc, hc, e, n, 1.0f / float(hc), s);
    return cudaGetLastError() == cudaSuccess;
}

// The group's feeder, first half: the residency follows the device's split row by row, in row order (the sequence
// sequential steps would have made); the group's hits are stamped first, and a promotion only runs in the window's last
// group (a slot the other group's products read is never a victim). The misses go to the lane grouped by expert: rows
// that picked the same non-resident expert share ONE job, its rows contiguous in the output. The lane runs on while
// the caller queues the device's next work; q4e_rows_finish joins it.
bool Gpu::Impl::q4e_rows_submit(int l, int r0, int n, int g, bool last) {
    const Cfg& cc = c;
    Q4e& qq = q;
    Layer& y = L[static_cast<size_t>(l)];
    const int e = cc.n_embd, nu = cc.n_used, nff = cc.n_ff_exp;
    const int P0 = r0 * nu;
    Q4eClock& ck = Q4eClock::get();
    const auto t_wait = std::chrono::steady_clock::now();
    if (!q4e_bell_wait(2 * l + g)) return false;
    const auto t_bell = std::chrono::steady_clock::now();
    if (ck.on) ck.sync_wait += std::chrono::duration<double, std::milli>(t_bell - t_wait).count();

    q4e_split_book(l, n * nu, P0);
    struct Miss {
        int32_t ex, pair, row;
        Q4e::Blob src;
    };
    Miss ms[kQ4ePairs];
    int nm = 0;
    for (int t = r0; t < r0 + n; ++t) {
        const int32_t* sel = qq.split_h + static_cast<size_t>(t) * nu;
        const int32_t* dslot = qq.split_h + kQ4ePairs + static_cast<size_t>(t) * nu;
        bool is_hit[32];
        Q4e::Blob src[32];
        q4e_cache_admit(l, sel, dslot, nu, is_hit, src, last);
        for (int i = 0; i < nu; ++i) {
            if (is_hit[i]) ++qq.cache_hits;
            else ms[nm++] = Miss{sel[i], t * nu + i, t - r0, src[i]};
        }
    }
    // the route-tail skip: an expert whose every miss in the group is at rank tail_skip or lower is not computed
    if (qq.tail_skip > 0) {
        bool keep[kQ4ePairs] = {};
        for (int a = 0; a < nm; ++a)
            if (ms[a].pair % nu < qq.tail_skip)
                for (int b = 0; b < nm; ++b) keep[b] = keep[b] || ms[b].ex == ms[a].ex;
        int w = 0;
        for (int a = 0; a < nm; ++a) {
            if (keep[a]) ms[w++] = ms[a];
            else ++qq.skipped;
        }
        nm = w;
    }
    cpu::ExpertJob* jobs = qrw.jobs[g];
    Q4eCtl& ctl = *qq.ctl_h[g];
    int n_miss = 0, n_jobs = 0;
    {
        bool taken[kQ4ePairs] = {};
        for (int a = 0; a < nm; ++a) {
            if (taken[a]) continue;
            cpu::ExpertJob& job = jobs[n_jobs++];
            job.w.gate = ms[a].src.g;
            job.w.up = ms[a].src.u;
            job.w.down = ms[a].src.d;
            job.w.gate_type = y.geh.type;
            job.w.up_type = y.ueh.type;
            job.w.down_type = y.deh.type;
            job.w.n_embd = e;
            job.w.n_ff = nff;
            job.act_row = qrw.miss_row + P0 + n_miss;
            job.out = qrw.dn_h + static_cast<size_t>(P0 + n_miss) * e;
            job.nrows = 0;
            for (int b = a; b < nm; ++b) {
                if (taken[b] || ms[b].ex != ms[a].ex) continue;
                taken[b] = true;
                qrw.miss_row[P0 + n_miss] = ms[b].row;
                ctl.miss_idx[n_miss] = ms[b].pair;
                ++n_miss;
                ++job.nrows;
            }
        }
    }
    qrw.n_jobs = n_jobs;
    std::string lerr;
    const auto t_lane = std::chrono::steady_clock::now();
    ctl.n_miss = n_miss;
    ctl.n_dma = 0;
    qrw.n_miss[g] = n_miss;
    if (n_miss > 0 && !qq.lane->submit(qrw.post + static_cast<size_t>(r0) * e, n, jobs, n_jobs, &lerr)) return false;
    if (ck.on) ck.sync += std::chrono::duration<double, std::milli>(t_lane - t_bell).count();
    return true;
}

// The second half: the caller helps the lane to its end, and the device's wait opens.
bool Gpu::Impl::q4e_rows_finish(int l, int g, bool last) {
    Q4eClock& ck = Q4eClock::get();
    const auto t0 = std::chrono::steady_clock::now();
    if (qrw.n_miss[g] > 0) q.lane->wait();
    if (ck.on) ck.lane += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    q4e_done(2 * l + g);
    if (last) q4e_pr_poll(l);
    return true;
}

bool Gpu::Impl::step_q4e(int32_t token, int pos, bool head) {
    Impl& I = *this;
    const Cfg& cc = c;
    Q4e& qq = q;
    cudaStream_t s = stream;
    const int e = cc.n_embd, hc = qq.hc;

    Q4eClock& ck = Q4eClock::get();
    const auto t_wall = std::chrono::steady_clock::now();
    if (ck.on) {
        ck.begin_token();
        rusage ru{};
        getrusage(RUSAGE_SELF, &ru);
        ck.maj = ru.ru_majflt;
        ck.minf = ru.ru_minflt;
    }
    if (pos < 0 || pos >= n_ctx) return false;
    I.qsa_pos = pos;
    if (!I.ensure_dec_scratch() || !I.pages_for(pos, pos + 1)) return false;
    qq.tok_hist[static_cast<size_t>(pos)] = token;

    // The token, the position and this call's doorbell number reach the kernels through device memory.
    const int32_t args[2] = {token, pos};
    const uint32_t sq = ++qq.seq;
    if (cudaMemcpyAsync(d_args, args, sizeof args, cudaMemcpyHostToDevice, s) != cudaSuccess ||
        cudaMemcpyAsync(qq.d_seq, &sq, sizeof sq, cudaMemcpyHostToDevice, s) != cudaSuccess)
        return false;

    // The PLE table's rows are read on a host thread (the hash only needs the token window) into a mapped buffer, and the
    // block's layer waits on the device for its flag.
    bool has_ple = false;
    for (const Layer& y : I.L)
        if (y.ple_key.p) has_ple = true;
    const int pb = int(sq & 1u);
    std::future<void> ple_rows;
    if (has_ple)
        ple_rows = std::async(std::launch::async, [&I, token, pos, pb, sq]() {
            q4e_unpin_this_thread();
            I.q4e_ple_rows(token, pos, I.q.emb_h[pb]);
            __atomic_store_n(I.q.flag_h + Q4e::kPle + pb, sq, __ATOMIC_RELEASE);
        });
    auto fail = [&]() {
        I.q4e_abort();
        return false;
    };

    if (!embed_row(I.embd.type, I.embd.p, d_args, x, e, s)) return false;
    ops::stack_fill(qq.st, x, hc, e, s);
    // Every token ages the routing heat that decides what stays resident (6.4: gamma 0.998, half-life ~350).
    if (!qq.eheat.empty()) I.q4e_cache_decay();

    // Pipelined: layer l is enqueued before the feeder serves layer l - 1, so the device always has the next layer's dense
    // half queued behind the wait it is in, and the launch queue never holds more than about two layers.
    const float inv_hc = 1.0f / float(hc);
    for (int l = 0; l < cc.n_layer_main; ++l) {
        Layer& y = I.L[static_cast<size_t>(l)];
        const auto t_launch = std::chrono::steady_clock::now();
        if (ck.on && l < 64) (void)cudaEventRecord(ck.in[static_cast<size_t>(l)], s);
        // The PLE block of its layer opens layer 1: the stack grows by gated + silu(dilated conv).
        if (y.ple_key.p) {
            if (!I.q4e_wait(Q4e::kPle + pb, qq.emb_hd[pb], qq.emb, qq.n_heads * qq.np)) return fail();
            const int hcd = hc * e;
            if (!gemv(y.ple_key, qq.emb, qq.pkey, hcd, qq.n_heads * qq.np)) return fail();
            if (!gemv(y.ple_value, qq.emb, qq.pval, e, qq.n_heads * qq.np)) return fail();
            ops::ple_gate(qq.pkey, qq.st, qq.pval, y.ple_gk, y.ple_gq, qq.pgated, hc, e, static_cast<float>(cc.eps), s);
            if (!dec::rmsnorm(qq.pgated, y.ple_gc, qq.pnrm, e, static_cast<float>(cc.eps), 1, s, 0, 0, hc)) return fail();
            ops::ple_conv_silu(qq.pnrm, qq.phist, y.ple_conv, qq.pconv, hcd, qq.kern, qq.ngram, s);
            ops::add_into(qq.st, qq.pgated, qq.pconv, hcd, s);
        }

        // The attention half: HC read, mixer, HC write.
        if (!I.q4e_hc_read(y.hca, qq.st)) return fail();
        if (y.full) {
            if (!I.q4e_attn(l, qq.mixed, qq.out)) return fail();
        } else if (!I.q4e_gdn(l, qq.mixed, qq.out)) {
            return fail();
        }
        ops::hc_combine(qq.st, qq.out, qq.inj, hc, e, inv_hc, s);

        // The FFN half: HC read, the MoE's device half, HC write; then the feeder serves the layer before.
        if (!I.q4e_hc_read(y.hcf, qq.st)) return fail();
        if (!I.q4e_moe(l)) return fail();
        ops::hc_combine(qq.st, qq.out, qq.inj, hc, e, inv_hc, s);
        if (ck.on) ck.launch += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_launch).count();
        if (l > 0 && !I.q4e_moe_serve(l - 1)) return fail();
    }

    if (head) {
        if (!I.q4e_hc_read(qq.head_hc, qq.st)) return fail();
        if (!gemv(I.head, qq.mixed, logits, cc.n_vocab, e)) return fail();
    }
    if (!I.q4e_moe_serve(cc.n_layer_main - 1)) return fail();
    if (!I.q4e_end()) return false;
    if (ck.on) {
        (void)cudaStreamSynchronize(s);
        ck.end_token(cc.n_layer_main, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_wall).count(),
                     "step", pos, qq.cache_hits, qq.cache_lookups, qq.miss_ram, qq.miss_file);
    }
    if (qq.pf_n > 0 && ck.on && (++qq.pf_steps & 63) == 0) {
        std::fprintf(stderr, "q4e prefetch (%llu steps): %.2f experts staged a step, %.2f of them used (%.0f%%), %.2f by the PCIe share\n",
                     static_cast<unsigned long long>(qq.pf_steps), double(qq.pf_issued) / double(qq.pf_steps),
                     double(qq.pf_used) / double(qq.pf_steps), 100.0 * double(qq.pf_used) / std::max(1.0, double(qq.pf_issued)),
                     double(qq.dma_n) / double(qq.pf_steps));
    }
    if (Q4ePredict& ps = Q4ePredict::get(); ps.on && ps.pred) {
        const int nl = cc.n_layer_main, nu = cc.n_used;
        std::vector<int32_t> pr(size_t(nl) * 16), ac(size_t(nl) * 16);
        if (cudaStreamSynchronize(s) == cudaSuccess &&
            cudaMemcpy(pr.data(), ps.pred, sizeof(int32_t) * pr.size(), cudaMemcpyDeviceToHost) == cudaSuccess &&
            cudaMemcpy(ac.data(), ps.act, sizeof(int32_t) * ac.size(), cudaMemcpyDeviceToHost) == cudaSuccess) {
            static const int ks[4] = {2, 4, 6, 10};
            for (int l = 1; l < nl; ++l) {
                const int32_t* P = pr.data() + size_t(l) * 16;
                const int32_t* A = ac.data() + size_t(l) * 16;
                const int32_t* so = qq.slab[size_t(l)].slot_of;
                auto picked = [&](int32_t x) { return std::find(A, A + nu, x) != A + nu; };
                for (int i = 0; i < nu; ++i) ps.overlap += picked(P[i]) ? 1 : 0;
                for (int i = 0; i < nu; ++i) ps.misses += so && so[A[i]] >= 0 ? 0 : 1;
                for (int k = 0; k < 4; ++k) {
                    int n = 0;
                    for (int i = 0; i < 16 && n < ks[k]; ++i) {
                        if (so && so[P[i]] >= 0) continue;   // resident: nothing to fetch
                        ++n;
                        ps.useful[k] += picked(P[i]) ? 1 : 0;
                    }
                    ps.taken[k] += n;
                }
            }
            ps.steps += 1;
            if (int64_t(ps.steps) % 32 == 0) {
                const double ln = ps.steps * (nl - 1);
                std::fprintf(stderr,
                             "q4e predict (%.0f steps): next layer's top-%d from this layer's input: %.2f of %d right; misses %.2f a "
                             "layer; the first 2/4/6/10 predicted non-resident: %.2f/%.2f/%.2f/%.2f of them picked (%.0f/%.0f/%.0f/%.0f%%)\n",
                             ps.steps, nu, ps.overlap / ln, nu, ps.misses / ln, ps.useful[0] / ln, ps.useful[1] / ln, ps.useful[2] / ln,
                             ps.useful[3] / ln, 100 * ps.useful[0] / std::max(1.0, ps.taken[0]), 100 * ps.useful[1] / std::max(1.0, ps.taken[1]),
                             100 * ps.useful[2] / std::max(1.0, ps.taken[2]), 100 * ps.useful[3] / std::max(1.0, ps.taken[3]));
            }
        }
    }
    return cudaGetLastError() == cudaSuccess;
}

// ---- the verify window (docs/flash-next.md 6.6a) ------------------------------------------------------------
//
// Why a window exists at all. The dense half of a qwen4exp token is 3.83 GiB of weights; at the measured 216 GB/s
// that is 19.0 ms, which is 52 tok/s at 100 % of the roofline with zero overhead of any kind, and the measured
// dense half alone is 19.7 ms a token (LIMBO_Q4E_TIME, 48 layers, device events). So a SINGLE-TOKEN decode cannot
// pass ~50 tok/s however the rest is tuned -- the slab already took the rest of the token to 49 ms. A window reads
// those weights once for T tokens instead of once per token, and that is the only way past the bar.
//
// Every op below is the _rows sibling of the op step_q4e calls, so row t of a window runs the same arithmetic
// step_q4e would have run for that token. Two things make that structural rather than something to be argued:
// the attention is the SAME dec::attn_prep / dec::attention the single-row step calls (its `1` is the row count,
// which is why no rope table is involved), and the per-stream hyper-connection norms use the same dec::rmsnorm
// with a row pitch in place of the single-row offsets.
//
// write_back is the one deliberate difference. A window's rows are candidates: the conv history and the delta-net
// state are read and left exactly as they were found, and only launch_q4e_commit(n) may move them, by replaying
// the accepted prefix out of the same buffers the window ran.

bool Gpu::Impl::ensure_q4e_rows() {
    if (!qrw.ready) return false;
    const Cfg& cc = c;
    if (!qrw.d_args && !(qrw.d_args = dalloc<int32_t>(4))) return false;
    if (!qrw.Stmp && !(qrw.Stmp = dalloc<float>(size_t(cc.n_v_head) * size_t(cc.gdn_head) * size_t(cc.gdn_head)))) return false;
    if (!qrw.d_tok && !(qrw.d_tok = dalloc<int32_t>(kQ4eRows))) return false;
    if (!qrw.attn_scr && !(qrw.attn_scr = dalloc<float>(dec::attn_scratch_floats(n_ctx, kQ4eRows, cc.n_head))))
        return false;
    if (!qrw.ab && !(qrw.ab = dalloc<float>(size_t(cc.n_layer_main) * size_t(kQ4eRows) * 2 * size_t(cc.dt_rank))))
        return false;
    if (!qrw.qkvk && !(qrw.qkvk = dalloc<float>(size_t(cc.n_layer_main) * size_t(kQ4eRows) * size_t(cc.qkv_w))))
        return false;
    // The MoE half's batch state. The device side is the carve's per-row buffers one row deeper; the host side is
    // pinned, because the ids and the tile come back over PCIe every layer and a pageable D2H makes the driver
    // stage the copy (0.57 ms measured, against 0.1 ms for the stream sync that follows it).
    const size_t npair = size_t(kQ4eRows) * size_t(cc.n_used);
    if (!qrw.d_pair && !(qrw.d_pair = dalloc<int32_t>(npair))) return false;
    if (!qrw.d_pair) return false;
    // The combine's pos_of: `dn` is [T * n_used][n_embd] and a pair's index IS its row of it.
    {
        std::vector<int32_t> ident(npair);
        for (size_t i = 0; i < npair; ++i) ident[i] = static_cast<int32_t>(i);
        if (cudaMemcpy(qrw.d_pair, ident.data(), npair * sizeof(int32_t), cudaMemcpyHostToDevice) != cudaSuccess)
            return false;
    }
    if (!qrw.post && (cudaHostAlloc(reinterpret_cast<void**>(&qrw.post), sizeof(float) * size_t(kQ4eRows) * size_t(cc.n_embd),
                                    cudaHostAllocMapped) != cudaSuccess ||
                      cudaHostGetDevicePointer(reinterpret_cast<void**>(&qrw.post_d), qrw.post, 0) != cudaSuccess))
        return false;
    if (!qrw.miss_row && cudaMallocHost(reinterpret_cast<void**>(&qrw.miss_row), sizeof(int32_t) * npair) != cudaSuccess)
        return false;
    if (!qrw.dn_h && (cudaHostAlloc(reinterpret_cast<void**>(&qrw.dn_h), sizeof(float) * npair * size_t(cc.n_embd), cudaHostAllocMapped) !=
                          cudaSuccess ||
                      cudaHostGetDevicePointer(reinterpret_cast<void**>(&qrw.dn_hd), qrw.dn_h, 0) != cudaSuccess))
        return false;
    if (!qrw.d_am && !(qrw.d_am = dalloc<int32_t>(kQ4eRows))) return false;
    return true;
}

// gemv over T rows. One batched launch where the input width is a whole number of 256-blocks; otherwise one plain
// gemv a row, because gemv_cols rejects n_in % 256 != 0 (its kernels index shared memory by block) and has no tail
// path while plain gemv() does. The geometry guard already requires n_embd % 256 == 0, so this costs nothing on the
// dense bulk and only covers the two narrow widths q4e actually has: hc_rank and the PLE table's row width.
static bool q4e_gemv_rows(uint32_t type, const void* W, const float* x, int x_stride, float* y, int y_stride,
                          int n_out, int n_in, int T, cudaStream_t s) {
    if (limbo::cuda::gemv_cols(type, W, x, x_stride, y, y_stride, n_out, n_in, T, false, s)) return true;
    (void)cudaGetLastError();
    for (int t = 0; t < T; ++t)
        if (!limbo::cuda::gemv(type, W, x + static_cast<size_t>(t) * x_stride, y + static_cast<size_t>(t) * y_stride,
                               n_out, n_in, false, s))
            return false;
    return true;
}

// The hyper-connection read over T rows: the five moves of q4e_hc_read with the stack's row pitch where the
// single-row form had fixed offsets. qrw.st is [T][hc*e], so stream i of row t sits at t*(hc*e) + i*e.
//
// The per-stream norms stay one launch per stream. dec::rmsnorm applies ONE weight vector to every row it is
// given, and q4e's hc streams have distinct weights, so the hc streams cannot share a launch -- that fusion needs
// per-row weight addressing and is a later step. What the window buys here is that each of those hc launches now
// covers all T rows: hc launches a layer where T sequential tokens would have cost hc*T.
bool Gpu::Impl::q4e_hc_read_rows(const Layer::Hc& m, const float* st, int T, int r0) {
    const Cfg& cc = c;
    const int e = cc.n_embd, hc = q.hc, rank = q.rank;
    const float inv_hc = 1.0f / float(hc);
    const float eps = static_cast<float>(cc.eps);
    cudaStream_t s = stream;
    const int pitch = hc * e;
    float *xn = qrw.xn + size_t(r0) * pitch, *lo = qrw.lo + size_t(r0) * rank, *gate = qrw.gate + size_t(r0) * pitch,
          *mixed = qrw.mixed + size_t(r0) * e, *inj = qrw.inj + size_t(r0) * hc;
    if (!dec::rmsnorm(st, m.norm, xn, e, eps, T, s, pitch, pitch, hc)) return false;
    if (!q4e_gemv_rows(m.down.type, m.down.p, xn, pitch, lo, rank, rank, pitch, T, s)) return false;
    ops::silu_scale(lo, lo, T * rank, inv_hc, s);
    if (!q4e_gemv_rows(m.up.type, m.up.p, lo, rank, gate, pitch, pitch, rank, T, s)) return false;
    ops::hc_gate_mix_rows(xn, gate, mixed, hc, e, T, inv_hc, s);
    if (m.inject.p && !limbo::cuda::gemv_narrow(m.inject.type, m.inject.p, xn, pitch, inj, hc, hc, pitch, T, false, s) &&
        !q4e_gemv_rows(m.inject.type, m.inject.p, xn, pitch, inj, hc, hc, pitch, T, s))
        return false;
    return true;
}

// Advances the delta-net state and the conv history by the first n rows of the last write_back = false window.
// Mirrors rung 1's launch_commit (gpu_rows.cu:454): the same conv + recurrence replay, over the rows the window
// already ran, so the arithmetic that lands in the state is the arithmetic that produced the accepted tokens.
//
// No gdn_coef_rows here on purpose: the window already left qrw.alpha / qrw.beta in the coef'd form the rows scan
// consumes, and re-applying the transform would apply it twice.
bool Gpu::Impl::launch_q4e_commit(int n) {
    Impl& I = *this;
    const Cfg& cc = c;
    Q4e& qq = q;
    cudaStream_t s = stream;
    const int qn = cc.n_k_head * cc.gdn_head;
    const int qkn_w = 2 * qn, vn = cc.n_v_head * cc.gdn_head;
    const int pitch = qq.hc * cc.n_embd;
    const float gdn_scale = static_cast<float>(1.0 / std::sqrt(double(cc.gdn_head)));
    const float gdn_eps = static_cast<float>(cc.eps / cc.gdn_head);
    if (!qrw.ready || n < 0 || n > kQ4eRows || n > qrw.pending) return false;
    if (n == 0) return true;
    for (int l = 0; l < cc.n_layer_main; ++l) {
        Layer& y = I.L[static_cast<size_t>(l)];
        // The PLE conv's history is the one piece of state the rows ops advance unconditionally: the window ran
        // on its own copy, and replaying the accepted prefix through the same kernel into the real history is
        // what advances it. Before the full-layer skip, because the PLE layer is not necessarily a GDN one.
        if (y.ple_key.p && qq.phist)
            ops::ple_conv_silu_rows(qrw.pnrm, qq.phist, y.ple_conv, qrw.pconv, pitch, qq.kern, qq.ngram, n, s);
        if (y.full) continue;  // a full layer's state is the K/V cache, which the prep wrote as the window ran
        // This layer's own alpha/beta slot, already coef'd by the window: re-applying the transform here would
        // apply it twice.
        const float* al = qrw.ab + static_cast<size_t>(l) * size_t(kQ4eRows) * 2 * size_t(cc.dt_rank);
        const float* be = al + cc.dt_rank;
        const float* qkvbuf = qrw.qkvk + static_cast<size_t>(l) * size_t(kQ4eRows) * size_t(cc.qkv_w);
        ops::conv_silu_rows(qkvbuf, I.hist[static_cast<size_t>(l)], y.conv, qrw.conv, cc.qkv_w, n, s, true);
        ops::head_norm_rows(qrw.conv, cc.qkv_w, qrw.qk, qkn_w, n, 2 * cc.n_k_head, cc.gdn_head, gdn_eps, gdn_scale, s);
        ops::gdn_scan_rows(I.S[static_cast<size_t>(l)], qrw.qk, qkn_w, qrw.conv + 2 * qn, cc.qkv_w, al, be,
                           2 * cc.dt_rank, qrw.o, vn, n, cc.n_v_head, cc.n_k_head, cc.gdn_head, cc.dt_rank, gdn_scale,
                           s, true);
    }
    return cudaGetLastError() == cudaSuccess;
}

// T rows through all n_layer_main layers, every dense weight read once for all of them. Row t runs the arithmetic
// step_q4e would have run for token t: each move below is the _rows sibling of the move in step_q4e, in the same
// order, with the same constants.
//
// write_back reaches exactly two places, the conv and the recurrence, and it is the whole contract: with false the
// window is a pure read of the state and launch_q4e_commit(n) is the only thing that may move it.
//
// The MoE half is batched too (q4e_moe_rows): the router and the top-k run for all T rows on the device and one
// round trip a layer brings back the ids and the tile, so the window pays one sync a layer where T sequential
// steps paid T -- that was the last thing standing between this and the dense half's own cost.
bool Gpu::Impl::step_q4e_rows(const int32_t* tokens, int T, int pos0, bool write_back) {
    Impl& I = *this;
    const Cfg& cc = c;
    Q4e& qq = q;
    cudaStream_t s = stream;
    const int e = cc.n_embd, hc = qq.hc;
    const int pitch = hc * e;

    // The window's own clock. Without it the only sample LIMBO_Q4E_TIME ever printed was the occasional
    // single-row step, so a decode that runs entirely through windows showed one line and a residency
    // percentage that counted nothing (q4e_moe_rows did not add to cache_hits).
    Q4eClock& ck = Q4eClock::get();
    const auto t_wall = std::chrono::steady_clock::now();
    uint64_t h0 = 0, l0 = 0, r0 = 0, f0 = 0;
    if (ck.on) {
        ck.begin_token();
        rusage ru{};
        getrusage(RUSAGE_SELF, &ru);
        ck.maj = ru.ru_majflt;
        ck.minf = ru.ru_minflt;
        h0 = qq.cache_hits;
        l0 = qq.cache_lookups;
        r0 = qq.miss_ram;
        f0 = qq.miss_file;
    }

    if (!I.ensure_q4e_rows() || T < 1 || T > kQ4eRows) return false;
    if (pos0 < 0 || pos0 + T > n_ctx) return false;
    if (!I.ensure_dec_scratch() || !I.pages_for(pos0, pos0 + T)) return false;
    for (int t = 0; t < T; ++t) qq.tok_hist[static_cast<size_t>(pos0 + t)] = tokens[t];

    // The rows attention reads its position from device memory, exactly as the single-row step does (that is why
    // the same dec::attn_prep serves both: its row count was always a parameter).
    const int32_t args[2] = {tokens[0], pos0};
    const uint32_t sq = ++qq.seq;
    if (cudaMemcpyAsync(qrw.d_args, args, sizeof args, cudaMemcpyHostToDevice, s) != cudaSuccess ||
        cudaMemcpyAsync(qq.d_seq, &sq, sizeof sq, cudaMemcpyHostToDevice, s) != cudaSuccess)
        return false;
    if (cudaMemcpyAsync(qrw.d_tok, tokens, sizeof(int32_t) * size_t(T), cudaMemcpyHostToDevice, s) != cudaSuccess)
        return false;

    // The PLE table's rows are read on the host (the hash only needs the token window).
    bool has_ple = false;
    for (const Layer& y : I.L)
        if (y.ple_key.p) has_ple = true;
    const int pb = int(sq & 1u);
    std::future<void> ple_rows;
    if (has_ple) {
        // on a host thread into a mapped buffer, as the single step does; the block's layer waits for the flag
        std::array<int32_t, kQ4eRows> tk{};
        for (int t = 0; t < T; ++t) tk[size_t(t)] = tokens[t];
        ple_rows = std::async(std::launch::async, [&I, tk, T, pos0, pb, sq]() {
            const int ew = I.q.n_heads * I.q.np;
            for (int t = 0; t < T; ++t) I.q4e_ple_rows(tk[size_t(t)], pos0 + t, I.q.emb_h[pb] + size_t(t) * size_t(ew));
            __atomic_store_n(I.q.flag_h + Q4e::kPle + pb, sq, __ATOMIC_RELEASE);
        });
    }

    if (!embed_rows(I.embd.type, I.embd.p, qrw.d_tok, T, qrw.x, e, s)) return false;
    for (int t = 0; t < T; ++t) ops::stack_fill(qrw.st + size_t(t) * pitch, qrw.x + size_t(t) * e, hc, e, s);
    // The PLE conv's history is one row's worth shared across the window's rows. The window runs on its own copy
    // so that a rejected row cannot move what a single token left behind -- and it must be SEEDED from the real
    // one, or the window's PLE conv reads whatever the carve left in that buffer. launch_q4e_commit advances the
    // real history by the accepted prefix.
    if (has_ple && qq.phist) {
        const size_t ph_n = size_t(qq.kern - 1) * size_t(qq.ngram) * size_t(pitch);
        if (cudaMemcpyAsync(qrw.phist, qq.phist, ph_n * sizeof(float), cudaMemcpyDeviceToDevice, s) != cudaSuccess)
            return false;
    }
    // Every token ages the routing heat that decides what stays resident (6.4: gamma 0.998, half-life ~350). One
    // decay a row, as T sequential steps would each have applied: the window's rows are tokens, and this is the
    // only place they are counted as anything else.
    if (!qq.eheat.empty())
        for (int t = 0; t < T; ++t) I.q4e_cache_decay();

    auto fail = [&]() {
        I.q4e_abort();
        return false;
    };
    // Two groups when the window is long enough: the device runs one group's dense half while the lane computes the
    // other's misses (the dense weights are read twice, which the lane's time pays for from split_min rows on).
    const int nl = cc.n_layer_main;
    const bool split = qq.split_min > 0 && T >= qq.split_min;
    const int a = split ? (T + 1) / 2 : T;
    if (split) {
        const int32_t a2[2] = {tokens[a], pos0 + a};
        if (cudaMemcpyAsync(qrw.d_args + 2, a2, sizeof a2, cudaMemcpyHostToDevice, s) != cudaSuccess) return fail();
    }
    // Each lane batch is submitted before the host queues the device's next work and joined after it, so the
    // launches never sit on the lane's path.
    if (!split) {
        for (int l = 0; l < nl; ++l) {
            if (l > 0 && !I.q4e_rows_submit(l - 1, 0, T, 0, true)) return fail();
            if (!I.q4e_rows_front(l, 0, T, 0, T, pos0, write_back, false, pb) || !I.q4e_rows_back(l, 0, T, 0)) return fail();
            if (l > 0 && !I.q4e_rows_finish(l - 1, 0, true)) return fail();
        }
        if (!I.q4e_rows_submit(nl - 1, 0, T, 0, true)) return fail();
    } else {
        // device: F(0,A) F(0,B) | B(l-1,A) F(l,A) | B(l-1,B) F(l,B) | ... ; the lane serves (l-1, A) while the device
        // runs B's front, and (l-1, B) while it runs A's back and next front
        if (!I.q4e_rows_front(0, 0, a, 0, T, pos0, write_back, true, pb) || !I.q4e_rows_front(0, a, T - a, 1, T, pos0, write_back, true, pb))
            return fail();
        for (int l = 1; l < nl; ++l) {
            if (!I.q4e_rows_submit(l - 1, 0, a, 0, false) || !I.q4e_rows_back(l - 1, 0, a, 0) ||
                !I.q4e_rows_front(l, 0, a, 0, T, pos0, write_back, true, pb) || !I.q4e_rows_finish(l - 1, 0, false))
                return fail();
            if (!I.q4e_rows_submit(l - 1, a, T - a, 1, true) || !I.q4e_rows_back(l - 1, a, T - a, 1) ||
                !I.q4e_rows_front(l, a, T - a, 1, T, pos0, write_back, true, pb) || !I.q4e_rows_finish(l - 1, 1, true))
                return fail();
        }
        if (!I.q4e_rows_submit(nl - 1, 0, a, 0, false) || !I.q4e_rows_back(nl - 1, 0, a, 0) || !I.q4e_rows_finish(nl - 1, 0, false) ||
            !I.q4e_rows_submit(nl - 1, a, T - a, 1, true))
            return fail();
        if (!I.q4e_rows_back(nl - 1, a, T - a, 1)) return fail();
    }

    // The window decides acceptance from all T rows, so the head is always computed (the single-row step makes it
    // optional because only the last token of a speculative pass needs it).
    if (!I.q4e_hc_read_rows(qq.head_hc, qrw.st, T)) return fail();
    if (!q4e_gemv_rows(I.head.type, I.head.p, qrw.mixed, e, qrw.logits, cc.n_vocab, cc.n_vocab, e, T, s)) return fail();
    if (ck.on) {
        (void)cudaEventRecord(ck.fin, s);
        ck.has_fin = true;
    }
    if (!I.q4e_rows_finish(nl - 1, split ? 1 : 0, true)) return fail();
    if (!I.q4e_end()) return false;

    // A window that IS the accepted sequence (write_back) advanced its own copy of the PLE conv history; hand it
    // back. ple_conv_silu_rows has no write_back -- it always rolls the history it is given -- so the window's
    // copy is ahead by T rows and the real one is what the next single step will read.
    if (write_back && has_ple && qq.phist) {
        const size_t ph_n = size_t(qq.kern - 1) * size_t(qq.ngram) * size_t(pitch);
        if (cudaMemcpyAsync(qq.phist, qrw.phist, ph_n * sizeof(float), cudaMemcpyDeviceToDevice, s) != cudaSuccess)
            return false;
    }

    // What commit may take a prefix of: a window that moved the state itself owes nothing.
    qrw.pos0 = pos0;
    qrw.last_T = T;
    qrw.pending = write_back ? 0 : T;
    if (ck.on) {
        (void)cudaStreamSynchronize(s);
        char what[8];
        std::snprintf(what, sizeof what, "w%d", T);
        ck.end_token(cc.n_layer_main,
                     std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_wall).count(), what,
                     pos0, qq.cache_hits - h0, qq.cache_lookups - l0, qq.miss_ram - r0, qq.miss_file - f0);
    }
    return cudaGetLastError() == cudaSuccess;
}

// The public face of the window (include/limbo/cuda/gpu.hpp). The window's rows live in its own workspace, which
// is why these are separate from step_rows / commit / read_logits_rows rather than a mode of them.
bool Gpu::q4e_step_rows(const int32_t* tokens, int T, int pos0, bool write_back) {
    Impl& I = *p_;
    if (!tokens || T < 1 || T > Impl::kQ4eRows || pos0 < 0 || pos0 + T > I.n_ctx) return false;
    // A plain row (the speculation's): the single step, its last residual and logits where the window's would be (the MTP
    // block reads the residual, argmax_rows / read_logits_rows the logits).
    if (T == 1 && write_back && I.qrw.ready) {
        if (!I.step_q4e(tokens[0], pos0, true)) return false;
        const size_t pitch = size_t(I.q.hc) * size_t(I.c.n_embd);
        if (cudaMemcpyAsync(I.qrw.st, I.q.st, sizeof(float) * pitch, cudaMemcpyDeviceToDevice, I.stream) != cudaSuccess ||
            cudaMemcpyAsync(I.qrw.logits, I.logits, sizeof(float) * size_t(I.c.n_vocab), cudaMemcpyDeviceToDevice, I.stream) != cudaSuccess)
            return false;
        I.qrw.pos0 = pos0;
        I.qrw.last_T = 1;
        I.qrw.pending = 0;
        return true;
    }
    return I.step_q4e_rows(tokens, T, pos0, write_back);
}

bool Gpu::q4e_commit(int n) {
    Impl& I = *p_;
    if (n < 0 || n > I.qrw.pending) return false;
    if (n == 0) {
        I.qrw.pending = 0;
        return true;
    }
    if (!I.launch_q4e_commit(n)) return false;
    I.qrw.pending = 0;
    return true;
}

bool Gpu::q4e_read_logits_rows(float* out, int T) {
    Impl& I = *p_;
    if (!I.qrw.ready || !out || T < 1 || T > Impl::kQ4eRows) return false;
    if (cudaStreamSynchronize(I.stream) != cudaSuccess) return false;
    return cudaMemcpy(out, I.qrw.logits, size_t(T) * size_t(I.c.n_vocab) * sizeof(float), cudaMemcpyDeviceToHost) ==
           cudaSuccess;
}

// The rows' greedy tokens. read_logits_rows' sibling for the greedy path: it is the same reduction argmax_rows runs
// on rung 1's window, over this window's own logits, and it stages through device memory for the same reason.
bool Gpu::q4e_argmax_rows(int32_t* out, int T) {
    Impl& I = *p_;
    if (!I.qrw.ready || !out || T < 1 || T > Impl::kQ4eRows) return false;
    if (!I.qrw.d_am && !(I.qrw.d_am = I.dalloc<int32_t>(Impl::kQ4eRows))) return false;
    ops::argmax_rows(I.qrw.logits, I.c.n_vocab, T, I.qrw.d_am, I.stream);
    if (cudaMemcpyAsync(out, I.qrw.d_am, size_t(T) * sizeof(int32_t), cudaMemcpyDeviceToHost, I.stream) != cudaSuccess)
        return false;
    return cudaStreamSynchronize(I.stream) == cudaSuccess;
}

bool Gpu::is_q4e() const { return p_->q4e; }

bool Gpu::mtp_front(const int32_t* next_tokens, int n, int first_row, int q0) {
    Impl& I = *p_;
    if (!I.q4e || !I.q4mtp.ready || n < 1 || first_row < 0 || first_row + n > I.qrw.last_T) return false;
    return I.q4e_mtp_front(I.qrw.st + size_t(first_row) * size_t(I.q.hc) * size_t(I.c.n_embd), next_tokens, n, q0);
}

int Gpu::q4e_rows_end() const { return p_->qrw.pos0 + p_->qrw.last_T; }
int Gpu::q4e_rows_last() const { return p_->qrw.last_T; }

// The same guard step_q4e_rows applies, solved for T: it refuses when (pos0 + T) / idx_kpool exceeds the selection's
// budget, so the longest window it takes is the largest T with pos0 + T <= n_top * kpool + kpool - 1. Below that
// budget the visible set IS the plain causal one (step.cpp qsa_select returns false exactly when nv <= top_k /
// kpool) and the window is exact; above it there is no window at all and the caller takes single rows.
int Gpu::q4e_window_rows(int pos0) const {
    const Impl& I = *p_;
    if (!I.qrw.ready || pos0 < 0 || pos0 >= I.n_ctx) return 0;
    const int T = std::min(Impl::kQ4eRows, I.n_ctx - pos0);   // QSA runs on the device at every length
    return T > 0 ? T : 0;
}


// ---- QSA (docs/flash-next.md 2.3) on the device ------------------------------------------------------------
bool Gpu::Impl::ensure_qsa() {
    if (qsa.ready) return true;
    const Cfg& cc = c;
    const int kp = q.idx_kpool, n_top = std::max(1, q.idx_top_k / std::max(1, kp));
    if (kp <= 0) return false;
    auto al = [&](size_t bytes) -> void* {
        void* p = nullptr;
        if (cudaMalloc(&p, bytes) != cudaSuccess) return nullptr;
        qsa.allocs.push_back(p);
        return p;
    };
    const size_t nb = size_t(n_ctx) / size_t(kp) + 1;
    qsa.ring.assign(size_t(cc.n_layer_main), nullptr);
    qsa.pool.assign(size_t(cc.n_layer_main), nullptr);
    for (int l = 0; l < cc.n_layer_main; ++l) {
        if (!L[size_t(l)].iq.p) continue;
        qsa.ring[size_t(l)] = static_cast<__half*>(al(16 * 128 * sizeof(__half)));
        qsa.pool[size_t(l)] = static_cast<__half*>(al(nb * 128 * sizeof(__half)));
        if (!qsa.ring[size_t(l)] || !qsa.pool[size_t(l)]) return false;
    }
    const size_t R = size_t(kQ4eRows);
    qsa.stride = nb;
    qsa.kraw = static_cast<float*>(al(R * 128 * sizeof(float)));
    qsa.qraw = static_cast<float*>(al(R * 4 * 128 * sizeof(float)));
    qsa.qn = static_cast<float*>(al(R * 4 * 128 * sizeof(float)));
    qsa.scores = static_cast<float*>(al(R * nb * sizeof(float)));
    qsa.sel = static_cast<int32_t*>(al(R * size_t(n_top) * sizeof(int32_t)));
    qsa.scr = static_cast<float*>(al(qsa::attn_scratch_floats(kQ4eRows, cc.n_head, n_top, kp) * sizeof(float)));
    if (!qsa.kraw || !qsa.qraw || !qsa.qn || !qsa.scores || !qsa.sel || !qsa.scr) return false;
    qsa.ready = true;
    return true;
}

void Gpu::Impl::q4e_prep(int l, const float* qfull, const float* kbuf, const float* vsrc, const float* qn, const float* kn, float* qbuf,
                         const int* pos_dev, int T) {
    const Cfg& cc = c;
    const size_t L = size_t(l);
    if (kv.fmt == KvFormat::Q8)
        dec::attn_prep_q8(qfull, kbuf, vsrc, qn, kn, qbuf, kv.kq[L], kv.ks[L], kv.vq[L], kv.vs[L], pos_dev, T, cc.eps, cc.rope_base,
                          cc.rope_dim, cc.n_head, cc.n_head_kv, cc.hd, stream, pt());
    else
        dec::attn_prep(qfull, kbuf, vsrc, qn, kn, qbuf, kc[L], vc[L], pos_dev, T, 0, cc.eps, cc.rope_base, cc.rope_dim, cc.n_head,
                       cc.n_head_kv, cc.hd, stream, pt());
}

bool Gpu::Impl::q4e_attend(int l, const float* qbuf, const float* qfull, float* attn, float* scr, const int* pos_dev, int T) {
    const Cfg& cc = c;
    const size_t L = size_t(l);
    const float scale = static_cast<float>(1.0 / std::sqrt(double(cc.hd)));
    if (kv.fmt == KvFormat::Q8)
        return dec::attention_q8(qbuf, kv.kq[L], kv.ks[L], kv.vq[L], kv.vs[L], qfull, attn, scr, pos_dev, T, n_ctx, cc.n_head,
                                 cc.n_head_kv, cc.hd, scale, stream, pt());
    return dec::attention(qbuf, kc[L], vc[L], qfull, attn, scr, pos_dev, T, n_ctx, 0, cc.n_head, cc.n_head_kv, cc.hd, scale, stream,
                          pt());
}

bool Gpu::Impl::q4e_qsa(int l, const float* mixed, int x_stride, int T, int pos0, const int* pos_dev, const float* qbuf,
                        const float* qfull, float* attn, bool* sparse) {
    *sparse = false;
    const Layer& y = L[size_t(l)];
    const Cfg& cc = c;
    const int kp = q.idx_kpool;
    if (kp <= 0) return true;
    if (!ensure_qsa()) return false;
    cudaStream_t s = stream;
    const int heads = q.idx_heads;
    if (!q4e_gemv_rows(y.ik.type, y.ik.p, mixed, x_stride, qsa.kraw, 128, 128, cc.n_embd, T, s) ||
        !q4e_gemv_rows(y.iq.type, y.iq.p, mixed, x_stride, qsa.qraw, heads * 128, heads * 128, cc.n_embd, T, s))
        return false;
    if (!qsa::index_rows(qsa.kraw, qsa.qraw, y.ikn, y.iqn, qsa.ring[size_t(l)], qsa.pool[size_t(l)], qsa.qn, pos_dev, T,
                         kp, heads, cc.rope_dim, cc.rope_base, s))
        return false;
    const int n_top = std::max(1, q.idx_top_k / kp);
    const int nv_max = (pos0 + T) / kp;
    if (nv_max <= n_top) return true;
    if (!qsa::select_rows(qsa.qn, qsa.pool[size_t(l)], qsa.scores, qsa.stride, qsa.sel, pos_dev, T, nv_max, kp, heads,
                          n_top, s))
        return false;
    const float scale = float(1.0 / std::sqrt(double(cc.hd)));
    const bool q8 = kv.fmt == KvFormat::Q8;
    if (!qsa::attention_rows(qbuf, q8 ? nullptr : kc[size_t(l)], q8 ? nullptr : vc[size_t(l)], qfull, attn, qsa.scr, qsa.sel, pos_dev,
                             pt(), 0, T, kp, n_top, cc.n_head, cc.n_head_kv, scale, s, q8 ? kv.kq[size_t(l)] : nullptr,
                             q8 ? kv.ks[size_t(l)] : nullptr, q8 ? kv.vq[size_t(l)] : nullptr, q8 ? kv.vs[size_t(l)] : nullptr))
        return false;
    *sparse = true;
    return true;
}

}  // namespace limbo::cuda
