// Copyright (c) 2026 ITERFORM. All rights reserved.
// src/cpu_lane_pool.cpp — the CPU lane's pool and layer scheduler.
//
// The pool (docs/donors/strata.md §5 "CPU pool", kindi.md §7 "threads", modchip.md §4):
//   - one worker per PHYSICAL core, never an SMT sibling; the first core stays free for the GPU feeder thread and
//     the last core of the other L3 domain for the OS and the desktop, so a busy pool cannot convoy either;
//   - a batch is a flat list of tasks claimed from ONE atomic 64-bit word: [epoch:24 | ntasks:20 | next:20]. A
//     claim is a CAS on the whole word, so a worker that wakes late can only ever claim from the batch that is
//     current when its CAS lands, never a stale one, and a batch is a barrier by construction (the dispatcher
//     returns when the done counter reaches ntasks);
//   - idle workers spin for `spin_us` then park on a futex; the publisher bumps a futex word and wakes only if a
//     worker is actually parked (Dekker with seq_cst on the sleepers counter and the control word, so a wake
//     cannot be lost);
//   - no std::mutex, no condition_variable, no lock_guard, no std::atomic<std::shared_ptr>. Every atomic here is
//     a plain integer (is_lock_free is static_asserted).
//
// The layer scheduler (strata.md §5 "activation quantized once per layer", kindi.md §7 "union batching"):
//   stage Q  one task per input row: quantize the layer's normed input into int8 planes, once, for every expert
//   stage A  (expert, 16 rows of gate+up): both matrices in one pass over the shared columns, SwiGLU into h;
//            the thread that finishes an expert's LAST A task quantizes that expert's h (a few microseconds) and
//            raises its ready flag
//   stage B  (expert, 32 rows of down): waits for its expert's flag, writes the unweighted FFN output
// The tasks are claimed in that order, so an A task only ever waits for Q tasks already running and a B task
// only for A tasks already running: no wait can deadlock, and no barrier sits between the stages.
#include <linux/futex.h>
#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <thread>

#include "limbo/cpu_lane.hpp"
#include "limbo/types.hpp"

namespace limbo::cpu {

namespace {

inline void cpu_relax() { __builtin_ia32_pause(); }

long futex(void* addr, int op, uint32_t val) {
    return syscall(SYS_futex, addr, op | FUTEX_PRIVATE_FLAG, val, nullptr, nullptr, 0);
}

double now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct AlignedBuf {
    void* p = nullptr;
    size_t cap = 0;
    ~AlignedBuf() { std::free(p); }
    void* ensure(size_t n) {
        if (n > cap) {
            std::free(p);
            cap = (n + 4095) & ~size_t(4095);
            p = std::aligned_alloc(64, cap);
            if (!p) std::abort();
        }
        return p;
    }
};

// ---- topology -------------------------------------------------------------------------------------------
std::string slurp(const std::string& path) {
    std::ifstream in(path);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

int first_cpu_of(const std::string& list) {   // "0,8" "0-3,8-11" -> 0
    return list.empty() ? -1 : std::atoi(list.c_str());
}

}  // namespace

std::vector<int> default_cpus(int max_workers, bool reserve_os) {
    const int n = int(std::thread::hardware_concurrency());
    // physical core = lowest cpu among SMT siblings; L3 domain = lowest cpu sharing the L3
    std::map<int, std::vector<int>> domains;   // L3 id -> physical cores (lowest sibling), ascending
    std::vector<int> seen;
    for (int c = 0; c < n; ++c) {
        const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(c);
        const int sib = first_cpu_of(slurp(base + "/topology/thread_siblings_list"));
        if (sib != c) continue;   // an SMT sibling of a lower cpu
        int l3 = first_cpu_of(slurp(base + "/cache/index3/shared_cpu_list"));
        if (l3 < 0) l3 = 0;
        domains[l3].push_back(c);
    }
    if (domains.empty()) {   // no sysfs: assume half the logical cpus are physical
        for (int c = 0; c < std::max(1, n / 2); ++c) domains[0].push_back(c);
    }
    // reserve the first core overall (GPU feeder) and the last core of the last domain (OS + desktop)
    std::vector<std::vector<int>> d;
    for (auto& kv : domains) d.push_back(kv.second);
    if (!d.empty() && !d.front().empty()) d.front().erase(d.front().begin());
    if (reserve_os) {
        if (d.size() > 1 && !d.back().empty()) d.back().pop_back();
        else if (!d.back().empty() && d.back().size() > 1) d.back().pop_back();
    }
    // interleave the domains so any prefix is balanced across L3s
    std::vector<int> out;
    for (size_t i = 0;; ++i) {
        bool any = false;
        for (auto& dom : d)
            if (i < dom.size()) {
                out.push_back(dom[i]);
                any = true;
            }
        if (!any) break;
    }
    if (max_workers > 0 && int(out.size()) > max_workers) out.resize(size_t(max_workers));
    return out;
}

// ---- pool ------------------------------------------------------------------------------------------------
class Pool {
public:
    using Fn = void (*)(void* ctx, int task, int worker);

    Pool(const PoolConfig& cfg, std::vector<int> cpus) : cpus_(std::move(cpus)), pin_(cfg.pin) {
        spin_us_.store(cfg.spin_us, std::memory_order_relaxed);
        threads_.reserve(cpus_.size());
        for (size_t i = 0; i < cpus_.size(); ++i) threads_.emplace_back([this, i] { worker_main(int(i)); });
    }
    ~Pool() {
        stop_.store(1, std::memory_order_seq_cst);
        wake_.fetch_add(1, std::memory_order_seq_cst);
        futex(&wake_, FUTEX_WAKE, 0x7fffffff);
        for (auto& t : threads_) t.join();
    }
    int workers() const { return int(cpus_.size()); }
    void set_spin_us(int us) { spin_us_.store(us, std::memory_order_relaxed); }

    // One batch in flight. fn(ctx, task, worker) for task in [0, n).
    void start(Fn fn, void* ctx, int n) {
        fn_ = fn;
        ctx_ = ctx;
        total_ = uint32_t(n);
        done_.store(0, std::memory_order_relaxed);
        waiter_.store(0, std::memory_order_relaxed);
        epoch_ = (epoch_ + 1) & 0xFFFFFFu;
        batches_.fetch_add(1, std::memory_order_relaxed);
        tasks_.fetch_add(uint64_t(n), std::memory_order_relaxed);
        ctl_.store((uint64_t(epoch_) << 40) | (uint64_t(n) << 20), std::memory_order_seq_cst);
        wake_.fetch_add(1, std::memory_order_seq_cst);
        if (sleepers_.load(std::memory_order_seq_cst) > 0) futex(&wake_, FUTEX_WAKE, 0x7fffffff);
    }
    bool done() const { return done_.load(std::memory_order_acquire) == total_; }
    // The caller claims tasks of the batch in flight like a worker (id workers()), so the thread that waits for the
    // lane computes in it instead of holding a core idle. A task only ever waits on tasks claimed before it, so this
    // cannot wait on itself.
    void help() {
        uint64_t w = ctl_.load(std::memory_order_acquire);
        for (;;) {
            const uint32_t next = uint32_t(w & 0xFFFFF), n = uint32_t((w >> 20) & 0xFFFFF);
            if (next >= n) return;
            if (ctl_.compare_exchange_weak(w, w + 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
                fn_(ctx_, int(next), workers());
                done_.fetch_add(1, std::memory_order_seq_cst);
                w = ctl_.load(std::memory_order_acquire);
            }
        }
    }
    void wait() {
        help();
        const double t0 = now_us();
        const int spin = spin_us_.load(std::memory_order_relaxed);
        while (done_.load(std::memory_order_acquire) != total_) {
            for (int i = 0; i < 64 && done_.load(std::memory_order_relaxed) != total_; ++i) cpu_relax();
            if (now_us() - t0 > double(std::max(spin, 200))) {
                waiter_.store(1, std::memory_order_seq_cst);
                const uint32_t v = done_.load(std::memory_order_seq_cst);
                if (v != total_) futex(&done_, FUTEX_WAIT, v);
                waiter_.store(0, std::memory_order_relaxed);
            }
        }
    }
    uint64_t batches() const { return batches_.load(std::memory_order_relaxed); }
    uint64_t tasks() const { return tasks_.load(std::memory_order_relaxed); }
    uint64_t parks() const { return parks_.load(std::memory_order_relaxed); }

private:
    static_assert(std::atomic<uint64_t>::is_always_lock_free && std::atomic<uint32_t>::is_always_lock_free,
                  "the pool's words must be lock-free");

    void worker_main(int wid) {
        if (pin_) {
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(cpus_[size_t(wid)], &set);
            pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
        }
        uint32_t seen = 0;   // the epoch this worker has already drained
        double idle_since = now_us();
        for (;;) {
            uint64_t w = ctl_.load(std::memory_order_acquire);
            if (uint32_t(w >> 40) != seen) {
                // a batch this worker has not drained: claim until nothing is left
                for (;;) {
                    const uint32_t next = uint32_t(w & 0xFFFFF), n = uint32_t((w >> 20) & 0xFFFFF);
                    if (next >= n) break;
                    if (ctl_.compare_exchange_weak(w, w + 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
                        fn_(ctx_, int(next), wid);   // fn_/ctx_ are stable from a won claim until done_ reaches total_
                        const uint32_t d = done_.fetch_add(1, std::memory_order_seq_cst) + 1;
                        if (d == total_ && waiter_.load(std::memory_order_seq_cst)) futex(&done_, FUTEX_WAKE, 1);
                        w = ctl_.load(std::memory_order_acquire);
                    }
                }
                seen = uint32_t(w >> 40);
                idle_since = now_us();
                continue;
            }
            if (stop_.load(std::memory_order_acquire)) return;
            // idle: spin, then park
            const int spin = spin_us_.load(std::memory_order_relaxed);
            if (now_us() - idle_since < double(spin)) {
                for (int i = 0; i < 32; ++i) cpu_relax();
                continue;
            }
            const uint32_t wv = wake_.load(std::memory_order_seq_cst);
            sleepers_.fetch_add(1, std::memory_order_seq_cst);
            if (uint32_t(ctl_.load(std::memory_order_seq_cst) >> 40) == seen && !stop_.load(std::memory_order_seq_cst)) {
                parks_.fetch_add(1, std::memory_order_relaxed);
                futex(&wake_, FUTEX_WAIT, wv);
            }
            sleepers_.fetch_sub(1, std::memory_order_seq_cst);
            idle_since = now_us();
        }
    }

    std::vector<int> cpus_;
    bool pin_;
    std::vector<std::thread> threads_;
    alignas(64) std::atomic<uint64_t> ctl_{0};
    alignas(64) std::atomic<uint32_t> done_{0};
    std::atomic<uint32_t> waiter_{0};
    alignas(64) std::atomic<uint32_t> wake_{0};
    std::atomic<uint32_t> sleepers_{0};
    std::atomic<uint32_t> stop_{0};
    std::atomic<int> spin_us_{150};
    std::atomic<uint64_t> batches_{0}, tasks_{0}, parks_{0};
    Fn fn_ = nullptr;
    void* ctx_ = nullptr;
    uint32_t total_ = 0;
    uint32_t epoch_ = 0;
};

// ---- the lane --------------------------------------------------------------------------------------------
namespace {

struct Task {
    uint8_t stage;   // 0 quantize, 1 gate+up, 2 down
    uint8_t pad;
    uint16_t job;
    int32_t r0, r1;  // stage 0: r0 = the input row; stages 1, 2: the weight rows [r0, r1)
};

struct alignas(64) JobState {
    std::atomic<int32_t> a_left{0};
    std::atomic<int32_t> h_ready{0};
    int cols[kMaxCols];   // activation columns this job's rows read in the input tile
    AlignedBuf h;         // float [nrows][n_ff]
    AlignedBuf hq;        // the quantized h
    ActQ hact;
};

constexpr int kScratchRows = 64;

struct Scratch {
    AlignedBuf g, u;   // [kMaxCols][kScratchRows] floats each
};

constexpr std::array<int, kMaxCols> make_ident() {
    std::array<int, kMaxCols> a{};
    for (int i = 0; i < kMaxCols; ++i) a[size_t(i)] = i;
    return a;
}
constexpr std::array<int, kMaxCols> kIdent = make_ident();

}  // namespace

struct Lane::Impl {
    Config cfg;
    Isa isa = Isa::Scalar;
    int planes = 3;
    std::unique_ptr<Pool> pool;
    std::vector<Scratch> scratch;
    std::unique_ptr<JobState[]> js;
    std::vector<Task> tasks;
    AlignedBuf xbuf[2];   // the shared input, quantized for blk 256 and blk 32
    ActQ xact[2];
    int xK = 0;
    unsigned xkinds = 0;   // bit0: blk 256 in use, bit1: blk 32 in use
    std::atomic<int32_t> q_done{0};
    int n_q = 0;
    // the batch in flight
    const float* x = nullptr;
    int T = 0;
    const ExpertJob* jobs = nullptr;
    int njobs = 0;

    static void task_entry(void* ctx, int idx, int worker) { static_cast<Impl*>(ctx)->run_task(idx, worker); }
    void run_task(int idx, int worker);
    const ActQ& xq_for(uint32_t type) const { return xact[act_block(type) == 256 ? 0 : 1]; }
};

namespace {
inline float silu_f(float x) { return x / (1.0f + std::exp(-x)); }
}

void Lane::Impl::run_task(int idx, int worker) {
    const Task t = tasks[size_t(idx)];
    Scratch& S = scratch[size_t(worker)];
    const int P = planes;
    if (t.stage == 0) {
        const int row = t.r0;
        for (int k = 0; k < 2; ++k)
            if (xkinds & (1u << k)) actq_quantize_row(isa, x + size_t(row) * size_t(xK), row, P, &xact[k]);
        q_done.fetch_add(1, std::memory_order_release);
        return;
    }
    const ExpertJob& j = jobs[t.job];
    JobState& st = js[t.job];
    const int R = j.nrows;
    const int ncol = R * P;
    const int nf = j.w.n_ff, ne = j.w.n_embd;
    if (t.stage == 1) {
        while (q_done.load(std::memory_order_acquire) < n_q) cpu_relax();
        const int rows = t.r1 - t.r0;
        float* yg = static_cast<float*>(S.g.ensure(sizeof(float) * kMaxCols * kScratchRows));
        float* yu = static_cast<float*>(S.u.ensure(sizeof(float) * kMaxCols * kScratchRows));
        const uint64_t grow = type_bytes(j.w.gate_type, uint64_t(ne)), urow = type_bytes(j.w.up_type, uint64_t(ne));
        gemv_cols(isa, j.w.gate_type, j.w.gate + uint64_t(t.r0) * grow, grow, rows, xq_for(j.w.gate_type), st.cols, ncol, yg, kScratchRows);
        gemv_cols(isa, j.w.up_type, j.w.up + uint64_t(t.r0) * urow, urow, rows, xq_for(j.w.up_type), st.cols, ncol, yu, kScratchRows);
        float* h = static_cast<float*>(st.h.p);
        for (int i = 0; i < R; ++i) {
            for (int r = 0; r < rows; ++r) {
                float g = 0.0f, u = 0.0f;
                for (int p = 0; p < P; ++p) {
                    g += yg[size_t(i * P + p) * kScratchRows + size_t(r)];
                    u += yu[size_t(i * P + p) * kScratchRows + size_t(r)];
                }
                h[size_t(i) * size_t(nf) + size_t(t.r0 + r)] = silu_f(g) * u;
            }
        }
        if (st.a_left.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            actq_quantize(isa, h, R, P, &st.hact);
            st.h_ready.store(1, std::memory_order_release);
        }
        return;
    }
    // stage 2: down rows [r0, r1)
    while (st.h_ready.load(std::memory_order_acquire) == 0) cpu_relax();
    const int rows = t.r1 - t.r0;
    float* y = static_cast<float*>(S.g.ensure(sizeof(float) * kMaxCols * kScratchRows));
    const uint64_t drow = type_bytes(j.w.down_type, uint64_t(nf));
    gemv_cols(isa, j.w.down_type, j.w.down + uint64_t(t.r0) * drow, drow, rows, st.hact, kIdent.data(), ncol, y, kScratchRows);
    for (int i = 0; i < R; ++i)
        for (int r = 0; r < rows; ++r) {
            float s = 0.0f;
            for (int p = P - 1; p >= 0; --p) s += y[size_t(i * P + p) * kScratchRows + size_t(r)];   // smallest plane first
            j.out[size_t(i) * size_t(ne) + size_t(t.r0 + r)] = s;
        }
}

Lane::Lane() = default;
Lane::~Lane() = default;

std::unique_ptr<Lane> Lane::create(const Config& cfg, std::string* err) {
    auto fail = [&](const char* m) -> std::unique_ptr<Lane> {
        if (err) *err = m;
        return nullptr;
    };
    if (cfg.planes < 1 || cfg.planes > kMaxPlanes) return fail("planes must be 1..3");
    if (cfg.max_tokens < 1 || cfg.max_tokens > kMaxRows) return fail("max_tokens must be 1..16");
    if (cfg.slice_a < 1 || cfg.slice_a > kScratchRows || cfg.slice_b < 1 || cfg.slice_b > kScratchRows) return fail("slice rows must be 1..64");
    if (cfg.max_jobs < 1 || cfg.max_jobs > 4096) return fail("max_jobs must be 1..4096");
    std::unique_ptr<Lane> l(new Lane());
    l->p_ = std::make_unique<Impl>();
    Impl& p = *l->p_;
    p.cfg = cfg;
    p.isa = cfg.isa == Isa::Scalar ? detect_isa() : cfg.isa;
    p.planes = cfg.planes;
    std::vector<int> cpus = cfg.pool.cpus.empty() ? default_cpus() : cfg.pool.cpus;
    if (cpus.empty()) return fail("no cpus for the pool");
    p.pool = std::make_unique<Pool>(cfg.pool, cpus);
    p.scratch.resize(cpus.size() + 1);   // + the caller, which helps in wait()
    p.js.reset(new JobState[size_t(cfg.max_jobs)]);
    p.tasks.reserve(size_t(cfg.max_jobs) * 256 + 64);
    return l;
}

int Lane::workers() const { return p_->pool->workers(); }
Isa Lane::isa() const { return p_->isa; }
int Lane::planes() const { return p_->planes; }
void Lane::set_spin_us(int us) { p_->pool->set_spin_us(us); }
LaneStats Lane::stats() const {
    LaneStats s;
    s.batches = p_->pool->batches();
    s.tasks = p_->pool->tasks();
    s.parks = p_->pool->parks();
    return s;
}

bool Lane::submit(const float* x, int T, const ExpertJob* jobs, int njobs, std::string* err) {
    Impl& p = *p_;
    auto fail = [&](const char* m) {
        if (err) *err = m;
        return false;
    };
    if (T < 1 || T > p.cfg.max_tokens) return fail("T out of range");
    if (njobs < 0 || njobs > p.cfg.max_jobs) return fail("too many jobs");
    const int P = p.planes;
    const int K = njobs ? jobs[0].w.n_embd : 0;
    unsigned kinds = 0;
    for (int i = 0; i < njobs; ++i) {
        const ExpertJob& j = jobs[i];
        if (j.nrows < 1 || j.nrows > kMaxRows || j.nrows * P > kMaxCols) return fail("job rows out of range");
        if (!lane_supports(j.w.gate_type) || !lane_supports(j.w.up_type) || !lane_supports(j.w.down_type)) return fail("unsupported expert type");
        if (j.w.n_embd != K || K > 8192 || j.w.n_ff > 8192) return fail("n_embd differs between jobs or is too large");
        if (K % weight_block_elems(j.w.gate_type) || K % weight_block_elems(j.w.up_type) || j.w.n_ff % weight_block_elems(j.w.down_type)) return fail("dimension not a multiple of the weight block");
        if (K % 32 || j.w.n_ff % 32) return fail("dimension not a multiple of 32");
        for (int r = 0; r < j.nrows; ++r)
            if ((j.act_row ? j.act_row[r] : r) >= T) return fail("act_row out of range");
        kinds |= 1u << (act_block(j.w.gate_type) == 256 ? 0 : 1);
        kinds |= 1u << (act_block(j.w.up_type) == 256 ? 0 : 1);
    }
    p.x = x;
    p.T = T;
    p.jobs = jobs;
    p.njobs = njobs;
    p.xK = K;
    p.xkinds = kinds;
    if (njobs == 0) {
        p.tasks.clear();
        p.n_q = 0;
        p.pool->start(&Impl::task_entry, &p, 0);
        return true;
    }
    for (int k = 0; k < 2; ++k)
        if (kinds & (1u << k)) {
            void* mem = p.xbuf[k].ensure(actq_bytes(K, T * P));
            actq_bind(&p.xact[k], mem, K, k == 0 ? 256 : 32, T * P);
        }
    p.tasks.clear();
    for (int r = 0; r < T; ++r) p.tasks.push_back({0, 0, 0, r, r + 1});
    p.n_q = T;
    p.q_done.store(0, std::memory_order_relaxed);
    for (int i = 0; i < njobs; ++i) {
        const ExpertJob& j = jobs[i];
        JobState& st = p.js[size_t(i)];
        const int nf = j.w.n_ff;
        for (int r = 0; r < j.nrows; ++r)
            for (int q = 0; q < P; ++q) st.cols[r * P + q] = (j.act_row ? j.act_row[r] : r) * P + q;
        st.h.ensure(sizeof(float) * size_t(j.nrows) * size_t(nf));
        const int hblk = act_block(j.w.down_type);
        void* hm = st.hq.ensure(actq_bytes(nf, j.nrows * P));
        actq_bind(&st.hact, hm, nf, hblk, j.nrows * P);
        const int na = (nf + p.cfg.slice_a - 1) / p.cfg.slice_a;
        st.a_left.store(na, std::memory_order_relaxed);
        st.h_ready.store(0, std::memory_order_relaxed);
    }
    for (int i = 0; i < njobs; ++i)
        for (int r0 = 0; r0 < jobs[i].w.n_ff; r0 += p.cfg.slice_a)
            p.tasks.push_back({1, 0, uint16_t(i), r0, std::min(jobs[i].w.n_ff, r0 + p.cfg.slice_a)});
    for (int i = 0; i < njobs; ++i)
        for (int r0 = 0; r0 < jobs[i].w.n_embd; r0 += p.cfg.slice_b)
            p.tasks.push_back({2, 0, uint16_t(i), r0, std::min(jobs[i].w.n_embd, r0 + p.cfg.slice_b)});
    if (p.tasks.size() >= (1u << 20)) return fail("too many tasks");
    p.pool->start(&Impl::task_entry, &p, int(p.tasks.size()));
    return true;
}

bool Lane::done() const { return p_->pool->done(); }
void Lane::wait() { p_->pool->wait(); }

bool Lane::run(const float* x, int T, const ExpertJob* jobs, int njobs, std::string* err) {
    if (!submit(x, T, jobs, njobs, err)) return false;
    wait();
    return true;
}

}  // namespace limbo::cpu
