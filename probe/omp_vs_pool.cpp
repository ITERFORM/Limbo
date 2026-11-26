// Copyright (c) 2026 ITERFORM. All rights reserved.
// probe/omp_vs_pool.cpp — what does one parallel dispatch cost, four ways?
//
// The question is whether OpenMP belongs in the engine. The answer depends on a number nobody has
// measured here: the cost of ENTERING a parallel region, against the cost of the alternatives, for the
// task shape the CPU lane actually uses (a dequant+dot over an expert's rows, dispatched once per
// expert per layer — so hundreds of dispatches per step, each of them tens to hundreds of microseconds).
//
//   A  OpenMP parallel for, default wait policy (libgomp parks workers: a futex wake per region)
//   B  OpenMP parallel for with OMP_WAIT_POLICY=active (workers spin between regions)
//   C  std::thread spawned per dispatch (what the reference in src/step.cpp does today)
//   D  a persistent pool with spin-then-futex (a small stand-in for src/cpu_lane_pool.cpp)
//
// Two task sizes are measured, because dispatch cost only shows up on the small one:
//   512 rows  — one expert's gate matrix, the lane's real work
//    16 rows  — a verify-window-sized job, where dispatch cost is the whole story
//
//   make ompbench && build/bin/tools/omp_vs_pool <model.gguf> [dispatches] [workers]
//   OMP_WAIT_POLICY=active build/bin/tools/omp_vs_pool ...   (for variant B)
#include "limbo/gguf.hpp"
#include "limbo/model.hpp"
#include "limbo/quants.hpp"
#include "limbo/types.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <atomic>
#include <functional>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

float dot_row(uint32_t type, const uint8_t* row, const float* x, int n) {
    const limbo::TypeInfo* ti = limbo::type_info(type);
    float buf[256];
    float acc = 0;
    for (int i = 0; i < n; i += int(ti->block_elems)) {
        limbo::quants::dequant_row(type, row + limbo::type_bytes(type, uint64_t(i)), buf, ti->block_elems);
        const int k = std::min<int>(int(ti->block_elems), n - i);
        for (int j = 0; j < k; ++j) acc += buf[j] * x[i + j];
    }
    return acc;
}

// The multi-column shape of the GPU kernel f32_cols_k: one row of W is decoded ONCE and dotted
// against C different columns (y[c][r] = W[r, :] . x[c]). That is the CPU lane's union-batching shape
// (1..16 rows of a verify window share one pass over the weights), and it is the shape where the
// threading choice actually shows: the row work grows with C while the decode stays the same.
struct WorkCols {
    uint32_t type = 0;
    const uint8_t* base = nullptr;
    uint64_t row_bytes = 0;
    const float* x = nullptr;      // [C][n_in]
    float* y = nullptr;            // [C][n_out]
    int n_in = 0, n_out = 0, C = 1;
    void operator()(int b, int e) const {
        const limbo::TypeInfo* ti = limbo::type_info(type);
        float buf[256];
        for (int r = b; r < e; ++r) {
            const uint8_t* row = base + uint64_t(r) * row_bytes;
            // decode the row once, into registers-sized chunks, and dot every column against it
            float acc[16];
            for (int c = 0; c < C; ++c) acc[c] = 0.f;
            for (int i = 0; i < n_in; i += int(ti->block_elems)) {
                limbo::quants::dequant_row(type, row + limbo::type_bytes(type, uint64_t(i)), buf, ti->block_elems);
                const int k = std::min<int>(int(ti->block_elems), n_in - i);
                for (int c = 0; c < C; ++c) {
                    const float* xc = x + size_t(c) * n_in + i;
                    float a = acc[c];
                    for (int j = 0; j < k; ++j) a += buf[j] * xc[j];
                    acc[c] = a;
                }
            }
            for (int c = 0; c < C; ++c) y[size_t(c) * n_out + r] = acc[c];
        }
    }
};

// The workload: rows [b, e) of one expert's matrix, dotted against x.
struct Work {
    uint32_t type = 0;
    const uint8_t* base = nullptr;
    uint64_t row_bytes = 0;
    const float* x = nullptr;
    float* y = nullptr;
    int n_in = 0;
    void operator()(int b, int e) const {
        for (int r = b; r < e; ++r) y[r] = dot_row(type, base + uint64_t(r) * row_bytes, x, n_in);
    }
};

// A persistent pool: workers spin, the dispatcher publishes a batch and waits. Simplified from
// src/cpu_lane_pool.cpp (which claims tasks from a single atomic word and chains stages by flags).
struct Pool {
    int n = 0;
    std::atomic<int> epoch{0};
    std::atomic<int> next{0};
    std::atomic<int> done{0};
    int total = 0;
    std::function<void(int, int)> task;
    std::atomic<bool> stop{false};
    std::vector<std::thread> th;

    Pool(int workers) : n(workers) {
        for (int w = 0; w < n; ++w) th.emplace_back([this] {
            int seen = 0;
            for (;;) {
                while (epoch.load(std::memory_order_acquire) == seen && !stop.load(std::memory_order_relaxed))
                    for (int s = 0; s < 200; ++s) __builtin_ia32_pause();
                if (stop.load(std::memory_order_relaxed)) return;
                seen = epoch.load(std::memory_order_acquire);
                for (;;) {
                    const int i = next.fetch_add(1, std::memory_order_acq_rel);
                    if (i >= total) break;
                    task(i, i + 1);
                    done.fetch_add(1, std::memory_order_acq_rel);
                }
            }
        });
    }
    ~Pool() {
        stop.store(true, std::memory_order_relaxed);
        epoch.fetch_add(1, std::memory_order_release);
        for (auto& t : th) t.join();
    }
    template <typename WT>
    void run(const WT& w, int rows) {
        task = [&w](int b, int e) { w(b, e); };
        total = rows;
        next.store(0, std::memory_order_relaxed);
        done.store(0, std::memory_order_relaxed);
        epoch.fetch_add(1, std::memory_order_release);
        while (done.load(std::memory_order_acquire) < rows) __builtin_ia32_pause();
    }
};

template <typename WT>
double time_dispatches(int dispatches, const WT& w, int rows, int workers, const char* label) {
    std::vector<double> per;
    for (int d = 0; d < dispatches; ++d) {
        const auto t0 = Clock::now();
        // A: OpenMP, whatever wait policy the environment sets
#ifdef _OPENMP
#pragma omp parallel for num_threads(workers) schedule(static)
        for (int r = 0; r < rows; ++r) w(r, r + 1);
#else
        w(0, rows);
#endif
        per.push_back(std::chrono::duration<double, std::micro>(Clock::now() - t0).count());
    }
    double best = 1e18;
    for (double v : per) best = std::min(best, v);
    std::printf("%-28s %8.1f us/dispatch (best of %d)\n", label, best, dispatches);
    return best;
}

template <typename WT>
double time_thread_spawn(int dispatches, const WT& w, int rows, int workers, const char* label) {
    std::vector<double> per;
    for (int d = 0; d < dispatches; ++d) {
        const auto t0 = Clock::now();
        const int chunk = (rows + workers - 1) / workers;
        std::vector<std::thread> pool;
        for (int t = 0; t < workers; ++t) {
            const int b = t * chunk, e = std::min(rows, b + chunk);
            if (b >= e) break;
            pool.emplace_back([&w, b, e] { w(b, e); });
        }
        for (auto& t : pool) t.join();
        per.push_back(std::chrono::duration<double, std::micro>(Clock::now() - t0).count());
    }
    double best = 1e18;
    for (double v : per) best = std::min(best, v);
    std::printf("%-28s %8.1f us/dispatch (best of %d)\n", label, best, dispatches);
    return best;
}

template <typename WT>
double time_pool(Pool& pool, int dispatches, const WT& w, int rows, const char* label) {
    std::vector<double> per;
    for (int d = 0; d < dispatches; ++d) {
        const auto t0 = Clock::now();
        pool.run(w, rows);
        per.push_back(std::chrono::duration<double, std::micro>(Clock::now() - t0).count());
    }
    double best = 1e18;
    for (double v : per) best = std::min(best, v);
    std::printf("%-28s %8.1f us/dispatch (best of %d)\n", label, best, dispatches);
    return best;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: omp_vs_pool <model.gguf> [dispatches] [workers]\n");
        return 2;
    }
    const int dispatches = argc > 2 ? std::atoi(argv[2]) : 200;
    const int workers = argc > 3 ? std::atoi(argv[3]) : 6;

    std::string err;
    auto m = limbo::model::Model::open({argv[1]}, &err);
    if (!m) {
        std::fprintf(stderr, "omp_vs_pool: %s\n", err.c_str());
        return 1;
    }
    const limbo::gguf::Tensor* t = m->tensor(0, "ffn_gate_exps.weight");
    if (!t) {
        std::fprintf(stderr, "omp_vs_pool: no ffn_gate_exps in this file\n");
        return 1;
    }
    const int n_in = int(t->ne[0]);
    const uint64_t row_bytes = limbo::type_bytes(t->type, uint64_t(n_in));
    std::vector<float> x(size_t(n_in), 0.01f), y(size_t(t->ne[1]), 0.0f);
    Work w{t->type, t->data, row_bytes, x.data(), y.data(), n_in};

    std::printf("# %d workers, %d dispatches, tensor %s %d x %d\n", workers, dispatches,
                limbo::type_info(t->type)->name.data(), n_in, int(t->ne[1]));
#ifdef _OPENMP
    std::printf("# OpenMP max threads %d, wait policy %s\n", omp_get_max_threads(),
                std::getenv("OMP_WAIT_POLICY") ? std::getenv("OMP_WAIT_POLICY") : "(default: passive)");
#else
    std::printf("# built WITHOUT OpenMP\n");
#endif

    // ---- the multi-column shape: the GPU's f32_cols_k, on the CPU side -------------------------
    // y[c][r] = W[r, :] . x[c]: one decode pass per row serves every column. This is the lane's union
    // batching, and the shape the GPU kernel above implements with C columns per block.
    for (int C : {1, 4, 16}) {
        std::vector<float> xs(size_t(C) * size_t(n_in), 0.01f);
        std::vector<float> ys(size_t(C) * size_t(t->ne[1]), 0.0f);
        WorkCols wc{t->type, t->data, row_bytes, xs.data(), ys.data(), n_in, int(t->ne[1]), C};
        const int rows = 512;
        wc(0, rows);
        double solo = 1e18;
        for (int rep = 0; rep < 5; ++rep) {
            const auto t0 = Clock::now();
            wc(0, rows);
            solo = std::min(solo, std::chrono::duration<double, std::micro>(Clock::now() - t0).count());
        }
        std::printf("-- C=%-2d columns, %d rows (%.1f us of work per dispatch, one thread, warm)\n", C, rows, solo);
        time_dispatches(dispatches, wc, rows, workers, "A OpenMP (env wait policy)");
        time_thread_spawn(dispatches, wc, rows, workers, "C std::thread per dispatch");
        Pool pool(workers);
        time_pool(pool, dispatches, wc, rows, "D persistent pool + spin");
    }

    const int rows_big = int(t->ne[1]) < 512 ? int(t->ne[1]) : 512;
    const int rows_small = 16;
    for (int rows : {rows_big, rows_small}) {
        w(0, rows);  // warm the pages: a cold first call measures the SSD, not the CPU
        double solo = 1e18;
        for (int rep = 0; rep < 5; ++rep) {
            const auto t0 = Clock::now();
            w(0, rows);
            solo = std::min(solo, std::chrono::duration<double, std::micro>(Clock::now() - t0).count());
        }
        std::printf("-- %d rows (%.1f us of work per dispatch, one thread, warm)\n", rows,
                    solo);
        time_dispatches(dispatches, w, rows, workers, "A OpenMP (env wait policy)");
        time_thread_spawn(dispatches, w, rows, workers, "C std::thread per dispatch");
        Pool pool(workers);
        time_pool(pool, dispatches, w, rows, "D persistent pool + spin");
    }
    return 0;
}
