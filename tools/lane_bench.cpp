// Copyright (c) 2026 ITERFORM. All rights reserved.
// tools/lane_bench.cpp — the CPU lane's expert kernels on a real file: GB/s a core per format at 1, 2, 4 and 8 rows, and the
// pool (the engine's worker placement) on one layer's batch of 10 single-row experts.
//
//   lane_bench <shard1.gguf> [shard2.gguf ...]     LIMBO_LANE_PLANES (1), LIMBO_LANE_REPS (40)
//
// Weights are copied off the mapping first (16 experts a tensor, ~35 MB a format: past the L3, so DRAM speed).
#include "limbo/cpu_lane.hpp"
#include "limbo/gguf.hpp"
#include "limbo/model.hpp"
#include "limbo/types.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using namespace limbo;

struct Stack {
    uint32_t type = 0;
    int n_in = 0, n_out = 0, n = 0;
    uint64_t row_bytes = 0, expert_bytes = 0;
    std::vector<uint8_t> bytes;
    std::string what;
};

Stack copy_experts(const gguf::Tensor& t, int n, const std::string& what) {
    Stack s;
    s.type = t.type;
    s.n_in = int(t.ne[0]);
    s.n_out = int(t.ne[1]);
    s.n = std::min<int>(n, int(t.ne[2]));
    s.row_bytes = type_bytes(t.type, uint64_t(s.n_in));
    s.expert_bytes = s.row_bytes * uint64_t(s.n_out);
    s.bytes.resize(size_t(s.expert_bytes) * size_t(s.n));
    std::memcpy(s.bytes.data(), t.data, s.bytes.size());
    s.what = what;
    return s;
}

double kernel_gbs(const Stack& s, int rows, int planes, int reps) {
    const cpu::Isa isa = cpu::detect_isa();
    const int ncols = rows * planes, blk = cpu::act_block(s.type);
    std::vector<float> x(size_t(rows) * size_t(s.n_in));
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (float& v : x) v = nd(rng);
    std::vector<uint8_t> mem(cpu::actq_bytes(s.n_in, ncols) + 64);
    void* al = reinterpret_cast<void*>((reinterpret_cast<uintptr_t>(mem.data()) + 63) & ~uintptr_t(63));
    cpu::ActQ a;
    cpu::actq_bind(&a, al, s.n_in, blk, ncols);
    cpu::actq_quantize(isa, x.data(), rows, planes, &a);
    std::vector<int> cols(static_cast<size_t>(ncols));
    for (int i = 0; i < ncols; ++i) cols[size_t(i)] = i;
    std::vector<float> y(size_t(ncols) * size_t(s.n_out));
    const auto t0 = Clock::now();
    for (int r = 0; r < reps; ++r)
        for (int e = 0; e < s.n; ++e)
            cpu::gemv_cols(isa, s.type, s.bytes.data() + uint64_t(e) * s.expert_bytes, s.row_bytes, s.n_out, a, cols.data(), ncols,
                           y.data(), s.n_out);
    const double sec = std::chrono::duration<double>(Clock::now() - t0).count();
    return double(s.expert_bytes) * double(s.n) * double(reps) / sec / 1e9;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: lane_bench <shard.gguf>...\n");
        return 2;
    }
    const int planes = std::getenv("LIMBO_LANE_PLANES") ? std::atoi(std::getenv("LIMBO_LANE_PLANES")) : 1;
    const int reps = std::getenv("LIMBO_LANE_REPS") ? std::atoi(std::getenv("LIMBO_LANE_REPS")) : 40;
    std::vector<std::string> paths(argv + 1, argv + argc);
    std::string err;
    auto m = model::Model::open(paths, &err);
    if (!m) {
        std::fprintf(stderr, "lane_bench: %s\n", err.c_str());
        return 1;
    }
    // the first layer of each (role, type) the file has
    std::map<std::pair<std::string, uint32_t>, Stack> st;
    for (int l = 0; l < 64; ++l)
        for (const char* role : {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"}) {
            const gguf::Tensor* t = m->tensor(l, role);
            if (!t || t->n_dims != 3 || !cpu::lane_supports(t->type)) continue;
            const auto key = std::make_pair(std::string(role), t->type);
            if (st.count(key)) continue;
            st[key] = copy_experts(*t, 16, "layer " + std::to_string(l) + " " + role);
        }
    std::printf("isa %s, planes %d, %d reps over 16 experts\n", cpu::isa_name(cpu::detect_isa()), planes, reps);
    std::printf("%-34s %-8s %5s x %-5s  %8s %8s %8s %8s   GB/s a core at 1 / 2 / 4 / 8 rows\n", "tensor", "type", "out", "in", "1", "2",
                "4", "8");
    for (auto& [key, s] : st) {
        const TypeInfo* ti = type_info(s.type);
        std::printf("%-34s %-8s %5d x %-5d ", s.what.c_str(), ti ? ti->name.data() : "?", s.n_out, s.n_in);
        for (int rows : {1, 2, 4, 8}) std::printf(" %8.2f", kernel_gbs(s, rows, planes, rows >= 4 ? reps / 2 : reps));
        std::printf("\n");
    }

    // the pool: one layer's batch, 10 single-row experts of the first layer that has all three stacks
    const gguf::Tensor *g = nullptr, *u = nullptr, *d = nullptr;
    int lay = -1;
    for (int l = 0; l < 64 && lay < 0; ++l) {
        g = m->tensor(l, "ffn_gate_exps.weight");
        u = m->tensor(l, "ffn_up_exps.weight");
        d = m->tensor(l, "ffn_down_exps.weight");
        if (g && u && d && cpu::lane_supports(g->type) && cpu::lane_supports(u->type) && cpu::lane_supports(d->type)) lay = l;
    }
    if (lay < 0) return 0;
    const Stack sg = copy_experts(*g, 16, "g"), su = copy_experts(*u, 16, "u"), sd = copy_experts(*d, 16, "d");
    cpu::Lane::Config cfg;
    cfg.planes = planes;
    cfg.max_jobs = 256;
    cfg.pool.cpus = cpu::default_cpus(0, false);
    cfg.pool.spin_us = 3000;
    auto lane = cpu::Lane::create(cfg, &err);
    if (!lane) {
        std::fprintf(stderr, "lane_bench: %s\n", err.c_str());
        return 1;
    }
    const int n_embd = sg.n_in, n_ff = sg.n_out;
    for (int rows : {1, 2, 4}) {
        std::vector<float> x(size_t(rows) * size_t(n_embd), 0.0f);
        std::mt19937 rng(3);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (float& v : x) v = nd(rng);
        std::vector<float> out(size_t(10) * size_t(rows) * size_t(n_embd));
        cpu::ExpertJob jobs[10];
        for (int j = 0; j < 10; ++j) {
            const int e = (j * 7) % 16;
            jobs[j].w = cpu::ExpertRef{sg.bytes.data() + uint64_t(e) * sg.expert_bytes, su.bytes.data() + uint64_t(e) * su.expert_bytes,
                                       sd.bytes.data() + uint64_t(e) * sd.expert_bytes, g->type, u->type, d->type, n_embd, n_ff};
            jobs[j].nrows = rows;
            jobs[j].out = out.data() + size_t(j) * size_t(rows) * size_t(n_embd);
        }
        for (int w = 0; w < 20; ++w) (void)lane->run(x.data(), rows, jobs, 10, &err);
        const int n = 400;
        const auto t0 = Clock::now();
        for (int i = 0; i < n; ++i)
            if (!lane->run(x.data(), rows, jobs, 10, &err)) {
                std::fprintf(stderr, "lane_bench: %s\n", err.c_str());
                return 1;
            }
        const double us = std::chrono::duration<double, std::micro>(Clock::now() - t0).count() / n;
        const double gb = double(sg.expert_bytes + su.expert_bytes + sd.expert_bytes) * 10.0 / (us * 1e-6) / 1e9;
        std::printf("pool layer %d, %d workers, 10 experts x %d row(s): %7.1f us a batch, %6.1f us an expert, %5.2f GB/s (%.2f a worker)\n",
                    lay, lane->workers(), rows, us, us / 10.0, gb, gb / lane->workers());
    }
    return 0;
}
