// Copyright (c) 2026 ITERFORM. All rights reserved.
// tools/dot_bench.cpp — how fast is one decoded row, really?
//
// The reference forward pass is allowed to be slow, but it is not allowed to be mysteriously slow: this
// times the fused dequant+dot on real tensors of a real file, single-threaded and across threads, so a
// pathology is a number instead of a suspicion.
#include "limbo/gguf.hpp"
#include "limbo/model.hpp"
#include "limbo/quants.hpp"
#include "limbo/types.hpp"

#include <chrono>
#include <cstdio>
#include <algorithm>
#include <cstring>
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
        const uint64_t off = limbo::type_bytes(type, uint64_t(i));
        limbo::quants::dequant_row(type, row + off, buf, ti->block_elems);
        const int k = std::min<int>(int(ti->block_elems), n - i);
        for (int j = 0; j < k; ++j) acc += buf[j] * x[i + j];
    }
    return acc;
}

void bench(const limbo::gguf::Tensor& t, int threads, const char* label) {
    const int n_in = int(t.ne[0]);
    const int n_out = int(t.ne[1]);
    const uint64_t row_bytes = limbo::type_bytes(t.type, uint64_t(n_in));
    std::vector<float> x(size_t(n_in), 0.01f), y(size_t(n_out), 0.f);
    const auto t0 = Clock::now();
    auto work = [&](int b, int e) {
        for (int i = b; i < e; ++i) y[size_t(i)] = dot_row(t.type, t.data + uint64_t(i) * row_bytes, x.data(), n_in);
    };
    if (threads <= 1) {
        work(0, n_out);
    } else {
        const int chunk = (n_out + threads - 1) / threads;
        std::vector<std::thread> pool;
        for (int th = 0; th < threads; ++th) {
            const int b = th * chunk, e = std::min(n_out, b + chunk);
            if (b >= e) break;
            pool.emplace_back(work, b, e);
        }
        for (auto& th : pool) th.join();
    }
    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    const double elems = double(n_in) * double(n_out);
    const double bytes = double(row_bytes) * double(n_out);
    std::printf("%-34s %-8s %6d x %-7d %6d threads  %8.1f ms  %7.2f Gelem/s  %6.2f GB/s\n", label,
                limbo::type_info(t.type) ? limbo::type_info(t.type)->name.data() : "?", n_in, n_out, threads, ms,
                elems / (ms / 1000.0) / 1e9, bytes / (ms / 1000.0) / 1e9);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: dot_bench <file.gguf>\n");
        return 2;
    }
    std::string err;
    auto m = limbo::model::Model::open({argv[1]}, &err);
    if (!m) {
        std::fprintf(stderr, "dot_bench: %s\n", err.c_str());
        return 1;
    }
    for (const char* role : {"attn_qkv.weight", "ffn_gate_exps.weight", "ffn_gate_shexp.weight"}) {
        const limbo::gguf::Tensor* t = m->tensor(0, role);
        if (!t) continue;
        if (t->n_dims == 3) {
            // one expert of a routed tensor: rows are the second dim
            limbo::gguf::Tensor one = *t;
            one.ne[1] = t->ne[1];
            one.n_dims = 2;
            one.nbytes = limbo::type_bytes(t->type, uint64_t(t->ne[0]) * uint64_t(t->ne[1]));
            bench(one, 1, "expert 0 of ffn_gate_exps");
            bench(one, 8, "expert 0 of ffn_gate_exps");
        } else {
            bench(*t, 1, role);
            bench(*t, 8, role);
        }
    }
    if (const limbo::gguf::Tensor* head = m->tensor(-1, "output.weight")) {
        bench(*head, 1, "output.weight (the head)");
        bench(*head, 8, "output.weight (the head)");
    }
    return 0;
}
