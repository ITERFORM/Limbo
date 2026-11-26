// Copyright (c) 2026 ITERFORM. All rights reserved.
// tests/cpu_lane_test.cpp — the CPU lane's kernels and pool, held to the verified decoder.
//
// Reference: quants::dequant_row (checked against ggml by tools/quant_parity) and a double-precision dot. The
// lane computes on int8 activation planes, so what it can agree to depends on the plane count: 1 plane is
// llama.cpp's Q8 numerics, 3 planes is fp32-like. Every check prints the measured error.
//
//   1. synthetic blocks of every lane format: gemv_cols (AVX2 and scalar) against the reference
//   2. real expert rows of the ladder's files (skipped when a file is absent): every distinct expert tensor
//      type found, gemv_cols against the reference, per format and per plane count
//   3. whole expert FFNs through the Lane (gate, up, SwiGLU, down) against expert_ffn_reference, 1/3/16 rows
//   4. the pool: every task runs exactly once across many batches, with parking in between
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "limbo/check.hpp"
#include "limbo/cpu_lane.hpp"
#include "limbo/gguf.hpp"
#include "limbo/quants.hpp"
#include "limbo/types.hpp"

using namespace limbo;
using namespace limbo::cpu;

namespace {

struct Err {
    double norm_rel = 0;   // max |y - ref| / (|w| |x|)
    double rel = 0;        // max |y - ref| / |ref| over products whose |ref| is at least 1e-3 of |w||x|
    long n = 0;
};

const char* tname(uint32_t t) {
    const TypeInfo* ti = type_info(t);
    return ti ? ti->name.data() : "?";
}

// Heavy-tailed activations: gaussian bulk with a few large outliers per block, like a normed residual.
void make_x(std::mt19937& rng, float* x, int K) {
    std::normal_distribution<float> g(0.0f, 1.0f);
    for (int i = 0; i < K; ++i) x[i] = g(rng);
    for (int i = 0; i < K / 64; ++i) x[size_t(rng() % uint32_t(K))] = g(rng) * 12.0f;
}

// y: [ncols][ldy] from gemv_cols for columns = (row r, plane p), merged over planes; compare with the reference.
Err compare(Isa isa, uint32_t type, const uint8_t* W, int nrows, int K, int nx, int planes, std::mt19937& rng) {
    const uint64_t row_bytes = type_bytes(type, uint64_t(K));
    const int blk = act_block(type);
    std::vector<float> xs(size_t(nx) * size_t(K));
    for (int r = 0; r < nx; ++r) make_x(rng, &xs[size_t(r) * size_t(K)], K);
    std::vector<uint8_t> mem(actq_bytes(K, nx * planes) + 64);
    void* aligned = reinterpret_cast<void*>((reinterpret_cast<uintptr_t>(mem.data()) + 63) & ~uintptr_t(63));
    ActQ a;
    actq_bind(&a, aligned, K, blk, nx * planes);
    actq_quantize(isa, xs.data(), nx, planes, &a);
    std::vector<int> cols(size_t(nx) * size_t(planes));
    for (int i = 0; i < nx * planes; ++i) cols[size_t(i)] = i;
    std::vector<float> y(size_t(nx) * size_t(planes) * size_t(nrows));
    const bool ok = gemv_cols(isa, type, W, row_bytes, nrows, a, cols.data(), nx * planes, y.data(), nrows);
    Err e;
    if (!ok) {
        e.norm_rel = 1e9;
        return e;
    }
    std::vector<float> wf(static_cast<size_t>(K));
    for (int i = 0; i < nrows; ++i) {
        quants::dequant_row(type, W + uint64_t(i) * row_bytes, wf.data(), uint64_t(K));
        double wn = 0;
        for (int k = 0; k < K; ++k) wn += double(wf[size_t(k)]) * double(wf[size_t(k)]);
        wn = std::sqrt(wn);
        for (int r = 0; r < nx; ++r) {
            double ref = 0, xn = 0;
            for (int k = 0; k < K; ++k) {
                ref += double(wf[size_t(k)]) * double(xs[size_t(r) * size_t(K) + size_t(k)]);
                xn += double(xs[size_t(r) * size_t(K) + size_t(k)]) * double(xs[size_t(r) * size_t(K) + size_t(k)]);
            }
            xn = std::sqrt(xn);
            double got = 0;
            for (int p = planes - 1; p >= 0; --p) got += double(y[size_t(r * planes + p) * size_t(nrows) + size_t(i)]);
            const double scale = wn * xn;
            if (scale == 0) continue;
            const double d = std::fabs(got - ref);
            e.norm_rel = std::max(e.norm_rel, d / scale);
            if (std::fabs(ref) >= 1e-3 * scale) e.rel = std::max(e.rel, d / std::fabs(ref));
            ++e.n;
        }
    }
    return e;
}

// ---- 1. synthetic blocks -------------------------------------------------------------------------------
void fill_blocks(std::mt19937& rng, uint32_t type, uint8_t* p, size_t nblocks, size_t block_bytes) {
    for (size_t i = 0; i < nblocks * block_bytes; ++i) p[i] = uint8_t(rng());
    // scales: small positive f16 so nothing is inf/nan; the positions of the f16 fields per format
    auto put = [&](uint8_t* b, size_t off, float v) {
        // f16 of v in [0.0005, 0.02]
        uint32_t bits;
        std::memcpy(&bits, &v, 4);
        const uint32_t sign = (rng() & 1u) && type == 42 ? 0x8000u : 0;   // Q2_0 may carry a negative scale
        const int exp = int((bits >> 23) & 0xff) - 127 + 15;
        const uint16_t h = uint16_t(sign | (uint32_t(exp) << 10) | ((bits >> 13) & 0x3ff));
        b[off] = uint8_t(h & 0xff);
        b[off + 1] = uint8_t(h >> 8);
    };
    std::uniform_real_distribution<float> u(0.0005f, 0.02f);
    for (size_t i = 0; i < nblocks; ++i) {
        uint8_t* b = p + i * block_bytes;
        switch (static_cast<Type>(type)) {
            case Type::Q4_K: case Type::Q5_K: put(b, 0, u(rng)); put(b, 2, u(rng)); break;
            case Type::Q2_K: put(b, 80, u(rng)); put(b, 82, u(rng)); break;
            case Type::Q3_K: put(b, 108, u(rng)); break;
            case Type::Q6_K: put(b, 208, u(rng)); break;
            default: put(b, 0, u(rng)); break;   // IQ*, Q2_0, IQ4_NL: d first
        }
    }
}

void test_synthetic(Isa isa) {
    std::mt19937 rng(7);
    const uint32_t types[] = {12, 13, 14, 11, 10, 22, 21, 23, 18, 20, 42};
    for (uint32_t t : types) {
        const TypeInfo* ti = type_info(t);
        const int K = 1024;   // a multiple of every block
        const int nrows = 40;
        const uint64_t rb = type_bytes(t, uint64_t(K));
        std::vector<uint8_t> W(rb * uint64_t(nrows));
        fill_blocks(rng, t, W.data(), W.size() / ti->block_bytes, ti->block_bytes);
        for (int planes = 1; planes <= 3; ++planes) {
            const Err e = compare(isa, t, W.data(), nrows, K, 5, planes, rng);
            // planes 1: ~1e-3 of |w||x|; planes 3: float rounding
            const double lim = planes == 1 ? 6e-3 : planes == 2 ? 1e-4 : 2e-5;
            char what[160];
            std::snprintf(what, sizeof what, "synthetic %-7s %s planes=%d  norm_rel %.2e rel %.2e (limit %.0e)", tname(t), isa_name(isa), planes, e.norm_rel, e.rel, lim);
            CHECK(e.norm_rel < lim, what);
        }
    }
}

// ---- 2/3. real files -----------------------------------------------------------------------------------
std::string find_one(const std::string& root, const std::string& fname) {
    namespace fs = std::filesystem;
    std::error_code ec;
    if (!fs::exists(root, ec)) return "";
    for (auto& d : fs::directory_iterator(root, ec)) {
        const fs::path c = d.path() / fname;
        if (fs::exists(c, ec)) return c.string();
    }
    return "";
}

std::vector<std::string> model_files() {
    std::vector<std::string> v;
    const std::string hub = std::string(std::getenv("HOME") ? std::getenv("HOME") : "") + "/.cache/huggingface/hub/";
    if (const char* e = std::getenv("LIMBO_MODEL_R1")) v.push_back(e);
    else {
        auto f = find_one(hub + "models--unsloth--Qwen3.6-35B-A3B-MTP-GGUF/snapshots", "Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf");
        if (!f.empty()) v.push_back(f);
    }
    if (const char* e = std::getenv("LIMBO_MODEL_R2")) v.push_back(e);
    else {
        auto f = find_one(hub + "models--peculiar-ragdoll--Cyber-Tiel-Coder-35B-A3B-GGUF-MTP/snapshots", "Cyber-Tiel-Coder-35B-A3B-MTP-UD-Q4_K_M.gguf");
        if (!f.empty()) v.push_back(f);
    }
    if (const char* e = std::getenv("LIMBO_MODEL_R3")) v.push_back(e);
    else {
        const std::string f = "/home/alca/projects/Strata-data/models/coder-IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf";
        if (std::filesystem::exists(f)) v.push_back(f);
    }
    return v;
}

struct ExpTensors {
    const gguf::Tensor *g, *u, *d;
};

ExpTensors layer_tensors(const gguf::File& f, int l) {
    const std::string b = "blk." + std::to_string(l) + ".ffn_";
    return {f.tensor(b + "gate_exps.weight"), f.tensor(b + "up_exps.weight"), f.tensor(b + "down_exps.weight")};
}

const uint8_t* expert_slice(const gguf::Tensor* t, int e) {
    const uint64_t row = type_bytes(t->type, uint64_t(t->ne[0]));
    return t->data + uint64_t(e) * uint64_t(t->ne[1]) * row;
}

ExpertRef make_ref(const ExpTensors& t, int e) {
    ExpertRef r;
    r.gate = expert_slice(t.g, e);
    r.up = expert_slice(t.u, e);
    r.down = expert_slice(t.d, e);
    r.gate_type = t.g->type;
    r.up_type = t.u->type;
    r.down_type = t.d->type;
    r.n_embd = int(t.g->ne[0]);
    r.n_ff = int(t.g->ne[1]);
    return r;
}

void test_real(const std::string& path, Isa isa, std::map<uint32_t, Err>* worst1, std::map<uint32_t, Err>* worst3) {
    std::string err;
    auto f = gguf::File::open(path, &err);
    if (!f) {
        std::printf("SKIP %s: %s\n", path.c_str(), err.c_str());
        return;
    }
    std::printf("-- %s\n", std::filesystem::path(path).filename().c_str());
    std::mt19937 rng(11);
    // every distinct (role, type) among the expert tensors
    std::map<std::pair<std::string, uint32_t>, int> first_layer;
    int n_layer = 0;
    for (auto& t : f->tensors()) {
        const std::string n(t.name);
        const auto p = n.find("ffn_");
        if (n.rfind("blk.", 0) != 0 || p == std::string::npos || n.find("_exps.weight") == std::string::npos) continue;
        const int l = std::atoi(n.c_str() + 4);
        n_layer = std::max(n_layer, l + 1);
        const std::string role = n.substr(p, n.find("_exps") - p);
        auto key = std::make_pair(role, t.type);
        if (!first_layer.count(key)) first_layer[key] = l;
    }
    for (auto& [key, l] : first_layer) {
        const uint32_t type = key.second;
        if (!lane_supports(type)) continue;
        const ExpTensors t = layer_tensors(*f, l);
        const gguf::Tensor* tt = key.first == "ffn_gate" ? t.g : key.first == "ffn_up" ? t.u : t.d;
        const int K = int(tt->ne[0]);
        const int nrows = int(std::min<int64_t>(64, tt->ne[1]));
        const int e = int(tt->ne[2] / 3);
        for (int planes : {1, 3}) {
            const Err er = compare(isa, type, expert_slice(tt, e), nrows, K, 4, planes, rng);
            const double lim = planes == 1 ? 6e-3 : 2e-5;
            char what[200];
            std::snprintf(what, sizeof what, "real %-6s %-7s blk.%-2d K=%-4d planes=%d  norm_rel %.2e rel %.2e", key.first.c_str() + 4, tname(type), l, K, planes, er.norm_rel, er.rel);
            CHECK(er.norm_rel < lim, what);
            auto& w = planes == 1 ? *worst1 : *worst3;
            Err& cur = w[type];
            cur.norm_rel = std::max(cur.norm_rel, er.norm_rel);
            cur.rel = std::max(cur.rel, er.rel);
            cur.n += er.n;
        }
    }
    // whole expert FFNs through the lane: one layer per distinct (gate, up, down) type triple
    std::map<std::array<uint32_t, 3>, int> triple;
    for (int l = 0; l < n_layer; ++l) {
        const ExpTensors t = layer_tensors(*f, l);
        if (!t.g || !t.u || !t.d) continue;
        triple.emplace(std::array<uint32_t, 3>{t.g->type, t.u->type, t.d->type}, l);
    }
    for (int planes : {1, 3}) {
        Lane::Config cfg;
        cfg.planes = planes;
        cfg.isa = isa;
        cfg.pool.cpus = default_cpus(3);
        if (cfg.pool.cpus.empty()) cfg.pool.cpus = {0};
        auto lane = Lane::create(cfg, &err);
        if (!lane) {
            std::printf("FAIL lane create: %s\n", err.c_str());
            ++limbo::test::failures();
            return;
        }
        for (auto& [tr, l] : triple) {
            if (!lane_supports(tr[0]) || !lane_supports(tr[1]) || !lane_supports(tr[2])) continue;
            const ExpTensors t = layer_tensors(*f, l);
            const int ne = int(t.g->ne[0]);
            for (int R : {1, 3, 16}) {
                std::vector<float> x(size_t(R) * size_t(ne));
                for (int r = 0; r < R; ++r) make_x(rng, &x[size_t(r) * size_t(ne)], ne);
                // three experts, each taking R rows of the tile (rows rotated so act_row is exercised)
                const int NJ = 3;
                std::vector<ExpertJob> jobs(NJ);
                std::vector<std::vector<float>> out(NJ, std::vector<float>(size_t(R) * size_t(ne)));
                std::vector<std::vector<int>> arow(NJ, std::vector<int>(size_t(R)));
                for (int j = 0; j < NJ; ++j) {
                    for (int r = 0; r < R; ++r) arow[size_t(j)][size_t(r)] = (r + j) % R;
                    jobs[size_t(j)].w = make_ref(t, 5 + 40 * j);
                    jobs[size_t(j)].nrows = R;
                    jobs[size_t(j)].act_row = arow[size_t(j)].data();
                    jobs[size_t(j)].out = out[size_t(j)].data();
                }
                if (!lane->run(x.data(), R, jobs.data(), NJ, &err)) {
                    std::printf("FAIL lane run: %s\n", err.c_str());
                    ++limbo::test::failures();
                    continue;
                }
                double worst = 0;
                std::vector<float> ref(static_cast<size_t>(ne));
                for (int j = 0; j < NJ; ++j)
                    for (int r = 0; r < R; ++r) {
                        expert_ffn_reference(jobs[size_t(j)].w, &x[size_t(arow[size_t(j)][size_t(r)]) * size_t(ne)], ref.data());
                        double ss = 0, dm = 0;
                        for (int i = 0; i < ne; ++i) ss += double(ref[size_t(i)]) * double(ref[size_t(i)]);
                        const double rms = std::sqrt(ss / ne);
                        for (int i = 0; i < ne; ++i) dm = std::max(dm, std::fabs(double(out[size_t(j)][size_t(r) * size_t(ne) + size_t(i)]) - double(ref[size_t(i)])));
                        worst = std::max(worst, dm / std::max(rms, 1e-30));
                    }
                const double lim = planes == 1 ? 6e-2 : 5e-4;
                char what[220];
                std::snprintf(what, sizeof what, "ffn %s/%s/%s blk.%-2d rows=%-2d planes=%d  max|dy|/rms %.2e", tname(tr[0]), tname(tr[1]), tname(tr[2]), l, R, planes, worst);
                CHECK(worst < lim, what);
            }
        }
    }
}

// ---- 4. the pool ----------------------------------------------------------------------------------------
void test_pool() {
    // The Lane's pool is exercised through submit/run with an empty-weight job set? The raw pool is internal, so the
    // check goes through a lane with many tiny batches and a parking gap, counting completed batches by the output.
    Lane::Config cfg;
    cfg.planes = 1;
    cfg.pool.cpus = default_cpus(4);
    if (cfg.pool.cpus.empty()) cfg.pool.cpus = {0};
    cfg.pool.spin_us = 0;   // park at once: every batch is a wake
    std::string err;
    auto lane = Lane::create(cfg, &err);
    CHECK(lane != nullptr, "pool: lane created");
    if (!lane) return;
    // one synthetic expert in Q4_K (all-zero blocks except a scale), compared batch after batch
    const int ne = 512, nf = 256;
    std::mt19937 rng(3);
    const uint64_t rg = type_bytes(12, uint64_t(ne)), rd = type_bytes(12, uint64_t(nf));
    std::vector<uint8_t> g(rg * size_t(nf)), u(rg * size_t(nf)), d(rd * size_t(ne));
    fill_blocks(rng, 12, g.data(), g.size() / 144, 144);
    fill_blocks(rng, 12, u.data(), u.size() / 144, 144);
    fill_blocks(rng, 12, d.data(), d.size() / 144, 144);
    ExpertJob job;
    job.w = {g.data(), u.data(), d.data(), 12, 12, 12, ne, nf};
    job.nrows = 2;
    std::vector<float> x(size_t(2) * ne), out0(size_t(2) * ne), out(size_t(2) * ne);
    for (int r = 0; r < 2; ++r) make_x(rng, &x[size_t(r) * ne], ne);
    job.out = out0.data();
    CHECK(lane->run(x.data(), 2, &job, 1, &err), "pool: first batch");
    bool all_same = true;
    for (int it = 0; it < 300; ++it) {
        if (it % 3 == 0) std::this_thread::sleep_for(std::chrono::microseconds(it % 7 == 0 ? 400 : 20));
        job.out = out.data();
        std::fill(out.begin(), out.end(), 1e30f);
        if (!lane->run(x.data(), 2, &job, 1, &err)) all_same = false;
        if (std::memcmp(out.data(), out0.data(), out.size() * 4) != 0) all_same = false;
    }
    CHECK(all_same, "pool: 300 batches with parking gaps give bit-identical output");
    const LaneStats s = lane->stats();
    std::printf("     pool stats: %llu batches, %llu tasks, %llu parks (spin_us=0)\n", (unsigned long long)s.batches,
                (unsigned long long)s.tasks, (unsigned long long)s.parks);
    CHECK(s.parks > 0, "pool: workers parked on the futex");
    // asynchronous: submit, poll done(), the output is complete when done() is true
    job.out = out.data();
    std::fill(out.begin(), out.end(), 1e30f);
    CHECK(lane->submit(x.data(), 2, &job, 1, &err), "pool: async submit");
    while (!lane->done()) {
    }
    CHECK(std::memcmp(out.data(), out0.data(), out.size() * 4) == 0, "pool: done() means the output is written");
    // bad input is refused, not run
    ExpertJob bad = job;
    bad.nrows = 17;
    CHECK(!lane->submit(x.data(), 2, &bad, 1, &err), "pool: 17 rows refused");
    bad = job;
    bad.w.gate_type = 8;   // Q8_0 is not a lane format
    CHECK(!lane->submit(x.data(), 2, &bad, 1, &err), "pool: unsupported type refused");
}

}  // namespace

int main() {
    std::printf("cpu isa %s, lane isa %s\n", isa_name(cpu_isa()), isa_name(detect_isa()));
    std::printf("AVX-512: no kernel exists; it has never run on this box or the ThinkPad.\n");
    CHECK(detect_isa() != Isa::Avx512, "detect_isa never reports the reserved AVX-512 slot");
    test_synthetic(Isa::Scalar);
    if (detect_isa() == Isa::Avx2) test_synthetic(Isa::Avx2);
    std::map<uint32_t, Err> w1, w3;
    const Isa isa = detect_isa();
    for (const std::string& f : model_files()) test_real(f, isa, &w1, &w3);
    if (model_files().empty()) std::printf("SKIP real-model checks: no GGUF found\n");
    std::printf("\nworst error per format on real expert rows (isa %s), |y-ref| / (|w||x|) and |y-ref|/|ref| where |ref| >= 1e-3|w||x|:\n", isa_name(isa));
    std::printf("  %-8s %-26s %-26s\n", "format", "planes=1 (norm_rel / rel)", "planes=3 (norm_rel / rel)");
    for (auto& [t, e1] : w1) {
        const Err& e3 = w3[t];
        std::printf("  %-8s %.2e / %.2e         %.2e / %.2e\n", tname(t), e1.norm_rel, e1.rel, e3.norm_rel, e3.rel);
    }
    test_pool();
    return limbo::test::finish("cpu_lane_test");
}
