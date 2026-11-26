// Copyright (c) 2026 ITERFORM. All rights reserved.
// tools/oracle_gate_gpu.cu — the GPU's teacher-forced gate against an oracle capture, the same record as
// tools/oracle_gate writes for the CPU reference (so the two records diff directly).
//
//   oracle_gate_gpu -m model.gguf --oracle bench/oracle/qwen4exp-coder-hi.jsonl --out build/tmp/gpu-hi.jsonl
//                   [--ctx N] [--min-free-gb G] [--max-pred N]
//
// The capture (oracle/oracle_dump) holds one prompt: its ids, the greedy tokens and per generated position
// the oracle's top-2 as [id1, logit1, id2, logit2, gap]. This tool feeds the prompt and then the ORACLE's
// greedy tokens through Gpu (so a disagreement at one position cannot snowball into the next) and writes one
// JSONL line per generated position: the GPU's top-2 next to the oracle's, the GPU logit at the oracle's two
// ids, and the differences (d1 / d2 / dgap). Agreement is the GPU's argmax == the oracle's greedy token.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "limbo/cuda/gpu.hpp"
#include "limbo/model.hpp"

namespace {

struct Capture {
    std::string name;
    std::vector<int32_t> prompt, greedy;
    std::vector<std::vector<double>> top2;
};

bool read_numbers(const std::string& s, size_t from, std::vector<double>& out, size_t* end) {
    size_t i = from;
    while (i < s.size() && s[i] != '[') ++i;
    if (i >= s.size()) return false;
    ++i;
    while (i < s.size() && s[i] != ']') {
        char* e = nullptr;
        const double v = std::strtod(s.c_str() + i, &e);
        if (e == s.c_str() + i) {
            ++i;
            continue;
        }
        out.push_back(v);
        i = size_t(e - s.c_str());
    }
    if (end) *end = i + 1;
    return true;
}

bool parse_capture(const std::string& line, Capture& c) {
    const size_t pn = line.find("\"name\":\"");
    if (pn == std::string::npos) return false;
    const size_t q = line.find('"', pn + 8);
    c.name = line.substr(pn + 8, q - (pn + 8));
    const size_t pp = line.find("\"prompt_ids\":");
    const size_t pg = line.find("\"greedy\":");
    const size_t pt = line.find("\"top2\":");
    if (pp == std::string::npos || pg == std::string::npos || pt == std::string::npos) return false;
    std::vector<double> v;
    read_numbers(line, pp, v, nullptr);
    for (double d : v) c.prompt.push_back(int32_t(d));
    v.clear();
    read_numbers(line, pg, v, nullptr);
    for (double d : v) c.greedy.push_back(int32_t(d));
    v.clear();
    size_t i = pt + 7;
    while (i < line.size() && line[i] != '[') ++i;
    ++i;
    while (i < line.size()) {
        while (i < line.size() && line[i] != '[' && line[i] != ']') ++i;
        if (i >= line.size() || line[i] == ']') break;
        std::vector<double> row;
        size_t e = 0;
        if (!read_numbers(line, i, row, &e)) break;
        if (row.size() == 5) c.top2.push_back(row);
        i = e;
    }
    return !c.prompt.empty();
}

struct Top2 {
    int i1 = 0, i2 = 0;
    float v1 = -INFINITY, v2 = -INFINITY;
};

Top2 top2(const float* l, int n) {
    Top2 t;
    for (int v = 0; v < n; ++v) {
        const float x = l[v];
        if (x > t.v1) {
            t.v2 = t.v1;
            t.i2 = t.i1;
            t.v1 = x;
            t.i1 = v;
        } else if (x > t.v2) {
            t.v2 = x;
            t.i2 = v;
        }
    }
    return t;
}

}  // namespace

int main(int argc, char** argv) {
    const char* model = nullptr;
    const char* oracle = nullptr;
    const char* out_path = nullptr;
    int n_ctx = 0, max_pred = 0;
    double min_free_gb = 14.6;
    for (int i = 1; i < argc; ++i) {
        auto is = [&](const char* a) { return std::strcmp(argv[i], a) == 0; };
        if ((is("-m") || is("--model")) && i + 1 < argc) model = argv[++i];
        else if (is("--oracle") && i + 1 < argc) oracle = argv[++i];
        else if (is("--out") && i + 1 < argc) out_path = argv[++i];
        else if (is("--ctx") && i + 1 < argc) n_ctx = std::atoi(argv[++i]);
        else if (is("--max-pred") && i + 1 < argc) max_pred = std::atoi(argv[++i]);
        else if (is("--min-free-gb") && i + 1 < argc) min_free_gb = std::atof(argv[++i]);
        else {
            std::fprintf(stderr, "usage: oracle_gate_gpu -m model.gguf --oracle capture.jsonl --out out.jsonl [--ctx N] [--min-free-gb G] [--max-pred N]\n");
            return 2;
        }
    }
    if (!model || !oracle || !out_path) {
        std::fprintf(stderr, "oracle_gate_gpu: -m, --oracle and --out are required\n");
        return 2;
    }

    Capture cap;
    {
        std::ifstream in(oracle, std::ios::binary);
        std::string line;
        bool ok = false;
        while (std::getline(in, line))
            if (line.find("\"prompt_ids\"") != std::string::npos) {
                ok = parse_capture(line, cap);
                break;
            }
        if (!ok || cap.greedy.size() != cap.top2.size() || cap.greedy.empty()) {
            std::fprintf(stderr, "oracle_gate_gpu: cannot read a capture from %s\n", oracle);
            return 1;
        }
    }
    const int n_prompt = int(cap.prompt.size());
    int n_pred = int(cap.greedy.size());
    if (max_pred > 0 && max_pred < n_pred) n_pred = max_pred;
    if (n_ctx <= 0) n_ctx = n_prompt + n_pred + 8;

    std::string err;
    auto m = limbo::model::Model::open({model}, &err);
    if (!m) {
        std::fprintf(stderr, "oracle_gate_gpu: %s\n", err.c_str());
        return 1;
    }
    const auto t_open = std::chrono::steady_clock::now();
    // LIMBO_TEST_KV=q8: the same run over a Q8 K/V pool
    const char* tkv = std::getenv("LIMBO_TEST_KV");
    const limbo::cuda::KvFormat kv_fmt = tkv && std::string(tkv) == "q8" ? limbo::cuda::KvFormat::Q8 : limbo::cuda::KvFormat::F16;
    auto gpu = limbo::cuda::Gpu::open(*m, n_ctx, static_cast<uint64_t>(min_free_gb * 1e9), &err, kv_fmt);
    if (!gpu) {
        std::fprintf(stderr, "oracle_gate_gpu: %s\n", err.c_str());
        return 1;
    }
    auto since = [](std::chrono::steady_clock::time_point t) {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
    };
    std::fprintf(stderr, "oracle_gate_gpu: %s: %d prompt tokens, %d generated positions, n_ctx %d; open %.1f s, weights %.2f GB\n",
                 cap.name.c_str(), n_prompt, n_pred, n_ctx, since(t_open), double(gpu->weight_bytes()) / 1e9);
    const int nv = gpu->n_vocab();
    std::vector<float> logits(static_cast<size_t>(nv));
    std::FILE* out = std::fopen(out_path, "w");
    if (!out) {
        std::fprintf(stderr, "oracle_gate_gpu: cannot write %s\n", out_path);
        return 1;
    }

    const auto t_start = std::chrono::steady_clock::now();
    if (std::getenv("LIMBO_GATE_PREFILL")) {   // the prompt through the batched prefill instead of single steps
        if (!gpu->prefill(cap.prompt.data(), n_prompt, 0, true)) {
            std::fprintf(stderr, "oracle_gate_gpu: prefill failed\n");
            return 1;
        }
    } else {
        for (int i = 0; i < n_prompt; ++i)
            if (!gpu->step(cap.prompt[static_cast<size_t>(i)], i, i == n_prompt - 1)) {
                std::fprintf(stderr, "oracle_gate_gpu: step failed at position %d\n", i);
                return 1;
            }
    }
    gpu->read_logits(logits.data());
    const double t_prefill = since(t_start);
    std::fprintf(stderr, "  prefill %d tokens in %.2f s (%.0f tok/s)\n", n_prompt, t_prefill, n_prompt / t_prefill);

    int agree = 0, tie_dis = 0, big_dis = 0;
    double max_d1 = 0, max_dgap = 0;
    const auto t_gen = std::chrono::steady_clock::now();
    for (int g = 0; g < n_pred; ++g) {
        const Top2 t = top2(logits.data(), nv);
        const std::vector<double>& o = cap.top2[static_cast<size_t>(g)];
        const int o1 = int(o[0]), o2 = int(o[2]);
        const double ol1 = o[1], ol2 = o[3], ogap = o[4];
        const double l_o1 = logits[static_cast<size_t>(o1)], l_o2 = logits[static_cast<size_t>(o2)];
        const double d1 = l_o1 - ol1, d2 = l_o2 - ol2;
        const double dgap = (l_o1 - l_o2) - ogap;
        const bool ok = (t.i1 == cap.greedy[static_cast<size_t>(g)]);
        agree += ok;
        if (!ok) (ogap < 0.1 ? tie_dis : big_dis)++;
        max_d1 = std::max(max_d1, std::fabs(d1));
        max_dgap = std::max(max_dgap, std::fabs(dgap));
        const int pos = n_prompt - 1 + g;
        std::fprintf(out,
                     "{\"name\":\"%s\",\"gen\":%d,\"pos\":%d,\"agree\":%s,\"oracle_greedy\":%d,"
                     "\"limbo\":[%d,%.6f,%d,%.6f,%.6f],\"oracle\":[%d,%.6f,%d,%.6f,%.6f],"
                     "\"limbo_at_o1\":%.6f,\"limbo_at_o2\":%.6f,\"d1\":%.6f,\"d2\":%.6f,\"dgap\":%.6f}\n",
                     cap.name.c_str(), g, pos, ok ? "true" : "false", cap.greedy[static_cast<size_t>(g)], t.i1, double(t.v1),
                     t.i2, double(t.v2), double(t.v1) - double(t.v2), o1, ol1, o2, ol2, ogap, l_o1, l_o2, d1, d2, dgap);
        std::fflush(out);
        std::fprintf(stderr, "  gen %2d pos %4d  %s  gpu %d (%.4f) vs oracle %d (%.4f)  ogap %.4f  d1 %+.4f  dgap %+.4f\n", g, pos,
                     ok ? "ok  " : "DIFF", t.i1, double(t.v1), o1, ol1, ogap, d1, dgap);
        if (g + 1 < n_pred) {
            if (!gpu->step(cap.greedy[static_cast<size_t>(g)], pos + 1, true)) {
                std::fprintf(stderr, "oracle_gate_gpu: step failed at position %d\n", pos + 1);
                return 1;
            }
            gpu->read_logits(logits.data());
        }
    }
    const double t_gen_s = since(t_gen);
    std::fprintf(out,
                 "{\"summary\":true,\"name\":\"%s\",\"n_prompt\":%d,\"n_gen\":%d,\"agree\":%d,\"disagree_gap_lt_0.1\":%d,"
                 "\"disagree_gap_ge_0.1\":%d,\"max_abs_d1\":%.6f,\"max_abs_dgap\":%.6f,\"arith\":\"gpu\","
                 "\"prefill_s\":%.2f,\"gen_s\":%.2f,\"tok_per_s_prefill\":%.3f,\"tok_per_s_gen\":%.3f}\n",
                 cap.name.c_str(), n_prompt, n_pred, agree, tie_dis, big_dis, max_d1, max_dgap, t_prefill, t_gen_s,
                 n_prompt / t_prefill, n_pred > 1 ? (n_pred - 1) / t_gen_s : 0.0);
    std::fclose(out);
    std::fprintf(stderr, "oracle_gate_gpu: %s: %d/%d agree, %d disagree with oracle gap < 0.1, %d with gap >= 0.1, max|d1| %.4f, max|dgap| %.4f\n",
                 cap.name.c_str(), agree, n_pred, tie_dis, big_dis, max_d1, max_dgap);
    return big_dis ? 1 : 0;
}
