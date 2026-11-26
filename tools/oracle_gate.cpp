// Copyright (c) 2026 ITERFORM. All rights reserved.
// tools/oracle_gate.cpp — teacher-forced comparison of the reference forward pass against an oracle capture.
//
//   oracle_gate -m model.gguf --oracle bench/oracle/rung1-prose.jsonl --out build/tmp/cpugate/prose.jsonl
//               [-t 8] [--ctx N] [--max-pred N] [--oracle-arith]
//
// --oracle-arith switches the reference to the oracle's own activation rounding (Ref::set_oracle_arith):
// the default keeps fp32 activations, which is the exact arithmetic and what the gate is defined on.
//
// The capture (oracle/oracle_dump) holds, for one prompt, the prompt ids, the greedy tokens and per
// generated position the top-2 as [id1, logit1, id2, logit2, gap]. This tool feeds the prompt and then the
// ORACLE's greedy tokens through Ref (so a disagreement at one position cannot snowball into the next) and
// writes one JSONL line per generated position: Limbo's top-2 next to the oracle's, the Limbo logit at the
// oracle's two ids, and the differences. Agreement is Limbo argmax == oracle greedy.
#include "limbo/step.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

// The capture is one JSON object per line, written by our own oracle_dump, so a tiny scanner is enough:
// find `"key":` and read the numbers up to the matching close bracket.
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
    // top2 is a list of 5-number lists: read the whole bracket flat and cut it in fives.
    v.clear();
    size_t i = pt + 7;
    while (i < line.size() && line[i] != '[') ++i;
    ++i;  // inside the outer list
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

}  // namespace

int main(int argc, char** argv) {
    const char* model = nullptr;
    const char* oracle = nullptr;
    const char* out_path = nullptr;
    int threads = 8, n_ctx = 0, max_pred = 0;
    bool oracle_arith = false;
    double rss_budget_gb = 0;   // > 0: mapped weight pages above this go back to the page cache between tokens
    for (int i = 1; i < argc; ++i) {
        auto is = [&](const char* a) { return std::strcmp(argv[i], a) == 0; };
        if ((is("-m") || is("--model")) && i + 1 < argc) model = argv[++i];
        else if (is("--oracle") && i + 1 < argc) oracle = argv[++i];
        else if (is("--out") && i + 1 < argc) out_path = argv[++i];
        else if ((is("-t") || is("--threads")) && i + 1 < argc) threads = std::atoi(argv[++i]);
        else if (is("--ctx") && i + 1 < argc) n_ctx = std::atoi(argv[++i]);
        else if (is("--max-pred") && i + 1 < argc) max_pred = std::atoi(argv[++i]);
        else if (is("--oracle-arith")) oracle_arith = true;
        else if (is("--rss-budget-gb") && i + 1 < argc) rss_budget_gb = std::atof(argv[++i]);
        else {
            std::fprintf(stderr, "usage: oracle_gate -m model.gguf --oracle capture.jsonl --out out.jsonl [-t N] [--ctx N] [--max-pred N] [--oracle-arith] [--rss-budget-gb G]\n");
            return 2;
        }
    }
    if (!model || !oracle || !out_path) {
        std::fprintf(stderr, "oracle_gate: -m, --oracle and --out are required\n");
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
            std::fprintf(stderr, "oracle_gate: cannot read a capture from %s\n", oracle);
            return 1;
        }
    }
    const int n_prompt = int(cap.prompt.size());
    int n_pred = int(cap.greedy.size());
    if (max_pred > 0 && max_pred < n_pred) n_pred = max_pred;
    if (n_ctx <= 0) n_ctx = n_prompt + n_pred + 8;

    std::string err;
    auto ref = limbo::step::Ref::open({model}, n_ctx, threads, &err);
    if (!ref) {
        std::fprintf(stderr, "oracle_gate: %s\n", err.c_str());
        return 1;
    }
    ref->set_oracle_arith(oracle_arith);
    if (rss_budget_gb > 0) ref->set_rss_budget(uint64_t(rss_budget_gb * 1073741824.0));
    const int nv = ref->n_vocab();
    std::FILE* out = std::fopen(out_path, "w");
    if (!out) {
        std::fprintf(stderr, "oracle_gate: cannot write %s\n", out_path);
        return 1;
    }
    std::fprintf(stderr, "oracle_gate: %s: %d prompt tokens, %d generated positions, %d threads, n_ctx %d, arithmetic %s\n",
                 cap.name.c_str(), n_prompt, n_pred, threads, n_ctx, oracle_arith ? "oracle (Q8 activations, f16 attention)" : "fp32");

    using clk = std::chrono::steady_clock;
    const auto t_start = clk::now();
    auto since = [](clk::time_point t) { return std::chrono::duration<double>(clk::now() - t).count(); };

    // Prompt: only the last token's logits are read, so the others skip the output head.
    const float* logits = nullptr;
    for (int i = 0; i < n_prompt; ++i) {
        const bool last = i == n_prompt - 1;
        logits = ref->forward(cap.prompt[size_t(i)], i, last);
        if ((i + 1) % 50 == 0 || last)
            std::fprintf(stderr, "  prefill %d/%d  %.1f s\n", i + 1, n_prompt, since(t_start));
    }
    const double t_prefill = since(t_start);

    int agree = 0, tie_dis = 0, big_dis = 0;
    double max_d1 = 0, max_dgap = 0;
    const auto t_gen = clk::now();
    for (int g = 0; g < n_pred; ++g) {
        int i1 = 0, i2 = 0;
        float v1 = -INFINITY, v2 = -INFINITY;
        for (int v = 0; v < nv; ++v) {
            const float x = logits[v];
            if (x > v1) {
                v2 = v1;
                i2 = i1;
                v1 = x;
                i1 = v;
            } else if (x > v2) {
                v2 = x;
                i2 = v;
            }
        }
        const std::vector<double>& o = cap.top2[size_t(g)];
        const int o1 = int(o[0]), o2 = int(o[2]);
        const double ol1 = o[1], ol2 = o[3], ogap = o[4];
        const double l_o1 = logits[o1], l_o2 = logits[o2];
        const double d1 = l_o1 - ol1, d2 = l_o2 - ol2;
        const double dgap = (l_o1 - l_o2) - ogap;   // the quantity that decides an argmax flip
        const bool ok = (i1 == cap.greedy[size_t(g)]);
        agree += ok;
        if (!ok) (ogap < 0.1 ? tie_dis : big_dis)++;
        max_d1 = std::max(max_d1, std::fabs(d1));
        max_dgap = std::max(max_dgap, std::fabs(dgap));
        const int pos = n_prompt - 1 + g;
        std::fprintf(out,
                     "{\"name\":\"%s\",\"gen\":%d,\"pos\":%d,\"agree\":%s,\"oracle_greedy\":%d,"
                     "\"limbo\":[%d,%.6f,%d,%.6f,%.6f],\"oracle\":[%d,%.6f,%d,%.6f,%.6f],"
                     "\"limbo_at_o1\":%.6f,\"limbo_at_o2\":%.6f,\"d1\":%.6f,\"d2\":%.6f,\"dgap\":%.6f}\n",
                     cap.name.c_str(), g, pos, ok ? "true" : "false", cap.greedy[size_t(g)], i1, v1, i2, v2,
                     double(v1) - double(v2), o1, ol1, o2, ol2, ogap, l_o1, l_o2, d1, d2, dgap);
        std::fflush(out);
        std::fprintf(stderr, "  gen %2d pos %4d  %s  limbo %d (%.4f) vs oracle %d (%.4f)  ogap %.4f  d1 %+.4f  dgap %+.4f\n", g,
                     pos, ok ? "ok  " : "DIFF", i1, double(v1), o1, ol1, ogap, d1, dgap);
        if (g + 1 < n_pred) logits = ref->forward(cap.greedy[size_t(g)], pos + 1);
    }
    const double t_gen_s = since(t_gen);
    std::fprintf(out,
                 "{\"summary\":true,\"name\":\"%s\",\"n_prompt\":%d,\"n_gen\":%d,\"agree\":%d,\"disagree_gap_lt_0.1\":%d,"
                 "\"disagree_gap_ge_0.1\":%d,\"max_abs_d1\":%.6f,\"max_abs_dgap\":%.6f,\"threads\":%d,\"arith\":\"%s\","
                 "\"prefill_s\":%.2f,\"gen_s\":%.2f,\"tok_per_s_prefill\":%.3f,\"tok_per_s_gen\":%.3f}\n",
                 cap.name.c_str(), n_prompt, n_pred, agree, tie_dis, big_dis, max_d1, max_dgap, threads,
                 oracle_arith ? "oracle" : "fp32", t_prefill, t_gen_s,
                 n_prompt / t_prefill, n_pred > 1 ? (n_pred - 1) / t_gen_s : 0.0);
    std::fclose(out);
    // peak resident set (anon = heap, file = touched mmap pages) from the kernel
    {
        std::ifstream st("/proc/self/status");
        std::string ln;
        while (std::getline(st, ln))
            if (ln.rfind("VmHWM", 0) == 0 || ln.rfind("RssAnon", 0) == 0 || ln.rfind("RssFile", 0) == 0)
                std::fprintf(stderr, "oracle_gate: %s\n", ln.c_str());
    }
    std::fprintf(stderr, "oracle_gate: %s: %d/%d agree, %d disagree with oracle gap < 0.1, %d with gap >= 0.1, max|d1| %.4f, max|dgap| %.4f\n",
                 cap.name.c_str(), agree, n_pred, tie_dis, big_dis, max_d1, max_dgap);
    return big_dis ? 1 : 0;
}
