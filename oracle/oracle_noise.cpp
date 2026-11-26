// Copyright (c) 2026 ITERFORM. All rights reserved.
// oracle_noise — llama.cpp against itself: the capture's prompt and greedy tokens, teacher-forced, but with the
// prompt decoded ONE token per llama_decode (like a decode step) instead of as one batch. Everything else is
// oracle_dump's context. It writes the same JSONL shape as the capture, with top2 = [o1, logit(o1), o2, logit(o2),
// gap] taken at the CAPTURE's ids, so the two files diff position by position.
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static std::vector<double> nums(const std::string& s, size_t from, size_t* end) {
    std::vector<double> out;
    size_t i = from;
    while (i < s.size() && s[i] != '[') ++i;
    ++i;
    while (i < s.size() && s[i] != ']') {
        char* e = nullptr;
        const double v = std::strtod(s.c_str() + i, &e);
        if (e == s.c_str() + i) { ++i; continue; }
        out.push_back(v);
        i = size_t(e - s.c_str());
    }
    if (end) *end = i + 1;
    return out;
}

int main(int argc, char** argv) {
    const char *model_path = nullptr, *cap_path = nullptr, *out_path = nullptr;
    int threads = 8, chunk = 1;
    bool op_offload = true;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-m")) model_path = argv[++i];
        else if (!std::strcmp(argv[i], "--oracle")) cap_path = argv[++i];
        else if (!std::strcmp(argv[i], "--out")) out_path = argv[++i];
        else if (!std::strcmp(argv[i], "-t")) threads = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--no-op-offload")) op_offload = false;
        else if (!std::strcmp(argv[i], "--chunk")) chunk = std::atoi(argv[++i]);   // prompt tokens per llama_decode
    }
    if (!model_path || !cap_path || !out_path) return 2;
    std::ifstream in(cap_path);
    std::string line, cap;
    while (std::getline(in, line)) if (line.find("\"prompt_ids\"") != std::string::npos) { cap = line; break; }
    const size_t pn = cap.find("\"name\":\"");
    const std::string name = cap.substr(pn + 8, cap.find('"', pn + 8) - (pn + 8));
    std::vector<double> pid = nums(cap, cap.find("\"prompt_ids\":"), nullptr);
    std::vector<double> gr = nums(cap, cap.find("\"greedy\":"), nullptr);
    std::vector<std::vector<double>> t2;
    {
        size_t i = cap.find("\"top2\":") + 7;
        while (i < cap.size() && cap[i] != '[') ++i;
        ++i;
        while (i < cap.size()) {
            while (i < cap.size() && cap[i] != '[' && cap[i] != ']') ++i;
            if (i >= cap.size() || cap[i] == ']') break;
            size_t e = 0;
            t2.push_back(nums(cap, i, &e));
            i = e;
        }
    }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.use_extra_bufts = false;   // no weight repack: it copies the whole model into RAM (the 54 GiB coder file does not fit)
    llama_model* model = llama_model_load_from_file(model_path, mp);
    if (!model) return 1;
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int n = int(pid.size()), n_pred = int(gr.size());
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = uint32_t(n + n_pred + 8);
    cp.n_batch = uint32_t(std::max(2048, n + 8));
    cp.n_ubatch = 512;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.offload_kqv = false;
    cp.no_perf = true;
    cp.op_offload = op_offload;
    llama_context* ctx = llama_init_from_model(model, cp);
    llama_batch b = llama_batch_init(std::max(2048, n + 8), 0, 1);
    auto decode = [&](int first, int count, int last_logits_idx) {
        for (int i = 0; i < count; ++i) {
            b.token[i] = (first < n) ? llama_token(pid[size_t(first + i)]) : llama_token(gr[size_t(first - n + i)]);
            b.pos[i] = first + i;
            b.n_seq_id[i] = 1;
            b.seq_id[0 + 0][0] = 0;
            b.seq_id[i][0] = 0;
            b.logits[i] = (i == last_logits_idx) ? 1 : 0;
        }
        b.n_tokens = count;
        return llama_decode(ctx, b);
    };
    for (int i = 0; i < n; i += chunk) {
        const int c = std::min(chunk, n - i);
        if (decode(i, c, i + c == n ? c - 1 : -1) != 0) return 1;
    }
    FILE* out = std::fopen(out_path, "w");
    std::fprintf(out, "{\"name\":\"%s\",\"n_prompt\":%d,\"chunk\":%d,\"top2\":[", name.c_str(), n, chunk);
    for (int g = 0; g < n_pred; ++g) {
        const float* lg = llama_get_logits_ith(ctx, -1);
        const int o1 = int(t2[size_t(g)][0]), o2 = int(t2[size_t(g)][2]);
        int a1 = 0;
        for (int v = 1; v < n_vocab; ++v) if (lg[v] > lg[a1]) a1 = v;
        std::fprintf(out, "%s[%d,%.6f,%d,%.6f,%.6f,%d]", g ? "," : "", o1, lg[o1], o2, lg[o2], double(lg[o1]) - double(lg[o2]), a1);
        if (g + 1 < n_pred && decode(n + g, 1, 0) != 0) return 1;
    }
    std::fprintf(out, "]}\n");
    std::fclose(out);
    return 0;
}
