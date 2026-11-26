// Copyright (c) 2026 ITERFORM. All rights reserved.
// oracle_nodes — runs the capture's prompt through llama.cpp as ONE CPU batch (no op offload, flash attention off)
// and writes every l_out-N node (or attn_residual-N with --node attn_residual; --layers a,b,c keeps only those), in full, to a binary file:  int32 n_tokens, int32 n_layers, int32 n_embd, then
// the prompt ids, then for each layer: int32 rows, rows*n_embd floats (layer 39 is cropped to the last row).
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

static std::vector<double> nums(const std::string& s, size_t from) {
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
    return out;
}

struct Grab { bool wide = false; std::string node = "l_out"; std::vector<int> want; bool wants(int l) const { if (want.empty()) return true; for (int w : want) if (w == l) return true; return false; } std::map<int, std::vector<float>> rows; std::map<int, int> nrows; int n_embd = 0; std::map<int, std::vector<int32_t>> topk; std::map<int, int> ntopk; std::vector<float> logits; };

static bool cb(struct ggml_tensor* t, bool ask, void* ud) {
    Grab* g = static_cast<Grab*>(ud);
    int layer = -1;
    if (std::strcmp(t->name, "result_output") == 0) {
        if (ask) return true;
        g->logits.resize(size_t(t->ne[0]) * size_t(t->ne[1]));
        ggml_backend_tensor_get(t, g->logits.data(), 0, g->logits.size() * sizeof(float));
        return true;
    }
    if (std::strncmp(t->name, "ffn_moe_topk-", 13) == 0 && std::sscanf(t->name + 13, "%d", &layer) == 1) {
        if (!g->wants(layer)) return !ask;
        if (ask) return true;
        // a view of the argsort result: rows are nb[1] apart, not packed
        std::vector<int32_t> buf(size_t(t->ne[0]) * size_t(t->ne[1]));
        for (int64_t r = 0; r < t->ne[1]; ++r)
            ggml_backend_tensor_get(t, buf.data() + size_t(r) * size_t(t->ne[0]), size_t(r) * t->nb[1], size_t(t->ne[0]) * sizeof(int32_t));
        auto& dst = g->topk[layer];
        dst.insert(dst.end(), buf.begin(), buf.end());      // one call per micro-batch, in order
        g->ntopk[layer] += int(t->ne[1]);
        return true;
    }
    const size_t nl_ = g->node.size();
    if (std::strncmp(t->name, g->node.c_str(), nl_) != 0 || t->name[nl_] != '-' || std::sscanf(t->name + nl_ + 1, "%d", &layer) != 1) return !ask ? true : false;
    if (!g->wants(layer)) return !ask;
    // a "(reshaped)" or "(view)" of a named node is the same data again
    if (std::strchr(t->name, ' ')) return !ask;
    if (ask) return true;
    // --wide: a [n_embd, streams, T] node (hyper-connections) is T rows of n_embd * streams
    const int ne0 = g->wide ? int(t->ne[0] * t->ne[1]) : int(t->ne[0]), ne1 = g->wide ? int(t->ne[2]) : int(t->ne[1]);
    g->n_embd = ne0;
    if (std::getenv("ORACLE_NODES_DEBUG")) std::fprintf(stderr, "node %s ne %ld %ld %ld %ld\n", t->name, (long) t->ne[0], (long) t->ne[1], (long) t->ne[2], (long) t->ne[3]);
    std::vector<float> buf(size_t(ne0) * ne1);
    ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));
    auto& dst = g->rows[layer];
    dst.insert(dst.end(), buf.begin(), buf.end());          // one call per micro-batch, in order
    g->nrows[layer] += ne1;
    return true;
}

int main(int argc, char** argv) {
    const char *model_path = nullptr, *cap_path = nullptr, *out_path = nullptr;
    int threads = 8;
    std::vector<int> want;
    std::string node = "l_out";
    bool wide = false;
    long top_k = 0;   // --top-k N: override the indexer's budget, so a short prompt exercises the block selection
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-m")) model_path = argv[++i];
        else if (!std::strcmp(argv[i], "--oracle")) cap_path = argv[++i];
        else if (!std::strcmp(argv[i], "--out")) out_path = argv[++i];
        else if (!std::strcmp(argv[i], "-t")) threads = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--node")) node = argv[++i];
        else if (!std::strcmp(argv[i], "--wide")) wide = true;
        else if (!std::strcmp(argv[i], "--top-k")) top_k = std::atol(argv[++i]);
        else if (!std::strcmp(argv[i], "--layers")) { for (char* p = std::strtok(argv[++i], ","); p; p = std::strtok(nullptr, ",")) want.push_back(std::atoi(p)); }
    }
    if (!model_path || !cap_path || !out_path) return 2;
    std::ifstream in(cap_path);
    std::string line, cap;
    while (std::getline(in, line)) if (line.find("\"prompt_ids\"") != std::string::npos) { cap = line; break; }
    std::vector<double> pid = nums(cap, cap.find("\"prompt_ids\":"));
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model_kv_override ov[2] = {};
    if (top_k > 0) {
        ov[0].tag = LLAMA_KV_OVERRIDE_TYPE_INT;
        std::snprintf(ov[0].key, sizeof(ov[0].key), "qwen4exp.attention.indexer.top_k");
        ov[0].val_i64 = top_k;
        mp.kv_overrides = ov;   // the array ends at the empty key
    }
    mp.use_extra_bufts = false;   // no weight repack: it copies the whole model into RAM; the mmap stays the only copy
    llama_model* model = llama_model_load_from_file(model_path, mp);
    if (!model) return 1;
    const int n = int(pid.size());
    Grab grab;
    grab.want = want;
    grab.node = node;
    grab.wide = wide;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = uint32_t(n + 8);
    cp.n_batch = uint32_t(std::max(2048, n + 8));
    cp.n_ubatch = 512;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.offload_kqv = false;
    cp.op_offload = false;
    cp.no_perf = true;
    cp.cb_eval = cb;
    cp.cb_eval_user_data = &grab;
    llama_context* ctx = llama_init_from_model(model, cp);
    llama_batch b = llama_batch_init(std::max(2048, n + 8), 0, 1);
    for (int i = 0; i < n; ++i) {
        b.token[i] = llama_token(pid[size_t(i)]);
        b.pos[i] = i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
        b.logits[i] = (i == n - 1) ? 1 : 0;
    }
    b.n_tokens = n;
    if (llama_decode(ctx, b) != 0) return 1;
    FILE* out = std::fopen(out_path, "wb");
    const int32_t hdr[3] = {n, int(grab.rows.size()), grab.n_embd};
    std::fwrite(hdr, 4, 3, out);
    for (double d : pid) { const int32_t v = int32_t(d); std::fwrite(&v, 4, 1, out); }
    for (auto& kv : grab.rows) {
        const int32_t layer = kv.first, rows = grab.nrows[kv.first];
        std::fwrite(&layer, 4, 1, out);
        std::fwrite(&rows, 4, 1, out);
        std::fwrite(kv.second.data(), 4, kv.second.size(), out);
    }
    const int32_t nt = int32_t(grab.topk.size());
    std::fwrite(&nt, 4, 1, out);
    for (auto& kv : grab.topk) {
        const int32_t layer = kv.first, rows = grab.ntopk[kv.first];
        std::fwrite(&layer, 4, 1, out);
        std::fwrite(&rows, 4, 1, out);
        std::fwrite(kv.second.data(), 4, kv.second.size(), out);
    }
    const int32_t nl = int32_t(grab.logits.size());
    std::fwrite(&nl, 4, 1, out);
    std::fwrite(grab.logits.data(), 4, grab.logits.size(), out);
    std::fclose(out);
    std::fprintf(stderr, "wrote %zu layers, %d tokens\n", grab.rows.size(), n);
    return 0;
}
