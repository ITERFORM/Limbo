// Copyright (c) 2026 ITERFORM. All rights reserved.
// oracle/oracle_dump.cpp — the reference captures: llama.cpp's own greedy tokens and top-2 gaps.
//
// This is a measuring device, not engine code. It links the operator's llama.cpp build (libllama) and
// is never linked into liblimbo.a — see the note in docs/CLEANROOM.md. PLAN.md §6 asks for it in W0:
// W1 compares Limbo's logits against these captures using the near-tie rule of §3.3 (a first
// difference is a tie when the reference's top-2 gap is under 0.01 nats).
//
//   oracle_dump -m model.gguf -f bench/prompts/code.txt [--n-predict 16] [--out file.jsonl]
//               [-t threads] [--ngl 0] [--gpu] [--truncate N] [--ubatch N]
//
// The prompt set is a text file: a line "@@ <name>" starts a prompt, and the bytes up to the next such
// line are its text verbatim (the file must end with a newline).
//
// The default capture is deliberately CPU-only and without flash attention, so it does not depend on which
// kernels a build happens to have: n_gpu_layers 0, flash attention off, KQV offload off, greedy
// sampling. Everything that could change the numbers is written into the first line of the output.
//
// --gpu is the capture of llama.cpp's CUDA arithmetic, as a user of llama.cpp gets it: n_gpu_layers 99, flash
// attention on, KQV offload and op_offload at the library's defaults (on). It is the reference for a
// "llama.cpp numerics" mode of Limbo's prefill (int8 activations in blocks of 32). --truncate N cuts every prompt
// to its first N tokens (after llama.cpp's own tokenisation) and names it "<name>:N", so one long prompt gives
// the 1024, 2048 and 4096 token captures; the prompt ids are in the output either way. --ubatch N changes the
// micro-batch (default 512, the library's; the prompt is read in pieces of that size): llama.cpp's own greedy tokens
// against a different micro-batch show how much of a difference to it is its own noise.
#include "llama.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

struct Prompt {
    std::string name;
    std::string text;
};

std::vector<Prompt> read_prompts(const char* path) {
    std::ifstream in(path, std::ios::binary);
    std::vector<Prompt> out;
    std::string line;
    Prompt cur;
    bool have = false;
    while (std::getline(in, line)) {
        if (line.rfind("@@ ", 0) == 0) {
            if (have) out.push_back(cur);
            cur = Prompt{line.substr(3), ""};
            have = true;
            continue;
        }
        if (have) {
            cur.text += line;
            cur.text += '\n';
        }
    }
    if (have) out.push_back(cur);
    return out;
}

uint64_t fnv1a(const std::string& s) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ull;
    }
    return h;
}

}  // namespace

int main(int argc, char** argv) {
    const char* model_path = nullptr;
    const char* prompts_path = nullptr;
    const char* out_path = nullptr;
    int n_predict = 16;
    int threads = 8;
    int ngl = 0;
    bool gpu = false;
    int truncate = 0;
    int ubatch = 512;

    for (int i = 1; i < argc; ++i) {
        auto is = [&](const char* a) { return std::strcmp(argv[i], a) == 0; };
        if ((is("-m") || is("--model")) && i + 1 < argc) model_path = argv[++i];
        else if ((is("-f") || is("--file")) && i + 1 < argc) prompts_path = argv[++i];
        else if (is("--out") && i + 1 < argc) out_path = argv[++i];
        else if (is("--n-predict") && i + 1 < argc) n_predict = std::atoi(argv[++i]);
        else if ((is("-t") || is("--threads")) && i + 1 < argc) threads = std::atoi(argv[++i]);
        else if (is("--ngl") && i + 1 < argc) ngl = std::atoi(argv[++i]);
        else if (is("--gpu")) gpu = true;
        else if (is("--truncate") && i + 1 < argc) truncate = std::atoi(argv[++i]);
        else if (is("--ubatch") && i + 1 < argc) ubatch = std::atoi(argv[++i]);
        else {
            std::fprintf(stderr,
                         "usage: oracle_dump -m model.gguf -f prompts.txt [--n-predict N] [--out f.jsonl] "
                         "[-t threads] [--ngl N] [--gpu] [--truncate N] [--ubatch N]\n");
            return 2;
        }
    }
    if (!model_path || !prompts_path) {
        std::fprintf(stderr, "oracle_dump: -m and -f are required\n");
        return 2;
    }

    if (gpu && ngl == 0) ngl = 99;
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model* model = llama_model_load_from_file(model_path, mp);
    if (!model) {
        std::fprintf(stderr, "oracle_dump: cannot load %s\n", model_path);
        return 1;
    }
    const llama_vocab* vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    const std::vector<Prompt> prompts = read_prompts(prompts_path);
    FILE* out = out_path ? std::fopen(out_path, "w") : stdout;
    if (!out) {
        std::fprintf(stderr, "oracle_dump: cannot write %s\n", out_path);
        return 1;
    }

    if (gpu) {
        std::fprintf(out,
                     "{\"settings\":{\"model\":\"%s\",\"prompts\":\"%s\",\"n_predict\":%d,\"threads\":%d,"
                     "\"n_gpu_layers\":%d,\"flash_attn\":\"on\",\"offload_kqv\":true,\"op_offload\":true,"
                     "\"n_ubatch\":\"min(%d,n_ctx)\",\"n_batch\":\"min(max(2048,n+8),n_ctx)\",\"sampling\":\"greedy\",\"add_special\":false,\"parse_special\":true,"
                     "\"n_vocab\":%d,\"prompts_count\":%zu,\"truncate\":%d,\"backend\":\"cuda\"}}\n",
                     model_path, prompts_path, n_predict, threads, ngl, ubatch, n_vocab, prompts.size(), truncate);
    } else {
        std::fprintf(out,
                     "{\"settings\":{\"model\":\"%s\",\"prompts\":\"%s\",\"n_predict\":%d,\"threads\":%d,"
                     "\"n_gpu_layers\":%d,\"flash_attn\":\"off\",\"offload_kqv\":false,\"sampling\":\"greedy\","
                     "\"add_special\":false,\"parse_special\":true,\"n_vocab\":%d,\"prompts_count\":%zu}}\n",
                     model_path, prompts_path, n_predict, threads, ngl, n_vocab, prompts.size());
    }

    for (const Prompt& p : prompts) {
        std::vector<llama_token> toks(p.text.size() + 16);
        int n = llama_tokenize(vocab, p.text.data(), int(p.text.size()), toks.data(), int(toks.size()), false, true);
        if (n < 0) {
            toks.resize(size_t(-n));
            n = llama_tokenize(vocab, p.text.data(), int(p.text.size()), toks.data(), int(toks.size()), false, true);
        }
        if (n <= 0) {
            std::fprintf(stderr, "oracle_dump: prompt %s tokenized to %d tokens\n", p.name.c_str(), n);
            continue;
        }
        toks.resize(size_t(n));
        std::string pname = p.name;
        if (truncate > 0) {
            if (n < truncate) {
                std::fprintf(stderr, "oracle_dump: %s has %d tokens, fewer than --truncate %d\n", p.name.c_str(), n, truncate);
                continue;
            }
            n = truncate;
            toks.resize(size_t(n));
            pname += ":" + std::to_string(truncate);
        }

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = uint32_t(n + n_predict + 8);
        cp.n_batch = uint32_t(std::max(2048, n + 8));
        cp.n_ubatch = uint32_t(ubatch);
        cp.n_threads = threads;
        cp.n_threads_batch = threads;
        if (gpu) {
            cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;  // offload_kqv and op_offload stay at the defaults (on)
        } else {
            cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
            cp.offload_kqv = false;
        }
        cp.no_perf = true;
        llama_context* ctx = llama_init_from_model(model, cp);
        if (!ctx) {
            std::fprintf(stderr, "oracle_dump: cannot create a context for %s\n", p.name.c_str());
            continue;
        }

        llama_batch b = llama_batch_init(std::max(2048, n + 8), 0, 1);
        for (int i = 0; i < n; ++i) {
            b.token[i] = toks[size_t(i)];
            b.pos[i] = i;
            b.n_seq_id[i] = 1;
            b.seq_id[i][0] = 0;
            b.logits[i] = (i == n - 1) ? 1 : 0;
        }
        b.n_tokens = n;
        if (llama_decode(ctx, b) != 0) {
            std::fprintf(stderr, "oracle_dump: decode failed for %s\n", p.name.c_str());
            llama_batch_free(b);
            llama_free(ctx);
            continue;
        }

        std::vector<int32_t> greedy;
        std::vector<std::array<double, 5>> top2;  // id1, logit1, id2, logit2, gap
        for (int step = 0; step < n_predict; ++step) {
            const float* lg = llama_get_logits_ith(ctx, -1);
            if (!lg) break;
            int i1 = 0, i2 = 0;
            float v1 = -INFINITY, v2 = -INFINITY;
            for (int v = 0; v < n_vocab; ++v) {
                const float x = lg[v];
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
            greedy.push_back(i1);
            top2.push_back({double(i1), double(v1), double(i2), double(v2), double(v1) - double(v2)});

            b.n_tokens = 0;
            b.token[0] = i1;
            b.pos[0] = n + step;
            b.n_seq_id[0] = 1;
            b.seq_id[0][0] = 0;
            b.logits[0] = 1;
            b.n_tokens = 1;
            if (llama_decode(ctx, b) != 0) break;
        }

        std::fprintf(out, "{\"name\":\"%s\",\"prompt_bytes\":%zu,\"prompt_fnv\":\"%016llx\",\"n_prompt\":%d,"
                          "\"prompt_ids\":[",
                     pname.c_str(), p.text.size(), (unsigned long long)fnv1a(p.text), n);
        for (int i = 0; i < n; ++i) std::fprintf(out, "%s%d", i ? "," : "", int(toks[size_t(i)]));
        std::fprintf(out, "],\"n_predict\":%zu,\"greedy\":[", greedy.size());
        for (size_t i = 0; i < greedy.size(); ++i) std::fprintf(out, "%s%d", i ? "," : "", greedy[i]);
        std::fprintf(out, "],\"top2\":[");
        for (size_t i = 0; i < top2.size(); ++i)
            std::fprintf(out, "%s[%d,%.6f,%d,%.6f,%.6f]", i ? "," : "", int(top2[i][0]), top2[i][1], int(top2[i][2]),
                         top2[i][3], top2[i][4]);
        std::fprintf(out, "]}\n");
        std::fflush(out);

        llama_batch_free(b);
        llama_free(ctx);
        std::fprintf(stderr, "oracle_dump: %s: %d prompt tokens, %zu generated\n", pname.c_str(), n, greedy.size());
    }

    if (out_path) std::fclose(out);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
