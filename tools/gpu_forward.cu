// Copyright (c) 2026 ITERFORM. All rights reserved.
// tools/gpu_forward.cu — the rung-1 forward pass on the GPU, held to the CPU reference.
//
//   gpu_forward -m model.gguf -f bench/prompts/prose.txt --names hi,prose,code [--parity 32] [--gen 64]
//               [--ctx 512] [--threads 8] [--no-cpu] [--trace0] [--trace-pos N] [--oracle-dir bench/oracle]
//               [--min-free-gb 14.6] [--planes N] [--chunk N] [--prefill-check 1,2,3] [--bench-prefill 512,2048]
//               [--cpu-dump DIR | --cpu-load DIR] [--no-graphs] [--layer-diff]
//
// Per prompt:
//   1. parity: limbo::step::Ref (CPU) and limbo::cuda::Gpu run the SAME token sequence — the prompt, then
//      --parity greedy tokens taken from the CPU run (teacher forcing) — and every position's logits are
//      compared: max |dlogit| over the vocabulary, and whether the GPU's top-1 is the CPU's.
//   1b. --prefill-check P,...: for each prompt the logits after Gpu::prefill with P activation planes (0 = the
//      token-by-token fp32 path) against the token-by-token logits (top-1, max|dlogit|) and against the CPU's,
//      and the greedy continuation of --parity tokens from each state against the CPU's.
//   2. generation: the GPU alone, free-running greedy for --gen tokens from the prompt, decoded to text,
//      timed (Gpu::prefill with --planes, then decode), and compared with the CPU's greedy tokens and the
//      oracle's capture where there is one.
//   --cpu-dump DIR runs only the CPU reference (no GPU touched, so no lock needed): the prompt, then --parity
//   greedy tokens, and saves the logits at the last prompt position and the greedy ids per prompt. --cpu-load DIR
//   feeds those files to the checks above instead of running the CPU beside the GPU.
//   3. --bench-prefill N,...: Gpu::prefill throughput over N tokens of bench/prompts/long.txt (repeated).
//
// The card is shared with the desktop: open() refuses to load unless cudaMemGetInfo shows --min-free-gb free.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "limbo/cuda/gpu.hpp"
#include "limbo/gguf.hpp"
#include "limbo/model.hpp"
#include "limbo/step.hpp"
#include "limbo/tok.hpp"

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

const char* kRung1 =
    "/home/alca/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-MTP-GGUF/snapshots/"
    "5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf";

// The same prompt-file format as oracle/oracle_dump and tools/ref_forward: "@@ <name>" then the text.
std::string read_prompt(const std::string& path, const std::string& want) {
    std::ifstream in(path, std::ios::binary);
    std::string line, cur_name, cur_text, found;
    bool have = false, got = false;
    while (std::getline(in, line)) {
        if (line.rfind("@@ ", 0) == 0) {
            if (have && cur_name == want) {
                found = cur_text;
                got = true;
            }
            cur_name = line.substr(3);
            cur_text.clear();
            have = true;
            continue;
        }
        if (have) {
            cur_text += line;
            cur_text += '\n';
        }
    }
    if (have && cur_name == want) {
        found = cur_text;
        got = true;
    }
    return got ? found : std::string();
}

std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out;
    size_t at = 0;
    while (at <= s.size()) {
        size_t p = s.find(',', at);
        if (p == std::string::npos) p = s.size();
        if (p > at) out.push_back(s.substr(at, p - at));
        at = p + 1;
    }
    return out;
}

// "greedy":[1,2,3] out of one oracle line.
std::vector<int32_t> oracle_greedy(const std::string& path, const std::string& name) {
    std::ifstream in(path);
    std::string line;
    std::vector<int32_t> out;
    const std::string tag = "\"name\":\"" + name + "\"";
    while (std::getline(in, line)) {
        if (line.find(tag) == std::string::npos) continue;
        const size_t g = line.find("\"greedy\":[");
        if (g == std::string::npos) continue;
        size_t i = g + 10;
        while (i < line.size() && line[i] != ']') {
            out.push_back(static_cast<int32_t>(std::strtol(line.c_str() + i, nullptr, 10)));
            while (i < line.size() && line[i] != ',' && line[i] != ']') ++i;
            if (i < line.size() && line[i] == ',') ++i;
        }
    }
    return out;
}

std::string escape(const std::string& s) {
    std::string o;
    for (unsigned char ch : s) {
        if (ch == '\n') o += "\\n";
        else if (ch == '\t') o += "\\t";
        else if (ch == '\\') o += "\\\\";
        else if (ch < 0x20) {
            char b[8];
            std::snprintf(b, sizeof b, "\\x%02x", ch);
            o += b;
        } else o += static_cast<char>(ch);
    }
    return o;
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

size_t vram_free() {
    size_t f = 0, t = 0;
    cudaMemGetInfo(&f, &t);
    return f;
}

struct PosRow {
    int pos, token;
    bool prompt;
    double max_abs;
    int at;
    int cpu_top, gpu_top;
    float cpu_gap;
};

}  // namespace

int main(int argc, char** argv) {
    std::string model_path = kRung1, prompt_file = "bench/prompts/prose.txt", oracle_dir = "bench/oracle";
    std::string names = "hi,prose,code";
    int n_parity = 32, n_gen = 64, n_ctx = 512, threads = 8;
    double min_free_gb = 14.6;
    bool use_cpu = true, trace0 = false;
    int trace_pos = -1;
    std::string prompt_dir = "bench/prompts";
    int planes = -1, chunk = 0;  // -1 / 0: the engine's defaults
    std::string check_planes, bench_prefill;
    bool profile = false;
    std::string cpu_dump, cpu_load;
    bool graphs = true, layer_diff = false;

    for (int i = 1; i < argc; ++i) {
        auto is = [&](const char* a) { return std::strcmp(argv[i], a) == 0; };
        auto val = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
        if (is("-m") || is("--model")) model_path = val();
        else if (is("-f") || is("--file")) prompt_file = val();
        else if (is("--prompt-dir")) prompt_dir = val();
        else if (is("--names")) names = val();
        else if (is("--parity")) n_parity = std::atoi(val());
        else if (is("--gen")) n_gen = std::atoi(val());
        else if (is("--ctx")) n_ctx = std::atoi(val());
        else if (is("--threads") || is("-t")) threads = std::atoi(val());
        else if (is("--min-free-gb")) min_free_gb = std::atof(val());
        else if (is("--oracle-dir")) oracle_dir = val();
        else if (is("--no-cpu")) use_cpu = false;
        else if (is("--trace0")) trace0 = true;
        else if (is("--trace-pos")) trace_pos = std::atoi(val());
        else if (is("--planes")) planes = std::atoi(val());
        else if (is("--chunk")) chunk = std::atoi(val());
        else if (is("--prefill-check")) check_planes = val();
        else if (is("--bench-prefill")) bench_prefill = val();
        else if (is("--profile")) profile = true;
        else if (is("--no-graphs")) graphs = false;
        else if (is("--layer-diff")) layer_diff = true;
        else if (is("--cpu-dump")) cpu_dump = val();
        else if (is("--cpu-load")) cpu_load = val();
        else {
            std::fprintf(stderr,
                         "usage: gpu_forward -m model.gguf [--prompt-dir D | -f prompts.txt] --names a,b [--parity N] [--gen N]\n"
                         "       [--ctx N] [--threads N] [--no-cpu] [--trace0] [--trace-pos N] [--oracle-dir D] [--min-free-gb G]\n"
                         "       [--planes N] [--chunk N] [--prefill-check P,P,..] [--bench-prefill N,N,..]\n");
            return 2;
        }
    }
    const int parity_req = n_parity;
    if (!use_cpu || !cpu_load.empty()) n_parity = 0;
    if (!cpu_load.empty()) use_cpu = false;

    // ---- the model: through the CPU reference when it is needed (it owns its Model), else on its own ----
    std::string err;
    std::unique_ptr<limbo::step::Ref> ref;
    std::unique_ptr<limbo::model::Model> own_model;
    const limbo::model::Model* model = nullptr;
    if (use_cpu) {
        ref = limbo::step::Ref::open({model_path}, n_ctx, threads, &err);
        if (!ref) {
            std::fprintf(stderr, "gpu_forward: %s\n", err.c_str());
            return 1;
        }
        model = &ref->model();
    } else {
        own_model = limbo::model::Model::open({model_path}, &err);
        if (!own_model) {
            std::fprintf(stderr, "gpu_forward: %s\n", err.c_str());
            return 1;
        }
        model = own_model.get();
    }
    limbo::tok::Vocab vocab;
    if (!limbo::tok::Vocab::load(*model->shards()[0], vocab, &err)) {
        std::fprintf(stderr, "gpu_forward: %s\n", err.c_str());
        return 1;
    }
    const limbo::tok::Tokenizer tok(vocab);

    if (!cpu_dump.empty()) {
        if (!ref) {
            std::fprintf(stderr, "gpu_forward: --cpu-dump needs the CPU reference (no --no-cpu)\n");
            return 2;
        }
        for (const std::string& name : split_csv(names)) {
            std::vector<int32_t> ids;
            limbo::tok::Options o;
            o.parse_special = true;
            if (name.rfind("long:", 0) == 0) {
                std::ifstream in(prompt_dir + "/long.txt", std::ios::binary);
                std::stringstream ss;
                ss << in.rdbuf();
                std::vector<int32_t> base;
                tok.encode(ss.str(), base, o);
                for (int i = 0, n = std::atoi(name.c_str() + 5); i < n && !base.empty(); ++i) ids.push_back(base[static_cast<size_t>(i) % base.size()]);
            } else {
                std::string text = read_prompt(prompt_file, name);
                if (text.empty()) text = read_prompt(prompt_dir + "/" + name + ".txt", name);
                tok.encode(text, ids, o);
            }
            const int n_prompt = static_cast<int>(ids.size()), nv = ref->config().n_vocab;
            ref->reset();
            std::vector<float> last(static_cast<size_t>(nv));
            std::vector<int32_t> greedy;
            int token = ids[0];
            const auto t0 = Clock::now();
            for (int pos = 0; pos < n_prompt + n_parity; ++pos) {
                if (pos < n_prompt) token = ids[static_cast<size_t>(pos)];
                const float* cl = ref->forward(token, pos);
                if (pos == n_prompt - 1) last.assign(cl, cl + nv);
                if (pos >= n_prompt - 1 && pos < n_prompt + n_parity - 1) {
                    token = top2(cl, nv).i1;
                    greedy.push_back(token);
                }
            }
            std::string fname = name;
            for (char& ch : fname) if (ch == ':') ch = '_';
            const std::string path = cpu_dump + "/" + fname + ".cpuref";
            std::ofstream f(path, std::ios::binary);
            const int32_t hdr[3] = {n_prompt, static_cast<int32_t>(greedy.size()), nv};
            f.write(reinterpret_cast<const char*>(hdr), sizeof hdr);
            f.write(reinterpret_cast<const char*>(last.data()), static_cast<std::streamsize>(last.size() * 4));
            f.write(reinterpret_cast<const char*>(greedy.data()), static_cast<std::streamsize>(greedy.size() * 4));
            std::printf("cpu-dump '%s': %d prompt tokens + %zu greedy in %.1f s -> %s\n", name.c_str(), n_prompt, greedy.size(),
                        ms_since(t0) / 1e3, path.c_str());
            std::fflush(stdout);
        }
        return 0;
    }

    // ---- the GPU: refuses unless the card has room beside the desktop -------------------------------
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        std::fprintf(stderr, "gpu_forward: no CUDA device\n");
        return 3;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const auto t_open = Clock::now();
    auto gpu = limbo::cuda::Gpu::open(*model, n_ctx, static_cast<uint64_t>(min_free_gb * 1e9), &err);
    if (!gpu) {
        std::fprintf(stderr, "gpu_forward: %s\n", err.c_str());
        return 3;
    }
    const double open_ms = ms_since(t_open);
    std::printf("gpu: %s sm_%d%d; cudaMemGetInfo free %.2f GB before the load, %.2f GB after\n", prop.name, prop.major,
                prop.minor, double(gpu->free_before_bytes()) / 1e9, double(gpu->free_after_bytes()) / 1e9);
    std::printf("vram: allocated by the engine %.3f GB = weights %.3f GB + KV cache/state/scratch %.3f GB (ctx %d)\n",
                double(gpu->vram_bytes()) / 1e9, double(gpu->weight_bytes()) / 1e9,
                double(gpu->vram_bytes() - gpu->weight_bytes()) / 1e9, n_ctx);
    std::printf("load: weights to VRAM %.2f s (%.2f GB/s); open() total %.2f s\n", gpu->load_ms() / 1e3,
                double(gpu->weight_bytes()) / 1e9 / (gpu->load_ms() / 1e3), open_ms / 1e3);
    std::fflush(stdout);
    size_t min_free = gpu->free_after_bytes();
    gpu->set_graphs(graphs);
    if (planes < 0) planes = gpu->prefill_planes();
    if (chunk <= 0) chunk = gpu->prefill_chunk();
    if (!gpu->set_prefill(planes, chunk)) {
        std::fprintf(stderr, "gpu_forward: bad --planes/--chunk\n");
        return 2;
    }
    std::printf("prefill: %s, chunk %d\n", gpu->prefill_planes() == 0 ? "token by token (fp32 GEMV)" : (std::to_string(gpu->prefill_planes()) + " int8 plane(s) on the tensor cores").c_str(), gpu->prefill_chunk());

    const int nv = gpu->n_vocab();
    std::vector<float> gl(static_cast<size_t>(nv));

    for (const std::string& name : split_csv(names.empty() ? std::string() : names)) {
        std::vector<int32_t> ids;
        limbo::tok::Options o;
        o.parse_special = true;
        if (name.rfind("long:", 0) == 0) {
            // "long:N": the first N tokens of bench/prompts/long.txt, repeated if it is shorter
            std::ifstream in(prompt_dir + "/long.txt", std::ios::binary);
            std::stringstream ss;
            ss << in.rdbuf();
            std::vector<int32_t> base;
            tok.encode(ss.str(), base, o);
            const int n = std::atoi(name.c_str() + 5);
            if (base.empty() || n <= 0) {
                std::fprintf(stderr, "gpu_forward: bad prompt '%s'\n", name.c_str());
                return 2;
            }
            for (int i = 0; i < n; ++i) ids.push_back(base[static_cast<size_t>(i) % base.size()]);
        } else {
            std::string text = read_prompt(prompt_file, name);
            if (text.empty()) text = read_prompt(prompt_dir + "/" + name + ".txt", name);
            if (text.empty()) {
                std::fprintf(stderr, "gpu_forward: no prompt named '%s' in %s or %s/%s.txt\n", name.c_str(),
                             prompt_file.c_str(), prompt_dir.c_str(), name.c_str());
                return 2;
            }
            tok.encode(text, ids, o);
        }
        const int n_prompt = static_cast<int>(ids.size());
        std::printf("\n=== prompt '%s': %d tokens ===\n", name.c_str(), n_prompt);
        if (n_prompt + std::max(n_parity, n_gen) + 1 > n_ctx) {
            std::fprintf(stderr, "gpu_forward: prompt + generation exceed --ctx %d\n", n_ctx);
            return 2;
        }

        std::vector<int32_t> cpu_greedy;
        std::vector<float> cpu_last;  // the CPU's logits at the last prompt position
        if (!cpu_load.empty()) {
            std::string fname = name;
            for (char& ch : fname) if (ch == ':') ch = '_';
            std::ifstream f(cpu_load + "/" + fname + ".cpuref", std::ios::binary);
            int32_t hdr[3] = {0, 0, 0};
            if (f.read(reinterpret_cast<char*>(hdr), sizeof hdr) && hdr[0] == n_prompt && hdr[2] == nv) {
                cpu_last.resize(static_cast<size_t>(nv));
                cpu_greedy.resize(static_cast<size_t>(hdr[1]));
                f.read(reinterpret_cast<char*>(cpu_last.data()), static_cast<std::streamsize>(cpu_last.size() * 4));
                f.read(reinterpret_cast<char*>(cpu_greedy.data()), static_cast<std::streamsize>(cpu_greedy.size() * 4));
                std::printf("   CPU reference loaded from %s: %d greedy tokens\n", (cpu_load + "/" + fname + ".cpuref").c_str(), hdr[1]);
            } else {
                std::printf("   no matching CPU reference file for '%s' in %s\n", name.c_str(), cpu_load.c_str());
            }
        }

        // ---- 1. teacher-forced parity ---------------------------------------------------------------
        if (n_parity > 0) {
            ref->reset();
            gpu->reset();
            std::vector<PosRow> rows;
            std::vector<std::vector<float>> gtrace, gmoe;
            double cpu_ms = 0, gpu_ms = 0;
            int token = ids[0];
            const int total = n_prompt + n_parity;
            for (int pos = 0; pos < total; ++pos) {
                if (pos < n_prompt) token = ids[static_cast<size_t>(pos)];
                const bool tr = (trace0 && pos == 0) || pos == trace_pos;
                if (tr) {
                    ref->trace().on = true;
                    ref->trace().all_pos = true;
                    gpu->set_layer_trace(&gtrace);
                    gpu->set_moe_trace(&gmoe);
                }
                auto t0 = Clock::now();
                const float* cl = ref->forward(token, pos);
                cpu_ms += ms_since(t0);
                t0 = Clock::now();
                if (!gpu->step(token, pos, true)) {
                    std::fprintf(stderr, "gpu_forward: a launch failed at pos %d\n", pos);
                    return 1;
                }
                if (!gpu->read_logits(gl.data())) {
                    std::fprintf(stderr, "gpu_forward: cannot read logits at pos %d\n", pos);
                    return 1;
                }
                gpu_ms += ms_since(t0);
                min_free = std::min(min_free, vram_free());
                if (tr) {
                    gpu->set_layer_trace(nullptr);
                    gpu->set_moe_trace(nullptr);
                    ref->trace().on = false;
                    ref->trace().all_pos = false;
                    std::printf("-- layer by layer at position %d (residual stream after each layer, GPU vs CPU; routed experts)\n", pos);
                    const auto& lo = ref->trace().l_out_pos[static_cast<size_t>(pos)];
                    const auto& mo = ref->trace().moe_pos[static_cast<size_t>(pos)];
                    for (size_t l = 0; l < gtrace.size() && l < lo.size(); ++l) {
                        double mx = 0, den = 0, num = 0;
                        for (size_t i = 0; i < gtrace[l].size() && i < lo[l].size(); ++i) {
                            mx = std::max(mx, std::fabs(double(gtrace[l][i]) - double(lo[l][i])));
                            num += (double(gtrace[l][i]) - double(lo[l][i])) * (double(gtrace[l][i]) - double(lo[l][i]));
                            den += double(lo[l][i]) * double(lo[l][i]);
                        }
                        // experts: the CPU's ids/weights (8 + 8) against the GPU's, and how close the 8th and 9th
                        // router logits are on the GPU (a near tie is where summation order can flip the choice)
                        std::string ex = "no moe trace";
                        if (l < mo.size() && mo[l].size() >= 16 && l < gmoe.size() && gmoe[l].size() >= 16 + 256) {
                            std::vector<int> cid(8), gid(8);
                            double wd = 0;
                            for (int i = 0; i < 8; ++i) {
                                cid[static_cast<size_t>(i)] = static_cast<int>(mo[l][static_cast<size_t>(i)]);
                                gid[static_cast<size_t>(i)] = static_cast<int>(gmoe[l][static_cast<size_t>(i)]);
                            }
                            std::vector<int> cs = cid, gs = gid;
                            std::sort(cs.begin(), cs.end());
                            std::sort(gs.begin(), gs.end());
                            std::vector<float> lg(gmoe[l].begin() + 16, gmoe[l].end());
                            std::sort(lg.begin(), lg.end(), std::greater<float>());
                            if (cs == gs)
                                for (int i = 0; i < 8; ++i)
                                    for (int j = 0; j < 8; ++j)
                                        if (cid[static_cast<size_t>(i)] == gid[static_cast<size_t>(j)])
                                            wd = std::max(wd, std::fabs(double(mo[l][static_cast<size_t>(8 + i)]) - double(gmoe[l][static_cast<size_t>(8 + j)])));
                            char b[200];
                            std::snprintf(b, sizeof b, "experts %s, max|dw| %.1e, gpu logit gap 8th-9th %.2e", cs == gs ? (cid == gid ? "same" : "same set, order differs") : "SET DIFFERS", wd,
                                          double(lg[7]) - double(lg[8]));
                            ex = b;
                        }
                        std::printf("   layer %2zu  max|d| %.3e  rel L2 %.3e  (%s)  %s\n", l, mx, std::sqrt(num / std::max(den, 1e-30)),
                                    (l % 4 == 3) ? "attention" : "gdn", ex.c_str());
                    }
                }

                if (pos == n_prompt - 1) cpu_last.assign(cl, cl + nv);
                const Top2 ct = top2(cl, nv);
                const Top2 gt = top2(gl.data(), nv);
                PosRow r{};
                r.pos = pos;
                r.token = token;
                r.prompt = pos < n_prompt;
                for (int v = 0; v < nv; ++v) {
                    const double d = std::fabs(double(cl[v]) - double(gl[static_cast<size_t>(v)]));
                    if (d > r.max_abs) {
                        r.max_abs = d;
                        r.at = v;
                    }
                }
                r.cpu_top = ct.i1;
                r.gpu_top = gt.i1;
                r.cpu_gap = ct.v1 - ct.v2;
                rows.push_back(r);
                // the device argmax must agree with the host's reading of the same logits
                if (pos == total - 1 || pos == n_prompt - 1) {
                    const int32_t dev = gpu->argmax();
                    if (dev != gt.i1) std::printf("   NOTE device argmax %d != host argmax %d at pos %d\n", dev, gt.i1, pos);
                }
                if (pos >= n_prompt - 1 && pos < total - 1) {
                    token = ct.i1;  // the CPU's greedy token feeds the next position
                    cpu_greedy.push_back(ct.i1);
                }
            }
            std::printf("-- parity, teacher-forced: %d prompt positions + %d CPU-greedy positions; CPU %.1f s, GPU %.1f s "
                        "(incl. logits copy)\n",
                        n_prompt, n_parity, cpu_ms / 1e3, gpu_ms / 1e3);
            std::printf("   pos  kind    token   max|dlogit|  at-token   cpu-top1  gpu-top1  agree  cpu-gap\n");
            int agree = 0, disagree = 0, gen_agree = 0, gen_n = 0;
            double worst = 0, sum = 0;
            std::vector<double> maxes;
            for (const PosRow& r : rows) {
                const bool same = r.cpu_top == r.gpu_top;
                agree += same;
                disagree += !same;
                worst = std::max(worst, r.max_abs);
                sum += r.max_abs;
                maxes.push_back(r.max_abs);
                if (!r.prompt) {
                    ++gen_n;
                    gen_agree += same;
                }
                std::printf("  %4d  %-6s %7d   %.3e  %8d  %8d  %8d  %-5s  %.4f\n", r.pos, r.prompt ? "prompt" : "gen",
                            r.token, r.max_abs, r.at, r.cpu_top, r.gpu_top, same ? "yes" : "NO", r.cpu_gap);
            }
            std::sort(maxes.begin(), maxes.end());
            std::printf("   summary '%s': top-1 agreement %d/%zu (generated positions %d/%d); max|dlogit| worst %.3e, "
                        "median %.3e, mean %.3e\n",
                        name.c_str(), agree, rows.size(), gen_agree, gen_n, worst, maxes[maxes.size() / 2], sum / double(rows.size()));
            for (const PosRow& r : rows)
                if (r.cpu_top != r.gpu_top)
                    std::printf("   disagreement at pos %d: cpu %d vs gpu %d, cpu gap %.4f nats%s\n", r.pos, r.cpu_top,
                                r.gpu_top, r.cpu_gap, r.cpu_gap < 0.01f ? " (a near tie by PLAN.md 3.3)" : "");
            (void)disagree;
            std::fflush(stdout);
        }

        // ---- 1b. prefill against the token-by-token path ---------------------------------------------
        if (!check_planes.empty()) {
            const int keep_planes = gpu->prefill_planes();
            std::vector<float> lt(static_cast<size_t>(nv)), lb(static_cast<size_t>(nv));
            const int n_cont = std::max(parity_req, 16);
            auto run_prefill = [&](int pl, std::vector<float>& out, std::vector<int32_t>& cont, double* ms) {
                gpu->set_prefill(pl, chunk);
                gpu->reset();
                gpu->sync();
                const auto t0 = Clock::now();
                const bool ok = gpu->prefill(ids.data(), n_prompt, 0, true);
                gpu->sync();
                *ms = ms_since(t0);
                if (!ok) return false;
                if (!gpu->read_logits(out.data())) return false;
                cont.clear();
                cont.push_back(gpu->argmax());
                for (int k = 1; k < n_cont; ++k) {
                    if (!gpu->step(cont.back(), n_prompt + k - 1, true)) return false;
                    cont.push_back(gpu->argmax());
                }
                return true;
            };
            std::vector<int32_t> cont_t;
            double ms_t = 0;
            if (!run_prefill(0, lt, cont_t, &ms_t)) {
                std::fprintf(stderr, "gpu_forward: token-by-token prefill failed\n");
                return 1;
            }
            const Top2 tt = top2(lt.data(), nv);
            std::printf("-- prefill check '%s': %d prompt tokens; reference = token by token on the GPU (%.0f ms), top-1 %d gap %.4f\n",
                        name.c_str(), n_prompt, ms_t, tt.i1, tt.v1 - tt.v2);
            if (!cpu_last.empty()) {
                double mx = 0;
                for (int v = 0; v < nv; ++v) mx = std::max(mx, std::fabs(double(cpu_last[static_cast<size_t>(v)]) - double(lt[static_cast<size_t>(v)])));
                std::printf("   token by token vs CPU: max|dlogit| %.3e, top-1 %s\n", mx, top2(cpu_last.data(), nv).i1 == tt.i1 ? "equal" : "DIFFERENT");
            }
            for (const std::string& ps : split_csv(check_planes)) {
                const int pl = std::atoi(ps.c_str());
                std::vector<int32_t> cont_b;
                double ms_b = 0;
                if (!run_prefill(pl, lb, cont_b, &ms_b)) {
                    std::fprintf(stderr, "gpu_forward: prefill with %d planes failed\n", pl);
                    return 1;
                }
                double mx = 0, sq = 0;
                for (int v = 0; v < nv; ++v) {
                    const double d = std::fabs(double(lb[static_cast<size_t>(v)]) - double(lt[static_cast<size_t>(v)]));
                    mx = std::max(mx, d);
                    sq += d * d;
                }
                const Top2 bt = top2(lb.data(), nv);
                size_t first = cont_b.size();
                for (size_t i = 0; i < cont_b.size(); ++i)
                    if (cont_b[i] != cont_t[i]) {
                        first = i;
                        break;
                    }
                std::printf("   planes %d: %.0f ms (%.0f tok/s); top-1 %s; max|dlogit| %.3e rms %.3e vs token-by-token; "
                            "greedy continuation (%d tokens) vs token-by-token: %s",
                            pl, ms_b, double(n_prompt) / (ms_b / 1e3), bt.i1 == tt.i1 ? "equal" : "DIFFERENT", mx,
                            std::sqrt(sq / nv), n_cont, first == cont_b.size() ? "identical" : ("differs at index " + std::to_string(first)).c_str());
                if (!cpu_last.empty()) {
                    double mc = 0;
                    for (int v = 0; v < nv; ++v) mc = std::max(mc, std::fabs(double(lb[static_cast<size_t>(v)]) - double(cpu_last[static_cast<size_t>(v)])));
                    size_t nc = std::min(cpu_greedy.size(), cont_b.size());
                    size_t fc = nc;
                    // cpu_greedy[i] is the token after prompt position n_prompt-1+i, the same indexing as cont_b
                    for (size_t i = 0; i < nc; ++i)
                        if (cpu_greedy[i] != cont_b[i]) {
                            fc = i;
                            break;
                        }
                    std::printf("; vs CPU: max|dlogit| %.3e, continuation (%zu tokens) %s", mc, nc,
                                fc == nc ? "identical" : ("differs at index " + std::to_string(fc)).c_str());
                }
                std::printf("\n");
                std::fflush(stdout);
            }
            gpu->set_prefill(keep_planes, chunk);
        }

        // ---- 1c. --layer-diff: the residual stream after every layer at the last prompt token, batched vs token by token
        if (layer_diff) {
            std::vector<std::vector<float>> tt, tb;
            const int keep_planes = gpu->prefill_planes();
            gpu->set_prefill(0, chunk);
            gpu->reset();
            gpu->prefill(ids.data(), n_prompt - 1, 0, false);
            gpu->set_layer_trace(&tt);
            gpu->prefill(ids.data() + n_prompt - 1, 1, n_prompt - 1, true);
            gpu->set_layer_trace(nullptr);
            gpu->set_prefill(keep_planes, chunk);
            gpu->reset();
            gpu->set_layer_trace(&tb);
            gpu->prefill(ids.data(), n_prompt, 0, true);
            gpu->set_layer_trace(nullptr);
            gpu->sync();
            std::printf("-- layer diff '%s': residual after each layer at the last of %d prompt tokens, %d planes vs token by token\n",
                        name.c_str(), n_prompt, keep_planes);
            for (size_t l = 0; l < tt.size() && l < tb.size(); ++l) {
                double mx = 0, num = 0, den = 0;
                for (size_t i = 0; i < tt[l].size() && i < tb[l].size(); ++i) {
                    const double d = double(tt[l][i]) - double(tb[l][i]);
                    mx = std::max(mx, std::fabs(d));
                    num += d * d;
                    den += double(tt[l][i]) * double(tt[l][i]);
                }
                std::printf("   layer %2zu  max|d| %.3e  rel L2 %.3e  (%s)\n", l, mx, std::sqrt(num / std::max(den, 1e-30)), (l % 4 == 3) ? "attention" : "gdn");
            }
        }

        // ---- 2. free-running greedy generation on the GPU --------------------------------------------
        if (n_gen > 0) {
            gpu->reset();
            gpu->sync();
            std::vector<int32_t> gen;
            const auto t_pre = Clock::now();
            if (!gpu->prefill(ids.data(), n_prompt, 0, true)) {
                std::fprintf(stderr, "gpu_forward: a launch failed in prefill\n");
                return 1;
            }
            gen.push_back(gpu->argmax());  // synchronises: the prefill is done
            const double pre_ms = ms_since(t_pre);
            const auto t_dec = Clock::now();
            for (int k = 1; k < n_gen; ++k) {
                if (!gpu->step(gen.back(), n_prompt + k - 1, true)) {
                    std::fprintf(stderr, "gpu_forward: a launch failed in decode at %d\n", k);
                    return 1;
                }
                gen.push_back(gpu->argmax());
            }
            const double dec_ms = ms_since(t_dec);
            if (profile) {
                std::vector<std::pair<std::string, double>> prof;
                gpu->set_decode_profile(&prof);
                const int n_prof = 32;
                for (int k = 0; k < n_prof; ++k) {
                    gpu->step(gen.back(), n_prompt + n_gen - 1 + k, true);
                    gpu->sync();
                }
                gpu->set_decode_profile(nullptr);
                double tot = 0;
                for (auto& kv : prof) tot += kv.second;
                std::printf("   decode phase profile (events, eager, %d steps, %.2f ms per step):\n", n_prof, tot / n_prof);
                for (auto& kv : prof) std::printf("      %-22s %8.3f ms/step  %5.1f%%\n", kv.first.c_str(), kv.second / n_prof, 100.0 * kv.second / tot);
            }
            min_free = std::min(min_free, vram_free());

            std::printf("-- generation '%s': %d prompt tokens + %d generated (greedy, free-running on the GPU)\n", name.c_str(),
                        n_prompt, n_gen);
            std::printf("   prefill %d tokens in %.1f ms = %.1f tok/s (Gpu::prefill, head on the last only)\n", n_prompt,
                        pre_ms, double(n_prompt) / (pre_ms / 1e3));
            std::printf("   decode  %d tokens in %.1f ms = %.2f tok/s (%.2f ms/token, one sync per token, %s)\n", n_gen - 1, dec_ms,
                        double(n_gen - 1) / (dec_ms / 1e3), dec_ms / double(n_gen - 1), graphs ? "CUDA graph" : "kernel by kernel");
            std::printf("   ids:");
            for (int32_t t : gen) std::printf(" %d", t);
            std::printf("\n   text: \"%s\"\n", escape(tok.decode(gen)).c_str());

            if (!cpu_greedy.empty()) {
                size_t n = std::min(cpu_greedy.size(), gen.size()), first = n;
                for (size_t i = 0; i < n; ++i)
                    if (cpu_greedy[i] != gen[i]) {
                        first = i;
                        break;
                    }
                std::printf("   vs the CPU's greedy: first %zu tokens %s\n", n,
                            first == n ? "identical" : ("differ first at index " + std::to_string(first)).c_str());
            }
            const std::vector<int32_t> og = oracle_greedy(oracle_dir + "/rung1-" + name + ".jsonl", name);
            if (!og.empty()) {
                const size_t n = std::min(og.size(), gen.size());
                size_t first = n;
                for (size_t i = 0; i < n; ++i)
                    if (og[i] != gen[i]) {
                        first = i;
                        break;
                    }
                std::printf("   vs the oracle capture (rung1-%s.jsonl, %zu tokens): %s\n", name.c_str(), n,
                            first == n ? "identical" : ("differ first at index " + std::to_string(first)).c_str());
            }
            std::fflush(stdout);
        }
    }


    // ---- 3. prefill throughput -----------------------------------------------------------------------
    if (!bench_prefill.empty()) {
        std::string text = read_prompt(prompt_dir + "/long.txt", "long");
        if (text.empty()) {
            std::ifstream in(prompt_dir + "/long.txt", std::ios::binary);
            std::stringstream ss;
            ss << in.rdbuf();
            text = ss.str();
        }
        std::vector<int32_t> base;
        limbo::tok::Options o;
        o.parse_special = true;
        tok.encode(text, base, o);
        std::printf("\n=== prefill throughput: long.txt = %zu tokens, repeated to length ===\n", base.size());
        for (const std::string& ns : split_csv(bench_prefill)) {
            const int n = std::atoi(ns.c_str());
            if (n + 1 > n_ctx) {
                std::fprintf(stderr, "gpu_forward: --bench-prefill %d needs --ctx > %d\n", n, n);
                return 2;
            }
            std::vector<int32_t> ids(static_cast<size_t>(n));
            for (int i = 0; i < n; ++i) ids[static_cast<size_t>(i)] = base[static_cast<size_t>(i) % base.size()];
            double best = 1e30, sum = 0;
            const int reps = 3;
            for (int r = 0; r < reps + 1; ++r) {
                gpu->reset();
                gpu->sync();
                const auto t0 = Clock::now();
                if (!gpu->prefill(ids.data(), n, 0, true)) {
                    std::fprintf(stderr, "gpu_forward: prefill failed\n");
                    return 1;
                }
                gpu->sync();
                const double ms = ms_since(t0);
                if (r == 0) continue;  // warm-up
                best = std::min(best, ms);
                sum += ms;
            }
            if (profile) {
                std::vector<std::pair<std::string, double>> prof;
                gpu->set_prefill_profile(&prof);
                gpu->reset();
                gpu->sync();
                gpu->prefill(ids.data(), n, 0, true);
                gpu->sync();
                gpu->set_prefill_profile(nullptr);
                double tot = 0;
                for (auto& kv : prof) tot += kv.second;
                std::printf("   phase profile of one run of %d tokens (events, sum %.1f ms):\n", n, tot);
                for (auto& kv : prof) std::printf("      %-22s %8.1f ms  %5.1f%%\n", kv.first.c_str(), kv.second, 100.0 * kv.second / tot);
            }
            std::printf("   prefill %5d tokens, %d planes, chunk %d: mean %.1f ms = %.0f tok/s (best %.1f ms = %.0f tok/s)\n", n,
                        gpu->prefill_planes(), gpu->prefill_chunk(), sum / reps, double(n) / (sum / reps / 1e3), best,
                        double(n) / (best / 1e3));
            std::fflush(stdout);
        }
        min_free = std::min(min_free, vram_free());
    }

    std::printf("\nvram peak: cudaMemGetInfo free was %.2f GB before the load and never fell below %.2f GB during the run "
                "(%.3f GB taken: allocations + CUDA context + any drift of the desktop's own use)\n",
                double(gpu->free_before_bytes()) / 1e9, double(min_free) / 1e9,
                double(gpu->free_before_bytes() - min_free) / 1e9);
    return 0;
}
