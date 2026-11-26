// Copyright (c) 2026 ITERFORM. All rights reserved.
// tools/bench_prefill.cu — prefill throughput and the llama.cpp parity gate of Limbo's batched prompt pass.
//
//   bench_prefill -m model.gguf [--planes N] [--chunk N] [--ctx N] [--min-free-gb G]
//                 [--bench 512,2048] [--reps 3] [--profile] [--golden-save F | --golden-check F] [--no-streams]
//                 [--oracle a.jsonl,b.jsonl,...] [--teacher]
//   bench_prefill -m model.gguf [--kv f16|q8] --depth 8192,32768 [--depth-chunks 2] [--text FILE] [--profile] --ctx N
//   bench_prefill -m model.gguf --long-greedy 32768 [--n-gen 16] [--planes-list 0,3,2,1] [--text FILE] --ctx N
//   bench_prefill -m model.gguf --kernels [--ext-formats] [--layer L] [--kt 512,2048] [--kplanes 1,2,3] [--sigma S] [--only name]
//
// --bench N,...: Gpu::prefill throughput over the first N tokens of bench/prompts/long.txt (repeated if it is
//   shorter), head on the last token, mean and best of --reps timed runs after one warm-up.
// --golden-save / --golden-check: the logits after prefilling the first 700 tokens of long.txt (chunk 256, so two
//   chunk boundaries) are written to / compared bit for bit with a file. A change that must not move a number
//   (a faster scan, a fused kernel) is held to "0 elements differ".
// --oracle: for every capture (oracle_dump output, e.g. bench/oracle/rung1-gpu-*.jsonl) the prompt ids of the
//   capture are prefilled with the chosen mode and decoded greedily for the capture's n_predict tokens. Reported:
//   the first token where Limbo's greedy differs from the capture, with Limbo's top-2 gap and the capture's, and
//   the logit differences at the captured top-2 tokens. --teacher also feeds the capture's own tokens through
//   the decode path (teacher forcing) and counts the positions whose top-1 agrees, which separates the prefill's
//   state from the (fp32-activation) decode path's own drift.
//
// --depth D,...: prefill speed over a long cached prefix. For every D: the first D tokens of --text (default
//   bench/prompts/long.txt, repeated) are prefilled, then --depth-chunks chunks of the chunk size at positions D, D+chunk,
//   ... are timed one by one (attention over the D cached positions plus the chunk). The prefix's own average is printed.
// --long-greedy N: prefill N tokens of --text and decode --n-gen greedy tokens, once per mode of --planes-list (0 = token by
//   token on the fp32 GEMV path with the f16 KV cache, the reference); the first listed mode is the reference and every
//   other one is compared with it (first differing index, both top-2 gaps).
// --kernels: no engine; one layer's tensors go to the card and each batched product is timed in isolation
//   (cudaEvents, mean of 20 calls) at T rows for 1, 2 and 3 planes, with the weight bytes per second it implies.
//   MoE routing is synthetic: per-expert popularity lognormal with --sigma, 8 distinct experts per token.
//
// The card is shared: open() refuses unless --min-free-gb is free. Wrap in the build/gpu.lock flock.
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "limbo/cuda/gemm.hpp"
#include "limbo/cuda/gpu.hpp"
#include "limbo/gguf.hpp"
#include "limbo/model.hpp"
#include "limbo/tok.hpp"
#include "limbo/types.hpp"

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

const char* kRung1 =
    "/home/alca/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-MTP-GGUF/snapshots/"
    "5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf";

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

struct Capture {
    std::string name;
    std::vector<int32_t> prompt, greedy;
    std::vector<std::array<double, 5>> top2;  // id1, logit1, id2, logit2, gap
};

// The numbers of the array that follows `key` (e.g. "greedy":[...]) in one jsonl line.
std::vector<double> numbers_after(const std::string& line, const char* key) {
    std::vector<double> out;
    const size_t k = line.find(key);
    if (k == std::string::npos) return out;
    size_t i = line.find('[', k) + 1;
    int depth = 1;
    while (i < line.size() && depth > 0) {
        const char c = line[i];
        if (c == '[') {
            ++depth;
            ++i;
        } else if (c == ']') {
            --depth;
            ++i;
        } else if (c == ',' || c == ' ') {
            ++i;
        } else {
            char* e = nullptr;
            out.push_back(std::strtod(line.c_str() + i, &e));
            i = size_t(e - line.c_str());
        }
    }
    return out;
}

bool load_captures(const std::string& path, std::vector<Capture>& out) {
    std::ifstream in(path);
    if (!in) return false;
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("{\"name\"", 0) != 0) continue;
        Capture c;
        const size_t a = line.find("\"name\":\"") + 8;
        c.name = line.substr(a, line.find('"', a) - a);
        for (double v : numbers_after(line, "\"prompt_ids\":")) c.prompt.push_back(int32_t(v));
        for (double v : numbers_after(line, "\"greedy\":")) c.greedy.push_back(int32_t(v));
        const std::vector<double> t = numbers_after(line, "\"top2\":");
        for (size_t i = 0; i + 4 < t.size(); i += 5) c.top2.push_back({t[i], t[i + 1], t[i + 2], t[i + 3], t[i + 4]});
        out.push_back(std::move(c));
    }
    return true;
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


// ---- --kernels: one layer's products in isolation -----------------------------------------------------------
struct DevT {
    void* p = nullptr;
    uint32_t type = 0;
    int64_t ne0 = 0, ne1 = 0;
    uint64_t bytes = 0;
};

DevT upload(const limbo::model::Model& m, const std::string& name) {
    DevT d;
    const limbo::gguf::Tensor* t = m.tensor(name);
    if (!t) {
        std::fprintf(stderr, "bench_prefill: no tensor %s\n", name.c_str());
        std::exit(2);
    }
    d.type = t->type;
    d.ne0 = t->ne[0];
    d.ne1 = t->ne[1];
    d.bytes = t->nbytes;
    if (cudaMalloc(&d.p, d.bytes + 256) != cudaSuccess || cudaMemcpy(d.p, t->data, d.bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "bench_prefill: cannot upload %s\n", name.c_str());
        std::exit(3);
    }
    return d;
}

struct Timer {
    cudaEvent_t a, b;
    Timer() {
        cudaEventCreate(&a);
        cudaEventCreate(&b);
    }
    template <typename F>
    double ms(F&& f, int reps = 20) {
        for (int i = 0; i < 3; ++i) f();
        cudaEventRecord(a);
        for (int i = 0; i < reps; ++i) f();
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float t = 0;
        cudaEventElapsedTime(&t, a, b);
        return double(t) / reps;
    }
};

int run_kernels(const limbo::model::Model& m, int layer, const std::vector<int>& Ts, const std::vector<int>& planes, double sigma,
                const std::string& only, bool ext) {
    namespace cu = limbo::cuda;
    std::string err;
    if (!cu::gemm_init(&err)) {
        std::fprintf(stderr, "bench_prefill: %s\n", err.c_str());
        return 3;
    }
    const std::string b = "blk." + std::to_string(layer) + ".";
    const DevT qkv = upload(m, b + "attn_qkv.weight"), gate = upload(m, b + "attn_gate.weight"), out = upload(m, b + "ssm_out.weight"),
               sg = upload(m, b + "ffn_gate_shexp.weight"), sd = upload(m, b + "ffn_down_shexp.weight"),
               ge = upload(m, b + "ffn_gate_exps.weight"), rt = upload(m, b + "ffn_gate_inp.weight"), ue = upload(m, b + "ffn_up_exps.weight"), de = upload(m, b + "ffn_down_exps.weight");
    const int E = 2048, NE = 256, NU = 8, FF = 512, Tmax = *std::max_element(Ts.begin(), Ts.end());
    const uint64_t ge_bytes = ge.bytes / NE, de_bytes = de.bytes / NE;
    std::printf("layer %d: qkv type %u, gate %u, out %u, shexp %u/%u, experts gate/up %u, down %u\n", layer, qkv.type, gate.type,
                out.type, sg.type, sd.type, ge.type, de.type);

    // activations
    std::vector<float> hx(size_t(Tmax) * 4096);
    uint32_t seed = 12345u;
    for (float& v : hx) {
        float g = 0;
        for (int i = 0; i < 4; ++i) {
            seed = seed * 1664525u + 1013904223u;
            g += float(seed >> 8) / 16777216.0f - 0.5f;
        }
        v = g * 1.7f;
    }
    float *dx, *dy, *dh, *dsc;
    int8_t* dq;
    float* dxs;
    cudaMalloc(&dx, hx.size() * 4);
    cudaMalloc(&dy, size_t(Tmax) * NU * E * 4);
    cudaMalloc(&dh, size_t(Tmax) * NU * FF * 4);
    cudaMalloc(&dsc, size_t(Tmax) * NU * FF * 4);
    cudaMalloc(&dq, size_t(3) * Tmax * NU * 4096);
    cudaMalloc(&dxs, size_t(Tmax) * NU * 128 * 4);
    cudaMemcpy(dx, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice);
    // routing
    std::vector<double> pop(NE);
    for (double& v : pop) {
        double g = 0;
        for (int i = 0; i < 12; ++i) {
            seed = seed * 1664525u + 1013904223u;
            g += double(seed >> 8) / 16777216.0 - 0.5;
        }
        v = std::exp(sigma * g);
    }
    std::vector<int32_t> hids(size_t(Tmax) * NU);
    for (int t = 0; t < Tmax; ++t) {
        std::vector<double> w = pop;
        for (int u = 0; u < NU; ++u) {
            double tot = 0;
            for (double v : w) tot += v;
            seed = seed * 1664525u + 1013904223u;
            double r = double(seed >> 8) / 16777216.0 * tot;
            int e = 0;
            for (; e < NE - 1; ++e) {
                r -= w[size_t(e)];
                if (r <= 0) break;
            }
            hids[size_t(t) * NU + size_t(u)] = e;
            w[size_t(e)] = 0;
        }
    }
    int32_t* ids;
    cudaMalloc(&ids, hids.size() * 4);
    cudaMemcpy(ids, hids.data(), hids.size() * 4, cudaMemcpyHostToDevice);
    cu::MoeLists L;
    const size_t pairs = size_t(Tmax) * NU, ts = size_t(cu::moe_max_tiles_small(NE)), tb = size_t(cu::moe_max_tiles_big(int(pairs), NE));
    cudaMalloc(&L.cnt, NE * 4);
    cudaMalloc(&L.start, (NE + 1) * 4);
    cudaMalloc(&L.list_tok, pairs * 4);
    cudaMalloc(&L.pos_of, pairs * 4);
    cudaMalloc(&L.tiles_small, ts * 16);
    cudaMalloc(&L.tiles_big, tb * 16);
    cudaMalloc(&L.n_tiles, 64);
    Timer tm;
    auto want = [&](const char* n) { return only.empty() || only == n; };
    for (int T : Ts) {
        cu::moe_build(ids, T, NE, NU, L, nullptr);
        std::vector<int32_t> cnt(NE), nt2(2);
        cudaMemcpy(cnt.data(), L.cnt, NE * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(nt2.data(), L.n_tiles, 8, cudaMemcpyDeviceToHost);
        int mx = 0, big = 0, nz = 0;
        for (int c : cnt) {
            mx = std::max(mx, c);
            big += c > cu::kMoeBmBig;
            nz += c > 0;
        }
        std::printf("\nT = %d: %d pairs, %d experts used, busiest %d pairs, %d experts above %d pairs; tiles small %d big %d\n", T,
                    T * NU, nz, mx, big, cu::kMoeBmBig, nt2[0], nt2[1]);
        std::printf("  %-14s %6s  %9s  %9s\n", "op", "planes", "ms/call", "weights GB/s");
        for (int nt : planes) {
            auto row = [&](const char* name, double ms, double wbytes) {
                std::printf("  %-14s %6d  %9.3f  %9.1f\n", name, nt, ms, wbytes / (ms * 1e6));
            };
            if (nt == planes.front() && want("router"))
                std::printf("  %-14s %6s  %9.3f\n", "router f32", "-", tm.ms([&] { cu::gemm_f32(static_cast<const float*>(rt.p), dx, dy, T, NE, E, nullptr); }));
            if (want("quant")) {
                row("act_quant 2048", tm.ms([&] { cu::act_quant(dx, T, E, nt, dq, dxs, nullptr); }), double(T) * E * 4);
            }
            cu::act_quant(dx, T, E, nt, dq, dxs, nullptr);
            if (want("qkv")) row("gdn qkv", tm.ms([&] { cu::gemm(qkv.type, qkv.p, dq, dxs, nt, T, dy, 8192, E, false, nullptr); }), double(qkv.bytes));
            if (want("z")) row("gdn gate (z)", tm.ms([&] { cu::gemm(gate.type, gate.p, dq, dxs, nt, T, dy, 4096, E, false, nullptr); }), double(gate.bytes));
            if (want("shexp")) row("shexp gate", tm.ms([&] { cu::gemm(sg.type, sg.p, dq, dxs, nt, T, dy, FF, E, false, nullptr); }), double(sg.bytes));
            cu::act_quant(dx, T, 4096, nt, dq, dxs, nullptr);
            if (want("out")) row("gdn out", tm.ms([&] { cu::gemm(out.type, out.p, dq, dxs, nt, T, dy, E, 4096, true, nullptr); }), double(out.bytes));
            cu::act_quant(dx, T, E, nt, dq, dxs, nullptr);
            if (want("gateup")) row("moe gate/up", tm.ms([&] { cu::moe_gate_up(ge.type, ge.p, ue.p, ge_bytes, dq, dxs, nt, T, T * NU, L, NE, dh, FF, E, nullptr); }), double(ge.bytes + ue.bytes));
            cu::moe_gate_up(ge.type, ge.p, ue.p, ge_bytes, dq, dxs, nt, T, T * NU, L, NE, dh, FF, E, nullptr);
            cu::act_quant(dh, T * NU, FF, nt, dq, dxs, nullptr);
            if (want("hquant")) row("act_quant 512", tm.ms([&] { cu::act_quant(dh, T * NU, FF, nt, dq, dxs, nullptr); }), double(T) * NU * FF * 4);
            if (want("down")) row("moe down", tm.ms([&] { cu::moe_down(de.type, de.p, de_bytes, dq, dxs, nt, T * NU, L, NE, dy, E, FF, nullptr); }), double(de.bytes));
            cudaDeviceSynchronize();
            if (const cudaError_t ce = cudaGetLastError(); ce != cudaSuccess) {
                std::fprintf(stderr, "bench_prefill: a kernel failed: %s\n", cudaGetErrorString(ce));
                return 1;
            }
        }
    }
    if (ext) {
        // the formats of the other models, random bytes (timing only): a dense product like gdn qkv (4096 x 2048), the fused
        // gate/up and the down product of an expert stack like layer 1's (256 experts, 512 x 2048 and 2048 x 512)
        struct F {
            uint32_t type;
            const char* name;
        };
        const F fm[] = {{uint32_t(limbo::Type::IQ3_XXS), "IQ3_XXS"}, {uint32_t(limbo::Type::IQ2_XXS), "IQ2_XXS"}, {uint32_t(limbo::Type::IQ2_XS), "IQ2_XS"},
                        {uint32_t(limbo::Type::IQ1_S), "IQ1_S"},     {uint32_t(limbo::Type::IQ1_M), "IQ1_M"},     {uint32_t(limbo::Type::IQ4_NL), "IQ4_NL"},
                        {uint32_t(limbo::Type::Q2_0), "Q2_0"},       {uint32_t(limbo::Type::Q4_K), "Q4_K"},       {uint32_t(limbo::Type::Q5_K), "Q5_K"},
                        {uint32_t(limbo::Type::Q2_K), "Q2_K"},       {uint32_t(limbo::Type::F16), "F16"}};
        float* dsum;
        cudaMalloc(&dsum, size_t(Tmax) * NU * 4096 / 16 * 4);
        auto fill = [&](void* dev, size_t bytes) {
            std::vector<uint8_t> h(bytes);
            for (size_t i = 0; i < bytes; ++i) {
                seed = seed * 1664525u + 1013904223u;
                h[i] = static_cast<uint8_t>(seed >> 16);
            }
            cudaMemcpy(dev, h.data(), bytes, cudaMemcpyHostToDevice);
        };
        for (const F& f : fm) {
            const uint64_t rb_in = limbo::type_bytes(f.type, E), rb_ff = limbo::type_bytes(f.type, FF);
            void *w, *g, *u, *d;
            cudaMalloc(&w, rb_in * 4096 + 256);
            fill(w, rb_in * 4096);
            const bool moe = f.type != uint32_t(limbo::Type::F16);
            if (moe) {
                cudaMalloc(&g, rb_in * FF * NE + 256);
                cudaMalloc(&u, rb_in * FF * NE + 256);
                cudaMalloc(&d, rb_ff * E * NE + 256);
                fill(g, rb_in * FF * NE);
                fill(u, rb_in * FF * NE);
                fill(d, rb_ff * E * NE);
            }
            for (int T : Ts) {
                cu::moe_build(ids, T, NE, NU, L, nullptr);
                for (int nt : planes) {
                    cu::act_quant_sums(dx, T, E, nt, dq, dxs, dsum, nullptr);
                    const double ms = tm.ms([&] { cu::gemm(f.type, w, dq, dxs, nt, T, dy, 4096, E, false, nullptr, dsum); });
                    std::printf("  %-8s T %4d planes %d  dense 4096x2048 %7.3f ms (%6.1f GB/s)", f.name, T, nt, ms, double(rb_in) * 4096 / (ms * 1e6));
                    if (moe) {
                        const double mg = tm.ms([&] { cu::moe_gate_up(f.type, g, u, rb_in * FF, dq, dxs, nt, T, T * NU, L, NE, dh, FF, E, nullptr, dsum); });
                        cu::moe_gate_up(f.type, g, u, rb_in * FF, dq, dxs, nt, T, T * NU, L, NE, dh, FF, E, nullptr, dsum);
                        cu::act_quant_sums(dh, T * NU, FF, nt, dq, dxs, dsum, nullptr);
                        const double md = tm.ms([&] { cu::moe_down(f.type, d, rb_ff * E, dq, dxs, nt, T * NU, L, NE, dy, E, FF, nullptr, dsum); });
                        std::printf("  gate/up %7.3f ms (%6.1f GB/s)  down %7.3f ms (%6.1f GB/s)", mg, double(rb_in) * FF * NE * 2 / (mg * 1e6), md,
                                    double(rb_ff) * E * NE / (md * 1e6));
                    }
                    std::printf("\n");
                    if (const cudaError_t ce = cudaGetLastError(); ce != cudaSuccess) {
                        std::fprintf(stderr, "bench_prefill: %s failed: %s\n", f.name, cudaGetErrorString(ce));
                        return 1;
                    }
                }
            }
            cudaFree(w);
            if (moe) {
                cudaFree(g);
                cudaFree(u);
                cudaFree(d);
            }
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path = kRung1, prompt_dir = "bench/prompts", bench, oracle;
    int planes = -1, chunk = 0, n_ctx = 4608, reps = 3;
    double min_free_gb = 14.6;
    bool profile = false, teacher = false, kernels = false, ext_formats = false;
    std::string depth_list, text_file, planes_list, kv_fmt = "f16";
    int depth_chunks = 2, long_greedy = 0, n_gen = 16;
    std::string golden_save, golden_check;
    bool streams = true;
    int klayer = 1;
    double sigma = 0.8;
    std::string kt = "512,2048", kp = "1,2,3", only;
    for (int i = 1; i < argc; ++i) {
        auto is = [&](const char* a) { return std::strcmp(argv[i], a) == 0; };
        auto val = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
        if (is("-m") || is("--model")) model_path = val();
        else if (is("--prompt-dir")) prompt_dir = val();
        else if (is("--planes")) planes = std::atoi(val());
        else if (is("--chunk")) chunk = std::atoi(val());
        else if (is("--ctx")) n_ctx = std::atoi(val());
        else if (is("--reps")) reps = std::atoi(val());
        else if (is("--min-free-gb")) min_free_gb = std::atof(val());
        else if (is("--bench")) bench = val();
        else if (is("--depth")) depth_list = val();
        else if (is("--kv")) kv_fmt = val();
        else if (is("--depth-chunks")) depth_chunks = std::atoi(val());
        else if (is("--text")) text_file = val();
        else if (is("--long-greedy")) long_greedy = std::atoi(val());
        else if (is("--n-gen")) n_gen = std::atoi(val());
        else if (is("--planes-list")) planes_list = val();
        else if (is("--oracle")) oracle = val();
        else if (is("--teacher")) teacher = true;
        else if (is("--profile")) profile = true;
        else if (is("--kernels")) kernels = true;
        else if (is("--no-streams")) streams = false;
        else if (is("--ext-formats")) ext_formats = true;
        else if (is("--golden-save")) golden_save = val();
        else if (is("--golden-check")) golden_check = val();
        else if (is("--layer")) klayer = std::atoi(val());
        else if (is("--kt")) kt = val();
        else if (is("--kplanes")) kp = val();
        else if (is("--sigma")) sigma = std::atof(val());
        else if (is("--only")) only = val();
        else {
            std::fprintf(stderr,
                         "usage: bench_prefill -m model.gguf [--planes N] [--chunk N] [--ctx N] [--bench 512,2048] [--reps R]\n"
                         "       [--profile] [--oracle a.jsonl,b.jsonl] [--teacher] [--min-free-gb G]\n");
            return 2;
        }
    }

    std::string err;
    auto model = limbo::model::Model::open({model_path}, &err);
    if (!model) {
        std::fprintf(stderr, "bench_prefill: %s\n", err.c_str());
        return 1;
    }
    limbo::tok::Vocab vocab;
    if (!limbo::tok::Vocab::load(*model->shards()[0], vocab, &err)) {
        std::fprintf(stderr, "bench_prefill: %s\n", err.c_str());
        return 1;
    }
    const limbo::tok::Tokenizer tok(vocab);

    if (kernels) {
        std::vector<int> Ts, Ps;
        for (const std::string& x : split_csv(kt)) Ts.push_back(std::atoi(x.c_str()));
        for (const std::string& x : split_csv(kp)) Ps.push_back(std::atoi(x.c_str()));
        return run_kernels(*model, klayer, Ts, Ps, sigma, only, ext_formats);
    }
    auto gpu = limbo::cuda::Gpu::open(*model, n_ctx, static_cast<uint64_t>(min_free_gb * 1e9), &err);
    if (!gpu) {
        std::fprintf(stderr, "bench_prefill: %s\n", err.c_str());
        return 3;
    }
    if (planes < 0) planes = gpu->prefill_planes();
    if (chunk <= 0) chunk = gpu->prefill_chunk();
    gpu->set_prefill_streams(streams);
    if (kv_fmt == "q8" || kv_fmt == "q4") {
        const auto f = kv_fmt == "q8" ? limbo::cuda::KvFormat::Q8 : limbo::cuda::KvFormat::Q4;
        if (!gpu->set_kv_format(f, 0)) {
            std::fprintf(stderr, "bench_prefill: set_kv_format(%s) refused\n", kv_fmt.c_str());
            return 3;
        }
        std::printf("kv cache: %s (%.3f GB)\n", kv_fmt.c_str(), double(gpu->kv_cache_bytes()) / 1e9);
    }
    if (!gpu->set_prefill(planes, chunk)) {
        std::fprintf(stderr, "bench_prefill: bad --planes/--chunk\n");
        return 2;
    }
    std::printf("bench_prefill: planes %d, chunk %d, ctx %d; free before the load %.2f GB, after %.2f GB\n", planes, chunk,
                n_ctx, double(gpu->free_before_bytes()) / 1e9, double(gpu->free_after_bytes()) / 1e9);
    size_t min_free = gpu->free_after_bytes();
    auto track = [&]() {
        size_t f = 0, t = 0;
        cudaMemGetInfo(&f, &t);
        min_free = std::min(min_free, f);
    };
    const int nv = gpu->n_vocab();
    int status = 0;

    // ---- golden logits ---------------------------------------------------------------------------------------
    if (!golden_save.empty() || !golden_check.empty()) {
        std::ifstream in(prompt_dir + "/long.txt", std::ios::binary);
        std::stringstream ss;
        ss << in.rdbuf();
        limbo::tok::Options o;
        o.parse_special = true;
        std::vector<int32_t> base;
        tok.encode(ss.str(), base, o);
        const int n = 700;
        if (base.size() < size_t(n) || n + 1 > n_ctx) {
            std::fprintf(stderr, "bench_prefill: golden needs 700 tokens of long.txt and --ctx > 700\n");
            return 2;
        }
        gpu->set_prefill(planes, 256);
        gpu->reset();
        gpu->sync();
        if (!gpu->prefill(base.data(), n, 0, true)) return 1;
        std::vector<float> lg(static_cast<size_t>(nv));
        if (!gpu->read_logits(lg.data())) return 1;
        if (!golden_save.empty()) {
            std::ofstream f(golden_save, std::ios::binary);
            f.write(reinterpret_cast<const char*>(lg.data()), std::streamsize(lg.size() * 4));
            std::printf("golden: planes %d, %d tokens, logits saved to %s\n", planes, n, golden_save.c_str());
        } else {
            std::vector<float> ref(lg.size());
            std::ifstream f(golden_check, std::ios::binary);
            if (!f.read(reinterpret_cast<char*>(ref.data()), std::streamsize(ref.size() * 4))) {
                std::fprintf(stderr, "bench_prefill: cannot read %s\n", golden_check.c_str());
                return 2;
            }
            size_t nd = 0;
            double mx = 0;
            for (size_t i = 0; i < lg.size(); ++i)
                if (std::memcmp(&lg[i], &ref[i], 4) != 0) {
                    ++nd;
                    mx = std::max(mx, std::fabs(double(lg[i]) - double(ref[i])));
                }
            const Top2 a = top2(lg.data(), nv), b = top2(ref.data(), nv);
            std::printf("golden: planes %d: %zu of %d logits differ, max|d| %.3e, top-1 %s (%d vs %d)\n", planes, nd, nv, mx,
                        a.i1 == b.i1 ? "equal" : "DIFFERENT", a.i1, b.i1);
            if (nd) status = 5;
        }
        gpu->set_prefill(planes, chunk);
    }

    // ---- the parity gate against llama.cpp captures -----------------------------------------------------
    int total_caps = 0, total_ident = 0;
    for (const std::string& path : split_csv(oracle)) {
        std::vector<Capture> caps;
        if (!load_captures(path, caps) || caps.empty()) {
            std::fprintf(stderr, "bench_prefill: cannot read captures from %s\n", path.c_str());
            return 2;
        }
        for (const Capture& c : caps) {
            const int n = int(c.prompt.size()), np = int(c.greedy.size());
            if (n + np + 1 > n_ctx) {
                std::fprintf(stderr, "bench_prefill: %s needs --ctx > %d\n", c.name.c_str(), n + np);
                return 2;
            }
            std::vector<float> lg(static_cast<size_t>(nv));
            // free-running: the prompt, then Limbo's own greedy tokens
            gpu->reset();
            gpu->sync();
            const auto t0 = Clock::now();
            if (!gpu->prefill(c.prompt.data(), n, 0, true)) {
                std::fprintf(stderr, "bench_prefill: prefill failed\n");
                return 1;
            }
            gpu->sync();
            const double pre_ms = ms_since(t0);
            std::vector<int32_t> gen;
            std::vector<std::array<double, 5>> mine;  // Limbo's top-2 per step
            double d1_sum = 0, d1_max = 0, dgap_max = 0, first_d1 = 0, first_d2 = 0;
            int d_n = 0;
            int first = -1;
            for (int k = 0; k < np; ++k) {
                if (k > 0 && !gpu->step(gen.back(), n + k - 1, true)) {
                    std::fprintf(stderr, "bench_prefill: step failed\n");
                    return 1;
                }
                if (!gpu->read_logits(lg.data())) return 1;
                const Top2 t = top2(lg.data(), nv);
                gen.push_back(t.i1);
                mine.push_back({double(t.i1), double(t.v1), double(t.i2), double(t.v2), double(t.v1) - double(t.v2)});
                if (first < 0 && t.i1 != c.greedy[size_t(k)]) first = k;
                if (k == 0) {
                    const double e1 = double(lg[size_t(c.top2[0][0])]) - c.top2[0][1];
                    const double e2 = double(lg[size_t(c.top2[0][2])]) - c.top2[0][3];
                    first_d1 = e1;
                    first_d2 = e2;
                }
                if (first < 0 || first == k) {
                    // the logit of the capture's top-1 and top-2 tokens, while the contexts are identical
                    const double d1 = double(lg[size_t(c.top2[size_t(k)][0])]) - c.top2[size_t(k)][1];
                    const double d2 = double(lg[size_t(c.top2[size_t(k)][2])]) - c.top2[size_t(k)][3];
                    const double dgap = (double(lg[size_t(c.top2[size_t(k)][0])]) - double(lg[size_t(c.top2[size_t(k)][2])])) - c.top2[size_t(k)][4];
                    d1_sum += std::fabs(d1);
                    d1_max = std::max(d1_max, std::max(std::fabs(d1), std::fabs(d2)));
                    dgap_max = std::max(dgap_max, std::fabs(dgap));
                    ++d_n;
                }
                if (first >= 0) break;  // diverged: later steps are a different context
            }
            ++total_caps;
            const bool ident = first < 0;
            total_ident += ident;
            std::printf("oracle %-10s %4d prompt tokens, planes %d: prefill %.0f ms (%.0f tok/s); free-running %d tokens: ",
                        c.name.c_str(), n, planes, pre_ms, double(n) / (pre_ms / 1e3), np);
            if (ident) {
                std::printf("IDENTICAL\n");
            } else {
                const auto& m = mine[size_t(first)];
                const auto& o = c.top2[size_t(first)];
                std::printf("differs at index %d: limbo %d (gap %.4f to %d), capture %d (gap %.4f to %d)%s%s\n", first,
                            int(m[0]), m[4], int(m[2]), int(o[0]), o[4], int(o[2]), int(m[2]) == int(o[0]) ? ", limbo's runner-up is the capture's top-1" : "",
                            o[4] < 0.01 ? " (near tie by PLAN 3.3)" : "");
                status = 4;
            }
            std::printf("    first token (prefill only): d logit at the capture's top-1 %+.4f, top-2 %+.4f; over %d matched steps: mean|d1| %.3e, max %.3e, max|d gap| %.3e\n",
                        first_d1, first_d2, d_n, d_n ? d1_sum / d_n : 0.0, d1_max, dgap_max);
            if (teacher) {
                // the capture's own tokens through the decode path: top-1 per position
                gpu->reset();
                gpu->sync();
                if (!gpu->prefill(c.prompt.data(), n, 0, true)) return 1;
                int agree = 0;
                std::string bad;
                double dmean = 0;
                for (int k = 0; k < np; ++k) {
                    if (k > 0 && !gpu->step(c.greedy[size_t(k - 1)], n + k - 1, true)) return 1;
                    if (!gpu->read_logits(lg.data())) return 1;
                    const Top2 t = top2(lg.data(), nv);
                    agree += t.i1 == c.greedy[size_t(k)];
                    dmean += std::fabs(double(lg[size_t(c.top2[size_t(k)][0])]) - c.top2[size_t(k)][1]);
                    if (t.i1 != c.greedy[size_t(k)]) {
                        char b[160];
                        std::snprintf(b, sizeof b, " [%d: limbo %d gap %.3f, capture %d gap %.3f]", k, t.i1, double(t.v1 - t.v2),
                                      c.greedy[size_t(k)], c.top2[size_t(k)][4]);
                        bad += b;
                    }
                }
                std::printf("    teacher-forced: top-1 agrees at %d of %d positions, mean|d logit1| %.3e%s\n", agree, np,
                            dmean / np, bad.c_str());
            }
            track();
            std::fflush(stdout);
        }
    }
    if (total_caps > 0) std::printf("oracle gate: %d of %d captures identical\n", total_ident, total_caps);

    // ---- long cached prefix ------------------------------------------------------------------------------------
    if (!depth_list.empty() || long_greedy > 0) {
        std::vector<int32_t> base;
        {
            std::ifstream in(text_file.empty() ? prompt_dir + "/long.txt" : text_file, std::ios::binary);
            std::stringstream ss;
            ss << in.rdbuf();
            limbo::tok::Options o;
            o.parse_special = true;
            tok.encode(ss.str(), base, o);
        }
        if (base.empty()) {
            std::fprintf(stderr, "bench_prefill: no text to tokenise\n");
            return 2;
        }
        std::printf("text: %zu tokens%s\n", base.size(), text_file.empty() ? " (long.txt, repeated to length)" : "");
        auto tokens = [&](int n) {
            std::vector<int32_t> ids(static_cast<size_t>(n));
            for (int i = 0; i < n; ++i) ids[static_cast<size_t>(i)] = base[static_cast<size_t>(i) % base.size()];
            return ids;
        };
        for (const std::string& ds : split_csv(depth_list)) {
            const int D = std::atoi(ds.c_str());
            if (D + depth_chunks * chunk + 1 > n_ctx) {
                std::fprintf(stderr, "bench_prefill: --depth %d + %d chunks of %d needs --ctx > %d\n", D, depth_chunks, chunk, D + depth_chunks * chunk);
                return 2;
            }
            const std::vector<int32_t> ids = tokens(D + depth_chunks * chunk);
            gpu->reset();
            gpu->sync();
            auto t0 = Clock::now();
            if (D > 0 && !gpu->prefill(ids.data(), D, 0, false)) {
                std::fprintf(stderr, "bench_prefill: prefill of the prefix failed\n");
                return 1;
            }
            gpu->sync();
            const double pre_ms = ms_since(t0);
            if (D > 0) std::printf("depth %6d: the prefix itself, %d tokens in %.0f ms = %.0f tok/s (average over 0..%d)\n", D, D, pre_ms, double(D) / (pre_ms / 1e3), D);
            for (int c = 0; c < depth_chunks; ++c) {
                const int pos0 = D + c * chunk;
                std::vector<std::pair<std::string, double>> prof;
                if (profile && c == 0) gpu->set_prefill_profile(&prof);
                t0 = Clock::now();
                if (!gpu->prefill(ids.data() + pos0, chunk, pos0, c == depth_chunks - 1)) {
                    std::fprintf(stderr, "bench_prefill: prefill at depth %d failed\n", pos0);
                    return 1;
                }
                gpu->sync();
                const double ms = ms_since(t0);
                if (profile && c == 0) {
                    gpu->set_prefill_profile(nullptr);
                    double tot = 0;
                    for (auto& kv : prof) tot += kv.second;
                    for (auto& kv : prof)
                        if (kv.second > 0.03 * tot) std::printf("      %-22s %8.1f ms  %5.1f%%\n", kv.first.c_str(), kv.second, 100.0 * kv.second / tot);
                }
                std::printf("depth %6d: chunk of %d tokens at positions %d..%d: %.1f ms = %.0f tok/s (planes %d)\n", pos0, chunk, pos0, pos0 + chunk - 1, ms,
                            double(chunk) / (ms / 1e3), planes);
                std::fflush(stdout);
            }
            track();
        }
        if (long_greedy > 0) {
            const int N = long_greedy;
            if (N + n_gen + 1 > n_ctx) {
                std::fprintf(stderr, "bench_prefill: --long-greedy %d needs --ctx > %d\n", N, N + n_gen);
                return 2;
            }
            const std::vector<int32_t> ids = tokens(N);
            std::vector<float> lg(static_cast<size_t>(nv));
            std::vector<int32_t> ref;
            std::vector<std::array<double, 2>> ref_gap;  // top-2 gap and top-2 logit of the reference
            std::vector<int> modes;
            for (const std::string& ms : split_csv(planes_list.empty() ? std::to_string(planes) : planes_list)) modes.push_back(std::atoi(ms.c_str()));
            for (size_t mi = 0; mi < modes.size(); ++mi) {
                const int pl = modes[mi];
                gpu->set_prefill(pl, chunk);
                gpu->reset();
                gpu->sync();
                const auto t0 = Clock::now();
                if (!gpu->prefill(ids.data(), N, 0, true)) {
                    std::fprintf(stderr, "bench_prefill: prefill (planes %d) failed\n", pl);
                    return 1;
                }
                if (!gpu->read_logits(lg.data())) return 1;
                const double pre_ms = ms_since(t0);
                std::vector<int32_t> gen;
                std::vector<std::array<double, 2>> gaps;
                for (int k = 0; k < n_gen; ++k) {
                    if (k > 0) {
                        if (!gpu->step(gen.back(), N + k - 1, true) || !gpu->read_logits(lg.data())) return 1;
                    }
                    const Top2 t = top2(lg.data(), nv);
                    gen.push_back(t.i1);
                    gaps.push_back({double(t.v1) - double(t.v2), double(t.v1)});
                }
                std::printf("long %d tokens, planes %d: prefill %.1f s (%.0f tok/s); first %d greedy:", N, pl, pre_ms / 1e3, double(N) / (pre_ms / 1e3), n_gen);
                for (int32_t v : gen) std::printf(" %d", v);
                std::printf("\n");
                if (mi == 0) {
                    ref = gen;
                    ref_gap = gaps;
                } else {
                    int first = -1;
                    for (int k = 0; k < n_gen; ++k)
                        if (gen[static_cast<size_t>(k)] != ref[static_cast<size_t>(k)]) {
                            first = k;
                            break;
                        }
                    if (first < 0) std::printf("    identical to the reference (planes %d) over %d tokens\n", modes[0], n_gen);
                    else
                        std::printf("    differs from the reference (planes %d) at index %d: mine %d (gap %.4f), reference %d (gap %.4f)\n", modes[0], first,
                                    gen[static_cast<size_t>(first)], gaps[static_cast<size_t>(first)][0], ref[static_cast<size_t>(first)], ref_gap[static_cast<size_t>(first)][0]);
                    std::printf("    first token: logit %.4f against the reference's %.4f\n", gaps[0][1], ref_gap[0][1]);
                }
                std::fflush(stdout);
                track();
            }
        }
        gpu->set_prefill(planes, chunk);
    }

    // ---- throughput ----------------------------------------------------------------------------------------
    if (!bench.empty()) {
        std::ifstream in(prompt_dir + "/long.txt", std::ios::binary);
        std::stringstream ss;
        ss << in.rdbuf();
        limbo::tok::Options o;
        o.parse_special = true;
        std::vector<int32_t> base;
        tok.encode(ss.str(), base, o);
        if (base.empty()) {
            std::fprintf(stderr, "bench_prefill: cannot read %s/long.txt\n", prompt_dir.c_str());
            return 2;
        }
        for (const std::string& ns : split_csv(bench)) {
            const int n = std::atoi(ns.c_str());
            if (n + 1 > n_ctx) {
                std::fprintf(stderr, "bench_prefill: --bench %d needs --ctx > %d\n", n, n);
                return 2;
            }
            std::vector<int32_t> ids(static_cast<size_t>(n));
            for (int i = 0; i < n; ++i) ids[size_t(i)] = base[size_t(i) % base.size()];
            double best = 1e30, sum = 0;
            for (int r = 0; r < reps + 1; ++r) {
                gpu->reset();
                gpu->sync();
                const auto t0 = Clock::now();
                if (!gpu->prefill(ids.data(), n, 0, true)) {
                    std::fprintf(stderr, "bench_prefill: prefill failed\n");
                    return 1;
                }
                gpu->sync();
                const double ms = ms_since(t0);
                if (r == 0) continue;
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
            track();
            std::printf("prefill %5d tokens, planes %d, chunk %d: mean %.1f ms = %.0f tok/s (best %.1f ms = %.0f tok/s)\n", n, planes,
                        chunk, sum / reps, double(n) / (sum / reps / 1e3), best, double(n) / (best / 1e3));
            std::fflush(stdout);
        }
    }
    std::printf("vram: free %.2f GB before the load, never below %.2f GB during the run\n", double(gpu->free_before_bytes()) / 1e9,
                double(min_free) / 1e9);
    return status;
}
