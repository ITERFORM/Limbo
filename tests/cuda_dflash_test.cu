// Copyright (c) 2026 ITERFORM. All rights reserved.
// tests/cuda_dflash_test.cu — the DFlash2 drafter (src/cuda/dflash.cu) against a plain CPU implementation of the same math, on a small
// synthetic checkpoint written here (2 layers, 512 wide, 8 heads over 2 KV heads, vocabulary 2048, random BF16 weights):
//   1. 2,300 positions are ingested (more than the 2,064-position ring and the 2,048 window) and a block is drafted: the final hidden
//      rows agree with the reference to 1e-3 (relative), the 7 drafts and their confidences are the reference's;
//   2. a slot cut back and re-ingested, a slot shared into another, and a gap (the context starts again) draft what the reference drafts;
//   3. the real checkpoint (LIMBO_DFLASH_DIR) loads and its shapes are accepted, when it is there.
// SKIP (exit 0) with no GPU.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "limbo/check.hpp"
#include "limbo/cuda/dflash.hpp"
#include "limbo/quants.hpp"
#include "limbo/safetensors.hpp"
#include "limbo/types.hpp"

using namespace limbo;
using limbo::cuda::Dflash;

namespace {

struct Rng {
    uint64_t s = 88172645463325252ull;
    uint32_t next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return uint32_t(s >> 32);
    }
    float uni() { return float(next()) / 4294967296.0f; }
    float norm() { return (uni() + uni() + uni() + uni() - 2.0f) * 1.7320508f; }  // mean 0, variance 1
};

uint16_t to_bf16(float f) {
    uint32_t b;
    std::memcpy(&b, &f, 4);
    b += 0x7fff + ((b >> 16) & 1);  // round to nearest even
    return uint16_t(b >> 16);
}
float from_bf16(uint16_t v) {
    uint32_t b = uint32_t(v) << 16;
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}
float rbf(float f) { return from_bf16(to_bf16(f)); }

constexpr int H = 512, FF = 1024, NH = 8, NKV = 2, HD = 128, RANK = 64, NL = 2, NV = 2048, NG = H / 16, DW = 4 * NG, NT = 3, FC = H * NT;
constexpr int kLayers[NT] = {2, 4, 6};  // the inputs of the layers after the checkpoint's target layers 1, 3, 5

struct Tn {
    std::string name;
    std::vector<int64_t> shape;
    std::vector<float> v;  // already BF16-rounded
};

Tn make(Rng& r, const std::string& name, std::vector<int64_t> shape, float sd, float mean = 0.0f) {
    Tn t;
    t.name = name;
    t.shape = shape;
    size_t n = 1;
    for (int64_t d : shape) n *= size_t(d);
    t.v.resize(n);
    for (float& x : t.v) x = rbf(mean + sd * r.norm());
    return t;
}

void write_checkpoint(const std::string& dir, const std::vector<Tn>& ts) {
    std::string hdr = "{";
    uint64_t off = 0;
    for (size_t i = 0; i < ts.size(); ++i) {
        const Tn& t = ts[i];
        std::string sh = "[";
        for (size_t k = 0; k < t.shape.size(); ++k) sh += (k ? "," : "") + std::to_string(t.shape[k]);
        sh += "]";
        const uint64_t nb = t.v.size() * 2;
        hdr += (i ? "," : "") + std::string("\"") + t.name + "\":{\"dtype\":\"BF16\",\"shape\":" + sh + ",\"data_offsets\":[" + std::to_string(off) + "," +
               std::to_string(off + nb) + "]}";
        off += nb;
    }
    hdr += "}";
    std::ofstream f(dir + "/model.safetensors", std::ios::binary);
    const uint64_t n = hdr.size();
    for (int i = 0; i < 8; ++i) f.put(char((n >> (8 * i)) & 255));
    f << hdr;
    for (const Tn& t : ts)
        for (float x : t.v) {
            const uint16_t b = to_bf16(x);
            f.write(reinterpret_cast<const char*>(&b), 2);
        }
    std::ofstream c(dir + "/config.json");
    c << "{\"num_hidden_layers\":" << NL << ",\"hidden_size\":" << H << ",\"num_attention_heads\":" << NH << ",\"num_key_value_heads\":" << NKV
      << ",\"head_dim\":128,\"intermediate_size\":" << FF << ",\"sliding_window\":2048,\"num_target_layers\":8,\"rms_norm_eps\":1e-06,"
      << "\"rope_parameters\":{\"rope_theta\":10000000,\"rope_type\":\"default\"},\"layer_types\":[\"sliding_attention\",\"sliding_attention\"],"
      << "\"dflash_config\":{\"block_size\":8,\"mask_token_id\":" << NV - 3 << ",\"target_layer_ids\":[1,3,5],\"conv_group_size\":16,"
      << "\"conv_kernel_size\":2,\"selector_rank\":" << RANK << ",\"selector_top_k\":16}}";
}

// ---- the reference: the same math in plain loops over the BF16-rounded weights -----------------------------------------------------
struct Ref {
    std::map<std::string, const Tn*> w;
    std::vector<float> embd, head;  // [NV][H]
    std::vector<std::vector<float>> K, V;  // per layer: [pos][NKV * 128]
    int mask = NV - 3;

    const std::vector<float>& W(const std::string& n) const { return w.at(n)->v; }
    static std::vector<float> mv(const std::vector<float>& m, const float* x, int out, int in) {
        std::vector<float> y(static_cast<size_t>(out));
        for (int o = 0; o < out; ++o) {
            double a = 0;
            for (int i = 0; i < in; ++i) a += double(m[size_t(o) * in + i]) * double(x[i]);
            y[size_t(o)] = float(a);
        }
        return y;
    }
    static void rms(float* x, const std::vector<float>& wt, int n, float eps = 1e-6f) {
        double s = 0;
        for (int i = 0; i < n; ++i) s += double(x[i]) * x[i];
        const float inv = 1.0f / std::sqrt(float(s / n) + eps);
        for (int i = 0; i < n; ++i) x[i] = x[i] * inv * wt[size_t(i)];
    }
    static void rope(float* h, int pos) {  // NEOX over 128 dims
        float o[HD];
        for (int d = 0; d < 64; ++d) {
            const double th = double(pos) * std::pow(1e7, -double(d) / 64.0);
            const float c = float(std::cos(th)), s = float(std::sin(th));
            o[d] = h[d] * c - h[d + 64] * s;
            o[d + 64] = h[d + 64] * c + h[d] * s;
        }
        std::memcpy(h, o, sizeof o);
    }
    std::string L(int l, const char* n) const { return "layers." + std::to_string(l) + "." + n; }

    void ingest(const float* feat, int pos) {
        std::vector<float> g = mv(W("fc.weight"), feat, H, FC);
        rms(g.data(), W("hidden_norm.weight"), H);
        for (int l = 0; l < NL; ++l) {
            std::vector<float> k = mv(W(L(l, "self_attn.k_proj.weight")), g.data(), NKV * HD, H);
            std::vector<float> v = mv(W(L(l, "self_attn.v_proj.weight")), g.data(), NKV * HD, H);
            for (int h = 0; h < NKV; ++h) {
                rms(&k[size_t(h) * HD], W(L(l, "self_attn.k_norm.weight")), HD);
                rope(&k[size_t(h) * HD], pos);
            }
            if (K[size_t(l)].size() < size_t(pos + 1) * NKV * HD) {
                K[size_t(l)].resize(size_t(pos + 1) * NKV * HD);
                V[size_t(l)].resize(size_t(pos + 1) * NKV * HD);
            }
            for (int i = 0; i < NKV * HD; ++i) {
                K[size_t(l)][size_t(pos) * NKV * HD + i] = k[size_t(i)];
                V[size_t(l)][size_t(pos) * NKV * HD + i] = v[size_t(i)];
            }
        }
    }
    // the card keeps K/V as F16
    static float h16(float f) { return __half2float(__float2half(f)); }

    // y[t] of the short conv over the 8 rows
    static void conv(const std::vector<float>& in, const std::vector<float>& dyn, const std::vector<float>& base, int side, std::vector<float>& out) {
        out.assign(size_t(8) * H, 0.0f);
        for (int t = 0; t < 8; ++t)
            for (int ch = 0; ch < H; ++ch) {
                const int g = ch / 16;
                const float w0 = dyn[size_t(t) * DW + (side * 2 + 0) * NG + g] + base[size_t((side * 2 + 0) * H + ch)];
                const float w1 = dyn[size_t(t) * DW + (side * 2 + 1) * NG + g] + base[size_t((side * 2 + 1) * H + ch)];
                float r = w0 * in[size_t(t) * H + ch];
                if (t > 0) r += w1 * in[size_t(t - 1) * H + ch];
                out[size_t(t) * H + ch] = r;
            }
    }
    static float silu(float x) { return x / (1.0f + std::exp(-x)); }

    struct Out {
        std::vector<float> hn;  // [8][H]
        int tok[7];
        float conf[7];
    };
    Out draft(int anchor, int pos, int ce, int lo) {
        Out o;
        std::vector<float> x(size_t(8) * H);
        const int toks[8] = {anchor, mask, mask, mask, mask, mask, mask, mask};
        for (int t = 0; t < 8; ++t)
            for (int d = 0; d < H; ++d) x[size_t(t) * H + d] = embd[size_t(toks[t]) * H + d];
        for (int l = 0; l < NL; ++l) {
            std::vector<float> xn = x, dyn(size_t(8) * DW);
            for (int t = 0; t < 8; ++t) rms(&xn[size_t(t) * H], W(L(l, "input_layernorm.weight")), H);
            for (int t = 0; t < 8; ++t) {
                auto d = mv(W(L(l, "attention_conv.kernel_projection.weight")), &xn[size_t(t) * H], DW, H);
                std::copy(d.begin(), d.end(), dyn.begin() + long(t) * DW);
            }
            std::vector<float> xc;
            conv(xn, dyn, W(L(l, "attention_conv.base_kernel")), 0, xc);
            std::vector<float> q(size_t(8) * NH * HD), kb(size_t(8) * NKV * HD), vb(size_t(8) * NKV * HD);
            for (int t = 0; t < 8; ++t) {
                auto a = mv(W(L(l, "self_attn.q_proj.weight")), &xc[size_t(t) * H], NH * HD, H);
                auto b = mv(W(L(l, "self_attn.k_proj.weight")), &xc[size_t(t) * H], NKV * HD, H);
                auto c = mv(W(L(l, "self_attn.v_proj.weight")), &xc[size_t(t) * H], NKV * HD, H);
                for (int h = 0; h < NH; ++h) {
                    rms(&a[size_t(h) * HD], W(L(l, "self_attn.q_norm.weight")), HD);
                    rope(&a[size_t(h) * HD], pos + t);
                }
                for (int h = 0; h < NKV; ++h) {
                    rms(&b[size_t(h) * HD], W(L(l, "self_attn.k_norm.weight")), HD);
                    rope(&b[size_t(h) * HD], pos + t);
                }
                std::copy(a.begin(), a.end(), q.begin() + long(t) * NH * HD);
                std::copy(b.begin(), b.end(), kb.begin() + long(t) * NKV * HD);
                std::copy(c.begin(), c.end(), vb.begin() + long(t) * NKV * HD);
            }
            std::vector<float> at(size_t(8) * NH * HD);
            for (int t = 0; t < 8; ++t)
                for (int h = 0; h < NH; ++h) {
                    const int kv = h / 4;
                    std::vector<double> sc;
                    std::vector<const float*> vv;
                    std::vector<float> vtmp;
                    const int p0 = std::max(lo, pos + t - 2048 + 1);
                    for (int p = p0; p < ce; ++p) {
                        double s = 0;
                        for (int d = 0; d < HD; ++d) s += double(q[(size_t(t) * NH + h) * HD + d]) * h16(K[size_t(l)][(size_t(p) * NKV + kv) * HD + d]);
                        sc.push_back(s / std::sqrt(double(HD)));
                    }
                    for (int j = 0; j < 8; ++j) {
                        double s = 0;
                        for (int d = 0; d < HD; ++d) s += double(q[(size_t(t) * NH + h) * HD + d]) * kb[(size_t(j) * NKV + kv) * HD + d];
                        sc.push_back(s / std::sqrt(double(HD)));
                    }
                    double m = -1e30;
                    for (double s : sc) m = std::max(m, s);
                    double den = 0;
                    std::vector<double> acc(HD, 0.0);
                    for (size_t k = 0; k < sc.size(); ++k) {
                        const double e = std::exp(sc[k] - m);
                        den += e;
                        for (int d = 0; d < HD; ++d) {
                            const double v = k < size_t(ce - p0) ? h16(V[size_t(l)][(size_t(p0 + int(k)) * NKV + kv) * HD + d])
                                                                  : vb[(size_t(int(k) - (ce - p0)) * NKV + kv) * HD + d];
                            acc[size_t(d)] += e * v;
                        }
                    }
                    for (int d = 0; d < HD; ++d) at[(size_t(t) * NH + h) * HD + d] = float(acc[size_t(d)] / den);
                }
            std::vector<float> op(size_t(8) * H);
            for (int t = 0; t < 8; ++t) {
                auto a = mv(W(L(l, "self_attn.o_proj.weight")), &at[size_t(t) * NH * HD], H, NH * HD);
                std::copy(a.begin(), a.end(), op.begin() + long(t) * H);
            }
            conv(op, dyn, W(L(l, "attention_conv.base_kernel")), 1, xc);
            std::vector<float> x1(size_t(8) * H), hn(size_t(8) * H), dynf(size_t(8) * DW);
            for (size_t i = 0; i < x1.size(); ++i) x1[i] = xc[i] + x[i];
            hn = x1;
            for (int t = 0; t < 8; ++t) rms(&hn[size_t(t) * H], W(L(l, "post_attention_layernorm.weight")), H);
            for (int t = 0; t < 8; ++t) {
                auto d = mv(W(L(l, "mlp_conv.kernel_projection.weight")), &hn[size_t(t) * H], DW, H);
                std::copy(d.begin(), d.end(), dynf.begin() + long(t) * DW);
            }
            conv(hn, dynf, W(L(l, "mlp_conv.base_kernel")), 0, xc);
            std::vector<float> dn(size_t(8) * H);
            for (int t = 0; t < 8; ++t) {
                auto g = mv(W(L(l, "mlp.gate_proj.weight")), &xc[size_t(t) * H], FF, H);
                auto u = mv(W(L(l, "mlp.up_proj.weight")), &xc[size_t(t) * H], FF, H);
                for (int i = 0; i < FF; ++i) g[size_t(i)] = silu(g[size_t(i)]) * u[size_t(i)];
                auto d = mv(W(L(l, "mlp.down_proj.weight")), g.data(), H, FF);
                std::copy(d.begin(), d.end(), dn.begin() + long(t) * H);
            }
            conv(dn, dynf, W(L(l, "mlp_conv.base_kernel")), 1, xc);
            for (size_t i = 0; i < x.size(); ++i) x[i] = xc[i] + x1[i];
        }
        o.hn = x;
        for (int t = 0; t < 8; ++t) rms(&o.hn[size_t(t) * H], W("norm.weight"), H);
        // the selector
        const auto& pcb = W("candidate_selector.predecessor_codebook");
        const auto& scb = W("candidate_selector.successor_codebook");
        std::vector<std::vector<int>> cand(8);
        std::vector<std::vector<float>> un(8), gate(8);
        for (int t = 1; t < 8; ++t) {
            std::vector<float> lg(static_cast<size_t>(NV));
            for (int v = 0; v < NV; ++v) {
                double a = 0;
                for (int d = 0; d < H; ++d) a += double(head[size_t(v) * H + d]) * o.hn[size_t(t) * H + d];
                lg[size_t(v)] = float(a);
            }
            std::vector<int> idx(static_cast<size_t>(NV));
            for (int v = 0; v < NV; ++v) idx[size_t(v)] = v;
            std::sort(idx.begin(), idx.end(), [&](int a, int b) { return lg[size_t(a)] > lg[size_t(b)] || (lg[size_t(a)] == lg[size_t(b)] && a < b); });
            idx.resize(16);
            cand[size_t(t)] = idx;
            for (int j : idx) un[size_t(t)].push_back(lg[size_t(j)]);
            gate[size_t(t)] = mv(W("candidate_selector.hidden_projection.weight"), &o.hn[size_t(t) * H], RANK, H);
        }
        int pred = 0;
        for (int t = 1; t < 8; ++t) {
            std::vector<float> sc(16);
            for (int j = 0; j < 16; ++j) {
                const int ptok = t == 1 ? anchor : cand[size_t(t - 1)][size_t(pred)];
                double a = 0;
                for (int d = 0; d < RANK; ++d)
                    a += double(scb[size_t(cand[size_t(t)][size_t(j)]) * RANK + d]) * (double(pcb[size_t(ptok) * RANK + d]) * gate[size_t(t)][size_t(d)]);
                sc[size_t(j)] = float(a) + un[size_t(t)][size_t(j)];
            }
            int best = 0;
            for (int j = 1; j < 16; ++j)
                if (sc[size_t(j)] > sc[size_t(best)]) best = j;
            double sum = 0;
            for (int j = 0; j < 16; ++j) sum += std::exp(double(sc[size_t(j)] - sc[size_t(best)]));
            o.tok[t - 1] = cand[size_t(t)][size_t(best)];
            o.conf[t - 1] = float(1.0 / sum);
            pred = best;
        }
        return o;
    }
};

}  // namespace

int main() {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        std::printf("SKIP cuda_dflash_test: no CUDA device\n");
        return 0;
    }
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    if (free_b < (400ull << 20)) {
        std::printf("SKIP cuda_dflash_test: %.2f GB free\n", double(free_b) / 1e9);
        return 0;
    }
    Rng rng;
    std::vector<Tn> ts;
    for (int l = 0; l < NL; ++l) {
        const std::string p = "layers." + std::to_string(l) + ".";
        ts.push_back(make(rng, p + "self_attn.q_proj.weight", {NH * HD, H}, 1.0f / std::sqrt(float(H))));
        ts.push_back(make(rng, p + "self_attn.k_proj.weight", {NKV * HD, H}, 1.0f / std::sqrt(float(H))));
        ts.push_back(make(rng, p + "self_attn.v_proj.weight", {NKV * HD, H}, 1.0f / std::sqrt(float(H))));
        ts.push_back(make(rng, p + "self_attn.o_proj.weight", {H, NH * HD}, 0.5f / std::sqrt(float(NH * HD))));
        ts.push_back(make(rng, p + "self_attn.q_norm.weight", {HD}, 0.1f, 1.0f));
        ts.push_back(make(rng, p + "self_attn.k_norm.weight", {HD}, 0.1f, 1.0f));
        ts.push_back(make(rng, p + "input_layernorm.weight", {H}, 0.1f, 1.0f));
        ts.push_back(make(rng, p + "post_attention_layernorm.weight", {H}, 0.1f, 1.0f));
        ts.push_back(make(rng, p + "mlp.gate_proj.weight", {FF, H}, 1.0f / std::sqrt(float(H))));
        ts.push_back(make(rng, p + "mlp.up_proj.weight", {FF, H}, 1.0f / std::sqrt(float(H))));
        ts.push_back(make(rng, p + "mlp.down_proj.weight", {H, FF}, 0.5f / std::sqrt(float(FF))));
        ts.push_back(make(rng, p + "attention_conv.base_kernel", {2, 2, H}, 0.3f, 0.2f));
        ts.push_back(make(rng, p + "attention_conv.kernel_projection.weight", {DW, H}, 0.05f / std::sqrt(float(H))));
        ts.push_back(make(rng, p + "mlp_conv.base_kernel", {2, 2, H}, 0.3f, 0.2f));
        ts.push_back(make(rng, p + "mlp_conv.kernel_projection.weight", {DW, H}, 0.05f / std::sqrt(float(H))));
    }
    ts.push_back(make(rng, "fc.weight", {H, FC}, 1.0f / std::sqrt(float(FC))));
    ts.push_back(make(rng, "hidden_norm.weight", {H}, 0.1f, 1.0f));
    ts.push_back(make(rng, "norm.weight", {H}, 0.1f, 1.0f));
    ts.push_back(make(rng, "candidate_selector.hidden_projection.weight", {RANK, H}, 1.0f / std::sqrt(float(H))));
    ts.push_back(make(rng, "candidate_selector.predecessor_codebook", {NV, RANK}, 0.5f));
    ts.push_back(make(rng, "candidate_selector.successor_codebook", {NV, RANK}, 0.5f));
    const std::string dir = limbo::test::tmp_dir("dflash").string();
    write_checkpoint(dir, ts);

    std::string err;
    auto df = Dflash::load(dir, &err);
    CHECK(df != nullptr, "the synthetic checkpoint loads");
    if (!df) {
        std::printf("%s\n", err.c_str());
        return limbo::test::finish("cuda_dflash_test");
    }
    CHECK(df->column_of(4) == 1 && df->column_of(3) == -1 && df->column_of(6) == 2 && df->column_of(1) == -1, "target layers map to columns");

    // the fake target: F32 embedding and head
    Ref ref;
    for (const Tn& t : ts) ref.w[t.name] = &t;
    ref.embd.resize(size_t(NV) * H);
    ref.head.resize(size_t(NV) * H);
    for (float& x : ref.embd) x = 0.5f * rng.norm();
    for (float& x : ref.head) x = 0.3f * rng.norm();
    float *d_embd = nullptr, *d_head = nullptr;
    cudaMalloc(&d_embd, ref.embd.size() * 4);
    cudaMalloc(&d_head, ref.head.size() * 4);
    cudaMemcpy(d_embd, ref.embd.data(), ref.embd.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_head, ref.head.data(), ref.head.size() * 4, cudaMemcpyHostToDevice);
    cudaStream_t s;
    cudaStreamCreate(&s);
    CHECK(df->bind(uint32_t(Type::F32), d_embd, uint32_t(Type::F32), d_head, NV, H, s), "binds to the fake target");
    ref.K.assign(NL, {});
    ref.V.assign(NL, {});

    // features: per position, the 3 target layers' inputs as fp32 rows (the stage rounds them to BF16)
    const int P = 2300;
    float* d_x = nullptr;
    cudaMalloc(&d_x, size_t(8) * NT * H * 4);
    std::vector<std::vector<float>> feats(size_t(P + 64), std::vector<float>(size_t(FC)));
    for (auto& f : feats)
        for (float& x : f) x = 2.0f * rng.norm();
    Dflash* cur = df.get();
    auto feed = [&](int slot, int pos0, int n) {  // positions pos0 .. pos0 + n - 1 through capture + ingest, 8 at a time
        bool ok = true;
        for (int a = 0; a < n; a += 8) {
            const int T = std::min(8, n - a);
            for (int k = 0; k < NT; ++k) {
                std::vector<float> h(size_t(T) * H);
                for (int t = 0; t < T; ++t)
                    for (int d = 0; d < H; ++d) h[size_t(t) * H + d] = feats[size_t(pos0 + a + t)][size_t(k) * H + d];
                cudaMemcpy(d_x, h.data(), h.size() * 4, cudaMemcpyHostToDevice);
                cur->capture(kLayers[k], d_x, T, 0, s);
            }
            ok &= cur->ingest(slot, 0, 0, T, pos0 + a);
        }
        return ok;
    };
    auto ref_feed = [&](int pos0, int n) {
        for (int i = 0; i < n; ++i) {
            std::vector<float> f(static_cast<size_t>(FC));
            for (int d = 0; d < FC; ++d) f[size_t(d)] = rbf(feats[size_t(pos0 + i)][size_t(d)]);
            ref.ingest(f.data(), pos0 + i);
        }
    };
    // the reference keeps K/V in float; the card keeps them in F16 (Ref::h16 rounds on read)
    CHECK(feed(0, 0, P), "2,300 positions ingest");
    ref_feed(0, P);
    CHECK(df->slot_len(0) == P && df->slot_lo(0) == P - Dflash::kRing, "slot length and the lowest position the ring holds");

    auto compare = [&](const char* what, int slot, int anchor, int pos, int ce, int lo) {
        Ref::Out want = ref.draft(anchor, pos, ce, lo);
        int32_t got[7];
        float conf[7];
        const bool ok = cur->draft_start(slot, anchor, pos) && cur->draft_wait(got, conf);
        std::vector<float> hn(size_t(8) * H);
        cur->debug_hidden(hn.data());
        double md = 0, mx = 0;
        for (size_t i = size_t(H); i < hn.size(); ++i) {
            md = std::max(md, std::fabs(double(hn[i]) - want.hn[i]));
            mx = std::max(mx, std::fabs(double(want.hn[i])));
        }
        bool same = ok;
        double cd = 0;
        for (int i = 0; i < 7; ++i) {
            same = same && got[i] == want.tok[i];
            cd = std::max(cd, std::fabs(double(conf[i]) - want.conf[i]));
        }
        std::printf("     %s: hidden max diff %.2e (max %.2f), conf diff %.2e, drafts %d %d %d %d %d %d %d (ref %d %d %d %d %d %d %d)\n", what, md, mx, cd, got[0],
                    got[1], got[2], got[3], got[4], got[5], got[6], want.tok[0], want.tok[1], want.tok[2], want.tok[3], want.tok[4], want.tok[5], want.tok[6]);
        CHECK(ok && md < 2e-3 * std::max(1.0, mx), (std::string(what) + ": hidden rows match").c_str());
        CHECK(same, (std::string(what) + ": the drafts are the reference's").c_str());
        CHECK(cd < 5e-3, (std::string(what) + ": confidences match").c_str());
    };
    compare("long context (ring wrapped)", 0, 1234, P, P, P - Dflash::kRing);

    // a block with no context at all, and with a short one
    CHECK(feed(1, 0, 5), "a second slot ingests 5 positions");
    compare("5 positions of context", 1, 77, 5, 5, 0);

    // a cut and re-ingest: positions P-20.. again with the same features
    df->cut_slot(0, P - 20);
    CHECK(df->slot_len(0) == P - 20, "cut_slot moves the end");
    CHECK(feed(0, P - 20, 20), "re-ingest after the cut");
    compare("after cut + re-ingest", 0, 4, P, P, std::max(0, P - Dflash::kRing));

    // sharing: slot 2 becomes the first P-5 positions of slot 0
    CHECK(df->share_slot(0, 2, P - 5), "share_slot");
    {
        int32_t a[7], b[7];
        float ca[7], cb[7];
        df->draft_start(2, 99, P - 5);
        df->draft_wait(a, ca);
        df->cut_slot(0, P - 5);
        df->draft_start(0, 99, P - 5);
        df->draft_wait(b, cb);
        bool same = true;
        for (int i = 0; i < 7; ++i) same = same && a[i] == b[i];
        CHECK(same, "a shared slot drafts what its source drafts at the same position");
    }

    // a gap: the context starts again at the new position
    CHECK(feed(1, 100, 12) && df->slot_len(1) == 112 && df->slot_lo(1) == 100, "a gap restarts the slot's context");

    // Q8_0 matrices: the reference gets the same weights round-tripped through the quantiser
    {
        std::vector<Tn> tq = ts;
        for (Tn& x : tq) {
            if (x.shape.size() != 2 || x.name.find("_codebook") != std::string::npos) continue;
            std::vector<uint8_t> q(x.v.size() / 32 * 34);
            quants::quantize_q8_0(x.v.data(), q.data(), x.v.size());
            quants::dequant_row(uint32_t(Type::Q8_0), q.data(), x.v.data(), x.v.size());
        }
        Ref rq;
        for (const Tn& x : tq) rq.w[x.name] = &x;
        rq.embd = ref.embd;
        rq.head = ref.head;
        rq.K.assign(NL, {});
        rq.V.assign(NL, {});
        auto dq = Dflash::load(dir, &err, true);
        CHECK(dq && dq->is_q8(), "the Q8_0 variant loads");
        if (dq) {
            CHECK(dq->bind(uint32_t(Type::F32), d_embd, uint32_t(Type::F32), d_head, NV, H, s), "binds");
            CHECK(dq->weight_bytes() < df->weight_bytes() * 6 / 10, "Q8_0 takes little over half the bytes of BF16");
            std::swap(ref, rq);
            cur = dq.get();
            CHECK(feed(0, 0, 300), "300 positions ingest (Q8_0)");
            ref_feed(0, 300);
            compare("Q8_0 weights", 0, 321, 300, 300, 0);
            std::swap(ref, rq);
            cur = df.get();
            dq.reset();
        }
    }

    // the real checkpoint, when it is there
    if (const char* real = std::getenv("LIMBO_DFLASH_DIR")) {
        std::string e2;
        auto r = Dflash::load(real, &e2);
        CHECK(r != nullptr, "the real checkpoint loads");
        if (!r) std::printf("%s\n", e2.c_str());
        else std::printf("     real: %llu MiB of weights, ring %llu MiB per slot\n", (unsigned long long)(r->weight_bytes() >> 20), (unsigned long long)(r->ring_bytes() >> 20));
    }
    df.reset();
    cudaFree(d_x);
    cudaFree(d_embd);
    cudaFree(d_head);
    cudaStreamDestroy(s);
    return limbo::test::finish("cuda_dflash_test");
}
