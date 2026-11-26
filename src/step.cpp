// Copyright (c) 2026 ITERFORM. All rights reserved.
// src/step.cpp — the reference forward pass.
//
// Every step below is the node sequence of bench/rung1-graph-spec.md, in order, with the same shapes.
// The weights are decoded on the fly by the verified decoder (src/quants.cpp, bit-exact against ggml),
// one row at a time, fused into the dot product so no float copy of a weight ever lands in memory.
#include "limbo/step.hpp"

#include <algorithm>
#include <unistd.h>
#include <atomic>
#include <cmath>
#include <cstring>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <set>
#include <thread>

#include "limbo/quants.hpp"
#include "limbo/types.hpp"

namespace limbo::step {

namespace {

constexpr float kSqrt2OverPi = 0.7978845608028654f;

// float -> IEEE half, round to nearest even (what the F16C instruction the oracle's cache writes use does).
uint16_t f32_to_f16(float x) {
    uint32_t bits;
    std::memcpy(&bits, &x, 4);
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = int32_t((bits >> 23) & 0xff) - 127 + 15;
    uint32_t man = bits & 0x7fffffu;
    if (((bits >> 23) & 0xff) == 0xff) return uint16_t(sign | 0x7c00u | (man ? 0x200u : 0));  // inf/nan
    if (exp >= 31) return uint16_t(sign | 0x7c00u);
    uint32_t half, rem, halfway;
    if (exp <= 0) {
        if (exp < -10) return uint16_t(sign);
        man |= 0x800000u;
        const uint32_t shift = uint32_t(14 - exp);
        half = man >> shift;
        rem = man & ((1u << shift) - 1);
        halfway = 1u << (shift - 1);
    } else {
        half = (uint32_t(exp) << 10) | (man >> 13);
        rem = man & 0x1fffu;
        halfway = 0x1000u;
    }
    if (rem > halfway || (rem == halfway && (half & 1))) ++half;   // a carry into the exponent is the right answer
    return uint16_t(sign | half);
}

// ---- the oracle's activation arithmetic (opt-in, Ref::set_oracle_arith) ----
// llama.cpp's CPU matmul never multiplies a quantised weight by the fp32 activation: it first rounds the
// activation to the weight type's partner format (Q8_K blocks of 256 for the K-quants and the i-quants,
// Q8_0 blocks of 32 for Q8_0 and IQ4_NL, f16 for f16 weights; fp32 weights keep fp32) and then takes an
// integer dot. The integer products are exact, so "weight * dequantised(rounded activation)" in float is
// the same number up to float summation order. These helpers write that rounded activation back as float.
enum class ActFmt { None, Q8K, Q80, F16, BF16 };

ActFmt act_fmt(uint32_t type) {
    switch (static_cast<Type>(type)) {
        case Type::F32: return ActFmt::None;
        case Type::F16: return ActFmt::F16;
        case Type::BF16: return ActFmt::BF16;   // a bf16 weight takes its activation as bf16
        case Type::Q8_0: case Type::IQ4_NL: case Type::Q4_0: case Type::Q5_0: case Type::Q2_0: return ActFmt::Q80;
        default: return ActFmt::Q8K;
    }
}

// One Q8_K block: the signed extreme maps to -127, the scale is its inverse, round to nearest, clamp at 127.
void round_q8k(const float* x, float* y, int n) {
    for (int b = 0; b < n; b += 256) {
        float mx = 0, amax = 0;
        for (int j = 0; j < 256; ++j) {
            const float ax = std::fabs(x[b + j]);
            if (ax > amax) { amax = ax; mx = x[b + j]; }
        }
        if (amax == 0.0f) { for (int j = 0; j < 256; ++j) y[b + j] = 0.0f; continue; }
        const float iscale = -127.0f / mx;
        const float d = 1.0f / iscale;
        for (int j = 0; j < 256; ++j) {
            const int v = std::min(127, int(std::nearbyint(iscale * x[b + j])));
            y[b + j] = float(v) * d;
        }
    }
}

// One Q8_0 block: scale amax/127 stored as f16, round to nearest.
void round_q80(const float* x, float* y, int n) {
    for (int b = 0; b < n; b += 32) {
        float amax = 0;
        for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(x[b + j]));
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        const float d16 = quants::f16_to_f32(f32_to_f16(d));
        for (int j = 0; j < 32; ++j) y[b + j] = float(std::nearbyint(x[b + j] * id)) * d16;
    }
}

void round_f16(const float* x, float* y, int n) {
    for (int j = 0; j < n; ++j) y[j] = quants::f16_to_f32(f32_to_f16(x[j]));
}

// float -> bfloat16 -> float, round to nearest even.
void round_bf16(const float* x, float* y, int n) {
    for (int j = 0; j < n; ++j) {
        uint32_t bits;
        std::memcpy(&bits, &x[j], 4);
        if ((bits & 0x7fffffffu) > 0x7f800000u) { y[j] = x[j]; continue; }   // nan stays nan
        bits += 0x7fffu + ((bits >> 16) & 1u);
        bits &= 0xffff0000u;
        std::memcpy(&y[j], &bits, 4);
    }
}

// y = the activation as the oracle's dot product sees it for a weight of `type`; false when it is unchanged.
bool round_act(uint32_t type, const float* x, float* y, int n) {
    switch (act_fmt(type)) {
        case ActFmt::None: return false;
        case ActFmt::Q8K: round_q8k(x, y, n); return true;
        case ActFmt::Q80: round_q80(x, y, n); return true;
        case ActFmt::F16: round_f16(x, y, n); return true;
        case ActFmt::BF16: round_bf16(x, y, n); return true;
    }
    return false;
}

float silu(float x) { return x / (1.0f + std::exp(-x)); }
float softplus(float x) { return x > 20.0f ? x : std::log1p(std::exp(x)); }
float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// Splits [0, n) over `threads` workers, each calling fn(begin, end).
namespace {
bool ref_time_on() {
    static const bool on = std::getenv("LIMBO_REF_TIME") != nullptr;
    return on;
}
struct Phase {
    const char* name;
    std::chrono::steady_clock::time_point t0;
    explicit Phase(const char* n) : name(n), t0(std::chrono::steady_clock::now()) {}
    ~Phase() {
        if (!ref_time_on()) return;
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "  %-22s %8.1f ms\n", name, ms);
    }
};
}  // namespace

void parallel_for(int n, int threads, const std::function<void(int, int)>& fn) {
    if (threads <= 1 || n <= 1) {
        fn(0, n);
        return;
    }
    const int chunk = (n + threads - 1) / threads;
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t) {
        const int b = t * chunk, e = std::min(n, b + chunk);
        if (b >= e) break;
        pool.emplace_back([&fn, b, e] { fn(b, e); });
    }
    for (auto& th : pool) th.join();
}

// One row of a quantised tensor, decoded and dotted with x. The decode happens one block at a time into
// a stack buffer, so nothing is materialised beyond the current block.
float dot_row(uint32_t type, const uint8_t* row, const float* x, int n) {
    const TypeInfo* ti = type_info(type);
    if (!ti) return 0;
    float buf[256];
    float acc = 0;
    // A one-element "block" (F32, F16, BF16) is decoded 256 at a time; the products and the sum are the same.
    const int step = ti->block_elems == 1 ? 256 : int(ti->block_elems);
    for (int i = 0; i < n; i += step) {
        const uint64_t off = ti->block_elems == 1 ? uint64_t(i) * ti->block_bytes : type_bytes(type, uint64_t(i));
        const int k = std::min<int>(step, n - i);
        quants::dequant_row(type, row + off, buf, uint64_t(ti->block_elems == 1 ? k : step));
        for (int j = 0; j < k; ++j) acc += buf[j] * x[i + j];
    }
    return acc;
}

// y[i] = dot(row i of w, x), w of shape [n_in, n_out].
void matvec(uint32_t type, const uint8_t* w, int n_out, int n_in, const float* x, float* y, int threads) {
    const uint64_t row_bytes = type_bytes(type, uint64_t(n_in));
    parallel_for(n_out, threads, [&](int b, int e) {
        for (int i = b; i < e; ++i) y[i] = dot_row(type, w + uint64_t(i) * row_bytes, x, n_in);
    });
}

void rmsnorm(const float* x, const float* w, float* y, int n, float eps) {
    double ss = 0;
    for (int i = 0; i < n; ++i) ss += double(x[i]) * double(x[i]);
    const float inv = float(1.0 / std::sqrt(ss / n + eps));
    for (int i = 0; i < n; ++i) y[i] = x[i] * inv * (w ? w[i] : 1.0f);
}


// Per-head RMS norm over head_dim, no weight, then a fixed scale (the graph's SCALE node).
void head_norm(const float* x, float* y, int n_head, int head_dim, float eps, float scale) {
    for (int h = 0; h < n_head; ++h) {
        const float* xh = x + static_cast<size_t>(h) * head_dim;
        float* yh = y + static_cast<size_t>(h) * head_dim;
        double ss = 0;
        for (int d = 0; d < head_dim; ++d) ss += double(xh[d]) * double(xh[d]);
        const float inv = float(1.0 / std::sqrt(ss / head_dim + eps)) * scale;
        for (int d = 0; d < head_dim; ++d) yh[d] = xh[d] * inv;
    }
}

// Rotary embedding on the first `rope_dim` dims of each head, in pairs (i, i + rope_dim/2) (NEOX pairing).
// `factors`, when given, divides each pair's angle (gemma's proportional rope on its global layers).
void rope(float* vec, int n_head, int head_dim, int rope_dim, int pos, double base, const float* factors) {
    const int pairs = rope_dim / 2;
    for (int i = 0; i < pairs; ++i) {
        double theta = double(pos) / std::pow(base, 2.0 * i / double(rope_dim));
        if (factors) theta /= double(factors[i]);
        const float c = float(std::cos(theta)), s = float(std::sin(theta));
        for (int h = 0; h < n_head; ++h) {
            float* v = vec + static_cast<size_t>(h) * head_dim;
            const float a = v[i], b = v[i + pairs];
            v[i] = a * c - b * s;
            v[i + pairs] = a * s + b * c;
        }
    }
}

void softmax_inplace(float* x, int n) {
    float m = -INFINITY;
    for (int i = 0; i < n; ++i) m = std::max(m, x[i]);
    float sum = 0;
    for (int i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - m);
        sum += x[i];
    }
    const float inv = sum > 0 ? 1.0f / sum : 0.0f;
    for (int i = 0; i < n; ++i) x[i] *= inv;
}

float gelu(float x) { return 0.5f * x * (1.0f + std::tanh(kSqrt2OverPi * x * (1.0f + 0.044715f * x * x))); }
float act_fn(Act a, float x) { return a == Act::Gelu ? gelu(x) : silu(x); }

// A tensor decoded to floats (norm weights, small vectors): the file's own values, whatever the type.
std::vector<float> decode_all(const gguf::Tensor* t) {
    std::vector<float> v(static_cast<size_t>(t->n_elements()));
    quants::dequant_row(t->type, t->data, v.data(), uint64_t(v.size()));
    return v;
}

}  // namespace

// ---- the resolved weights ----------------------------------------------------------------------------
// Everything a layer reads, looked up once at open: big matrices stay as tensors over the mapping, small
// vectors (norm weights, the GDN's decay terms and conv taps) are decoded to floats.
struct Ref::Proj {
    const gguf::Tensor* w = nullptr;
    float scale = 1.0f;   // an optional per-tensor `<name>.scale`, applied after the product
    bool has_scale = false;
};

// A hyper-connection mixer: a grouped norm weight, the low-rank down and up projections and the inject projection.
struct Ref::Hc {
    std::vector<float> norm;
    Proj down, up, inject;
};

struct Ref::Weights {
    struct Layer {
        Hc hc_attn, hc_ffn;
        // the n-gram hash embedding block (qwen4exp)
        Proj ple_key, ple_value;
        std::vector<float> ple_norm_key, ple_norm_query, ple_norm_conv, ple_conv;
        std::vector<float> n_in;        // attn_norm: the mixer's input norm
        std::vector<float> n_mix_out;   // post_attention_norm as gemma uses it: on the mixer's output
        std::vector<float> n_ffn_in;    // the FFN's input norm (qwen: post_attention_norm, gemma: ffn_norm)
        std::vector<float> n_ffn_out;   // post_ffw_norm: on the FFN's output
        std::vector<float> n_moe_in, n_mlp_out, n_moe_out;   // gemma's pre_ffw_norm_2, post_ffw_norm_1, post_ffw_norm_2
        // attention
        Proj wq, wk, wv, wo;
        std::vector<float> q_norm, k_norm;
        // the sparse-attention indexer (qwen4exp)
        Proj idx_q, idx_k;
        std::vector<float> idx_qn, idx_kn;
        // gated delta net
        Proj qkv, z, alpha, beta, ssm_out;
        std::vector<float> conv, dt_bias, ssm_a, ssm_norm;
        // dense MLP
        Proj ffn_gate, ffn_up, ffn_down;
        // routed experts
        Proj router;
        std::vector<float> router_scale, down_scale;
        const gguf::Tensor *exp_gate = nullptr, *exp_up = nullptr, *exp_gate_up = nullptr, *exp_down = nullptr;
        // shared expert
        Proj sh_gate, sh_up, sh_down;
        std::vector<float> sh_gate_inp;
        // per-layer embeddings, layer scalar
        Proj ple_gate, ple_proj;
        std::vector<float> ple_norm;
        float out_scale = 1.0f;
    };
    std::vector<Layer> L;
    Hc hc_head;
    Proj tok_embd, output, ple_model_proj;
    const gguf::Tensor* ple_embd = nullptr;
    std::vector<float> out_norm, ple_proj_norm, rope_freqs;
    // inkling2_loop's own tensors, kept apart from the qwen/gemma recipe fields: the graph that reads
    // them is a different graph (docs/inkling2.md), not a variation of the one above.
    struct InklingLayer {
        Proj wq, wk, wv, wo, wr, wgate, sink, rel_bias;   // rel_bias: [n_rel, extent], the bias table          // wr [n_embd, n_head*n_rel]; sink [n_embd, n_head]
        std::vector<float> q_clip, rel_clip;           // [head_dim, n_head], [n_rel, n_head]
        std::vector<float> q_norm, k_norm;             // per-head RMSNorm weights
        Proj k_conv, v_conv, attn_post_conv;           // depthwise, [sconv, channels]
        Proj k_conv_proj, v_conv_proj;                 // global layers: input-dependent taps
        std::vector<float> q_inp_norm, k_inp_norm, v_inp_norm;   // global layers
        std::vector<float> attn_norm;                  // sliding layers
        Proj res_q_attn;                               // [n_embd, 1 sliding / 3 global]
        std::vector<float> ffn_norm;
        Proj router;                                   // ffn_gate_inp
        std::vector<float> exp_bias;                   // exp_probs_b.bias
        const gguf::Tensor *gate_up = nullptr, *down = nullptr;
        std::vector<float> gate_up_scale, down_scale;  // per-expert dequant scales
        Proj ffn_post_conv, res_q_ffn;
        Proj ctrl_down[4], ctrl_gain[4], ctrl_forget[4], msg_query;
    };
    std::vector<InklingLayer> ink_L;
    std::vector<float> ink_tok_norm, ink_out_norm, ink_out_res_query;
};

Ref::Ref() = default;
Ref::~Ref() = default;

// Reads the file's recipe: which kind of layer each block is and how its norms sit, from the metadata and
// from which tensors exist. An architecture the reference does not implement is refused here.
bool Ref::resolve(std::string* err) {
    const model::Model& m = *model_;
    const model::Geometry& g = m.geometry();
    Config& c = cfg_;
    const bool is_gemma = g.arch == "gemma4";
    const bool is_q4e = g.arch == "qwen4exp";
    const bool is_qwen = g.arch == "qwen35" || g.arch == "qwen35moe" || is_q4e;
    const bool is_inkling = g.arch == "inkling2_loop";
    if (!is_gemma && !is_qwen && !is_inkling) {
        *err = "the reference forward pass has no recipe for architecture \"" + g.arch +
               "\" (qwen35, qwen35moe, qwen4exp, gemma4 and inkling2_loop are implemented)";
        return false;
    }
    c.arch = g.arch;
    w_.reset(new Weights());
    std::set<const gguf::Tensor*> claimed;
    std::string missing;
    auto get = [&](int l, const std::string& role, bool required) -> const gguf::Tensor* {
        const gguf::Tensor* t = m.tensor(l, role);
        if (t) claimed.insert(t);
        else if (required) missing += (l >= 0 ? "blk." + std::to_string(l) + "." : std::string()) + role + " ";
        return t;
    };
    auto vec = [&](int l, const std::string& role, bool required) -> std::vector<float> {
        const gguf::Tensor* t = get(l, role, required);
        return t ? decode_all(t) : std::vector<float>();
    };
    auto proj = [&](int l, const std::string& base, bool required) -> Proj {
        Proj p;
        p.w = get(l, base + ".weight", required);
        // A scalar `<name>.scale` multiplies the product. A longer one is something else's (gemma's
        // ffn_gate_inp.scale is the router's per-channel input multiplier) and is claimed by that consumer.
        if (const gguf::Tensor* s = m.tensor(l, base + ".scale"); s && s->n_elements() == 1) {
            claimed.insert(s);
            p.scale = decode_all(s)[0];
            p.has_scale = true;
        }
        return p;
    };

    // ---- model level -------------------------------------------------------------------------------
    Weights& W = *w_;
    W.tok_embd = proj(-1, "token_embd", true);
    auto hc_mod = [&](int l, const std::string& base, bool inject) {
        Hc h;
        h.norm = vec(l, base + "_norm.weight", true);
        h.down = proj(l, base + "_down", true);
        h.up = proj(l, base + "_up", true);
        if (inject) h.inject = proj(l, base + "_inject", true);
        return h;
    };
    if (is_q4e) {
        // Hyper-connections: no output_norm, the head mixer carries it.
        c.hc_count = g.hc_count;
        c.hc_rank = g.hc_rank;
        if (c.hc_count < 2 || c.hc_rank < 1) {
            *err = "qwen4exp without hyper-connection metadata";
            return false;
        }
        W.hc_head = hc_mod(-1, "output_hc", false);
    } else {
        W.out_norm = vec(-1, "output_norm.weight", true);
    }
    W.output = proj(-1, "output", false);
    c.tied_head = W.output.w == nullptr;
    if (c.tied_head) W.output = W.tok_embd;
    c.logit_softcap = float(g.logit_softcap);
    c.embed_scale = is_gemma ? std::sqrt(float(g.n_embd)) : 1.0f;
    if (g.n_embd_per_layer > 0) {
        c.n_embd_per_layer = g.n_embd_per_layer;
        W.ple_embd = get(-1, "per_layer_token_embd.weight", true);
        if (is_q4e) {
            c.ple_ngram = g.ple_ngram;
            c.ple_heads_per_ngram = g.ple_heads_per_ngram;
            c.ple_conv_kernel = g.ple_conv_kernel;
            c.ple_eos = g.ple_eos;
            c.ple_multipliers = g.ple_multipliers;
            c.ple_head_offsets = g.ple_head_offsets;
            c.ple_head_vocab = g.ple_head_vocab;
            const int n_heads = (c.ple_ngram - 1) * c.ple_heads_per_ngram;
            if (c.ple_ngram < 2 || int(c.ple_multipliers.size()) < c.ple_ngram || int(c.ple_head_offsets.size()) < n_heads ||
                int(c.ple_head_vocab.size()) < n_heads || c.ple_conv_kernel < 1) {
                *err = "the n-gram hash embedding metadata is incomplete";
                return false;
            }
        } else {
            W.ple_model_proj = proj(-1, "per_layer_model_proj", true);
            W.ple_proj_norm = vec(-1, "per_layer_proj_norm.weight", true);
        }
    }
    const bool has_rope_freqs = m.tensor(-1, "rope_freqs.weight") != nullptr;
    if (has_rope_freqs) W.rope_freqs = vec(-1, "rope_freqs.weight", true);

    // inkling2_loop: the constants and the two model-level tensors the graph needs beyond the common
    // embedding and head (docs/inkling2.md). Every one is the file's own key; nothing is defaulted in.
    if (is_inkling) {
        c.ink.on = true;
        c.ink.rel_length = g.rel_length;
        c.ink.rel_extent = g.rel_extent;
        c.ink.qk_clip = float(g.qk_clip);
        c.ink.rel_clip = float(g.rel_clip);
        c.ink.log_floor = g.log_scale_floor;
        c.ink.log_alpha = float(g.log_scale_alpha);
        c.ink.situ_beta = float(g.situ_beta);
        c.ink.situ_lin_beta = float(g.situ_linear_beta);
        c.ink.sconv = g.sconv;
        c.ink.route_scale = float(g.expert_weights_scale);
        c.ink.loop_start = g.loop_start;
        c.ink.loop_end = g.loop_end;
        c.ink.loop_count = g.loop_count;
        c.ink.gate_rank = g.loop_gate_rank;
        c.ink.res_block = g.attn_res_block;
        if (c.ink.rel_extent <= 0 || c.ink.rel_length <= 0 || c.ink.sconv < 2 || c.ink.res_block <= 0 ||
            c.ink.loop_count < 2 || c.ink.gate_rank < 1 || c.ink.situ_beta <= 0 || c.ink.situ_lin_beta <= 0) {
            *err = "inkling2_loop without its relative-bias, short-conv, attn-res, loop or SiTU metadata";
            return false;
        }
        W.ink_tok_norm = vec(-1, "token_embd_norm.weight", true);
        W.ink_out_res_query = vec(-1, "output_res_query.weight", true);
        c.ink.vocab_unpadded = g.n_vocab_unpadded;
        c.logit_softcap = float(g.logit_softcap);
    }

    // ---- layers ------------------------------------------------------------------------------------
    const int n_main = g.n_layer_main;
    c.layers.assign(static_cast<size_t>(n_main), LayerRecipe());
    W.L.assign(static_cast<size_t>(n_main), Weights::Layer());
    if (is_inkling) W.ink_L.assign(static_cast<size_t>(n_main), Weights::InklingLayer());
    for (int l = 0; l < n_main; ++l) {
        LayerRecipe& r = c.layers[static_cast<size_t>(l)];
        Weights::Layer& L = W.L[static_cast<size_t>(l)];
        if (is_inkling) {
            Weights::InklingLayer& K = W.ink_L[l];
            const bool swa = g.is_swa(l);
            r.mixer = Mixer::Attention;
            r.attn.sliding = swa;
            r.attn.window = g.sliding_window;
            r.attn.n_head = g.n_head_at(l);
            r.attn.n_head_kv = g.n_head_kv_at(l);
            r.attn.head_dim = g.head_dim;
            r.attn.rope_dim = 0;  // no RoPE at all: the relative bias carries the positions
            r.attn.v_norm = true; // V is RMS-normalised per head, WITHOUT a gamma (no tensor for it)
            r.ffn.kind = FfnKind::MoeShared;   // routed experts only; n_ff_shexp stays 0
            r.ffn.n_expert = g.n_expert;
            r.ffn.n_used = g.n_expert_used;
            r.ffn.n_ff_exp = g.n_ff_exp;
            r.ffn.n_ff_shexp = 0;
            r.ffn.fused_gate_up = true;
            r.ffn.expert_scale = true;
            K.wq = proj(l, "attn_q", true);
            K.wk = proj(l, "attn_k", true);
            K.wv = proj(l, "attn_v", true);
            K.wo = proj(l, "attn_output", true);
            K.wr = proj(l, "attn_rel", true);
            K.rel_bias = proj(l, "attn_rel_bias", true);
            K.wgate = proj(l, "attn_gate", true);
            K.sink = proj(l, "attn_sink_proj", true);
            K.q_clip = vec(l, "attn_q_clip.weight", true);
            K.rel_clip = vec(l, "attn_rel_clip.weight", true);
            K.k_conv = proj(l, "attn_k_conv", true);
            K.v_conv = proj(l, "attn_v_conv", true);
            K.attn_post_conv = proj(l, "attn_post_conv", true);
            if (swa) {
                K.attn_norm = vec(l, "attn_norm.weight", true);
            } else {
                K.q_inp_norm = vec(l, "attn_q_inp_norm.weight", true);
                K.k_inp_norm = vec(l, "attn_k_inp_norm.weight", true);
                K.v_inp_norm = vec(l, "attn_v_inp_norm.weight", true);
                K.k_conv_proj = proj(l, "attn_k_conv_proj", true);
                K.v_conv_proj = proj(l, "attn_v_conv_proj", true);
            }
            K.res_q_attn = proj(l, "attn_res_query", l > 0);   // layer 0 has none
            K.ffn_norm = vec(l, "ffn_norm.weight", true);
            K.router = proj(l, "ffn_gate_inp", true);
            K.exp_bias = vec(l, "exp_probs_b.bias", true);
            K.gate_up = get(l, "ffn_gate_up_exps.weight", true);
            K.down = get(l, "ffn_down_exps.weight", true);
            K.gate_up_scale = vec(l, "ffn_gate_up_exps.scale", false);
            K.down_scale = vec(l, "ffn_down_exps.scale", false);
            K.ffn_post_conv = proj(l, "ffn_post_conv", true);
            K.res_q_ffn = proj(l, "ffn_res_query", true);
            if (l >= c.ink.loop_start && l < c.ink.loop_end) {
                static const char* kCtrl[4] = {"q", "k", "v", "ffn"};
                for (int ci = 0; ci < 4; ++ci) {
                    const std::string b = std::string("loop_") + kCtrl[ci] + "_ctrl_";
                    K.ctrl_down[ci] = proj(l, b + "down", true);
                    K.ctrl_gain[ci] = proj(l, b + "gain", true);
                    K.ctrl_forget[ci] = proj(l, b + "forget", true);
                }
                K.msg_query = proj(l, "loop_msg_query", true);
            }
            continue;
        }
        if (!is_q4e) L.n_in = vec(l, "attn_norm.weight", true);
        const bool linear = std::find(m.linear_layers().begin(), m.linear_layers().end(), l) != m.linear_layers().end();
        r.mixer = linear ? Mixer::Gdn : Mixer::Attention;

        if (linear) {
            L.qkv = proj(l, "attn_qkv", true);
            L.z = proj(l, "attn_gate", true);
            r.gdn_sigmoid_gate = is_q4e;
            L.alpha = proj(l, "ssm_alpha", true);
            L.beta = proj(l, "ssm_beta", true);
            L.ssm_out = proj(l, "ssm_out", true);
            L.conv = vec(l, "ssm_conv1d.weight", true);
            L.dt_bias = vec(l, "ssm_dt.bias", true);
            L.ssm_a = vec(l, "ssm_a", true);
            L.ssm_norm = vec(l, "ssm_norm.weight", true);
        } else {
            AttnRecipe& a = r.attn;
            a.sliding = g.is_swa(l);
            a.window = a.sliding ? g.sliding_window : 0;
            a.n_head = g.n_head;
            a.n_head_kv = g.n_head_kv_at(l);
            a.head_dim = a.sliding && g.head_dim_swa > 0 ? g.head_dim_swa : g.head_dim;
            a.rope_dim = a.sliding && g.rope_dim_swa > 0 ? g.rope_dim_swa : g.rope_dim;
            a.rope_base = a.sliding && g.rope_base_swa > 0 ? g.rope_base_swa : g.rope_base;
            a.rope_factors = has_rope_freqs && !a.sliding;
            a.own_kv = g.has_own_kv(l);
            a.kv_src = g.kv_source(l);
            a.scale = is_gemma ? 1.0f : float(1.0 / std::sqrt(double(a.head_dim)));
            L.wq = proj(l, "attn_q", true);
            const int64_t q_rows = L.wq.w ? L.wq.w->ne[1] : 0;
            a.q_gate = q_rows == 2 * int64_t(a.n_head) * a.head_dim;
            if (!a.q_gate && q_rows != int64_t(a.n_head) * a.head_dim && L.wq.w) {
                *err = "layer " + std::to_string(l) + ": attn_q has " + std::to_string(q_rows) + " rows for " +
                       std::to_string(a.n_head) + " heads of " + std::to_string(a.head_dim);
                return false;
            }
            L.wo = proj(l, "attn_output", true);
            L.q_norm = vec(l, "attn_q_norm.weight", true);
            if (is_q4e && l < int(g.compress_ratios.size()) && g.compress_ratios[size_t(l)] > 0) {
                a.qsa_top_k = g.indexer_top_k;
                // test hook: a smaller budget makes the selection act on a short prompt (a multiple of the block size)
                if (const char* ov = std::getenv("LIMBO_QSA_TOP_K")) a.qsa_top_k = std::atoi(ov);
                a.qsa_kpool = g.compress_ratios[size_t(l)];
                a.qsa_heads = g.indexer_heads;
                a.qsa_dim = g.indexer_dim;
                L.idx_q = proj(l, "indexer.q_proj", true);
                L.idx_k = proj(l, "indexer.k_proj", true);
                L.idx_qn = vec(l, "indexer.q_norm.weight", true);
                L.idx_kn = vec(l, "indexer.k_norm.weight", true);
            }
            if (a.own_kv) {
                L.wk = proj(l, "attn_k", true);
                L.k_norm = vec(l, "attn_k_norm.weight", true);
                L.wv = proj(l, "attn_v", false);
                a.v_from_k = L.wv.w == nullptr;
                a.v_norm = is_gemma;
            }
        }

        // The norms. Qwen names the FFN's input norm post_attention_norm; gemma uses that name for a norm on
        // the mixer's output and has its own ffn_norm for the FFN's input.
        if (is_gemma) {
            r.post_mixer_norm = true;
            L.n_mix_out = vec(l, "post_attention_norm.weight", true);
            L.n_ffn_in = vec(l, "ffn_norm.weight", true);
            r.post_ffn_norm = true;
            L.n_ffn_out = vec(l, "post_ffw_norm.weight", true);
        } else if (!is_q4e) {
            L.n_ffn_in = vec(l, "post_attention_norm.weight", true);
        }
        if (is_q4e) {
            L.hc_attn = hc_mod(l, "hc_attn", true);
            L.hc_ffn = hc_mod(l, "hc_ffn", true);
            if (m.tensor(l, "ple_key.weight")) {
                r.ple_hash = true;
                L.ple_key = proj(l, "ple_key", true);
                L.ple_value = proj(l, "ple_value", true);
                L.ple_norm_key = vec(l, "ple_norm_key.weight", true);
                L.ple_norm_query = vec(l, "ple_norm_query.weight", true);
                L.ple_norm_conv = vec(l, "ple_norm_conv.weight", true);
                L.ple_conv = vec(l, "ple_conv1d.weight", true);
            }
        }

        // The FFN.
        FfnRecipe& f = r.ffn;
        f.act = is_gemma ? Act::Gelu : Act::Silu;
        f.n_ff = g.n_ff_at(l);
        f.n_expert = g.n_expert;
        f.n_used = g.n_expert_used;
        f.n_ff_exp = g.n_ff_exp;
        const bool routed = m.tensor(l, "ffn_gate_inp.weight") != nullptr;
        if (!routed) {
            f.kind = FfnKind::Dense;
            L.ffn_gate = proj(l, "ffn_gate", true);
            L.ffn_up = proj(l, "ffn_up", true);
            L.ffn_down = proj(l, "ffn_down", true);
        } else {
            L.router = proj(l, "ffn_gate_inp", true);
            L.exp_gate_up = get(l, "ffn_gate_up_exps.weight", false);
            f.fused_gate_up = L.exp_gate_up != nullptr;
            if (!f.fused_gate_up) {
                L.exp_gate = get(l, "ffn_gate_exps.weight", true);
                L.exp_up = get(l, "ffn_up_exps.weight", true);
            }
            L.exp_down = get(l, "ffn_down_exps.weight", true);
            if (const gguf::Tensor* ds = get(l, "ffn_down_exps.scale", false)) {
                L.down_scale = decode_all(ds);
                f.expert_scale = true;
            }
            if (is_gemma) {
                f.kind = FfnKind::DenseAndMoe;
                L.ffn_gate = proj(l, "ffn_gate", true);
                L.ffn_up = proj(l, "ffn_up", true);
                L.ffn_down = proj(l, "ffn_down", true);
                L.router_scale = vec(l, "ffn_gate_inp.scale", true);
                f.router_scaled = true;
                L.n_moe_in = vec(l, "pre_ffw_norm_2.weight", true);
                L.n_mlp_out = vec(l, "post_ffw_norm_1.weight", true);
                L.n_moe_out = vec(l, "post_ffw_norm_2.weight", true);
            } else {
                f.kind = FfnKind::MoeShared;
                f.n_ff_shexp = g.n_ff_shexp;
                L.sh_gate = proj(l, "ffn_gate_shexp", true);
                L.sh_up = proj(l, "ffn_up_shexp", true);
                L.sh_down = proj(l, "ffn_down_shexp", true);
                L.sh_gate_inp = vec(l, "ffn_gate_inp_shexp.weight", true);
            }
        }

        if (g.n_embd_per_layer > 0 && !is_q4e) {
            r.ple = true;
            L.ple_gate = proj(l, "inp_gate", true);
            L.ple_proj = proj(l, "proj", true);
            L.ple_norm = vec(l, "post_norm.weight", true);
        }
        if (const gguf::Tensor* os = get(l, "layer_output_scale.weight", false)) {
            L.out_scale = decode_all(os)[0];
            r.out_scale = true;
        }
    }
    if (!missing.empty()) {
        *err = "tensors the " + g.arch + " recipe needs are not in the file: " + missing;
        return false;
    }

    // ---- every tensor of the file is either run or named as not run --------------------------------
    std::string stray;
    for (const gguf::Tensor* t : m.tensors()) {
        if (claimed.count(t)) continue;
        const std::string name(t->name);
        int layer = -1;
        if (name.rfind("blk.", 0) == 0) layer = std::atoi(name.c_str() + 4);
        if (layer >= 0 && layer == g.mtp_layer) {
            unused_.push_back(name + "  (the MTP block: not run by the reference)");
        } else if (layer >= 0 && layer < g.n_layer && !g.has_own_kv(layer) &&
                   (name.find(".attn_k.") != std::string::npos || name.find(".attn_v.") != std::string::npos ||
                    name.find(".attn_k_norm.") != std::string::npos)) {
            unused_.push_back(name + "  (reads the K/V cache of layer " + std::to_string(g.kv_source(layer)) + ")");
        } else {
            stray += name + " ";
        }
    }
    if (!stray.empty()) {
        *err = "tensors of the file the " + g.arch + " recipe does not read: " + stray;
        return false;
    }
    return true;
}

std::unique_ptr<Ref> Ref::open(const std::vector<std::string>& shards, int n_ctx, int threads, std::string* err) {
    std::unique_ptr<Ref> r(new Ref());
    r->model_ = model::Model::open(shards, err);
    if (!r->model_) return nullptr;
    r->threads_ = std::max(1, threads);
    const model::Geometry& g = r->model_->geometry();
    Config& c = r->cfg_;
    c.n_layer = g.n_layer;
    c.n_layer_main = g.n_layer_main;
    c.n_embd = g.n_embd;
    c.n_head = g.n_head;
    c.n_head_kv = g.n_head_kv;
    c.head_dim = g.head_dim;
    c.rope_dim = g.rope_dim;
    c.n_ctx = n_ctx;
    c.n_vocab = g.n_vocab;
    c.n_expert = g.n_expert;
    c.n_expert_used = g.n_expert_used;
    c.n_ff_exp = g.n_ff_exp;
    c.n_ff_shexp = g.n_ff_shexp;
    c.rms_eps = g.rms_eps > 0 ? g.rms_eps : 1e-6;
    c.rope_base = g.rope_base > 0 ? g.rope_base : 1e7;
    c.full_layers = r->model_->full_layers();

    // The GDN widths come from the file's own keys and are checked against the tensors: qkv is one fused
    // projection whose width splits into q, k and v, with value heads tiled over the key heads.
    if (g.ssm_inner > 0) {
        c.n_k_head = g.ssm_groups;
        c.n_v_head = g.ssm_dt_rank;
        c.gdn_head = g.ssm_state;
        c.dt_rank = g.ssm_dt_rank;
        c.conv_kernel = g.ssm_conv;
        c.qkv_width = 2 * g.ssm_groups * g.ssm_state + g.ssm_inner;
        if (c.n_v_head % c.n_k_head != 0 || c.n_v_head * c.gdn_head != g.ssm_inner) {
            *err = "the GDN geometry in the metadata is inconsistent";
            return nullptr;
        }
        if (!r->model_->linear_layers().empty()) {
            const int l0 = r->model_->linear_layers()[0];
            const gguf::Tensor* qkv = r->model_->tensor(l0, "attn_qkv.weight");
            const gguf::Tensor* conv = r->model_->tensor(l0, "ssm_conv1d.weight");
            const gguf::Tensor* norm = r->model_->tensor(l0, "ssm_norm.weight");
            const gguf::Tensor* a = r->model_->tensor(l0, "ssm_a");
            // ssm_conv1d.weight is {kernel, channels} in ggml's own order (ne[0] = kernel), which the
            // oracle's node line confirms: SSM_CONV(conv_input{4, 8192}, blk.0.ssm_conv1d.weight{4, 8192}).
            if (!qkv || !conv || !norm || !a || qkv->ne[1] != c.qkv_width || conv->ne[0] != c.conv_kernel ||
                conv->ne[1] != c.qkv_width || norm->ne[0] != c.gdn_head || a->ne[0] != c.dt_rank) {
                *err = "the GDN tensors of layer " + std::to_string(l0) + " disagree with the metadata";
                return nullptr;
            }
        }
    }
    if (!r->resolve(err)) return nullptr;
    if (c.ink.on) {
        // The head keeps only the unpadded vocabulary rows: token_embd and output are padded to 201024,
        // the graph keeps 200058 (inkling2.cpp:442). Everything downstream sizes on cfg_.n_vocab.
        if (c.ink.vocab_unpadded <= 0 || c.ink.vocab_unpadded > c.n_vocab) {
            *err = "inkling2_loop without a usable vocab_size_unpadded";
            return nullptr;
        }
        c.n_vocab = c.ink.vocab_unpadded;
        // One state set per visit slot: the physical layers' own, then one per looped pass beyond the
        // first. Weights are shared by layer_phys, state is not (inkling2-loop.cpp post_arch_tensors).
        const int n_slot = c.n_layer + (c.ink.loop_count - 1) * (c.ink.loop_end - c.ink.loop_start);
        const int kvw = c.n_head_kv * c.head_dim;
        const int conv_ch = 2 * kvw + 2 * c.n_embd;   // k, v, the attention output, the FFN output
        r->ink_k_cache_.assign(static_cast<size_t>(n_slot), {});
        r->ink_v_cache_.assign(static_cast<size_t>(n_slot), {});
        r->ink_conv_hist_.assign(static_cast<size_t>(n_slot), {});
        for (int s = 0; s < n_slot; ++s) {
            r->ink_k_cache_[static_cast<size_t>(s)].assign(static_cast<size_t>(n_ctx) * kvw, 0);
            r->ink_v_cache_[static_cast<size_t>(s)].assign(static_cast<size_t>(n_ctx) * kvw, 0);
            r->ink_conv_hist_[static_cast<size_t>(s)].assign(
                static_cast<size_t>(c.ink.sconv - 1) * conv_ch, 0.0f);
        }
    }

    const int e = c.n_embd, qkvw = c.qkv_width > 0 ? c.qkv_width : 4 * c.n_embd;
    r->x_.assign(static_cast<size_t>(e), 0);
    r->xn_.assign(static_cast<size_t>(e), 0);
    r->tmp_.assign(static_cast<size_t>(qkvw), 0);
    r->q_.assign(static_cast<size_t>(qkvw), 0);
    r->k_.assign(static_cast<size_t>(qkvw), 0);
    r->v_.assign(static_cast<size_t>(qkvw), 0);
    r->attn_.assign(static_cast<size_t>(qkvw), 0);
    r->logits_.assign(static_cast<size_t>(c.n_vocab), 0);
    if (c.n_embd_per_layer > 0) r->ple_in_.assign(static_cast<size_t>(c.n_layer_main) * c.n_embd_per_layer, 0.0f);
    if (c.hc_count > 0) {
        r->stack_.assign(static_cast<size_t>(c.hc_count) * e, 0.0f);
        r->tok_hist_.assign(static_cast<size_t>(n_ctx), -1);
        if (c.ple_ngram > 0)
            r->ple_hist_.assign(static_cast<size_t>(c.ple_conv_kernel - 1) * c.ple_ngram * c.hc_count * e, 0.0f);
    }

    const int n_main = c.n_layer_main > 0 ? c.n_layer_main : c.n_layer;
    r->gdn_state_.assign(static_cast<size_t>(n_main), {});
    r->conv_state_.assign(static_cast<size_t>(n_main), {});
    r->k_cache_.assign(static_cast<size_t>(n_main), {});
    r->v_cache_.assign(static_cast<size_t>(n_main), {});
    for (int l = 0; l < n_main; ++l) {
        const LayerRecipe& lr = c.layers[static_cast<size_t>(l)];
        if (lr.mixer == Mixer::Gdn) {
            r->gdn_state_[static_cast<size_t>(l)].assign(static_cast<size_t>(c.n_v_head) * c.gdn_head * c.gdn_head, 0.0f);
            r->conv_state_[static_cast<size_t>(l)].assign(static_cast<size_t>(c.conv_kernel - 1) * qkvw, 0.0f);
        } else if (lr.attn.own_kv) {
            const size_t kv = static_cast<size_t>(n_ctx) * lr.attn.n_head_kv * lr.attn.head_dim;
            r->k_cache_[static_cast<size_t>(l)].assign(kv, 0);
            r->v_cache_[static_cast<size_t>(l)].assign(kv, 0);
            if (lr.attn.qsa_kpool > 0) {
                r->idx_raw_.resize(static_cast<size_t>(n_main));
                r->idx_pool_.resize(static_cast<size_t>(n_main));
                r->idx_raw_[static_cast<size_t>(l)].assign(static_cast<size_t>(n_ctx) * lr.attn.qsa_dim, 0);
                r->idx_pool_[static_cast<size_t>(l)].assign(static_cast<size_t>(n_ctx / lr.attn.qsa_kpool + 1) * lr.attn.qsa_dim, 0);
            }
        }
    }
    return r;
}

void Ref::reset() {
    for (auto& s : gdn_state_) std::fill(s.begin(), s.end(), 0.0f);
    for (auto& s : conv_state_) std::fill(s.begin(), s.end(), 0.0f);
    for (auto& s : k_cache_) std::fill(s.begin(), s.end(), uint16_t(0));
    for (auto& s : v_cache_) std::fill(s.begin(), s.end(), uint16_t(0));
    for (auto& s : idx_raw_) std::fill(s.begin(), s.end(), uint16_t(0));
    for (auto& s : idx_pool_) std::fill(s.begin(), s.end(), uint16_t(0));
    for (auto& s : ink_k_cache_) std::fill(s.begin(), s.end(), uint16_t(0));
    for (auto& s : ink_v_cache_) std::fill(s.begin(), s.end(), uint16_t(0));
    for (auto& s : ink_conv_hist_) std::fill(s.begin(), s.end(), 0.0f);
    std::fill(ple_hist_.begin(), ple_hist_.end(), 0.0f);
    std::fill(tok_hist_.begin(), tok_hist_.end(), int32_t(-1));
}

void Ref::mv(const Proj& p, int n_out, int n_in, const float* x, float* y) {
    const gguf::Tensor* t = p.w;
    if (oracle_arith_) {
        std::vector<float> xq(static_cast<size_t>(n_in));
        if (round_act(t->type, x, xq.data(), n_in)) {
            matvec(t->type, t->data, n_out, n_in, xq.data(), y, threads_);
            if (p.has_scale) for (int i = 0; i < n_out; ++i) y[i] *= p.scale;
            return;
        }
    }
    matvec(t->type, t->data, n_out, n_in, x, y, threads_);
    if (p.has_scale) for (int i = 0; i < n_out; ++i) y[i] *= p.scale;
}

// The token's embedding (scaled when the recipe says so) and, for per-layer-embedding models, the token's
// input to every layer.
void Ref::embed_token(int32_t token, float* x) {
    const Config& c = cfg_;
    const gguf::Tensor* t = w_->tok_embd.w;
    const uint64_t row_bytes = type_bytes(t->type, uint64_t(c.n_embd));
    quants::dequant_row(t->type, t->data + uint64_t(token) * row_bytes, x, uint64_t(c.n_embd));
    if (c.embed_scale != 1.0f) for (int i = 0; i < c.n_embd; ++i) x[i] *= c.embed_scale;

    if (c.n_embd_per_layer > 0 && c.ple_ngram == 0) {
        // input_l = rms_norm(model_proj(x) , weight / sqrt 2) + embedding_row_l * sqrt(width) / sqrt 2, where the
        // projection's own 1/sqrt(n_embd) scale is folded into the norm's epsilon (rms_norm(a*s, eps) =
        // rms_norm(a, eps / s^2)), exactly as the oracle's graph folds them.
        const int np = c.n_embd_per_layer, nl = c.n_layer_main;
        const Weights& W = *w_;
        const uint64_t prow = type_bytes(W.ple_embd->type, uint64_t(np) * uint64_t(c.n_layer));
        std::vector<float> emb(static_cast<size_t>(np) * c.n_layer), proj(static_cast<size_t>(np) * c.n_layer);
        quants::dequant_row(W.ple_embd->type, W.ple_embd->data + uint64_t(token) * prow, emb.data(),
                            uint64_t(np) * uint64_t(c.n_layer));
        mv(W.ple_model_proj, np * c.n_layer, c.n_embd, x, proj.data());
        const float inv_sqrt2 = 1.0f / std::sqrt(2.0f);
        const float proj_scale = 1.0f / std::sqrt(float(c.n_embd));
        const float eps = float(c.rms_eps) / (proj_scale * proj_scale);
        std::vector<float> nw(static_cast<size_t>(np));
        for (int i = 0; i < np; ++i) nw[static_cast<size_t>(i)] = W.ple_proj_norm[static_cast<size_t>(i)] * inv_sqrt2;
        const float emb_scale = std::sqrt(float(np)) / std::sqrt(2.0f);
        for (int l = 0; l < nl; ++l) {
            float* out = ple_in_.data() + static_cast<size_t>(l) * np;
            rmsnorm(proj.data() + static_cast<size_t>(l) * np, nw.data(), out, np, eps);
            for (int i = 0; i < np; ++i) out[i] += emb[static_cast<size_t>(l) * np + i] * emb_scale;
        }
    }
}

void Ref::embed(int32_t token, float* x) { embed_token(token, x); }

// Mapped file pages (the weights) this process holds, from /proc/self/statm's "shared" column.
void Ref::trim_resident() {
    std::FILE* f = std::fopen("/proc/self/statm", "r");
    if (!f) return;
    unsigned long size = 0, resident = 0, shared = 0;
    const int got = std::fscanf(f, "%lu %lu %lu", &size, &resident, &shared);
    std::fclose(f);
    if (got == 3 && uint64_t(shared) * uint64_t(sysconf(_SC_PAGESIZE)) > rss_budget_) model_->drop_resident();
}

const float* Ref::forward(int32_t token, int pos, bool want_logits) {
    if (rss_budget_ > 0) trim_resident();
    const Config& c = cfg_;
    if (c.ink.on) return forward_inkling(token, pos, want_logits);
    if (c.hc_count > 0) return forward_hc(token, pos, want_logits);
    cur_pos_ = pos;
    embed_token(token, x_.data());
    if (trace_.on && pos == 0) {
        // Only the first position is traced, and the buffers are cleared only then: clearing on every
        // call leaves the last token's (empty) pass as the trace, which is how the per-layer dump came
        // out blank while the layer-0 nodes were fine.
        trace_.x0.assign(x_.begin(), x_.begin() + 8);
        trace_.attn_out.assign(static_cast<size_t>(c.n_layer_main), {});
        trace_.l_out.assign(static_cast<size_t>(c.n_layer_main), {});
    }

    if (trace_.on && trace_.all_pos) {
        if (trace_.l_out_pos.size() <= static_cast<size_t>(pos)) {
            trace_.l_out_pos.resize(static_cast<size_t>(pos) + 1);
            trace_.attn_out_pos.resize(static_cast<size_t>(pos) + 1);
            trace_.moe_pos.resize(static_cast<size_t>(pos) + 1);
        }
        trace_.moe_pos[static_cast<size_t>(pos)].assign(static_cast<size_t>(c.n_layer_main), {});
        trace_.l_out_pos[static_cast<size_t>(pos)].assign(static_cast<size_t>(c.n_layer_main), {});
        trace_.attn_out_pos[static_cast<size_t>(pos)].assign(static_cast<size_t>(c.n_layer_main), {});
    }
    const int n_layers = (layer_limit_ > 0 && layer_limit_ < c.n_layer_main) ? layer_limit_ : c.n_layer_main;
    for (int l = 0; l < n_layers; ++l) {
        forward_layer(l, pos, x_.data());
        if (trace_.on && pos == 0) trace_.l_out[static_cast<size_t>(l)] = x_;
        if (trace_.on && trace_.all_pos) trace_.l_out_pos[static_cast<size_t>(pos)][static_cast<size_t>(l)] = x_;
    }

    if (!want_logits) return nullptr;
    return head(x_.data());
}

const float* Ref::head(const float* x) {
    const Config& c = cfg_;
    const Weights& W = *w_;
    std::vector<float> rn(static_cast<size_t>(c.n_embd));
    if (c.hc_count > 0) {
        // the head's own hyper-connection mixer is the output norm: x is the stack of streams
        std::vector<float> unused(static_cast<size_t>(c.hc_count));
        hc_mix(W.hc_head, x, rn.data(), unused.data());
    } else {
        rmsnorm(x, W.out_norm.data(), rn.data(), c.n_embd, float(c.rms_eps));
    }
    if (trace_.on) trace_.result_norm = rn;

    { Phase ph("head matvec"); mv(W.output, c.n_vocab, c.n_embd, rn.data(), logits_.data()); }
    if (c.logit_softcap > 0.0f) {
        const float inv = 1.0f / c.logit_softcap;
        for (int i = 0; i < c.n_vocab; ++i) logits_[static_cast<size_t>(i)] = c.logit_softcap * std::tanh(logits_[static_cast<size_t>(i)] * inv);
    }
    if (trace_.on) trace_.logits = logits_;
    return logits_.data();
}

// One layer in place: x += mixer(norm(x)), then the FFN half.
void Ref::forward_layer(int l, int pos, float* x) {
    cur_pos_ = pos;
    const Config& c = cfg_;
    if (c.hc_count > 0) {   // x is the stack of hc_count streams
        layer_hc(l, pos, x);
        return;
    }
    const LayerRecipe& r = c.layers[static_cast<size_t>(l)];
    const Weights::Layer& W = w_->L[static_cast<size_t>(l)];
    const int e = c.n_embd;
    rmsnorm(x, W.n_in.data(), xn_.data(), e, float(c.rms_eps));
    if (trace_.on && l == 0 && pos == 0) trace_.xn0.assign(xn_.begin(), xn_.begin() + 8);
    std::vector<float> mix(static_cast<size_t>(e));
    if (r.mixer == Mixer::Gdn) layer_linear(l, xn_.data(), mix.data());
    else layer_attention(l, pos, xn_.data(), mix.data());
    if (r.post_mixer_norm) rmsnorm(mix.data(), W.n_mix_out.data(), mix.data(), e, float(c.rms_eps));
    for (int i = 0; i < e; ++i) x[i] += mix[static_cast<size_t>(i)];
    if (trace_.on && !trace_.attn_out.empty()) trace_.attn_out[static_cast<size_t>(l)] = std::vector<float>(x, x + e);
    if (trace_.on && trace_.all_pos) trace_.attn_out_pos[static_cast<size_t>(cur_pos_)][static_cast<size_t>(l)] = std::vector<float>(x, x + e);
    if (attn_only_) return;
    layer_ffn(l, x);
}

void Ref::forward_layer_attn(int l, int pos, float* x) {
    attn_only_ = true;
    forward_layer(l, pos, x);
    attn_only_ = false;
}

void Ref::forward_layer_ffn(int l, float* x) {
    if (cfg_.hc_count > 0) layer_hc_ffn(l, x);
    else layer_ffn(l, x);
}

std::vector<int> Ref::route_from_stack(int l, const float* stack) {
    const Config& c = cfg_;
    const FfnRecipe& f = c.layers[static_cast<size_t>(l)].ffn;
    const Weights::Layer& W = w_->L[static_cast<size_t>(l)];
    std::vector<float> mixed(static_cast<size_t>(c.n_embd)), inject(static_cast<size_t>(c.hc_count)), probs(static_cast<size_t>(f.n_expert));
    hc_mix(W.hc_ffn, stack, mixed.data(), inject.data());
    mv(W.router, f.n_expert, c.n_embd, mixed.data(), probs.data());
    std::vector<int> idx(static_cast<size_t>(f.n_expert));
    for (int i = 0; i < f.n_expert; ++i) idx[static_cast<size_t>(i)] = i;
    std::partial_sort(idx.begin(), idx.begin() + f.n_used, idx.end(), [&](int a, int b) { return probs[static_cast<size_t>(a)] > probs[static_cast<size_t>(b)]; });
    idx.resize(static_cast<size_t>(f.n_used));
    return idx;
}

void Ref::note_token(int pos, int32_t token) { tok_hist_[static_cast<size_t>(pos)] = token; }

// The gated delta net mixer: out = ssm_out(gated per-head norm of the delta-rule scan over conv(qkv)).
void Ref::layer_linear(int l, const float* xn, float* attn_out) {
    const Config& c = cfg_;
    const Weights::Layer& W = w_->L[static_cast<size_t>(l)];
    const int e = c.n_embd;

    // qkv, then the depthwise conv over the last `conv_kernel` rows of it.
    { Phase ph("qkv matvec"); mv(W.qkv, c.qkv_width, e, xn, q_.data()); }
    if (trace_.on && l == 0 && cur_pos_ == 0) trace_.qkv0.assign(q_.begin(), q_.begin() + 8);
    {
        const int k = c.conv_kernel;
        std::vector<float>& hist = conv_state_[static_cast<size_t>(l)];
        const int hist_rows = k - 1;
        const std::vector<float>& wconv = W.conv;
        // The weight is {kernel, channels} with ne[0] contiguous, so tap j of channel ch is at
        // ch * kernel + j — not j * channels + ch. Getting this backwards produces a conv that looks
        // plausible and is wrong, which is what the oracle's node values caught.
        for (int ch = 0; ch < c.qkv_width; ++ch) {
            float acc = 0;
            for (int j = 0; j < hist_rows; ++j)
                acc += wconv[static_cast<size_t>(ch) * k + j] * hist[static_cast<size_t>(j) * c.qkv_width + ch];
            acc += wconv[static_cast<size_t>(ch) * k + hist_rows] * q_[static_cast<size_t>(ch)];
            attn_[static_cast<size_t>(ch)] = silu(acc);
        }
        // shift the history: drop the oldest row, append the new one
        if (hist_rows > 0) {
            std::memmove(hist.data(), hist.data() + static_cast<size_t>(c.qkv_width), static_cast<size_t>(hist_rows - 1) * c.qkv_width * sizeof(float));
            std::memcpy(hist.data() + static_cast<size_t>(hist_rows - 1) * c.qkv_width, q_.data(), static_cast<size_t>(c.qkv_width) * sizeof(float));
        }
    }

    // Split into q [n_k_head x gdn_head], k the same, v [n_v_head x gdn_head]; normalise q and k.
    if (trace_.on && l == 0 && cur_pos_ == 0) {
        trace_.conv0.assign(attn_.begin(), attn_.begin() + 3);
        trace_.conv0.insert(trace_.conv0.end(), attn_.begin() + 2045, attn_.begin() + 2048);  // q head 15
        trace_.conv0.insert(trace_.conv0.end(), attn_.begin() + 4096, attn_.begin() + 4099);  // v head 0
        trace_.conv0.insert(trace_.conv0.end(), attn_.end() - 3, attn_.end());
    }
    const int qn = c.n_k_head * c.gdn_head, vn = c.n_v_head * c.gdn_head;
    const float scale = float(1.0 / std::sqrt(double(c.gdn_head)));
    std::vector<float> qh(static_cast<size_t>(qn)), kh(static_cast<size_t>(qn));
    // L2 norm with eps inside the sum: x / sqrt(|x|^2 + eps), i.e. an RMS norm with eps/n, times 1/sqrt(n).
    head_norm(attn_.data(), qh.data(), c.n_k_head, c.gdn_head, float(c.rms_eps / c.gdn_head), scale);
    head_norm(attn_.data() + qn, kh.data(), c.n_k_head, c.gdn_head, float(c.rms_eps / c.gdn_head), scale);
    if (trace_.on && l == 0 && cur_pos_ == 0) {
        trace_.qhat0.assign(qh.begin(), qh.begin() + 3);
        trace_.qhat0.insert(trace_.qhat0.end(), qh.end() - 3, qh.end());
        trace_.khat0.assign(kh.begin(), kh.begin() + 3);
        trace_.khat0.insert(trace_.khat0.end(), kh.end() - 3, kh.end());
    }
    const float* vv = attn_.data() + 2 * qn;

    // Decay and beta: softplus(ssm_alpha @ x + dt.bias) * ssm_a, and sigmoid(ssm_beta @ x).
    std::vector<float> alpha(static_cast<size_t>(c.dt_rank)), beta(static_cast<size_t>(c.dt_rank));
    mv(W.alpha, c.dt_rank, e, xn, alpha.data());
    mv(W.beta, c.dt_rank, e, xn, beta.data());
    const std::vector<float>& dtb = W.dt_bias;
    const std::vector<float>& sa = W.ssm_a;
    std::vector<float> gate(static_cast<size_t>(c.n_v_head)), betav(static_cast<size_t>(c.n_v_head));
    for (int h = 0; h < c.n_v_head; ++h) {
        const int r = h % c.dt_rank;   // one decay and one beta per value head
        gate[static_cast<size_t>(h)] = softplus(alpha[static_cast<size_t>(r)] + dtb[static_cast<size_t>(r)]) * sa[static_cast<size_t>(r)];
        betav[static_cast<size_t>(h)] = sigmoid(beta[static_cast<size_t>(r)]);
    }

    // The delta-rule recurrence, per value head. State S is [gdn_head x gdn_head], k over the key dim.
    std::vector<float>& S = gdn_state_[static_cast<size_t>(l)];
    std::vector<float> sk(static_cast<size_t>(c.gdn_head)), d(static_cast<size_t>(c.gdn_head)), o(static_cast<size_t>(vn));
    { Phase ph("gdn scan");
    for (int h = 0; h < c.n_v_head; ++h) {
        // Value heads share the key heads by tiling (h % n_k_head), not by pairs (h / 2): measured per layer
        // against the oracle's own l_out nodes over every position (bench/w1-cpu-gate-2026-11-19.md).
        const int kh_i = h % c.n_k_head;
        const float* kvec = kh.data() + static_cast<size_t>(kh_i) * c.gdn_head;
        const float* qvec = qh.data() + static_cast<size_t>(kh_i) * c.gdn_head;
        const float* vvec = vv + static_cast<size_t>(h) * c.gdn_head;
        float* s = S.data() + static_cast<size_t>(h) * c.gdn_head * c.gdn_head;
        // ssm_a is stored NEGATIVE (measured: blk.0.ssm_a spans -72.3 to -0.0186), so `gate` is
        // already the negative decay and the state is multiplied by exp(gate) < 1. Taking exp(-gate)
        // turns the decay into growth and the residual stream reaches -inf within a few layers.
        const float a = std::exp(gate[static_cast<size_t>(h)]);
        const float b = betav[static_cast<size_t>(h)];
        for (int i = 0; i < c.gdn_head * c.gdn_head; ++i) s[i] *= a;
        for (int j = 0; j < c.gdn_head; ++j) {
            float acc = 0;
            for (int i = 0; i < c.gdn_head; ++i) acc += s[static_cast<size_t>(i) * c.gdn_head + j] * kvec[i];
            sk[static_cast<size_t>(j)] = acc;
        }
        for (int j = 0; j < c.gdn_head; ++j) d[static_cast<size_t>(j)] = (vvec[j] - sk[static_cast<size_t>(j)]) * b;
        for (int i = 0; i < c.gdn_head; ++i) {
            const float ki = kvec[i];
            for (int j = 0; j < c.gdn_head; ++j) s[static_cast<size_t>(i) * c.gdn_head + j] += ki * d[static_cast<size_t>(j)];
        }
        for (int j = 0; j < c.gdn_head; ++j) {
            float acc = 0;
            for (int i = 0; i < c.gdn_head; ++i) acc += s[static_cast<size_t>(i) * c.gdn_head + j] * qvec[i];
            o[static_cast<size_t>(h) * c.gdn_head + j] = acc * scale;   // the op's own 1/sqrt(S_v)
        }
    }
    }

    // Per-head norm with the ssm_norm weight, gated by silu(attn_gate @ x), then ssm_out.
    const std::vector<float>& sn = W.ssm_norm;
    std::vector<float> on_(static_cast<size_t>(vn));
    for (int h = 0; h < c.n_v_head; ++h) {
        const float* oh = o.data() + static_cast<size_t>(h) * c.gdn_head;
        float* out = on_.data() + static_cast<size_t>(h) * c.gdn_head;
        double ss = 0;
        for (int j = 0; j < c.gdn_head; ++j) ss += double(oh[j]) * double(oh[j]);
        const float inv = float(1.0 / std::sqrt(ss / c.gdn_head + c.rms_eps));
        for (int j = 0; j < c.gdn_head; ++j) out[j] = oh[j] * inv * sn[static_cast<size_t>(j)];
    }
    std::vector<float> z(static_cast<size_t>(vn));
    { Phase ph("attn_gate matvec"); mv(W.z, vn, e, xn, z.data()); }
    if (trace_.on && l == 0 && cur_pos_ == 0) {
        trace_.z0.assign(z.begin(), z.begin() + 3);
        trace_.z0.insert(trace_.z0.end(), z.begin() + 125, z.begin() + 128);  // row 0 tail
    }
    const bool sig_gate = c.layers[static_cast<size_t>(l)].gdn_sigmoid_gate;
    for (int i = 0; i < vn; ++i)
        attn_[static_cast<size_t>(i)] = on_[static_cast<size_t>(i)] * (sig_gate ? sigmoid(z[static_cast<size_t>(i)]) : silu(z[static_cast<size_t>(i)]));
    if (trace_.on && l == 0 && cur_pos_ == 0) {
        trace_.gated0.assign(attn_.begin(), attn_.begin() + 3);
        trace_.gated0.insert(trace_.gated0.end(), attn_.begin() + 125, attn_.begin() + 128);  // row 0 tail
    }

    if (trace_.on && l == 0 && cur_pos_ == 0) {
        trace_.gdn0.assign(on_.begin(), on_.begin() + 3);
        trace_.gdn0.insert(trace_.gdn0.end(), on_.begin() + 125, on_.begin() + 128);  // row 0 tail
    }
    { Phase ph("ssm_out matvec"); mv(W.ssm_out, e, vn, attn_.data(), attn_out); }
    if (trace_.on && l == 0 && cur_pos_ == 0) trace_.linattn0.assign(attn_out, attn_out + 8);
}

// The indexer of one sparse-attention layer for the token at `pos`: caches its raw key (f16), pools every block that
// this token completes (mean of the block's raw keys, RMS norm, rotary at the block's first position, f16), and, when
// more complete blocks are visible than the budget holds, fills `sel` with the positions it attends to (the best
// blocks by the sum over heads of relu(q.k) / sqrt(dim), then the incomplete tail, ascending) and returns true.
bool Ref::qsa_select(int l, int pos, const float* xn, std::vector<int>& sel) {
    const Config& c = cfg_;
    const AttnRecipe& a = c.layers[static_cast<size_t>(l)].attn;
    const Weights::Layer& W = w_->L[static_cast<size_t>(l)];
    const int e = c.n_embd, dim = a.qsa_dim, nh = a.qsa_heads, kp = a.qsa_kpool;
    const float eps = float(c.rms_eps);
    std::vector<uint16_t>& raw = idx_raw_[static_cast<size_t>(l)];
    std::vector<uint16_t>& pool = idx_pool_[static_cast<size_t>(l)];

    std::vector<float> k(static_cast<size_t>(dim));
    mv(W.idx_k, dim, e, xn, k.data());
    for (int d = 0; d < dim; ++d) raw[static_cast<size_t>(pos) * dim + d] = f32_to_f16(k[static_cast<size_t>(d)]);
    if ((pos + 1) % kp == 0) {
        const int j = (pos + 1) / kp - 1;
        std::vector<float> m(static_cast<size_t>(dim), 0.0f), mn(static_cast<size_t>(dim));
        for (int i = 0; i < kp; ++i)
            for (int d = 0; d < dim; ++d) m[static_cast<size_t>(d)] += quants::f16_to_f32(raw[static_cast<size_t>(j * kp + i) * dim + d]);
        for (int d = 0; d < dim; ++d) m[static_cast<size_t>(d)] *= 1.0f / float(kp);
        rmsnorm(m.data(), W.idx_kn.data(), mn.data(), dim, eps);
        rope(mn.data(), 1, dim, a.rope_dim, j * kp, a.rope_base, nullptr);
        for (int d = 0; d < dim; ++d) pool[static_cast<size_t>(j) * dim + d] = f32_to_f16(mn[static_cast<size_t>(d)]);
    }

    const int nv = (pos + 1) / kp;                       // complete blocks, this token's own included when it ends one
    const int n_top = std::max(1, a.qsa_top_k / kp);
    if (nv <= n_top) return false;

    std::vector<float> q(static_cast<size_t>(nh) * dim);
    mv(W.idx_q, nh * dim, e, xn, q.data());
    for (int h = 0; h < nh; ++h) rmsnorm(q.data() + static_cast<size_t>(h) * dim, W.idx_qn.data(), q.data() + static_cast<size_t>(h) * dim, dim, eps);
    rope(q.data(), nh, dim, a.rope_dim, pos, a.rope_base, nullptr);
    const float w = 1.0f / std::sqrt(float(dim));
    std::vector<float> score(static_cast<size_t>(nv));
    std::vector<float> kf(static_cast<size_t>(dim));
    for (int j = 0; j < nv; ++j) {
        for (int d = 0; d < dim; ++d) kf[static_cast<size_t>(d)] = quants::f16_to_f32(pool[static_cast<size_t>(j) * dim + d]);
        float sc = 0;
        for (int h = 0; h < nh; ++h) {
            float qk = 0;
            for (int d = 0; d < dim; ++d) qk += q[static_cast<size_t>(h) * dim + d] * kf[static_cast<size_t>(d)];
            sc += std::max(qk, 0.0f) * w;
        }
        score[static_cast<size_t>(j)] = sc;
    }
    std::vector<int> order(static_cast<size_t>(nv));
    for (int j = 0; j < nv; ++j) order[static_cast<size_t>(j)] = j;
    std::partial_sort(order.begin(), order.begin() + n_top, order.end(), [&](int x, int y) {
        return score[static_cast<size_t>(x)] > score[static_cast<size_t>(y)] || (score[static_cast<size_t>(x)] == score[static_cast<size_t>(y)] && x < y);
    });
    std::sort(order.begin(), order.begin() + n_top);
    sel.clear();
    for (int i = 0; i < n_top; ++i)
        for (int m = 0; m < kp; ++m) sel.push_back(order[static_cast<size_t>(i)] * kp + m);
    for (int t = nv * kp; t <= pos; ++t) sel.push_back(t);
    return true;
}

// Attention over the f16 K/V cache: grouped-query, optionally sliding-window, optionally gated, optionally
// reading another layer's cache. out = wo(attention).
void Ref::layer_attention(int l, int pos, const float* xn, float* out) {
    const Config& c = cfg_;
    const Weights::Layer& W = w_->L[static_cast<size_t>(l)];
    const AttnRecipe& a = c.layers[static_cast<size_t>(l)].attn;
    const int e = c.n_embd;
    const int hd = a.head_dim, nh = a.n_head, nkv = a.n_head_kv;
    const float eps = float(c.rms_eps);
    const float* factors = a.rope_factors ? w_->rope_freqs.data() : nullptr;

    // q is [q_0 | gate_0 | q_1 | gate_1 | ...] (head_dim each) when the layer gates its output, else [q_0 | q_1 | ...].
    const int q_stride = a.q_gate ? 2 * hd : hd;
    std::vector<float> qfull(static_cast<size_t>(nh) * q_stride);
    mv(W.wq, nh * q_stride, e, xn, qfull.data());
    std::vector<float> q(static_cast<size_t>(nh) * hd);
    for (int h = 0; h < nh; ++h) {
        const float* src = qfull.data() + static_cast<size_t>(h) * q_stride;
        float* dst = q.data() + static_cast<size_t>(h) * hd;
        double ss = 0;
        for (int d = 0; d < hd; ++d) ss += double(src[d]) * double(src[d]);
        const float inv = float(1.0 / std::sqrt(ss / hd + c.rms_eps));
        for (int d = 0; d < hd; ++d) dst[d] = src[d] * inv * W.q_norm[static_cast<size_t>(d)];
    }
    rope(q.data(), nh, hd, a.rope_dim, pos, a.rope_base, factors);
    if (oracle_arith_) round_f16(q.data(), q.data(), nh * hd);   // the oracle's K.Q product takes q as f16

    const int kvw = nkv * hd;
    const int src_layer = a.own_kv ? l : a.kv_src;
    if (a.own_kv) {
        std::vector<float> k(static_cast<size_t>(kvw)), v(static_cast<size_t>(kvw));
        mv(W.wk, kvw, e, xn, k.data());
        if (a.v_from_k) v = k;   // V is the raw K projection, before K's norm and rope
        else mv(W.wv, kvw, e, xn, v.data());
        for (int h = 0; h < nkv; ++h) {
            float* kk = k.data() + static_cast<size_t>(h) * hd;
            double ss = 0;
            for (int d = 0; d < hd; ++d) ss += double(kk[d]) * double(kk[d]);
            const float inv = float(1.0 / std::sqrt(ss / hd + c.rms_eps));
            for (int d = 0; d < hd; ++d) kk[d] *= inv * W.k_norm[static_cast<size_t>(d)];
        }
        if (a.v_norm) head_norm(v.data(), v.data(), nkv, hd, eps, 1.0f);
        rope(k.data(), nkv, hd, a.rope_dim, pos, a.rope_base, factors);

        // The cache holds f16, as the oracle's does: a numeric difference there is a real difference.
        std::vector<uint16_t>& kc = k_cache_[static_cast<size_t>(l)];
        std::vector<uint16_t>& vc = v_cache_[static_cast<size_t>(l)];
        for (int i = 0; i < kvw; ++i) {
            kc[static_cast<size_t>(pos) * kvw + static_cast<size_t>(i)] = f32_to_f16(k[static_cast<size_t>(i)]);
            vc[static_cast<size_t>(pos) * kvw + static_cast<size_t>(i)] = f32_to_f16(v[static_cast<size_t>(i)]);
        }
    }
    const std::vector<uint16_t>& kc = k_cache_[static_cast<size_t>(src_layer)];
    const std::vector<uint16_t>& vc = v_cache_[static_cast<size_t>(src_layer)];

    // Sparse attention (qwen4exp): the indexer pools its keys in blocks of qsa_kpool positions, scores every complete
    // block for this query, and the query attends to the best qsa_top_k / kpool blocks plus the incomplete tail. While
    // the complete blocks fit the budget every position is visible and this is the plain causal attention.
    std::vector<int> sel;
    bool use_sel = false;
    if (a.qsa_kpool > 0) use_sel = qsa_select(l, pos, xn, sel);
    const int t0 = a.sliding ? std::max(0, pos - a.window + 1) : 0;
    const int n_vis = use_sel ? int(sel.size()) : pos - t0 + 1;
    const int n_grp = nkv > 0 ? nh / nkv : 1;
    auto tpos = [&](int i) { return use_sel ? sel[static_cast<size_t>(i)] : t0 + i; };
    std::vector<float> attn(static_cast<size_t>(nh) * hd, 0.0f);
    std::vector<float> scores(static_cast<size_t>(n_vis));
    for (int h = 0; h < nh; ++h) {
        const int kh = h / n_grp;
        const float* qh = q.data() + static_cast<size_t>(h) * hd;
        for (int i = 0; i < n_vis; ++i) {
            const uint16_t* ksrc = kc.data() + static_cast<size_t>(tpos(i)) * kvw + static_cast<size_t>(kh) * hd;
            float acc = 0;
            for (int d = 0; d < hd; ++d) acc += qh[d] * quants::f16_to_f32(ksrc[d]);
            scores[static_cast<size_t>(i)] = acc * a.scale;
        }
        softmax_inplace(scores.data(), n_vis);
        if (oracle_arith_) round_f16(scores.data(), scores.data(), n_vis);   // and the V product takes the weights as f16
        float* oh = attn.data() + static_cast<size_t>(h) * hd;
        for (int i = 0; i < n_vis; ++i) {
            const float w = scores[static_cast<size_t>(i)];
            const uint16_t* vsrc = vc.data() + static_cast<size_t>(tpos(i)) * kvw + static_cast<size_t>(kh) * hd;
            for (int d = 0; d < hd; ++d) oh[d] += w * quants::f16_to_f32(vsrc[d]);
        }
    }
    // The gate half of q multiplies the attention output.
    if (a.q_gate)
        for (int h = 0; h < nh; ++h)
            for (int d = 0; d < hd; ++d) {
                const float g = sigmoid(qfull[static_cast<size_t>(h) * q_stride + hd + d]);
                attn[static_cast<size_t>(h) * hd + d] *= g;
            }
    mv(W.wo, e, nh * hd, attn.data(), out);
}

// The FFN half of a layer, in place: x += ffn(x), then the per-layer-embedding block and the layer scalar.
void Ref::layer_ffn(int l, float* x) {
    const Config& c = cfg_;
    const LayerRecipe& r = c.layers[static_cast<size_t>(l)];
    const Weights::Layer& W = w_->L[static_cast<size_t>(l)];
    const int e = c.n_embd;
    const float eps = float(c.rms_eps);
    std::vector<float> pre(static_cast<size_t>(e)), out(static_cast<size_t>(e), 0.0f);
    rmsnorm(x, W.n_ffn_in.data(), pre.data(), e, eps);
    const bool rec = trace_.on && trace_.node_layer == l;
    if (rec) trace_.nodes["ffn_norm"] = pre;
    ffn_block(l, x, pre.data(), out.data());
    if (r.post_ffn_norm) rmsnorm(out.data(), W.n_ffn_out.data(), out.data(), e, eps);
    if (rec) trace_.nodes["ffn_post_norm"] = out;
    for (int j = 0; j < e; ++j) x[j] += out[static_cast<size_t>(j)];

    if (r.ple) ple_block(l, x);
    if (r.out_scale) for (int j = 0; j < e; ++j) x[j] *= W.out_scale;
}

// The FFN proper: out = ffn(pre), where `pre` is the already-normed (or hyper-mixed) input and `x` the residual
// stream the router of a DenseAndMoe layer reads.
void Ref::ffn_block(int l, const float* x, const float* pre, float* out) {
    const Config& c = cfg_;
    const LayerRecipe& r = c.layers[static_cast<size_t>(l)];
    const Weights::Layer& W = w_->L[static_cast<size_t>(l)];
    const int e = c.n_embd;
    const float eps = float(c.rms_eps);
    const bool rec = trace_.on && trace_.node_layer == l;
    auto note = [&](const char* name, const std::vector<float>& v) { if (rec) trace_.nodes[name] = v; };
    std::fill(out, out + e, 0.0f);

    switch (r.ffn.kind) {
        case FfnKind::Dense:
            ffn_dense(l, pre, out);
            break;
        case FfnKind::MoeShared: {
            // Routed experts, then the shared expert behind a scalar sigmoid gate.
            moe(l, pre, pre, out);
            const int ns = r.ffn.n_ff_shexp;
            std::vector<float> gs(static_cast<size_t>(ns)), us(static_cast<size_t>(ns)), hs(static_cast<size_t>(ns));
            mv(W.sh_gate, ns, e, pre, gs.data());
            mv(W.sh_up, ns, e, pre, us.data());
            for (int j = 0; j < ns; ++j) hs[static_cast<size_t>(j)] = silu(gs[static_cast<size_t>(j)]) * us[static_cast<size_t>(j)];
            std::vector<float> shared(static_cast<size_t>(e));
            mv(W.sh_down, e, ns, hs.data(), shared.data());
            float dot = 0;
            for (int j = 0; j < e; ++j) dot += W.sh_gate_inp[static_cast<size_t>(j)] * pre[j];
            const float sg = sigmoid(dot);
            for (int j = 0; j < e; ++j) out[j] += shared[static_cast<size_t>(j)] * sg;
            break;
        }
        case FfnKind::DenseAndMoe: {
            // A dense MLP and the routed experts side by side, each on its own input norm and with its own
            // output norm; the router reads the residual stream itself, rms-normalised, scaled by
            // 1/sqrt(n_embd) and by a learned vector.
            std::vector<float> mlp(static_cast<size_t>(e)), routed(static_cast<size_t>(e), 0.0f);
            ffn_dense(l, pre, mlp.data());
            rmsnorm(mlp.data(), W.n_mlp_out.data(), mlp.data(), e, eps);
            note("ffn_mlp", mlp);
            std::vector<float> in2(static_cast<size_t>(e)), route(static_cast<size_t>(e));
            rmsnorm(x, W.n_moe_in.data(), in2.data(), e, eps);
            rmsnorm(x, nullptr, route.data(), e, eps);
            const float s = 1.0f / std::sqrt(float(e));
            for (int j = 0; j < e; ++j) route[static_cast<size_t>(j)] = (route[static_cast<size_t>(j)] * s) * W.router_scale[static_cast<size_t>(j)];
            moe(l, route.data(), in2.data(), routed.data());
            note("ffn_moe_out", routed);
            rmsnorm(routed.data(), W.n_moe_out.data(), routed.data(), e, eps);
            note("ffn_moe", routed);
            for (int j = 0; j < e; ++j) out[j] = mlp[static_cast<size_t>(j)] + routed[static_cast<size_t>(j)];
            note("ffn_moe_combined", std::vector<float>(out, out + e));
            break;
        }
    }
}

// down(act(gate(x)) * up(x)) with the layer's own dense weights.
void Ref::ffn_dense(int l, const float* pre, float* out) {
    const Config& c = cfg_;
    const FfnRecipe& f = c.layers[static_cast<size_t>(l)].ffn;
    const Weights::Layer& W = w_->L[static_cast<size_t>(l)];
    const int e = c.n_embd, nf = f.n_ff;
    std::vector<float> g(static_cast<size_t>(nf)), u(static_cast<size_t>(nf)), h(static_cast<size_t>(nf));
    mv(W.ffn_gate, nf, e, pre, g.data());
    mv(W.ffn_up, nf, e, pre, u.data());
    for (int j = 0; j < nf; ++j) h[static_cast<size_t>(j)] = act_fn(f.act, g[static_cast<size_t>(j)]) * u[static_cast<size_t>(j)];
    mv(W.ffn_down, e, nf, h.data(), out);
}

// The per-layer-embedding block: x += norm(proj(act(gate(x)) * input_l)).
void Ref::ple_block(int l, float* x) {
    const Config& c = cfg_;
    const Weights::Layer& W = w_->L[static_cast<size_t>(l)];
    const int e = c.n_embd, np = c.n_embd_per_layer;
    std::vector<float> g(static_cast<size_t>(np)), p(static_cast<size_t>(e));
    mv(W.ple_gate, np, e, x, g.data());
    const float* in = ple_in_.data() + static_cast<size_t>(l) * np;
    for (int j = 0; j < np; ++j) g[static_cast<size_t>(j)] = gelu(g[static_cast<size_t>(j)]) * in[j];
    mv(W.ple_proj, e, np, g.data(), p.data());
    rmsnorm(p.data(), W.ple_norm.data(), p.data(), e, float(c.rms_eps));
    for (int j = 0; j < e; ++j) x[j] += p[static_cast<size_t>(j)];
}

// The routed experts: out = sum over the top-n_used experts (softmax gate, weights renormalised to sum 1) of
// weight * down_e(act(gate_e(post)) * up_e(post)). `route_in` feeds the router, `post` the experts.
void Ref::moe(int l, const float* route_in, const float* post, float* out) {
    Phase ph_moe("moe (whole)");
    const Config& c = cfg_;
    const FfnRecipe& f = c.layers[static_cast<size_t>(l)].ffn;
    const Weights::Layer& W = w_->L[static_cast<size_t>(l)];
    const int e = c.n_embd, nf = f.n_ff_exp;

    // Router.
    std::vector<float> probs(static_cast<size_t>(f.n_expert));
    mv(W.router, f.n_expert, e, route_in, probs.data());
    softmax_inplace(probs.data(), f.n_expert);
    if (trace_.on && l == 0 && cur_pos_ == 0) trace_.moe_logits0.assign(probs.begin(), probs.begin() + 8);

    std::vector<int> idx(static_cast<size_t>(f.n_expert));
    for (int i = 0; i < f.n_expert; ++i) idx[static_cast<size_t>(i)] = i;
    const int n_sorted = std::min(f.n_expert, f.n_used + 1);   // one past the cut, to see how close the 9th was
    std::partial_sort(idx.begin(), idx.begin() + n_sorted, idx.end(),
                      [&](int a, int b) { return probs[static_cast<size_t>(a)] > probs[static_cast<size_t>(b)]; });
    float wsum = 0;
    for (int i = 0; i < f.n_used; ++i) wsum += probs[static_cast<size_t>(idx[static_cast<size_t>(i)])];
    wsum = std::max(wsum, 6.103515625e-05f);  // the graph's CLAMP
    std::vector<float> acc(static_cast<size_t>(e), 0.0f);
    if (trace_.on) {
        trace_.moe_last_ids.assign(idx.begin(), idx.begin() + f.n_used);
        trace_.moe_last_p.clear();
        for (int i = 0; i < n_sorted; ++i) trace_.moe_last_p.push_back(probs[static_cast<size_t>(idx[static_cast<size_t>(i)])]);
    }
    if (trace_.on && trace_.all_pos) {
        std::vector<float>& mt = trace_.moe_pos[static_cast<size_t>(cur_pos_)][static_cast<size_t>(l)];
        for (int i = 0; i < f.n_used; ++i) mt.push_back(float(idx[static_cast<size_t>(i)]));
        for (int i = 0; i < f.n_used; ++i) mt.push_back(probs[static_cast<size_t>(idx[static_cast<size_t>(i)])] / wsum);
    }

    const gguf::Tensor* wg = f.fused_gate_up ? W.exp_gate_up : W.exp_gate;
    const gguf::Tensor* wu = f.fused_gate_up ? W.exp_gate_up : W.exp_up;
    const gguf::Tensor* wd = W.exp_down;
    const uint64_t gate_row = type_bytes(wg->type, uint64_t(e));
    // a fused tensor holds 2 * nf rows per expert, gate rows first and up rows after them
    const uint64_t gate_expert = uint64_t(f.fused_gate_up ? 2 * nf : nf) * gate_row;
    const uint64_t up_off = f.fused_gate_up ? uint64_t(nf) * gate_row : 0;
    const uint64_t down_row = type_bytes(wd->type, uint64_t(nf));
    const uint64_t down_expert = uint64_t(e) * down_row;
    std::vector<float> hq(static_cast<size_t>(nf));
    std::vector<float> g(static_cast<size_t>(nf)), u(static_cast<size_t>(nf)), hh(static_cast<size_t>(nf)), dn(static_cast<size_t>(e));
    // The oracle rounds `post` once per weight type and the swiglu output once per expert.
    std::vector<float> post_g(static_cast<size_t>(e)), post_u(static_cast<size_t>(e));
    const float* pg = post;
    const float* pu = post;
    if (oracle_arith_) {
        if (round_act(wg->type, post, post_g.data(), e)) pg = post_g.data();
        if (round_act(wu->type, post, post_u.data(), e)) pu = post_u.data();
    }
    Phase ph_experts("  moe routed experts");
    for (int i = 0; i < f.n_used; ++i) {
        const int ex = idx[static_cast<size_t>(i)];
        const float w = probs[static_cast<size_t>(ex)] / wsum;
        // The expert's rows go across the pool as well: one expert is 3.1M elements, and a serial
        // decode of IQ2_S runs at 0.29 Gelem/s, which is 109 ms a layer if it is left on one core.
        const uint8_t* gbase = wg->data + uint64_t(ex) * gate_expert;
        const uint8_t* ubase = wu->data + uint64_t(ex) * gate_expert + up_off;
        const uint8_t* dbase = wd->data + uint64_t(ex) * down_expert;
        parallel_for(nf, threads_, [&](int b, int en) {
            for (int j = b; j < en; ++j) {
                g[static_cast<size_t>(j)] = dot_row(wg->type, gbase + uint64_t(j) * gate_row, pg, e);
                u[static_cast<size_t>(j)] = dot_row(wu->type, ubase + uint64_t(j) * gate_row, pu, e);
            }
        });
        for (int j = 0; j < nf; ++j) hh[static_cast<size_t>(j)] = act_fn(f.act, g[static_cast<size_t>(j)]) * u[static_cast<size_t>(j)];
        const float* hd_in = hh.data();
        if (oracle_arith_ && round_act(wd->type, hh.data(), hq.data(), nf)) hd_in = hq.data();
        parallel_for(e, threads_, [&](int b, int en) {
            for (int j = b; j < en; ++j)
                dn[static_cast<size_t>(j)] = dot_row(wd->type, dbase + uint64_t(j) * down_row, hd_in, nf);
        });
        if (f.expert_scale) {
            const float sc = W.down_scale[static_cast<size_t>(ex)];
            for (int j = 0; j < e; ++j) acc[static_cast<size_t>(j)] += (dn[static_cast<size_t>(j)] * sc) * w;
        } else {
            for (int j = 0; j < e; ++j) acc[static_cast<size_t>(j)] += dn[static_cast<size_t>(j)] * w;
        }
    }
    for (int j = 0; j < e; ++j) out[j] = acc[static_cast<size_t>(j)];
}

// ---- hyper-connections ----------------------------------------------------------------------------
// The residual is hc_count parallel streams. A mixer reads them through a gated low-rank module (a grouped RMS
// norm with a per-stream weight, a down projection scaled by 1/hc, silu, an up projection, a sigmoid gate, the mean
// of the gated streams) and writes its output back to every stream with its own weight, 2 * sigmoid(inject).
void Ref::hc_mix(const Hc& m, const float* st, float* mixed, float* inject) {
    const Config& c = cfg_;
    const int e = c.n_embd, hc = c.hc_count, hcd = hc * e;
    const float inv_hc = 1.0f / float(hc);
    std::vector<float> xn(static_cast<size_t>(hcd)), lo(static_cast<size_t>(c.hc_rank)), gate(static_cast<size_t>(hcd));
    for (int s = 0; s < hc; ++s)
        rmsnorm(st + static_cast<size_t>(s) * e, m.norm.data() + static_cast<size_t>(s) * e, xn.data() + static_cast<size_t>(s) * e, e,
                float(c.rms_eps));
    mv(m.down, c.hc_rank, hcd, xn.data(), lo.data());
    for (int i = 0; i < c.hc_rank; ++i) lo[static_cast<size_t>(i)] = silu(lo[static_cast<size_t>(i)] * inv_hc);
    mv(m.up, hcd, c.hc_rank, lo.data(), gate.data());
    for (int i = 0; i < e; ++i) {
        float acc = xn[static_cast<size_t>(i)] * sigmoid(gate[static_cast<size_t>(i)]);
        for (int s = 1; s < hc; ++s)
            acc += xn[static_cast<size_t>(s) * e + i] * sigmoid(gate[static_cast<size_t>(s) * e + i]);
        mixed[i] = acc * inv_hc;
    }
    if (m.inject.w) {
        mv(m.inject, hc, hcd, xn.data(), inject);
        for (int s = 0; s < hc; ++s) inject[s] *= inv_hc;
    }
}

void Ref::hc_combine(float* st, const float* block_out, const float* inject) {
    const int e = cfg_.n_embd, hc = cfg_.hc_count;
    for (int s = 0; s < hc; ++s) {
        const float w = 2.0f * sigmoid(inject[s]);
        float* row = st + static_cast<size_t>(s) * e;
        for (int i = 0; i < e; ++i) row[i] = row[i] + block_out[i] * w;
    }
}

void Ref::hc_note(int l, int pos, const char* name, const float* v, size_t n) {
    // node_layer == -2 records every layer's nodes under "<name>-<layer>@<pos>"; == l records that layer's, by name
    if (!trace_.on || !(trace_.node_layer == l || trace_.node_layer == -2)) return;
    trace_.nodes[trace_.node_layer == -2 ? std::string(name) + "-" + std::to_string(l) + "@" + std::to_string(pos) : std::string(name)] =
        std::vector<float>(v, v + n);
}

void Ref::layer_hc_attn(int l, int pos, float* st) {
    const Config& c = cfg_;
    const LayerRecipe& r = c.layers[static_cast<size_t>(l)];
    const Weights::Layer& W = w_->L[static_cast<size_t>(l)];
    const int e = c.n_embd;
    if (r.ple_hash) ple_hash_block(l, pos, st);
    std::vector<float> mixed(static_cast<size_t>(e)), inject(static_cast<size_t>(c.hc_count)), out(static_cast<size_t>(e));
    hc_mix(W.hc_attn, st, mixed.data(), inject.data());
    hc_note(l, pos, "hc_mixed_attn", mixed.data(), mixed.size());
    hc_note(l, pos, "hc_inject_attn", inject.data(), inject.size());
    if (r.mixer == Mixer::Gdn) layer_linear(l, mixed.data(), out.data());
    else layer_attention(l, pos, mixed.data(), out.data());
    hc_note(l, pos, "mixer_out", out.data(), out.size());
    hc_combine(st, out.data(), inject.data());
    hc_note(l, pos, "hc_combine_attn", st, static_cast<size_t>(c.hc_count) * e);
    if (trace_.on && !trace_.attn_out.empty()) trace_.attn_out[static_cast<size_t>(l)] = std::vector<float>(st, st + static_cast<size_t>(c.hc_count) * e);
    if (trace_.on && trace_.all_pos) trace_.attn_out_pos[static_cast<size_t>(pos)][static_cast<size_t>(l)] = std::vector<float>(st, st + static_cast<size_t>(c.hc_count) * e);
}

void Ref::layer_hc_ffn(int l, float* st) {
    const Config& c = cfg_;
    const Weights::Layer& W = w_->L[static_cast<size_t>(l)];
    const int e = c.n_embd;
    std::vector<float> mixed(static_cast<size_t>(e)), inject(static_cast<size_t>(c.hc_count)), out(static_cast<size_t>(e));
    hc_mix(W.hc_ffn, st, mixed.data(), inject.data());
    hc_note(l, cur_pos_, "hc_mixed_ffn", mixed.data(), mixed.size());
    ffn_block(l, nullptr, mixed.data(), out.data());
    hc_note(l, cur_pos_, "ffn_out", out.data(), out.size());
    hc_combine(st, out.data(), inject.data());
    hc_note(l, cur_pos_, "hc_combine_ffn", st, static_cast<size_t>(c.hc_count) * e);
}

void Ref::layer_hc(int l, int pos, float* st) {
    cur_pos_ = pos;
    layer_hc_attn(l, pos, st);
    if (attn_only_) return;
    layer_hc_ffn(l, st);
}

const float* Ref::forward_hc(int32_t token, int pos, bool want_logits) {
    const Config& c = cfg_;
    cur_pos_ = pos;
    const int e = c.n_embd;
    tok_hist_[static_cast<size_t>(pos)] = token;
    std::vector<float> emb(static_cast<size_t>(e));
    embed_token(token, emb.data());
    for (int s = 0; s < c.hc_count; ++s) std::copy(emb.begin(), emb.end(), stack_.begin() + static_cast<size_t>(s) * e);
    if (trace_.on && pos == 0) {
        trace_.x0.assign(emb.begin(), emb.begin() + 8);
        trace_.attn_out.assign(static_cast<size_t>(c.n_layer_main), {});
        trace_.l_out.assign(static_cast<size_t>(c.n_layer_main), {});
    }
    if (trace_.on && trace_.all_pos) {
        if (trace_.l_out_pos.size() <= static_cast<size_t>(pos)) {
            trace_.l_out_pos.resize(static_cast<size_t>(pos) + 1);
            trace_.attn_out_pos.resize(static_cast<size_t>(pos) + 1);
            trace_.moe_pos.resize(static_cast<size_t>(pos) + 1);
        }
        trace_.moe_pos[static_cast<size_t>(pos)].assign(static_cast<size_t>(c.n_layer_main), {});
        trace_.l_out_pos[static_cast<size_t>(pos)].assign(static_cast<size_t>(c.n_layer_main), {});
        trace_.attn_out_pos[static_cast<size_t>(pos)].assign(static_cast<size_t>(c.n_layer_main), {});
    }
    const int n_layers = (layer_limit_ > 0 && layer_limit_ < c.n_layer_main) ? layer_limit_ : c.n_layer_main;
    for (int l = 0; l < n_layers; ++l) {
        layer_hc(l, pos, stack_.data());
        if (trace_.on && pos == 0) trace_.l_out[static_cast<size_t>(l)] = stack_;
        if (trace_.on && trace_.all_pos) trace_.l_out_pos[static_cast<size_t>(pos)][static_cast<size_t>(l)] = stack_;
    }
    if (!want_logits) return nullptr;
    return head(stack_.data());
}

// The n-gram hash embedding block: the token and its predecessors are hashed into one row per head of a very large
// table; the rows, read through a key and a value projection, gate a value into every stream (a signed square root
// of a per-stream key.query dot product, then a sigmoid), and a dilated depthwise causal conv over the gated values
// joins them. h' = h + gated + silu(conv(norm(gated))).
void Ref::ple_hash_block(int l, int pos, float* st) {
    const Config& c = cfg_;
    const Weights& Wg = *w_;
    const Weights::Layer& W = Wg.L[static_cast<size_t>(l)];
    const int e = c.n_embd, hc = c.hc_count, hcd = hc * e, np = c.n_embd_per_layer;
    const int n_heads = (c.ple_ngram - 1) * c.ple_heads_per_ngram;
    const float eps = float(c.rms_eps);

    // The window: the token, then its predecessors; a missing or EOS predecessor cuts the window, and everything
    // behind the cut reads as EOS (the token's own EOS does not cut its context).
    std::vector<int64_t> ctx(static_cast<size_t>(c.ple_ngram));
    ctx[0] = tok_hist_[static_cast<size_t>(pos)];
    bool cut = false;
    for (int s = 1; s < c.ple_ngram; ++s) {
        const int64_t t = pos - s >= 0 ? int64_t(tok_hist_[static_cast<size_t>(pos - s)]) : -1;
        cut = cut || t < 0 || t == c.ple_eos;
        ctx[static_cast<size_t>(s)] = cut ? c.ple_eos : t;
    }
    // One row per head: n-gram n takes the first n window entries, mixes them with the per-position multipliers
    // (unsigned 64-bit, wrapping) and reduces the result into the head's own range of the table.
    std::vector<float> emb(static_cast<size_t>(n_heads) * np);
    const gguf::Tensor* table = Wg.ple_embd;
    const uint64_t row_bytes = type_bytes(table->type, uint64_t(np));
    for (int n = 2; n <= c.ple_ngram; ++n) {
        uint64_t mixed = uint64_t(ctx[0]) * c.ple_multipliers[0];
        for (int j = 1; j < n; ++j) mixed ^= uint64_t(ctx[static_cast<size_t>(j)]) * c.ple_multipliers[static_cast<size_t>(j)];
        for (int g = 0; g < c.ple_heads_per_ngram; ++g) {
            const int h = (n - 2) * c.ple_heads_per_ngram + g;
            const uint64_t row = mixed % c.ple_head_vocab[static_cast<size_t>(h)] + c.ple_head_offsets[static_cast<size_t>(h)];
            quants::dequant_row(table->type, table->data + row * row_bytes, emb.data() + static_cast<size_t>(h) * np, uint64_t(np));
        }
    }

    std::vector<float> key(static_cast<size_t>(hcd)), value(static_cast<size_t>(e));
    if (trace_.on && (trace_.node_layer == l || trace_.node_layer == -2))
        trace_.nodes[trace_.node_layer == -2 ? "ple_embd-" + std::to_string(l) + "@" + std::to_string(pos) : std::string("ple_embd")] = emb;
    mv(W.ple_key, hcd, n_heads * np, emb.data(), key.data());
    mv(W.ple_value, e, n_heads * np, emb.data(), value.data());
    std::vector<float> kn(static_cast<size_t>(hcd)), qn(static_cast<size_t>(hcd));
    for (int s = 0; s < hc; ++s) {
        rmsnorm(key.data() + static_cast<size_t>(s) * e, W.ple_norm_key.data() + static_cast<size_t>(s) * e, kn.data() + static_cast<size_t>(s) * e, e, eps);
        rmsnorm(st + static_cast<size_t>(s) * e, W.ple_norm_query.data() + static_cast<size_t>(s) * e, qn.data() + static_cast<size_t>(s) * e, e, eps);
    }
    std::vector<float> gated(static_cast<size_t>(hcd));
    const float inv_sqrt_e = 1.0f / std::sqrt(float(e));
    for (int s = 0; s < hc; ++s) {
        double dot = 0;
        for (int i = 0; i < e; ++i) dot += double(kn[static_cast<size_t>(s) * e + i] * qn[static_cast<size_t>(s) * e + i]);
        const float sc = float(dot) * inv_sqrt_e;
        const float mag = std::sqrt(std::min(std::max(std::fabs(sc), 1e-6f), INFINITY));
        const float sgn = sc > 0.0f ? 1.0f : sc < 0.0f ? -1.0f : 0.0f;
        const float gate = sigmoid(sgn * mag);
        for (int i = 0; i < e; ++i) gated[static_cast<size_t>(s) * e + i] = value[static_cast<size_t>(i)] * gate;
    }
    std::vector<float> normalized(static_cast<size_t>(hcd));
    for (int s = 0; s < hc; ++s)
        rmsnorm(gated.data() + static_cast<size_t>(s) * e, W.ple_norm_conv.data() + static_cast<size_t>(s) * e, normalized.data() + static_cast<size_t>(s) * e, e, eps);

    // The dilated causal conv: tap k reads (kernel-1-k) * dilation positions back; the history holds the last
    // (kernel-1) * dilation inputs, oldest first, zeros before the start of the sequence.
    const int kern = c.ple_conv_kernel, dil = c.ple_ngram, hist_rows = (kern - 1) * dil;
    std::vector<float> conv(static_cast<size_t>(hcd));
    for (int ch = 0; ch < hcd; ++ch) {
        float acc = 0;
        for (int k = 0; k < kern; ++k) {
            const int back = (kern - 1 - k) * dil;
            const float xv = back == 0 ? normalized[static_cast<size_t>(ch)]
                                       : ple_hist_[static_cast<size_t>(hist_rows - back) * hcd + ch];
            const float term = W.ple_conv[static_cast<size_t>(ch) * kern + k] * xv;
            acc = k == 0 ? term : acc + term;
        }
        conv[static_cast<size_t>(ch)] = silu(acc);
    }
    if (hist_rows > 0) {
        std::memmove(ple_hist_.data(), ple_hist_.data() + hcd, static_cast<size_t>(hist_rows - 1) * hcd * sizeof(float));
        std::memcpy(ple_hist_.data() + static_cast<size_t>(hist_rows - 1) * hcd, normalized.data(), static_cast<size_t>(hcd) * sizeof(float));
    }
    if (trace_.on && (trace_.node_layer == l || trace_.node_layer == -2)) {
        const std::string suffix = trace_.node_layer == -2 ? "-" + std::to_string(l) + "@" + std::to_string(pos) : std::string();
        trace_.nodes["ple_gated_value" + suffix] = gated;
        trace_.nodes["ple_conv_out" + suffix] = conv;
    }
    for (int i = 0; i < hcd; ++i) st[i] = st[i] + (gated[static_cast<size_t>(i)] + conv[static_cast<size_t>(i)]);
}

// ---- inkling2_loop (docs/inkling2.md) ---------------------------------------------------------------
// The graph is the reference's (llama.cpp src/models/inkling2.cpp + inkling2-loop.cpp, read as facts
// about the architecture), written in Limbo's own step idiom. One token per call, all
// n_layer + (loop_count-1)*(loop_end-loop_start) visits. Notable numerics, all verified against the
// oracle's own numbers (bench, 2026-11-23):
//  - the attention logits are scale*q.k + rel_bias[d].r - sink, with scale = 1/sqrt(head_dim): the
//    relative bias and the sink enter AFTER the q.k scale (the reference's softmax carries a zero
//    sink column in the denominator; the Python reference adds position_bias after the scaling too),
//    and the denominator carries the sink column at logit 0;
//  - V gets a per-head RMS norm with no weight; q and k get no norm at all, only the L2 clips;
//  - the short convolutions apply no activation: the ssm_conv op is a plain depthwise sum here.
namespace {

// An L2 clip, per head over `row` elements, after the optional per-channel scale: the head's values are
// scaled down to an L2 norm of at most `threshold` (norm is sqrt of the sum + 1e-6, the factor clamped
// to [0, 1]).
void ink_clip(float* v, int n_head, int row, const float* scale, float threshold) {
    if (scale)
        for (size_t i = 0; i < static_cast<size_t>(n_head) * row; ++i) v[i] *= scale[i];
    for (int h = 0; h < n_head; ++h) {
        float* vh = v + static_cast<size_t>(h) * row;
        double ss = 0;
        for (int i = 0; i < row; ++i) ss += double(vh[i]) * double(vh[i]);
        const float norm = float(std::sqrt(ss)) + 1e-6f;
        const float factor = std::min(std::max(threshold / norm, 0.0f), 1.0f);
        for (int i = 0; i < row; ++i) vh[i] *= factor;
    }
}

}  // namespace

// An attention over the depth: the bank of earlier block outputs plus cur, each RMS-normalised (eps
// 1e-6, no weight), scored per query by dot(state, query) / sqrt(n_embd), softmax over the states, and
// the weighted sum per query. The query is [n_embd, q_count]; out is [q_count x n_embd].
void Ref::ink_mix(const std::vector<std::vector<float>>& blocks, const float* cur, const float* query,
                  int q_count, float* out) {
    const int e = cfg_.n_embd;
    const size_t n = blocks.size() + 1;
    std::memset(out, 0, size_t(q_count) * size_t(e) * sizeof(float));
    std::vector<float> st(n * size_t(e));
    for (size_t i = 0; i < blocks.size(); ++i)
        rmsnorm(blocks[i].data(), nullptr, st.data() + i * size_t(e), e, 1e-6f);
    rmsnorm(cur, nullptr, st.data() + blocks.size() * size_t(e), e, 1e-6f);
    const float w = float(1.0 / std::sqrt(double(e)));
    std::vector<float> sc(n);
    for (int q = 0; q < q_count; ++q) {
        const float* qv = query + size_t(q) * e;
        for (size_t s = 0; s < n; ++s) {
            const float* row = st.data() + s * size_t(e);
            float acc = 0;
            for (int i = 0; i < e; ++i) acc += row[i] * qv[i];
            sc[s] = acc * w;
        }
        softmax_inplace(sc.data(), int(n));
        float* o = out + size_t(q) * e;
        for (size_t s = 0; s < n; ++s) {
            const float p = sc[s];
            const float* row = st.data() + s * size_t(e);
            for (int i = 0; i < e; ++i) o[i] += p * row[i];
        }
    }
}

// The causal short convolution of one slot at channel `offset` of the shared history (rows of 2*n_kv +
// 2*n_embd channels: k, v, the attention output, the FFN output). out = (cur + conv(x)) * inv, x being
// the history rows plus cur; a `dyn` tensor of [n_conv taps per head] adds input-dependent taps, tap
// `lag` of head h weighting x row k-1-lag over that head's channels. No activation.
void Ref::ink_convolve(int slot, int offset, const float* cur, int n_ch, int n_heads, const float* dyn,
                       const gguf::Tensor* weight, float* out) {
    const Config& c = cfg_;
    const int k = c.ink.sconv;
    const int nch = 2 * (c.n_head_kv * c.head_dim) + 2 * c.n_embd;
    std::vector<float>& hist = ink_conv_hist_[size_t(slot)];
    std::vector<float> x(size_t(k) * n_ch);
    for (int j = 0; j < k - 1; ++j)
        std::memcpy(x.data() + size_t(j) * n_ch, hist.data() + size_t(j) * nch + offset,
                    size_t(n_ch) * sizeof(float));
    std::memcpy(x.data() + size_t(k - 1) * n_ch, cur, size_t(n_ch) * sizeof(float));
    for (int j = 0; j < k - 1; ++j)   // the state keeps the last k-1 rows, one row forward
        std::memcpy(hist.data() + size_t(j) * nch + offset, x.data() + size_t(j + 1) * n_ch,
                    size_t(n_ch) * sizeof(float));

    const int taps = int(weight->ne[0]);   // == k, {n_conv, channels}, taps contiguous per channel
    const float inv = float(1.0 / std::sqrt(double((dyn ? 2 : 1) * k + 1)));
    const size_t nchs = size_t(n_ch);
    std::vector<float> acc(nchs);
    if (weight->type == 0 /* F32 */) {   // the file's conv weights are F32
        const float* w = reinterpret_cast<const float*>(weight->data);
        for (int ch = 0; ch < n_ch; ++ch) {
            float sum = 0;
            for (int j = 0; j < taps; ++j)
                sum += w[size_t(ch) * taps + j] * x[size_t(j) * n_ch + ch];
            acc[size_t(ch)] = sum;
        }
    } else {
        const uint8_t* w = weight->data;
        const uint64_t row_bytes = type_bytes(weight->type, uint64_t(taps));
        for (int ch = 0; ch < n_ch; ++ch) {
            float buf[64];
            quants::dequant_row(weight->type, w + uint64_t(ch) * row_bytes, buf, uint64_t(taps));
            float sum = 0;
            for (int j = 0; j < taps; ++j) sum += buf[j] * x[size_t(j) * n_ch + ch];
            acc[size_t(ch)] = sum;
        }
    }
    if (dyn) {
        const int per_head = n_ch / n_heads;
        for (int lag = 0; lag < taps; ++lag)
            for (int h = 0; h < n_heads; ++h) {
                const float tap = dyn[size_t(h) * taps + lag];
                const int base = h * per_head;
                for (int d = 0; d < per_head; ++d)
                    acc[size_t(base + d)] += tap * x[size_t(k - 1 - lag) * n_ch + base + d];
            }
    }
    for (int ch = 0; ch < n_ch; ++ch) out[ch] = (cur[ch] + acc[size_t(ch)]) * inv;
}

// The routed experts: sigmoid scores, top-k by scores + selection bias, the unbiased scores
// renormalised (+1e-20) and scaled by expert_weights_scale, then per expert the fused gate/up rows
// (per-expert dequant scale on both), SiTU, and the down projection (its own per-expert scale).
void Ref::ink_moe(int il, const float* post, float* out) {
    const Config& c = cfg_;
    const int phys = il < c.n_layer ? il : c.ink.loop_start + (il - c.n_layer) % (c.ink.loop_end - c.ink.loop_start);
    const Weights::InklingLayer& K = w_->ink_L[size_t(phys)];
    const int e = c.n_embd, nf = c.layers[size_t(phys)].ffn.n_ff_exp;
    const int n_exp = c.n_expert, n_used = c.n_expert_used;
    const size_t ne_ = size_t(n_exp), nfh = size_t(nf), es = size_t(e);

    std::vector<float> logits(ne_);
    mv(K.router, n_exp, e, post, logits.data());
    if (trace_.on && cur_pos_ == 0) trace_.nodes["ffn_moe_logits-" + std::to_string(il)] = logits;
    std::vector<float> probs(ne_);
    for (int i = 0; i < n_exp; ++i) probs[size_t(i)] = sigmoid(logits[size_t(i)]);
    std::vector<int> idx(ne_);
    for (int i = 0; i < n_exp; ++i) idx[size_t(i)] = i;
    const int n_sorted = std::min(n_exp, n_used + 1);   // one past the cut, to see how close the 9th was
    std::partial_sort(idx.begin(), idx.begin() + n_sorted, idx.end(), [&](int a, int b) {
        const float sa = probs[size_t(a)] + K.exp_bias[size_t(a)];
        const float sb = probs[size_t(b)] + K.exp_bias[size_t(b)];
        return sa > sb || (sa == sb && a < b);
    });
    if (trace_.on && cur_pos_ == 0) {
        trace_.moe_last_ids.assign(idx.begin(), idx.begin() + n_used);
        trace_.moe_last_p.clear();
        for (int i = 0; i < n_sorted; ++i) trace_.moe_last_p.push_back(probs[size_t(idx[size_t(i)])]);
        const size_t nso = size_t(n_sorted);
        std::vector<float> idf(nso);
        for (int i = 0; i < n_sorted; ++i) idf[size_t(i)] = float(idx[size_t(i)]);
        trace_.nodes["ffn_moe_topk-" + std::to_string(il)] = idf;
    }
    float wsum = 0;
    for (int i = 0; i < n_used; ++i) wsum += probs[size_t(idx[size_t(i)])];
    wsum += 1e-20f;   // the reference's expert_weights_norm_eps
    if (trace_.on && cur_pos_ == 0) {
        const size_t nu_ = size_t(n_used);
        std::vector<float> ws(nu_);
        for (int i = 0; i < n_used; ++i)
            ws[size_t(i)] = probs[size_t(idx[size_t(i)])] / wsum * c.ink.route_scale;
        trace_.nodes["ffn_moe_weights_scaled-" + std::to_string(il)] = ws;
    }

    const gguf::Tensor* wg = K.gate_up;
    const gguf::Tensor* wd = K.down;
    const uint64_t row_gu = type_bytes(wg->type, uint64_t(e));
    const uint64_t exp_gu = uint64_t(2 * nf) * row_gu;   // one expert: 2*nf rows, gate first, up after
    const uint64_t row_dn = type_bytes(wd->type, uint64_t(nf));
    const uint64_t exp_dn = uint64_t(e) * row_dn;
    const float beta = c.ink.situ_beta, lb = c.ink.situ_lin_beta, rs = c.ink.route_scale;
    std::vector<float> g(nfh), u(nfh), h(nfh), dn(es);
    std::vector<float> acc(es, 0.0f);
    for (int i = 0; i < n_used; ++i) {
        const int ex = idx[size_t(i)];
        const float w = probs[size_t(ex)] / wsum * rs;
        const uint8_t* gbase = wg->data + uint64_t(ex) * exp_gu;
        const uint8_t* dbase = wd->data + uint64_t(ex) * exp_dn;
        const float gus = K.gate_up_scale.empty() ? 1.0f : K.gate_up_scale[size_t(ex)];
        const float ds = K.down_scale.empty() ? 1.0f : K.down_scale[size_t(ex)];
        parallel_for(nf, threads_, [&](int b, int en) {
            for (int j = b; j < en; ++j) {
                g[size_t(j)] = dot_row(wg->type, gbase + uint64_t(j) * row_gu, post, e) * gus;
                u[size_t(j)] = dot_row(wg->type, gbase + uint64_t(nf + j) * row_gu, post, e) * gus;
            }
        });
        for (int j = 0; j < nf; ++j) {
            const float gj = g[size_t(j)], uj = u[size_t(j)];
            h[size_t(j)] = (beta * std::tanh(gj / beta) * sigmoid(gj)) * (lb * std::tanh(uj / lb));
        }
        parallel_for(e, threads_, [&](int b, int en) {
            for (int j = b; j < en; ++j)
                dn[size_t(j)] = dot_row(wd->type, dbase + uint64_t(j) * row_dn, h.data(), nf) * ds;
        });
        for (int j = 0; j < e; ++j) acc[size_t(j)] += dn[size_t(j)] * w;
    }
    for (int j = 0; j < e; ++j) out[j] = acc[size_t(j)];
}

// One visit of the layer graph: the depth mixes, the attention with its relative bias and sinks, the
// residual bank, and the FFN half. `il` is the visit (the cache slot), `ip` the visit index within the
// block structure; the weights come from layer_phys(il), the state from slot il. `messages`, when
// given, are this pass's four controller messages (q, k, v, ffn).
void Ref::ink_visit(int il, int ip, int pos, float* cur, std::vector<std::vector<float>>& blocks,
                    const std::vector<float>* messages) {
    const Config& c = cfg_;
    const int e = c.n_embd;
    const int phys = il < c.n_layer ? il : c.ink.loop_start + (il - c.n_layer) % (c.ink.loop_end - c.ink.loop_start);
    const Weights::InklingLayer& K = w_->ink_L[size_t(phys)];
    const LayerRecipe& r = c.layers[size_t(phys)];
    const bool swa = r.attn.sliding;
    const int n_head = r.attn.n_head, nkv = r.attn.n_head_kv, hd = r.attn.head_dim;
    const int kvw = nkv * hd;
    const int rel_len = c.ink.rel_length;
    const float eps = float(c.rms_eps);
    const size_t es = size_t(e), nhs = size_t(n_head), kvws = size_t(kvw), gr = size_t(c.ink.gate_rank);

    const std::vector<float> inpSA(cur, cur + e);

    // The loop controller: gate = ctrl_down @ [rms_norm(cur) | rms_norm(message)] (eps 1e-6, both, no
    // weight), then out = (1 - tanh(forget @ gate)) * cur + tanh(gain @ gate) * message.
    auto control = [&](float* io, const float* message, int ci) {
        std::vector<float> gate(size_t(2) * size_t(e)), lo(gr);
        rmsnorm(io, nullptr, gate.data(), e, 1e-6f);
        rmsnorm(message, nullptr, gate.data() + e, e, 1e-6f);
        matvec(K.ctrl_down[ci].w->type, K.ctrl_down[ci].w->data, c.ink.gate_rank, 2 * e, gate.data(),
               lo.data(), threads_);
        std::vector<float> g(es), f(es);
        matvec(K.ctrl_gain[ci].w->type, K.ctrl_gain[ci].w->data, e, c.ink.gate_rank, lo.data(),
               g.data(), threads_);
        matvec(K.ctrl_forget[ci].w->type, K.ctrl_forget[ci].w->data, e, c.ink.gate_rank, lo.data(),
               f.data(), threads_);
        for (int i = 0; i < e; ++i) {
            const float gn = std::tanh(g[size_t(i)]), fn = std::tanh(f[size_t(i)]);
            io[i] = io[i] - fn * io[i] + gn * message[i];
        }
    };

    // The three attention inputs. The depth mix runs first (every layer but 0), then, in a looped pass,
    // the controller, then the norms: one attn_norm for each input on a sliding layer with messages,
    // one norm for the Q input otherwise (K and V inputs equal it), and the three input norms on the
    // global layers.
    std::vector<float> in0(es), in1(es), in2(es);
    if (K.res_q_attn.w) {
        const int qn = swa ? 1 : 3;
        std::vector<float> mixedq(size_t(qn) * size_t(e));
        ink_mix(blocks, cur, decode_all(K.res_q_attn.w).data(), qn, mixedq.data());
        in0.assign(mixedq.begin(), mixedq.begin() + e);
        in1.assign(mixedq.begin() + size_t(swa ? 0 : 1) * e, mixedq.begin() + size_t(swa ? 1 : 2) * e);
        in2.assign(mixedq.begin() + size_t(swa ? 0 : 2) * e, mixedq.end());
    } else {
        in0 = inpSA;
        in1 = inpSA;
        in2 = inpSA;
    }
    if (messages) {
        control(in0.data(), messages->data(), 0);
        control(in1.data(), messages->data() + e, 1);
        control(in2.data(), messages->data() + 2 * e, 2);
    }
    if (swa) {
        if (messages) {
            rmsnorm(in0.data(), K.attn_norm.data(), in0.data(), e, eps);
            rmsnorm(in1.data(), K.attn_norm.data(), in1.data(), e, eps);
            rmsnorm(in2.data(), K.attn_norm.data(), in2.data(), e, eps);
        } else {
            rmsnorm(in0.data(), K.attn_norm.data(), in0.data(), e, eps);
            in1 = in0;
            in2 = in0;
        }
    } else {
        rmsnorm(in0.data(), K.q_inp_norm.data(), in0.data(), e, eps);
        rmsnorm(in1.data(), K.k_inp_norm.data(), in1.data(), e, eps);
        rmsnorm(in2.data(), K.v_inp_norm.data(), in2.data(), e, eps);
    }

    // q, k, v through their projections; k and v then through their short convolutions (the state keeps
    // the raw projections). r for the relative bias.
    std::vector<float> q(nhs * size_t(hd)), kraw(kvws), vraw(kvws), rv(nhs * size_t(rel_len));
    mv(K.wq, n_head * hd, e, in0.data(), q.data());
    mv(K.wk, kvw, e, in1.data(), kraw.data());
    mv(K.wv, kvw, e, in2.data(), vraw.data());
    mv(K.wr, n_head * rel_len, e, in0.data(), rv.data());

    std::vector<float> k(kvws), v(kvws);
    if (swa) {
        ink_convolve(il, 0, kraw.data(), kvw, nkv, nullptr, K.k_conv.w, k.data());
        ink_convolve(il, kvw, vraw.data(), kvw, nkv, nullptr, K.v_conv.w, v.data());
    } else {
        std::vector<float> kdyn(size_t(c.ink.sconv) * nkv), vdyn(size_t(c.ink.sconv) * nkv);
        mv(K.k_conv_proj, c.ink.sconv * nkv, e, in1.data(), kdyn.data());
        mv(K.v_conv_proj, c.ink.sconv * nkv, e, in2.data(), vdyn.data());
        ink_convolve(il, 0, kraw.data(), kvw, nkv, kdyn.data(), K.k_conv.w, k.data());
        ink_convolve(il, kvw, vraw.data(), kvw, nkv, vdyn.data(), K.v_conv.w, v.data());
    }
    // The clips bound q, k and r in place of the q/k norms the file carries no tensors for; V gets a
    // per-head RMS norm with no weight.
    ink_clip(q.data(), n_head, hd, K.q_clip.data(), c.ink.qk_clip);
    ink_clip(k.data(), nkv, hd, nullptr, c.ink.qk_clip);
    ink_clip(rv.data(), n_head, rel_len, K.rel_clip.data(), c.ink.rel_clip);
    head_norm(v.data(), v.data(), nkv, hd, eps, 1.0f);
    if (!swa) {
        // the global layers' log-scaled positions multiply q and r, not k
        const float tau = float(1.0 + c.ink.log_alpha *
            std::log(std::max(1.0, double(pos + 1) / double(c.ink.log_floor))));
        for (size_t i = 0; i < q.size(); ++i) q[i] *= tau;
        for (size_t i = 0; i < rv.size(); ++i) rv[i] *= tau;
    }
    if (trace_.on && pos == 0) {
        trace_.nodes["Qcur-" + std::to_string(il)] = q;
        trace_.nodes["Kcur-" + std::to_string(il)] = k;
        trace_.nodes["Vcur-" + std::to_string(il)] = v;
        trace_.nodes["Rcur-" + std::to_string(il)] = rv;
    }

    // The bias table: one row per backward distance, per head -- rel_bias row d dotted with r's head
    // column, minus this token's sink logit. The table enters the softmax AFTER the q.k scale (see the
    // score line below): logits = scale*q.k + bias, and the softmax carries the sink column with logit 0.
    std::vector<float> sink(nhs);
    mv(K.sink, n_head, e, in0.data(), sink.data());
    const int extent = swa ? r.attn.window : c.ink.rel_extent;
    const uint64_t rb_row = type_bytes(K.rel_bias.w->type, uint64_t(rel_len));
    std::vector<float> table(size_t(extent + 1) * size_t(n_head), 0.0f);   // last row = the zero pad
    for (int d = 0; d < extent; ++d)
        for (int h = 0; h < n_head; ++h)
            table[size_t(d) * n_head + size_t(h)] =
                dot_row(K.rel_bias.w->type, K.rel_bias.w->data + uint64_t(d) * rb_row,
                        rv.data() + size_t(h) * rel_len, rel_len) - sink[size_t(h)];

    // The attention itself: GQA over the slot's f16 cache, scores scale*(q.k + bias), causal, SWA's
    // window, and the folded sink column (logit 0) in the denominator.
    std::vector<uint16_t>& kc = ink_k_cache_[size_t(il)];
    std::vector<uint16_t>& vc = ink_v_cache_[size_t(il)];
    for (int i = 0; i < kvw; ++i) {
        kc[size_t(pos) * kvw + size_t(i)] = f32_to_f16(k[size_t(i)]);
        vc[size_t(pos) * kvw + size_t(i)] = f32_to_f16(v[size_t(i)]);
    }
    const int n_grp = n_head / nkv;
    const float scale = float(1.0 / std::sqrt(double(hd)));
    const int t0 = swa ? std::max(0, pos - r.attn.window + 1) : 0;
    const int n_vis = pos - t0 + 1;
    const size_t nvs = size_t(n_vis);
    std::vector<float> attnv(nhs * size_t(hd), 0.0f);
    std::vector<float> sc(nvs);
    std::vector<float> kq_rec;
    if (trace_.on && pos == 0) kq_rec.resize(size_t(n_head));
    for (int h = 0; h < n_head; ++h) {
        const int kh = h / n_grp;
        const float* qh = q.data() + size_t(h) * hd;
        for (int j = t0; j <= pos; ++j) {
            const uint16_t* ks = kc.data() + size_t(j) * kvw + size_t(kh) * hd;
            float qk = 0;
            for (int d = 0; d < hd; ++d) qk += qh[d] * quants::f16_to_f32(ks[d]);
            if (trace_.on && pos == 0) kq_rec[size_t(h)] = qk;
            const int dist = std::min(pos - j, extent);
            sc[size_t(j - t0)] = scale * qk + table[size_t(dist) * n_head + size_t(h)];
        }
        float m = 0.0f;   // the sink column's logit is 0 and joins the max, as the reference does
        for (int i = 0; i < n_vis; ++i) m = std::max(m, sc[size_t(i)]);
        float denom = std::exp(-m);   // the sink column
        for (int i = 0; i < n_vis; ++i) {
            sc[size_t(i)] = std::exp(sc[size_t(i)] - m);
            denom += sc[size_t(i)];
        }
        const float inv = denom > 0 ? 1.0f / denom : 0.0f;
        float* oh = attnv.data() + size_t(h) * hd;
        for (int i = 0; i < n_vis; ++i) {
            const float p = sc[size_t(i)] * inv;
            const uint16_t* vs = vc.data() + size_t(t0 + i) * kvw + size_t(kh) * hd;
            for (int d = 0; d < hd; ++d) oh[d] += p * quants::f16_to_f32(vs[d]);
        }
    }

    // The output gate, then the output projection and its short convolution.
    std::vector<float> gate(nhs * size_t(hd));
    mv(K.wgate, n_head * hd, e, in0.data(), gate.data());
    for (size_t i = 0; i < gate.size(); ++i) attnv[i] *= sigmoid(gate[i]);
    std::vector<float> attn_out(es), attn_conv(es);
    mv(K.wo, e, n_head * hd, attnv.data(), attn_out.data());
    ink_convolve(il, 2 * kvw, attn_out.data(), e, 1, nullptr, K.attn_post_conv.w, attn_conv.data());
    if (trace_.on && pos == 0) {
        trace_.nodes["kq-" + std::to_string(il)] = kq_rec;
        trace_.nodes["attn_out-" + std::to_string(il)] = attn_conv;
    }

    // Every res_block-th visit starts a new residual stream by banking the pre-attention state; the
    // others add the attention output to it.
    if (ip % c.ink.res_block == 0) {
        blocks.push_back(inpSA);
        std::copy(attn_conv.begin(), attn_conv.end(), cur);
    } else {
        for (int i = 0; i < e; ++i) cur[i] = inpSA[i] + attn_conv[size_t(i)];
    }
    if (trace_.on && pos == 0) trace_.attn_out[size_t(il)] = std::vector<float>(cur, cur + e);

    // The FFN half: the depth mix, the loop controller on its own message, the norm, the experts, and
    // the FFN output's short convolution, added back to the pre-FFN state.
    const std::vector<float> ffn_inp(cur, cur + e);
    if (trace_.on && pos == 0) trace_.nodes["ffn_inp-" + std::to_string(il)] = ffn_inp;
    {
        std::vector<float> mixed(es);
        ink_mix(blocks, cur, decode_all(K.res_q_ffn.w).data(), 1, mixed.data());
        if (messages) control(mixed.data(), messages->data() + 3 * e, 3);
        rmsnorm(mixed.data(), K.ffn_norm.data(), mixed.data(), e, eps);
        std::vector<float> moe_out(es), ffn_conv(es);
        ink_moe(il, mixed.data(), moe_out.data());
        ink_convolve(il, 2 * kvw + e, moe_out.data(), e, 1, nullptr, K.ffn_post_conv.w, ffn_conv.data());
        for (int i = 0; i < e; ++i) cur[i] = ffn_inp[i] + ffn_conv[size_t(i)];
    }
    if (trace_.on && pos == 0) trace_.l_out[size_t(il)] = std::vector<float>(cur, cur + e);
}

const float* Ref::forward_inkling(int32_t token, int pos, bool want_logits) {
    const Config& c = cfg_;
    cur_pos_ = pos;
    const int e = c.n_embd;
    const size_t es = size_t(e);
    const int core = c.ink.loop_end - c.ink.loop_start;
    const int n_vis = c.n_layer + (c.ink.loop_count - 1) * core;
    if (trace_.on && pos == 0) {
        trace_.attn_out.assign(size_t(n_vis), {});
        trace_.l_out.assign(size_t(n_vis), {});
    }

    std::vector<float> cur(es);
    embed_token(token, cur.data());
    rmsnorm(cur.data(), w_->ink_tok_norm.data(), cur.data(), e, float(c.rms_eps));   // token_embd_norm
    if (trace_.on && pos == 0) trace_.x0.assign(cur.begin(), cur.begin() + 8);

    std::vector<std::vector<float>> blocks;
    for (int il = 0; il < c.ink.loop_start; ++il) ink_visit(il, il, pos, cur.data(), blocks, nullptr);

    // Every pass starts from the prelude output; its bank reads the previous pass's core blocks after
    // the first core block, plus the previous pass's output, through the layer's own message queries.
    const std::vector<float> prelude = cur;
    const std::vector<std::vector<float>> prelude_blocks = blocks;
    for (int pass = 0; pass < c.ink.loop_count; ++pass) {
        std::vector<std::vector<float>> bank;
        std::vector<float> bank_out;
        if (pass > 0) {
            bank.assign(blocks.begin() + ptrdiff_t(prelude_blocks.size() + 1), blocks.end());
            bank_out = cur;
            cur = prelude;
            blocks = prelude_blocks;
        }
        for (int ip = c.ink.loop_start; ip < c.ink.loop_end; ++ip) {
            const int il = pass == 0 ? ip : c.n_layer + (pass - 1) * core + (ip - c.ink.loop_start);
            if (pass == 0) {
                ink_visit(il, ip, pos, cur.data(), blocks, nullptr);
                continue;
            }
            std::vector<float> msgs(size_t(4) * size_t(e));
            ink_mix(bank, bank_out.data(), decode_all(w_->ink_L[size_t(ip)].msg_query.w).data(), 4,
                    msgs.data());
            ink_visit(il, ip, pos, cur.data(), blocks, &msgs);
        }
    }

    for (int il = c.ink.loop_end; il < c.n_layer; ++il) ink_visit(il, il, pos, cur.data(), blocks, nullptr);

    if (!want_logits) return nullptr;
    std::vector<float> mixed(es);
    ink_mix(blocks, cur.data(), w_->ink_out_res_query.data(), 1, mixed.data());
    return head(mixed.data());
}

}  // namespace limbo::step
