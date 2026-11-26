// Copyright (c) 2026 ITERFORM. All rights reserved.
// tests/cuda_dense_test.cu — qwen35 dense on the GPU engine at a geometry that is not the 27B's: a synthetic file (written here, random
// weights in fourteen formats, tied head, 8 query heads on 2 KV heads, n_embd 1024, an FFN of 7 super-blocks) run by limbo::cuda::Gpu and
// by the CPU reference (limbo::step::Ref). Checks: the decode step's logits against the reference's, the batched prefill (3 and 2 planes)
// against the same, the prefill's state (a decode step after each), and the T-row pass (rows == sequential steps bit for bit, commit). SKIP (exit 0) without a GPU.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "limbo/check.hpp"
#include "limbo/cuda/gpu.hpp"
#include "limbo/model.hpp"
#include "limbo/step.hpp"
#include "limbo/types.hpp"

namespace {

using Bytes = std::vector<uint8_t>;
using limbo::Type;

struct Wr {
    Bytes b;
    void raw(const void* p, size_t n) { b.insert(b.end(), static_cast<const uint8_t*>(p), static_cast<const uint8_t*>(p) + n); }
    void u32(uint32_t v) { raw(&v, 4); }
    void u64(uint64_t v) { raw(&v, 8); }
    void f32(float v) { raw(&v, 4); }
    void str(const std::string& s) { u64(s.size()); raw(s.data(), s.size()); }
    void kv_u32(const std::string& k, uint32_t v) { str(k); u32(4); u32(v); }
    void kv_f32(const std::string& k, float v) { str(k); u32(6); f32(v); }
    void pad(size_t a) { while (b.size() % a) b.push_back(0); }
};

struct Tn {
    std::string name;
    uint32_t type;
    uint64_t ne0, ne1;  // ne1 == 0: one dimension
    Bytes data;
};

uint32_t g_seed = 12345u;
uint32_t rnd() { g_seed = g_seed * 1664525u + 1013904223u; return g_seed >> 8; }
float frand() { return float(rnd() & 0xffff) / 65536.0f - 0.5f; }

// the byte offsets of the fp16 scales a block of each format carries (d, then dmin where there is one)
void patch_scales(uint32_t t, Bytes& d, size_t n_blocks, size_t bb) {
    auto put = [&](size_t at, uint16_t h) { d[at] = uint8_t(h & 0xff); d[at + 1] = uint8_t(h >> 8); };
    const uint16_t small = 0x1018;  // 0.0005
    for (size_t i = 0; i < n_blocks; ++i) {
        uint8_t* p = d.data() + i * bb;
        const size_t o = size_t(p - d.data());
        if (t == uint32_t(Type::Q4_K) || t == uint32_t(Type::Q5_K)) { put(o, small); put(o + 2, small); }
        else if (t == uint32_t(Type::Q2_K)) { put(o + 80, small); put(o + 82, small); }
        else if (t == uint32_t(Type::Q3_K)) put(o + 108, small);
        else if (t == uint32_t(Type::Q6_K)) put(o + 208, small);
        else if (t == uint32_t(Type::IQ1_M)) {  // d is spread over the top nibbles of the four 16-bit scale words (0x1018 = nibbles 8, 1, 0, 1)
            const int nib[4] = {8, 1, 0, 1};
            for (int k = 0; k < 4; ++k) d[o + 49 + 2 * size_t(k)] = uint8_t((d[o + 49 + 2 * size_t(k)] & 0x0f) | (nib[k] << 4));
        }
        else put(o, small);  // IQ*, Q8_0, IQ4_NL: d first
    }
}

Tn quant(const std::string& name, uint32_t type, uint64_t n_in, uint64_t n_out) {
    Tn t{name, type, n_in, n_out, {}};
    const uint64_t bytes = limbo::type_bytes(type, n_in) * n_out;
    t.data.resize(bytes);
    for (uint8_t& v : t.data) v = uint8_t(rnd());
    const uint64_t row = limbo::type_bytes(type, n_in);
    // block size in bytes: a 32-element block for Q8_0 / IQ4_NL, a 256-element super-block for the rest
    const uint64_t bb = (type == uint32_t(Type::Q8_0)) ? 34 : (type == uint32_t(Type::IQ4_NL)) ? 18 : limbo::type_bytes(type, 256);
    patch_scales(type, t.data, size_t(bytes / bb), size_t(bb));
    (void)row;
    return t;
}
Tn f32t(const std::string& name, uint64_t ne0, uint64_t ne1, float base, float spread) {
    Tn t{name, uint32_t(Type::F32), ne0, ne1, {}};
    const uint64_t n = ne0 * std::max<uint64_t>(ne1, 1);
    t.data.resize(n * 4);
    for (uint64_t i = 0; i < n; ++i) { const float v = base + spread * frand(); std::memcpy(t.data.data() + i * 4, &v, 4); }
    return t;
}

constexpr int kLayers = 8, kEmbd = 1024, kHead = 8, kKv = 2, kHd = 256, kKHead = 4, kVHead = 8, kGdn = 128, kConv = 4, kFf = 1792, kVocab = 512;

std::string build(const std::string& dir) {
    std::vector<Tn> ts;
    const uint32_t ty[] = {uint32_t(Type::Q4_K),   uint32_t(Type::Q2_K),   uint32_t(Type::Q3_K),    uint32_t(Type::Q6_K),   uint32_t(Type::IQ4_XS),
                           uint32_t(Type::IQ3_S),  uint32_t(Type::IQ2_XXS), uint32_t(Type::IQ2_XS),  uint32_t(Type::IQ2_S), uint32_t(Type::IQ3_XXS),
                           uint32_t(Type::IQ1_S),  uint32_t(Type::IQ1_M),  uint32_t(Type::Q8_0),   uint32_t(Type::Q5_K),    uint32_t(Type::IQ4_NL)};
    int pick = 0;
    auto next = [&]() { return ty[size_t(pick++) % (sizeof ty / sizeof ty[0])]; };
    const int qkv_w = 2 * kKHead * kGdn + kVHead * kGdn, vn = kVHead * kGdn, qw = kHead * kHd, kvw = kKv * kHd;
    ts.push_back(quant("token_embd.weight", next(), kEmbd, kVocab));  // no output.weight: the head is tied to it
    ts.push_back(f32t("output_norm.weight", kEmbd, 0, 1.0f, 0.2f));
    for (int l = 0; l < kLayers; ++l) {
        const std::string b = "blk." + std::to_string(l) + ".";
        ts.push_back(f32t(b + "attn_norm.weight", kEmbd, 0, 1.0f, 0.2f));
        ts.push_back(f32t(b + "post_attention_norm.weight", kEmbd, 0, 1.0f, 0.2f));
        if ((l + 1) % 4 == 0) {
            ts.push_back(quant(b + "attn_q.weight", next(), kEmbd, 2 * qw));
            ts.push_back(quant(b + "attn_k.weight", next(), kEmbd, kvw));
            ts.push_back(quant(b + "attn_v.weight", next(), kEmbd, kvw));
            ts.push_back(quant(b + "attn_output.weight", next(), qw, kEmbd));
            ts.push_back(f32t(b + "attn_q_norm.weight", kHd, 0, 1.0f, 0.2f));
            ts.push_back(f32t(b + "attn_k_norm.weight", kHd, 0, 1.0f, 0.2f));
        } else {
            ts.push_back(quant(b + "attn_qkv.weight", next(), kEmbd, qkv_w));
            ts.push_back(quant(b + "attn_gate.weight", next(), kEmbd, vn));
            ts.push_back(quant(b + "ssm_out.weight", next(), vn, kEmbd));
            ts.push_back(f32t(b + "ssm_alpha.weight", kEmbd, kVHead, 0.0f, 0.1f));
            ts.push_back(f32t(b + "ssm_beta.weight", kEmbd, kVHead, 0.0f, 0.1f));
            ts.push_back(f32t(b + "ssm_conv1d.weight", kConv, qkv_w, 0.0f, 0.8f));
            ts.push_back(f32t(b + "ssm_dt.bias", kVHead, 0, 0.1f, 0.2f));
            ts.push_back(f32t(b + "ssm_a", kVHead, 0, -0.6f, 0.8f));
            ts.push_back(f32t(b + "ssm_norm.weight", kGdn, 0, 1.0f, 0.2f));
        }
        ts.push_back(quant(b + "ffn_gate.weight", next(), kEmbd, kFf));
        ts.push_back(quant(b + "ffn_up.weight", next(), kEmbd, kFf));
        ts.push_back(quant(b + "ffn_down.weight", next(), kFf, kEmbd));
    }
    Wr w;
    w.u32(0x46554747);
    w.u32(3);
    w.u64(ts.size());
    w.u64(18);
    w.str("general.architecture"); w.u32(8); w.str("qwen35");
    const std::string a = "qwen35.";
    w.kv_u32(a + "block_count", kLayers);
    w.kv_u32(a + "context_length", 4096);
    w.kv_u32(a + "embedding_length", kEmbd);
    w.kv_u32(a + "feed_forward_length", kFf);
    w.kv_u32(a + "attention.head_count", kHead);
    w.kv_u32(a + "attention.head_count_kv", kKv);
    w.kv_u32(a + "attention.key_length", kHd);
    w.kv_u32(a + "attention.value_length", kHd);
    w.kv_f32(a + "rope.freq_base", 1e7f);
    w.kv_f32(a + "attention.layer_norm_rms_epsilon", 1e-6f);
    w.kv_u32(a + "rope.dimension_count", 64);
    w.kv_u32(a + "full_attention_interval", 4);
    w.kv_u32(a + "ssm.conv_kernel", kConv);
    w.kv_u32(a + "ssm.state_size", kGdn);
    w.kv_u32(a + "ssm.group_count", kKHead);
    w.kv_u32(a + "ssm.time_step_rank", kVHead);
    w.kv_u32(a + "ssm.inner_size", vn);
    uint64_t off = 0;
    std::vector<uint64_t> offs;
    for (const Tn& t : ts) { offs.push_back(off); off += (t.data.size() + 31) / 32 * 32; }
    for (size_t i = 0; i < ts.size(); ++i) {
        w.str(ts[i].name);
        w.u32(ts[i].ne1 ? 2 : 1);
        w.u64(ts[i].ne0);
        if (ts[i].ne1) w.u64(ts[i].ne1);
        w.u32(ts[i].type);
        w.u64(offs[i]);
    }
    w.pad(32);
    for (size_t i = 0; i < ts.size(); ++i) { w.raw(ts[i].data.data(), ts[i].data.size()); w.pad(32); }
    const std::string path = dir + "/dense_synth.gguf";
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(w.b.data()), std::streamsize(w.b.size()));
    return path;
}

double rel_diff(const float* a, const float* b, int n) {
    double mx = 0, ref = 0;
    for (int i = 0; i < n; ++i) { mx = std::max(mx, std::fabs(double(a[i]) - double(b[i]))); ref = std::max(ref, std::fabs(double(b[i]))); }
    return mx / std::max(ref, 1e-9);
}
int argmax(const float* v, int n) { int k = 0; for (int i = 1; i < n; ++i) if (v[i] > v[k]) k = i; return k; }

}  // namespace

int main() {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) { std::printf("SKIP: no CUDA device\n"); return 0; }
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    if (free_b < (2ull << 30)) { std::printf("SKIP: under 2 GB of VRAM free\n"); return 0; }
    const std::string path = build(limbo::test::tmp_dir("cuda_dense").string());
    std::string err;
    auto ref = limbo::step::Ref::open({path}, 64, 1, &err);
    CHECK(ref != nullptr, "the CPU reference opens the synthetic qwen35 file");
    if (!ref) { std::printf("  %s\n", err.c_str()); return limbo::test::finish("cuda_dense"); }
    auto model = limbo::model::Model::open({path}, &err);
    auto gpu = model ? limbo::cuda::Gpu::open(*model, 64, 0, &err) : nullptr;
    CHECK(gpu != nullptr, "Gpu::open takes the dense file (tied head, group 4, n_embd 1024)");
    if (!gpu) { std::printf("  %s\n", err.c_str()); return limbo::test::finish("cuda_dense"); }

    const int n = 20;
    std::vector<int32_t> toks(static_cast<size_t>(n));
    for (int32_t& t : toks) t = int32_t(rnd() % kVocab);
    std::vector<std::vector<float>> want(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const float* l = ref->forward(toks[size_t(i)], i, true);
        want[size_t(i)].assign(l, l + kVocab);
    }
    std::vector<float> lg(kVocab);
    double worst = 0;
    int agree = 0;
    gpu->reset();
    gpu->set_prefill(0, 1024);
    for (int i = 0; i < n; ++i) {
        gpu->step(toks[size_t(i)], i, true);
        gpu->read_logits(lg.data());
        worst = std::max(worst, rel_diff(lg.data(), want[size_t(i)].data(), kVocab));
        agree += argmax(lg.data(), kVocab) == argmax(want[size_t(i)].data(), kVocab);
    }
    std::printf("     decode steps vs the reference: worst relative logit difference %.3e, top-1 agrees at %d of %d\n", worst, agree, n);
    CHECK(worst < 2e-3, "decode step logits within 2e-3 of the reference's");
    CHECK(agree >= n - 1, "decode step top-1 agrees with the reference");

    for (int planes : {3, 2}) {
        gpu->reset();
        gpu->set_prefill(planes, 1024);
        gpu->prefill(toks.data(), n, 0, true);
        gpu->read_logits(lg.data());
        const double d = rel_diff(lg.data(), want[size_t(n - 1)].data(), kVocab);
        std::printf("     batched prefill, %d planes, %d tokens: relative logit difference %.3e\n", planes, n, d);
        CHECK(d < (planes == 3 ? 2e-3 : 1e-2), planes == 3 ? "batched prefill (3 planes) matches the reference" : "batched prefill (2 planes) matches the reference");
        // the state it leaves: a decode step after it
        const int32_t nxt = int32_t(rnd() % kVocab);
        gpu->step(nxt, n, true);
        gpu->read_logits(lg.data());
        const float* l = ref->forward(nxt, n, true);
        const double d2 = rel_diff(lg.data(), l, kVocab);
        std::printf("     a decode step after it: %.3e\n", d2);
        CHECK(d2 < (planes == 3 ? 2e-3 : 1e-2), "the state the prefill leaves continues like the reference's");
        // the reference advanced one position: rebuild it for the next round
        ref = limbo::step::Ref::open({path}, 64, 1, &err);
        for (int i = 0; i < n; ++i) ref->forward(toks[size_t(i)], i, false);
    }
    // the T-row pass: the rows of a window are the sequential steps' logits bit for bit, and commit(n) leaves the state of n steps
    auto same = [&](const float* a, const float* b) { return std::memcmp(a, b, kVocab * sizeof(float)) == 0; };
    const int pre = 6;
    std::vector<int32_t> win(8);
    for (int32_t& t : win) t = int32_t(rnd() % kVocab);
    std::vector<std::vector<float>> seq(9);
    gpu->reset();
    gpu->set_prefill(0, 1024);
    gpu->prefill(toks.data(), pre, 0, false);
    for (int i = 0; i < 9; ++i) {  // win[0..7] then one more token (win[0] again): logits after each
        gpu->step(i < 8 ? win[size_t(i)] : win[0], pre + i, true);
        seq[size_t(i)].resize(kVocab);
        gpu->read_logits(seq[size_t(i)].data());
    }
    for (int T : {1, 2, 3, 4, 8}) {
        gpu->reset();
        gpu->prefill(toks.data(), pre, 0, false);
        std::vector<float> rows(size_t(T) * kVocab);
        const bool ok = gpu->step_rows(win.data(), T, pre, true) && gpu->read_logits_rows(rows.data(), T);
        bool eq = ok;
        for (int t = 0; t < T && eq; ++t) eq = same(rows.data() + size_t(t) * kVocab, seq[size_t(t)].data());
        char what[96];
        std::snprintf(what, sizeof what, "step_rows T=%d: every row bit-identical to the sequential step", T);
        CHECK(eq, what);
        if (T >= 2) {  // write_back off, commit 1 of the window: the next step continues as after one step
            gpu->reset();
            gpu->prefill(toks.data(), pre, 0, false);
            gpu->step_rows(win.data(), T, pre, false);
            gpu->commit(1);
            std::vector<float> l1(kVocab);
            gpu->step(win[1], pre + 1, true);
            gpu->read_logits(l1.data());
            std::snprintf(what, sizeof what, "commit(1) of a T=%d window: the next step equals the sequential one", T);
            CHECK(same(l1.data(), seq[1].data()), what);
        }
    }
    return limbo::test::finish("cuda_dense");
}
