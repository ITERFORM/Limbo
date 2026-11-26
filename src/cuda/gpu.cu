// Copyright (c) 2026 ITERFORM. All rights reserved.
// src/cuda/gpu.cu — the rung-1 forward pass on the device. See include/limbo/cuda/gpu.hpp.
//
// The order of operations is src/step.cpp's (itself the oracle's graph, bench/rung1-graph-spec.md):
//
//   x = token_embd[token]
//   for each of the 40 main layers:  x += attn(norm(x));  x += moe(post_norm(x))
//   logits = output @ (norm(x) * output_norm)
//
// attn is the gated delta net on 30 layers and gated GQA attention on 10. Every kernel of a step goes on
// one stream with no host synchronisation: the router's top-8 is chosen on the device and the batched expert
// GEMVs read their ids from device memory.
#include "limbo/cuda/gpu.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>

#include "gpu_impl.hpp"
#include "limbo/cuda/gemm.hpp"
#include "limbo/cuda/gemv.hpp"
#include "limbo/quants.hpp"
#include "limbo/types.hpp"
#include "ops.hpp"

namespace limbo::cuda {

namespace {

constexpr uint64_t kAlign = 256;
constexpr size_t kStageBytes = 32u << 20;

uint64_t align_up(uint64_t v) { return (v + kAlign - 1) / kAlign * kAlign; }

}  // namespace

namespace {

#define CU_TRY(call, what)                                                                      \
    do {                                                                                        \
        cudaError_t e_ = (call);                                                                \
        if (e_ != cudaSuccess) {                                                                \
            if (err) *err = std::string(what) + ": " + cudaGetErrorString(e_);                  \
            return nullptr;                                                                     \
        }                                                                                       \
    } while (0)

// Double-buffered pinned staging: the mmap'd file is copied into one buffer while the other is in flight.
struct Uploader {
    void* stage[2] = {nullptr, nullptr};
    cudaEvent_t ev[2] = {nullptr, nullptr};
    cudaStream_t s = nullptr;
    int b = 0;

    bool init() {
        if (cudaStreamCreate(&s) != cudaSuccess) return false;
        for (int i = 0; i < 2; ++i) {
            if (cudaMallocHost(&stage[i], kStageBytes) != cudaSuccess) return false;
            if (cudaEventCreateWithFlags(&ev[i], cudaEventDisableTiming) != cudaSuccess) return false;
        }
        return true;
    }
    bool copy(void* dst, const void* src, size_t n) {
        const uint8_t* in = static_cast<const uint8_t*>(src);
        uint8_t* out = static_cast<uint8_t*>(dst);
        for (size_t off = 0; off < n; off += kStageBytes) {
            const size_t c = std::min(kStageBytes, n - off);
            if (cudaEventSynchronize(ev[b]) != cudaSuccess) return false;
            std::memcpy(stage[b], in + off, c);
            if (cudaMemcpyAsync(out + off, stage[b], c, cudaMemcpyHostToDevice, s) != cudaSuccess) return false;
            if (cudaEventRecord(ev[b], s) != cudaSuccess) return false;
            b ^= 1;
        }
        return true;
    }
    ~Uploader() {
        if (s) cudaStreamSynchronize(s);
        for (int i = 0; i < 2; ++i) {
            if (ev[i]) cudaEventDestroy(ev[i]);
            if (stage[i]) cudaFreeHost(stage[i]);
        }
        if (s) cudaStreamDestroy(s);
    }
};

}  // namespace

Gpu::Gpu() = default;
Gpu::~Gpu() = default;

int Gpu::n_vocab() const { return p_->c.n_vocab; }
int Gpu::n_ctx() const { return p_->n_ctx; }
double Gpu::load_ms() const { return p_->load_ms; }
uint64_t Gpu::weight_bytes() const { return p_->weight_bytes; }
uint64_t Gpu::vram_bytes() const { return p_->vram_bytes + p_->pf.bytes; }
uint64_t Gpu::free_before_bytes() const { return p_->free_before; }
uint64_t Gpu::free_after_bytes() const { return p_->free_after; }

uint64_t Gpu::expert_cache_bytes() const { return p_->q.slab_bytes; }
void Gpu::set_layer_trace(std::vector<std::vector<float>>* t) { p_->trace = t; }
void Gpu::set_moe_trace(std::vector<std::vector<float>>* t) { p_->moe_trace = t; }
void Gpu::set_prefill_profile(std::vector<std::pair<std::string, double>>* t) { p_->pf_prof = t; }
void Gpu::set_decode_profile(std::vector<std::pair<std::string, double>>* t) { p_->dec_prof = t; }
void Gpu::sync() { cudaStreamSynchronize(p_->stream); }

std::unique_ptr<Gpu> Gpu::open(const model::Model& m, int n_ctx, uint64_t min_free_bytes, std::string* err, KvFormat kv_fmt, int pool_pages,
                               bool q4e_mtp) {
    std::unique_ptr<Gpu> g(new Gpu());
    g->p_ = std::make_unique<Impl>();
    Impl& I = *g->p_;
    Cfg& c = I.c;
    const model::Geometry& geo = m.geometry();

    CU_TRY(cudaSetDevice(0), "cudaSetDevice");
    CU_TRY(cudaFree(nullptr), "CUDA context");
    size_t free_b = 0, total_b = 0;
    CU_TRY(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
    I.free_before = free_b;
    if (free_b < min_free_bytes) {
        if (err)
            *err = "only " + std::to_string(double(free_b) / 1e9) + " GB of VRAM free (of " +
                   std::to_string(double(total_b) / 1e9) + " GB), need >= " + std::to_string(double(min_free_bytes) / 1e9) +
                   " GB: the desktop or its engine holds the rest";
        return nullptr;
    }
    std::string terr;
    if (!init_tables(&terr) || !gemm_init(&terr)) {
        if (err) *err = terr;
        return nullptr;
    }

    // ---- geometry, from the file ------------------------------------------------------------------
    const bool q4e = geo.arch == "qwen4exp";
    if (geo.arch != "qwen35moe" && geo.arch != "qwen35" && !q4e) {
        if (err) *err = "gpu: only qwen35moe, qwen35 and qwen4exp are implemented, this file is " + geo.arch;
        return nullptr;
    }
    I.q4e = q4e;
    c.n_layer_main = geo.n_layer_main > 0 ? geo.n_layer_main : geo.n_layer;
    c.n_embd = geo.n_embd;
    c.n_head = geo.n_head;
    c.n_head_kv = geo.n_head_kv;
    c.hd = geo.head_dim;
    c.rope_dim = geo.rope_dim;
    c.n_vocab = geo.n_vocab;
    c.n_expert = geo.n_expert;
    c.n_used = geo.n_expert_used;
    c.n_ff_exp = geo.n_ff_exp;
    c.n_ff_shexp = geo.n_ff_shexp;
    c.n_ff = geo.arch == "qwen35" ? geo.n_ff : 0;
    c.n_k_head = geo.ssm_groups;
    c.n_v_head = geo.ssm_dt_rank;
    c.gdn_head = geo.ssm_state;
    c.conv_k = geo.ssm_conv;
    c.dt_rank = geo.ssm_dt_rank;
    c.qkv_w = 2 * c.n_k_head * c.gdn_head + c.n_v_head * c.gdn_head;
    c.eps = geo.rms_eps > 0 ? geo.rms_eps : 1e-6;
    c.rope_base = geo.rope_base > 0 ? geo.rope_base : 1e7;
    I.n_ctx = n_ctx;
    const bool dense = geo.arch == "qwen35";
    if (dense) I.pf_planes = 2;  // no router to flip on activation rounding: 2 planes (1 part in 16192 of a block's amplitude) is 1.3x faster than 3
    // qwen4exp's q/kv group is 24/2 = 12 (the scorers instantiate G = 8 and 12), its FF widths are 640 (a gemv
    // tail), and the hyper-connection mixers replace the norms the other recipes carry.
    const int grp_ok = q4e ? 12 : 8;
    if (c.hd != 256 || c.gdn_head != 128 || c.conv_k != 4 || c.n_expert > 1024 || c.n_used > 16 || c.n_embd % 256 ||
        c.n_head_kv <= 0 || c.n_head % c.n_head_kv || c.n_head / c.n_head_kv > grp_ok || c.rope_dim > c.hd || n_ctx < 1 ||
        n_ctx > 262144 || c.n_v_head % c.n_k_head || c.n_embd > 16384 || (!dense && !q4e && c.n_embd != 2048) || c.n_vocab <= 0 ||
        c.dt_rank > 256 || (dense && (c.n_ff <= 0 || c.n_ff % 256 || c.n_expert != 0 || (c.n_v_head * c.gdn_head) % 256)) ||
        (q4e && (c.n_v_head * c.gdn_head != geo.ssm_inner || c.n_ff_exp < 32 || c.n_ff_shexp < 32 || c.n_ff_exp % 8 || c.n_ff_shexp % 8))) {
        if (err)
            *err = "gpu: geometry outside what the kernels are written for (hd 256, gdn 128, conv 4, at most 12 query heads per KV head, "
                   "n_embd, n_ff and the delta-net width multiples of 256 or a whole number of blocks, n_ctx <= 262144)";
        return nullptr;
    }
    if (q4e) {
        // The hyper-connection stack and the PLE hash block (step.cpp's hc / ple recipes).
        I.q.hc = geo.hc_count;
        I.q.rank = geo.hc_rank;
        I.q.np = geo.n_embd_per_layer;
        I.q.ngram = geo.ple_ngram;
        I.q.kern = geo.ple_conv_kernel;
        I.q.n_heads = I.q.ngram > 0 ? (I.q.ngram - 1) * geo.ple_heads_per_ngram : 0;
        I.q.eos = geo.ple_eos;
        I.q.mult = geo.ple_multipliers;
        I.q.off = geo.ple_head_offsets;
        I.q.vocab = geo.ple_head_vocab;
        I.q.idx_top_k = geo.indexer_top_k;
        I.q.idx_heads = geo.indexer_heads;
        if (geo.has_indexer && (geo.indexer_dim != 128 || geo.indexer_heads < 1 || geo.indexer_heads > 4)) {
            if (err) *err = "gpu: qwen4exp's QSA kernels take an indexer of 1..4 heads of 128";
            return nullptr;
        }
        for (int ratio : geo.compress_ratios)
            if (ratio > 0) I.q.idx_kpool = ratio;
        if (I.q.hc < 2 || I.q.rank < 1 || I.q.np < 1 || I.q.n_heads < 1 || I.q.kern < 1 || I.q.ngram < 2 ||
            int(I.q.mult.size()) < I.q.ngram || int(I.q.off.size()) < I.q.n_heads || int(I.q.vocab.size()) < I.q.n_heads) {
            if (err) *err = "gpu: qwen4exp without its hyper-connection / PLE metadata";
            return nullptr;
        }
    }

    // The main stream has the higher priority: when the prefill's second stream shares the card, the delta net's
    // scan chain (few, latency-bound blocks) is scheduled ahead of the big product's next blocks.
    int prio_lo = 0, prio_hi = 0;
    cudaDeviceGetStreamPriorityRange(&prio_lo, &prio_hi);
    CU_TRY(cudaStreamCreateWithPriority(&I.stream, cudaStreamDefault, prio_hi), "cudaStreamCreate");
    CU_TRY(cudaStreamCreateWithPriority(&I.stream2, cudaStreamDefault, prio_lo), "cudaStreamCreate");
    // fragment-kernel decode products (gemm.cu); not qwen4exp: its many small products made a step 47 -> 57 ms
    if (!I.q4e) {
        std::string ferr;
        if (!frag_reserve(I.stream, &ferr) || !frag_reserve(I.stream2, &ferr)) {
            if (err) *err = "gpu: " + ferr;
            return nullptr;
        }
    }
    CU_TRY(cudaEventCreateWithFlags(&I.ev_a, cudaEventDisableTiming), "cudaEventCreate");
    CU_TRY(cudaEventCreateWithFlags(&I.ev_b, cudaEventDisableTiming), "cudaEventCreate");

    // ---- weights: two passes over the same traversal, the first sizes the arena, the second uploads ----
    Uploader up;
    if (!up.init()) {
        if (err) *err = "gpu: cannot allocate the pinned staging buffers";
        return nullptr;
    }
    std::string missing;
    uint64_t used = 0;
    bool dry = true;
    bool upload_ok = true;
    auto put = [&](const gguf::Tensor* t, bool as_f32) -> W {
        W w;
        if (!t) return w;
        w.type = as_f32 ? static_cast<uint32_t>(Type::F32) : t->type;
        const uint64_t bytes = as_f32 ? uint64_t(t->n_elements()) * 4 : t->nbytes;
        const uint64_t off = used;
        used += align_up(bytes);
        if (dry) return w;
        w.p = I.arena + off;
        if (as_f32 && t->type != static_cast<uint32_t>(Type::F32)) {
            std::vector<float> tmp(static_cast<size_t>(t->n_elements()));
            if (!quants::dequant_row(t->type, t->data, tmp.data(), uint64_t(t->n_elements()))) upload_ok = false;
            else upload_ok = upload_ok && up.copy(I.arena + off, tmp.data(), bytes);
        } else {
            upload_ok = upload_ok && up.copy(I.arena + off, t->data, bytes);
        }
        return w;
    };
    auto need = [&](int layer, const char* role) -> const gguf::Tensor* {
        const gguf::Tensor* t = m.tensor(layer, role);
        if (!t && missing.empty()) missing = (layer >= 0 ? "blk." + std::to_string(layer) + "." : std::string()) + role;
        return t;
    };

    // The token embedding: its rows are read one at a time, so by default it stays in mapped host memory.
    auto embd_load = [&]() {
        if (I.embd_in_vram) {
            I.embd = put(need(-1, "token_embd.weight"), false);
        } else if (const gguf::Tensor* te = need(-1, "token_embd.weight")) {
            I.embd.type = te->type;
            if (!dry) {
                void* dp = nullptr;
                if (cudaHostAlloc(&I.embd_host, te->nbytes, cudaHostAllocMapped) != cudaSuccess) {
                    cudaGetLastError();
                    I.embd_host = nullptr;
                    upload_ok = false;
                } else {
                    std::memcpy(I.embd_host, te->data, te->nbytes);
                    if (cudaHostGetDevicePointer(&dp, I.embd_host, 0) != cudaSuccess) upload_ok = false;
                    I.embd.p = static_cast<const uint8_t*>(dp);
                }
            }
        }
    };
    auto hc_mod = [&](int l, const std::string& base, Layer::Hc* out, bool inject = true) {
        out->norm = reinterpret_cast<const float*>(put(need(l, (base + "_norm.weight").c_str()), true).p);
        out->down = put(need(l, (base + "_down.weight").c_str()), false);
        out->up = put(need(l, (base + "_up.weight").c_str()), false);
        if (inject) out->inject = put(need(l, (base + "_inject.weight").c_str()), false);
    };
    auto traverse = [&]() {
        I.L.assign(static_cast<size_t>(c.n_layer_main), Layer());
        embd_load();
        I.out_norm = reinterpret_cast<const float*>(put(need(-1, "output_norm.weight"), true).p);
        // no output matrix (tied embeddings): the head is the token embedding, on the device in its own copy
        I.head = put(m.tensor(-1, "output.weight") ? need(-1, "output.weight") : need(-1, "token_embd.weight"), false);
        for (int l = 0; l < c.n_layer_main; ++l) {
            Layer& y = I.L[static_cast<size_t>(l)];
            y.full = geo.is_full_attention(l);
            y.attn_norm = reinterpret_cast<const float*>(put(need(l, "attn_norm.weight"), true).p);
            y.post_norm = reinterpret_cast<const float*>(put(need(l, "post_attention_norm.weight"), true).p);
            if (y.full) {
                y.q = put(need(l, "attn_q.weight"), false);
                y.k = put(need(l, "attn_k.weight"), false);
                y.v = put(need(l, "attn_v.weight"), false);
                y.o = put(need(l, "attn_output.weight"), false);
                y.qn = reinterpret_cast<const float*>(put(need(l, "attn_q_norm.weight"), true).p);
                y.kn = reinterpret_cast<const float*>(put(need(l, "attn_k_norm.weight"), true).p);
            } else {
                y.qkv = put(need(l, "attn_qkv.weight"), false);
                y.gate = put(need(l, "attn_gate.weight"), false);
                y.out = put(need(l, "ssm_out.weight"), false);
                y.alpha = put(need(l, "ssm_alpha.weight"), dense);  // the 27B stores them BF16: F32 on the device, as rung 1's file has them
                y.beta = put(need(l, "ssm_beta.weight"), dense);
                y.conv = reinterpret_cast<const float*>(put(need(l, "ssm_conv1d.weight"), true).p);
                y.dtb = reinterpret_cast<const float*>(put(need(l, "ssm_dt.bias"), true).p);
                y.ssm_a = reinterpret_cast<const float*>(put(need(l, "ssm_a"), true).p);
                y.ssm_norm = reinterpret_cast<const float*>(put(need(l, "ssm_norm.weight"), true).p);
            }
            if (dense) {
                y.fgate = put(need(l, "ffn_gate.weight"), false);
                y.fup = put(need(l, "ffn_up.weight"), false);
                y.fdown = put(need(l, "ffn_down.weight"), false);
                continue;
            }
            y.router = put(need(l, "ffn_gate_inp.weight"), false);
            y.ge = put(need(l, "ffn_gate_exps.weight"), false);
            y.ue = put(need(l, "ffn_up_exps.weight"), false);
            y.de = put(need(l, "ffn_down_exps.weight"), false);
            y.sgate = put(need(l, "ffn_gate_shexp.weight"), false);
            y.sup = put(need(l, "ffn_up_shexp.weight"), false);
            y.sdown = put(need(l, "ffn_down_shexp.weight"), false);
            y.shgate = reinterpret_cast<const float*>(put(need(l, "ffn_gate_inp_shexp.weight"), true).p);
        }
    };

    // qwen4exp: the hyper-connection mixers and the GDN / sparse-attention mixers of every block, the PLE
    // block's dense projections, and the MoE with its routed stacks left in the file's own mapping (the CPU
    // lane computes them: 23.4 GiB of experts do not fit beside the dense weights).
    auto traverse_q4e = [&]() {
        I.L.assign(static_cast<size_t>(c.n_layer_main), Layer());
        embd_load();
        I.head = put(need(-1, "output.weight"), false);
        hc_mod(-1, "output_hc", &I.q.head_hc, false);
        if (const gguf::Tensor* te = need(-1, "per_layer_token_embd.weight")) {
            if (!dry) {
                I.q.table = te->data;
                I.q.table_type = te->type;
                I.q.row_bytes = type_bytes(te->type, uint64_t(I.q.np));
                // rows are read at hashed, scattered offsets: fault-around readahead only evicts what is cached
                const uintptr_t lo = reinterpret_cast<uintptr_t>(te->data) & ~uintptr_t(4095);
                const uintptr_t hi = (reinterpret_cast<uintptr_t>(te->data) + te->nbytes + 4095) & ~uintptr_t(4095);
                (void)::madvise(reinterpret_cast<void*>(lo), size_t(hi - lo), MADV_RANDOM);
                uint64_t mb = 512;
                if (const char* e = std::getenv("LIMBO_Q4E_PLE_CACHE_MB")) mb = uint64_t(std::atoll(e));
                uint64_t n = 1;
                while (n * 2 * sizeof(Impl::Q4e::PleSlot) <= (mb << 20)) n *= 2;
                if (mb > 0 && n >= 1024) {
                    I.q.ple_cache.reset(new Impl::Q4e::PleSlot[n]);
                    I.q.ple_mask = n - 1;
                }
            }
        }
        for (int l = 0; l < c.n_layer_main; ++l) {
            Layer& y = I.L[static_cast<size_t>(l)];
            y.full = geo.is_full_attention(l);
            y.gdn_sig = true;
            hc_mod(l, "hc_attn", &y.hca);
            hc_mod(l, "hc_ffn", &y.hcf);
            if (y.full) {
                y.q = put(need(l, "attn_q.weight"), false);
                y.k = put(need(l, "attn_k.weight"), false);
                y.v = put(m.tensor(l, "attn_v.weight") ? need(l, "attn_v.weight") : nullptr, false);  // v may be the raw k projection
                y.o = put(need(l, "attn_output.weight"), false);
                y.qn = reinterpret_cast<const float*>(put(need(l, "attn_q_norm.weight"), true).p);
                y.kn = reinterpret_cast<const float*>(put(need(l, "attn_k_norm.weight"), true).p);
                y.iq = put(need(l, "indexer.q_proj.weight"), false);
                y.ik = put(need(l, "indexer.k_proj.weight"), false);
                y.iqn = reinterpret_cast<const float*>(put(need(l, "indexer.q_norm.weight"), true).p);
                y.ikn = reinterpret_cast<const float*>(put(need(l, "indexer.k_norm.weight"), true).p);
            } else {
                y.qkv = put(need(l, "attn_qkv.weight"), false);
                y.gate = put(need(l, "attn_gate.weight"), false);
                y.out = put(need(l, "ssm_out.weight"), false);
                y.alpha = put(need(l, "ssm_alpha.weight"), false);
                y.beta = put(need(l, "ssm_beta.weight"), false);
                y.conv = reinterpret_cast<const float*>(put(need(l, "ssm_conv1d.weight"), true).p);
                y.dtb = reinterpret_cast<const float*>(put(need(l, "ssm_dt.bias"), true).p);
                y.ssm_a = reinterpret_cast<const float*>(put(need(l, "ssm_a"), true).p);
                y.ssm_norm = reinterpret_cast<const float*>(put(need(l, "ssm_norm.weight"), true).p);
            }
            if (m.tensor(l, "ple_key.weight")) {
                y.ple_key = put(need(l, "ple_key.weight"), false);
                y.ple_value = put(need(l, "ple_value.weight"), false);
                y.ple_gk = reinterpret_cast<const float*>(put(need(l, "ple_norm_key.weight"), true).p);
                y.ple_gq = reinterpret_cast<const float*>(put(need(l, "ple_norm_query.weight"), true).p);
                y.ple_gc = reinterpret_cast<const float*>(put(need(l, "ple_norm_conv.weight"), true).p);
                y.ple_conv = reinterpret_cast<const float*>(put(need(l, "ple_conv1d.weight"), true).p);
            }
            y.router = put(need(l, "ffn_gate_inp.weight"), false);
            y.sgate = put(need(l, "ffn_gate_shexp.weight"), false);
            y.sup = put(need(l, "ffn_up_shexp.weight"), false);
            y.sdown = put(need(l, "ffn_down_shexp.weight"), false);
            y.shgate = reinterpret_cast<const float*>(put(need(l, "ffn_gate_inp_shexp.weight"), true).p);
            // the routed stacks: host pointers for the CPU lane, one expert's slices ge_bytes / de_bytes apart
            const gguf::Tensor* ge = need(l, "ffn_gate_exps.weight");
            const gguf::Tensor* ue = need(l, "ffn_up_exps.weight");
            const gguf::Tensor* de = need(l, "ffn_down_exps.weight");
            if (ge && ue && de && !dry) {
                y.geh = {ge->data, ge->type};
                y.ueh = {ue->data, ue->type};
                y.deh = {de->data, de->type};
                y.ge_bytes = type_bytes(ge->type, uint64_t(c.n_embd)) * uint64_t(c.n_ff_exp);
                y.de_bytes = type_bytes(de->type, uint64_t(c.n_ff_exp)) * uint64_t(c.n_embd);
            }
        }
    };

    if (const char* e = std::getenv("LIMBO_EMBD_VRAM")) I.embd_in_vram = *e == '1';
    if (q4e) traverse_q4e(); else traverse();
    if (!missing.empty()) {
        if (err) *err = "gpu: the model has no tensor " + missing;
        return nullptr;
    }
    I.arena_bytes = used;
    I.weight_bytes = used;
    // min_free_bytes 0: what this model needs, the weights and 1 GiB for the cache and the scratch
    if (min_free_bytes == 0 && free_b < used + (1ull << 30)) {
        if (err)
            *err = "only " + std::to_string(double(free_b) / 1e9) + " GB of VRAM free, the weights need " + std::to_string(double(used) / 1e9) +
                   " GB and the cache 1 GiB more: the desktop or another engine holds the rest";
        return nullptr;
    }
    const auto t0 = std::chrono::steady_clock::now();
    // 256 bytes of slack: the batched MoE kernel copies whole 16-byte windows around a block, which can run up to 15
    // bytes past the end of the last tensor
    CU_TRY(cudaMalloc(reinterpret_cast<void**>(&I.arena), I.arena_bytes + 256), "cudaMalloc(weights)");
    used = 0;
    dry = false;
    if (q4e) traverse_q4e(); else traverse();
    if (!upload_ok) {
        if (err) *err = "gpu: a weight upload failed";
        return nullptr;
    }
    CU_TRY(cudaStreamSynchronize(up.s), "weight upload");
    I.load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    // ---- what the kernels will index, checked against the tensors ------------------------------------
    for (int l = 0; l < c.n_layer_main; ++l) {
        Layer& y = I.L[static_cast<size_t>(l)];
        if (dense) {
            const W* gemvs[] = {&y.q, &y.k, &y.v, &y.o, &y.qkv, &y.gate, &y.out, &y.alpha, &y.beta, &y.fgate, &y.fup, &y.fdown};
            bool ok = true;
            for (const W* w : gemvs)
                if (w->p && !gemv_supported(w->type)) ok = false;
            if (!ok) {
                if (err) *err = "gpu: layer " + std::to_string(l) + " has a tensor type the GEMV kernels do not handle";
                return nullptr;
            }
            continue;
        }
        if (q4e) {
            const W* gemvs[] = {&y.q,   &y.k,     &y.v,      &y.o,      &y.qkv,   &y.gate,  &y.out,   &y.alpha,
                                &y.beta, &y.router, &y.sgate, &y.sup,   &y.sdown, &y.iq,   &y.ik,    &y.ple_key,
                                &y.ple_value, &y.hca.down, &y.hca.up, &y.hca.inject, &y.hcf.down, &y.hcf.up, &y.hcf.inject};
            bool ok = true;
            for (const W* w : gemvs)
                if (w->p && !gemv_supported(w->type)) ok = false;
            const gguf::Tensor* ge = m.tensor(l, "ffn_gate_exps.weight");
            const gguf::Tensor* ue = m.tensor(l, "ffn_up_exps.weight");
            const gguf::Tensor* de = m.tensor(l, "ffn_down_exps.weight");
            if (!ge || !ue || !de || !y.geh.p || y.ueh.type != y.geh.type ||
                !cpu::lane_supports(y.geh.type) || !cpu::lane_supports(y.deh.type) || y.ge_bytes == 0 || y.de_bytes == 0 ||
                ge->nbytes != y.ge_bytes * uint64_t(c.n_expert) || de->nbytes != y.de_bytes * uint64_t(c.n_expert))
                ok = false;
            if (!ok) {
                if (err) *err = "gpu: layer " + std::to_string(l) + " has a tensor type or shape the qwen4exp kernels or the CPU lane do not handle";
                return nullptr;
            }
            continue;
        }
        const gguf::Tensor* ge = m.tensor(l, "ffn_gate_exps.weight");
        const gguf::Tensor* ue = m.tensor(l, "ffn_up_exps.weight");
        const gguf::Tensor* de = m.tensor(l, "ffn_down_exps.weight");
        y.ge_bytes = type_bytes(ge->type, uint64_t(c.n_embd)) * uint64_t(c.n_ff_exp);
        y.de_bytes = type_bytes(de->type, uint64_t(c.n_ff_exp)) * uint64_t(c.n_embd);
        bool ok = ge->nbytes == y.ge_bytes * uint64_t(c.n_expert) && ue->type == ge->type && ue->nbytes == ge->nbytes &&
                  de->nbytes == y.de_bytes * uint64_t(c.n_expert);
        const W* gemvs[] = {&y.q, &y.k, &y.v, &y.o, &y.qkv, &y.gate, &y.out, &y.alpha, &y.beta, &y.router,
                            &y.ge, &y.ue, &y.de, &y.sgate, &y.sup, &y.sdown};
        for (const W* w : gemvs)
            if (w->p && !gemv_supported(w->type)) ok = false;
        if (!ok) {
            if (err) *err = "gpu: layer " + std::to_string(l) + " has a tensor type or shape the PoC kernels do not handle";
            return nullptr;
        }
    }
    if (!gemv_supported(I.embd.type) || !gemv_supported(I.head.type) || (!dense && !q4e && (c.n_ff_exp % 256 || c.n_ff_shexp % 256))) {
        if (err) *err = "gpu: embedding or head type not supported";
        return nullptr;
    }

    // The batched prefill needs the int8 path for every weight matrix and fp32 for the router and the delta net's
    // alpha/beta; otherwise prefill() runs token by token.
    {
        bool able = dense ? (c.n_embd % 256 == 0 && c.n_ff % 256 == 0 && c.n_vocab > 0 && (c.n_head * c.hd) % 128 == 0 &&
                             (c.n_v_head * c.gdn_head) % 128 == 0)
                          : c.n_embd % 128 == 0 && c.n_ff_exp % 128 == 0 && c.n_ff_shexp % 128 == 0 && c.n_expert <= 1024 &&
                    c.n_used == 8 && c.n_vocab > 0 && (c.n_head * c.hd) % 128 == 0 && (c.n_v_head * c.gdn_head) % 128 == 0 &&
                    c.n_ff_exp % 64 == 0;
        const uint32_t f32 = static_cast<uint32_t>(Type::F32);
        for (int l = 0; l < c.n_layer_main && able; ++l) {
            const Layer& y = I.L[static_cast<size_t>(l)];
            const W* dw[] = {&y.q, &y.k, &y.v, &y.o, &y.qkv, &y.gate, &y.out, &y.sgate, &y.sup, &y.sdown, &y.fgate, &y.fup, &y.fdown};
            for (const W* w : dw)
                if (w->p && !gemm_supported(w->type)) able = false;
            if (!dense && (!gemm_supported(y.ge.type) || !gemm_supported(y.de.type) || y.ge.type != y.ue.type)) able = false;
            if (!dense && y.router.type != f32) able = false;
            if (!y.full && (y.alpha.type != f32 || y.beta.type != f32)) able = false;
        }
        I.pf_able = able && !q4e;  // qwen4exp: per-step prefill only (the batched int8 path has no HC / PLE recipe yet)
    }

    // ---- state and scratch ---------------------------------------------------------------------------
    const size_t S_n = size_t(c.n_v_head) * c.gdn_head * c.gdn_head;
    const size_t hist_n = size_t(c.conv_k - 1) * c.qkv_w;
    I.S.assign(size_t(c.n_layer_main), nullptr);
    I.hist.assign(size_t(c.n_layer_main), nullptr);
    const int pages = pool_pages > 0 ? pool_pages : I.n_chunks();
    if (!I.alloc_kv(kv_fmt, pages)) {
        if (err)
            *err = std::string("gpu: cannot allocate the ") + kv_format_name(kv_fmt) + " K/V pool (" + std::to_string(pages) + " pages of 256 positions, " +
                   std::to_string(I.page_bytes(kv_fmt) * uint64_t(pages) >> 20) + " MiB)";
        return nullptr;
    }
    for (int l = 0; l < c.n_layer_main; ++l) {
        const size_t li = size_t(l);
        if (!I.L[li].full) {
            I.S[li] = I.dalloc<float>(S_n);
            I.hist[li] = I.dalloc<float>(hist_n);
            if (!I.S[li] || !I.hist[li]) {
                if (err) *err = "gpu: cannot allocate the GDN state";
                return nullptr;
            }
        }
    }
    const size_t E = size_t(c.n_embd);
    const size_t qkv_w = size_t(c.qkv_w), vn = size_t(c.n_v_head) * c.gdn_head;
    const size_t qw = size_t(c.n_head) * c.hd, kvw = size_t(c.n_head_kv) * c.hd;
    const size_t ffe = std::max(size_t(c.n_used) * c.n_ff_exp, size_t(c.n_ff));
    struct Carve {
        float** dst;
        size_t n;
    };
    const Carve carve[] = {
        {&I.x, E},          {&I.xn, E},          {&I.qkv, qkv_w},      {&I.conv, qkv_w},    {&I.qk, 2 * size_t(c.n_k_head) * c.gdn_head},
        {&I.alpha, size_t(std::max(64, c.dt_rank))}, {&I.beta, size_t(std::max(64, c.dt_rank))},       {&I.o, vn},           {&I.z, vn},          {&I.gated, vn},
        {&I.qfull, 2 * qw}, {&I.kbuf, kvw},      {&I.vbuf, kvw},       {&I.qbuf, qw},       {&I.attn, qw},
        {&I.rlog, size_t(c.n_expert)}, {&I.post, E}, {&I.gbuf, ffe},   {&I.ubuf, ffe},      {&I.hbuf, ffe},
        {&I.dn, size_t(c.n_used) * E}, {&I.gs, size_t(c.n_ff_shexp)}, {&I.us, size_t(c.n_ff_shexp)},
        {&I.hs, size_t(c.n_ff_shexp)}, {&I.shared, E}, {&I.sg, 32},    {&I.wts, 32},        {&I.rn, E},
        {&I.logits, size_t(c.n_vocab)},
    };
    size_t total = 0;
    for (const Carve& cv : carve) total += (cv.n + 63) / 64 * 64;
    CU_TRY(cudaMalloc(reinterpret_cast<void**>(&I.scratch), total * sizeof(float)), "cudaMalloc(scratch)");
    size_t at = 0;
    for (const Carve& cv : carve) {
        *cv.dst = I.scratch + at;
        at += (cv.n + 63) / 64 * 64;
    }
    I.ids = I.dalloc<int32_t>(32);
    I.tok = I.dalloc<int32_t>(4);
    I.d_args = I.dalloc<int32_t>(4);
    if (!I.ids || !I.tok || !I.d_args ||
        cudaMallocHost(reinterpret_cast<void**>(&I.pin), sizeof(int32_t) * 2 * Impl::kRing) != cudaSuccess) {
        if (err) *err = "gpu: cannot allocate the id buffers";
        return nullptr;
    }
    for (cudaEvent_t& e : I.pin_ev)
        if (cudaEventCreateWithFlags(&e, cudaEventDisableTiming) != cudaSuccess) {
            if (err) *err = "gpu: cannot create the staging events";
            return nullptr;
        }
    if (cudaMallocHost(reinterpret_cast<void**>(&I.pt_pin), sizeof(int32_t) * size_t(Impl::kPtRing) * size_t(I.n_chunks() + 1)) != cudaSuccess) {
        if (err) *err = "gpu: cannot allocate the page-table staging buffer";
        return nullptr;
    }
    for (cudaEvent_t& e : I.pt_ev)
        if (cudaEventCreateWithFlags(&e, cudaEventDisableTiming) != cudaSuccess) {
            if (err) *err = "gpu: cannot create the page-table events";
            return nullptr;
        }
    if (!I.state_ready()) {
        if (err) *err = "gpu: cannot allocate the page table";
        return nullptr;
    }
    if (q4e) {
        // The residual stack and the PLE chain's scratch, the CPU lane and its pinned host buffers.
        const size_t hc = size_t(I.q.hc), e = size_t(c.n_embd), hcd = hc * e;
        const size_t phist_n = size_t(I.q.kern - 1) * size_t(I.q.ngram) * hcd;
        struct QCarve {
            float** dst;
            size_t n;
        };
        const QCarve qcarve[] = {
            {&I.q.st, hcd},      {&I.q.xn, hcd},      {&I.q.lo, size_t(I.q.rank)}, {&I.q.gate, hcd},
            {&I.q.mixed, e},     {&I.q.inj, hc},      {&I.q.out, e},               {&I.q.emb, size_t(I.q.n_heads) * size_t(I.q.np)},
            {&I.q.pkey, hcd},    {&I.q.pval, e},      {&I.q.pgated, hcd},          {&I.q.pnrm, hcd},
            {&I.q.pconv, hcd},   {&I.q.phist, phist_n},
        };
        size_t qtot = 0;
        for (const QCarve& cv : qcarve) qtot += (cv.n + 63) / 64 * 64;
        if (cudaMalloc(reinterpret_cast<void**>(&I.q.scratch), qtot * sizeof(float)) != cudaSuccess) {
            if (err) *err = "gpu: cannot allocate the qwen4exp scratch";
            return nullptr;
        }
        size_t qat = 0;
        for (const QCarve& cv : qcarve) {
            *cv.dst = I.q.scratch + qat;
            qat += (cv.n + 63) / 64 * 64;
        }
        if (cudaHostAlloc(reinterpret_cast<void**>(&I.q.host_x), sizeof(float) * e, cudaHostAllocMapped) != cudaSuccess ||
            cudaHostAlloc(reinterpret_cast<void**>(&I.q.host_dn), sizeof(float) * size_t(c.n_used) * e, cudaHostAllocMapped) != cudaSuccess ||
            cudaHostGetDevicePointer(reinterpret_cast<void**>(&I.q.host_x_d), I.q.host_x, 0) != cudaSuccess ||
            cudaHostGetDevicePointer(reinterpret_cast<void**>(&I.q.host_dn_d), I.q.host_dn, 0) != cudaSuccess) {
            if (err) *err = "gpu: cannot allocate the CPU lane's pinned buffers";
            return nullptr;
        }
        cpu::Lane::Config lcfg;
        // The verify window's MoE submits one job per (row, used) pair -- up to kQ4eRows * n_used, which is 64 at
        // T = 8 -- where a single token submits n_used. The default 64 is exactly the window's worst case, so it
        // is raised rather than left as a trap for a wider window.
        lcfg.max_jobs = 256;
        // The plan's two-worker lane (docs/flash-next.md 6.4) was a roofline assumption for the state where
        // 91 % of the experts are resident, not a law: measured, the lane scales 2.87x from 2 to 6 workers
        // (361 -> 126 ms of exposed lane per token on the `hi` capture), so it is per-core-bound and every
        // physical core it may have is worth taking. default_cpus reserves the GPU feeder and the OS/desktop
        // cores itself. LIMBO_Q4E_WORKERS overrides it for the bench.
        int lane_workers = 0;
        if (const char* w = std::getenv("LIMBO_Q4E_WORKERS")) lane_workers = std::atoi(w);
        // every physical core but the GPU feeder's (the step thread pins itself there): the lane is per-core-bound
        lcfg.pool.cpus = cpu::default_cpus(lane_workers, std::getenv("LIMBO_Q4E_RESERVE_OS") != nullptr);
        // The gap between two layers' lane batches is the GPU's dense half of a layer (~0.5 ms): a worker that parks
        // in it pays a futex wake and cold caches 48 times a token. Spin through it (an idle server parks after 3 ms).
        lcfg.pool.spin_us = 3000;
        if (const char* sp = std::getenv("LIMBO_Q4E_SPIN_US")) lcfg.pool.spin_us = std::atoi(sp);
        // 3 planes is fp32-like (cpu_lane.hpp); 1 is llama.cpp's own Q8 numerics. 1 by default: the lane is
        // arithmetic-bound and 1 plane is 185 against 240 us an expert (6 workers, IQ2_S + IQ4_NL), and the gate
        // holds (code 16/16, hi 14/16 against 12/16 at 3).
        lcfg.planes = 1;
        if (const char* p = std::getenv("LIMBO_Q4E_PLANES")) lcfg.planes = std::atoi(p);
        // The lane's task granularity. One stage-1 task is `slice_a` rows of gate (and, separately, of up) and one
        // stage-2 task is `slice_b` rows of down, so a job costs n_ff/slice_a * 2 + n_embd/slice_b tasks -- 88 + 128
        // at the defaults on this model. Every task pays the claim, the activation-column gather and the partial
        // write, so the count is overhead per unit of weight read; the ceiling is the scratch's 64 rows.
        //
        // Measured, and the defaults are kept: at 64/64 (the ceiling) the `code` capture gave 9.2 tok/s against
        // 8.6 to 12.4 for 16/32 on the same box in the same hour. Fewer, larger tasks are not the win that the
        // per-task overhead suggests -- a 16-row stage-1 task is already 16 x K weights, so it is streaming-bound
        // and not claim-bound -- and 64 rows is a narrower tail to balance over 6 workers. The knobs are here
        // because they are the instrument, not because there is a better setting to find by guessing.
        if (const char* a = std::getenv("LIMBO_Q4E_SLICE_A")) lcfg.slice_a = std::atoi(a);
        if (const char* b = std::getenv("LIMBO_Q4E_SLICE_B")) lcfg.slice_b = std::atoi(b);
        I.q.lane = cpu::Lane::create(lcfg, err);
        if (!I.q.lane) return nullptr;
        I.q.tok_hist.assign(size_t(n_ctx), -1);
        // The verify window's workspace: one buffer per row-bearing tensor, `kQ4eRows` rows deep, row t
        // contiguous. Its own carve rather than the single-row scratch, so a window cannot move a number a
        // single token produced. The PLE conv's history is one row's worth shared across the window's rows, and
        // the logits are the one big item (kQ4eRows x n_vocab), because a window decides acceptance from all T.
        {
            const size_t E = size_t(c.n_embd), hc = size_t(I.q.hc), hcd = hc * E, R = size_t(Impl::kQ4eRows);
            const size_t phist_n = size_t(I.q.kern - 1) * size_t(I.q.ngram) * hcd;
            const size_t ffe = std::max(size_t(c.n_used) * size_t(c.n_ff_exp), size_t(c.n_ff));
            struct RW {
                float** dst;
                size_t per_row;
                bool rows;
            };
            const RW carve[] = {
                {&I.qrw.st, hcd, true},       {&I.qrw.xn, hcd, true},       {&I.qrw.lo, size_t(I.q.rank), true},
                {&I.qrw.gate, hcd, true},     {&I.qrw.mixed, E, true},      {&I.qrw.inj, hc, true},
                {&I.qrw.out, E, true},        {&I.qrw.emb, size_t(I.q.n_heads) * size_t(I.q.np), true},
                {&I.qrw.x, E, true},
                {&I.qrw.pkey, hcd, true},     {&I.qrw.pval, E, true},       {&I.qrw.pgated, hcd, true},
                {&I.qrw.pnrm, hcd, true},     {&I.qrw.pconv, hcd, true},    {&I.qrw.phist, phist_n, false},
                {&I.qrw.qkv, size_t(c.qkv_w), true},   {&I.qrw.conv, size_t(c.qkv_w), true},
                {&I.qrw.qk, 2 * size_t(c.n_k_head) * size_t(c.gdn_head), true},
                {&I.qrw.alpha, size_t(c.dt_rank), true}, {&I.qrw.beta, size_t(c.dt_rank), true},
                {&I.qrw.o, size_t(c.n_v_head) * size_t(c.gdn_head), true},
                {&I.qrw.gated, size_t(c.n_v_head) * size_t(c.gdn_head), true},
                {&I.qrw.z, size_t(c.n_v_head) * size_t(c.gdn_head), true},
                {&I.qrw.attn, size_t(c.n_head) * size_t(c.hd), true},
                {&I.qrw.qfull, 2 * size_t(c.n_head) * size_t(c.hd), true},
                {&I.qrw.kbuf, size_t(c.n_head_kv) * size_t(c.hd), true},
                {&I.qrw.vbuf, size_t(c.n_head_kv) * size_t(c.hd), true},
                {&I.qrw.qbuf, size_t(c.n_head) * size_t(c.hd), true},
                {&I.qrw.dn, size_t(c.n_used) * E, true},  {&I.qrw.rlog, size_t(c.n_expert), true},
                {&I.qrw.wts, size_t(c.n_used), true},     {&I.qrw.gbuf, ffe, true},
                {&I.qrw.ubuf, ffe, true},     {&I.qrw.hbuf, ffe, true},
                {&I.qrw.gs, size_t(c.n_ff_shexp), true},  {&I.qrw.us, size_t(c.n_ff_shexp), true},
                {&I.qrw.hs, size_t(c.n_ff_shexp), true},  {&I.qrw.shared, E, true},
                {&I.qrw.sg, 32, false},       {&I.qrw.logits, size_t(c.n_vocab), true},
            };
            size_t total = 0;
            for (const RW& r : carve) total += (r.per_row * (r.rows ? R : 1) + 63) / 64 * 64;
            float* base = nullptr;
            if (cudaMalloc(reinterpret_cast<void**>(&base), total * sizeof(float)) != cudaSuccess) {
                if (err) *err = "gpu: cannot allocate the qwen4exp verify window's workspace";
                return nullptr;
            }
            I.owned.push_back(base);
            I.owned_bytes += total * sizeof(float);
            size_t at = 0;
            for (const RW& r : carve) {
                *r.dst = base + at;
                at += (r.per_row * (r.rows ? R : 1) + 63) / 64 * 64;
            }
            I.qrw.ids = I.dalloc<int32_t>(size_t(Impl::kQ4eRows) * size_t(c.n_used));
            if (!I.qrw.ids) {
                if (err) *err = "gpu: cannot allocate the verify window's expert ids";
                return nullptr;
            }
            I.qrw.ready = true;
        }
        // The resident expert cache (docs/flash-next.md 6.4): everything the card has left after the dense
        // weights, the state, the scratch and the lane's buffers, minus a reserve for the desktop and the
        // driver. 0 disables it and leaves every expert on the lane — the correctness floor the gate is run
        // against, and what LIMBO_Q4E_CACHE_GB=0 selects.
        // The prompt pass's workspace is made now, so the slab below is measured without it; its stage is lent from the
        // slab (q4e_lend). The card is filled to LIMBO_Q4E_HEADROOM_MB (512) of free memory, the desktop's and the
        // driver's: every other byte is either needed by decode or holds experts.
        if (!I.ensure_q4e_pf()) {
            if (err) *err = "gpu: cannot allocate the qwen4exp prompt pass's workspace";
            return nullptr;
        }
        size_t cache_free = 0, cache_total = 0;
        (void)cudaMemGetInfo(&cache_free, &cache_total);
        uint64_t headroom = 512ull << 20;   // + the server's --vram-reserve-mib (256): ~768 MiB stay free
        if (const char* h = std::getenv("LIMBO_Q4E_HEADROOM_MB")) headroom = uint64_t(std::atoll(h)) << 20;
        uint64_t vram_budget = cache_free > headroom ? uint64_t(cache_free) - headroom : 0;
        // what the backend makes after this: the K/V pool of n_ctx positions in the format it asked for (f16 or Q8), the
        // sparse indexer's pool and the split-decode partials, ~3 KiB a position; and the MTP block
        {
            const KvFormat pf_fmt = kv_fmt == KvFormat::Q8 ? KvFormat::Q8 : KvFormat::F16;
            const uint64_t ctx_b = uint64_t(n_ctx + 512) * (I.page_bytes(pf_fmt) / 256 + 3072);
            vram_budget = vram_budget > ctx_b ? vram_budget - ctx_b : 0;
            uint64_t pf = 0;
            // the MTP layer beside the slab (its tensors, its K/V and rows), when its file is there
            if (const std::string mp = q4e_mtp ? q4e_mtp_path(m) : std::string(); !mp.empty()) {
                struct stat st{};
                if (::stat(mp.c_str(), &st) == 0) pf += uint64_t(st.st_size) + (192ull << 20);   // the file + its K/V and partials at 32k cells (~158 MiB)
            }
            if (const char* m = std::getenv("LIMBO_Q4E_PF_MB")) pf = uint64_t(std::atoll(m)) << 20;
            vram_budget = vram_budget > pf ? vram_budget - pf : 0;
        }
        if (const char* g = std::getenv("LIMBO_Q4E_CACHE_GB")) vram_budget = uint64_t(std::atof(g) * 1e9);
        // The RAM tier: by default the complement of the VRAM slab, as far as the box can pin it (q4e_cache_open).
        uint64_t ram_budget = UINT64_MAX;
        if (const char* g = std::getenv("LIMBO_Q4E_RAM_GB")) ram_budget = uint64_t(std::atof(g) * 1e9);
        for (const auto& f : m.shards()) {
            Impl::Q4e::Shard sh;
            sh.base = f->base();
            sh.size = f->size();
            sh.fd = ::open(f->path().c_str(), O_RDONLY | O_CLOEXEC);
            I.q.shards.push_back(sh);
        }
        if (I.q.table) (void)I.q.file_of(I.q.table, &I.q.table_fd, &I.q.table_off);
        if (const char* pp = std::getenv("LIMBO_Q4E_PROFILE")) {
            I.q.profile_path = pp;   // "" disables
        } else if (!m.shards().empty()) {
            std::string dir;
            if (const char* x = std::getenv("XDG_CACHE_HOME"); x && *x) dir = std::string(x) + "/limbo";
            else if (const char* h = std::getenv("HOME")) dir = std::string(h) + "/.cache/limbo";
            if (!dir.empty()) {
                (void)::mkdir(dir.c_str(), 0755);
                std::string base = m.shards()[0]->path();
                if (const size_t sl = base.rfind('/'); sl != std::string::npos) base = base.substr(sl + 1);
                I.q.profile_path = dir + "/" + base + "." + std::to_string(m.shards()[0]->size()) + ".routing";
            }
        }
        if (!I.q4e_cache_open(vram_budget, ram_budget, err)) return nullptr;
    }
    g->reset();
    CU_TRY(cudaStreamSynchronize(I.stream), "init");
    // the weights are on the card (or in the RAM tier): the mapping's pages the load touched go back to the page cache
    if (q4e) m.drop_resident();

    size_t free_a = 0, total_a = 0;
    cudaMemGetInfo(&free_a, &total_a);
    I.free_after = free_a;
    // What this object allocated, exactly (the free-memory delta also moves with the desktop's own use, and
    // with the CUDA context, which nvidia-smi's per-process figure includes).
    I.vram_bytes = I.arena_bytes + I.owned_bytes + total * sizeof(float);
    return g;
}

void Gpu::reset() {
    Impl& I = *p_;
    const Cfg& c = I.c;
    slot_cut(0);
    const size_t S_n = size_t(c.n_v_head) * c.gdn_head * c.gdn_head * sizeof(float);
    const size_t hist_n = size_t(c.conv_k - 1) * c.qkv_w * sizeof(float);
    for (size_t l = 0; l < I.S.size(); ++l) {
        if (I.S[l]) cudaMemsetAsync(I.S[l], 0, S_n, I.stream);
        if (I.hist[l]) cudaMemsetAsync(I.hist[l], 0, hist_n, I.stream);
    }
    if (I.q4e && I.q.phist) {
        cudaMemsetAsync(I.q.phist, 0, size_t(I.q.kern - 1) * size_t(I.q.ngram) * size_t(I.q.hc) * size_t(c.n_embd) * sizeof(float), I.stream);
        std::fill(I.q.tok_hist.begin(), I.q.tok_hist.end(), int32_t(-1));
    }
    cudaStreamSynchronize(I.stream);
}

bool Gpu::reserve_scratch() {
    Impl& I = *p_;
    bool ok = I.ensure_dec_scratch();
    if (I.pf_able) ok = I.ensure_prefill(I.pf_chunk) && ok;
    return ok;
}

int32_t Gpu::argmax() {
    Impl& I = *p_;
    ops::argmax(I.logits, I.c.n_vocab, I.tok, I.stream);
    int32_t t = -1;
    cudaMemcpyAsync(&t, I.tok, sizeof t, cudaMemcpyDeviceToHost, I.stream);
    cudaStreamSynchronize(I.stream);
    return t;
}

bool Gpu::read_logits(float* out) {
    Impl& I = *p_;
    cudaStreamSynchronize(I.stream);
    return cudaMemcpy(out, I.logits, size_t(I.c.n_vocab) * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
}

}  // namespace limbo::cuda
