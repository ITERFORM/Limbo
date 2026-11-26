// Copyright (c) 2026 ITERFORM. All rights reserved.
// src/cuda/gpu_q4e_mtp.cu — qwen4exp's multi-token-prediction layer as the draft source of the verify window.
//
// The checkpoint ships one MTP layer (a separate GGUF here: mtp-q2_0.gguf, BF16 dense parts and Q2_0 experts). Interop
// facts of its wiring (vLLM's qwen4_exp MTP, as docs/donors/strata.md records them): cell i pairs the main model's
// final multi-stream residual R_i with the token at i + 1; its output predicts the token at i + 2 and its own
// residual feeds the next draft step.
//
//   e = fc_embedding(rms(embed(tok)) * (1 + w))        h = fc_hidden per stream (rms_10240(R) * (1 + w))
//   R' = h + e  ->  hc read  ->  attention (its own K/V, dense)  ->  hc write  ->  hc read  ->  MoE (512 experts,
//   top-10, shared expert)  ->  hc write  ->  final mixer (hc read)  ->  the model's head
//
// A draft need not be exact: the verify window decides every emitted token. The attention is dense over the cells
// the block has seen since the caller's base (identical to the sparse selection below 2,051 cells), and every expert
// sits in VRAM.
#include <cuda_fp16.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "dec_ops.hpp"
#include "gpu_impl.hpp"
#include "limbo/cuda/gemv.hpp"
#include "limbo/cuda/gpu.hpp"
#include "limbo/gguf.hpp"
#include "limbo/quants.hpp"
#include "limbo/types.hpp"
#include "ops.hpp"

namespace limbo::cuda {

using namespace impl;

namespace {

bool gemv_rows(uint32_t type, const void* W, const float* x, int x_stride, float* y, int y_stride, int n_out, int n_in,
               int T, cudaStream_t s) {
    if (n_in % 256 == 0 && T <= 8) return gemv_cols(type, W, x, x_stride, y, y_stride, n_out, n_in, T, false, s);
    for (int t = 0; t < T; ++t)
        if (!gemv(type, W, x + size_t(t) * x_stride, y + size_t(t) * y_stride, n_out, n_in, false, s)) return false;
    return true;
}

// st[t][s][j] = h[t][s][j] + e[t][j]
__global__ void add_e_k(float* st, const float* h, const float* e, int hc, int n, int T) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i >= hc * n || t >= T) return;
    st[size_t(t) * hc * n + i] = h[size_t(t) * hc * n + i] + e[size_t(t) * n + i % n];
}

// h[slot][i] = silu(gu[slot][i]) * gu[slot][nff + i]: the fused gate|up rows of an expert
// Per row: the argmax of n logits (ties: lowest index) and its softmax probability, 1 / sum exp(l - max).
__global__ void argmax_p_k(const float* logits, int n, int32_t* am, float* pr) {
    __shared__ float sm[1024];
    __shared__ int si[1024];
    const float* l = logits + size_t(blockIdx.x) * size_t(n);
    float m = -INFINITY;
    int mi = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        if (l[i] > m) {
            m = l[i];
            mi = i;
        }
    sm[threadIdx.x] = m;
    si[threadIdx.x] = mi;
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w) {
            const float o = sm[threadIdx.x + w];
            const int oi = si[threadIdx.x + w];
            if (o > sm[threadIdx.x] || (o == sm[threadIdx.x] && oi < si[threadIdx.x])) {
                sm[threadIdx.x] = o;
                si[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    const float mx = sm[0];
    const int best = si[0];
    __syncthreads();
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) acc += expf(l[i] - mx);
    sm[threadIdx.x] = acc;
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w) sm[threadIdx.x] += sm[threadIdx.x + w];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        am[blockIdx.x] = best;
        pr[blockIdx.x] = 1.0f / sm[0];
    }
}

__global__ void swiglu_split_k(const float* gu, float* h, int nff, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int slot = i / nff, j = i % nff;
    const float g = gu[size_t(slot) * 2 * nff + j], u = gu[size_t(slot) * 2 * nff + nff + j];
    h[i] = g / (1.0f + __expf(-g)) * u;
}

bool exists(const std::string& p) {
    struct stat st{};
    return ::stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

}  // namespace

// LIMBO_Q4E_MTP, else mtp-q2_0.gguf beside the model or in ../mtp, ../../mtp (the Strata-data layout); "" none.
std::string q4e_mtp_path(const model::Model& m) {
    if (const char* e = std::getenv("LIMBO_Q4E_MTP")) return std::string(e) == "0" ? std::string() : std::string(e);
    if (m.shards().empty()) return {};
    std::string dir = m.shards()[0]->path();
    const size_t sl = dir.rfind('/');
    dir = sl == std::string::npos ? std::string(".") : dir.substr(0, sl);
    for (const std::string& c : {dir + "/mtp-q2_0.gguf", dir + "/../mtp/mtp-q2_0.gguf", dir + "/../../mtp/mtp-q2_0.gguf"})
        if (exists(c)) return c;
    return {};
}

bool Gpu::Impl::q4e_mtp_load(const model::Model& m) {
    Q4eMtp& M = q4mtp;
    if (M.ready) return true;
    const std::string path = q4e_mtp_path(m);
    if (path.empty()) return false;
    std::string err;
    auto f = gguf::File::open(path, &err);
    if (!f) {
        std::fprintf(stderr, "q4e MTP: %s: %s\n", path.c_str(), err.c_str());
        return false;
    }
    const Cfg& cc = c;
    const int e = cc.n_embd, hc = q.hc, pitch = hc * e;
    bool ok = true;
    auto al = [&](size_t bytes) -> void* {
        void* p = nullptr;
        if (cudaMalloc(&p, bytes) != cudaSuccess) {
            (void)cudaGetLastError();
            ok = false;
            return nullptr;
        }
        M.allocs.push_back(p);
        return p;
    };
    auto tensor = [&](const std::string& name) -> const gguf::Tensor* {
        const gguf::Tensor* t = f->tensor(name);
        if (!t) {
            std::fprintf(stderr, "q4e MTP: %s has no %s\n", path.c_str(), name.c_str());
            ok = false;
        }
        return t;
    };
    auto put = [&](const std::string& name) -> W {
        W w;
        const gguf::Tensor* t = tensor(name);
        if (!t) return w;
        void* p = al(t->nbytes);
        if (p && cudaMemcpy(p, t->data, t->nbytes, cudaMemcpyHostToDevice) != cudaSuccess) ok = false;
        w.type = t->type;
        w.p = static_cast<const uint8_t*>(p);
        return w;
    };
    // an F32 vector on the device; shift: the zero-centred norms store w for (1 + w)
    auto vec = [&](const std::string& name, float shift) -> const float* {
        const gguf::Tensor* t = tensor(name);
        if (!t) return nullptr;
        std::vector<float> v(size_t(t->n_elements()));
        if (!quants::dequant_row(t->type, t->data, v.data(), uint64_t(v.size()))) {
            ok = false;
            return nullptr;
        }
        for (float& x : v) x += shift;
        float* p = static_cast<float*>(al(v.size() * sizeof(float)));
        if (p && cudaMemcpy(p, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) ok = false;
        return p;
    };
    // the hyper-connection norms are zero-centred too: (1 + w) gives 68 % of drafts kept, w alone 40 %
    const float hc_shift = std::getenv("LIMBO_MTP_HC_SHIFT") ? float(std::atof(std::getenv("LIMBO_MTP_HC_SHIFT"))) : 1.0f;
    const std::string L = "mtp.layers.0.";
    M.fc_e = put("mtp.fc_embedding.weight");
    M.fc_h = put("mtp.fc_hidden.weight");
    M.pre_e = vec("mtp.pre_fc_norm_embedding.weight", 1.0f);
    M.pre_h = vec("mtp.pre_fc_norm_hidden.weight", 1.0f);
    auto hcm = [&](const std::string& base, Layer::Hc* o, bool inject) {
        o->norm = vec(base + "hc_norm.weight", hc_shift);
        o->down = put(base + "input_mix_weight_down.weight");
        o->up = put(base + "input_mix_weight_up.weight");
        if (inject) o->inject = put(base + "block_inject_weight.weight");
    };
    hcm(L + "attn_hyper_connection.", &M.hca, true);
    hcm(L + "mlp_hyper_connection.", &M.hcf, true);
    hcm("mtp.hyper_connection_mixer.", &M.mix, false);
    M.q = put(L + "self_attn.q_proj.weight");
    M.k = put(L + "self_attn.k_proj.weight");
    M.v = put(L + "self_attn.v_proj.weight");
    M.o = put(L + "self_attn.o_proj.weight");
    M.qn = vec(L + "self_attn.q_norm.weight", 1.0f);
    M.kn = vec(L + "self_attn.k_norm.weight", 1.0f);
    M.router = put(L + "mlp.gate.weight");
    M.sh_gate = put(L + "mlp.shared_expert.gate_proj.weight");
    M.sh_up = put(L + "mlp.shared_expert.up_proj.weight");
    M.sh_down = put(L + "mlp.shared_expert.down_proj.weight");
    M.shgate = vec(L + "mlp.shared_expert_gate.weight", 0.0f);
    M.gu = put(L + "mlp.experts.gate_up_proj");
    M.dn = put(L + "mlp.experts.down_proj");
    if (!ok) return false;
    {
        const gguf::Tensor* gu = f->tensor(L + "mlp.experts.gate_up_proj");
        const gguf::Tensor* dn = f->tensor(L + "mlp.experts.down_proj");
        M.n_exp = int(gu->ne[2]);
        M.nff = int(gu->ne[1]) / 2;
        M.gu_bytes = gu->nbytes / uint64_t(M.n_exp);
        M.dn_bytes = dn->nbytes / uint64_t(M.n_exp);
        const gguf::Tensor* sg = f->tensor(L + "mlp.shared_expert.gate_proj.weight");
        M.nsh = int(sg->ne[1]);
    }
    M.n_used = cc.n_used;
    const int R = kQ4eRows, pairs = R * M.n_used;
    auto fa = [&](size_t n) { return static_cast<float*>(al(n * sizeof(float))); };
    M.cap = std::min(n_ctx, 32768);   // cells are the model's positions: the block's attention sees the prompt (q4e_mtp_front)
    M.kc = static_cast<__half*>(al(size_t(M.cap) * size_t(cc.n_head_kv) * size_t(cc.hd) * sizeof(__half)));
    M.vc = static_cast<__half*>(al(size_t(M.cap) * size_t(cc.n_head_kv) * size_t(cc.hd) * sizeof(__half)));
    M.hin = fa(size_t(R) * pitch);
    M.hn = fa(size_t(R) * pitch);
    M.h = fa(size_t(R) * pitch);
    M.st = fa(size_t(R) * pitch);
    M.emb = fa(size_t(R) * e);
    M.en = fa(size_t(R) * e);
    M.ev = fa(size_t(R) * e);
    const size_t qw = size_t(cc.n_head) * cc.hd, kvw = size_t(cc.n_head_kv) * cc.hd;
    M.qfull = fa(size_t(R) * 2 * qw);
    M.kbuf = fa(size_t(R) * kvw);
    M.vbuf = fa(size_t(R) * kvw);
    M.qbuf = fa(size_t(R) * qw);
    M.attn = fa(size_t(R) * qw);
    M.out = fa(size_t(R) * e);
    M.scr = fa(dec::attn_scratch_floats(M.cap, R, cc.n_head));
    M.rlog = fa(size_t(R) * M.n_exp);
    M.wts = fa(size_t(pairs));
    M.ids = static_cast<int32_t*>(al(size_t(pairs) * sizeof(int32_t)));
    M.gub = fa(size_t(pairs) * 2 * M.nff);
    M.hh = fa(size_t(pairs) * M.nff);
    M.dnb = fa(size_t(pairs) * e);
    M.gs = fa(size_t(R) * M.nsh);
    M.us = fa(size_t(R) * M.nsh);
    M.hs = fa(size_t(R) * M.nsh);
    M.shared = fa(size_t(R) * e);
    M.sg = fa(size_t(R));
    M.logits = fa(size_t(R) * cc.n_vocab);
    M.d_am = static_cast<int32_t*>(al(size_t(R) * sizeof(int32_t)));
    M.d_tok = static_cast<int32_t*>(al(size_t(R) * sizeof(int32_t)));
    M.d_pos = static_cast<int32_t*>(al(sizeof(int32_t)));
    M.d_pr = fa(size_t(R));
    if (!ok || !ensure_q4e_rows() || !qrw.d_pair) {
        std::fprintf(stderr, "q4e MTP: not loaded (%s)\n", !ok ? "a device allocation or copy failed" : "no window workspace");
        return false;
    }
    uint64_t bytes = 0;
    for (const auto& t : f->tensors()) bytes += t.nbytes;
    std::fprintf(stderr, "q4e MTP: %s, %d experts (type %u), %.0f MiB on the device, %d cells\n", path.c_str(), M.n_exp,
                 M.gu.type, double(bytes) / 1048576.0, M.cap);
    M.ready = true;
    return true;
}

bool Gpu::Impl::q4e_mtp_rows(const int32_t* tokens, int T, int q0, bool h_from_mtp) {
    Q4eMtp& M = q4mtp;
    const Cfg& cc = c;
    cudaStream_t s = stream;
    const int e = cc.n_embd, hc = q.hc, pitch = hc * e;
    if (!M.ready || T < 1 || T > kQ4eRows || q0 < 0 || q0 + T > M.cap || (h_from_mtp && T != 1)) { std::fprintf(stderr, "q4e MTP: failed at line %d (%s)\n", __LINE__, cudaGetErrorString(cudaPeekAtLastError())); return false; }
    // the hidden state: the main model's final stack of the last window's rows, or the block's own last output row
    if (h_from_mtp) {
        if (cudaMemcpyAsync(M.hin, M.st + size_t(M.last_T - 1) * pitch, sizeof(float) * pitch, cudaMemcpyDeviceToDevice, s) !=
            cudaSuccess)
            { std::fprintf(stderr, "q4e MTP: failed at line %d (%s)\n", __LINE__, cudaGetErrorString(cudaPeekAtLastError())); return false; }
    } else if (cudaMemcpyAsync(M.hin, qrw.st, sizeof(float) * size_t(T) * pitch, cudaMemcpyDeviceToDevice, s) != cudaSuccess) {
        { std::fprintf(stderr, "q4e MTP: failed at line %d (%s)\n", __LINE__, cudaGetErrorString(cudaPeekAtLastError())); return false; }
    }
    int32_t args[kQ4eRows];
    for (int t = 0; t < T; ++t) args[t] = tokens[t];
    if (cudaMemcpyAsync(M.d_tok, args, sizeof(int32_t) * size_t(T), cudaMemcpyHostToDevice, s) != cudaSuccess ||
        cudaMemcpyAsync(M.d_pos, &q0, sizeof(int32_t), cudaMemcpyHostToDevice, s) != cudaSuccess)
        { std::fprintf(stderr, "q4e MTP: failed at line %d (%s)\n", __LINE__, cudaGetErrorString(cudaPeekAtLastError())); return false; }
    const float eps = float(cc.eps), inv_hc = 1.0f / float(hc);
    // the two inputs
    if (!embed_rows(embd.type, embd.p, M.d_tok, T, M.emb, e, s)) { std::fprintf(stderr, "q4e MTP: failed at line %d (%s)\n", __LINE__, cudaGetErrorString(cudaPeekAtLastError())); return false; }
    ops::rmsnorm_rows(M.emb, M.pre_e, M.en, T, e, eps, s);
    if (!gemv_rows(M.fc_e.type, M.fc_e.p, M.en, e, M.ev, e, e, e, T, s)) { std::fprintf(stderr, "q4e MTP: failed at line %d (%s)\n", __LINE__, cudaGetErrorString(cudaPeekAtLastError())); return false; }
    ops::rmsnorm_rows(M.hin, M.pre_h, M.hn, T, pitch, eps, s);
    for (int st = 0; st < hc; ++st)
        if (!gemv_rows(M.fc_h.type, M.fc_h.p, M.hn + size_t(st) * e, pitch, M.h + size_t(st) * e, pitch, e, e, T, s)) { std::fprintf(stderr, "q4e MTP: failed at line %d (%s)\n", __LINE__, cudaGetErrorString(cudaPeekAtLastError())); return false; }
    add_e_k<<<dim3(unsigned((pitch + 255) / 256), unsigned(T)), 256, 0, s>>>(M.st, M.h, M.ev, hc, e, T);
    // attention
    const int qw = cc.n_head * cc.hd, kvw = cc.n_head_kv * cc.hd;
    if (!q4e_hc_read_rows(M.hca, M.st, T)) { std::fprintf(stderr, "q4e MTP: failed at line %d (%s)\n", __LINE__, cudaGetErrorString(cudaPeekAtLastError())); return false; }
    if (!gemv_rows(M.q.type, M.q.p, qrw.mixed, e, M.qfull, 2 * qw, 2 * qw, e, T, s) ||
        !gemv_rows(M.k.type, M.k.p, qrw.mixed, e, M.kbuf, kvw, kvw, e, T, s) ||
        !gemv_rows(M.v.type, M.v.p, qrw.mixed, e, M.vbuf, kvw, kvw, e, T, s))
        { std::fprintf(stderr, "q4e MTP: failed at line %d (%s)\n", __LINE__, cudaGetErrorString(cudaPeekAtLastError())); return false; }
    dec::attn_prep(M.qfull, M.kbuf, M.vbuf, M.qn, M.kn, M.qbuf, M.kc, M.vc, M.d_pos, T, M.cap, cc.eps, cc.rope_base,
                   cc.rope_dim, cc.n_head, cc.n_head_kv, cc.hd, s, nullptr);
    if (!dec::attention(M.qbuf, M.kc, M.vc, M.qfull, M.attn, M.scr, M.d_pos, T, M.cap, M.cap, cc.n_head, cc.n_head_kv, cc.hd,
                        float(1.0 / std::sqrt(double(cc.hd))), s, nullptr))
        { std::fprintf(stderr, "q4e MTP: failed at line %d (%s)\n", __LINE__, cudaGetErrorString(cudaPeekAtLastError())); return false; }
    if (!gemv_rows(M.o.type, M.o.p, M.attn, qw, M.out, e, e, qw, T, s)) { std::fprintf(stderr, "q4e MTP: failed at line %d (%s)\n", __LINE__, cudaGetErrorString(cudaPeekAtLastError())); return false; }
    ops::hc_combine_rows(M.st, M.out, qrw.inj, hc, e, T, inv_hc, s);
    // the MoE
    if (!q4e_hc_read_rows(M.hcf, M.st, T)) { std::fprintf(stderr, "q4e MTP: failed at line %d (%s)\n", __LINE__, cudaGetErrorString(cudaPeekAtLastError())); return false; }
    const int nu = M.n_used, pairs = T * nu;
    if (!gemv_rows(M.router.type, M.router.p, qrw.mixed, e, M.rlog, M.n_exp, M.n_exp, e, T, s)) { std::fprintf(stderr, "q4e MTP: failed at line %d (%s)\n", __LINE__, cudaGetErrorString(cudaPeekAtLastError())); return false; }
    if (!dec::topk(M.rlog, M.n_exp, nu, T, M.ids, M.wts, s)) { std::fprintf(stderr, "q4e MTP: failed at line %d (%s)\n", __LINE__, cudaGetErrorString(cudaPeekAtLastError())); return false; }
    if (!gemv_experts(M.gu.type, M.gu.p, M.gu_bytes, M.ids, pairs, qrw.mixed, e, M.gub, 2 * M.nff, e, s, nu)) { std::fprintf(stderr, "q4e MTP: failed at line %d (%s)\n", __LINE__, cudaGetErrorString(cudaPeekAtLastError())); return false; }
    swiglu_split_k<<<(pairs * M.nff + 255) / 256, 256, 0, s>>>(M.gub, M.hh, M.nff, pairs * M.nff);
    if (!gemv_experts(M.dn.type, M.dn.p, M.dn_bytes, M.ids, pairs, M.hh, M.nff, M.dnb, e, M.nff, s, 1)) { std::fprintf(stderr, "q4e MTP: failed at line %d (%s)\n", __LINE__, cudaGetErrorString(cudaPeekAtLastError())); return false; }
    if (!gemv_rows(M.sh_gate.type, M.sh_gate.p, qrw.mixed, e, M.gs, M.nsh, M.nsh, e, T, s) ||
        !gemv_rows(M.sh_up.type, M.sh_up.p, qrw.mixed, e, M.us, M.nsh, M.nsh, e, T, s))
        { std::fprintf(stderr, "q4e MTP: failed at line %d (%s)\n", __LINE__, cudaGetErrorString(cudaPeekAtLastError())); return false; }
    ops::swiglu(M.gs, M.us, M.hs, T * M.nsh, s);
    if (!gemv_rows(M.sh_down.type, M.sh_down.p, M.hs, M.nsh, M.shared, e, e, M.nsh, T, s)) { std::fprintf(stderr, "q4e MTP: failed at line %d (%s)\n", __LINE__, cudaGetErrorString(cudaPeekAtLastError())); return false; }
    ops::shexp_gate_rows(M.shgate, qrw.mixed, M.sg, e, T, s);
    if (cudaMemsetAsync(M.out, 0, sizeof(float) * size_t(T) * e, s) != cudaSuccess) { std::fprintf(stderr, "q4e MTP: failed at line %d (%s)\n", __LINE__, cudaGetErrorString(cudaPeekAtLastError())); return false; }
    ops::moe_combine_rows(M.out, M.dnb, qrw.d_pair, M.wts, M.shared, M.sg, e, nu, T, s);
    ops::hc_combine_rows(M.st, M.out, qrw.inj, hc, e, T, inv_hc, s);
    M.last_T = T;
    return cudaGetLastError() == cudaSuccess;
}

// The block's front for n cells from cell0: residual row R_rows + i * pitch (device) with the embedding of next_tokens[i]
// (host: the token at cell0 + i + 1) through fc_embedding / fc_hidden and the attention's hyper-connection to its K/V.
// No attention, MoE or head: what a later draft attends to of a prompt, or of rows no draft started from.
bool Gpu::Impl::q4e_mtp_front(const float* R_rows, const int32_t* next_tokens, int n, int cell0) {
    Q4eMtp& M = q4mtp;
    if (!M.ready || n <= 0) return true;
    const Cfg& cc = c;
    cudaStream_t s = stream;
    const int e = cc.n_embd, hc = q.hc, pitch = hc * e;
    const float eps = float(cc.eps);
    const int kvw = cc.n_head_kv * cc.hd;
    for (int i0 = 0; i0 < n; i0 += kQ4eRows) {
        const int T = std::min(kQ4eRows, n - i0), q0 = cell0 + i0;
        if (q0 < 0 || q0 + T > M.cap) break;   // past the block's room: not kept
        int32_t args[kQ4eRows];
        for (int t = 0; t < T; ++t) args[t] = next_tokens[i0 + t];
        if (cudaMemcpyAsync(M.hin, R_rows + size_t(i0) * pitch, sizeof(float) * size_t(T) * pitch, cudaMemcpyDeviceToDevice, s) != cudaSuccess ||
            cudaMemcpyAsync(M.d_tok, args, sizeof(int32_t) * size_t(T), cudaMemcpyHostToDevice, s) != cudaSuccess ||
            cudaMemcpyAsync(M.d_pos, &q0, sizeof(int32_t), cudaMemcpyHostToDevice, s) != cudaSuccess ||
            !embed_rows(embd.type, embd.p, M.d_tok, T, M.emb, e, s))
            return false;
        ops::rmsnorm_rows(M.emb, M.pre_e, M.en, T, e, eps, s);
        if (!gemv_rows(M.fc_e.type, M.fc_e.p, M.en, e, M.ev, e, e, e, T, s)) return false;
        ops::rmsnorm_rows(M.hin, M.pre_h, M.hn, T, pitch, eps, s);
        for (int st = 0; st < hc; ++st)
            if (!gemv_rows(M.fc_h.type, M.fc_h.p, M.hn + size_t(st) * e, pitch, M.h + size_t(st) * e, pitch, e, e, T, s)) return false;
        add_e_k<<<dim3(unsigned((pitch + 255) / 256), unsigned(T)), 256, 0, s>>>(M.st, M.h, M.ev, hc, e, T);
        if (!q4e_hc_read_rows(M.hca, M.st, T) ||   // no q: the prep's query output is not read
            !gemv_rows(M.k.type, M.k.p, qrw.mixed, e, M.kbuf, kvw, kvw, e, T, s) ||
            !gemv_rows(M.v.type, M.v.p, qrw.mixed, e, M.vbuf, kvw, kvw, e, T, s))
            return false;
        dec::attn_prep(M.qfull, M.kbuf, M.vbuf, M.qn, M.kn, M.qbuf, M.kc, M.vc, M.d_pos, T, M.cap, cc.eps, cc.rope_base, cc.rope_dim,
                       cc.n_head, cc.n_head_kv, cc.hd, s, nullptr);
    }
    return cudaGetLastError() == cudaSuccess;
}

// q4e_mtp_front for a prompt piece: blocks of up to kQ4eMb rows through the prompt pass's GEMMs and workspace (the
// block's projections are BF16, and the rows form reads a BF16 matrix once a row).
bool Gpu::Impl::q4e_mtp_prompt(const float* R_rows, const int32_t* next_tokens, int n, int cell0) {
    Q4eMtp& M = q4mtp;
    if (!M.ready || n <= 0) return true;
    if (!ensure_q4e_pf()) return false;
    const Cfg& cc = c;
    cudaStream_t s = stream;
    const int e = cc.n_embd, hc = q.hc, pitch = hc * e, kvw = cc.n_head_kv * cc.hd;
    const float eps = float(cc.eps);
    for (int i0 = 0; i0 < n; i0 += kQ4eMb) {
        const int T = std::min(kQ4eMb, n - i0), q0 = cell0 + i0;
        if (q0 < 0 || q0 + T > M.cap) break;
        if (cudaMemcpyAsync(qpf.ids, next_tokens + i0, sizeof(int32_t) * size_t(T), cudaMemcpyHostToDevice, s) != cudaSuccess ||
            cudaMemcpyAsync(M.d_pos, &q0, sizeof(int32_t), cudaMemcpyHostToDevice, s) != cudaSuccess ||
            !embed_rows(embd.type, embd.p, qpf.ids, T, qpf.x, e, s))
            return false;
        ops::rmsnorm_rows(qpf.x, M.pre_e, qpf.out, T, e, eps, s);
        if (!q4e_mm(M.fc_e, qpf.out, e, qpf.shared, e, e, e, T)) return false;
        ops::rmsnorm_rows(R_rows + size_t(i0) * pitch, M.pre_h, qpf.pkey, T, pitch, eps, s);
        for (int st = 0; st < hc; ++st)
            if (!q4e_mm(M.fc_h, qpf.pkey + size_t(st) * e, pitch, qpf.pgated + size_t(st) * e, pitch, e, e, T)) return false;
        add_e_k<<<dim3(unsigned((pitch + 255) / 256), unsigned(T)), 256, 0, s>>>(qpf.pnrm, qpf.pgated, qpf.shared, hc, e, T);
        if (!q4e_hc_read_pf(M.hca, qpf.pnrm, T) || !q4e_mm(M.k, qpf.mixed, e, qpf.kbuf, kvw, kvw, e, T) ||
            !q4e_mm(M.v, qpf.mixed, e, qpf.vbuf, kvw, kvw, e, T))
            return false;
        dec::attn_prep(qpf.qfull, qpf.kbuf, qpf.vbuf, M.qn, M.kn, qpf.qbuf, M.kc, M.vc, M.d_pos, T, M.cap, cc.eps, cc.rope_base, cc.rope_dim,
                       cc.n_head, cc.n_head_kv, cc.hd, s, nullptr);
    }
    return cudaGetLastError() == cudaSuccess;
}

// q4e_mtp_argmax with each draft's probability under the block (over its draft vocabulary).
bool Gpu::Impl::q4e_mtp_argmax_p(int32_t* out, float* prob, int T) {
    Q4eMtp& M = q4mtp;
    const Cfg& cc = c;
    cudaStream_t s = stream;
    if (!M.ready || T < 1 || T > M.last_T) return false;
    const int e = cc.n_embd, nv = M.nd > 0 ? M.nd : cc.n_vocab;
    if (!q4e_hc_read_rows(M.mix, M.st, T)) return false;
    if (!gemv_rows(head.type, head.p, qrw.mixed, e, M.logits, nv, nv, e, T, s)) return false;
    argmax_p_k<<<T, 1024, 0, s>>>(M.logits, nv, M.d_am, M.d_pr);
    if (cudaMemcpyAsync(out, M.d_am, sizeof(int32_t) * size_t(T), cudaMemcpyDeviceToHost, s) != cudaSuccess ||
        cudaMemcpyAsync(prob, M.d_pr, sizeof(float) * size_t(T), cudaMemcpyDeviceToHost, s) != cudaSuccess)
        return false;
    return cudaStreamSynchronize(s) == cudaSuccess;
}

bool Gpu::Impl::q4e_mtp_argmax(int32_t* out, int T) {
    Q4eMtp& M = q4mtp;
    const Cfg& cc = c;
    cudaStream_t s = stream;
    if (!M.ready || T < 1 || T > M.last_T) { std::fprintf(stderr, "q4e MTP: failed at line %d (%s)\n", __LINE__, cudaGetErrorString(cudaPeekAtLastError())); return false; }
    const int e = cc.n_embd, nv = M.nd > 0 ? M.nd : cc.n_vocab;
    if (!q4e_hc_read_rows(M.mix, M.st, T)) { std::fprintf(stderr, "q4e MTP: failed at line %d (%s)\n", __LINE__, cudaGetErrorString(cudaPeekAtLastError())); return false; }
    if (!gemv_rows(head.type, head.p, qrw.mixed, e, M.logits, nv, nv, e, T, s)) { std::fprintf(stderr, "q4e MTP: failed at line %d (%s)\n", __LINE__, cudaGetErrorString(cudaPeekAtLastError())); return false; }
    ops::argmax_rows(M.logits, nv, T, M.d_am, s);
    if (cudaMemcpyAsync(out, M.d_am, sizeof(int32_t) * size_t(T), cudaMemcpyDeviceToHost, s) != cudaSuccess) { std::fprintf(stderr, "q4e MTP: failed at line %d (%s)\n", __LINE__, cudaGetErrorString(cudaPeekAtLastError())); return false; }
    return cudaStreamSynchronize(s) == cudaSuccess;
}

}  // namespace limbo::cuda
