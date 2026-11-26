// Copyright (c) 2026 ITERFORM. All rights reserved.
// src/cuda/gpu_rows.cu — a window of T = 1..8 tokens in one pass (verify windows for speculation), and the commit of the
// accepted prefix. Owner: the decode agent. Private state: gpu_impl.hpp (decode section, `rw`).
//
// The pass is the decode step with a row dimension: every weight is read once for all T rows (gemv_cols), the attention
// scores T queries causally (dec::attention), the delta net scans the T tokens in order with the state in registers, and
// the routed experts are processed per (row, choice) pair by the same kernels step() uses. Each row's arithmetic is the
// statement-for-statement arithmetic of step(), so the logits of row t equal those of the sequential steps (bit for bit
// except where a kernel's launch shape changes a float reduction, and none does here).
//
// write_back = false (docs/donors/kindi.md 2.5): the pass does not advance the delta-net state or the conv history; the
// inputs of the scan and the conv are kept per layer, and commit(n) advances both by exactly the first n tokens. The KV
// rows of rejected tokens are overwritten when those positions are next written.
#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "dec_ops.hpp"
#include "gpu_impl.hpp"
#include "limbo/cuda/gemm.hpp"
#include "limbo/cuda/gemv.hpp"
#include "limbo/model.hpp"
#include "limbo/quants.hpp"
#include "limbo/types.hpp"
#include "ops.hpp"

namespace limbo::cuda {

constexpr uint32_t F32 = static_cast<uint32_t>(Type::F32);

namespace {

__device__ __forceinline__ float silu_r(float x) { return x / (1.0f + expf(-x)); }
__device__ __forceinline__ float sigmoid_r(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float softplus_r(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

template <typename T>
__device__ __forceinline__ T warp_sum_r(T v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
template <typename T>
__device__ __forceinline__ T block_sum_r(T v, T* sm) {
    v = warp_sum_r(v);
    if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = v;
    __syncthreads();
    T r = 0;
    for (int i = 0; i < static_cast<int>(blockDim.x >> 5); ++i) r += sm[i];
    __syncthreads();
    return r;
}
__device__ __forceinline__ bool better_r(float v1, int i1, float v2, int i2) { return v1 > v2 || (v1 == v2 && i1 < i2); }
__device__ __forceinline__ void block_argmax_r(float& v, int& idx, float* sv, int* si) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, v, o);
        const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
        if (better_r(ov, oi, v, idx)) {
            v = ov;
            idx = oi;
        }
    }
    if ((threadIdx.x & 31) == 0) {
        sv[threadIdx.x >> 5] = v;
        si[threadIdx.x >> 5] = idx;
    }
    __syncthreads();
    float bv = sv[0];
    int bi = si[0];
    for (int i = 1; i < static_cast<int>(blockDim.x >> 5); ++i)
        if (better_r(sv[i], si[i], bv, bi)) {
            bv = sv[i];
            bi = si[i];
        }
    __syncthreads();
    v = bv;
    idx = bi;
}

// ---- row kernels: the single-token kernels of ops.cu with a row index (blockIdx.x or .y) -----------------------
// Depthwise conv over T consecutive tokens, channel by channel. out == nullptr: only the history moves (the commit).
// write_back false: the history is left alone.
__global__ void conv_rows_k(const float* qkv, int in_stride, float* hist, const float* w, float* out, int out_stride, int width,
                            int T, bool write_back) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= width) return;
    float h0 = hist[ch], h1 = hist[width + ch], h2 = hist[2 * width + ch];
    const float* wc = w + static_cast<int64_t>(ch) * 4;
    for (int t = 0; t < T; ++t) {
        const float xn = qkv[static_cast<int64_t>(t) * in_stride + ch];
        if (out) {
            float acc = 0.f;
            acc += wc[0] * h0;
            acc += wc[1] * h1;
            acc += wc[2] * h2;
            acc += wc[3] * xn;
            out[static_cast<int64_t>(t) * out_stride + ch] = silu_r(acc);
        }
        h0 = h1;
        h1 = h2;
        h2 = xn;
    }
    if (write_back) {
        hist[ch] = h0;
        hist[width + ch] = h1;
        hist[2 * width + ch] = h2;
    }
}

// grid (n_heads, T), block 128
__global__ void head_norm_rows_k(const float* in, int in_stride, float* out, int out_stride, int hd, float eps, float scale) {
    __shared__ double sm[32];
    const float* src = in + static_cast<int64_t>(blockIdx.y) * in_stride + static_cast<int64_t>(blockIdx.x) * hd;
    float* dst = out + static_cast<int64_t>(blockIdx.y) * out_stride + static_cast<int64_t>(blockIdx.x) * hd;
    double ss = 0;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        const double v = src[d];
        ss += v * v;
    }
    ss = block_sum_r(ss, sm);
    __shared__ float s_inv;
    if (threadIdx.x == 0) s_inv = static_cast<float>(1.0 / sqrt(ss / hd + static_cast<double>(eps))) * scale;
    __syncthreads();
    const float inv = s_inv;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) dst[d] = src[d] * inv;
}

// The delta rule over T tokens: one block per value head, one thread per value column, the state column in 128
// registers for all T tokens; per token the statements of ops.cu's gdn_scan_kernel. o == nullptr skips the output (the
// commit). write_back false: the state is not stored.
__global__ void __launch_bounds__(128)
    scan_rows_k(float* S, const float* qk, int qk_stride, const float* vsrc, int v_stride, const float* alpha, const float* beta,
                int ab_stride, const float* dt_bias, const float* ssm_a, float* o, int o_stride, int n_k_head, int hd, int dt_rank,
                float scale, int T, bool write_back) {
    __shared__ float ks[2][128], qs[2][128];
    const int h = blockIdx.x, j = threadIdx.x;
    const int kh = h % n_k_head;
    const int r = h % dt_rank;
    float* s = S + static_cast<int64_t>(h) * hd * hd;
    float st[128];
#pragma unroll
    for (int i = 0; i < 128; ++i) st[i] = s[i * 128 + j];
    for (int t = 0; t < T; ++t) {
        const int b = t & 1;
        qs[b][j] = qk[static_cast<int64_t>(t) * qk_stride + static_cast<int64_t>(kh) * hd + j];
        ks[b][j] = qk[static_cast<int64_t>(t) * qk_stride + static_cast<int64_t>(n_k_head) * hd + static_cast<int64_t>(kh) * hd + j];
        __syncthreads();
        const float gate = softplus_r(alpha[static_cast<int64_t>(t) * ab_stride + r] + dt_bias[r]) * ssm_a[r];
        const float a = expf(gate);
        const float bb = sigmoid_r(beta[static_cast<int64_t>(t) * ab_stride + r]);
        float skj = 0.f;
#pragma unroll
        for (int i = 0; i < 128; ++i) {
            const float val = st[i] * a;
            st[i] = val;
            skj += val * ks[b][i];
        }
        const float d = (vsrc[static_cast<int64_t>(t) * v_stride + static_cast<int64_t>(h) * hd + j] - skj) * bb;
        float oj = 0.f;
#pragma unroll
        for (int i = 0; i < 128; ++i) {
            const float val = st[i] + ks[b][i] * d;
            st[i] = val;
            oj += val * qs[b][i];
        }
        if (o) o[static_cast<int64_t>(t) * o_stride + static_cast<int64_t>(h) * hd + j] = oj * scale;
    }
    if (write_back) {
#pragma unroll
        for (int i = 0; i < 128; ++i) s[i * 128 + j] = st[i];
    }
}

// grid (n_heads, T), block 128
__global__ void gate_rows_k(const float* o, const float* z, const float* sn, float* gated, int hd, double eps, int stride) {
    __shared__ double sm[32];
    const int64_t base = static_cast<int64_t>(blockIdx.y) * stride + static_cast<int64_t>(blockIdx.x) * hd;
    const float* oh = o + base;
    double ss = 0;
    for (int j = threadIdx.x; j < hd; j += blockDim.x) {
        const double v = oh[j];
        ss += v * v;
    }
    ss = block_sum_r(ss, sm);
    __shared__ float s_inv;
    if (threadIdx.x == 0) s_inv = static_cast<float>(1.0 / sqrt(ss / hd + eps));
    __syncthreads();
    const float inv = s_inv;
    for (int j = threadIdx.x; j < hd; j += blockDim.x) {
        const float normed = oh[j] * inv * sn[j];
        gated[base + j] = normed * silu_r(z[base + j]);
    }
}

// grid (T), block 256
__global__ void shexp_rows_k(const float* gw, const float* post, float* sg, int n) {
    __shared__ float sm[32];
    post += static_cast<int64_t>(blockIdx.x) * n;
    float acc = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) acc += gw[i] * post[i];
    acc = block_sum_r(acc, sm);
    if (threadIdx.x == 0) sg[blockIdx.x] = sigmoid_r(acc);
}

// grid (ceil(n / 256), T): x[t][j] += (sum_i dn[t][i][j] * wts[t][i]) + shared[t][j] * sg[t]
__global__ void combine_rows_k(float* x, const float* dn, const float* wts, const float* shared, const float* sg, int n, int n_used) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    const int t = blockIdx.y;
    float acc = 0.f;
    for (int i = 0; i < n_used; ++i) acc += dn[(static_cast<int64_t>(t) * n_used + i) * n + j] * wts[t * n_used + i];
    x[static_cast<int64_t>(t) * n + j] += acc + shared[static_cast<int64_t>(t) * n + j] * sg[t];
}

// grid (T), block 1024
__global__ void argmax_rows_k(const float* logits, int n, int32_t* out) {
    __shared__ float sv[32];
    __shared__ int si[32];
    logits += static_cast<int64_t>(blockIdx.x) * n;
    float bv = -INFINITY;
    int bi = INT_MAX;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = logits[i];
        if (v > bv) {
            bv = v;
            bi = i;
        }
    }
    block_argmax_r(bv, bi, sv, si);
    if (threadIdx.x == 0) out[blockIdx.x] = bi;
}

}  // namespace

constexpr int kMtpCap = 8192;  // positions of the MTP block's own K/V (Gpu::mtp_capacity)

// ---- the pass ---------------------------------------------------------------------------------------------------

// Debug (set_decode_profile): per-phase milliseconds of an eager T-row or MTP pass, summed into the profile.
struct PassProf {
    std::vector<std::pair<std::string, double>>* out;
    cudaStream_t s;
    std::vector<std::pair<const char*, cudaEvent_t>> evs;
    PassProf(std::vector<std::pair<std::string, double>>* o, cudaStream_t st) : out(o), s(st) {}
    void mark(const char* name) {
        if (!out) return;
        cudaEvent_t e;
        cudaEventCreate(&e);
        cudaEventRecord(e, s);
        evs.emplace_back(name, e);
    }
    ~PassProf() {
        if (!out || evs.empty()) return;
        cudaStreamSynchronize(s);
        for (size_t i = 1; i < evs.size(); ++i) {
            float ms = 0;
            cudaEventElapsedTime(&ms, evs[i - 1].second, evs[i].second);
            bool found = false;
            for (auto& kv : *out)
                if (kv.first == evs[i].first) {
                    kv.second += ms;
                    found = true;
                }
            if (!found) out->emplace_back(evs[i].first, ms);
        }
        for (auto& e : evs) cudaEventDestroy(e.second);
    }
};

bool Gpu::Impl::ensure_rows() {
    if (rw.ready) return true;
    if (c.dt_rank > 64) return false;  // the per-layer scan inputs keep 64 floats a row
    if (!ensure_dec_scratch()) return false;
    if (!rw.attn_scr && !(rw.attn_scr = dalloc<float>(dec::attn_scratch_floats(n_ctx, dec::kMaxRows, c.n_head)))) return false;
    const size_t T = dec::kMaxRows;
    const size_t E = size_t(c.n_embd), vn = size_t(c.n_v_head) * c.gdn_head, qw = size_t(c.n_head) * c.hd, kvw = size_t(c.n_head_kv) * c.hd;
    const size_t qkv_w = size_t(c.qkv_w), qkn_w = 2 * size_t(c.n_k_head) * c.gdn_head;
    int n_gdn = 0;
    for (int l = 0; l < c.n_layer_main; ++l) n_gdn += L[size_t(l)].full ? 0 : 1;
    struct Carve {
        float** dst;
        size_t n;
    };
    float *x, *xn, *o, *z, *gated, *qfull, *kbuf, *vbuf, *qbuf, *attn, *rlog, *post, *wts, *hbuf, *dn, *gs, *us, *hs, *shared, *sg, *rn,
        *logits, *qkv_k, *cv_k, *qkn_k, *al_k, *be_k;
    const size_t ffe = std::max(size_t(c.n_used) * size_t(c.n_ff_exp), size_t(c.n_ff));
    const size_t ffs = std::max(size_t(c.n_ff_shexp), size_t(c.n_ff));  // the shared expert's width, or the dense FFN's
    const Carve carve[] = {
        {&x, T * E},     {&xn, T * E},     {&o, T * vn},      {&z, T * vn},         {&gated, T * vn},
        {&qfull, T * 2 * qw}, {&kbuf, T * kvw}, {&vbuf, T * kvw}, {&qbuf, T * qw}, {&attn, T * qw},
        {&rlog, T * size_t(c.n_expert)}, {&post, T * E}, {&wts, T * size_t(c.n_used)}, {&hbuf, T * ffe},
        {&dn, T * size_t(c.n_used) * E}, {&gs, T * ffs}, {&us, T * ffs},
        {&hs, T * ffs}, {&shared, T * E}, {&sg, T}, {&rn, T * E}, {&logits, T * size_t(c.n_vocab)},
        {&qkv_k, size_t(n_gdn) * T * qkv_w}, {&cv_k, size_t(n_gdn) * T * qkv_w}, {&qkn_k, size_t(n_gdn) * T * qkn_w},
        {&al_k, size_t(n_gdn) * T * 64}, {&be_k, size_t(n_gdn) * T * 64},
    };
    size_t total = 0;
    for (const Carve& cv : carve) total += (cv.n + 63) / 64 * 64;
    float* blob = dalloc<float>(total);
    int32_t* ids = dalloc<int32_t>(T * size_t(c.n_used) + 16);
    int32_t* args = dalloc<int32_t>(16);
    int32_t* am = dalloc<int32_t>(16);
    if (!blob || !ids || !args || !am) return false;
    size_t at = 0;
    for (const Carve& cv : carve) {
        *cv.dst = blob + at;
        at += (cv.n + 63) / 64 * 64;
    }
    rw.x = x; rw.xn = xn; rw.o = o; rw.z = z; rw.gated = gated; rw.qfull = qfull; rw.kbuf = kbuf; rw.vbuf = vbuf; rw.qbuf = qbuf;
    rw.attn = attn; rw.rlog = rlog; rw.post = post; rw.wts = wts; rw.hbuf = hbuf; rw.dn = dn; rw.gs = gs; rw.us = us; rw.hs = hs;
    rw.shared = shared; rw.sg = sg; rw.rn = rn; rw.logits = logits;
    rw.qkv_k = qkv_k; rw.cv_k = cv_k; rw.qkn_k = qkn_k; rw.al_k = al_k; rw.be_k = be_k;
    rw.ids = ids;
    rw.d_args = args;
    rw.d_am = am;
    if (cudaMallocHost(reinterpret_cast<void**>(&rw.pin), sizeof(int32_t) * 16 * kRing) != cudaSuccess) return false;
    rw.ready = true;
    return true;
}

bool Gpu::Impl::launch_rows(int T, bool wb) {
    Impl& I = *this;
    const Cfg& c = I.c;
    cudaStream_t s = I.stream;
    const int E = c.n_embd;
    const int vn = c.n_v_head * c.gdn_head, qw = c.n_head * c.hd, kvw = c.n_head_kv * c.hd;
    const int qkn_w = 2 * c.n_k_head * c.gdn_head;
    const float epsf = static_cast<float>(c.eps);
    const float gdn_scale = static_cast<float>(1.0 / std::sqrt(double(c.gdn_head)));
    const float gdn_eps = static_cast<float>(c.eps / c.gdn_head);
    const float attn_scale = static_cast<float>(1.0 / std::sqrt(double(c.hd)));
    cudaStream_t s2 = I.da.s2;
    int evi = 0;
    auto fork = [&]() {
        cudaEvent_t e = I.da.ev[evi++ & 7];
        cudaEventRecord(e, s);
        cudaStreamWaitEvent(s2, e, 0);
    };
    auto join = [&]() {
        cudaEvent_t e = I.da.ev[evi++ & 7];
        cudaEventRecord(e, s2);
        cudaStreamWaitEvent(s, e, 0);
    };
    bool ok = true;
    int gi = 0;  // index of the delta-net layer among the delta-net layers
    PassProf pp(I.dec_prof, s);
    pp.mark("start");
    const int* pt = I.pt();  // the live slot's page table
    ok &= embed_rows(I.embd.type, I.embd.p, rw.d_args + 1, T, rw.x, E, s);
    for (int l = 0; l < c.n_layer_main; ++l) {
        const Layer& y = I.L[size_t(l)];
        if (I.df) I.df->capture(l, rw.x, T, 0, s);  // the drafter reads some layers' inputs
        ok &= dec::rmsnorm(rw.x, y.attn_norm, rw.xn, E, epsf, T, s);
        if (y.full) {
            ok &= gemv_cols(y.q.type, y.q.p, rw.xn, E, rw.qfull, 2 * qw, 2 * qw, E, T, false, s);
            ok &= gemv_cols(y.k.type, y.k.p, rw.xn, E, rw.kbuf, kvw, kvw, E, T, false, s);
            ok &= gemv_cols(y.v.type, y.v.p, rw.xn, E, rw.vbuf, kvw, kvw, E, T, false, s);
            if (I.kv.fmt == KvFormat::Q4) {
                dec::attn_prep_q4(rw.qfull, rw.kbuf, rw.vbuf, y.qn, y.kn, rw.qbuf, I.kv.kq[size_t(l)], I.kv.ks[size_t(l)], I.kv.vq[size_t(l)],
                                  I.kv.vs[size_t(l)], rw.d_args, T, c.eps, c.rope_base, c.rope_dim, c.n_head, c.n_head_kv, c.hd, s, pt);
                ok &= dec::attention_q4(rw.qbuf, I.kv.kq[size_t(l)], I.kv.ks[size_t(l)], I.kv.vq[size_t(l)], I.kv.vs[size_t(l)], rw.qfull,
                                        rw.attn, rw.attn_scr, rw.d_args, T, I.n_ctx, c.n_head, c.n_head_kv, c.hd, attn_scale, s, pt);
            } else if (I.kv.fmt == KvFormat::Q8) {
                dec::attn_prep_q8(rw.qfull, rw.kbuf, rw.vbuf, y.qn, y.kn, rw.qbuf, I.kv.kq[size_t(l)], I.kv.ks[size_t(l)], I.kv.vq[size_t(l)],
                                  I.kv.vs[size_t(l)], rw.d_args, T, c.eps, c.rope_base, c.rope_dim, c.n_head, c.n_head_kv, c.hd, s, pt);
                ok &= dec::attention_q8(rw.qbuf, I.kv.kq[size_t(l)], I.kv.ks[size_t(l)], I.kv.vq[size_t(l)], I.kv.vs[size_t(l)], rw.qfull,
                                        rw.attn, rw.attn_scr, rw.d_args, T, I.n_ctx, c.n_head, c.n_head_kv, c.hd, attn_scale, s, pt);
            } else {
                dec::attn_prep(rw.qfull, rw.kbuf, rw.vbuf, y.qn, y.kn, rw.qbuf, I.kc[size_t(l)], I.vc[size_t(l)], rw.d_args, T, 0,
                               c.eps, c.rope_base, c.rope_dim, c.n_head, c.n_head_kv, c.hd, s, pt);
                ok &= dec::attention(rw.qbuf, I.kc[size_t(l)], I.vc[size_t(l)], rw.qfull, rw.attn, rw.attn_scr, rw.d_args, T, I.n_ctx,
                                     0, c.n_head, c.n_head_kv, c.hd, attn_scale, s, pt);
            }
            ok &= gemv_cols(y.o.type, y.o.p, rw.attn, qw, rw.x, E, E, qw, T, true, s);
            pp.mark("rows attn layer");
        } else {
            const size_t gb = size_t(gi) * dec::kMaxRows;
            float* qkv = rw.qkv_k + gb * size_t(c.qkv_w);
            float* cv = rw.cv_k + gb * size_t(c.qkv_w);
            float* qkn = rw.qkn_k + gb * size_t(qkn_w);
            float* al = rw.al_k + gb * 64;
            float* be = rw.be_k + gb * 64;
            fork();
            ok &= gemv_cols(y.gate.type, y.gate.p, rw.xn, E, rw.z, vn, vn, E, T, false, s2);
            ok &= gemv_cols(y.qkv.type, y.qkv.p, rw.xn, E, qkv, c.qkv_w, c.qkv_w, E, T, false, s);
            conv_rows_k<<<(c.qkv_w + 255) / 256, 256, 0, s>>>(qkv, c.qkv_w, I.hist[size_t(l)], y.conv, cv, c.qkv_w, c.qkv_w, T, wb);
            head_norm_rows_k<<<dim3(unsigned(2 * c.n_k_head), unsigned(T)), 128, 0, s>>>(cv, c.qkv_w, qkn, qkn_w, c.gdn_head, gdn_eps,
                                                                                          gdn_scale);
            if (y.alpha.type == F32 && y.beta.type == F32 && E <= 2048) {
                ok &= dec::f32_cols(reinterpret_cast<const float*>(y.alpha.p), reinterpret_cast<const float*>(y.beta.p), rw.xn, E, al, be, 64,
                                    c.dt_rank, E, T, 2, s);
            } else {
                ok &= gemv_cols(y.alpha.type, y.alpha.p, rw.xn, E, al, 64, c.dt_rank, E, T, false, s);
                ok &= gemv_cols(y.beta.type, y.beta.p, rw.xn, E, be, 64, c.dt_rank, E, T, false, s);
            }
            scan_rows_k<<<c.n_v_head, c.gdn_head, 0, s>>>(I.S[size_t(l)], qkn, qkn_w, cv + qkn_w, c.qkv_w, al, be, 64, y.dtb, y.ssm_a, rw.o,
                                                         vn, c.n_k_head, c.gdn_head, c.dt_rank, gdn_scale, T, wb);
            join();
            gate_rows_k<<<dim3(unsigned(c.n_v_head), unsigned(T)), 128, 0, s>>>(rw.o, rw.z, y.ssm_norm, rw.gated, c.gdn_head, c.eps, vn);
            ok &= gemv_cols(y.out.type, y.out.p, rw.gated, vn, rw.x, E, E, vn, T, true, s);
            ++gi;
            pp.mark("rows gdn layer");
        }

        ok &= dec::rmsnorm(rw.x, y.post_norm, rw.post, E, epsf, T, s);
        if (c.n_ff > 0) {  // dense SwiGLU: x += down(silu(gate(post)) * up(post))
            ok &= gemv_cols(y.fgate.type, y.fgate.p, rw.post, E, rw.gs, c.n_ff, c.n_ff, E, T, false, s);
            ok &= gemv_cols(y.fup.type, y.fup.p, rw.post, E, rw.us, c.n_ff, c.n_ff, E, T, false, s);
            pp.mark("rows ffn gate/up");
            ops::swiglu(rw.gs, rw.us, rw.hs, T * c.n_ff, s);
            ok &= gemv_cols(y.fdown.type, y.fdown.p, rw.hs, c.n_ff, rw.x, E, E, c.n_ff, T, true, s);
            pp.mark("rows ffn down");
            continue;
        }
        fork();
        ok &= gemv_cols(y.sgate.type, y.sgate.p, rw.post, E, rw.gs, c.n_ff_shexp, c.n_ff_shexp, E, T, false, s2);
        ok &= gemv_cols(y.sup.type, y.sup.p, rw.post, E, rw.us, c.n_ff_shexp, c.n_ff_shexp, E, T, false, s2);
        ops::swiglu(rw.gs, rw.us, rw.hs, T * c.n_ff_shexp, s2);
        ok &= gemv_cols(y.sdown.type, y.sdown.p, rw.hs, c.n_ff_shexp, rw.shared, E, E, c.n_ff_shexp, T, false, s2);
        shexp_rows_k<<<T, 256, 0, s2>>>(y.shgate, rw.post, rw.sg, E);
        if (y.router.type == F32)
            ok &= dec::f32_cols(reinterpret_cast<const float*>(y.router.p), nullptr, rw.post, E, rw.rlog, nullptr, c.n_expert, c.n_expert, E, T, 1, s);
        else
            ok &= gemv_cols(y.router.type, y.router.p, rw.post, E, rw.rlog, c.n_expert, c.n_expert, E, T, false, s);
        ok &= dec::topk(rw.rlog, c.n_expert, c.n_used, T, rw.ids, rw.wts, s);
        const int pairs = T * c.n_used;
        if (!gemv_experts_gate_up(y.ge.type, y.ge.p, y.ue.p, y.ge_bytes, rw.ids, pairs, rw.post, rw.hbuf, c.n_ff_exp, E, s, c.n_used, E)) {
            for (int t = 0; t < T; ++t) {  // a layer whose experts have no fast kernel: the generic pair per row
                ok &= gemv_experts_dual(y.ge.type, y.ge.p, y.ue.p, y.ge_bytes, rw.ids + t * c.n_used, c.n_used, rw.post + size_t(t) * E,
                                        I.gbuf, I.ubuf, c.n_ff_exp, E, s);
                ops::swiglu(I.gbuf, I.ubuf, rw.hbuf + size_t(t) * c.n_used * c.n_ff_exp, c.n_used * c.n_ff_exp, s);
            }
        }
        if (!gemv_experts_down_fast(y.de.type, y.de.p, y.de_bytes, rw.ids, pairs, rw.hbuf, rw.dn, E, c.n_ff_exp, s)) {
            for (int t = 0; t < T; ++t)
                ok &= gemv_experts(y.de.type, y.de.p, y.de_bytes, rw.ids + t * c.n_used, c.n_used,
                                   rw.hbuf + size_t(t) * c.n_used * c.n_ff_exp, c.n_ff_exp, rw.dn + size_t(t) * c.n_used * E, E,
                                   c.n_ff_exp, s);
        }
        join();
        combine_rows_k<<<dim3(unsigned((E + 255) / 256), unsigned(T)), 256, 0, s>>>(rw.x, rw.dn, rw.wts, rw.shared, rw.sg, E, c.n_used);
    }
    ok &= dec::rmsnorm(rw.x, I.out_norm, rw.rn, E, epsf, T, s);
    ok &= gemv_cols(I.head.type, I.head.p, rw.rn, E, rw.logits, c.n_vocab, c.n_vocab, E, T, false, s);
    pp.mark("rows head");
    return ok && cudaGetLastError() == cudaSuccess;
}

// Advances the delta-net state and the conv history by the first n tokens of the last write_back = false window.
bool Gpu::Impl::launch_commit(int n) {
    Impl& I = *this;
    const Cfg& c = I.c;
    cudaStream_t s = I.stream;
    const int vn = c.n_v_head * c.gdn_head, qkn_w = 2 * c.n_k_head * c.gdn_head;
    const float gdn_scale = static_cast<float>(1.0 / std::sqrt(double(c.gdn_head)));
    int gi = 0;
    for (int l = 0; l < c.n_layer_main; ++l) {
        const Layer& y = I.L[size_t(l)];
        if (y.full) continue;
        const size_t gb = size_t(gi) * dec::kMaxRows;
        const float* qkv = rw.qkv_k + gb * size_t(c.qkv_w);
        const float* cv = rw.cv_k + gb * size_t(c.qkv_w);
        const float* qkn = rw.qkn_k + gb * size_t(qkn_w);
        const float* al = rw.al_k + gb * 64;
        const float* be = rw.be_k + gb * 64;
        conv_rows_k<<<(c.qkv_w + 255) / 256, 256, 0, s>>>(qkv, c.qkv_w, I.hist[size_t(l)], y.conv, nullptr, c.qkv_w, c.qkv_w, n, true);
        scan_rows_k<<<c.n_v_head, c.gdn_head, 0, s>>>(I.S[size_t(l)], qkn, qkn_w, cv + qkn_w, c.qkv_w, al, be, 64, y.dtb, y.ssm_a, nullptr, vn,
                                                     c.n_k_head, c.gdn_head, c.dt_rank, gdn_scale, n, true);
        ++gi;
    }
    return cudaGetLastError() == cudaSuccess;
}

bool Gpu::Impl::rows_exec(int slot, int T, bool flag, int mode) {
    Impl& I = *this;
    cudaStream_t s = I.stream;
    auto body = [&]() { return mode == 1 ? I.launch_commit(T) : (mode == 2 ? I.launch_mtp(T, flag) : I.launch_rows(T, flag)); };
    if (!I.graphs) return body();
    // A captured graph bakes in the state pointers and the page table of the slot it was captured for: select_slot swaps the
    // graph sets, so the live ones are always this slot's.
    if (!rw.gx[slot]) {
        if (rw.warm[slot]++ == 0) return body();  // the first pass of each variant runs eagerly so every module is loaded before capture
        cudaGraph_t g = nullptr;
        if (cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal) != cudaSuccess) return false;
        const bool ok = body();
        if (cudaStreamEndCapture(s, &g) != cudaSuccess || !ok || !g) {
            if (g) cudaGraphDestroy(g);
            cudaGetLastError();
            return false;
        }
        const cudaError_t e = cudaGraphInstantiate(&rw.gx[slot], g, 0);
        cudaGraphDestroy(g);
        if (e != cudaSuccess) {
            rw.gx[slot] = nullptr;
            cudaGetLastError();
            return body();
        }
    }
    return cudaGraphLaunch(rw.gx[slot], s) == cudaSuccess;
}

bool Gpu::step_rows(const int32_t* tokens, int T, int pos0, bool write_back) {
    Impl& I = *p_;
    if (T < 1 || T > dec::kMaxRows || pos0 < 0 || pos0 + T > I.n_ctx || !tokens) return false;
    if (!I.ensure_rows()) return false;
    if (!I.pages_for(pos0, pos0 + T)) return false;
    cudaStream_t s = I.stream;
    const int slot = I.ring++ % Impl::kRing;
    if (cudaEventSynchronize(I.pin_ev[slot]) != cudaSuccess) return false;
    int32_t* pin = I.rw.pin + 16 * slot;
    pin[0] = pos0;
    for (int t = 0; t < T; ++t) pin[1 + t] = tokens[t];
    if (cudaMemcpyAsync(I.rw.d_args, pin, 16 * sizeof(int32_t), cudaMemcpyHostToDevice, s) != cudaSuccess) return false;
    if (cudaEventRecord(I.pin_ev[slot], s) != cudaSuccess) return false;
    I.rw.pending = write_back ? 0 : T;
    I.rw.pos0 = pos0;
    if (!I.rows_exec((T - 1) * 2 + (write_back ? 1 : 0), T, write_back, 0)) return false;
    if (write_back) I.df_ingest(0, 0, T, pos0);  // a window that is not written back is ingested by commit(n), for the rows it keeps
    return true;
}

bool Gpu::commit(int n) {
    Impl& I = *p_;
    if (n < 0 || n > I.rw.pending) return false;
    if (n == 0) {
        I.rw.pending = 0;
        return true;
    }
    if (!I.rows_exec(16 + (n - 1), n, true, 1)) return false;
    I.rw.pending = 0;
    I.df_ingest(0, 0, n, I.rw.pos0);
    return true;
}

bool Gpu::read_logits_rows(float* out, int T) {
    Impl& I = *p_;
    if (!I.rw.ready || T < 1 || T > dec::kMaxRows) return false;
    cudaStreamSynchronize(I.stream);
    return cudaMemcpy(out, I.rw.logits, size_t(T) * size_t(I.c.n_vocab) * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
}

bool Gpu::argmax_rows(int32_t* out, int T) {
    Impl& I = *p_;
    if (!I.rw.ready || T < 1 || T > dec::kMaxRows) return false;
    argmax_rows_k<<<T, 1024, 0, I.stream>>>(I.rw.logits, I.c.n_vocab, I.rw.d_am);
    if (cudaMemcpyAsync(out, I.rw.d_am, size_t(T) * sizeof(int32_t), cudaMemcpyDeviceToHost, I.stream) != cudaSuccess) return false;
    return cudaStreamSynchronize(I.stream) == cudaSuccess;
}

// ---- the MTP block ---------------------------------------------------------------------------------------------------
// qwen35moe's nextn block (llama.cpp src/models/qwen35moe.cpp, graph_mtp): x = eh_proj([rms(embed(token)) ; rms(h)]), then one
// full-attention + MoE layer of the usual shape on x (residuals around both halves), then the model's own head after the
// block's head norm. h is the previous position's hidden state before the final norm.

bool Gpu::mtp_ready() const { return p_->q4e ? p_->q4mtp.ready : p_->mtp.ready; }

int Gpu::mtp_capacity() const { return p_->q4e ? p_->q4mtp.cap : p_->mtp.cap; }

bool Gpu::mtp_set_draft_vocab(int n) {
    Impl& I = *p_;
    if (I.q4e) {
        if (!I.q4mtp.ready || n < 0 || n > I.c.n_vocab) return false;
        I.q4mtp.nd = n >= I.c.n_vocab ? 0 : n;
        return true;
    }
    if (!I.mtp.ready || n < 0 || n > I.c.n_vocab) return false;
    cudaStreamSynchronize(I.stream);
    for (int slot = 24; slot < 40; ++slot) {
        if (I.rw.gx[slot]) cudaGraphExecDestroy(I.rw.gx[slot]);
        I.rw.gx[slot] = nullptr;
        I.rw.warm[slot] = 0;
        for (size_t i = 0; i < I.st.slots.size(); ++i) {  // the stashed slots' graphs bake the old draft vocabulary in too
            if (int(i) == I.st.cur) continue;
            if (I.st.slots[i].rgx[slot]) cudaGraphExecDestroy(I.st.slots[i].rgx[slot]);
            I.st.slots[i].rgx[slot] = nullptr;
            I.st.slots[i].rwarm[slot] = 0;
        }
    }
    I.mtp.nd = n >= I.c.n_vocab ? 0 : n;
    return true;
}

bool Gpu::mtp_load(const model::Model& m) {
    Impl& I = *p_;
    if (I.q4e) return I.q4e_mtp_load(m);
    const model::Geometry& geo = m.geometry();
    if (!geo.has_mtp || geo.mtp_layer < 0 || I.mtp.ready) return I.mtp.ready;
    if (!I.ensure_rows()) return false;
    const Cfg& c = I.c;
    const int L = geo.mtp_layer;
    Impl::Mtp& M = I.mtp;
    bool ok = true;
    auto up = [&](const char* role, bool as_f32) -> W {
        W w;
        const gguf::Tensor* t = m.tensor(L, role);
        if (!t) {
            ok = false;
            return w;
        }
        std::vector<float> tmp;
        const uint8_t* src = t->data;
        uint64_t bytes = t->nbytes;
        uint32_t type = t->type;
        if (as_f32 && type != static_cast<uint32_t>(Type::F32)) {
            tmp.resize(static_cast<size_t>(t->n_elements()));
            if (!quants::dequant_row(t->type, t->data, tmp.data(), static_cast<uint64_t>(t->n_elements()))) {
                ok = false;
                return w;
            }
            src = reinterpret_cast<const uint8_t*>(tmp.data());
            bytes = tmp.size() * 4;
            type = static_cast<uint32_t>(Type::F32);
        } else if (as_f32) {
            type = static_cast<uint32_t>(Type::F32);
        }
        uint8_t* d = I.dalloc<uint8_t>(bytes + 256);
        if (!d) {
            ok = false;
            return w;
        }
        for (uint64_t off = 0; off < bytes; off += (64u << 20)) {
            const uint64_t n = std::min<uint64_t>(64u << 20, bytes - off);
            if (cudaMemcpy(d + off, src + off, n, cudaMemcpyHostToDevice) != cudaSuccess) ok = false;
        }
        w.p = d;
        w.type = type;
        return w;
    };
    auto f32 = [&](const char* role) { return reinterpret_cast<const float*>(up(role, true).p); };
    M.eh = up("nextn.eh_proj.weight", false);
    M.enorm = f32("nextn.enorm.weight");
    M.hnorm = f32("nextn.hnorm.weight");
    M.shn = f32("nextn.shared_head_norm.weight");
    M.attn_norm = f32("attn_norm.weight");
    M.post_norm = f32("post_attention_norm.weight");
    M.q = up("attn_q.weight", false);
    M.k = up("attn_k.weight", false);
    M.v = up("attn_v.weight", false);
    M.o = up("attn_output.weight", false);
    M.qn = f32("attn_q_norm.weight");
    M.kn = f32("attn_k_norm.weight");
    if (c.n_ff > 0) {  // qwen35: a dense SwiGLU in the block
        M.fgate = up("ffn_gate.weight", false);
        M.fup = up("ffn_up.weight", false);
        M.fdown = up("ffn_down.weight", false);
    } else {
        M.router = f32("ffn_gate_inp.weight");
        M.shgate = f32("ffn_gate_inp_shexp.weight");
        M.ge = up("ffn_gate_exps.weight", false);
        M.ue = up("ffn_up_exps.weight", false);
        M.de = up("ffn_down_exps.weight", false);
        M.sgate = up("ffn_gate_shexp.weight", false);
        M.sup = up("ffn_up_shexp.weight", false);
        M.sdown = up("ffn_down_shexp.weight", false);
        if (!ok) return false;
        M.ge_bytes = type_bytes(M.ge.type, uint64_t(c.n_embd)) * uint64_t(c.n_ff_exp);
        M.de_bytes = type_bytes(M.de.type, uint64_t(c.n_ff_exp)) * uint64_t(c.n_embd);
    }
    if (!ok) return false;
    for (const W* w : {&M.eh, &M.q, &M.k, &M.v, &M.o, &M.ge, &M.ue, &M.de, &M.sgate, &M.sup, &M.sdown, &M.fgate, &M.fup, &M.fdown})
        if (w->p && !gemv_supported(w->type)) return false;
    // K/V cache of the block (kMtpCap positions of its own) and the activation buffers of up to 8 rows
    M.cap = std::min(I.n_ctx, kMtpCap);
    M.vpitch = (M.cap + 63) / 64 * 64;
    const size_t kv_n = size_t(M.cap) * c.n_head_kv * c.hd, v_n = size_t(c.n_head_kv) * c.hd * size_t(M.vpitch);
    M.kc = I.dalloc<__half>(kv_n);
    M.vc = I.dalloc<__half>(v_n);
    if (!M.kc || !M.vc) return false;
    cudaMemset(M.kc, 0, kv_n * sizeof(__half));
    cudaMemset(M.vc, 0, v_n * sizeof(__half));
    const size_t T = dec::kMaxRows, E = size_t(c.n_embd), qw = size_t(c.n_head) * c.hd, kvw = size_t(c.n_head_kv) * c.hd;
    const size_t ffe = std::max(size_t(c.n_used) * size_t(c.n_ff_exp), size_t(c.n_ff));
    const size_t ffs = std::max(size_t(c.n_ff_shexp), size_t(c.n_ff));
    struct Carve {
        float** dst;
        size_t n;
    };
    const Carve carve[] = {
        {&M.x, T * E},       {&M.cat, T * 2 * E}, {&M.xn, T * E},      {&M.qfull, T * 2 * qw}, {&M.kbuf, T * kvw},
        {&M.vbuf, T * kvw},  {&M.qbuf, T * qw},   {&M.attn, T * qw},   {&M.rlog, T * size_t(c.n_expert)}, {&M.post, T * E},
        {&M.wts, T * size_t(c.n_used)}, {&M.hbuf, T * ffe}, {&M.dn, T * size_t(c.n_used) * E}, {&M.gs, T * ffs},
        {&M.us, T * ffs}, {&M.hs, T * ffs}, {&M.shared, T * E}, {&M.sg, T}, {&M.rn, T * E},
        {&M.logits, T * size_t(c.n_vocab)}, {&M.hout, T * E},
    };
    size_t total = 0;
    for (const Carve& cv : carve) total += (cv.n + 63) / 64 * 64;
    float* blob = I.dalloc<float>(total);
    M.ids = I.dalloc<int32_t>(T * size_t(c.n_used) + 16);
    M.d_args = I.dalloc<int32_t>(16);
    M.d_am = I.dalloc<int32_t>(16);
    if (!blob || !M.ids || !M.d_args || !M.d_am) return false;
    size_t at = 0;
    for (const Carve& cv : carve) {
        *cv.dst = blob + at;
        at += (cv.n + 63) / 64 * 64;
    }
    if (cudaMallocHost(reinterpret_cast<void**>(&M.pin), sizeof(int32_t) * 16 * Impl::kRing) != cudaSuccess) return false;
    cudaMemset(M.hout, 0, T * E * sizeof(float));
    M.ready = true;
    return true;
}

bool Gpu::Impl::launch_mtp(int T, bool hm) {
    Impl& I = *this;
    const Cfg& c = I.c;
    Mtp& M = I.mtp;
    cudaStream_t s = I.stream, s2 = I.da.s2;
    const int E = c.n_embd, qw = c.n_head * c.hd, kvw = c.n_head_kv * c.hd;
    const float epsf = static_cast<float>(c.eps);
    const float attn_scale = static_cast<float>(1.0 / std::sqrt(double(c.hd)));
    int evi = 0;
    auto fork = [&]() {
        cudaEvent_t e = I.da.ev[evi++ & 7];
        cudaEventRecord(e, s);
        cudaStreamWaitEvent(s2, e, 0);
    };
    auto join = [&]() {
        cudaEvent_t e = I.da.ev[evi++ & 7];
        cudaEventRecord(e, s2);
        cudaStreamWaitEvent(s, e, 0);
    };
    bool ok = true;
    PassProf pp(I.dec_prof, s);
    pp.mark("start");
    // x = eh_proj([rms(embed(token)) ; rms(h)])
    ok &= embed_rows(I.embd.type, I.embd.p, M.d_args + 1, T, M.xn, E, s);
    ok &= dec::rmsnorm(M.xn, M.enorm, M.cat, E, epsf, T, s, 2 * E);
    ok &= dec::rmsnorm(hm ? M.hout : rw.x, M.hnorm, M.cat + E, E, epsf, T, s, 2 * E);
    ok &= gemv_cols(M.eh.type, M.eh.p, M.cat, 2 * E, M.x, E, E, 2 * E, T, false, s);
    pp.mark("mtp embed+eh");
    // attention, residual on x
    ok &= dec::rmsnorm(M.x, M.attn_norm, M.xn, E, epsf, T, s);
    ok &= gemv_cols(M.q.type, M.q.p, M.xn, E, M.qfull, 2 * qw, 2 * qw, E, T, false, s);
    ok &= gemv_cols(M.k.type, M.k.p, M.xn, E, M.kbuf, kvw, kvw, E, T, false, s);
    ok &= gemv_cols(M.v.type, M.v.p, M.xn, E, M.vbuf, kvw, kvw, E, T, false, s);
    dec::attn_prep(M.qfull, M.kbuf, M.vbuf, M.qn, M.kn, M.qbuf, M.kc, M.vc, M.d_args, T, M.vpitch, c.eps, c.rope_base, c.rope_dim, c.n_head,
                   c.n_head_kv, c.hd, s);
    ok &= dec::attention(M.qbuf, M.kc, M.vc, M.qfull, M.attn, I.rw.attn_scr, M.d_args, T, M.cap, M.vpitch, c.n_head, c.n_head_kv, c.hd,
                         attn_scale, s);
    ok &= gemv_cols(M.o.type, M.o.p, M.attn, qw, M.x, E, E, qw, T, true, s);
    pp.mark("mtp attention");
    // the FFN half
    ok &= dec::rmsnorm(M.x, M.post_norm, M.post, E, epsf, T, s);
    if (c.n_ff > 0) {
        ok &= gemv_cols(M.fgate.type, M.fgate.p, M.post, E, M.gs, c.n_ff, c.n_ff, E, T, false, s);
        ok &= gemv_cols(M.fup.type, M.fup.p, M.post, E, M.us, c.n_ff, c.n_ff, E, T, false, s);
        ops::swiglu(M.gs, M.us, M.hs, T * c.n_ff, s);
        ok &= gemv_cols(M.fdown.type, M.fdown.p, M.hs, c.n_ff, M.x, E, E, c.n_ff, T, true, s);
    } else {
    fork();
    ok &= gemv_cols(M.sgate.type, M.sgate.p, M.post, E, M.gs, c.n_ff_shexp, c.n_ff_shexp, E, T, false, s2);
    ok &= gemv_cols(M.sup.type, M.sup.p, M.post, E, M.us, c.n_ff_shexp, c.n_ff_shexp, E, T, false, s2);
    ops::swiglu(M.gs, M.us, M.hs, T * c.n_ff_shexp, s2);
    ok &= gemv_cols(M.sdown.type, M.sdown.p, M.hs, c.n_ff_shexp, M.shared, E, E, c.n_ff_shexp, T, false, s2);
    shexp_rows_k<<<T, 256, 0, s2>>>(M.shgate, M.post, M.sg, E);
    ok &= dec::f32_cols(M.router, nullptr, M.post, E, M.rlog, nullptr, c.n_expert, c.n_expert, E, T, 1, s);
    ok &= dec::topk(M.rlog, c.n_expert, c.n_used, T, M.ids, M.wts, s);
    for (int t = 0; t < T; ++t) {  // the block's experts are Q2_K / Q3_K: the generic products, a row at a time
        ok &= gemv_experts_dual(M.ge.type, M.ge.p, M.ue.p, M.ge_bytes, M.ids + t * c.n_used, c.n_used, M.post + size_t(t) * E, I.gbuf, I.ubuf, c.n_ff_exp, E, s);
        ops::swiglu(I.gbuf, I.ubuf, M.hbuf + size_t(t) * c.n_used * c.n_ff_exp, c.n_used * c.n_ff_exp, s);
        ok &= gemv_experts(M.de.type, M.de.p, M.de_bytes, M.ids + t * c.n_used, c.n_used, M.hbuf + size_t(t) * c.n_used * c.n_ff_exp, c.n_ff_exp,
                           M.dn + size_t(t) * c.n_used * E, E, c.n_ff_exp, s);
    }
    join();
    combine_rows_k<<<dim3(unsigned((E + 255) / 256), unsigned(T)), 256, 0, s>>>(M.x, M.dn, M.wts, M.shared, M.sg, E, c.n_used);
    }
    pp.mark("mtp ffn");
    cudaMemcpyAsync(M.hout, M.x, size_t(T) * size_t(E) * sizeof(float), cudaMemcpyDeviceToDevice, s);
    // the model's own head
    ok &= dec::rmsnorm(M.x, M.shn, M.rn, E, epsf, T, s);
    const int nd = M.nd > 0 ? M.nd : c.n_vocab;
    ok &= gemv_cols(I.head.type, I.head.p, M.rn, E, M.logits, nd, nd, E, T, false, s);
    pp.mark("mtp head");
    return ok && cudaGetLastError() == cudaSuccess;
}

bool Gpu::mtp_rows(const int32_t* tokens, int T, int q0, bool h_from_mtp) {
    Impl& I = *p_;
    if (I.q4e) return I.q4e_mtp_rows(tokens, T, q0, h_from_mtp);
    if (!I.mtp.ready || T < 1 || T > dec::kMaxRows || q0 < 0 || q0 + T > I.mtp.cap || !tokens || (h_from_mtp && T != 1)) return false;
    cudaStream_t s = I.stream;
    const int slot = I.ring++ % Impl::kRing;
    if (cudaEventSynchronize(I.pin_ev[slot]) != cudaSuccess) return false;
    int32_t* pin = I.mtp.pin + 16 * slot;
    pin[0] = q0;
    for (int t = 0; t < T; ++t) pin[1 + t] = tokens[t];
    if (cudaMemcpyAsync(I.mtp.d_args, pin, 16 * sizeof(int32_t), cudaMemcpyHostToDevice, s) != cudaSuccess) return false;
    if (cudaEventRecord(I.pin_ev[slot], s) != cudaSuccess) return false;
    return I.rows_exec(24 + (T - 1) * 2 + (h_from_mtp ? 1 : 0), T, h_from_mtp, 2);
}

bool Gpu::mtp_argmax_p(int32_t* out, float* prob, int T) {
    Impl& I = *p_;
    if (I.q4e) return I.q4e_mtp_argmax_p(out, prob, T);
    for (int t = 0; t < T; ++t) prob[t] = 1.0f;
    return mtp_argmax(out, T);
}

bool Gpu::mtp_argmax(int32_t* out, int T) {
    Impl& I = *p_;
    if (I.q4e) return I.q4e_mtp_argmax(out, T);
    if (!I.mtp.ready || T < 1 || T > dec::kMaxRows) return false;
    argmax_rows_k<<<T, 1024, 0, I.stream>>>(I.mtp.logits, I.mtp.nd > 0 ? I.mtp.nd : I.c.n_vocab, I.mtp.d_am);
    if (cudaMemcpyAsync(out, I.mtp.d_am, size_t(T) * sizeof(int32_t), cudaMemcpyDeviceToHost, I.stream) != cudaSuccess) return false;
    return cudaStreamSynchronize(I.stream) == cudaSuccess;
}

}  // namespace limbo::cuda
