// Copyright (c) 2026 ITERFORM. All rights reserved.
// src/cuda/ink_draft.cu — Inkling-2's DSpark block drafter (see ink_draft.hpp).
//
// Written from the checkpoint's GGUF (dspark-Inkling-2-Mini-Q8_0.gguf: arch "dflash", 3 layers, block 8, target
// features at visits 5 / 23 / 41 plus the final norm, has_markov_context) and from reading llama.cpp PR #32218's
// description of the math (src/models/dflash.cpp, dspark.cpp, common/speculative.cpp of that branch). No code is
// taken from it. What the drafter computes:
//
// Context (per target position p, per layer l):  g = rms(sum_i ctx_mix_l[i] * f_i(p)) * ctx_norm_l;
//   K = rope(rms_head(k_ctx_l g) * k_ctx_norm_l, p), V = v_ctx_l g  -- the layer's context K/V.
//   f_rnn(p) = rms(sum_i markov_ctx_mix[i] * f_i(p)) * markov_ctx_norm  -- the Markov head's context state.
// Block (anchor at position pos, rows t = 0..7 at pos + t): x_0 = rms(embed(anchor)) * token_embd_norm,
//   x_t = mask_embd[t - 1]. Per layer: a = rms(x) * attn_norm; q, k, v = wq a, wk a, wv a; q, k per-head rms * their
//   norms, NeoX RoPE (theta 1e6); attention of every row over the context keys before pos (the sliding layers those
//   with p > pos - window, one window for the whole block) and all 8 block keys, no causal mask, scale 1/sqrt(128);
//   x += wo(attn); x += down(silu(gate h) * up h), h = rms(x) * ffn_norm. hn = rms(x) * output_norm.
// Markov head (greedy): prev = anchor, state = 0; for row i = 1..7: j = joint_proj [state, w1[prev], hn_i, f_rnn(pos-1)]
//   + b; gate = sigmoid(j0), cand = tanh(j1), state = cand + gate * (state - cand), m = tanh(j2);
//   col = softcap_30(head(hn_i) + w2 m + b2); prev = argmax col -- draft i.
#include "ink_draft.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "limbo/cuda/gemv.hpp"
#include "limbo/gguf.hpp"
#include "limbo/quants.hpp"
#include "limbo/types.hpp"

namespace limbo::cuda {

namespace {

constexpr int kB = 8;       // block rows: the anchor and 7 masks
constexpr int kHD = 128;    // head dim
constexpr int kCh = 256;    // context positions per attention chunk
constexpr int kMaxNeed = 4;

__device__ __forceinline__ float wsum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float bsum256(float v, float* sh) {   // 256-thread block sum
    v = wsum(v);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    float r = 0.f;
    for (int i = 0; i < 8; ++i) r += sh[i];
    __syncthreads();
    return r;
}

// y[t] = rms(sum_i mix[i] * f[t][i]) * w   (f: [T][4][n]); mix of `need` features
__global__ void mix_norm_k(const float* f, const float* mix, int need, const float* w, float* y, int n, float eps) {
    __shared__ float sh[8];
    const float* ft = f + size_t(blockIdx.x) * 4 * n;
    float ss = 0.f;
    for (int d = threadIdx.x; d < n; d += 256) {
        float v = 0.f;
        for (int i = 0; i < need; ++i) v += mix[i] * ft[size_t(i) * n + d];
        ss += v * v;
    }
    const float inv = rsqrtf(bsum256(ss, sh) / float(n) + eps);
    for (int d = threadIdx.x; d < n; d += 256) {
        float v = 0.f;
        for (int i = 0; i < need; ++i) v += mix[i] * ft[size_t(i) * n + d];
        y[size_t(blockIdx.x) * n + d] = v * inv * w[d];
    }
}

__global__ void rms_rows_k(const float* x, const float* w, float* y, int n, float eps) {
    __shared__ float sh[8];
    const float* xr = x + size_t(blockIdx.x) * n;
    float ss = 0.f;
    for (int d = threadIdx.x; d < n; d += 256) ss += xr[d] * xr[d];
    const float inv = rsqrtf(bsum256(ss, sh) / float(n) + eps);
    for (int d = threadIdx.x; d < n; d += 256) y[size_t(blockIdx.x) * n + d] = xr[d] * inv * (w ? w[d] : 1.f);
}

// per (row, head): rms over 128 with weight w, then NeoX RoPE at position pos0 + row (rope == true)
__global__ void head_norm_rope_k(float* x, const float* w, int n_head, int pos0, float theta, float eps, bool rope) {
    __shared__ float sh[4], v[kHD];
    const int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    float* xh = x + (size_t(t) * n_head + h) * kHD;
    const float a = xh[d];
    float ss = wsum(a * a);
    if ((d & 31) == 0) sh[d >> 5] = ss;
    __syncthreads();
    const float inv = rsqrtf((sh[0] + sh[1] + sh[2] + sh[3]) / float(kHD) + eps);
    v[d] = a * inv * (w ? w[d] : 1.f);
    __syncthreads();
    if (!rope) {
        xh[d] = v[d];
        return;
    }
    const int i = d & 63;
    const float ang = float(pos0 + t) * powf(theta, -float(i) / 64.f);
    float sn, cs;
    sincosf(ang, &sn, &cs);
    xh[d] = d < 64 ? v[i] * cs - v[i + 64] * sn : v[i + 64] * cs + v[i] * sn;
}

__global__ void store_ctx_k(const float* k, const float* v, __half* kc, __half* vc, int kvw, int pos0, int cap) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i >= kvw) return;
    const size_t row = size_t((pos0 + t) % cap);
    kc[row * kvw + i] = __float2half(k[size_t(t) * kvw + i]);
    vc[row * kvw + i] = __float2half(v[size_t(t) * kvw + i]);
}

// The block attention, split over chunks of context positions plus one chunk of the 8 block keys (the last):
// block (chunk, kv head, row), warps = the group's query heads, lanes over 4 dims each, positions in order.
__global__ void attn_part_k(const float* q, const __half* kc, const __half* vc, int cap, int lo, int hi,
                            const float* bk, const float* bv, float* scratch, int n_chunks, int n_head, int n_kv,
                            float scale) {
    const int c = blockIdx.x, kv = blockIdx.y, t = blockIdx.z, w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int G = n_head / n_kv;
    if (w >= G) return;
    const int h = kv * G + w;
    const float* qh = q + (size_t(t) * n_head + h) * kHD;
    float qv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) qv[j] = qh[lane * 4 + j];
    float m = -INFINITY, s = 0.f, num[4] = {0.f, 0.f, 0.f, 0.f};
    const bool blk = c == n_chunks - 1;
    const int first = blk ? 0 : lo + c * kCh;
    const int last = blk ? kB - 1 : min(hi, lo + (c + 1) * kCh - 1);
    for (int p = first; p <= last; ++p) {
        float kk[4], vv[4];
        if (blk) {
            const float* kr = bk + (size_t(p) * n_kv + kv) * kHD + lane * 4;
            const float* vr = bv + (size_t(p) * n_kv + kv) * kHD + lane * 4;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                kk[j] = kr[j];
                vv[j] = vr[j];
            }
        } else {
            const size_t row = size_t(p % cap) * n_kv * kHD + size_t(kv) * kHD + lane * 4;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                kk[j] = __half2float(kc[row + j]);
                vv[j] = __half2float(vc[row + j]);
            }
        }
        float dot = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) dot += qv[j] * kk[j];
        const float l = scale * wsum(dot);
        const float m2 = fmaxf(m, l), e = __expf(m - m2), el = __expf(l - m2);
        s = s * e + el;
#pragma unroll
        for (int j = 0; j < 4; ++j) num[j] = num[j] * e + el * vv[j];
        m = m2;
    }
    float* row = scratch + ((size_t(t) * n_chunks + c) * n_head + h) * (2 + kHD);
    if (lane == 0) {
        row[0] = m;
        row[1] = s;
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) row[2 + lane * 4 + j] = num[j];
}

__global__ void attn_merge_k(const float* scratch, float* out, int n_chunks, int n_head) {
    const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    float m = -INFINITY;
    for (int c = 0; c < n_chunks; ++c) m = fmaxf(m, scratch[((size_t(t) * n_chunks + c) * n_head + h) * (2 + kHD)]);
    float s = 0.f, num = 0.f;
    for (int c = 0; c < n_chunks; ++c) {
        const float* r = scratch + ((size_t(t) * n_chunks + c) * n_head + h) * (2 + kHD);
        if (r[0] == -INFINITY) continue;
        const float e = __expf(r[0] - m);
        s += r[1] * e;
        num += r[2 + d] * e;
    }
    out[(size_t(t) * n_head + h) * kHD + d] = s > 0.f ? num / s : 0.f;
}

__global__ void silu_mul_k(const float* g, const float* u, float* h, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] = g[i] / (1.f + __expf(-g[i])) * u[i];
}
__global__ void add_k(float* x, const float* y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}
__global__ void scale_k(float* x, float s, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}

// joint = [state, w1[prev] (BF16 row), hn, f_rnn]
__global__ void joint_k(const float* state, const uint16_t* w1, const int32_t* prev, const float* hn, const float* frnn,
                        float* joint, int R, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < R) joint[i] = state[i];
    else if (i < 2 * R) joint[i] = __uint_as_float(uint32_t(w1[size_t(*prev) * R + (i - R)]) << 16);
    else if (i < 2 * R + n) joint[i] = hn[i - 2 * R];
    else if (i < 2 * R + 2 * n) joint[i] = frnn[i - 2 * R - n];
}

// the GRU-like state update and the Markov vector
__global__ void gru_k(const float* j3, const float* b, float* state, float* mk, int R) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= R) return;
    const float g = 1.f / (1.f + __expf(-(j3[i] + b[i])));
    const float c = tanhf(j3[R + i] + b[R + i]);
    state[i] = c + g * (state[i] - c);
    mk[i] = tanhf(j3[2 * R + i] + b[2 * R + i]);
}

// col = softcap(base + bias + b2); per-block (max, lowest index) partials
// ... and the block's sum of exp(v - its max) (ps), so the final pass has the softmax probability of the argmax
// (the draft's confidence)
__global__ void col_argmax_part_k(const float* base, const float* bias, const float* b2, int n, float cap, float* pv,
                                  int32_t* pi, float* ps) {
    __shared__ float sv[256];
    __shared__ int32_t si[256];
    __shared__ float sm[256], ssum[256];
    float bvv = -INFINITY, run_m = -INFINITY, run_s = 0.f;
    int32_t bi = INT32_MAX;
    for (int i = blockIdx.x * 256 + threadIdx.x; i < n; i += gridDim.x * 256) {
        float v = base[i] + bias[i] + (b2 ? b2[i] : 0.f);
        if (cap > 0.f) v = cap * tanhf(v / cap);
        if (v > bvv || (v == bvv && i < bi)) {
            bvv = v;
            bi = i;
        }
        const float m2 = fmaxf(run_m, v);
        run_s = run_s * __expf(run_m - m2) + __expf(v - m2);
        run_m = m2;
    }
    sm[threadIdx.x] = run_m;
    ssum[threadIdx.x] = run_s;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) {
        if (int(threadIdx.x) < o) {
            const float m1 = sm[threadIdx.x], m2 = sm[threadIdx.x + o], mm = fmaxf(m1, m2);
            ssum[threadIdx.x] = (m1 == -INFINITY ? 0.f : ssum[threadIdx.x] * __expf(m1 - mm)) +
                                (m2 == -INFINITY ? 0.f : ssum[threadIdx.x + o] * __expf(m2 - mm));
            sm[threadIdx.x] = mm;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) ps[blockIdx.x] = ssum[0];   // relative to the block max, which is sv's max below
    sv[threadIdx.x] = bvv;
    si[threadIdx.x] = bi;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) {
        if (int(threadIdx.x) < o) {
            const float v2 = sv[threadIdx.x + o];
            const int32_t i2 = si[threadIdx.x + o];
            if (v2 > sv[threadIdx.x] || (v2 == sv[threadIdx.x] && i2 < si[threadIdx.x])) {
                sv[threadIdx.x] = v2;
                si[threadIdx.x] = i2;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        pv[blockIdx.x] = sv[0];
        pi[blockIdx.x] = si[0];
    }
}
__global__ void argmax_final_k(const float* pv, const int32_t* pi, const float* ps, int nb, int32_t* prev, int32_t* out,
                               float* conf) {
    if (threadIdx.x != 0) return;
    float bv = -INFINITY;
    int32_t bi = INT32_MAX;
    for (int b = 0; b < nb; ++b)
        if (pv[b] > bv || (pv[b] == bv && pi[b] < bi)) {
            bv = pv[b];
            bi = pi[b];
        }
    float z = 0.f;   // sum over the vocabulary of exp(v - max)
    for (int b = 0; b < nb; ++b)
        if (pv[b] > -INFINITY) z += ps[b] * __expf(pv[b] - bv);
    *prev = bi;
    *out = bi;
    *conf = z > 0.f ? 1.f / z : 0.f;
}

}  // namespace

struct InkDraft::St {
    struct Layer {
        const uint8_t *wq = nullptr, *wk = nullptr, *wv = nullptr, *wo = nullptr, *wkc = nullptr, *wvc = nullptr;
        const uint8_t *gate = nullptr, *up = nullptr, *down = nullptr;
        uint32_t tq = 0, tk = 0, tv = 0, to = 0, tkc = 0, tvc = 0, tg = 0, tu = 0, td = 0;
        float *attn_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr, *kc_norm = nullptr, *ffn_norm = nullptr;
        float *ctx_norm = nullptr, *ctx_mix = nullptr;
        __half *kc = nullptr, *vc = nullptr;
        int cap = 0, window = 0;   // window 0: full attention
    };
    std::vector<Layer> L;
    InkGpu::Shared tg{};
    int n_embd = 0, n_head = 0, n_kv = 0, kvw = 0, n_ff = 0, R = 0, n_vocab = 0, need = 4, visits[3] = {-1, -1, -1};
    float eps = 1e-6f, theta = 1e6f, softcap = 0.f;
    float *out_norm = nullptr, *tok_norm = nullptr, *mask = nullptr, *mk_mix = nullptr, *mk_norm = nullptr;
    const uint8_t *w1 = nullptr, *w2 = nullptr, *jw = nullptr;
    uint32_t w2_type = 0, jw_type = 0;
    float *w2_b = nullptr, *jb = nullptr;
    // scratch
    float *x = nullptr, *a = nullptr, *q = nullptr, *k = nullptr, *v = nullptr, *att = nullptr, *o = nullptr;
    float *g = nullptr, *u = nullptr, *h = nullptr, *hn = nullptr, *ig = nullptr, *ik = nullptr, *iv = nullptr;
    float *frnn = nullptr, *base = nullptr, *state = nullptr, *joint = nullptr, *j3 = nullptr, *mk = nullptr;
    float *bias = nullptr, *pv = nullptr, *ps = nullptr, *conf = nullptr, *scratch = nullptr;
    int32_t *d_anchor = nullptr, *prev = nullptr, *drafts = nullptr, *pi = nullptr, *pinned = nullptr;
    int max_chunks = 0, last = -1;
    std::vector<void*> allocs;
    bool failed = false;

    void* alloc(size_t b) {
        void* p = nullptr;
        if (cudaMalloc(&p, b) != cudaSuccess) {
            failed = true;
            return nullptr;
        }
        allocs.push_back(p);
        return p;
    }
    float* af(size_t n) { return static_cast<float*>(alloc(n * sizeof(float))); }
    ~St() {
        if (tg.stream) cudaStreamSynchronize(tg.stream);
        for (void* p : allocs) cudaFree(p);
        if (pinned) cudaFreeHost(pinned);
    }

    // y[t] = W x[t] for T rows, in pieces of 8 columns
    bool mm(const uint8_t* W, uint32_t type, const float* x_, float* y, int n_out, int n_in, int T) {
        for (int t = 0; t < T; t += 8) {
            const int c = std::min(8, T - t);
            if (!gemv_cols(type, W, x_ + size_t(t) * n_in, n_in, y + size_t(t) * n_out, n_out, n_out, n_in, c, false,
                           tg.stream))
                for (int i = 0; i < c; ++i)
                    if (!gemv(type, W, x_ + size_t(t + i) * n_in, y + size_t(t + i) * n_out, n_out, n_in, false, tg.stream))
                        return false;
        }
        return true;
    }
    bool ok_launch() { return cudaGetLastError() == cudaSuccess; }

    bool inject(const float* feats, int T, int pos0) {
        cudaStream_t s = tg.stream;
        bool ok = true;
        for (int base_t = 0; base_t < T; base_t += 64) {
            const int n = std::min(64, T - base_t);
            const float* f = feats + size_t(base_t) * 4 * n_embd;
            for (Layer& l : L) {
                mix_norm_k<<<n, 256, 0, s>>>(f, l.ctx_mix, need, l.ctx_norm, ig, n_embd, eps);
                ok = ok_launch() && ok;
                ok = mm(l.wkc, l.tkc, ig, ik, kvw, n_embd, n) && mm(l.wvc, l.tvc, ig, iv, kvw, n_embd, n) && ok;
                head_norm_rope_k<<<dim3(n_kv, n), kHD, 0, s>>>(ik, l.kc_norm, n_kv, pos0 + base_t, theta, eps, true);
                store_ctx_k<<<dim3((kvw + 255) / 256, n), 256, 0, s>>>(ik, iv, l.kc, l.vc, kvw, pos0 + base_t, l.cap);
                ok = ok_launch() && ok;
            }
        }
        // the Markov head's context state: the last row only
        mix_norm_k<<<1, 256, 0, s>>>(feats + size_t(T - 1) * 4 * n_embd, mk_mix, need, mk_norm, frnn, n_embd, eps);
        ok = ok_launch() && ok;
        if (ok) last = pos0 + T - 1;
        return ok;
    }

    int draft(int32_t anchor, int pos, int kk, int32_t* out, float* probs) {
        if (kk < 1 || last != pos - 1 || pos < 1) return 0;
        kk = std::min(kk, kB - 1);
        cudaStream_t s = tg.stream;
        const size_t es = size_t(n_embd);
        bool ok = true;
        pinned[0] = anchor;
        ok = cudaMemcpyAsync(d_anchor, pinned, sizeof(int32_t), cudaMemcpyHostToDevice, s) == cudaSuccess;
        // the block's inputs
        ok = embed_row(tg.tok_type, tg.tok_embd, d_anchor, x, n_embd, s) && ok;
        rms_rows_k<<<1, 256, 0, s>>>(x, tok_norm, x, n_embd, eps);
        ok = cudaMemcpyAsync(x + es, mask, size_t(kB - 1) * es * sizeof(float), cudaMemcpyDeviceToDevice, s) ==
                 cudaSuccess &&
             ok;
        const int qw = n_head * kHD;
        for (Layer& l : L) {
            rms_rows_k<<<kB, 256, 0, s>>>(x, l.attn_norm, a, n_embd, eps);
            ok = mm(l.wq, l.tq, a, q, qw, n_embd, kB) && mm(l.wk, l.tk, a, k, kvw, n_embd, kB) &&
                 mm(l.wv, l.tv, a, v, kvw, n_embd, kB) && ok;
            head_norm_rope_k<<<dim3(n_head, kB), kHD, 0, s>>>(q, l.q_norm, n_head, pos, theta, eps, true);
            head_norm_rope_k<<<dim3(n_kv, kB), kHD, 0, s>>>(k, l.k_norm, n_kv, pos, theta, eps, true);
            // the context keys this block sees: before pos, within the window of the anchor (sliding layers) and
            // within what the ring still holds
            int lo = std::max(0, pos - (l.cap - kB));
            if (l.window > 0) lo = std::max(lo, pos - l.window + 1);
            const int hi = pos - 1;
            const int nc = (hi >= lo ? (hi - lo) / kCh + 1 : 0) + 1;
            attn_part_k<<<dim3(nc, n_kv, kB), 128, 0, s>>>(q, l.kc, l.vc, l.cap, lo, hi, k, v, scratch, nc, n_head,
                                                           n_kv, 1.f / std::sqrt(float(kHD)));
            attn_merge_k<<<dim3(kB, n_head), kHD, 0, s>>>(scratch, att, nc, n_head);
            ok = ok_launch() && ok;
            ok = mm(l.wo, l.to, att, o, n_embd, qw, kB) && ok;
            add_k<<<(kB * n_embd + 255) / 256, 256, 0, s>>>(x, o, kB * n_embd);
            rms_rows_k<<<kB, 256, 0, s>>>(x, l.ffn_norm, a, n_embd, eps);
            ok = mm(l.gate, l.tg, a, g, n_ff, n_embd, kB) && mm(l.up, l.tu, a, u, n_ff, n_embd, kB) && ok;
            silu_mul_k<<<(kB * n_ff + 255) / 256, 256, 0, s>>>(g, u, h, kB * n_ff);
            ok = mm(l.down, l.td, h, o, n_embd, n_ff, kB) && ok;
            add_k<<<(kB * n_embd + 255) / 256, 256, 0, s>>>(x, o, kB * n_embd);
            ok = ok_launch() && ok;
        }
        rms_rows_k<<<kB, 256, 0, s>>>(x, out_norm, hn, n_embd, eps);
        // the base logits of rows 1..kk through the target's head
        ok = mm(tg.head, tg.head_type, hn + es, base, n_vocab, n_embd, kk) && ok;
        if (tg.head_scale != 1.f) scale_k<<<(kk * n_vocab + 255) / 256, 256, 0, s>>>(base, tg.head_scale, kk * n_vocab);
        // the Markov chain
        ok = cudaMemcpyAsync(prev, d_anchor, sizeof(int32_t), cudaMemcpyDeviceToDevice, s) == cudaSuccess && ok;
        ok = cudaMemsetAsync(state, 0, size_t(R) * sizeof(float), s) == cudaSuccess && ok;
        const int nb = 64;
        for (int i = 1; i <= kk; ++i) {
            joint_k<<<(2 * R + 2 * n_embd + 255) / 256, 256, 0, s>>>(state, reinterpret_cast<const uint16_t*>(w1), prev,
                                                                    hn + size_t(i) * es, frnn, joint, R, n_embd);
            ok = gemv(jw_type, jw, joint, j3, 3 * R, 2 * R + 2 * n_embd, false, s) && ok;
            gru_k<<<(R + 255) / 256, 256, 0, s>>>(j3, jb, state, mk, R);
            ok = gemv(w2_type, w2, mk, bias, n_vocab, R, false, s) && ok;
            col_argmax_part_k<<<nb, 256, 0, s>>>(base + size_t(i - 1) * n_vocab, bias, w2_b, n_vocab, softcap, pv, pi, ps);
            argmax_final_k<<<1, 32, 0, s>>>(pv, pi, ps, nb, prev, drafts + (i - 1), conf + (i - 1));
            ok = ok_launch() && ok;
        }
        ok = ok && cudaStreamSynchronize(s) == cudaSuccess &&
             cudaMemcpy(out, drafts, size_t(kk) * sizeof(int32_t), cudaMemcpyDeviceToHost) == cudaSuccess &&
             (!probs || cudaMemcpy(probs, conf, size_t(kk) * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
        return ok ? kk : 0;
    }
};

InkDraft::InkDraft() = default;
InkDraft::~InkDraft() = default;
const int* InkDraft::visits() const { return st_->visits; }
bool InkDraft::inject(const float* feats, int T, int pos0) { return st_->inject(feats, T, pos0); }
int InkDraft::draft(int32_t anchor, int pos, int k, int32_t* out, float* probs) {
    return st_->draft(anchor, pos, k, out, probs);
}
int InkDraft::last_pos() const { return st_->last; }
void InkDraft::reset() { st_->last = -1; }

std::unique_ptr<InkDraft> InkDraft::open(const std::string& path, const InkGpu& target, int full_cap, std::string* err) {
    const auto fail = [&](const std::string& m) -> std::unique_ptr<InkDraft> {
        if (err) *err = "InkDraft: " + m;
        return nullptr;
    };
    std::string e;
    auto f = gguf::File::open(path, &e);
    if (!f) return fail(e);
    const auto arch = f->get_str("general.architecture");
    if (!arch || *arch != "dflash" || !f->tensor("markov_w1.weight") || !f->tensor("blk.0.ctx_mix.weight"))
        return fail("not an Inkling-2 DSpark drafter (arch dflash with a Markov head and per-layer context mixes)");
    std::unique_ptr<InkDraft> d(new InkDraft());
    d->st_.reset(new St());
    St& S = *d->st_;
    S.tg = target.shared();
    const auto gi = [&](const char* k, int64_t def) { return f->get_int(k).value_or(def); };
    const int n_layer = int(gi("dflash.block_count", 0));
    S.n_embd = int(gi("dflash.embedding_length", 0));
    S.n_head = int(gi("dflash.attention.head_count", 0));
    S.n_kv = int(gi("dflash.attention.head_count_kv", 0));
    S.n_ff = int(gi("dflash.feed_forward_length", 0));
    S.kvw = S.n_kv * kHD;
    S.theta = float(f->get_float("dflash.rope.freq_base").value_or(1e6));
    S.eps = float(f->get_float("dflash.attention.layer_norm_rms_epsilon").value_or(1e-6));
    S.softcap = float(f->get_float("dflash.final_logit_softcapping").value_or(0.0));
    const int window = int(gi("dflash.attention.sliding_window", 0));
    if (gi("dflash.block_size", 0) != kB || n_layer <= 0 || S.n_embd != S.tg.n_embd ||
        int(gi("dflash.attention.key_length", 0)) != kHD || S.n_head % S.n_kv != 0 || S.n_head / S.n_kv > 4)
        return fail("geometry this pass does not run (block 8, head dim 128, the target's width)");
    const std::vector<int32_t> tl = f->get_i32_array("dflash.target_layers");
    if (tl.size() != 4) return fail("expected 4 target layers");
    for (int i = 0; i < 3; ++i) S.visits[i] = tl[size_t(i)];
    S.n_vocab = S.tg.n_vocab;

    const auto up = [&](const char* name, const uint8_t*& p, uint32_t* type) {
        const gguf::Tensor* t = f->tensor(name);
        if (!t) return false;
        void* dp = S.alloc(t->nbytes);
        if (!dp || cudaMemcpy(dp, t->data, t->nbytes, cudaMemcpyHostToDevice) != cudaSuccess) return false;
        p = static_cast<const uint8_t*>(dp);
        if (type) *type = t->type;
        return true;
    };
    const auto vec = [&](const std::string& name, float*& p, int64_t want) {
        const gguf::Tensor* t = f->tensor(name);
        if (!t || (want > 0 && t->n_elements() != want)) return false;
        std::vector<float> tmp(size_t(t->n_elements()));
        if (!quants::dequant_row(t->type, t->data, tmp.data(), tmp.size())) return false;
        p = S.af(tmp.size());
        return p && cudaMemcpy(p, tmp.data(), tmp.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess;
    };
    if (!vec("output_norm.weight", S.out_norm, S.n_embd) || !vec("token_embd_norm.weight", S.tok_norm, S.n_embd) ||
        !vec("mask_embd.weight", S.mask, int64_t(kB - 1) * S.n_embd) || !vec("markov_ctx_mix.weight", S.mk_mix, 4) ||
        !vec("markov_ctx_norm.weight", S.mk_norm, S.n_embd) || !vec("markov_w2.bias", S.w2_b, S.n_vocab) ||
        !vec("joint_proj.bias", S.jb, 0) || !up("markov_w1.weight", S.w1, nullptr) ||
        !up("markov_w2.weight", S.w2, &S.w2_type) || !up("joint_proj.weight", S.jw, &S.jw_type))
        return fail("missing or mis-shaped global tensor (or out of device memory)");
    const gguf::Tensor* w1t = f->tensor("markov_w1.weight");
    S.R = int(w1t->ne[0]);
    if (w1t->type != uint32_t(Type::BF16) || w1t->ne[1] != S.n_vocab || f->tensor("markov_w2.weight")->ne[1] != S.n_vocab)
        return fail("the Markov head's vocabulary is not the target's");
    S.L.resize(size_t(n_layer));
    for (int l = 0; l < n_layer; ++l) {
        St::Layer& Ly = S.L[size_t(l)];
        const auto nm = [&](const char* role) { return "blk." + std::to_string(l) + "." + role; };
        if (!up(nm("attn_q.weight").c_str(), Ly.wq, &Ly.tq) || !up(nm("attn_k.weight").c_str(), Ly.wk, &Ly.tk) ||
            !up(nm("attn_v.weight").c_str(), Ly.wv, &Ly.tv) || !up(nm("attn_output.weight").c_str(), Ly.wo, &Ly.to) ||
            !up(nm("attn_k_ctx.weight").c_str(), Ly.wkc, &Ly.tkc) || !up(nm("attn_v_ctx.weight").c_str(), Ly.wvc, &Ly.tvc) ||
            !up(nm("ffn_gate.weight").c_str(), Ly.gate, &Ly.tg) || !up(nm("ffn_up.weight").c_str(), Ly.up, &Ly.tu) ||
            !up(nm("ffn_down.weight").c_str(), Ly.down, &Ly.td) ||
            !vec(nm("attn_norm.weight"), Ly.attn_norm, S.n_embd) || !vec(nm("attn_q_norm.weight"), Ly.q_norm, kHD) ||
            !vec(nm("attn_k_norm.weight"), Ly.k_norm, kHD) || !vec(nm("attn_k_ctx_norm.weight"), Ly.kc_norm, kHD) ||
            !vec(nm("ffn_norm.weight"), Ly.ffn_norm, S.n_embd) || !vec(nm("ctx_norm.weight"), Ly.ctx_norm, S.n_embd) ||
            !vec(nm("ctx_mix.weight"), Ly.ctx_mix, 4))
            return fail("missing or mis-shaped tensor in layer " + std::to_string(l));
        // the sliding pattern [true, false, true]: the middle layer attends the whole context
        Ly.window = (n_layer == 3 && l == 1) ? 0 : window;
        Ly.cap = Ly.window > 0 ? Ly.window + 2 * kB : std::max(kCh, std::min(full_cap, S.tg.n_ctx) + kB);
        Ly.kc = static_cast<__half*>(S.alloc(size_t(Ly.cap) * S.kvw * sizeof(__half)));
        Ly.vc = static_cast<__half*>(S.alloc(size_t(Ly.cap) * S.kvw * sizeof(__half)));
        S.max_chunks = std::max(S.max_chunks, Ly.cap / kCh + 3);
    }
    const size_t es = size_t(S.n_embd);
    S.x = S.af(kB * es);
    S.a = S.af(kB * es);
    S.q = S.af(size_t(kB) * S.n_head * kHD);
    S.k = S.af(size_t(kB) * S.kvw);
    S.v = S.af(size_t(kB) * S.kvw);
    S.att = S.af(size_t(kB) * S.n_head * kHD);
    S.o = S.af(kB * es);
    S.g = S.af(size_t(kB) * S.n_ff);
    S.u = S.af(size_t(kB) * S.n_ff);
    S.h = S.af(size_t(kB) * S.n_ff);
    S.hn = S.af(kB * es);
    S.ig = S.af(64 * es);
    S.ik = S.af(size_t(64) * S.kvw);
    S.iv = S.af(size_t(64) * S.kvw);
    S.frnn = S.af(es);
    S.base = S.af(size_t(kB) * S.n_vocab);
    S.state = S.af(size_t(S.R));
    S.joint = S.af(size_t(2 * S.R) + 2 * es);
    S.j3 = S.af(size_t(3 * S.R));
    S.mk = S.af(size_t(S.R));
    S.bias = S.af(size_t(S.n_vocab));
    S.pv = S.af(64);
    S.ps = S.af(64);
    S.conf = S.af(kB);
    S.pi = static_cast<int32_t*>(S.alloc(64 * sizeof(int32_t)));
    S.scratch = S.af(size_t(kB) * S.max_chunks * S.n_head * (2 + kHD));
    S.d_anchor = static_cast<int32_t*>(S.alloc(sizeof(int32_t)));
    S.prev = static_cast<int32_t*>(S.alloc(sizeof(int32_t)));
    S.drafts = static_cast<int32_t*>(S.alloc(kB * sizeof(int32_t)));
    if (S.failed) return fail("out of device memory");
    if (cudaMallocHost(reinterpret_cast<void**>(&S.pinned), 4 * sizeof(int32_t)) != cudaSuccess)
        return fail("out of pinned memory");
    (void)kMaxNeed;
    return d;
}

}  // namespace limbo::cuda
