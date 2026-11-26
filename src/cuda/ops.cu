// Copyright (c) 2026 ITERFORM. All rights reserved.
// src/cuda/ops.cu — the small kernels of the forward pass. See ops.hpp; every one mirrors a step of
// src/step.cpp, including where it sums in double and the association of its float products.
#include "ops.hpp"

#include <cfloat>
#include <climits>

#include "ops_common.cuh"

namespace limbo::cuda::ops {

namespace {

// ---- RMS norm ---------------------------------------------------------------------------------------
__global__ void rmsnorm_kernel(const float* x, const float* w, float* y, int n, float eps) {
    __shared__ double sm[32];
    double ss = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const double v = x[i];
        ss += v * v;
    }
    ss = block_sum(ss, sm);
    __shared__ float s_inv;
    if (threadIdx.x == 0) s_inv = static_cast<float>(1.0 / sqrt(ss / n + static_cast<double>(eps)));  // one double sqrt
    __syncthreads();
    const float inv = s_inv;
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = x[i] * inv * w[i];
}

// ---- GDN --------------------------------------------------------------------------------------------
__global__ void conv_silu_kernel(const float* qkv, float* hist, const float* w, float* out, int width) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= width) return;
    const float h0 = hist[ch], h1 = hist[width + ch], h2 = hist[2 * width + ch], xn = qkv[ch];
    const float* wc = w + static_cast<int64_t>(ch) * 4;  // {kernel, channels} with the kernel contiguous
    float acc = 0.f;
    acc += wc[0] * h0;
    acc += wc[1] * h1;
    acc += wc[2] * h2;
    acc += wc[3] * xn;
    out[ch] = silu_f(acc);
    hist[ch] = h1;
    hist[width + ch] = h2;
    hist[2 * width + ch] = xn;
}

__global__ void head_norm_kernel(const float* in, float* out, int hd, float eps, float scale) {
    __shared__ double sm[32];
    const float* src = in + static_cast<int64_t>(blockIdx.x) * hd;
    float* dst = out + static_cast<int64_t>(blockIdx.x) * hd;
    double ss = 0;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        const double v = src[d];
        ss += v * v;
    }
    ss = block_sum(ss, sm);
    __shared__ float s_inv;
    if (threadIdx.x == 0) s_inv = static_cast<float>(1.0 / sqrt(ss / hd + static_cast<double>(eps))) * scale;
    __syncthreads();
    const float inv = s_inv;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) dst[d] = src[d] * inv;
}

// One block per value head, one thread per value column j: thread j owns column j of S[h] (S[i][j], i the
// key dimension), so every access to S is coalesced and the sums over i run in the reference's order. The column
// sits in 128 registers for the step (hd is 128 for the rung-1 family), so all the loads are issued at once
// instead of one dependent load per iteration.
__global__ void __launch_bounds__(128) gdn_scan_kernel(float* S, const float* qk, const float* v, const float* alpha,
                                                       const float* beta, const float* dt_bias, const float* ssm_a,
                                                       float* o, int n_k_head, int hd, int dt_rank, float scale) {
    __shared__ float ks[128], qs[128];
    const int h = blockIdx.x, j = threadIdx.x;
    const int kh = h % n_k_head;
    const int r = h % dt_rank;
    const float* qh = qk + static_cast<int64_t>(kh) * hd;
    const float* kk = qk + static_cast<int64_t>(n_k_head) * hd + static_cast<int64_t>(kh) * hd;
    qs[j] = qh[j];
    ks[j] = kk[j];
    float* s = S + static_cast<int64_t>(h) * hd * hd;
    float st[128];
#pragma unroll
    for (int i = 0; i < 128; ++i) st[i] = s[i * 128 + j];
    __syncthreads();
    // ssm_a is stored negative, so `gate` is already the negative decay and the state is scaled by exp(gate).
    const float gate = softplus_f(alpha[r] + dt_bias[r]) * ssm_a[r];
    const float a = expf(gate);
    const float b = sigmoid_f(beta[r]);
    float skj = 0.f;
#pragma unroll
    for (int i = 0; i < 128; ++i) {
        const float val = st[i] * a;
        st[i] = val;
        skj += val * ks[i];
    }
    const float d = (v[static_cast<int64_t>(h) * hd + j] - skj) * b;
    float oj = 0.f;
#pragma unroll
    for (int i = 0; i < 128; ++i) {
        const float val = st[i] + ks[i] * d;
        s[i * 128 + j] = val;
        oj += val * qs[i];
    }
    o[static_cast<int64_t>(h) * hd + j] = oj * scale;
}

__global__ void gdn_gate_kernel(const float* o, const float* z, const float* sn, float* gated, int hd, double eps,
                                bool sigmoid_z) {
    __shared__ double sm[32];
    const float* oh = o + static_cast<int64_t>(blockIdx.x) * hd;
    double ss = 0;
    for (int j = threadIdx.x; j < hd; j += blockDim.x) {
        const double v = oh[j];
        ss += v * v;
    }
    ss = block_sum(ss, sm);
    __shared__ float s_inv;
    if (threadIdx.x == 0) s_inv = static_cast<float>(1.0 / sqrt(ss / hd + eps));
    __syncthreads();
    const float inv = s_inv;
    for (int j = threadIdx.x; j < hd; j += blockDim.x) {
        const int64_t i = static_cast<int64_t>(blockIdx.x) * hd + j;
        const float normed = oh[j] * inv * sn[j];
        gated[i] = normed * (sigmoid_z ? sigmoid_f(z[i]) : silu_f(z[i]));
    }
}

// ---- full attention ---------------------------------------------------------------------------------
// NEOX partial RoPE on the first rope_dim dims: pair (i, i + rope_dim/2), theta = pos / base^(2i/rope_dim),
// computed in double and rounded to float as the reference does. The products are not fused: a*c - b*s as two
// roundings, like the host code.
__device__ __forceinline__ float rope_elem(const float* sv, int d, int pos, double base, int rope_dim) {
    const int pairs = rope_dim / 2;
    if (d >= rope_dim) return sv[d];
    const int i = d < pairs ? d : d - pairs;
    const double theta = static_cast<double>(pos) / pow(base, 2.0 * i / static_cast<double>(rope_dim));
    const float c = static_cast<float>(cos(theta)), sn = static_cast<float>(sin(theta));
    const float a = sv[i], b = sv[i + pairs];
    return d < pairs ? __fsub_rn(__fmul_rn(a, c), __fmul_rn(b, sn)) : __fadd_rn(__fmul_rn(a, sn), __fmul_rn(b, c));
}

__global__ void attn_prep_kernel(const float* qfull, const float* kbuf, const float* vbuf, const float* qn,
                                 const float* kn, float* q_out, __half* kc, __half* vc, const int* pos_dev, int vpitch,
                                 double eps, double base, int rope_dim, int n_head, int n_head_kv, int hd) {
    __shared__ double sm[32];
    __shared__ float sv[256];
    const int b = blockIdx.x, d = threadIdx.x;  // blockDim.x == hd == 256
    const int pos = *pos_dev;
    const int64_t kvw = static_cast<int64_t>(n_head_kv) * hd;
    if (b < n_head) {
        const float* src = qfull + static_cast<int64_t>(b) * 2 * hd;
        const double v = src[d];
        const double ss = block_sum(v * v, sm);
        __shared__ float s_inv;
        if (d == 0) s_inv = static_cast<float>(1.0 / sqrt(ss / hd + eps));
        __syncthreads();
        const float inv = s_inv;
        sv[d] = src[d] * inv * qn[d];
        __syncthreads();
        q_out[static_cast<int64_t>(b) * hd + d] = rope_elem(sv, d, pos, base, rope_dim);
    } else if (b < n_head + n_head_kv) {
        const int kh = b - n_head;
        const float* src = kbuf + static_cast<int64_t>(kh) * hd;
        const double v = src[d];
        const double ss = block_sum(v * v, sm);
        __shared__ float s_inv;
        if (d == 0) s_inv = static_cast<float>(1.0 / sqrt(ss / hd + eps));
        __syncthreads();
        const float inv = s_inv;
        sv[d] = src[d] * (inv * kn[d]);
        __syncthreads();
        kc[pos * kvw + static_cast<int64_t>(kh) * hd + d] = __float2half_rn(rope_elem(sv, d, pos, base, rope_dim));
    } else {
        const int vh = b - n_head - n_head_kv;
        vc[(static_cast<int64_t>(vh) * hd + d) * vpitch + pos] = __float2half_rn(vbuf[static_cast<int64_t>(vh) * hd + d]);
    }
}

// One block per query head. Warps split the cache positions for the scores (lanes split the 256 dims, 8
// each, one 16-byte load of the f16 key); the softmax is over shared memory; then thread d accumulates
// output dim d over the positions in order, as the reference does.
__global__ void attention_kernel(const float* q, const __half* kc, const __half* vc, const float* qfull, float* out,
                                 const int* pos_dev, int vpitch, int n_head_kv, int group, float scale) {
    extern __shared__ float sc[];
    __shared__ float s_q[256];
    __shared__ float sm[32];
    const int h = blockIdx.x, d = threadIdx.x, warp = d >> 5, lane = d & 31;
    const int pos = *pos_dev;
    const int kh = h / group;
    const int64_t kvw = static_cast<int64_t>(n_head_kv) * 256;
    s_q[d] = q[static_cast<int64_t>(h) * 256 + d];
    __syncthreads();

    for (int t = warp; t <= pos; t += 8) {
        const uint4 raw = *reinterpret_cast<const uint4*>(kc + t * kvw + static_cast<int64_t>(kh) * 256 + lane * 8);
        const __half2* h2 = reinterpret_cast<const __half2*>(&raw);
        float acc = 0.f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 f = __half22float2(h2[i]);
            acc += s_q[lane * 8 + 2 * i] * f.x;
            acc += s_q[lane * 8 + 2 * i + 1] * f.y;
        }
        acc = warp_sum(acc);
        if (lane == 0) sc[t] = acc * scale;
    }
    __syncthreads();

    float m = -FLT_MAX;
    for (int t = d; t <= pos; t += blockDim.x) m = fmaxf(m, sc[t]);
    m = block_max(m, sm);
    float s = 0.f;
    for (int t = d; t <= pos; t += blockDim.x) {
        const float e = expf(sc[t] - m);
        sc[t] = e;
        s += e;
    }
    s = block_sum(s, sm);
    const float inv = s > 0.f ? 1.0f / s : 0.0f;
    __syncthreads();

    float o = 0.f;
    const __half* vcol = vc + (static_cast<int64_t>(kh) * 256 + d) * vpitch;
    for (int t = 0; t <= pos; ++t) o += (sc[t] * inv) * __half2float(vcol[t]);
    const float g = sigmoid_f(qfull[static_cast<int64_t>(h) * 512 + 256 + d]);
    out[static_cast<int64_t>(h) * 256 + d] = o * g;
}

// ---- MoE --------------------------------------------------------------------------------------------
__global__ void swiglu_kernel(const float* g, const float* u, float* h, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] = silu_f(g[i]) * u[i];
}

__global__ void router_topk_kernel(const float* logits, int n, int k, int32_t* ids, float* wts) {
    __shared__ float p[1024];
    __shared__ float sm[32];
    __shared__ float sv[32];
    __shared__ int si[32];
    __shared__ float chosen[16];
    float m = -FLT_MAX;
    for (int i = threadIdx.x; i < n; i += blockDim.x) m = fmaxf(m, logits[i]);
    m = block_max(m, sm);
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float e = expf(logits[i] - m);
        p[i] = e;
        s += e;
    }
    s = block_sum(s, sm);
    const float inv = s > 0.f ? 1.0f / s : 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) p[i] *= inv;
    __syncthreads();
    for (int r = 0; r < k; ++r) {
        float bv = -2.0f;
        int bi = INT_MAX;
        for (int i = threadIdx.x; i < n; i += blockDim.x)
            if (better(p[i], i, bv, bi)) {
                bv = p[i];
                bi = i;
            }
        block_argmax(bv, bi, sv, si);
        if (threadIdx.x == 0) {
            ids[r] = bi;
            chosen[r] = bv;
            p[bi] = -1.0f;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float wsum = 0.f;
        for (int r = 0; r < k; ++r) wsum += chosen[r];
        wsum = fmaxf(wsum, 6.103515625e-05f);  // the graph's CLAMP
        for (int r = 0; r < k; ++r) wts[r] = chosen[r] / wsum;
    }
}

__global__ void shexp_gate_kernel(const float* gw, const float* post, float* sg, int n) {
    __shared__ float sm[32];
    float acc = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) acc += gw[i] * post[i];
    acc = block_sum(acc, sm);
    if (threadIdx.x == 0) *sg = sigmoid_f(acc);
}

__global__ void moe_combine_kernel(float* x, const float* dn, const float* wts, const float* shared, const float* sg,
                                   int n, int n_used) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    float acc = 0.f;
    for (int i = 0; i < n_used; ++i) acc += dn[static_cast<int64_t>(i) * n + j] * wts[i];
    x[j] += acc + shared[j] * *sg;
}

__global__ void moe_scatter_add_kernel(float* dn, const float* packed, const int32_t* miss_i, int n_miss, int n) {
    const int m = blockIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n || m >= n_miss) return;
    dn[static_cast<int64_t>(miss_i[m]) * n + j] += packed[static_cast<int64_t>(m) * n + j];
}

// ---- hyper-connections and the PLE block (qwen4exp) -------------------------------------------------
__global__ void silu_scale_kernel(const float* x, float* y, int n, float scale) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = silu_f(x[i] * scale);
}

__global__ void hc_gate_mix_kernel(const float* xn, const float* gate, float* mixed, int hc, int n, float inv_hc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = 0.f;
    for (int s = 0; s < hc; ++s) acc += xn[static_cast<int64_t>(s) * n + i] * sigmoid_f(gate[static_cast<int64_t>(s) * n + i]);
    mixed[i] = acc * inv_hc;
}

__global__ void hc_combine_kernel(float* st, const float* out, const float* inject, int hc, int n, float inv_hc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    for (int s = 0; s < hc; ++s) {
        const float w = 2.0f * sigmoid_f(inject[s] * inv_hc);
        float* row = st + static_cast<int64_t>(s) * n + i;
        *row = *row + out[i] * w;
    }
}

__global__ void stack_fill_kernel(float* st, const float* x, int hc, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    for (int s = 0; s < hc; ++s) st[static_cast<int64_t>(s) * n + i] = x[i];
}

// One block per stream: the stream's kn and qn norms, their dot in double (the reference sums in double),
// the signed-square-root gate and the value's broadcast. The norms are the reference's rmsnorm (ss in double).
__global__ void ple_gate_kernel(const float* key, const float* st, const float* val, const float* gk, const float* gq,
                                float* gated, int hc, int n, float eps) {
    __shared__ double sm[32];
    __shared__ float s_gate;
    const int s = blockIdx.x;
    const float* ks = key + static_cast<int64_t>(s) * n;
    const float* ss = st + static_cast<int64_t>(s) * n;
    float* gn = gated + static_cast<int64_t>(s) * n;
    double kn_ss = 0, qn_ss = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const double a = ks[i], b = ss[i];
        kn_ss += a * a;
        qn_ss += b * b;
    }
    kn_ss = block_sum(kn_ss, sm);
    qn_ss = block_sum(qn_ss, sm);
    const float kinv = static_cast<float>(1.0 / sqrt(kn_ss / n + static_cast<double>(eps)));
    const float qinv = static_cast<float>(1.0 / sqrt(qn_ss / n + static_cast<double>(eps)));
    // the reference normed the two rows in float and summed their products in double
    double dot = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float a = ks[i] * kinv * gk[static_cast<int64_t>(s) * n + i];
        const float b = ss[i] * qinv * gq[static_cast<int64_t>(s) * n + i];
        dot += static_cast<double>(a) * static_cast<double>(b);
    }
    dot = block_sum(dot, sm);
    if (threadIdx.x == 0) {
        const float sc = static_cast<float>(dot) * (1.0f / sqrtf(static_cast<float>(n)));
        const float mag = sqrtf(fmaxf(fabsf(sc), 1e-6f));
        const float sgn = sc > 0.0f ? 1.0f : (sc < 0.0f ? -1.0f : 0.0f);
        s_gate = sigmoid_f(sgn * mag);
    }
    __syncthreads();
    const float g = s_gate;
    for (int i = threadIdx.x; i < n; i += blockDim.x) gn[i] = val[i] * g;
}

// The dilated depthwise conv over history + the shift, one thread per channel. hist is [hist_rows][W] row-major,
// hist_rows = (kern-1) * dilation, newest last: tap k reads (kern-1-k) * dilation positions back.
__global__ void ple_conv_silu_kernel(const float* nrm, float* hist, const float* taps, float* conv, int W, int kern,
                                     int dil, int hist_rows) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= W) return;
    const float* tc = taps + static_cast<int64_t>(ch) * kern;
    float acc = 0.f;
    for (int k = 0; k < kern; ++k) {
        const int back = (kern - 1 - k) * dil;
        const float xv = back == 0 ? nrm[ch] : hist[static_cast<int64_t>(hist_rows - back) * W + ch];
        acc += tc[k] * xv;
    }
    conv[ch] = silu_f(acc);
    for (int r = 0; r < hist_rows - 1; ++r) hist[static_cast<int64_t>(r) * W + ch] = hist[static_cast<int64_t>(r + 1) * W + ch];
    if (hist_rows > 0) hist[static_cast<int64_t>(hist_rows - 1) * W + ch] = nrm[ch];
}

__global__ void add_into_kernel(float* st, const float* a, const float* b, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) st[i] += a[i] + b[i];
}

// ---- the same four ops over a window of T rows (src/cuda/gpu_rows.cu's shape) --------------------------
// Row t is contiguous, so every pointer is [T][...] and each row runs the single-row kernel's arithmetic
// unchanged -- which is what makes a window's rows equal T sequential steps.
__global__ void hc_gate_mix_rows_kernel(const float* xn, const float* gate, float* mixed, int hc, int n, int T,
                                        float inv_hc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;   // a row a grid row: every element as before
    if (i >= n || t >= T) return;
    {
        const int64_t base = static_cast<int64_t>(t) * hc * n;
        float acc = 0.f;
        for (int s = 0; s < hc; ++s)
            acc += xn[base + static_cast<int64_t>(s) * n + i] * sigmoid_f(gate[base + static_cast<int64_t>(s) * n + i]);
        mixed[static_cast<int64_t>(t) * n + i] = acc * inv_hc;
    }
}

__global__ void hc_combine_rows_kernel(float* st, const float* out, const float* inject, int hc, int n, int T,
                                       float inv_hc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i >= n || t >= T) return;
    {
        const int64_t base = static_cast<int64_t>(t) * hc * n;
        for (int s = 0; s < hc; ++s) {
            const float w = 2.0f * sigmoid_f(inject[static_cast<int64_t>(t) * hc + s] * inv_hc);
            float* row = st + base + static_cast<int64_t>(s) * n + i;
            *row = *row + out[static_cast<int64_t>(t) * n + i] * w;
        }
    }
}

// One block per (row, stream): blockIdx.x = t * hc + s.
__global__ void ple_gate_rows_kernel(const float* key, const float* st, const float* val, const float* gk,
                                     const float* gq, float* gated, int hc, int n, int T, float eps) {
    __shared__ double sm[32];
    __shared__ float s_gate;
    const int s = blockIdx.x % hc, t = blockIdx.x / hc;
    const int64_t kb = static_cast<int64_t>(t) * hc * n + static_cast<int64_t>(s) * n;
    const float* ks = key + kb;
    const float* ss = st + kb;
    const float* vs = val + static_cast<int64_t>(t) * n;
    float* gn = gated + kb;
    double kn_ss = 0, qn_ss = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const double a = ks[i], b = ss[i];
        kn_ss += a * a;
        qn_ss += b * b;
    }
    kn_ss = block_sum(kn_ss, sm);
    qn_ss = block_sum(qn_ss, sm);
    const float kinv = static_cast<float>(1.0 / sqrt(kn_ss / n + static_cast<double>(eps)));
    const float qinv = static_cast<float>(1.0 / sqrt(qn_ss / n + static_cast<double>(eps)));
    double dot = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float a = ks[i] * kinv * gk[static_cast<int64_t>(s) * n + i];
        const float b = ss[i] * qinv * gq[static_cast<int64_t>(s) * n + i];
        dot += static_cast<double>(a) * static_cast<double>(b);
    }
    dot = block_sum(dot, sm);
    if (threadIdx.x == 0) {
        const float sc = static_cast<float>(dot) * (1.0f / sqrtf(static_cast<float>(n)));
        const float mag = sqrtf(fmaxf(fabsf(sc), 1e-6f));
        const float sgn = sc > 0.0f ? 1.0f : (sc < 0.0f ? -1.0f : 0.0f);
        s_gate = sigmoid_f(sgn * mag);
    }
    __syncthreads();
    const float g = s_gate;
    for (int i = threadIdx.x; i < n; i += blockDim.x) gn[i] = vs[i] * g;
}

// The dilated depthwise conv over the window and the history, one thread per channel, the rows in time order:
// row t reads the taps and whatever history rows 0..t-1 left, exactly as T sequential steps would.
__global__ void ple_conv_silu_rows_kernel(const float* nrm, float* hist, const float* taps, float* conv, int W,
                                          int kern, int dil, int hist_rows, int T) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= W) return;
    const float* tc = taps + static_cast<int64_t>(ch) * kern;
    for (int t = 0; t < T; ++t) {
        const float* row = nrm + static_cast<int64_t>(t) * W;
        float acc = 0.f;
        for (int k = 0; k < kern; ++k) {
            const int back = (kern - 1 - k) * dil;
            const float xv = back == 0 ? row[ch] : hist[static_cast<int64_t>(hist_rows - back) * W + ch];
            acc += tc[k] * xv;
        }
        conv[static_cast<int64_t>(t) * W + ch] = silu_f(acc);
        for (int r = 0; r < hist_rows - 1; ++r)
            hist[static_cast<int64_t>(r) * W + ch] = hist[static_cast<int64_t>(r + 1) * W + ch];
        if (hist_rows > 0) hist[static_cast<int64_t>(hist_rows - 1) * W + ch] = row[ch];
    }
}

// ---- head -------------------------------------------------------------------------------------------
__global__ void argmax_kernel(const float* logits, int n, int32_t* tok) {
    __shared__ float sv[32];
    __shared__ int si[32];
    float bv = -INFINITY;
    int bi = INT_MAX;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = logits[i];
        if (v > bv) {
            bv = v;
            bi = i;
        }
    }
    block_argmax(bv, bi, sv, si);
    if (threadIdx.x == 0) *tok = bi;
}

// One row a block: the same reduction as argmax_kernel, on row blockIdx.x of a [T][n] tile.
__global__ void argmax_rows_kernel(const float* logits, int n, int32_t* out) {
    __shared__ float sv[32];
    __shared__ int si[32];
    const float* row = logits + static_cast<int64_t>(blockIdx.x) * n;
    float bv = -INFINITY;
    int bi = INT_MAX;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = row[i];
        if (v > bv) {
            bv = v;
            bi = i;
        }
    }
    block_argmax(bv, bi, sv, si);
    if (threadIdx.x == 0) out[blockIdx.x] = bi;
}


}  // namespace

void rmsnorm(const float* x, const float* w, float* y, int n, float eps, cudaStream_t s) {
    rmsnorm_kernel<<<1, 256, 0, s>>>(x, w, y, n, eps);
}

void conv_silu(const float* qkv, float* hist, const float* w, float* out, int width, cudaStream_t s) {
    conv_silu_kernel<<<(width + 255) / 256, 256, 0, s>>>(qkv, hist, w, out, width);
}

void head_norm(const float* in, float* out, int n_heads, int hd, float eps, float scale, cudaStream_t s) {
    head_norm_kernel<<<n_heads, 128, 0, s>>>(in, out, hd, eps, scale);
}

void gdn_scan(float* S, const float* qk, const float* v, const float* alpha, const float* beta, const float* dt_bias,
              const float* ssm_a, float* o, int n_v_head, int n_k_head, int hd, int dt_rank, float scale,
              cudaStream_t s) {
    gdn_scan_kernel<<<n_v_head, hd, 0, s>>>(S, qk, v, alpha, beta, dt_bias, ssm_a, o, n_k_head, hd, dt_rank, scale);
}

void gdn_gate(const float* o, const float* z, const float* sn, float* gated, int n_heads, int hd, double eps,
              cudaStream_t s, bool sigmoid_z) {
    gdn_gate_kernel<<<n_heads, 128, 0, s>>>(o, z, sn, gated, hd, eps, sigmoid_z);
}

void attn_prep(const float* qfull, const float* kbuf, const float* vbuf, const float* qn, const float* kn,
               float* q_out, __half* kcache, __half* vcache, const int* pos, int vpitch, double eps, double rope_base,
               int rope_dim, int n_head, int n_head_kv, int hd, cudaStream_t s) {
    attn_prep_kernel<<<n_head + 2 * n_head_kv, hd, 0, s>>>(qfull, kbuf, vbuf, qn, kn, q_out, kcache, vcache, pos, vpitch,
                                                          eps, rope_base, rope_dim, n_head, n_head_kv, hd);
}

void attention(const float* q, const __half* kcache, const __half* vcache, const float* qfull, float* out,
               const int* pos, int max_ctx, int vpitch, int n_head, int n_head_kv, int hd, float scale, cudaStream_t s) {
    (void)hd;  // 256: the kernel is written for the rung-1 head size
    attention_kernel<<<n_head, 256, static_cast<size_t>(max_ctx) * sizeof(float), s>>>(
        q, kcache, vcache, qfull, out, pos, vpitch, n_head_kv, n_head / n_head_kv, scale);
}

void swiglu(const float* g, const float* u, float* h, int n, cudaStream_t s) {
    swiglu_kernel<<<(n + 255) / 256, 256, 0, s>>>(g, u, h, n);
}

void router_topk(const float* logits, int n_expert, int n_used, int32_t* ids, float* wts, cudaStream_t s) {
    router_topk_kernel<<<1, 256, 0, s>>>(logits, n_expert, n_used, ids, wts);
}

void shexp_gate(const float* gw, const float* post, float* sg, int n, cudaStream_t s) {
    shexp_gate_kernel<<<1, 256, 0, s>>>(gw, post, sg, n);
}

void moe_combine(float* x, const float* dn, const float* wts, const float* shared, const float* sg, int n,
                 int n_used, cudaStream_t s) {
    moe_combine_kernel<<<(n + 255) / 256, 256, 0, s>>>(x, dn, wts, shared, sg, n, n_used);
}

void moe_scatter_add(float* dn, const float* packed, const int32_t* miss_i, int n_miss, int n, cudaStream_t s) {
    if (n_miss <= 0) return;
    const dim3 grid(static_cast<unsigned>((n + 255) / 256), static_cast<unsigned>(n_miss));
    moe_scatter_add_kernel<<<grid, 256, 0, s>>>(dn, packed, miss_i, n_miss, n);
}

void argmax(const float* logits, int n, int32_t* tok, cudaStream_t s) {
    argmax_kernel<<<1, 1024, 0, s>>>(logits, n, tok);
}

void argmax_rows(const float* logits, int n, int T, int32_t* out, cudaStream_t s) {
    argmax_rows_kernel<<<T, 1024, 0, s>>>(logits, n, out);
}

void silu_scale(const float* x, float* y, int n, float scale, cudaStream_t s) {
    silu_scale_kernel<<<(n + 255) / 256, 256, 0, s>>>(x, y, n, scale);
}

void hc_gate_mix(const float* xn, const float* gate, float* mixed, int hc, int n, float inv_hc, cudaStream_t s) {
    hc_gate_mix_kernel<<<(n + 255) / 256, 256, 0, s>>>(xn, gate, mixed, hc, n, inv_hc);
}

void hc_combine(float* st, const float* out, const float* inject, int hc, int n, float inv_hc, cudaStream_t s) {
    hc_combine_kernel<<<(n + 255) / 256, 256, 0, s>>>(st, out, inject, hc, n, inv_hc);
}

void stack_fill(float* st, const float* x, int hc, int n, cudaStream_t s) {
    stack_fill_kernel<<<(n + 255) / 256, 256, 0, s>>>(st, x, hc, n);
}

void ple_gate(const float* key, const float* st, const float* val, const float* gk, const float* gq, float* gated,
              int hc, int n, float eps, cudaStream_t s) {
    ple_gate_kernel<<<hc, 256, 0, s>>>(key, st, val, gk, gq, gated, hc, n, eps);
}

void ple_conv_silu(const float* nrm, float* hist, const float* taps, float* conv, int W, int kern, int dil,
                   cudaStream_t s) {
    ple_conv_silu_kernel<<<(W + 255) / 256, 256, 0, s>>>(nrm, hist, taps, conv, W, kern, dil, (kern - 1) * dil);
}

void add_into(float* st, const float* a, const float* b, int n, cudaStream_t s) {
    add_into_kernel<<<(n + 255) / 256, 256, 0, s>>>(st, a, b, n);
}

void hc_gate_mix_rows(const float* xn, const float* gate, float* mixed, int hc, int n, int T, float inv_hc,
                      cudaStream_t s) {
    hc_gate_mix_rows_kernel<<<dim3(unsigned((n + 255) / 256), unsigned(T)), 256, 0, s>>>(xn, gate, mixed, hc, n, T, inv_hc);
}

void hc_combine_rows(float* st, const float* out, const float* inject, int hc, int n, int T, float inv_hc,
                     cudaStream_t s) {
    hc_combine_rows_kernel<<<dim3(unsigned((n + 255) / 256), unsigned(T)), 256, 0, s>>>(st, out, inject, hc, n, T, inv_hc);
}

void ple_gate_rows(const float* key, const float* st, const float* val, const float* gk, const float* gq, float* gated,
                   int hc, int n, int T, float eps, cudaStream_t s) {
    ple_gate_rows_kernel<<<static_cast<unsigned>(T * hc), 256, 0, s>>>(key, st, val, gk, gq, gated, hc, n, T, eps);
}

void ple_conv_silu_rows(const float* nrm, float* hist, const float* taps, float* conv, int W, int kern, int dil,
                        int T, cudaStream_t s) {
    const int hist_rows = (kern - 1) * dil;
    ple_conv_silu_rows_kernel<<<(W + 255) / 256, 256, 0, s>>>(nrm, hist, taps, conv, W, kern, dil, hist_rows, T);
}

}  // namespace limbo::cuda::ops
