// Copyright (c) 2026 ITERFORM. All rights reserved.
// src/cuda/ops_rows.cu — the batched (prompt prefill) forms of ops.hpp: the same arithmetic over T rows.
// Owner: the prefill agent. Decode kernels live in ops.cu.
#include "ops.hpp"

#include <cfloat>
#include <type_traits>
#include <climits>
#include <cstdint>

#include "limbo/cuda/kv.hpp"
#include "ops_common.cuh"

namespace limbo::cuda::ops {

namespace {

// ---- batched kernels (prefill) ----------------------------------------------------------------------
__global__ void rmsnorm_rows_kernel(const float* x, const float* w, float* y, int n, float eps) {
    __shared__ double sm[32];
    const float* xr = x + static_cast<int64_t>(blockIdx.x) * n;
    float* yr = y + static_cast<int64_t>(blockIdx.x) * n;
    double ss = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const double v = xr[i];
        ss += v * v;
    }
    ss = block_sum(ss, sm);
    __shared__ float s_inv;
    if (threadIdx.x == 0) s_inv = static_cast<float>(1.0 / sqrt(ss / n + static_cast<double>(eps)));
    __syncthreads();
    const float inv = s_inv;
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * inv * w[i];
}

// One thread per (token, channel): the 4 taps are rows t-3..t of the qkv sequence, rows before token 0 come from
// the history; the products are added in the decode order, so every output is the decode kernel's.
__global__ void conv_silu_rows_kernel(const float* qkv, const float* hist, const float* w, float* out, int width, int T, int t0) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = t0 + static_cast<int>(blockIdx.y);
    if (ch >= width) return;
    auto in = [&](int u) -> float { return u >= 0 ? qkv[static_cast<int64_t>(u) * width + ch] : hist[static_cast<int64_t>(3 + u) * width + ch]; };
    const float* wc = w + static_cast<int64_t>(ch) * 4;
    float acc = 0.f;
    acc += wc[0] * in(t - 3);
    acc += wc[1] * in(t - 2);
    acc += wc[2] * in(t - 1);
    acc += wc[3] * in(t);
    out[static_cast<int64_t>(t) * width + ch] = silu_f(acc);
    (void)T;
}

// The history after the chunk: its last three input rows (older rows from the old history when T < 3).
__global__ void conv_hist_kernel(const float* qkv, float* hist, int width, int T) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= width) return;
    float h[3];
    for (int k = 0; k < 3; ++k) {
        const int u = T - 3 + k;
        h[k] = u >= 0 ? qkv[static_cast<int64_t>(u) * width + ch] : hist[static_cast<int64_t>(3 + u) * width + ch];
    }
    for (int k = 0; k < 3; ++k) hist[static_cast<int64_t>(k) * width + ch] = h[k];
}

// One warp per 128-wide row: the sum of squares in double over the lane's four elements, then a shuffle tree.
__global__ void head_norm_rows_kernel(const float* in, int in_stride, float* out, int out_stride, int n_heads, int T,
                                      float eps, float scale) {
    const int w = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (w >= n_heads * T) return;
    const int t = w / n_heads, h = w % n_heads;
    const float4 v = *reinterpret_cast<const float4*>(in + static_cast<int64_t>(t) * in_stride + static_cast<int64_t>(h) * 128 + lane * 4);
    double ss = static_cast<double>(v.x) * v.x + static_cast<double>(v.y) * v.y + static_cast<double>(v.z) * v.z +
                static_cast<double>(v.w) * v.w;
    ss = warp_sum(ss);
    float inv = 0.f;
    if (lane == 0) inv = static_cast<float>(1.0 / sqrt(ss / 128 + static_cast<double>(eps))) * scale;  // double sqrt: once per row
    inv = __shfl_sync(0xffffffffu, inv, 0);
    *reinterpret_cast<float4*>(out + static_cast<int64_t>(t) * out_stride + static_cast<int64_t>(h) * 128 + lane * 4) =
        make_float4(v.x * inv, v.y * inv, v.z * inv, v.w * inv);
}

// gated[i] = (o[i] * inv_rms(o_head) * sn[j]) * silu(z[i]), one warp per 128-wide (token, head) row.
__global__ void gdn_gate_rows_kernel(const float* o, const float* z, const float* sn, float* gated, int rows, double eps,
                                     bool sigmoid_z) {
    const int w = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (w >= rows) return;
    const int64_t base = static_cast<int64_t>(w) * 128 + lane * 4;
    const float4 ov = *reinterpret_cast<const float4*>(o + base), zv = *reinterpret_cast<const float4*>(z + base);
    const float4 sv = *reinterpret_cast<const float4*>(sn + lane * 4);
    double ss = static_cast<double>(ov.x) * ov.x + static_cast<double>(ov.y) * ov.y + static_cast<double>(ov.z) * ov.z +
                static_cast<double>(ov.w) * ov.w;
    ss = warp_sum(ss);
    float inv = 0.f;
    if (lane == 0) inv = static_cast<float>(1.0 / sqrt(ss / 128 + eps));
    inv = __shfl_sync(0xffffffffu, inv, 0);
    float4 r;
    r.x = (ov.x * inv * sv.x) * (sigmoid_z ? sigmoid_f(zv.x) : silu_f(zv.x));
    r.y = (ov.y * inv * sv.y) * (sigmoid_z ? sigmoid_f(zv.y) : silu_f(zv.y));
    r.z = (ov.z * inv * sv.z) * (sigmoid_z ? sigmoid_f(zv.z) : silu_f(zv.z));
    r.w = (ov.w * inv * sv.w) * (sigmoid_z ? sigmoid_f(zv.w) : silu_f(zv.w));
    *reinterpret_cast<float4*>(gated + base) = r;
}

// The per-token decay and gate of the scan, for every (token, value-head group) at once: alpha[t][r] becomes
// exp(softplus(alpha + dt_bias) * ssm_a), beta[t][r] becomes sigmoid(beta). The scan kernel computed these itself,
// 128 threads redundantly, on its critical path; the values are the same function of the same inputs.
__global__ void gdn_coef_rows_kernel(float* alpha, float* beta, int ab_stride, const float* dt_bias, const float* ssm_a,
                                     int dt_rank) {
    const int t = blockIdx.x, r = threadIdx.x;
    if (r >= dt_rank) return;
    const int64_t at = static_cast<int64_t>(t) * ab_stride + r;
    const float gate = softplus_f(alpha[at] + dt_bias[r]) * ssm_a[r];
    alpha[at] = expf(gate);
    beta[at] = sigmoid_f(beta[at]);
}

// The delta-rule scan over T tokens for one value head: thread j owns column j of S in 128 registers, the
// per-token arithmetic is gdn_scan_kernel's, statement for statement (so every result is the decode kernel's bit
// for bit). What differs is the schedule. One token is two dependent chains of 128 FMAs (skj = S^T k, then
// o = S^T q) and the chain of token t+1 needs only the updated element i of token t, not o_t: so the loop that
// finishes token t (state update, o_t) also starts token t+1 (decay, skj_{t+1}), the two chains running side by
// side. A token costs one chain latency instead of two. The decay, gate and sigmoid of a token are computed one
// token ahead, off the critical path; k and q live in shared memory rings, one barrier per token.
__global__ void __launch_bounds__(128) gdn_scan_rows_kernel(float* S, const float* qk, int qk_stride, const float* v,
                                                            int v_stride, const float* alpha, const float* beta,
                                                            int ab_stride, float* o, int o_stride, int T, int n_k_head,
                                                            int hd, int dt_rank, float scale, bool write_back, float* S_out) {
    __shared__ __align__(16) float ks[3][128], qs[2][128];
    const int h = blockIdx.x, j = threadIdx.x;
    const int kh = h % n_k_head;
    const int r = h % dt_rank;
    float* sp = S + static_cast<int64_t>(h) * hd * hd;
    float st[128];
#pragma unroll
    for (int i = 0; i < 128; ++i) st[i] = sp[i * hd + j];
    auto kld = [&](int t) { return qk[static_cast<int64_t>(t) * qk_stride + static_cast<int64_t>(n_k_head) * hd + static_cast<int64_t>(kh) * hd + j]; };
    auto qld = [&](int t) { return qk[static_cast<int64_t>(t) * qk_stride + static_cast<int64_t>(kh) * hd + j]; };
    // alpha and beta hold the decay and the sigmoid already (gdn_coef_rows)
    auto coef = [&](int t, float& a, float& b, float& vj) {
        a = alpha[static_cast<int64_t>(t) * ab_stride + r];
        b = beta[static_cast<int64_t>(t) * ab_stride + r];
        vj = v[static_cast<int64_t>(t) * v_stride + static_cast<int64_t>(h) * hd + j];
    };
    ks[0][j] = kld(0);
    qs[0][j] = qld(0);
    if (T > 1) ks[1][j] = kld(1);
    float nq = T > 1 ? qld(1) : 0.f, nk = T > 2 ? kld(2) : 0.f;  // q of token t+1 and k of token t+2, staged at token t
    float a0, b0, vj0;
    coef(0, a0, b0, vj0);
    float an = 0.f, bn = 0.f, vn = 0.f;  // token t+1's decay, sigmoid and v
    if (T > 1) coef(1, an, bn, vn);
    __syncthreads();
    // token 0: decay and the first chain
    float skj = 0.f;
    {
        const float* kq = ks[0];
#pragma unroll
        for (int i = 0; i < 128; ++i) {
            const float val = st[i] * a0;
            st[i] = val;
            skj += val * kq[i];
            if ((i & 15) == 15) asm volatile("" ::: "memory");
        }
    }
    float d = (vj0 - skj) * b0;
    for (int t = 0; t + 1 < T; ++t) {
        // stage the next iteration's q and k, and its coefficients, while this one computes
        qs[(t + 1) & 1][j] = nq;
        if (t + 2 < T) ks[(t + 2) % 3][j] = nk;
        nq = t + 2 < T ? qld(t + 2) : 0.f;
        nk = t + 3 < T ? kld(t + 3) : 0.f;
        float an2 = 0.f, bn2 = 0.f, vn2 = 0.f;
        if (t + 2 < T) coef(t + 2, an2, bn2, vn2);
        const float* kt = ks[t % 3];
        const float* kn = ks[(t + 1) % 3];
        const float* qt = qs[t & 1];
        float oj = 0.f, sk = 0.f;
#pragma unroll
        for (int i = 0; i < 128; ++i) {
            const float val = st[i] + kt[i] * d;
            oj += val * qt[i];
            const float v2 = val * an;
            st[i] = v2;
            sk += v2 * kn[i];
            if ((i & 15) == 15) asm volatile("" ::: "memory");
        }
        o[static_cast<int64_t>(t) * o_stride + static_cast<int64_t>(h) * hd + j] = oj * scale;
        d = (vn - sk) * bn;
        an = an2;
        bn = bn2;
        vn = vn2;
        __syncthreads();
    }
    {
        const float* kt = ks[(T - 1) % 3];
        const float* qt = qs[(T - 1) & 1];
        float oj = 0.f;
#pragma unroll
        for (int i = 0; i < 128; ++i) {
            const float val = st[i] + kt[i] * d;
            st[i] = val;
            oj += val * qt[i];
            if ((i & 15) == 15) asm volatile("" ::: "memory");
        }
        o[static_cast<int64_t>(T - 1) * o_stride + static_cast<int64_t>(h) * hd + j] = oj * scale;
    }
    // write_back false: the state is not stored (a verify window's rows are candidates; commit replays the accepted
    // prefix). The output o above is still written -- each row's own output is needed to score the row.
    float* dst = write_back ? sp : S_out ? S_out + static_cast<int64_t>(h) * hd * hd : nullptr;
    if (!dst) return;
#pragma unroll
    for (int i = 0; i < 128; ++i) dst[i * hd + j] = st[i];
}

// cos/sin of the NEOX partial RoPE for positions pos0 .. pos0+T-1: theta in double, rounded to float, exactly as
// rope_elem computes them.
__global__ void rope_table_kernel(float* cs, int pos0, int T, double base, int rope_dim) {
    const int i = threadIdx.x, t = blockIdx.x;
    const int pairs = rope_dim / 2;
    if (i >= pairs) return;
    const double theta = static_cast<double>(pos0 + t) / pow(base, 2.0 * i / static_cast<double>(rope_dim));
    cs[(static_cast<int64_t>(t) * 2) * pairs + i] = static_cast<float>(cos(theta));
    cs[(static_cast<int64_t>(t) * 2 + 1) * pairs + i] = static_cast<float>(sin(theta));
    (void)T;
}

__device__ __forceinline__ float rope_elem_tab(const float* sv, int d, const float* cs, int rope_dim) {
    const int pairs = rope_dim / 2;
    if (d >= rope_dim) return sv[d];
    const int i = d < pairs ? d : d - pairs;
    const float c = cs[i], sn = cs[pairs + i];
    const float a = sv[i], b = sv[i + pairs];
    return d < pairs ? __fsub_rn(__fmul_rn(a, c), __fmul_rn(b, sn)) : __fadd_rn(__fmul_rn(a, sn), __fmul_rn(b, c));
}

// Paged K/V (docs/paged-kv.md): pt[c] is the pool page of chunk c (null: the chunk is its own page, the contiguous layout); a
// position's row on the pool's position axis is page * 256 + position-in-page, an f16 V page is [kv head][256 dims][256 positions].
__device__ __forceinline__ int page_of_pos(const int* pt, int pos) { return pt ? pt[pos >> 8] : pos >> 8; }
__device__ __forceinline__ int64_t pool_pos(const int* pt, int pos) { return static_cast<int64_t>(page_of_pos(pt, pos)) * 256 + (pos & 255); }

// F: the cache format (0 f16, 1 Q8, 2 Q4; include/limbo/cuda/kv.hpp). Q4 rotates the query and the key (kv_rotate256).
template <int F>
__global__ void attn_prep_rows_kernel(const float* qfull, const float* kbuf, const float* vbuf, const float* qn,
                                      const float* kn, float* q_out, __half* kc, __half* vc, int8_t* kq, uint16_t* ks, int8_t* vq,
                                      uint16_t* vs, const float* cs, int pos0, const int* pt, int vpg, int vst, double eps, int rope_dim,
                                      int n_head, int n_head_kv, int hd) {
    __shared__ double sm[32];
    __shared__ float sv[256];
    __shared__ float s_rot[256];
    const int b = blockIdx.x, d = threadIdx.x, t = blockIdx.y;  // blockDim.x == hd == 256
    const int pos = pos0 + t;
    const int64_t kvw = static_cast<int64_t>(n_head_kv) * hd;
    const int64_t qw = static_cast<int64_t>(n_head) * hd;
    if (b < n_head) {
        const float* src = qfull + static_cast<int64_t>(t) * 2 * qw + static_cast<int64_t>(b) * 2 * hd;
        const double v = src[d];
        const double ss = block_sum(v * v, sm);
        __shared__ float s_inv;
        if (d == 0) s_inv = static_cast<float>(1.0 / sqrt(ss / hd + eps));
        __syncthreads();
        const float inv = s_inv;
        sv[d] = src[d] * inv * qn[d];
        __syncthreads();
        const float qr = rope_elem_tab(sv, d, cs + static_cast<int64_t>(t) * rope_dim, rope_dim);
        q_out[static_cast<int64_t>(t) * qw + static_cast<int64_t>(b) * hd + d] = F == 2 ? kv_rotate256(qr, s_rot) : qr;
    } else if (b < n_head + n_head_kv) {
        const int kh = b - n_head;
        const float* src = kbuf + static_cast<int64_t>(t) * kvw + static_cast<int64_t>(kh) * hd;
        const double v = src[d];
        const double ss = block_sum(v * v, sm);
        __shared__ float s_inv;
        if (d == 0) s_inv = static_cast<float>(1.0 / sqrt(ss / hd + eps));
        __syncthreads();
        const float inv = s_inv;
        sv[d] = src[d] * (inv * kn[d]);
        __syncthreads();
        const float kr = rope_elem_tab(sv, d, cs + static_cast<int64_t>(t) * rope_dim, rope_dim);
        if constexpr (F == 2) {
            uint16_t dh;
            const int byte = kv_pack4_lane(kv_quant4_lane(kv_rotate256(kr, s_rot), &dh));
            const int64_t row = pool_pos(pt, pos) * n_head_kv + kh;
            if (byte >= 0) reinterpret_cast<uint8_t*>(kq)[row * (hd / 2) + (d >> 1)] = static_cast<uint8_t>(byte);
            if ((d & 31) == 0) ks[row * (hd / kKvBlock) + (d >> 5)] = dh;
        } else if constexpr (F == 1) {  // include/limbo/cuda/kv.hpp: int8 + one half scale per 32 dimensions, one row per (position, kv head)
            uint16_t dh;
            const int8_t qv = kv_quant_lane(kr, &dh);
            const int64_t row = pool_pos(pt, pos) * n_head_kv + kh;
            kq[row * hd + d] = qv;
            if ((d & 31) == 0) ks[row * (hd / kKvBlock) + (d >> 5)] = dh;
        } else {
            kc[pool_pos(pt, pos) * kvw + static_cast<int64_t>(kh) * hd + d] = __float2half_rn(kr);
        }
    } else {
        const int vh = b - n_head - n_head_kv;
        const float vv = vbuf[static_cast<int64_t>(t) * kvw + static_cast<int64_t>(vh) * hd + d];
        if constexpr (F == 2) {
            uint16_t dh;
            const int byte = kv_pack4_lane(kv_quant4_lane(kv_rotate256(vv, s_rot), &dh));
            const int64_t row = pool_pos(pt, pos) * n_head_kv + vh;
            if (byte >= 0) reinterpret_cast<uint8_t*>(vq)[row * (hd / 2) + (d >> 1)] = static_cast<uint8_t>(byte);
            if ((d & 31) == 0) vs[row * (hd / kKvBlock) + (d >> 5)] = dh;
        } else if constexpr (F == 1) {
            uint16_t dh;
            const int8_t qv = kv_quant_lane(vv, &dh);
            const int64_t row = pool_pos(pt, pos) * n_head_kv + vh;
            vq[row * hd + d] = qv;
            if ((d & 31) == 0) vs[row * (hd / kKvBlock) + (d >> 5)] = dh;
        } else {
            vc[static_cast<int64_t>(page_of_pos(pt, pos)) * vpg + (static_cast<int64_t>(vh) * hd + d) * vst + (pos & 255)] = __float2half_rn(vv);
        }
    }
}

// ---- prompt attention on the tensor cores -----------------------------------------------------------------
//
// One warp = one query head and 16 consecutive queries; a block is 4 warps = 64 queries of a head. The keys
// are walked in blocks of 64 with an online softmax. Q K^T and P V run on fp16 mma.m16n8k16 with fp32
// accumulation; the keys and values are the cache's own f16 (exact), Q and P are split into a high and a low
// half (x = hi + lo, 22 bits) and both go through the tensor cores, so the products are fp32-accurate and
// the only difference from the reference is the order of the sums.
//
// Layouts. The K cache is [pos][kv head][256] f16; the V cache is stored transposed, [kv head][256][vpitch],
// so a B fragment of P V is one 8-byte load of four consecutive positions of one dimension. The contraction
// order inside an mma is ours to choose as long as both operands agree: for Q K^T a thread holds 8 consecutive
// dimensions of a 32-dimension chunk (one 16-byte load of K), for P V four consecutive positions (the C layout
// of two S tiles is then already the A fragment, no shuffle), so the key row that column n of an S tile stands
// for is chosen to make that so.
__device__ __forceinline__ void ldsm_x4_f16(uint32_t* r, const void* p) {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
}
__device__ __forceinline__ void mma_f16(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_h2(float lo_elem, float hi_elem) {
    const __half2 h = __floats2half2_rn(lo_elem, hi_elem);
    return *reinterpret_cast<const uint32_t*>(&h);
}

constexpr int kQP = 264;  // halves per row of the staged Q: 256 + 8, so the ldmatrix rows are 528 B apart

// Paged (pt set, kbase 0, n_ctx the positions the table addresses, vpitch 256, vpg the V elements of a page): a key's row is its
// pool row, so the whole sequence is one pass over the pool with the bits of the contiguous kernel.
// The keys are read from a window: kc / vt hold the keys of absolute positions [kbase, kbase + n_ctx) (kbase a multiple of 64,
// n_ctx the rows of the window, vpitch the pitch of its transposed V), and several windows in ascending order make one
// attention: flags bit 0 = the first window (the running state starts empty), bit 1 = the last (the output is finalised).
// Between windows the unnormalised numerator lives in `out` and the running (max, sum) in `ml` [row][head][2]; the loop over key
// blocks is the same one, so the windows give the bits of a single pass. flags = 3, kbase = 0 is the plain kernel.
template <bool LO, bool PAGED>
__global__ void __launch_bounds__(128) attention_tc_kernel(const float* q, const __half* kc, const __half* vt,
                                                           const float* qfull, float* out, int pos0, int T,
                                                           int n_head, int group, int n_ctx, int vpitch, float scale,
                                                           int kbase, float* ml, int flags, const int* pt, int vpg) {
    extern __shared__ __align__(16) uint8_t smem[];
    const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int h = blockIdx.y, kh = h / group;
    const int t0 = blockIdx.x * 64 + wid * 16;
    if (t0 >= T) return;
    const int64_t qw = static_cast<int64_t>(n_head) * 256, kvw = static_cast<int64_t>(n_head / group) * 256;  // kvw: a K row of every KV head
    __half* sQh = reinterpret_cast<__half*>(smem) + static_cast<size_t>(wid) * 2 * 16 * kQP;
    __half* sQl = sQh + 16 * kQP;

    // stage Q: slot u of a 32-dimension chunk holds the dimension the A fragment of that slot stands for
    {
        const int step = lane >> 4, s = lane & 15;
        const int dim = (s < 8) ? (8 * (s >> 1) + (s & 1) + 4 * step) : (8 * ((s - 8) >> 1) + 2 + ((s - 8) & 1) + 4 * step);
        for (int r = 0; r < 16; ++r) {
            const int qr = min(t0 + r, T - 1);
            const float* src = q + static_cast<int64_t>(qr) * qw + static_cast<int64_t>(h) * 256;
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                const float x = src[32 * c + dim];
                const __half hi = __float2half_rn(x);
                sQh[r * kQP + 32 * c + lane] = hi;
                if constexpr (LO) sQl[r * kQP + 32 * c + lane] = __float2half_rn(x - __half2float(hi));
            }
        }
    }
    __syncwarp();

    const bool first = (flags & 1) != 0, last = (flags & 2) != 0;
    float o[32][4];
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
    if (first) {
#pragma unroll
        for (int j = 0; j < 32; ++j)
#pragma unroll
            for (int i = 0; i < 4; ++i) o[j][i] = 0.f;
    } else {  // the state of the windows before this one
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int qr = min(t0 + g + 8 * r, T - 1);
            const float2 st = *reinterpret_cast<const float2*>(ml + (static_cast<int64_t>(qr) * n_head + h) * 2);
            m[r] = st.x;
            l[r] = t == 0 ? st.y : 0.f;  // the quad's partial sums are added at the end: one lane carries the total
        }
#pragma unroll
        for (int j = 0; j < 32; ++j)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int qr = min(t0 + g + 8 * (i >> 1), T - 1);
                o[j][i] = out[static_cast<int64_t>(qr) * qw + static_cast<int64_t>(h) * 256 + j * 8 + 2 * t + (i & 1)];
            }
    }
    const int qpos[2] = {pos0 + t0 + g, pos0 + t0 + g + 8};
    const int kb_max = (pos0 + min(t0 + 15, T - 1)) / 64;
    const int kb_lo = kbase / 64, kb_hi = min(kb_max, (kbase + n_ctx - 1) / 64);
    if (kb_lo > kb_hi && !last) return;  // no key of this window is visible to these queries: the state stands

    for (int kb = kb_lo; kb <= kb_hi; ++kb) {
        float S[8][4];
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int i = 0; i < 4; ++i) S[j][i] = 0.f;

#pragma unroll 2
        for (int cc = 0; cc < 8; ++cc) {
            uint32_t ah[2][4], al[2][4];
#pragma unroll
            for (int st = 0; st < 2; ++st) {
                const int col = (cc * 2 + st) * 16 + (lane >> 4) * 8;
                const int row = (lane & 7) + ((lane >> 3) & 1) * 8;
                ldsm_x4_f16(ah[st], sQh + row * kQP + col);
                if constexpr (LO) ldsm_x4_f16(al[st], sQl + row * kQP + col);
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                // key row of this lane's column: tile pair (j >> 1) of 16 keys, the first tile holds keys
                // 4t'+e, the second 4t'+2+e for the column n = 2t'+e
                const int kpos = kb * 64 + (j >> 1) * 16 + 4 * (g >> 1) + 2 * (j & 1) + (g & 1);
                const int64_t krow = PAGED ? pool_pos(pt, kpos) : static_cast<int64_t>(min(kpos - kbase, n_ctx - 1));
                const uint4 kv = __ldg(reinterpret_cast<const uint4*>(kc + krow * kvw + static_cast<int64_t>(kh) * 256 + 32 * cc + 8 * t));
                mma_f16(S[j], ah[0], kv.x, kv.y);
                if constexpr (LO) mma_f16(S[j], al[0], kv.x, kv.y);
                mma_f16(S[j], ah[1], kv.z, kv.w);
                if constexpr (LO) mma_f16(S[j], al[1], kv.z, kv.w);
            }
        }

        // scale, causal mask, online softmax (rows g and g+8; the four threads of a quad share a row)
        float bm[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int kp = kb * 64 + (j >> 1) * 16 + 4 * t + 2 * (j & 1) + (i & 1);
                const int r = i >> 1;
                float v = S[j][i] * scale;
                if (kp > qpos[r]) v = -INFINITY;
                S[j][i] = v;
                bm[r] = fmaxf(bm[r], v);
            }
        float alpha[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            bm[r] = fmaxf(bm[r], __shfl_xor_sync(0xffffffffu, bm[r], 1));
            bm[r] = fmaxf(bm[r], __shfl_xor_sync(0xffffffffu, bm[r], 2));
            const float mn = fmaxf(m[r], bm[r]);
            alpha[r] = mn == -INFINITY ? 1.f : expf(m[r] - mn);
            m[r] = mn;
            l[r] *= alpha[r];
        }
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int r = i >> 1;
                const float p = S[j][i] == -INFINITY ? 0.f : expf(S[j][i] - m[r]);
                S[j][i] = p;
                l[r] += p;
            }
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            o[j][0] *= alpha[0];
            o[j][1] *= alpha[0];
            o[j][2] *= alpha[1];
            o[j][3] *= alpha[1];
        }

        // O += P V, 16 keys at a time: the A fragment is two S tiles
#pragma unroll
        for (int c16 = 0; c16 < 4; ++c16) {
            uint32_t ph[4], pl[4];
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                const int j = 2 * c16 + (w >> 1);
                const float x0 = S[j][2 * (w & 1)], x1 = S[j][2 * (w & 1) + 1];
                const __half2 hh = __floats2half2_rn(x0, x1);
                ph[w] = *reinterpret_cast<const uint32_t*>(&hh);
                if constexpr (LO) {
                    const float2 hf = __half22float2(hh);
                    pl[w] = pack_h2(x0 - hf.x, x1 - hf.y);
                } else {
                    pl[w] = 0;
                }
            }
            // fragment order a0 = (row g, slots 2t..), a1 = (row g+8, same), a2 = (row g, slots 2t+8..), a3 = (row g+8)
            const uint32_t a_hi[4] = {ph[0], ph[1], ph[2], ph[3]};
            const uint32_t a_lo[4] = {pl[0], pl[1], pl[2], pl[3]};
            // paged: a block of 64 keys is inside one page, whose V is [kv head][256 dims][256 positions]
            const __half* vrow = (PAGED ? vt + static_cast<int64_t>(page_of_pos(pt, kb * 64)) * vpg + (kb * 64 & 255) : vt + (kb * 64 - kbase)) +
                                 (static_cast<int64_t>(kh) * 256 + g) * vpitch + c16 * 16 + 4 * t;
#pragma unroll
            for (int dt = 0; dt < 32; ++dt) {
                const uint2 b = __ldg(reinterpret_cast<const uint2*>(vrow + static_cast<int64_t>(dt) * 8 * vpitch));
                mma_f16(o[dt], a_hi, b.x, b.y);
                if constexpr (LO) mma_f16(o[dt], a_lo, b.x, b.y);
            }
        }
    }

    if (!last) {  // hand the state to the next window
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            l[r] += __shfl_xor_sync(0xffffffffu, l[r], 1);
            l[r] += __shfl_xor_sync(0xffffffffu, l[r], 2);
            const int qr = t0 + g + 8 * r;
            if (t == 0 && qr < T) *reinterpret_cast<float2*>(ml + (static_cast<int64_t>(qr) * n_head + h) * 2) = make_float2(m[r], l[r]);
        }
#pragma unroll
        for (int j = 0; j < 32; ++j)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int qr = t0 + g + 8 * (i >> 1);
                if (qr < T) out[static_cast<int64_t>(qr) * qw + static_cast<int64_t>(h) * 256 + j * 8 + 2 * t + (i & 1)] = o[j][i];
            }
        return;
    }
    // the quad's partial row sums, then out = O / l * sigmoid(gate)
    float inv[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        l[r] += __shfl_xor_sync(0xffffffffu, l[r], 1);
        l[r] += __shfl_xor_sync(0xffffffffu, l[r], 2);
        inv[r] = l[r] > 0.f ? 1.0f / l[r] : 0.f;
    }
#pragma unroll
    for (int dt = 0; dt < 32; ++dt)
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int qr = t0 + g + 8 * r;
            if (qr >= T) continue;
            const int d = dt * 8 + 2 * t;
            const float2 gate = *reinterpret_cast<const float2*>(qfull + static_cast<int64_t>(qr) * 2 * qw + static_cast<int64_t>(h) * 512 + 256 + d);
            float2 res;
            res.x = o[dt][2 * r] * inv[r] * sigmoid_f(gate.x);
            res.y = o[dt][2 * r + 1] * inv[r] * sigmoid_f(gate.y);
            *reinterpret_cast<float2*>(out + static_cast<int64_t>(qr) * qw + static_cast<int64_t>(h) * 256 + d) = res;
        }
}

__global__ void router_topk_rows_kernel(const float* logits, int n, int k, int32_t* ids, float* wts) {
    __shared__ float p[1024];
    __shared__ float sm[32];
    __shared__ float sv[32];
    __shared__ int si[32];
    __shared__ float chosen[16];
    const float* lg = logits + static_cast<int64_t>(blockIdx.x) * n;
    int32_t* idt = ids + static_cast<int64_t>(blockIdx.x) * k;
    float* wt = wts + static_cast<int64_t>(blockIdx.x) * k;
    float m = -FLT_MAX;
    for (int i = threadIdx.x; i < n; i += blockDim.x) m = fmaxf(m, lg[i]);
    m = block_max(m, sm);
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float e = expf(lg[i] - m);
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
            idt[r] = bi;
            chosen[r] = bv;
            p[bi] = -1.0f;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float wsum = 0.f;
        for (int r = 0; r < k; ++r) wsum += chosen[r];
        wsum = fmaxf(wsum, 6.103515625e-05f);
        for (int r = 0; r < k; ++r) wt[r] = chosen[r] / wsum;
    }
}

__global__ void shexp_gate_rows_kernel(const float* gw, const float* post, float* sg, int n) {
    __shared__ float sm[32];
    const float* pr = post + static_cast<int64_t>(blockIdx.x) * n;
    float acc = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) acc += gw[i] * pr[i];
    acc = block_sum(acc, sm);
    if (threadIdx.x == 0) sg[blockIdx.x] = sigmoid_f(acc);
}

__global__ void moe_combine_rows_kernel(float* x, const float* dn, const int32_t* pos_of, const float* wts,
                                        const float* shared, const float* sg, int n, int n_used) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (j >= n) return;
    float acc = 0.f;
    for (int i = 0; i < n_used; ++i) acc += dn[static_cast<int64_t>(pos_of[t * n_used + i]) * n + j] * wts[t * n_used + i];
    x[static_cast<int64_t>(t) * n + j] += acc + shared[static_cast<int64_t>(t) * n + j] * sg[t];
}

// Q8 -> f16 for a window of positions [k0, k0 + W): K as [row][kv head][256], V transposed [kv head][256][wp]. Position >= valid_end
// (not written yet: the cache holds whatever it held) becomes 0 so that a masked key never multiplies garbage. One value is
// fp16(q * d): the dequantised element is exact in fp32 (7 + 11 bits) and takes one f16 rounding, the one the f16 cache has.
// grid (W / 32, n_head_kv), block 256 = the head dimension; a thread walks 32 positions of its dimension.
// Q4: false for Q8's int8 rows, true for Q4's nibble rows (the keys stay rotated, as the Q4 prep's query is; the values are
// rotated back).
template <bool Q4>
__global__ void kv_expand_kernel(const int8_t* kq, const uint16_t* ks, const int8_t* vq, const uint16_t* vs, __half* kw, __half* vw,
                                 int k0, int valid_end, int n_head_kv, int wp, const int* pt) {
    __shared__ float s_rot[256];
    const int d = threadIdx.x, kh = blockIdx.y, p0 = blockIdx.x * 32;
    const int64_t pb = pool_pos(pt, k0 + p0);  // the 32 positions of a block share a page (k0 and p0 are multiples of 32)
    __half h[32];
#pragma unroll
    for (int p = 0; p < 32; ++p) {
        const int pos = k0 + p0 + p;
        const int64_t row = (pb + p) * n_head_kv + kh;
        float kv = 0.f, vv = 0.f;
        if (pos < valid_end) {
            const float sk = __half2float(__ushort_as_half(ks[row * 8 + (d >> 5)])), sv = __half2float(__ushort_as_half(vs[row * 8 + (d >> 5)]));
            if constexpr (Q4) {
                const uint8_t* k4 = reinterpret_cast<const uint8_t*>(kq) + row * 128 + (d >> 1);
                const uint8_t* v4 = reinterpret_cast<const uint8_t*>(vq) + row * 128 + (d >> 1);
                kv = kv_nib(*k4, d & 1) * sk;
                vv = kv_nib(*v4, d & 1) * sv;
            } else {
                kv = static_cast<float>(kq[row * 256 + d]) * sk;
                vv = static_cast<float>(vq[row * 256 + d]) * sv;
            }
        }
        kw[(static_cast<int64_t>(p0 + p) * n_head_kv + kh) * 256 + d] = __float2half_rn(kv);
        if constexpr (Q4) vv = kv_unrotate256(vv, s_rot);  // every thread of the block is at the same position: uniform
        h[p] = __float2half_rn(vv);
    }
    uint4* dst = reinterpret_cast<uint4*>(vw + (static_cast<int64_t>(kh) * 256 + d) * wp + p0);
#pragma unroll
    for (int i = 0; i < 4; ++i) dst[i] = *reinterpret_cast<const uint4*>(h + 8 * i);
}

}  // namespace

void gdn_gate_rows(const float* o, const float* z, const float* sn, float* gated, int rows, int hd, double eps,
                   cudaStream_t s, bool sigmoid_z) {
    (void)hd;  // 128
    gdn_gate_rows_kernel<<<static_cast<unsigned>((rows + 3) / 4), 128, 0, s>>>(o, z, sn, gated, rows, eps, sigmoid_z);
}

void rmsnorm_rows(const float* x, const float* w, float* y, int rows, int n, float eps, cudaStream_t s) {
    rmsnorm_rows_kernel<<<rows, 256, 0, s>>>(x, w, y, n, eps);
}

void conv_silu_rows(const float* qkv, float* hist, const float* w, float* out, int width, int T, cudaStream_t s,
                    bool write_back, int t0) {
    if (T <= t0) return;
    conv_silu_rows_kernel<<<dim3(static_cast<unsigned>((width + 255) / 256), static_cast<unsigned>(T - t0)), 256, 0, s>>>(qkv, hist, w, out,
                                                                                                             width, T, t0);
    // A verify window's rows are candidates, not committed: the conv reads the history (the kernel above already
    // rolls it inside the window) but must not move it until commit says how many rows were accepted.
    if (write_back) conv_hist_kernel<<<(width + 255) / 256, 256, 0, s>>>(qkv, hist, width, T);
}

void head_norm_rows(const float* in, int in_stride, float* out, int out_stride, int T, int n_heads, int hd, float eps,
                    float scale, cudaStream_t s) {
    (void)hd;  // 128: the kernel is written for the rung-1 delta-net head size
    head_norm_rows_kernel<<<static_cast<unsigned>((n_heads * T + 3) / 4), 128, 0, s>>>(in, in_stride, out, out_stride,
                                                                                     n_heads, T, eps, scale);
}

void gdn_coef_rows(float* alpha, float* beta, int ab_stride, const float* dt_bias, const float* ssm_a, int T, int dt_rank,
                   cudaStream_t s) {
    gdn_coef_rows_kernel<<<T, (dt_rank + 31) / 32 * 32, 0, s>>>(alpha, beta, ab_stride, dt_bias, ssm_a, dt_rank);
}

void gdn_scan_rows(float* S, const float* qk, int qk_stride, const float* v, int v_stride, const float* alpha,
                   const float* beta, int ab_stride, float* o, int o_stride, int T, int n_v_head, int n_k_head, int hd,
                   int dt_rank, float scale, cudaStream_t s, bool write_back, float* S_out) {
    gdn_scan_rows_kernel<<<n_v_head, hd, 0, s>>>(S, qk, qk_stride, v, v_stride, alpha, beta, ab_stride, o, o_stride, T,
                                                 n_k_head, hd, dt_rank, scale, write_back, S_out);
}

void rope_table(float* cs, int pos0, int T, double rope_base, int rope_dim, cudaStream_t s) {
    rope_table_kernel<<<T, 64, 0, s>>>(cs, pos0, T, rope_base, rope_dim);
}

void attn_prep_rows(const float* qfull, const float* kbuf, const float* vbuf, const float* qn, const float* kn,
                    float* q_out, __half* kcache, __half* vcache, const float* rope_cs, int pos0, int T, int vpitch,
                    double eps, int rope_dim, int n_head, int n_head_kv, int hd, cudaStream_t s, const int* pt) {
    attn_prep_rows_kernel<0><<<dim3(static_cast<unsigned>(n_head + 2 * n_head_kv), static_cast<unsigned>(T)), hd, 0, s>>>(
        qfull, kbuf, vbuf, qn, kn, q_out, kcache, vcache, nullptr, nullptr, nullptr, nullptr, rope_cs, pos0, pt,
        pt ? n_head_kv * hd * 256 : 256, pt ? 256 : vpitch, eps, rope_dim, n_head, n_head_kv, hd);
}

void attn_prep_rows_q8(const float* qfull, const float* kbuf, const float* vbuf, const float* qn, const float* kn, float* q_out,
                       int8_t* kq, uint16_t* ks, int8_t* vq, uint16_t* vs, const float* rope_cs, int pos0, int T, double eps,
                       int rope_dim, int n_head, int n_head_kv, int hd, cudaStream_t s, const int* pt) {
    attn_prep_rows_kernel<1><<<dim3(static_cast<unsigned>(n_head + 2 * n_head_kv), static_cast<unsigned>(T)), hd, 0, s>>>(
        qfull, kbuf, vbuf, qn, kn, q_out, nullptr, nullptr, kq, ks, vq, vs, rope_cs, pos0, pt, 0, 0, eps, rope_dim, n_head, n_head_kv, hd);
}

void attn_prep_rows_q4(const float* qfull, const float* kbuf, const float* vbuf, const float* qn, const float* kn, float* q_out,
                       int8_t* kq, uint16_t* ks, int8_t* vq, uint16_t* vs, const float* rope_cs, int pos0, int T, double eps,
                       int rope_dim, int n_head, int n_head_kv, int hd, cudaStream_t s, const int* pt) {
    attn_prep_rows_kernel<2><<<dim3(static_cast<unsigned>(n_head + 2 * n_head_kv), static_cast<unsigned>(T)), hd, 0, s>>>(
        qfull, kbuf, vbuf, qn, kn, q_out, nullptr, nullptr, kq, ks, vq, vs, rope_cs, pos0, pt, 0, 0, eps, rope_dim, n_head, n_head_kv, hd);
}

void attention_rows(const float* q, const __half* kcache, const __half* vcache, const float* qfull, float* out,
                    int pos0, int T, int n_ctx, int vpitch, int n_head, int n_head_kv, int hd, float scale,
                    cudaStream_t s, bool exact, const int* pt) {
    attention_rows_window(q, kcache, vcache, qfull, out, pos0, T, 0, n_ctx, pt ? 256 : vpitch, n_head, n_head_kv, hd, scale, 3, nullptr, s, exact, pt);
}

void attention_rows_window(const float* q, const __half* kwin, const __half* vwin, const float* qfull, float* out, int pos0, int T,
                           int kbase, int n_rows, int vpitch, int n_head, int n_head_kv, int hd, float scale, int flags, float* ml,
                           cudaStream_t s, bool exact, const int* pt) {
    (void)hd;  // 256
    constexpr size_t smem = size_t(4) * 2 * 16 * kQP * sizeof(__half);
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(attention_tc_kernel<true, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
        cudaFuncSetAttribute(attention_tc_kernel<false, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
        cudaFuncSetAttribute(attention_tc_kernel<true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
        cudaFuncSetAttribute(attention_tc_kernel<false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
        configured = true;
    }
    const dim3 grid(static_cast<unsigned>((T + 63) / 64), static_cast<unsigned>(n_head));
    const int vpg = n_head_kv * 256 * 256;
    auto go = [&](auto lo, auto paged) {
        attention_tc_kernel<decltype(lo)::value, decltype(paged)::value><<<grid, 128, smem, s>>>(q, kwin, vwin, qfull, out, pos0, T, n_head, n_head / n_head_kv,
                                                                                               n_rows, vpitch, scale, kbase, ml, flags, pt, vpg);
    };
    using T_ = std::true_type;
    using F_ = std::false_type;
    if (pt) exact ? go(T_{}, T_{}) : go(F_{}, T_{});
    else exact ? go(T_{}, F_{}) : go(F_{}, F_{});
}

void kv_expand_window(const int8_t* kq, const uint16_t* ks, const int8_t* vq, const uint16_t* vs, __half* kw, __half* vw, int k0,
                      int n_rows, int valid_end, int n_head_kv, int wp, cudaStream_t s, const int* pt) {
    kv_expand_kernel<false><<<dim3(static_cast<unsigned>(n_rows / 32), static_cast<unsigned>(n_head_kv)), 256, 0, s>>>(kq, ks, vq, vs, kw, vw, k0,
                                                                                                                 valid_end, n_head_kv, wp, pt);
}

void kv_expand_window_q4(const int8_t* kq, const uint16_t* ks, const int8_t* vq, const uint16_t* vs, __half* kw, __half* vw, int k0,
                         int n_rows, int valid_end, int n_head_kv, int wp, cudaStream_t s, const int* pt) {
    kv_expand_kernel<true><<<dim3(static_cast<unsigned>(n_rows / 32), static_cast<unsigned>(n_head_kv)), 256, 0, s>>>(kq, ks, vq, vs, kw, vw, k0,
                                                                                                                valid_end, n_head_kv, wp, pt);
}

void router_topk_rows(const float* logits, int n_expert, int n_used, int T, int32_t* ids, float* wts, cudaStream_t s) {
    router_topk_rows_kernel<<<T, 256, 0, s>>>(logits, n_expert, n_used, ids, wts);
}

void shexp_gate_rows(const float* gw, const float* post, float* sg, int n, int T, cudaStream_t s) {
    shexp_gate_rows_kernel<<<T, 256, 0, s>>>(gw, post, sg, n);
}

void moe_combine_rows(float* x, const float* dn, const int32_t* pos_of, const float* wts, const float* shared,
                      const float* sg, int n, int n_used, int T, cudaStream_t s) {
    moe_combine_rows_kernel<<<dim3(static_cast<unsigned>((n + 255) / 256), static_cast<unsigned>(T)), 256, 0, s>>>(
        x, dn, pos_of, wts, shared, sg, n, n_used);
}

}  // namespace limbo::cuda::ops
