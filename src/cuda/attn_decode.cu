// Copyright (c) 2026 ITERFORM. All rights reserved.
// src/cuda/attn_decode.cu — full attention for the decode step and verify windows (T = 1..8 query rows).
//
// Model math (bench/rung1-graph-spec.md): q and k are RMS-normed per head, partially rotated (NEOX, first
// rope_dim dims), k and v go to the f16 cache, scores are q.k * scale in float, softmax in float, the output is
// the probability-weighted V times the sigmoid of the gate half of q_full.
//
// The attention itself is a split flash-decode (docs/donors/kindi.md 2.4): the positions are cut into chunks of
// 256 at fixed absolute boundaries, a block per (KV head, chunk, row) scores its chunk for the 8 query heads that
// share the KV head, so every K row and every V column is read from memory once for all 8 heads, and writes the
// chunk's (max, sum, numerator). A second kernel merges the chunks in position order. The chunk boundaries, the
// lane mapping and every reduction tree are fixed, so a row's result is the same whatever the number of rows in
// the launch or the length of the context beyond it.
//
// Q8/Q4 pools with a group of at most 8 take attn_tc_kernel at every row count (K/V read once per 8 row-head pairs).
//
// Paging (docs/paged-kv.md): a chunk's rows live in page pt[chunk] of the pool (pt null: the chunk is its own page, the contiguous
// layout, which the MTP block's small cache keeps). Page p owns positions [p*256, p*256+256): K and the Q8 / Q4 planes are indexed
// by p*256 + position-in-page, an f16 V page is [kv head][256 dims][256 positions] (vpg elements a page, vst a dimension row).
#include <cfloat>
#include <cmath>
#include <cstdlib>

#include "dec_ops.hpp"
#include "limbo/cuda/kv.hpp"

namespace limbo::cuda::dec {

namespace {

constexpr int kGMax = 12;    // widest query group per KV head the scorers instantiate (G = 8 and 12)
constexpr int kHD = 256;     // head dimension
constexpr int kCH = kAttnChunk;

__device__ __forceinline__ void cp_async16_a(void* dst, const void* src) {
    const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(d), "l"(src));
}
__device__ __forceinline__ void cp_async_commit_a() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_async_wait_all_a() { asm volatile("cp.async.wait_group 0;\n" ::); }

__device__ __forceinline__ float sigmoid_d(float x) { return 1.0f / (1.0f + expf(-x)); }

template <typename T>
__device__ __forceinline__ T warp_sum_d(T v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max_d(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

template <typename T>
__device__ __forceinline__ T block_sum_d(T v, T* sm) {
    v = warp_sum_d(v);
    if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = v;
    __syncthreads();
    T r = 0;
    for (int i = 0; i < static_cast<int>(blockDim.x >> 5); ++i) r += sm[i];
    __syncthreads();
    return r;
}

// The page of a position (pt null: the chunk index itself) and its row on the pool's position axis.
__device__ __forceinline__ int page_of(const int* pt, int pos) { return pt ? pt[pos / kCH] : pos / kCH; }
__device__ __forceinline__ int64_t page_pos(const int* pt, int pos) { return static_cast<int64_t>(page_of(pt, pos)) * kCH + pos % kCH; }

// ---- preparation: the same arithmetic as ops::attn_prep, for T rows ---------------------------------------
__device__ __forceinline__ float rope_elem_d(const float* sv, int d, int pos, double base, int rope_dim) {
    const int pairs = rope_dim / 2;
    if (d >= rope_dim) return sv[d];
    const int i = d < pairs ? d : d - pairs;
    const double theta = static_cast<double>(pos) / pow(base, 2.0 * i / static_cast<double>(rope_dim));
    const float c = static_cast<float>(cos(theta)), sn = static_cast<float>(sin(theta));
    const float a = sv[i], b = sv[i + pairs];
    return d < pairs ? __fsub_rn(__fmul_rn(a, c), __fmul_rn(b, sn)) : __fadd_rn(__fmul_rn(a, sn), __fmul_rn(b, c));
}

// grid (n_head + 2 * n_head_kv, T), block hd (256).
__global__ void attn_prep_kernel(const float* qfull, const float* kbuf, const float* vbuf, const float* qn, const float* kn,
                                 float* q_out, __half* kc, __half* vc, const int* pos0_dev, const int* pt, int vpg, int vst,
                                 double eps, double base, int rope_dim, int n_head, int n_head_kv, int hd) {
    __shared__ double sm[32];
    __shared__ float sv[256];
    __shared__ float s_inv;
    const int b = blockIdx.x, t = blockIdx.y, d = threadIdx.x;  // blockDim.x == hd == 256
    const int pos = *pos0_dev + t;
    const int64_t kvw = static_cast<int64_t>(n_head_kv) * hd;
    if (b < n_head) {
        const float* src = qfull + (static_cast<int64_t>(t) * n_head + b) * 2 * hd;
        const double v = src[d];
        const double ss = block_sum_d(v * v, sm);
        if (d == 0) s_inv = static_cast<float>(1.0 / sqrt(ss / hd + eps));
        __syncthreads();
        const float inv = s_inv;
        sv[d] = src[d] * inv * qn[d];
        __syncthreads();
        q_out[(static_cast<int64_t>(t) * n_head + b) * hd + d] = rope_elem_d(sv, d, pos, base, rope_dim);
    } else if (b < n_head + n_head_kv) {
        const int kh = b - n_head;
        const float* src = kbuf + t * kvw + static_cast<int64_t>(kh) * hd;
        const double v = src[d];
        const double ss = block_sum_d(v * v, sm);
        if (d == 0) s_inv = static_cast<float>(1.0 / sqrt(ss / hd + eps));
        __syncthreads();
        const float inv = s_inv;
        sv[d] = src[d] * (inv * kn[d]);
        __syncthreads();
        kc[page_pos(pt, pos) * kvw + static_cast<int64_t>(kh) * hd + d] = __float2half_rn(rope_elem_d(sv, d, pos, base, rope_dim));
    } else {
        const int vh = b - n_head - n_head_kv;
        vc[static_cast<int64_t>(page_of(pt, pos)) * vpg + (static_cast<int64_t>(vh) * hd + d) * vst + pos % kCH] =
            __float2half_rn(vbuf[t * kvw + static_cast<int64_t>(vh) * hd + d]);
    }
}

// ---- split flash-decode -------------------------------------------------------------------------------------

__device__ __forceinline__ void unpack8(const uint4& r, float* f) {
    const __half2* h = reinterpret_cast<const __half2*>(&r);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 t = __half22float2(h[i]);
        f[2 * i] = t.x;
        f[2 * i + 1] = t.y;
    }
}

// Butterfly reduce-scatter of 32 values per lane over the warp: lane L returns the sum over the warp of value L.
// 31 shuffles and 31 adds for 32 sums, against 160 shuffles for 32 separate warp reductions.
__device__ __forceinline__ float reduce_scatter32(const float (&v)[32], int lane) {
    float a[16], b[8], c[4], d[2];
    {
        const bool hi = (lane & 16) != 0;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const float keep = hi ? v[i + 16] : v[i], send = hi ? v[i] : v[i + 16];
            a[i] = keep + __shfl_xor_sync(0xffffffffu, send, 16);
        }
    }
    {
        const bool hi = (lane & 8) != 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float keep = hi ? a[i + 8] : a[i], send = hi ? a[i] : a[i + 8];
            b[i] = keep + __shfl_xor_sync(0xffffffffu, send, 8);
        }
    }
    {
        const bool hi = (lane & 4) != 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float keep = hi ? b[i + 4] : b[i], send = hi ? b[i] : b[i + 4];
            c[i] = keep + __shfl_xor_sync(0xffffffffu, send, 4);
        }
    }
    {
        const bool hi = (lane & 2) != 0;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const float keep = hi ? c[i + 2] : c[i], send = hi ? c[i] : c[i + 2];
            d[i] = keep + __shfl_xor_sync(0xffffffffu, send, 2);
        }
    }
    const bool hi = (lane & 1) != 0;
    const float keep = hi ? d[1] : d[0], send = hi ? d[0] : d[1];
    return keep + __shfl_xor_sync(0xffffffffu, send, 1);
}

// grid (n_chunks_max, n_head_kv, T), block 256 (8 warps). Scratch layout: po[T][n_head][n_chunks_max][256] then
// pml[T][n_head][n_chunks_max][2].
template <int G, int GR = G>  // head slots per KV head: 8 (rung 1) or 12 (qwen4exp); GR of them real (6 for the 27Bs)
__global__ void __launch_bounds__(256, 2)
    attn_part_kernel(const float* __restrict__ q, const __half* __restrict__ kc, const __half* __restrict__ vc, float* po,
                     float* pml, const int* pos0_dev, int n_chunks_max, const int* pt, int vpg, int vst, float scale, int n_head,
                     int n_head_kv) {
    __shared__ __align__(16) float s_p[8][kCH];  // one band of 8 heads
    const int t = blockIdx.x, kh = blockIdx.y, c = blockIdx.z;   // rows fastest: a chunk's rows run together and share it in L2
    const int grp = n_head / n_head_kv;  // query heads per KV head (<= G; the slots past it are zero and not written)
    const int pos = *pos0_dev + t;
    const int start = c * kCH;
    if (start > pos) return;
    const int len = min(kCH, pos + 1 - start);
    const int pg = pt ? pt[c] : c, pbase = pg * kCH;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t kvw = static_cast<int64_t>(n_head_kv) * kHD;

    // The query dims of every head of the group load once; the three phases below then run per band of 8
    // heads (reduce_scatter32 scatters 4 positions x 8 heads across the warp's 32 lanes), so a group of 12
    // is two bands and the arithmetic of a group of 8 is unchanged.
    float qr[G][8];
    {
        const float* qb = q + (static_cast<int64_t>(t) * n_head + kh * grp) * kHD + lane * 8;
#pragma unroll
        for (int h = 0; h < G; ++h) {
            const float4 a = (h < grp ? *reinterpret_cast<const float4*>(qb + h * kHD) : make_float4(0.f, 0.f, 0.f, 0.f)), b = (h < grp ? *reinterpret_cast<const float4*>(qb + h * kHD + 4) : make_float4(0.f, 0.f, 0.f, 0.f));
            qr[h][0] = a.x; qr[h][1] = a.y; qr[h][2] = a.z; qr[h][3] = a.w;
            qr[h][4] = b.x; qr[h][5] = b.y; qr[h][6] = b.z; qr[h][7] = b.w;
        }
    }
    for (int h0 = 0; h0 < G; h0 += 8) {
    // ---- scores: this lane owns dims 8*lane .. 8*lane+7 of every K row, a warp scores 4 positions per pass
    {
        const __half* kbase = kc + static_cast<int64_t>(kh) * kHD + lane * 8;
        uint4 kn[4];
        auto load_quad = [&](int qd, uint4 (&dst)[4]) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int pl = warp * 32 + qd * 4 + i;
                dst[i] = pl < len ? __ldg(reinterpret_cast<const uint4*>(kbase + (pbase + pl) * kvw)) : make_uint4(0, 0, 0, 0);
            }
        };
        load_quad(0, kn);
#pragma unroll 1
        for (int qd = 0; qd < 8; ++qd) {
            uint4 kr[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) kr[i] = kn[i];
            if (qd + 1 < 8) load_quad(qd + 1, kn);
            float v[32];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float kf[8];
                unpack8(kr[i], kf);
#pragma unroll
                for (int h = 0; h < 8; ++h) {
                    float acc = 0.f;
                    if (h0 + h < GR) {
#pragma unroll
                        for (int j = 0; j < 8; ++j) acc = fmaf(qr[h0 + h][j], kf[j], acc);
                    }
                    v[i * 8 + h] = acc;
                }
            }
            const float sum = reduce_scatter32(v, lane);
            s_p[lane & 7][warp * 32 + qd * 4 + (lane >> 3)] = sum * scale;
        }
    }
    __syncthreads();

    // ---- softmax over the chunk: head h0 + warp belongs to warp warp
    {
        float m = -FLT_MAX;
        for (int p = lane; p < len; p += 32) m = fmaxf(m, s_p[warp][p]);
        m = warp_max_d(m);
        float l = 0.f;
        for (int p = lane; p < kCH; p += 32) {
            const float e = p < len ? expf(s_p[warp][p] - m) : 0.f;
            s_p[warp][p] = e;
            l += e;
        }
        l = warp_sum_d(l);
        if (lane == 0 && h0 + warp < grp) {
            float* o = pml + ((static_cast<int64_t>(t) * n_head + kh * grp + h0 + warp) * n_chunks_max + c) * 2;
            o[0] = m;
            o[1] = l;
        }
    }
    __syncthreads();

    // ---- numerators: V is [kv head][dim][position], so a warp reads 256 consecutive positions of one dimension
    // (a lane 8 of them), forms the 8 per-head partial dots, and reduce-scatters 4 dimensions x 8 heads at once
    {
        float pr[8][8];
#pragma unroll
        for (int h = 0; h < 8; ++h) {
            const float4 a = *reinterpret_cast<const float4*>(&s_p[h][lane * 8]), b = *reinterpret_cast<const float4*>(&s_p[h][lane * 8 + 4]);
            pr[h][0] = a.x; pr[h][1] = a.y; pr[h][2] = a.z; pr[h][3] = a.w;
            pr[h][4] = b.x; pr[h][5] = b.y; pr[h][6] = b.z; pr[h][7] = b.w;
        }
        const bool live = lane * 8 < len;
        const int rem = len - lane * 8;  // valid positions of this lane's group when it straddles the end
        const __half* vbase = vc + static_cast<int64_t>(pg) * vpg + (static_cast<int64_t>(kh) * kHD) * vst + lane * 8;
        uint4 vn[4];
        auto load_quad = [&](int qd, uint4 (&dst)[4]) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int d = warp * 32 + qd * 4 + i;
                dst[i] = live ? __ldg(reinterpret_cast<const uint4*>(vbase + static_cast<int64_t>(d) * vst)) : make_uint4(0, 0, 0, 0);
            }
        };
        load_quad(0, vn);
#pragma unroll 1
        for (int qd = 0; qd < 8; ++qd) {
            uint4 vr[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) vr[i] = vn[i];
            if (qd + 1 < 8) load_quad(qd + 1, vn);
            float v[32];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float vf[8];
                unpack8(vr[i], vf);
                if (rem < 8) {
#pragma unroll
                    for (int j = 0; j < 8; ++j) vf[j] = j < rem ? vf[j] : 0.f;  // stale cache rows past the end
                }
#pragma unroll
                for (int h = 0; h < 8; ++h) {
                    float acc = 0.f;
                    if (h0 + h < GR) {
#pragma unroll
                        for (int j = 0; j < 8; ++j) acc = fmaf(pr[h][j], vf[j], acc);
                    }
                    v[i * 8 + h] = acc;
                }
            }
            const float sum = reduce_scatter32(v, lane);
            const int d = warp * 32 + qd * 4 + (lane >> 3), h = lane & 7;
            if (h0 + h < grp) po[((static_cast<int64_t>(t) * n_head + kh * grp + h0 + h) * n_chunks_max + c) * kHD + d] = sum;
        }
    }
    __syncthreads();
    }
}

// grid (n_head, T), block 256, dynamic smem n_chunks_max floats. out = (sum_c w_c o_c) / (sum_c w_c l_c) * gate; with unrot the
// merged output is rotated back first (a Q4 cache holds rotated values).
__global__ void __launch_bounds__(256)
    attn_combine_kernel(const float* __restrict__ po, const float* __restrict__ pml, const float* __restrict__ qfull,
                        float* out, const int* pos0_dev, int n_chunks_max, int n_head, bool unrot = false, int span = kCH) {
    extern __shared__ float s_w[];
    __shared__ float s_red[32];
    __shared__ float s_rot[256];
    const int head = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    const int pos = *pos0_dev + t;
    const int nc = pos / span + 1;
    const int64_t row = static_cast<int64_t>(t) * n_head + head;
    const float* ml = pml + row * n_chunks_max * 2;
    float m = -FLT_MAX;
    for (int c = d; c < nc; c += blockDim.x) m = fmaxf(m, ml[2 * c]);
    m = warp_max_d(m);
    if ((d & 31) == 0) s_red[d >> 5] = m;
    __syncthreads();
    float M = -FLT_MAX;
    for (int i = 0; i < static_cast<int>(blockDim.x >> 5); ++i) M = fmaxf(M, s_red[i]);
    for (int c = d; c < nc; c += blockDim.x) s_w[c] = expf(ml[2 * c] - M);
    __syncthreads();
    float L = 0.f, O = 0.f;
    const float* ob = po + row * n_chunks_max * kHD + d;
    for (int c = 0; c < nc; ++c) {
        const float w = s_w[c];
        L = fmaf(w, ml[2 * c + 1], L);
        O = fmaf(w, ob[static_cast<int64_t>(c) * kHD], O);
    }
    const float inv = L > 0.f ? 1.0f / L : 0.0f;
    const float g = sigmoid_d(qfull[row * 2 * kHD + kHD + d]);
    out[row * kHD + d] = (unrot ? kv_unrotate256(O * inv, s_rot) : O * inv) * g;
}


// ---- the same, over a Q8 cache (include/limbo/cuda/kv.hpp) ----------------------------------------------------------
// grid (n_head + 2 * n_head_kv, T), block hd (256): K and V of the position are quantised per block of 32 dimensions (a warp).
__global__ void attn_prep_q8_kernel(const float* qfull, const float* kbuf, const float* vbuf, const float* qn, const float* kn,
                                    float* q_out, int8_t* kq, uint16_t* ks, int8_t* vq, uint16_t* vs, const int* pos0_dev, const int* pt,
                                    double eps, double base, int rope_dim, int n_head, int n_head_kv, int hd) {
    __shared__ double sm[32];
    __shared__ float sv[256];
    __shared__ float s_inv;
    const int b = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    const int pos = *pos0_dev + t;
    const int64_t kvw = static_cast<int64_t>(n_head_kv) * hd;
    if (b < n_head) {
        const float* src = qfull + (static_cast<int64_t>(t) * n_head + b) * 2 * hd;
        const double v = src[d];
        const double ss = block_sum_d(v * v, sm);
        if (d == 0) s_inv = static_cast<float>(1.0 / sqrt(ss / hd + eps));
        __syncthreads();
        const float inv = s_inv;
        sv[d] = src[d] * inv * qn[d];
        __syncthreads();
        q_out[(static_cast<int64_t>(t) * n_head + b) * hd + d] = rope_elem_d(sv, d, pos, base, rope_dim);
    } else if (b < n_head + n_head_kv) {
        const int kh = b - n_head;
        const float* src = kbuf + t * kvw + static_cast<int64_t>(kh) * hd;
        const double v = src[d];
        const double ss = block_sum_d(v * v, sm);
        if (d == 0) s_inv = static_cast<float>(1.0 / sqrt(ss / hd + eps));
        __syncthreads();
        const float inv = s_inv;
        sv[d] = src[d] * (inv * kn[d]);
        __syncthreads();
        uint16_t dh;
        const int8_t q = kv_quant_lane(rope_elem_d(sv, d, pos, base, rope_dim), &dh);
        const int64_t row = page_pos(pt, pos) * n_head_kv + kh;
        kq[row * hd + d] = q;
        if ((d & 31) == 0) ks[row * (hd / kKvBlock) + (d >> 5)] = dh;
    } else {
        const int vh = b - n_head - n_head_kv;
        uint16_t dh;
        const int8_t q = kv_quant_lane(vbuf[t * kvw + static_cast<int64_t>(vh) * hd + d], &dh);
        const int64_t row = page_pos(pt, pos) * n_head_kv + vh;
        vq[row * hd + d] = q;
        if ((d & 31) == 0) vs[row * (hd / kKvBlock) + (d >> 5)] = dh;
    }
}

__device__ __forceinline__ void unpack8_i8(const uint2& r, float* f) {
    const uint32_t w[2] = {r.x, r.y};
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) f[4 * i + j] = static_cast<float>(static_cast<int8_t>(w[i] >> (8 * j)));
}

// grid (n_chunks_max, n_head_kv, T), block 256. Scores: a warp scores 4 positions per pass (a lane owns 8 dimensions of a K row,
// the scale of its block of 32 multiplies the lane's partial dot); numerators: warp w owns dimensions 32w..32w+31 and walks the
// chunk's positions, one int8 per lane per position, so the partial numerators need no cross-lane reduction.
template <int G, int GR = G>  // head slots per KV head: 8 (rung 1) or 12 (qwen4exp); GR of them real (6 for the 27Bs)
__global__ void __launch_bounds__(256, 2)
    attn_part_q8_kernel(const float* __restrict__ q, const int8_t* __restrict__ kq, const uint16_t* __restrict__ ks,
                        const int8_t* __restrict__ vq, const uint16_t* __restrict__ vs, float* po, float* pml, const int* pos0_dev,
                        int n_chunks_max, const int* pt, float scale, int n_head, int n_head_kv) {
    __shared__ __align__(16) float s_pt[kCH][G];  // scores, then probabilities: [position][head]
    const int t = blockIdx.x, kh = blockIdx.y, c = blockIdx.z;   // rows fastest: a chunk's rows run together and share it in L2
    const int grp = n_head / n_head_kv;  // query heads per KV head (<= G; the slots past it are zero and not written)
    const int pos = *pos0_dev + t;
    const int start = c * kCH;
    if (start > pos) return;
    const int len = min(kCH, pos + 1 - start);
    const int pbase = (pt ? pt[c] : c) * kCH;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t nbl = kHD / kKvBlock;  // scales per row

    {
        float qr[G][8];
        const float* qb = q + (static_cast<int64_t>(t) * n_head + kh * grp) * kHD + lane * 8;
#pragma unroll
        for (int h = 0; h < G; ++h) {
            const float4 a = (h < grp ? *reinterpret_cast<const float4*>(qb + h * kHD) : make_float4(0.f, 0.f, 0.f, 0.f)), b = (h < grp ? *reinterpret_cast<const float4*>(qb + h * kHD + 4) : make_float4(0.f, 0.f, 0.f, 0.f));
            qr[h][0] = a.x; qr[h][1] = a.y; qr[h][2] = a.z; qr[h][3] = a.w;
            qr[h][4] = b.x; qr[h][5] = b.y; qr[h][6] = b.z; qr[h][7] = b.w;
        }
        const int64_t kstride = static_cast<int64_t>(n_head_kv);
        // per band of 8 heads (reduce_scatter32 scatters 4 positions x 8 heads), as the f16 scorer: a group of 12 is two
        // bands, and a group of 8 does exactly what it did
        for (int h0 = 0; h0 < G; h0 += 8) {
        uint2 kn[4];
        uint16_t sn[4];
        auto load_quad = [&](int qd, uint2 (&dst)[4], uint16_t (&sc)[4]) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int pl = warp * 32 + qd * 4 + i;
                const int64_t row = (static_cast<int64_t>(pbase + pl) * kstride + kh);
                dst[i] = pl < len ? __ldg(reinterpret_cast<const uint2*>(kq + row * kHD + lane * 8)) : make_uint2(0, 0);
                sc[i] = pl < len ? __ldg(ks + row * nbl + (lane >> 2)) : static_cast<uint16_t>(0);
            }
        };
        load_quad(0, kn, sn);
#pragma unroll 1
        for (int qd = 0; qd < 8; ++qd) {
            uint2 kr[4];
            uint16_t sr[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                kr[i] = kn[i];
                sr[i] = sn[i];
            }
            if (qd + 1 < 8) load_quad(qd + 1, kn, sn);
            float v[32];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float kf[8];
                unpack8_i8(kr[i], kf);
                const float sc = __half2float(__ushort_as_half(sr[i]));
#pragma unroll
                for (int h = 0; h < 8; ++h) {
                    float acc = 0.f;
                    if (h0 + h < GR) {
#pragma unroll
                        for (int j = 0; j < 8; ++j) acc = fmaf(qr[h0 + h][j], kf[j], acc);
                    }
                    v[i * 8 + h] = acc * sc;
                }
            }
            const float sum = reduce_scatter32(v, lane);
            if (h0 + (lane & 7) < G) s_pt[warp * 32 + qd * 4 + (lane >> 3)][h0 + (lane & 7)] = sum * scale;
        }
        }
    }
    __syncthreads();

    for (int hh = warp; hh < G; hh += 8) {
        float m = -FLT_MAX;
        for (int p = lane; p < len; p += 32) m = fmaxf(m, s_pt[p][hh]);
        m = warp_max_d(m);
        float l = 0.f;
        for (int p = lane; p < kCH; p += 32) {
            const float e = p < len ? expf(s_pt[p][hh] - m) : 0.f;
            s_pt[p][hh] = e;
            l += e;
        }
        l = warp_sum_d(l);
        if (lane == 0 && hh < grp) {
            float* o = pml + ((static_cast<int64_t>(t) * n_head + kh * grp + hh) * n_chunks_max + c) * 2;
            o[0] = m;
            o[1] = l;
        }
    }
    __syncthreads();

    {
        float acc[G];
#pragma unroll
        for (int h = 0; h < G; ++h) acc[h] = 0.f;
        const int64_t kstride = static_cast<int64_t>(n_head_kv);
#pragma unroll 4
        for (int p = 0; p < len; ++p) {
            const int64_t row = (static_cast<int64_t>(pbase + p) * kstride + kh);
            const float v = static_cast<float>(__ldg(vq + row * kHD + warp * 32 + lane)) *
                            __half2float(__ushort_as_half(__ldg(vs + row * nbl + warp)));
#pragma unroll
            for (int h4 = 0; h4 < GR; h4 += 4) {
                if (h4 + 4 <= GR) {
                    const float4 a = *reinterpret_cast<const float4*>(&s_pt[p][h4]);
                    acc[h4] = fmaf(a.x, v, acc[h4]);
                    acc[h4 + 1] = fmaf(a.y, v, acc[h4 + 1]);
                    acc[h4 + 2] = fmaf(a.z, v, acc[h4 + 2]);
                    acc[h4 + 3] = fmaf(a.w, v, acc[h4 + 3]);
                } else {
                    const float2 a = *reinterpret_cast<const float2*>(&s_pt[p][h4]);
                    acc[h4] = fmaf(a.x, v, acc[h4]);
                    acc[h4 + 1] = fmaf(a.y, v, acc[h4 + 1]);
                }
            }
        }
#pragma unroll
        for (int h = 0; h < G; ++h)
            if (h < grp) po[((static_cast<int64_t>(t) * n_head + kh * grp + h) * n_chunks_max + c) * kHD + warp * 32 + lane] = acc[h];
    }
}

// ---- the same, over a Q4 cache (include/limbo/cuda/kv.hpp) ----------------------------------------------------------
// grid (n_head + 2 * n_head_kv, T), block hd (256): the query, the key and the value are rotated (kv_rotate256), K and V become
// 4-bit codes per block of 32 dimensions (a warp), two to a byte.
__global__ void attn_prep_q4_kernel(const float* qfull, const float* kbuf, const float* vbuf, const float* qn, const float* kn,
                                    float* q_out, uint8_t* kq, uint16_t* ks, uint8_t* vq, uint16_t* vs, const int* pos0_dev, const int* pt,
                                    double eps, double base, int rope_dim, int n_head, int n_head_kv, int hd) {
    __shared__ double sm[32];
    __shared__ float sv[256];
    __shared__ float s_rot[256];
    __shared__ float s_inv;
    const int b = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    const int pos = *pos0_dev + t;
    const int64_t kvw = static_cast<int64_t>(n_head_kv) * hd;
    if (b < n_head + n_head_kv) {
        const bool is_q = b < n_head;
        const int kh = b - n_head;
        const float* src = is_q ? qfull + (static_cast<int64_t>(t) * n_head + b) * 2 * hd : kbuf + t * kvw + static_cast<int64_t>(kh) * hd;
        const double v = src[d];
        const double ss = block_sum_d(v * v, sm);
        if (d == 0) s_inv = static_cast<float>(1.0 / sqrt(ss / hd + eps));
        __syncthreads();
        const float inv = s_inv;
        sv[d] = is_q ? src[d] * inv * qn[d] : src[d] * (inv * kn[d]);
        __syncthreads();
        const float r = kv_rotate256(rope_elem_d(sv, d, pos, base, rope_dim), s_rot);
        if (is_q) {
            q_out[(static_cast<int64_t>(t) * n_head + b) * hd + d] = r;
        } else {
            uint16_t dh;
            const int byte = kv_pack4_lane(kv_quant4_lane(r, &dh));
            const int64_t row = page_pos(pt, pos) * n_head_kv + kh;
            if (byte >= 0) kq[row * (hd / 2) + (d >> 1)] = static_cast<uint8_t>(byte);
            if ((d & 31) == 0) ks[row * (hd / kKvBlock) + (d >> 5)] = dh;
        }
    } else {
        const int vh = b - n_head - n_head_kv;
        uint16_t dh;
        const int byte = kv_pack4_lane(kv_quant4_lane(kv_rotate256(vbuf[t * kvw + static_cast<int64_t>(vh) * hd + d], s_rot), &dh));
        const int64_t row = page_pos(pt, pos) * n_head_kv + vh;
        if (byte >= 0) vq[row * (hd / 2) + (d >> 1)] = static_cast<uint8_t>(byte);
        if ((d & 31) == 0) vs[row * (hd / kKvBlock) + (d >> 5)] = dh;
    }
}

// attn_part_q8_kernel over 4-bit codes: a lane's 8 key dimensions are one 32-bit word of the 128-byte row, a warp's 32 value
// dimensions 16 bytes of it.
template <int G, int GR = G>  // head slots per KV head (8); GR of them real (6 for the 27Bs)
__global__ void __launch_bounds__(256, 2)
    attn_part_q4_kernel(const float* __restrict__ q, const uint8_t* __restrict__ kq, const uint16_t* __restrict__ ks,
                        const uint8_t* __restrict__ vq, const uint16_t* __restrict__ vs, float* po, float* pml, const int* pos0_dev,
                        int n_chunks_max, const int* pt, float scale, int n_head, int n_head_kv) {
    __shared__ __align__(16) float s_pt[kCH][G];  // scores, then probabilities: [position][head]
    constexpr int kRB = kHD / 2;                    // bytes per row
    // the chunk's V, copied in while the scores run (the value loop held few loads in flight)
    __shared__ __align__(16) uint8_t s_v[kCH * kRB];
    __shared__ __align__(16) uint16_t s_vs[kCH * (kHD / kKvBlock)];
    const int t = blockIdx.x, kh = blockIdx.y, c = blockIdx.z;   // rows fastest: a chunk's rows run together and share it in L2
    const int grp = n_head / n_head_kv;  // query heads per KV head (<= G; the slots past it are zero and not written)
    const int pos = *pos0_dev + t;
    const int start = c * kCH;
    if (start > pos) return;
    const int len = min(kCH, pos + 1 - start);
    const int pbase = (pt ? pt[c] : c) * kCH;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t nbl = kHD / kKvBlock;
    const int64_t kstride = static_cast<int64_t>(n_head_kv);
    for (int u = threadIdx.x; u < len * (kRB / 16 + 1); u += blockDim.x) {   // 8 chunks of codes and 1 of scales per row
        const int p = u / (kRB / 16 + 1), k = u % (kRB / 16 + 1);
        const int64_t row = static_cast<int64_t>(pbase + p) * kstride + kh;
        if (k < kRB / 16) cp_async16_a(s_v + p * kRB + k * 16, vq + row * kRB + k * 16);
        else cp_async16_a(s_vs + p * nbl, vs + row * nbl);
    }
    cp_async_commit_a();

    {
        float qr[G][8];
        const float* qb = q + (static_cast<int64_t>(t) * n_head + kh * grp) * kHD + lane * 8;
#pragma unroll
        for (int h = 0; h < G; ++h) {
            const float4 a = (h < grp ? *reinterpret_cast<const float4*>(qb + h * kHD) : make_float4(0.f, 0.f, 0.f, 0.f)), b = (h < grp ? *reinterpret_cast<const float4*>(qb + h * kHD + 4) : make_float4(0.f, 0.f, 0.f, 0.f));
            qr[h][0] = a.x; qr[h][1] = a.y; qr[h][2] = a.z; qr[h][3] = a.w;
            qr[h][4] = b.x; qr[h][5] = b.y; qr[h][6] = b.z; qr[h][7] = b.w;
        }
        uint32_t kn[4];
        uint16_t sn[4];
        auto load_quad = [&](int qd, uint32_t (&dst)[4], uint16_t (&sc)[4]) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int pl = warp * 32 + qd * 4 + i;
                const int64_t row = static_cast<int64_t>(pbase + pl) * kstride + kh;
                dst[i] = pl < len ? __ldg(reinterpret_cast<const unsigned int*>(kq + row * kRB + lane * 4)) : 0x88888888u;
                sc[i] = pl < len ? __ldg(ks + row * nbl + (lane >> 2)) : static_cast<uint16_t>(0);
            }
        };
        load_quad(0, kn, sn);
#pragma unroll 1
        for (int qd = 0; qd < 8; ++qd) {
            uint32_t kr[4];
            uint16_t sr[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                kr[i] = kn[i];
                sr[i] = sn[i];
            }
            if (qd + 1 < 8) load_quad(qd + 1, kn, sn);
            float v[32];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float kf[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) kf[j] = kv_nib(kr[i], j);
                const float sc = __half2float(__ushort_as_half(sr[i]));
#pragma unroll
                for (int h = 0; h < G; ++h) {
                    float acc = 0.f;
                    if (h < GR) {
#pragma unroll
                        for (int j = 0; j < 8; ++j) acc = fmaf(qr[h][j], kf[j], acc);
                    }
                    v[i * G + h] = acc * sc;
                }
            }
            const float sum = reduce_scatter32(v, lane);
            s_pt[warp * 32 + qd * 4 + (lane >> 3)][lane & 7] = sum * scale;
        }
    }
    __syncthreads();

    {
        float m = -FLT_MAX;
        for (int p = lane; p < len; p += 32) m = fmaxf(m, s_pt[p][warp]);
        m = warp_max_d(m);
        float l = 0.f;
        for (int p = lane; p < kCH; p += 32) {
            const float e = p < len ? expf(s_pt[p][warp] - m) : 0.f;
            s_pt[p][warp] = e;
            l += e;
        }
        l = warp_sum_d(l);
        if (lane == 0 && warp < grp) {
            float* o = pml + ((static_cast<int64_t>(t) * n_head + kh * grp + warp) * n_chunks_max + c) * 2;
            o[0] = m;
            o[1] = l;
        }
    }
    cp_async_wait_all_a();   // this thread's V copies; the barrier makes everyone's visible
    __syncthreads();

    {
        float acc[G];
#pragma unroll
        for (int h = 0; h < G; ++h) acc[h] = 0.f;
#pragma unroll 4
        for (int p = 0; p < len; ++p) {
            const uint32_t byte = s_v[p * kRB + warp * 16 + (lane >> 1)];
            const float v = kv_nib(byte, lane & 1) * __half2float(__ushort_as_half(s_vs[p * nbl + warp]));
            const float4 a = *reinterpret_cast<const float4*>(&s_pt[p][0]);
            acc[0] = fmaf(a.x, v, acc[0]);
            acc[1] = fmaf(a.y, v, acc[1]);
            acc[2] = fmaf(a.z, v, acc[2]);
            acc[3] = fmaf(a.w, v, acc[3]);
            if constexpr (GR > 6) {
                const float4 b = *reinterpret_cast<const float4*>(&s_pt[p][4]);
                acc[4] = fmaf(b.x, v, acc[4]);
                acc[5] = fmaf(b.y, v, acc[5]);
                acc[6] = fmaf(b.z, v, acc[6]);
                acc[7] = fmaf(b.w, v, acc[7]);
            } else {
                const float2 b = *reinterpret_cast<const float2*>(&s_pt[p][4]);
                acc[4] = fmaf(b.x, v, acc[4]);
                acc[5] = fmaf(b.y, v, acc[5]);
            }
        }
#pragma unroll
        for (int h = 0; h < G; ++h)
            if (h < grp) po[((static_cast<int64_t>(t) * n_head + kh * grp + h) * n_chunks_max + c) * kHD + warp * 32 + lane] = acc[h];
    }
}

// ---- decode attention over a Q8 / Q4 pool on the tensor cores ---------------------------------------------------------------
// Block = KV head x key span x 8 (row, head) pairs; Q in registers as hi/lo f16 rows of one mma tile (fp32-accurate).
// Codes exact in f16, K scale in fp32 per 32 dims, V scale folded into P with a fixed 2^10 lift (bits independent of the window).
constexpr int kTcSpan = 512;   // keys per block (one partial per span)
constexpr float kTcLift = 1024.f;
constexpr int kTcPS = 40;      // floats per row of the shared P tile

template <bool Q4>
struct TcLay {
    static constexpr int kRow = Q4 ? 128 : 256;  // bytes of a pool row (one KV head of one position)
    static constexpr int kKS = Q4 ? 144 : 288;   // smem strides: conflict-free fragment reads
    static constexpr int kVS = Q4 ? 144 : 272;
    static constexpr int kVOff = 32 * kKS;
    static constexpr int kKSc = kVOff + 32 * kVS;
    static constexpr int kVSc = kKSc + 512;
    static constexpr int kSlot = kVSc + 512;
};

__device__ __forceinline__ void tc_mma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t tc_h2(float x0, float x1) {
    const __half2 h = __floats2half2_rn(x0, x1);
    return *reinterpret_cast<const uint32_t*>(&h);
}
// x = hi + lo, two f16 pairs
__device__ __forceinline__ void tc_split(float x0, float x1, uint32_t& hi, uint32_t& lo) {
    const __half2 h = __floats2half2_rn(x0, x1);
    hi = *reinterpret_cast<const uint32_t*>(&h);
    const float2 f = __half22float2(h);
    lo = tc_h2(x0 - f.x, x1 - f.y);
}
__device__ __forceinline__ uint32_t tc_sub2(uint32_t a, uint32_t b) {
    uint32_t r;
    asm("sub.f16x2 %0, %1, %2;\n" : "=r"(r) : "r"(a), "r"(b));
    return r;
}
__device__ __forceinline__ float tc_half(uint32_t packed, int hi) {
    return __half2float(__ushort_as_half(static_cast<unsigned short>(hi ? packed >> 16 : packed & 0xffffu)));
}

__device__ __forceinline__ void cp_async16_z(void* dst, const void* src, bool valid) {
    const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    const int sz = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(sz));
}
template <int N>
__device__ __forceinline__ void cp_async_wait_n() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// The 8 dims 8t..8t+7 of a 32-dim block as four f16 pairs (dim 8t+i, dim 8t+4+i): the k slots of the A tile follow the same order.
__device__ __forceinline__ void tc_kcodes_q4(uint32_t w, uint32_t* b) {
#pragma unroll
    for (int i = 0; i < 4; ++i) b[i] = tc_sub2(((w >> (4 * i)) & 0x000F000Fu) | 0x64006400u, 0x64086408u);
}
__device__ __forceinline__ void tc_kcodes_q8(uint2 w, uint32_t* b) {
#pragma unroll
    for (int i = 0; i < 4; ++i) b[i] = tc_sub2((__byte_perm(w.x, w.y, 0x4400 + 0x1111 * i) & 0x00FF00FFu) ^ 0x64806480u, 0x64806480u);
}

template <bool Q4, int R>
__global__ void __launch_bounds__(128, Q4 ? 3 : 2)
    attn_tc_kernel(const float* __restrict__ q, const uint8_t* __restrict__ kq, const uint16_t* __restrict__ ks, const uint8_t* __restrict__ vq,
                   const uint16_t* __restrict__ vs, float* po, float* pml, const int* pos0_dev, int nsp_max, const int* pt, float scale,
                   int n_head, int n_head_kv, int T) {
    using L = TcLay<Q4>;
    __shared__ __align__(16) uint8_t sm[R * L::kSlot + 8 * kTcPS * 4 + 8 * 4 * 8];
    float* Pb = reinterpret_cast<float*>(sm + R * L::kSlot);
    float* ML = Pb + 8 * kTcPS;
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int gi = blockIdx.x, sp = blockIdx.y, kh = blockIdx.z;
    const int grp = n_head / n_head_kv, npairs = T * grp, pos0 = *pos0_dev;
    const int span_start = sp * kTcSpan;
    if (span_start > pos0 + T - 1) return;
    const int pend = pos0 + T;
    const int lim = min(span_start + kTcSpan, pend);
    const int nblk = (lim - span_start + 31) / 32;
    const int pg = min(gi * 8 + g, npairs - 1);
    const int rowt = pg / grp, hh = pg - rowt * grp;
    const int qpos = pos0 + rowt;

    // Q as A fragments: kstep 2c + s, regs {hi (row g), lo (row g+8), hi (k+8), lo (k+8)}
    uint32_t A[16][4];
    {
        const float* qp = q + (static_cast<int64_t>(rowt) * n_head + kh * grp + hh) * kHD + 8 * t;
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            const float4 x0 = *reinterpret_cast<const float4*>(qp + 32 * c), x1 = *reinterpret_cast<const float4*>(qp + 32 * c + 4);
            tc_split(x0.x, x1.x, A[2 * c][0], A[2 * c][1]);
            tc_split(x0.y, x1.y, A[2 * c][2], A[2 * c][3]);
            tc_split(x0.z, x1.z, A[2 * c + 1][0], A[2 * c + 1][1]);
            tc_split(x0.w, x1.w, A[2 * c + 1][2], A[2 * c + 1][3]);
        }
    }

    auto issue_stage = [&](int s) {
        const int base = span_start + 32 * s;
        const int c = base / kCH;
        const int64_t pb = static_cast<int64_t>(pt ? pt[c] : c) * kCH + (base % kCH);
        uint8_t* slot = sm + (s % R) * L::kSlot;
        constexpr int PK = L::kRow / 16;
        for (int u = tid; u < 64 * PK + 64; u += 128) {
            if (u < 64 * PK) {
                const int isv = u >= 32 * PK, v = isv ? u - 32 * PK : u, key = v / PK, piece = v % PK;
                const bool ok = base + key < pend;
                const int64_t row = (pb + (ok ? key : 0)) * n_head_kv + kh;
                cp_async16_z(slot + (isv ? L::kVOff + key * L::kVS : key * L::kKS) + piece * 16, (isv ? vq : kq) + row * L::kRow + piece * 16, ok);
            } else {
                const int j = u - 64 * PK, key = j & 31, isv = j >> 5;
                const bool ok = base + key < pend;
                const int64_t row = (pb + (ok ? key : 0)) * n_head_kv + kh;
                cp_async16_z(slot + (isv ? L::kVSc : L::kKSc) + key * 16, (isv ? vs : ks) + row * 8, ok);
            }
        }
    };

#pragma unroll
    for (int s = 0; s < R - 1; ++s) {
        if (s < nblk) issue_stage(s);
        asm volatile("cp.async.commit_group;\n" ::);
    }

    float o[2][4][4];
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
        for (int d = 0; d < 4; ++d)
#pragma unroll
            for (int i = 0; i < 4; ++i) o[a][d][i] = 0.f;
    float m_run = -INFINITY, l_run = 0.f;
    const uint32_t vsel = 0x4400u + 0x1111u * static_cast<uint32_t>(Q4 ? (g >> 1) : (g & 3));
    const int vsh = Q4 ? 4 * (g & 1) : 0;

    for (int kb = 0; kb < nblk; ++kb) {
        cp_async_wait_n<R - 2>();
        __syncthreads();
        if (kb + R - 1 < nblk) issue_stage(kb + R - 1);
        asm volatile("cp.async.commit_group;\n" ::);
        const uint8_t* slot = sm + (kb % R) * L::kSlot;

        // ---- scores of the keys 8w..8w+7 of the stage
        {
            const uint8_t* kr = slot + (8 * w + g) * L::kKS;
            const uint4 sc0 = *reinterpret_cast<const uint4*>(slot + L::kKSc + (8 * w + 2 * t) * 16);
            const uint4 sc1 = *reinterpret_cast<const uint4*>(slot + L::kKSc + (8 * w + 2 * t + 1) * 16);
            const uint32_t s0r[4] = {sc0.x, sc0.y, sc0.z, sc0.w}, s1r[4] = {sc1.x, sc1.y, sc1.z, sc1.w};
            float S0 = 0.f, S1 = 0.f;
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                uint32_t b[4];
                if constexpr (Q4) tc_kcodes_q4(*reinterpret_cast<const uint32_t*>(kr + 16 * c + 4 * t), b);
                else tc_kcodes_q8(*reinterpret_cast<const uint2*>(kr + 32 * c + 8 * t), b);
                float C[4] = {0.f, 0.f, 0.f, 0.f};
                tc_mma(C, A[2 * c], b[0], b[1]);
                tc_mma(C, A[2 * c + 1], b[2], b[3]);
                S0 = fmaf(tc_half(s0r[c >> 1], c & 1), C[0] + C[2], S0);
                S1 = fmaf(tc_half(s1r[c >> 1], c & 1), C[1] + C[3], S1);
            }
            const int kabs = span_start + 32 * kb + 8 * w + 2 * t;
            const float s0 = kabs <= qpos ? S0 * scale : -INFINITY, s1 = kabs + 1 <= qpos ? S1 * scale : -INFINITY;
            float mw = fmaxf(s0, s1);
            mw = fmaxf(mw, __shfl_xor_sync(0xffffffffu, mw, 1));
            mw = fmaxf(mw, __shfl_xor_sync(0xffffffffu, mw, 2));
            const float mr = mw == -INFINITY ? 0.f : mw;
            const float p0 = s0 == -INFINITY ? 0.f : expf(s0 - mr), p1 = s1 == -INFINITY ? 0.f : expf(s1 - mr);
            float lw = p0 + p1;
            lw += __shfl_xor_sync(0xffffffffu, lw, 1);
            lw += __shfl_xor_sync(0xffffffffu, lw, 2);
            *reinterpret_cast<float2*>(Pb + g * kTcPS + 8 * w + 2 * t) = make_float2(p0, p1);
            if (t == 0) *reinterpret_cast<float2*>(ML + (g * 4 + w) * 2) = make_float2(mw, lw);
        }
        __syncthreads();

        // ---- softmax state over the 4 score groups, then P V over this warp's 64 value dims
        const float4 ma = *reinterpret_cast<const float4*>(ML + g * 8), mb = *reinterpret_cast<const float4*>(ML + g * 8 + 4);
        const float mj[4] = {ma.x, ma.z, mb.x, mb.z}, lj[4] = {ma.y, ma.w, mb.y, mb.w};
        const float mn = fmaxf(m_run, fmaxf(fmaxf(mj[0], mj[1]), fmaxf(mj[2], mj[3])));
        float alpha = 1.f, f[4] = {0.f, 0.f, 0.f, 0.f};
        if (mn != -INFINITY) {
            alpha = m_run == -INFINITY ? 0.f : expf(m_run - mn);
#pragma unroll
            for (int j = 0; j < 4; ++j) f[j] = mj[j] == -INFINITY ? 0.f : expf(mj[j] - mn);
        }
        l_run = fmaf(f[3], lj[3], fmaf(f[2], lj[2], fmaf(f[1], lj[1], fmaf(f[0], lj[0], l_run * alpha))));
        m_run = mn;
        if (!__all_sync(0xffffffffu, alpha == 1.f)) {   // the running max rarely moves: x1.0 is exact, skipping it changes no bit
#pragma unroll
            for (int a = 0; a < 2; ++a)
#pragma unroll
                for (int d = 0; d < 4; ++d)
#pragma unroll
                    for (int i = 0; i < 4; ++i) o[a][d][i] *= alpha;
        }

#pragma unroll
        for (int s = 0; s < 2; ++s) {
            const float2 pA = *reinterpret_cast<const float2*>(Pb + g * kTcPS + 16 * s + 2 * t);
            const float2 pB = *reinterpret_cast<const float2*>(Pb + g * kTcPS + 16 * s + 8 + 2 * t);
            const float fA = f[2 * s] * kTcLift, fB = f[2 * s + 1] * kTcLift;
            const int k0 = 16 * s + 2 * t;
            const uint8_t* vsb = slot + L::kVSc + 4 * w;
            const uint32_t vsA0 = *reinterpret_cast<const uint32_t*>(vsb + k0 * 16), vsA1 = *reinterpret_cast<const uint32_t*>(vsb + (k0 + 1) * 16),
                           vsB0 = *reinterpret_cast<const uint32_t*>(vsb + (k0 + 8) * 16), vsB1 = *reinterpret_cast<const uint32_t*>(vsb + (k0 + 9) * 16);
            const uint8_t* vb = slot + L::kVOff;
#pragma unroll
            for (int bi = 0; bi < 2; ++bi) {
                const float x0 = fminf(pA.x * (fA * tc_half(vsA0, bi)), 65504.f), x1 = fminf(pA.y * (fA * tc_half(vsA1, bi)), 65504.f);
                const float x2 = fminf(pB.x * (fB * tc_half(vsB0, bi)), 65504.f), x3 = fminf(pB.y * (fB * tc_half(vsB1, bi)), 65504.f);
                uint32_t a[4];
                tc_split(x0, x1, a[0], a[1]);
                tc_split(x2, x3, a[2], a[3]);
                const int bk = 2 * w + bi;
                if constexpr (Q4) {
                    const uint4 rA0 = *reinterpret_cast<const uint4*>(vb + k0 * L::kVS + 16 * bk), rA1 = *reinterpret_cast<const uint4*>(vb + (k0 + 1) * L::kVS + 16 * bk),
                                rB0 = *reinterpret_cast<const uint4*>(vb + (k0 + 8) * L::kVS + 16 * bk), rB1 = *reinterpret_cast<const uint4*>(vb + (k0 + 9) * L::kVS + 16 * bk);
                    const uint32_t wa0[4] = {rA0.x, rA0.y, rA0.z, rA0.w}, wa1[4] = {rA1.x, rA1.y, rA1.z, rA1.w}, wb0[4] = {rB0.x, rB0.y, rB0.z, rB0.w},
                                   wb1[4] = {rB1.x, rB1.y, rB1.z, rB1.w};
#pragma unroll
                    for (int dt = 0; dt < 4; ++dt) {
                        const uint32_t b0 = tc_sub2(((__byte_perm(wa0[dt], wa1[dt], vsel) >> vsh) & 0x000F000Fu) | 0x64006400u, 0x64086408u);
                        const uint32_t b1 = tc_sub2(((__byte_perm(wb0[dt], wb1[dt], vsel) >> vsh) & 0x000F000Fu) | 0x64006400u, 0x64086408u);
                        tc_mma(o[bi][dt], a, b0, b1);
                    }
                } else {
#pragma unroll
                    for (int dt = 0; dt < 4; ++dt) {
                        const int off = 32 * bk + 8 * dt + 4 * (g >> 2);
                        const uint32_t wa0 = *reinterpret_cast<const uint32_t*>(vb + k0 * L::kVS + off), wa1 = *reinterpret_cast<const uint32_t*>(vb + (k0 + 1) * L::kVS + off),
                                       wb0 = *reinterpret_cast<const uint32_t*>(vb + (k0 + 8) * L::kVS + off), wb1 = *reinterpret_cast<const uint32_t*>(vb + (k0 + 9) * L::kVS + off);
                        const uint32_t b0 = tc_sub2((__byte_perm(wa0, wa1, vsel) & 0x00FF00FFu) ^ 0x64806480u, 0x64806480u);
                        const uint32_t b1 = tc_sub2((__byte_perm(wb0, wb1, vsel) & 0x00FF00FFu) ^ 0x64806480u, 0x64806480u);
                        tc_mma(o[bi][dt], a, b0, b1);
                    }
                }
            }
        }
    }

    if (gi * 8 + g < npairs && span_start <= qpos) {
        const int64_t base = (static_cast<int64_t>(rowt) * n_head + kh * grp + hh) * nsp_max + sp;
        const float inv = 1.f / kTcLift;
#pragma unroll
        for (int bi = 0; bi < 2; ++bi)
#pragma unroll
            for (int dt = 0; dt < 4; ++dt)
                *reinterpret_cast<float2*>(po + base * kHD + 64 * w + 32 * bi + 8 * dt + 2 * t) =
                    make_float2((o[bi][dt][0] + o[bi][dt][2]) * inv, (o[bi][dt][1] + o[bi][dt][3]) * inv);
        if (w == 0 && t == 0) *reinterpret_cast<float2*>(pml + base * 2) = make_float2(m_run, l_run);
    }
}

// LIMBO_ATTN_TC=0: the scalar kernels for every row count (dev switch)
bool tc_on() {
    static const bool on = [] {
        const char* e = std::getenv("LIMBO_ATTN_TC");
        return !(e && *e == '0');
    }();
    return on;
}

}  // namespace

// The f16 V layout a launch uses: paged (pt set) a page is [kv head][256 dims][256 positions], contiguous a chunk's columns
// start c*256 into rows vpitch apart.
static int v_page_elems(const int* pt, int n_head_kv, int hd) { return pt ? n_head_kv * hd * kCH : kCH; }
static int v_row_stride(const int* pt, int vpitch) { return pt ? kCH : vpitch; }

void attn_prep(const float* qfull, const float* kbuf, const float* vbuf, const float* qn, const float* kn, float* q_out,
               __half* kcache, __half* vcache, const int* pos0, int T, int vpitch, double eps, double rope_base,
               int rope_dim, int n_head, int n_head_kv, int hd, cudaStream_t s, const int* pt) {
    const dim3 grid(static_cast<unsigned>(n_head + 2 * n_head_kv), static_cast<unsigned>(T));
    attn_prep_kernel<<<grid, hd, 0, s>>>(qfull, kbuf, vbuf, qn, kn, q_out, kcache, vcache, pos0, pt, v_page_elems(pt, n_head_kv, hd),
                                         v_row_stride(pt, vpitch), eps, rope_base, rope_dim, n_head, n_head_kv, hd);
}

size_t attn_scratch_floats(int max_ctx, int max_rows, int n_head) {
    const size_t nc = static_cast<size_t>((max_ctx + kCH - 1) / kCH);
    return static_cast<size_t>(max_rows) * static_cast<size_t>(n_head) * nc * (kHD + 2);
}

bool attention(const float* q, const __half* kcache, const __half* vcache, const float* qfull, float* out, float* scratch,
               const int* pos0, int T, int max_ctx, int vpitch, int n_head, int n_head_kv, int hd, float scale,
               cudaStream_t s, const int* pt) {
    const int grp = n_head / n_head_kv;
    if (hd != kHD || n_head_kv <= 0 || n_head % n_head_kv != 0 || grp > kGMax || T < 1 || T > kMaxRows) return false;
    const int nc = (max_ctx + kCH - 1) / kCH;
    float* po = scratch;
    float* pml = scratch + static_cast<size_t>(T) * n_head * nc * kHD;
    const dim3 grid(static_cast<unsigned>(T), static_cast<unsigned>(n_head_kv), static_cast<unsigned>(nc));
    if (grp == 6)
        attn_part_kernel<8, 6><<<grid, 256, 0, s>>>(q, kcache, vcache, po, pml, pos0, nc, pt, v_page_elems(pt, n_head_kv, hd),
                                                 v_row_stride(pt, vpitch), scale, n_head, n_head_kv);
    else if (grp <= 8)
        attn_part_kernel<8><<<grid, 256, 0, s>>>(q, kcache, vcache, po, pml, pos0, nc, pt, v_page_elems(pt, n_head_kv, hd),
                                                 v_row_stride(pt, vpitch), scale, n_head, n_head_kv);
    else
        attn_part_kernel<12><<<grid, 256, 0, s>>>(q, kcache, vcache, po, pml, pos0, nc, pt, v_page_elems(pt, n_head_kv, hd),
                                                  v_row_stride(pt, vpitch), scale, n_head, n_head_kv);
    attn_combine_kernel<<<dim3(static_cast<unsigned>(n_head), static_cast<unsigned>(T)), 256, static_cast<size_t>(nc) * sizeof(float), s>>>(
        po, pml, qfull, out, pos0, nc, n_head);
    return cudaGetLastError() == cudaSuccess;
}

void attn_prep_q8(const float* qfull, const float* kbuf, const float* vbuf, const float* qn, const float* kn, float* q_out, int8_t* kq,
                  uint16_t* ks, int8_t* vq, uint16_t* vs, const int* pos0, int T, double eps, double rope_base, int rope_dim, int n_head,
                  int n_head_kv, int hd, cudaStream_t s, const int* pt) {
    const dim3 grid(static_cast<unsigned>(n_head + 2 * n_head_kv), static_cast<unsigned>(T));
    attn_prep_q8_kernel<<<grid, hd, 0, s>>>(qfull, kbuf, vbuf, qn, kn, q_out, kq, ks, vq, vs, pos0, pt, eps, rope_base, rope_dim, n_head,
                                           n_head_kv, hd);
}

// Tensor-core split flash-decode over a Q8 / Q4 pool: partials per span of kTcSpan keys, merged by the combine kernel.
template <bool Q4>
static bool attention_tc(const float* q, const int8_t* kq, const uint16_t* ks, const int8_t* vq, const uint16_t* vs, const float* qfull,
                         float* out, float* scratch, const int* pos0, int T, int max_ctx, int n_head, int n_head_kv, float scale,
                         cudaStream_t s, const int* pt) {
    const int grp = n_head / n_head_kv;
    const int nsp = (max_ctx + kTcSpan - 1) / kTcSpan;
    float* po = scratch;
    float* pml = scratch + static_cast<size_t>(T) * n_head * nsp * kHD;
    const dim3 grid(static_cast<unsigned>((T * grp + 7) / 8), static_cast<unsigned>(nsp), static_cast<unsigned>(n_head_kv));
    attn_tc_kernel<Q4, Q4 ? 3 : 2><<<grid, 128, 0, s>>>(q, reinterpret_cast<const uint8_t*>(kq), ks, reinterpret_cast<const uint8_t*>(vq), vs, po, pml,
                                                       pos0, nsp, pt, scale, n_head, n_head_kv, T);
    attn_combine_kernel<<<dim3(static_cast<unsigned>(n_head), static_cast<unsigned>(T)), 256, static_cast<size_t>(nsp) * sizeof(float), s>>>(
        po, pml, qfull, out, pos0, nsp, n_head, Q4, kTcSpan);
    return cudaGetLastError() == cudaSuccess;
}

bool attention_q8(const float* q, const int8_t* kq, const uint16_t* ks, const int8_t* vq, const uint16_t* vs, const float* qfull, float* out,
                  float* scratch, const int* pos0, int T, int max_ctx, int n_head, int n_head_kv, int hd, float scale, cudaStream_t s,
                  const int* pt) {
    const int grp = n_head / n_head_kv;
    // groups of 8 (rung 1) and 12 (qwen4exp: two bands of 8 in the scorer)
    if (hd != kHD || n_head_kv <= 0 || n_head % n_head_kv != 0 || grp > kGMax || T < 1 || T > kMaxRows) return false;
    if (tc_on() && grp <= 8) return attention_tc<false>(q, kq, ks, vq, vs, qfull, out, scratch, pos0, T, max_ctx, n_head, n_head_kv, scale, s, pt);
    const int nc = (max_ctx + kCH - 1) / kCH;
    float* po = scratch;
    float* pml = scratch + static_cast<size_t>(T) * n_head * nc * kHD;
    const dim3 grid(static_cast<unsigned>(T), static_cast<unsigned>(n_head_kv), static_cast<unsigned>(nc));
    if (grp == 6)
        attn_part_q8_kernel<8, 6><<<grid, 256, 0, s>>>(q, kq, ks, vq, vs, po, pml, pos0, nc, pt, scale, n_head, n_head_kv);
    else if (grp <= 8)
        attn_part_q8_kernel<8><<<grid, 256, 0, s>>>(q, kq, ks, vq, vs, po, pml, pos0, nc, pt, scale, n_head, n_head_kv);
    else
        attn_part_q8_kernel<12><<<grid, 256, 0, s>>>(q, kq, ks, vq, vs, po, pml, pos0, nc, pt, scale, n_head, n_head_kv);
    attn_combine_kernel<<<dim3(static_cast<unsigned>(n_head), static_cast<unsigned>(T)), 256, static_cast<size_t>(nc) * sizeof(float), s>>>(
        po, pml, qfull, out, pos0, nc, n_head);
    return cudaGetLastError() == cudaSuccess;
}

void attn_prep_q4(const float* qfull, const float* kbuf, const float* vbuf, const float* qn, const float* kn, float* q_out, int8_t* kq,
                  uint16_t* ks, int8_t* vq, uint16_t* vs, const int* pos0, int T, double eps, double rope_base, int rope_dim, int n_head,
                  int n_head_kv, int hd, cudaStream_t s, const int* pt) {
    const dim3 grid(static_cast<unsigned>(n_head + 2 * n_head_kv), static_cast<unsigned>(T));
    attn_prep_q4_kernel<<<grid, hd, 0, s>>>(qfull, kbuf, vbuf, qn, kn, q_out, reinterpret_cast<uint8_t*>(kq), ks, reinterpret_cast<uint8_t*>(vq),
                                           vs, pos0, pt, eps, rope_base, rope_dim, n_head, n_head_kv, hd);
}

bool attention_q4(const float* q, const int8_t* kq, const uint16_t* ks, const int8_t* vq, const uint16_t* vs, const float* qfull, float* out,
                  float* scratch, const int* pos0, int T, int max_ctx, int n_head, int n_head_kv, int hd, float scale, cudaStream_t s,
                  const int* pt) {
    const int grp = n_head / n_head_kv;
    // the Q8/Q4 scorers keep the 8-head reduce-scatter layout: groups beyond 8 take the f16 path
    if (hd != kHD || n_head_kv <= 0 || n_head % n_head_kv != 0 || grp > 8 || T < 1 || T > kMaxRows) return false;
    if (tc_on()) return attention_tc<true>(q, kq, ks, vq, vs, qfull, out, scratch, pos0, T, max_ctx, n_head, n_head_kv, scale, s, pt);
    const int nc = (max_ctx + kCH - 1) / kCH;
    float* po = scratch;
    float* pml = scratch + static_cast<size_t>(T) * n_head * nc * kHD;
    const dim3 grid(static_cast<unsigned>(T), static_cast<unsigned>(n_head_kv), static_cast<unsigned>(nc));
    if (grp == 6)
        attn_part_q4_kernel<8, 6><<<grid, 256, 0, s>>>(q, reinterpret_cast<const uint8_t*>(kq), ks, reinterpret_cast<const uint8_t*>(vq), vs,
                                                       po, pml, pos0, nc, pt, scale, n_head, n_head_kv);
    else
        attn_part_q4_kernel<8><<<grid, 256, 0, s>>>(q, reinterpret_cast<const uint8_t*>(kq), ks, reinterpret_cast<const uint8_t*>(vq), vs,
                                                    po, pml, pos0, nc, pt, scale, n_head, n_head_kv);
    attn_combine_kernel<<<dim3(static_cast<unsigned>(n_head), static_cast<unsigned>(T)), 256, static_cast<size_t>(nc) * sizeof(float), s>>>(
        po, pml, qfull, out, pos0, nc, n_head, true);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace limbo::cuda::dec
