// Copyright (c) 2026 ITERFORM. All rights reserved.
// src/cuda/q4e_qsa.cu — qwen4exp's sparse attention ("QSA", docs/flash-next.md 2.3) on the device: the indexer's
// key ring and pooled block keys, the block scores, the top-k selection and the attention over the selected cells.
// The arithmetic of the scores and the pooled keys follows Ref::qsa_select (src/step.cpp) operation for operation
// (no fused multiply-add where the reference has none), so the selection is the reference's.
#include "q4e_qsa.hpp"

#include <cfloat>
#include <cmath>

namespace limbo::cuda::qsa {

namespace {

constexpr int kDim = 128;   // indexer head width
constexpr int kHD = 256;    // attention head width
constexpr int kSelChunk = 64;   // cells gathered at a time
constexpr int kPart = 512;      // cells a partial covers (a block's run of chunks)
constexpr int kKS = 258, kVS = 264;   // shared row strides (halves): K odd in words (column reads), V 16-byte rows

__device__ __forceinline__ float h2f(__half h) { return __half2float(h); }

// rmsnorm over 128 values held one per thread (double sum, as Ref's rmsnorm), times w.
__device__ __forceinline__ float rms128(float v, const float* w, double* sh) {
    const int d = threadIdx.x;
    double x = double(v) * double(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
    if ((d & 31) == 0) sh[d >> 5] = x;
    __syncthreads();
    const double ss = sh[0] + sh[1] + sh[2] + sh[3];
    __syncthreads();
    const float inv = float(1.0 / sqrt(ss / kDim + double(kEps)));
    return __fmul_rn(__fmul_rn(v, inv), w ? w[d] : 1.0f);
}

// NeoX rotation of the first rope_dim of 128 values (one per thread), at position p (Ref's rope).
__device__ __forceinline__ float rope128(float v, int p, int rope_dim, double base, float* sv) {
    const int d = threadIdx.x, pairs = rope_dim / 2;
    sv[d] = v;
    __syncthreads();
    float r = v;
    if (d < rope_dim) {
        const int i = d < pairs ? d : d - pairs;
        const double theta = double(p) / pow(base, 2.0 * i / double(rope_dim));
        const float c = float(cos(theta)), s = float(sin(theta));
        const float a = sv[i], b = sv[i + pairs];
        r = d < pairs ? __fsub_rn(__fmul_rn(a, c), __fmul_rn(b, s)) : __fadd_rn(__fmul_rn(a, s), __fmul_rn(b, c));
    }
    __syncthreads();
    return r;
}

// One block of 128 threads, the rows in order: each row's raw key into the ring, the block it completes pooled,
// and its query normed and rotated.
__global__ void index_k(const float* kraw, const float* qraw, const float* ikn, const float* iqn, __half* ring,
                        __half* pool, float* qn, const int* pos0, int T, int kp, int heads, int rope_dim, double base) {
    __shared__ double sh[4];
    __shared__ float sv[kDim];
    const int d = threadIdx.x;
    for (int t = 0; t < T; ++t) {
        const int pos = pos0[0] + t;
        ring[size_t(pos & 15) * kDim + d] = __float2half_rn(kraw[size_t(t) * kDim + d]);
        __syncthreads();
        if ((pos + 1) % kp == 0) {
            const int j = (pos + 1) / kp - 1;
            float m = 0.0f;
            for (int i = 0; i < kp; ++i) m = __fadd_rn(m, h2f(ring[size_t((j * kp + i) & 15) * kDim + d]));
            m = __fmul_rn(m, 1.0f / float(kp));
            float y = rms128(m, ikn, sh);
            y = rope128(y, j * kp, rope_dim, base, sv);
            pool[size_t(j) * kDim + d] = __float2half_rn(y);
        }
        for (int h = 0; h < heads; ++h) {
            float y = rms128(qraw[(size_t(t) * heads + h) * kDim + d], iqn, sh);
            y = rope128(y, pos, rope_dim, base, sv);
            qn[(size_t(t) * heads + h) * kDim + d] = y;
        }
        __syncthreads();
    }
}

// s_j = sum_h relu(q_h . pooled_j) / sqrt(128), a thread a block j, the reference's order of operations.
__global__ void score_k(const float* qn, const __half* pool, float* scores, size_t stride, const int* pos0, int kp,
                        int heads, float w) {
    __shared__ float q[4 * kDim];
    const int t = blockIdx.y;
    for (int i = threadIdx.x; i < heads * kDim; i += blockDim.x) q[i] = qn[size_t(t) * heads * kDim + i];
    __syncthreads();
    const int nv = (pos0[0] + t + 1) / kp;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= nv) return;
    const __half* pj = pool + size_t(j) * kDim;
    float kf[kDim];
#pragma unroll 16
    for (int d = 0; d < kDim; ++d) kf[d] = h2f(pj[d]);
    float sc = 0.0f;
    for (int h = 0; h < heads; ++h) {
        float qk = 0.0f;
#pragma unroll 16
        for (int d = 0; d < kDim; ++d) qk = __fadd_rn(qk, __fmul_rn(q[h * kDim + d], kf[d]));
        sc = __fadd_rn(sc, __fmul_rn(fmaxf(qk, 0.0f), w));
    }
    scores[size_t(t) * stride + j] = sc;
}

// The same three results for a long run of rows, in parallel (index_rows_par). Pooling: a block per block j this run
// completes, its kp raw keys from the run (rounded as the ring rounds them) or, before the run, from the ring.
__global__ void pool_par_k(const float* kraw, const float* ikn, const __half* ring, __half* pool, int pos0, int jlo, int kp,
                           int rope_dim, double base) {
    __shared__ double sh[4];
    __shared__ float sv[kDim];
    const int d = threadIdx.x, j = jlo + blockIdx.x;
    float m = 0.0f;
    for (int i = 0; i < kp; ++i) {
        const int p = j * kp + i;
        const float v = p >= pos0 ? h2f(__float2half_rn(kraw[size_t(p - pos0) * kDim + d])) : h2f(ring[size_t(p & 15) * kDim + d]);
        m = __fadd_rn(m, v);
    }
    m = __fmul_rn(m, 1.0f / float(kp));
    float y = rms128(m, ikn, sh);
    y = rope128(y, j * kp, rope_dim, base, sv);
    pool[size_t(j) * kDim + d] = __float2half_rn(y);
}

__global__ void qn_par_k(const float* qraw, const float* iqn, float* qn, int pos0, int heads, int rope_dim, double base) {
    __shared__ double sh[4];
    __shared__ float sv[kDim];
    const int d = threadIdx.x, t = blockIdx.x, h = blockIdx.y;
    float y = rms128(qraw[(size_t(t) * heads + h) * kDim + d], iqn, sh);
    y = rope128(y, pos0 + t, rope_dim, base, sv);
    qn[(size_t(t) * heads + h) * kDim + d] = y;
}

// The ring after the run: its last 16 positions' raw keys.
__global__ void ring_par_k(const float* kraw, __half* ring, int pos0, int T) {
    const int p = pos0 + T - 1 - int(blockIdx.x), d = threadIdx.x;
    ring[size_t(p & 15) * kDim + d] = __float2half_rn(kraw[size_t(p - pos0) * kDim + d]);
}

// Block-wide exclusive scan of one int per thread (1024 threads); returns the exclusive prefix, *total the sum.
__device__ int block_scan(int v, int* sh, int* total) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    int x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) sh[w] = x;
    __syncthreads();
    if (w == 0) {
        int s = lane < int(blockDim.x >> 5) ? sh[lane] : 0;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, s, o);
            if (lane >= o) s += y;
        }
        sh[lane] = s;
    }
    __syncthreads();
    const int before = (w > 0 ? sh[w - 1] : 0) + x - v;
    *total = sh[int(blockDim.x >> 5) - 1];
    __syncthreads();
    return before;
}

__device__ __forceinline__ uint32_t key_of(float s) { return __float_as_uint(fmaxf(s, 0.0f) + 0.0f); }

// The n_top best blocks of each row (score descending, the lower index on a tie: Ref's partial_sort), written in
// ascending block order. A radix select finds the n_top-th key; one ordered pass then keeps every key above it and
// the first ones equal to it. One block of 1024 threads per row.
__global__ void select_k(const float* scores, size_t stride, int32_t* sel, const int* pos0, int kp, int n_top) {
    __shared__ int hist[256];
    __shared__ int sh[32];
    __shared__ uint32_t s_prefix;
    __shared__ int s_rem;
    const int t = blockIdx.x;
    const int nv = (pos0[0] + t + 1) / kp;
    if (nv <= n_top) return;
    const float* sc = scores + size_t(t) * stride;
    uint32_t prefix = 0, mask = 0;
    int rem = n_top;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
        __syncthreads();
        for (int j = threadIdx.x; j < nv; j += blockDim.x) {
            const uint32_t k = key_of(sc[j]);
            if ((k & mask) == prefix) atomicAdd(&hist[(k >> shift) & 255], 1);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int cum = 0, b = 255;
            for (; b > 0; --b) {
                if (cum + hist[b] >= rem) break;
                cum += hist[b];
            }
            s_prefix = prefix | (uint32_t(b) << shift);
            s_rem = rem - cum;
        }
        __syncthreads();
        prefix = s_prefix;
        rem = s_rem;
        mask |= 255u << shift;
        __syncthreads();
    }
    const uint32_t tau = prefix;   // the n_top-th key; `rem` of the keys equal to it are kept
    const int seg = (nv + int(blockDim.x) - 1) / int(blockDim.x);
    const int lo = threadIdx.x * seg, hi = min(nv, lo + seg);
    int n_eq = 0;
    for (int j = lo; j < hi; ++j) n_eq += key_of(sc[j]) == tau;
    int tot;
    const int eq_before = block_scan(n_eq, sh, &tot);
    int n_sel = 0, e = eq_before;
    for (int j = lo; j < hi; ++j) {
        const uint32_t k = key_of(sc[j]);
        if (k > tau) ++n_sel;
        else if (k == tau && e++ < rem) ++n_sel;
    }
    int out = block_scan(n_sel, sh, &tot);
    e = eq_before;
    int32_t* o = sel + size_t(t) * n_top;
    for (int j = lo; j < hi; ++j) {
        const uint32_t k = key_of(sc[j]);
        if (k > tau || (k == tau && e++ < rem)) o[out++] = j;
    }
}

__device__ __forceinline__ int page_of(const int* pt, int pos) { return pt ? pt[pos / kPage] : pos / kPage; }

// The attention over the selected cells: block (chunk of kSelChunk cells, KV head, row), a warp per query head of the
// group. The chunk's cells -- the n_top blocks' kp positions each, then the tail nv*kp .. pos -- are gathered once
// into shared memory for all the group's heads: K rows first (a lane a cell for the scores), then V (the paged V
// is [dims][positions], so it is read cell-fastest and transposed; a lane takes 8 dims of the products).
// A Q8 pool (kq non-null; include/limbo/cuda/kv.hpp: rows [position][kv head][256] int8, a half scale per 32): the
// gathers dequantise into the same f16 buffer, so everything after them is the f16 path's.
__global__ void attn_part_k(const float* q, const __half* kc, const __half* vc, const int32_t* sel, const int* pos0,
                            const int* pt, int vpg, int vst, int kp, int n_top, int n_head, int n_kv, float scale,
                            float* po, float* pml, int n_chunks, const int8_t* kq, const uint16_t* ks, const int8_t* vq,
                            const uint16_t* vs) {
    extern __shared__ __align__(16) unsigned char sm[];
    const int c = blockIdx.x, kh = blockIdx.y, t = blockIdx.z, w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int G = n_head / n_kv, nthr = blockDim.x;
    __half* skv = reinterpret_cast<__half*>(sm);                                       // [kSelChunk][kVS]
    float* sq = reinterpret_cast<float*>(sm + size_t(kSelChunk) * kVS * sizeof(__half));  // [G][kHD]
    float* sp = sq + size_t(G) * kHD;                                                   // [G][kSelChunk]
    int* spos = reinterpret_cast<int*>(sp + size_t(G) * kSelChunk);                     // [kSelChunk] K row offsets
    int* svb = spos + kSelChunk;                                                        // [kSelChunk] V page + position
    const int pos = pos0[0] + t, nv = (pos + 1) / kp;
    const bool plain = nv <= n_top;   // no selection yet: every position up to the row's own (at most n_top * kp + kp - 1)
    const int n_cells = plain ? pos + 1 : n_top * kp + (pos + 1 - nv * kp);
    if (c * kPart >= n_cells) return;   // the combine reads only the row's own partials
    const int32_t* sl = sel + size_t(t) * n_top;
    for (int i = threadIdx.x; i < G * kHD; i += nthr) sq[i] = q[(size_t(t) * n_head + size_t(kh) * G) * kHD + i];
    const int h = kh * G + w;
    constexpr int kCpl = kSelChunk / 32;   // cells a lane scores
    float m_run = -INFINITY, s_run = 0.0f, num[8] = {};
    for (int e0 = c * kPart; e0 < min(n_cells, (c + 1) * kPart); e0 += kSelChunk) {
        const int nc = min(n_cells, e0 + kSelChunk) - e0;
        __syncthreads();   // the previous chunk's V and probabilities are read
        for (int i = threadIdx.x; i < nc; i += nthr) {
            const int e = e0 + i;
            const int p = plain ? e : e < n_top * kp ? sl[e / kp] * kp + e % kp : nv * kp + (e - n_top * kp);
            const int pg = page_of(pt, p), pp = p % kPage;
            if (kq) {
                spos[i] = (pg * kPage + pp) * n_kv + kh;   // the K and V row
            } else {
                spos[i] = (pg * kPage + pp) * n_kv * kHD + kh * kHD;
                svb[i] = pg * vpg + pp + kh * kHD * vst;
            }
        }
        __syncthreads();
        for (int i = w; i < nc; i += G) {
            __half2* dst = reinterpret_cast<__half2*>(skv + i * kKS + lane * 8);
            if (kq) {
                const int64_t row = spos[i];
                const uint2 raw = *reinterpret_cast<const uint2*>(kq + row * kHD + lane * 8);
                const float d = __half2float(__ushort_as_half(ks[row * (kHD / 32) + (lane >> 2)]));
                const int8_t* c8 = reinterpret_cast<const int8_t*>(&raw);
#pragma unroll
                for (int j = 0; j < 4; ++j) dst[j] = __floats2half2_rn(float(c8[2 * j]) * d, float(c8[2 * j + 1]) * d);
            } else {
                const uint4 v = *reinterpret_cast<const uint4*>(kc + int64_t(spos[i]) + lane * 8);
                const __half2* src = reinterpret_cast<const __half2*>(&v);
#pragma unroll
                for (int j = 0; j < 4; ++j) dst[j] = src[j];
            }
        }
        __syncthreads();
        float sc[kCpl];
        float m = -INFINITY;
        if (w < G) {
            const float2* qh = reinterpret_cast<const float2*>(sq + w * kHD);
            float a0[kCpl] = {}, a1[kCpl] = {};
            const __half2* kr = reinterpret_cast<const __half2*>(skv + lane * kKS);
#pragma unroll 4
            for (int d2 = 0; d2 < kHD / 2; ++d2) {
                const float2 qv = qh[d2];
#pragma unroll
                for (int k = 0; k < kCpl; ++k) {
                    const float2 kf = __half22float2(kr[k * 32 * (kKS / 2) + d2]);
                    a0[k] = fmaf(qv.x, kf.x, a0[k]);
                    a1[k] = fmaf(qv.y, kf.y, a1[k]);
                }
            }
#pragma unroll
            for (int k = 0; k < kCpl; ++k) {
                sc[k] = lane + 32 * k < nc ? (a0[k] + a1[k]) * scale : -INFINITY;
                m = fmaxf(m, sc[k]);
            }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        }
        __syncthreads();   // K is read: the buffer takes V
        // cells come in runs of 4 consecutive positions from a multiple of 4 (a block, the plain prefix, the tail),
        // inside one page: a run's V for one dim is 8 bytes
        if (kq) {   // a Q8 V row is the cell's 256 dims: consecutive threads take consecutive dims of one row
            for (int idx = threadIdx.x; idx < kSelChunk * kHD; idx += nthr) {
                const int i = idx / kHD, d = idx % kHD;
                if (i >= nc) continue;
                const int64_t row = spos[i];
                skv[i * kVS + d] = __float2half_rn(float(vq[row * kHD + d]) * __half2float(__ushort_as_half(vs[row * (kHD / 32) + d / 32])));
            }
        }
        for (int idx = threadIdx.x; !kq && idx < (kSelChunk / 4) * kHD; idx += nthr) {
            const int g = idx & (kSelChunk / 4 - 1), d = idx / (kSelChunk / 4), i = 4 * g;
            if (i >= nc) continue;
            const uint2 raw = *reinterpret_cast<const uint2*>(vc + int64_t(svb[i]) + int64_t(d) * vst);
            const __half* hv = reinterpret_cast<const __half*>(&raw);
#pragma unroll
            for (int j = 0; j < 4; ++j) skv[(i + j) * kVS + d] = hv[j];
        }
        if (w < G) {
            const float m2 = fmaxf(m_run, m), ea = __expf(m_run - m2);
            float ssum = 0.0f;
#pragma unroll
            for (int k = 0; k < kCpl; ++k) {
                const int i = lane + 32 * k;
                const float e = i < nc ? __expf(sc[k] - m2) : 0.0f;
                sp[w * kSelChunk + i] = e;
                ssum += e;
            }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) ssum += __shfl_xor_sync(0xffffffffu, ssum, o);
            s_run = s_run * ea + ssum;
#pragma unroll
            for (int j = 0; j < 8; ++j) num[j] *= ea;
            m_run = m2;
        }
        __syncthreads();
        if (w < G) {
            const float* ph = sp + w * kSelChunk;
            for (int i = 0; i < nc; ++i) {
                const float pv = ph[i];
                const uint4 raw = *reinterpret_cast<const uint4*>(skv + i * kVS + lane * 8);
                const __half2* v2 = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 f = __half22float2(v2[j]);
                    num[2 * j] = fmaf(pv, f.x, num[2 * j]);
                    num[2 * j + 1] = fmaf(pv, f.y, num[2 * j + 1]);
                }
            }
        }
    }
    if (w >= G) return;
    const size_t row = (size_t(t) * n_head + h) * n_chunks + c;
    if (lane == 0) {
        pml[row * 2] = m_run;
        pml[row * 2 + 1] = s_run;
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) po[row * kHD + lane * 8 + j] = num[j];
}

// out = merged / sum * sigmoid(gate): block (head, row), a thread a dim.
__global__ void attn_combine_k(const float* po, const float* pml, const float* qfull, float* out, const int* pos0, int kp,
                               int n_top, int n_head, int n_chunks) {
    const int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    const int pos = pos0[0] + t, nv = (pos + 1) / kp;
    const int n_cells = nv <= n_top ? pos + 1 : n_top * kp + (pos + 1 - nv * kp);
    const int nc = (n_cells + kPart - 1) / kPart;
    const size_t base = (size_t(t) * n_head + h) * n_chunks;
    float M = -INFINITY;
    for (int c = 0; c < nc; ++c) M = fmaxf(M, pml[(base + c) * 2]);
    float L = 0.0f, O = 0.0f;
    for (int c = 0; c < nc; ++c) {
        const float mc = pml[(base + c) * 2];
        if (mc == -INFINITY) continue;
        const float wc = __expf(mc - M);
        L += wc * pml[(base + c) * 2 + 1];
        O += wc * po[(base + c) * kHD + d];
    }
    const size_t row = size_t(t) * n_head + h;
    const float g = 1.0f / (1.0f + expf(-qfull[row * 2 * kHD + kHD + d]));
    out[row * kHD + d] = (L > 0.0f ? O / L : 0.0f) * g;
}

}  // namespace

bool index_rows_par(const float* kraw, const float* qraw, const float* ikn, const float* iqn, __half* ring, __half* pool,
                    float* qn, int pos0, int T, int kp, int heads, int rope_dim, double base, cudaStream_t s) {
    if (T < 1 || heads > 4 || kp < 1 || kp > 8) return false;
    const int jlo = (pos0 + 1 + kp - 1) / kp - 1, jhi = (pos0 + T) / kp - 1;
    if (jhi >= jlo) pool_par_k<<<unsigned(jhi - jlo + 1), kDim, 0, s>>>(kraw, ikn, ring, pool, pos0, jlo, kp, rope_dim, base);
    qn_par_k<<<dim3(unsigned(T), unsigned(heads)), kDim, 0, s>>>(qraw, iqn, qn, pos0, heads, rope_dim, base);
    ring_par_k<<<unsigned(std::min(T, 16)), kDim, 0, s>>>(kraw, ring, pos0, T);
    return cudaGetLastError() == cudaSuccess;
}

bool index_rows(const float* kraw, const float* qraw, const float* ikn, const float* iqn, __half* ring, __half* pool,
                float* qn, const int* pos0, int T, int kp, int heads, int rope_dim, double base, cudaStream_t s) {
    if (T < 1 || heads > 4 || kp < 1 || kp > 8) return false;
    index_k<<<1, kDim, 0, s>>>(kraw, qraw, ikn, iqn, ring, pool, qn, pos0, T, kp, heads, rope_dim, base);
    return cudaGetLastError() == cudaSuccess;
}

bool select_rows(const float* qn, const __half* pool, float* scores, size_t stride, int32_t* sel, const int* pos0, int T,
                 int nv_max, int kp, int heads, int n_top, cudaStream_t s) {
    if (T < 1 || nv_max < 1) return false;
    score_k<<<dim3(unsigned((nv_max + 255) / 256), unsigned(T)), 256, 0, s>>>(qn, pool, scores, stride, pos0, kp, heads,
                                                                            1.0f / std::sqrt(float(kDim)));
    select_k<<<T, 1024, 0, s>>>(scores, stride, sel, pos0, kp, n_top);
    return cudaGetLastError() == cudaSuccess;
}

size_t attn_scratch_floats(int T, int n_head, int n_top, int kp) {
    const int nc = (n_top * kp + kp + kPart - 1) / kPart;
    return size_t(T) * n_head * nc * (kHD + 2);
}

bool attention_rows(const float* q, const __half* kc, const __half* vc, const float* qfull, float* out, float* scratch,
                    const int32_t* sel, const int* pos0, const int* pt, int vpitch, int T, int kp, int n_top, int n_head,
                    int n_kv, float scale, cudaStream_t s, const int8_t* kq, const uint16_t* ks, const int8_t* vq, const uint16_t* vs) {
    const int G = n_head / n_kv;
    if (T < 1 || G > 32 || n_head % n_kv || kp % 4) return false;
    const int nc = (n_top * kp + kp + kPart - 1) / kPart;
    float* po = scratch;
    float* pml = scratch + size_t(T) * n_head * nc * kHD;
    const int vpg = pt ? n_kv * kHD * kPage : kPage, vst = pt ? kPage : vpitch;
    const size_t smem = size_t(kSelChunk) * kVS * sizeof(__half) + size_t(G) * (kHD + kSelChunk) * sizeof(float) +
                        2 * size_t(kSelChunk) * sizeof(int);
    static size_t set = 0;
    if (smem > set) {
        if (cudaFuncSetAttribute(attn_part_k, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem)) != cudaSuccess) return false;
        set = smem;
    }
    attn_part_k<<<dim3(unsigned(nc), unsigned(n_kv), unsigned(T)), 32 * G, smem, s>>>(
        q, kc, vc, sel, pos0, pt, vpg, vst, kp, n_top, n_head, n_kv, scale, po, pml, nc, kq, ks, vq, vs);
    attn_combine_k<<<dim3(unsigned(n_head), unsigned(T)), kHD, 0, s>>>(po, pml, qfull, out, pos0, kp, n_top, n_head, nc);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace limbo::cuda::qsa
