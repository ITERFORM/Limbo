// Copyright (c) 2026 ITERFORM. All rights reserved.
// src/cuda/ink_attn.cu — the inkling2 relative-bias attention with per-head sinks (docs/inkling2.md §3, §6).
//
// Its own kernel, not a variation of dec::attn: the logits are scale*q.k + bias[d][h] where d is the key's
// backward distance (clamped to the bias table's reach, the last row the zero pad), the table already carries
// -sink[h], and the softmax denominator carries the sink column at logit 0. The bias and the sink enter AFTER
// the q.k scale — the fold the oracle's own numbers settled (bench/inkling2-cpu-gate-2026-11-23.md §3).
//
// Split flash-decode like the decode attention (src/cuda/attn_decode.cu): fixed 256-position chunks, one block
// per (KV head, chunk, row), each block's warps one query head of the group (the K row is read once for the
// group), and a combine kernel merges the chunks in position order. Every reduction tree is a fixed shuffle
// tree and the merge order is fixed, so a row's result does not depend on how many rows share the launch.
#include <cfloat>
#include <algorithm>
#include <cmath>

#include "ink_ops.hpp"

namespace limbo::cuda::ink {

namespace {

constexpr int kHD = 128;    // head dimension (the file's, fixed for this kernel)
constexpr int kGMax = 4;    // widest query group per KV head the scorers instantiate (G = 3 here)
constexpr int kGridChunks = 160;
// The decode scorer's chunk for a sliding window: 1,024 positions in chunks of 256 were 5 blocks per KV head, each a
// chain of 256 dependent steps; chunks of 32 are 33 blocks of 32 steps.
constexpr int kChunkW = 32;   // blocks per (KV head, row) of the decode scorer; each walks its chunks

// Four K or V values (dims 4 * lane ..) of cache row `row`, KV head `kv`: f16, or int8 times the row's scale.
template <bool Q8>
__device__ __forceinline__ void kv4(const void* c, const float* sc, size_t row, int n_kv, int kv, int lane, float o[4]) {
    if constexpr (Q8) {
        const char4 b = reinterpret_cast<const char4*>(static_cast<const int8_t*>(c) + (row * n_kv + kv) * kHD)[lane];
        const float s = sc[row * n_kv + kv];
        o[0] = float(b.x) * s;
        o[1] = float(b.y) * s;
        o[2] = float(b.z) * s;
        o[3] = float(b.w) * s;
    } else {
        const __half* h = static_cast<const __half*>(c) + (row * n_kv + kv) * kHD + lane * 4;
#pragma unroll
        for (int j = 0; j < 4; ++j) o[j] = __half2float(h[j]);
    }
}

__device__ __forceinline__ float warp_sum_d(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// The absolute position chunk 0 of a query row starts at: 0 for full attention, the kChunk boundary at or
// below the window's first position otherwise. The grid's chunks are absolute chunks either way.
__device__ __forceinline__ int chunk_base(int q_pos, int window) {
    return window > 0 ? (max(0, q_pos - window + 1) / kChunk) * kChunk : 0;
}
// The same for the decode scorer's chunk size cs (kChunk for full attention, kChunkW for a window).
__device__ __forceinline__ int chunk_base_cs(int q_pos, int window, int cs) {
    return window > 0 ? (max(0, q_pos - window + 1) / cs) * cs : 0;
}

// Chunks the grid launches: every chunk of max_ctx, or with a window the at most window/kChunk + 1 absolute
// chunks a window can touch (never more than max_ctx has).
int grid_chunks(int max_ctx, int window) {
    if (window > 0) return std::min((max_ctx + kChunkW - 1) / kChunkW, (window + kChunkW - 1) / kChunkW + 1);
    return (max_ctx + kChunk - 1) / kChunk;
}

// The chunk scorer: block (chunk, KV head, row), 4 warps = the group's query heads. Each warp keeps one
// online-softmax state per lane over its strided positions (ascending), merges the 32 lane states in fixed
// lane order through shared memory, and writes the chunk's (max, sum, num[hd]) for the combine.
template <bool Q8>
__global__ void ink_score_k(const float* q, const void* kcache, const void* vcache, const float* kscale,
                            const float* vscale, const float* table,
                            float* scratch, const int* pos0, int n_chunks, int ring, int n_head, int n_head_kv,
                            float scale, int extent, int window, size_t table_stride) {
    const int kv = blockIdx.y, t = blockIdx.z;
    const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int G = n_head / n_head_kv;
    if (wid >= G) return;
    const int h = kv * G + wid;

    const int q_pos = pos0[0] + t;
    table += size_t(t) * table_stride;
    // The grid holds at most kGridChunks blocks per (KV head, row): a block walks its chunks, and stops at the first
    // past the row's own position -- the allocated length beyond the filled one costs nothing. The partials of a
    // chunk are the same whichever block computes them, and the combine reads only chunks up to the position.
    const int cs = window > 0 ? kChunkW : kChunk;
    for (int c = blockIdx.x; c < n_chunks; c += gridDim.x) {
    const int base = chunk_base_cs(q_pos, window, cs) + c * cs;
    if (base > q_pos) break;
    int first = base, last = base + cs - 1;
    if (window > 0) first = max(first, q_pos - window + 1);
    last = min(last, q_pos);

    float m = -INFINITY, s = 0.0f, num[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float* qh = q + size_t(t) * n_head * kHD + size_t(h) * kHD;
    // One position per iteration: its dot is split over the lanes and summed in the fixed shuffle tree (the
    // lanes all work the same position, so the reduction is well defined), and each lane keeps its own 4 dims
    // of the numerator. The online-softmax state (m, s) is warp-uniform by construction.
    const int n_pos = max(0, last - first + 1);
    int crow = ring > 0 ? first % ring : first;   // the cache row, advanced by one a step (no modulo in the loop)
    float tb = n_pos > 0 ? table[size_t(min(q_pos - first, extent)) * n_head + size_t(h)] : 0.0f;
    for (int i = 0; i < n_pos; ++i) {
        const int pos = first + i;
        // the next step's bias, loaded a step ahead (its latency hides behind this step)
        const float tb_next =
            i + 1 < n_pos ? table[size_t(min(q_pos - pos - 1, extent)) * n_head + size_t(h)] : 0.0f;
        float kk[4], vv[4];
        kv4<Q8>(kcache, kscale, size_t(crow), n_head_kv, kv, lane, kk);
        kv4<Q8>(vcache, vscale, size_t(crow), n_head_kv, kv, lane, vv);
        float qk = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int d = lane * 4 + j;
            qk += qh[d] * kk[j];
        }
        qk = warp_sum_d(qk);
        const float l = scale * qk + tb;
        const float m2 = fmaxf(m, l);
        const float e = __expf(m - m2);
        const float el = __expf(l - m2);
        s = s * e + el;
#pragma unroll
        for (int j = 0; j < 4; ++j) num[j] = num[j] * e + el * vv[j];
        m = m2;
        tb = tb_next;
        if (++crow == ring) crow = 0;
    }

    float* row = scratch + ((size_t(t) * n_chunks + c) * n_head + size_t(h)) * (2 + kHD);
    if (lane == 0) {
        row[0] = m;
        row[1] = s;
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) row[2 + lane * 4 + i] = num[i];
    }
}

// The combine: block (row, head), 128 threads (one per dim); the chunks merge in position order, then the sink
// column at logit 0 joins the max and the denominator.
__global__ void ink_combine_k(const float* scratch, float* out, const int* pos0, int n_chunks, int n_head,
                              int window) {
    const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const int q_pos = pos0[0] + t;
    const int cs = window > 0 ? kChunkW : kChunk;
    const int base = chunk_base_cs(q_pos, window, cs);

    float m = 0.0f;   // the sink column's logit is 0 and joins the max
    for (int c = 0; c < n_chunks; ++c) {
        if (base + c * cs > q_pos) break;
        const float* row = scratch + ((size_t(t) * n_chunks + c) * n_head + size_t(h)) * (2 + kHD);
        m = fmaxf(m, row[0]);
    }
    float s = __expf(-m);   // the sink column
    float num = 0.0f;
    for (int c = 0; c < n_chunks; ++c) {
        if (base + c * cs > q_pos) break;
        const float* row = scratch + ((size_t(t) * n_chunks + c) * n_head + size_t(h)) * (2 + kHD);
        if (row[0] <= -INFINITY) continue;
        const float e = __expf(row[0] - m);
        s += row[1] * e;
        num += row[2 + d] * e;
    }
    out[size_t(t) * n_head * kHD + size_t(h) * kHD + d] = num / s;
}

// ---- the prompt pass's attention: up to kMaxRows consecutive query rows of one KV head share every K/V tile ----
//
// Block (chunk, KV head), 8 warps. The T rows x G query heads (at most 32 queries) sit in shared memory; the chunk's
// keys are walked in tiles of 32 (one key per lane), each tile's K and V staged in shared memory once for all the
// queries of the block. blockIdx.z picks 8 of the queries, warp w owns one; per tile it scores its query against the 32 keys (the
// lanes), folds the tile into its online softmax and accumulates V (lane L owns dims 4L..4L+3). A query's row reads
// its own bias table (table + t * tstride). The chunk grid starts at the window's first kChunk boundary of the first
// row; the partials are combined per row by ink_combine_rows_k, the sink column at logit 0 joining as before.
constexpr int kKT = 32;   // keys per tile
constexpr int kKP = kHD + 2;   // K row pitch in halves (odd word pitch: the lanes' rows hit distinct banks)

__device__ __forceinline__ float warp_max_d(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

template <bool Q8>
__global__ void __launch_bounds__(256) ink_score_rows_k(const float* q, const void* kcache, const void* vcache,
                                                        const float* kscale, const float* vscale,
                                                        const float* table, size_t tstride, float* scratch,
                                                        const int* pos0, int T, int n_chunks, int ring, int n_head,
                                                        int n_head_kv, float scale, int extent, int window) {
    __shared__ float Qs[8][kHD];
    __shared__ __half Ks[kKT][kKP];
    __shared__ __half Vs[kKT][kHD];
    const int kv = blockIdx.y, tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
    const int G = n_head / n_head_kv, nq = T * G;
    const int q_lo = blockIdx.z * 8;   // this block's 8 queries (one per warp): q_lo .. q_lo + 7
    const int p0 = pos0[0];
    const int lo_all = window > 0 ? max(0, p0 - window + 1) : 0, hi_all = p0 + T - 1;

    for (int i = tid; i < 8 * kHD; i += 256) {
        const int qi = q_lo + i / kHD, d = i % kHD;
        if (qi >= nq) break;
        const int t = qi / G, h = kv * G + qi % G;
        Qs[qi - q_lo][d] = q[size_t(t) * n_head * kHD + size_t(h) * kHD + d];
    }

    // a capped grid: a block walks its chunks up to the group's last row (the allocated length beyond costs nothing)
    for (int c = blockIdx.x; c < n_chunks; c += gridDim.x) {
    const int base = chunk_base(p0, window) + c * kChunk;
    if (base > hi_all) break;
    float m[4], s[4], acc[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        m[j] = -INFINITY;
        s[j] = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[j][i] = 0.0f;
    }

    for (int k0 = base; k0 < base + kChunk; k0 += kKT) {
        if (k0 > hi_all) break;
        if (k0 + kKT - 1 < lo_all) continue;
        __syncthreads();
        for (int i = tid; i < kKT * (kHD / 2); i += 256) {
            const int r = i / (kHD / 2), d2 = i % (kHD / 2), pos = k0 + r;
            __half2 kk = __float2half2_rn(0.0f), vv = __float2half2_rn(0.0f);
            if (pos >= lo_all && pos <= hi_all) {
                const size_t row = size_t(ring > 0 ? pos % ring : pos);
                if constexpr (Q8) {
                    const size_t o = (row * n_head_kv + kv) * kHD + 2 * d2;
                    const int8_t* kb = static_cast<const int8_t*>(kcache) + o;
                    const int8_t* vb = static_cast<const int8_t*>(vcache) + o;
                    const float ks = kscale[row * n_head_kv + kv], vs = vscale[row * n_head_kv + kv];
                    kk = __floats2half2_rn(float(kb[0]) * ks, float(kb[1]) * ks);
                    vv = __floats2half2_rn(float(vb[0]) * vs, float(vb[1]) * vs);
                } else {
                    kk = reinterpret_cast<const __half2*>(static_cast<const __half*>(kcache) + row * n_head_kv * kHD +
                                                          size_t(kv) * kHD)[d2];
                    vv = reinterpret_cast<const __half2*>(static_cast<const __half*>(vcache) + row * n_head_kv * kHD +
                                                          size_t(kv) * kHD)[d2];
                }
            }
            Ks[r][2 * d2] = __low2half(kk);
            Ks[r][2 * d2 + 1] = __high2half(kk);
            reinterpret_cast<__half2*>(&Vs[r][0])[d2] = vv;
        }
        __syncthreads();
        const int key = k0 + lane;
#pragma unroll
        for (int j = 0; j < 1; ++j) {
            const int qi = q_lo + w;
            if (qi >= nq) break;
            const int t = qi / G, h = kv * G + qi % G, qpos = p0 + t;
            const int first = window > 0 ? max(0, qpos - window + 1) : 0;
            const bool valid = key >= first && key <= qpos;
            float dot = 0.0f;
            const float2* qv = reinterpret_cast<const float2*>(&Qs[w][0]);
#pragma unroll 8
            for (int d2 = 0; d2 < kHD / 2; ++d2) {
                const float2 qq = qv[d2];
                dot += qq.x * __half2float(Ks[lane][2 * d2]);
                dot += qq.y * __half2float(Ks[lane][2 * d2 + 1]);
            }
            const float l = valid ? scale * dot + table[size_t(t) * tstride + size_t(min(qpos - key, extent)) * n_head + h]
                                  : -INFINITY;
            const float mt = warp_max_d(l);
            if (mt == -INFINITY) continue;
            const float m2 = fmaxf(m[j], mt);
            const float e = __expf(m[j] - m2);
            const float p = valid ? __expf(l - m2) : 0.0f;
            s[j] = s[j] * e + warp_sum_d(p);
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[j][i] *= e;
#pragma unroll 8
            for (int kk = 0; kk < kKT; ++kk) {
                const float pk = __shfl_sync(0xffffffffu, p, kk);
                const __half2* vr = reinterpret_cast<const __half2*>(&Vs[kk][lane * 4]);
                const float2 v01 = __half22float2(vr[0]), v23 = __half22float2(vr[1]);
                acc[j][0] += pk * v01.x;
                acc[j][1] += pk * v01.y;
                acc[j][2] += pk * v23.x;
                acc[j][3] += pk * v23.y;
            }
            m[j] = m2;
        }
    }

#pragma unroll
    for (int j = 0; j < 1; ++j) {
        const int qi = q_lo + w;
        if (qi >= nq) break;
        const int t = qi / G, h = kv * G + qi % G;
        float* row = scratch + ((size_t(t) * n_chunks + c) * n_head + size_t(h)) * (2 + kHD);
        if (lane == 0) {
            row[0] = m[j];
            row[1] = s[j];
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) row[2 + lane * 4 + i] = acc[j][i];
    }
    __syncthreads();   // the next chunk reuses the tiles
    }
}

// The combine of ink_score_rows_k's partials: block (row, head), one thread per dim, the chunks in position order
// from the group's base, the sink column at logit 0.
__global__ void ink_combine_rows_k(const float* scratch, float* out, const int* pos0, int n_chunks, int n_head,
                                   int window) {
    const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const int p0 = pos0[0], q_pos = p0 + t;
    const int base = chunk_base(p0, window);
    float m = 0.0f;
    for (int c = 0; c < n_chunks; ++c) {
        if (base + c * kChunk > q_pos) break;
        m = fmaxf(m, scratch[((size_t(t) * n_chunks + c) * n_head + size_t(h)) * (2 + kHD)]);
    }
    float s = __expf(-m), num = 0.0f;
    for (int c = 0; c < n_chunks; ++c) {
        if (base + c * kChunk > q_pos) break;
        const float* row = scratch + ((size_t(t) * n_chunks + c) * n_head + size_t(h)) * (2 + kHD);
        if (row[0] <= -INFINITY) continue;
        const float e = __expf(row[0] - m);
        s += row[1] * e;
        num += row[2 + d] * e;
    }
    out[size_t(t) * n_head * kHD + size_t(h) * kHD + d] = num / s;
}

}  // namespace

bool attention_rows(const float* q, const __half* kcache, const __half* vcache, const float* table, size_t tstride,
                    float* out, float* scratch, const int* pos0, int T, int max_ctx, int ring, int n_head,
                    int n_head_kv, int hd, float scale, int extent, int window, cudaStream_t s, int last_pos,
                    const float* kscale, const float* vscale) {
    if (hd != kHD || n_head % n_head_kv != 0 || n_head / n_head_kv > kGMax || T < 1 || T > kMaxRows) return false;
    if (T * (n_head / n_head_kv) > 32) return false;
    if (ring < 0 || (ring > 0 && (window <= 0 || ring < window + T - 1))) return false;
    const int all = (max_ctx + kChunk - 1) / kChunk;
    // the chunks the group's keys can touch: from the first row's window base to the last row
    // full attention: the chunks up to the group's last position (last_pos, host-known), not the allocated length
    const int n_chunks = window > 0 ? std::min(all, (window + T - 1 + kChunk - 1) / kChunk + 1)
                                    : (last_pos >= 0 ? std::min(all, last_pos / kChunk + 1) : all);
    (kscale ? ink_score_rows_k<true> : ink_score_rows_k<false>)<<<dim3(std::min(n_chunks, kGridChunks), n_head_kv, (T * (n_head / n_head_kv) + 7) / 8), 256, 0, s>>>(q, kcache, vcache, kscale, vscale, table, tstride, scratch, pos0, T,
                                                               n_chunks, ring, n_head, n_head_kv, scale, extent,
                                                               window);
    ink_combine_rows_k<<<dim3(T, n_head), kHD, 0, s>>>(scratch, out, pos0, n_chunks, n_head, window);
    return cudaGetLastError() == cudaSuccess;
}

size_t attention_scratch_floats(int max_ctx, int max_rows, int n_head, int hd) {
    (void)hd;
    // full attention's chunks, or a window's of kChunkW (up to 4,096 positions of window)
    const size_t n_chunks = std::max((size_t(max_ctx) + kChunk - 1) / kChunk,
                                     std::min((size_t(max_ctx) + kChunkW - 1) / kChunkW, size_t(4096 / kChunkW + 1)));
    return size_t(max_rows) * n_chunks * size_t(n_head) * (2 + kHD);
}

bool attention(const float* q, const __half* kcache, const __half* vcache, const float* table, float* out,
               float* scratch, const int* pos0, int T, int max_ctx, int ring, int n_head, int n_head_kv,
               int hd, float scale, int extent, int window, cudaStream_t s, size_t table_stride, const float* kscale,
               const float* vscale) {
    if (hd != kHD || n_head % n_head_kv != 0 || n_head / n_head_kv > kGMax || T < 1 || T > kMaxRows) return false;
    // A ring must hold the window plus the rows stored ahead of the earliest query row.
    if (ring < 0 || (ring > 0 && (window <= 0 || ring < window + T - 1))) return false;
    const int n_chunks = grid_chunks(max_ctx, window);
    const dim3 grid(std::min(n_chunks, kGridChunks), n_head_kv, T);
    (kscale ? ink_score_k<true> : ink_score_k<false>)<<<grid, 4 * 32, 0, s>>>(q, kcache, vcache, kscale, vscale, table, scratch, pos0, n_chunks, ring, n_head,
                                        n_head_kv, scale, extent, window, table_stride);
    ink_combine_k<<<dim3(T, n_head), kHD, 0, s>>>(scratch, out, pos0, n_chunks, n_head, window);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace limbo::cuda::ink
