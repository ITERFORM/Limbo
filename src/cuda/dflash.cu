// Copyright (c) 2026 ITERFORM. All rights reserved.
// src/cuda/dflash.cu — the DFlash2 block drafter (include/limbo/cuda/dflash.hpp, docs/dflash2.md): loader, kernels, ingest, draft.
#include "limbo/cuda/dflash.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <sstream>

#include "dec_ops.hpp"
#include "limbo/cuda/gemv.hpp"
#include "limbo/gguf.hpp"
#include "limbo/jinja.hpp"
#include "limbo/quants.hpp"
#include "limbo/safetensors.hpp"
#include "limbo/types.hpp"
#include "ops.hpp"

namespace limbo::cuda {

namespace {

constexpr uint32_t kBF16 = static_cast<uint32_t>(Type::BF16);
constexpr int kRows = 8;      // the block, and the most rows of one ingest launch
constexpr int kMaxDraftRows = kRows - 1;
constexpr int kSplits = 32;   // key slices of the attention
constexpr int kTopParts = 64; // slices of the vocabulary in the first top-k pass

// what changes from one draft to the next, read by the kernels from device memory so one captured graph serves every slot and position
struct DArgs {
    int32_t anchor, pos, lo_eff, nctx;  // the anchor token and its position, the lowest context position seen, the context positions
    uint64_t ring;                      // the slot's context cache (device pointer)
};

__device__ __forceinline__ float bf2f(__nv_bfloat16 v) { return __bfloat162float(v); }

__device__ __forceinline__ float warp_sum_f(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float block_sum_f(float v, float* sm) {
    v = warp_sum_f(v);
    if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = v;
    __syncthreads();
    float r = 0;
    for (int i = 0; i < static_cast<int>(blockDim.x >> 5); ++i) r += sm[i];
    __syncthreads();
    return r;
}

// (value, index): the larger value wins, the lower index on a tie
__device__ __forceinline__ bool better_vi(float v1, int i1, float v2, int i2) { return v1 > v2 || (v1 == v2 && i1 < i2); }

__device__ __forceinline__ void block_argmax_vi(float& v, int& idx, float* sv, int* si) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, v, o);
        const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
        if (better_vi(ov, oi, v, idx)) {
            v = ov;
            idx = oi;
        }
    }
    if ((threadIdx.x & 31) == 0) {
        sv[threadIdx.x >> 5] = v;
        si[threadIdx.x >> 5] = idx;
    }
    __syncthreads();
    v = sv[0];
    idx = si[0];
    for (int i = 1; i < static_cast<int>(blockDim.x >> 5); ++i)
        if (better_vi(sv[i], si[i], v, idx)) {
            v = sv[i];
            idx = si[i];
        }
    __syncthreads();
}

// the features of a layer's input: fp32 residual rows -> BF16 columns col0.. of the stage
__global__ void df_capture_k(const float* __restrict__ x, int E, __nv_bfloat16* __restrict__ stage, int stride, int col0) {
    const int t = blockIdx.x;
    for (int d = threadIdx.x; d < E; d += blockDim.x) stage[size_t(t) * stride + col0 + d] = __float2bfloat16(x[size_t(t) * E + d]);
}

// stage rows r0 .. r0 + n -> fp32 rows of `stride` columns
__global__ void df_unstage_k(const __nv_bfloat16* __restrict__ stage, int r0, int stride, float* __restrict__ out) {
    const int t = blockIdx.x;
    for (int d = threadIdx.x; d < stride; d += blockDim.x) out[size_t(t) * stride + d] = bf2f(stage[size_t(r0 + t) * stride + d]);
}

__global__ void df_tokens_k(int32_t* tok, const DArgs* a, int32_t mask) { tok[threadIdx.x] = threadIdx.x == 0 ? a->anchor : mask; }

// The short causal convolution with per-token coefficients (docs/dflash2.md): channels in groups of `group`, one coefficient per
// (token, side, tap, group) from the projection plus a per-(side, tap, channel) base; tap 1 reads the previous row of the block
// (zero before row 0). dyn rows are [side][tap][group] = 2 * 2 * 128 floats.
__global__ void df_conv_k(const float* __restrict__ in, const float* __restrict__ dyn, const float* __restrict__ base, int side,
                          float* __restrict__ out, int E, int group, int n_groups) {
    const int t = blockIdx.x;
    const int dyn_w = 2 * 2 * n_groups;
    for (int ch = threadIdx.x; ch < E; ch += blockDim.x) {
        const int g = ch / group;
        const float w0 = dyn[size_t(t) * dyn_w + (side * 2 + 0) * n_groups + g] + base[(side * 2 + 0) * E + ch];
        const float w1 = dyn[size_t(t) * dyn_w + (side * 2 + 1) * n_groups + g] + base[(side * 2 + 1) * E + ch];
        float r = w0 * in[size_t(t) * E + ch];
        if (t > 0) r += w1 * in[size_t(t - 1) * E + ch];
        out[size_t(t) * E + ch] = r;
    }
}

__global__ void df_add_k(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

// RMS norm over one head's 128 dims, then NEOX RoPE at position pos; each block is one (row, head), 128 threads.
__device__ __forceinline__ void head_norm_rope(const float* __restrict__ src, const float* __restrict__ w, int pos, float eps, float rope_base, float* sm,
                                               float* sy, float* dst /*[128] or null*/, __half* dsth /*[128] or null*/) {
    const int d = threadIdx.x;
    const float v = src[d];
    const float ss = block_sum_f(v * v, sm);
    const float y = v * rsqrtf(ss / 128.0f + eps) * w[d];
    sy[d] = y;
    __syncthreads();
    float r;
    if (d < 64) {
        const float theta = powf(rope_base, -float(d) / 64.0f);
        float sn, cs;
        sincosf(float(pos) * theta, &sn, &cs);
        r = sy[d] * cs - sy[d + 64] * sn;
    } else {
        const float theta = powf(rope_base, -float(d - 64) / 64.0f);
        float sn, cs;
        sincosf(float(pos) * theta, &sn, &cs);
        r = sy[d] * cs + sy[d - 64] * sn;
    }
    if (dst) dst[d] = r;
    if (dsth) dsth[d] = __float2half(r);
}

// q (32 heads) and k (8 heads) of the block's rows: norm + rope at pos + row. in: q [8][4096], k [8][1024].
__global__ void df_qkprep_k(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ qn, const float* __restrict__ kn,
                            float* __restrict__ qr, float* __restrict__ kr, const DArgs* a, int n_head, int n_kv, float eps, float rope_base) {
    __shared__ float sm[4], sy[128];
    const int t = blockIdx.x, h = blockIdx.y, pos = a->pos;
    if (h < n_head) head_norm_rope(q + (size_t(t) * n_head + h) * 128, qn, pos + t, eps, rope_base, sm, sy, qr + (size_t(t) * n_head + h) * 128, nullptr);
    else head_norm_rope(k + (size_t(t) * n_kv + (h - n_head)) * 128, kn, pos + t, eps, rope_base, sm, sy, kr + (size_t(t) * n_kv + (h - n_head)) * 128, nullptr);
}

// K (norm + rope at pos0 + row) and V of rows 0..n-1 into the ring: ring is [which][kv head][C][128] of one layer. Block (row, 0..15).
__global__ void df_store_kv_k(const float* __restrict__ kb, const float* __restrict__ vb, const float* __restrict__ kn, __half* __restrict__ ring, int C, int pos0,
                              int n_kv, float eps, float rope_base) {
    __shared__ float sm[4], sy[128];
    const int t = blockIdx.x, h = blockIdx.y;
    const int p = pos0 + t, r = p % C;
    if (h < n_kv) {
        head_norm_rope(kb + (size_t(t) * n_kv + h) * 128, kn, p, eps, rope_base, sm, sy, nullptr, ring + (size_t(h) * C + r) * 128);
    } else {
        const int hv = h - n_kv;
        ring[(size_t(n_kv + hv) * C + r) * 128 + threadIdx.x] = __float2half(vb[(size_t(t) * n_kv + hv) * 128 + threadIdx.x]);
    }
}

// Attention of the block's 8 rows (32 heads, 8 KV heads) over the context in the ring and the block's own 8 keys (non-causal inside
// the block; a context key p is seen by row i when p > pos + i - window). One block = (KV head, key slice); one warp = one head of the
// group of 4, 8 rows. Each warp writes (max, sum, numerator[128]) per row and slice; df_attn_combine_k merges the slices.
__global__ void __launch_bounds__(128) df_attn_k(const float* __restrict__ qr, const float* __restrict__ kr, const float* __restrict__ vb,
                                                 const DArgs* __restrict__ args, int layer_stride, int C, int n_head, int n_kv, int window, float scale,
                                                 float* __restrict__ part) {
    const int kvh = blockIdx.x, split = blockIdx.y;
    const int pos = args->pos, lo_eff = args->lo_eff, nctx = args->nctx;
    const __half* ring = reinterpret_cast<const __half*>(args->ring) + size_t(layer_stride);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int head = kvh * 4 + warp;
    float q[kRows][4], m[kRows], l[kRows], acc[kRows][4];
#pragma unroll
    for (int i = 0; i < kRows; ++i) {
        const float4 v = *reinterpret_cast<const float4*>(qr + (size_t(i) * n_head + head) * 128 + lane * 4);
        q[i][0] = v.x * scale;
        q[i][1] = v.y * scale;
        q[i][2] = v.z * scale;
        q[i][3] = v.w * scale;
        m[i] = -FLT_MAX;
        l[i] = 0.0f;
        acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.0f;
    }
    const int total = nctx + kRows;
    const int per = (total + kSplits - 1) / kSplits;
    const int u0 = split * per, u1 = min(total, u0 + per);
    const __half* kring = ring + size_t(kvh) * C * 128;
    const __half* vring = ring + size_t(n_kv + kvh) * C * 128;
    for (int u = u0; u < u1; ++u) {
        float k4[4], v4[4];
        const bool is_ctx = u < nctx;
        const int p = lo_eff + u;
        if (is_ctx) {
            const int r = p % C;
            const __half2* kp = reinterpret_cast<const __half2*>(kring + size_t(r) * 128 + lane * 4);
            const __half2* vp = reinterpret_cast<const __half2*>(vring + size_t(r) * 128 + lane * 4);
            const float2 a = __half22float2(kp[0]), b = __half22float2(kp[1]), c = __half22float2(vp[0]), d = __half22float2(vp[1]);
            k4[0] = a.x; k4[1] = a.y; k4[2] = b.x; k4[3] = b.y;
            v4[0] = c.x; v4[1] = c.y; v4[2] = d.x; v4[3] = d.y;
        } else {
            const int j = u - nctx;
            const float4 a = *reinterpret_cast<const float4*>(kr + (size_t(j) * n_kv + kvh) * 128 + lane * 4);
            const float4 c = *reinterpret_cast<const float4*>(vb + (size_t(j) * n_kv + kvh) * 128 + lane * 4);
            k4[0] = a.x; k4[1] = a.y; k4[2] = a.z; k4[3] = a.w;
            v4[0] = c.x; v4[1] = c.y; v4[2] = c.z; v4[3] = c.w;
        }
#pragma unroll
        for (int i = 0; i < kRows; ++i) {
            if (is_ctx && p <= pos + i - window) continue;  // out of row i's window (uniform across the warp)
            float s = q[i][0] * k4[0] + q[i][1] * k4[1] + q[i][2] * k4[2] + q[i][3] * k4[3];
            s = warp_sum_f(s);
            const float mn = fmaxf(m[i], s);
            const float a = __expf(m[i] - mn), e = __expf(s - mn);
            l[i] = l[i] * a + e;
            m[i] = mn;
#pragma unroll
            for (int d = 0; d < 4; ++d) acc[i][d] = acc[i][d] * a + e * v4[d];
        }
    }
#pragma unroll
    for (int i = 0; i < kRows; ++i) {
        float* o = part + ((size_t(head) * kRows + i) * kSplits + split) * 132;
        if (lane == 0) {
            o[0] = m[i];
            o[1] = l[i];
        }
        *reinterpret_cast<float4*>(o + 4 + lane * 4) = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
    }
}

__global__ void df_attn_combine_k(const float* __restrict__ part, float* __restrict__ out, int n_head) {
    const int head = blockIdx.x, i = blockIdx.y, d = threadIdx.x;
    const float* base = part + (size_t(head) * kRows + i) * kSplits * 132;
    float M = -FLT_MAX;
    for (int s = 0; s < kSplits; ++s)
        if (base[s * 132 + 1] > 0.0f) M = fmaxf(M, base[s * 132]);
    float L = 0.0f, A = 0.0f;
    for (int s = 0; s < kSplits; ++s) {
        const float ls = base[s * 132 + 1];
        if (ls <= 0.0f) continue;
        const float w = __expf(base[s * 132] - M);
        L += ls * w;
        A += base[s * 132 + 4 + d] * w;
    }
    out[(size_t(i) * n_head + head) * 128 + d] = L > 0.0f ? A / L : 0.0f;
}

// top-16 of 7 rows of logits, first pass: block (slice, row) keeps its slice's best 16 by 16 rounds of block argmax
__global__ void df_top_part_k(const float* __restrict__ logits, int nv, int chunk, int top_k, float* __restrict__ pv, int32_t* __restrict__ pi) {
    extern __shared__ float sv_all[];
    float* v = sv_all;
    __shared__ float rv[32];
    __shared__ int ri[32];
    const int part = blockIdx.x, row = blockIdx.y;
    const int lo = part * chunk, hi = min(nv, lo + chunk);
    const float* src = logits + size_t(row) * nv;
    for (int j = threadIdx.x; j < hi - lo; j += blockDim.x) v[j] = src[lo + j];
    __syncthreads();
    for (int r = 0; r < top_k; ++r) {
        float bv = -FLT_MAX;
        int bi = INT_MAX;
        for (int j = threadIdx.x; j < hi - lo; j += blockDim.x)
            if (better_vi(v[j], lo + j, bv, bi)) {
                bv = v[j];
                bi = lo + j;
            }
        block_argmax_vi(bv, bi, rv, ri);
        if (threadIdx.x == 0) {
            const size_t o = (size_t(row) * kTopParts + part) * top_k + r;
            pv[o] = bv;
            pi[o] = bi == INT_MAX ? -1 : bi;
            if (bi != INT_MAX) v[bi - lo] = -FLT_MAX;
        }
        __syncthreads();
    }
}

// second pass: the best 16 of a row's 64 x 16 candidates, descending
__global__ void df_top_merge_k(const float* __restrict__ pv, const int32_t* __restrict__ pi, int top_k, float* __restrict__ tv, int32_t* __restrict__ ti) {
    __shared__ float rv[32];
    __shared__ int ri[32];
    __shared__ float cv[kTopParts * 16];
    __shared__ int ci[kTopParts * 16];
    const int row = blockIdx.x, n = kTopParts * top_k;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        cv[j] = pv[size_t(row) * n + j];
        ci[j] = pi[size_t(row) * n + j];
    }
    __syncthreads();
    for (int r = 0; r < top_k; ++r) {
        float bv = -FLT_MAX;
        int bi = INT_MAX;
        for (int j = threadIdx.x; j < n; j += blockDim.x)
            if (ci[j] >= 0 && better_vi(cv[j], ci[j], bv, bi)) {
                bv = cv[j];
                bi = ci[j];
            }
        block_argmax_vi(bv, bi, rv, ri);
        if (bi != INT_MAX) {
            for (int j = threadIdx.x; j < n; j += blockDim.x)
                if (ci[j] == bi) {
                    ci[j] = -1;
                    cv[j] = -FLT_MAX;
                }
        }
        if (threadIdx.x == 0) {
            tv[row * top_k + r] = bv;
            ti[row * top_k + r] = bi == INT_MAX ? 0 : bi;
        }
        __syncthreads();
    }
}

// The selector's pairwise scores of block position i = row + 1: scores[row][k][j] = unary[row][j] + succ[cand_i[j]] . (pred[prev_k] * gate_i)
// where prev_k is the anchor token (row 0: every k the same) or candidate k of position i - 1. 256 threads = (k, j).
__global__ void df_scores_k(const float* __restrict__ tv, const int32_t* __restrict__ ti, const float* __restrict__ gate, const __nv_bfloat16* __restrict__ pred_cb,
                            const __nv_bfloat16* __restrict__ succ_cb, const DArgs* __restrict__ args, int top_k, int rank, float* __restrict__ scores) {
    extern __shared__ float sh[];
    float* succ = sh;                       // [top_k][rank + 1]
    float* pred = succ + top_k * (rank + 1);  // [top_k][rank + 1]
    float* g = pred + top_k * (rank + 1);   // [rank]
    const int row = blockIdx.x;
    const int32_t anchor = args->anchor;
    const int words = rank / 2;
    for (int idx = threadIdx.x; idx < top_k * words; idx += blockDim.x) {
        const int j = idx / words, w = idx % words;
        const int32_t tok = ti[row * top_k + j];
        const __nv_bfloat162 s2 = *reinterpret_cast<const __nv_bfloat162*>(succ_cb + size_t(tok) * rank + 2 * w);
        succ[j * (rank + 1) + 2 * w] = __low2float(s2);
        succ[j * (rank + 1) + 2 * w + 1] = __high2float(s2);
        const int32_t ptok = row == 0 ? anchor : ti[(row - 1) * top_k + j];
        const __nv_bfloat162 p2 = *reinterpret_cast<const __nv_bfloat162*>(pred_cb + size_t(ptok) * rank + 2 * w);
        pred[j * (rank + 1) + 2 * w] = __low2float(p2);
        pred[j * (rank + 1) + 2 * w + 1] = __high2float(p2);
    }
    for (int d = threadIdx.x; d < rank; d += blockDim.x) g[d] = gate[size_t(row) * rank + d];
    __syncthreads();
    const int k = threadIdx.x / top_k, j = threadIdx.x % top_k;
    if (k >= top_k) return;
    const int kk = row == 0 ? 0 : k;  // position 1 has a single predecessor, the anchor
    float s = 0.0f;
    for (int d = 0; d < rank; ++d) s += succ[j * (rank + 1) + d] * (pred[kk * (rank + 1) + d] * g[d]);
    scores[(size_t(row) * top_k + k) * top_k + j] = s + tv[row * top_k + j];
}

// The greedy walk over the lattice: position 1 from predecessor 0, then each position from the candidate the last one chose.
__global__ void df_walk_k(const float* __restrict__ scores, const int32_t* __restrict__ ti, int top_k, int n_rows, int32_t* __restrict__ out_tok,
                          float* __restrict__ out_conf) {
    if (threadIdx.x != 0) return;
    int pred = 0;
    for (int row = 0; row < n_rows; ++row) {
        const float* sc = scores + (size_t(row) * top_k + pred) * top_k;
        int best = 0;
        for (int j = 1; j < top_k; ++j)
            if (sc[j] > sc[best]) best = j;
        float sum = 0.0f;
        for (int j = 0; j < top_k; ++j) sum += expf(sc[j] - sc[best]);
        out_tok[row] = ti[row * top_k + best];
        out_conf[row] = 1.0f / sum;
        pred = best;
    }
}

}  // namespace

// ============================================================================================================================
struct Dflash::Impl {
    DflashConfig c;
    struct W {   // a matrix on the card and its format (gemv_cols decodes every one it holds)
        const uint8_t* p = nullptr;
        uint32_t t = 0;
    };
    struct Layer {
        W q, k, v, o, gate, up, down, cap, cfp;
        const float *in_norm = nullptr, *post_norm = nullptr, *qn = nullptr, *kn = nullptr, *cab = nullptr, *cfb = nullptr;
    };
    std::vector<Layer> L;
    W fc, hproj;
    const float *hidden_norm = nullptr, *out_norm = nullptr;
    const __nv_bfloat16 *pred_cb = nullptr, *succ_cb = nullptr;
    void *pred_host = nullptr, *succ_host = nullptr, *arena = nullptr, *farena = nullptr;
    uint64_t wbytes = 0, sbytes = 0, stage_b = 0;
    uint32_t wtype = kBF16;  // the format a matrix is decoded to when it is not kept: BF16, or Q8_0 under kQ8
    bool native = false;     // kNative: the file's own formats kept
    int nd = 0;              // candidates come from the first nd ids (0: all)
    int vocab_cb = 0;
    // the target
    uint32_t embd_type = 0, head_type = 0;
    const void *embd = nullptr, *head = nullptr;
    int nv = 0;
    cudaStream_t s = nullptr;
    bool is_bound = false;
    // scratch
    void* scratch = nullptr;
    float *x = nullptr, *xn = nullptr, *xc = nullptr, *dyna = nullptr, *dynf = nullptr, *qb = nullptr, *kb = nullptr, *vb = nullptr, *qr = nullptr, *kr = nullptr,
          *attn = nullptr, *o = nullptr, *x1 = nullptr, *hn = nullptr, *g = nullptr, *u = nullptr, *h = nullptr, *dn = nullptr, *gate = nullptr, *part = nullptr,
          *logits = nullptr, *pv = nullptr, *tv = nullptr, *scores = nullptr, *conf = nullptr, *xin = nullptr, *g2 = nullptr, *conf_h = nullptr;
    int32_t *tok = nullptr, *pi = nullptr, *ti = nullptr, *res = nullptr, *res_h = nullptr;
    __nv_bfloat16 *stage0 = nullptr, *stage1 = nullptr;
    int stage1_rows = 0;
    struct Slot {
        __half* kv = nullptr;
        int lo = 0, n = 0;
    };
    std::vector<Slot> slots;
    std::vector<int> col;  // layer -> column block

    ~Impl() {
        if (prof_n > 0) {
            for (const auto& e : prof) std::fprintf(stderr, "dflash prof: %-14s %7.3f ms per draft\n", e.first.c_str(), e.second / double(prof_n));
        }
        if (s) cudaStreamSynchronize(s);
        if (arena) cudaFree(arena);
        if (farena) cudaFree(farena);
        if (scratch) cudaFree(scratch);
        if (stage0) cudaFree(stage0);
        if (stage1) cudaFree(stage1);
        if (pred_host) cudaFreeHost(pred_host);
        if (succ_host) cudaFreeHost(succ_host);
        if (res_h) cudaFreeHost(res_h);
        if (h_args) cudaFreeHost(h_args);
        if (d_args) cudaFree(d_args);
        if (gx) cudaGraphExecDestroy(gx);
        for (cudaEvent_t& e : args_ev)
            if (e) cudaEventDestroy(e);
        for (Slot& sl : slots)
            if (sl.kv) cudaFree(sl.kv);
    }
    uint64_t ring_elems() const { return uint64_t(c.n_layer) * 2 * c.n_kv * Dflash::kRing * c.hd; }
    __half* layer_ring(const Slot& sl, int l) const { return sl.kv + uint64_t(l) * 2 * c.n_kv * Dflash::kRing * c.hd; }
    bool forward(int slot, int32_t anchor, int pos);
    bool body();  // the kernels of one block pass
    static constexpr int kArgsRing = 8;
    DArgs* d_args = nullptr;
    DArgs* h_args = nullptr;  // pinned, kArgsRing entries
    cudaEvent_t args_ev[kArgsRing] = {};
    int args_i = 0;
    cudaGraphExec_t gx = nullptr;  // the captured block pass
    int warm = 0;
    bool use_graph = true;
    // LIMBO_DF_PROF=1: every draft is timed per phase with events (synchronising) and the sums are printed when the drafter goes
    std::vector<std::pair<std::string, double>> prof;
    bool prof_on = std::getenv("LIMBO_DF_PROF") != nullptr;
    long prof_n = 0;
};

Dflash::Dflash() : p_(new Impl) {}
Dflash::~Dflash() = default;
const DflashConfig& Dflash::cfg() const { return p_->c; }
uint64_t Dflash::weight_bytes() const { return p_->wbytes; }
uint64_t Dflash::scratch_bytes() const { return p_->sbytes; }
uint64_t Dflash::ring_bytes() const { return p_->ring_elems() * sizeof(__half); }
bool Dflash::bound() const { return p_->is_bound; }
void Dflash::set_vocab_limit(int n) {
    Impl& I = *p_;
    const int nd = n > 0 && n < I.nv ? std::max(n, 1024) : 0;
    if (nd == I.nd) return;
    if (I.s) cudaStreamSynchronize(I.s);
    if (I.gx) cudaGraphExecDestroy(I.gx);  // it bakes the vocabulary prefix in
    I.gx = nullptr;
    I.warm = 0;
    I.nd = nd;
}
bool Dflash::is_q8() const { return p_->wtype != kBF16; }
const char* Dflash::quant_name() const { return p_->native ? "native" : p_->wtype != kBF16 ? "q8" : "bf16"; }
int Dflash::vocab_limit() const { return p_->nd > 0 ? p_->nd : p_->nv; }
uint64_t Dflash::stage_bytes() const { return p_->stage_b; }

namespace {

bool read_file(const std::string& path, std::string* out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    *out = ss.str();
    return true;
}

float bf16_to_f32(uint16_t v) {
    uint32_t b = uint32_t(v) << 16;
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}

// float to bfloat16, round to nearest even on the 16 dropped bits.
uint16_t f32_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t r = u + 0x7fffu + ((u >> 16) & 1u);
    return uint16_t(r >> 16);
}

// ---- one weight, either backend ------------------------------------------------------------------------------------
// The safetensors branch reads BF16 matrices and uploads them as they are. A GGUF drafter is Q6_K, so the loader needs
// one place that says "these are the bytes, this is the type, this is the shape", and one place that turns any type into
// what the arena wants. `ne` is the LOGICAL shape -- (out, in) for a matrix, (n) for a vector -- for both backends:
// safetensors stores it that way already, a GGUF stores ne[0] contiguous, so it is reversed on the way in.
struct WT {
    const uint8_t* data = nullptr;
    uint64_t nbytes = 0;
    uint64_t n_elem = 0;
    uint32_t type = kBF16;
    int64_t ne[4] = {1, 1, 1, 1};
    bool shape_is(std::initializer_list<int64_t> s) const {
        int i = 0;
        for (int64_t v : s) { if (i > 3 || ne[i] != v) return false; ++i; }
        for (; i < 4; ++i) if (ne[i] != 1) return false;
        return true;
    }
};

}  // namespace

std::unique_ptr<Dflash> Dflash::load(const std::string& path, std::string* err, int quant) {
    const bool q8 = quant == kQ8;
    auto fail = [&](const std::string& m) -> std::unique_ptr<Dflash> {
        if (err) *err = "dflash: " + m;
        return nullptr;
    };
    // Two backends, one checkpoint. A safetensors DFlash2 is a directory (or a .safetensors path) with config.json and a
    // `dflash_config` section; the GGUF conversion carries the SAME configuration in `dflash.*` metadata and the target's
    // mask id under the tokenizer's own key, so nothing has to be guessed. `gsel` picks the tensor names below.
    const bool is_gguf = path.size() > 5 && path.compare(path.size() - 5, 5, ".gguf") == 0;
    auto d = std::unique_ptr<Dflash>(new Dflash);
    Impl& I = *d->p_;
    I.wtype = q8 ? static_cast<uint32_t>(Type::Q8_0) : kBF16;
    I.native = quant == kNative;
    DflashConfig& c = I.c;
    std::string dir = path, file, text, jerr;   // `dir` stays `path` for the directory form, which is the common one
    std::unique_ptr<gguf::File> gf;
    std::unique_ptr<st::File> sf;
    jinja::Value cfg;   // the safetensors branch fills it; the GGUF branch reads the same fields from metadata
    if (is_gguf) {
        gf = gguf::File::open(path, &jerr);
        if (!gf) return fail(jerr);
        auto gi = [&](const char* key, int fb) { auto v = gf->get_int(key); return v ? int(*v) : fb; };
        auto gfl = [&](const char* key, double fb) { auto v = gf->get_float(key); return v ? *v : fb; };
        c.n_layer = gi("dflash.block_count", 0);
        c.hidden = gi("dflash.embedding_length", 0);
        c.n_head = gi("dflash.attention.head_count", 0);
        c.n_kv = gi("dflash.attention.head_count_kv", 0);
        c.hd = gi("dflash.attention.key_length", 0);
        c.ffn = gi("dflash.feed_forward_length", 0);
        c.window = gi("dflash.attention.sliding_window", 2048);
        c.eps = float(gfl("dflash.attention.layer_norm_rms_epsilon", 1e-6));
        c.rope_base = gfl("dflash.rope.freq_base", 1e7);
        c.block = gi("dflash.block_size", 0);
        c.conv_k = gi("dflash.conv_kernel_size", 0);
        c.conv_group = gi("dflash.conv_group_size", 0);
        c.rank = gi("dflash.selector_rank", 0);
        c.top_k = gi("dflash.selector_top_k", 0);
        c.mask_token = gi("tokenizer.ggml.mask_token_id", -1);
        if (const gguf::Value* tl = gf->find("dflash.target_layers"))
            if (tl->elem == gguf::VType::I32 && tl->raw)
                for (uint64_t k = 0; k < tl->count; ++k) c.target_layers.push_back(reinterpret_cast<const int32_t*>(tl->raw)[k]);
        // The conversion does not write `dflash.num_target_layers` (it is the TARGET's block count, not this file's), so it
        // comes off the ids the file does carry. `col` is only ever indexed by the target's own layer ids plus one, so a
        // bound that covers the highest of them is all this needs.
        for (int t : c.target_layers) c.n_target = std::max(c.n_target, t + 1);
        if (!c.target_layers.empty()) c.n_target = std::max(c.n_target, c.target_layers.back() + 2);
    } else {
        if (path.size() > 12 && path.compare(path.size() - 12, 12, ".safetensors") == 0) {
            file = path;
            const size_t sl = path.rfind('/');
            dir = sl == std::string::npos ? "." : path.substr(0, sl);
        } else {
            file = path + "/model.safetensors";
        }
        if (!read_file(dir + "/config.json", &text)) return fail("no config.json beside " + file + " (a GGUF drafter is passed as its own path)");
        if (!jinja::json_parse(text, &cfg, &jerr)) return fail("config.json: " + jerr);
        const jinja::Value* df = cfg.find("dflash_config");
        if (!df) return fail("config.json has no dflash_config (not a DFlash2 checkpoint)");
        c.n_layer = int(cfg.get_int("num_hidden_layers", 0));
        c.hidden = int(cfg.get_int("hidden_size", 0));
        c.n_head = int(cfg.get_int("num_attention_heads", 0));
        c.n_kv = int(cfg.get_int("num_key_value_heads", 0));
        c.hd = int(cfg.get_int("head_dim", 0));
        c.ffn = int(cfg.get_int("intermediate_size", 0));
        c.window = int(cfg.get_int("sliding_window", 2048));
        c.n_target = int(cfg.get_int("num_target_layers", 0));
        if (const jinja::Value* e = cfg.find("rms_norm_eps")) c.eps = float(e->as_double());
        if (const jinja::Value* rp = cfg.find("rope_parameters"))
            if (const jinja::Value* th = rp->find("rope_theta")) c.rope_base = th->as_double();
        c.block = int(df->get_int("block_size", 0));
        c.mask_token = int(df->get_int("mask_token_id", -1));
        c.conv_k = int(df->get_int("conv_kernel_size", 0));
        c.conv_group = int(df->get_int("conv_group_size", 0));
        c.rank = int(df->get_int("selector_rank", 0));
        c.top_k = int(df->get_int("selector_top_k", 0));
        if (const jinja::Value* tl = df->find("target_layer_ids"))
            if (tl->is_list())
                for (const jinja::Value& v : tl->list()) c.target_layers.push_back(int(v.i()));
    }
    if (c.n_layer < 1 || c.n_layer > 16 || c.hd != 128 || c.n_kv < 1 || c.n_head != 4 * c.n_kv || c.block != 8 || c.conv_k != 2 || c.conv_group != 16 ||
        c.top_k != 16 || c.rank < 2 || c.rank % 2 != 0 || c.rank > 512 || c.hidden % 256 != 0 || c.ffn % 256 != 0 || (c.n_head * c.hd) % 256 != 0 ||
        c.target_layers.empty() || c.target_layers.size() > 16)
        return fail("config.json: a shape this engine does not run (head_dim 128, 4 query heads per KV head, block 8, conv kernel 2 in groups of 16, top-16 selector)");
    if (c.mask_token < 0 || c.n_target <= 0) return fail("config.json: mask_token_id or num_target_layers missing");
    for (int t : c.target_layers)
        if (t < 0 || t >= c.n_target) return fail("config.json: target layer id out of range");
    if (!is_gguf)
    if (const jinja::Value* lt = cfg.find("layer_types"))
        if (lt->is_list())
            for (const jinja::Value& v : lt->list())
                if (!v.is_string() || v.s() != "sliding_attention") return fail("config.json: a layer that is not sliding_attention");
    // the checkpoint's ids count layer OUTPUTS (hidden_states[id + 1] of the reference): layer id's output is the input of layer id + 1,
    // which is what the pass can capture (the llama.cpp converter writes id + 1 for the same reason)
    I.col.assign(size_t(c.n_target) + 1, -1);
    for (size_t k = 0; k < c.target_layers.size(); ++k) I.col[size_t(c.target_layers[k]) + 1] = int(k);

    std::string serr;
    if (!is_gguf) {
        sf = st::File::open(file, &serr);
        if (!sf) return fail(serr);
    }
    // One lookup over two name worlds. The GGUF conversion of a DFlash2 checkpoint renames every tensor to this engine's
    // own `blk.<N>.<role>` scheme, so each role is asked for under both names and only one of them exists. The returned
    // view is stable (a deque), and `ne` is normalised to the LOGICAL shape so the checks below are the same either way.
    std::deque<WT> keep;
    auto look = [&](const std::string& st_name, const std::string& gg_name) -> const WT* {
        WT w;
        if (is_gguf) {
            const gguf::Tensor* t = gf->tensor(gg_name);
            if (!t) { serr = "tensor " + gg_name + " missing"; return nullptr; }
            w.data = t->data;
            w.nbytes = t->nbytes;
            w.n_elem = uint64_t(t->n_elements());
            w.type = t->type;
            for (uint32_t i = 0; i < t->n_dims && i < 4; ++i) w.ne[i] = t->ne[t->n_dims - 1 - i];
        } else {
            const st::Tensor* t = sf->find(st_name);
            if (!t) t = sf->find("model." + st_name);
            if (!t) { serr = "tensor " + st_name + " missing"; return nullptr; }
            if (t->dtype != "BF16") { serr = "tensor " + st_name + " is " + t->dtype + ", not BF16"; return nullptr; }
            w.data = t->data;
            w.nbytes = t->nbytes;
            w.n_elem = uint64_t(t->n_elements());
            w.type = kBF16;
            for (size_t i = 0; i < t->shape.size() && i < 4; ++i) w.ne[i] = t->shape[i];
        }
        keep.push_back(w);
        return &keep.back();
    };
    // The two are the same call with the shape folded in; a wrong shape is a wrong name mapping, and it has to be loud.
    auto pick = [&](const std::string& st_name, const std::string& gg_name, std::initializer_list<int64_t> sh) -> const WT* {
        const WT* w = look(st_name, gg_name);
        if (w && !w->shape_is(sh)) { serr = "tensor " + (is_gguf ? gg_name : st_name) + " has an unexpected shape"; return nullptr; }
        return w;
    };
    const int H = c.hidden, QW = c.n_head * c.hd, KW = c.n_kv * c.hd, NG = H / c.conv_group;
    struct M {
        const WT* t;
        Impl::W* dst;
    };
    struct F {
        const WT* t;
        const float** dst;
    };
    std::vector<M> mats;
    std::vector<F> fs;
    I.L.resize(size_t(c.n_layer));
    bool ok = true;
    auto m = [&](const WT* t, Impl::W* dst) {
        if (!t) ok = false;
        else mats.push_back({t, dst});
    };
    auto f = [&](const WT* t, const float** dst) {
        if (!t) ok = false;
        else fs.push_back({t, dst});
    };
    for (int l = 0; l < c.n_layer && ok; ++l) {
        Impl::Layer& y = I.L[size_t(l)];
        const std::string p = "layers." + std::to_string(l) + ".";
        const std::string g = "blk." + std::to_string(l) + ".";
        auto L = [&](const char* a, const char* b, std::initializer_list<int64_t> sh) { return pick(p + a, g + b, sh); };
        m(L("self_attn.q_proj.weight", "attn_q.weight", {QW, H}), &y.q);
        m(L("self_attn.k_proj.weight", "attn_k.weight", {KW, H}), &y.k);
        m(L("self_attn.v_proj.weight", "attn_v.weight", {KW, H}), &y.v);
        m(L("self_attn.o_proj.weight", "attn_output.weight", {H, QW}), &y.o);
        m(L("mlp.gate_proj.weight", "ffn_gate.weight", {c.ffn, H}), &y.gate);
        m(L("mlp.up_proj.weight", "ffn_up.weight", {c.ffn, H}), &y.up);
        m(L("mlp.down_proj.weight", "ffn_down.weight", {H, c.ffn}), &y.down);
        m(L("attention_conv.kernel_projection.weight", "attn_conv_proj.weight", {2 * c.conv_k * NG, H}), &y.cap);
        m(L("mlp_conv.kernel_projection.weight", "ffn_conv_proj.weight", {2 * c.conv_k * NG, H}), &y.cfp);
        f(L("input_layernorm.weight", "attn_norm.weight", {H}), &y.in_norm);
        f(L("post_attention_layernorm.weight", "ffn_norm.weight", {H}), &y.post_norm);
        f(L("self_attn.q_norm.weight", "attn_q_norm.weight", {c.hd}), &y.qn);
        f(L("self_attn.k_norm.weight", "attn_k_norm.weight", {c.hd}), &y.kn);
        f(L("attention_conv.base_kernel", "attn_conv_base", {2, c.conv_k, H}), &y.cab);
        f(L("mlp_conv.base_kernel", "ffn_conv_base", {2, c.conv_k, H}), &y.cfb);
    }
    if (ok) {
        m(pick("fc.weight", "fc.weight", {H, int64_t(c.target_layers.size()) * H}), &I.fc);
        m(pick("candidate_selector.hidden_projection.weight", "selector_hidden.weight", {c.rank, H}), &I.hproj);
        // `hidden_norm` normalises the target's stacked hidden states going in; `norm` normalises the draft coming out. The
        // GGUF names them enc.output_norm and output_norm. Both are [H], so a swap would not fail a shape check -- this is
        // the one pairing in the mapping that has to be right by name and cannot be proved by shape.
        f(pick("hidden_norm.weight", "enc.output_norm.weight", {H}), &I.hidden_norm);
        f(pick("norm.weight", "output_norm.weight", {H}), &I.out_norm);
    }
    const WT* pc = ok ? look("candidate_selector.predecessor_codebook", "selector_predecessor.weight") : nullptr;
    const WT* sc = ok ? look("candidate_selector.successor_codebook", "selector_successor.weight") : nullptr;
    if (!ok) return fail(serr);
    if (!pc) pc = look("model.candidate_selector.predecessor_codebook", "model.selector_predecessor.weight");
    if (!sc) sc = look("model.candidate_selector.successor_codebook", "model.selector_successor.weight");
    if (!pc || !sc || pc->ne[0] != sc->ne[0] || pc->ne[1] != c.rank || sc->ne[1] != c.rank)
        return fail("the selector's codebooks are missing or are not [vocab][rank]");
    I.vocab_cb = int(pc->ne[0]);
    if (c.mask_token >= I.vocab_cb) return fail("mask token outside the vocabulary");

    // device: the BF16 matrices in one arena (256-byte aligned), the small tensors as fp32 in another
    const uint32_t kF32 = static_cast<uint32_t>(Type::F32);
    // A weight the decoder cannot read cannot be turned into what the pass wants, so say so before the cudaMalloc.
    for (const M& e : mats)
        if (e.t->type != kBF16 && e.t->type != kF32 && !quants::supported(e.t->type))
            return fail("a drafter weight is type " + std::to_string(e.t->type) + ", which the decoder does not read");
    for (const F& e : fs)
        if (e.t->type != kBF16 && e.t->type != kF32 && !quants::supported(e.t->type))
            return fail("a drafter norm is of a type the decoder does not read");
    uint64_t total = 0;
    std::vector<uint64_t> off;
    // A quantised drafter is decoded once at load and stored as bf16 (the pass's own dtype) or, under --draft-quant q8, as
    // Q8_0 rows the same way a bf16 checkpoint becomes Q8_0 here. Either way the pass below reads one layout.
    // kNative keeps every matrix whose format the pass decodes as it is in the file
    auto kept_as_is = [&](const WT* t) { return I.native && t->type != kF32 && gemv_supported(t->type); };
    auto dev_bytes = [&](const WT* t) {
        if (kept_as_is(t)) return t->nbytes;
        if (q8) return uint64_t(t->n_elem) / 32 * 34;
        return t->type == kBF16 ? t->nbytes : uint64_t(t->n_elem) * 2;
    };
    for (const M& e : mats) {
        off.push_back(total);
        total += (dev_bytes(e.t) + 255) / 256 * 256;
    }
    uint64_t ftotal = 0;
    std::vector<uint64_t> foff;
    for (const F& e : fs) {
        foff.push_back(ftotal);
        ftotal += (uint64_t(e.t->n_elem) * 4 + 255) / 256 * 256;
    }
    if (cudaMalloc(&I.arena, total) != cudaSuccess) {
        cudaGetLastError();
        I.arena = nullptr;
        return fail("out of device memory for the weights (" + std::to_string(total >> 20) + " MiB)");
    }
    if (cudaMalloc(&I.farena, ftotal) != cudaSuccess) {
        cudaGetLastError();
        I.farena = nullptr;
        return fail("out of device memory for the norms");
    }
    I.wbytes = total + ftotal;
    for (size_t i = 0; i < mats.size(); ++i) {
        uint8_t* dst = static_cast<uint8_t*>(I.arena) + off[i];
        const WT& w = *mats[i].t;
        bool copied = false;
        const bool kept = kept_as_is(&w);
        if (kept || (w.type == kBF16 && !q8)) {
            if (cudaMemcpy(dst, w.data, w.nbytes, cudaMemcpyHostToDevice) != cudaSuccess) return fail("weight upload failed");
            copied = true;
        }
        if (!copied) {
            std::vector<float> f32(size_t(w.n_elem));
            const uint16_t* s16 = reinterpret_cast<const uint16_t*>(w.data);
            if (w.type == kBF16)
                for (size_t j = 0; j < f32.size(); ++j) f32[j] = bf16_to_f32(s16[j]);
            else if (w.type == kF32)
                std::memcpy(f32.data(), w.data, f32.size() * 4);
            else if (!quants::dequant_row(w.type, w.data, f32.data(), w.n_elem))
                return fail("weight decode failed");
            if (q8) {
                std::vector<uint8_t> q(size_t(w.n_elem) / 32 * 34);
                quants::quantize_q8_0(f32.data(), q.data(), w.n_elem);
                if (cudaMemcpy(dst, q.data(), q.size(), cudaMemcpyHostToDevice) != cudaSuccess) return fail("weight upload failed");
            } else {
                std::vector<uint16_t> b(f32.size());
                for (size_t j = 0; j < f32.size(); ++j) b[j] = f32_to_bf16(f32[j]);
                if (cudaMemcpy(dst, b.data(), b.size() * 2, cudaMemcpyHostToDevice) != cudaSuccess) return fail("weight upload failed");
            }
        }
        *mats[i].dst = Impl::W{dst, kept ? w.type : (w.type == kBF16 && !q8 ? kBF16 : I.wtype)};
    }
    for (size_t i = 0; i < fs.size(); ++i) {
        const WT& w = *fs[i].t;
        std::vector<float> h(size_t(w.n_elem));
        const uint16_t* s16 = reinterpret_cast<const uint16_t*>(w.data);
        if (w.type == kBF16)
            for (size_t j = 0; j < h.size(); ++j) h[j] = bf16_to_f32(s16[j]);
        else if (w.type == kF32)
            std::memcpy(h.data(), w.data, h.size() * 4);
        else if (!quants::dequant_row(w.type, w.data, h.data(), w.n_elem))
            return fail("norm decode failed");
        float* dst = reinterpret_cast<float*>(static_cast<uint8_t*>(I.farena) + foff[i]);
        if (cudaMemcpy(dst, h.data(), h.size() * 4, cudaMemcpyHostToDevice) != cudaSuccess) return fail("norm upload failed");
        *fs[i].dst = dst;
    }
    // the codebooks: mapped host memory (a draft reads 32 rows of each)
    for (int w = 0; w < 2; ++w) {
        const WT& t = w == 0 ? *pc : *sc;
        void*& host = w == 0 ? I.pred_host : I.succ_host;
        const uint64_t nb = t.n_elem * 2;
        if (cudaHostAlloc(&host, nb, cudaHostAllocMapped) != cudaSuccess) {
            cudaGetLastError();
            host = nullptr;
            return fail("out of host memory for the codebooks");
        }
        if (t.type == kBF16) {
            std::memcpy(host, t.data, nb);
        } else {
            std::vector<float> f32(size_t(t.n_elem));
            if (t.type == kF32) std::memcpy(f32.data(), t.data, f32.size() * 4);
            else if (!quants::dequant_row(t.type, t.data, f32.data(), t.n_elem)) return fail("codebook decode failed");
            uint16_t* b = static_cast<uint16_t*>(host);
            for (size_t j = 0; j < f32.size(); ++j) b[j] = f32_to_bf16(f32[j]);
        }
        void* dp = nullptr;
        if (cudaHostGetDevicePointer(&dp, host, 0) != cudaSuccess) return fail("codebooks: no device alias of the host memory");
        (w == 0 ? I.pred_cb : I.succ_cb) = static_cast<const __nv_bfloat16*>(dp);
    }
    return d;
}

uint64_t Dflash::estimate(const std::string& path, int slots, int stage_rows, int quant) {
    const bool q8 = quant == kQ8;
    // A GGUF drafter: the geometry comes from `dflash.*`, so the two numbers this needs -- the target's hidden width and
    // how many of its layers the block sees -- are read from the metadata rather than from a tensor's shape, and the
    // weights are sized from what they will BE in the arena, not from what they are on disk (Q6_K is repacked).
    if (path.size() > 5 && path.compare(path.size() - 5, 5, ".gguf") == 0) {
        std::string err;
        auto gf = gguf::File::open(path, &err);
        if (!gf) return 0;
        uint64_t w = 0, nv = 0;
        for (const gguf::Tensor& t : gf->tensors()) {
            const uint64_t n = uint64_t(t.n_elements());
            if (t.name.find("codebook") != std::string::npos || t.name.find("selector_successor") != std::string::npos ||
                t.name.find("selector_predecessor") != std::string::npos) {
                // host memory, but it is still allocated: the codebook is [vocab][rank] bf16 once decoded
                nv = std::max(nv, t.n_dims == 2 ? uint64_t(t.ne[1]) : 0);
                continue;
            }
            const bool kept = quant == kNative && t.n_dims > 1 && t.type != static_cast<uint32_t>(Type::F32) && gemv_supported(t.type);
            w += (t.n_dims == 1 ? n * 4 : kept ? uint64_t(t.nbytes) : q8 ? n / 32 * 34 : n * 2) + 256;
        }
        auto hi = gf->get_int("dflash.embedding_length");
        auto ci = gf->get_int("dflash.block_count");
        if (!hi || !ci || nv == 0) return 0;
        const uint64_t hidden = uint64_t(*hi);
        const uint64_t ntl = uint64_t(*ci);
        const uint64_t fc = hidden * ntl;
        const uint64_t scratch = 4 * (uint64_t(kRows) * (fc + 40 * hidden) + 7 * nv + uint64_t(32) * kRows * kSplits * 132 + (4ull << 20)) + (2ull << 20);
        const uint64_t stage = uint64_t(kRows + std::max(0, stage_rows)) * fc * 2;
        const uint64_t ring = 6ull * 2 * 8 * kRing * 128 * 2;
        return w + scratch + stage + uint64_t(std::max(1, slots)) * ring;
    }
    std::string dir = path, file;
    if (path.size() > 12 && path.compare(path.size() - 12, 12, ".safetensors") == 0) {
        file = path;
        const size_t sl = path.rfind('/');
        dir = sl == std::string::npos ? "." : path.substr(0, sl);
    } else {
        file = path + "/model.safetensors";
    }
    std::string err;
    auto sf = st::File::open(file, &err);
    if (!sf) return 0;
    uint64_t w = 0, nv = 0, hidden = 0, ntl = 1;
    for (const st::Tensor& t : sf->tensors()) {
        if (t.name.find("_codebook") != std::string::npos) {
            nv = uint64_t(t.shape.empty() ? 0 : t.shape[0]);
            continue;  // host memory
        }
        w += (t.shape.size() == 1 ? t.nbytes * 2 : q8 ? t.nbytes / 64 * 34 : t.nbytes) + 256;
        if (t.name == "hidden_norm.weight" || t.name == "model.hidden_norm.weight") hidden = uint64_t(t.shape[0]);
        if (t.name == "fc.weight" || t.name == "model.fc.weight") ntl = uint64_t(t.shape.at(1)) / std::max<uint64_t>(1, uint64_t(t.shape.at(0)));
    }
    if (nv == 0 || hidden == 0) return 0;
    const uint64_t fc = hidden * ntl;
    const uint64_t scratch = 4 * (uint64_t(kRows) * (fc + 40 * hidden) + 7 * nv + uint64_t(32) * kRows * kSplits * 132 + (4ull << 20)) + (2ull << 20);
    const uint64_t stage = uint64_t(kRows + std::max(0, stage_rows)) * fc * 2;
    const uint64_t ring = 6ull * 2 * 8 * kRing * 128 * 2;
    return w + scratch + stage + uint64_t(std::max(1, slots)) * ring;
}

bool Dflash::bind(uint32_t embd_type, const void* embd, uint32_t head_type, const void* head, int n_vocab, int n_embd, cudaStream_t s) {
    Impl& I = *p_;
    if (I.is_bound) return true;
    if (n_embd != I.c.hidden || n_vocab != I.vocab_cb || n_vocab < 1024 || !embd || !head) return false;
    I.embd_type = embd_type;
    I.embd = embd;
    I.head_type = head_type;
    I.head = head;
    I.nv = n_vocab;
    I.s = s;
    const size_t R = kRows, H = size_t(I.c.hidden), QW = size_t(I.c.n_head) * I.c.hd, KW = size_t(I.c.n_kv) * I.c.hd, FF = size_t(I.c.ffn);
    const size_t NV = size_t(n_vocab), FC = H * I.c.target_layers.size(), DW = H / 4;  // dyn: 2 sides x 2 taps x hidden/16 groups
    struct Carve {
        float** dst;
        size_t n;
    };
    float *conf_d;
    const Carve carve[] = {
        {&I.x, R * H},     {&I.xn, R * H},    {&I.xc, R * H},   {&I.dyna, R * DW}, {&I.dynf, R * DW}, {&I.qb, R * QW},  {&I.kb, R * KW},  {&I.vb, R * KW},
        {&I.qr, R * QW},   {&I.kr, R * KW},   {&I.attn, R * QW}, {&I.o, R * H},     {&I.x1, R * H},     {&I.hn, R * H},   {&I.g, R * FF},   {&I.u, R * FF},
        {&I.h, R * FF},    {&I.dn, R * H},    {&I.gate, R * size_t(I.c.rank)}, {&I.part, size_t(I.c.n_head) * R * kSplits * 132},
        {&I.logits, 7 * NV}, {&I.pv, 7 * kTopParts * 16}, {&I.tv, R * 16}, {&I.scores, R * 16 * 16}, {&conf_d, 16}, {&I.xin, R * FC}, {&I.g2, R * H},
    };
    size_t tot = 0;
    for (const Carve& cv : carve) tot += (cv.n + 63) / 64 * 64;
    // int32 tail: tok[8], pi[7 * 64 * 16], ti[8 * 16], res[8]
    const size_t ints = 8 + 7 * kTopParts * 16 + R * 16 + 8 + 64;
    if (cudaMalloc(&I.scratch, tot * 4 + ints * 4) != cudaSuccess) {
        cudaGetLastError();
        I.scratch = nullptr;
        return false;
    }
    I.sbytes = tot * 4 + ints * 4;
    size_t at = 0;
    for (const Carve& cv : carve) {
        *cv.dst = static_cast<float*>(I.scratch) + at;
        at += (cv.n + 63) / 64 * 64;
    }
    I.conf = conf_d;
    int32_t* ip = reinterpret_cast<int32_t*>(static_cast<float*>(I.scratch) + tot);
    I.tok = ip;
    I.pi = ip + 8;
    I.ti = ip + 8 + 7 * kTopParts * 16;
    I.res = ip + 8 + 7 * kTopParts * 16 + R * 16;
    if (cudaMallocHost(reinterpret_cast<void**>(&I.res_h), 16 * sizeof(int32_t)) != cudaSuccess) return false;
    if (cudaMallocHost(reinterpret_cast<void**>(&I.h_args), sizeof(DArgs) * Impl::kArgsRing) != cudaSuccess) return false;
    if (cudaMalloc(reinterpret_cast<void**>(&I.d_args), sizeof(DArgs)) != cudaSuccess) {
        cudaGetLastError();
        I.d_args = nullptr;
        return false;
    }
    I.conf_h = reinterpret_cast<float*>(I.res_h + 8);
    // stage 0: 8 rows of fc's input
    if (cudaMalloc(reinterpret_cast<void**>(&I.stage0), R * FC * sizeof(__nv_bfloat16)) != cudaSuccess) {
        cudaGetLastError();
        I.stage0 = nullptr;
        return false;
    }
    I.stage_b = R * FC * sizeof(__nv_bfloat16);
    I.is_bound = true;
    return true;
}

int Dflash::column_of(int layer) const { return layer >= 0 && size_t(layer) < p_->col.size() ? p_->col[size_t(layer)] : -1; }

void Dflash::capture(int layer, const float* x, int T, int stage, cudaStream_t s) const {
    const Impl& I = *p_;
    const int k = column_of(layer);
    if (k < 0 || T < 1) return;
    __nv_bfloat16* dst = stage == 0 ? I.stage0 : I.stage1;
    if (!dst) return;
    df_capture_k<<<T, 256, 0, s>>>(x, I.c.hidden, dst, I.c.hidden * int(I.c.target_layers.size()), k * I.c.hidden);
}

bool Dflash::reserve_stage(int rows) {
    Impl& I = *p_;
    if (rows <= I.stage1_rows) return true;
    const size_t FC = size_t(I.c.hidden) * I.c.target_layers.size();
    __nv_bfloat16* nb = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&nb), size_t(rows) * FC * sizeof(__nv_bfloat16)) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    if (I.stage1) cudaFree(I.stage1);
    I.stage1 = nb;
    I.stage_b += uint64_t(rows - I.stage1_rows) * FC * sizeof(__nv_bfloat16);
    I.stage1_rows = rows;
    return true;
}

// ---- slots ------------------------------------------------------------------------------------------------------------------
bool Dflash::ensure_slot(int slot) {
    Impl& I = *p_;
    if (slot < 0) return false;
    if (size_t(slot) >= I.slots.size()) I.slots.resize(size_t(slot) + 1);
    Impl::Slot& sl = I.slots[size_t(slot)];
    if (sl.kv) return true;
    if (cudaMalloc(reinterpret_cast<void**>(&sl.kv), ring_bytes()) != cudaSuccess) {
        cudaGetLastError();
        sl.kv = nullptr;
        return false;
    }
    sl.lo = sl.n = 0;
    return true;
}

void Dflash::reset_slot(int slot) {
    if (slot >= 0 && size_t(slot) < p_->slots.size()) p_->slots[size_t(slot)].lo = p_->slots[size_t(slot)].n = 0;
}

void Dflash::cut_slot(int slot, int keep) {
    if (slot < 0 || size_t(slot) >= p_->slots.size()) return;
    Impl::Slot& sl = p_->slots[size_t(slot)];
    if (keep >= sl.n) return;
    // rows past `keep` were written over the ring's older positions: only [n - ring, keep) is intact
    sl.lo = std::max(sl.lo, sl.n - kRing);
    sl.n = std::max(0, keep);
    sl.lo = std::min(sl.lo, sl.n);
}

bool Dflash::share_slot(int from, int to, int n) {
    Impl& I = *p_;
    if (from < 0 || size_t(from) >= I.slots.size() || !I.slots[size_t(from)].kv || n <= 0) return false;
    if (from == to) return true;
    if (!ensure_slot(to)) return false;
    Impl::Slot &a = I.slots[size_t(from)], &b = I.slots[size_t(to)];
    if (a.n < n) return false;  // `from` has not read that many positions
    // a ring entry is position % kRing, so the same layout works in both: copy the lot
    if (cudaMemcpyAsync(b.kv, a.kv, ring_bytes(), cudaMemcpyDeviceToDevice, I.s) != cudaSuccess) return false;
    b.n = n;
    b.lo = std::min(std::max(a.lo, a.n - kRing), b.n);  // entries of positions in [n, a.n) overwrote older ones
    return true;
}

int Dflash::slot_len(int slot) const { return slot >= 0 && size_t(slot) < p_->slots.size() ? p_->slots[size_t(slot)].n : 0; }
int Dflash::slot_lo(int slot) const { return slot >= 0 && size_t(slot) < p_->slots.size() ? p_->slots[size_t(slot)].lo : 0; }

bool Dflash::free_slot(int slot) {
    Impl& I = *p_;
    if (slot < 0 || size_t(slot) >= I.slots.size()) return false;
    Impl::Slot& sl = I.slots[size_t(slot)];
    if (I.s) cudaStreamSynchronize(I.s);
    if (sl.kv) cudaFree(sl.kv);
    sl = Impl::Slot{};
    return true;
}

// ---- ingest -----------------------------------------------------------------------------------------------------------------
bool Dflash::ingest(int slot, int stage, int r0, int n, int pos0) {
    Impl& I = *p_;
    if (!I.is_bound || n <= 0 || pos0 < 0 || r0 < 0) return false;
    const __nv_bfloat16* sg = stage == 0 ? I.stage0 : I.stage1;
    if (!sg || (stage == 0 && r0 + n > kRows) || (stage == 1 && r0 + n > I.stage1_rows)) return false;
    if (!ensure_slot(slot)) return false;
    Impl::Slot& sl = I.slots[size_t(slot)];
    if (pos0 != sl.n) {
        if (pos0 > sl.n) {  // a gap: the context starts again here
            sl.lo = pos0;
        } else {  // a rewind: only [n - ring, pos0) is intact
            sl.lo = std::max(sl.lo, sl.n - kRing);
            sl.lo = std::min(sl.lo, pos0);
        }
        sl.n = pos0;
    }
    const DflashConfig& c = I.c;
    const int H = c.hidden, FC = H * int(c.target_layers.size());
    cudaStream_t s = I.s;
    bool ok = true;
    for (int a = 0; a < n; a += kRows) {
        const int T = std::min(kRows, n - a);
        df_unstage_k<<<T, 256, 0, s>>>(sg, r0 + a, FC, I.xin);
        ok &= gemv_cols(I.fc.t, I.fc.p, I.xin, FC, I.g2, H, H, FC, T, false, s);
        ok &= dec::rmsnorm(I.g2, I.hidden_norm, I.xn, H, c.eps, T, s);
        for (int l = 0; l < c.n_layer; ++l) {
            const Impl::Layer& y = I.L[size_t(l)];
            ok &= gemv_cols(y.k.t, y.k.p, I.xn, H, I.kb, c.n_kv * c.hd, c.n_kv * c.hd, H, T, false, s);
            ok &= gemv_cols(y.v.t, y.v.p, I.xn, H, I.vb, c.n_kv * c.hd, c.n_kv * c.hd, H, T, false, s);
            df_store_kv_k<<<dim3(unsigned(T), unsigned(2 * c.n_kv)), 128, 0, s>>>(I.kb, I.vb, y.kn, I.layer_ring(sl, l), kRing, pos0 + a, c.n_kv, c.eps,
                                                                                  float(c.rope_base));
        }
    }
    sl.n = pos0 + n;
    sl.lo = std::max(sl.lo, sl.n - kRing);
    return ok && cudaGetLastError() == cudaSuccess;
}

// ---- the block pass -----------------------------------------------------------------------------------------------------------
// Everything the pass reads that changes between drafts is in d_args (DArgs), so the same captured graph serves every slot and position.
bool Dflash::Impl::body() {
    const DflashConfig& cc = c;
    const int H = cc.hidden, QW = cc.n_head * cc.hd, KW = cc.n_kv * cc.hd, FF = cc.ffn, NG = H / cc.conv_group, DW = 4 * NG;
    const float scale = 1.0f / std::sqrt(float(cc.hd));
    bool ok = true;
    std::vector<std::pair<const char*, cudaEvent_t>> evs;
    auto mark = [&](const char* name) {
        if (!prof_on) return;
        cudaEvent_t e;
        cudaEventCreate(&e);
        cudaEventRecord(e, s);
        evs.emplace_back(name, e);
    };
    mark("start");
    df_tokens_k<<<1, kRows, 0, s>>>(tok, d_args, cc.mask_token);
    ok &= embed_rows(embd_type, embd, tok, kRows, x, H, s);
    mark("embed");
    for (int l = 0; l < cc.n_layer; ++l) {
        const Layer& y = L[size_t(l)];
        ok &= dec::rmsnorm(x, y.in_norm, xn, H, cc.eps, kRows, s);
        ok &= gemv_cols(y.cap.t, y.cap.p, xn, H, dyna, DW, DW, H, kRows, false, s);
        df_conv_k<<<kRows, 256, 0, s>>>(xn, dyna, y.cab, 0, xc, H, cc.conv_group, NG);
        ok &= gemv_cols(y.q.t, y.q.p, xc, H, qb, QW, QW, H, kRows, false, s);
        ok &= gemv_cols(y.k.t, y.k.p, xc, H, kb, KW, KW, H, kRows, false, s);
        ok &= gemv_cols(y.v.t, y.v.p, xc, H, vb, KW, KW, H, kRows, false, s);
        mark("qkv");
        df_qkprep_k<<<dim3(kRows, unsigned(cc.n_head + cc.n_kv)), 128, 0, s>>>(qb, kb, y.qn, y.kn, qr, kr, d_args, cc.n_head, cc.n_kv, cc.eps, float(cc.rope_base));
        df_attn_k<<<dim3(unsigned(cc.n_kv), kSplits), 128, 0, s>>>(qr, kr, vb, d_args, l * 2 * cc.n_kv * Dflash::kRing * cc.hd, Dflash::kRing, cc.n_head, cc.n_kv,
                                                                  cc.window, scale, part);
        df_attn_combine_k<<<dim3(unsigned(cc.n_head), kRows), 128, 0, s>>>(part, attn, cc.n_head);
        mark("attention");
        ok &= gemv_cols(y.o.t, y.o.p, attn, QW, o, H, H, QW, kRows, false, s);
        df_conv_k<<<kRows, 256, 0, s>>>(o, dyna, y.cab, 1, xc, H, cc.conv_group, NG);
        df_add_k<<<(kRows * H + 255) / 256, 256, 0, s>>>(xc, x, x1, kRows * H);
        ok &= dec::rmsnorm(x1, y.post_norm, hn, H, cc.eps, kRows, s);
        ok &= gemv_cols(y.cfp.t, y.cfp.p, hn, H, dynf, DW, DW, H, kRows, false, s);
        df_conv_k<<<kRows, 256, 0, s>>>(hn, dynf, y.cfb, 0, xc, H, cc.conv_group, NG);
        mark("o+conv");
        ok &= gemv_cols(y.gate.t, y.gate.p, xc, H, g, FF, FF, H, kRows, false, s);
        ok &= gemv_cols(y.up.t, y.up.p, xc, H, u, FF, FF, H, kRows, false, s);
        ops::swiglu(g, u, h, kRows * FF, s);
        ok &= gemv_cols(y.down.t, y.down.p, h, FF, dn, H, H, FF, kRows, false, s);
        df_conv_k<<<kRows, 256, 0, s>>>(dn, dynf, y.cfb, 1, xc, H, cc.conv_group, NG);
        df_add_k<<<(kRows * H + 255) / 256, 256, 0, s>>>(xc, x1, x, kRows * H);
        mark("ffn");
    }
    ok &= dec::rmsnorm(x, out_norm, hn, H, cc.eps, kRows, s);
    // the drafts are rows 1 .. 7: the target's head over them, the selector's gate from the same normed rows
    mark("norm");
    const int nvd = nd > 0 ? nd : nv;  // candidates come from the first nvd ids
    ok &= gemv_cols(head_type, head, hn + H, H, logits, nvd, nvd, H, kMaxDraftRows, false, s);
    mark("head");
    ok &= gemv_cols(hproj.t, hproj.p, hn + H, H, gate, cc.rank, cc.rank, H, kMaxDraftRows, false, s);
    const int chunk = (nvd + kTopParts - 1) / kTopParts;
    df_top_part_k<<<dim3(kTopParts, kMaxDraftRows), 256, size_t(chunk) * sizeof(float), s>>>(logits, nvd, chunk, cc.top_k, pv, pi);
    df_top_merge_k<<<kMaxDraftRows, 256, 0, s>>>(pv, pi, cc.top_k, tv, ti);
    const size_t sh = size_t(2 * cc.top_k * (cc.rank + 1) + cc.rank) * sizeof(float);
    df_scores_k<<<kMaxDraftRows, 256, sh, s>>>(tv, ti, gate, pred_cb, succ_cb, d_args, cc.top_k, cc.rank, scores);
    df_walk_k<<<1, 32, 0, s>>>(scores, ti, cc.top_k, kMaxDraftRows, res, conf);
    if (prof_on && !evs.empty()) {
        mark("select");
        cudaEventSynchronize(evs.back().second);
        for (size_t i = 1; i < evs.size(); ++i) {
            float ms = 0;
            cudaEventElapsedTime(&ms, evs[i - 1].second, evs[i].second);
            auto it = std::find_if(prof.begin(), prof.end(), [&](const auto& e) { return e.first == evs[i].first; });
            if (it == prof.end()) prof.emplace_back(evs[i].first, double(ms));
            else it->second += double(ms);
        }
        for (auto& e : evs) cudaEventDestroy(e.second);
        ++prof_n;
    }
    return ok && cudaGetLastError() == cudaSuccess;
}

bool Dflash::Impl::forward(int slot, int32_t anchor, int pos) {
    const Slot& sl = slots[size_t(slot)];
    const int ce = std::min(sl.n, pos);
    const int lo_eff = std::max(sl.lo, pos - c.window + 1);
    // the args go through a ring of pinned slots: one is reused only after the copy that read it has run
    const int i = args_i++ % kArgsRing;
    if (args_ev[i] && cudaEventSynchronize(args_ev[i]) != cudaSuccess) return false;
    h_args[i] = DArgs{anchor, pos, lo_eff, std::max(0, ce - lo_eff), reinterpret_cast<uint64_t>(sl.kv)};
    if (cudaMemcpyAsync(d_args, &h_args[i], sizeof(DArgs), cudaMemcpyHostToDevice, s) != cudaSuccess) return false;
    if (!args_ev[i] && cudaEventCreateWithFlags(&args_ev[i], cudaEventDisableTiming) != cudaSuccess) return false;
    if (cudaEventRecord(args_ev[i], s) != cudaSuccess) return false;
    if (!use_graph || prof_on) return body();
    if (!gx) {
        if (warm++ == 0) return body();  // the first pass runs eagerly so every kernel's module is loaded before a capture
        cudaGraph_t gr = nullptr;
        if (cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal) != cudaSuccess) return false;
        const bool ok = body();
        if (cudaStreamEndCapture(s, &gr) != cudaSuccess || !ok || !gr) {
            if (gr) cudaGraphDestroy(gr);
            cudaGetLastError();
            use_graph = false;
            return body();
        }
        const cudaError_t e = cudaGraphInstantiate(&gx, gr, 0);
        cudaGraphDestroy(gr);
        if (e != cudaSuccess) {
            gx = nullptr;
            cudaGetLastError();
            use_graph = false;
            return body();
        }
    }
    return cudaGraphLaunch(gx, s) == cudaSuccess;
}

bool Dflash::draft_start(int slot, int32_t anchor, int pos) {
    Impl& I = *p_;
    if (!I.is_bound || pos < 0 || !ensure_slot(slot)) return false;
    if (!I.forward(slot, anchor, pos)) return false;
    if (cudaMemcpyAsync(I.res_h, I.res, 8 * sizeof(int32_t), cudaMemcpyDeviceToHost, I.s) != cudaSuccess) return false;
    return cudaMemcpyAsync(I.conf_h, I.conf, 8 * sizeof(float), cudaMemcpyDeviceToHost, I.s) == cudaSuccess;
}

bool Dflash::draft_wait(int32_t* out, float* conf) {
    Impl& I = *p_;
    if (cudaStreamSynchronize(I.s) != cudaSuccess) return false;
    for (int i = 0; i < kMaxDraft; ++i) {
        out[i] = I.res_h[i];
        if (conf) conf[i] = I.conf_h[i];
    }
    return true;
}

bool Dflash::debug_hidden(float* out8) {
    Impl& I = *p_;
    cudaStreamSynchronize(I.s);
    return cudaMemcpy(out8, I.hn, size_t(kRows) * I.c.hidden * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
}

}  // namespace limbo::cuda
