// Copyright (c) 2026 ITERFORM. All rights reserved.
// src/cuda/ink_gpu.cu — the whole inkling2_loop forward pass on the device (docs/inkling2.md), the graph of
// Ref::forward_inkling / ink_visit (src/step.cpp) as kernel launches on one stream. The visit graph verbatim:
// the prelude (slots 0..loop_start), the loop_count passes over the core with the bank reset and the
// per-layer controller messages (mix(bank, bank_out, msg_query) per visit), the coda, then build_output.
//
// Weights are uploaded once from the GGUF mapping in their block layout; every product goes through
// limbo/cuda/gemv.hpp (the experts through src/cuda/ink_moe.cu, the bias table's per-head dots through
// gemv_experts with one shared matrix), the mechanisms through src/cuda/ink_ops.hpp and ink_flow.hpp. The
// small folds around them (the RMS norms, the sink fold into the bias table, the f16 cache stores, the
// softcap, the argmax) are kernels here; every reduction they make is a fixed-order double sum or a fixed
// shuffle tree, so a result never depends on the launch geometry.
//
// State is per visit slot (weights shared, state not): one f16 K/V pair [rows][n_head_kv * head_dim] -- rows
// n_ctx, or for a sliding slot a ring of its window -- and one short-conv history of (sconv-1) rows of the
// shared 2*n_head_kv*head_dim + 2*n_embd channels, exactly the layout Ref::ink_convolve documents. The host sees one token in and, if it asks, the logits or the argmax
// out; step() itself never reads the device.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ink_gpu.hpp"
#include "ink_moe.hpp"
#include "ink_ops.hpp"
#include "ink_flow.hpp"
#include "limbo/cuda/gemm.hpp"
#include "limbo/cuda/gemv.hpp"
#include "limbo/quants.hpp"
#include "limbo/types.hpp"

namespace limbo::cuda {

namespace {

// ---- the small folds ------------------------------------------------------------------------------------
__device__ __forceinline__ double warp_sum_f64(double v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// y[i] = x[i] * (1 / sqrt(sum(x^2)/n + eps)) * (w ? w[i] : 1) over one row of n (src/step.cpp rmsnorm: the
// sum is a fixed-order double reduction and the weight multiplies last).
__global__ void rmsnorm_kernel(const float* __restrict__ x, const float* __restrict__ w, float* y,
                               int n, float eps) {
    __shared__ double sh[8];
    __shared__ double sinv;
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    double ss = 0.0;
    for (int i = int(threadIdx.x); i < n; i += int(blockDim.x)) ss += double(x[i]) * double(x[i]);
    ss = warp_sum_f64(ss);
    if (lane == 0) sh[wid] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        double r = 0.0;
        for (int i = 0; i < int(blockDim.x) >> 5; ++i) r += sh[i];
        sinv = 1.0 / sqrt(r / double(n) + double(eps));
    }
    __syncthreads();
    const float inv = float(sinv);
    for (int i = int(threadIdx.x); i < n; i += int(blockDim.x)) y[i] = x[i] * inv * (w ? w[i] : 1.0f);
}

// The same per head of head_dim elements, no weight (src/step.cpp head_norm at scale 1). One block per head.
__global__ void headnorm_kernel(const float* x, float* y, int head_dim, float eps) {
    __shared__ double sh[4];
    __shared__ double sinv;
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    const float* xh = x + size_t(blockIdx.x) * head_dim;
    float* yh = y + size_t(blockIdx.x) * head_dim;
    double ss = 0.0;
    if (int(threadIdx.x) < head_dim) ss = double(xh[threadIdx.x]) * double(xh[threadIdx.x]);
    ss = warp_sum_f64(ss);
    if (lane == 0) sh[wid] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        double r = 0.0;
        for (int i = 0; i < int(blockDim.x) >> 5; ++i) r += sh[i];
        sinv = 1.0 / sqrt(r / double(head_dim) + double(eps));
    }
    __syncthreads();
    if (int(threadIdx.x) < head_dim) yh[threadIdx.x] = xh[threadIdx.x] * float(sinv);
}

// v[i] *= tau[pos]: the global layers' log-scaled position factor, read on the device so a captured pass
// serves every position (the table holds exactly the floats the host formula gives).
__global__ void mul_tau_kernel(float* v, const float* __restrict__ tau, const int* __restrict__ d_pos, int n) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) v[i] = v[i] * tau[d_pos[0]];
}

__global__ void mul_scalar_kernel(float* v, float s, int n) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) v[i] = v[i] * s;
}

// v[i] *= sigmoid(g[i]) (src/step.cpp's output gate: attnv[i] *= sigmoid(gate[i])).
__global__ void sigmoid_gate_kernel(float* __restrict__ v, const float* __restrict__ g, int n) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) v[i] = v[i] * (1.0f / (1.0f + expf(-g[i])));
}

__global__ void add_kernel(float* __restrict__ out, const float* __restrict__ a, const float* __restrict__ b, int n) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) out[i] = a[i] + b[i];
}

// The f16 K/V cache row of position pos (src/step.cpp's f32_to_f16: round to nearest even, as __float2half).
__global__ void store_kv_kernel(const float* __restrict__ k, const float* __restrict__ v, __half* __restrict__ kc,
                                __half* __restrict__ vc, const int* __restrict__ d_pos, int kvw, int cap) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    const size_t row = size_t(d_pos[0] % cap);   // a sliding slot is a ring; a global slot's cap is n_ctx
    if (i < kvw) {
        kc[row * kvw + i] = __float2half(k[i]);
        vc[row * kvw + i] = __float2half(v[i]);
    }
}

// The sink fold into the bias table: table[d][h] = rel_bias[d] . r_h - sink[h] for d < extent, the pad row
// zero (src/step.cpp's table build). The gemv wrote the products transposed ([h][extent]).
__global__ void bias_fold_kernel(const float* __restrict__ sc, const float* __restrict__ sink,
                                 float* __restrict__ table, int extent, int n_head) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= (extent + 1) * n_head) return;
    const int d = i / n_head, h = i % n_head;
    table[i] = d < extent ? sc[h * extent + d] - sink[h] : 0.0f;
}

// The decode step's bias table in one kernel: table[d][h] = rel_bias[d] . r_h - sink[h] (d < extent), the pad row
// zero -- with the exact arithmetic gemv_experts gave it for an F32 matrix of rel_len <= 16 columns: the row is the
// GEMV's partial super-block, lane L sums its 8 products in order, and the warp tree adds lane 0's and lane 1's
// sums (the others are zero). So the bits are the ones bias_fold_kernel wrote over gemv_experts' products.
__global__ void bias_table_kernel(const float* __restrict__ relb, int rel_len, const float* __restrict__ r,
                                  const float* __restrict__ sink, float* __restrict__ table, int extent, int n_head,
                                  size_t t_stride = 0) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    r += size_t(blockIdx.y) * n_head * rel_len;   // row blockIdx.y of the batched form
    sink += size_t(blockIdx.y) * n_head;
    table += size_t(blockIdx.y) * t_stride;
    if (i >= (extent + 1) * n_head) return;
    const int d = i / n_head, h = i % n_head;
    float v = 0.0f;
    if (d < extent) {
        const float* w = relb + size_t(d) * rel_len;
        const float* x = r + size_t(h) * rel_len;
        float a = 0.0f, b = 0.0f;
        for (int j = 0; j < 8 && j < rel_len; ++j) a += w[j] * x[j];
        for (int j = 8; j < rel_len; ++j) b += w[j] * x[j];
        v = (a + b) - sink[h];
    }
    table[i] = v;
}

// y[t][n] = W[n] . x[t] for an F32 matrix with n_in < 256 (the controller's gain/forget, n_in = the gate rank), with
// gemv_kernel's arithmetic for such a row -- it is all partial super-block: lane L sums its 8 products in order, the
// warp tree adds the lanes -- so every row's bits are gemv()'s. A warp per (output, row).
__global__ void gemv_tail_rows_kernel(const float* __restrict__ W, const float* __restrict__ x, float* __restrict__ y,
                                      int n_out, int n_in) {
    const int n = blockIdx.x * 8 + (threadIdx.x >> 5), t = blockIdx.y, lane = threadIdx.x & 31;
    if (n >= n_out) return;
    float tail = 0.f;
    if (lane * 8 < n_in) {
        const float* w = W + size_t(n) * n_in + lane * 8;
        const float* xv = x + size_t(t) * n_in + lane * 8;
#pragma unroll
        for (int i = 0; i < 8; ++i) tail += w[i] * xv[i];
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) tail += __shfl_xor_sync(0xffffffffu, tail, o);
    if (lane == 0) y[size_t(t) * n_out + n] = tail;
}

// logits[i] = cap * tanh(logits[i] * inv_cap) (src/step.cpp head's softcap fold).
__global__ void softcap_kernel(float* v, int n, float cap, float inv_cap) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) v[i] = cap * tanhf(v[i] * inv_cap);
}

// The conv history after the first `keep` rows of a recorded batched pass: the last sconv - 1 inputs up to row
// keep - 1, from the recorded rows or, before the pass, from the snapshot of the history it started from.
__global__ void conv_rollback_kernel(float* __restrict__ hist, const float* __restrict__ snap,
                                     const float* __restrict__ rec, int nch, int sconv, int rows_cap, int keep) {
    const int ch = int(blockIdx.x * blockDim.x + threadIdx.x), slot = blockIdx.y;
    if (ch >= nch) return;
    const size_t hs = size_t(slot) * (sconv - 1) * nch;
    for (int j = 0; j < sconv - 1; ++j) {
        const int r = (keep - 1) - (sconv - 2) + j;
        hist[hs + size_t(j) * nch + ch] = r < 0 ? snap[hs + size_t(sconv - 1 + r) * nch + ch]
                                                : rec[(size_t(slot) * rows_cap + r) * nch + ch];
    }
}

// The greedy token: the highest logit, the lowest id on a tie, one block, a fixed-order fold. blockIdx.x is the row
// of the batched form (v and out at blockIdx.x times n and 1).
__global__ void argmax_kernel(const float* __restrict__ v, int n, int32_t* __restrict__ out) {
    v += size_t(blockIdx.x) * n;
    out += blockIdx.x;
    __shared__ float sm[256];
    __shared__ int32_t si[256];
    float m = -INFINITY;
    int32_t idx = INT32_MAX;
    for (int i = int(threadIdx.x); i < n; i += int(blockDim.x))
        if (v[i] > m) {
            m = v[i];
            idx = i;
        }
    sm[threadIdx.x] = m;
    si[threadIdx.x] = idx;
    __syncthreads();
    if (threadIdx.x == 0) {
        float bm = sm[0];
        int32_t bi = si[0];
        for (int t = 1; t < int(blockDim.x); ++t)
            if (sm[t] > bm || (sm[t] == bm && si[t] < bi)) {
                bm = sm[t];
                bi = si[t];
            }
        *out = bi;
    }
}

// ---- the prompt pass's row kernels (prefill_rows): one block or one grid row per token row -------------
__global__ void rmsnorm_rows_kernel(const float* __restrict__ x, size_t xs, const float* __restrict__ w, float* y,
                                    size_t ys, int n, float eps) {
    __shared__ double sh[8];
    __shared__ double sinv;
    x += size_t(blockIdx.x) * xs;
    y += size_t(blockIdx.x) * ys;
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    double ss = 0.0;
    for (int i = int(threadIdx.x); i < n; i += int(blockDim.x)) ss += double(x[i]) * double(x[i]);
    ss = warp_sum_f64(ss);
    if (lane == 0) sh[wid] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        double r = 0.0;
        for (int i = 0; i < int(blockDim.x) >> 5; ++i) r += sh[i];
        sinv = 1.0 / sqrt(r / double(n) + double(eps));
    }
    __syncthreads();
    const float inv = float(sinv);
    for (int i = int(threadIdx.x); i < n; i += int(blockDim.x)) y[i] = x[i] * inv * (w ? w[i] : 1.0f);
}

// v[t][i] *= tau[pos[t]]
__global__ void mul_tau_rows_kernel(float* v, const float* __restrict__ tau, const int* __restrict__ d_pos, int n) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x), t = blockIdx.y;
    if (i < n) v[size_t(t) * n + i] *= tau[d_pos[t]];
}

// q8 K/V: block (KV head, row t, K or V), 128 threads = the head's dims; scale = amax / 127 of the head's values,
// q = round(v / scale). Row t's position d_pos[t], ring row pos % cap.
__global__ void store_kv_q8_kernel(const float* __restrict__ k, const float* __restrict__ v, int8_t* __restrict__ kq,
                                   int8_t* __restrict__ vq, float* __restrict__ ks, float* __restrict__ vs,
                                   const int* __restrict__ d_pos, int n_kv, int cap) {
    __shared__ float sh[4];
    const int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    const bool isv = blockIdx.z == 1;
    const float x = (isv ? v : k)[(size_t(t) * n_kv + h) * 128 + d];
    float a = fabsf(x);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    if ((d & 31) == 0) sh[d >> 5] = a;
    __syncthreads();
    const float amax = fmaxf(fmaxf(sh[0], sh[1]), fmaxf(sh[2], sh[3]));
    const float sc = amax / 127.0f;
    const size_t row = size_t(d_pos[t] % cap);
    const float qv = sc > 0.0f ? rintf(x / sc) : 0.0f;
    (isv ? vq : kq)[(row * n_kv + h) * 128 + d] = int8_t(fmaxf(-127.0f, fminf(127.0f, qv)));
    if (d == 0) (isv ? vs : ks)[row * n_kv + h] = sc;
}

// K/V rows t of a piece into the slot at their positions (ring row pos % cap).
__global__ void store_kv_rows_kernel(const float* __restrict__ k, const float* __restrict__ v, __half* __restrict__ kc,
                                     __half* __restrict__ vc, const int* __restrict__ d_pos, int kvw, int cap) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x), t = blockIdx.y;
    if (i >= kvw) return;
    const size_t row = size_t(d_pos[t] % cap);
    kc[row * kvw + i] = __float2half(k[size_t(t) * kvw + i]);
    vc[row * kvw + i] = __float2half(v[size_t(t) * kvw + i]);
}

// The bias table of every row: table[t][d][h] = relb[d] . r[t][h] - sink[t][h] for d < extent, the pad row zero.
// A block is 64 distances of one row, the row's r and sinks and the 64 relb rows in shared memory. Distances a row
// can never read (beyond its own position) are skipped: attention clamps to min(q_pos - pos, extent).
constexpr int kBiasD = 64, kRelMax = 32, kHeadMax = 32;
__global__ void __launch_bounds__(256) bias_rows_kernel(const float* __restrict__ relb, int rel_len,
                                                        const float* __restrict__ r, const float* __restrict__ sink,
                                                        float* __restrict__ tables, size_t t_stride, int extent,
                                                        int n_head, const int* __restrict__ d_pos) {
    __shared__ float sr[kHeadMax * kRelMax], ss[kHeadMax], sb[kBiasD * kRelMax];
    const int t = blockIdx.y, d0 = blockIdx.x * kBiasD, tid = threadIdx.x;
    const int q_pos = d_pos[t];
    if (d0 > q_pos && d0 <= extent - 1) return;   // never read by this row (the pad row at extent is kept)
    for (int i = tid; i < n_head * rel_len; i += 256) sr[i] = r[size_t(t) * n_head * rel_len + i];
    for (int i = tid; i < n_head; i += 256) ss[i] = sink[size_t(t) * n_head + i];
    for (int i = tid; i < kBiasD * rel_len; i += 256) {
        const int d = d0 + i / rel_len;
        sb[i] = d < extent ? relb[size_t(d0) * rel_len + i] : 0.f;
    }
    __syncthreads();
    for (int i = tid; i < kBiasD * n_head; i += 256) {
        const int dl = i / n_head, h = i % n_head, d = d0 + dl;
        if (d > extent) break;
        float v = 0.0f;
        if (d < extent) {
            float acc = 0.0f;
            for (int j = 0; j < rel_len; ++j) acc += sb[dl * rel_len + j] * sr[h * rel_len + j];
            v = acc - ss[h];
        }
        tables[size_t(t) * t_stride + size_t(d) * n_head + h] = v;
    }
}

// y[t][n] = W[n] . x[t] for a small F32 matrix with a short row (the controller's gain/forget: n_in = the gate
// rank, 64): a block is 64 outputs x 16 rows, W and x staged through shared memory in 64-column slabs, each thread
// 4 outputs of one row.
__global__ void __launch_bounds__(256) f32_rows_kernel(const float* __restrict__ W, const float* __restrict__ x,
                                                       float* __restrict__ y, int n_out, int n_in, int T) {
    __shared__ float sw[64][65], sx[16][65];
    const int n0 = blockIdx.x * 64, t0 = blockIdx.y * 16, tid = threadIdx.x;
    const int tr = tid >> 4, nc = tid & 15;   // row tr, outputs nc, nc + 16, nc + 32, nc + 48
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int k0 = 0; k0 < n_in; k0 += 64) {
        for (int i = tid; i < 64 * 64; i += 256) {
            const int r = i >> 6, c = i & 63, n = n0 + r, k = k0 + c;
            sw[r][c] = n < n_out && k < n_in ? W[size_t(n) * n_in + k] : 0.f;
        }
        for (int i = tid; i < 16 * 64; i += 256) {
            const int r = i >> 6, c = i & 63, t = t0 + r, k = k0 + c;
            sx[r][c] = t < T && k < n_in ? x[size_t(t) * n_in + k] : 0.f;
        }
        __syncthreads();
#pragma unroll 8
        for (int c = 0; c < 64; ++c) {
            const float xv = sx[tr][c];
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[j] += sw[nc + 16 * j][c] * xv;
        }
        __syncthreads();
    }
    const int t = t0 + tr;
    if (t < T)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            if (n0 + nc + 16 * j < n_out) y[size_t(t) * n_out + n0 + nc + 16 * j] = acc[j];
}

}  // namespace

// ---- the resolved model and the step's scratch ----------------------------------------------------------
struct InkGpu::St {
    // The geometry the graph runs on (docs/inkling2.md §1). InkGpu runs the file as it stands: uniform head
    // counts and one conv-history width across the layers; open() refuses anything else loudly.
    int n_layer = 0, n_embd = 0, n_head = 0, n_head_kv = 0, head_dim = 0, kvw = 0, nch = 0;
    int n_expert = 0, n_expert_used = 0, n_ff_exp = 0;
    int rel_length = 0, rel_extent = 0, window = 0, tbl_extent = 0, sconv = 0, res_block = 0;
    int log_floor = 0, gate_rank = 0, loop_start = 0, loop_end = 0, loop_count = 0;
    float qk_clip = 0, rel_clip = 0, log_alpha = 0, situ_beta = 0, situ_lin_beta = 0;
    float route_scale = 1.0f, rms_eps = 1e-6f, softcap = 0, attn_scale = 1.0f;
    int n_vocab = 0, n_ctx = 0, n_slot = 0;

    cudaStream_t stream = nullptr;
    std::vector<void*> allocs;   // everything device-side, freed by ~St
    int32_t* pinned = nullptr;   // [2]: the token and the position for the async H2D copies

    struct Proj {
        const uint8_t* w = nullptr;
        uint32_t type = 0;
        float scale = 1.0f;   // the optional scalar <name>.scale, applied after the product
        const float* f32 = nullptr;   // a small BF16/F16 matrix decoded to F32 too (the prompt pass's gemm_f32)
    };
    struct InkLayer {
        Proj wq, wk, wv, wo, wr, wgate, sink, rel_bias;
        Proj k_conv_proj, v_conv_proj;   // global layers: input-dependent conv taps
        Proj router, gate_up, down;
        Proj ctrl_down[4], ctrl_gain[4], ctrl_forget[4];
        float *q_clip = nullptr, *rel_clip = nullptr;
        float *attn_norm = nullptr;   // sliding layers
        float *q_inp_norm = nullptr, *k_inp_norm = nullptr, *v_inp_norm = nullptr;   // global layers
        float *k_conv = nullptr, *v_conv = nullptr, *attn_post_conv = nullptr, *ffn_post_conv = nullptr;
        float *res_q_attn = nullptr, *res_q_ffn = nullptr, *ffn_norm = nullptr;
        float *exp_bias = nullptr, *gu_scale = nullptr, *dn_scale = nullptr;
        float* msg_query = nullptr;   // loop layers: [4][n_embd]
        uint64_t gu_expert_bytes = 0, dn_expert_bytes = 0;
        float* relb32 = nullptr;   // attn_rel_bias decoded to F32 [rows][rel_length] (the prompt pass's tables)
        bool swa = false;
    };
    std::vector<InkLayer> L;
    Proj tok_embd, output;
    float *tok_norm = nullptr, *out_norm = nullptr, *out_res_query = nullptr;

    // Per-slot state: f16 K/V caches, the short-conv histories, and the visit outputs for the gate.
    // A global slot holds n_ctx rows; a sliding slot is a ring of window + kMaxRows - 1 rows (40 of the 48
    // slots: the ring is what lets the context exceed ~37k positions on a 16 GB card).
    __half *k_cache = nullptr, *v_cache = nullptr;   // f16 rows, or int8 rows (kv_q8) behind the same pointers
    float *k_scale = nullptr, *v_scale = nullptr;    // kv_q8: [kv_rows][n_head_kv]
    bool kv_q8 = false;
    std::vector<size_t> slot_off;   // [n_slot] first row of the slot in k_cache/v_cache
    std::vector<int> slot_cap;      // [n_slot] rows the slot holds
    size_t kv_rows = 0;             // sum of slot_cap
    float *conv_hist = nullptr, *l_out = nullptr;

    // The step's scratch, sized once.
    float *cur = nullptr, *inpSA = nullptr, *prelude = nullptr, *bank_out = nullptr;
    float *blocks = nullptr, *bank = nullptr;   // [max_blocks + 1][n_embd]: rows plus one mix-states row
    float *in0 = nullptr, *in1 = nullptr, *in2 = nullptr, *mixedq = nullptr, *mixed = nullptr, *msgs = nullptr;
    float *gate_in = nullptr, *lo = nullptr, *gvec = nullptr, *fvec = nullptr;
    float *q = nullptr, *gate = nullptr, *attnv = nullptr;
    float *kraw = nullptr, *vraw = nullptr, *k = nullptr, *v = nullptr, *kdyn = nullptr, *vdyn = nullptr;
    float *rv = nullptr, *sinkv = nullptr, *tblsc = nullptr, *table = nullptr;
    float *attn_out = nullptr, *attn_conv = nullptr, *ffn_inp = nullptr, *moe_out = nullptr, *ffn_conv = nullptr;
    float *moe_logits = nullptr, *moe_h = nullptr, *moe_raw = nullptr, *moe_w = nullptr, *rn = nullptr;
    float *logits = nullptr, *attn_scratch = nullptr;
    int32_t *ids = nullptr, *head_ids = nullptr, *d_token = nullptr, *d_pos = nullptr, *d_argmax = nullptr;

    // The row-batched pass's own buffers (step_rows): every per-token vector as [kRows][n], plus the tokens
    // and positions of the rows. The scratch used strictly one row at a time (the controller's, the table's,
    // the experts', the attention's) is the single-row step's.
    struct Rows {
        float *cur = nullptr, *inpSA = nullptr, *prelude = nullptr, *bank_out = nullptr;
        float *blocks = nullptr, *bank = nullptr, *in0 = nullptr, *in1 = nullptr, *in2 = nullptr;
        float *mixedq = nullptr, *mixed = nullptr, *msgs = nullptr, *q = nullptr, *gate = nullptr, *attnv = nullptr;
        float *kraw = nullptr, *vraw = nullptr, *k = nullptr, *v = nullptr, *kdyn = nullptr, *vdyn = nullptr;
        float *rv = nullptr, *sinkv = nullptr, *attn_out = nullptr, *attn_conv = nullptr, *ffn_inp = nullptr;
        float *moe_out = nullptr, *ffn_conv = nullptr;
        int32_t *d_tok = nullptr, *d_pos = nullptr, *pinned = nullptr;   // pinned: [kPiece tokens | kPiece positions]
        // the exact batched pass (step_rows, kRows rows): per-row router, experts and bias tables
        float *logits = nullptr, *moe_w = nullptr, *moe_h = nullptr, *moe_raw = nullptr, *tables = nullptr;
        float *gate_in = nullptr, *lo = nullptr, *gvec = nullptr, *fvec = nullptr;
        int32_t* ids = nullptr;
    } R;
    // The products of step_rows with gemv()'s bits per row: gemv_cols in pieces of kRows, the exact tail kernel for
    // F32 rows shorter than a super-block.
    bool mm_exact_rows(const Proj& p, const float* x, float* y, int n_out, int n_in, int T) {
        if (T > 1 && p.type == static_cast<uint32_t>(Type::F32) && n_in < 256 && n_in % 8 == 0) {
            gemv_tail_rows_kernel<<<dim3((n_out + 7) / 8, T), 256, 0, stream>>>(reinterpret_cast<const float*>(p.w), x,
                                                                               y, n_out, n_in);
            return cudaGetLastError() == cudaSuccess && (p.scale == 1.0f || mul(y, p.scale, n_out * T));
        }
        return gemvw_rows(p, x, y, n_out, n_in, T);
    }
    // control() over T rows with its bits: io and message rows at their strides.
    bool control_exact_rows(const InkLayer& W, float* io, size_t io_stride, const float* msg, size_t msg_stride, int ci,
                            int T) {
        return ink::control_gate_rows(io, io_stride, msg, msg_stride, R.gate_in, n_embd, T, stream) &&
               mm_exact_rows(W.ctrl_down[ci], R.gate_in, R.lo, gate_rank, 2 * n_embd, T) &&
               mm_exact_rows(W.ctrl_gain[ci], R.lo, R.gvec, n_embd, gate_rank, T) &&
               mm_exact_rows(W.ctrl_forget[ci], R.lo, R.fvec, n_embd, gate_rank, T) &&
               ink::control_combine_rows(io, io_stride, msg, msg_stride, R.gvec, R.fvec, n_embd, T, stream);
    }   // kPiece rows: step_rows uses the first kRows, prefill_rows all of them
    bool visit_rows(int il, int ip, int pos0, int T, bool msgs_, int& nb);
    bool step_rows(const int32_t* tokens, int T, int pos0, bool head, int head_mode = -1);
    bool rows_body(int T, int head);   // head: 0 none, 1 the last row, 2 every row (logits_rows, argmax_rows)

    // The draft model's view (InkGpu::set_capture): the features of every row a pass reads -- the inputs of three
    // visits and the final norm output -- and the verify window's rollback state.
    bool capture = false;
    int tap_il[3] = {-1, -1, -1};
    float* feats = nullptr;          // [kPiece][4][n_embd]
    float* logits_rows = nullptr;    // [kRows][n_vocab]
    int32_t* argmax_rows = nullptr;  // [kRows]
    float *conv_rec = nullptr, *hist_snap = nullptr;   // [n_slot][kRows][nch], [n_slot][sconv - 1][nch]
    int verify_pos = -1, verify_rows = 0;
    // slot il's caches (int8 rows behind the __half pointers when kv_q8) and scales (null for f16)
    __half* kc_of(int il) const {
        return reinterpret_cast<__half*>(reinterpret_cast<uint8_t*>(k_cache) +
                                         slot_off[size_t(il)] * size_t(kvw) * (kv_q8 ? 1 : 2));
    }
    __half* vc_of(int il) const {
        return reinterpret_cast<__half*>(reinterpret_cast<uint8_t*>(v_cache) +
                                         slot_off[size_t(il)] * size_t(kvw) * (kv_q8 ? 1 : 2));
    }
    float* ks_of(int il) const { return kv_q8 ? k_scale + slot_off[size_t(il)] * n_head_kv : nullptr; }
    float* vs_of(int il) const { return kv_q8 ? v_scale + slot_off[size_t(il)] * n_head_kv : nullptr; }
    // rows t < T of k / v ([T][kvw]) into slot il at positions pos[t]
    bool store_rows(int il, const float* k_, const float* v_, const int* pos, int T) {
        const int cap = slot_cap[size_t(il)];
        if (kv_q8)
            store_kv_q8_kernel<<<dim3(unsigned(n_head_kv), unsigned(T), 2), 128, 0, stream>>>(
                k_, v_, reinterpret_cast<int8_t*>(kc_of(il)), reinterpret_cast<int8_t*>(vc_of(il)), ks_of(il), vs_of(il),
                pos, n_head_kv, cap);
        else
            store_kv_rows_kernel<<<dim3(unsigned((kvw + 255) / 256), unsigned(T)), 256, 0, stream>>>(
                k_, v_, kc_of(il), vc_of(il), pos, kvw, cap);
        return cudaGetLastError() == cudaSuccess;
    }
    void set_capture(bool on, const int* visits);
    bool verify(const int32_t* tokens, int T, int pos0, int32_t* out, float* logits_out = nullptr);
    bool commit(int n);
    bool tap(int il, const float* src, int T) {
        if (!capture) return true;
        for (int k = 0; k < 3; ++k)
            if (il == tap_il[k])
                return cpy2(feats + size_t(k) * n_embd, 4 * size_t(n_embd), src, n_embd, n_embd, T);
        return true;
    }
    // record row t's conv inputs of slot il (the verify window's rollback): [width] floats at channel `off`
    bool rec_conv(int il, int off, const float* src, size_t src_stride, int width, int T) {
        return cpy2(conv_rec + size_t(il) * InkGpu::kRows * nch + off, nch, src, src_stride, size_t(width), T);
    }
    bool upload_rows(const int32_t* tokens, int T, int pos0);
    cudaGraphExec_t graph_rows[InkGpu::kRows][3] = {};

    // The prompt pass (prefill_rows): the products of a piece of up to kPiece tokens on the int8 tensor cores
    // (`planes` activation planes, limbo/cuda/gemm.hpp), the experts grouped (every expert's weights read once per
    // piece), the rest in row kernels. Not the decode pass's bits: the activation rounding of the planes.
    struct Piece {
        int8_t* xq[3] = {nullptr, nullptr, nullptr};   // three quantised inputs [planes][kPiece][3072]
        float* xs[3] = {nullptr, nullptr, nullptr};    // their scales [kPiece][3072 / 32]
        float* xsum[3] = {nullptr, nullptr, nullptr};  // their sums per 16 columns (the K-quants' minimum term)
        float *gate_in = nullptr, *lo = nullptr, *gvec = nullptr, *fvec = nullptr;
        float *logits = nullptr, *w = nullptr, *raw = nullptr, *h = nullptr, *hs = nullptr, *dn = nullptr;
        float* tables = nullptr;   // [kPiece][tbl_extent + 1][n_head]
        int8_t* hq = nullptr;
        int32_t *ids = nullptr, *pe = nullptr;
        MoeLists lists;
        int qk = 0;   // the widest quantised input (max of n_embd and n_head * head_dim)
    } P;
    int planes = 2;
    int piece_pos0 = 0;
    bool attn_rows = false;   // decode and verify through attention_rows (LIMBO_INK_ATTN=rows); both or neither   // the prompt piece's first position (host): bounds the attention's chunk grid
    // the decode step's bias table through bias_table_kernel (F32 rel_bias of at most 16 columns: its bits are
    // gemv_experts'); otherwise the GEMV and the fold
    bool exact_bias = false;
    bool visit_piece(int il, int ip, int T, bool msgs_, int& nb);
    bool prefill_rows(const int32_t* tokens, int T, int pos0, bool head);
    bool quant(const float* x, int rows, int k, int slot) {
        return act_quant_sums(x, rows, k, planes, P.xq[slot], P.xs[slot], P.xsum[slot], stream);
    }
    // y [T][n_out] = W x[t]: the int8 GEMM over quantised slot `slot` of x where the format has one, the F32
    // GEMM for F32 weights, the column GEMV in pieces of 8 otherwise (BF16).
    bool mm_rows(const Proj& p, const float* x, int slot, float* y, int n_out, int n_in, int T) {
        bool ok;
        const bool f32 = p.type == static_cast<uint32_t>(Type::F32) || p.f32;
        const float* wf = p.f32 ? p.f32 : reinterpret_cast<const float*>(p.w);
        if (!f32 && gemm_supported(p.type) && n_in % 128 == 0 && slot >= 0)
            ok = gemm(p.type, p.w, P.xq[slot], P.xs[slot], planes, T, y, n_out, n_in, false, stream, P.xsum[slot]);
        else if (f32 && n_in % 256 == 0)
            ok = gemm_f32(wf, x, y, T, n_out, n_in, stream);
        else if (f32) {
            f32_rows_kernel<<<dim3((n_out + 63) / 64, (T + 15) / 16), 256, 0, stream>>>(wf, x, y, n_out, n_in, T);
            ok = cudaGetLastError() == cudaSuccess;
        } else {
            ok = true;
            for (int t = 0; t < T && ok; t += InkGpu::kRows) {
                const int c = std::min(InkGpu::kRows, T - t);
                ok = gemvw_rows(Proj{p.w, p.type, 1.0f}, x + size_t(t) * n_in, y + size_t(t) * n_out, n_out, n_in, c);
            }
        }
        return ok && (p.scale == 1.0f || mul(y, p.scale, n_out * T));
    }
    bool rms_rows(const float* x, size_t xs, const float* w, float* y, size_t ys, int T) {
        rmsnorm_rows_kernel<<<T, 256, 0, stream>>>(x, xs, w, y, ys, n_embd, rms_eps);
        return cudaGetLastError() == cudaSuccess;
    }
    // dst rows (pitch dp floats) = src rows (pitch sp floats), n floats each, T rows
    bool cpy2(float* dst, size_t dp, const float* src, size_t sp, size_t n, int T) {
        return cudaMemcpy2DAsync(dst, dp * sizeof(float), src, sp * sizeof(float), n * sizeof(float), size_t(T),
                                 cudaMemcpyDeviceToDevice, stream) == cudaSuccess;
    }
    // control() over T rows: io rows at io_stride, message rows at msg_stride.
    bool control_rows(const InkLayer& W, float* io, size_t io_stride, const float* msg, size_t msg_stride, int ci,
                      int T) {
        return ink::control_gate_rows(io, io_stride, msg, msg_stride, P.gate_in, n_embd, T, stream) &&
               mm_rows(W.ctrl_down[ci], P.gate_in, -1, P.lo, gate_rank, 2 * n_embd, T) &&
               mm_rows(W.ctrl_gain[ci], P.lo, -1, P.gvec, n_embd, gate_rank, T) &&
               mm_rows(W.ctrl_forget[ci], P.lo, -1, P.fvec, n_embd, gate_rank, T) &&
               ink::control_combine_rows(io, io_stride, msg, msg_stride, P.gvec, P.fvec, n_embd, T, stream);
    }

    // y[t] = W x[t] for T rows (x rows n_in apart, y rows n_out apart): one read of W when gemv_cols takes the
    // type and shape, else T gemv calls -- the same bits either way (gemv_cols' contract).
    bool gemvw_rows(const Proj& p, const float* x, float* y, int n_out, int n_in, int T) {
        if (T == 1) return gemvw(p, x, y, n_out, n_in);
        if (!gemv_cols(p.type, p.w, x, n_in, y, n_out, n_out, n_in, T, false, stream))
            for (int t = 0; t < T; ++t)
                if (!gemv(p.type, p.w, x + size_t(t) * n_in, y + size_t(t) * n_out, n_out, n_in, false, stream))
                    return false;
        return p.scale == 1.0f || mul(y, p.scale, n_out * T);
    }

    // The decode pass as two captured CUDA graphs (without / with the head): the token and the position
    // reach it through device memory, so one capture serves every position. copied: recorded after each
    // upload from a pinned staging buffer, waited on before the buffer is written again.
    float* d_tau = nullptr;   // [n_ctx] the global layers' position factor
    cudaGraphExec_t graph[2] = {nullptr, nullptr};
    cudaEvent_t copied = nullptr;
    bool use_graph = true;
    bool pass(bool head);   // the decode pass after the uploads: what the graphs hold

    int max_blocks = 0;
    int next_pos = 0;
    bool head_ran = false;

    ~St();

    void* alloc(size_t bytes) {
        void* p = nullptr;
        if (cudaMalloc(&p, bytes) != cudaSuccess) {
            alloc_failed = true;
            return nullptr;
        }
        allocs.push_back(p);
        return p;
    }
    bool alloc_failed = false;
    float* alloc_f(size_t n) { return static_cast<float*>(alloc(n * sizeof(float))); }

    bool rms1(const float* x, const float* w, float* y, int n) {
        rmsnorm_kernel<<<1, 256, 0, stream>>>(x, w, y, n, rms_eps);
        return cudaGetLastError() == cudaSuccess;
    }
    bool mul_tau(float* v, const int* pos, int n) {
        mul_tau_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(v, d_tau, pos, n);
        return cudaGetLastError() == cudaSuccess;
    }
    bool mul(float* v, float s, int n) {
        mul_scalar_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(v, s, n);
        return cudaGetLastError() == cudaSuccess;
    }

    // The products with the Proj's optional scalar scale folded in after (src/step.cpp Ref::mv).
    bool gemvw(const Proj& p, const float* x, float* y, int n_out, int n_in) {
        if (!gemv(p.type, p.w, x, y, n_out, n_in, false, stream)) return false;
        return p.scale == 1.0f || mul(y, p.scale, n_out);
    }

    // The loop controller around its F32 GEMVs (src/step.cpp ink_visit's control, docs/inkling2.md §5).
    // gain and forget share the input, so they are one dual product (W2/y2, blockIdx.z picks) unless the
    // two tensors carry different types; the optional scales fold in after, exactly as gemvw does.
    bool control(const InkLayer& W, float* io, const float* message, int ci) {
        const Proj& gain = W.ctrl_gain[ci];
        const Proj& forget = W.ctrl_forget[ci];
        const bool dual = gain.type == forget.type;
        return ink::control_gate(io, message, gate_in, n_embd, stream) &&
               gemvw(W.ctrl_down[ci], gate_in, lo, gate_rank, 2 * n_embd) &&
               (dual ? gemv_dual(gain.type, gain.w, forget.w, lo, gvec, fvec, n_embd, gate_rank, stream) &&
                           (gain.scale == 1.0f || mul(gvec, gain.scale, n_embd)) &&
                           (forget.scale == 1.0f || mul(fvec, forget.scale, n_embd))
                     : gemvw(gain, lo, gvec, n_embd, gate_rank) && gemvw(forget, lo, fvec, n_embd, gate_rank)) &&
               ink::control_combine(io, message, gvec, fvec, io, n_embd, stream);
    }

    bool init(const model::Model& m, int n_ctx_, std::string* err, uint64_t reserve = 0);
    bool fit_ctx(int file_ctx, uint64_t reserve, std::string* err);
    bool load(const model::Model& m, std::string* err);
    bool make_state(std::string* err);
    bool visit(int il, int ip, int pos, const float* msgs_, int& nb);
    bool step(int32_t token, int pos, bool head);
    void reset();
};

// One visit of the layer graph (src/step.cpp ink_visit): the depth mixes, the loop controller, the
// relative-bias attention, the residual bank, and the FFN half. `nb` is the residual bank's row count (the
// caller owns the pass-level bookkeeping), `msgs_` the four controller messages or null.
bool InkGpu::St::visit(int il, int ip, int /*pos: the kernels read d_pos*/, const float* msgs_, int& nb) {
    St& S = *this;
    const int core = S.loop_end - S.loop_start;
    const int phys = il < S.n_layer ? il : S.loop_start + (il - S.n_layer) % core;
    const InkGpu::St::InkLayer& W = S.L[static_cast<size_t>(phys)];
    const bool swa = W.swa;
    const int e = S.n_embd;
    const size_t es = static_cast<size_t>(e);
    cudaStream_t s = S.stream;
    bool ok = true;
    const auto cpy = [&](float* dst, const float* src, size_t n) {
        return cudaMemcpyAsync(dst, src, n * sizeof(float), cudaMemcpyDeviceToDevice, s) == cudaSuccess;
    };

    // inpSA is the pre-attention state; the residual bank reads it, not the mutated cur.
    ok = cpy(S.inpSA, S.cur, es) && ok;

    // The attention's inputs: the depth mix first (every layer but 0), then, in a looped pass, the
    // controller, then the norms.
    if (W.res_q_attn) {
        const int qn = swa ? 1 : 3;
        ok = cpy(S.blocks + size_t(nb) * es, S.cur, es) && ok;
        ok = ink::mix(S.blocks, W.res_q_attn, S.mixedq, nb + 1, e, qn, s) && ok;
        ok = cpy(S.in0, S.mixedq, es) && ok;
        ok = cpy(S.in1, S.mixedq + size_t(swa ? 0 : 1) * es, es) && ok;
        ok = cpy(S.in2, S.mixedq + size_t(swa ? 0 : 2) * es, es) && ok;
    } else {
        ok = cpy(S.in0, S.inpSA, es) && ok;
        ok = cpy(S.in1, S.inpSA, es) && ok;
        ok = cpy(S.in2, S.inpSA, es) && ok;
    }
    if (msgs_) {
        ok = S.control(W, S.in0, msgs_, 0) && ok;
        ok = S.control(W, S.in1, msgs_ + es, 1) && ok;
        ok = S.control(W, S.in2, msgs_ + 2 * es, 2) && ok;
    }
    if (swa) {
        ok = S.rms1(S.in0, W.attn_norm, S.in0, e) && ok;
        if (msgs_) {
            ok = S.rms1(S.in1, W.attn_norm, S.in1, e) && ok;
            ok = S.rms1(S.in2, W.attn_norm, S.in2, e) && ok;
        } else {
            ok = cpy(S.in1, S.in0, es) && ok;
            ok = cpy(S.in2, S.in0, es) && ok;
        }
    } else {
        ok = S.rms1(S.in0, W.q_inp_norm, S.in0, e) && ok;
        ok = S.rms1(S.in1, W.k_inp_norm, S.in1, e) && ok;
        ok = S.rms1(S.in2, W.v_inp_norm, S.in2, e) && ok;
    }

    // q, k, v through their projections; k and v then through their short convolutions (the state keeps the
    // raw projections). r for the relative bias.
    const int nh = S.n_head, nkv = S.n_head_kv, hd = S.head_dim, kvw = S.kvw, rel_len = S.rel_length;
    ok = S.gemvw(W.wq, S.in0, S.q, nh * hd, e) && ok;
    ok = S.gemvw(W.wk, S.in1, S.kraw, kvw, e) && ok;
    ok = S.gemvw(W.wv, S.in2, S.vraw, kvw, e) && ok;
    ok = S.gemvw(W.wr, S.in0, S.rv, nh * rel_len, e) && ok;
    const float* kdyn = nullptr;
    const float* vdyn = nullptr;
    if (!swa) {
        ok = S.gemvw(W.k_conv_proj, S.in1, S.kdyn, S.sconv * nkv, e) && ok;
        ok = S.gemvw(W.v_conv_proj, S.in2, S.vdyn, S.sconv * nkv, e) && ok;
        kdyn = S.kdyn;
        vdyn = S.vdyn;
    }
    float* hist = S.conv_hist + size_t(il) * (S.sconv - 1) * size_t(S.nch);
    ok = ink::conv(S.sconv, S.kraw, kdyn, W.k_conv, S.k, hist, 0, kvw, nkv, S.nch, 1, s) && ok;
    ok = ink::conv(S.sconv, S.vraw, vdyn, W.v_conv, S.v, hist, kvw, kvw, nkv, S.nch, 1, s) && ok;

    // The clips bound q, k and r in place of the q/k norms the file carries no tensors for; V gets a
    // per-head RMS norm with no weight.
    ok = ink::clip(S.q, W.q_clip, nh, hd, S.qk_clip, 1, s) && ok;
    ok = ink::clip(S.k, nullptr, nkv, hd, S.qk_clip, 1, s) && ok;
    ok = ink::clip(S.rv, W.rel_clip, nh, rel_len, S.rel_clip, 1, s) && ok;
    {
        headnorm_kernel<<<nkv, 128, 0, s>>>(S.v, S.v, hd, S.rms_eps);
        ok = cudaGetLastError() == cudaSuccess && ok;
    }
    if (!swa) {
        // the global layers' log-scaled positions multiply q and r, not k
        ok = S.mul_tau(S.q, S.d_pos, nh * hd) && ok;
        ok = S.mul_tau(S.rv, S.d_pos, nh * rel_len) && ok;
    }

    // The bias table: one row per backward distance, per head -- rel_bias row d dotted with r's head column,
    // minus this token's sink logit. gemv_experts runs the n_head dots as one launch (one shared matrix,
    // expert_bytes 0), writing the products transposed; the fold subtracts the sink and pads the zero row.
    ok = S.gemvw(W.sink, S.in0, S.sinkv, nh, e) && ok;
    const int extent = swa ? S.window : S.rel_extent;
    if (S.exact_bias) {
        bias_table_kernel<<<static_cast<unsigned>(((extent + 1) * nh + 255) / 256), 256, 0, s>>>(
            W.relb32, rel_len, S.rv, S.sinkv, S.table, extent, nh);
        ok = cudaGetLastError() == cudaSuccess && ok;
    } else {
        ok = gemv_experts(W.rel_bias.type, W.rel_bias.w, 0, S.head_ids, nh, S.rv, rel_len, S.tblsc, extent, rel_len,
                          s) &&
             ok;
        bias_fold_kernel<<<static_cast<unsigned>(((extent + 1) * nh + 255) / 256), 256, 0, s>>>(
            S.tblsc, S.sinkv, S.table, extent, nh);
        ok = cudaGetLastError() == cudaSuccess && ok;
    }

    // The cache holds the clipped k and the normalised v of this position (src/step.cpp writes after them).
    ok = S.store_rows(il, S.k, S.v, S.d_pos, 1) && ok;

    // The attention itself: GQA over the slot's f16 cache, scores scale*(q.k + bias), causal, SWA's window,
    // and the folded sink column (logit 0) in the denominator.
    const int ring = S.slot_cap[size_t(il)] < S.n_ctx ? S.slot_cap[size_t(il)] : 0;
    ok = (S.attn_rows ? ink::attention_rows(S.q, S.kc_of(il), S.vc_of(il), S.table, 0, S.attnv, S.attn_scratch,
                                            S.d_pos, 1, S.n_ctx, ring, nh, nkv, hd, S.attn_scale, extent,
                                            swa ? S.window : 0, s, -1, S.ks_of(il), S.vs_of(il))
                      : ink::attention(S.q, S.kc_of(il), S.vc_of(il), S.table, S.attnv, S.attn_scratch, S.d_pos, 1,
                                       S.n_ctx, ring, nh, nkv, hd, S.attn_scale, extent, swa ? S.window : 0, s, 0,
                                       S.ks_of(il), S.vs_of(il))) &&
         ok;

    // The output gate, then the output projection and its short convolution.
    ok = S.gemvw(W.wgate, S.in0, S.gate, nh * hd, e) && ok;
    {
        sigmoid_gate_kernel<<<static_cast<unsigned>((nh * hd + 255) / 256), 256, 0, s>>>(S.attnv, S.gate, nh * hd);
        ok = cudaGetLastError() == cudaSuccess && ok;
    }
    ok = S.gemvw(W.wo, S.attnv, S.attn_out, e, nh * hd) && ok;
    ok = ink::conv(S.sconv, S.attn_out, nullptr, W.attn_post_conv, S.attn_conv, hist, 2 * kvw, e, 1, S.nch, 1, s) &&
         ok;

    // Every res_block-th visit starts a new residual stream by banking the pre-attention state; the others
    // add the attention output to it.
    if (ip % S.res_block == 0) {
        ok = cpy(S.blocks + size_t(nb) * es, S.inpSA, es) && ok;
        ++nb;
        ok = cpy(S.cur, S.attn_conv, es) && ok;
    } else {
        add_kernel<<<static_cast<unsigned>((e + 255) / 256), 256, 0, s>>>(S.cur, S.inpSA, S.attn_conv, e);
        ok = cudaGetLastError() == cudaSuccess && ok;
    }

    // The FFN half: the depth mix, the loop controller on its own message, the norm, the experts, and the
    // FFN output's short convolution, added back to the pre-FFN state.
    ok = cpy(S.ffn_inp, S.cur, es) && ok;
    ok = cpy(S.blocks + size_t(nb) * es, S.cur, es) && ok;
    ok = ink::mix(S.blocks, W.res_q_ffn, S.mixed, nb + 1, e, 1, s) && ok;
    if (msgs_) ok = S.control(W, S.mixed, msgs_ + 3 * es, 3) && ok;
    ok = S.rms1(S.mixed, W.ffn_norm, S.mixed, e) && ok;
    ok = ink::moe_route(S.mixed, W.router.w, W.router.type, W.exp_bias, S.n_expert, S.n_expert_used, e, S.route_scale,
                        S.ids, S.moe_w, S.moe_logits, s) &&
         ok;
    ok = ink::moe_gate_up_situ(W.gate_up.w, W.gate_up.type, W.gu_expert_bytes, S.ids, S.n_expert_used, S.mixed,
                               W.gu_scale, S.n_ff_exp, e, S.situ_beta, S.situ_lin_beta, S.moe_h, S.moe_raw, s) &&
         ok;
    ok = ink::moe_down_accum(W.down.w, W.down.type, W.dn_expert_bytes, S.ids, S.n_expert_used, S.moe_h, W.dn_scale,
                             S.moe_w, S.n_ff_exp, e, S.moe_out, S.moe_raw, s) &&
         ok;
    ok = ink::conv(S.sconv, S.moe_out, nullptr, W.ffn_post_conv, S.ffn_conv, hist, 2 * kvw + e, e, 1, S.nch, 1, s) &&
         ok;
    add_kernel<<<static_cast<unsigned>((e + 255) / 256), 256, 0, s>>>(S.cur, S.ffn_inp, S.ffn_conv, e);
    ok = cudaGetLastError() == cudaSuccess && ok;

    ok = cpy(S.l_out + size_t(il) * es, S.cur, es) && ok;
    return ok;
}

// ---- open: the geometry, the weights, the state and the scratch ------------------------------------------
bool InkGpu::St::init(const model::Model& m, int n_ctx_, std::string* err, uint64_t reserve) {
    if (const char* kv = std::getenv("LIMBO_INK_KV")) kv_q8 = kv_q8 || std::string(kv) == "q8";
    const model::Geometry& g = m.geometry();
    const auto fail = [&](const std::string& msg) {
        *err = "InkGpu: " + msg;
        return false;
    };
    if (g.arch != "inkling2_loop") return fail("not an inkling2_loop file (arch \"" + g.arch + "\")");
    std::string terr;
    if (!init_tables(&terr)) return fail("codebook upload: " + terr);

    n_layer = g.n_layer;
    n_embd = g.n_embd;
    head_dim = g.head_dim;
    n_head = g.n_head_at(0);
    n_head_kv = g.n_head_kv_at(0);
    kvw = n_head_kv * head_dim;
    nch = 2 * kvw + 2 * n_embd;
    n_expert = g.n_expert;
    n_expert_used = g.n_expert_used;
    n_ff_exp = g.n_ff_exp;
    rel_length = g.rel_length;
    rel_extent = g.rel_extent;
    window = g.sliding_window;
    tbl_extent = std::max(window, rel_extent);
    sconv = g.sconv;
    res_block = g.attn_res_block;
    log_floor = g.log_scale_floor;
    gate_rank = g.loop_gate_rank;
    loop_start = g.loop_start;
    loop_end = g.loop_end;
    loop_count = g.loop_count;
    qk_clip = float(g.qk_clip);
    rel_clip = float(g.rel_clip);
    log_alpha = float(g.log_scale_alpha);
    situ_beta = float(g.situ_beta);
    situ_lin_beta = float(g.situ_linear_beta);
    route_scale = float(g.expert_weights_scale);
    rms_eps = float(g.rms_eps);
    softcap = float(g.logit_softcap);
    n_vocab = g.n_vocab_unpadded;
    n_ctx = n_ctx_;
    attn_scale = float(1.0 / std::sqrt(double(head_dim)));

    // The file must be the graph this pass runs (docs/inkling2.md): its constants present, its shapes what
    // the kernels take. Anything else is refused here, never approximated.
    if (n_layer <= 0 || n_embd <= 0 || n_embd % 8 != 0) return fail("bad geometry");
    if (rel_extent <= 0 || rel_length <= 0 || sconv < 2 || res_block <= 0 || loop_count < 2 || gate_rank < 1 ||
        situ_beta <= 0 || situ_lin_beta <= 0)
        return fail("inkling2_loop without its relative-bias, short-conv, attn-res, loop or SiTU metadata");
    if (loop_start < 0 || loop_start >= loop_end || loop_end > n_layer) return fail("loop bounds out of range");
    if (n_expert <= 0 || n_expert > 1024 || n_expert_used <= 0 || n_expert_used > n_expert || n_ff_exp <= 0)
        return fail("expert geometry out of range");
    if (route_scale <= 0) return fail("expert_weights_scale must be positive");
    if (n_vocab <= 0 || n_vocab > g.n_vocab) return fail("vocab_size_unpadded is unusable");
    if (head_dim != 128 || n_head <= 0 || n_head_kv <= 0 || n_head % n_head_kv != 0 || n_head / n_head_kv > 4)
        return fail("the attention kernels take head_dim 128 and a GQA group of at most 4");
    for (int l = 0; l < n_layer; ++l)
        if (g.n_head_at(l) != n_head || g.n_head_kv_at(l) != n_head_kv)
            return fail("per-layer head counts are not uniform (this pass runs the file as it stands)");
    n_slot = n_layer + (loop_count - 1) * (loop_end - loop_start);

    // The residual bank's high-water mark over the visit sequence (the depth mix takes 8 states at most).
    {
        int nb = 0, maxb = 0;
        const auto push = [&](int ip) {
            if (ip % res_block == 0) {
                ++nb;
                maxb = std::max(maxb, nb);
            }
        };
        for (int ip = 0; ip < loop_start; ++ip) push(ip);
        const int prelude_nb = nb;
        for (int pass = 0; pass < loop_count; ++pass) {
            if (pass > 0) nb = prelude_nb;
            for (int ip = loop_start; ip < loop_end; ++ip) push(ip);
        }
        for (int ip = loop_end; ip < n_layer; ++ip) push(ip);
        max_blocks = maxb;
    }
    if (max_blocks + 1 > 8) return fail("the residual bank exceeds the depth mix's 8 states");
    if (rel_length > kRelMax || n_head > kHeadMax) return fail("the prompt pass's bias tables take rel_length and n_head up to 32");
    if (!load(m, err)) return false;
    exact_bias = rel_length <= 16 && !std::getenv("LIMBO_INK_BIAS_GEMV");
    for (const InkLayer& W : L) exact_bias = exact_bias && W.rel_bias.type == static_cast<uint32_t>(Type::F32) && W.relb32;
    if (n_ctx <= 0 && !fit_ctx(g.n_ctx, reserve, err)) return false;
    return make_state(err);
}

// The context that fits: what is free once the weights are resident, minus the reserve, the sliding rings and
// a fixed margin for the step's scratch, over what one more position costs -- K and V of every global slot
// plus the attention scratch's per-chunk partials. Rounded down to the kChunk grid and capped at the file's
// context_length. The state account is the one ink_budget verifies (192 KiB per position before the ring).
bool InkGpu::St::fit_ctx(int file_ctx, uint64_t reserve, std::string* err) {
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
        *err = "InkGpu: cudaMemGetInfo failed";
        return false;
    }
    const int core = loop_end - loop_start;
    int n_global = 0, n_swa = 0;
    for (int il = 0; il < n_slot; ++il) {
        const int phys = il < n_layer ? il : loop_start + (il - n_layer) % core;
        (L[size_t(phys)].swa && window > 0 ? n_swa : n_global)++;
    }
    // K and V of one position in one slot (q8: int8 values and a float scale per KV head)
    const uint64_t kv_row = kv_q8 ? uint64_t(kvw) * 2 + uint64_t(n_head_kv) * 2 * sizeof(float)
                                  : uint64_t(kvw) * 2 * sizeof(__half);
    const uint64_t per_pos = uint64_t(n_global) * kv_row +
                             (uint64_t(InkGpu::kRows) * n_head * (2 + 128) * sizeof(float)) / ink::kChunk + 1;
    const uint64_t rings = uint64_t(n_swa) * uint64_t(window + ink::kMaxRows - 1) * kv_row;
    const uint64_t margin = uint64_t(448) << 20;   // the prompt pass's piece buffers, scratch, the allocator's slack
    const uint64_t fixed = reserve + rings + margin;
    if (free_b <= fixed + per_pos * ink::kChunk) {
        *err = "InkGpu: no room for context: " + std::to_string(free_b >> 20) + " MiB free after the weights, " +
               std::to_string(fixed >> 20) + " MiB reserved + rings + scratch";
        return false;
    }
    uint64_t fit = (free_b - fixed) / per_pos;
    const uint64_t cap = file_ctx > 0 ? uint64_t(file_ctx) : uint64_t(1) << 20;
    fit = std::min(fit, cap);
    n_ctx = int(fit / ink::kChunk * ink::kChunk);
    return true;
}

// Every tensor the graph reads, uploaded once in its GGUF block layout (the matrices as they are, the small
// vectors and the conv taps decoded to F32, exactly the values Ref's decode_all feeds it).
bool InkGpu::St::load(const model::Model& m, std::string* err) {
    const auto fail = [&](const std::string& msg) {
        *err = "InkGpu: " + msg;
        return false;
    };
    auto up_raw = [&](const gguf::Tensor* t, const std::string& what, Proj& p) {
        if (!t) return true;
        if (!gemv_supported(t->type)) return fail("no GEMV kernel for " + what);
        void* d = alloc(t->nbytes);
        if (!d) return fail("out of device memory for " + what);
        if (cudaMemcpy(d, t->data, t->nbytes, cudaMemcpyHostToDevice) != cudaSuccess)
            return fail("upload failed for " + what);
        p.w = static_cast<const uint8_t*>(d);
        p.type = t->type;
        if ((t->type == static_cast<uint32_t>(Type::BF16) || t->type == static_cast<uint32_t>(Type::F16)) &&
            t->n_elements() <= (int64_t(1) << 22)) {
            std::vector<float> tmp(static_cast<size_t>(t->n_elements()));
            if (!quants::dequant_row(t->type, t->data, tmp.data(), static_cast<uint64_t>(tmp.size())))
                return fail("cannot decode " + what);
            float* f = alloc_f(tmp.size());
            if (!f || cudaMemcpy(f, tmp.data(), tmp.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
                return fail("upload failed for " + what);
            p.f32 = f;
        }
        return true;
    };
    // A decoded F32 vector (norm weights, clip scales, conv taps, queries, biases): `want` elements or 0.
    auto up_vec = [&](const gguf::Tensor* t, const std::string& what, int want, float*& dst) {
        if (!t) return true;
        if (want > 0 && t->n_elements() != want) return fail(what + " has the wrong size");
        std::vector<float> tmp(static_cast<size_t>(t->n_elements()));
        if (!quants::dequant_row(t->type, t->data, tmp.data(), static_cast<uint64_t>(tmp.size())))
            return fail("cannot decode " + what);
        dst = alloc_f(tmp.size());
        if (!dst) return fail("out of device memory for " + what);
        return cudaMemcpy(dst, tmp.data(), tmp.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess ||
               fail("upload failed for " + what);
    };
    // A matrix with its optional scalar <name>.scale (src/step.cpp Ref::Proj).
    auto proj = [&](int l, const std::string& base, bool required, Proj& p) {
        const gguf::Tensor* t = m.tensor(l, base + ".weight");
        if (!t) return !required || fail("missing tensor " + (l >= 0 ? "blk." + std::to_string(l) + "." : "") + base);
        if (const gguf::Tensor* sc = m.tensor(l, base + ".scale"); sc && sc->n_elements() == 1) {
            std::vector<float> tmp(1);
            quants::dequant_row(sc->type, sc->data, tmp.data(), 1);
            p.scale = tmp[0];
        }
        return up_raw(t, base, p);
    };
    auto vec = [&](int l, const std::string& role, bool required, int want, float*& dst) {
        const gguf::Tensor* t = m.tensor(l, role);
        if (!t) return !required || fail("missing tensor " + (l >= 0 ? "blk." + std::to_string(l) + "." : "") + role);
        return up_vec(t, role, want, dst);
    };

    // ---- model level ---------------------------------------------------------------------------------
    if (!proj(-1, "token_embd", true, tok_embd) || !vec(-1, "token_embd_norm.weight", true, n_embd, tok_norm) ||
        !vec(-1, "output_res_query.weight", true, n_embd, out_res_query) ||
        !vec(-1, "output_norm.weight", true, n_embd, out_norm) || !proj(-1, "output", true, output))
        return false;

    // ---- layers ---------------------------------------------------------------------------------------
    L.assign(static_cast<size_t>(n_layer), InkLayer());
    for (int l = 0; l < n_layer; ++l) {
        InkLayer& W = L[static_cast<size_t>(l)];
        W.swa = m.geometry().is_swa(l);
        const int qn = W.swa ? 1 : 3;   // attn_res_query: one row on the sliding layers, three on the global
        if (!proj(l, "attn_q", true, W.wq) || !proj(l, "attn_k", true, W.wk) || !proj(l, "attn_v", true, W.wv) ||
            !proj(l, "attn_output", true, W.wo) || !proj(l, "attn_rel", true, W.wr) ||
            !proj(l, "attn_rel_bias", true, W.rel_bias) || !proj(l, "attn_gate", true, W.wgate) ||
            !proj(l, "attn_sink_proj", true, W.sink))
            return false;
        if (!vec(l, "attn_rel_bias.weight", true, 0, W.relb32)) return false;
        if (!vec(l, "attn_q_clip.weight", true, n_head * head_dim, W.q_clip) ||
            !vec(l, "attn_rel_clip.weight", true, n_head * rel_length, W.rel_clip))
            return false;
        // the convs: F32 taps, sconv per channel, contiguous per channel (ink::conv's w)
        if (!vec(l, "attn_k_conv.weight", true, sconv * kvw, W.k_conv) ||
            !vec(l, "attn_v_conv.weight", true, sconv * kvw, W.v_conv) ||
            !vec(l, "attn_post_conv.weight", true, sconv * n_embd, W.attn_post_conv) ||
            !vec(l, "ffn_post_conv.weight", true, sconv * n_embd, W.ffn_post_conv))
            return false;
        if (W.swa) {
            if (!vec(l, "attn_norm.weight", true, n_embd, W.attn_norm)) return false;
        } else {
            if (!vec(l, "attn_q_inp_norm.weight", true, n_embd, W.q_inp_norm) ||
                !vec(l, "attn_k_inp_norm.weight", true, n_embd, W.k_inp_norm) ||
                !vec(l, "attn_v_inp_norm.weight", true, n_embd, W.v_inp_norm))
                return false;
            if (!proj(l, "attn_k_conv_proj", true, W.k_conv_proj) || !proj(l, "attn_v_conv_proj", true, W.v_conv_proj))
                return false;
        }
        if (l > 0 && !vec(l, "attn_res_query.weight", true, qn * n_embd, W.res_q_attn)) return false;
        if (!vec(l, "ffn_norm.weight", true, n_embd, W.ffn_norm) || !proj(l, "ffn_gate_inp", true, W.router) ||
            !vec(l, "exp_probs_b.bias", true, n_expert, W.exp_bias) ||
            !proj(l, "ffn_gate_up_exps", true, W.gate_up) || !proj(l, "ffn_down_exps", true, W.down))
            return false;
        // the per-expert dequantisation scales (docs/inkling2.md §4): some layers carry none and
        // dequantise to unit scale; identity stays null so the kernels multiply by exactly 1.
        if (!vec(l, "ffn_gate_up_exps.scale", false, n_expert, W.gu_scale) ||
            !vec(l, "ffn_down_exps.scale", false, n_expert, W.dn_scale))
            return false;
        if (!vec(l, "ffn_res_query.weight", true, n_embd, W.res_q_ffn)) return false;
        if (l >= loop_start && l < loop_end) {
            static const char* kCtrl[4] = {"q", "k", "v", "ffn"};
            for (int ci = 0; ci < 4; ++ci) {
                const std::string b = std::string("loop_") + kCtrl[ci] + "_ctrl_";
                if (!proj(l, b + "down", true, W.ctrl_down[ci]) || !proj(l, b + "gain", true, W.ctrl_gain[ci]) ||
                    !proj(l, b + "forget", true, W.ctrl_forget[ci]))
                    return false;
            }
            if (!vec(l, "loop_msg_query.weight", true, 4 * n_embd, W.msg_query)) return false;
        }
        // the expert stacks' geometry is what the grouped products assume
        const gguf::Tensor* gu = m.tensor(l, "ffn_gate_up_exps.weight");
        const gguf::Tensor* dn = m.tensor(l, "ffn_down_exps.weight");
        if (!gu || gu->ne[0] != n_embd || gu->ne[1] != 2 * n_ff_exp || gu->ne[2] != n_expert)
            return fail("blk." + std::to_string(l) + ".ffn_gate_up_exps.weight has the wrong shape");
        if (!dn || dn->ne[0] != n_ff_exp || dn->ne[1] != n_embd || dn->ne[2] != n_expert)
            return fail("blk." + std::to_string(l) + ".ffn_down_exps.weight has the wrong shape");
        const gguf::Tensor* rb = m.tensor(l, "attn_rel_bias.weight");
        if (!rb || rb->ne[0] != rel_length || rb->ne[1] < tbl_extent)
            return fail("blk." + std::to_string(l) + ".attn_rel_bias.weight has the wrong shape");
        W.gu_expert_bytes = uint64_t(2 * n_ff_exp) * type_bytes(W.gate_up.type, uint64_t(n_embd));
        W.dn_expert_bytes = uint64_t(n_embd) * type_bytes(W.down.type, uint64_t(n_ff_exp));
    }
    return !alloc_failed || fail("out of device memory");
}

// The per-slot state and the step's scratch, sized once; everything zeroed like Ref::reset leaves it.
bool InkGpu::St::make_state(std::string* err) {
    const auto fail = [&](const std::string& msg) {
        *err = "InkGpu: " + msg;
        return false;
    };
    const size_t es = size_t(n_embd), kvws = size_t(kvw), nhs = size_t(n_head), gr = size_t(gate_rank);
    const size_t hd = size_t(head_dim), rels = size_t(rel_length), nes = size_t(n_expert);
    const size_t nus = size_t(n_expert_used), nfs = size_t(n_ff_exp);
    slot_off.assign(size_t(n_slot), 0);
    slot_cap.assign(size_t(n_slot), n_ctx);
    kv_rows = 0;
    {
        const int core = loop_end - loop_start;
        for (int il = 0; il < n_slot; ++il) {
            const int phys = il < n_layer ? il : loop_start + (il - n_layer) % core;
            if (L[size_t(phys)].swa && window > 0) slot_cap[size_t(il)] = std::min(n_ctx, window + ink::kMaxRows - 1);
            slot_off[size_t(il)] = kv_rows;
            kv_rows += size_t(slot_cap[size_t(il)]);
        }
    }
    const size_t elt = kv_q8 ? 1 : sizeof(__half);
    k_cache = static_cast<__half*>(alloc(kv_rows * kvws * elt));
    v_cache = static_cast<__half*>(alloc(kv_rows * kvws * elt));
    if (kv_q8) {
        k_scale = alloc_f(kv_rows * size_t(n_head_kv));
        v_scale = alloc_f(kv_rows * size_t(n_head_kv));
    }
    conv_hist = alloc_f(size_t(n_slot) * size_t(sconv - 1) * size_t(nch));
    l_out = alloc_f(size_t(n_slot) * es);

    cur = alloc_f(es);
    inpSA = alloc_f(es);
    prelude = alloc_f(es);
    bank_out = alloc_f(es);
    blocks = alloc_f(size_t(max_blocks + 1) * es);
    bank = alloc_f(size_t(max_blocks + 1) * es);
    in0 = alloc_f(es);
    in1 = alloc_f(es);
    in2 = alloc_f(es);
    mixedq = alloc_f(3 * es);
    mixed = alloc_f(es);
    msgs = alloc_f(4 * es);
    gate_in = alloc_f(2 * es);
    lo = alloc_f(gr);
    gvec = alloc_f(es);
    fvec = alloc_f(es);
    q = alloc_f(nhs * hd);
    gate = alloc_f(nhs * hd);
    attnv = alloc_f(nhs * hd);
    kraw = alloc_f(kvws);
    vraw = alloc_f(kvws);
    k = alloc_f(kvws);
    v = alloc_f(kvws);
    kdyn = alloc_f(size_t(sconv) * size_t(n_head_kv));
    vdyn = alloc_f(size_t(sconv) * size_t(n_head_kv));
    rv = alloc_f(nhs * rels);
    sinkv = alloc_f(nhs);
    tblsc = alloc_f(nhs * size_t(tbl_extent));
    table = alloc_f(size_t(tbl_extent + 1) * nhs);
    attn_out = alloc_f(es);
    attn_conv = alloc_f(es);
    ffn_inp = alloc_f(es);
    moe_out = alloc_f(es);
    ffn_conv = alloc_f(es);
    moe_logits = alloc_f(nes);
    moe_w = alloc_f(nus);
    moe_h = alloc_f(nus * nfs);
    moe_raw = alloc_f(ink::moe_scratch_floats(n_expert_used, n_ff_exp, n_embd));
    rn = alloc_f(es);
    logits = alloc_f(size_t(n_vocab));
    attn_scratch = alloc_f(ink::attention_scratch_floats(n_ctx, InkGpu::kRows, n_head, head_dim));
    ids = static_cast<int32_t*>(alloc(nus * sizeof(int32_t)));
    head_ids = static_cast<int32_t*>(alloc(nhs * sizeof(int32_t)));
    d_token = static_cast<int32_t*>(alloc(sizeof(int32_t)));
    d_pos = static_cast<int32_t*>(alloc(sizeof(int32_t)));
    d_argmax = static_cast<int32_t*>(alloc(sizeof(int32_t)));
    d_tau = alloc_f(size_t(n_ctx));
    {
        const size_t kr = size_t(InkGpu::kPiece);
        const auto rows = [&](size_t n) { return alloc_f(kr * n); };
        R.cur = rows(es);
        R.inpSA = rows(es);
        R.prelude = rows(es);
        R.bank_out = rows(es);
        R.blocks = rows(size_t(max_blocks + 1) * es);
        R.bank = rows(size_t(max_blocks + 1) * es);
        R.in0 = rows(es);
        R.in1 = rows(es);
        R.in2 = rows(es);
        R.mixedq = rows(3 * es);
        R.mixed = rows(es);
        R.msgs = rows(4 * es);
        R.q = rows(nhs * hd);
        R.gate = rows(nhs * hd);
        R.attnv = rows(nhs * hd);
        R.kraw = rows(kvws);
        R.vraw = rows(kvws);
        R.k = rows(kvws);
        R.v = rows(kvws);
        R.kdyn = rows(size_t(sconv) * size_t(n_head_kv));
        R.vdyn = rows(size_t(sconv) * size_t(n_head_kv));
        R.rv = rows(nhs * rels);
        R.sinkv = rows(nhs);
        R.attn_out = rows(es);
        R.attn_conv = rows(es);
        R.ffn_inp = rows(es);
        R.moe_out = rows(es);
        R.ffn_conv = rows(es);
        R.d_tok = static_cast<int32_t*>(alloc(kr * sizeof(int32_t)));
        R.d_pos = static_cast<int32_t*>(alloc(kr * sizeof(int32_t)));
        const size_t k8 = size_t(InkGpu::kRows);
        R.logits = alloc_f(k8 * nes);
        R.moe_w = alloc_f(k8 * nus);
        R.ids = static_cast<int32_t*>(alloc(k8 * nus * sizeof(int32_t)));
        R.moe_h = alloc_f(k8 * nus * nfs);
        R.moe_raw = alloc_f(k8 * ink::moe_scratch_floats(n_expert_used, n_ff_exp, n_embd));
        R.tables = alloc_f(k8 * size_t(tbl_extent + 1) * nhs);
        R.gate_in = alloc_f(k8 * 2 * es);
        R.lo = alloc_f(k8 * gr);
        R.gvec = alloc_f(k8 * es);
        R.fvec = alloc_f(k8 * es);
    }
    feats = alloc_f(size_t(InkGpu::kPiece) * 4 * es);
    logits_rows = alloc_f(size_t(InkGpu::kRows) * size_t(n_vocab));
    argmax_rows = static_cast<int32_t*>(alloc(size_t(InkGpu::kRows) * sizeof(int32_t)));
    conv_rec = alloc_f(size_t(n_slot) * InkGpu::kRows * size_t(nch));
    hist_snap = alloc_f(size_t(n_slot) * size_t(sconv - 1) * size_t(nch));
    if (alloc_failed) return fail("out of device memory");
    if (cudaMallocHost(reinterpret_cast<void**>(&R.pinned), 2 * size_t(InkGpu::kPiece) * sizeof(int32_t)) != cudaSuccess)
        return fail("out of pinned host memory");
    {
        std::string gerr;
        if (!gemm_init(&gerr)) return fail("gemm codebooks: " + gerr);
        const size_t kp = size_t(InkGpu::kPiece), pairs = kp * size_t(n_expert_used), nff = size_t(n_ff_exp);
        P.qk = std::max(n_embd, n_head * head_dim);
        for (int i = 0; i < 3; ++i) {
            P.xq[i] = static_cast<int8_t*>(alloc(size_t(3) * kp * size_t(P.qk)));
            P.xs[i] = alloc_f(kp * size_t(P.qk / 32));
            P.xsum[i] = alloc_f(kp * size_t(P.qk / 16));
        }
        P.gate_in = alloc_f(kp * 2 * es);
        P.lo = alloc_f(kp * gr);
        P.gvec = alloc_f(kp * es);
        P.fvec = alloc_f(kp * es);
        P.logits = alloc_f(kp * nes);
        P.w = alloc_f(pairs);
        P.ids = static_cast<int32_t*>(alloc(pairs * sizeof(int32_t)));
        P.pe = static_cast<int32_t*>(alloc(pairs * sizeof(int32_t)));
        P.raw = alloc_f(pairs * 2 * nff);
        P.h = alloc_f(pairs * nff);
        P.hq = static_cast<int8_t*>(alloc(size_t(3) * pairs * nff));
        P.hs = alloc_f(pairs * (nff / 32 + 1));
        P.dn = alloc_f(pairs * es);
        P.tables = alloc_f(kp * size_t(tbl_extent + 1) * nhs);
        P.lists.cnt = static_cast<int32_t*>(alloc(nes * sizeof(int32_t)));
        P.lists.start = static_cast<int32_t*>(alloc((nes + 1) * sizeof(int32_t)));
        P.lists.list_tok = static_cast<int32_t*>(alloc(pairs * sizeof(int32_t)));
        P.lists.pos_of = static_cast<int32_t*>(alloc(pairs * sizeof(int32_t)));
        P.lists.tiles_small = static_cast<int4*>(alloc(size_t(moe_max_tiles_small(n_expert)) * sizeof(int4)));
        P.lists.tiles_big =
            static_cast<int4*>(alloc(size_t(moe_max_tiles_big(int(pairs), n_expert)) * sizeof(int4)));
        P.lists.n_tiles = static_cast<int32_t*>(alloc(2 * sizeof(int32_t)));
        if (alloc_failed) return fail("out of device memory for the prompt pass");
        if (const char* e = std::getenv("LIMBO_INK_PLANES")) planes = std::max(1, std::min(3, std::atoi(e)));
    }
    if (cudaMallocHost(reinterpret_cast<void**>(&pinned), 2 * sizeof(int32_t)) != cudaSuccess)
        return fail("out of pinned host memory");
    std::vector<int32_t> zeros(nhs, 0);
    if (cudaMemcpy(head_ids, zeros.data(), nhs * sizeof(int32_t), cudaMemcpyHostToDevice) != cudaSuccess)
        return fail("upload failed");
    if (cudaStreamCreate(&stream) != cudaSuccess) return fail("cudaStreamCreate failed");
    if (cudaEventCreateWithFlags(&copied, cudaEventDisableTiming) != cudaSuccess) return fail("cudaEventCreate failed");
    {
        // tau(pos) = 1 + alpha * log(max(1, (pos + 1) / floor)), in double, rounded once to float (the formula
        // the per-position host code used)
        std::vector<float> tau(static_cast<size_t>(n_ctx));
        for (int p = 0; p < n_ctx; ++p)
            tau[size_t(p)] = float(1.0 + double(log_alpha) *
                                             std::log(std::max(1.0, double(p + 1) / double(std::max(1, log_floor)))));
        if (cudaMemcpy(d_tau, tau.data(), tau.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
            return fail("upload failed");
    }
    if (const char* g = std::getenv("LIMBO_INK_GRAPH")) use_graph = std::string(g) != "0";
    if (const char* a = std::getenv("LIMBO_INK_ATTN")) attn_rows = std::string(a) == "rows";
    reset();
    return true;
}

InkGpu::St::~St() {
    if (stream) cudaStreamSynchronize(stream);
    for (void* p : allocs) cudaFree(p);
    for (cudaGraphExec_t& g : graph)
        if (g) cudaGraphExecDestroy(g);
    for (auto& gr : graph_rows)
        for (cudaGraphExec_t& g : gr)
            if (g) cudaGraphExecDestroy(g);
    if (copied) cudaEventDestroy(copied);
    if (pinned) cudaFreeHost(pinned);
    if (R.pinned) cudaFreeHost(R.pinned);
    if (stream) cudaStreamDestroy(stream);
}

void InkGpu::St::reset() {
    if (stream) cudaStreamSynchronize(stream);
    cudaMemset(k_cache, 0, kv_rows * size_t(kvw) * (kv_q8 ? 1 : sizeof(__half)));
    cudaMemset(v_cache, 0, kv_rows * size_t(kvw) * (kv_q8 ? 1 : sizeof(__half)));
    if (kv_q8) {
        cudaMemset(k_scale, 0, kv_rows * size_t(n_head_kv) * sizeof(float));
        cudaMemset(v_scale, 0, kv_rows * size_t(n_head_kv) * sizeof(float));
    }
    cudaMemset(conv_hist, 0, size_t(n_slot) * size_t(sconv - 1) * size_t(nch) * sizeof(float));
    next_pos = 0;
    head_ran = false;
}

// One whole forward pass (src/step.cpp forward_inkling): the prelude, the looped core with its bank reset
// and per-layer messages, the coda, and with head the build_output mix and the LM head. Everything queues on
// one stream; the only host-to-device traffic is the token and the position.
bool InkGpu::St::step(int32_t token, int pos, bool head) {
    if (pos != next_pos || pos < 0 || pos >= n_ctx) return false;
    // The staging buffer is written again only once the last upload from it has run.
    if (cudaEventSynchronize(copied) != cudaSuccess) return false;
    pinned[0] = token;
    pinned[1] = pos;
    bool ok = cudaMemcpyAsync(d_token, pinned, sizeof(int32_t), cudaMemcpyHostToDevice, stream) == cudaSuccess &&
              cudaMemcpyAsync(d_pos, pinned + 1, sizeof(int32_t), cudaMemcpyHostToDevice, stream) == cudaSuccess &&
              cudaEventRecord(copied, stream) == cudaSuccess;
    cudaGraphExec_t& g = graph[head ? 1 : 0];
    if (use_graph && !g) {
        // One capture per head flag; a pass that fails to capture runs uncaptured from then on.
        cudaGraph_t cg = nullptr;
        bool cap = cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed) == cudaSuccess;
        const bool body = cap && pass(head);
        cap = cudaStreamEndCapture(stream, &cg) == cudaSuccess && cap && body &&
              cudaGraphInstantiate(&g, cg, 0) == cudaSuccess;
        if (cg) cudaGraphDestroy(cg);
        cudaGetLastError();
        if (!cap) {
            use_graph = false;
            g = nullptr;
        }
    }
    ok = (use_graph ? cudaGraphLaunch(g, stream) == cudaSuccess : pass(head)) && ok;
    if (head) head_ran = true;
    next_pos = pos + 1;
    return ok;
}

bool InkGpu::St::pass(bool head) {
    const size_t es = size_t(n_embd);
    const int pos = next_pos;   // unused by the kernels (they read d_pos); kept for visit()'s signature
    bool ok = embed_row(tok_embd.type, tok_embd.w, d_token, cur, n_embd, stream);
    ok = rms1(cur, tok_norm, cur, n_embd) && ok;

    int nb = 0;
    for (int il = 0; il < loop_start; ++il) ok = tap(il, cur, 1) && visit(il, il, pos, nullptr, nb) && ok;
    const int prelude_nb = nb;
    ok = cudaMemcpyAsync(prelude, cur, es * sizeof(float), cudaMemcpyDeviceToDevice, stream) == cudaSuccess && ok;

    // Every pass starts from the prelude output; its bank reads the previous pass's core blocks after the
    // first core block, plus the previous pass's output, through the layer's own message queries.
    const int core = loop_end - loop_start;
    for (int pass = 0; pass < loop_count; ++pass) {
        int bank_n = 0;
        if (pass > 0) {
            bank_n = std::max(0, nb - (prelude_nb + 1));
            if (bank_n > 0)
                ok = cudaMemcpyAsync(bank, blocks + size_t(prelude_nb + 1) * es,
                                     size_t(bank_n) * es * sizeof(float), cudaMemcpyDeviceToDevice, stream) ==
                         cudaSuccess &&
                     ok;
            ok = cudaMemcpyAsync(bank_out, cur, es * sizeof(float), cudaMemcpyDeviceToDevice, stream) ==
                     cudaSuccess &&
                 ok;
            ok = cudaMemcpyAsync(cur, prelude, es * sizeof(float), cudaMemcpyDeviceToDevice, stream) ==
                     cudaSuccess &&
                 ok;
            nb = prelude_nb;
        }
        for (int ip = loop_start; ip < loop_end; ++ip) {
            const int il = pass == 0 ? ip : n_layer + (pass - 1) * core + (ip - loop_start);
            if (pass == 0) {
                ok = tap(il, cur, 1) && visit(il, ip, pos, nullptr, nb) && ok;
                continue;
            }
            // The messages: mix(bank, bank_out, msg_query) with this layer's own queries (the bank and its
            // output are the pass's, the query is the layer's).
            ok = cudaMemcpyAsync(bank + size_t(bank_n) * es, bank_out, es * sizeof(float),
                                 cudaMemcpyDeviceToDevice, stream) == cudaSuccess &&
                 ok;
            ok = ink::mix(bank, L[static_cast<size_t>(ip)].msg_query, msgs, bank_n + 1, n_embd, 4, stream) && ok;
            ok = tap(il, cur, 1) && visit(il, ip, pos, msgs, nb) && ok;
        }
    }
    for (int il = loop_end; il < n_layer; ++il) ok = tap(il, cur, 1) && visit(il, il, pos, nullptr, nb) && ok;

    if (head || capture) {
        ok = cudaMemcpyAsync(blocks + size_t(nb) * es, cur, es * sizeof(float), cudaMemcpyDeviceToDevice, stream) ==
                 cudaSuccess &&
             ok;
        ok = ink::mix(blocks, out_res_query, mixed, nb + 1, n_embd, 1, stream) && ok;
        ok = rms1(mixed, out_norm, rn, n_embd) && ok;
        if (capture)
            ok = cudaMemcpyAsync(feats + 3 * es, rn, es * sizeof(float), cudaMemcpyDeviceToDevice, stream) == cudaSuccess &&
                 ok;
    }
    if (head) {
        ok = gemvw(output, rn, logits, n_vocab, n_embd) && ok;
        if (softcap > 0.0f) {
            softcap_kernel<<<static_cast<unsigned>((n_vocab + 255) / 256), 256, 0, stream>>>(logits, n_vocab, softcap,
                                                                                             1.0f / softcap);
            ok = cudaGetLastError() == cudaSuccess && ok;
        }
    }
    return ok;
}

// visit() over T consecutive rows (step_rows). The order of operations within a row is visit()'s; the
// stateful ones (the convolutions' histories, the K/V stores, the attention) run row by row in position
// order, and the dense projections run once for all rows through gemvw_rows. Row t of a [kRows][n] buffer
// is at buffer + t * n.
bool InkGpu::St::visit_rows(int il, int ip, int /*pos0: the kernels read R.d_pos*/, int T, bool msgs_, int& nb) {
    St& S = *this;
    const int core = S.loop_end - S.loop_start;
    const int phys = il < S.n_layer ? il : S.loop_start + (il - S.n_layer) % core;
    const InkGpu::St::InkLayer& W = S.L[static_cast<size_t>(phys)];
    const bool swa = W.swa;
    const int e = S.n_embd;
    const size_t es = static_cast<size_t>(e), bs = size_t(S.max_blocks + 1) * es;
    cudaStream_t s = S.stream;
    bool ok = true;
    const auto cpy = [&](float* dst, const float* src, size_t n) {
        return cudaMemcpyAsync(dst, src, n * sizeof(float), cudaMemcpyDeviceToDevice, s) == cudaSuccess;
    };
    const int nh = S.n_head, nkv = S.n_head_kv, hd = S.head_dim, kvw = S.kvw, rel_len = S.rel_length;
    const size_t qw = size_t(nh) * hd, kw = size_t(kvw), rw = size_t(nh) * rel_len, dw = size_t(S.sconv) * nkv;
    float* hist = S.conv_hist + size_t(il) * (S.sconv - 1) * size_t(S.nch);

    ok = cpy(R.inpSA, R.cur, size_t(T) * es) && ok;

    // The attention's inputs: the depth mix, the controller, the norms -- every row kernel runs the single-row
    // kernel's arithmetic on its row.
    if (W.res_q_attn) {
        const int qn = swa ? 1 : 3;
        ok = S.cpy2(R.blocks + size_t(nb) * es, bs, R.cur, es, es, T) && ok;
        ok = ink::mix_rows(R.blocks, bs, W.res_q_attn, R.mixedq, 3 * es, nb + 1, e, qn, T, s) && ok;
        ok = S.cpy2(R.in0, es, R.mixedq, 3 * es, es, T) && ok;
        ok = S.cpy2(R.in1, es, R.mixedq + size_t(swa ? 0 : 1) * es, 3 * es, es, T) && ok;
        ok = S.cpy2(R.in2, es, R.mixedq + size_t(swa ? 0 : 2) * es, 3 * es, es, T) && ok;
    } else {
        ok = cpy(R.in0, R.inpSA, size_t(T) * es) && ok;
        ok = cpy(R.in1, R.inpSA, size_t(T) * es) && ok;
        ok = cpy(R.in2, R.inpSA, size_t(T) * es) && ok;
    }
    if (msgs_) {
        ok = S.control_exact_rows(W, R.in0, es, R.msgs, 4 * es, 0, T) && ok;
        ok = S.control_exact_rows(W, R.in1, es, R.msgs + es, 4 * es, 1, T) && ok;
        ok = S.control_exact_rows(W, R.in2, es, R.msgs + 2 * es, 4 * es, 2, T) && ok;
    }
    if (swa) {
        ok = S.rms_rows(R.in0, es, W.attn_norm, R.in0, es, T) && ok;
        if (msgs_) {
            ok = S.rms_rows(R.in1, es, W.attn_norm, R.in1, es, T) && ok;
            ok = S.rms_rows(R.in2, es, W.attn_norm, R.in2, es, T) && ok;
        } else {
            ok = cpy(R.in1, R.in0, size_t(T) * es) && ok;
            ok = cpy(R.in2, R.in0, size_t(T) * es) && ok;
        }
    } else {
        ok = S.rms_rows(R.in0, es, W.q_inp_norm, R.in0, es, T) && ok;
        ok = S.rms_rows(R.in1, es, W.k_inp_norm, R.in1, es, T) && ok;
        ok = S.rms_rows(R.in2, es, W.v_inp_norm, R.in2, es, T) && ok;
    }

    // The projections, once for all rows (gemv_cols: gemv()'s bits per column).
    ok = S.gemvw_rows(W.wq, R.in0, R.q, int(qw), e, T) && ok;
    ok = S.gemvw_rows(W.wk, R.in1, R.kraw, kvw, e, T) && ok;
    ok = S.gemvw_rows(W.wv, R.in2, R.vraw, kvw, e, T) && ok;
    ok = S.gemvw_rows(W.wr, R.in0, R.rv, int(rw), e, T) && ok;
    if (!swa) {
        ok = S.gemvw_rows(W.k_conv_proj, R.in1, R.kdyn, int(dw), e, T) && ok;
        ok = S.gemvw_rows(W.v_conv_proj, R.in2, R.vdyn, int(dw), e, T) && ok;
    }
    ok = S.gemvw_rows(W.sink, R.in0, R.sinkv, nh, e, T) && ok;
    ok = S.gemvw_rows(W.wgate, R.in0, R.gate, int(qw), e, T) && ok;

    if (S.capture) {   // the verify window's rollback: every row's conv inputs
        ok = S.rec_conv(il, 0, R.kraw, kw, kvw, T) && S.rec_conv(il, kvw, R.vraw, kw, kvw, T) && ok;
    }
    // The short convolutions: one FIR launch per input over the rows, each output the single-row sum.
    ok = ink::conv_rows(S.sconv, R.kraw, kw, swa ? nullptr : R.kdyn, dw, W.k_conv, R.k, kw, hist, 0, kvw, nkv, S.nch,
                        T, s) &&
         ok;
    ok = ink::conv_rows(S.sconv, R.vraw, kw, swa ? nullptr : R.vdyn, dw, W.v_conv, R.v, kw, hist, kvw, kvw, nkv,
                        S.nch, T, s) &&
         ok;
    ok = ink::clip(R.q, W.q_clip, nh, hd, S.qk_clip, T, s) && ok;
    ok = ink::clip(R.k, nullptr, nkv, hd, S.qk_clip, T, s) && ok;
    ok = ink::clip(R.rv, W.rel_clip, nh, rel_len, S.rel_clip, T, s) && ok;
    headnorm_kernel<<<unsigned(T * nkv), 128, 0, s>>>(R.v, R.v, hd, S.rms_eps);
    ok = cudaGetLastError() == cudaSuccess && ok;
    if (!swa) {
        mul_tau_rows_kernel<<<dim3(unsigned((qw + 255) / 256), unsigned(T)), 256, 0, s>>>(R.q, S.d_tau, R.d_pos,
                                                                                         int(qw));
        mul_tau_rows_kernel<<<dim3(unsigned((rw + 255) / 256), unsigned(T)), 256, 0, s>>>(R.rv, S.d_tau, R.d_pos,
                                                                                         int(rw));
        ok = cudaGetLastError() == cudaSuccess && ok;
    }

    // The bias tables of every row, all K/V rows, then the attention of all rows in one launch (each row sees exactly
    // the positions up to its own; the ring holds the window plus the kRows - 1 rows stored ahead).
    const int extent = swa ? S.window : S.rel_extent;
    const size_t tstride = size_t(S.tbl_extent + 1) * nh;
    if (S.exact_bias) {
        bias_table_kernel<<<dim3(unsigned(((extent + 1) * nh + 255) / 256), unsigned(T)), 256, 0, s>>>(
            W.relb32, rel_len, R.rv, R.sinkv, R.tables, extent, nh, tstride);
        ok = cudaGetLastError() == cudaSuccess && ok;
    } else {
        for (int t = 0; t < T; ++t) {
            ok = gemv_experts(W.rel_bias.type, W.rel_bias.w, 0, S.head_ids, nh, R.rv + size_t(t) * rw, rel_len,
                              S.tblsc, extent, rel_len, s) &&
                 ok;
            bias_fold_kernel<<<static_cast<unsigned>(((extent + 1) * nh + 255) / 256), 256, 0, s>>>(
                S.tblsc, R.sinkv + size_t(t) * nh, R.tables + size_t(t) * tstride, extent, nh);
        }
    }
    const int cap = S.slot_cap[size_t(il)];
    const int ring = cap < S.n_ctx ? cap : 0;
    __half* kc = S.kc_of(il);
    __half* vc = S.vc_of(il);
    ok = S.store_rows(il, R.k, R.v, R.d_pos, T) && ok;
    ok = (S.attn_rows ? ink::attention_rows(R.q, kc, vc, R.tables, tstride, R.attnv, S.attn_scratch, R.d_pos, T,
                                            S.n_ctx, ring, nh, nkv, hd, S.attn_scale, extent, swa ? S.window : 0, s, -1,
                                            S.ks_of(il), S.vs_of(il))
                      : ink::attention(R.q, kc, vc, R.tables, R.attnv, S.attn_scratch, R.d_pos, T, S.n_ctx, ring, nh,
                                       nkv, hd, S.attn_scale, extent, swa ? S.window : 0, s, tstride, S.ks_of(il),
                                       S.vs_of(il))) &&
         ok;

    // The output gate and the output projection, the post convolution, the bank.
    sigmoid_gate_kernel<<<static_cast<unsigned>((T * int(qw) + 255) / 256), 256, 0, s>>>(R.attnv, R.gate, T * int(qw));
    ok = cudaGetLastError() == cudaSuccess && ok;
    ok = S.gemvw_rows(W.wo, R.attnv, R.attn_out, e, int(qw), T) && ok;
    if (S.capture) ok = S.rec_conv(il, 2 * kvw, R.attn_out, es, e, T) && ok;
    ok = ink::conv_rows(S.sconv, R.attn_out, es, nullptr, 0, W.attn_post_conv, R.attn_conv, es, hist, 2 * kvw, e, 1,
                        S.nch, T, s) &&
         ok;
    const bool new_block = ip % S.res_block == 0;
    if (new_block) {
        ok = S.cpy2(R.blocks + size_t(nb) * es, bs, R.inpSA, es, es, T) && ok;
        ok = cpy(R.cur, R.attn_conv, size_t(T) * es) && ok;
        ++nb;
    } else {
        add_kernel<<<static_cast<unsigned>((T * e + 255) / 256), 256, 0, s>>>(R.cur, R.inpSA, R.attn_conv, T * e);
        ok = cudaGetLastError() == cudaSuccess && ok;
    }

    // The FFN half: the router per row, every row's experts in one grouped launch each.
    ok = cpy(R.ffn_inp, R.cur, size_t(T) * es) && ok;
    ok = S.cpy2(R.blocks + size_t(nb) * es, bs, R.cur, es, es, T) && ok;
    ok = ink::mix_rows(R.blocks, bs, W.res_q_ffn, R.mixed, es, nb + 1, e, 1, T, s) && ok;
    if (msgs_) ok = S.control_exact_rows(W, R.mixed, es, R.msgs + 3 * es, 4 * es, 3, T) && ok;
    ok = S.rms_rows(R.mixed, es, W.ffn_norm, R.mixed, es, T) && ok;
    ok = S.mm_exact_rows(W.router, R.mixed, R.logits, S.n_expert, e, T) && ok;
    ok = ink::route_rows(R.logits, W.exp_bias, S.n_expert, S.n_expert_used, S.route_scale, R.ids, R.moe_w, T, s) && ok;
    ok = ink::moe_gate_up_situ_rows(W.gate_up.w, W.gate_up.type, W.gu_expert_bytes, R.ids, S.n_expert_used, R.mixed, T,
                                    W.gu_scale, S.n_ff_exp, e, S.situ_beta, S.situ_lin_beta, R.moe_h, R.moe_raw, s) &&
         ok;
    ok = ink::moe_down_accum_rows(W.down.w, W.down.type, W.dn_expert_bytes, R.ids, S.n_expert_used, T, R.moe_h,
                                  W.dn_scale, R.moe_w, S.n_ff_exp, e, R.moe_out, R.moe_raw, s) &&
         ok;
    if (S.capture) ok = S.rec_conv(il, 2 * kvw + e, R.moe_out, es, e, T) && ok;
    ok = ink::conv_rows(S.sconv, R.moe_out, es, nullptr, 0, W.ffn_post_conv, R.ffn_conv, es, hist, 2 * kvw + e, e, 1,
                        S.nch, T, s) &&
         ok;
    add_kernel<<<static_cast<unsigned>((T * e + 255) / 256), 256, 0, s>>>(R.cur, R.ffn_inp, R.ffn_conv, T * e);
    ok = cudaGetLastError() == cudaSuccess && ok;
    ok = cpy(S.l_out + size_t(il) * es, R.cur + size_t(T - 1) * es, es) && ok;
    return ok;
}

// step() over T consecutive tokens: the same pass structure, every per-token vector a row of R.
bool InkGpu::St::step_rows(const int32_t* tokens, int T, int pos0, bool head, int head_mode) {
    if (T < 1 || T > InkGpu::kRows || pos0 != next_pos || pos0 < 0 || pos0 + T > n_ctx) return false;
    bool ok = upload_rows(tokens, T, pos0);
    // One capture per (rows, head): the pass reads its tokens and positions from R.d_tok / R.d_pos.
    const int hm = head_mode >= 0 ? head_mode : (head ? 1 : 0);
    cudaGraphExec_t& g = graph_rows[T - 1][hm];
    if (use_graph && !g) {
        cudaGraph_t cg = nullptr;
        bool cap = cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed) == cudaSuccess;
        const bool body = cap && rows_body(T, hm);
        cap = cudaStreamEndCapture(stream, &cg) == cudaSuccess && cap && body &&
              cudaGraphInstantiate(&g, cg, 0) == cudaSuccess;
        if (cg) cudaGraphDestroy(cg);
        cudaGetLastError();
        if (!cap) g = nullptr;
    }
    ok = (g ? cudaGraphLaunch(g, stream) == cudaSuccess : rows_body(T, hm)) && ok;
    if (hm == 1) head_ran = true;
    next_pos = pos0 + T;
    return ok;
}

// The body of step_rows (what its graphs hold).
bool InkGpu::St::rows_body(int T, int head) {
    const int pos0 = 0;   // unused by the kernels (R.d_pos)
    const size_t es = size_t(n_embd), bs = size_t(max_blocks + 1) * es;
    const auto row = [](float* base, size_t n, int t) { return base + size_t(t) * n; };
    const auto cpy = [&](float* dst, const float* src, size_t n) {
        return cudaMemcpyAsync(dst, src, n * sizeof(float), cudaMemcpyDeviceToDevice, stream) == cudaSuccess;
    };
    bool ok = true;
    for (int t = 0; t < T; ++t) {
        ok = embed_row(tok_embd.type, tok_embd.w, R.d_tok + t, row(R.cur, es, t), n_embd, stream) && ok;
        ok = rms1(row(R.cur, es, t), tok_norm, row(R.cur, es, t), n_embd) && ok;
    }

    int nb = 0;
    for (int il = 0; il < loop_start; ++il) ok = tap(il, R.cur, T) && visit_rows(il, il, pos0, T, false, nb) && ok;
    const int prelude_nb = nb;
    ok = cpy(R.prelude, R.cur, size_t(T) * es) && ok;

    const int core = loop_end - loop_start;
    for (int pass = 0; pass < loop_count; ++pass) {
        int bank_n = 0;
        if (pass > 0) {
            bank_n = std::max(0, nb - (prelude_nb + 1));
            for (int t = 0; t < T; ++t)
                if (bank_n > 0)
                    ok = cpy(row(R.bank, bs, t), row(R.blocks, bs, t) + size_t(prelude_nb + 1) * es,
                             size_t(bank_n) * es) &&
                         ok;
            ok = cpy(R.bank_out, R.cur, size_t(T) * es) && ok;
            ok = cpy(R.cur, R.prelude, size_t(T) * es) && ok;
            nb = prelude_nb;
        }
        for (int ip = loop_start; ip < loop_end; ++ip) {
            const int il = pass == 0 ? ip : n_layer + (pass - 1) * core + (ip - loop_start);
            if (pass == 0) {
                ok = tap(il, R.cur, T) && visit_rows(il, ip, pos0, T, false, nb) && ok;
                continue;
            }
            for (int t = 0; t < T; ++t) {
                float* bank_t = row(R.bank, bs, t);
                ok = cpy(bank_t + size_t(bank_n) * es, row(R.bank_out, es, t), es) && ok;
                ok = ink::mix(bank_t, L[static_cast<size_t>(ip)].msg_query, row(R.msgs, 4 * es, t), bank_n + 1, n_embd,
                              4, stream) &&
                     ok;
            }
            ok = tap(il, R.cur, T) && visit_rows(il, ip, pos0, T, true, nb) && ok;
        }
    }
    for (int il = loop_end; il < n_layer; ++il) ok = tap(il, R.cur, T) && visit_rows(il, il, pos0, T, false, nb) && ok;

    if (capture || head == 2) {
        // every row's final norm: the draft model's fourth feature and, for a verify window, every row's head
        ok = cpy2(R.blocks + size_t(nb) * es, bs, R.cur, es, es, T) && ok;
        ok = ink::mix_rows(R.blocks, bs, out_res_query, R.in1, es, nb + 1, n_embd, 1, T, stream) && ok;
        ok = rms_rows(R.in1, es, out_norm, R.in0, es, T) && ok;
        if (capture) ok = cpy2(feats + 3 * es, 4 * es, R.in0, es, es, T) && ok;
        if (head == 2) {
            ok = gemvw_rows(output, R.in0, logits_rows, n_vocab, n_embd, T) && ok;
            if (softcap > 0.0f) {
                softcap_kernel<<<static_cast<unsigned>((T * n_vocab + 255) / 256), 256, 0, stream>>>(
                    logits_rows, T * n_vocab, softcap, 1.0f / softcap);
                ok = cudaGetLastError() == cudaSuccess && ok;
            }
            argmax_kernel<<<T, 256, 0, stream>>>(logits_rows, n_vocab, argmax_rows);
            ok = cudaGetLastError() == cudaSuccess && ok;
        }
    }
    if (head == 1) {
        float* blocks = row(R.blocks, bs, T - 1);
        ok = cpy(blocks + size_t(nb) * es, row(R.cur, es, T - 1), es) && ok;
        ok = ink::mix(blocks, out_res_query, mixed, nb + 1, n_embd, 1, stream) && ok;
        ok = rms1(mixed, out_norm, rn, n_embd) && ok;
        ok = gemvw(output, rn, logits, n_vocab, n_embd) && ok;
        if (softcap > 0.0f) {
            softcap_kernel<<<static_cast<unsigned>((n_vocab + 255) / 256), 256, 0, stream>>>(logits, n_vocab, softcap,
                                                                                             1.0f / softcap);
            ok = cudaGetLastError() == cudaSuccess && ok;
        }
    }
    return ok;
}

// The tokens and positions of a piece into R.d_tok / R.d_pos, through the pinned staging buffer (written only once
// its last upload has run).
bool InkGpu::St::upload_rows(const int32_t* tokens, int T, int pos0) {
    if (cudaEventSynchronize(copied) != cudaSuccess) return false;
    for (int t = 0; t < T; ++t) {
        R.pinned[t] = tokens[t];
        R.pinned[InkGpu::kPiece + t] = pos0 + t;
    }
    return cudaMemcpyAsync(R.d_tok, R.pinned, size_t(T) * sizeof(int32_t), cudaMemcpyHostToDevice, stream) ==
               cudaSuccess &&
           cudaMemcpyAsync(R.d_pos, R.pinned + InkGpu::kPiece, size_t(T) * sizeof(int32_t), cudaMemcpyHostToDevice,
                           stream) == cudaSuccess &&
           cudaEventRecord(copied, stream) == cudaSuccess;
}

// visit() over a piece of T tokens, batched: the same operations in the same order per row, the products as int8
// GEMMs over the whole piece, the experts grouped, the stateful parts (convolutions, K/V, attention) in position
// order.
bool InkGpu::St::visit_piece(int il, int ip, int T, bool msgs_, int& nb) {
    St& S = *this;
    const int core = S.loop_end - S.loop_start;
    const int phys = il < S.n_layer ? il : S.loop_start + (il - S.n_layer) % core;
    const InkGpu::St::InkLayer& W = S.L[static_cast<size_t>(phys)];
    const bool swa = W.swa;
    const int e = S.n_embd;
    const size_t es = size_t(e), bs = size_t(S.max_blocks + 1) * es;
    cudaStream_t s = S.stream;
    const int nh = S.n_head, nkv = S.n_head_kv, hd = S.head_dim, kvw = S.kvw, rel_len = S.rel_length;
    const size_t qw = size_t(nh) * hd, kw = size_t(kvw), rw = size_t(nh) * rel_len, dw = size_t(S.sconv) * nkv;
    float* hist = S.conv_hist + size_t(il) * (S.sconv - 1) * size_t(S.nch);
    const auto cpy = [&](float* dst, const float* src, size_t n) {
        return cudaMemcpyAsync(dst, src, n * sizeof(float), cudaMemcpyDeviceToDevice, s) == cudaSuccess;
    };
    bool ok = cpy(R.inpSA, R.cur, size_t(T) * es);

    // the attention's inputs
    if (W.res_q_attn) {
        const int qn = swa ? 1 : 3;
        ok = S.cpy2(R.blocks + size_t(nb) * es, bs, R.cur, es, es, T) && ok;
        ok = ink::mix_rows(R.blocks, bs, W.res_q_attn, R.mixedq, 3 * es, nb + 1, e, qn, T, s) && ok;
        ok = S.cpy2(R.in0, es, R.mixedq, 3 * es, es, T) && ok;
        ok = S.cpy2(R.in1, es, R.mixedq + size_t(swa ? 0 : 1) * es, 3 * es, es, T) && ok;
        ok = S.cpy2(R.in2, es, R.mixedq + size_t(swa ? 0 : 2) * es, 3 * es, es, T) && ok;
    } else {
        ok = cpy(R.in0, R.inpSA, size_t(T) * es) && ok;
        ok = cpy(R.in1, R.inpSA, size_t(T) * es) && ok;
        ok = cpy(R.in2, R.inpSA, size_t(T) * es) && ok;
    }
    if (msgs_) {
        ok = S.control_rows(W, R.in0, es, R.msgs, 4 * es, 0, T) && ok;
        ok = S.control_rows(W, R.in1, es, R.msgs + es, 4 * es, 1, T) && ok;
        ok = S.control_rows(W, R.in2, es, R.msgs + 2 * es, 4 * es, 2, T) && ok;
    }
    if (swa) {
        ok = S.rms_rows(R.in0, es, W.attn_norm, R.in0, es, T) && ok;
        if (msgs_) {
            ok = S.rms_rows(R.in1, es, W.attn_norm, R.in1, es, T) && ok;
            ok = S.rms_rows(R.in2, es, W.attn_norm, R.in2, es, T) && ok;
        } else {
            ok = cpy(R.in1, R.in0, size_t(T) * es) && ok;
            ok = cpy(R.in2, R.in0, size_t(T) * es) && ok;
        }
    } else {
        ok = S.rms_rows(R.in0, es, W.q_inp_norm, R.in0, es, T) && ok;
        ok = S.rms_rows(R.in1, es, W.k_inp_norm, R.in1, es, T) && ok;
        ok = S.rms_rows(R.in2, es, W.v_inp_norm, R.in2, es, T) && ok;
    }

    // the projections
    ok = S.quant(R.in0, T, e, 0) && S.quant(R.in1, T, e, 1) && S.quant(R.in2, T, e, 2) && ok;
    ok = S.mm_rows(W.wq, R.in0, 0, R.q, int(qw), e, T) && ok;
    ok = S.mm_rows(W.wk, R.in1, 1, R.kraw, kvw, e, T) && ok;
    ok = S.mm_rows(W.wv, R.in2, 2, R.vraw, kvw, e, T) && ok;
    ok = S.mm_rows(W.wr, R.in0, 0, R.rv, int(rw), e, T) && ok;
    if (!swa) {
        ok = S.mm_rows(W.k_conv_proj, R.in1, 1, R.kdyn, int(dw), e, T) && ok;
        ok = S.mm_rows(W.v_conv_proj, R.in2, 2, R.vdyn, int(dw), e, T) && ok;
    }
    ok = S.mm_rows(W.sink, R.in0, 0, R.sinkv, nh, e, T) && ok;
    ok = S.mm_rows(W.wgate, R.in0, 0, R.gate, int(qw), e, T) && ok;

    ok = ink::conv_rows(S.sconv, R.kraw, kw, swa ? nullptr : R.kdyn, dw, W.k_conv, R.k, kw, hist, 0, kvw, nkv, S.nch,
                        T, s) &&
         ok;
    ok = ink::conv_rows(S.sconv, R.vraw, kw, swa ? nullptr : R.vdyn, dw, W.v_conv, R.v, kw, hist, kvw, kvw, nkv,
                        S.nch, T, s) &&
         ok;
    ok = ink::clip(R.q, W.q_clip, nh, hd, S.qk_clip, T, s) && ok;
    ok = ink::clip(R.k, nullptr, nkv, hd, S.qk_clip, T, s) && ok;
    ok = ink::clip(R.rv, W.rel_clip, nh, rel_len, S.rel_clip, T, s) && ok;
    // the per-head V norm: every (row, KV head) is one head of kvw / nkv contiguous values
    headnorm_kernel<<<unsigned(T * nkv), 128, 0, s>>>(R.v, R.v, hd, S.rms_eps);
    ok = cudaGetLastError() == cudaSuccess && ok;
    if (!swa) {
        mul_tau_rows_kernel<<<dim3(unsigned((qw + 255) / 256), unsigned(T)), 256, 0, s>>>(R.q, S.d_tau, R.d_pos,
                                                                                         int(qw));
        mul_tau_rows_kernel<<<dim3(unsigned((rw + 255) / 256), unsigned(T)), 256, 0, s>>>(R.rv, S.d_tau, R.d_pos,
                                                                                         int(rw));
        ok = cudaGetLastError() == cudaSuccess && ok;
    }

    // the bias tables of every row, then the K/V and the attention in groups of kRows (the sliding ring holds the
    // window plus kRows - 1 rows stored ahead)
    const int extent = swa ? S.window : S.rel_extent;
    const size_t tstride = size_t(S.tbl_extent + 1) * nh;
    bias_rows_kernel<<<dim3(unsigned((extent + 1 + kBiasD - 1) / kBiasD), unsigned(T)), 256, 0, s>>>(
        W.relb32, rel_len, R.rv, R.sinkv, P.tables, tstride, extent, nh, R.d_pos);
    ok = cudaGetLastError() == cudaSuccess && ok;
    const int cap = S.slot_cap[size_t(il)];
    const int ring = cap < S.n_ctx ? cap : 0;
    __half* kc = S.kc_of(il);
    __half* vc = S.vc_of(il);
    for (int t0 = 0; t0 < T; t0 += InkGpu::kRows) {
        const int g = std::min(InkGpu::kRows, T - t0);
        ok = S.store_rows(il, R.k + size_t(t0) * kw, R.v + size_t(t0) * kw, R.d_pos + t0, g) && ok;
        ok = ink::attention_rows(R.q + size_t(t0) * qw, kc, vc, P.tables + size_t(t0) * tstride, tstride,
                                 R.attnv + size_t(t0) * qw, S.attn_scratch, R.d_pos + t0, g, S.n_ctx, ring, nh, nkv, hd,
                                 S.attn_scale, extent, swa ? S.window : 0, s, S.piece_pos0 + t0 + g - 1, S.ks_of(il),
                                 S.vs_of(il)) &&
             ok;
    }

    // the output gate and projection, the post convolution, the bank
    sigmoid_gate_kernel<<<static_cast<unsigned>((T * int(qw) + 255) / 256), 256, 0, s>>>(R.attnv, R.gate, T * int(qw));
    ok = cudaGetLastError() == cudaSuccess && ok;
    ok = S.quant(R.attnv, T, int(qw), 0) && ok;
    ok = S.mm_rows(W.wo, R.attnv, 0, R.attn_out, e, int(qw), T) && ok;
    ok = ink::conv_rows(S.sconv, R.attn_out, es, nullptr, 0, W.attn_post_conv, R.attn_conv, es, hist, 2 * kvw, e, 1,
                        S.nch, T, s) &&
         ok;
    const bool new_block = ip % S.res_block == 0;
    if (new_block) {
        ok = S.cpy2(R.blocks + size_t(nb) * es, bs, R.inpSA, es, es, T) && ok;
        ok = cpy(R.cur, R.attn_conv, size_t(T) * es) && ok;
        ++nb;
    } else {
        add_kernel<<<static_cast<unsigned>((T * e + 255) / 256), 256, 0, s>>>(R.cur, R.inpSA, R.attn_conv, T * e);
        ok = cudaGetLastError() == cudaSuccess && ok;
    }

    // the FFN half: the depth mix, the controller, the norm, the router, the grouped experts
    ok = cpy(R.ffn_inp, R.cur, size_t(T) * es) && ok;
    ok = S.cpy2(R.blocks + size_t(nb) * es, bs, R.cur, es, es, T) && ok;
    ok = ink::mix_rows(R.blocks, bs, W.res_q_ffn, R.mixed, es, nb + 1, e, 1, T, s) && ok;
    if (msgs_) ok = S.control_rows(W, R.mixed, es, R.msgs + 3 * es, 4 * es, 3, T) && ok;
    ok = S.rms_rows(R.mixed, es, W.ffn_norm, R.mixed, es, T) && ok;
    ok = S.mm_rows(W.router, R.mixed, -1, P.logits, S.n_expert, e, T) && ok;
    ok = ink::route_rows(P.logits, W.exp_bias, S.n_expert, S.n_expert_used, S.route_scale, P.ids, P.w, T, s) && ok;
    const int pairs = T * S.n_expert_used, nff = S.n_ff_exp;
    ok = moe_build(P.ids, T, S.n_expert, S.n_expert_used, P.lists, s) && ok;
    ok = S.quant(R.mixed, T, e, 0) && ok;
    ok = moe_rows(W.gate_up.type, W.gate_up.w, W.gu_expert_bytes, P.xq[0], P.xs[0], S.planes, T, pairs, P.lists,
                  S.n_expert, P.raw, 2 * nff, e, s) &&
         ok;
    ok = ink::situ_pairs(P.raw, P.ids, P.lists.pos_of, P.pe, W.gu_scale, pairs, nff, S.situ_beta, S.situ_lin_beta, P.h,
                         s) &&
         ok;
    ok = act_quant(P.h, pairs, nff, S.planes, P.hq, P.hs, s) && ok;
    ok = moe_down(W.down.type, W.down.w, W.dn_expert_bytes, P.hq, P.hs, S.planes, pairs, P.lists, S.n_expert, P.dn, e,
                  nff, s) &&
         ok;
    ok = ink::combine_rows(P.dn, P.lists.pos_of, P.ids, W.dn_scale, P.w, S.n_expert_used, e, T, R.moe_out, s) && ok;
    ok = ink::conv_rows(S.sconv, R.moe_out, es, nullptr, 0, W.ffn_post_conv, R.ffn_conv, es, hist, 2 * kvw + e, e, 1,
                        S.nch, T, s) &&
         ok;
    add_kernel<<<static_cast<unsigned>((T * e + 255) / 256), 256, 0, s>>>(R.cur, R.ffn_inp, R.ffn_conv, T * e);
    ok = cudaGetLastError() == cudaSuccess && ok;
    ok = cpy(S.l_out + size_t(il) * es, R.cur + size_t(T - 1) * es, es) && ok;
    return ok;
}

bool InkGpu::St::prefill_rows(const int32_t* tokens, int T, int pos0, bool head) {
    if (T < 1 || T > InkGpu::kPiece || pos0 != next_pos || pos0 < 0 || pos0 + T > n_ctx) return false;
    const size_t es = size_t(n_embd), bs = size_t(max_blocks + 1) * es;
    const auto cpy = [&](float* dst, const float* src, size_t n) {
        return cudaMemcpyAsync(dst, src, n * sizeof(float), cudaMemcpyDeviceToDevice, stream) == cudaSuccess;
    };
    bool ok = upload_rows(tokens, T, pos0);
    piece_pos0 = pos0;
    ok = embed_rows(tok_embd.type, tok_embd.w, R.d_tok, T, R.cur, n_embd, stream) && ok;
    ok = rms_rows(R.cur, es, tok_norm, R.cur, es, T) && ok;

    int nb = 0;
    for (int il = 0; il < loop_start; ++il) ok = tap(il, R.cur, T) && visit_piece(il, il, T, false, nb) && ok;
    const int prelude_nb = nb;
    ok = cpy(R.prelude, R.cur, size_t(T) * es) && ok;

    const int core = loop_end - loop_start;
    for (int pass = 0; pass < loop_count; ++pass) {
        int bank_n = 0;
        if (pass > 0) {
            bank_n = std::max(0, nb - (prelude_nb + 1));
            if (bank_n > 0) ok = cpy2(R.bank, bs, R.blocks + size_t(prelude_nb + 1) * es, bs, size_t(bank_n) * es, T) && ok;
            ok = cpy(R.bank_out, R.cur, size_t(T) * es) && ok;
            ok = cpy(R.cur, R.prelude, size_t(T) * es) && ok;
            nb = prelude_nb;
            ok = cpy2(R.bank + size_t(bank_n) * es, bs, R.bank_out, es, es, T) && ok;
        }
        for (int ip = loop_start; ip < loop_end; ++ip) {
            const int il = pass == 0 ? ip : n_layer + (pass - 1) * core + (ip - loop_start);
            if (pass == 0) {
                ok = tap(il, R.cur, T) && visit_piece(il, ip, T, false, nb) && ok;
                continue;
            }
            ok = ink::mix_rows(R.bank, bs, L[static_cast<size_t>(ip)].msg_query, R.msgs, 4 * es, bank_n + 1, n_embd, 4,
                               T, stream) &&
                 ok;
            ok = tap(il, R.cur, T) && visit_piece(il, ip, T, true, nb) && ok;
        }
    }
    for (int il = loop_end; il < n_layer; ++il) ok = tap(il, R.cur, T) && visit_piece(il, il, T, false, nb) && ok;

    if (capture) {
        ok = cpy2(R.blocks + size_t(nb) * es, bs, R.cur, es, es, T) && ok;
        ok = ink::mix_rows(R.blocks, bs, out_res_query, R.in1, es, nb + 1, n_embd, 1, T, stream) && ok;
        ok = rms_rows(R.in1, es, out_norm, R.in0, es, T) && ok;
        ok = cpy2(feats + 3 * es, 4 * es, R.in0, es, es, T) && ok;
    }
    if (head) {
        float* blocks = R.blocks + size_t(T - 1) * bs;
        ok = cpy(blocks + size_t(nb) * es, R.cur + size_t(T - 1) * es, es) && ok;
        ok = ink::mix(blocks, out_res_query, mixed, nb + 1, n_embd, 1, stream) && ok;
        ok = rms1(mixed, out_norm, rn, n_embd) && ok;
        ok = gemvw(output, rn, logits, n_vocab, n_embd) && ok;
        if (softcap > 0.0f) {
            softcap_kernel<<<static_cast<unsigned>((n_vocab + 255) / 256), 256, 0, stream>>>(logits, n_vocab, softcap,
                                                                                             1.0f / softcap);
            ok = cudaGetLastError() == cudaSuccess && ok;
        }
        head_ran = true;
    }
    next_pos = pos0 + T;
    return ok;
}

// The draft model's taps: visits are indices in the visit sequence (prelude, the loop's passes, coda).
void InkGpu::St::set_capture(bool on, const int* visits) {
    if (stream) cudaStreamSynchronize(stream);
    capture = on;
    const int core = loop_end - loop_start;
    for (int k = 0; k < 3; ++k) {
        const int v = visits ? visits[k] : -1;
        int il = -1;
        if (v >= 0 && v < loop_start) il = v;
        else if (v >= loop_start && v < loop_start + loop_count * core) {
            const int p = (v - loop_start) / core, w = (v - loop_start) % core;
            il = p == 0 ? loop_start + w : n_layer + (p - 1) * core + w;
        } else if (v >= loop_start + loop_count * core && v < n_layer + (loop_count - 1) * core)
            il = loop_end + (v - loop_start - loop_count * core);
        tap_il[k] = il;
    }
    // the captured passes change: every graph is captured again
    for (cudaGraphExec_t& g : graph)
        if (g) { cudaGraphExecDestroy(g); g = nullptr; }
    for (auto& gr : graph_rows)
        for (cudaGraphExec_t& g : gr)
            if (g) { cudaGraphExecDestroy(g); g = nullptr; }
}

bool InkGpu::St::verify(const int32_t* tokens, int T, int pos0, int32_t* out, float* logits_out) {
    if (!capture || T < 1 || T > InkGpu::kRows) return false;
    bool ok = cudaMemcpyAsync(hist_snap, conv_hist, size_t(n_slot) * size_t(sconv - 1) * size_t(nch) * sizeof(float),
                              cudaMemcpyDeviceToDevice, stream) == cudaSuccess;
    ok = step_rows(tokens, T, pos0, false, 2) && ok;
    ok = ok && cudaStreamSynchronize(stream) == cudaSuccess &&
         cudaMemcpy(out, argmax_rows, size_t(T) * sizeof(int32_t), cudaMemcpyDeviceToHost) == cudaSuccess &&
         (!logits_out || cudaMemcpy(logits_out, logits_rows, size_t(T) * size_t(n_vocab) * sizeof(float),
                                    cudaMemcpyDeviceToHost) == cudaSuccess);
    verify_pos = ok ? pos0 : -1;
    verify_rows = ok ? T : 0;
    return ok;
}

bool InkGpu::St::commit(int n) {
    if (verify_pos < 0 || n < 1 || n > verify_rows) return false;
    if (n < verify_rows) {
        conv_rollback_kernel<<<dim3(unsigned((nch + 255) / 256), unsigned(n_slot)), 256, 0, stream>>>(
            conv_hist, hist_snap, conv_rec, nch, sconv, InkGpu::kRows, n);
        if (cudaGetLastError() != cudaSuccess) return false;
    }
    next_pos = verify_pos + n;
    verify_pos = -1;
    return true;
}

// ---- the class ------------------------------------------------------------------------------------------
InkGpu::InkGpu() = default;
InkGpu::~InkGpu() = default;

std::unique_ptr<InkGpu> InkGpu::open(const model::Model& m, int n_ctx, uint64_t min_free_bytes, std::string* err,
                                     uint64_t reserve_bytes, bool kv_q8) {
    uint64_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
        *err = "InkGpu::open: cudaMemGetInfo failed";
        return nullptr;
    }
    if (free_b < min_free_bytes) {
        *err = "InkGpu::open: " + std::to_string(free_b) + " bytes of VRAM free, " + std::to_string(min_free_bytes) +
               " required (the card is shared with the desktop)";
        return nullptr;
    }
    std::unique_ptr<InkGpu> g(new InkGpu());
    g->st_.reset(new St());
    g->st_->kv_q8 = kv_q8;
    if (!g->st_->init(m, n_ctx, err, reserve_bytes)) return nullptr;
    return g;
}

int InkGpu::n_vocab() const { return st_->n_vocab; }
int InkGpu::n_ctx() const { return st_->n_ctx; }
int InkGpu::n_slots() const { return st_->n_slot; }

bool InkGpu::step(int32_t token, int pos, bool head) { return st_->step(token, pos, head); }
bool InkGpu::step_rows(const int32_t* tokens, int T, int pos0, bool head) {
    return st_->step_rows(tokens, T, pos0, head);
}
bool InkGpu::prefill_rows(const int32_t* tokens, int T, int pos0, bool head) {
    return st_->prefill_rows(tokens, T, pos0, head);
}
void InkGpu::set_capture(bool on, const int* visits) { st_->set_capture(on, visits); }
const float* InkGpu::features() const { return st_->feats; }
bool InkGpu::verify(const int32_t* tokens, int T, int pos0, int32_t* out, float* logits_out) {
    return st_->verify(tokens, T, pos0, out, logits_out);
}
bool InkGpu::commit(int n) { return st_->commit(n); }
int InkGpu::next_pos() const { return st_->next_pos; }
InkGpu::Shared InkGpu::shared() const {
    const St& S = *st_;
    return Shared{S.tok_embd.w, S.tok_embd.type, S.output.w, S.output.type, S.output.scale, S.n_vocab, S.n_embd,
                  S.n_ctx, S.stream};
}

int32_t InkGpu::argmax() {
    St& S = *st_;
    if (!S.head_ran) return -1;
    argmax_kernel<<<1, 256, 0, S.stream>>>(S.logits, S.n_vocab, S.d_argmax);
    if (cudaGetLastError() != cudaSuccess || cudaStreamSynchronize(S.stream) != cudaSuccess) return -1;
    int32_t out = -1;
    if (cudaMemcpy(&out, S.d_argmax, sizeof(int32_t), cudaMemcpyDeviceToHost) != cudaSuccess) return -1;
    return out;
}

bool InkGpu::read_logits(float* out) {
    St& S = *st_;
    if (!out || !S.head_ran) return false;
    if (cudaStreamSynchronize(S.stream) != cudaSuccess) return false;
    return cudaMemcpy(out, S.logits, size_t(S.n_vocab) * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
}

bool InkGpu::read_l_out(int slot, float* out) {
    St& S = *st_;
    if (!out || slot < 0 || slot >= S.n_slot) return false;
    if (cudaStreamSynchronize(S.stream) != cudaSuccess) return false;
    return cudaMemcpy(out, S.l_out + size_t(slot) * size_t(S.n_embd), size_t(S.n_embd) * sizeof(float),
                      cudaMemcpyDeviceToHost) == cudaSuccess;
}

void InkGpu::sync() { cudaStreamSynchronize(st_->stream); }
void InkGpu::reset() { st_->reset(); }

}  // namespace limbo::cuda
