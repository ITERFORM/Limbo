// Copyright (c) 2026 ITERFORM. All rights reserved.
#pragma once
// src/cuda/ink_ops.hpp — the inkling2_loop decode kernels (docs/inkling2.md §3), one row at a time or a verify
// window of T rows. Only .cu translation units include this header.
//
// The attention is its own kernel (docs/inkling2.md §6: not a variation of dec::attn — it carries a per-key
// relative-bias gather and a per-head attention sink in the denominator). Its arithmetic is the reference's
// (src/step.cpp ink_visit): for query row t of head h at position q_pos + t, over the visible keys j:
//
//   logit_j = scale * q.k_j + table[min(q_pos + t - kv_pos_j, extent)][h]   (the table has the sink subtracted)
//   p       = softmax over the visible keys with a sink column at logit 0 in the denominator
//   out     = sum_j p_j * v_j
//
// The split flash-decode structure is the decode attention's (src/cuda/attn_decode.cu): fixed 256-position
// chunks, one block per (KV head, chunk, row) scoring the chunk for every query head that shares the KV head,
// a combine kernel merging the chunks in position order. Every reduction tree and the merge order are fixed, so
// a row's result does not depend on how many rows share the launch.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace limbo::cuda::ink {

constexpr int kChunk = 256;   // positions per flash-decode chunk (a chunk boundary is a fixed absolute position)
constexpr int kMaxRows = 8;   // the widest verify window

// Scratch floats for up to max_rows rows at max_ctx positions: per (head, chunk, row) the chunk's
// (max, sum, numerator[hd]) partials, plus the merge's own lane state.
size_t attention_scratch_floats(int max_ctx, int max_rows, int n_head, int hd);

// The split flash-decode. K and V are the f16 cache of a contiguous slot: K [pos][kv head * hd], V the same.
// `table` is [extent + 1][n_head] f32 (row extent is the zero pad beyond the bias's reach), each entry already
// carrying -sink[h]; `scale` multiplies q.k only. Window > 0: each query attends the last `window` positions
// (its own included); window 0: everything up to it. out is [T][n_head * hd]; `scratch` holds
// attention_scratch_floats() floats for max_ctx / max_rows. n_head / n_head_kv must be at most 4, hd 128.
// `ring` > 0: the slot is a ring of `ring` rows and position p lives at row p % ring (a sliding layer keeps
// only its window, docs/inkling2.md §7); 0: row p is position p. With a window the chunk grid covers only the
// window's chunks -- still aligned to absolute kChunk boundaries, so the partials and their merge order are
// the ones the full grid produces.
bool attention(const float* q, const __half* kcache, const __half* vcache, const float* table, float* out,
               float* scratch, const int* pos0, int T, int max_ctx, int ring, int n_head, int n_head_kv,
               int hd, float scale, int extent, int window, cudaStream_t s, size_t table_stride = 0,
               const float* kscale = nullptr, const float* vscale = nullptr);
// kscale / vscale non-null: the caches are int8 (kcache / vcache point at int8 rows of n_head_kv * hd values) and
// value d of KV head h at cache row r is int8 * scale[r * n_head_kv + h] (InkGpu's q8 K/V).
// The prompt pass's form: the same attention for T <= kMaxRows consecutive rows (q, out [T][n_head * hd]), each
// row with its own table (table + t * tstride), every K/V tile read once for all the rows and heads that share it.
// Not attention()'s bits (the softmax folds per tile of 32 keys); the same scratch.
bool attention_rows(const float* q, const __half* kcache, const __half* vcache, const float* table, size_t tstride,
                    float* out, float* scratch, const int* pos0, int T, int max_ctx, int ring, int n_head,
                    int n_head_kv, int hd, float scale, int extent, int window, cudaStream_t s, int last_pos = -1,
                    const float* kscale = nullptr, const float* vscale = nullptr);

// table_stride > 0: row t reads its own table at table + t * table_stride (the prompt pass: every token has its
// own bias row, it carries the token's sink); 0: every row shares `table`.

// ---- the short convolutions and the L2 clips (docs/inkling2.md §2/§3) ------------------------------------------
//
// The causal short convolution of one visit's slot: x is the history rows (sconv-1 of them, the shared
// n_sconv_channels of the slot at channel `offset`) plus the current input `cur`, the weight {sconv taps,
// n_ch} with the taps contiguous per channel. out = (cur + conv(x)) * 1/sqrt((dyn ? 2 : 1)*sconv + 1), no
// activation; the state keeps the last sconv-1 rows (shift one row forward). `dyn` (n_sconv taps per head)
// adds input-dependent taps: tap lag of head h weights x row sconv-1-lag over that head's channels. The
// history is row-major: row j at hist + j * n_sconv_channels + offset. T rows share the launch (row t's
// history is its own slice of the state buffer).
bool conv(int sconv, const float* cur, const float* dyn, const float* w, float* out, float* hist, int offset,
          int n_ch, int n_heads, int n_sconv_channels, int T, cudaStream_t s);

// conv() over T consecutive rows in position order (row t's cur / dyn / out at t times cs / ds / os floats): the
// bits of T conv() calls, one launch.
bool conv_rows(int sconv, const float* cur, size_t cs, const float* dyn, size_t ds, const float* w, float* out,
               size_t os, float* hist, int offset, int n_ch, int n_heads, int n_sconv_channels, int T, cudaStream_t s);

// An L2 clip, per head over `row` elements, after the optional per-channel scale: the head's values are scaled
// down to an L2 norm of at most `threshold` (the norm is sqrt of the sum + 1e-6, the factor clamped to [0, 1]).
// v is [T][n_head * row] in place; scale [n_head * row] or null.
bool clip(float* v, const float* scale, int n_head, int row, float threshold, int T, cudaStream_t s);

}  // namespace limbo::cuda::ink
