// Copyright (c) 2026 ITERFORM. All rights reserved.
#pragma once
// limbo/cuda/kv.hpp — the layouts of the full-attention K/V cache. Owner: the decode agent; the prefill agent writes and reads
// the same layouts in its batched attention.
//
// One cache per full-attention layer, `n_ctx` positions, `n_head_kv` KV heads of `hd` = 256 dimensions (rung 1: 2 x 256).
//
// F16 (the default, exact): K is [pos][kv head][256] half; V is stored transposed, [kv head][256][vpitch] half with
//   vpitch = n_ctx rounded up to 64, so a lane that owns a value dimension reads consecutive positions.
//
// Q8 (q8_0 numerics, half the bytes: 1088 B per position and layer against 2048): K and V alike are [pos][kv head][256] int8 with a
//   second plane of scales [pos][kv head][8] half, one scale per block of 32 consecutive dimensions of one position:
//       d = amax / 127 (amax over the block), q = round(x / d) (to nearest even), x ~ q * d, d stored as a half.
//   V is NOT transposed: a position's row is written when the position is, and a block never spans positions, so a token can
//   be quantised alone (a transposed V would need 32 positions to form a block). The planes are separate so the int8 rows
//   are 16-byte aligned (a ggml q8_0 block is 34 bytes); the numerics are q8_0's.
//
// Q4 (q4_0 numerics, 576 B per position and layer): K and V are [pos][kv head][128] bytes, dimension 2j in the low nibble of byte
//   j and 2j+1 in the high one, with Q8's scale plane (one half per block of 32 dimensions):
//       m = the block's value of largest magnitude (sign kept), d = m / -8, q = clamp(round(x / d) + 8, 0, 15), x ~ (q - 8) * d.
//   K is rotated before it is quantised: k' = R k with R = H D / 16 (H the 256-point Walsh-Hadamard matrix, D the fixed signs of
//   kv_rot_sign), an orthogonal map, so q'.k' = q.k when the query is rotated the same way (the attention preps write the
//   rotated query). V is rotated the same way and the attention's output rotated back (R^T = D H / 16), since sum_i p_i R v_i =
//   R sum_i p_i v_i. The rotation spreads a head's few large dimensions over all 256, which is what keeps 4-bit codes near 8-bit.
//
// Positions past the one being decoded are never read, so neither format needs clearing between sequences.
#include <cstddef>
#include <cstdint>

namespace limbo::cuda {

enum class KvFormat : int { F16 = 0, Q8 = 1, Q4 = 2 };

constexpr int kKvBlock = 32;  // dimensions per Q8 / Q4 scale

inline const char* kv_format_name(KvFormat f) { return f == KvFormat::F16 ? "f16" : f == KvFormat::Q8 ? "q8" : "q4"; }

// The bytes of one full-attention layer's planes for n_ctx positions: K values, V values, K scales, V scales (F16: no scales,
// V transposed with vpitch positions per row).
struct KvPlanes {
    uint64_t k = 0, v = 0, ks = 0, vs = 0;
    uint64_t total() const { return k + v + ks + vs; }
};
inline KvPlanes kv_planes(KvFormat f, int n_ctx, int vpitch, int n_head_kv, int hd) {
    const uint64_t rows = static_cast<uint64_t>(n_ctx) * static_cast<uint64_t>(n_head_kv);
    KvPlanes p;
    if (f == KvFormat::F16) {
        p.k = rows * static_cast<uint64_t>(hd) * 2;
        p.v = static_cast<uint64_t>(n_head_kv) * static_cast<uint64_t>(hd) * static_cast<uint64_t>(vpitch) * 2;
        return p;
    }
    p.k = p.v = f == KvFormat::Q8 ? rows * static_cast<uint64_t>(hd) : rows * static_cast<uint64_t>(hd) / 2;
    p.ks = p.vs = rows * static_cast<uint64_t>(hd / kKvBlock) * 2;
    return p;
}

// Bytes of K plus V per position per full-attention layer.
inline uint64_t kv_bytes_per_position(KvFormat f, int n_head_kv, int hd) {
    const uint64_t kvw = static_cast<uint64_t>(n_head_kv) * static_cast<uint64_t>(hd);
    const uint64_t sc = kvw / kKvBlock * 2;
    return f == KvFormat::F16 ? 2 * kvw * 2 : f == KvFormat::Q8 ? 2 * (kvw + sc) : 2 * (kvw / 2 + sc);
}

}  // namespace limbo::cuda

#ifdef __CUDACC__
#include <cuda_fp16.h>

namespace limbo::cuda {

// Quantises one block of 32 dimensions held one per lane of a warp: returns the int8 of this lane's value and writes the block's
// scale to *d_half (the same on every lane). Use with all 32 lanes active.
__device__ __forceinline__ int8_t kv_quant_lane(float x, uint16_t* d_half) {
    float a = fabsf(x);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    const float d = a / 127.0f;
    const float inv = d > 0.f ? 127.0f / a : 0.f;
    *d_half = __half_as_ushort(__float2half_rn(d));
    return static_cast<int8_t>(__float2int_rn(x * inv));  // round to nearest even; the tie case is measure-zero
}

__device__ __forceinline__ float kv_dequant(int8_t q, uint16_t d_half) { return static_cast<float>(q) * __half2float(__ushort_as_half(d_half)); }

// Q4: one block of 32 dimensions held one per lane of a warp becomes this lane's code (0..15); the block's scale goes to *d_half.
// The value of largest magnitude is found with a total order (larger magnitude, then larger value), so every lane agrees on it.
__device__ __forceinline__ int kv_quant4_lane(float x, uint16_t* d_half) {
    float m = x;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float y = __shfl_xor_sync(0xffffffffu, m, o);
        const float ay = fabsf(y), am = fabsf(m);
        if (ay > am || (ay == am && y > m)) m = y;
    }
    const float d = m / -8.0f;
    const float id = d != 0.f ? 1.0f / d : 0.f;
    *d_half = __half_as_ushort(__float2half_rn(d));
    return min(15, max(0, __float2int_rn(x * id) + 8));
}

// Packs the codes of a warp's 32 lanes into 16 bytes: the even lane of a pair returns its byte (low nibble its own code, high
// nibble the odd neighbour's), the odd lane returns -1. Use with all 32 lanes active.
__device__ __forceinline__ int kv_pack4_lane(int q) {
    const int hi = __shfl_down_sync(0xffffffffu, q, 1);
    return (threadIdx.x & 1) ? -1 : (q | (hi << 4));
}

__device__ __forceinline__ float kv_nib(uint32_t w, int j) { return static_cast<float>(static_cast<int>((w >> (4 * j)) & 15u) - 8); }

// The fixed signs D of the Q4 key rotation.
__device__ __forceinline__ float kv_rot_sign(int d) {
    uint32_t h = static_cast<uint32_t>(d) * 2654435761u;
    h ^= h >> 16;
    h *= 2246822519u;
    h ^= h >> 13;
    return (h & 1u) ? -1.0f : 1.0f;
}

// R x for a block of 256 threads holding one dimension each (thread d holds x_d): returns (R x)_d. s is 256 floats of shared
// memory. Every thread of the block must call it.
__device__ __forceinline__ float kv_rotate256(float x, float* s) {
    const int d = static_cast<int>(threadIdx.x);
    float v = x * kv_rot_sign(d);
#pragma unroll
    for (int h = 1; h < 32; h <<= 1) {
        const float o = __shfl_xor_sync(0xffffffffu, v, h);
        v = (d & h) ? o - v : v + o;
    }
#pragma unroll
    for (int h = 32; h < 256; h <<= 1) {
        s[d] = v;
        __syncthreads();
        const float o = s[d ^ h];
        __syncthreads();
        v = (d & h) ? o - v : v + o;
    }
    return v * 0.0625f;
}

// R^T x, the inverse of kv_rotate256 (the Walsh-Hadamard butterflies first, then the signs). Same contract.
__device__ __forceinline__ float kv_unrotate256(float x, float* s) {
    const int d = static_cast<int>(threadIdx.x);
    float v = x;
#pragma unroll
    for (int h = 1; h < 32; h <<= 1) {
        const float o = __shfl_xor_sync(0xffffffffu, v, h);
        v = (d & h) ? o - v : v + o;
    }
#pragma unroll
    for (int h = 32; h < 256; h <<= 1) {
        s[d] = v;
        __syncthreads();
        const float o = s[d ^ h];
        __syncthreads();
        v = (d & h) ? o - v : v + o;
    }
    return v * 0.0625f * kv_rot_sign(d);
}

}  // namespace limbo::cuda
#endif
