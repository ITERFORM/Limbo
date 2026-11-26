// Copyright (c) 2026 ITERFORM. All rights reserved.
// src/cuda/gemm.cu — batched weight products on the int8 tensor cores. See include/limbo/cuda/gemm.hpp.
//
// Written from the block layouts in include/limbo/quants.hpp and the decode math already verified in
// src/cuda/gemv.cu; the tiling is our own. Ideas: decode a weight tile once into shared memory as int8 and
// apply it to every token row (docs/donors/kindi.md §2.3), the int32 "magic" accumulator whose bits read as a
// float so one FADD replaces an int-to-float conversion (kindi §2.3), grouped MoE GEMM over per-expert token
// lists (kindi §2.3).
//
// One block = 4 warps = a tile of 128 shared-memory weight rows x BM token rows, walked over K in stages of
// BK = 128 columns. Per stage the block decodes its weight rows to int8 (+ one fp32 scale per 16/32 columns),
// copies the activation planes of its token rows, then every warp runs mma.m16n8k16 over its 2 x 16 weight
// rows and all BM tokens. After each group of columns that shares a weight scale the int32 planes are
// recombined in fp32 and scaled into the fp32 accumulators.
#include <cuda_fp16.h>

#include <cfloat>
#include <cstdio>
#include <vector>
#include <type_traits>
#include <algorithm>
#include <cstdlib>
#include <mutex>

#include "limbo/cuda/gemm.hpp"
#include "ops.hpp"
#include "ops_common.cuh"
#include "limbo/quants.hpp"
#include "limbo/types.hpp"

#include "gemm_mmq.cuh"

namespace limbo::cuda {

namespace {

using namespace mmq;

uint64_t* g_t2s = nullptr;
uint32_t* g_t3s = nullptr;
int8_t* g_tnl = nullptr;
uint32_t* g_sel2 = nullptr;
bool g_sel2_ok = false;
uint32_t* g_g3xxs = nullptr;   // the fragment kernel's copies of the IQ3_XXS / IQ2_XXS / IQ2_XS grids and their 7-bit sign table
uint64_t* g_g2xxs = nullptr;
uint64_t* g_g2xs = nullptr;
uint8_t* g_sgn = nullptr;

template <int NT, int BM, int MODE>
bool dispatch_type(uint32_t type, const Args& a, dim3 grid, cudaStream_t s) {
    switch (type) {
        case kQ8_0: return launch_mmq<kQ8_0, NT, BM, MODE>(a, grid, s);
        case kQ6_K: return launch_mmq<kQ6_K, NT, BM, MODE>(a, grid, s);
        case kIQ2_S: return launch_mmq<kIQ2_S, NT, BM, MODE>(a, grid, s);
        case kIQ3_S: return launch_mmq<kIQ3_S, NT, BM, MODE>(a, grid, s);
        case kIQ4_XS: return launch_mmq<kIQ4_XS, NT, BM, MODE>(a, grid, s);
        default: return false;
    }
}

template <int BM, int MODE>
bool dispatch(int nt, uint32_t type, const Args& a, dim3 grid, cudaStream_t s) {
    switch (nt) {
        case 1: return dispatch_type<1, BM, MODE>(type, a, grid, s);
        case 2: return dispatch_type<2, BM, MODE>(type, a, grid, s);
        case 3: return dispatch_type<3, BM, MODE>(type, a, grid, s);
        default: return false;
    }
}

// The dense tile: 64 token rows, or 32 with three planes (the register file cannot hold 64 rows of them).
template <int MODE>
bool dispatch_dense(int nt, uint32_t type, const Args& a, int n_out, int T, cudaStream_t s) {
    if (nt == 3) {
        const dim3 grid(static_cast<unsigned>((n_out + kBN - 1) / kBN), static_cast<unsigned>((T + 31) / 32));
        return dispatch_type<3, 32, MODE>(type, a, grid, s);
    }
    const dim3 grid(static_cast<unsigned>((n_out + kBN - 1) / kBN), static_cast<unsigned>((T + 63) / 64));
    return nt == 1 ? dispatch_type<1, 64, MODE>(type, a, grid, s) : dispatch_type<2, 64, MODE>(type, a, grid, s);
}

// ---- the MoE kernel: weights decoded straight into the A fragments ---------------------------------------
//
// An expert tile has few token rows (BM = 16), so decoding the weights is what costs, and decoding them into a
// shared int8 tile and reading that back is the same work twice. Here every warp owns its weight rows, stages
// the raw block bytes of those rows into shared memory with wide asynchronous copies (double buffered: the
// next super-block is in flight while this one is used), and each thread decodes the 4-byte pieces of the
// mma A fragments it holds (rows g and g+8, columns 4t..4t+3 of every 16) from there. The tensor-core step,
// the scales and the epilogue are the same as in mmq_kernel.
//
// A super-block of an i-quant row is 82 or 110 bytes at an address that is only 2-byte aligned, so each row's
// window is the 16-byte chunks covering it (RC chunks, the offset of the block inside the window is its
// address & 15).
constexpr int kXP = 256 + 16;  // pitch in bytes of an activation row of one super-block stage

template <uint32_t TY>
__host__ __device__ constexpr int raw_chunks() {
    return TY == kIQ2_S ? 7 : TY == kQ6_K ? 15 : TY == kQ4_K ? 9 : TY == kQ5_K ? 11 : TY == kIQ3_XXS ? 7 : TY == kIQ2_XXS ? 5 : TY == kIQ2_XS ? 6
         : TY == kQ2_K ? 7 : TY == kQ8_0 ? 17 : TY == kIQ4_XS ? 9 : 8;   // the super-block from an even offset in a 16-byte window (IQ2_S: 82 bytes from any offset);
}
// Byte offset of the fp16 super-block scale d inside a super-block.
template <uint32_t TY>
__host__ __device__ constexpr int d_offset() {
    return TY == kQ6_K ? 208 : TY == kQ2_K ? 80 : TY == kQ3_K ? 108 : 0;
}

__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
    const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(d), "l"(src));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::); }

// The A-fragment word (columns 4t..4t+3 of the 16 of step i) and the weight scale of that step for one row,
// from the raw super-block at rp in shared memory; mn: the group's minimum (formats with one, has_min).
template <uint32_t TY>
__device__ __forceinline__ void frag_decode(const uint8_t* rp, float d, int i, int t, const uint32_t* tab, const uint8_t* sgn,
                                            uint32_t& areg, float& scale, float& mn) {
    const int m = 2 * i + (t >> 1);  // the 8-column group
    // columns 16i + 4t .. +3: 32-group i / 2, 8-group l inside it, the low (e 0) or high half of that 8-group
    const int ib32 = i >> 1, l = m & 3, e = t & 1;
    if constexpr (TY == kQ8_0) {  // 8 blocks of (f16 d, 32 codes)
        const uint8_t* blk = rp + 34 * ib32;
        scale = __half2float(__ushort_as_half(*reinterpret_cast<const unsigned short*>(blk)));
        areg = ld32_2(blk + 2 + 16 * (i & 1) + 4 * t);
    } else if constexpr (TY == kIQ4_XS) {  // 4-bit codes into the non-linear table (sgn holds it), 6-bit scales minus 32
        const int sh = rp[2] | (rp[3] << 8), sl = rp[4 + (ib32 >> 1)];
        const int ls = ((sl >> (4 * (ib32 & 1))) & 0xf) | (((sh >> (2 * ib32)) & 3) << 4);
        scale = d * static_cast<float>(ls - 32);
        const uint32_t w = ld32_2(rp + 8 + 16 * ib32 + 4 * t);
        const int sh4 = (i & 1) ? 4 : 0;
        const int8_t* nl = reinterpret_cast<const int8_t*>(sgn);
        uint32_t o = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) o |= (static_cast<uint32_t>(nl[(w >> (8 * k + sh4)) & 0xf]) & 0xffu) << (8 * k);
        areg = o;
    } else if constexpr (TY == kIQ3_XXS) {
        const uint32_t word = ld32_2(rp + 66 + 4 * ib32);
        scale = d * (0.5f + static_cast<float>(word >> 28)) * 0.5f;
        const unsigned sg = sgn[(word >> (7 * l)) & 127];
        areg = negate_where(tab[rp[2 + 4 * i + t]], sign_bytes(e ? (sg >> 4) : sg));
    } else if constexpr (TY == kIQ2_XXS) {
        const uint32_t idxw = ld32_2(rp + 2 + 8 * ib32), sgnw = ld32_2(rp + 6 + 8 * ib32);
        scale = d * (0.5f + static_cast<float>(sgnw >> 28)) * 0.25f;
        const uint2 gg = reinterpret_cast<const uint2*>(tab)[(idxw >> (8 * l)) & 0xff];
        const unsigned sg = sgn[(sgnw >> (7 * l)) & 127];
        areg = negate_where(e ? gg.y : gg.x, sign_bytes(e ? (sg >> 4) : sg));
    } else if constexpr (TY == kIQ2_XS) {
        const unsigned q = ld16(rp + 2 + 2 * m);
        const uint8_t scb = rp[66 + ib32];
        scale = d * (0.5f + static_cast<float>((i & 1) ? (scb >> 4) : (scb & 0xf))) * 0.25f;
        const uint2 gg = reinterpret_cast<const uint2*>(tab)[q & 511];
        const unsigned sg = sgn[q >> 9];
        areg = negate_where(e ? gg.y : gg.x, sign_bytes(e ? (sg >> 4) : sg));
    } else if constexpr (TY == kQ2_K) {
        // value 128 half + 32 j + 16 h + l: 2-bit field j of byte 32 half + 16 h + l, scale and minimum of 16-group 8 half + 2 j + h
        const int half = i >> 3, j = (i >> 1) & 3, h = i & 1;
        const unsigned scb = rp[8 * half + 2 * j + h];
        scale = d * static_cast<float>(scb & 15);
        mn = __half2float(__ushort_as_half(*reinterpret_cast<const unsigned short*>(rp + 82))) * static_cast<float>(scb >> 4);
        areg = (ld32_2(rp + 16 + 32 * half + 16 * h + 4 * t) >> (2 * j)) & 0x03030303u;
    } else if constexpr (TY == kQ3_K) {
        // as Q2_K plus the high bit (bit 4 half + j of hmask[16 h + l]: set = the code, clear = 4 less), the 6-bit scale minus 32
        const int half = i >> 3, j = (i >> 1) & 3, h = i & 1;
        const int is = 8 * half + 2 * j + h, g = is >> 2, i4 = is & 3;
        const unsigned lob = rp[96 + (g & 1) * 4 + i4];
        const int slo = (g >> 1) ? static_cast<int>((lob >> 4) & 15) : static_cast<int>(lob & 15);
        const int shi = (rp[104 + i4] >> (2 * g)) & 3;
        scale = d * static_cast<float>((slo | (shi << 4)) - 32);
        const int o16 = 16 * h + 4 * t;
        areg = __vsub4(((ld32_2(rp + 32 + 32 * half + o16) >> (2 * j)) & 0x03030303u) | (((ld32_2(rp + o16) >> (4 * half + j)) & 0x01010101u) << 2),
                       0x04040404u);
    } else if constexpr (TY == kQ4_K || TY == kQ5_K) {
        // group s32 of 32 columns, its 64-column chunk c and nibble g, position p inside it
        const int s32 = i >> 1, c = s32 >> 1, g = s32 & 1, p = 16 * (i & 1) + 4 * t;
        int sc, mi;
        k4_scale_min(s32, rp + 4, sc, mi);
        scale = d * static_cast<float>(sc);
        mn = __half2float(__ushort_as_half(*reinterpret_cast<const unsigned short*>(rp + 2))) * static_cast<float>(mi);
        if constexpr (TY == kQ4_K) {
            areg = (ld32_2(rp + 16 + 32 * c + p) >> (4 * g)) & 0x0F0F0F0Fu;
        } else {   // the fifth bit: bit s32 of the bytes qh[p .. p + 3]
            areg = ((ld32_2(rp + 48 + 32 * c + p) >> (4 * g)) & 0x0F0F0F0Fu) | (((ld32_2(rp + 16 + p) >> s32) & 0x01010101u) << 4);
        }
    } else if constexpr (TY == kIQ2_S) {
        const int ib32 = i >> 1, l4 = 2 * (i & 1) + (t >> 1);
        const unsigned idx = static_cast<unsigned>(rp[2 + m]) | ((static_cast<unsigned>(rp[66 + ib32]) << (8 - 2 * l4)) & 0x300u);
        const uint32_t sel = tab[idx];
        const uint32_t s16 = (t & 1) ? (sel >> 16) : (sel & 0xFFFFu);
        const uint32_t g = __byte_perm(0x002b1908u, 0u, s16);
        const unsigned sg = rp[34 + m];
        areg = negate_where(g, sign_bytes((t & 1) ? (sg >> 4) : sg));
        const uint8_t scb = rp[74 + ib32];
        const int nib = (i & 1) ? (scb >> 4) : (scb & 0xf);
        scale = d * (0.5f + static_cast<float>(nib)) * 0.25f;
    } else if constexpr (TY == kQ6_K) {
        // columns 16i + 4t .. +3: half-block h, 32-run qi, offset l inside the run (the natural order of decode8q)
        const int half = i >> 3, qi = (i >> 1) & 3, l = (i & 1) * 16 + t * 4;
        const uint32_t l4 = ld32_2(rp + half * 64 + (qi & 1) * 32 + l), h4 = ld32_2(rp + 128 + half * 32 + l);
        const uint32_t v = ((l4 >> ((qi >> 1) * 4)) & 0x0F0F0F0Fu) | (((h4 >> (2 * qi)) & 0x03030303u) << 4);
        areg = ((v | 0x80808080u) - 0x20202020u) ^ 0x80808080u;  // v - 32 per byte
        scale = d * static_cast<float>(static_cast<int8_t>(rp[192 + half * 8 + qi * 2 + (i & 1)]));
    } else {  // IQ3_S
        const int l = 2 * (i & 1) + (t >> 1);
        const unsigned qhb = rp[66 + (i >> 1)];
        const unsigned idx = static_cast<unsigned>(rp[2 + 2 * m + (t & 1)]) | ((qhb << (8 - 2 * l - (t & 1))) & 256u);
        const uint32_t g = tab[idx];
        const unsigned sg = rp[74 + m];
        areg = negate_where(g, sign_bytes((t & 1) ? (sg >> 4) : sg));
        const uint8_t scb = rp[106 + (i >> 2)];
        const int nib = ((i >> 1) & 1) ? (scb >> 4) : (scb & 0xf);
        scale = d * (1.f + 2.f * static_cast<float>(nib));
    }
}

template <uint32_t TY, int NT, int BM, int MODE, int NBUF>
__host__ __device__ constexpr size_t moe_smem_bytes() {
    return size_t(NBUF) * (size_t(128) * raw_chunks<TY>() * 16 + size_t(NT) * BM * kXP + size_t(BM) * 8 * 4 + (has_min<TY>() ? size_t(BM) * 16 * 4 : 0));
}

// MT = m-tiles of 16 weight rows per warp: 2 gives 4 warps per block, 1 gives 8 warps (the same 128 rows, half the
// registers per thread, twice the warps to hide latency). With MT = 1 the gate rows and the up rows of the fused
// gate/up sit in different warps (an even and an odd one): the up warp hands its accumulators over through shared
// memory at the end.
template <uint32_t TY, int NT, int BM, int MODE, int NBUF, int MT>
__global__ void __launch_bounds__(256 / MT) moe_kernel(const Args a) {
    constexpr int NTH = 256 / MT;
    extern __shared__ __align__(16) uint8_t smem[];
    constexpr int RC = raw_chunks<TY>(), RP = RC * 16;
    constexpr int GS = group_size<TY>(), KS = GS / 16, TN = BM / 8, SBB = sb_bytes(TY);
    constexpr size_t kRawBytes = size_t(128) * RP, kXBytes = size_t(NT) * BM * kXP, kScBytes = size_t(BM) * 8 * 4;
    constexpr size_t kSmBytes = has_min<TY>() ? size_t(BM) * 16 * 4 : 0;   // activation sums per 16 columns
    constexpr size_t kBufBytes = kRawBytes + kXBytes + kScBytes + kSmBytes;
    __shared__ __align__(16) uint32_t s_tab[TY == kIQ2_S || TY == kIQ2_XS ? 1024 : TY == kIQ3_S || TY == kIQ2_XXS ? 512 : TY == kIQ3_XXS ? 256 : 1];
    __shared__ uint8_t s_sgn[TY == kIQ3_XXS || TY == kIQ2_XXS || TY == kIQ2_XS ? 128 : TY == kIQ4_XS ? 16 : 1];   // signs, or IQ4_XS's table
    __shared__ int s_xrow[BM];

    const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
    const int gq = lane >> 2, tq = lane & 3;

    if constexpr (TY == kIQ2_S) {
        for (int i = tid; i < 1024; i += NTH) s_tab[i] = a.tb.sel2[i];
    } else if constexpr (TY == kIQ3_S) {
        for (int i = tid; i < 512; i += NTH) s_tab[i] = a.tb.t3s[i];
    } else if constexpr (TY == kIQ3_XXS) {
        for (int i = tid; i < 256; i += NTH) s_tab[i] = a.tb.g3xxs[i];
    } else if constexpr (TY == kIQ2_XXS) {
        for (int i = tid; i < 512; i += NTH) s_tab[i] = reinterpret_cast<const uint32_t*>(a.tb.g2xxs)[i];
    } else if constexpr (TY == kIQ2_XS) {
        for (int i = tid; i < 1024; i += NTH) s_tab[i] = reinterpret_cast<const uint32_t*>(a.tb.g2xs)[i];
    }
    if constexpr (TY == kIQ3_XXS || TY == kIQ2_XXS || TY == kIQ2_XS) {
        for (int i = tid; i < 128; i += NTH) s_sgn[i] = a.tb.signs[i];
    } else if constexpr (TY == kIQ4_XS) {
        if (tid < 16) s_sgn[tid] = static_cast<uint8_t>(a.tb.tnl[tid]);
    }

    int e = 0, off, cnt;
    if (a.tiles) {
        const int4 tl = a.tiles[blockIdx.y < static_cast<unsigned>(*a.n_tiles) ? blockIdx.y : 0];
        if (static_cast<int>(blockIdx.y) >= *a.n_tiles) return;
        e = tl.x;
        off = tl.y;
        cnt = tl.z;
    } else {  // dense: a tile is BM consecutive rows of xq
        off = blockIdx.y * BM;
        cnt = min(BM, a.T - off);
    }
    const uint8_t* eb_ = a.eptr ? a.eptr[e] : nullptr;
    const uint8_t* wb0 = eb_ ? eb_ + reinterpret_cast<uintptr_t>(a.W0) : a.W0 + static_cast<uint64_t>(e) * a.expert_bytes;
    const uint8_t* wb1 = MODE == kGateUp ? (eb_ ? eb_ + reinterpret_cast<uintptr_t>(a.W1)
                                                : a.W1 + static_cast<uint64_t>(e) * a.expert_bytes)
                                         : nullptr;
    const int n0 = blockIdx.x * (MODE == kGateUp ? 64 : 128);

    for (int i = tid; i < BM; i += NTH) {
        const int p = off + (i < cnt ? i : 0);
        s_xrow[i] = a.xrow_list ? a.xrow_list[p] : p;
    }
    __syncthreads();

    // smem row r of this block -> (weights, global row): warp w owns r in [16 MT w, 16 MT (w + 1)), m-tile j = (r >> 4) & 1 - gate/up: the odd m-tiles are up rows
    auto row_ptr = [&](int r, int sb) -> const uint8_t* {
        const uint8_t* wb;
        int grow;
        if constexpr (MODE == kGateUp) {
            wb = ((r >> 4) & 1) ? wb1 : wb0;
            grow = n0 + (r >> 5) * 16 + (r & 15);
        } else {
            wb = wb0;
            grow = n0 + r;
        }
        if (grow >= a.n_out) grow = a.n_out - 1;
        return wb + static_cast<uint64_t>(grow) * a.row_bytes + static_cast<uint64_t>(sb) * SBB;
    };

    auto issue = [&](int st, int buf) {
        uint8_t* raw = smem + buf * kBufBytes;
        uint8_t* xs = raw + kRawBytes;
        float* xsc = reinterpret_cast<float*>(xs + kXBytes);
        for (int c = tid; c < 128 * RC; c += NTH) {
            const int r = c / RC, ch = c % RC;
            const uint8_t* p = row_ptr(r, st);
            const uint8_t* win = reinterpret_cast<const uint8_t*>(reinterpret_cast<uintptr_t>(p) & ~static_cast<uintptr_t>(15));
            cp_async16(raw + r * RP + ch * 16, win + ch * 16);
        }
        for (int c = tid; c < NT * BM * 16; c += NTH) {
            const int t = c / (BM * 16), rem = c % (BM * 16);
            const int row = rem >> 4, c16 = rem & 15;
            cp_async16(xs + (t * BM + row) * kXP + c16 * 16,
                       a.xq + t * a.xterm_stride + static_cast<uint64_t>(s_xrow[row]) * a.K + st * 256 + c16 * 16);
        }
        for (int c = tid; c < BM * 2; c += NTH) {
            const int row = c >> 1, h = c & 1;
            cp_async16(xsc + row * 8 + h * 4, a.xs + static_cast<uint64_t>(s_xrow[row]) * (a.K / 32) + st * 8 + h * 4);
        }
        if constexpr (has_min<TY>()) {
            float* xsm = xsc + BM * 8;
            for (int c = tid; c < BM * 4; c += NTH) {
                const int row = c >> 2, q4 = c & 3;
                cp_async16(xsm + row * 16 + q4 * 4, a.xsum + static_cast<uint64_t>(s_xrow[row]) * (a.K / 16) + st * 16 + q4 * 4);
            }
        }
        cp_async_commit();
    };

    float acc[MT][TN][4];
#pragma unroll
    for (int j = 0; j < MT; ++j)
#pragma unroll
        for (int n = 0; n < TN; ++n)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[j][n][i] = 0.f;

    // dense K slice blockIdx.z (a.ksplit > 1): its stages only, its sums to its own plane of a.part
    int st0 = 0, nstage = a.K / 256;
    if (!a.tiles && a.ksplit > 1) {
        const int per = (nstage + a.ksplit - 1) / a.ksplit;
        st0 = min(nstage, static_cast<int>(blockIdx.z) * per);
        nstage = min(nstage, st0 + per);
    }
    if (st0 < nstage) issue(st0, 0);
    for (int st = st0; st < nstage; ++st) {
        const int buf = NBUF == 2 ? ((st - st0) & 1) : 0;
        cp_async_wait_all();
        __syncthreads();
        if constexpr (NBUF == 2) {
            if (st + 1 < nstage) issue(st + 1, buf ^ 1);
        }
        const uint8_t* raw = smem + buf * kBufBytes;
        const uint8_t* xs = raw + kRawBytes;
        const float* xsc = reinterpret_cast<const float*>(xs + kXBytes);

        // this thread's 4 rows: warp wid, m-tile j, rows gq and gq+8
        const uint8_t* rp[MT][2];
        float dsc[MT][2];
#pragma unroll
        for (int j = 0; j < MT; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = wid * (16 * MT) + j * 16 + gq + h * 8;
                const uint8_t* p = row_ptr(r, st);
                rp[j][h] = raw + r * RP + (reinterpret_cast<uintptr_t>(p) & 15);
                dsc[j][h] = __half2float(__ushort_as_half(*reinterpret_cast<const unsigned short*>(rp[j][h] + d_offset<TY>())));
            }

#pragma unroll
        for (int i = 0; i < 16; i += KS) {
            uint32_t A[KS][MT][2];
            float SW[MT][2], MN[MT][2];
            uint32_t B[KS][NT][TN];
#pragma unroll
            for (int s = 0; s < KS; ++s) {
#pragma unroll
                for (int j = 0; j < MT; ++j)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        float sc, mnv = 0.f;
                        frag_decode<TY>(rp[j][h], dsc[j][h], i + s, tq, s_tab, s_sgn, A[s][j][h], sc, mnv);
                        if (s == 0) {
                            SW[j][h] = sc;
                            MN[j][h] = mnv;
                        }
                    }
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    if constexpr (TN >= 4) {
#pragma unroll
                        for (int p = 0; p < TN / 4; ++p)
                            ldsm_x4(B[s][t][4 * p], B[s][t][4 * p + 1], B[s][t][4 * p + 2], B[s][t][4 * p + 3],
                                    xs + (t * BM + (4 * p + (lane >> 3)) * 8 + (lane & 7)) * kXP + (i + s) * 16);
                    } else {
#pragma unroll
                        for (int p = 0; p < TN / 2; ++p)
                            ldsm_x2(B[s][t][2 * p], B[s][t][2 * p + 1],
                                    xs + (t * BM + (2 * p + ((lane >> 3) & 1)) * 8 + (lane & 7)) * kXP + (i + s) * 16);
                    }
                }
            }
            const int xb = i / 2;
            float sxr[TN][2];
#pragma unroll
            for (int tn = 0; tn < TN; ++tn) {
                sxr[tn][0] = xsc[(tn * 8 + 2 * tq) * 8 + xb];
                sxr[tn][1] = xsc[(tn * 8 + 2 * tq + 1) * 8 + xb];
            }
#pragma unroll
            for (int j = 0; j < MT; ++j)
#pragma unroll
                for (int tn = 0; tn < TN; ++tn) {
                    int d[NT][4];
#pragma unroll
                    for (int t = 0; t < NT; ++t) {
                        mma_s8(d[t], A[0][j][0], A[0][j][1], B[0][t][tn], kMagic, kMagic, kMagic, kMagic);
#pragma unroll
                        for (int s = 1; s < KS; ++s)
                            mma_s8(d[t], A[s][j][0], A[s][j][1], B[s][t][tn], d[t][0], d[t][1], d[t][2], d[t][3]);
                    }
                    const float sx0 = sxr[tn][0], sx1 = sxr[tn][1];
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        float v = __int_as_float(d[0][q]) - kMagicF;
#pragma unroll
                        for (int t = 1; t < NT; ++t) v = fmaf(v, 128.0f, __int_as_float(d[t][q]) - kMagicF);
                        const float sw = SW[j][q >> 1];
                        const float sx = (q & 1) ? sx1 : sx0;
                        acc[j][tn][q] = fmaf(sw * sx, v, acc[j][tn][q]);
                        if constexpr (has_min<TY>()) {   // minus min x sum of the group's activations
                            const float* xsm = xsc + BM * 8;
                            const int tok = tn * 8 + 2 * tq + (q & 1);
                            const float xg = GS == 32 ? xsm[tok * 16 + 2 * xb] + xsm[tok * 16 + 2 * xb + 1] : xsm[tok * 16 + i];
                            acc[j][tn][q] = fmaf(-MN[j][q >> 1], xg, acc[j][tn][q]);
                        }
                    }
                }
        }
        if constexpr (NBUF == 1) __syncthreads();
        if constexpr (NBUF == 1) {
            if (st + 1 < nstage) issue(st + 1, 0);
        }
    }

    if constexpr (MODE == kGateUp && MT == 1) {
        // an odd warp holds the up rows of the gate rows its even neighbour holds: hand them over through shared memory
        float* ex = reinterpret_cast<float*>(smem);  // [warp pair][BM tokens][16 rows]
        __syncthreads();
        if (wid & 1) {
#pragma unroll
            for (int tn = 0; tn < TN; ++tn)
#pragma unroll
                for (int q = 0; q < 4; ++q)
                    ex[((wid >> 1) * BM + tn * 8 + 2 * tq + (q & 1)) * 16 + gq + (q >= 2 ? 8 : 0)] = acc[0][tn][q];
        }
        __syncthreads();
        if (!(wid & 1)) {
#pragma unroll
            for (int tn = 0; tn < TN; ++tn)
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int tok = tn * 8 + 2 * tq + (q & 1);
                    if (tok >= cnt) continue;
                    const int rr = gq + (q >= 2 ? 8 : 0), n = n0 + (wid >> 1) * 16 + rr;
                    if (n < a.n_out)
                        a.y[static_cast<int64_t>(off + tok) * a.ldy + n] = silu_f(acc[0][tn][q]) * ex[((wid >> 1) * BM + tok) * 16 + rr];
                }
        }
        return;
    }
    float* ybase = a.y;
    if constexpr (MODE != kGateUp) {
        if (!a.tiles && a.ksplit > 1) ybase = a.part + static_cast<int64_t>(blockIdx.z) * a.T * a.ldy;
    }
#pragma unroll
    for (int tn = 0; tn < TN; ++tn) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int tok = tn * 8 + 2 * tq + (q & 1);
            if (tok >= cnt) continue;
            const int rr = gq + (q >= 2 ? 8 : 0);
            float* yrow = ybase + static_cast<int64_t>(off + tok) * a.ldy;
            if constexpr (MODE == kGateUp) {
                if constexpr (MT == 2) {
                    const int n = n0 + wid * 16 + rr;
                    if (n < a.n_out) yrow[n] = silu_f(acc[0][tn][q]) * acc[1][tn][q];
                }
            } else {
#pragma unroll
                for (int j = 0; j < MT; ++j) {
                    const int n = n0 + wid * (16 * MT) + j * 16 + rr;
                    if (n >= a.n_out) continue;
                    if (MODE == kAcc && !(a.ksplit > 1 && !a.tiles)) yrow[n] += acc[j][tn][q];
                    else yrow[n] = acc[j][tn][q];
                }
            }
        }
    }
}

template <uint32_t TY, int NT, int BM, int MODE, int NBUF, int MT>
bool launch_moe(const Args& a, dim3 grid, cudaStream_t s) {
    constexpr size_t smem = moe_smem_bytes<TY, NT, BM, MODE, NBUF>();
    static bool configured = false;
    if (!configured) {
        if (cudaFuncSetAttribute(moe_kernel<TY, NT, BM, MODE, NBUF, MT>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(smem)) != cudaSuccess)
            return false;
        configured = true;
    }
    moe_kernel<TY, NT, BM, MODE, NBUF, MT><<<grid, 256 / MT, smem, s>>>(a);
    return cudaGetLastError() == cudaSuccess;
}

// ---- the dense prefill kernel: moe_kernel's maths and order, weights and half super-block activations double buffered ----
// Raw rows padded to an odd chunk count (8-row byte reads hit one bank at 128 B); a slab grid keeps activations in L2.
constexpr int kDXP = 144;        // pitch in bytes of an activation row of a half stage: 128 columns + 16
constexpr int kDenseSlab = 16;   // token tiles per slab

__device__ __forceinline__ void cp_async4(void* dst, const void* src) {
    const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(d), "l"(src));
}

template <uint32_t TY>
__host__ __device__ constexpr int dn_pitch() { return raw_chunks<TY>() * 16 + ((raw_chunks<TY>() & 1) ? 0 : 16); }
template <uint32_t TY>
__host__ __device__ constexpr size_t dn_wbytes() { return size_t(128) * dn_pitch<TY>(); }
template <uint32_t TY, int NT, int BM>
__host__ __device__ constexpr size_t dn_xbytes() {   // one activation buffer: the planes, 4 scale groups, 8 sums of 16
    return size_t(NT) * BM * kDXP + size_t(BM) * 4 * 4 + (has_min<TY>() ? size_t(BM) * 8 * 4 : 0);
}
// two weight buffers when they fit beside the activations (the codebook tables take up to 4.4 KB of the 99 KB), else one
template <uint32_t TY, int NT, int BM>
__host__ __device__ constexpr int dn_wb() { return 2 * dn_wbytes<TY>() + 2 * dn_xbytes<TY, NT, BM>() + 4400 <= 101376 ? 2 : 1; }
template <uint32_t TY, int NT, int BM>
__host__ __device__ constexpr size_t dn_smem() { return size_t(dn_wb<TY, NT, BM>()) * dn_wbytes<TY>() + 2 * dn_xbytes<TY, NT, BM>(); }

template <uint32_t TY, int NT, int BM, int MODE>
__global__ void __launch_bounds__(256, 1) dense_kernel(const Args a) {
    constexpr int WB = dn_wb<TY, NT, BM>();
    constexpr int RC = raw_chunks<TY>(), RP = dn_pitch<TY>();
    constexpr int GS = group_size<TY>(), KS = GS / 16, TN = BM / 8, SBB = sb_bytes(TY);
    constexpr size_t kWBytes = dn_wbytes<TY>(), kXBuf = dn_xbytes<TY, NT, BM>();
    constexpr int XCH = NT * BM * 8 / 256, WCH = (128 * RC + 255) / 256, MCH = BM * 8 / 256;
    extern __shared__ __align__(16) uint8_t smem[];
    __shared__ __align__(16) uint32_t s_tab[TY == kIQ2_S || TY == kIQ2_XS ? 1024 : TY == kIQ3_S || TY == kIQ2_XXS ? 512 : TY == kIQ3_XXS ? 256 : 1];
    __shared__ uint8_t s_sgn[TY == kIQ3_XXS || TY == kIQ2_XXS || TY == kIQ2_XS ? 128 : TY == kIQ4_XS ? 16 : 1];

    const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
    const int gq = lane >> 2, tq = lane & 3;

    if constexpr (TY == kIQ2_S) {
        for (int i = tid; i < 1024; i += 256) s_tab[i] = a.tb.sel2[i];
    } else if constexpr (TY == kIQ3_S) {
        for (int i = tid; i < 512; i += 256) s_tab[i] = a.tb.t3s[i];
    } else if constexpr (TY == kIQ3_XXS) {
        for (int i = tid; i < 256; i += 256) s_tab[i] = a.tb.g3xxs[i];
    } else if constexpr (TY == kIQ2_XXS) {
        for (int i = tid; i < 512; i += 256) s_tab[i] = reinterpret_cast<const uint32_t*>(a.tb.g2xxs)[i];
    } else if constexpr (TY == kIQ2_XS) {
        for (int i = tid; i < 1024; i += 256) s_tab[i] = reinterpret_cast<const uint32_t*>(a.tb.g2xs)[i];
    }
    if constexpr (TY == kIQ3_XXS || TY == kIQ2_XXS || TY == kIQ2_XS) {
        for (int i = tid; i < 128; i += 256) s_sgn[i] = a.tb.signs[i];
    } else if constexpr (TY == kIQ4_XS) {
        if (tid < 16) s_sgn[tid] = static_cast<uint8_t>(a.tb.tnl[tid]);
    }

    // block -> (weight tile, token tile): a slab of kDenseSlab token tiles walks every weight tile
    const int nrow = (a.n_out + 127) >> 7, ntok = (a.T + BM - 1) / BM;
    const int slab = static_cast<int>(blockIdx.x) / (kDenseSlab * nrow);
    const int ws = min(kDenseSlab, ntok - slab * kDenseSlab);
    const int rem0 = static_cast<int>(blockIdx.x) - slab * kDenseSlab * nrow;
    const int n0 = (rem0 / ws) * 128, off = (slab * kDenseSlab + rem0 % ws) * BM;
    const int cnt = min(BM, a.T - off);
    const uint8_t* wb0 = a.W0;

    // this thread's copy descriptors: source bases (the K offset is added per stage) and shared-memory offsets
    const int8_t* xsrc[XCH];
    int xdst[XCH];
#pragma unroll
    for (int k = 0; k < XCH; ++k) {
        const int c = tid + 256 * k, t = c / (BM * 8), rem = c % (BM * 8), row = rem >> 3, c16 = rem & 7;
        const int p = off + (row < cnt ? row : 0);
        xsrc[k] = a.xq + t * a.xterm_stride + static_cast<uint64_t>(p) * a.K + c16 * 16;
        xdst[k] = (t * BM + row) * kDXP + c16 * 16;
    }
    const float* ssrc;
    int sdst;
    {
        const int row = tid >> 2, g = tid & 3, p = off + (row < cnt ? row : 0);
        ssrc = a.xs + static_cast<uint64_t>(p) * (a.K / 32) + g;
        sdst = g * BM + row;
    }
    [[maybe_unused]] const float* msrc[MCH];
    [[maybe_unused]] int mdst[MCH];
    if constexpr (has_min<TY>()) {
#pragma unroll
        for (int k = 0; k < MCH; ++k) {
            const int c = tid + 256 * k, row = c >> 3, g = c & 7, p = off + (row < cnt ? row : 0);
            msrc[k] = a.xsum + static_cast<uint64_t>(p) * (a.K / 16) + g;
            mdst[k] = g * BM + row;
        }
    }
    const uint8_t* wsrc[WCH];
    int wdst[WCH];
#pragma unroll
    for (int k = 0; k < WCH; ++k) {
        const int c = tid + 256 * k, r = c / RC, ch = c % RC;
        int grow = n0 + r;
        if (grow >= a.n_out) grow = a.n_out - 1;
        wsrc[k] = wb0 + static_cast<uint64_t>(grow) * a.row_bytes + ch * 16;
        wdst[k] = r * RP + ch * 16;
    }

    auto issue_w = [&](int sb, int wbuf) {
        uint8_t* raw = smem + wbuf * kWBytes;
#pragma unroll
        for (int k = 0; k < WCH; ++k) {
            if (256 * k + tid < 128 * RC) {   // the row's window starts at the 16-byte floor of its super-block
                const uint8_t* p = wsrc[k] + static_cast<uint64_t>(sb) * SBB;
                cp_async16(raw + wdst[k], reinterpret_cast<const uint8_t*>(reinterpret_cast<uintptr_t>(p) & ~static_cast<uintptr_t>(15)));
            }
        }
    };
    auto issue_x = [&](int sb, int h, int buf) {
        uint8_t* xb = smem + WB * kWBytes + buf * kXBuf;
        float* sxs = reinterpret_cast<float*>(xb + NT * BM * kDXP);
        const int ko = sb * 256 + h * 128;
#pragma unroll
        for (int k = 0; k < XCH; ++k) cp_async16(xb + xdst[k], xsrc[k] + ko);
        if (BM * 4 >= 256 || tid < BM * 4) cp_async4(sxs + sdst, ssrc + sb * 8 + h * 4);
        if constexpr (has_min<TY>()) {
            float* sms = sxs + BM * 4;
#pragma unroll
            for (int k = 0; k < MCH; ++k) cp_async4(sms + mdst[k], msrc[k] + sb * 16 + h * 8);
        }
    };

    float acc[TN][4];
#pragma unroll
    for (int n = 0; n < TN; ++n)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[n][i] = 0.f;

    // one half super-block (hc: 0 or 1) from an activation buffer and this thread's two rows
    auto compute = [&](auto hc, const uint8_t* const* rp, const float* dsc, const uint8_t* xb) {
        constexpr int h = decltype(hc)::value;
        const float* sxs = reinterpret_cast<const float*>(xb + NT * BM * kDXP);
        [[maybe_unused]] const float* sms = sxs + BM * 4;
#pragma unroll
        for (int ii = 0; ii < 8; ii += KS) {
            const int i = 8 * h + ii;
            uint32_t A[KS][2];
            float SW[2], MN[2];
            uint32_t B[KS][NT][TN];
#pragma unroll
            for (int s = 0; s < KS; ++s) {
#pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                    float sc, mnv = 0.f;
                    frag_decode<TY>(rp[hh], dsc[hh], i + s, tq, s_tab, s_sgn, A[s][hh], sc, mnv);
                    if (s == 0) {
                        SW[hh] = sc;
                        MN[hh] = mnv;
                    }
                }
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    if constexpr (TN >= 4) {
#pragma unroll
                        for (int p = 0; p < TN / 4; ++p)
                            ldsm_x4(B[s][t][4 * p], B[s][t][4 * p + 1], B[s][t][4 * p + 2], B[s][t][4 * p + 3],
                                    xb + (t * BM + (4 * p + (lane >> 3)) * 8 + (lane & 7)) * kDXP + (ii + s) * 16);
                    } else {
#pragma unroll
                        for (int p = 0; p < TN / 2; ++p)
                            ldsm_x2(B[s][t][2 * p], B[s][t][2 * p + 1],
                                    xb + (t * BM + (2 * p + ((lane >> 3) & 1)) * 8 + (lane & 7)) * kDXP + (ii + s) * 16);
                    }
                }
            }
#pragma unroll
            for (int tn = 0; tn < TN; ++tn) {
                // two planes: the exact integer 128 a + b converted once (the fp32 recombination's value)
                constexpr int kInit = NT == 2 ? 0 : kMagic;
                int d[NT][4];
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    mma_s8(d[t], A[0][0], A[0][1], B[0][t][tn], kInit, kInit, kInit, kInit);
#pragma unroll
                    for (int s = 1; s < KS; ++s) mma_s8(d[t], A[s][0], A[s][1], B[s][t][tn], d[t][0], d[t][1], d[t][2], d[t][3]);
                }
                const float2 sx = *reinterpret_cast<const float2*>(sxs + (ii >> 1) * BM + tn * 8 + 2 * tq);
                [[maybe_unused]] float2 m0 = make_float2(0.f, 0.f), m1 = m0;
                if constexpr (has_min<TY>()) {
                    m0 = *reinterpret_cast<const float2*>(sms + ii * BM + tn * 8 + 2 * tq);
                    if constexpr (GS == 32) m1 = *reinterpret_cast<const float2*>(sms + (ii + 1) * BM + tn * 8 + 2 * tq);
                }
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    float v;
                    if constexpr (NT == 2) {
                        v = __int2float_rn(d[0][q] * 128 + d[1][q]);
                    } else {
                        v = __int_as_float(d[0][q]) - kMagicF;
#pragma unroll
                        for (int t = 1; t < NT; ++t) v = fmaf(v, 128.0f, __int_as_float(d[t][q]) - kMagicF);
                    }
                    const float sw = SW[q >> 1];
                    const float sxv = (q & 1) ? sx.y : sx.x;
                    acc[tn][q] = fmaf(sw * sxv, v, acc[tn][q]);
                    if constexpr (has_min<TY>()) {   // minus min x sum of the group's activations
                        const float xg = GS == 32 ? ((q & 1) ? m0.y + m1.y : m0.x + m1.x) : ((q & 1) ? m0.y : m0.x);
                        acc[tn][q] = fmaf(-MN[q >> 1], xg, acc[tn][q]);
                    }
                }
            }
        }
    };

    uint32_t rlow[2];   // low bits of the address of this thread's rows, super-block 0
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
        int grow = n0 + wid * 16 + gq + hh * 8;
        if (grow >= a.n_out) grow = a.n_out - 1;
        rlow[hh] = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(wb0 + static_cast<uint64_t>(grow) * a.row_bytes));
    }

    const int nsb = a.K >> 8;
    __syncthreads();
    if constexpr (WB == 2) issue_w(0, 0);
    issue_x(0, 0, 0);
    cp_async_commit();
    for (int sb = 0; sb < nsb; ++sb) {
        const uint8_t* wraw = smem + (WB == 2 ? (sb & 1) : 0) * kWBytes;
        cp_async_wait_all();
        __syncthreads();
        if constexpr (WB == 1) {
            issue_w(sb, 0);
            cp_async_commit();
            cp_async_wait_all();
            __syncthreads();
        }
        issue_x(sb, 1, 1);
        if constexpr (WB == 2) {
            if (sb + 1 < nsb) issue_w(sb + 1, (sb + 1) & 1);
        }
        cp_async_commit();

        const uint8_t* rp[2];
        float dsc[2];
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int r = wid * 16 + gq + hh * 8;
            rp[hh] = wraw + r * RP + ((rlow[hh] + sb * SBB) & 15);
            dsc[hh] = __half2float(__ushort_as_half(*reinterpret_cast<const unsigned short*>(rp[hh] + d_offset<TY>())));
        }
        compute(std::integral_constant<int, 0>{}, rp, dsc, smem + WB * kWBytes);

        cp_async_wait_all();
        __syncthreads();
        if (sb + 1 < nsb) {
            issue_x(sb + 1, 0, 0);
            cp_async_commit();
        }
        compute(std::integral_constant<int, 1>{}, rp, dsc, smem + WB * kWBytes + kXBuf);
    }

    // all the loads of an accumulating product go out before the first add
    [[maybe_unused]] float yv[TN][4];
    if constexpr (MODE == kAcc) {
#pragma unroll
        for (int tn = 0; tn < TN; ++tn)
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int tok = tn * 8 + 2 * tq + (q & 1), n = n0 + wid * 16 + gq + (q >= 2 ? 8 : 0);
                yv[tn][q] = (tok < cnt && n < a.n_out) ? a.y[static_cast<int64_t>(off + tok) * a.ldy + n] : 0.f;
            }
    }
#pragma unroll
    for (int tn = 0; tn < TN; ++tn) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int tok = tn * 8 + 2 * tq + (q & 1);
            if (tok >= cnt) continue;
            const int n = n0 + wid * 16 + gq + (q >= 2 ? 8 : 0);
            if (n >= a.n_out) continue;
            float* yp = a.y + static_cast<int64_t>(off + tok) * a.ldy + n;
            if constexpr (MODE == kAcc) *yp = yv[tn][q] + acc[tn][q];
            else *yp = acc[tn][q];
        }
    }
}

template <uint32_t TY, int NT, int BM, int MODE>
bool launch_dense(const Args& a, int T, int n_out, cudaStream_t s) {
    constexpr size_t smem = dn_smem<TY, NT, BM>();
    static bool configured = false;
    if (!configured) {
        if (cudaFuncSetAttribute(dense_kernel<TY, NT, BM, MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)) != cudaSuccess)
            return false;
        configured = true;
    }
    const unsigned grid = static_cast<unsigned>(((n_out + 127) / 128) * ((T + BM - 1) / BM));
    dense_kernel<TY, NT, BM, MODE><<<grid, 256, smem, s>>>(a);
    return cudaGetLastError() == cudaSuccess;
}

template <int NT, int BM, int MODE>
bool dispatch_dn(uint32_t type, const Args& a, int T, int n_out, cudaStream_t s) {
    auto go = [&](auto ty) { return launch_dense<decltype(ty)::value, NT, BM, MODE>(a, T, n_out, s); };
    switch (type) {
        case kIQ2_S: return go(std::integral_constant<uint32_t, kIQ2_S>{});
        case kIQ3_S: return go(std::integral_constant<uint32_t, kIQ3_S>{});
        case kQ6_K: return go(std::integral_constant<uint32_t, kQ6_K>{});
        case kQ4_K: return go(std::integral_constant<uint32_t, kQ4_K>{});
        case kQ5_K: return go(std::integral_constant<uint32_t, kQ5_K>{});
        case kIQ3_XXS: return go(std::integral_constant<uint32_t, kIQ3_XXS>{});
        case kIQ2_XXS: return go(std::integral_constant<uint32_t, kIQ2_XXS>{});
        case kIQ2_XS: return go(std::integral_constant<uint32_t, kIQ2_XS>{});
        case kQ2_K: return go(std::integral_constant<uint32_t, kQ2_K>{});
        case kQ3_K: return go(std::integral_constant<uint32_t, kQ3_K>{});
        case kQ8_0: return go(std::integral_constant<uint32_t, kQ8_0>{});
        case kIQ4_XS: return go(std::integral_constant<uint32_t, kIQ4_XS>{});
        default: return false;
    }
}

// Whether the fragment kernel can take this expert stack: an i-quant with 16-byte aligned rows (IQ2_S), or any
// row alignment (IQ3_S's window is wide enough), and K a multiple of 256.
bool moe_fast_ok(uint32_t type, const Args& a) {
    if (a.K % 256 != 0 || a.expert_bytes % 16 != 0) return false;
    if (a.eptr && ((reinterpret_cast<uintptr_t>(a.W0) | reinterpret_cast<uintptr_t>(a.W1)) % 16) != 0) return false;
    if (type == kIQ2_S) return g_sel2_ok;
    return type == kIQ3_S || type == kQ6_K;
}

template <int NT, int BM, int MODE, int NBUF, int MT>
bool dispatch_moe_type(uint32_t type, const Args& a, dim3 grid, cudaStream_t s) {
    switch (type) {
        case kIQ2_S: return launch_moe<kIQ2_S, NT, BM, MODE, NBUF, MT>(a, grid, s);
        case kIQ3_S: return launch_moe<kIQ3_S, NT, BM, MODE, NBUF, MT>(a, grid, s);
        case kQ6_K: return launch_moe<kQ6_K, NT, BM, MODE, NBUF, MT>(a, grid, s);
        default: return false;
    }
}

// The dense products' fragment formats (prefill and decode; MoE stacks keep dispatch_moe_type's three).
bool frag_type(uint32_t type) {
    return type == kIQ3_S || type == kQ6_K || type == kQ4_K || type == kQ5_K || type == kIQ3_XXS || type == kIQ2_XXS || type == kIQ2_XS ||
           type == kQ2_K || type == kQ3_K || type == kQ8_0 || type == kIQ4_XS || (type == kIQ2_S && g_sel2_ok);
}
template <int BM, int MODE, int NBUF, int MT>
bool dispatch_frag(int nt, uint32_t type, const Args& a, dim3 grid, cudaStream_t s) {
    auto go = [&](auto ty) {
        constexpr uint32_t TY = decltype(ty)::value;
        switch (nt) {
            case 1: return launch_moe<TY, 1, BM, MODE, NBUF, MT>(a, grid, s);
            case 2: return launch_moe<TY, 2, BM, MODE, NBUF, MT>(a, grid, s);
            default: return launch_moe<TY, 3, BM, MODE, NBUF, MT>(a, grid, s);
        }
    };
    switch (type) {
        case kIQ2_S: return go(std::integral_constant<uint32_t, kIQ2_S>{});
        case kIQ3_S: return go(std::integral_constant<uint32_t, kIQ3_S>{});
        case kQ6_K: return go(std::integral_constant<uint32_t, kQ6_K>{});
        case kQ4_K: return go(std::integral_constant<uint32_t, kQ4_K>{});
        case kQ5_K: return go(std::integral_constant<uint32_t, kQ5_K>{});
        case kIQ3_XXS: return go(std::integral_constant<uint32_t, kIQ3_XXS>{});
        case kIQ2_XXS: return go(std::integral_constant<uint32_t, kIQ2_XXS>{});
        case kIQ2_XS: return go(std::integral_constant<uint32_t, kIQ2_XS>{});
        case kQ2_K: return go(std::integral_constant<uint32_t, kQ2_K>{});
        case kQ3_K: return go(std::integral_constant<uint32_t, kQ3_K>{});
        case kQ8_0: return go(std::integral_constant<uint32_t, kQ8_0>{});
        case kIQ4_XS: return go(std::integral_constant<uint32_t, kIQ4_XS>{});
        default: return false;
    }
}

// MT: m-tiles per warp (2: 4 warps per block, 1: 8 warps); the MoE kernels use 2, the dense ones 1.
template <int BM, int MODE, int NBUF, int MT = 2>
bool dispatch_moe(int nt, uint32_t type, const Args& a, dim3 grid, cudaStream_t s) {
    switch (nt) {
        case 1: return dispatch_moe_type<1, BM, MODE, NBUF, MT>(type, a, grid, s);
        case 2: return dispatch_moe_type<2, BM, MODE, NBUF, MT>(type, a, grid, s);
        case 3: return dispatch_moe_type<3, BM, MODE, NBUF, MT>(type, a, grid, s);
        default: return false;
    }
}

// The bytes of a row of n_in elements (n_in need not be a multiple of 256 for the 32- and 64-element block formats:
// Inkling-2's expert down rows are 1,152 wide).
uint64_t row_bytes_of(uint32_t type, int n_in) {
    return n_in % 256 == 0 ? static_cast<uint64_t>(n_in / 256) * sb_bytes(type)
                           : limbo::type_bytes(type, static_cast<uint64_t>(n_in));
}

Args base_args(uint32_t type, const void* W, const int8_t* xq, const float* xs, int xrows, int K, float* y, int ldy,
               int n_out) {
    Args a{};
    a.W0 = static_cast<const uint8_t*>(W);
    a.row_bytes = row_bytes_of(type, K);
    a.xq = xq;
    a.xs = xs;
    a.xterm_stride = static_cast<uint64_t>(xrows) * K;
    a.y = y;
    a.ldy = ldy;
    a.n_out = n_out;
    a.K = K;
    a.tb = Tabs{g_t2s, g_t3s, g_tnl, g_sel2, g_g2xxs, g_g2xs, g_g3xxs, nullptr, g_sgn};
    return a;
}

}  // namespace

bool gemm_init(std::string* err) {
    if (g_t2s) return true;
    std::vector<int8_t> nl(quants::kIq4NlValues, quants::kIq4NlValues + 16);
    cudaError_t e = cudaMalloc(reinterpret_cast<void**>(&g_t2s), sizeof(uint64_t) * 1024);
    if (e == cudaSuccess) e = cudaMalloc(reinterpret_cast<void**>(&g_t3s), sizeof(uint32_t) * 512);
    if (e == cudaSuccess) e = cudaMalloc(reinterpret_cast<void**>(&g_tnl), 16);
    if (e == cudaSuccess) e = cudaMemcpy(g_t2s, quants::kIq2sGrid, sizeof(uint64_t) * 1024, cudaMemcpyHostToDevice);
    if (e == cudaSuccess) e = cudaMemcpy(g_t3s, quants::kIq3sGrid, sizeof(uint32_t) * 512, cudaMemcpyHostToDevice);
    if (e == cudaSuccess) e = cudaMemcpy(g_tnl, nl.data(), 16, cudaMemcpyHostToDevice);
    if (e == cudaSuccess) e = cudaMalloc(reinterpret_cast<void**>(&g_g3xxs), sizeof(uint32_t) * 256);
    if (e == cudaSuccess) e = cudaMalloc(reinterpret_cast<void**>(&g_g2xxs), sizeof(uint64_t) * 256);
    if (e == cudaSuccess) e = cudaMalloc(reinterpret_cast<void**>(&g_g2xs), sizeof(uint64_t) * 512);
    if (e == cudaSuccess) e = cudaMalloc(reinterpret_cast<void**>(&g_sgn), 128);
    if (e == cudaSuccess) e = cudaMemcpy(g_g3xxs, quants::kIq3xxsGrid, sizeof(uint32_t) * 256, cudaMemcpyHostToDevice);
    if (e == cudaSuccess) e = cudaMemcpy(g_g2xxs, quants::kIq2xxsGrid, sizeof(uint64_t) * 256, cudaMemcpyHostToDevice);
    if (e == cudaSuccess) e = cudaMemcpy(g_g2xs, quants::kIq2xsGrid, sizeof(uint64_t) * 512, cudaMemcpyHostToDevice);
    if (e == cudaSuccess) e = cudaMemcpy(g_sgn, quants::kSignsIq2xs, 128, cudaMemcpyHostToDevice);
    // The IQ2_S grid only holds the byte values 8, 25 and 43: an entry is 8 codes of 0/1/2, i.e. a byte-permute
    // selector (one nibble per output byte) into the bytes {8, 25, 43} of a constant. Two selectors (elements
    // 0-3 and 4-7) per entry, 4 KB for the whole grid instead of 8.
    if (e == cudaSuccess) {
        std::vector<uint32_t> sel(1024);
        g_sel2_ok = true;
        for (int i = 0; i < 1024; ++i) {
            uint32_t lo = 0, hi = 0;
            for (int k = 0; k < 8; ++k) {
                const uint32_t v = static_cast<uint32_t>((quants::kIq2sGrid[i] >> (8 * k)) & 0xff);
                const uint32_t code = v == 8 ? 0u : v == 25 ? 1u : v == 43 ? 2u : 3u;
                if (code == 3u) g_sel2_ok = false;
                (k < 4 ? lo : hi) |= code << (4 * (k & 3));
            }
            sel[static_cast<size_t>(i)] = lo | (hi << 16);
        }
        e = cudaMalloc(reinterpret_cast<void**>(&g_sel2), sizeof(uint32_t) * 1024);
        if (e == cudaSuccess) e = cudaMemcpy(g_sel2, sel.data(), sizeof(uint32_t) * 1024, cudaMemcpyHostToDevice);
    }
    if (e != cudaSuccess) {
        if (err) *err = std::string("gemm_init: ") + cudaGetErrorString(e);
        return false;
    }
    return ext_init_a(err) && ext_init_b(err) && ext_init_c(err);
}

// The int8 path: F16, the formats of dispatch_type and the extension groups (not BF16, F32: the callers fall back to the GEMV).
bool gemm_supported(uint32_t type) {
    return type == static_cast<uint32_t>(Type::F16) || type == kQ8_0 || type == kQ6_K || type == kIQ2_S || type == kIQ3_S ||
           type == kIQ4_XS || mmq::ext_group(type) != 0;
}

namespace {

// One block of 32 values of one row, quantised by one warp (lane = position in the block): the block's amplitude as an
// integer of R levels, split into nt balanced base-128 digits (digit nt-1 in [-64, 63], the others likewise, digit 0 in
// [-127, 127]); nt = 1 is Q8_1. q is [nt][rows][K], sc is [rows][K/32].
template <int NT>
__device__ __forceinline__ void quant_block(float v, int lane, int64_t row, int b, int rows, int K, int8_t* q, float* sc,
                                            float* sums = nullptr) {
    constexpr float R = NT == 1 ? 127.0f : NT == 2 ? 16192.0f : 2000000.0f;
    const int nb = K / 32;
    float amax = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    if constexpr (NT == 1) {
        // Q8_1: d = amax / 127, q = round(x * (127 / amax)) with ties away from zero (roundf), the scale kept in
        // fp32, as the int8 tensor-core path of llama.cpp lays it out (one scale per 32 values).
        int qi = 0;
        if (amax > 0.f) qi = static_cast<int>(roundf(v * __fdiv_rn(127.0f, amax)));
        q[row * K + b * 32 + lane] = static_cast<int8_t>(qi);
        const float sx = amax > 0.f ? __fdiv_rn(amax, 127.0f) : 0.f;
        if (lane == 0) sc[row * nb + b] = sx;
        if (sums) {
            int s16 = qi;
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) s16 += __shfl_xor_sync(0xffffffffu, s16, o);
            if ((lane & 15) == 0) sums[row * (K / 16) + b * 2 + (lane >> 4)] = sx * static_cast<float>(s16);
        }
        return;
    }
    int X = 0;
    if (amax > 0.f) X = __float2int_rn(v * __fdiv_rn(R, amax));
    int8_t dig[NT];
    int rem = X;
#pragma unroll
    for (int t = NT - 1; t >= 1; --t) {
        const int d = ((rem + 64) & 127) - 64;
        dig[t] = static_cast<int8_t>(d);
        rem = (rem - d) >> 7;
    }
    dig[0] = static_cast<int8_t>(rem);
#pragma unroll
    for (int t = 0; t < NT; ++t) q[(static_cast<int64_t>(t) * rows + row) * K + b * 32 + lane] = dig[t];
    const float sx = amax > 0.f ? __fdiv_rn(amax, R) : 0.f;
    if (lane == 0) sc[row * nb + b] = sx;
    if (sums) {
        int s16 = X;
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) s16 += __shfl_xor_sync(0xffffffffu, s16, o);
        if ((lane & 15) == 0) sums[row * (K / 16) + b * 2 + (lane >> 4)] = sx * static_cast<float>(s16);
    }
}

// One warp per (row, block of 32).
template <int NT>
__global__ void act_quant_kernel(const float* x, int64_t x_stride, int rows, int K, int8_t* q, float* sc, float* sums) {
    const int nb = K / 32;
    const int64_t gw = static_cast<int64_t>(blockIdx.x) * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (gw >= static_cast<int64_t>(rows) * nb) return;
    const int64_t row = gw / nb;
    const int b = static_cast<int>(gw % nb);
    quant_block<NT>(x[row * x_stride + b * 32 + lane], lane, row, b, rows, K, q, sc, sums);
}

// rmsnorm_rows_kernel (ops_rows.cu) and the quantiser of its output in one pass: one block of 256 threads per row, the
// sum of squares in double over the same element order, y written as the norm does, and a warp quantises each block
// of 32 of y straight from the registers. Same values as the two kernels, one launch and one pass less.
template <int NT>
__global__ void __launch_bounds__(256) rmsnorm_quant_kernel(const float* x, const float* w, float* y, int n, float eps, int rows,
                                                            int8_t* q, float* sc) {
    __shared__ double sm[32];
    const float* xr = x + static_cast<int64_t>(blockIdx.x) * n;
    float* yr = y + static_cast<int64_t>(blockIdx.x) * n;
    double ss = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const double v = xr[i];
        ss += v * v;
    }
    ss = ops::block_sum(ss, sm);
    __shared__ float s_inv;
    if (threadIdx.x == 0) s_inv = static_cast<float>(1.0 / sqrt(ss / n + static_cast<double>(eps)));
    __syncthreads();
    const float inv = s_inv;
    for (int i = threadIdx.x, k = 0; i < n; i += blockDim.x, ++k) {
        const float v = xr[i] * inv * w[i];
        yr[i] = v;
        quant_block<NT>(v, threadIdx.x & 31, blockIdx.x, (i >> 5), rows, n, q, sc);
    }
}

}  // namespace

bool act_quant_sums(const float* x, int rows, int K, int nt, int8_t* q, float* sc, float* sums, cudaStream_t s) {
    if (rows <= 0 || K % 128 != 0) return false;
    const int64_t warps = static_cast<int64_t>(rows) * (K / 32);
    const unsigned grid = static_cast<unsigned>((warps + 7) / 8);
    switch (nt) {
        case 1: act_quant_kernel<1><<<grid, 256, 0, s>>>(x, K, rows, K, q, sc, sums); break;
        case 2: act_quant_kernel<2><<<grid, 256, 0, s>>>(x, K, rows, K, q, sc, sums); break;
        case 3: act_quant_kernel<3><<<grid, 256, 0, s>>>(x, K, rows, K, q, sc, sums); break;
        default: return false;
    }
    return cudaGetLastError() == cudaSuccess;
}

bool act_quant(const float* x, int rows, int K, int nt, int8_t* q, float* sc, cudaStream_t s) {
    return act_quant_sums(x, rows, K, nt, q, sc, nullptr, s);
}

bool gemm_needs_sums(uint32_t type) {
    return type == static_cast<uint32_t>(Type::Q4_K) || type == static_cast<uint32_t>(Type::Q5_K) || type == static_cast<uint32_t>(Type::Q2_K);
}

bool rmsnorm_quant(const float* x, const float* w, float* y, int rows, int n, float eps, int nt, int8_t* q, float* sc,
                   cudaStream_t s) {
    if (rows <= 0 || n % 256 != 0) return false;
    const unsigned grid = static_cast<unsigned>(rows);
    switch (nt) {
        case 1: rmsnorm_quant_kernel<1><<<grid, 256, 0, s>>>(x, w, y, n, eps, rows, q, sc); break;
        case 2: rmsnorm_quant_kernel<2><<<grid, 256, 0, s>>>(x, w, y, n, eps, rows, q, sc); break;
        case 3: rmsnorm_quant_kernel<3><<<grid, 256, 0, s>>>(x, w, y, n, eps, rows, q, sc); break;
        default: return false;
    }
    return cudaGetLastError() == cudaSuccess;
}

namespace {
// One launch of an extension format's kernel (gemm_ext_*.cu).
bool ext_launch(uint32_t type, const Args& a, int nt, int bm, int mode, dim3 grid, cudaStream_t s) {
    switch (mmq::ext_group(type)) {
        case 1: return ext_launch_a(type, a, nt, bm, mode, grid, s);
        case 2: return ext_launch_b(type, a, nt, bm, mode, grid, s);
        case 3: return ext_launch_c(type, a, nt, bm, mode, grid, s);
        default: return false;
    }
}
}  // namespace

// ---- decode products through the fragment-direct kernel: one weight decode serves 16 rows, int8 activation planes -------
// The path depends on format, width and stream only, never on the row count: a step and every window share the bits.
namespace {

__global__ void ksplit_reduce_kernel(const float* part, int S, int T, int n_out, float* y, int ldy, bool accumulate) {
    const int64_t n = static_cast<int64_t>(T) * n_out;
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = part[i];
    for (int z = 1; z < S; ++z) v += part[static_cast<int64_t>(z) * n + i];
    float* yp = y + (i / n_out) * ldy + i % n_out;
    *yp = accumulate ? *yp + v : v;
}

int env_int(const char* k, int d) {
    const char* v = std::getenv(k);
    return v && *v ? std::atoi(v) : d;
}
// LIMBO_FRAG=0: FP32 kernels only; LIMBO_FRAG_NT: activation planes (1..3)
const int g_frag_on = env_int("LIMBO_FRAG", 1);
const bool g_pf_frag = env_int("LIMBO_PF_FRAG", 1) != 0;   // prefill products on the fragment kernel
const int g_frag_nt = std::min(3, std::max(1, env_int("LIMBO_FRAG_NT", 2)));
const int g_frag_blocks = env_int("LIMBO_FRAG_BLOCKS", 272);   // blocks a product aims for before K is cut
constexpr int kFragMaxIn = 32768;                              // widest input (the DFlash2 fc: 25,600)
constexpr int kFragMaxCols = 16;
constexpr size_t kFragPart = size_t(1) << 20;                  // floats of K-slice sums

// a stream's scratch, made before any graph capture
struct FragScratch {
    cudaStream_t s = nullptr;
    int8_t* xq = nullptr;   // [3][16][kFragMaxIn]
    float* xs = nullptr;    // [16][kFragMaxIn / 32]
    float* xsum = nullptr;  // [16][kFragMaxIn / 16]: the formats with a minimum
    float* part = nullptr;  // [kFragPart]
    float* h = nullptr;     // [kFragMaxIn]: gemv_swiglu's silu(g) * u
};

// 16-row tiles, one stage buffer, 4 warps of 32 weight rows (the fastest of the four forms measured)
template <int MODE>
bool dispatch_window(int nt, uint32_t type, const Args& a, dim3 grid, cudaStream_t s) {
    return dispatch_frag<16, MODE, 1, 2>(nt, type, a, grid, s);
}
std::mutex g_frag_mu;
std::vector<FragScratch> g_frag;

}  // namespace

bool frag_reserve(cudaStream_t s, std::string* err) {
    if (g_frag_on <= 0) return true;
    std::lock_guard<std::mutex> lk(g_frag_mu);
    for (const FragScratch& f : g_frag)
        if (f.s == s) return true;
    FragScratch f;
    f.s = s;
    if (cudaMalloc(&f.xq, size_t(3) * kFragMaxCols * kFragMaxIn) != cudaSuccess ||
        cudaMalloc(&f.xs, size_t(kFragMaxCols) * (kFragMaxIn / 32) * 4) != cudaSuccess ||
        cudaMalloc(&f.xsum, size_t(kFragMaxCols) * (kFragMaxIn / 16) * 4) != cudaSuccess || cudaMalloc(&f.part, kFragPart * 4) != cudaSuccess ||
        cudaMalloc(&f.h, size_t(kFragMaxIn) * 4) != cudaSuccess) {
        cudaGetLastError();
        if (f.xq) cudaFree(f.xq);
        if (f.xs) cudaFree(f.xs);
        if (f.xsum) cudaFree(f.xsum);
        if (f.part) cudaFree(f.part);
        if (err) *err = "out of device memory for the window scratch";
        return false;
    }
    g_frag.push_back(f);
    return true;
}

void frag_release(cudaStream_t s) {
    std::lock_guard<std::mutex> lk(g_frag_mu);
    for (size_t i = 0; i < g_frag.size(); ++i)
        if (g_frag[i].s == s) {
            cudaStreamSynchronize(s);
            cudaFree(g_frag[i].xq);
            cudaFree(g_frag[i].xs);
            cudaFree(g_frag[i].xsum);
            cudaFree(g_frag[i].part);
            cudaFree(g_frag[i].h);
            g_frag.erase(g_frag.begin() + static_cast<std::ptrdiff_t>(i));
            return;
        }
}

namespace {
bool frag_scratch(uint32_t type, int n_in, cudaStream_t s, FragScratch* out) {
    if (g_frag_on <= 0 || n_in % 256 != 0 || n_in > kFragMaxIn || !g_t2s) return false;
    if (!frag_type(type)) return false;
    std::lock_guard<std::mutex> lk(g_frag_mu);
    for (const FragScratch& e : g_frag)   // (the legacy default stream is a null handle: found or not is its own answer)
        if (e.s == s) {
            if (out) *out = e;
            return true;
        }
    return false;   // a stream without scratch keeps the FP32 kernels
}
}  // namespace

bool frag_takes(uint32_t type, int n_in, cudaStream_t s) { return frag_scratch(type, n_in, s, nullptr); }

bool frag_swiglu(uint32_t type, const void* W, const float* g, const float* u, float* y, int n_out, int n_in, bool accumulate,
                 cudaStream_t s) {
    FragScratch f;
    if (!frag_scratch(type, n_in, s, &f)) return false;
    ops::swiglu(g, u, f.h, n_in, s);   // the T-row pass's own h kernel
    return frag_cols(type, W, f.h, n_in, y, n_out, n_out, n_in, 1, accumulate, s);
}

bool frag_cols(uint32_t type, const void* W, const float* x, int x_stride, float* y, int y_stride, int n_out, int n_in, int C,
               bool accumulate, cudaStream_t s) {
    if (C < 1 || C > kFragMaxCols || n_out <= 0) return false;
    FragScratch f;
    if (!frag_scratch(type, n_in, s, &f)) return false;
    const int nt = g_frag_nt;
    Args a = base_args(type, W, f.xq, f.xs, C, n_in, y, y_stride, n_out);
    a.T = C;
    float* xsum = gemm_needs_sums(type) ? f.xsum : nullptr;
    a.xsum = xsum;
    // from the shape alone, never from C: a column's sums are the same in every window
    const int rb = (n_out + kBN - 1) / kBN, nst = n_in / 256;
    int S = std::max(1, std::min(nst / 4, (g_frag_blocks + rb - 1) / rb));
    S = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(S, static_cast<int64_t>(kFragPart / (size_t(kFragMaxCols) * n_out)))));
    S = (nst + (nst + S - 1) / S - 1) / ((nst + S - 1) / S);   // no empty slices
    const int64_t warps = static_cast<int64_t>(C) * (n_in / 32);
    switch (nt) {
        case 1: act_quant_kernel<1><<<static_cast<unsigned>((warps + 7) / 8), 256, 0, s>>>(x, x_stride, C, n_in, f.xq, f.xs, xsum); break;
        case 2: act_quant_kernel<2><<<static_cast<unsigned>((warps + 7) / 8), 256, 0, s>>>(x, x_stride, C, n_in, f.xq, f.xs, xsum); break;
        default: act_quant_kernel<3><<<static_cast<unsigned>((warps + 7) / 8), 256, 0, s>>>(x, x_stride, C, n_in, f.xq, f.xs, xsum); break;
    }
    a.ksplit = S;
    a.part = S > 1 ? f.part : nullptr;
    if (S > 1) a.ldy = n_out;
    const dim3 grid(static_cast<unsigned>(rb), 1, static_cast<unsigned>(S));
    const bool acc = accumulate && S == 1;
    if (!(acc ? dispatch_window<kAcc>(nt, type, a, grid, s) : dispatch_window<kStore>(nt, type, a, grid, s))) return false;
    if (S == 1) return true;
    const int64_t n = static_cast<int64_t>(C) * n_out;
    ksplit_reduce_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, s>>>(f.part, S, C, n_out, y, y_stride, accumulate);
    return cudaGetLastError() == cudaSuccess;
}

bool gemm(uint32_t type, const void* W, const int8_t* xq, const float* xs, int nt, int T, float* y, int n_out,
          int n_in, bool accumulate, cudaStream_t s, const float* xsum) {
    if (!gemm_supported(type) || n_in % kBK != 0 || n_out <= 0 || T <= 0 || !g_t2s) return false;
    if (type == static_cast<uint32_t>(Type::F16)) return gemm_f16(W, xq, xs, nt, T, y, n_out, n_in, accumulate, s);
    Args a = base_args(type, W, xq, xs, T, n_in, y, n_out, n_out);
    a.T = T;
    a.xsum = xsum;
    if (nt < 1 || nt > 3) return false;
    // the fragment kernel for the formats it decodes (IQ3_S at 1024 rows: 1.7x the shared-tile kernel)
    if (n_in % 256 == 0 && g_pf_frag && frag_type(type) && (!gemm_needs_sums(type) || xsum)) {
        const int bm = nt == 3 ? 32 : 64;
        const dim3 grid(static_cast<unsigned>((n_out + kBN - 1) / kBN), static_cast<unsigned>((T + bm - 1) / bm));
        if (nt == 1) return accumulate ? dispatch_frag<64, kAcc, 1, 2>(nt, type, a, grid, s) : dispatch_frag<64, kStore, 1, 2>(nt, type, a, grid, s);
        if (nt == 2) return accumulate ? dispatch_dn<2, 64, kAcc>(type, a, T, n_out, s) : dispatch_dn<2, 64, kStore>(type, a, T, n_out, s);
        return accumulate ? dispatch_dn<3, 32, kAcc>(type, a, T, n_out, s) : dispatch_dn<3, 32, kStore>(type, a, T, n_out, s);
    }
    if (mmq::ext_group(type)) {
        if (gemm_needs_sums(type) && !xsum) return false;
        const int bm = nt == 3 ? 32 : 64;
        const dim3 grid(static_cast<unsigned>((n_out + kBN - 1) / kBN), static_cast<unsigned>((T + bm - 1) / bm));
        return ext_launch(type, a, nt, bm, accumulate ? kAcc : kStore, grid, s);
    }
    return accumulate ? dispatch_dense<kAcc>(nt, type, a, n_out, T, s) : dispatch_dense<kStore>(nt, type, a, n_out, T, s);
}

namespace {

constexpr int kMoeSmallMax = kMoeBmBig;  // an expert with up to this many pairs is cut into tiles of kMoeBmSmall
// (a threshold of 16 or 24 instead measured the same within 3 percent at 512 and 2048 tokens)

// One block: counts the pairs per expert, scans, places each pair, cuts the tiles.
__global__ void moe_build_kernel(const int32_t* ids, int T, int n_expert, int n_used, MoeLists L) {
    __shared__ int s_cnt[1024], s_start[1025], s_cur[1024], s_t0[1025], s_t1[1025];
    const int tid = threadIdx.x, total = T * n_used;
    for (int e = tid; e < n_expert; e += blockDim.x) {
        s_cnt[e] = 0;
        s_cur[e] = 0;
    }
    __syncthreads();
    for (int p = tid; p < total; p += blockDim.x) atomicAdd(&s_cnt[ids[p]], 1);
    __syncthreads();
    if (tid == 0) {
        int acc = 0, t0 = 0, t1 = 0;
        for (int e = 0; e < n_expert; ++e) {
            s_start[e] = acc;
            s_t0[e] = t0;
            s_t1[e] = t1;
            acc += s_cnt[e];
            if (s_cnt[e] <= kMoeSmallMax) t0 += (s_cnt[e] + kMoeBmSmall - 1) / kMoeBmSmall;
            else t1 += (s_cnt[e] + L.big_bm - 1) / L.big_bm;
        }
        s_start[n_expert] = acc;
        L.n_tiles[0] = t0;
        L.n_tiles[1] = t1;
    }
    __syncthreads();
    for (int e = tid; e < n_expert; e += blockDim.x) {
        L.cnt[e] = s_cnt[e];
        L.start[e] = s_start[e];
    }
    if (tid == 0) L.start[n_expert] = s_start[n_expert];
    for (int p = tid; p < total; p += blockDim.x) {
        const int e = ids[p];
        const int pos = s_start[e] + atomicAdd(&s_cur[e], 1);
        L.list_tok[pos] = p / n_used;
        L.pos_of[p] = pos;
    }
    for (int e = tid; e < n_expert; e += blockDim.x) {
        const int c = s_cnt[e];
        if (c == 0) continue;
        if (c <= kMoeSmallMax) {
            for (int m = 0; m * kMoeBmSmall < c; ++m)
                L.tiles_small[s_t0[e] + m] = make_int4(e, s_start[e] + m * kMoeBmSmall, min(kMoeBmSmall, c - m * kMoeBmSmall), 0);
        } else {
            const int bb = L.big_bm;
            for (int m = 0; m * bb < c; ++m)
                L.tiles_big[s_t1[e] + m] = make_int4(e, s_start[e] + m * bb, min(bb, c - m * bb), 0);
        }
    }
}

}  // namespace

bool moe_bm64_ok(uint32_t type, int K, uint64_t expert_bytes) {
    if (!gemm_supported(type) || K % kBK != 0 || !g_t2s) return false;
    if (mmq::ext_group(type)) return true;
    Args a = base_args(type, nullptr, nullptr, nullptr, 1, K, nullptr, 1, 1);
    a.expert_bytes = expert_bytes;
    return moe_fast_ok(type, a);
}

bool moe_build(const int32_t* ids, int T, int n_expert, int n_used, const MoeLists& L, cudaStream_t s) {
    if (n_expert > 1024 || T <= 0 || (L.big_bm != kMoeBmBig && L.big_bm != 64)) return false;
    moe_build_kernel<<<1, 1024, 0, s>>>(ids, T, n_expert, n_used, L);
    return cudaGetLastError() == cudaSuccess;
}

bool moe_gate_up(uint32_t type, const void* Wg, const void* Wu, uint64_t expert_bytes, const int8_t* xq,
                 const float* xs, int nt, int T, int n_pairs, const MoeLists& L, int n_expert, float* h, int n_ff,
                 int K, cudaStream_t s, const float* xsum) {
    if (!gemm_supported(type) || K % kBK != 0 || n_ff % 64 != 0 || !g_t2s) return false;
    Args a = base_args(type, Wg, xq, xs, T, K, h, n_ff, n_ff);
    a.W1 = static_cast<const uint8_t*>(Wu);
    a.expert_bytes = expert_bytes;
    a.eptr = L.eptr;
    a.xrow_list = L.list_tok;
    a.xsum = xsum;
    const dim3 gs(static_cast<unsigned>(n_ff / 64), static_cast<unsigned>(moe_max_tiles_small(n_expert)));
    const dim3 gb(static_cast<unsigned>(n_ff / 64), static_cast<unsigned>(moe_max_tiles_big(n_pairs, n_expert)));
    if (mmq::ext_group(type)) {
        if (gemm_needs_sums(type) && !xsum) return false;
        a.tiles = L.tiles_small;
        a.n_tiles = L.n_tiles;
        if (!ext_launch(type, a, nt, kMoeBmSmall, kGateUp, gs, s)) return false;
        a.tiles = L.tiles_big;
        a.n_tiles = L.n_tiles + 1;
        return ext_launch(type, a, nt, L.big_bm, kGateUp, gb, s);
    }
    if (moe_fast_ok(type, a)) {
        a.tiles = L.tiles_small;
        a.n_tiles = L.n_tiles;
        if (!dispatch_moe<kMoeBmSmall, kGateUp, 1>(nt, type, a, gs, s)) return false;
        a.tiles = L.tiles_big;
        a.n_tiles = L.n_tiles + 1;
        return L.big_bm == 64 ? dispatch_moe<64, kGateUp, 1>(nt, type, a, gb, s) : dispatch_moe<kMoeBmBig, kGateUp, 1>(nt, type, a, gb, s);
    }
    // the generic kernel handles any tile size: one pass over the big class' tiles of 16 would need a third list,
    // so this path is reserved for the 16-wide small class and 32-wide big class through the same kernel
    a.tiles = L.tiles_small;
    a.n_tiles = L.n_tiles;
    if (!dispatch<kMoeBmSmall, kGateUp>(nt, type, a, gs, s)) return false;
    a.tiles = L.tiles_big;
    a.n_tiles = L.n_tiles + 1;
    if (L.big_bm != kMoeBmBig) return false;
    return dispatch<kMoeBmBig, kGateUp>(nt, type, a, gb, s);
}

bool moe_down(uint32_t type, const void* Wd, uint64_t expert_bytes, const int8_t* hq, const float* hs, int nt,
              int n_pairs, const MoeLists& L, int n_expert, float* dn, int n_out, int K, cudaStream_t s,
              const float* hsum) {
    if (!gemm_supported(type) || K % kBK != 0 || !g_t2s) return false;
    Args a = base_args(type, Wd, hq, hs, n_pairs, K, dn, n_out, n_out);
    a.xsum = hsum;
    a.expert_bytes = expert_bytes;
    a.eptr = L.eptr;
    const dim3 gs(static_cast<unsigned>((n_out + kBN - 1) / kBN), static_cast<unsigned>(moe_max_tiles_small(n_expert)));
    const dim3 gb(static_cast<unsigned>((n_out + kBN - 1) / kBN), static_cast<unsigned>(moe_max_tiles_big(n_pairs, n_expert)));
    if (mmq::ext_group(type)) {
        if (gemm_needs_sums(type) && !hsum) return false;
        a.tiles = L.tiles_small;
        a.n_tiles = L.n_tiles;
        if (!ext_launch(type, a, nt, kMoeBmSmall, kStore, gs, s)) return false;
        a.tiles = L.tiles_big;
        a.n_tiles = L.n_tiles + 1;
        return ext_launch(type, a, nt, L.big_bm, kStore, gb, s);
    }
    if (moe_fast_ok(type, a)) {
        a.tiles = L.tiles_small;
        a.n_tiles = L.n_tiles;
        if (!dispatch_moe<kMoeBmSmall, kStore, 1>(nt, type, a, gs, s)) return false;
        a.tiles = L.tiles_big;
        a.n_tiles = L.n_tiles + 1;
        return L.big_bm == 64 ? dispatch_moe<64, kStore, 1>(nt, type, a, gb, s) : dispatch_moe<kMoeBmBig, kStore, 1>(nt, type, a, gb, s);
    }
    a.tiles = L.tiles_small;
    a.n_tiles = L.n_tiles;
    if (!dispatch<kMoeBmSmall, kStore>(nt, type, a, gs, s)) return false;
    a.tiles = L.tiles_big;
    a.n_tiles = L.n_tiles + 1;
    if (L.big_bm != kMoeBmBig) return false;
    return dispatch<kMoeBmBig, kStore>(nt, type, a, gb, s);
}

bool moe_rows(uint32_t type, const void* W, uint64_t expert_bytes, const int8_t* xq, const float* xs, int nt, int T,
              int n_pairs, const MoeLists& L, int n_expert, float* y, int n_out, int K, cudaStream_t s,
              const float* xsum) {
    if (!gemm_supported(type) || K % kBK != 0 || !g_t2s) return false;
    Args a = base_args(type, W, xq, xs, T, K, y, n_out, n_out);
    a.xsum = xsum;
    a.xrow_list = L.list_tok;
    a.expert_bytes = expert_bytes;
    a.eptr = L.eptr;
    const dim3 gs(static_cast<unsigned>((n_out + kBN - 1) / kBN), static_cast<unsigned>(moe_max_tiles_small(n_expert)));
    const dim3 gb(static_cast<unsigned>((n_out + kBN - 1) / kBN), static_cast<unsigned>(moe_max_tiles_big(n_pairs, n_expert)));
    if (mmq::ext_group(type)) {
        if (gemm_needs_sums(type) && !xsum) return false;
        a.tiles = L.tiles_small;
        a.n_tiles = L.n_tiles;
        if (!ext_launch(type, a, nt, kMoeBmSmall, kStore, gs, s)) return false;
        a.tiles = L.tiles_big;
        a.n_tiles = L.n_tiles + 1;
        return ext_launch(type, a, nt, L.big_bm, kStore, gb, s);
    }
    if (moe_fast_ok(type, a)) {
        a.tiles = L.tiles_small;
        a.n_tiles = L.n_tiles;
        if (!dispatch_moe<kMoeBmSmall, kStore, 1>(nt, type, a, gs, s)) return false;
        a.tiles = L.tiles_big;
        a.n_tiles = L.n_tiles + 1;
        return L.big_bm == 64 ? dispatch_moe<64, kStore, 1>(nt, type, a, gb, s) : dispatch_moe<kMoeBmBig, kStore, 1>(nt, type, a, gb, s);
    }
    a.tiles = L.tiles_small;
    a.n_tiles = L.n_tiles;
    if (!dispatch<kMoeBmSmall, kStore>(nt, type, a, gs, s)) return false;
    a.tiles = L.tiles_big;
    a.n_tiles = L.n_tiles + 1;
    if (L.big_bm != kMoeBmBig) return false;
    return dispatch<kMoeBmBig, kStore>(nt, type, a, gb, s);
}

namespace {

// y[t][n] for a 32-row tile and a 16-token tile: warp w owns weight rows n0+4w..n0+4w+3; lane L keeps one partial
// sum per (row, token) over its elements 8L..8L+7 of every 256 columns, then the shuffle tree of gemv.cu. A lane's
// x elements are read once for all four rows, and the tile of x sits in shared memory as float4 slots
// [token][half][lane] so that the 16-byte reads of a warp are consecutive (no bank conflicts).
constexpr int kF32Rows = 4, kF32Tok = 16;
__global__ void __launch_bounds__(256) gemm_f32_kernel(const float* W, const float* x, float* y, int T, int n_out,
                                                       int n_in) {
    extern __shared__ float4 sx4[];  // [kF32Tok][64]
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = blockIdx.x * (8 * kF32Rows) + warp * kF32Rows, t0 = blockIdx.y * kF32Tok;
    float acc[kF32Rows][kF32Tok];
#pragma unroll
    for (int r = 0; r < kF32Rows; ++r)
#pragma unroll
        for (int i = 0; i < kF32Tok; ++i) acc[r][i] = 0.f;
    const int nsb = n_in / 256;
    for (int s = 0; s < nsb; ++s) {
        __syncthreads();
        for (int i = threadIdx.x; i < kF32Tok * 64; i += 256) {
            const int tk = i >> 6, c4 = i & 63;  // c4 = 2 * lane + half
            const int t = min(t0 + tk, T - 1);
            sx4[tk * 64 + (c4 & 1) * 32 + (c4 >> 1)] = *reinterpret_cast<const float4*>(x + static_cast<int64_t>(t) * n_in + s * 256 + c4 * 4);
        }
        __syncthreads();
        float w[kF32Rows][8];
#pragma unroll
        for (int r = 0; r < kF32Rows; ++r) {
            const int row = min(row0 + r, n_out - 1);
            const float4* wv = reinterpret_cast<const float4*>(W + static_cast<int64_t>(row) * n_in + s * 256 + lane * 8);
            const float4 w0 = __ldg(wv), w1 = __ldg(wv + 1);
            w[r][0] = w0.x; w[r][1] = w0.y; w[r][2] = w0.z; w[r][3] = w0.w;
            w[r][4] = w1.x; w[r][5] = w1.y; w[r][6] = w1.z; w[r][7] = w1.w;
        }
#pragma unroll
        for (int tk = 0; tk < kF32Tok; ++tk) {
            const float4 x0 = sx4[tk * 64 + lane], x1 = sx4[tk * 64 + 32 + lane];
#pragma unroll
            for (int r = 0; r < kF32Rows; ++r) {
                float a = acc[r][tk];
                a += w[r][0] * x0.x;
                a += w[r][1] * x0.y;
                a += w[r][2] * x0.z;
                a += w[r][3] * x0.w;
                a += w[r][4] * x1.x;
                a += w[r][5] * x1.y;
                a += w[r][6] * x1.z;
                a += w[r][7] * x1.w;
                acc[r][tk] = a;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < kF32Rows; ++r) {
        if (row0 + r >= n_out) continue;
#pragma unroll
        for (int tk = 0; tk < kF32Tok; ++tk) {
            float v = acc[r][tk];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0 && t0 + tk < T) y[static_cast<int64_t>(t0 + tk) * n_out + row0 + r] = v;
        }
    }
}

}  // namespace

bool gemm_f32(const float* W, const float* x, float* y, int T, int n_out, int n_in, cudaStream_t s) {
    if (n_in % 256 != 0 || T <= 0 || n_out <= 0) return false;
    constexpr size_t smem = kF32Tok * 64 * sizeof(float4);
    const dim3 grid(static_cast<unsigned>((n_out + 8 * kF32Rows - 1) / (8 * kF32Rows)), static_cast<unsigned>((T + kF32Tok - 1) / kF32Tok));
    gemm_f32_kernel<<<grid, 256, smem, s>>>(W, x, y, T, n_out, n_in);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace limbo::cuda
