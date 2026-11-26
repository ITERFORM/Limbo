// Copyright (c) 2026 ITERFORM. All rights reserved.
// src/cuda/gpu_kv.cu — the K/V pool's planes in each format (F16, Q8, Q4), their allocation and the format switch.
// Layouts: include/limbo/cuda/kv.hpp; the pages and slots on top: gpu_pages.cu, gpu_state.cu.
#include <algorithm>

#include "gpu_impl.hpp"
#include "limbo/cuda/kv.hpp"

namespace limbo::cuda {

namespace {

// grid (n_pool_positions, n_head_kv), block 256: one K row and one V row (gathered from the page-transposed f16 layout) become Q8.
// The Q8 pool has the f16 pool's pages: page p stays page p.
__global__ void kv_convert_k(const __half* kc, const __half* vc, int8_t* kq, uint16_t* ks, int8_t* vq, uint16_t* vs, int n_head_kv, int hd) {
    const int pos = blockIdx.x, kh = blockIdx.y, d = threadIdx.x;
    const int64_t row = static_cast<int64_t>(pos) * n_head_kv + kh;
    uint16_t dh;
    int8_t q = kv_quant_lane(__half2float(kc[row * hd + d]), &dh);
    kq[row * hd + d] = q;
    if ((d & 31) == 0) ks[row * (hd / kKvBlock) + (d >> 5)] = dh;
    q = kv_quant_lane(__half2float(vc[static_cast<int64_t>(pos >> 8) * n_head_kv * hd * 256 + (static_cast<int64_t>(kh) * hd + d) * 256 + (pos & 255)]), &dh);
    vq[row * hd + d] = q;
    if ((d & 31) == 0) vs[row * (hd / kKvBlock) + (d >> 5)] = dh;
}

// rb: bytes of codes per row (hd for Q8, hd / 2 for Q4)
__global__ void kv_fill_k(int8_t* q, uint16_t* sc, int64_t n_rows, int hd, int rb, uint32_t seed) {
    const int64_t row = blockIdx.x;
    if (row >= n_rows) return;
    uint32_t h = static_cast<uint32_t>(row) * 2654435761u ^ seed;
    for (int d = threadIdx.x; d < rb; d += blockDim.x) {
        uint32_t x = h + static_cast<uint32_t>(d) * 40503u;
        x ^= x >> 15;
        x *= 2246822519u;
        x ^= x >> 13;
        q[row * rb + d] = static_cast<int8_t>(static_cast<int>(x & 0xff) - 128 + ((x >> 8) & 1));
    }
    if (threadIdx.x < hd / kKvBlock) sc[row * (hd / kKvBlock) + threadIdx.x] = __half_as_ushort(__float2half_rn(0.004f + 0.0001f * float((h >> (3 + threadIdx.x)) & 15)));
}

}  // namespace

bool Gpu::kv_fill_synthetic(int n, uint32_t seed) {
    Impl& I = *p_;
    if (I.kv.fmt == KvFormat::F16 || n < 0 || n > I.n_ctx || !I.pool) return false;
    if (!slot_reserve(n) || !I.sync_pt()) return false;
    const int nkv = I.c.n_head_kv, rb = I.kv.fmt == KvFormat::Q8 ? I.c.hd : I.c.hd / 2;
    const std::vector<int32_t>& tab = I.pool->table(I.st.cur);
    // the pages in runs of consecutive ids (a fresh pool hands them out in order: one run), 256 * n_head_kv rows to a page
    for (int c0 = 0; c0 * 256 < n;) {
        int c1 = c0;
        while ((c1 + 1) * 256 < n && tab[size_t(c1 + 1)] == tab[size_t(c1)] + 1) ++c1;
        const int64_t rows = static_cast<int64_t>(std::min(n, (c1 + 1) * 256) - c0 * 256) * nkv, row0 = static_cast<int64_t>(tab[size_t(c0)]) * 256 * nkv;
        for (size_t l = 0; l < I.L.size(); ++l) {
            if (!I.L[l].full || rows == 0) continue;
            kv_fill_k<<<static_cast<unsigned>(rows), 256, 0, I.stream>>>(I.kv.kq[l] + row0 * rb, I.kv.ks[l] + row0 * (I.c.hd / kKvBlock), rows, I.c.hd, rb,
                                                                       seed + static_cast<uint32_t>(l) * 7919u + static_cast<uint32_t>(c0));
            kv_fill_k<<<static_cast<unsigned>(rows), 256, 0, I.stream>>>(I.kv.vq[l] + row0 * rb, I.kv.vs[l] + row0 * (I.c.hd / kKvBlock), rows, I.c.hd, rb,
                                                                       seed * 31u + static_cast<uint32_t>(l) * 104729u + static_cast<uint32_t>(c0));
        }
        c0 = c1 + 1;
    }
    return cudaStreamSynchronize(I.stream) == cudaSuccess;
}

KvFormat Gpu::kv_format() const { return p_->kv.fmt; }

uint64_t Gpu::kv_cache_bytes() const { return p_->kv_bytes(p_->kv.fmt); }

KvPlanes Gpu::Impl::pool_planes(KvFormat f, int pages) const {
    KvPlanes pg = kv_planes(f, 256, 256, c.n_head_kv, c.hd);  // one page (an f16 V page has 256 positions per row)
    const uint64_t n = static_cast<uint64_t>(pages);
    return {pg.k * n, pg.v * n, pg.ks * n, pg.vs * n};
}

uint64_t Gpu::Impl::page_bytes(KvFormat f) const {
    uint64_t n_full = 0;
    for (const Layer& y : L) n_full += y.full ? 1 : 0;
    return n_full * kv_planes(f, 256, 256, c.n_head_kv, c.hd).total();
}

uint64_t Gpu::Impl::kv_bytes(KvFormat f) const { return page_bytes(f) * static_cast<uint64_t>(pool_pages); }

uint64_t Gpu::kv_cache_bytes(KvFormat f) const { return p_->kv_bytes(f); }

uint64_t Gpu::kv_page_bytes(KvFormat f) const { return p_->page_bytes(f); }

int Gpu::pool_pages() const { return p_->pool_pages; }

bool Gpu::Impl::alloc_kv(KvFormat f, int pages) {
    const size_t n = L.size();
    kc.assign(n, nullptr);
    vc.assign(n, nullptr);
    kv.kq.assign(n, nullptr);
    kv.vq.assign(n, nullptr);
    kv.ks.assign(n, nullptr);
    kv.vs.assign(n, nullptr);
    kv.fmt = f;
    pool_pages = pages;
    const KvPlanes pl = pool_planes(f, pages);
    for (size_t l = 0; l < n; ++l) {
        if (!L[l].full) continue;
        bool ok;
        if (f == KvFormat::F16) {
            kc[l] = dalloc<__half>(pl.k / 2);
            vc[l] = dalloc<__half>(pl.v / 2);
            ok = kc[l] && vc[l];
        } else {
            kv.kq[l] = dalloc<int8_t>(pl.k);
            kv.vq[l] = dalloc<int8_t>(pl.v);
            kv.ks[l] = dalloc<uint16_t>(pl.ks / 2);
            kv.vs[l] = dalloc<uint16_t>(pl.vs / 2);
            ok = kv.kq[l] && kv.vq[l] && kv.ks[l] && kv.vs[l];
        }
        if (!ok) {
            cudaGetLastError();
            free_kv();
            return false;
        }
    }
    // zero codes and scales (a Q4 zero code reads as -8 x scale 0): a page never holds NaN bits, whatever it held last is finite
    for (size_t l = 0; l < n; ++l) {
        if (!L[l].full) continue;
        if (f == KvFormat::F16) {
            cudaMemsetAsync(kc[l], 0, pl.k, stream);
            cudaMemsetAsync(vc[l], 0, pl.v, stream);
        } else {
            cudaMemsetAsync(kv.kq[l], f == KvFormat::Q4 ? 0x88 : 0, pl.k, stream);
            cudaMemsetAsync(kv.vq[l], f == KvFormat::Q4 ? 0x88 : 0, pl.v, stream);
            cudaMemsetAsync(kv.ks[l], 0, pl.ks, stream);
            cudaMemsetAsync(kv.vs[l], 0, pl.vs, stream);
        }
    }
    pool = std::make_unique<kv::PagePool>(pages, n_chunks(), kMaxSlots);
    return cudaStreamSynchronize(stream) == cudaSuccess;
}

void Gpu::Impl::free_kv() {
    cudaStreamSynchronize(stream);
    auto drop = [&](void* q, uint64_t bytes) {
        if (!q) return;
        auto it = std::find(owned.begin(), owned.end(), q);
        if (it != owned.end()) {
            owned.erase(it);
            owned_bytes -= std::min(owned_bytes, bytes);
        }
        cudaFree(q);
    };
    const KvPlanes pl = pool_planes(kv.fmt, pool_pages);
    for (size_t l = 0; l < L.size(); ++l) {
        if (l < kc.size()) drop(kc[l], pl.k);
        if (l < vc.size()) drop(vc[l], pl.v);
        if (l < kv.kq.size()) drop(kv.kq[l], pl.k);
        if (l < kv.vq.size()) drop(kv.vq[l], pl.v);
        if (l < kv.ks.size()) drop(kv.ks[l], pl.ks);
        if (l < kv.vs.size()) drop(kv.vs[l], pl.vs);
    }
    std::fill(kc.begin(), kc.end(), nullptr);
    std::fill(vc.begin(), vc.end(), nullptr);
    std::fill(kv.kq.begin(), kv.kq.end(), nullptr);
    std::fill(kv.vq.begin(), kv.vq.end(), nullptr);
    std::fill(kv.ks.begin(), kv.ks.end(), nullptr);
    std::fill(kv.vs.begin(), kv.vs.end(), nullptr);
}

void Gpu::Impl::drop_graphs() {
    cudaStreamSynchronize(stream);
    for (int v = 0; v < 2; ++v) {
        if (gexec[v]) cudaGraphExecDestroy(gexec[v]);
        gexec[v] = nullptr;
        warm[v] = 0;
    }
    for (int slot = 0; slot < 40; ++slot) {
        if (rw.gx[slot]) cudaGraphExecDestroy(rw.gx[slot]);
        rw.gx[slot] = nullptr;
        rw.warm[slot] = 0;
    }
    for (size_t i = 0; i < st.slots.size(); ++i) {  // the stashed slots' (the live slot's entry is empty)
        if (int(i) == st.cur) continue;
        Slot& sl = st.slots[i];
        for (int v = 0; v < 2; ++v) {
            if (sl.gexec[v]) cudaGraphExecDestroy(sl.gexec[v]);
            sl.gexec[v] = nullptr;
            sl.warm[v] = 0;
        }
        for (int g = 0; g < 40; ++g) {
            if (sl.rgx[g]) cudaGraphExecDestroy(sl.rgx[g]);
            sl.rgx[g] = nullptr;
            sl.rwarm[g] = 0;
        }
    }
}

bool Gpu::set_pool(KvFormat f, int pages) {
    Impl& I = *p_;
    if (pages < 1) return false;
    // Nothing to keep: the old planes go before the new ones are made (never both at once), every slot's pages are gone.
    const KvFormat old = I.kv.fmt;
    const int old_pages = I.pool_pages;
    I.drop_graphs();
    I.free_kv();
    const bool ok = I.alloc_kv(f, pages) || I.alloc_kv(old, old_pages);
    I.drop_graphs();
    I.state_sizes();
    reset();
    return ok && I.kv.fmt == f;
}

bool Gpu::set_kv_format(KvFormat f, int n_filled) {
    Impl& I = *p_;
    if (f == I.kv.fmt) return true;
    if (I.st.slots.size() > 1) return false;
    if (n_filled == 0) return set_pool(f, I.pool_pages);
    if (f != KvFormat::Q8 || I.kv.fmt != KvFormat::F16) return false;
    // F16 -> Q8: the same pages in a new pool, every pool position converted (free pages hold finite stale rows: harmless)
    const Cfg& c = I.c;
    cudaStreamSynchronize(I.stream);
    const size_t layers = I.L.size();
    std::vector<int8_t*> kq(layers, nullptr), vq(layers, nullptr);
    std::vector<uint16_t*> ks(layers, nullptr), vs(layers, nullptr);
    const KvPlanes q8 = I.pool_planes(KvFormat::Q8, I.pool_pages);
    for (size_t l = 0; l < layers; ++l) {
        if (!I.L[l].full) continue;
        kq[l] = I.dalloc<int8_t>(q8.k);
        vq[l] = I.dalloc<int8_t>(q8.v);
        ks[l] = I.dalloc<uint16_t>(q8.ks / 2);
        vs[l] = I.dalloc<uint16_t>(q8.vs / 2);
        if (!kq[l] || !vq[l] || !ks[l] || !vs[l]) return false;
    }
    const unsigned n_pos = static_cast<unsigned>(I.pool_pages) * 256u;
    for (size_t l = 0; l < layers; ++l) {
        if (!I.L[l].full) continue;
        kv_convert_k<<<dim3(n_pos, static_cast<unsigned>(c.n_head_kv)), 256, 0, I.stream>>>(I.kc[l], I.vc[l], kq[l], ks[l], vq[l], vs[l], c.n_head_kv, c.hd);
    }
    if (cudaStreamSynchronize(I.stream) != cudaSuccess) return false;
    // the f16 pool goes, and every captured graph with it
    I.drop_graphs();
    I.free_kv();
    I.kv.kq = kq;
    I.kv.vq = vq;
    I.kv.ks = ks;
    I.kv.vs = vs;
    I.kv.fmt = KvFormat::Q8;
    I.state_sizes();
    return true;
}

}  // namespace limbo::cuda
