// Copyright (c) 2026 ITERFORM. All rights reserved.
// src/cuda/gpu_pages.cu — the paged K/V pool on the device (docs/paged-kv.md): taking pages for the positions a pass writes, copy on
// write, the table upload, cutting and sharing. The bookkeeping itself (tables, refcounts) is limbo::kv::PagePool; this file does
// what it asks of the device. Everything is queued on the engine's stream, so a copy or an upload is ordered after the passes
// that read the old page or table and before the ones that read the new.
#include <algorithm>
#include <vector>

#include "gpu_impl.hpp"
#include "limbo/cuda/kv.hpp"

namespace limbo::cuda {

const int* Gpu::Impl::pt() const { return st.slots.empty() ? nullptr : st.slots[size_t(st.cur)].d_pt; }

// One page, every plane of every full-attention layer, dst <- src.
void Gpu::Impl::page_copy(int dst, int src) {
    ++n_page_copies;
    const KvPlanes pg = kv_planes(kv.fmt, 256, 256, c.n_head_kv, c.hd);
    auto cp = [&](void* base, uint64_t bytes) {
        if (!base || !bytes) return;
        uint8_t* b = static_cast<uint8_t*>(base);
        cudaMemcpyAsync(b + uint64_t(dst) * bytes, b + uint64_t(src) * bytes, bytes, cudaMemcpyDeviceToDevice, stream);
    };
    for (size_t l = 0; l < L.size(); ++l) {
        if (!L[l].full) continue;
        if (kv.fmt == KvFormat::F16) {
            cp(kc[l], pg.k);
            cp(vc[l], pg.v);
        } else {
            cp(kv.kq[l], pg.k);
            cp(kv.vq[l], pg.v);
            cp(kv.ks[l], pg.ks);
            cp(kv.vs[l], pg.vs);
        }
    }
}

// The live slot's table to the device through a staging ring (a slot is reused only after the copy that read it has run).
bool Gpu::Impl::sync_pt() {
    if (!pool || !pool->take_dirty(st.cur)) return true;
    const int n = n_chunks() + 1;
    const int r = pt_ring++ % kPtRing;
    if (cudaEventSynchronize(pt_ev[r]) != cudaSuccess) return false;
    int32_t* host = pt_pin + size_t(r) * size_t(n);
    const std::vector<int32_t>& t = pool->table(st.cur);
    std::copy(t.begin(), t.end(), host);
    host[n - 1] = -1;
    if (cudaMemcpyAsync(st.slots[size_t(st.cur)].d_pt, host, size_t(n) * sizeof(int32_t), cudaMemcpyHostToDevice, stream) != cudaSuccess) return false;
    return cudaEventRecord(pt_ev[r], stream) == cudaSuccess;
}

bool Gpu::Impl::pages_for(int lo, int hi) {
    if (!pool || st.slots.empty()) return false;
    std::vector<kv::PageCopy> cp;
    const bool ok = pool->write(st.cur, lo, hi, &cp);
    for (const kv::PageCopy& x : cp) page_copy(x.dst, x.src);
    return sync_pt() && ok;
}

void Gpu::slot_cut(int keep) {
    Impl& I = *p_;
    if (!I.pool || I.st.slots.empty()) return;
    I.pool->cut(I.st.cur, keep);
    I.sync_pt();
    if (I.df) I.df->cut_slot(I.st.cur, keep);
}

bool Gpu::slot_reserve(int n) {
    Impl& I = *p_;
    if (!I.pool || I.st.slots.empty() || n < 0 || n > I.n_ctx) return false;
    const bool ok = I.pool->fill(I.st.cur, n);
    return I.sync_pt() && ok;
}

bool Gpu::slot_share(int from, int n) {
    Impl& I = *p_;
    if (!I.pool || from < 0 || from >= int(I.st.slots.size()) || from == I.st.cur || n < 1 || n > I.n_ctx) return false;
    std::vector<kv::PageCopy> cp;
    if (!I.pool->share(I.st.cur, from, n, &cp)) return false;
    for (const kv::PageCopy& x : cp) I.page_copy(x.dst, x.src);
    if (I.df && !I.df->share_slot(from, I.st.cur, n)) I.df->reset_slot(I.st.cur);  // the drafter's context follows the pages when it can
    return I.sync_pt();
}

int Gpu::slot_room(int at) const {
    const Impl& I = *p_;
    return I.pool ? I.pool->room(I.st.cur, at) : 0;
}

void Gpu::release_slot(int slot) {
    Impl& I = *p_;
    if (!I.pool || slot < 0 || slot >= int(I.st.slots.size())) return;
    I.pool->cut(slot, 0);
    if (slot == I.st.cur) I.sync_pt();
    if (I.df) I.df->reset_slot(slot);
}

Gpu::PoolStats Gpu::pool_stats() const {
    const Impl& I = *p_;
    PoolStats s;
    if (!I.pool) return s;
    s.pages = I.pool->pages();
    s.used = I.pool->used_pages();
    s.shared = I.pool->shared_pages();
    s.page_bytes = I.page_bytes(I.kv.fmt);
    s.copies = I.n_page_copies;
    return s;
}

int Gpu::slot_pages(int slot) const {
    const Impl& I = *p_;
    return I.pool && slot >= 0 && slot < int(I.st.slots.size()) ? I.pool->held(slot) : 0;
}

int Gpu::page_of(int slot, int chunk) const {
    const Impl& I = *p_;
    if (!I.pool || slot < 0 || slot >= int(I.st.slots.size()) || chunk < 0 || chunk >= I.pool->n_chunks()) return -1;
    return I.pool->table(slot)[size_t(chunk)];
}

int Gpu::page_refs(int page) const { return p_->pool ? p_->pool->refs(page) : 0; }

}  // namespace limbo::cuda
