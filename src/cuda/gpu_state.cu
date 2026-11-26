// Copyright (c) 2026 ITERFORM. All rights reserved.
// src/cuda/gpu_state.cu — the conversation's device state: recurrent-state snapshots, rewind and slots.
// Owner: the tape agent. See the state API in include/limbo/cuda/gpu.hpp and the state section of gpu_impl.hpp.
//
// A snapshot is every delta-net layer's S (32 heads x 128 x 128 fp32) and conv history (3 rows of the qkv
// width), packed layer after layer into one VRAM block per checkpoint slot. The full-attention layers' K/V is positional
// and never copied: a rewind to position p restores the snapshot taken at p and reads on from p, and the rows
// past p are overwritten as the next pass writes them (nothing attends past the position it is at). A snapshot belongs to no
// slot: restore() writes it into whichever slot is live, which is how a slot adopts another's prefix (gpu_pages.cu shares the pages).
//
// A slot is one conversation's state (page table + recurrent state); the live one is the members of Gpu::Impl the
// kernels read, the others are stashed in st.slots, and select_slot() swaps pointers (no copy). A captured
// decode graph bakes the state pointers and the page table in, so the graphs travel with their slot.
#include <algorithm>
#include <cstdint>
#include <utility>

#include "gpu_impl.hpp"
#include "limbo/cuda/kv.hpp"

namespace limbo::cuda {

namespace {

constexpr uint64_t kAlign = 256;
uint64_t up(uint64_t v) { return (v + kAlign - 1) / kAlign * kAlign; }

struct Sizes {
    uint64_t S = 0, hist = 0;  // bytes of one layer's S and conv history
    int n_gdn = 0, n_full = 0;
};

// A template only because Gpu::Impl is private: the type is deduced, never named here.
template <class I_>
Sizes sizes_of(const I_& I) {
    const Cfg& c = I.c;
    Sizes z;
    z.S = uint64_t(c.n_v_head) * uint64_t(c.gdn_head) * uint64_t(c.gdn_head) * sizeof(float);
    z.hist = uint64_t(c.conv_k - 1) * uint64_t(c.qkv_w) * sizeof(float);
    for (const Layer& y : I.L) (y.full ? z.n_full : z.n_gdn)++;
    return z;
}

}  // namespace

// ---- teardown and sizes ----------------------------------------------------------------------------

Gpu::Impl::State::~State() {
    for (void* p : snap)
        if (p) cudaFree(p);
    for (Slot& h : slots) {
        for (cudaGraphExec_t& g : h.gexec)
            if (g) cudaGraphExecDestroy(g);  // only a stashed slot holds graphs here; the live one's are Impl's
        for (cudaGraphExec_t& g : h.rgx)
            if (g) cudaGraphExecDestroy(g);
        if (h.d_pt) cudaFree(h.d_pt);
        if (h.blob) cudaFree(h.blob);
    }
}

bool Gpu::Impl::state_ready() {
    if (!st.slots.empty()) return true;
    if (S.empty()) return false;
    state_sizes();
    st.slots.resize(1);
    st.cur = 0;
    if (cudaMalloc(reinterpret_cast<void**>(&st.slots[0].d_pt), size_t(n_chunks() + 1) * sizeof(int32_t)) != cudaSuccess) {
        cudaGetLastError();
        st.slots.clear();
        return false;
    }
    cudaMemsetAsync(st.slots[0].d_pt, 0xff, size_t(n_chunks() + 1) * sizeof(int32_t), stream);  // -1 = no page
    return true;
}

void Gpu::Impl::state_sizes() {
    const Sizes z = sizes_of(*this);
    st.snap_bytes = uint64_t(z.n_gdn) * (up(z.S) + up(z.hist));
    st.slot_bytes = st.snap_bytes + uint64_t(n_chunks() + 1) * sizeof(int32_t);  // the recurrent state and the page table
}

// ---- snapshots ----------------------------------------------------------------------------------------

uint64_t Gpu::snapshot_bytes() const {
    p_->state_ready();
    return p_->st.snap_bytes;
}

int Gpu::snapshot_slots() const { return int(p_->st.snap.size()); }

uint64_t Gpu::slot_bytes() const {
    p_->state_ready();
    return p_->st.slot_bytes;
}

uint64_t Gpu::state_vram_bytes() const {
    const Impl& I = *p_;
    uint64_t n = uint64_t(I.st.snap.size()) * I.st.snap_bytes;
    for (size_t h = 1; h < I.st.slots.size(); ++h) n += I.st.slot_bytes;
    return n;
}

namespace {

// True when `bytes` more can be taken and keep_free_bytes are still free afterwards.
bool room_for(uint64_t bytes, uint64_t keep_free_bytes) {
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return uint64_t(free_b) >= bytes + keep_free_bytes;
}

}  // namespace

int Gpu::set_snapshot_slots(int want, uint64_t keep_free_bytes) {
    Impl& I = *p_;
    if (!I.state_ready()) return 0;
    want = std::max(0, want);
    while (int(I.st.snap.size()) > want) {
        cudaFree(I.st.snap.back());  // implicitly waits for a copy still queued on the stream
        I.st.snap.pop_back();
        I.st.filled.pop_back();
    }
    while (int(I.st.snap.size()) < want) {
        if (!room_for(I.st.snap_bytes, keep_free_bytes)) break;
        void* p = nullptr;
        if (cudaMalloc(&p, I.st.snap_bytes) != cudaSuccess) {
            cudaGetLastError();
            break;
        }
        I.st.snap.push_back(p);
        I.st.filled.push_back(0);
    }
    return int(I.st.snap.size());
}

bool Gpu::snapshot(int slot) {
    Impl& I = *p_;
    if (!I.state_ready() || slot < 0 || slot >= int(I.st.snap.size())) return false;
    const Sizes z = sizes_of(I);
    uint8_t* at = static_cast<uint8_t*>(I.st.snap[size_t(slot)]);
    bool ok = true;
    for (size_t l = 0; l < I.S.size(); ++l) {
        if (!I.S[l]) continue;
        ok &= cudaMemcpyAsync(at, I.S[l], z.S, cudaMemcpyDeviceToDevice, I.stream) == cudaSuccess;
        at += up(z.S);
        ok &= cudaMemcpyAsync(at, I.hist[l], z.hist, cudaMemcpyDeviceToDevice, I.stream) == cudaSuccess;
        at += up(z.hist);
    }
    I.st.filled[size_t(slot)] = ok ? 1 : 0;
    return ok;
}

bool Gpu::restore(int slot) {
    Impl& I = *p_;
    if (!I.state_ready() || slot < 0 || slot >= int(I.st.snap.size()) || !I.st.filled[size_t(slot)]) return false;
    const Sizes z = sizes_of(I);
    const uint8_t* at = static_cast<const uint8_t*>(I.st.snap[size_t(slot)]);
    bool ok = true;
    for (size_t l = 0; l < I.S.size(); ++l) {
        if (!I.S[l]) continue;
        ok &= cudaMemcpyAsync(I.S[l], at, z.S, cudaMemcpyDeviceToDevice, I.stream) == cudaSuccess;
        at += up(z.S);
        ok &= cudaMemcpyAsync(I.hist[l], at, z.hist, cudaMemcpyDeviceToDevice, I.stream) == cudaSuccess;
        at += up(z.hist);
    }
    return ok;
}

// ---- slots --------------------------------------------------------------------------------------------

int Gpu::slots() const {
    p_->state_ready();
    return int(p_->st.slots.size());
}

int Gpu::slot() const { return p_->st.cur; }

int Gpu::set_slots(int want, uint64_t keep_free_bytes) {
    Impl& I = *p_;
    if (!I.state_ready()) return 0;
    want = std::max(1, std::min(want, Impl::kMaxSlots));
    while (int(I.st.slots.size()) > want && int(I.st.slots.size()) - 1 > I.st.cur) {  // never the live slot
        Impl::Slot& h = I.st.slots.back();
        const int idx = int(I.st.slots.size()) - 1;
        cudaStreamSynchronize(I.stream);
        for (cudaGraphExec_t& g : h.gexec)
            if (g) cudaGraphExecDestroy(g);
        for (cudaGraphExec_t& g : h.rgx)
            if (g) cudaGraphExecDestroy(g);
        if (I.pool) I.pool->cut(idx, 0);
        if (I.df) I.df->free_slot(idx);
        if (h.d_pt) cudaFree(h.d_pt);
        if (h.blob) cudaFree(h.blob);
        I.st.slots.pop_back();
    }
    const Sizes z = sizes_of(I);
    const size_t n_pt = size_t(I.n_chunks() + 1) * sizeof(int32_t);
    while (int(I.st.slots.size()) < want) {
        if (!room_for(I.st.slot_bytes + (I.df ? I.df->ring_bytes() : 0), keep_free_bytes)) break;
        Impl::Slot h;
        if (cudaMalloc(&h.blob, I.st.snap_bytes) != cudaSuccess) {
            cudaGetLastError();
            break;
        }
        if (cudaMalloc(reinterpret_cast<void**>(&h.d_pt), n_pt) != cudaSuccess) {
            cudaGetLastError();
            cudaFree(h.blob);
            break;
        }
        h.bytes = I.st.snap_bytes;
        cudaMemsetAsync(h.blob, 0, h.bytes, I.stream);  // a new slot starts as after reset()
        cudaMemsetAsync(h.d_pt, 0xff, n_pt, I.stream);
        uint8_t* at = static_cast<uint8_t*>(h.blob);
        const size_t n = I.L.size();
        h.S.assign(n, nullptr);
        h.hist.assign(n, nullptr);
        for (size_t l = 0; l < n; ++l) {
            if (I.L[l].full) continue;
            h.S[l] = reinterpret_cast<float*>(at);
            at += up(z.S);
            h.hist[l] = reinterpret_cast<float*>(at);
            at += up(z.hist);
        }
        if (I.df && !I.df->ensure_slot(int(I.st.slots.size()))) {  // the slot's context cache beside its state
            cudaGetLastError();
            cudaFree(h.blob);
            cudaFree(h.d_pt);
            break;
        }
        I.st.slots.push_back(std::move(h));
    }
    return int(I.st.slots.size());
}

bool Gpu::select_slot(int h) {
    Impl& I = *p_;
    if (!I.state_ready() || h < 0 || h >= int(I.st.slots.size())) return false;
    if (h == I.st.cur) return true;
    Impl::Slot& a = I.st.slots[size_t(I.st.cur)];
    Impl::Slot& b = I.st.slots[size_t(h)];
    // The live members go into the leaving slot, the arriving slot's into the live members.
    a.S.swap(I.S), I.S.swap(b.S);
    a.hist.swap(I.hist), I.hist.swap(b.hist);
    for (int v = 0; v < 2; ++v) {
        a.gexec[v] = I.gexec[v];
        a.warm[v] = I.warm[v];
        I.gexec[v] = b.gexec[v];
        I.warm[v] = b.warm[v];
        b.gexec[v] = nullptr;
        b.warm[v] = 0;
    }
    for (int g = 0; g < 40; ++g) {
        a.rgx[g] = I.rw.gx[g];
        a.rwarm[g] = I.rw.warm[g];
        I.rw.gx[g] = b.rgx[g];
        I.rw.warm[g] = b.rwarm[g];
        b.rgx[g] = nullptr;
        b.rwarm[g] = 0;
    }
    I.st.cur = h;
    return true;
}

}  // namespace limbo::cuda
