// Copyright (c) 2026 ITERFORM. All rights reserved.
// src/kvpool.cpp — the page tables, reference counts and copy on write of the paged K/V cache (limbo/kvpool.hpp).
#include "limbo/kvpool.hpp"

#include <algorithm>

namespace limbo::kv {

PagePool::PagePool(int pages, int n_chunks, int n_slots)
    : pages_(std::max(0, pages)),
      n_chunks_(std::max(1, n_chunks)),
      refs_(size_t(pages_), 0),
      tab_(size_t(std::max(1, n_slots)), std::vector<int32_t>(size_t(n_chunks_), -1)),
      own_(size_t(std::max(1, n_slots)), std::vector<uint8_t>(size_t(n_chunks_), uint8_t(Owner::Gpu))),
      dirty_(size_t(std::max(1, n_slots)), 1) {
    free_.reserve(size_t(pages_));
    for (int p = pages_ - 1; p >= 0; --p) free_.push_back(p);
}

int PagePool::shared_pages() const {
    int n = 0;
    for (int r : refs_) n += r > 1 ? 1 : 0;
    return n;
}

int PagePool::held(int slot) const {
    int n = 0;
    for (int p : tab_[size_t(slot)]) n += p >= 0 ? 1 : 0;
    return n;
}

bool PagePool::take_dirty(int slot) {
    const bool d = dirty_[size_t(slot)] != 0;
    dirty_[size_t(slot)] = 0;
    return d;
}

int PagePool::take() {
    if (free_.empty()) return -1;
    const int p = free_.back();
    free_.pop_back();
    refs_[size_t(p)] = 1;
    return p;
}

void PagePool::drop(int page) {
    if (--refs_[size_t(page)] == 0) free_.push_back(page);
}

bool PagePool::write(int slot, int lo, int hi, std::vector<PageCopy>* copies) {
    if (lo < 0 || hi <= lo) return hi <= lo;
    const int c0 = lo / kPagePositions, c1 = (hi - 1) / kPagePositions;
    if (c1 >= n_chunks_) return false;
    std::vector<int32_t>& t = tab_[size_t(slot)];
    for (int c = c0; c <= c1; ++c) {
        const int p = t[size_t(c)];
        if (p >= 0 && refs_[size_t(p)] == 1) continue;
        const int q = take();
        if (q < 0) return false;
        if (p >= 0) {
            const bool whole = lo <= c * kPagePositions && hi >= (c + 1) * kPagePositions;
            if (!whole && copies) copies->push_back({q, p});
            drop(p);
        }
        t[size_t(c)] = q;
        dirty_[size_t(slot)] = 1;
    }
    return true;
}

bool PagePool::fill(int slot, int n_pos) {
    if (n_pos <= 0) return true;
    const int chunks = (n_pos + kPagePositions - 1) / kPagePositions;
    if (chunks > n_chunks_) return false;
    std::vector<int32_t>& t = tab_[size_t(slot)];
    for (int c = 0; c < chunks; ++c) {
        if (t[size_t(c)] >= 0) continue;
        const int q = take();
        if (q < 0) return false;
        t[size_t(c)] = q;
        dirty_[size_t(slot)] = 1;
    }
    return true;
}

void PagePool::cut(int slot, int keep) {
    std::vector<int32_t>& t = tab_[size_t(slot)];
    for (int c = (std::max(0, keep) + kPagePositions - 1) / kPagePositions; c < n_chunks_; ++c)
        if (t[size_t(c)] >= 0) {
            drop(t[size_t(c)]);
            t[size_t(c)] = -1;
            dirty_[size_t(slot)] = 1;
        }
}

bool PagePool::share(int dst, int src, int n_pos, std::vector<PageCopy>* copies) {
    if (dst == src || n_pos <= 0 || held(dst) != 0) return false;
    const int whole = n_pos / kPagePositions, chunks = (n_pos + kPagePositions - 1) / kPagePositions;
    if (chunks > n_chunks_) return false;
    const std::vector<int32_t>& s = tab_[size_t(src)];
    for (int c = 0; c < chunks; ++c)
        if (s[size_t(c)] < 0) return false;
    std::vector<int32_t>& t = tab_[size_t(dst)];
    int tail = -1;
    if (chunks > whole) {
        tail = take();
        if (tail < 0) return false;
    }
    for (int c = 0; c < whole; ++c) {
        t[size_t(c)] = s[size_t(c)];
        ++refs_[size_t(s[size_t(c)])];
    }
    if (tail >= 0) {
        t[size_t(whole)] = tail;
        if (copies) copies->push_back({tail, s[size_t(whole)]});
    }
    dirty_[size_t(dst)] = 1;
    return true;
}

int PagePool::room(int slot, int at) const {
    const std::vector<int32_t>& t = tab_[size_t(slot)];
    int lead = 0;
    while (lead < n_chunks_ && t[size_t(lead)] >= 0) ++lead;
    const int have = std::max(0, lead * kPagePositions - at);
    return std::min(have + free_pages() * kPagePositions, n_chunks_ * kPagePositions - at);
}

bool PagePool::consistent() const {
    std::vector<int> hold(size_t(pages_), 0);
    for (const auto& t : tab_)
        for (int p : t) {
            if (p >= pages_) return false;
            if (p >= 0) ++hold[size_t(p)];
        }
    int unheld = 0;
    for (int p = 0; p < pages_; ++p) {
        if (hold[size_t(p)] != refs_[size_t(p)]) return false;
        unheld += hold[size_t(p)] == 0 ? 1 : 0;
    }
    if (unheld != int(free_.size())) return false;
    std::vector<uint8_t> seen(size_t(pages_), 0);
    for (int p : free_) {
        if (p < 0 || p >= pages_ || seen[size_t(p)] || refs_[size_t(p)] != 0) return false;
        seen[size_t(p)] = 1;
    }
    return true;
}

}  // namespace limbo::kv
