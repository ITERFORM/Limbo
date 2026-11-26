// Copyright (c) 2026 ITERFORM. All rights reserved.
#pragma once
// limbo/kvpool.hpp — the page bookkeeping of the paged K/V cache (docs/paged-kv.md), host side only.
//
// A page holds 256 consecutive positions (= the split flash-decode's chunk) of one sequence for every full-attention layer. A slot
// (sequence) owns a table, one page id per chunk (-1: none); the kernels read the table through device memory. Pages carry a
// reference count: a page with more than one holder is a shared prefix, and writing into it replaces it by a private copy first
// (copy on write). The pool never touches the device: write() and share() say which pages the caller must copy (dst <- src, every
// plane of every layer).
//
// Page ids below pages() are device pages. Ids from pages() on are reserved for the host tier (pinned host memory the CPU lane
// reads), and the per-chunk owner flag says which lane computes a chunk's partials; neither is used yet.
#include <cstdint>
#include <vector>

namespace limbo::kv {

constexpr int kPagePositions = 256;

struct PageCopy {
    int dst = -1, src = -1;
};

enum class Owner : uint8_t { Gpu = 0, Cpu = 1 };

class PagePool {
public:
    // `pages` device pages, `n_slots` tables of `n_chunks` entries.
    PagePool(int pages, int n_chunks, int n_slots);

    int pages() const { return pages_; }
    int free_pages() const { return int(free_.size()); }
    int used_pages() const { return pages_ - free_pages(); }
    int shared_pages() const;  // pages with more than one holder
    int n_chunks() const { return n_chunks_; }
    int n_slots() const { return int(tab_.size()); }
    int refs(int page) const { return page >= 0 && page < pages_ ? refs_[size_t(page)] : 0; }

    // The slot's table (n_chunks entries) and the chunks that hold a page. take_dirty: true once after the table changed since
    // the last call (the device copy of the table is stale).
    const std::vector<int32_t>& table(int slot) const { return tab_[size_t(slot)]; }
    int held(int slot) const;
    bool take_dirty(int slot);
    Owner owner(int slot, int chunk) const { return Owner(own_[size_t(slot)][size_t(chunk)]); }
    void set_owner(int slot, int chunk, Owner o) { own_[size_t(slot)][size_t(chunk)] = uint8_t(o); }

    // Positions [lo, hi) of the slot are about to be written: chunks without a page take one, a chunk whose page is shared takes a
    // private page instead (a copy is listed unless the range overwrites the whole chunk). False when the pool runs dry; the pages
    // taken so far stay (cut() gives them back). A fresh pool hands out 0, 1, 2 ...; a page given back is the next one taken.
    bool write(int slot, int lo, int hi, std::vector<PageCopy>* copies);

    // The first n_pos positions of the slot have pages: chunks without one take one, shared pages stay shared (no copy on write: this
    // reserves, it does not write). False when the pool runs dry; what was taken stays.
    bool fill(int slot, int n_pos);

    // Positions >= keep of the slot go back to the pool (a shared page only loses this holder). keep = 0: everything.
    void cut(int slot, int keep);

    // The empty slot `dst` starts as the first n_pos positions of `src`: whole chunks are shared (one more holder each), the chunk
    // that n_pos cuts in two is copied (the slot writes into it from n_pos on). False, with dst left empty, when src does not hold
    // those chunks, dst is not empty or the pool has no page for the copy.
    bool share(int dst, int src, int n_pos, std::vector<PageCopy>* copies);

    // Positions the slot can still hold from position `at` on: the pages it has in a row from the start and the free ones.
    int room(int slot, int at) const;

    // Consistency check for tests: every page's count equals its holders and the free list is exactly the unheld pages.
    bool consistent() const;

private:
    int take();
    void drop(int page);

    int pages_, n_chunks_;
    std::vector<int32_t> refs_;
    std::vector<int32_t> free_;  // a stack; a fresh pool has page 0 on top
    std::vector<std::vector<int32_t>> tab_;
    std::vector<std::vector<uint8_t>> own_;
    std::vector<uint8_t> dirty_;
};

}  // namespace limbo::kv
