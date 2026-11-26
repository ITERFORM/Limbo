// Copyright (c) 2026 ITERFORM. All rights reserved.
// tests/kvpool_test.cpp — the page bookkeeping of the paged K/V cache (limbo/kvpool.hpp): allocation on demand, release on cut,
// reference counts, copy on write, prefix sharing, and the invariant after a long random mix. Host only.
#include <cstdint>
#include <cstdio>
#include <vector>

#include "limbo/check.hpp"
#include "limbo/kvpool.hpp"

using limbo::kv::PageCopy;
using limbo::kv::PagePool;

namespace {

void allocation() {
    PagePool pool(8, 16, 2);
    CHECK(pool.pages() == 8 && pool.free_pages() == 8 && pool.used_pages() == 0, "1. a fresh pool is all free");
    std::vector<PageCopy> cp;
    CHECK(pool.write(0, 0, 1, &cp) && pool.table(0)[0] == 0 && pool.used_pages() == 1, "1. the first write takes page 0");
    CHECK(pool.write(0, 1, 256, &cp) && pool.used_pages() == 1, "1. positions inside a held chunk take nothing");
    CHECK(pool.write(0, 256, 257, &cp) && pool.table(0)[1] == 1 && pool.held(0) == 2, "1. the next chunk takes the next page, on demand");
    CHECK(pool.write(0, 300, 1300, &cp) && pool.held(0) == 6 && pool.table(0)[5] == 5, "1. a long write takes every chunk it touches");
    CHECK(cp.empty(), "1. nothing was copied: no page was shared");
    CHECK(pool.take_dirty(0) && !pool.take_dirty(0), "1. the table is dirty once after a change");
    CHECK(!pool.write(1, 0, 4 * 256, &cp) && pool.free_pages() == 0, "1. a write the pool cannot hold fails and keeps what it took");
    pool.cut(1, 0);
    CHECK(pool.free_pages() == 2 && pool.held(1) == 0, "1. cut(0) gives the partial allocation back");
    pool.cut(0, 700);
    CHECK(pool.held(0) == 3 && pool.free_pages() == 5, "1. cut(700) keeps the chunk that holds position 699 (chunks 0..2)");
    pool.cut(0, 512);
    CHECK(pool.held(0) == 2 && pool.free_pages() == 6, "1. cut at a chunk boundary frees that chunk");
    CHECK(pool.write(0, 512, 513, &cp) && pool.table(0)[2] == 2, "1. a page given back is the next one taken");
    CHECK(!pool.write(0, 16 * 256, 16 * 256 + 1, &cp), "1. a position past the table is refused");
    CHECK(pool.fill(1, 700) && pool.held(1) == 3 && pool.fill(1, 700) && pool.held(1) == 3, "1. fill takes the pages of the first n positions, once");
    pool.cut(1, 0);
    CHECK(pool.consistent(), "1. counts and free list agree");
}

void sharing() {
    PagePool pool(16, 32, 3);
    std::vector<PageCopy> cp;
    pool.write(1, 0, 3000, &cp);  // slot 1: 12 chunks
    CHECK(pool.held(1) == 12 && pool.used_pages() == 12, "2. the source holds 12 pages");
    cp.clear();
    CHECK(pool.share(0, 1, 2600, &cp), "2. slot 0 takes the first 2600 positions of slot 1");
    CHECK(pool.used_pages() == 13, "2. 10 whole chunks are shared, the chunk 2600 cuts is copied: one new page, not 11");
    CHECK(cp.size() == 1 && cp[0].src == pool.table(1)[10] && cp[0].dst == pool.table(0)[10] && cp[0].dst != cp[0].src,
          "2. the copy is the source's page 10 into a private page");
    bool same = true;
    for (int c = 0; c < 10; ++c) same &= pool.table(0)[size_t(c)] == pool.table(1)[size_t(c)] && pool.refs(pool.table(1)[size_t(c)]) == 2;
    CHECK(same && pool.shared_pages() == 10, "2. the first 10 chunks are the same pages with two holders");
    CHECK(pool.table(0)[11] == -1, "2. nothing past the copied chunk");
    CHECK(!pool.share(0, 1, 100, &cp), "2. a slot that holds pages cannot be shared into");
    CHECK(!pool.share(2, 1, 4000, &cp) && pool.held(2) == 0, "2. more than the source holds is refused and leaves the slot empty");
    cp.clear();
    CHECK(pool.share(2, 1, 1024, &cp) && cp.empty() && pool.used_pages() == 13, "2. a page-aligned share copies nothing and takes no page");
    CHECK(pool.refs(pool.table(1)[0]) == 3, "2. three holders on the first page");
    // writes into a shared page: slot 1 overwrites inside chunk 3 (positions 800..900): copy on write, listed
    cp.clear();
    CHECK(pool.write(1, 800, 900, &cp) && cp.size() == 1 && cp[0].src == pool.table(0)[3], "2. a partial write into a shared chunk copies the page");
    CHECK(pool.table(1)[3] != pool.table(0)[3] && pool.refs(pool.table(0)[3]) == 2 && pool.refs(pool.table(1)[3]) == 1,
          "2. the writer has its own page now, the others still share the old one");
    cp.clear();
    CHECK(pool.write(1, 1024, 1280, &cp) && cp.empty(), "2. a write that covers a whole shared chunk replaces it without a copy");
    CHECK(pool.table(1)[4] != pool.table(0)[4], "2. ... and the replacement is private");
    // releasing: slot 0 leaves, shared pages only lose one holder
    const int before = pool.free_pages();
    pool.cut(0, 0);
    CHECK(pool.held(0) == 0 && pool.free_pages() > before, "2. a slot that leaves frees its private pages");
    CHECK(pool.refs(pool.table(2)[0]) == 2 && pool.refs(pool.table(2)[1]) == 2, "2. the pages it shared stay with the others");
    CHECK(pool.consistent(), "2. counts and free list agree");
    CHECK(pool.fill(2, 1100) && pool.refs(pool.table(2)[0]) == 2 && pool.held(2) == 5, "2. fill reserves without copying a shared page");
    pool.cut(1, 0);
    pool.cut(2, 0);
    CHECK(pool.free_pages() == 16 && pool.used_pages() == 0 && pool.consistent(), "2. everything given back: all 16 pages are free");
    // a share that cannot take its copy page fails whole
    PagePool tight(3, 8, 2);
    tight.write(1, 0, 3 * 256, nullptr);
    CHECK(!tight.share(0, 1, 300, &cp) && tight.held(0) == 0 && tight.refs(tight.table(1)[0]) == 1 && tight.consistent(),
          "2. no page for the copy: refused, no holder added");
}

void room() {
    PagePool pool(10, 16, 2);
    pool.write(0, 0, 600, nullptr);
    CHECK(pool.room(0, 600) == 3 * 256 - 600 + 7 * 256, "3. room = the tail of the held pages plus the free ones");
    CHECK(pool.room(1, 0) == 7 * 256, "3. an empty slot may take every free page");
    PagePool big(100, 4, 1);
    CHECK(big.room(0, 0) == 4 * 256, "3. never more than the table can address");
}

void shuffle() {
    // a long random mix: the counts must still equal the holders and no page may leak
    PagePool pool(40, 64, 4);
    uint32_t r = 12345;
    auto rnd = [&](uint32_t n) {
        r = r * 1664525u + 1013904223u;
        return (r >> 8) % n;
    };
    std::vector<PageCopy> cp;
    bool ok = true;
    for (int i = 0; i < 20000 && ok; ++i) {
        const int s = int(rnd(4));
        switch (rnd(4)) {
            case 0: {
                const int lo = int(rnd(8000)), hi = lo + 1 + int(rnd(900));
                pool.write(s, lo - lo % 256 == 0 ? lo : 0, hi, &cp);  // a write from 0 or from a chunk start (a slot is contiguous)
                break;
            }
            case 1: pool.cut(s, int(rnd(9000))); break;
            case 2: {
                const int d = int(rnd(4));
                if (d != s) pool.share(d, s, 1 + int(rnd(4000)), &cp);
                break;
            }
            default: pool.cut(s, 0); break;
        }
        ok = pool.consistent();
    }
    CHECK(ok, "4. 20000 random writes, cuts, shares and releases keep every count right");
    for (int s = 0; s < 4; ++s) pool.cut(s, 0);
    CHECK(pool.free_pages() == 40 && pool.consistent(), "4. after releasing every slot the pool is whole again");
}

}  // namespace

int main() {
    allocation();
    sharing();
    room();
    shuffle();
    return limbo::test::finish("kvpool_test");
}
