// Copyright (c) 2026 ITERFORM. All rights reserved.
// tests/ngram_test.cpp — limbo::ngram::Index against a naive reference: chains stay exact through put, truncate and assign.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "limbo/check.hpp"
#include "limbo/ngram.hpp"

using limbo::ngram::Draft;
using limbo::ngram::Index;

namespace {

// Earlier positions whose last n tokens equal the end's, newest first.
std::vector<int> naive_matches(const std::vector<int32_t>& t, int n) {
    std::vector<int> r;
    if (t.size() < size_t(n)) return r;
    const size_t end = t.size() - 1;
    for (size_t p = end; p-- > size_t(n) - 1;)
        if (std::equal(t.begin() + long(p) + 1 - n, t.begin() + long(p) + 1, t.begin() + long(end) + 1 - n)) r.push_back(int(p));
    return r;
}

int naive_len(const std::vector<int32_t>& t, size_t a, size_t b) {
    int k = 0;
    while (k < 255 && size_t(k) <= a && t[a - size_t(k)] == t[b - size_t(k)]) ++k;
    return k;
}

// The chain of the tape's end holds every earlier position with the same last n tokens, newest first (a collision may add others).
bool chain_ok(const Index& ix, const std::vector<int32_t>& ref, int n) {
    if (ref.size() < size_t(n)) return true;
    const std::vector<int> got = ix.debug_chain(ref.size() - 1), want = naive_matches(ref, n);
    std::vector<int> only;
    for (int p : got)
        if (std::equal(ref.begin() + long(p) + 1 - n, ref.begin() + long(p) + 1, ref.end() - n)) only.push_back(p);
    return only == want && std::is_sorted(got.begin(), got.end(), std::greater<int>());
}

// What propose() may return: a copy of the continuation of some earlier match (periodic past the end).
bool plausible(const std::vector<int32_t>& t, const Draft& d, int n, int max) {
    if (d.n == 0) return true;
    if (d.n != max || d.src < 0 || size_t(d.src) > t.size() - 1) return false;
    std::vector<int32_t> seq(t.begin(), t.end());
    for (int j = 0; j < d.n; ++j) {
        const int32_t want = seq[size_t(d.src) + size_t(j)];
        if (d.tok[j] != want) return false;
        seq.push_back(d.tok[j]);
    }
    if (!d.aligned) return d.match >= n && naive_len(t, size_t(d.src) - 1, t.size() - 1) == d.match;
    return true;
}

}  // namespace

int main() {
    // ---- a copy is found, with its match length ----
    {
        Index ix(4, 1024);
        std::vector<int32_t> a;
        for (int i = 0; i < 40; ++i) a.push_back(100 + i);
        ix.put(0, a.data(), int(a.size()), false);
        const int32_t pre[] = {7, 8, 9};
        ix.put(40, pre, 3, true);
        ix.put(43, a.data(), 6, true);  // the first six of the copy: its last four match a[2..5]
        const Draft d = ix.propose(7);
        CHECK(d.n == 7 && d.src == 6 && d.tok[0] == 106 && d.tok[6] == 112 && !d.aligned && d.match == 6 && d.cls == 1, "ngram: after six copied tokens the draft is the next seven of the original");
    }
    // ---- nothing to say when nothing repeats ----
    {
        Index ix(4, 1024);
        std::vector<int32_t> a;
        for (int i = 0; i < 100; ++i) a.push_back(i);
        ix.put(0, a.data(), 100, false);
        CHECK(ix.propose(7).n == 0, "ngram: no repeat, no draft");
        Index e(4, 16);
        CHECK(e.propose(7).n == 0 && e.size() == 0, "ngram: an empty tape proposes nothing");
    }
    // ---- a periodic tape: the draft runs past the end by repeating ----
    {
        Index ix(3, 256);
        std::vector<int32_t> a;
        for (int i = 0; i < 12; ++i) a.push_back(1 + i % 3);
        ix.put(0, a.data(), 12, false);
        const Draft d = ix.propose(7);
        bool ok = d.n == 7;
        std::vector<int32_t> s = a;
        for (int j = 0; ok && j < 7; ++j) {
            ok &= d.tok[j] == 1 + int(s.size()) % 3;
            s.push_back(d.tok[j]);
        }
        CHECK(ok && d.match >= 9, "ngram: a period-3 tape is continued past its end with the same period");
    }
    // ---- the longest context wins among candidates ----
    {
        Index ix(3, 1024);
        // two earlier occurrences of "5 6 7": one preceded by 1 2 3 (long match with the end), one by 9 9 9
        const int32_t t[] = {1, 2, 3, 5, 6, 7, 50, 51, 52, 53, 54, 55, 56, 57, 9, 9, 9, 5, 6, 7, 60, 61, 62, 63, 64, 65, 66, 67, 1, 2, 3, 5, 6, 7};
        ix.put(0, t, int(sizeof t / sizeof t[0]), false);
        const Draft d = ix.propose(4);
        CHECK(d.n == 4 && d.src == 6 && d.tok[0] == 50 && d.match == 6, "ngram: of two earlier occurrences the one whose preceding tokens match too is used, not the newest");
    }
    // ---- the alignment survives a substituted token ----
    {
        Index ix(4, 1024);
        std::vector<int32_t> a;
        for (int i = 0; i < 60; ++i) a.push_back(300 + i);
        ix.put(0, a.data(), 60, false);
        std::vector<int32_t> out(a.begin(), a.begin() + 10);  // the copy starts; the 11th token is changed (a rename)
        ix.put(60, out.data(), 10, true);
        Draft d = ix.propose(7);
        CHECK(d.n == 7 && d.src == 10 && d.tok[0] == 310 && !d.aligned, "ngram: the copy is found by its hash first");
        const int32_t sub = 9999;
        ix.put(70, &sub, 1, true);  // the model said 9999 where the source has 310
        d = ix.propose(7);
        CHECK(d.n == 7 && d.aligned && d.tok[0] == 311 && d.tok[6] == 317 && d.cls == 0, "ngram: after a substituted token the draft goes on at the next source token (aligned, recovering)");
        ix.put(71, a.data() + 11, 2, true);
        d = ix.propose(7);
        CHECK(d.n == 7 && d.tok[0] == 313, "ngram: and keeps going while the copy goes on");
        // three misses in a row end the alignment
        const int32_t junk[] = {7001, 7002, 7003, 7004, 7005};
        ix.put(73, junk, 5, true);
        CHECK(ix.propose(7).n == 0, "ngram: a run of misses drops the alignment");
    }
    // ---- consistency: random puts, truncations and copies against the naive reference ----
    {
        std::mt19937 rng(12345);
        bool chains = true, drafts = true, tapes = true;
        long checked = 0, with_draft = 0;
        for (int n : {2, 3, 5, 8}) {
            Index ix(n, 4096), other(n, 4096);
            std::vector<int32_t> ref, ref_other;
            auto token = [&] { return int32_t(rng() % 7 + (rng() % 5 == 0 ? rng() % 40 : 0)); };  // a small alphabet repeats a lot
            for (int step = 0; step < 4000; ++step) {
                const unsigned op = rng() % 100;
                if (op < 70) {  // extend by 1..9 tokens
                    int m = int(rng() % 9) + 1;
                    std::vector<int32_t> v;
                    for (int k = 0; k < m; ++k) v.push_back(token());
                    ix.put(int(ref.size()), v.data(), m, rng() % 2 == 0);
                    ref.insert(ref.end(), v.begin(), v.end());
                } else if (op < 85 && !ref.empty()) {  // rewind and write something else (a restore)
                    const size_t cut = size_t(rng() % (ref.size() + 1));
                    std::vector<int32_t> v;
                    for (int k = 0, m = int(rng() % 6); k < m; ++k) v.push_back(token());
                    if (v.empty()) ix.truncate(cut);
                    else ix.put(int(cut), v.data(), int(v.size()), true);
                    ref.resize(cut);
                    ref.insert(ref.end(), v.begin(), v.end());
                } else if (op < 90) {  // truncate
                    const size_t cut = size_t(rng() % (ref.size() + 1));
                    ix.truncate(cut);
                    ref.resize(cut);
                } else if (op < 95) {  // the other slot takes a prefix of this one
                    const size_t cut = size_t(rng() % (ref.size() + 1));
                    other.assign(ix, cut);
                    ref_other.assign(ref.begin(), ref.begin() + long(cut));
                    tapes &= other.tape() == ref_other;
                    std::vector<int32_t> v{token(), token()};
                    other.put(int(cut), v.data(), 2, true);
                    ref_other.insert(ref_other.end(), v.begin(), v.end());
                    tapes &= other.tape() == ref_other;
                    chains &= chain_ok(other, ref_other, n);
                } else if (ref.size() > 3000) {
                    ix.truncate(0);
                    ref.clear();
                }
                tapes &= ix.tape() == ref;
                chains &= chain_ok(ix, ref, n);
                if (step % 7 == 0 && !ref.empty()) {
                    const Draft d = ix.propose(1 + int(rng() % 7));
                    ++checked;
                    with_draft += d.n > 0;
                    drafts &= d.n == 0 || plausible(ref, d, n, d.n);
                }
            }
        }
        CHECK(tapes, "ngram: the tape equals the reference after every put / rewind / truncate / assign");
        CHECK(chains, "ngram: every earlier position with the same last n tokens is in the chain, newest first, through rewinds and copies");
        CHECK(drafts && with_draft > 100, "ngram: every draft is a real continuation of an earlier match (and drafts are made)");
        std::printf("     ngram: %ld lookups, %ld drafts\n", checked, with_draft);
    }
    // ---- a put that skips positions stops the index until the tape starts over ----
    {
        Index ix(3, 256);
        const int32_t a[] = {1, 2, 3, 1, 2, 3, 1, 2, 3};
        ix.put(0, a, 9, false);
        CHECK(ix.propose(4).n == 4, "ngram: a repeating tape drafts");
        ix.put(20, a, 3, true);
        CHECK(!ix.synced() && ix.propose(4).n == 0 && ix.size() == 0, "ngram: a put past the end leaves a gap: nothing is proposed");
        ix.put(5, a, 3, true);
        CHECK(!ix.synced() && ix.size() == 0, "ngram: and it stays that way until position 0 is written");
        ix.put(0, a, 9, false);
        CHECK(ix.synced() && ix.propose(4).n == 4, "ngram: a reset and refill brings it back");
    }
    // ---- a long tape: lookups stay cheap and rewinds exact ----
    {
        Index ix(8, 1 << 18);
        std::mt19937 rng(7);
        std::vector<int32_t> v(200000);
        for (auto& x : v) x = int32_t(rng() % 50000);
        ix.put(0, v.data(), int(v.size()), false);
        const int32_t tail[] = {v[1000], v[1001], v[1002], v[1003], v[1004], v[1005], v[1006], v[1007]};
        ix.put(int(v.size()), tail, 8, true);
        const Draft d = ix.propose(7);
        CHECK(d.n == 7 && d.src == 1008 && d.tok[0] == v[1008], "ngram: a quoted passage in a 200k-token tape is found");
        ix.truncate(1500);
        CHECK(ix.size() == 1500 && ix.debug_chain(1499).size() <= 1, "ngram: a rewind to 1,500 forgets the rest");
    }
    return limbo::test::finish("ngram");
}
