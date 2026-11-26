// Copyright (c) 2026 ITERFORM. All rights reserved.
// src/ngram.cpp — limbo::ngram::Index (include/limbo/ngram.hpp).
#include "limbo/ngram.hpp"

#include <algorithm>

namespace limbo::ngram {

namespace {
constexpr int kMaxChain = 32;   // candidates examined per lookup
constexpr int kMaxMatch = 255;  // how far back a match is measured
}  // namespace

Index::Index(int n, int positions) : n_(std::clamp(n, 2, 64)) {
    int bits = 12;
    while (bits < 22 && (1 << bits) < 2 * std::max(positions, 1)) ++bits;
    bits_ = bits;
}

void Index::clear() {
    tape_.clear();
    prev_.clear();
    if (!head_.empty()) std::fill(head_.begin(), head_.end(), -1);
    gap_ = false;
    a_ = -1;
    a_run_ = a_miss_ = 0;
}

uint32_t Index::hash_at(size_t end) const {
    uint64_t h = 0x9E3779B97F4A7C15ull;
    for (size_t k = end + 1 - size_t(n_); k <= end; ++k) {
        h = (h ^ uint32_t(tape_[k])) * 0x100000001B3ull;
        h ^= h >> 29;
    }
    h *= 0xFF51AFD7ED558CCDull;
    h ^= h >> 32;
    return uint32_t(h & ((1u << bits_) - 1));
}

void Index::insert(size_t i) {
    if (head_.empty()) head_.assign(size_t(1) << bits_, -1);
    const uint32_t b = hash_at(i);
    prev_[i] = head_[b];
    head_[b] = int32_t(i);
}

void Index::remove(size_t i) {
    const uint32_t b = hash_at(i);
    if (head_[b] == int32_t(i)) head_[b] = prev_[i];
}

void Index::truncate(size_t len) {
    if (len >= tape_.size()) return;
    const size_t removed = tape_.size() - len;
    if (a_ >= int(len)) a_ = -1;
    if (len == 0) {
        clear();
        return;
    }
    if (!head_.empty() && removed > len + head_.size() / 16) {  // cheaper to chain what stays again than to unchain what goes
        tape_.resize(len);
        prev_.assign(len, -2);
        std::fill(head_.begin(), head_.end(), -1);
        for (size_t i = size_t(n_) - 1; i < len; ++i) insert(i);
        return;
    }
    for (size_t i = tape_.size(); i-- > len;)  // newest first: each is the head of its chain when its turn comes
        if (i + 1 >= size_t(n_)) remove(i);
    tape_.resize(len);
    prev_.resize(len);
}

void Index::put(int pos, const int32_t* tokens, int m, bool track) {
    if (pos < 0 || m <= 0) return;
    if (!track) a_ = -1;  // tokens read from a prompt break any copy being followed
    if (gap_) {
        if (pos != 0) return;
        clear();
    }
    if (size_t(pos) > tape_.size()) {  // tokens the index never saw: it cannot say what the tape holds
        clear();
        gap_ = true;
        return;
    }
    if (size_t(pos) < tape_.size()) truncate(size_t(pos));
    tape_.reserve(tape_.size() + size_t(m));
    for (int k = 0; k < m; ++k) {
        if (track) follow(tokens[k]);
        tape_.push_back(tokens[k]);
        prev_.push_back(-2);
        if (tape_.size() >= size_t(n_)) insert(tape_.size() - 1);
    }
}

void Index::assign(const Index& from, size_t len) {
    if (&from == this) {
        truncate(len);
        return;
    }
    n_ = from.n_;
    bits_ = from.bits_;
    tape_ = from.tape_;
    prev_ = from.prev_;
    head_ = from.head_;
    gap_ = from.gap_;
    a_ = -1;
    a_run_ = a_miss_ = 0;
    if (gap_) return;
    truncate(len);
}

void Index::follow(int32_t t) {
    if (a_ < 0) return;
    if (size_t(a_) >= tape_.size()) {
        a_ = -1;
        return;
    }
    if (tape_[size_t(a_)] == t) {
        if (++a_run_ >= 2) a_miss_ = 0;
    } else {
        a_run_ = 0;
        if (++a_miss_ > 2) {  // three in a row: the copy is over, not a substitution
            a_ = -1;
            return;
        }
    }
    ++a_;
}

int Index::match_len(size_t a, size_t b) const {
    int k = 0;
    while (k < kMaxMatch && size_t(k) <= a && tape_[a - size_t(k)] == tape_[b - size_t(k)]) ++k;
    return k;
}

Draft Index::propose(int max) {
    Draft d;
    max = std::min(max, 8);
    if (gap_ || tape_.empty() || max < 1) return d;
    const size_t end = tape_.size() - 1;
    int best = -1, best_len = 0;
    if (tape_.size() >= size_t(n_)) {
        int p = prev_[end], steps = 0;
        while (p >= 0 && steps++ < kMaxChain) {
            const int L = match_len(size_t(p), end);
            if (L >= n_ && L > best_len) {
                best = p;
                best_len = L;
                if (L >= kMaxMatch) break;
            }
            p = prev_[size_t(p)];
        }
    }
    int src = -1;
    if (best >= 0 && (a_ < 0 || best + 1 == a_ || a_run_ < best_len)) {  // the hash match is at least as sure as the alignment
        src = best + 1;
        d.match = best_len;
        d.aligned = false;
        d.cls = best_len < 8 ? 1 : best_len < 24 ? 2 : 3;
        a_ = src;
        a_run_ = best_len;
        a_miss_ = 0;
    } else if (a_ >= 0 && size_t(a_) <= end) {
        src = a_;
        d.match = a_run_;
        d.aligned = true;
        d.cls = 0;
    } else return d;
    // tokens src, src+1, ...; past the tape's end they are the draft itself (the copy repeats with period end + 1 - src)
    const size_t period = end + 1 - size_t(src);
    for (int j = 0; j < max; ++j) {
        const size_t at = size_t(src) + size_t(j);
        d.tok[j] = at <= end ? tape_[at] : d.tok[size_t(j) - period];
    }
    d.n = max;
    d.src = src;
    return d;
}

uint64_t Index::bytes() const { return uint64_t(tape_.capacity() + prev_.capacity() + head_.capacity()) * sizeof(int32_t); }

std::vector<int> Index::debug_chain(size_t end) const {
    std::vector<int> r;
    if (end >= tape_.size() || end + 1 < size_t(n_)) return r;
    for (int p = prev_[end]; p >= 0; p = prev_[size_t(p)]) r.push_back(p);
    return r;
}

}  // namespace limbo::ngram
