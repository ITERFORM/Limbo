// Copyright (c) 2026 ITERFORM. All rights reserved.
#pragma once
// limbo/ngram.hpp — prompt-lookup drafts for speculative decoding: an index over one slot's whole tape.
//
// A conversation that repeats itself (a function reproduced with a rename, a file quoted back with an edit, a tool result echoed)
// lets the next tokens be guessed from the tokens that followed the same words earlier. The Index keeps the slot's tape (every token
// the backend has read: prompt pieces, then generated ones) and, for each position, the previous position whose last `n` tokens are
// the same (a chain per hash bucket, newest first). propose() looks the tape's end up in that chain, takes the candidate whose
// context matches furthest back, and returns the tokens that followed it, up to `max` (a window holds at most 8 rows), continuing
// periodically when the copy runs into the end of the tape. When no candidate matches (the words just changed: a renamed identifier)
// the draft keeps following the earlier alignment for a few tokens (a substitution does not end a copy).
//
// A draft is only a guess: the backend checks it in one pass and keeps the prefix the model agrees with, so what the model says
// does not depend on this file. The structure is exact, though, so a miss is the text's, not a stale entry's: truncate() removes
// every position past the cut from its chain (rewind after a restore or a cut), assign() copies another slot's tape.
#include <cstdint>
#include <vector>

namespace limbo::ngram {

constexpr int kClasses = 4;

struct Draft {
    int n = 0;               // tokens in tok (0: nothing to propose)
    int32_t tok[8] = {};
    int match = 0;           // tokens before the end of the tape that equal the ones before the source (capped at 255)
    bool aligned = false;    // taken from the running alignment, not found through the hash (it follows a substitution)
    int src = -1;            // tape position of tok[0] in the source copy
    int cls = 0;             // how far to trust it: 0 aligned (recovering), 1 a match of < 8 tokens, 2 of < 24, 3 longer (kClasses of them)
};

class Index {
public:
    // n: match length (tokens that must agree before a draft is made); positions: the largest tape the index will hold (sizes the table).
    Index(int n = 8, int positions = 1 << 16);

    int n() const { return n_; }
    size_t size() const { return tape_.size(); }
    const std::vector<int32_t>& tape() const { return tape_; }
    bool synced() const { return !gap_; }  // false after a put that skipped positions: nothing is proposed until the next reset

    void clear();
    // Tokens tokens[0..m) now sit at positions pos.. of the tape: whatever was past pos is forgotten first. track: they were chosen by
    // the model (not read from a prompt), so the running alignment follows them. A put that starts past the end of the tape leaves a
    // gap the index cannot bridge: it stops proposing (until the tape is cleared and refilled from 0).
    void put(int pos, const int32_t* tokens, int m, bool track);
    void truncate(size_t len);
    // This index becomes the first `len` tokens of `from` (a slot that starts as another's prefix).
    void assign(const Index& from, size_t len);

    // The draft for the tape's end: at most `max` (<= 8) tokens. Keeps the alignment (see above) when it picks one.
    Draft propose(int max);

    // Bytes held, for the start-up report.
    uint64_t bytes() const;
    // The positions before `end` whose last n tokens hash like end's, newest first (a test hook: each is a candidate, a collision is not a match).
    std::vector<int> debug_chain(size_t end) const;

private:
    uint32_t hash_at(size_t end) const;  // of tape_[end-n+1 .. end]
    void insert(size_t i);               // chain position i (its last n tokens exist)
    void remove(size_t i);               // position i is the newest in its chain: unchain it
    int match_len(size_t a, size_t b) const;  // tokens equal going back from tape positions a and b (a < b)
    void follow(int32_t t);                   // the model chose t after the tape as it was: the alignment moves

    int n_;
    int bits_;
    std::vector<int32_t> tape_;
    std::vector<int32_t> prev_;  // per position: the previous position in its chain, -1 when none (-2: no hash, fewer than n tokens)
    std::vector<int32_t> head_;  // per bucket: the newest position, -1 when none
    bool gap_ = false;
    // the alignment: the tape position expected to repeat next (a_ < 0: none), the tokens that agreed since it last missed, the misses in a row
    int a_ = -1, a_run_ = 0, a_miss_ = 0;
};

// One Index per slot of a backend, with the calls a backend sees mapped onto them: the tape of the live slot is what the backend has read.
class Slots {
public:
    void configure(int n, int positions, int slots) {
        n_ = n;
        positions_ = positions;
        resize(slots);
    }
    void resize(int slots) {
        while (int(ix_.size()) < slots) ix_.emplace_back(n_, positions_);
    }
    int size() const { return int(ix_.size()); }
    bool select(int h) {
        if (h < 0 || h >= size()) return false;
        cur_ = h;
        return true;
    }
    Index& cur() { return ix_[size_t(cur_)]; }
    const Index& at(int h) const { return ix_[size_t(h)]; }
    void share(int from, size_t len) {  // the live slot starts as the first len tokens of `from`
        if (from >= 0 && from < size() && from != cur_) cur().assign(ix_[size_t(from)], len);
    }

private:
    std::vector<Index> ix_;
    int cur_ = 0, n_ = 8, positions_ = 1 << 16;
};

}  // namespace limbo::ngram
