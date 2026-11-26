// Copyright (c) 2026 ITERFORM. All rights reserved.
// tests/cuda_state_test.cu — the conversation state of limbo::cuda::Gpu: recurrent-state snapshots, rewind and slots.
//
// K/V is positional, the delta-net state is not, so the rule under test is: state at position p =
// restore(the snapshot taken at p), then prefill(tokens[p..]). Everything is compared bit for bit against
// an uninterrupted run (max|dlogit| is printed; it must be 0):
//   1. taking a snapshot does not disturb the run it was taken in;
//   2. snapshot at p, run on (a prefill, then decode steps), run a DIFFERENT branch over the same positions
//      (stale K/V rows and a different recurrent state), restore, re-prefill from p: the logits of the last
//      prompt token and the next 8 greedy decode steps equal the uninterrupted run's;
//   3. two snapshots (p1 < p2) restore either;
//   4. bad slots and never-written slots are refused;
//   5. what one snapshot and one restore cost, and how many bytes it is;
//   6. slots: a side conversation in slot 1 between two steps of slot 0 leaves slot 0's next logits bit-identical
//      to a run where it never happened, and the decode graphs are captured per home;
//   7. information only: prefill chunking (one call, or split at p) against the same tokens.
//
// The model is LIMBO_TEST_GGUF or the rung-1 file. SKIP (exit 0) when there is no GPU or under 15.5 GB free.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "limbo/check.hpp"
#include "limbo/cuda/gpu.hpp"
#include "limbo/model.hpp"

namespace {

using limbo::cuda::Gpu;

constexpr const char* kRung1 =
    "/home/alca/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-MTP-GGUF/snapshots/"
    "5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf";

std::vector<int32_t> random_ids(int n, uint32_t seed) {
    std::vector<int32_t> ids(static_cast<size_t>(n));
    uint32_t s = seed;
    for (int32_t& v : ids) {
        s = s * 1664525u + 1013904223u;
        v = 1000 + static_cast<int32_t>((s >> 8) % 200000u);
    }
    return ids;
}

double max_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(double(a[i]) - double(b[i])));
    return m;
}
bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

// What a conversation run leaves behind: the logits of the last prompt token and of each decode step.
struct Trace {
    std::vector<std::vector<float>> logits;
    std::vector<int32_t> toks;  // the greedy token chosen after each of them
};

// Reads ids[from..n-1) in one prefill call, runs ids[n-1] as a step with logits, then `steps` greedy decode steps.
bool finish_run(Gpu& g, const std::vector<int32_t>& ids, int from, int n, int steps, Trace* t) {
    const int nv = g.n_vocab();
    if (from < n - 1 && !g.prefill(ids.data() + from, n - 1 - from, from, false)) return false;
    int32_t tok = ids[static_cast<size_t>(n - 1)];
    for (int i = 0; i <= steps; ++i) {
        if (!g.step(tok, n - 1 + i, true)) return false;
        t->logits.emplace_back(static_cast<size_t>(nv));
        if (!g.read_logits(t->logits.back().data())) return false;
        tok = g.argmax();
        t->toks.push_back(tok);
    }
    return true;
}

bool traces_equal(const Trace& a, const Trace& b, double* worst) {
    if (a.logits.size() != b.logits.size()) return false;
    bool eq = a.toks == b.toks;
    *worst = 0;
    for (size_t i = 0; i < a.logits.size(); ++i) {
        *worst = std::max(*worst, max_diff(a.logits[i], b.logits[i]));
        eq = eq && same_bits(a.logits[i], b.logits[i]);
    }
    return eq;
}

}  // namespace

int main() {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        std::printf("SKIP cuda_state_test: no CUDA device\n");
        return 0;
    }
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    if (free_b < static_cast<size_t>(15.5e9)) {
        std::printf("SKIP cuda_state_test: %.2f GB free of %.2f GB, need 15.5 GB (the desktop or another engine holds the rest)\n",
                    double(free_b) / 1e9, double(total_b) / 1e9);
        return 0;
    }
    const char* env = std::getenv("LIMBO_TEST_GGUF");
    const std::string path = env && *env && !std::strchr(env, ':') ? env : kRung1;
    std::string err;
    auto model = limbo::model::Model::open({path}, &err);
    if (!model) {
        std::printf("SKIP cuda_state_test: cannot open %s: %s\n", path.c_str(), err.c_str());
        return 0;
    }
    constexpr int kCtx = 512;
    auto gpu = Gpu::open(*model, kCtx, static_cast<uint64_t>(14.6e9), &err, limbo::cuda::KvFormat::F16, 6);
    CHECK(gpu != nullptr, "the engine opens");
    if (!gpu) {
        std::printf("     %s\n", err.c_str());
        return limbo::test::finish("cuda_state_test");
    }
    Gpu& g = *gpu;
    g.set_prefill(3, 64);  // chunks of 64: a prompt of 160 tokens is three chunks, and the snapshots sit between calls

    // ---- sizes -----------------------------------------------------------------------------------------
    const uint64_t sb = g.snapshot_bytes();
    // 30 delta-net layers x (S 32 x 128 x 128 fp32 + conv history 3 x 8192 fp32), each part 256-byte aligned
    CHECK(sb == 30ull * (2097152 + 98304), "a snapshot is 30 x (2,097,152 + 98,304) bytes");
    std::printf("     one snapshot: %llu bytes (%.2f MiB)\n", (unsigned long long)sb, double(sb) / 1048576.0);
    CHECK(g.set_snapshot_slots(3) == 3 && g.snapshot_slots() == 3, "three snapshot slots allocate");
    std::printf("     3 slots: %.1f MiB of VRAM (state_vram_bytes %llu)\n", 3.0 * double(sb) / 1048576.0, (unsigned long long)g.state_vram_bytes());

    constexpr int N = 160, P1 = 48, P2 = 96, STEPS = 8;
    const std::vector<int32_t> A = random_ids(N + STEPS, 7), B = random_ids(N + STEPS, 4242);

    // ---- the uninterrupted reference ---------------------------------------------------------------------
    Trace ref;
    g.reset();
    bool ok = g.prefill(A.data(), P1, 0, false) && g.prefill(A.data() + P1, P2 - P1, P1, false) && finish_run(g, A, P2, N, STEPS, &ref);
    CHECK(ok && ref.toks.size() == size_t(STEPS) + 1, "the reference run (prefill 0..48, 48..96, 96..159, step 159, 8 greedy steps)");

    // ---- 1. a snapshot does not disturb the run -----------------------------------------------------------
    {
        Trace t;
        g.reset();
        ok = g.prefill(A.data(), P1, 0, false) && g.snapshot(0) && g.prefill(A.data() + P1, P2 - P1, P1, false) && g.snapshot(1) &&
             finish_run(g, A, P2, N, STEPS, &t);
        double w = 0;
        const bool eq = ok && traces_equal(t, ref, &w);
        std::printf("     snapshots taken at %d and %d inside the run: max|dlogit| %.3e vs the reference\n", P1, P2, w);
        CHECK(eq, "1. taking snapshots leaves the logits and the next 8 greedy tokens bit-identical");
    }

    // ---- 2. restore over a different branch ---------------------------------------------------------------
    // State now: A's whole run (K/V of positions 0..167, recurrent state at 168). Run branch B over positions
    // P1.. instead (stale K/V everywhere past P1, a different recurrent state), then rewind to P1 / P2.
    for (int from : {P1, P2}) {
        Trace dirty, t;
        const int slot = from == P1 ? 0 : 1;
        ok = g.restore(slot) && g.prefill(B.data() + from, N - from, from, false);  // the other branch, prompt part
        for (int i = 0; ok && i < STEPS; ++i) ok = g.step(B[static_cast<size_t>(N + i)], N + i, i == STEPS - 1);  // and decode steps
        CHECK(ok, "2. a different branch runs over the same positions");
        if (from == P1) {  // the control: without the restore the same re-prefill must NOT reproduce the run
            Trace c;
            const bool cok = g.prefill(A.data() + P1, P2 - P1, P1, false) && finish_run(g, A, P2, N, STEPS, &c);
            double cw = 0;
            std::printf("     control, no restore (the other branch's recurrent state): max|dlogit| %.3e\n", cw = [&] { double m = 0; for (size_t i = 0; i < c.logits.size() && i < ref.logits.size(); ++i) m = std::max(m, max_diff(c.logits[i], ref.logits[i])); return m; }());
            CHECK(cok && cw > 0, "2. control: re-prefilling over the other branch WITHOUT restore does not reproduce the run (the test can fail)");
        }
        // rewind: restore(from), then the uninterrupted run's own tokens from `from` on, in the same chunks
        ok = g.restore(slot);
        if (ok && from == P1) ok = g.prefill(A.data() + P1, P2 - P1, P1, false) && finish_run(g, A, P2, N, STEPS, &t);
        else if (ok) ok = finish_run(g, A, P2, N, STEPS, &t);
        double w = 0;
        const bool eq = ok && traces_equal(t, ref, &w);
        std::printf("     rewind to %d over a diverged branch (stale K/V past %d): max|dlogit| %.3e over %zu logit vectors, tokens %s\n", from,
                    from, w, t.logits.size(), t.toks == ref.toks ? "equal" : "DIFFERENT");
        CHECK(eq, from == P1 ? "2. restore(snapshot at 48) + re-prefill == the uninterrupted run, bit for bit"
                             : "2. restore(snapshot at 96) + re-prefill == the uninterrupted run, bit for bit");
    }

    // ---- 3. the older snapshot is still there after the newer one was used ---------------------------------
    {
        Trace t;
        ok = g.restore(0) && g.prefill(A.data() + P1, P2 - P1, P1, false) && finish_run(g, A, P2, N, STEPS, &t);
        double w = 0;
        CHECK(ok && traces_equal(t, ref, &w), "3. a slot restores more than once (restore(0) again after restore(1))");
    }

    // ---- 4. refusals ---------------------------------------------------------------------------------------
    CHECK(!g.snapshot(-1) && !g.snapshot(3) && !g.restore(-1) && !g.restore(3), "4. out-of-range slots are refused");
    CHECK(!g.restore(2), "4. a slot that was never written is refused");

    // ---- 5. cost ---------------------------------------------------------------------------------------------
    {
        constexpr int R = 50;
        g.sync();
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < R; ++i) g.snapshot(2);
        g.sync();
        const double snap_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / R;
        const auto t1 = std::chrono::steady_clock::now();
        for (int i = 0; i < R; ++i) g.restore(2);
        g.sync();
        const double rest_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count() / R;
        std::printf("     snapshot %.3f ms, restore %.3f ms (%d each, back to back, synchronised once; %.2f MiB)\n", snap_ms, rest_ms, R,
                    double(sb) / 1048576.0);
        CHECK(snap_ms < 5.0 && rest_ms < 5.0, "5. a snapshot and a restore each take under 5 ms");
    }

    // ---- 7 (information). does the prefill chunking move the logits? ---------------------------------------
    {
        Trace t;
        g.reset();
        ok = finish_run(g, A, 0, N, STEPS, &t);  // one prefill call over 0..158 in chunks of 64
        double w = 0;
        const bool eq = ok && traces_equal(t, ref, &w);
        std::printf("     [info] one prefill call 0..158 vs three calls split at %d and %d: max|dlogit| %.3e, tokens %s, logits %s\n", P1, P2, w,
                    t.toks == ref.toks ? "equal" : "DIFFERENT", eq ? "bit-identical" : "not bit-identical");
    }

    // ---- 7b (information). a piece shorter than 8 tokens takes the token-by-token path -----------------------
    {
        Trace t;
        g.reset();
        ok = g.prefill(A.data(), N - 8, 0, false) && g.prefill(A.data() + (N - 8), 7, N - 8, false);  // the last 7 prompt tokens: below the batched path's 8
        int32_t tok = A[size_t(N - 1)];
        for (int i = 0; ok && i <= STEPS; ++i) {
            ok = g.step(tok, N - 1 + i, true);
            t.logits.emplace_back(static_cast<size_t>(g.n_vocab()));
            ok = ok && g.read_logits(t.logits.back().data());
            tok = g.argmax();
            t.toks.push_back(tok);
        }
        double w = 0;
        const bool eq = ok && traces_equal(t, ref, &w);
        std::printf("     [info] a 7-token tail piece (token by token) vs the batched path: max|dlogit| %.3e, tokens %s, logits %s\n", w,
                    t.toks == ref.toks ? "equal" : "DIFFERENT", eq ? "bit-identical" : "not bit-identical");
    }

    // ---- 6. slots ----------------------------------------------------------------------------------------------
    {
        const uint64_t keep_free = 256ull << 20;
        const int n_slots = g.set_slots(2, keep_free);
        CHECK(n_slots == 2 && g.slots() == 2 && g.slot() == 0, "6. a second slot allocates");
        std::printf("     one extra slot at n_ctx %d: %.1f MiB (recurrent state; the K/V is the pool's)\n", kCtx, double(g.slot_bytes()) / 1048576.0);
        if (n_slots == 2) {
            const int nv = g.n_vocab();
            // Home 0 reads 104 tokens, then (optionally) a side conversation runs in slot 1, then slot 0 reads on
            // and decodes. The calls on slot 0 are the same either way.
            auto main_run = [&](bool side_call, Trace* out, Trace* side) {
                g.reset();
                bool r = g.prefill(A.data(), P2, 0, false) && g.prefill(A.data() + P2, 8, P2, false);
                if (side_call) {
                    r = r && g.select_slot(1) && g.slot() == 1;
                    g.reset();  // acts on the live slot: the side slot starts empty (it already is)
                    r = r && finish_run(g, B, 0, 120, STEPS, side);  // a prompt, a step and 8 decode steps: the graphs get captured in slot 1
                    r = r && g.select_slot(0) && g.slot() == 0;
                }
                r = r && g.prefill(A.data() + P2 + 8, N - 1 - (P2 + 8), P2 + 8, false);
                int32_t tok = A[size_t(N - 1)];
                for (int i = 0; r && i <= STEPS; ++i) {
                    r = g.step(tok, N - 1 + i, true);
                    out->logits.emplace_back(size_t(nv));
                    r = r && g.read_logits(out->logits.back().data());
                    tok = g.argmax();
                    out->toks.push_back(tok);
                }
                return r;
            };
            Trace alone, with_side, side, unused;
            ok = main_run(false, &alone, &unused);
            CHECK(ok, "6. slot 0 alone");
            ok = main_run(true, &with_side, &side);
            CHECK(ok, "6. slot 0 with a conversation in slot 1 in the middle");
            CHECK(!g.select_slot(2) && !g.select_slot(-1) && g.slot() == 0, "6. slots that do not exist are refused");
            double w = 0;
            const bool eq = traces_equal(with_side, alone, &w);
            std::printf("     slot 0 across a side conversation in slot 1: max|dlogit| %.3e, tokens %s\n", w, with_side.toks == alone.toks ? "equal" : "DIFFERENT");
            CHECK(eq, "6. a side call in slot 1 leaves slot 0's logits and greedy tokens bit-identical (the pointer select moved no state)");
            // slot 1 kept its own conversation: its next step equals the same conversation run alone in slot 0
            std::vector<float> h1(static_cast<size_t>(nv)), h0(static_cast<size_t>(nv));
            ok = g.select_slot(1) && g.step(side.toks.back(), 120 + STEPS, true) && g.read_logits(h1.data());
            ok = ok && g.select_slot(0);
            g.reset();
            Trace side0;
            ok = ok && finish_run(g, B, 0, 120, STEPS, &side0) && g.step(side0.toks.back(), 120 + STEPS, true) && g.read_logits(h0.data());
            std::printf("     slot 1 resumed after slot 0 ran in between: max|dlogit| %.3e against the same conversation alone in slot 0\n", max_diff(h1, h0));
            CHECK(ok && side0.toks == side.toks && same_bits(h1, h0), "6. slot 1 kept its own K/V and recurrent state while slot 0 was live");
            std::printf("     state VRAM: %.1f MiB (3 snapshot slots + 1 extra slot)\n", double(g.state_vram_bytes()) / 1048576.0);
        }
    }

    // ---- 7. pages: a slot adopts another's prefix, copy on write, release ----------------------------------------------
    if (g.slots() == 2) {
        const std::vector<int32_t> C = random_ids(300, 99), D = random_ids(200, 5);
        const int nv = g.n_vocab();
        const int pool_pages = g.pool_stats().pages;
        struct Seen {
            std::vector<float> l0, l1;
            Gpu::PoolStats after_read, after_share, after_write, after_release;
            int p0 = -1, p1 = -1;
            bool ok = true;
        };
        // Slot 0 reads C[0..300) with checkpoints at 100 and 280; slot 1 adopts the first 280 positions (pages shared, the page that cuts
        // 280 copied, the recurrent state restored from the checkpoint), reads C[280..300) and steps. With `overwrite`, slot 0 then goes
        // back to 100 and reads other tokens over the page it shares with slot 1 before slot 1 steps.
        auto scenario = [&](bool overwrite) {
            Seen r;
            r.ok = g.select_slot(0);
            g.reset();
            r.ok = r.ok && g.select_slot(1);
            g.reset();
            r.ok = r.ok && g.select_slot(0);
            r.ok = r.ok && g.prefill(C.data(), 100, 0, false) && g.snapshot(1) && g.prefill(C.data() + 100, 180, 100, false) && g.snapshot(0) &&
                   g.prefill(C.data() + 280, 20, 280, false);
            r.after_read = g.pool_stats();
            if (!overwrite) {
                r.l0.resize(size_t(nv));
                r.ok = r.ok && g.step(4242, 300, true) && g.read_logits(r.l0.data());
            }
            r.ok = r.ok && g.select_slot(1) && g.slot_share(0, 280) && g.restore(0);
            r.after_share = g.pool_stats();
            r.ok = r.ok && g.prefill(C.data() + 280, 20, 280, false);
            if (overwrite) {
                r.ok = r.ok && g.select_slot(0) && g.restore(1) && g.prefill(D.data(), 150, 100, false);
                r.after_write = g.pool_stats();
                r.p0 = g.page_of(0, 0);
                r.p1 = g.page_of(1, 0);
                r.ok = r.ok && g.select_slot(1);
            }
            r.l1.resize(size_t(nv));
            r.ok = r.ok && g.step(4242, 300, true) && g.read_logits(r.l1.data());
            return r;
        };
        const Seen a = scenario(false), b = scenario(true);
        CHECK(a.ok && b.ok, "7. both scenarios ran");
        CHECK(a.after_read.used == 2 && a.after_share.used == 3 && a.after_share.shared == 1,
              "7. slot 0's 300 positions hold 2 pages; slot 1 adopting 280 of them takes ONE more (1 shared + the copied page), not two");
        CHECK(same_bits(a.l0, a.l1), "7. the slot that adopted the prefix continues bit for bit like the slot that read it");
        CHECK(b.after_write.used == 4 && b.p0 != b.p1 && b.p0 >= 0 && b.p1 >= 0,
              "7. slot 0 rewriting inside the shared page took a private page (copy on write): 4 pages, slot 0 and slot 1 now differ on chunk 0");
        CHECK(same_bits(a.l1, b.l1), "7. ... and slot 1's logits did not move");
        std::printf("     pages of 256 positions (pool %d): after slot 0 read 300: %d used; slot 1 adopts 280: %d used, %d shared; after the overwrite: %d used\n",
                    pool_pages, a.after_read.used, a.after_share.used, a.after_share.shared, b.after_write.used);
        g.release_slot(1);
        CHECK(g.pool_stats().used == 2 && g.slot_pages(1) == 0 && g.slot_pages(0) == 2, "7. release_slot(1) gives its pages back");
        g.select_slot(0);
        g.slot_cut(256);
        CHECK(g.slot_pages(0) == 1, "7. slot_cut(256) keeps the one page that holds positions 0..255");
        CHECK(g.slot_reserve(300) && g.slot_pages(0) == 2, "7. slot_reserve(300) takes the page for 256..299");
        g.reset();
        CHECK(g.pool_stats().used == 0 && g.slot_pages(0) == 0, "7. reset() gives the slot's pages back: the pool is empty");
        CHECK(g.slot_room(0) == pool_pages * 256 || g.slot_room(0) == 2 * 256, "7. an empty slot may take the whole pool (up to what its table addresses)");
    }
    return limbo::test::finish("cuda_state_test");
}
