// Copyright (c) 2026 ITERFORM. All rights reserved.
// tests/cuda_ink_step_test.cu — the whole inkling2_loop forward pass on the GPU (src/cuda/ink_gpu.cu) against
// the CPU reference (limbo::step::Ref, src/step.cpp) on the real model, teacher-forced. The two run the same
// graph with the same weights; only the accumulation order differs. The gate, in the W1/oracle-gate
// methodology (docs/inkling2.md):
//
//   1. position 0: every one of the visit slots' l_out (Ref's Trace against InkGpu::read_l_out), bar 0.05;
//   2. the greedy token agrees at EVERY position (the behavioral gate);
//   3. max |dlogit| <= 0.05 over the first 8 positions (measured 1e-3: pure accumulation-order noise), and
//      <= 3.0 over the whole run;
//   4. reset() + a second identical run: the logits are bit-identical at every position.
//
// Why the split on 3: the graph is position-independent and the router's top-k is the only O(1)
// discontinuity in it. The two trajectories' MoE inputs differ by the accumulation-order noise (growing with
// the cached positions), and once that noise crosses an 8th/9th expert margin the two runs select different
// experts -- both valid under the model's own tie rule -- and their logits legitimately differ by the swapped
// expert's contribution. The measured profile shows exactly that: <= 0.001 through position 15, a
// discontinuous jump at the first crossing, and the crossing's effect carried by the caches afterwards. The
// greedy token is the gate that survives it (it agreed at all 32 positions), and the full per-position
// profile is printed so every position stays visible.
//
// The model is LIMBO_TEST_INK_GGUF or /home/alca/models/inkling/Inkling-2-Mini-UD-IQ2_XXS.gguf. No GPU, no
// model file, or under 2 GB of free VRAM beside the desktop: SKIP, exit 0.
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

#include "../src/cuda/ink_gpu.hpp"
#include "limbo/check.hpp"
#include "limbo/model.hpp"
#include "limbo/step.hpp"

namespace {

namespace cu = limbo::cuda;

constexpr const char* kModel = "/home/alca/models/inkling/Inkling-2-Mini-UD-IQ2_XXS.gguf";
constexpr int kCtx = 64, kNTok = 32;

// The oracle's prompt and the first 30 ids of its greedy continuation / runner-up choices
// (bench/oracle/inkling2-hi.jsonl): teacher-forced, every id below the model's vocab_size_unpadded.
const int32_t kTokens[kNTok] = {
    1910,  198,   72,    149952, 149878, 150383, 149899, 150084, 220,   42775, 150902,
    13,    198,   72,    149952, 149878, 220,    40059,  40,     150087, 149916, 150377,
    13,    149876, 150406, 17,     13,     150028, 149900, 1202,   150480, 153457};

// The greedy token of a host logits row: the highest logit, the lowest id on a tie (the same rule as
// InkGpu::argmax).
int32_t argmax_of(const float* v, int n) {
    int32_t best = 0;
    for (int i = 1; i < n; ++i)
        if (v[i] > v[best]) best = i;
    return best;
}

}  // namespace

int main() {
    int dev = 0;
    if (cudaGetDeviceCount(&dev) != cudaSuccess || dev == 0) {
        std::printf("cuda_ink_step_test: SKIP (no GPU)\n");
        return limbo::test::finish("cuda_ink_step");
    }
    {
        size_t free_b = 0, total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess || free_b < (size_t(2) << 30)) {
            std::printf("cuda_ink_step_test: SKIP (only %zu MB of VRAM free beside the desktop)\n",
                        free_b >> 20);
            return limbo::test::finish("cuda_ink_step");
        }
    }
    const char* env = std::getenv("LIMBO_TEST_INK_GGUF");
    const std::string path = env && *env ? env : kModel;
    if (std::FILE* f = std::fopen(path.c_str(), "rb")) {
        std::fclose(f);
    } else {
        std::printf("cuda_ink_step_test: SKIP (no model at %s)\n", path.c_str());
        return limbo::test::finish("cuda_ink_step");
    }

    std::string err;
    std::unique_ptr<limbo::model::Model> m = limbo::model::Model::open({path}, &err);
    if (!m) {
        std::printf("FAIL model: %s\n", err.c_str());
        return limbo::test::finish("cuda_ink_step");
    }
    std::unique_ptr<limbo::step::Ref> ref = limbo::step::Ref::open({path}, kCtx, 8, &err);
    if (!ref) {
        std::printf("FAIL Ref: %s\n", err.c_str());
        return limbo::test::finish("cuda_ink_step");
    }
    ref->trace().on = true;
    std::unique_ptr<cu::InkGpu> gpu = cu::InkGpu::open(*m, kCtx, 0, &err);
    if (!gpu) {
        std::printf("FAIL InkGpu: %s\n", err.c_str());
        return limbo::test::finish("cuda_ink_step");
    }
    const int nv = gpu->n_vocab();
    std::printf("cuda_ink_step: %s, %d slots, %d logits\n", path.c_str(), gpu->n_slots(), nv);

    // ---- the teacher-forced run ------------------------------------------------------------------------
    std::vector<float> lg(static_cast<size_t>(nv)), lr(static_cast<size_t>(nv));
    std::vector<float> run1(static_cast<size_t>(kNTok) * nv);
    std::vector<float> buf(static_cast<size_t>(nv));   // l_out reads (n_embd floats of them)
    bool steps_ok = true;
    double max_dlogit = 0;
    int agree = 0;
    double max_dl = 0;
    int worst = -1;
    bool louts_ok = true;
    std::vector<double> per_pos(static_cast<size_t>(kNTok), 0.0);
    for (int i = 0; i < kNTok; ++i) {
        const float* r = ref->forward(kTokens[i], i, true);
        std::memcpy(lr.data(), r, sizeof(float) * size_t(nv));
        steps_ok = gpu->step(kTokens[i], i, true) && steps_ok;
        if (!gpu->read_logits(lg.data())) steps_ok = false;
        std::memcpy(&run1[size_t(i) * nv], lg.data(), sizeof(float) * size_t(nv));
        double dpi = 0;
        for (int j = 0; j < nv; ++j) dpi = std::max(dpi, std::fabs(double(lg[size_t(j)]) - double(lr[size_t(j)])));
        per_pos[size_t(i)] = dpi;
        max_dlogit = std::max(max_dlogit, dpi);
        if (argmax_of(lr.data(), nv) == gpu->argmax()) ++agree;
        // position 0's visit slots: the Ref's trace against the GPU's l_out (the buffer is the last step's,
        // so it is read back before the next step overwrites it)
        if (i == 0) {
            const std::vector<std::vector<float>>& lref = ref->trace().l_out;
            louts_ok = int(lref.size()) == gpu->n_slots();
            for (int slot = 0; slot < gpu->n_slots() && slot < int(lref.size()); ++slot) {
                if (!gpu->read_l_out(slot, buf.data())) {
                    louts_ok = false;
                    continue;
                }
                double ds = 0;
                for (size_t j = 0; j < lref[size_t(slot)].size(); ++j)
                    ds = std::max(ds, std::fabs(double(buf[j]) - double(lref[size_t(slot)][j])));
                std::printf("     l_out[%2d] max |d| %.4f\n", slot, ds);
                if (ds > max_dl) {
                    max_dl = ds;
                    worst = slot;
                }
            }
        }
    }
    CHECK(steps_ok, "every step launched and its logits read back");
    std::printf("     max |dlogit| per position:");
    for (int i = 0; i < kNTok; ++i) std::printf(" %d:%.3f", i, per_pos[size_t(i)]);
    std::printf("\n");
    int first_over = kNTok;
    for (int i = 0; i < kNTok; ++i)
        if (per_pos[size_t(i)] > 0.05) {
            first_over = i;
            break;
        }
    std::printf("     first tie-break crossing at position %d; max |dlogit| %.4f over %d positions\n", first_over,
                max_dlogit, kNTok);
    CHECK(first_over >= 8,
          "at least 8 positions reproduce the CPU reference to 0.05 before the first expert tie-break crossing");
    CHECK(max_dlogit <= 3.0, "no gross divergence anywhere in the run");
    std::printf("     greedy tokens agreeing: %d / %d\n", agree, kNTok);
    CHECK(agree == kNTok, "the greedy token agrees at every position");

    CHECK(louts_ok, "Ref's trace holds one l_out per visit slot and the GPU reads them back");
    std::printf("     max |dl_out| over %d slots at position 0: %.4f (slot %d)\n", gpu->n_slots(), max_dl, worst);
    CHECK(max_dl <= 0.05, "every slot's l_out within 0.05 of the CPU reference");

    // ---- reset() + a second identical run ---------------------------------------------------------------
    gpu->reset();
    bool same = true, steps2_ok = true;
    for (int i = 0; i < kNTok; ++i) {
        steps2_ok = gpu->step(kTokens[i], i, true) && steps2_ok;
        if (!gpu->read_logits(lg.data())) steps2_ok = false;
        same = same && std::memcmp(&run1[size_t(i) * nv], lg.data(), sizeof(float) * size_t(nv)) == 0;
    }
    CHECK(steps2_ok, "the second run launched");
    CHECK(same, "after reset() a second run is bit-identical at every position");

    // ---- step_rows: the row-batched prompt pass is the bits of the per-token one ---------------------------
    // Uneven pieces (1..8 rows), the head on each piece's last row, against run1 at that position.
    {
        gpu->reset();
        const int pieces[] = {8, 1, 3, 8, 5, 7};
        bool rows_ok = true, rows_same = true;
        int pos = 0;
        for (int T : pieces) {
            rows_ok = gpu->step_rows(kTokens + pos, T, pos, true) && rows_ok;
            pos += T;
            if (!gpu->read_logits(lg.data())) rows_ok = false;
            rows_same = rows_same && std::memcmp(&run1[size_t(pos - 1) * nv], lg.data(), sizeof(float) * size_t(nv)) == 0;
        }
        CHECK(rows_ok && pos == kNTok, "step_rows launched over pieces of 8, 1, 3, 8, 5 and 7 rows");
        CHECK(rows_same, "step_rows' logits are bit-identical to step()'s at every piece end");
        CHECK(!gpu->step_rows(kTokens, cu::InkGpu::kRows + 1, pos, false), "step_rows refuses more than kRows rows");
    }

    // Past the sliding ring's wrap (1,031 rows): 1,100 positions at n_ctx 2,048, per token and in 8-row pieces,
    // compared at the piece ends 1,023 / 1,031 / 1,099.
    {
        gpu = nullptr;   // the first pass's VRAM back before the second opens
        std::unique_ptr<cu::InkGpu> g2 = cu::InkGpu::open(*m, 2048, 0, &err);
        if (!g2) {
            std::printf("FAIL InkGpu at n_ctx 2048: %s\n", err.c_str());
            ++limbo::test::failures();
        } else {
            constexpr int kN = 1100;
            const int checks[] = {1023, 1031, kN - 1};
            std::vector<int32_t> toks(kN);
            for (int i = 0; i < kN; ++i) toks[size_t(i)] = kTokens[i % kNTok];
            std::vector<float> want(3 * size_t(nv));
            bool ok1 = true;
            for (int i = 0, c = 0; i < kN; ++i) {
                const bool head = c < 3 && i == checks[c];
                ok1 = g2->step(toks[size_t(i)], i, head) && ok1;
                if (head && !g2->read_logits(&want[size_t(c++) * nv])) ok1 = false;
            }
            g2->reset();
            bool ok2 = true, eq = true;
            for (int pos = 0, c = 0; pos < kN;) {
                const int T = std::min(cu::InkGpu::kRows, kN - pos);
                const bool head = c < 3 && pos + T - 1 == checks[c];
                ok2 = g2->step_rows(toks.data() + pos, T, pos, head) && ok2;
                pos += T;
                if (head) {
                    if (!g2->read_logits(lg.data())) ok2 = false;
                    eq = eq && std::memcmp(&want[size_t(c++) * nv], lg.data(), sizeof(float) * size_t(nv)) == 0;
                }
            }
            CHECK(ok1 && ok2, "1,100 positions ran per token and in 8-row pieces");
            CHECK(eq, "past the ring's wrap step_rows is bit-identical to step() at 1,023 / 1,031 / 1,099");

            // The verify window (InkGpu::verify / commit, the draft model's check): positions 0..19 per token, then
            // [t20, t21, t22, wrong] as one verify pass, commit(2) keeps t20 and t21, and decoding goes on per token from
            // 22 -- its logits at 22..39 must be the bits of a run that never verified (the short-conv histories
            // rolled back, the rejected rows' K/V overwritten), and the verify's greedy tokens step()'s.
            {
                const int visits[3] = {5, 23, 41};
                g2->reset();
                std::vector<float> ref40(size_t(40) * nv);
                std::vector<int32_t> greedy(40);
                bool okv = true;
                for (int i = 0; i < 40; ++i) {
                    okv = g2->step(toks[size_t(i)], i, true) && okv;
                    okv = g2->read_logits(&ref40[size_t(i) * nv]) && okv;
                    greedy[size_t(i)] = argmax_of(&ref40[size_t(i) * nv], nv);
                }
                g2->set_capture(true, visits);
                g2->reset();
                for (int i = 0; i < 20; ++i) okv = g2->step(toks[size_t(i)], i, false) && okv;
                const int32_t win[4] = {toks[20], toks[21], toks[22], (toks[23] + 1) % 1000};
                int32_t out[4] = {};
                okv = g2->verify(win, 4, 20, out) && okv;
                bool same_greedy = true;
                for (int t = 0; t < 3; ++t) same_greedy = same_greedy && out[t] == greedy[size_t(20 + t)];   // row 3 read the wrong token
                okv = g2->commit(2) && okv;
                bool same = true;
                for (int i = 22; i < 40; ++i) {
                    okv = g2->step(toks[size_t(i)], i, true) && okv;
                    okv = g2->read_logits(lg.data()) && okv;
                    same = same && std::memcmp(lg.data(), &ref40[size_t(i) * nv], sizeof(float) * size_t(nv)) == 0;
                }
                g2->set_capture(false, nullptr);
                CHECK(okv, "verify, commit and the steps after them launched");
                CHECK(same_greedy, "the verify window's greedy tokens are step()'s");
                CHECK(same, "after commit(2) of a 4-row verify, decoding is bit-identical to a run that never verified");
            }

            // The prompt pass: pieces of up to kPiece rows over the int8 tensor cores. Not the bits: the planes round
            // the activations, and a router near-tie can then pick a different expert for one token (the same effect
            // that bounds step() against the CPU reference). Gated like that: per token run step() and keep its
            // logits at 24 piece ends of uneven sizes, then the same pieces through prefill_rows -- the greedy token
            // agrees at every end, the median |dlogit| is small, and nothing diverges grossly.
            {
                std::vector<int> ends;
                for (int pos = 0, i = 0; pos < kN; ++i) {
                    const int T = std::min(kN - pos, i % 3 == 0 ? cu::InkGpu::kPiece : 9 + 37 * i % 101);
                    pos += T;
                    ends.push_back(pos - 1);
                }
                g2->reset();
                std::vector<float> ref(ends.size() * size_t(nv));
                bool ok4 = true;
                for (int i = 0, c = 0; i < kN; ++i) {
                    const bool head = c < int(ends.size()) && i == ends[size_t(c)];
                    ok4 = g2->step(toks[size_t(i)], i, head) && ok4;
                    if (head && !g2->read_logits(&ref[size_t(c++) * nv])) ok4 = false;
                }
                g2->reset();
                int agree = 0;
                double maxd = 0;
                std::vector<double> med;
                const auto t0 = std::chrono::steady_clock::now();
                for (int pos = 0, c = 0; pos < kN; ++c) {
                    const int T = ends[size_t(c)] + 1 - pos;
                    ok4 = g2->prefill_rows(toks.data() + pos, T, pos, true) && ok4;
                    pos += T;
                    if (!g2->read_logits(lg.data())) ok4 = false;
                    const float* rv = &ref[size_t(c) * nv];
                    double d = 0;
                    for (int j = 0; j < nv; ++j) d = std::max(d, std::fabs(double(lg[size_t(j)]) - double(rv[j])));
                    med.push_back(d);
                    maxd = std::max(maxd, d);
                    const int ga = argmax_of(lg.data(), nv), gr = argmax_of(rv, nv);
                    // a near-tie in step()'s own logits is not a disagreement: the prompt pass's pick is within 0.5
                    // of step()'s best there
                    agree += ga == gr || rv[ga] >= rv[gr] - 0.5f;
                    if (ga != gr) std::printf("     piece end %d: greedy %d vs step() %d, step()'s gap %.4f\n", ends[size_t(c)], ga, gr, double(rv[gr] - rv[ga]));
                }
                const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                std::sort(med.begin(), med.end());
                std::printf("     prefill_rows: %d tokens in %zu pieces, %.3f s with a logits read per piece (%.0f tok/s);"
                            " greedy %d/%zu, median |dlogit| %.4f, max %.4f\n",
                            kN, ends.size(), secs, kN / secs, agree, ends.size(), med[med.size() / 2], maxd);
                CHECK(ok4, "prefill_rows ran 1,100 positions in uneven pieces");
                CHECK(agree == int(ends.size()), "prefill_rows' greedy token is step()'s at every piece end (or within 0.5 of it: a near-tie)");
                CHECK(med[med.size() / 2] <= 2.0, "prefill_rows' median |dlogit| against step() is at most 2 (measured floor at 3 planes: 0.81)");
                CHECK(maxd <= 8.0, "no gross divergence (a router near-tie moves one token, not the run)");
            }
        }
    }

    return limbo::test::finish("cuda_ink_step");
}
