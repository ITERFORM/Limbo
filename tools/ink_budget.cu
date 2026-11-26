// Copyright (c) 2026 ITERFORM. All rights reserved.
// tools/ink_budget.cu — the inkling2_loop memory and bandwidth budget on the desktop (docs/inkling2.md §7
// item 3). Every number is measured or computed from the file's own tensors; nothing is asserted:
//
//   1. the file account: bytes per weight class, the routed experts split into the looped core (read
//      loop_count times per token) and the rest;
//   2. the state account: the per-slot f16 K/V and short-conv histories and the step scratch, by geometry;
//   3. the measured residency: cudaMemGetInfo around InkGpu::open at several n_ctx, the fitted
//      bytes-per-position slope checked against the state account, and where the card runs out -- under
//      the current per-slot n_ctx allocation and under the sliding-window ring (the sliding slots hold
//      window-sized caches, not n_ctx, so the ring is what the account says they cost);
//   4. the per-token reads of one decode step at several context lengths and the tok/s those bytes bound
//      at the card's 288 GB/s, against the measured InkGpu step rate at short context;
//   5. the rung-2 case: the same per-token expert reads from host RAM over the measured 6.6 GB/s PCIe
//      (PLAN.md §2), i.e. what the loop's extra passes cost wherever the experts live.
//
//   ink_budget [file.gguf]      (default LIMBO_TEST_INK_GGUF or the Inkling-2-Mini UD-IQ2_XXS)
//
// Exits non-zero when a check fails.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "../src/cuda/ink_gpu.hpp"
#include "limbo/model.hpp"

namespace {

namespace cu = limbo::cuda;
using limbo::model::Geometry;
using limbo::model::Model;

constexpr const char* kModel = "/home/alca/models/inkling/Inkling-2-Mini-UD-IQ2_XXS.gguf";
constexpr double kGib = 1073741824.0, kMib = 1048576.0;
// The card's memory bandwidth (PLAN.md §2's box table) and the measured PCIe lane rate.
constexpr double kVramBs = 288e9, kPcieBs = 6.6e9;

int failures = 0;

void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}

int layer_of(const std::string& n) {
    if (n.rfind("blk.", 0) != 0) return -1;
    return std::atoi(n.c_str() + 4);
}

bool contains(const std::string& n, const char* s) { return n.find(s) != std::string::npos; }

size_t free_vram() {
    size_t f = 0, t = 0;
    return cudaMemGetInfo(&f, &t) == cudaSuccess ? f : 0;
}

}  // namespace

int main(int argc, char** argv) {
    const char* env = std::getenv("LIMBO_TEST_INK_GGUF");
    const std::string path = argc > 1 ? argv[1] : (env && *env ? env : kModel);

    int dev = 0;
    if (cudaGetDeviceCount(&dev) != cudaSuccess || dev == 0) {
        std::printf("ink_budget: SKIP (no GPU)\n");
        return 0;
    }
    size_t vram_total = 0, vram_free0 = 0;
    cudaMemGetInfo(&vram_free0, &vram_total);

    std::string err;
    std::unique_ptr<Model> m = Model::open({path}, &err);
    if (!m) {
        std::printf("ink_budget: %s\n", err.c_str());
        return 1;
    }
    const Geometry& g = m->geometry();

    const int kvw = g.n_head_kv * g.head_dim;
    const int nch = 2 * kvw + 2 * g.n_embd;
    const int n_slot = g.n_layer + (g.loop_count - 1) * (g.loop_end - g.loop_start);
    const double kv_pos = 2.0 * kvw * 2.0;   // one f16 K row and one V row per position per slot
    const double hist_slot = double(g.sconv - 1) * nch * 4.0;
    int n_swa_l = 0;
    for (uint8_t s : g.swa_l) n_swa_l += s != 0 ? 1 : 0;

    std::printf("ink_budget: %s\n", path.c_str());
    std::printf("geometry: %d layers, n_embd %d, %d heads / %d kv (head_dim %d), %d experts (%d used), "
                "ff %d\n",
                g.n_layer, g.n_embd, g.n_head, g.n_head_kv, g.head_dim, g.n_expert, g.n_expert_used,
                g.n_ff_exp);
    std::printf("          loop %d..%d x%d -> %d visits per token, %d state slots; sliding window %d on "
                "%d layers, global on %d\n",
                g.loop_start, g.loop_end, g.loop_count, n_slot, n_slot, g.sliding_window, n_swa_l,
                g.n_layer - n_swa_l);
    {
        double f = 0;
        for (const auto* t : m->tensors()) f += double(t->nbytes);
        std::printf("file:     %.2f GiB of tensors (%zu)\n\n", f / kGib, m->tensors().size());
    }

    // ---- 1. the file account ---------------------------------------------------------------------------
    // Classes: the routed experts (split looped core vs the rest), the per-layer dense weights, the
    // embeddings and the output head. A layer's dense bytes are every tensor of that block that is not an
    // expert tensor; the looped core's layers (loop_l) are visited loop_count times per token.
    std::vector<double> dense(size_t(g.n_layer), 0.0), expert(size_t(g.n_layer), 0.0);
    double exp_scale = 0, embd_b = 0, head_b = 0, other = 0;
    for (const auto* t : m->tensors()) {
        const std::string n(t->name);
        const int l = layer_of(n);
        if (contains(n, "_exps")) {
            if (contains(n, ".scale")) exp_scale += double(t->nbytes);
            else if (l >= 0) expert[size_t(l)] += double(t->nbytes);
            else other += double(t->nbytes);
        } else if (n == "output.weight") head_b += double(t->nbytes);
        else if (n == "token_embd.weight") embd_b += double(t->nbytes);
        else if (l >= 0) dense[size_t(l)] += double(t->nbytes);
        else other += double(t->nbytes);
    }
    double exp_loop = 0, exp_rest = 0, den_loop = 0, den_rest = 0, one_expert = 0;
    for (int l = 0; l < g.n_layer; ++l) {
        const bool looped = std::find(g.loop_l.begin(), g.loop_l.end(), l) != g.loop_l.end();
        (looped ? exp_loop : exp_rest) += expert[size_t(l)];
        (looped ? den_loop : den_rest) += dense[size_t(l)];
    }
    for (int l : g.loop_l) one_expert += expert[size_t(l)] / g.n_expert;
    one_expert /= std::max(1, int(g.loop_l.size()));
    {
        const double all = exp_loop + exp_rest;
        exp_loop += exp_scale * (exp_loop / all);
        exp_rest += exp_scale * (exp_rest / all);
    }

    std::printf("1. the file account\n");
    std::printf("  routed experts, looped layers %d..%d (read %dx per token)  %9.2f MiB\n", g.loop_start,
                g.loop_end - 1, g.loop_count, exp_loop / kMib);
    std::printf("  routed experts, prelude + coda (read 1x per token)        %9.2f MiB\n", exp_rest / kMib);
    std::printf("  dense per-layer weights, looped core (read %dx)           %9.2f MiB\n", g.loop_count,
                den_loop / kMib);
    std::printf("  dense per-layer weights, prelude + coda (read 1x)        %9.2f MiB\n", den_rest / kMib);
    std::printf("  token embeddings                                         %9.2f MiB\n", embd_b / kMib);
    std::printf("  output head                                              %9.2f MiB\n", head_b / kMib);
    std::printf("  norms / queries / biases (global)                        %9.2f MiB\n", other / kMib);
    std::printf("  one expert at the looped core's layers: %.3f MiB (gate/up + down), %d used per visit\n\n",
                one_expert / kMib, g.n_expert_used);

    // ---- 2. the state account --------------------------------------------------------------------------
    std::printf("2. the state account (per slot, per position)\n");
    std::printf("  f16 K/V: %.1f KiB per position per slot; %d slots -> %.1f KiB per position\n",
                kv_pos / 1024.0, n_slot, kv_pos * n_slot / 1024.0);
    std::printf("  short-conv histories: %.2f MiB total (%d slots x %d rows x %d floats)\n",
                hist_slot * n_slot / kMib, n_slot, g.sconv - 1, nch);
    std::printf("  the fused bias table: %.0f KiB of scratch (%d x %d floats), rebuilt per visit -- the\n"
                "  dense [q x kv] form is never materialised (docs/inkling2.md 6)\n",
                (std::max(g.sliding_window, g.rel_extent) + 1.0) * g.n_head * 4.0 / 1024.0,
                std::max(g.sliding_window, g.rel_extent) + 1, g.n_head);
    std::printf("  logits: %.2f MiB (%d unpadded rows)\n\n", g.n_vocab_unpadded * 4.0 / kMib,
                g.n_vocab_unpadded);

    // ---- 3. the measured residency ---------------------------------------------------------------------
    // InkGpu's state is f16 K/V [n_slot][n_ctx][kvw] plus the histories and l_out; the account's slope is
    // n_slot * kv_pos bytes per position. Measured as the free-VRAM delta around open() at several n_ctx.
    std::printf("3. the measured residency (InkGpu::open, free-VRAM delta)\n");
    std::vector<double> xs, ys;
    const int ctxs[] = {256, 1024, 4096, 16384};
    for (int c : ctxs) {
        const size_t before = free_vram();
        std::unique_ptr<cu::InkGpu> gpu = cu::InkGpu::open(*m, c, 0, &err);
        if (!gpu) {
            std::printf("  n_ctx %6d: does not fit (%s)\n", c, err.c_str());
            continue;
        }
        const double delta = double(before - free_vram());
        std::printf("  n_ctx %6d: %9.2f MiB resident (%.2f GiB of the %.2f GiB card free before)\n", c,
                    delta / kMib, double(before) / kGib, double(vram_total) / kGib);
        xs.push_back(double(c));
        ys.push_back(delta);
        gpu.reset();
    }
    double slope = 0, base = 0;
    {
        double sx = 0, sy = 0, sxx = 0, sxy = 0;
        const int n = int(xs.size());
        for (int i = 0; i < n; ++i) {
            sx += xs[size_t(i)];
            sy += ys[size_t(i)];
            sxx += xs[size_t(i)] * xs[size_t(i)];
            sxy += xs[size_t(i)] * ys[size_t(i)];
        }
        if (n >= 2) {
            slope = (n * sxy - sx * sy) / std::max(1e-9, n * sxx - sx * sx);
            base = (sy - slope * sx) / n;
        }
    }
    std::printf("  fit: %.1f KiB per position (account: %.1f KiB), %.2f MiB fixed at n_ctx 0\n",
                slope / 1024.0, kv_pos * n_slot / 1024.0, base / kMib);
    check(xs.size() >= 2 && std::fabs(slope - kv_pos * n_slot) / (kv_pos * n_slot) < 0.01,
          "the measured per-position slope matches the state account (48 slots x 4 KiB)");
    // Where the card runs out: the fit against the free VRAM this box has beside the desktop, and the
    // sliding-window ring account (the sliding slots only ever hold `window` positions).
    double swa_slots = 0, gl_slots = 0;
    for (int s = 0; s < n_slot; ++s) {
        const int l = s < g.n_layer ? s : g.loop_start + (s - g.n_layer) % (g.loop_end - g.loop_start);
        (g.is_swa(l) ? swa_slots : gl_slots) += 1;
    }
    {
        const double room = double(vram_free0) - base;
        const double ctx_now = room / (kv_pos * n_slot);
        const double ctx_ring =
            (room - swa_slots * g.sliding_window * kv_pos) / (kv_pos * gl_slots);
        std::printf("  resident context on this box: ~%d positions as allocated;\n", int(ctx_now));
        std::printf("                              ~%d positions with the sliding-window ring\n",
                    int(std::max(0.0, ctx_ring)));
    }
    std::printf("\n");

    // ---- 4. the per-token reads ------------------------------------------------------------------------
    // One decode step, one token: every visit reads its physical layer's dense weights and n_expert_used
    // experts, the slot's K/V up to the window on the sliding layers (the full history on the global ones),
    // and the conv history; with the head on, the output head's rows too.
    std::vector<int> reads(size_t(g.n_layer), 1);
    for (int l = 0; l < g.n_layer; ++l)
        if (std::find(g.loop_l.begin(), g.loop_l.end(), l) != g.loop_l.end()) reads[size_t(l)] = g.loop_count;
    double dense_token = 0, expert_token = 0, hist_token = 0;
    for (int l = 0; l < g.n_layer; ++l) {
        dense_token += reads[size_t(l)] * dense[size_t(l)];
        expert_token += reads[size_t(l)] * g.n_expert_used * (expert[size_t(l)] / g.n_expert);
        hist_token += reads[size_t(l)] * hist_slot;
    }
    const double embd_row = embd_b / std::max(1, g.n_vocab);
    std::printf("4. the per-token device reads of one decode step (MiB)\n");
    std::printf("  %-46s %9s %9s %9s %9s\n", "at context", "512", "4,096", "32,768", "262,144");
    const double ctx_list[] = {512, 4096, 32768, 262144};
    double kv_row[4], tot_row[4], tok_row[4];
    for (int i = 0; i < 4; ++i) {
        const double len = ctx_list[i];
        double kv = 0;
        for (int l = 0; l < g.n_layer; ++l) {
            const double per_visit = std::min(len, g.is_swa(l) ? double(g.sliding_window) : len);
            kv += reads[size_t(l)] * per_visit * kv_pos;
        }
        kv_row[i] = kv;
        tot_row[i] = dense_token + expert_token + head_b + embd_row + hist_token + kv;
        tok_row[i] = kVramBs / tot_row[i];
    }
    std::printf("  %-46s %9.0f %9.0f %9.0f %9.0f\n", "weights: dense (looped core 3x)", dense_token / kMib,
                dense_token / kMib, dense_token / kMib, dense_token / kMib);
    std::printf("  %-46s %9.0f %9.0f %9.0f %9.0f\n", "weights: 8 of 128 experts per visit",
                expert_token / kMib, expert_token / kMib, expert_token / kMib, expert_token / kMib);
    std::printf("  %-46s %9.0f %9.0f %9.0f %9.0f\n", "output head", head_b / kMib, head_b / kMib,
                head_b / kMib, head_b / kMib);
    std::printf("  %-46s %9.0f %9.0f %9.0f %9.0f\n", "K/V reads (sliding capped at the window)",
                kv_row[0] / kMib, kv_row[1] / kMib, kv_row[2] / kMib, kv_row[3] / kMib);
    std::printf("  %-46s %9.0f %9.0f %9.0f %9.0f\n", "conv histories + embedding row", hist_token / kMib,
                hist_token / kMib, hist_token / kMib, hist_token / kMib);
    std::printf("  %-46s %9.0f %9.0f %9.0f %9.0f\n", "total per token", tot_row[0] / kMib, tot_row[1] / kMib,
                tot_row[2] / kMib, tot_row[3] / kMib);
    std::printf("  %-46s %9.1f %9.1f %9.1f %9.1f\n", "bound at 288 GB/s (tok/s)", tok_row[0], tok_row[1],
                tok_row[2], tok_row[3]);
    std::printf("\n");

    // The measured InkGpu decode rate at short context: 8 warmup + 64 head steps, one stream, no host
    // round trip inside the pass. The traffic account must bound it from above.
    {
        std::unique_ptr<cu::InkGpu> gpu = cu::InkGpu::open(*m, 512, 0, &err);
        if (!gpu) {
            std::printf("  InkGpu bench: %s\n", err.c_str());
            ++failures;
        } else {
            for (int i = 0; i < 8; ++i) gpu->step(1910 + i % 32, i, true);
            gpu->sync();
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < 64; ++i) gpu->step(1910 + (i * 7919) % 32, 8 + i, true);
            gpu->sync();
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            const double tps = 64.0 / s;
            std::printf("  measured: InkGpu decode %.1f tok/s at n_ctx 512 (64 head steps, one stream)\n", tps);
            std::printf("            %.0f MiB/token in the account -> %.0f%% of the 288 GB/s bound\n",
                        tot_row[0] / kMib, 100.0 * (tot_row[0] * tps) / kVramBs);
            check(tps <= tok_row[0] * 1.01, "the measured rate is inside the traffic account's bound");
        }
    }
    std::printf("\n");

    // ---- 5. the rung-2 case: experts in host RAM --------------------------------------------------------
    // The loop multiplies every non-resident expert read by loop_count on the looped core: each of the
    // three passes routes its own 8 experts, so a host-resident expert set is pulled 3x per token there.
    // The lane is the measured 6.6 GB/s PCIe (PLAN.md 2); PLAN.md 3.1's crossover says a decode row count
    // (1 row per expert product) computes on the CPU from pinned RAM rather than crossing the bus.
    std::printf("5. the rung-2 case (experts wherever they live)\n");
    std::printf("  expert bytes read per token: %.0f MiB (%.0f MiB of it the looped core's extra %d passes)\n",
                expert_token / kMib, expert_token * (g.loop_count - 1.0) / g.loop_count / kMib,
                g.loop_count - 1);
    std::printf("  from host RAM over the 6.6 GB/s PCIe link: %.2f s per token -> %.2f tok/s\n",
                expert_token / kPcieBs, kPcieBs / expert_token);
    std::printf("  one expert pull: %.3f MiB, ~%.0f us on the link at that rate; the %d looped layers route\n"
                "  %d experts per pass, %d per token with the loop's third passes\n",
                one_expert / kMib, one_expert / kPcieBs * 1e6, int(g.loop_l.size()), g.n_expert_used,
                g.n_expert_used * g.loop_count * int(g.loop_l.size()));
    std::printf("  verdict: a decode row is 1 row per expert product -- PLAN.md 3.1 puts that under the\n"
                "  ~10-row crossover, so host experts compute on the CPU from pinned RAM (or stay pinned\n"
                "  in the kindi slot table), and the loop's extra passes are what multiply the pull count.\n");

    std::printf("\nink_budget: %s (%d failure(s))\n", failures ? "FAIL" : "all checks passed", failures);
    return failures ? 1 : 0;
}
