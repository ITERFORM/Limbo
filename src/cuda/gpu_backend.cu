// Copyright (c) 2026 ITERFORM. All rights reserved.
// src/cuda/gpu_backend.cu — limbo::cuda::GpuBackend: serve::Backend over limbo::cuda::Gpu.
#include "limbo/cuda/backend.hpp"
#include "limbo/cuda/dflash.hpp"
#include "limbo/vlog.hpp"

#include <algorithm>
#include <chrono>

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace limbo::cuda {

std::unique_ptr<GpuBackend> GpuBackend::open(const std::vector<std::string>& shards, int n_ctx, uint64_t min_free_bytes,
                                             std::string* err, const GpuOptions& plan) {
    KvFormat fixed_fmt = KvFormat::Q4;
    if (plan.want == "f16") fixed_fmt = KvFormat::F16;
    else if (plan.want == "q8") fixed_fmt = KvFormat::Q8;
    else if (plan.want != "q4" && plan.want != "auto") {
        if (err) *err = "--kv takes auto, f16, q8 or q4, not " + plan.want;
        return nullptr;
    }
    if (plan.spec != "auto" && plan.spec != "mtp" && plan.spec != "ngram" && plan.spec != "dflash" && plan.spec != "dspark" &&
        plan.spec != "off") {
        if (err) *err = "--spec takes auto, mtp, ngram, dflash, dspark or off, not " + plan.spec;
        return nullptr;
    }
    if ((plan.spec == "dflash" || plan.spec == "dspark") && plan.draft_model.empty()) {
        if (err) *err = "--spec " + plan.spec + " needs --draft-model (the DFlash2 checkpoint)";
        return nullptr;
    }
    if (plan.draft_k < 0 || plan.draft_k > 7 || plan.draft_vocab < 0 || (plan.draft_quant != "q8" && plan.draft_quant != "bf16" && plan.draft_quant != "native" && plan.draft_quant != "auto")) {
        if (err) *err = "--draft-k takes 0..7, --draft-vocab 0 or a count, --draft-quant q8 or bf16";
        return nullptr;
    }
    if (!plan.draft_model.empty() && plan.spec != "auto" && plan.spec != "dflash" && plan.spec != "dspark") {
        if (err) *err = "--draft-model is read by --spec auto, dflash or dspark, not --spec " + plan.spec;
        return nullptr;
    }
    if (plan.ngram_n < 0 || plan.ngram_n == 1 || plan.ngram_n > 64 || plan.ngram_max < 0 || plan.ngram_max > 7) {
        if (err) *err = "--ngram-n takes 2..64 and --ngram-max 1..7";
        return nullptr;
    }
    const bool use_mtp = plan.spec == "auto" || plan.spec == "mtp", use_ngram = plan.spec == "auto" || plan.spec == "ngram";

    const bool use_df = !plan.draft_model.empty();
    const bool df_gguf = plan.draft_model.size() > 5 && plan.draft_model.compare(plan.draft_model.size() - 5, 5, ".gguf") == 0;
    const int df_quant = plan.draft_quant == "bf16" ? Dflash::kBf16 : plan.draft_quant == "native" || (plan.draft_quant == "auto" && df_gguf)
                                                                         ? Dflash::kNative
                                                                         : Dflash::kQ8;
    std::unique_ptr<GpuBackend> b(new GpuBackend());
    b->model_ = model::Model::open(shards, err);
    if (!b->model_) return nullptr;
    const bool q4e = b->model_->geometry().arch == "qwen4exp";
    if (q4e && (plan.spec == "dflash" || plan.spec == "dspark" || use_df)) {
        if (err) *err = "qwen4exp drafts with its own MTP layer and the ngram tape (--spec auto); a DFlash drafter is not its architecture";
        return nullptr;
    }
    // qwen4exp's K/V: f16 when asked for, else Q8 (Strata's int8 rows at 128K). Its expert slab takes every byte the pool
    // leaves, so the format is fixed before the slab is sized (Gpu::open reserves exactly that pool), never Q4.
    if (q4e && plan.want == "q4") {
        if (err) *err = "qwen4exp's attention reads an f16 or a Q8 K/V pool (--kv f16|q8|auto), not q4";
        return nullptr;
    }
    const KvFormat q4e_fmt = plan.want == "f16" ? KvFormat::F16 : KvFormat::Q8;
    {
        size_t fb = 0, tb = 0;
        if (cudaMemGetInfo(&fb, &tb) == cudaSuccess) b->free_at_open_ = fb;
    }
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        if (err) *err = "no CUDA device";
        return nullptr;
    }
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) b->device_ = prop.name;
    // a two-page pool: the real one is sized once the budget is known; the q4e slab leaves the MTP block's room only if it loads
    b->gpu_ = Gpu::open(*b->model_, n_ctx, min_free_bytes, err, q4e ? q4e_fmt : KvFormat::Q4, 2, use_mtp);
    if (!b->gpu_) return nullptr;
    b->n_ctx_ = n_ctx;
    {
        Gpu& g = *b->gpu_;
        const int planes = plan.prefill_planes > 0 ? plan.prefill_planes : g.prefill_planes();
        const int chunk = n_ctx > 131072 ? std::min(512, g.prefill_chunk()) : g.prefill_chunk();
        if (!g.set_prefill(planes, chunk)) {
            if (err) *err = "--prefill-planes takes 1, 2 or 3";
            return nullptr;
        }
    }
    // The lazily made scratch is made now, so the free memory below (and the server's slots and checkpoint slots after it) is what remains.
    size_t scratch_free0 = 0, scratch_total = 0;
    cudaMemGetInfo(&scratch_free0, &scratch_total);
    if (!b->gpu_->reserve_scratch()) {
        if (err) *err = "cannot allocate the prefill and decode scratch";
        cudaGetLastError();
        return nullptr;
    }
    Gpu& g = *b->gpu_;
    {
        size_t fb = 0, tb = 0;
        cudaMemGetInfo(&fb, &tb);
        b->scratch_bytes_ = scratch_free0 > fb ? uint64_t(scratch_free0 - fb) : 0;
    }
    const double mib = 1048576.0;
    // The pool (docs/paged-kv.md): the most exact format whose pool covers n_ctx positions IN ALL (the slots share it, as llama.cpp's
    // -kvu does), plus one page per slot for the page a prefix cut copies; the MTP block beside it (it ranks above the cache's
    // precision), two checkpoint slots per slot, all above the reserve. When even Q4 cannot cover n_ctx the pool is what fits.
    {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        const uint64_t have = uint64_t(free_b) + g.kv_cache_bytes();  // the two-page pool goes before the real one is made
        const uint64_t slack = 160ull << 20;  // the block's graphs, the capture workspaces and what the estimates miss (a run without it lost its second slot)
        const uint64_t budget = have > plan.reserve + slack ? have - plan.reserve - slack : 0;
        const uint64_t snap = g.snapshot_bytes(), slot_b = g.slot_bytes();
        const int ck = std::max(0, plan.checkpoints);
        const int want = std::max(1, plan.slots);
        const int want_pages = (n_ctx + 255) / 256 + want;
        auto pool_b = [&](KvFormat f, int pages) { return g.kv_page_bytes(f) * uint64_t(pages); };
        auto others = [&](int slots) { return uint64_t(slots - 1) * slot_b + uint64_t(slots) * uint64_t(ck) * snap; };
        const uint64_t mtp = use_mtp ? mtp_bytes_estimate(*b->model_, n_ctx) : 0;
        const uint64_t rows_only = use_ngram || use_df ? rows_bytes_estimate(*b->model_, n_ctx) : 0;  // the window scratch without the block
        std::vector<std::string> dfs;
        if (use_df) {
            dfs.push_back(plan.draft_model);
            for (const std::string& a : plan.draft_alts)
                if (Gpu::df_estimate(a, 1, g.prefill_chunk(), df_quant) > 0) dfs.push_back(a);
        }
        size_t dfi = 0;   // the one chosen
        auto df_bi = [&](size_t i, int h) { return use_df ? Gpu::df_estimate(dfs[i], h, g.prefill_chunk(), df_quant) : uint64_t(0); };
        auto df_b = [&](int h) { return df_bi(dfi, h); };  // the drafter with h slots' contexts
        if (use_df && df_b(1) == 0) {
            if (err) *err = "--draft-model " + plan.draft_model + ": no readable model.safetensors (and config.json) there";
            return nullptr;
        }
        const std::vector<std::vector<KvFormat>> classes =
            g.is_q4e() ? std::vector<std::vector<KvFormat>>{{q4e_fmt}}
            : plan.want == "auto" ? std::vector<std::vector<KvFormat>>{{KvFormat::F16, KvFormat::Q8}, {KvFormat::Q4}}
                                  : std::vector<std::vector<KvFormat>>{{fixed_fmt}};
        KvFormat pick = classes.back().back();
        int slots = 0, pages = want_pages;
        bool with_mtp = false, with_df = false;
        // fit order: a drafter > K/V Q8+ (F16 only at full ctx) > drafter quant, MTP beside > context down to ctx_floor
        struct Src {
            bool m, d;
            size_t di;
        };
        std::vector<Src> drafted, plain;
        for (size_t di = 0; di < dfs.size(); ++di) {
            drafted.push_back({true, true, di});
            drafted.push_back({false, true, di});
        }
        plain.push_back({true, false, 0});
        plain.push_back({false, false, 0});
        const int floor_pages = plan.ctx_floor > 0 ? std::min(want_pages, (plan.ctx_floor + 255) / 256 + want) : want_pages;
        for (int h = want; h >= 1 && slots == 0; --h)
            for (const std::vector<Src>* group : {&drafted, &plain})
                for (size_t ci = 0; ci < classes.size() && slots == 0; ++ci)
                    for (size_t oi = 0; oi < group->size() && slots == 0; ++oi) {
                        const Src& o = (*group)[oi];
                        if ((o.m && mtp == 0) || (o.d && !use_df)) continue;
                        const uint64_t spec_b = (o.m ? mtp : rows_only) + (o.d ? df_bi(o.di, h) : 0);
                        for (int pg = want_pages; slots == 0; pg = std::max(floor_pages, pg - 32)) {   // 8K a step
                            for (KvFormat f : classes[ci]) {
                                if (f == KvFormat::F16 && classes[ci].size() > 1 && pg != want_pages) continue;
                                if (pool_b(f, pg) + others(h) + spec_b <= budget) {
                                    pick = f;
                                    slots = h;
                                    with_mtp = o.m;
                                    with_df = o.d;
                                    if (o.d) dfi = o.di;
                                    pages = pg;
                                    break;
                                }
                            }
                            if (pg == floor_pages) break;
                        }
                    }
        if (slots > 0 && pages < want_pages) b->n_ctx_ = (pages - want) * 256;   // the context auto kept beside the drafters
        const uint64_t dfe = with_df ? df_b(std::max(1, slots)) : 0;
        char note[400];
        if (slots == 0) {  // not even the least exact format covers n_ctx: the pool is what fits, with one slot and no MTP block
            const uint64_t rest = budget > others(1) ? budget - others(1) : 0;
            pages = int(std::min<uint64_t>(uint64_t(want_pages), rest / g.kv_page_bytes(pick)));
            if (pages < 2) {
                if (err) *err = "not enough VRAM above the reserve for even two K/V pages (" + std::to_string(budget >> 20) + " MiB)";
                return nullptr;
            }
            slots = 1;
        }
        if ((pick != g.kv_format() || pages != g.pool_pages()) && !g.set_pool(pick, pages)) {
            if (err) *err = "cannot allocate the K/V pool";
            cudaGetLastError();
            return nullptr;
        }
        // The checkpoint slots are the tape's survival (serve.hpp: a slot without snapshot slots loses
        // its tape at every new turn), and the plan budgeted them here — so take the planned ones
        // BEFORE the MTP block and the drafter load. They rank below the slots (GpuOptions): when the
        // block's real size overshoots its estimate, it must lose the VRAM, not the tape. serve's
        // later set_snapshot_slots(asked) only grows the count, best-effort.
        const int planned_snaps = slots * std::max(1, ck);
        const int snaps = g.set_snapshot_slots(planned_snaps, plan.reserve);
        std::string snap_note;
        if (snaps < planned_snaps)
            snap_note = vlog::fmt(" (%d of %d planned checkpoint slots)", snaps, planned_snaps);
        if (with_mtp) {
            size_t f0 = 0, f1 = 0, tb = 0;
            cudaMemGetInfo(&f0, &tb);
            b->load_mtp();
            cudaMemGetInfo(&f1, &tb);
            b->mtp_bytes_ = f0 > f1 ? uint64_t(f0 - f1) : 0;
        }
        if (use_df && !with_df) b->df_note_ = "it did not fit beside the slots (" + vlog::thou(static_cast<long long>(df_b(1) >> 20)) + " MiB with one context)";
        if (with_df) {
            size_t f0 = 0, f1 = 0, tb = 0;
            cudaMemGetInfo(&f0, &tb);
            std::string e2;
            if (g.df_load(dfs[dfi], &e2, df_quant, plan.draft_vocab > 0 ? plan.draft_vocab : 65536)) {
                b->spec_.dflash = true;
                b->spec_.dspark = plan.spec == "dspark";
                b->spec_.fixed_k = plan.draft_k;
                b->draft_dir_ = dfs[dfi];
            } else {
                b->df_note_ = e2;
                cudaGetLastError();
                if (plan.spec == "dflash" || plan.spec == "dspark") {
                    if (err) *err = e2;
                    return nullptr;
                }
            }
            cudaMemGetInfo(&f1, &tb);
            b->df_bytes_ = f0 > f1 ? uint64_t(f0 - f1) : 0;
        }
        {  // what each format needed, for the -v start-up block
            auto mb = [&](uint64_t v) { return vlog::thou(static_cast<long long>(v >> 20)); };
            std::string w = "kv: budget " + mb(budget) + " MiB (" + mb(have) + " free with the pool made at open, less the " + mb(plan.reserve) + " MiB reserve and a 160 MiB slack); " +
                            "the pool for " + std::to_string(want_pages) + " pages (" + std::to_string(n_ctx) + " positions + a page per slot) plus " + std::to_string(std::max(1, slots)) +
                            " slot(s) with " + std::to_string(ck) + " checkpoint slots each" + (with_mtp ? " and the MTP block" : "") + " needs:";
            for (KvFormat f : {KvFormat::F16, KvFormat::Q8, KvFormat::Q4}) {
                const uint64_t need = pool_b(f, want_pages) + others(slots) + (with_mtp ? mtp : rows_only);
                w += std::string(" ") + kv_format_name(f) + " " + mb(need) + " MiB" + (need <= budget ? " (fits)" : " (no)") + (f == pick ? " <- chosen" : "") + ";";
            }
            w += plan.want == "auto" ? " auto takes the most exact format that fits" : std::string(" --kv ") + plan.want + " fixes the format";
            if (!with_mtp && mtp > 0) w += "; the MTP block (" + mb(mtp) + " MiB) did not fit beside the slots";
            if (use_df) w += with_df ? "; the DFlash2 drafter takes " + mb(dfe) + " MiB with " + std::to_string(std::max(1, slots)) + " context(s)" : "; the DFlash2 drafter (" + mb(df_b(1)) + " MiB) did not fit beside the slots";
            b->kv_why_ = w;
        }
        if (pages >= want_pages)
            std::snprintf(note, sizeof note, "K/V %s (%s: a pool of %d pages = %d positions for all slots, %d slot%s planned with %d checkpoint slots each, in %.0f MiB above the reserve and a 160 MiB slack; f16 needs %.0f MiB)",
                          kv_format_name(pick), plan.want == "auto" ? "auto" : "fixed", pages, pages * 256, slots, slots == 1 ? "" : "s", ck, double(budget) / mib,
                          double(pool_b(KvFormat::F16, want_pages) + others(slots)) / mib);
        else
            std::snprintf(note, sizeof note, "K/V %s (%s: ONLY %d positions fit: %d pages, %.0f MiB above the reserve, %d positions were asked; a request that needs more gets a 400; f16 would need %.0f MiB)",
                          kv_format_name(pick), plan.want == "auto" ? "auto" : "fixed", pages * 256, pages, double(budget) / mib, n_ctx,
                          double(pool_b(KvFormat::F16, want_pages) + others(1)) / mib);
        b->kv_note_ = std::string(note) + snap_note;
    }
    b->spec_.ngram = use_ngram;
    // qwen4exp drafts with what its architecture brings and nothing else: its own MTP layer (gpu_q4e_mtp.cu) and the
    // tape's ngram, chosen per round by the policy, verified in its own window (q4e_step_rows). No DFlash drafter.
    if (b->gpu_->is_q4e()) {
        b->spec_.dflash = false;
        b->spec_.mtp = b->spec_.mtp && b->gpu_->mtp_ready();   // its own block (gpu_q4e_mtp.cu), or none
    }
    b->spec_.on = b->spec_.mtp || b->spec_.ngram || b->spec_.dflash;
    for (int j = 1; j < 8; ++j) {
        b->spec_.acc_df[j] = std::max(0.25, 0.8 - 0.07 * (j - 1));
        for (int c = 0; c < 4; ++c) b->spec_.acc_dc[j][c] = std::min(0.95, std::max(0.1, b->spec_.acc_df[j] + 0.15 * (c - 1.5)));
    }
    b->ng_n_ = plan.ngram_n > 0 ? plan.ngram_n : kDefaultNgramN;
    b->spec_.ng_max = plan.ngram_max > 0 ? plan.ngram_max : 7;
    for (auto& row : b->spec_.acc_ng) {  // what a match of that class has been worth: the first guesses, then the EMAs
        const double p0 = &row == &b->spec_.acc_ng[0] ? 0.5 : &row == &b->spec_.acc_ng[1] ? 0.55 : &row == &b->spec_.acc_ng[2] ? 0.8 : 0.93;
        for (double& a : row) a = p0;
    }
    if (b->spec_.on) {  // timed here, so no request pays for it; the windows are dropped and the device state cleared again
        try {
            b->spec_calibrate();
            g.reset();
        } catch (const std::exception& ex) {
            std::fprintf(stderr, "speculation calibration failed: %s (%s) -- speculation off\n", ex.what(),
                         cudaGetErrorString(cudaGetLastError()));
            b->spec_.on = b->spec_.mtp = b->spec_.ngram = b->spec_.dflash = false;
        }
    }
    if (b->spec_.ngram) b->ng_.configure(b->ng_n_, n_ctx, 1);
    std::string costs;
    if (b->spec_.on) {
        char cb[120];
        std::snprintf(cb, sizeof cb, " (row %.1f ms, windows of 2/3/4 rows %.1f/%.1f/%.1f, block step %.1f, drafter %.1f)", b->spec_.t_plain, b->spec_.t_win[2],
                      b->spec_.t_win[3], b->spec_.t_win[4], b->spec_.t_chain, b->spec_.t_df);
        costs = cb;
    }
    std::string what = std::string(b->spec_.mtp ? "MTP" : "") + (b->spec_.mtp && b->spec_.ngram ? " + " : "") +
                       (b->spec_.ngram ? "ngram (n " + std::to_string(b->ng_n_) + ", up to " + std::to_string(b->spec_.ng_max) + ")" : "");
    if (b->spec_.dflash) what += std::string(what.empty() ? "" : " + ") + (b->spec_.dspark ? "DFlash2/DSpark" : "DFlash2");
    b->kv_note_ = b->kv_note_ + ", prefill planes " + std::to_string(g.prefill_planes()) + " chunk " + std::to_string(g.prefill_chunk()) +
                  (b->spec_.on ? ", speculation " + what + costs : ", no speculation");
    if (cudaMallocHost(reinterpret_cast<void**>(&b->logits_), size_t(b->gpu_->n_vocab()) * sizeof(float)) != cudaSuccess) {
        if (err) *err = std::string("cannot allocate the pinned logits buffer: ") + cudaGetErrorString(cudaGetLastError()) +
                         " for " + std::to_string(size_t(b->gpu_->n_vocab()) * sizeof(float)) + " bytes";
        cudaGetLastError();
        return nullptr;
    }
    return b;
}

GpuBackend::~GpuBackend() {
    gpu_.reset();  // the device arena goes before the pinned buffer and the CUDA context's host side
    if (logits_) cudaFreeHost(logits_);
}

int GpuBackend::n_vocab() const { return gpu_->n_vocab(); }
int GpuBackend::n_ctx() const { return n_ctx_; }
void GpuBackend::reset() {
    spec_.live = false;
    if (spec_.ngram) ng_.cur().clear();
    gpu_->reset();
}

void GpuBackend::step(int32_t token, int pos, bool head) {
    spec_.live = false;
    if (spec_.ngram) ng_.cur().put(pos, &token, 1, true);
    if (!gpu_->step(token, pos, head)) {
        // A failed launch leaves an error pending on the stream; name it before the server resets the session.
        char msg[160];
        std::snprintf(msg, sizeof msg, "GPU step failed at position %d (%s)", pos, cudaGetErrorString(cudaGetLastError()));
        throw std::runtime_error(msg);
    }
}

const float* GpuBackend::forward(int32_t token, int pos) { return forward(token, pos, true); }

const float* GpuBackend::forward(int32_t token, int pos, bool want_logits) {
    step(token, pos, want_logits);
    if (!want_logits) return nullptr;  // queued, not waited for: the next call or a sync picks it up
    if (!gpu_->read_logits(logits_)) throw std::runtime_error("GPU logits copy failed");
    return logits_;
}

int32_t GpuBackend::forward_greedy(int32_t token, int pos) {
    step(token, pos, true);
    const int32_t t = gpu_->argmax();
    if (t < 0 || t >= gpu_->n_vocab()) throw std::runtime_error("GPU argmax failed");
    return t;
}

void GpuBackend::prefill(const int32_t* tokens, int n, int pos0) {
    spec_.live = false;
    if (gpu_->is_q4e()) spec_.base = 0;   // q4e: the block's cells are the positions (the prompt pass fills them)
    if (spec_.ngram) ng_.cur().put(pos0, tokens, n, false);
    if (!gpu_->prefill(tokens, n, pos0, false)) {
        char msg[160];
        std::snprintf(msg, sizeof msg, "GPU prefill of %d tokens at position %d failed (%s)", n, pos0, cudaGetErrorString(cudaGetLastError()));
        throw std::runtime_error(msg);
    }
}

// The device bytes the MTP block takes when loaded (Gpu::mtp_load and the T-row pass it needs): its tensors (the router and the
// shared gate go to F32), its own K/V, the T-row activations and the split flash-decode partials of 8 rows at n_ctx.
uint64_t GpuBackend::mtp_bytes_estimate(const model::Model& m, int n_ctx) {
    const model::Geometry& geo = m.geometry();
    if (geo.arch == "qwen4exp") return q4e_mtp_path(m).empty() ? 0 : 1;   // Gpu::open reserves it beside the expert slab
    if (!geo.has_mtp || geo.mtp_layer < 0) return 0;
    uint64_t w = 0;
    if (geo.n_expert == 0) {  // qwen35: a dense block
        for (const char* role : {"nextn.eh_proj.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight", "ffn_gate.weight",
                                 "ffn_up.weight", "ffn_down.weight"})
            if (const gguf::Tensor* t = m.tensor(geo.mtp_layer, role)) w += t->nbytes + 256;
        for (const char* role : {"nextn.enorm.weight", "nextn.hnorm.weight", "nextn.shared_head_norm.weight", "attn_norm.weight",
                                 "post_attention_norm.weight", "attn_q_norm.weight", "attn_k_norm.weight"})
            if (const gguf::Tensor* t = m.tensor(geo.mtp_layer, role)) w += uint64_t(t->n_elements()) * 4 + 256;
        const uint64_t chunks = (uint64_t(n_ctx) + 255) / 256;
        const uint64_t partials = 8ull * uint64_t(geo.n_head) * chunks * 258 * 4;
        // + the block's K/V, the T-row activations and logits (8 rows of the vocabulary), the per-layer scan inputs of the rows pass
        const uint64_t rows = 8ull * uint64_t(geo.n_vocab) * 4 * 2 + 8ull * uint64_t(geo.ssm_inner * 2 + 2 * geo.ssm_groups * geo.ssm_state) * 4 * 2 * uint64_t(geo.n_layer_main);
        return w + partials + rows + (96ull << 20);
    }
    for (const char* role : {"nextn.eh_proj.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
                             "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight", "ffn_gate_shexp.weight", "ffn_up_shexp.weight",
                             "ffn_down_shexp.weight"})
        if (const gguf::Tensor* t = m.tensor(geo.mtp_layer, role)) w += t->nbytes + 256;
    for (const char* role : {"nextn.enorm.weight", "nextn.hnorm.weight", "nextn.shared_head_norm.weight", "attn_norm.weight",
                             "post_attention_norm.weight", "attn_q_norm.weight", "attn_k_norm.weight", "ffn_gate_inp.weight",
                             "ffn_gate_inp_shexp.weight"})  // uploaded as F32
        if (const gguf::Tensor* t = m.tensor(geo.mtp_layer, role)) w += uint64_t(t->n_elements()) * 4 + 256;
    const uint64_t chunks = (uint64_t(n_ctx) + 255) / 256;
    const uint64_t partials = 8ull * uint64_t(geo.n_head) * chunks * 258 * 4;
    return w + partials + (96ull << 20);  // + the block's K/V (16 MiB) and the rows' activations and logits
}

// The window scratch alone (ngram without the block): the T-row activations and logits of 8 rows, the split flash-decode partials.
uint64_t GpuBackend::rows_bytes_estimate(const model::Model& m, int n_ctx) {
    const model::Geometry& geo = m.geometry();
    const uint64_t chunks = (uint64_t(n_ctx) + 255) / 256;
    uint64_t r = 8ull * uint64_t(geo.n_head) * chunks * 258 * 4 + (80ull << 20);
    if (geo.n_expert == 0)
        r += 8ull * uint64_t(geo.n_vocab) * 4 * 2 + 8ull * uint64_t(geo.ssm_inner * 2 + 2 * geo.ssm_groups * geo.ssm_state) * 4 * 2 * uint64_t(geo.n_layer_main);
    return r;
}

bool GpuBackend::load_mtp() {
    if (!gpu_->mtp_load(*model_)) {
        cudaGetLastError();
        return false;
    }
    gpu_->mtp_set_draft_vocab(65536);  // drafts score the first 65,536 ids of the head (bench/w1-decode-tune section 14)
    for (double& a : spec_.acc) a = 0.65;
    spec_.mtp = true;
    return true;
}

// Every variant the decode loop runs, once eagerly and once captured, then timed; at position 0 (the load pays). The windows are
// committed or dropped on the spot; the caller clears the device state afterwards. The block's passes only when it is loaded.
void GpuBackend::spec_calibrate() {
    Gpu& g = *gpu_;
    std::vector<int32_t> w(8, 220);
    auto bad = [] { throw std::runtime_error("GPU speculation calibration failed"); };
    auto ms = [](std::chrono::steady_clock::time_point t0) { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); };
    auto window = [&](int T) {
        if (!(g.is_q4e() ? g.q4e_step_rows(w.data(), T, 0, false) : g.step_rows(w.data(), T, 0, false)) ||
            !(g.is_q4e() ? g.q4e_commit(T) : g.commit(T)))
            bad();
    };
    for (int T = 1; T <= 8; ++T) {
        window(T);
        window(T);
        g.sync();
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < 3; ++r) window(T);
        g.sync();
        spec_.t_win[T] = ms(t0) / 3;
    }
    for (int r = 0; r < 5; ++r) {  // the plain row with write-back: two warm-up passes, three timed
        if (r == 2) g.sync();
        const auto t0 = std::chrono::steady_clock::now();
        if (!(g.is_q4e() ? g.q4e_step_rows(w.data(), 1, 0, true) : g.step_rows(w.data(), 1, 0, true))) bad();
        if (r >= 2) {
            g.sync();
            spec_.t_plain += ms(t0) / 3;
        }
    }
    if (spec_.mtp) {
        window(8);  // the block's refresh over R = 1..8 rows reads the hidden states of a window of at least R rows
        for (int R = 1; R <= 8; ++R)
            for (int r = 0; r < 2; ++r)
                if (!g.mtp_rows(w.data(), R, 8, false)) bad();
        int32_t tk = 220, o[8];
        if (!g.mtp_rows(&tk, 1, 8, false) || !g.mtp_argmax(o, 1)) bad();
        for (int r = 0; r < 2; ++r)  // the chained draft: eager, then captured
            if (!g.mtp_rows(&tk, 1, 9, true) || !g.mtp_argmax(o, 1)) bad();
        g.sync();
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < 4; ++r)
            if (!g.mtp_rows(&tk, 1, 9, true) || !g.mtp_argmax(o, 1)) bad();
        spec_.t_chain = ms(t0) / 4;
    }
    if (spec_.dflash) {  // the drafter's block over a context of the 8 positions the last window left
        window(8);
        int32_t dt[8];
        float dc[8];
        for (int r = 0; r < 2; ++r)
            if (!g.df_draft_start(220, 8) || !g.df_draft_wait(dt, dc)) bad();
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < 4; ++r)
            if (!g.df_draft_start(220, 8) || !g.df_draft_wait(dt, dc)) bad();
        spec_.t_df = ms(t0) / 4;
    }
    spec_.calibrated = true;
}

std::string GpuBackend::acc_note(int from, int to) const {
    std::string s;
    for (int j = from; j <= to; ++j) s += vlog::fmt(j > from ? "/%.2f" : "%.2f", spec_.acc[j]);
    return s;
}

// One decode step that may return several tokens. Four ways to make it, chosen per cycle by expected tokens per millisecond:
//   plain   one row (with write-back, so the MTP block can resume from its hidden state);
//   mtp     the block drafts k tokens (a refresh over the last window's rows, then k - 1 chained passes), one window checks them;
//   ngram   the slot's tape proposes k tokens (a copy of what followed the same words earlier), the same window checks them;
//   dflash  the DFlash2 drafter fills a block of 7 positions in one pass over its context (every pass feeds it, Gpu::df_*), the same window checks them.
// Expected tokens come from an EMA of each draft position's acceptance (per source; for ngram per match class); costs from the
// calibration scaled by what the calls measure. The check is the same for both drafts, so the tokens are exact: greedy equals
// the plain loop's, sampling goes through pick.row (Sampler::sample_draft), whatever the draft was.
int GpuBackend::decode_spec(int32_t fed, int pos, int max_out, const SpecPick& pick, int32_t* out) {
    if (!spec_.on) return Backend::decode_spec(fed, pos, max_out, pick, out);
    Gpu& g = *gpu_;
    // The window the loop below holds is whichever one this model has: rung 1's step_rows, or qwen4exp's
    // q4e_step_rows with its own workspace, its own logits and its own commit. Nothing else in this function
    // changes -- the draft sources, the cost model, the acceptance and the sampler are the same, and the tokens
    // are exact for the same reason: the check is a real forward pass either way.
    const bool q4e = g.is_q4e();
    auto win = [&](const int32_t* t, int T, int p0, bool wb) {
        return q4e ? g.q4e_step_rows(t, T, p0, wb) : g.step_rows(t, T, p0, wb);
    };
    auto cmt = [&](int n) { return q4e ? g.q4e_commit(n) : g.commit(n); };
    auto amax = [&](int32_t* o, int T) { return q4e ? g.q4e_argmax_rows(o, T) : g.argmax_rows(o, T); };
    auto rlog = [&](float* o, int T) { return q4e ? g.q4e_read_logits_rows(o, T) : g.read_logits_rows(o, T); };
    const auto t_call = std::chrono::steady_clock::now();
    auto ms_since = [](std::chrono::steady_clock::time_point t) { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count(); };
    // the calibration is at position 0: the cost of a row and of a window grow differently with the context, so each is scaled by what the calls measure
    // q4e: a window's cost from the windows that ran (SpecState::qw). A size not seen yet takes the cheaper of the line
    // through the plain row and the sizes seen, and the calibration: optimistic, so it gets tried and measured -- rows of
    // copied text share experts, and a straight line from 2..5 rows put an 8-row window at ~123 ms where it runs in ~98
    // (an edit copying a file: 78 tok/s through 8-row ngram windows, 69 when the line kept them out). `cal` is what the
    // other models use.
    auto row_ms = [&] { return spec_.s_row * (spec_.t_plain + 0.3); };
    auto win_ms = [&](int T, double cal) -> double {
        if (!q4e) return cal;
        if (spec_.qn[T] >= 2) return spec_.qw[T];
        double sl = 0;
        int ns = 0;
        for (int u = 2; u <= 8; ++u)
            if (spec_.qn[u] >= 2) {
                sl += (spec_.qw[u] - row_ms()) / double(u - 1);
                ++ns;
            }
        return ns ? std::min(cal, row_ms() + std::max(0.0, sl / ns) * double(T - 1)) : cal;
    };
    auto chain_ms = [&] { return spec_.q_chain > 0 ? spec_.q_chain : spec_.s_win * spec_.t_chain; };
    auto learn = [&](double& scale, double calibrated_ms, double w) {
        scale = std::min(3.0, std::max(0.5, (1 - w) * scale + w * ms_since(t_call) / calibrated_ms));
    };
    const int nv = g.n_vocab();
    // The acceptance of draft j given the drafter's own confidence in it. The bucket table is the discrete form and is what
    // `--spec dflash` uses; `--spec dspark` fits a logistic instead, so the four hand-placed edges stop being a guess and the
    // estimate becomes a function of how confident THIS drafter actually is on THIS text. a = 1, b = 0 is exactly p = conf.
    auto acc_of = [&](int j, float conf) -> double {
        if (!spec_.dspark) {
            const int c = conf < 0.5f ? 0 : conf < 0.8f ? 1 : conf < 0.95f ? 2 : 3;
            return spec_.acc_dc[j][c];
        }
        return 1.0 / (1.0 + std::exp(-(spec_.dsp_a * double(conf) + spec_.dsp_b)));
    };
    int32_t am[9] = {};
    auto fail = [&](const char* what) {
        spec_.live = false;
        char msg[160];
        std::snprintf(msg, sizeof msg, "GPU speculation: %s at position %d (%s)", what, pos, cudaGetErrorString(cudaGetLastError()));
        throw std::runtime_error(msg);
    };
    // the checked rows' tokens: on the device when greedy, else the rows' logits through the server's sampler
    auto fetch = [&](int T) {
        if (pick.greedy) {
            if (!amax(am, T)) fail("argmax");
        } else {
            rows_logits_.resize(size_t(T) * size_t(nv));
            if (!rlog(rows_logits_.data(), T)) fail("logits copy");
        }
    };
    auto choose = [&](int row, int32_t draft, bool* kept) -> int32_t {
        if (pick.greedy) {
            *kept = am[row] == draft;
            return am[row];
        }
        return pick.row(rows_logits_.data() + size_t(row) * size_t(nv), draft, kept);
    };
    ngram::Index* ix = spec_.ngram ? &ng_.cur() : nullptr;
    if (ix) ix->put(pos, &fed, 1, true);
    const bool cont = spec_.mtp && spec_.live && pos == spec_.next_pos && fed == spec_.c;
    const int cap = spec_.mtp ? g.mtp_capacity() : 0;
    // The longest window this model can actually run at this position. rung 1's step_rows takes 1..8 rows wherever
    // there is room; qwen4exp's q4e_step_rows refuses past the sparse selection's block budget, where its rows
    // would no longer be the plain causal ones. A window the engine would refuse has to become single rows here,
    // not a failed window call.
    const int wmax = q4e ? g.q4e_window_rows(pos) : 8;
    const bool room = max_out >= 2 && pos + 9 <= g.n_ctx() && wmax >= 2;
    const int kroom = std::min({max_out - 1, 7, g.n_ctx() - pos - 1, wmax - 1});
    enum Src { Plain, Mtp, Ngram, Dflash } src = Plain;
    int k = 0;
    double best = 1.0 / (spec_.s_row * (spec_.t_plain + 0.3));  // tokens per ms of a plain row
    int kmtp = 0;
    if (cont && room) {
        kmtp = std::min(spec_.kmax, kroom);
        for (int kk = 1; kk <= kmtp; ++kk) {
            double e = 1, pr = 1;
            for (int j = 1; j <= kk; ++j) {
                pr *= spec_.acc[j];
                e += pr;
            }
            const double cost = q4e ? chain_ms() * kk + win_ms(kk + 1, spec_.s_win * (spec_.t_win[kk + 1] + 0.3))
                                    : spec_.s_win * (spec_.t_chain * kk + spec_.t_win[kk + 1] + 0.3);
            if (e / cost > best) {
                best = e / cost;
                k = kk;
                src = Mtp;
            }
        }
    }
    // LIMBO_MTP_K=k: every continuing step drafts k with the block (the bench's way to see a window's real cost)
    if (cont && room && spec_.mtp) {
        static const int forced = std::getenv("LIMBO_MTP_K") ? std::atoi(std::getenv("LIMBO_MTP_K")) : 0;
        if (forced > 0) {
            k = std::min(forced, kroom);
            src = Mtp;
            best = 1e9;
        }
    }
    ngram::Draft dr;
    if (ix && room && kroom >= 1) {
        dr = ix->propose(std::min(spec_.ng_max, kroom));
        for (int kk = 1; kk <= dr.n; ++kk) {  // a window of kk + 1 rows, no block passes
            double e = 1, pr = 1;
            for (int j = 1; j <= kk; ++j) {
                pr *= spec_.acc_ng[dr.cls][j];
                e += pr;
            }
            const double cost = win_ms(kk + 1, spec_.s_ngw * (spec_.t_win[kk + 1] + 0.3));
            if (e / cost > best) {
                best = e / cost;
                k = kk;
                src = Ngram;
            }
        }
    }
    // the DFlash2 block: one pass for up to 7 drafts, whatever k; its context must end right before the token fed now
    const bool dfok = spec_.dflash && room && kroom >= 1 && g.df_len() == pos;
    if (dfok) {
        const int fk = spec_.fixed_k;
        for (int kk = fk ? std::min(fk, kroom) : 1; kk <= (fk ? std::min(fk, kroom) : std::min(7, kroom)); ++kk) {
            double e = 1, pr = 1;
            for (int j = 1; j <= kk; ++j) {
                pr *= spec_.acc_df[j];
                e += pr;
            }
            const double cost = spec_.s_dfw * (spec_.t_df + spec_.t_win[kk + 1] + 0.3);
            if (e / cost > best) {
                best = e / cost;
                k = kk;
                src = Dflash;
            }
        }
    }
    if (src == Plain) {  // an EMA only moves when its source runs: one cycle in a while tests each
        if (kmtp >= 1 && spec_.since_probe >= 16) {
            src = Mtp;
            k = 1;
        } else if (room && dr.n >= 1 && spec_.since_probe_ng >= 12) {
            // A window is worth taking only if it beats the same tokens as plain rows, and the ngram branch used
            // to take one unconditionally -- the MTP branch above weighs its expected tokens against its cost,
            // this one did not. That is free when a window is cheaper than a row, and it is not free when it is
            // not: on a box whose expert cache does not cover the working set, a window costs MORE per row than a
            // plain row, because the dense read it saves is not what is limiting. Measured there: 47.7 ms a plain
            // row against 246 ms for three rows through the window. The window still bought one token a pass, so
            // every pass paid 5x for it.
            //
            // e is the tokens the window is expected to return, from the acceptance this match class has actually
            // shown: 1 + p1 + p1*p2 + ... . e tokens need e plain rows to match.
            int kk = std::min({2, dr.n, kroom});
            while (kk >= 1) {
                double e = 1, pr = 1;
                for (int j = 1; j <= kk; ++j) {
                    pr *= spec_.acc_ng[dr.cls][j];
                    e += pr;
                }
                if (q4e ? e * row_ms() > win_ms(kk + 1, spec_.t_win[kk + 1]) : e * spec_.t_plain > spec_.t_win[kk + 1]) break;
                --kk;
            }
            if (kk >= 1) {
                src = Ngram;
                k = kk;
            }
        } else if (dfok && spec_.since_probe_df >= 12) {
            src = Dflash;
            k = std::min(3, kroom);
        }
    }
    if (dr.n >= 1 && src != Ngram) ++spec_.since_probe_ng;
    if (dfok && src != Dflash) ++spec_.since_probe_df;
    // q4e, the donor's rule (docs/donors/strata.md, Drafting): a continuing run always drafts; the block's drafts join the
    // window while each has p >= min_p under it, up to kmax; no draft passing it makes the pass a plain row.
    int32_t d[9];  // d[1..k]
    double t_block = 0;
    bool drafted = false;
    if (spec_.min_p < 0) spec_.min_p = std::getenv("LIMBO_SPEC_MIN_P") ? std::atof(std::getenv("LIMBO_SPEC_MIN_P")) : 0.5;
    if (q4e && spec_.mtp && spec_.min_p > 0 && cont && room && (src == Plain || src == Mtp) && !std::getenv("LIMBO_MTP_K")) {
        if (pos - spec_.base + 16 > cap) spec_.base = pos - spec_.R;
        const int R = spec_.R;
        int32_t nxt[9], rowarg[9];
        float rowp[9];
        for (int j = 0; j + 1 < R; ++j) nxt[j] = spec_.prev[size_t(j + 1)];
        nxt[R - 1] = fed;
        if (!g.mtp_rows(nxt, R, pos - R - spec_.base, false) || !g.mtp_argmax_p(rowarg, rowp, R)) fail("draft block");
        d[1] = rowarg[R - 1];
        float p = rowp[R - 1];
        const int kcap = std::min(spec_.kmax, kroom);
        int kk = 0;
        while (kk < kcap && p >= float(spec_.min_p)) {
            ++kk;
            if (kk < kcap && (!g.mtp_rows(&d[kk], 1, pos + kk - 1 - spec_.base, true) || !g.mtp_argmax_p(&d[kk + 1], &p, 1)))
                fail("chained draft");
        }
        k = kk;
        src = k > 0 ? Mtp : Plain;
        drafted = true;
        spec_.since_probe = 0;
        t_block = verbose_ >= 2 || q4e ? ms_since(t_call) : 0;
    }
    if (src == Plain) {
        // q4e: the block's cells are the positions; the rows of the last pass it has not read go into its K/V first
        if (q4e && spec_.mtp && !drafted) {
            if (cont) {
                int32_t nxt[9];
                for (int j = 0; j + 1 < spec_.R; ++j) nxt[j] = spec_.prev[size_t(j + 1)];
                nxt[spec_.R - 1] = fed;
                if (pos - spec_.R - spec_.base >= 0 && !g.mtp_front(nxt, spec_.R, 0, pos - spec_.R - spec_.base)) fail("draft block front");
            } else if (pos >= 1 && g.q4e_rows_end() == pos && pos - 1 - spec_.base >= 0) {
                if (!g.mtp_front(&fed, 1, g.q4e_rows_last() - 1, pos - 1 - spec_.base)) fail("draft block front");
            }
        }
        // a plain row with write-back: its hidden state is what the block reads if drafting resumes at the next position
        if (!win(&fed, 1, pos, true)) fail("window");
        fetch(1);
        bool kept = false;
        const int32_t c = choose(0, -1, &kept);
        ++spec_.runs;
        if (cont) ++spec_.plain;
        spec_.since_probe = cont ? spec_.since_probe + 1 : 0;
        if (!q4e) spec_.base = pos;  // rung 1: the block has not read this position; its positions start here again
        spec_.prev.assign(1, fed);
        spec_.R = 1;
        spec_.c = c;
        spec_.next_pos = pos + 1;
        spec_.live = true;
        ++spec_.tokens;
        if (verbose_ >= 2)
            vlog::line("spec", vlog::fmt("plain  row %s  -> 1 tok%s%s", vlog::ms(ms_since(t_call)).c_str(), cont ? "" : "  (not continuing a run)",
                                         dr.n >= 1 ? vlog::fmt("  (ngram draft of %d declined: class %d, acc %.2f)", dr.n, dr.cls, spec_.acc_ng[dr.cls][1]).c_str() : ""));
        learn(spec_.s_row, spec_.t_plain + 0.3, 0.3);
        out[0] = c;
        return 1;
    }
    float dconf[8] = {};
    if (src == Dflash) {
        spec_.since_probe_df = 0;
        int32_t dt[8];
        if (!g.df_draft_start(fed, pos) || !g.df_draft_wait(dt, dconf)) fail("dflash draft");
        // the drafts are in: the length of the window is chosen again from each draft's own confidence (the block's cost is spent)
        if (!spec_.fixed_k) {
            double bestr = 0, e = 1, pr = 1;
            for (int kk = 1; kk <= std::min(7, kroom); ++kk) {
                pr *= acc_of(kk, dconf[kk - 1]);
                e += pr;
                const double r = e / (spec_.t_win[kk + 1] + 0.3);
                if (r > bestr) {
                    bestr = r;
                    k = kk;
                }
            }
        }
        for (int j = 1; j <= k; ++j) d[j] = dt[j - 1];
        t_block = verbose_ >= 2 || q4e ? ms_since(t_call) : 0;
    } else if (src == Mtp && drafted) {
    } else if (src == Mtp) {
        spec_.since_probe = 0;
        // the block's positions start again when they run out (its older entries are then never read)
        if (pos - spec_.base + 16 > cap) spec_.base = pos - spec_.R;
        const int R = spec_.R;
        // refresh: the accepted rows of the last window with their next tokens (the last one's is c, the token fed now)
        int32_t nxt[9];
        for (int j = 0; j + 1 < R; ++j) nxt[j] = spec_.prev[size_t(j + 1)];
        nxt[R - 1] = fed;
        int32_t rowarg[9];
        if (!g.mtp_rows(nxt, R, pos - R - spec_.base, false) || !g.mtp_argmax(rowarg, R)) fail("draft block");
        d[1] = rowarg[R - 1];
        for (int j = 2; j <= k; ++j)
            if (!g.mtp_rows(&d[j - 1], 1, pos + j - 2 - spec_.base, true) || !g.mtp_argmax(&d[j], 1)) fail("chained draft");
        t_block = verbose_ >= 2 || q4e ? ms_since(t_call) : 0;
    } else {
        spec_.since_probe_ng = 0;
        for (int j = 1; j <= k; ++j) d[j] = dr.tok[j - 1];
    }
    int32_t w[9];
    w[0] = fed;
    for (int j = 1; j <= k; ++j) w[j] = d[j];
    if (!win(w, k + 1, pos, false)) fail("check window");
    fetch(k + 1);
    int m = 0;
    int32_t next = -1;
    for (int j = 0; j < k; ++j) {  // row j checks draft j + 1
        bool kept = false;
        const int32_t t = choose(j, d[j + 1], &kept);
        if (!kept || (pick.stop && pick.stop(t))) {
            next = t;
            break;
        }
        ++m;
    }
    if (next < 0) {
        bool kept = false;
        next = choose(k, -1, &kept);
    }
    if (!cmt(m + 1)) fail("commit");
    for (int j = 1; j <= m; ++j) out[j - 1] = d[j];
    out[m] = next;
    if (src == Mtp) {
        for (int j = 1; j <= std::min(k, m + 1); ++j) spec_.acc[j] = 0.9 * spec_.acc[j] + 0.1 * (j <= m ? 1.0 : 0.0);
        ++spec_.cycles;
        spec_.drafted += k;
        spec_.accepted += m;
        learn(spec_.s_win, spec_.t_chain * k + spec_.t_win[k + 1] + 0.3, 0.15);
        if (q4e && k > 0) spec_.q_chain = spec_.q_chain > 0 ? 0.85 * spec_.q_chain + 0.15 * t_block / k : t_block / k;
    } else if (src == Dflash) {
        for (int j = 1; j <= std::min(k, m + 1); ++j) {
            const double hit = j <= m ? 1.0 : 0.0;
            const int c = dconf[j - 1] < 0.5f ? 0 : dconf[j - 1] < 0.8f ? 1 : dconf[j - 1] < 0.95f ? 2 : 3;
            spec_.acc_df[j] = 0.92 * spec_.acc_df[j] + 0.08 * hit;
            spec_.acc_dc[j][c] = 0.9 * spec_.acc_dc[j][c] + 0.1 * hit;
            // DSpark: one logistic-regression step on the same observation. The gradient of the log loss with respect to the
            // logit is (p - hit), and the logit is a * conf + b, so the two partials are that times conf and that alone.
            // The last j of this loop is the draft that was dropped, which is the one negative example a cycle carries.
            if (spec_.dspark) {
                const double p = acc_of(j, dconf[j - 1]);
                const double g_ = p - hit;
                spec_.dsp_a = std::min(20.0, std::max(-20.0, spec_.dsp_a - 0.05 * g_ * double(dconf[j - 1])));
                spec_.dsp_b = std::min(20.0, std::max(-20.0, spec_.dsp_b - 0.05 * g_));
                ++spec_.dsp_n;
            }
        }
        ++spec_.df_cycles;
        spec_.df_drafted += k;
        spec_.df_accepted += m;
        if (!q4e) spec_.base = pos;
        learn(spec_.s_dfw, spec_.t_df + spec_.t_win[k + 1] + 0.3, 0.15);
    } else {
        for (int j = 1; j <= std::min(k, m + 1); ++j) spec_.acc_ng[dr.cls][j] = 0.85 * spec_.acc_ng[dr.cls][j] + 0.15 * (j <= m ? 1.0 : 0.0);
        ++spec_.ng_cycles;
        spec_.ng_drafted += k;
        spec_.ng_accepted += m;
        if (!q4e) spec_.base = pos;  // rung 1: the block did not read the window before this one
        learn(spec_.s_ngw, spec_.t_win[k + 1] + 0.3, 0.15);
    }
    if (q4e && src != Plain) {   // the window and its commit, the block's drafts not
        const double w = ms_since(t_call) - t_block;
        spec_.qw[k + 1] = spec_.qn[k + 1] ? 0.85 * spec_.qw[k + 1] + 0.15 * w : w;
        ++spec_.qn[k + 1];
    }
    spec_.tokens += m + 1;
    if (ix && m > 0) ix->put(pos + 1, d + 1, m, true);  // the sampled token goes in as the next call's `fed`
    spec_.prev.assign(w, w + k + 1);
    spec_.R = m + 1;
    spec_.c = next;
    spec_.next_pos = pos + m + 1;
    spec_.live = true;
    if (verbose_ >= 2) {
        if (src == Mtp)
            vlog::line("spec", vlog::fmt("mtp    k=%d kept %d/%d  block %s  window %s (%d rows)  -> %d tok  acc %s", k, m, k, vlog::ms(t_block).c_str(),
                                         vlog::ms(ms_since(t_call) - t_block).c_str(), k + 1, m + 1, acc_note(1, 4).c_str()));
        else if (src == Dflash)
            vlog::line("spec", vlog::fmt("dflash k=%d kept %d/%d  block %s  window %s (%d rows)  -> %d tok  conf %.2f/%.2f/%.2f/%.2f  acc %.2f/%.2f/%.2f/%.2f", k, m, k, vlog::ms(t_block).c_str(),
                                         vlog::ms(ms_since(t_call) - t_block).c_str(), k + 1, m + 1, double(dconf[0]), double(dconf[1]), double(dconf[2]), double(dconf[3]),
                                         spec_.acc_df[1], spec_.acc_df[2], spec_.acc_df[3], spec_.acc_df[4]));
        else
            vlog::line("spec", vlog::fmt("ngram  k=%d kept %d/%d  match %d%s from %s  window %s (%d rows)  -> %d tok  acc[%d] %.2f/%.2f/%.2f", k, m, k, dr.match,
                                         dr.aligned ? " (aligned)" : "", vlog::thou(dr.src).c_str(), vlog::ms(ms_since(t_call)).c_str(), k + 1, m + 1, dr.cls,
                                         spec_.acc_ng[dr.cls][1], spec_.acc_ng[dr.cls][2], spec_.acc_ng[dr.cls][3]));
    }
    return m + 1;
}

std::string GpuBackend::spec_stats() const {
    if (!spec_.on || spec_.tokens == 0) return {};
    const long passes = spec_.cycles + spec_.ng_cycles + spec_.df_cycles + spec_.runs;
    char buf[360];
    int n = std::snprintf(buf, sizeof buf, "%ld/%ld (%.2f tokens a pass, %ld of %ld drafts kept, %ld plain rows)", spec_.tokens, passes,
                          double(spec_.tokens) / double(std::max(1L, passes)), spec_.accepted + spec_.ng_accepted + spec_.df_accepted,
                          spec_.drafted + spec_.ng_drafted + spec_.df_drafted, spec_.plain);
    if ((spec_.ngram && spec_.ng_cycles > 0 || spec_.dflash && spec_.df_cycles > 0) && n > 0 && size_t(n) < sizeof buf)
        n += std::snprintf(buf + n, sizeof buf - size_t(n), " [mtp %ld cycles, ngram %ld cycles %ld of %ld kept, dflash %ld cycles %ld of %ld kept]", spec_.cycles, spec_.ng_cycles,
                      spec_.ng_accepted, spec_.ng_drafted, spec_.df_cycles, spec_.df_accepted, spec_.df_drafted);
    // the window costs the q4e policy has learned (ms, rows 2..8; - = not seen)
    const bool learned = std::any_of(spec_.qn + 2, spec_.qn + 9, [](int c) { return c > 0; });
    for (int T = 2; learned && T <= 8 && n > 0 && size_t(n) < sizeof buf; ++T) {
        const int w = spec_.qn[T] > 0 ? std::snprintf(buf + n, sizeof buf - size_t(n), T == 2 ? " [windows %.0f" : "/%.0f", spec_.qw[T])
                                       : std::snprintf(buf + n, sizeof buf - size_t(n), T == 2 ? " [windows -" : "/-");
        n += w;
        if (T == 8 && size_t(n) < sizeof buf) n += std::snprintf(buf + n, sizeof buf - size_t(n), " ms]");
    }
    spec_.runs = spec_.cycles = spec_.drafted = spec_.accepted = spec_.tokens = spec_.plain = 0;
    spec_.ng_cycles = spec_.ng_drafted = spec_.ng_accepted = 0;
    spec_.df_cycles = spec_.df_drafted = spec_.df_accepted = 0;
    return buf;
}

uint64_t GpuBackend::vram_free() const {
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    return uint64_t(free_b);
}

std::vector<std::string> GpuBackend::startup_report() const {
    using vlog::fmt;
    using vlog::thou;
    std::vector<std::string> r;
    const model::Geometry& geo = model_->geometry();
    const Gpu& g = *gpu_;
    r.push_back(fmt("model: %s, %d layers%s, hidden %s, %d heads / %d KV, head dim %d, vocab %s%s%s; weights %s in %s (%s)", geo.arch.c_str(), geo.n_layer,
                    geo.full_attn_interval > 0 ? fmt(" (full attention every %d)", geo.full_attn_interval).c_str() : "", thou(geo.n_embd).c_str(), geo.n_head, geo.n_head_kv,
                    geo.head_dim, thou(geo.n_vocab).c_str(), geo.n_expert > 0 ? fmt(", %d experts (%d used)", geo.n_expert, geo.n_expert_used).c_str() : ", dense",
                    geo.has_mtp ? ", MTP block in the file" : "", vlog::bytes(g.weight_bytes()).c_str(), vlog::ms(g.load_ms()).c_str(),
                    device_.empty() ? "GPU" : device_.c_str()));
    const Gpu::PoolStats ps = g.pool_stats();
    r.push_back(fmt("kv: %s, %d pages x %d positions = %s positions for all slots, %s (%s a position, every full-attention layer)", kv_format_name(g.kv_format()), ps.pages,
                    ps.page_positions, thou(ps.pages * ps.page_positions).c_str(), vlog::bytes(g.kv_cache_bytes()).c_str(),
                    vlog::bytes(ps.pages > 0 ? g.kv_cache_bytes() / uint64_t(ps.pages * ps.page_positions) : 0).c_str()));
    if (!kv_why_.empty()) r.push_back(kv_why_);
    r.push_back(fmt("boot: prefill planes %d, chunk %d tokens", g.prefill_planes(), g.prefill_chunk()));
    if (spec_.mtp)
        r.push_back(fmt("mtp: calibrated at position 0: plain row %.1f ms, windows of 2/3/4/8 rows %.1f/%.1f/%.1f/%.1f ms, block step %.1f ms; draft vocabulary 65,536 ids", spec_.t_plain,
                        spec_.t_win[2], spec_.t_win[3], spec_.t_win[4], spec_.t_win[8], spec_.t_chain));
    else if (spec_.on)
        r.push_back(fmt("mtp: not loaded (--spec ngram, no block in the file, or it did not fit); windows calibrated: plain row %.1f ms, windows of 2/3/4/8 rows %.1f/%.1f/%.1f/%.1f ms",
                        spec_.t_plain, spec_.t_win[2], spec_.t_win[3], spec_.t_win[4], spec_.t_win[8]));
    else r.push_back("mtp: off (--spec off or ngram unusable, no block in the file, or it did not fit)");
    if (spec_.dflash)
        r.push_back(fmt("dflash: %s, matrices in %s, candidates from the first %s ids, one pass drafts up to 7 tokens (%.1f ms; plain row %.1f ms); a context of %s positions a slot (%s); fed by every pass, chosen per cycle by tokens per ms, k by the drafts' own confidence%s",
                        draft_dir_.c_str(), g.df_quant_name(), thou(g.df_vocab()).c_str(), spec_.t_df, spec_.t_plain, thou(2048).c_str(), vlog::bytes(g.df_slot_bytes()).c_str(),
                        spec_.fixed_k ? fmt("; k fixed at %d", spec_.fixed_k).c_str() : ""));
    else if (!df_note_.empty())
        r.push_back("dflash: not loaded: " + df_note_);
    if (spec_.ngram)
        r.push_back(fmt("ngram: drafts from each slot's tape: %d tokens must repeat, up to %d drafted, a window of at most %d rows; index %s a slot at %s positions; a cycle takes the source with the most tokens per ms (plain row %.1f ms)",
                        ng_n_, spec_.ng_max, spec_.ng_max + 1, vlog::bytes(ng_.size() ? ng_.at(0).bytes() + (uint64_t(4) << 20) : 0).c_str(), thou(n_ctx_).c_str(), spec_.t_plain));
    else r.push_back("ngram: off (--spec mtp or off)");
    return r;
}

std::vector<serve::Backend::MemUse> GpuBackend::memory_report() const {
    using vlog::fmt;
    const Gpu& g = *gpu_;
    const Gpu::PoolStats ps = g.pool_stats();
    std::vector<MemUse> m;
    m.push_back({"weights", g.weight_bytes(), "the model's tensors, resident"});
    m.push_back({"K/V pool", g.kv_cache_bytes(), fmt("%s: %d pages x %d positions", kv_format_name(g.kv_format()), ps.pages, ps.page_positions)});
    m.push_back({"slot state", uint64_t(g.slots()) * g.slot_bytes(), fmt("%d slot%s x %s of recurrent state", g.slots(), g.slots() == 1 ? "" : "s", vlog::bytes(g.slot_bytes()).c_str())});
    m.push_back({"checkpoints", uint64_t(g.snapshot_slots()) * g.snapshot_bytes(), fmt("%d snapshot slots x %s", g.snapshot_slots(), vlog::bytes(g.snapshot_bytes()).c_str())});
    if (spec_.mtp) m.push_back({"MTP block", mtp_bytes_, "its tensors, K/V, window activations and graphs"});
    if (spec_.dflash) m.push_back({"DFlash2 drafter", df_bytes_ + uint64_t(g.slots() > 1 ? g.slots() - 1 : 0) * g.df_slot_bytes(), fmt("weights, scratch, stages and %d context%s of %s", g.slots(), g.slots() == 1 ? "" : "s", vlog::bytes(g.df_slot_bytes()).c_str())});
    m.push_back({"scratch", scratch_bytes_, "prefill buffers and split flash-decode partials"});
    uint64_t listed = 0;
    for (const MemUse& u : m) listed += u.bytes;
    const uint64_t now_free = vram_free();
    const uint64_t held = free_at_open_ > now_free ? free_at_open_ - now_free : 0;
    m.push_back({"other", held > listed ? held - listed : 0, "CUDA context, captured graphs, workspaces"});
    return m;
}

std::string GpuBackend::describe() const {
    char buf[160];
    std::snprintf(buf, sizeof buf, "%s, %.2f GB of VRAM", device_.empty() ? "GPU" : device_.c_str(), double(gpu_->vram_bytes()) / 1e9);
    return std::string(buf) + ", " + kv_note_;
}

}  // namespace limbo::cuda
