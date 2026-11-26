// Copyright (c) 2026 ITERFORM. All rights reserved.
// src/cuda/ink_backend.cu — limbo::cuda::InkGpu as a serve::Backend (include/limbo/cuda/backend.hpp):
// the inkling2_loop entry behind `limbo_serve_gpu`. The pass itself is src/cuda/ink_gpu.cu; this file is
// only the server's host side: one pinned logits buffer for requests that sample, the device argmax for
// greedy ones, and prompt tokens with no output head.
#include "limbo/cuda/backend.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <sys/stat.h>

#include "../src/cuda/ink_draft.hpp"
#include "../src/cuda/ink_gpu.hpp"
#include "limbo/model.hpp"

namespace limbo::cuda {

struct InkBackend::St {
    // Declared so the device goes before the pinned buffer in the teardown (destruction runs in reverse).
    std::unique_ptr<InkGpu> gpu;
    std::unique_ptr<InkDraft> draft;
    int draft_k = 3;
    float min_conf = 0.7f;   // measured best of 0.3 / 0.5 / 0.7 / 0.85 on code and prose (k 3)
    long cycles = 0, drafted = 0, accepted = 0;
    double ms_draft = 0, ms_verify = 0;
    float* logits = nullptr;  // pinned, n_vocab floats
    float* rows = nullptr;    // pinned, kRows x n_vocab: a sampling request's checked rows
    int vocab = 0;
    // the drafter's context follows every row the target reads
    void inject(int T, int pos0) {
        if (draft && !draft->inject(gpu->features(), T, pos0)) throw std::runtime_error("inkling2 draft injection failed");
    }
    ~St() {
        draft.reset();
        gpu.reset();
        if (logits) cudaFreeHost(logits);
        if (rows) cudaFreeHost(rows);
    }
};

InkBackend::InkBackend() : st_(new St()) {}
InkBackend::~InkBackend() = default;

std::unique_ptr<InkBackend> InkBackend::open(const std::vector<std::string>& shards, int n_ctx,
                                             uint64_t min_free_bytes, std::string* err, uint64_t reserve_bytes,
                                             const std::string& draft_model, int draft_k, bool kv_q8) {
    if (!draft_model.empty()) {
        // the drafter's weights, its context caches and scratch stay free when the target fits its context
        struct stat sb {};
        if (stat(draft_model.c_str(), &sb) != 0) {
            if (err) *err = "cannot read the draft model " + draft_model;
            return nullptr;
        }
        reserve_bytes += uint64_t(sb.st_size) + (uint64_t(224) << 20);
    }
    std::string e;
    std::unique_ptr<model::Model> m = model::Model::open(shards, &e);
    if (!m) {
        if (err) *err = e;
        return nullptr;
    }
    std::unique_ptr<InkGpu> gpu = InkGpu::open(*m, n_ctx, min_free_bytes, &e, reserve_bytes, kv_q8);
    if (!gpu) {
        if (err) *err = e;
        return nullptr;
    }
    std::unique_ptr<InkBackend> b(new InkBackend());
    if (!draft_model.empty()) {
        std::unique_ptr<InkDraft> d = InkDraft::open(draft_model, *gpu, 32768, &e);
        if (!d) {
            if (err) *err = e;
            return nullptr;
        }
        gpu->set_capture(true, d->visits());
        b->st_->draft = std::move(d);
        b->st_->draft_k = std::max(1, std::min(draft_k, 7));
        if (const char* c = std::getenv("LIMBO_INK_DRAFT_CONF")) b->st_->min_conf = float(std::atof(c));
    }
    b->st_->gpu = std::move(gpu);
    b->st_->vocab = b->st_->gpu->n_vocab();
    if (cudaMallocHost(reinterpret_cast<void**>(&b->st_->logits), size_t(b->st_->vocab) * sizeof(float)) !=
        cudaSuccess) {
        if (err)
            *err = std::string("cannot allocate the pinned logits buffer: ") +
                   cudaGetErrorString(cudaGetLastError());
        return nullptr;
    }
    return b;
}

int InkBackend::n_vocab() const { return st_->vocab; }
int InkBackend::n_ctx() const { return st_->gpu->n_ctx(); }
void InkBackend::reset() {
    st_->gpu->reset();
    if (st_->draft) st_->draft->reset();
}
void InkBackend::sync() { st_->gpu->sync(); }

const float* InkBackend::forward(int32_t token, int pos) { return forward(token, pos, true); }

const float* InkBackend::forward(int32_t token, int pos, bool want_logits) {
    if (!st_->gpu->step(token, pos, want_logits)) throw std::runtime_error("inkling2 device step failed");
    st_->inject(1, pos);
    if (!want_logits) return nullptr;
    if (!st_->gpu->read_logits(st_->logits)) throw std::runtime_error("inkling2 logits copy failed");
    return st_->logits;
}

int32_t InkBackend::forward_greedy(int32_t token, int pos) {
    if (!st_->gpu->step(token, pos, true)) throw std::runtime_error("inkling2 device step failed");
    st_->inject(1, pos);
    const int32_t t = st_->gpu->argmax();
    if (t < 0 || t >= st_->vocab) throw std::runtime_error("inkling2 argmax failed");
    return t;
}

void InkBackend::prefill(const int32_t* tokens, int n, int pos0) {
    for (int i = 0; i < n;) {
        const int T = std::min(InkGpu::kPiece, n - i);
        if (!st_->gpu->prefill_rows(tokens + i, T, pos0 + i, false))
            throw std::runtime_error("inkling2 device prefill failed");
        st_->inject(T, pos0 + i);
        i += T;
    }
}

std::string InkBackend::describe() const {
    return st_->draft ? "inkling2_loop, device pass (one conversation), DSpark draft k " + std::to_string(st_->draft_k)
                      : "inkling2_loop, device pass (one conversation)";
}

bool InkBackend::spec_ready() const { return st_->draft != nullptr; }

// One cycle: draft k tokens after `fed`, read [fed, drafts] in one exact pass (every row's greedy token is step()'s),
// keep the agreeing prefix plus the target's own next token. Sampling requests take the plain path.
int InkBackend::decode_spec(int32_t fed, int pos, int max_out, const SpecPick& pick, int32_t* out) {
    St& S = *st_;
    const int k = std::min({S.draft_k, max_out - 1, S.gpu->n_ctx() - pos - 1, InkGpu::kRows - 1});
    int32_t d[8] = {}, g[8] = {}, toks[8] = {};
    float conf[8] = {};
    const auto t0 = std::chrono::steady_clock::now();
    // a sampling request needs the server's sampler on every checked row: the rows' logits cross over
    if (!pick.greedy && !S.rows &&
        cudaMallocHost(reinterpret_cast<void**>(&S.rows), size_t(InkGpu::kRows) * size_t(S.vocab) * sizeof(float)) !=
            cudaSuccess)
        S.rows = nullptr;
    const bool can = S.draft && k >= 1 && (pick.greedy || (S.rows && pick.row));
    int nd = can ? S.draft->draft(fed, pos, k, d, conf) : 0;
    // check only the drafts the drafter is sure enough of: a checked row costs about a third of a step
    int keep = 0;
    while (keep < nd && conf[keep] >= S.min_conf) ++keep;
    nd = keep;
    const auto t1 = std::chrono::steady_clock::now();
    if (nd < 1) return Backend::decode_spec(fed, pos, max_out, pick, out);
    toks[0] = fed;
    for (int i = 0; i < nd; ++i) toks[i + 1] = d[i];
    if (!S.gpu->verify(toks, nd + 1, pos, g, pick.greedy ? nullptr : S.rows))
        throw std::runtime_error("inkling2 verify pass failed");
    S.ms_draft += std::chrono::duration<double, std::milli>(t1 - t0).count();
    S.ms_verify += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
    int j = 0;
    if (pick.greedy) {
        while (j < nd && g[j] == d[j] && !(pick.stop && pick.stop(d[j]))) {
            out[j] = d[j];
            ++j;
        }
        out[j] = g[j];
    } else {
        // the server's sampler on each row: it keeps the draft with the target's probability of it, else draws from
        // the target's distribution without it (an exact draw for a point-mass draft); the last row has no draft
        for (;; ++j) {
            bool kept = false;
            const int32_t t = pick.row(S.rows + size_t(j) * size_t(S.vocab), j < nd ? d[j] : -1, &kept);
            out[j] = t;
            if (!(j < nd && kept) || (pick.stop && pick.stop(t))) break;
        }
    }
    const int n = j + 1;   // rows kept: fed and the j accepted drafts
    if (!S.gpu->commit(n)) throw std::runtime_error("inkling2 verify commit failed");
    S.inject(n, pos);
    ++S.cycles;
    S.drafted += nd;
    S.accepted += j;
    return n;
}

std::string InkBackend::spec_stats() const {
    const St& S = *st_;
    if (!S.draft || S.cycles == 0) return {};
    char b[160];
    std::snprintf(b, sizeof b, "dspark %ld cycles, %ld/%ld drafts kept (%.0f%%), %.2f tokens a cycle, draft %.1f ms verify %.1f ms",
                  S.cycles, S.accepted, S.drafted, S.drafted ? 100.0 * double(S.accepted) / double(S.drafted) : 0.0,
                  double(S.accepted + S.cycles) / double(S.cycles), S.ms_draft / double(S.cycles),
                  S.ms_verify / double(S.cycles));
    return b;
}

uint64_t InkBackend::vram_free() const {
    size_t f = 0, t = 0;
    return cudaMemGetInfo(&f, &t) == cudaSuccess ? f : 0;
}

}  // namespace limbo::cuda
