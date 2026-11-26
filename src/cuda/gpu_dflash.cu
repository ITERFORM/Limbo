// Copyright (c) 2026 ITERFORM. All rights reserved.
// src/cuda/gpu_dflash.cu — Gpu's side of the DFlash2 drafter (include/limbo/cuda/dflash.hpp, docs/dflash2.md): loading, the ingest of
// what a pass read, the block draft. The passes themselves (gpu_decode.cu, gpu_rows.cu, gpu_prefill.cu) only call Dflash::capture at
// the layer inputs; the calls that make those rows the drafter's context are here.
#include <algorithm>

#include "gpu_impl.hpp"
#include "limbo/cuda/gpu.hpp"

namespace limbo::cuda {

bool Gpu::Impl::df_ingest(int stage, int r0, int n, int pos0) {
    if (!df || n <= 0) return true;
    return df->ingest(st.cur, stage, r0, n, pos0);
}

bool Gpu::df_load(const std::string& dir, std::string* err, int quant, int vocab) {
    Impl& I = *p_;
    if (I.df) return true;
    auto d = Dflash::load(dir, err, quant);
    if (!d) return false;
    if (!d->bind(I.embd.type, I.embd.p, I.head.type, I.head.p, I.c.n_vocab, I.c.n_embd, I.stream)) {
        cudaGetLastError();
        if (err) *err = "dflash: the drafter does not fit this model (hidden size / vocabulary) or the scratch did not fit";
        return false;
    }
    d->set_vocab_limit(vocab);
    for (int t : d->cfg().target_layers)
        if (t + 1 >= I.c.n_layer_main) {
            if (err) *err = "dflash: a target layer whose output the pass does not hold (past the model's main layers)";
            return false;
        }
    if (!I.state_ready() || !d->ensure_slot(I.st.cur)) {
        cudaGetLastError();
        if (err) *err = "dflash: no room for the context cache";
        return false;
    }
    // stage 1 holds a prefill chunk's rows (the window is all the drafter reads of a prompt)
    I.df_stage_rows = std::min(I.pf_chunk, d->cfg().window);
    if (!d->reserve_stage(I.df_stage_rows)) {
        cudaGetLastError();
        if (err) *err = "dflash: no room for the prefill stage";
        return false;
    }
    cudaStreamSynchronize(I.stream);
    I.drop_graphs();  // the passes now capture features: graphs made before bake nothing of it in
    I.df = std::move(d);
    return true;
}

bool Gpu::df_ready() const { return bool(p_->df); }
const char* Gpu::df_quant_name() const { return p_->df ? p_->df->quant_name() : "bf16"; }
int Gpu::df_vocab() const { return p_->df ? p_->df->vocab_limit() : 0; }

uint64_t Gpu::df_bytes() const {
    const Impl& I = *p_;
    return I.df ? I.df->weight_bytes() + I.df->scratch_bytes() + I.df->stage_bytes() : 0;
}

uint64_t Gpu::df_slot_bytes() const { return p_->df ? p_->df->ring_bytes() : 0; }

int Gpu::df_len() const { return p_->df ? p_->df->slot_len(p_->st.cur) : 0; }

void Gpu::df_prefill_from(int pos) { p_->df_from = std::max(0, pos); }

bool Gpu::df_draft_start(int32_t anchor, int pos) {
    Impl& I = *p_;
    return I.df && I.df->draft_start(I.st.cur, anchor, pos);
}

bool Gpu::df_draft_wait(int32_t* out7, float* conf7) { return p_->df && p_->df->draft_wait(out7, conf7); }

uint64_t Gpu::df_estimate(const std::string& dir, int slots, int chunk, int quant) { return Dflash::estimate(dir, slots, std::min(chunk, 2048), quant); }

}  // namespace limbo::cuda
