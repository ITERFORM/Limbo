// Copyright (c) 2026 ITERFORM. All rights reserved.
#pragma once
// src/cuda/ink_draft.hpp — Inkling-2's DSpark block drafter on the device (docs/inkling2.md, "DSpark"):
// the 3-layer draft model shipped as mtp.safetensors (dspark-Inkling-2-Mini-*.gguf, arch "dflash"). It reads no
// prompt itself: every position the target reads leaves its features (InkGpu::set_capture) and the drafter turns
// them into its own context K/V; a draft is one pass over the block [anchor, mask x 7] plus the Markov head's chain.
// Only .cu translation units include this header.

#include <cstdint>
#include <memory>
#include <string>

#include "ink_gpu.hpp"

namespace limbo::cuda {

class InkDraft {
public:
    // Loads the drafter next to `target` (its embedding table and LM head are the target's). full_cap: positions the
    // full-attention layer keeps (a ring; the sliding layers keep their window). nullptr and *err when the file is not
    // an Inkling-2 DSpark drafter or a CUDA call fails.
    static std::unique_ptr<InkDraft> open(const std::string& path, const InkGpu& target, int full_cap, std::string* err);
    ~InkDraft();
    InkDraft(const InkDraft&) = delete;
    InkDraft& operator=(const InkDraft&) = delete;

    // The target's visits whose inputs are the drafter's first three features (the fourth is the final norm).
    const int* visits() const;
    // Features of T target rows at positions pos0.. ([T][4][n_embd], device): their context K/V in every layer, and
    // the Markov head's context state of the last one. Queued on the target's stream.
    bool inject(const float* feats, int T, int pos0);
    // Drafts up to k (<= 7) tokens following `anchor` at `pos` (the context must hold pos - 1). Returns how many it
    // wrote to out, 0 when it cannot draft. Synchronises.
    // probs (optional): each draft's softmax probability under the drafter (its confidence).
    int draft(int32_t anchor, int pos, int k, int32_t* out, float* probs = nullptr);
    int last_pos() const;   // the last position injected, -1 none
    void reset();           // forget the context (the target was reset; positions are injected again)

private:
    InkDraft();
    struct St;
    std::unique_ptr<St> st_;
};

}  // namespace limbo::cuda
