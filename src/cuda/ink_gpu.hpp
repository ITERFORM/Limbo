// Copyright (c) 2026 ITERFORM. All rights reserved.
#pragma once
// src/cuda/ink_gpu.hpp — the whole inkling2_loop forward pass on the device (docs/inkling2.md), the
// inkling2 counterpart of limbo::cuda::Gpu's decode API (include/limbo/cuda/gpu.hpp).
//
// Op for op what limbo::step::Ref's forward_inkling does on the CPU: the weights are copied to VRAM once,
// in their GGUF block layout (the routed experts through src/cuda/ink_moe.cu, every other product through
// limbo/cuda/gemv.hpp, the mechanisms through src/cuda/ink_ops.hpp and ink_flow.hpp), and one step is the
// 36-visit graph — the prelude, the three looped passes with their bank reset and per-layer controller
// messages, the coda, then build_output — as a fixed sequence of kernel launches on one stream with no host
// round trip inside it. The per-slot f16 K/V caches and short-conv histories live on the device; the host
// sees one token in and, if it asks, the logits or the argmax out.

#include <cstdint>
#include <memory>
#include <cuda_runtime.h>

#include <string>

#include "limbo/model.hpp"

namespace limbo::cuda {

class InkGpu {
public:
    // Copies the tensors to VRAM. Refuses (nullptr, *err) when cudaMemGetInfo shows less than
    // min_free_bytes free at the start, or when the file is not an inkling2_loop model this pass runs.
    // n_ctx <= 0 fits the context: after the weights are resident, the largest multiple of 256 positions
    // whose state fits the free VRAM minus reserve_bytes, capped at the file's context_length.
    // kv_q8: K/V rows as int8 with a scale per position and KV head (half the bytes the attention reads).
    static std::unique_ptr<InkGpu> open(const model::Model& m, int n_ctx, uint64_t min_free_bytes, std::string* err,
                                        uint64_t reserve_bytes = 0, bool kv_q8 = false);
    ~InkGpu();
    InkGpu(const InkGpu&) = delete;
    InkGpu& operator=(const InkGpu&) = delete;

    int n_vocab() const;   // the unpadded vocabulary: the head's rows and the logits' width
    int n_ctx() const;
    int n_slots() const;   // the visit slots (each with its own K/V cache and conv history)

    // Queues one whole forward pass of `token` at position `pos` (positions in order from 0 after reset()).
    // With head, the build_output depth mix, the output norm and the LM head run too and the logits are
    // left on the device. Returns false when a launch failed or the position is out of order or does not
    // fit n_ctx(); the pass is not synchronised (argmax() / read_logits() / read_l_out() / sync() do that).
    bool step(int32_t token, int pos, bool head);

    // The same pass over T <= kRows consecutive tokens at positions pos0..pos0+T-1 (the prompt path). The
    // dense projections read each weight once for all T rows (gemv_cols, bit-identical to gemv); everything
    // with state -- the short convolutions, the K/V stores, the attention, the experts -- runs row by row in
    // position order, so the result is the bits T step() calls produce. With head, the head runs for the
    // last row only.
    static constexpr int kRows = 8;
    bool step_rows(const int32_t* tokens, int T, int pos0, bool head);

    // The prompt pass over up to kPiece consecutive tokens: the dense products on the int8 tensor cores over the
    // activation planes (LIMBO_INK_PLANES, default 2: the server's --prefill-planes arithmetic), the routed
    // experts grouped so every expert is read once per piece, the stateful parts in position order. Not
    // step()'s bits -- the planes round the activations -- and gated against it in tests/cuda_ink_step_test.cu.
    static constexpr int kPiece = 512;
    bool prefill_rows(const int32_t* tokens, int T, int pos0, bool head);

    // ---- the draft model's hooks (src/cuda/ink_draft.cu) ------------------------------------------------------
    // set_capture(true, visits): every pass also writes its rows' features to features() -- [row][4][n_embd]: the
    // inputs of the three visits (indices in the visit sequence) and the final norm output -- and the exact batched
    // pass records what a verify window needs to roll back. Recaptures the pass graphs.
    void set_capture(bool on, const int* visits);
    const float* features() const;
    // verify: the exact batched pass over tokens[0..T) at pos0.. with every row's head; out[t] is row t's greedy token
    // (step()'s bits). commit(n) keeps the first n rows (the state is then at pos0 + n): the short-conv histories
    // roll back to row n - 1, the K/V rows past it are overwritten when those positions are read again.
    // logits_out (optional): the rows' logits too, [T][n_vocab] (a sampling request's check, host memory).
    bool verify(const int32_t* tokens, int T, int pos0, int32_t* out, float* logits_out = nullptr);
    bool commit(int n);
    int next_pos() const;
    struct Shared {
        const uint8_t* tok_embd;
        uint32_t tok_type;
        const uint8_t* head;
        uint32_t head_type;
        float head_scale;
        int n_vocab, n_embd, n_ctx;
        cudaStream_t stream;
    };
    Shared shared() const;

    // The greedy token of the last head run (lowest id on a tie). Synchronises; -1 when no head has run.
    int32_t argmax();
    // The n_vocab() logits of the last head run into out. Synchronises.
    bool read_logits(float* out);
    // One visit slot's output (n_embd floats, as limbo::step::Trace's l_out holds them) at the last step
    // into out. Synchronises.
    bool read_l_out(int slot, float* out);

    void sync();
    void reset();   // the KV caches and conv histories zeroed, the position counter back at 0

private:
    InkGpu();
    struct St;
    std::unique_ptr<St> st_;
};

}  // namespace limbo::cuda
