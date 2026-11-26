// Copyright (c) 2026 ITERFORM. All rights reserved.
#pragma once
// limbo/cuda/dflash.hpp — the DFlash2 block drafter on the GPU (docs/dflash2.md). The checkpoint is a small 6-layer decoder that reads
// the target's hidden states (8 of its layers, fused by `fc` into one vector per position) as a K/V cache of context and fills a block
// of 8 positions [anchor, MASK x 7] in ONE non-causal pass; a selector over the top-16 candidates of each position turns the block into
// 7 draft tokens. It has no embedding or head of its own: the target's are used. Weights are BF16 in the file and stay BF16 on the card.
//
// The class owns the weights, the per-slot context cache (a ring of the last 2048 + 16 positions' K/V in 6 layers), its scratch and
// the kernels; the target's pass only calls capture() where a layer's input is live and the caller calls ingest() / draft().
// Everything is queued on the stream given at bind(); draft() synchronises. Only .cu translation units include this header.

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace limbo::cuda {

struct DflashConfig {
    int n_layer = 6, hidden = 2048, n_head = 32, n_kv = 8, hd = 128, ffn = 6144;
    int block = 8;                 // the block's positions, anchor first: block - 1 drafts
    int mask_token = 248077;
    int top_k = 16, rank = 256;    // selector
    int conv_k = 2, conv_group = 16;
    int window = 2048;             // sliding attention: a position sees the context positions within `window`
    int n_target = 40;             // layers of the target
    float eps = 1e-6f;
    double rope_base = 1e7;
    std::vector<int> target_layers;  // the target layers whose OUTPUTS feed fc (config.json's ids), in the order of fc's columns
};

class Dflash {
public:
    static constexpr int kRing = 2048 + 16;  // positions the context cache holds per slot (the window and the stale rows past it)
    static constexpr int kMaxDraft = 7;      // block - 1

    // Reads <path>/config.json and <path>/model.safetensors (or, when path names a .safetensors file, config.json beside it),
    // checks every tensor's shape against the config, uploads the weights. nullptr and *err on failure. The codebooks of the selector
    // (2 x vocab x 256 BF16) stay in mapped host memory: a draft reads 32 rows of each. quant: kBf16, kQ8 (Q8_0 made at load) or
    // kNative (a GGUF drafter's own formats: the file's bytes on the card).
    static constexpr int kBf16 = 0, kQ8 = 1, kNative = 2;
    static std::unique_ptr<Dflash> load(const std::string& path, std::string* err, int quant = kBf16);
    // Device bytes load() + bind() + reserve_stage(stage_rows) + `slots` contexts will take, from the files alone (0 when they cannot be read).
    static uint64_t estimate(const std::string& path, int slots, int stage_rows, int quant = kBf16);
    ~Dflash();
    Dflash(const Dflash&) = delete;
    Dflash& operator=(const Dflash&) = delete;

    const DflashConfig& cfg() const;
    uint64_t weight_bytes() const;   // device bytes of the weights
    uint64_t scratch_bytes() const;  // device bytes of the scratch bind() allocated
    uint64_t ring_bytes() const;     // device bytes of one slot's context cache

    // The target: its token embedding and head (device-visible pointers, `Type` ids of limbo/types.hpp), vocabulary and width, and
    // the stream everything runs on. n_embd must equal the drafter's hidden size. Allocates the scratch.
    bool bind(uint32_t embd_type, const void* embd, uint32_t head_type, const void* head, int n_vocab, int n_embd, cudaStream_t s);
    bool bound() const;
    // Score only the first n ids of the vocabulary as candidates (the head is then read over a prefix of its rows; ids are roughly in
    // order of use). 0: all.
    void set_vocab_limit(int n);
    bool is_q8() const;
    const char* quant_name() const;   // "bf16", "q8" or "native"
    int vocab_limit() const;  // the candidate prefix in effect (the whole vocabulary when none)

    // ---- the target's side ---------------------------------------------------------------------------------------------------
    // Rows of the target's residual stream x [T][n_embd] (fp32) at the input of its layer `layer` (= the output of layer - 1) go to the
    // feature stage, row r of `stage` at column block k (the position of layer - 1 in target_layers), as BF16. A no-op for any other layer.
    // The stage has 8 rows (stage 0) or `rows` rows after reserve_stage(rows) (stage 1, never part of a captured graph).
    void capture(int layer, const float* x, int T, int stage, cudaStream_t s) const;
    // the layer whose input is captured -> column block, -1 when it is not used
    int column_of(int layer) const;
    bool reserve_stage(int rows);  // stage 1 holds `rows` rows; false when it does not fit
    uint64_t stage_bytes() const;

    // ---- the context cache -----------------------------------------------------------------------------------------------------
    // A slot's cache holds the K/V of positions [lo, n) (the last 2048 + 16 at most); drafting attends the positions of the window
    // below the block. ensure_slot allocates the ring (ring_bytes()); false when it does not fit.
    bool ensure_slot(int slot);
    void reset_slot(int slot);                       // empty
    void cut_slot(int slot, int keep);               // positions >= keep are gone
    bool share_slot(int from, int to, int n);        // `to` becomes the first n positions of `from` (empty when `from` does not hold them)
    int slot_len(int slot) const;                    // n: where the next ingest must start
    int slot_lo(int slot) const;
    bool free_slot(int slot);                        // releases the ring

    // Rows r0 .. r0 + n - 1 of the stage (n <= 8 each call; larger n is looped) become positions pos0 .. pos0 + n - 1 of the slot's
    // cache: fc, the norm, K/V of the 6 layers. pos0 != slot_len starts the slot over from pos0 (a gap) or rewinds it (pos0 < len).
    bool ingest(int slot, int stage, int r0, int n, int pos0);

    // ---- drafting -----------------------------------------------------------------------------------------------------------
    // The block at positions pos .. pos + 7 of the slot (anchor = the token at pos): one pass, the selector, the walk. out[0..6] are
    // the drafts of positions pos + 1 .. pos + 7; conf[0..6] the softmax probability of the chosen candidate among the 16. Not
    // synchronised until the host reads the result: draft_start queues it, draft_wait fetches it.
    bool draft_start(int slot, int32_t anchor, int pos);
    bool draft_wait(int32_t* out, float* conf);
    // Debug/test: the block's final hidden rows (8 x hidden, fp32) and the logits' top-16 of the last draft.
    bool debug_hidden(float* out8);

private:
    Dflash();
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace limbo::cuda
