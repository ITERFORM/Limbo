// Copyright (c) 2026 ITERFORM. All rights reserved.
#pragma once
// limbo/cuda/backend.hpp — limbo::cuda::Gpu as a serve::Backend (rung 1 on the GPU behind `limbo_serve_gpu`).
//
// Prompt tokens go through Gpu::prefill() in the engine's chunks with no output head (the server looks after
// the client between chunks). A step that wants logits copies them into one reused
// pinned host buffer; at temperature 0 the server asks forward_greedy() instead, which takes the device
// argmax, so a greedy token costs one int over PCIe rather than 1 MB of logits.
//
// Gpu is one sequence on one stream and the card holds one rung-1 engine: construct at most one GpuBackend
// per process, and only after taking the machine's GPU lock (flock build/gpu.lock) when other jobs share it.
// Beside the live conversation it keeps Gpu's snapshot slots and extra slots for the server's tape.

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "limbo/cuda/gpu.hpp"
#include "limbo/model.hpp"
#include "limbo/ngram.hpp"
#include "limbo/serve.hpp"

namespace limbo::cuda {

// How the GPU backend is set up at open (limbo_serve_gpu). The K/V pool covers n_ctx positions for all slots together (plus a
// page per slot), in the most exact format (f16, then q8, then q4) that leaves room for `slots` slots, each with `checkpoints`
// snapshot slots beside it, the MTP block and `reserve`; "f16" / "q8" / "q4" take that format. When the pool cannot cover n_ctx
// even in q4 it is what fits, and the start-up line says so with the numbers. prefill_planes: the batched prefill's activation
// planes (Gpu::set_prefill; 0 keeps the engine's default). Past 131072 positions the prefill chunk is 512 (its buffers halve; the
// prompt attention dominates there anyway).
struct GpuOptions {
    std::string want = "f16";
    int slots = 1;
    int checkpoints = 0;
    uint64_t reserve = 0;
    int ctx_floor = 0;
    // the zoo's smaller quants of draft_model, best first: the planner takes the best that fits
    std::vector<std::string> draft_alts;   // --ctx auto: the pool may shrink to this many positions to keep a drafter (0: the context is fixed)
    int prefill_planes = 0;
    // Speculation: "mtp" loads the model's own next-token block when the file has one and it fits beside the pool and the slots (it
    // ranks above the cache's precision, below the slots); "ngram" drafts from the slot's own tape (src/ngram.cpp) and loads no block;
    // "auto" is both, chosen per cycle by expected tokens per millisecond; "off" is neither. "dflash" drafts from the DFlash2 block
    // drafter alone; "auto" adds it as a third source when draft_model is set (docs/dflash2.md). It ranks beside the MTP block: above
    // the cache's precision, below the slots.
    std::string spec = "auto";
    std::string draft_model;  // the DFlash2 checkpoint: a directory with config.json and model.safetensors, or the .safetensors file; "" = none
    std::string draft_quant = "auto";  // native | q8 | bf16; auto: native for a GGUF drafter, q8 for safetensors
    int draft_vocab = 0;      // candidates from the first N ids of the vocabulary (0: 65,536, as the MTP block; N >= the vocabulary: all)
    int draft_k = 0;          // a fixed number of drafts per cycle, 1..7 (0: chosen per cycle)
    int ngram_n = 0;    // tokens that must repeat before an ngram draft is made (0: kDefaultNgramN)
    int ngram_max = 0;  // the longest ngram draft, 1..7 (0: 7)
};

class GpuBackend final : public serve::Backend {
public:
    // Opens the model, uploads the main blocks to VRAM, allocates the scratch and the pinned logits buffer and sizes the K/V cache
    // by `plan`. nullptr and *err when the file does not open, the card has less than min_free_bytes free, or a CUDA call fails.
    static std::unique_ptr<GpuBackend> open(const std::vector<std::string>& shards, int n_ctx, uint64_t min_free_bytes,
                                            std::string* err, const GpuOptions& plan = {});
    ~GpuBackend() override;

    int n_vocab() const override;
    int n_ctx() const override;
    void reset() override;
    static constexpr int kDefaultNgramN = 8;  // rung 1 is indifferent to 4..12 on repeats; 8 drafts less from coincidences in prose
    const float* forward(int32_t token, int pos) override;
    const float* forward(int32_t token, int pos, bool want_logits) override;
    int32_t forward_greedy(int32_t token, int pos) override;
    void prefill(const int32_t* tokens, int n, int pos0) override;
    int prefill_chunk() const override { return gpu_->prefill_chunk(); }
    void sync() override { gpu_->sync(); }
    std::string describe() const override;

    // The tape (serve.hpp): recurrent-state snapshots in VRAM and slots (one conversation each), both taken only
    // while the reserve (set_vram_reserve, 384 MiB until the server says otherwise) stays free on the card: the desktop and the other jobs share it.
    int set_snapshot_slots(int want) override { return gpu_->set_snapshot_slots(want, reserve_); }
    int snapshot_slots() const override { return gpu_->snapshot_slots(); }
    bool snapshot(int slot) override { return gpu_->snapshot(slot); }
    bool restore(int slot) override {
        spec_.live = false;
        return gpu_->restore(slot);
    }
    // Gpu::prefill() reads a piece shorter than 8 tokens token by token (the batched path starts at 8), which is
    // slower per token and rounds differently.
    int prefill_batch_min() const override { return 8; }
    int set_slots(int want) override {
        const int n = gpu_->set_slots(want, reserve_);
        if (spec_.ngram) ng_.resize(n);
        return n;
    }
    int slots() const override { return gpu_->slots(); }
    bool select_slot(int h) override {
        spec_.live = false;
        const bool ok = gpu_->select_slot(h);
        if (ok) ng_.select(h);
        return ok;
    }
    void set_vram_reserve(uint64_t bytes) override { reserve_ = bytes; }
    uint64_t vram_free() const override;  // cudaMemGetInfo, now
    uint64_t slot_bytes() const override { return gpu_->slot_bytes(); }
    uint64_t snapshot_bytes() const override { return gpu_->snapshot_bytes(); }

    // The paged pool (docs/paged-kv.md): every slot's K/V lives in it, a slot starts as another's prefix by reference.
    Pool pool() const override {
        const Gpu::PoolStats s = gpu_->pool_stats();
        Pool p;
        p.positions = s.pages * s.page_positions;
        p.free_positions = (s.pages - s.used) * s.page_positions;
        p.used_pages = s.used;
        p.shared_pages = s.shared;
        p.total_pages = s.pages;
        p.page = s.page_positions;
        p.copies = s.copies;
        return p;
    }
    int slot_pages(int slot) const override { return gpu_->slot_pages(slot); }
    void cut(int keep) override {
        spec_.live = false;
        if (spec_.ngram) ng_.cur().truncate(size_t(std::max(0, keep)));
        gpu_->slot_cut(keep);
    }
    bool reserve(int n) override { return gpu_->slot_reserve(n); }
    bool share(int from, int n) override {
        spec_.live = false;
        const bool ok = gpu_->slot_share(from, n);
        if (ok && spec_.ngram) ng_.share(from, size_t(n));
        return ok;
    }
    int room(int at) const override { return gpu_->slot_room(at); }

    // Speculation with the model's MTP block (Gpu::mtp_rows): a run drafts k tokens through the block, checks [fed, drafts] in
    // one step_rows window and commits the accepted prefix. k is chosen per cycle from the measured window and block costs and
    // an EMA of each draft position's acceptance.
    bool spec_ready() const override { return spec_.on; }
    int decode_spec(int32_t fed, int pos, int max_out, const SpecPick& pick, int32_t* out) override;
    void prompt_end(int end) override { if (spec_.dflash) gpu_->df_prefill_from(end - 2048 - 16); }
    std::string spec_stats() const override;

    // The verbose log (serve.hpp): -vv adds a line per speculation cycle; the start-up block lists what open() decided and measured.
    void set_verbose(int level) override { verbose_ = level; }
    std::vector<std::string> startup_report() const override;
    std::vector<MemUse> memory_report() const override;

    const Gpu& gpu() const { return *gpu_; }
    const std::string& device_name() const { return device_; }

    // Tokens per Gpu::prefill() call: the engine's own batched chunk (Gpu::set_prefill, default 1024 —
    // about half a second of prompt at ~2,000 tok/s, so the server still looks after the client in time).

private:
    GpuBackend() = default;
    void step(int32_t token, int pos, bool head);  // throws std::runtime_error when a launch fails

    std::unique_ptr<model::Model> model_;
    std::unique_ptr<Gpu> gpu_;
    float* logits_ = nullptr;  // pinned, n_vocab floats
    std::string device_;
    std::string kv_note_;  // the K/V format and why (describe())
    std::string kv_why_;   // the same as lines for -v: what each format needed against the budget
    int verbose_ = 0;
    uint64_t free_at_open_ = 0;      // device bytes free before the weights went up
    uint64_t scratch_bytes_ = 0;     // measured: the prefill and decode scratch (reserve_scratch)
    uint64_t mtp_bytes_ = 0;         // measured: the MTP block
    struct Spec {
        bool on = false;            // some speculation is on: windows are calibrated and decode_spec runs
        bool mtp = false;           // the block is loaded
        bool ngram = false;         // drafts from the slot's tape
        bool dflash = false;        // the DFlash2 drafter is loaded
        bool live = false;          // the next call may continue the run: nothing else touched the state since
        int next_pos = -1;          // where that call must be
        int32_t c = -1;             // ... and the token it must feed (what the last call returned last)
        int base = 0;               // the block's position 0 among the model's positions
        int R = 0;                  // rows of the last window the block has not read with their next token
        std::vector<int32_t> prev;  // the last window's tokens
        double acc[16] = {};        // acceptance of draft j given the earlier ones, an EMA
        double t_win[9] = {};       // ms of a window of T rows and its commit, measured at load
        double t_plain = 0;         // ms of a plain row with write-back
        double s_row = 1, s_win = 1;  // measured / calibrated ms of a plain row and of a cycle (the calibration is at position 0): EMAs
        int since_probe = 0;        // plain rows since the last drafting cycle
        double t_chain = 0;         // ms of one chained draft through the block
        bool calibrated = false;
        int kmax = 4;
        double min_p = -1;          // q4e: a draft joins while p >= min_p under the block (LIMBO_SPEC_MIN_P, 0.5; 0 = the cost model)
        // ngram: acceptance of draft j by match class (ngram::Draft::cls), the cost scale of its windows, cycles since a draft was last tried
        double acc_ng[ngram::kClasses][8] = {};
        double s_ngw = 1;
        // qwen4exp: what a window of T rows really costs, learned per T from the passes themselves. The load-time
        // calibration runs one repeated token, so its rows all route alike and the lane's share -- every new row brings
        // experts of its own -- never shows: it reads flat where a real window climbs ~12 ms a row (54/67/80/92 ms for 2-5).
        double qw[9] = {};
        int qn[9] = {};
        double q_chain = 0;   // ms of one draft of the block, learned
        int since_probe_ng = 0;
        int ng_max = 7;
        long ng_cycles = 0, ng_drafted = 0, ng_accepted = 0;
        // dflash: acceptance of draft j given the earlier ones, the draft pass's ms (measured at load), the cost scale of its cycles, cycles since it was last tried
        double acc_df[8] = {};
        double acc_dc[8][4];       // the same by the drafter's confidence in the draft (bucket edges 0.5 / 0.8 / 0.95)
        // DSpark (--spec dspark): the same acceptance as a CALIBRATED continuous function of the draft's own confidence,
        // p_i = sigmoid(a * conf_i + b), with (a, b) fitted online from the observed keep/drop pairs. The bucket table above
        // is the same idea discretised into four fixed cells whose edges were chosen by hand; this is the continuous form
        // that converges to whatever this drafter's confidences actually mean.
        bool dspark = false;
        double dsp_a = 1.0, dsp_b = 0.0;   // a = 1, b = 0 starts at p = conf, which is already better than a fixed table
        long dsp_n = 0;
        int fixed_k = 0;
        double t_df = 0;
        double s_dfw = 1;
        int since_probe_df = 0;
        long df_cycles = 0, df_drafted = 0, df_accepted = 0;
        long runs = 0, cycles = 0, drafted = 0, accepted = 0, tokens = 0, plain = 0;  // since the last spec_stats()
    };
    mutable Spec spec_;
    std::vector<float> rows_logits_;  // a sampling pick's rows (read_logits_rows: T x n_vocab)
    void spec_calibrate();
    std::string acc_note(int from, int to) const;  // "0.81/0.62/0.40": the acceptance EMAs of draft positions from..to
    bool load_mtp();
    std::string draft_dir_;  // the DFlash2 checkpoint
    uint64_t df_bytes_ = 0;  // measured: the drafter (weights, scratch, stages, the first slot's context)
    std::string df_note_;    // why it is not loaded, when asked for
    static uint64_t mtp_bytes_estimate(const model::Model& m, int n_ctx);
    int n_ctx_ = 0;
    uint64_t reserve_ = 384ull << 20;
    ngram::Slots ng_;  // one index per slot: the slot's whole tape (every token it has read)
    int ng_n_ = kDefaultNgramN;
    static uint64_t rows_bytes_estimate(const model::Model& m, int n_ctx);
};

// limbo::cuda::InkGpu as a serve::Backend (`limbo_serve_gpu` on an inkling2_loop file).
//
// The whole 36-visit pass runs per token on the device; the adapter adds the host side the server
// needs: one reused pinned logits buffer (forward), or the device argmax at temperature 0
// (forward_greedy: one int over PCIe instead of 800 KB of logits), and prompt tokens run with no
// output head. InkGpu is one conversation on one stream: one InkBackend per process, no extra
// slots, no snapshot slots, no pool — the server's tape falls back to reset-and-re-read on a
// diverging prompt, which is exact.
class InkBackend final : public serve::Backend {
public:
    // Opens the model and uploads it to VRAM. nullptr and *err when the file does not open as an
    // inkling2_loop pass, the card has less than min_free_bytes free, or a CUDA call fails.
    // n_ctx <= 0 fits the context to the card (InkGpu::open), keeping reserve_bytes free.
    // draft_model: Inkling-2's DSpark drafter (dspark-Inkling-2-Mini-*.gguf); speculative decoding then drafts up to
    // draft_k tokens a cycle and checks them in one exact pass (src/cuda/ink_draft.cu). Its device memory is kept
    // free before the context is fitted.
    static std::unique_ptr<InkBackend> open(const std::vector<std::string>& shards, int n_ctx,
                                            uint64_t min_free_bytes, std::string* err, uint64_t reserve_bytes = 0,
                                            const std::string& draft_model = {}, int draft_k = 3, bool kv_q8 = false);
    ~InkBackend() override;

    int n_vocab() const override;
    int n_ctx() const override;
    void reset() override;
    const float* forward(int32_t token, int pos) override;
    const float* forward(int32_t token, int pos, bool want_logits) override;
    int32_t forward_greedy(int32_t token, int pos) override;  // throws std::runtime_error when a launch fails
    // The prompt in InkGpu::prefill_rows pieces (the int8 tensor-core pass, grouped experts).
    void prefill(const int32_t* tokens, int n, int pos0) override;
    int prefill_chunk() const override { return 512; }
    void sync() override;
    std::string describe() const override;
    uint64_t vram_free() const override;
    bool spec_ready() const override;
    int decode_spec(int32_t fed, int pos, int max_out, const SpecPick& pick, int32_t* out) override;
    std::string spec_stats() const override;

private:
    InkBackend();
    struct St;
    std::unique_ptr<St> st_;
};

}  // namespace limbo::cuda
