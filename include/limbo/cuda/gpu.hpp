// Copyright (c) 2026 ITERFORM. All rights reserved.
#pragma once
// limbo/cuda/gpu.hpp — the whole qwen35moe forward pass on the device (rung 1, W1 proof of concept).
//
// Op for op what limbo::step::Ref does on the CPU (include/limbo/step.hpp): the weights are copied to VRAM
// once, in their GGUF block layout, and a step is a fixed sequence of kernel launches on one stream with no
// host round trip inside it — the router's top-8 stays on the device and feeds the batched expert GEMVs
// through device memory. The host sees one token in and, if it asks, the logits or the argmax out.
//
// Two ways through the model. step() is one token: a CUDA graph captured after the first passes (the token and
// the position are read from device memory), the fp32-activation GEMVs of the reference's arithmetic.
// prefill() is a prompt: batched over chunks of tokens, the weight products on the int8 tensor cores (a weight
// tile is decoded once and applied to every token of the tile), the delta net scanned in token order inside one
// kernel per head, the attention as an online-softmax tensor-core kernel, the MoE as per-expert token lists.
// Both leave the same state behind (KV cache, delta-net state, conv history), so decode continues from either.
//
// Also qwen35 dense (Qwen3.8-27B and its smaller siblings): the same GDN + gated-attention stack with a dense SwiGLU FFN in place of the
// MoE, every size read from the file (n_embd, n_ff, head counts, any GQA group up to 8). step() and prefill() only (no T-row pass, no MTP
// block); prefill defaults to 2 planes there, and falls back to token-by-token when a weight type has no int8 GEMM (BF16, F32).
// Rung 1: 40 main layers (the MTP block is not loaded), full attention at the interval the model's
// geometry names, 256 experts top-8 with a gated shared expert, a K/V cache in F16, Q8 or Q4 (include/limbo/cuda/kv.hpp),
// GDN state per linear layer, n_ctx up to 262144 (the model's trained length). The token embedding stays in mapped host
// memory (one row per token crosses the bus); LIMBO_EMBD_VRAM=1 puts it on the card.
//
// The K/V cache is one pool of 256-position pages (docs/paged-kv.md) shared by every slot (a slot is one conversation: its page
// table, its recurrent state, its captured graphs). Pages are taken when a position is first written and given back by reset(),
// slot_cut() and erase; a slot can start as the shared prefix of another (slot_share), and a write into a shared page copies it
// first. n_ctx is the most positions one slot can address; the pool holds `pool_pages` pages in all.

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "limbo/cuda/kv.hpp"
#include "limbo/model.hpp"

namespace limbo::cuda {

// qwen4exp's MTP layer file (src/cuda/gpu_q4e_mtp.cu): LIMBO_Q4E_MTP, else mtp-q2_0.gguf beside the model, in ../mtp or ../../mtp; "" none.
std::string q4e_mtp_path(const model::Model& m);

class Gpu {
public:
    // Copies the main-block tensors to VRAM. Refuses (nullptr, *err) when cudaMemGetInfo shows less than
    // min_free_bytes free at the start: the card is shared with the operator's desktop and its engine.
    // kv: the format of the K/V pool (set_kv_format(f, 0) and set_pool change it later, while the state is empty); pool_pages: its
    // size in 256-position pages (0: enough for n_ctx positions).
    static std::unique_ptr<Gpu> open(const model::Model& m, int n_ctx, uint64_t min_free_bytes, std::string* err,
                                     KvFormat kv = KvFormat::F16, int pool_pages = 0, bool q4e_mtp = true);
    ~Gpu();
    Gpu(const Gpu&) = delete;
    Gpu& operator=(const Gpu&) = delete;

    int n_vocab() const;
    int n_ctx() const;

    // Bytes of the resident expert cache (qwen4exp): 0 when every routed expert runs on the CPU lane.
    uint64_t expert_cache_bytes() const;

    // The live slot starts over: its GDN states and conv histories are zeroed and its pages go back to the pool.
    void reset();

    // ---- decode API (owner: the decode agent; gpu_decode.cu) ----------------------------------------------

    // Whether step() replays a captured CUDA graph (default) or launches the ~900 kernels one by one. The first
    // two passes of each variant (head or not) run eagerly and the graph is captured at the second.
    void set_graphs(bool on);

    // Queues one whole forward pass of `token` at position `pos` (positions in order from 0 after reset()).
    // With head, the final norm and the LM head run too and the logits are left on the device.
    // Returns false if a launch failed; the pass is not synchronised (argmax()/read_logits()/sync() do that).
    bool step(int32_t token, int pos, bool head);

    // The format of the full-attention K/V cache (include/limbo/cuda/kv.hpp): F16 (exact, the default), Q8 (q8_0 numerics, half the
    // bytes) or Q4 (rotated keys, q4_0 numerics, 28 percent of the bytes). set_kv_format(f, 0) re-makes the cache empty in any format
    // (the old planes are freed before the new are made; the state is reset; the pool keeps its size). Refused when extra slots exist;
    // on an allocation failure the old format is kept and false returned.
    // set_kv_format(Q8, n) with n != 0 quantises the first n positions of the f16 cache into a new Q8 cache, frees the f16 one and
    // drops the captured decode graphs; decode (step, step_rows) then reads and writes Q8, from the same pages. Refused when extra slots
    // exist. n_filled < 0: every position.
    bool set_kv_format(KvFormat f, int n_filled = -1);
    KvFormat kv_format() const;
    // Debug/bench: fill the first n positions of the live slot (pages taken as needed) with pseudo-random int8 and scales of a plausible
    // size, so decode can be timed at a context that is too long to prefill or whose f16 cache does not fit. Q8 and Q4 only.
    bool kv_fill_synthetic(int n_positions, uint32_t seed = 1);
    // Device bytes of the K/V pool at the current format and size.
    uint64_t kv_cache_bytes() const;
    uint64_t kv_cache_bytes(KvFormat f) const;  // the same in format f
    uint64_t kv_page_bytes(KvFormat f) const;   // one page of 256 positions in format f, every full-attention layer
    // Re-makes the pool empty in format f with `pages` pages (the old planes are freed first; every slot's pages are gone, the
    // live slot's state is reset). False with the old pool kept when the new one does not fit.
    bool set_pool(KvFormat f, int pages);
    int pool_pages() const;
    // Allocates now the scratch the first prefill and the first step would allocate (the batched prefill's buffers, the split
    // flash-decode's partials), so cudaMemGetInfo tells the truth before slots and snapshot slots are sized.
    bool reserve_scratch();

    // A window of T = 1..8 tokens in one pass (verify windows for speculation; src/cuda/gpu_rows.cu): tokens[t] sits at
    // position pos0 + t, every weight is read once for all T rows, the attention is causal inside the window, and the logits
    // of all T rows are left on the device (read_logits_rows / argmax_rows). The rows of the window equal T sequential step()
    // calls. With write_back the delta-net state and the conv history advance by T tokens, as after T steps. Without it they
    // do not move: the per-token inputs are kept, and commit(n) then advances them by exactly the first n tokens (n <= T of
    // the last such window; commit(0) drops the window). K/V rows of rejected tokens are overwritten when those positions
    // are next written. One captured graph per T (and per n for the commit). Not synchronised.
    bool step_rows(const int32_t* tokens, int T, int pos0, bool write_back);
    bool commit(int n);
    // The same window over the qwen4exp (q4e) step, which has its own pass, its own workspace and its own
    // commit. T rows through all n_layer_main layers, every dense weight read once for all of them; row t runs
    // the arithmetic step() would have run for token t. write_back = false leaves the delta-net state and the
    // conv history exactly as they were and q4e_commit(n) then advances them by the first n rows (n <= T of the
    // last such window). read_logits_rows / argmax_rows do NOT read this path's rows: use q4e_read_logits_rows.
    bool q4e_step_rows(const int32_t* tokens, int T, int pos0, bool write_back);
    bool q4e_commit(int n);
    bool q4e_read_logits_rows(float* out, int T);
    // The greedy token of each of the T rows of the last q4e_step_rows (lowest index on a tie). Synchronises.
    bool q4e_argmax_rows(int32_t* out, int T);
    // True for a qwen4exp model: its step, its window and its MoE are a pass of their own, so the dispatch has to
    // know which window it is holding before it drafts anything.
    bool is_q4e() const;
    // The longest window q4e_step_rows can run EXACTLY at pos0 (0 if none): the sparse block selection is not on
    // the GPU yet, so past its budget the visible set is not the plain causal one and the window refuses. The
    // dispatch has to clamp its window length by this, or a refusal throws mid-generation.
    int q4e_window_rows(int pos0) const;
    // Row t's n_vocab logits of the last step_rows into out + t * n_vocab() (T rows). Synchronises.
    bool read_logits_rows(float* out, int T);
    // The greedy token of each of the T rows of the last step_rows (lowest index on a tie). Synchronises.
    bool argmax_rows(int32_t* out, int T);

    // The model's own multi-token-prediction head (the nextn block of a file such as rung 1's blk.40; src/cuda/gpu_rows.cu).
    // mtp_load uploads it and gives it a K/V cache of its own, mtp_capacity() positions (min(n_ctx(), 8192)); false when the file
    // has none or it does not fit. The block's positions (q0 of mtp_rows) are its own: the attention depends on position
    // differences only, so a caller counts them from where it starts the block (and starts again when they run out).
    bool mtp_load(const model::Model& m);
    bool mtp_ready() const;
    int mtp_capacity() const;
    // A draft need not be exact (the verify window decides what is kept): score only the first n_vocab_draft ids of the
    // vocabulary in the block's head (ids are roughly in order of use, so the head is read over a prefix of its rows). 0 or
    // n_vocab(): all of them. Drops the block's captured graphs.
    bool mtp_set_draft_vocab(int n_vocab_draft);
    // One pass of the block over T = 1..8 rows at its positions q0 .. q0 + T - 1 (< mtp_capacity()): row t takes the embedding of
    // tokens[t] and the hidden state of row t of the last step_rows window (h_from_mtp = false: the pass that produced that
    // hidden state must have been a step_rows of at least T rows) or, with h_from_mtp (T = 1), the block's own output of the
    // previous mtp_rows call (chained drafts). The block's K/V for those positions are written. Not synchronised.
    bool mtp_rows(const int32_t* tokens, int T, int q0, bool h_from_mtp);
    // qwen4exp: the block's K/V for n cells from q0, from rows first_row.. of the last window's (or plain row's) final residuals and
    // the token after each; no draft. q4e_rows_end: the position after the last row those residuals hold; q4e_rows_last: their rows.
    bool mtp_front(const int32_t* next_tokens, int n, int first_row, int q0);
    int q4e_rows_end() const;
    int q4e_rows_last() const;
    // The greedy token of each of the T rows of the last mtp_rows (through the model's own head). Synchronises.
    bool mtp_argmax(int32_t* out, int T);
    // mtp_argmax with each draft's probability under the block (qwen4exp; 1 elsewhere).
    bool mtp_argmax_p(int32_t* out, float* prob, int T);

    // The DFlash2 block drafter (docs/dflash2.md, src/cuda/gpu_dflash.cu). df_load reads <dir>/config.json and <dir>/model.safetensors (or a
    // .safetensors path), uploads them and the first slot's context cache; false (and *err) when it does not fit or is not a DFlash2
    // checkpoint for this model. From then on every pass that reads positions (step, prefill, step_rows) captures the target's hidden
    // states the drafter reads and turns them into its context: after commit(n) the n accepted rows, after step / prefill / a
    // write_back window all of theirs. Each slot has its own context (set_slots allocates it, slot_cut / slot_share / reset keep it
    // in step with the pages). df_draft_start queues a block for the token `anchor` at position `pos` of the live slot (the context
    // should end at pos: df_len() == pos); df_draft_wait returns the 7 drafts of pos + 1 .. pos + 7 and the drafter's confidence in each.
    // q8: the drafter's matrices are quantised to Q8_0 at load; vocab: candidates come from the first `vocab` ids (0: all).
    bool df_load(const std::string& dir, std::string* err, int quant = 0, int vocab = 0);   // quant: Dflash::kBf16 / kQ8 / kNative
    bool df_ready() const;
    const char* df_quant_name() const;  // "q8" or "bf16"
    int df_vocab() const;               // candidates come from the first df_vocab() ids
    uint64_t df_bytes() const;       // device bytes of the weights, the scratch and the stages
    uint64_t df_slot_bytes() const;  // one slot's context cache
    int df_len() const;              // positions the live slot's context holds (where the next pass must start for it to stay whole)
    void df_prefill_from(int pos);   // hint: prefill positions below `pos` are not read by the drafter (0: all)
    bool df_draft_start(int32_t anchor, int pos);
    bool df_draft_wait(int32_t* out7, float* conf7);
    // Device bytes df_load will take for `slots` slots, from the checkpoint's header alone; 0 when it cannot say.
    static uint64_t df_estimate(const std::string& dir, int slots, int chunk, int quant = 0);

    // ---- prefill API (owner: the prefill agent; gpu_prefill.cu) --------------------------------------------

    // Queues the forward pass of the n prompt tokens `tokens` at positions pos0 .. pos0+n-1 (pos0 is 0 after
    // reset(), or the number of tokens already in the cache). Same state afterwards as n calls to step(); with
    // head_on_last the final norm and the LM head run on the last token only and its logits are left on the
    // device. Returns false if a launch failed or the positions do not fit n_ctx(); not synchronised.
    bool prefill(const int32_t* tokens, int n, int pos0, bool head_on_last);

    // How prefill() computes. planes = 0: token by token on the fp32 GEMV path (the reference arithmetic, ~100
    // tok/s). planes = 1..3: batched, `chunk` tokens per pass, weight products on the int8 tensor cores with the
    // activations rounded to that many int8 planes (limbo/cuda/gemm.hpp): 3 is fp32-like (the default: each
    // product is within 3e-7 of the exact one), 2 resolves a block's amplitude to 1 part in 16192 (about 1.5x
    // faster, opt-in: it moves router near-ties more often), 1 is llama.cpp's own Q8 activations (not parity
    // safe). chunk is clamped to [16, 4096], default 1024; the batched scratch (~0.3 MB per chunk token) is
    // allocated at the first batched prefill. Returns false for an out-of-range value.
    bool set_prefill(int planes, int chunk);
    int prefill_planes() const;
    int prefill_chunk() const;
    // Whether the batched pass runs its independent work (delta-net alpha/beta/z, the shared expert) on the second
    // stream beside the critical path (default on). Off: one stream, the order of kernels is the same.
    void set_prefill_streams(bool on);

    // ---- shared: results and debug (owner: orchestrator; gpu.cu) -------------------------------------------

    // The greedy token of the last step's logits (lowest index on a tie). Synchronises.
    int32_t argmax();
    // Copies n_vocab() logits of the last step that ran the head to `out`. Synchronises.
    bool read_logits(float* out);
    void sync();

    // Debug: when set, step() copies the residual stream after every layer into (*t)[layer] (n_embd floats),
    // synchronising once per layer; the batched prefill does the same for the last token of each chunk. Pass
    // nullptr to switch it off.
    void set_layer_trace(std::vector<std::vector<float>>* t);
    // Debug: when set, step() records per layer the routed experts as 2*n_used + n_expert floats: the ids by
    // descending router weight, their weights, then all the router logits. Synchronises once per layer.
    void set_moe_trace(std::vector<std::vector<float>>* t);

    // Debug: when set, the batched prefill records CUDA events around its phases and adds the milliseconds of
    // each phase (keyed by name) into *t; the pass is synchronised once per chunk. Pass nullptr to switch it off.
    void set_prefill_profile(std::vector<std::pair<std::string, double>>* t);
    // Debug: the same for step(): each phase's milliseconds, summed over the steps run while it is set (eager
    // launches, synchronised once per step; the graph is bypassed).
    void set_decode_profile(std::vector<std::pair<std::string, double>>* t);

    // Measurements of open().
    double load_ms() const;               // wall time of the weight upload
    uint64_t weight_bytes() const;        // bytes of weights resident on the device
    uint64_t vram_bytes() const;          // device bytes this object allocated: weights, KV, state, scratch (not the CUDA context)
    uint64_t free_before_bytes() const;   // cudaMemGetInfo free at the start of open()
    uint64_t free_after_bytes() const;    // ... and when it returned

    // ---- state API: snapshots, rewind, slots, pages (owner: the tape agent; gpu_state.cu, gpu_pages.cu) ----------------
    //
    // What a conversation is made of: K/V on the 10 full-attention layers, which is positional (the row of
    // position p depends on tokens 0..p only and is written by whichever pass reads p), and on the 30 delta-net
    // layers a recurrent state (S, 32 heads x 128 x 128 fp32) plus a 3-row conv history, which is the whole
    // past folded into one value and cannot be rewound, only restored.
    //
    // Rewinding to position p therefore needs no K/V copy: restore the recurrent state that was snapshotted at
    // p and prefill/step from p on; the rows past p are overwritten as they are read and nothing attends past
    // the position it is at. The rule: state at p = restore(the snapshot taken at p), then prefill(tokens[p..]).
    // A snapshot is a device-to-device copy on the engine's stream, so it is ordered after every pass queued
    // before it and before every pass queued after it (no synchronisation, ~0.3 ms).

    // Bytes of one snapshot: every delta-net layer's S and conv history.
    uint64_t snapshot_bytes() const;

    // Makes `want` snapshot slots exist (more allocates, fewer frees the tail, 0 frees all; existing slots keep
    // their contents), one VRAM block per slot, and returns how many exist afterwards: the allocation stops at
    // the first cudaMalloc that fails or that would leave less than keep_free_bytes free on the card.
    int set_snapshot_slots(int want, uint64_t keep_free_bytes = 0);
    int snapshot_slots() const;

    // slot := the current home's recurrent state, as of every pass queued so far. False for a bad slot or a
    // failed copy.
    bool snapshot(int slot);
    // The current home's recurrent state := slot. The next pass must be at the position the snapshot was taken
    // at (the caller knows it; the engine does not track positions). False for a bad or never-written slot.
    bool restore(int slot);

    // Slots: one conversation each (page table + recurrent state + graphs), switching is a pointer select with no copy. Slot 0
    // is the one open() allocated; set_slots(n) makes slots 1..n-1 exist (empty, as after reset()), stopping at the first
    // allocation that fails or would leave less than keep_free_bytes free, and returns how many exist. reset(), snapshot(),
    // restore() and the page calls below act on the current slot; the graphs are captured per slot. All slots share the pool.
    int set_slots(int want, uint64_t keep_free_bytes = 0);
    int slots() const;
    bool select_slot(int slot);  // false when it does not exist
    int slot() const;
    uint64_t slot_bytes() const;        // device bytes of one extra slot: its recurrent state (its K/V is the pool's)
    uint64_t state_vram_bytes() const;  // device bytes of the snapshot slots and the extra slots

    // Pages. slot_cut: the live slot's positions >= keep go back to the pool (a shared page only loses this holder). slot_reserve:
    // pages for its first n positions exist now (false: the pool is out; what it took stays). slot_share: the live slot, which must
    // be empty, becomes the first n positions of `from` (whole pages shared, the page n cuts in two copied); false when `from` does
    // not hold them or the pool has no page for the copy. slot_room(at): positions the live slot can still hold from `at` on.
    // release_slot: the pages of any slot go back (the slot's state is untouched). All of them act on the stream's order.
    void slot_cut(int keep);
    bool slot_reserve(int n_positions);
    bool slot_share(int from, int n_positions);
    int slot_room(int at) const;
    void release_slot(int slot);
    struct PoolStats {
        int pages = 0, used = 0, shared = 0;  // device pages: all, held by at least one slot, held by more than one
        int page_positions = 256;
        uint64_t page_bytes = 0;
        uint64_t copies = 0;  // pages copied since the start
    };
    PoolStats pool_stats() const;
    int slot_pages(int slot) const;        // pages the slot's table holds
    int page_of(int slot, int chunk) const;  // -1 when none (test hook)
    int page_refs(int page) const;

private:
    Gpu();
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace limbo::cuda
