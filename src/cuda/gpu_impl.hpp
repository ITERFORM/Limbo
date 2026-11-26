// Copyright (c) 2026 ITERFORM. All rights reserved.
#pragma once
// src/cuda/gpu_impl.hpp — Gpu's private state, shared by gpu.cu (load), gpu_decode.cu, gpu_prefill.cu and
// gpu_state.cu. Members are grouped by owner; add members only inside your own section.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "limbo/cpu_lane.hpp"
#include "limbo/cuda/dflash.hpp"
#include "limbo/cuda/gemm.hpp"
#include "limbo/cuda/gemv.hpp"
#include "limbo/cuda/gpu.hpp"
#include "limbo/kvpool.hpp"

namespace limbo::cuda {

namespace impl {

struct W {
    const uint8_t* p = nullptr;
    uint32_t type = 0;
};

// What the q4e feeder hands a layer's device wait (mapped, then copied to the device): the lane's jobs -> dn rows,
// and the PCIe share's.
struct Q4eCtl {
    static constexpr int kDma = 8;   // device-computed misses: the prefetched ones, then the PCIe share (at most 4)
    int32_t n_miss = 0, n_dma = 0;
    int32_t dma_idx[kDma] = {};
    int32_t miss_idx[256] = {};
    static constexpr int kHead = 2 + kDma;   // words before miss_idx
};

struct Layer {
    bool full = false;
    const float* attn_norm = nullptr;
    const float* post_norm = nullptr;
    // gated delta net
    W qkv, gate, out, alpha, beta;
    const float *conv = nullptr, *dtb = nullptr, *ssm_a = nullptr, *ssm_norm = nullptr;
    // full attention
    W q, k, v, o;
    const float *qn = nullptr, *kn = nullptr;
    // dense SwiGLU FFN (qwen35; the MoE tensors below stay empty)
    W fgate, fup, fdown;
    // MoE
    W router, ge, ue, de, sgate, sup, sdown;
    const float* shgate = nullptr;
    uint64_t ge_bytes = 0, de_bytes = 0;  // bytes of one expert in the gate/up and down stacks

    // ---- qwen4exp (the Flash-Next stack; src/step.cpp's hc / PLE / indexer recipes) -------------------
    // A hyper-connection read mixer: a per-stream grouped norm, the low-rank down/up pair and the inject row.
    struct Hc {
        const float* norm = nullptr;  // [hc_count][n_embd]
        W down, up, inject;           // inject empty for the head's mixer
    };
    Hc hca, hcf;          // hc_attn, hc_ffn
    bool gdn_sig = false; // this GDN's output gate is sigmoid(z), not silu(z)
    W iq, ik;             // the sparse-attention indexer's projections (BF16) and norms
    const float *iqn = nullptr, *ikn = nullptr;
    // The PLE n-gram hash block (layer 1): the dense projections of the table's rows and the conv chain.
    W ple_key, ple_value;
    const float *ple_gk = nullptr, *ple_gq = nullptr, *ple_gc = nullptr, *ple_conv = nullptr;
    // The routed experts stay in the file's own mapping (23.4 GiB do not fit the card): p is HOST memory,
    // one expert's slices are p + ex * ge_bytes / de_bytes, computed by the CPU lane.
    W geh, ueh, deh;
};

struct Cfg {
    int n_layer_main = 0, n_embd = 0, n_head = 0, n_head_kv = 0, hd = 0, rope_dim = 0, n_vocab = 0;
    int n_expert = 0, n_used = 0, n_ff_exp = 0, n_ff_shexp = 0;
    int n_ff = 0;  // the dense FFN width (qwen35); n_expert == 0 means every layer's FFN is dense
    int n_k_head = 0, n_v_head = 0, gdn_head = 0, conv_k = 0, qkv_w = 0, dt_rank = 0;
    double eps = 1e-6, rope_base = 1e7;
};

}  // namespace impl

using impl::Cfg;
using impl::Layer;
using impl::W;
using impl::Q4eCtl;

struct Gpu::Impl {
    // ---- shared: config, streams, weights, per-layer state, step scratch (owner: orchestrator) ----
    Cfg c;
    int n_ctx = 0;  // positions a slot can address (its page table has n_chunks entries)
    cudaStream_t stream = nullptr;
    // A second stream for the batched prefill: the delta net's z product runs beside its latency-bound scan chain.
    cudaStream_t stream2 = nullptr;
    cudaEvent_t ev_a = nullptr, ev_b = nullptr;

    uint8_t* arena = nullptr;
    uint64_t arena_bytes = 0;
    std::vector<Layer> L;
    W embd, head;
    const float* out_norm = nullptr;
    // The token embedding is read one row per token: by default it stays in mapped pinned host memory that the kernels read over
    // the bus (embd.p is its device alias), which leaves its bytes on the card to the K/V cache. LIMBO_EMBD_VRAM=1 keeps it here.
    void* embd_host = nullptr;
    bool embd_in_vram = false;

    // Per-layer state of the live slot; null where a layer does not have it. kc / vc are the f16 pool's planes.
    std::vector<float*> S, hist;
    std::vector<__half*> kc, vc;
    std::vector<void*> owned;  // every cudaMalloc other than the arena and the scratch
    uint64_t owned_bytes = 0;

    // Scratch, carved from one allocation.
    float* scratch = nullptr;
    float *x, *xn, *qkv, *conv, *qk, *alpha, *beta, *o, *z, *gated, *qfull, *kbuf, *vbuf, *qbuf, *attn, *rlog, *post,
        *gbuf, *ubuf, *hbuf, *dn, *gs, *us, *hs, *shared, *sg, *wts, *rn, *logits;
    int32_t *ids = nullptr, *tok = nullptr;

    // ---- prefill (owner: the prefill agent) -------------------------------------------------------------
    // Batched prefill (lazy): scratch for up to pf.cap tokens, and how to run it.
    struct Prefill {
        int cap = 0;
        void* blob = nullptr;
        uint64_t bytes = 0;
        float *x, *xn, *qkv, *conv, *qk, *alpha, *beta, *o, *z, *gated, *qfull, *kbuf, *vbuf, *qbuf, *attn, *rlog, *post,
            *wts, *h, *dn, *gs, *us, *hs, *shared, *sg, *xs, *xsum, *rope;
        int32_t *ids, *tok;
        int8_t* xq;
        float *aml, *kwin, *vwin;  // Q8 cache reads: the running (max, sum) of the attention windows, and the f16 window of K and V
        MoeLists L;
    } pf;
    bool pf_able = false;
    int pf_planes = 3, pf_chunk = 1024;  // the exact mode by default (PLAN.md 3.3: nothing lossy unless asked)
    bool pf_overlap = true;              // independent work on stream2 (set_prefill_streams)

    // ---- shared: traces, profiles, accounting -------------------------------------------------------------
    std::vector<std::vector<float>>* trace = nullptr;
    std::vector<std::vector<float>>* moe_trace = nullptr;
    std::vector<std::pair<std::string, double>>* pf_prof = nullptr;
    std::vector<std::pair<std::string, double>>* dec_prof = nullptr;
    double load_ms = 0;
    uint64_t weight_bytes = 0, vram_bytes = 0, free_before = 0, free_after = 0;

    // ---- qwen4exp (owner: the q4e agent; gpu_q4e.cu) ------------------------------------------------------
    // The Flash-Next stack: a hyper-connection residual of `hc` streams, GDN / sparse-attention mixers, the
    // PLE hashed n-gram block and the MoE whose routed experts run on the CPU lane (the file's own mapping;
    // the model must outlive the Gpu for those host reads). Decode only: no batched prefill, no graph capture
    // (the PLE rows and the lane round trips visit the host inside the pass).
    bool q4e = false;
    struct Q4e {
        int hc = 0, rank = 0;       // residual streams, the mixers' low rank
        int np = 0;                 // one PLE table row's width (n_embd_per_layer)
        int n_heads = 0;            // (ngram - 1) * heads_per_ngram table rows per token
        int kern = 0, ngram = 0;    // the PLE conv's taps and its dilation (the n-gram size)
        int64_t eos = -1;
        std::vector<uint64_t> mult, off, vocab;  // the hash: per-position multipliers, per-head ranges
        Layer::Hc head_hc;          // output_hc: the head's read mixer (the output norm of the other recipes)
        const uint8_t* table = nullptr;   // per_layer_token_embd, mapped, never uploaded
        // The shards as files: the experts and the PLE rows are READ (pread) rather than faulted through the mapping,
        // and their pages dropped from the page cache once copied, so the start does not grow this process's
        // working set by the 23 GiB it reads and push the desktop into swap (Strata: read unbuffered, give back).
        struct Shard {
            const uint8_t* base = nullptr;
            uint64_t size = 0;
            int fd = -1;
        };
        std::vector<Shard> shards;
        int table_fd = -1;          // the PLE table's shard, and the table's offset in it
        uint64_t table_off = 0;
        // the file and offset of a pointer into a mapping; false when it is not in one
        bool file_of(const uint8_t* p, int* fd, uint64_t* off) const {
            for (const Shard& s : shards)
                if (p >= s.base && p < s.base + s.size) {
                    *fd = s.fd;
                    *off = uint64_t(p - s.base);
                    return true;
                }
            return false;
        }
        uint32_t table_type = 0;
        uint64_t row_bytes = 0;     // one table row (np elements)
        // The PLE rows already read, by row: the drive serves ~8.5k scattered reads a second on this box (any queue
        // depth, mmap or O_DIRECT or AIO alike), and the page cache spends 4 KiB on each 90-byte row. Slots are
        // seqlocked (tag = row << 1 | busy), 4-way probed, overwritten when full. LIMBO_Q4E_PLE_CACHE_MB, 0 = off.
        struct PleSlot {
            std::atomic<uint64_t> tag{~uint64_t(0)};
            uint8_t row[96];
        };
        std::unique_ptr<PleSlot[]> ple_cache;
        uint64_t ple_mask = 0;
        std::atomic<uint64_t> ple_hits{0}, ple_miss{0};
        int idx_kpool = 0, idx_top_k = 0;  // the sparse selection's block size and budget
        int idx_heads = 4;                 // the indexer's query heads (of 128)
        // Scratch, one allocation carved at open.
        float* scratch = nullptr;
        float *st = nullptr, *xn = nullptr, *lo = nullptr, *gate = nullptr, *mixed = nullptr, *inj = nullptr, *out = nullptr,
              *emb = nullptr, *pkey = nullptr, *pval = nullptr, *pgated = nullptr, *pnrm = nullptr, *pconv = nullptr,
              *phist = nullptr;
        // The routed experts: one layer's 10 rows in host memory per step, quantized and dotted by the lane.
        std::unique_ptr<cpu::Lane> lane;
        float *host_x = nullptr, *host_dn = nullptr;  // mapped [n_embd], [n_used * n_embd]
        float *host_x_d = nullptr, *host_dn_d = nullptr;
        std::vector<int32_t> tok_hist;  // the PLE window's predecessors

        // The resident expert cache (docs/flash-next.md 6.3/6.4). One slab per layer; a slot holds one expert's
        // gate, up and down bytes exactly as the file stores them, so a resident expert is computed by the same
        // fp32 gemv kernels as the dense weights. Admission is decayed routing frequency; a miss still goes to
        // the CPU lane. It changes where an expert is computed, never what: both paths multiply the file's own
        // dequantized weights into the same fp32 FFN input, and a resident expert stops touching the drive.
        struct Slab {
            uint8_t* base = nullptr;          // VRAM: n_slots * blob
            int n_slots = 0;
            uint64_t blob = 0, ge = 0, de = 0;  // gate | up | down, contiguous
            int32_t* slot_of = nullptr;       // [n_expert] -> slot, -1 = not resident: this layer's row of slot_map
            std::vector<int32_t> expert_of;   // [n_slots] -> expert, -1 = free
            int used = 0;                     // slots ever handed out
            // Tier 2, the same shape in pinned host memory. The CPU lane reads a resident expert from here at
            // DDR speed (a file mapping re-faults and evicts: 19 vs 42.8 GB/s measured by the donor), and a
            // VRAM admission copies from here over PCIe instead of from the drive.
            uint8_t* ram = nullptr;           // ram_slots * blob, pinned and cacheable
            int ram_slots = 0, ram_used = 0;
            std::vector<int32_t> ram_slot_of;  // [n_expert] -> ram slot, -1 = not in RAM
            std::vector<int32_t> ram_expert_of;
            std::vector<uint8_t> ram_ready;    // published by the filler once the bytes are in
            std::vector<uint8_t> ram_pending;  // a promotion is reading it: not evictable (the lane may still read it)
            std::vector<uint8_t*> ram_ptr;     // [ram_slots + 1] slot -> its bytes (slots trade buffers with the spare)
            int ram_spare = -1;                // the free buffer a promotion's victim is copied down into
            std::vector<uint32_t> stamp;       // [n_slots] the admission epoch that last read the slot
            // one promotion in flight a layer, on the copy stream: e into VRAM slot vs, its victim v down into spare
            bool pr_on = false;
            int32_t pr_e = -1, pr_v = -1, pr_vs = -1, pr_rt = -1;
        };
        // One expert's three matrices wherever they happen to live: the VRAM slab, the RAM tier, or the file's
        // mapping. A RAM-resident blob is contiguous (gate | up | down); the mapping's three tensors are not.
        struct Blob {
            const uint8_t *g = nullptr, *u = nullptr, *d = nullptr;
        };
        std::vector<Slab> slab;               // [n_layer_main]
        // RAM slots whose bytes were just copied into VRAM: freed at the next admission for that layer, by
        // which time the step's own per-layer stream sync has carried the copy. RAM then holds the exact
        // complement of VRAM instead of mirroring it, which is the difference between 14 and 24 GiB of cover.
        std::vector<std::vector<int32_t>> ram_release;
        std::vector<float> eheat;             // [n_layer_main][n_expert]: decayed lookups, the admission score
        // Undecayed routing counts across runs, kept on disk (profile_path): the next start puts each layer's most
        // routed experts in VRAM instead of experts 0..n, so the first replies do not climb from ~45 % hits.
        std::vector<float> lt;
        std::string profile_path;
        void profile_save() const;

        float decay = 0.998f;                 // per token: half-life ~350 (flash-next.md 6.4)
        // The prompt pass's stage is lent from the slab: the last lend_k slots of every layer go to RAM for a prompt and
        // hold the staged experts (layer l stages into the ranges lend_rng[l & 1]); the rebalance refills them after.
        int lend_k = 0;
        bool lend_single = false;   // one stage buffer: the next layer's waits for this one's MoE
        std::vector<std::pair<uint8_t*, uint64_t>> lend_rng[2];
        // Every slab's slot_of in one mapped pinned table [n_layer_main][n_expert]: the host's book IS what the
        // device's split reads, so the GPU sorts hits from misses after the top-k without a host round trip.
        int32_t* slot_map = nullptr;
        int32_t* slot_map_d = nullptr;        // its device alias
        // The split's output: d_sel [pairs] device, the slot of each selection (-1 = miss; the hits' gemv_experts
        // ids), and split_h, mapped [2][pairs] (the ids, then the slots), what the host's admission follows.
        int32_t* d_sel = nullptr;
        int32_t* split_h = nullptr;
        int32_t* split_d = nullptr;
        // The doorbell: a layer's split rings bell[l] = seq once the selections and the FFN input are in mapped memory;
        // the feeder (the step thread) admits, runs the lane, writes ctl and the lane's outputs and raises done[l] = seq;
        // the device's wait copies ctl and the layer goes on. The host never syncs inside a step.
        // bell/done: 2 * layer + group; pred: the layer whose experts a prediction names
        static constexpr int kBell = 0, kDone = 128, kPle = 256, kAbort = 258, kErr = 259, kPred = 260, kFlags = 388;
        uint32_t* flag_h = nullptr;   // mapped [kFlags]
        uint32_t* flag_d = nullptr;
        Q4eCtl* ctl_h[2] = {};        // mapped, one a window group
        Q4eCtl* ctl_d[2] = {};
        Q4eCtl* ctl_dev[2] = {};      // the wait's device copy
        int split_min = 0;            // LIMBO_Q4E_SPLIT_MIN: a window of this many rows or more runs as two overlapped groups (0 off:
                                      // the dense half costs a group nearly what it costs the window, measured w3 73 vs 65.6 ms)
        uint32_t* d_seq = nullptr;    // device: the running MoE call's sequence number
        uint32_t seq = 0;
        float* emb_h[2] = {nullptr, nullptr};    // mapped PLE rows, by seq parity
        float* emb_hd[2] = {nullptr, nullptr};
        uint64_t slab_bytes = 0, ram_bytes = 0, cache_hits = 0, cache_lookups = 0;
        uint64_t miss_ram = 0, miss_file = 0;   // where the lane's experts came from (bench)
        uint32_t fill_mbps = 0;                  // 0 = unpaced; the ring is staging, so it fills at drive speed
        // The grouped expert GEMV over the hits. ON: interleaved A/B on `code`, warm, 14.25/14.06 tok/s
        // grouped against 13.61 per-expert = +4.0 %, gate 16/16 in every run. It is worth what the tail path
        // is worth — with n_in % 256 == 0 it was neutral-to-negative, because the DOWN (n_in = n_ff_exp = 640)
        // could not group and the gate and up alone did not pay for the extra memcpy and scatter. It is also
        // the shape a verify window needs (its slots are the rows, x_stride the row's input).
        bool grouped_probe = true;               // LIMBO_Q4E_GROUPED=0 falls back to the per-expert loop
        // The RAM tier is filled by one background thread out of the file's mapping, so a SATA read never lands
        // on the step's critical path: the step only ever reads a slot the filler has published.
        struct FillReq {
            int32_t l, e, slot;
        };
        static constexpr int kFillRing = 4096;
        std::vector<FillReq> fill;
        std::atomic<uint32_t> fill_head{0}, fill_tail{0};
        std::atomic<bool> fill_stop{false};
        std::thread filler;
        cudaStream_t cstream = nullptr;          // promotions and the prompt pass's stage
        // The PCIe share of a decode layer's misses (docs/donors/strata.md, "PCIe share"): with dma_min or more misses,
        // one RAM-held expert is copied to the device on its own stream and computed there while the lane takes the rest.
        cudaStream_t dma_stream = nullptr;
        cudaEvent_t dma_ev = nullptr;
        uint8_t* dma_buf = nullptr;              // kDma blobs of the largest layer
        uint64_t dma_blob = 0;
        float* dn_dma = nullptr;                 // [kDma][n_embd]
        float *dma_g = nullptr, *dma_u = nullptr, *dma_h = nullptr;   // [kDma][n_ff_exp]
        int32_t* dev_iota = nullptr;
        static constexpr int kDma = Q4eCtl::kDma, kDmaShare = 4;
        int dma_min = 3;
        // The prefetch (LIMBO_Q4E_PREFETCH=n): layer l+1's router on layer l's MoE input names the experts l+1 will likely
        // pick (6.85 of 10 measured); the first n of them that are neither resident nor promoting are copied over the
        // bus while layer l runs, into one of two staging sets (by layer parity). A miss of l+1 that is staged is computed
        // on the device beside the lane instead of read by it.
        static constexpr int kPf = 8;
        int pf_n = 0;
        uint8_t* pf_buf = nullptr;               // [2][kPf] blobs of dma_blob
        cudaStream_t pf_stream = nullptr;
        cudaEvent_t pf_ev[2] = {};
        int32_t pf_ex[2][kPf] = {};
        int pf_cnt[2] = {}, pf_layer[2] = {-1, -1};
        uint32_t pf_seq[2] = {};
        int32_t *pred_h = nullptr, *pred_d = nullptr;   // mapped [n_layer][16]: the predicted ids, rank order
        int32_t* pred_top = nullptr;                    // device [16]
        float *pred_rlog = nullptr, *pred_w = nullptr;
        int32_t *pf_slot_h = nullptr, *pf_slot_d = nullptr;   // mapped [kPf]: the staged slot of each device-computed miss
        uint64_t pf_issued = 0, pf_used = 0, pf_steps = 0;
        // Route-tail skip (LIMBO_Q4E_TAIL_SKIP=r, off by default: it changes the output): a miss every row routes at rank
        // r or lower (0 = the top choice) is not computed, its share of the mix left out (the donor's default, r = 7).
        int tail_skip = 0;
        uint64_t skipped = 0;
        float pcie_share = 0.34f;                // of a layer's misses (RAM-held); the link is ~6.5 GB/s, 325 us an expert (LIMBO_Q4E_PCIE_SHARE)
        uint64_t dma_n = 0;
        std::vector<cudaEvent_t> pr_ev;          // [n_layer_main]
        uint32_t epoch = 0;                      // one per MoE call (a token, or a window)
        int pr_budget = 0;                       // promotions left this token
        uint64_t promotions = 0;
    } q;
    void q4e_pr_finish(int l);
    bool q4e_swap(int l, int32_t e, int vs, bool sync_book);
    void q4e_pr_drain();
    bool q4e_cache_warm(std::string* err);
    bool q4e_lend();
    // The qwen4exp step and its pieces (gpu_q4e.cu).
    bool step_q4e(int32_t token, int pos, bool head);
    // The verify window (docs/flash-next.md 6.6a): T rows through all 48 layers in one pass, every dense weight
    // read once for all of them, the delta-net state advanced only by commit(). Its own workspace, so a window
    // can never change what a single token does.
    static constexpr int kQ4eRows = 8;
    // The (row, used) pairs one window's MoE can carry: a row's selected experts, for every row of the window.
    static constexpr int kQ4ePairs = kQ4eRows * 32;
    struct Q4eRows {
        bool ready = false;
        int n_jobs = 0;   // the last window MoE's lane jobs (misses grouped by expert)
        cpu::ExpertJob jobs[2][kQ4ePairs];   // a group's jobs, alive from submit to finish (the lane keeps the pointer)
        int n_miss[2] = {};
        int pending = 0;   // the length of the last write_back = false window, what commit() may take a prefix of
        int pos0 = 0;      // its first position
        int last_T = 0;    // the rows of the last window (or 1: a plain row) whose residuals st holds
        float *st = nullptr, *xn = nullptr, *lo = nullptr, *gate = nullptr, *mixed = nullptr, *inj = nullptr,
              *out = nullptr, *x = nullptr, *emb = nullptr, *pkey = nullptr, *pval = nullptr, *pgated = nullptr, *pnrm = nullptr,
              *pconv = nullptr, *phist = nullptr;
        float *qkv = nullptr, *conv = nullptr, *qk = nullptr, *alpha = nullptr, *beta = nullptr, *o = nullptr,
              *gated = nullptr, *z = nullptr, *attn = nullptr, *dn = nullptr, *rlog = nullptr, *wts = nullptr;
        // The attention half of a full layer, per row (rung 1's window carries the same four).
        float *qfull = nullptr, *kbuf = nullptr, *vbuf = nullptr, *qbuf = nullptr;
        float *gbuf = nullptr, *ubuf = nullptr, *hbuf = nullptr, *gs = nullptr, *us = nullptr, *hs = nullptr,
              *shared = nullptr, *sg = nullptr, *logits = nullptr;
        int32_t* ids = nullptr;   // [T][n_used]
        // Allocated by ensure_q4e_rows, not carved: the rows attention reads its position from device memory (the
        // single-row step passes d_args + 1), embed_rows takes a token ARRAY where the single-row step passes the
        // one int2, and the split attention needs its own partials for kQ4eRows rows.
        int32_t* d_args = nullptr;   // [2 groups][token, position]
        int32_t* d_tok = nullptr;
        float* attn_scr = nullptr;
        float* Stmp = nullptr;       // one layer's delta-net state after the first group's rows (a split window)
        // alpha and beta per LAYER, [n_layer_main][kQ4eRows][alpha | beta] with row stride 2*dt_rank. They are
        // per layer because the window's gdn_coef_rows transforms them in place and the window walks every layer:
        // a single shared pair would leave commit -- which replays layer by layer after the window has finished --
        // using the LAST layer's coefficients for every layer.
        float* ab = nullptr;
        // The GDN conv's input per LAYER, [n_layer_main][kQ4eRows][qkv_w]. Same reason as `ab`: the window walks
        // every GDN layer and each one overwrites its qkv projection, so commit would otherwise replay every
        // layer with the last layer's conv input. (`pnrm` needs no such slot: only the one PLE layer writes it.)
        float* qkvk = nullptr;
        // The MoE half's batch state (gpu_q4e.cu q4e_moe_rows). The router and the top-k run for all T rows on the
        // device, and then ONE round trip brings back the ids and the FFN input tile -- so a window's MoE costs one
        // sync a layer where T sequential steps cost T, which is the whole reason to batch it. The per-row arrays
        // are [T][n_used], a row's hits packed at that row's own offset, so each row's grouped expert launch reads
        // a contiguous id list exactly as the single-row q4e_moe does (gemv_experts' x_stride is per *selection*,
        // not per pair, so the pairs of one row are what can share a launch).
        float* post = nullptr;         // mapped [T][n_embd]: the FFN input tile, the CPU lane's own input
        float* post_d = nullptr;
        int32_t* miss_row = nullptr;   // pinned [T][n_used]: the tile row each lane job reads (its act_row)
        float* dn_h = nullptr;         // mapped [T][n_used][n_embd]: the lane's outputs, packed by job
        float* dn_hd = nullptr;
        int32_t* d_pair = nullptr;
        int32_t* d_am = nullptr;       // device [T]: the rows' greedy tokens, the argmax's staging
    } qrw;
    bool step_q4e_rows(const int32_t* tokens, int T, int pos0, bool write_back);
    bool launch_q4e_commit(int n);
    bool ensure_q4e_rows();
    void q4e_ple_rows(int32_t token, int pos, float* emb, bool ahead = true);   // ahead: a readahead per row (one caller)
    bool q4e_hc_read(const Layer::Hc& m, const float* st);
    bool q4e_hc_read_rows(const Layer::Hc& m, const float* st, int T, int r0 = 0);
    bool q4e_gdn(int l, const float* mixed, float* out);
    bool q4e_attn(int l, const float* mixed, float* out);
    // QSA on the device (src/cuda/q4e_qsa.cu): every attention layer's indexer state, kept for every position read.
    struct QsaSt {
        bool ready = false;
        std::vector<__half*> ring, pool;   // [n_layer_main]: [16][128] raw keys, [n_ctx / kpool + 1][128] pooled
        float *kraw = nullptr, *qraw = nullptr, *qn = nullptr, *scores = nullptr, *scr = nullptr;
        int32_t* sel = nullptr;
        size_t stride = 0;
        std::vector<void*> allocs;
    } qsa;
    int qsa_pos = 0;   // the decode step's position (host), for the selection's grid
    // The prompt pass (src/cuda/gpu_q4e_pf.cu): a piece of up to kQ4ePfMax tokens, layer-major, micro-batches of kQ4eMb.
    static constexpr int kQ4ePfMax = 4096, kQ4eMb = 512, kQsaG = 64, kSplitK = 8;
    struct Q4ePf {
        bool ready = false, failed = false;
        float *st = nullptr, *xn = nullptr, *lo = nullptr, *gate = nullptr, *mixed = nullptr, *inj = nullptr, *out = nullptr,
              *x = nullptr, *emb = nullptr, *pkey = nullptr, *pval = nullptr, *pgated = nullptr, *pnrm = nullptr,
              *pconv = nullptr, *qfull = nullptr, *kbuf = nullptr, *vbuf = nullptr, *qbuf = nullptr, *attn = nullptr,
              *qkv = nullptr, *conv = nullptr, *qk = nullptr, *al = nullptr, *be = nullptr, *o = nullptr, *z = nullptr,
              *gated = nullptr, *rlog = nullptr, *wts = nullptr, *h = nullptr, *dn = nullptr, *gs = nullptr, *us = nullptr,
              *hs = nullptr, *shared = nullptr, *sg = nullptr, *xs = nullptr, *xsum = nullptr;
        float *aml = nullptr, *kwin = nullptr, *vwin = nullptr;   // a Q8 pool's dense rows: f16 windows of it and the running (max, sum)
        int32_t *posv = nullptr, *tok = nullptr, *ids = nullptr;
        int8_t* xq = nullptr;
        const uint8_t** eptr_d[2] = {nullptr, nullptr};  // expert -> blob, device
        const uint8_t** eptr_h[2] = {nullptr, nullptr};  // pinned
        cudaEvent_t staged[2] = {nullptr, nullptr}, done[2] = {nullptr, nullptr};
        float* emb_h[2] = {nullptr, nullptr};        // pinned [kQ4ePfMax][ple width]: a piece's PLE rows, two pieces
        // The PLE rows of a piece on host threads (scattered table reads: many in flight keep the drive's queue full).
        // The next piece's start while this one computes; `key` says which piece a buffer holds.
        struct PleJob {
            std::vector<std::thread> th;
            int pos0 = -1, n = 0, buf = 0;
            void join() {
                for (std::thread& t : th)
                    if (t.joinable()) t.join();
                th.clear();
            }
            ~PleJob() { join(); }
        } ple_job;
        int ple_have_pos0 = -1, ple_have_n = 0, ple_have_buf = 0;   // a finished job's piece
        // the sparse attention's indexer for a micro-batch, and its selection in groups of kQsaG rows
        float *ikraw = nullptr, *iqraw = nullptr, *iqn = nullptr, *qscores = nullptr, *qscr = nullptr;
        float* splitk = nullptr;   // [kSplitK][kQ4eMb][512]: the split-K partials of a narrow product
        int32_t* qsel = nullptr;
        MoeLists lists;
        int32_t* cnt_h = nullptr;                    // pinned [n_layer_main][n_expert]: the prompt's routing counts
        int32_t* cnt_d = nullptr;                    // device, accumulated over the prompt's pieces
        bool counted = false;
    } qpf;
    void q4e_rebalance();   // the prompt's routing into the heat, then VRAM re-sorted by it (after a prompt pass)
    // The MTP layer as the draft source (src/cuda/gpu_q4e_mtp.cu).
    struct Q4eMtp {
        bool ready = false;
        int cap = 0, nd = 0, last_T = 0, n_exp = 0, n_used = 0, nff = 0, nsh = 0;
        W fc_e, fc_h, q, k, v, o, router, sh_gate, sh_up, sh_down, gu, dn;
        uint64_t gu_bytes = 0, dn_bytes = 0;
        const float *pre_e = nullptr, *pre_h = nullptr, *qn = nullptr, *kn = nullptr, *shgate = nullptr;
        Layer::Hc hca, hcf, mix;
        __half *kc = nullptr, *vc = nullptr;
        float *hin = nullptr, *hn = nullptr, *h = nullptr, *st = nullptr, *emb = nullptr, *en = nullptr, *ev = nullptr;
        float *qfull = nullptr, *kbuf = nullptr, *vbuf = nullptr, *qbuf = nullptr, *attn = nullptr, *out = nullptr, *scr = nullptr;
        float *rlog = nullptr, *wts = nullptr, *gub = nullptr, *hh = nullptr, *dnb = nullptr, *gs = nullptr, *us = nullptr,
              *hs = nullptr, *shared = nullptr, *sg = nullptr, *logits = nullptr;
        int32_t *ids = nullptr, *d_am = nullptr, *d_tok = nullptr, *d_pos = nullptr;
        float* d_pr = nullptr;
        std::vector<void*> allocs;
    } q4mtp;
    bool q4e_mtp_load(const model::Model& m);
    bool q4e_mtp_rows(const int32_t* tokens, int T, int q0, bool h_from_mtp);
    bool q4e_mtp_front(const float* R_rows, const int32_t* next_tokens, int n, int cell0);
    bool q4e_mtp_prompt(const float* R_rows, const int32_t* next_tokens, int n, int cell0);
    bool q4e_mtp_argmax(int32_t* out, int T);
    bool q4e_mtp_argmax_p(int32_t* out, float* prob, int T);
    bool ensure_q4e_pf();
    bool q4e_mm(const W& w, const float* x, int xs, float* y, int ys, int n_out, int n_in, int T);
    bool q4e_hc_read_pf(const Layer::Hc& m, const float* st, int T);
    bool q4e_stage_layer(int l, int b);
    bool q4e_moe_pf(int l, int T, int b);
    bool prefill_q4e(const int32_t* tokens, int n, int pos0, const int32_t* next = nullptr, int next_n = 0);
    void q4e_ple_start(const int32_t* tokens, int n, int pos0, int buf);
    bool ensure_qsa();
    // The indexer for T rows of `mixed` (rows `x_stride` apart) at positions pos0 (host) / pos_dev; when any row has more
    // complete blocks than the budget, the selection and the attention over it into `attn` (*sparse = true); otherwise
    // the caller runs the plain attention.
    bool q4e_qsa(int l, const float* mixed, int x_stride, int T, int pos0, const int* pos_dev, const float* qbuf,
                 const float* qfull, float* attn, bool* sparse);
    bool q4e_moe(int l);
    // The expert cache (gpu_q4e.cu): open the slabs out of `budget` bytes, decay the heat, admit this step's
    // misses. `is_hit` is filled per selected expert in top-k order.
    bool q4e_cache_open(uint64_t vram_budget, uint64_t ram_budget, std::string* err);
    void q4e_cache_close();
    void q4e_fill_loop();
    void q4e_cache_decay();
    void q4e_cache_admit(int l, const int32_t* sel, const int32_t* dslot, int n_used, bool* is_hit, Q4e::Blob* src, bool may_promote = true);
    bool q4e_bell(int idx, int l, const int32_t* ids, int n, const float* x, float* xh_d, int nx, int pair0);
    bool q4e_wait(int flag, const void* src_d, void* dst, int nw);
    bool q4e_bell_wait(int idx);
    void q4e_prefetch(int l1);
    // A full-attention layer's K/V write and dense attention over the pool's format (f16, or Q8: int8 rows, a half scale a
    // block of 32; Strata's 128K layout).
    void q4e_prep(int l, const float* qfull, const float* kbuf, const float* vsrc, const float* qn, const float* kn, float* qbuf,
                  const int* pos_dev, int T);
    bool q4e_attend(int l, const float* qbuf, const float* qfull, float* attn, float* scr, const int* pos_dev, int T);
    void q4e_done(int idx);
    void q4e_abort();
    bool q4e_end();
    void q4e_split_book(int l, int n, int pair0 = 0);
    void q4e_pr_poll(int l);
    bool q4e_moe_serve(int l);
    bool q4e_rows_submit(int l, int r0, int n, int g, bool last);
    bool q4e_rows_finish(int l, int g, bool last);
    bool q4e_rows_front(int l, int r0, int n, int g, int T, int pos0, bool write_back, bool split, int pb);
    bool q4e_rows_back(int l, int r0, int n, int g);
    int32_t q4e_ram_fill(int l, int32_t e, const int32_t* held, int nheld);

    ~Impl() {
        q4e_cache_close();
        for (void* p : q4mtp.allocs) cudaFree(p);
        if (stream) cudaStreamSynchronize(stream);
        for (void* p : qsa.allocs) cudaFree(p);
        if (pf.blob) cudaFree(pf.blob);
        for (cudaGraphExec_t& g : gexec)
            if (g) cudaGraphExecDestroy(g);
        for (cudaEvent_t& e : pin_ev)
            if (e) cudaEventDestroy(e);
        if (pin) cudaFreeHost(pin);
        for (cudaEvent_t& e : pt_ev)
            if (e) cudaEventDestroy(e);
        if (pt_pin) cudaFreeHost(pt_pin);
        for (void* p : owned) cudaFree(p);
        if (scratch) cudaFree(scratch);
        if (q.scratch) cudaFree(q.scratch);
        if (q.host_x) cudaFreeHost(q.host_x);
        if (q.host_dn) cudaFreeHost(q.host_dn);
        if (q.split_h) cudaFreeHost(q.split_h);
        if (q.flag_h) cudaFreeHost(q.flag_h);
        if (q.ctl_h[0]) cudaFreeHost(q.ctl_h[0]);
        if (q.pred_h) cudaFreeHost(q.pred_h);
        if (q.pf_slot_h) cudaFreeHost(q.pf_slot_h);
        for (float* p : q.emb_h)
            if (p) cudaFreeHost(p);
        if (q.slot_map) cudaFreeHost(q.slot_map);
        // The verify window's pinned MoE state (ensure_q4e_rows).
        if (qrw.dn_h) cudaFreeHost(qrw.dn_h);
        if (qrw.miss_row) cudaFreeHost(qrw.miss_row);
        if (qrw.post) cudaFreeHost(qrw.post);
        if (arena) cudaFree(arena);
        if (embd_host) cudaFreeHost(embd_host);
        if (stream2) cudaStreamSynchronize(stream2);
        if (ev_a) cudaEventDestroy(ev_a);
        if (ev_b) cudaEventDestroy(ev_b);
        if (stream2) frag_release(stream2);
        if (stream) frag_release(stream);
        if (stream2) cudaStreamDestroy(stream2);
        if (stream) cudaStreamDestroy(stream);
    }

    template <typename T>
    T* dalloc(size_t n) {
        void* p = nullptr;
        if (cudaMalloc(&p, n * sizeof(T)) != cudaSuccess) return nullptr;
        owned.push_back(p);
        owned_bytes += n * sizeof(T);
        return static_cast<T*>(p);
    }

    bool gemv(const W& w, const float* in, float* out, int n_out, int n_in, bool acc = false) {
        return limbo::cuda::gemv(w.type, w.p, in, out, n_out, n_in, acc, stream);
    }

    // ---- decode (owner: the decode agent) ----------------------------------------------------------------
    // Decode as a captured graph: the token and the position come from device memory (d_args), staged through a
    // ring of pinned slots, so one graph replays every step. Variant 0 runs without the head, 1 with it.
    static constexpr int kRing = 256;
    int32_t* d_args = nullptr;
    int32_t* pin = nullptr;  // [kRing][2]
    cudaEvent_t pin_ev[kRing] = {};
    int ring = 0;
    cudaGraphExec_t gexec[2] = {nullptr, nullptr};
    int warm[2] = {0, 0};
    bool graphs = true;
    bool launch_step(bool head);
    bool step_run(int32_t token, int pos, bool head);  // Gpu::step without the drafter's ingest
    // Scratch the decode kernels need beyond open()'s: the split flash-decode's partials. Allocated on the first
    // step (before any capture) and freed with `owned`.
    float* attn_scr = nullptr;
    bool ensure_dec_scratch();
    // A side stream and events for the decode graph's parallel branches (the delta net's z product beside its scan
    // chain, the shared expert beside the router and the routed experts). Created with the scratch; the holder
    // frees them when the Impl goes.
    struct DecAux {
        cudaStream_t s2 = nullptr;
        cudaEvent_t ev[8] = {};
        DecAux() = default;
        DecAux(const DecAux&) = delete;
        DecAux& operator=(const DecAux&) = delete;
        ~DecAux() {
            for (cudaEvent_t& e : ev)
                if (e) cudaEventDestroy(e);
            if (s2) frag_release(s2);
            if (s2) cudaStreamDestroy(s2);
        }
    } da;

    // The T-row pass (gpu_rows.cu): activation buffers for up to dec::kMaxRows rows, the delta-net inputs kept per layer for
    // the commit, and one captured graph per (T, write_back), per commit length n and per MTP pass (slots 0..15, 16..23, 24..39).
    struct Rows {
        bool ready = false;
        int pending = 0;  // length of the last write_back = false window, what commit() may take a prefix of
        int pos0 = 0;     // its first position
        float *x = nullptr, *xn = nullptr, *o = nullptr, *z = nullptr, *gated = nullptr, *qfull = nullptr, *kbuf = nullptr,
              *vbuf = nullptr, *qbuf = nullptr, *attn = nullptr, *rlog = nullptr, *post = nullptr, *wts = nullptr, *hbuf = nullptr,
              *dn = nullptr, *gs = nullptr, *us = nullptr, *hs = nullptr, *shared = nullptr, *sg = nullptr, *rn = nullptr,
              *logits = nullptr, *qkv_k = nullptr, *cv_k = nullptr, *qkn_k = nullptr, *al_k = nullptr, *be_k = nullptr;
        float* attn_scr = nullptr;  // the split flash-decode's partials for kMaxRows rows (the decode step's attn_scr holds one)
        int32_t *ids = nullptr, *d_args = nullptr, *d_am = nullptr, *pin = nullptr;  // d_args = {pos0, tokens...}
        cudaGraphExec_t gx[40] = {};  // the live slot's graphs (select_slot swaps them with the slot's own)
        int warm[40] = {};
        Rows() = default;
        Rows(const Rows&) = delete;
        Rows& operator=(const Rows&) = delete;
        ~Rows() {
            for (cudaGraphExec_t& g : gx)
                if (g) cudaGraphExecDestroy(g);
            if (pin) cudaFreeHost(pin);
        }
    } rw;
    // The K/V pool (docs/paged-kv.md): per full-attention layer the planes of `pool_pages` pages, shared by every slot. Q8 / Q4: code
    // and scale planes of K and V (the Q4 planes are bytes of two codes, carried in the same int8_t pointers); F16 uses kc / vc.
    struct Kv {
        KvFormat fmt = KvFormat::F16;
        std::vector<int8_t*> kq, vq;
        std::vector<uint16_t*> ks, vs;
    } kv;
    std::unique_ptr<kv::PagePool> pool;  // the page tables of every slot, refcounts, copy on write (host side)
    int pool_pages = 0;
    static constexpr int kMaxSlots = 64;
    int n_chunks() const { return (n_ctx + kv::kPagePositions - 1) / kv::kPagePositions; }
    KvPlanes pool_planes(KvFormat f, int pages) const;  // one layer's planes for `pages` pages
    bool alloc_kv(KvFormat f, int pages);  // the pool's planes in format f, zeroed (kv.fmt = f, pool_pages = pages, the pool starts
                                           // empty); false with nothing left allocated
    void free_kv();                        // the pool's planes go (the pointers become null)
    void drop_graphs();                    // every captured decode and T-row graph of the live slot (they bake the pointers in)
    uint64_t kv_bytes(KvFormat f) const;   // the pool's bytes in format f at pool_pages
    uint64_t page_bytes(KvFormat f) const; // one page, every full-attention layer
    // Pages: the live slot's table is device memory (st.slots[st.cur].d_pt) that every attention launch reads. pages_for takes the
    // pages the positions [lo, hi) need (copying a shared one first), then uploads the table if it changed; all on the stream.
    bool pages_for(int lo, int hi);
    bool sync_pt();
    void page_copy(int dst, int src);
    uint64_t n_page_copies = 0;  // pages copied since the start (a prefix cut inside a page, a write into a shared page)
    const int* pt() const;  // the live slot's device table
    static constexpr int kPtRing = 16;
    int32_t* pt_pin = nullptr;  // [kPtRing][n_chunks + 1] staging for the table uploads
    cudaEvent_t pt_ev[kPtRing] = {};
    int pt_ring = 0;
    bool ensure_rows();
    bool launch_rows(int T, bool write_back);
    bool launch_commit(int n);
    bool launch_mtp(int T, bool h_from_mtp);
    bool rows_exec(int slot, int T, bool flag, int mode);  // mode 0: the window, 1: the commit, 2: the MTP pass

    // The model's own next-token head (the MTP block, blk.<mtp_layer>.nextn.*): uploaded by mtp_load(), with its own K/V cache.
    struct Mtp {
        bool ready = false;
        W eh, q, k, v, o, ge, ue, de, sgate, sup, sdown;
        W fgate, fup, fdown;  // the dense block's FFN (qwen35)
        const float *enorm = nullptr, *hnorm = nullptr, *shn = nullptr, *attn_norm = nullptr, *post_norm = nullptr, *qn = nullptr,
                    *kn = nullptr, *router = nullptr, *shgate = nullptr;  // router and shgate converted to F32 on the host
        uint64_t ge_bytes = 0, de_bytes = 0;
        int nd = 0;  // rows of the head the draft scores (the first nd ids of the vocabulary); 0 = all
        // The block's K/V holds `cap` positions of its own (relative to wherever the caller starts it: the attention depends on
        // position differences only), with its own V pitch.
        int cap = 0, vpitch = 0;
        __half *kc = nullptr, *vc = nullptr;
        float *x = nullptr, *cat = nullptr, *xn = nullptr, *qfull = nullptr, *kbuf = nullptr, *vbuf = nullptr, *qbuf = nullptr,
              *attn = nullptr, *rlog = nullptr, *post = nullptr, *wts = nullptr, *hbuf = nullptr, *dn = nullptr, *gs = nullptr,
              *us = nullptr, *hs = nullptr, *shared = nullptr, *sg = nullptr, *rn = nullptr, *logits = nullptr, *hout = nullptr;
        int32_t *ids = nullptr, *d_args = nullptr, *d_am = nullptr, *pin = nullptr;  // d_args = {q0, tokens...}
        ~Mtp() {
            if (pin) cudaFreeHost(pin);
        }
    } mtp;

    // The DFlash2 drafter (gpu_dflash.cu): fed by every pass that reads positions (the layer inputs it needs are captured inside the
    // pass, ingest() turns the accepted rows into its context K/V), asked for a block by decode_spec.
    std::unique_ptr<Dflash> df;
    int df_from = 0;  // prefill: positions below it are not ingested (the drafter's window never reaches them)
    int df_stage_rows = 0;
    bool df_ingest(int stage, int r0, int n, int pos0);

    // ---- prefill (owner: the prefill agent) -------------------------------------------------------------
    bool ensure_prefill(int cap);
    // The forward pass of T prompt tokens at positions pos0.. on the batched int8 path; the residual stream of
    // the chunk is left in pf.x.
    bool run_chunk(const int32_t* toks, int T, int pos0, int df_r0 = -1);  // df_r0 >= 0: rows df_r0.. go to the drafter's stage 1

    // ---- state: snapshots, rewind, slots (owner: the tape agent; gpu_state.cu) ----------------------
    // A slot is one conversation: its page table (into the shared K/V pool), its recurrent state and its captured graphs. The live
    // slot's state pointers and graphs are the members above (S, hist, gexec, warm, rw.gx, rw.warm); the others sit in st.slots, and
    // select_slot() swaps the two sets (pointers only). A captured graph bakes the state pointers and the page table in, so each
    // slot keeps its own graphs. The table's host side is st.pool's, indexed by slot.
    struct Slot {
        std::vector<float*> S, hist;
        cudaGraphExec_t gexec[2] = {nullptr, nullptr};
        int warm[2] = {0, 0};
        cudaGraphExec_t rgx[40] = {};
        int rwarm[40] = {};
        int32_t* d_pt = nullptr;  // the device page table: n_chunks + 1 entries, -1 = none
        void* blob = nullptr;     // the cudaMalloc behind an extra slot's state (null for slot 0: `owned` holds it)
        uint64_t bytes = 0;
    };
    struct State {
        uint64_t snap_bytes = 0, slot_bytes = 0;
        std::vector<void*> snap;        // one block per checkpoint slot
        std::vector<char> filled;       // a checkpoint slot written by snapshot()
        std::vector<Slot> slots;        // [s] holds slot s while it is not the live one; [0] exists from open()
        int cur = 0;
        State() = default;
        State(const State&) = delete;
        State& operator=(const State&) = delete;
        ~State();
    } st;
    bool state_ready();  // sizes st (first call): snapshot bytes, slot 0
    void state_sizes();  // st.snap_bytes and st.slot_bytes
};

}  // namespace limbo::cuda
