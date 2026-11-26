// Copyright (c) 2026 ITERFORM. All rights reserved.
#pragma once
// limbo/cpu_lane.hpp — the CPU lane: expert FFNs on the host cores, beside the GPU, in the same step.
//
// PLAN.md §3.1: compute where the bytes are, unless enough rows read them. An expert that does not live in
// VRAM sits in pinned RAM; for the few rows of a decode or verify window the CPU computes it there and only
// activations cross the bus (docs/donors/strata.md §5, §10; kindi.md §7; modchip.md §2, §4). This header is
// the lane's whole surface:
//
//   kernels   one expert's SwiGLU FFN (gate, up, down) for 1..16 rows, every weight row read once for all of
//             its rows (union batching: a block is decoded once into L1 and dotted against every column),
//             on int8 activations that are quantized ONCE per layer and shared by every expert job;
//   precision the activation is split into `planes` int8 planes (residuals of each other, the GPU prefill's
//             scheme, include/limbo/cuda/gpu.hpp set_prefill): 1 = llama.cpp's own Q8 numerics (fastest,
//             moves router near-ties), 2 = 1 part in 16k of a block's amplitude, 3 = fp32-like (the default:
//             a product within ~1e-6 of the exact one). A plane is just one more column of the same weight
//             decode, so the weight bytes are still read once;
//   pool      persistent workers on physical cores only, one core kept for the GPU feeder and one for the
//             OS and the desktop, tasks claimed lock-free from ONE atomic word (epoch | count | index),
//             spin-then-futex parking. No mutex, no condition variable anywhere.
//
// SIMD: runtime dispatch Scalar / AVX2 by __builtin_cpu_supports and target attributes. AVX-512: NO KERNEL
// EXISTS and none has ever run — neither box has it (PLAN.md §2) — so Isa::Avx512 is a reserved dispatch
// slot that today resolves to the AVX2 kernels. detect_isa() never reports it.
//
// Formats (limbo::Type ids; layouts in quants.hpp): Q4_K Q5_K Q6_K Q3_K Q2_K IQ2_S IQ3_S IQ4_XS IQ3_XXS IQ4_NL
// Q2_0. The kernels agree with the verified CPU decoder (quants.cpp): tests/cpu_lane_test.cpp holds every
// format to dequant + double-precision dot on real expert rows of the ladder's files.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace limbo::cpu {

enum class Isa : uint8_t { Scalar = 0, Avx2 = 1, Avx512 = 2 /* reserved: no kernels, never ran */ };

Isa cpu_isa();       // what this CPU reports (may say Avx512 on another machine)
Isa detect_isa();    // the best level this build has kernels for AND the CPU runs: Scalar or Avx2
const char* isa_name(Isa isa);

constexpr int kMaxRows = 16;     // tokens per expert job
constexpr int kMaxPlanes = 3;
constexpr int kMaxCols = kMaxRows * kMaxPlanes;

// True when gemv_cols / the lane implement this limbo::Type id.
bool lane_supports(uint32_t type);
// Activation block size a weight format wants: 256 for the super-block formats, 32 for IQ4_NL and Q2_0.
int act_block(uint32_t type);
// The weight block a format walks (elements) and its bytes.
int weight_block_elems(uint32_t type);

// ---- quantized activations -----------------------------------------------------------------------------
// `ncols` columns of K int8 values, in blocks of `blk` (256 or 32) with one float scale per block, and the
// sum of every 16 values (the K-quants' minimum terms need it). Column c of row r and plane p is r*planes+p.
struct ActQ {
    int K = 0, blk = 0, ncols = 0;
    int8_t* q = nullptr;     // [ncols][K], 32-byte aligned rows
    float* d = nullptr;      // [ncols][K/blk]
    int16_t* bs = nullptr;   // [ncols][K/16]
};
size_t actq_bytes(int K, int ncols);                  // storage for one ActQ (any blk), alignment included
void actq_bind(ActQ* a, void* mem, int K, int blk, int ncols);   // mem: 64-byte aligned, actq_bytes() long

// Quantizes nrows rows of x (each K floats) into `planes` planes; plane 0 is round(x / d0), plane p the
// quantization of what the planes before it left over. Writes columns [0, nrows*planes) of `a`.
void actq_quantize(Isa isa, const float* x, int nrows, int planes, ActQ* a);
// One row only, into columns [row*planes, row*planes+planes): the lane's per-row quantize task.
void actq_quantize_row(Isa isa, const float* x, int row, int planes, ActQ* a);

// ---- the primitive: weight rows against a set of activation columns ------------------------------------
// y[c * ldy + i] = dot(row i of W, column cols[c]) for i in [0, nrows), c in [0, ncols). W is nrows rows of
// row_bytes each (K elements per row, K = a.K, a multiple of the format's block). Each weight block is
// decoded once and applied to every column.
bool gemv_cols(Isa isa, uint32_t type, const uint8_t* W, uint64_t row_bytes, int nrows, const ActQ& a,
               const int* cols, int ncols, float* y, int ldy);

// ---- the lane ------------------------------------------------------------------------------------------
struct ExpertRef {              // one expert's three matrices as they sit in host memory (one slice of the GGUF tensors)
    const uint8_t* gate = nullptr;   // n_ff rows of n_embd
    const uint8_t* up = nullptr;     // n_ff rows of n_embd
    const uint8_t* down = nullptr;   // n_embd rows of n_ff
    uint32_t gate_type = 0, up_type = 0, down_type = 0;
    int n_embd = 0, n_ff = 0;
};

struct ExpertJob {
    ExpertRef w;
    int nrows = 0;                  // rows (tokens) that picked this expert, 1..kMaxRows
    const int* act_row = nullptr;   // which row of the layer's input tile row i reads (nullptr: i)
    float* out = nullptr;           // [nrows][n_embd]: down(silu(gate x) * up x), UNWEIGHTED; the consumer scales and sums
};

struct PoolConfig {
    std::vector<int> cpus;          // logical cpus of the workers, one per worker; empty = default_cpus()
    int spin_us = 150;              // idle spin before a worker parks on the futex
    bool pin = true;
};

// The default worker placement: one worker per physical core (never an SMT sibling), the first core kept for
// the GPU feeder thread and the last core of the other L3 domain for the OS and the desktop, the rest split
// over the two L3 domains. `max_workers` caps it (0 = all that remain; this box: 6).
// reserve_os = false keeps only the GPU feeder's core back (a dedicated inference box: the desktop shares the SMT
// siblings).
std::vector<int> default_cpus(int max_workers = 0, bool reserve_os = true);

class Pool;   // pool.cpp-internal; Lane owns one

struct LaneStats {
    uint64_t batches = 0, tasks = 0;
    uint64_t parks = 0;            // worker futex sleeps
};

class Lane {
public:
    struct Config {
        PoolConfig pool;
        int planes = 3;            // activation planes, 1..3 (see the header comment)
        int max_jobs = 64;         // experts per layer batch
        int max_tokens = kMaxRows; // rows of the layer input
        int slice_a = 16;          // gate/up rows per task
        int slice_b = 32;          // down rows per task
        Isa isa = Isa::Scalar;     // Scalar = detect_isa()
    };
    // nullptr and *err on a bad config.
    static std::unique_ptr<Lane> create(const Config& cfg, std::string* err = nullptr);
    ~Lane();
    Lane(const Lane&) = delete;
    Lane& operator=(const Lane&) = delete;

    int workers() const;
    Isa isa() const;
    int planes() const;
    // Idle spin before the workers park on the futex. The engine raises it for the length of a decode step (the
    // gap between layers is ~1 ms) and drops it to ~0 between generations, so an idle desktop never sees the pool.
    void set_spin_us(int us);

    // Starts one layer: x is T rows of n_embd floats (the normed MoE input), jobs the experts to run on it.
    // Returns at once; x, the jobs' weights, act_row and out must stay valid until done(). One batch in flight.
    // false when the batch is invalid (too many rows/jobs, unsupported type, ...), *err says which.
    bool submit(const float* x, int T, const ExpertJob* jobs, int njobs, std::string* err = nullptr);
    bool done() const;       // every job's out is written (acquire)
    void wait();             // spin, then futex
    // submit + wait.
    bool run(const float* x, int T, const ExpertJob* jobs, int njobs, std::string* err = nullptr);

    LaneStats stats() const;

private:
    Lane();
    struct Impl;
    std::unique_ptr<Impl> p_;
};

// Reference for the tests and the bench: down(silu(gate x) * up x) in double precision from the CPU decoder
// (quants::dequant_row), one row. out is n_embd floats.
void expert_ffn_reference(const ExpertRef& w, const float* x, float* out);

}  // namespace limbo::cpu
