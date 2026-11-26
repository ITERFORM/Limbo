# Does OpenMP belong in Limbo? Measured, 2026-11-20

The operator rule (PLAN.md §8) says "lock-free and OpenMP, with SIMD dispatch"; the engine's shape
(PLAN.md §4) says "one CPU pool, on physical cores only, CCX-aware, spin-then-futex... hand-offs are
lock-free (rings, atomics, futex park). No mutex." Those two lines only conflict if OpenMP is read as the
*engine's* threading substrate. This measures which reading is right.

`probe/omp_vs_pool.cpp` (built with `-fopenmp`, linked against the verified decoder) dispatches the same
work four ways: **A** `#pragma omp parallel for`, **C** `std::thread` spawned per dispatch (what the CPU
reference in `src/step.cpp` does today), **D** a persistent pool with spin-then-futex (a small stand-in
for `src/cpu_lane_pool.cpp`). The work is one expert's matrix: rows of `blk.0.ffn_gate_exps.weight`
(IQ2_S, 2048 wide) dotted against a vector, which is the CPU lane's actual task shape.

RTX-adjacent hardware: Ryzen 7 3800X, 6 workers (8 physical cores minus the GPU feeder and the OS, as
the lane's design reserves). Best of 80 dispatches, pages warm.

| work per dispatch (1 thread) | A OpenMP, passive | A OpenMP, active | C thread spawn | D pool + spin |
|---|---|---|---|---|
| **512 rows** (3,484 µs) | 747 µs | 639 µs | 705 µs | **589 µs** |
| **16 rows** (84 µs) | 21.2 µs | **12.3 µs** | 110.6 µs | 13.4 µs |

Speedup on 6 workers: pool **5.9x** at 512 rows (98% of linear), 6.3x at 16 rows.

## What the numbers say

1. **The prior I would have asserted is wrong.** I expected OpenMP's region entry (a futex wake per
   region with the default passive policy) to cost ~10 µs and to be visible at these sizes. It is not:
   passive OpenMP is 21 µs at 16 rows against the pool's 13 µs, and 747 vs 589 µs at 512 rows. OpenMP is
   within 20% at the big size and equal at the small one when set to active. Nothing here disqualifies it.
2. **`std::thread` per dispatch is a real wart in the reference.** At 16 rows it costs **110 µs to do
   84 µs of work** — worse than single-threaded. That is the one place in the tree where the dispatch
   mechanism dominates, and it is the reference, not the lane.
3. **The pool is the best or tied-best at both sizes** (589 µs / 13.4 µs), and it is the only one of the
   three that can express what the lane actually needs.

## Why the pool stays anyway — and it is not speed

The lane's scheduler (src/cpu_lane_pool.cpp, and the design in docs/donors/strata.md §5, kindi.md §7)
chains three stages **without a barrier between them**: Q quantizes the layer's input once, A computes
expert gate+up and the thread that finishes an expert's last A task quantizes that expert's h and raises
its ready flag, B waits on that flag. `#pragma omp parallel for` has an implicit barrier at the end of
every region; a worksharing construct cannot express a producer/consumer flag chain, and `nowait` plus
hand-rolled atomics would be the pool written inside OpenMP, with libgomp's team management on top.

The core map is the same argument: one worker per **physical** core, the first core left free for the GPU
feeder thread, the last core of the other L3 domain left to the OS. That is a persistent worker set with
a fixed map across a whole step, not a per-region place list.

So: **the pool is not there because it is faster — it is there because OpenMP cannot say what the lane
has to say.** The measurement says adding OpenMP would not cost much; the design says it would not buy
the thing that matters.

## The recommendation

- **The CPU lane keeps its pool.** No `#pragma omp` in the lane's scheduler or its kernels.
- **Add OpenMP (or better, reuse the pool) in exactly one place: the CPU reference's `parallel_for`.**
  The measured 110 µs per dispatch against 84 µs of work is the reference's largest avoidable cost, and
  the reference is the oracle for every kernel — it is the tool that decides whether a kernel is right.
  Reusing the lane's pool is preferable to a new dependency; OpenMP is the smaller diff if the pool's API
  does not fit a plain row range.
- **`#pragma omp simd` on the dequant+dot loops is a separate question from threading** and should be
  decided by measurement: at 512 rows the pool already reaches 5.9x on 6 workers, so the lane is not
  dispatch-bound there — any remaining headroom is in the inner loop, not in the dispatch.
- **Write the reading into PLAN.md §8**: "lock-free and OpenMP" means the hot path is lock-free and
  OpenMP is available for coarse paths (the reference, tools, tests) — not that `#pragma omp` may be
  dropped into a lane that is entered per expert per layer.

## Reproduce

    g++ -O2 -std=c++20 -Iinclude -fopenmp probe/omp_vs_pool.cpp build/liblimbo.a -o build/bin/tools/omp_vs_pool
    OMP_NUM_THREADS=6 OMP_WAIT_POLICY=active build/bin/tools/omp_vs_pool <rung1.gguf> 80 6

## The shape the GPU kernel implements, on the CPU side

`f32_cols_k` is a multi-column GEMV: one row of W per block, decoded once and dotted against C columns
(`y[c][r] = W[r,:] . x[c]`). The CPU lane's union batching is the same shape, so the probe now runs it:
C columns share one pass over the weights, 512 rows, 6 workers, best of 40, warm.

| C columns x 512 rows | work (1 thread) | OpenMP passive | OpenMP active | thread spawn | pool + spin |
|---|---|---|---|---|---|
| C=1 | 3.50 ms | 773 us | 715 us | 713 us | **598 us** |
| C=4 | 5.46 ms | 1086 us | 1105 us | 1049 us | **927 us** |
| C=16 | 13.39 ms | 2479 us | 2493 us | 2372 us | **2370 us** |

**As the work per dispatch grows, the mechanism stops mattering.** At C=16 the three are within 5% of
each other; at C=1 the pool leads OpenMP by 23%, and on a 16-row job by 37% (13.4 us against 21.2 us).
Speedup on 6 workers: 5.65-5.9x for the pool at every C, 4.5x (C=1) to 5.4x (C=16) for OpenMP.

## Where OpenMP could lead in this repo, in order of how much it buys

1. **The tools that gate everything** — `quant_parity` (2696 tensors x 5 sampled rows, serial today),
   the tokenizer corpus runs, the per-position layer-gate comparisons, `oracle_dump`'s prompt loop.
   Embarrassingly parallel over independent items, no shared state, no chaining: OpenMP is the smallest
   correct change and it cuts the wall time of the verification harness by ~5-6x on 6 workers. This is
   the strongest case in the tree and it does not touch the engine's design at all.
2. **The CPU reference's `parallel_for`** — measured 7-8x better than `std::thread` per dispatch at
   16 rows (12-21 us against 110 us). Reusing the lane's pool is the alternative.
3. **A CPU multi-column kernel written as a flat row loop** (the analogue of `f32_cols_k`): OpenMP ties
   the pool from C>=4, and only loses at C=1.
4. **Prefill chunks on the CPU backend** — large C, flat row range, a region per chunk that amortizes
   over milliseconds.
5. **`#pragma omp simd` on the dequant+dot inner loops** — a separate axis (vectorization, not
   threading). Note where the headroom is: at C=16 the pool reaches only 5.65x on 6 workers with
   13.4 ms of work per dispatch, so that path is not dispatch-bound — any remaining win is in the
   inner loop, and it should be measured before a pragma is added.

Unchanged boundaries: the lane's staged scheduler (per-expert flags, no barriers), the doorbell and
handoff paths (microsecond scale), the fixed core map across a step, and the GPU feeder thread.

One design note on the kernel itself: its structure — one row of W per block, 128 threads, four float4
each, a warp-shuffle tree then a four-way shared-memory reduction, C columns accumulated together — is
the GPU's answer to *small T with shared weights*. The CPU side has 7x the bandwidth (measured: DDR
47 GB/s against PCIe 6.6 GB/s) and its own answer is union batching in the lane. They are complements,
and the planner's crossover (~10 rows) is where they meet; the threading choice on the CPU side is worth
exactly what the table says: nothing at large C, 20-40% at small C, and never `std::thread` per dispatch.
