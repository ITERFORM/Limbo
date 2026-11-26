# Rung 1 against llama.cpp, rechecked (2026-11-26)

Rung 1 (Qwen3.6-35B-A3B UD-IQ3_XXS, `qwen35moe`) on the RTX 4060 Ti 16 GB, main at 5c41046, the commands of
`bench/w1-gpu-perf-2026-11-19.md` §6 run unchanged (`gpu_forward`, chunk 1024, n_ctx 4096 for prefill, 512 for decode).
The llama.cpp column is that file's `llama-bench` run (fork build `0beb5b84b`, `-ngl 99 -fa 1 -p 512,2048 -n 128 -r 3`);
it was not re-run. The X11 desktop drew about 16% of the GPU during these runs; single runs, not machine-gated.

| | llama.cpp | Limbo 2026-11-19 | Limbo 2026-11-26 |
|---|---|---|---|
| prefill pp512, 3 planes (exact) | 2260.98 ± 78.28 | 1943 | 2323 (mean of 3; best 2335) |
| prefill pp2048, 3 planes | 2078.44 ± 134.32 | 2085 | 2471 (best 2476) |
| prefill pp512, 2 planes | | 2474 | 2799 (best 2810) |
| prefill pp2048, 2 planes | | 2695 | 3063 (best 3069) |
| decode tg128 (128 tokens, CUDA graph) | 75.99 ± 1.22 | 91.0 | mean 88.4 (87.8 to 89.1) |

- The 14% gap at 512 tokens with 3 planes (2026-11-19 §7.1) is closed: 2323 against 2260.98 is inside llama.cpp's own
  spread, and pp2048 is 19% ahead. The dense products now run on the int8 fragment kernel (one decode of each weight
  into the tensor-core layout, Q6_K included) and the pipelined dense kernel (weights and half super-block activation
  stages double buffered), the two remedies §7.1 named.
- Decode: mean 88.4 over 9 runs (3 runs x the prose, code and tools prompts). With `LIMBO_FRAG=0` (every dense
  product on the FP32 kernels) the same command gives mean 94.8 (94.2 to 95.3). Single steps stay on the fragment
  kernel by default because the verify windows use it: a step and a window then compute a row with the same bits, so
  greedy speculation keeps the plain loop's text. Serving with speculation is faster on it
  (`bench/dense27b-2026-11-26.md`: 172 against 152 tok/s on code.txt).

## Raw lines

```
LIMBO_FRAG=1 (default), three runs:
decode  127 tokens ... = 88.81 / 88.02 / 89.11 tok/s
decode  127 tokens ... = 87.99 / 88.46 / 88.45 tok/s
decode  127 tokens ... = 87.78 / 88.34 / 88.39 tok/s
LIMBO_FRAG=0:
decode  127 tokens ... = 94.96 / 95.27 / 94.20 tok/s
--planes 3: prefill   512 tokens, 3 planes, chunk 1024: mean 220.4 ms = 2323 tok/s (best 219.3 ms = 2335 tok/s)
            prefill  2048 tokens, 3 planes, chunk 1024: mean 828.7 ms = 2471 tok/s (best 827.1 ms = 2476 tok/s)
--planes 2: prefill   512 tokens, 2 planes, chunk 1024: mean 182.9 ms = 2799 tok/s (best 182.2 ms = 2810 tok/s)
            prefill  2048 tokens, 2 planes, chunk 1024: mean 668.7 ms = 3063 tok/s (best 667.4 ms = 3069 tok/s)
```
