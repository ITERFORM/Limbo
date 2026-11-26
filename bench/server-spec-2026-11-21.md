# MTP speculation in the server (2026-11-21)

`limbo_serve_gpu --spec auto` (the default) loads the model's own MTP block (blk.40 of rung 1) and decodes through
`Backend::decode_spec`: a draft of k tokens from the block, checked in one T = k+1 row window (`Gpu::step_rows(write_back=false)`), the accepted
prefix kept by `commit(m+1)`. Greedy is exact; a sampled request keeps a draft with probability p(draft) and otherwise draws from p without it.
Rung 1 UD-IQ3_XXS, RTX 4060 Ti 16 GB (shared with the desktop), `--ctx 32768`, 256 tokens, every GPU run under `flock build/gpu.lock`.

## The "greedy differs" report was K/V precision, not the cycle path

`--kv auto` fits the block before the cache's precision (include/limbo/cuda/backend.hpp): at 32k with two homes it picks **q8 without the block and q4 with
it**. `spec_check.py` at its old default compared a q8 server with a q4 one; the first difference sat at char 23 of prompt 0 in both runs (also with every
cycle forced onto the plain one-row path). With one fixed `--kv` on both sides the text is identical on all three prompts at f16, q8 and q4. The cycle path (T-row window without write-back, then `commit(n < T)`) needed no change.
`spec_check.py` now takes `--kv` (default q8) and says when the two servers ended up with different formats.

## Numbers (`spec_check.py --kv F`, 3 greedy prompts + 1 sampled at temperature 0.7)

After the follow-up below (calibration at load, plain-row fallback). Plain tok/s is the `--spec off` server in the same run.

| K/V | prompt | plain tok/s | speculative tok/s | text |
|---|---|---|---|---|
| f16 | lighthouse story | 100.8 | 128.5 (x1.28) | identical |
| f16 | python function | 100.6 | 153.2 (x1.52) | identical |
| f16 | CPU cache | 100.6 | 140.4 (x1.40) | identical |
| f16 | sampled | 87.9 | 105.8 | runs |
| q8 | lighthouse story | 99.9 | 125.6 (x1.26) | identical |
| q8 | python function | 99.6 | 151.1 (x1.52) | identical |
| q8 | CPU cache | 99.8 | 144.6 (x1.45) | identical |
| q8 | sampled | 86.9 | 103.8 | runs |
| q4 | lighthouse story | 99.9 | 125.0 (x1.25) | identical |
| q4 | python function | 99.6 | 150.4 (x1.51) | identical |
| q4 | CPU cache | 99.8 | 143.1 (x1.43) | identical |
| q4 | sampled | 87.1 | 104.2 | runs |

Before the follow-up (calibration inside the first request, no fallback; the story prompt paid ~0.8 s of it): story x1.01-1.07, python x1.46-1.60,
CPU cache x1.39-1.46. Server log: `spec=244/155 (1.57 tokens a pass, 89 of 164 drafts kept, 0 plain rows)`, `255/135 (1.89, 120 of 134)`, `255/142 (1.80, 113 of 141)`.
Free prose is the hard case (about 1.6 tokens a pass); code and explanatory text reach 1.8-1.9.

## Calibration at load, and never slower than plain

- `GpuBackend::open` times every variant the decode loop runs (windows of 1..8 rows with their commit, the plain row, the block's refresh over 1..8 rows,
  the chained draft; each once eagerly, once captured, then timed) at position 0 right after the block loads, then resets the device state. The first
  request no longer pays for it (story prompt 95.8 -> 125.6 tok/s at q8). The startup line prints the costs:
  `MTP speculation (row 9.8 ms, windows of 2/3/4 rows 11.7/13.0/14.9, block step 0.7)`. The rows' scratch is now allocated at load, so the server's
  free-VRAM arithmetic for homes sees it (at 64k with the desktop holding VRAM that can cost the second home: 1283 MiB free before homes with the block, 1792 without).
- `decode_spec` picks k = 0..4 by tokens per ms; k = 0 is a plain one-row window with write-back (cost = the plain row). In plain mode one k = 1 cycle is
  tried every 16 tokens so the acceptance EMA can recover (`spec=... N plain rows` counts the rows). Plain rows move the block's base to their own
  position, so drafting that resumes starts the block's positions again at 0 instead of attending entries it never wrote.
- Position 0 does not predict the costs at depth (at 60k a plain row is 11.9 ms, a cycle about 18 ms, not the 13 ms of the table): a row's and a cycle's
  cost are each scaled by an EMA of measured / calibrated ms, so the fallback works at the run's own depth. Without it the short needle answers still drafted
  at 21-31% acceptance and ran at 72-87 tok/s (mean 80.6) against 83-87 plain (mean 84.9).
- Greedy stays exact: a plain row is the same pass the first token of a request already takes. Checked with a temporary 3x bias that forced plain rows
  (232 plain rows and 14 probe cycles of 253 tokens) and a 1.25x bias that mixed both: text identical at q8; the bias is not in the tree.

## 64k context (`needle_serve.py --ctx 65536 --tokens 60000`)

- `--kv auto`: both needles found, but **no speculation**: with two homes of 65,536 positions the block does not fit beside them (homes win over the block).
- `--kv q4 --gen 256` (the block loads first, then as many homes as fit; this run got one, the desktop held VRAM): both needles found twice, speculation on.
  Decode of the four short answers (10-19 tokens, hard to draft: 4 of 11 drafts kept, 2 plain rows per request): 74.2, 93.2, 83.7, 90.5 tok/s, mean 85.4;
  the same server with `--spec off` (two homes, answers 2 and 4 from the cache): 86.9, 83.3, 85.4, 83.8, mean 84.9. Requests of 0.1-0.25 s are noisy.
  A 256-token answer at 60,058 tokens of context: **95.7 tok/s speculative** (`spec=255/154 (1.66 tokens a pass, 101 of 154 drafts kept, 1 plain rows)`)
  against **80.3 plain**.

## Also changed

- `tools/bench_decode.cu` `run_spec`: the block's refresh took the next tokens from the last R entries of the previous window; the accepted rows are its
  first R, so `nxt[j] = prev[j + 1]` (only after a rejection with k >= 2). `--spec prose,code --gen 128 --draft-vocab 65536 --k 2`, output identical to
  plain before and after: before prose 132.7 tok/s, 2.06 tokens a cycle, 53% of 124 drafts; code 137.5 tok/s, 2.10, 55% of 122. After prose 132.8, 2.06, 53% of 124;
  code 135.2, 2.06, 53% of 124. Acceptance does not move beyond noise (a few drafts of ~120).
- `tests/serve_test.cpp`: `SpecFake` returns 1..3 tokens a `decode_spec` call; greedy text, finish_reason, token count and every read equal the plain
  loop's, each token is read once at consecutive positions, `max_tokens` holds, an end-of-generation token that closes a run is never read.
- `scripts/needle_serve.py --gen N` (decode speed at the long context), prints the server's `spec=` lines.

## Commands

    flock -w 3600 build/gpu.lock python3 scripts/spec_check.py --kv f16     # also q8, q4; --ctx 32768 by default
    flock -w 3600 build/gpu.lock python3 scripts/needle_serve.py --ctx 65536 --tokens 60000 --kv q4 --gen 256 [--extra "--spec off"]
    flock -w 3600 build/gpu.lock build/bin/tools/bench_decode -m rung1.gguf --spec prose,code --gen 128 --draft-vocab 65536 --k 2
    make test
