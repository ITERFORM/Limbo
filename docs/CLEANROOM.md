# Clean room

Limbo carries its own licence, and that only holds if its code is ours. Ideas, algorithms, measurements
and file-format facts are free to use. Expression is not. These rules keep the line visible.

## The two roles

- **Readers** study a donor and write `docs/donors/<donor>.md` in their own words: what the idea is,
  why it works, the measured effect, layouts and math, and `path:line` provenance. Every claim is
  tagged READ, MEASURED, CLAIMED or EST.
- **Writers** implement Limbo from those specs, from public papers and from our own measurements.
  When a spec is not enough, a writer asks for the spec to be extended. A writer does not open the
  donor source.

## Rules

0. The oracle (`oracle/`, `make oracle`) is a measuring device, not engine code: it links the
   operator's own llama.cpp build to record greedy tokens and top-2 gaps, and nothing from it is ever
   linked into `liblimbo.a`. Only its numbers cross the line.

1. No donor code in this repository. That means no copied blocks, no line-by-line translations
   (CUDA to CUDA, Python to C++, intrinsics sequence to intrinsics sequence), and no pasted kernel
   strings or tables.
2. Specs contain no code. Pseudo-code is allowed only at the level of an algorithm description, never
   as a transcription of a donor function.
3. Format facts are interoperability, not expression: GGUF block layouts, type ids, model math. They
   are written down exactly.
4. Codebook grids that must match bit for bit (the ggml i-quant grids) come from ggml under its MIT
   notice, or are regenerated, and are listed in `THIRD_PARTY_NOTICES`. They never come through a donor
   that copied them, such as Strata.
5. kindi is AGPL-3.0, so it gets the strictest reading: ideas and measurements only.
6. Every commit that implements a donor idea names the spec section it came from.

## Donors

| donor | licence | spec |
|---|---|---|
| ahtx/kindi 0.6.1 (`7fececb`) | AGPL-3.0 | `donors/kindi.md` |
| synchopate/llama-modchip v3.0 (`cfc906a`) | Apache-2.0 | `donors/modchip.md` |
| llama.cpp-adaptive-kv-streaming (itsXactlY `493e34eef`, Raymond v2 `4804b4044`) | MIT | `donors/kv-streaming.md` |
| Niko1221/Strata (local `d31fbc0`, engine 0.1.61) | MIT | `donors/strata.md` |
