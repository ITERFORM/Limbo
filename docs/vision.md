# Vision input for Qwen3.8-27B (qwen35 + mmproj)

What the engine must compute for an image, written from reading llama.cpp's mtmd (MIT; read, not copied) and measured
against it (`oracle/oracle_vision.cpp`). Our code is our own; this file is the contract.

Two files: the LLM GGUF (qwen35, `rope.dimension_sections = [11,11,10,0]`) and the projector
`mmproj-Qwen3.8-27B-BF16.gguf` (`clip.projector_type = qwen3vl_merger`). Request shape: OpenAI chat `content` parts,
`{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}`.

## 1. Projector hyper-parameters (read from the mmproj, nothing hard-coded)

patch 16, embd 1152, ffn 4304, 27 blocks, 16 heads (head_dim 72), LayerNorm eps 1e-6, merge 2x2, output 5120 (= LLM
n_embd), position table 2304 = 48 x 48, mean = std = 0.5 per channel, no deepstack layers (all flags false), no
class token. Weights are BF16 matrices, F32 vectors; `v.patch_embd.weight` and `.weight.1` are the two temporal
taps of a 3-D conv, F32 `[kw=16, kh=16, c=3, oc=1152]` (flat per output channel: `kw + 16*(kh + 16*c)`).

## 2. Preprocessing (CPU, deterministic)

1. Decode to 8-bit RGB (alpha dropped, not composited; grey and palette expanded).
2. Target size, "smart resize" with `align = patch*merge = 32`: `w' = max(32, round(w/32)*32)`, same for h. Pixel budget
   `[min_px, max_px] = [8, 4096] tokens * 1024 px` (a token covers 32x32 px). Over budget:
   `beta = sqrt(h*w/max_px)`, `h' = max(32, floor(h/beta/32)*32)`, same for w. Under the minimum:
   `beta = sqrt(min_px/(h*w))`, `h' = ceil(h*beta/32)*32`. (Our server lowers max_px to a token cap, default 1024, and to
   what the engine's scratch can hold; larger images are shrunk by the same rule.)
3. Resample to `w' x h'` with the separable Pillow bicubic filter (a = -0.5): for output pixel `i` the centre is
   `(i+0.5)*s` with `s = in/out`; the filter is widened by `max(1,s)` (support 2*max(1,s)), taps
   `x0 = int(c - support + 0.5)` clamped to 0 up to `x1 = int(c + support + 0.5)` clamped to `in`, weights
   `f((x - c + 0.5)/max(1,s))` normalised to sum 1. Horizontal pass first, then vertical, each result rounded and
   clamped back to 8 bits (22-bit fixed-point weights, `+0.5` bias then shift; this rounding is part of the contract).
   Equal size: copy.
4. Normalise: `(px/255 - 0.5)/0.5` per channel, planar `[3][h'][w']` for the patch conv.

Token grid: `px = w'/16`, `py = h'/16` patches, `nx = px/2`, `ny = py/2` merged tokens, `n = nx*ny` (<= 4096).

## 3. Vision tower

Patch embedding: the image is fed as both temporal frames, so the conv is `(W0 + W1) * patch` with stride 16, no
padding: `e[p] = bias + sum W[kw,kh,c] * img[c][16*py + kh][16*px + kw]`, i.e. a `[768] x [768,1152]` product per
patch, input order `c, kh, kw`.

Token order: patches are numbered in 2x2 merge groups: `t = ((y/2)*nx + x/2)*4 + (y%2)*2 + (x%2)` for patch (x, y).
All per-token data below (position embedding, rope positions) uses this order, so the merger reads 4 consecutive
rows.

Position embedding: the 48x48 table (`[2304][1152]`, row index `y*48 + x`) is resampled bilinearly with
align-corners to `py x px`: source coordinate `sy = y*47/(py-1)`, `sx = x*47/(px-1)`, `y0 = floor(sy)`, `y1 =
min(y0+1, 47)`, fractional weights `dy = sy-y0`, `dx = sx-x0`. It is added after the patch bias.

Block (27x, pre-norm, residual stream in fp32):

    x = x + out_proj( attn( rope( qkv(LN1(x)) ) ) )          LN = (x-mean)/sqrt(var+eps)*w + b   (biased variance)
    x = x + down( gelu_tanh( up(LN2(x)) ) )                  every matmul has a bias

`qkv` is one `[1152 -> 3456]` matmul, rows `[q (16x72) | k | v]`. Attention is full and bidirectional over all
patches of the image, scale `1/sqrt(72)`, per head. gelu is the tanh approximation
(`0.5 x (1 + tanh(0.7978845608 (x + 0.044715 x^3)))`), which is what the operator's llama.cpp computes for `use_gelu`
(torch's merger uses the exact erf form; the difference is below the bf16 noise and we follow the oracle).

Vision RoPE (per head, 72 dims): only pairs `(i, i+36)`, `i = 0..35`, rotate; angle
`theta_i = pos_i * 10000^(-2*(i mod 18)/36)` with `pos_i = y` for `i < 18` and `x` for `i >= 18`, where (y, x) is the
patch's row and column in the *image* grid (not the merge-group index). `q' = q cos - q_pair sin`-style NEOX rotation:
`out_i = a cos - b sin`, `out_{i+36} = b cos + a sin` with `a = in_i, b = in_{i+36}`.

Merger: `post_ln` (LayerNorm on each 1152 row), then 4 consecutive rows concatenated to a 4608 vector (order
`(0,0),(0,1),(1,0),(1,1)` = (y%2, x%2)), `mm.0` (4608 -> 4608, bias) -> gelu_tanh -> `mm.2` (4608 -> 5120, bias). Output
`n x 5120` rows, one per image token, in `t/4` order = raster order of the (nx, ny) grid.

## 4. Placement in the prompt

The chat template (the GGUF's own) renders each image part as `<|vision_start|><|image_pad|><|vision_end|>`. After
tokenising, every `<|image_pad|>` is replaced by `n` image tokens, the markers stay ordinary text tokens. The image
tokens carry no embedding lookup: row `r` of the projector output is the input embedding of the r-th token (used as is,
no scaling). They are attended causally like text.

## 5. Positions: interleaved M-RoPE on the LLM

qwen35 uses the *interleaved* variant (`IMROPE`). Only the 16 full-attention layers rotate (the 48 delta-net layers
have no positions). Each token has three positions `(pt, ph, pw)`; the rotation is the usual NEOX partial rope on the
first 64 of 256 dims, pairs `(i, i+32)`, frequency `base^(-2i/64)` with base 1e7, but the position that multiplies pair
`i` is picked by `i mod 3`: 0 -> pt, 1 -> ph, 2 -> pw, with the w-lane only for `i < 30` (i = 30 -> pt, 31 -> ph). That is
11 + 11 + 10 = the file's section counts. A text token has `pt = ph = pw = p`, so the table is the plain rope's, bit for
bit.

Image tokens of one image starting at text position `p0` (the position the first `<|vision_start|>`'s successor would
have): token `r` (raster index in the `nx x ny` grid) has `pt = p0`, `ph = p0 + r/nx`, `pw = p0 + r%nx`. The image
consumes `max(nx, ny)` positions: the next text token (`<|vision_end|>`) sits at `p0 + max(nx, ny)`, while the K/V
pool holds all `nx*ny` rows. Hence the cache row index and the rope position differ after an image by
`delta = max(nx, ny) - nx*ny` per image (<= 0): decode, the verify rows and the MTP block rotate at `row + delta_total`.

## 6. Engine consequences

* The tape (prefix reuse, checkpoints, K/V sharing) sees an image as `n` pseudo tokens: ids `0x80000000 | h31(H, r)` with
  `H` = a 64-bit hash of (decoded pixels after preprocessing, nx, ny) and `r` the row; equal images hit the cache.
* The tower runs once per image hash (embeddings kept in an LRU on the host), in the engine's idle prefill scratch.
* A prefill chunk with image rows gets: embeddings for those rows, and 3 positions per row.
