#!/usr/bin/env python3
# Copyright (c) 2026 ITERFORM. All rights reserved.
"""scripts/readme_claims.py — the README's claims gate (readme-claims-gate).

Asserts, for every measured number README.md states: the claim appears in the README exactly once
(a second copy is a substitute witness for a deleted claim), the artifact of record contains its own
form of the figure, the computed percentages are recomputed from the artifacts' own strings, and
every path the README asserts exists — backticked or linked — resolves in the tree.

  scripts/readme_claims.py            # gate README.md
  scripts/readme_claims.py /tmp/x.md  # gate another copy (the mutation test does this)

Exit 0 green, 1 with named FAIL lines. A number this script cannot parse is a FAIL, never a pass.
"""
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (claim as the README states it, artifact of record, the artifact's own form of the figure).
# Where the README restates a figure in a summary and a detail table, the two forms are separate
# rows: each pattern must appear exactly once, so deleting either one is caught.
# The ngram table ties each figure to its column: off, mtp, ngram, auto.
NGRAM_ROW = (r"\| off \| mtp \| ngram \| auto \|.*?\| 92\.4 \| 133\.2 \(1\.44x\) "
             r"\| 262\.9 \(2\.85x\) \| 267\.4 \(2\.89x\) \|")

CLAIMS = [
    # 06 — speculative decoding
    (r"1,200 tokens", "bench/ngram-2026-11-22.md", r"1,200 tok"),
    (r"Plain decode \| 92\.4 tok/s", "bench/ngram-2026-11-22.md", NGRAM_ROW),
    (r"MTP \| 133\.2 tok/s", "bench/ngram-2026-11-22.md", NGRAM_ROW),
    (r"N-gram \| 262\.9 tok/s", "bench/ngram-2026-11-22.md", NGRAM_ROW),
    (r"Auto \(per cycle\) \| 267\.4 tok/s", "bench/ngram-2026-11-22.md", NGRAM_ROW),
    (r"the repack of the 27B's drafter never fit", "models.json", r"the Q8_0 repack of the drafter never fit"),
    (r"at most eight rows", "tools/limbo_serve_gpu.cu", r"at most 8 rows"),
    # 08 — against llama.cpp: Limbo re-measured 2026-11-26, llama-bench from 2026-11-19
    (r"88\.4 tok/s", "bench/rung1-recheck-2026-11-26.md", r"mean 88\.4"),
    (r"75\.99 ± 1\.22 tok/s", "bench/w1-gpu-perf-2026-11-19.md", r"75\.99 ± 1\.22"),
    (r"2,323 tok/s", "bench/rung1-recheck-2026-11-26.md", r"2323 \(mean of 3"),
    (r"2,260\.98 ± 78\.28 tok/s", "bench/w1-gpu-perf-2026-11-19.md", r"2260\.98 ± 78\.28"),
    (r"2,471 tok/s", "bench/rung1-recheck-2026-11-26.md", r"2471 \(best 2476\)"),
    (r"2,078\.44 ± 134\.32 tok/s", "bench/w1-gpu-perf-2026-11-19.md", r"2078\.44 ± 134\.32"),
    (r"50\.0 tok/s", "bench/inkling2-decode-tune-2026-11-24.md", r"50\.0 tok/s"),
    (r"46\.19 tok/s", "bench/inkling2-budget-2026-11-23.md", r"46\.19"),
    (r"94\.8 tok/s", "bench/rung1-recheck-2026-11-26.md", r"mean 94\.8"),
    # 08 — dense 27B
    (r"21,253-token code prompt", "bench/dense27b-2026-11-26.md", r"21,253-token code prompt"),
    (r"280 tok/s", "bench/dense27b-2026-11-26.md", r"\| 280 tok/s \(the operator's log"),
    (r"554 tok/s", "bench/dense27b-2026-11-26.md", r"\| 554 tok/s \|"),
    (r"19\.7 tok/s", "bench/dense27b-2026-11-26.md", r"\| 19\.7 tok/s \(the operator's log\)"),
    (r"49\.9 tok/s", "bench/dense27b-2026-11-26.md", r"49\.9 tok/s, 3\.73 tokens a pass"),
    (r"329 tok/s", "bench/dense27b-2026-11-26.md", r"\| 329 tok/s \| 666 tok/s \|"),
    (r"666 tok/s", "bench/dense27b-2026-11-26.md", r"\| 329 tok/s \| 666 tok/s \|"),
    (r"90\.2 ms", "bench/dense27b-2026-11-26.md", r"\| 90\.2 ms \(scalar multi-row attention\) \| 75\.8 ms \|"),
    (r"75\.8 ms", "bench/dense27b-2026-11-26.md", r"\| 90\.2 ms \(scalar multi-row attention\) \| 75\.8 ms \|"),
    (r"2,741 MiB", "bench/dense27b-2026-11-26.md", r"2,741 MiB above the reserve gave the first"),
    (r"46\.2 tok/s", "bench/dense27b-2026-11-26.md", r"57,344 positions \| \| 46\.2 tok/s"),
    (r"2,493 MiB", "bench/dense27b-2026-11-26.md", r"2,493 MiB the second"),
    (r"37\.6 tok/s", "bench/dense27b-2026-11-26.md", r"49,152 positions \| \| 37\.6 tok/s"),
    (r"613 tok/s", "bench/dense27b-2026-11-26.md", r"\| 267 tok/s \| 613 tok/s \|"),
    (r"37\.5 tok/s", "bench/dense27b-2026-11-26.md", r"\| 37\.5 tok/s \|"),
    (r"172 tok/s", "bench/dense27b-2026-11-26.md", r"172 tok/s \(152 with `LIMBO_FRAG=0`"),
    (r"152 tok/s", "bench/dense27b-2026-11-26.md", r"172 tok/s \(152 with `LIMBO_FRAG=0`"),
    (r"62\.7 tok/s", "bench/dense27b-2026-11-26.md", r"62\.7 tok/s, 84% of drafts kept"),
    # 08 — long context
    (r"222,811-token prompt", "bench/long-context-2026-11-21.md", r"222,811-token prompt"),
    (r"364 s prefill \(612 tok/s\)", "bench/long-context-2026-11-21.md", r"364 s \(612 tok/s\)"),
    (r"51\.8 tok/s", "bench/long-context-2026-11-21.md", r"\| 364 s \(612 tok/s\) \| 51\.8 \|"),
    (r"0\.52 s", "bench/long-context-2026-11-21.md", r"cached, 0\.52 s"),
    # 09 — the validation gates
    (r"28 of 28", "bench/dense27b-2026-11-26.md", r"`make test`: 28 of 28"),
    (r"20/20 files identical", "bench/tok-gate-2026-11-19.md", r"20 of 20 comparisons identical"),
    (r"13/13 formats, 0 differences", "bench/arch-coverage-2026-11-20.md",
     r"13 of 13 formats, 0 differences"),
    (r"49/50 positions agree", "bench/w1-cpu-gate-2026-11-19.md", r"\*\*49 agree"),
    (r"32 fixtures, 76 interpreter cases", "bench/serve-2026-11-19.md",
     r"32 chat fixtures, 76 interpreter cases"),
    (r"Within 1e-5 of row sum", "tests/cuda_gemv_test.cu", r"within 1e-5 of the row's sum"),
    (r"Bit-identical", "tests/cuda_gemm_test.cu", r"bit for bit"),
    (r"0 of 248320 logits differ", "bench/dense27b-2026-11-26.md", r"`0 of 248320 logits differ`"),
    (r"\| The same bits \|", "bench/dense27b-2026-11-26.md",
     r"a step and every window size compute a row with the same bits"),
    (r"61/64 positions", "bench/inkling2-cpu-gate-2026-11-23.md", r"61 agree"),
    (r"32/32 greedy tokens", "tests/cuda_ink_step_test.cu", r"the greedy token agrees at every position"),
    (r"bit-identical reruns", "tests/cuda_ink_step_test.cu", r"bit-identical"),
    (r"Within 2e-6", "tests/cuda_ink_moe_test.cu", r"within 2e-6"),
    (r"192\.0 vs 191\.9 KiB/position", "bench/inkling2-budget-2026-11-23.md",
     r"191\.9 KiB per position[^.]*\(account: 192\.0\)"),
    # engine facts
    (r"byte-identical to ggml's `to_float`", "bench/arch-coverage-2026-11-20.md",
     r"0 differences against ggml"),
    (r"complete 36-visit graph", "src/cuda/ink_gpu.hpp", r"36-visit"),
    (r"256-position pages", "include/limbo/cuda/gpu.hpp", r"256-position pages"),
    (r"3e-7", "include/limbo/cuda/gpu.hpp", r"3e-7"),
    (r"approximately 1\.5x", "include/limbo/cuda/gpu.hpp", r"about 1\.5x"),
    (r"1/16192", "include/limbo/cuda/gpu.hpp", r"1 part in 16192"),
    (r"up to 262144", "include/limbo/cuda/gpu.hpp", r"up to 262144"),
    (r"48 greedy tokens after a 4,096-token prefill", "bench/dense27b-2026-11-26.md",
     r"48 greedy tokens after a 4096-token prefill were identical"),
    # 11 — the server's defaults, from the header that sets them
    (r"\| `--port` \| `8080` \|", "include/limbo/serve.hpp", r"int port = 8080;"),
    (r"\| `--vram-reserve-mib M` \| `256` \|", "include/limbo/serve.hpp", r"int vram_reserve_mib = 256;"),
    (r"auto is 65536 on the GPU", "include/limbo/serve.hpp", r"the GPU entry 65536"),
    (r"4096 on the CPU", "include/limbo/serve.hpp", r"the CPU entry 4096"),
    (r"\| `--slots N` \| `1` \|", "include/limbo/serve.hpp", r"int slots = 1;"),
    (r"\| `--checkpoints N` \| `8` \|", "include/limbo/serve.hpp", r"int checkpoints = 8;"),
    (r"\| `--threads N` \| `6` \|", "include/limbo/serve.hpp", r"int threads = 6;"),
    # the machine table
    (r"NVIDIA RTX 4060 Ti, 16 GB VRAM", "docs/PLAN.md", r"RTX 4060 Ti 16 GB"),
    (r"\| Interface \| PCIe Gen3 x8 \|", "docs/PLAN.md", r"PCIe Gen3 x8"),
    (r"RTX 4060 Ti 16 GB · Ryzen 7 3800X · PCIe Gen3 x8", "docs/PLAN.md", r"Ryzen 7 3800X"),
    (r"AMD Ryzen 7 3800X, 8 cores / 16 threads, AVX2", "docs/PLAN.md",
     r"Ryzen 7 3800X, 8 cores / 16 threads, AVX2"),
    (r"\| System RAM \| 31 GB \|", "docs/PLAN.md", r"31 GB of DDR4"),
]

failures = 0


def fail(msg):
    global failures
    print("FAIL " + msg)
    failures += 1


def read_flat(path):
    with open(path, encoding="utf-8") as f:
        return re.sub(r"\s+", " ", f.read())


def main():
    readme_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "README.md")
    with open(readme_path, encoding="utf-8") as f:
        md = f.read()
    flat = re.sub(r"\s+", " ", md)

    # 1. every measured claim: exactly once in the README, present in its artifact of record.
    for claim, src, src_pat in CLAIMS:
        n = len(re.findall(claim, flat))
        if n != 1:
            fail(f"claim /{claim}/ appears {n} times in the README (want exactly 1)")
        path = os.path.join(ROOT, src)
        if not os.path.exists(path):
            fail(f"artifact {src} does not exist")
            continue
        if not re.search(src_pat, read_flat(path)):
            fail(f"artifact {src} does not contain /{src_pat}/ for claim /{claim}/")

    # 2. computed figures, recomputed from the artifacts' own strings.
    def once(phrase, why):
        n = flat.count(phrase)
        if n != 1:
            fail(f"'{phrase}' appears {n} times in the README (want exactly 1: {why})")

    def num(path, pat):
        return float(re.search(pat, read_flat(os.path.join(ROOT, path))).group(1))

    try:
        ll512 = num("bench/w1-gpu-perf-2026-11-19.md", r"(2260\.98) ± 78\.28")
        spread = num("bench/w1-gpu-perf-2026-11-19.md", r"2260\.98 ± (78\.28)")
        ll2048 = num("bench/w1-gpu-perf-2026-11-19.md", r"(2078\.44) ± 134\.32")
        ours512 = num("bench/rung1-recheck-2026-11-26.md", r"(2323) \(mean of 3")
        ours2048 = num("bench/rung1-recheck-2026-11-26.md", r"(2471) \(best 2476\)")
        if abs(ours512 - ll512) > spread:
            fail(f"pp512 {ours512} is outside llama.cpp's spread {ll512} ± {spread}, the README says inside")
        once("inside llama.cpp's own spread", "pp512 against llama-bench")
        once(f"{round((ours2048 / ll2048 - 1) * 100)}% ahead", f"pp2048 recomputed {ours2048}/{ll2048}")
        frag = num("bench/rung1-recheck-2026-11-26.md", r"mean (88\.4) over 9 runs")
        fp32 = num("bench/rung1-recheck-2026-11-26.md", r"mean (94\.8) \(94\.2 to 95\.3\)")
        once(f"{round((1 - frag / fp32) * 100)}% slower", f"plain decode recomputed 1-{frag}/{fp32}")
        big = num("bench/dense27b-2026-11-26.md", r"57,344 positions \| \| (46\.2) tok/s")
        small = num("bench/dense27b-2026-11-26.md", r"49,152 positions \| \| (37\.6) tok/s")
        once(f"costs {round((1 - small / big) * 100)}% of decode speed", f"recomputed 1-{small}/{big}")
    except Exception as e:  # a claim we cannot parse is a FAIL (never a pass)
        fail(f"a computed figure could not be recomputed: {e}")
    try:
        kv = read_flat(os.path.join(ROOT, "include/limbo/cuda/kv.hpp"))
        q4 = float(re.search(r"(\d+) B per position and layer\)", kv).group(1))
        f16 = float(re.search(r"against (\d+)", kv).group(1))
        pct = round(q4 / f16 * 100)
        if f"{pct}% of F16" not in flat:
            fail(f"the Q4 storage ratio is not stated as {pct}% of F16 (recomputed {q4}/{f16})")
    except Exception as e:
        fail(f"the Q4 storage ratio could not be recomputed: {e}")

    # 3. the CUDA battery size is counted from the tree, not remembered.
    n_tests = len(glob.glob(os.path.join(ROOT, "tests/cuda_*_test.cu")))
    if f"{n_tests} tests green" not in flat:
        fail(f"the README does not state {n_tests} tests green (counted tests/cuda_*_test.cu)")

    # 4. every path the README asserts exists — backticked spans and markdown link targets.
    tokens = re.findall(r"`([^`]+)`", md) + re.findall(r"\]\(([^)]+)\)", md)
    for token in tokens:
        t = token.strip()
        if t.startswith(("--", "http", "/")) or "$" in t or " " in t or t.startswith("build"):
            continue
        if "/" not in t and not re.search(r"\.(md|cpp|cu|hpp|cuh|py|sh|mk|json)$", t):
            continue
        if not re.match(r"^(docs|bench|src|tests|tools|scripts|include|mk|oracle|probe)/", t) and \
                t not in ("models.json", "THIRD_PARTY_NOTICES"):
            continue  # HTTP routes, function pairs and bare tool names are not repository paths
        t = re.sub(r":\d+(-\d+)?$", "", t)  # path:line citations
        if any(ch in t for ch in "*?["):
            if not glob.glob(os.path.join(ROOT, t)):
                fail(f"glob `{token}` matches nothing in the tree")
        elif not os.path.exists(os.path.join(ROOT, t)):
            fail(f"referenced path `{token}` does not exist")

    # 5. the honest sections are present.
    for heading in ("## 08 — Performance on the target machine",
                    "### Known limitations",
                    "## 09 — Validation"):
        if heading not in md:
            fail(f"the '{heading}' section is missing")
    if "not machine-gated" not in flat:
        fail("the timing disclosure (results are not machine-gated) is missing")

    print(f"readme_claims: {len(CLAIMS)} claims + arithmetic + links + sections -> "
          + ("FAIL" if failures else "all green"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
