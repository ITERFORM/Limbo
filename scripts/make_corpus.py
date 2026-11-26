#!/usr/bin/env python3
# Copyright (c) 2026 ITERFORM. All rights reserved.
"""Write the tokenizer corpus under bench/corpus/.

The corpus is the falsification set for the W0 gate: scripts/tok_compare.sh tokenizes every file with
both Limbo and llama-tokenize and diffs the id lists. It is written here, not by hand, so the exact
bytes are visible and reproducible. Every file ends without a trailing newline unless the newline is
the point of the case.
"""
import os
import random

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench", "corpus")

FILES = {
    # Ordinary prose, contractions in every case, digits split one by one, punctuation runs.
    "prose.txt": (
        "The quick brown fox jumps over the lazy dog. It's 2026, and we're still arguing about\n"
        "tokenizers. DON'T shout: he'll fix it, they've done it, you'd think so, I'M sure.\n"
        "Numbers: 0 1 2 42 1234 3.14159 1,000,000 0x1F 12.5% $19.99 -273.15 1e-9\n"
        "Abbreviations, e.g. i.e. etc. Quotes: \"double\", 'single', `backtick`.\n"
        "Trailing spaces follow   \n"
        "and a tab\tseparated\trow.\n"
    ),
    # Code: indentation, operators, string literals, comments.
    "code.txt": (
        "#include <cstdint>\n"
        "namespace limbo::tok {\n"
        "struct Vocab { std::vector<std::string_view> tokens; };\n"
        "int f(int x) { return x * 2 + 1; }  // 2 + 1 = 3\n"
        "def g(n):\n"
        "    return [i ** 2 for i in range(n) if i % 2 == 0]\n"
        "let re = /^[a-z0-9_-]+$/gi;\n"
        "SELECT id, name FROM users WHERE age >= 18 AND name LIKE 'A%';\n"
        "git commit -m \"fix: don't drop the tail\"\n"
    ),
    # Tool-call shaped payloads: what Ariadne actually sends.
    "tools.txt": (
        "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"K\xf6ln\", \"units\": \"metric\"}}\n"
        "<tool_call>\n"
        "{\"name\": \"read_file\", \"arguments\": {\"path\": \"/home/alca/projects/limbo-engine\"}}\n"
        "</tool_call>\n"
        "<|im_start|>system\nYou are Ariadne.<|im_end|>\n"
        "<|im_start|>user\nrun it<|im_end|>\n"
        "<|im_start|>assistant\n<tool_call>\n{\"name\": \"job\", \"arguments\": {\"cmd\": \"make test\"}}\n"
        "</tool_call><|im_end|>\n"
        "{\"a\": [1, 2, 3], \"b\": {\"c\": null}, \"d\": true, \"e\": 1.5e-3, \"f\": \"\\n\\t\\\\\"}\n"
    ),
    # Scripts, marks, whitespace that is not ASCII, emoji, and a mix that forces every alternative.
    "unicode.txt": (
        "\u65e5\u672c\u8a9e\u306e\u30c6\u30ad\u30b9\u30c8\u3067\u3059\u3002\u4e2d\u6587\u6d4b\u8bd5\u3002\n"
        "\u0410\u0431\u0432\u0433\u0434 \u0438 \u0435\u0449\u0451 \u0416\u0443\u043a.\n"
        "Devanagari: \u0928\u092e\u0938\u094d\u0924\u0947 \u0926\u0941\u0928\u093f\u092f\u093e\n"
        "Arabic: \u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645\n"
        "Thai: \u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35\u0e04\u0e23\u0e31\u0e1a\n"
        "Korean: \uc548\ub155\ud558\uc138\uc694\n"
        "Combining: e\u0301 a\u0300 n\u0303 o\u0308 u\u0304\n"
        "Emoji: \U0001f600\U0001f469\u200d\U0001f4bb\u2764\ufe0f\U0001f1e9\U0001f1ea\n"
        "Symbols: \u00bd \u2167 \u00b2 \u2075 \u20ac \u00a3 \u2192 \u221e \u2211\n"
        "Spaces: a\u00a0b c\u2009d e\u2003f g\u3000h\n"
        "Soft hyphen and ZWSP: a\u00adb c\u200bd\n"
    ),
    # Boundaries: empty lines, whitespace-only lines, CRLF, a lone newline at the end, one long run.
    "edge.txt": (
        "\n"
        "\n"
        "   \n"
        "\t\n"
        "a\r\nb\r\n\r\n"
        "x" + " " * 40 + "y\n"
        "no trailing newline at the end of this line"
    ),
}


def main():
    os.makedirs(ROOT, exist_ok=True)
    for name, text in FILES.items():
        path = os.path.join(ROOT, name)
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        print(f"{path}: {len(text.encode())} bytes")

    # Deterministic random corpora: codepoints drawn from the pools the pattern distinguishes, so a
    # wrong class table or a wrong alternative order shows up as an id difference, not as prose that
    # happens to be easy. Seeds are fixed, so the files are reproducible.
    pools = [
        [(0x20, 0x7E, 60), (0x09, 0x0D, 4), (0xA0, 0xFF, 6)],
        [(0x300, 0x36F, 8), (0x370, 0x3FF, 6), (0x400, 0x4FF, 6), (0x590, 0x5FF, 4)],
        [(0x900, 0x97F, 5), (0xE00, 0xE7F, 4), (0x3040, 0x30FF, 4), (0x4E00, 0x4E80, 5)],
        [(0x1F300, 0x1F3FF, 4), (0x1F400, 0x1F4FF, 3), (0x2000, 0x206F, 6), (0x20A0, 0x20BF, 3)],
    ]
    rng = random.Random(20261119)
    for i, pool in enumerate(pools):
        out = []
        while sum(len(c.encode()) for c in out) < 24000:
            lo, hi, weight = rng.choices(pool, weights=[p[2] for p in pool])[0]
            cp = rng.randint(lo, hi)
            if 0xD800 <= cp <= 0xDFFF:
                continue
            out.append(chr(cp))
        path = os.path.join(ROOT, f"random{i + 1}.txt")
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("".join(out))
        print(f"{path}: {os.path.getsize(path)} bytes")

    # A long file: deep merge chains and enough length that a quadratic merge loop would be visible.
    words = ("tokenizer byte-level BPE merges residency planner PCIe Gen3 lane decode prefill "
             "MoE expert page table checkpoint session tape Ariadne Limbo Strata kindi modchip").split()
    out = []
    while sum(len(w) + 1 for w in out) < 200000:
        out.append(rng.choice(words))
        if len(out) % 17 == 0:
            out.append("\n")
    path = os.path.join(ROOT, "long.txt")
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(" ".join(out))
    print(f"{path}: {os.path.getsize(path)} bytes")


if __name__ == "__main__":
    main()
