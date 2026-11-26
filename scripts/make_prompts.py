#!/usr/bin/env python3
# Copyright (c) 2026 ITERFORM. All rights reserved.
"""Write the fixed prompt set under bench/prompts/.

PLAN.md §6: the oracle records greedy tokens and top-2 gaps for a fixed prompt set per rung — code,
prose, tool calls, long context. The set is written here so the bytes are visible and reproducible, and
it is committed: a capture is only comparable to a later capture if the prompts are the same bytes.

The format is the one oracle/oracle_dump.cpp reads: a line "@@ <name>" starts a prompt, everything up
to the next such line is its text verbatim. Every file ends with a newline.
"""
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench", "prompts")

SYSTEM = ("You are Ariadne, a coding agent on the operator's desktop. Be precise and brief. "
          "Use tools when they are cheaper than guessing.")

PROMPTS = {
    "code": SYSTEM + "\n\nReview this function and say what it does wrong:\n\n"
                     "```cpp\n"
                     "int parse_len(const char* s) {\n"
                     "    int n = 0;\n"
                     "    while (*s) { n = n * 10 + (*s - '0'); ++s; }\n"
                     "    return n;\n"
                     "}\n"
                     "```\n",
    "prose": SYSTEM + "\n\nIn two sentences, explain why a paged KV cache matters when the model fits "
                      "in VRAM but the context does not.\n",
    "tools": SYSTEM + "\n\n<|im_start|>user\nWhat is the weather in K\u00f6ln?<|im_end|>\n"
                      "<|im_start|>assistant\n<tool_call>\n"
                      "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"K\u00f6ln\", \"units\": \"metric\"}}\n"
                      "</tool_call><|im_end|>\n"
                      "<|im_start|>user\nNow check the file too.<|im_end|>\n"
                      "<|im_start|>assistant\n",
    "long": SYSTEM + "\n\nHere is the design note, then a question about it.\n\n" + "\n".join(
        f"Section {i}: the planner owns every budget. A non-resident expert is computed on the CPU from "
        f"pinned RAM when fewer than about ten rows read it in a step, and moved over PCIe when more do. "
        f"The same crossover decides a KV page. Currency is expected step-milliseconds saved per byte, "
        f"and routing counts decay with 1.5x hysteresis and per-layer quotas." for i in range(1, 61)
    ) + "\n\nQuestion: which lane does a verify window of six rows use for a missed expert, and why?\n",
}


def main():
    os.makedirs(ROOT, exist_ok=True)
    for name, text in PROMPTS.items():
        if not text.endswith("\n"):
            text += "\n"
        path = os.path.join(ROOT, f"{name}.txt")
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("@@ " + name + "\n" + text)
        print(f"{path}: {os.path.getsize(path)} bytes")


if __name__ == "__main__":
    main()
