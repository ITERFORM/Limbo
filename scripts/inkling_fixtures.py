#!/usr/bin/env python3
# Copyright (c) 2026 ITERFORM. All rights reserved.
"""Golden fixtures for the Inkling-2 chat template: rendered by the environment the template is written for.

    scripts/inkling_fixtures.py [--gguf MODEL.gguf] [--out bench/fixtures/chat-inkling]

The template carries `{% generation %}`, which plain jinja2 refuses -- it is a transformers extension.
The reference implementation is `AssistantTracker` in transformers/utils/chat_template_utils.py: it
renders the block's body UNCHANGED and records the character span of the assistant's turns beside it.
So this script renders with transformers' own compiled environment (`_compile_jinja_template`, the same
ImmutableSandboxedEnvironment with loopcontrols, a non-escaping `tojson`, `raise_exception` and
`strftime_now`), not with a hand-rolled one: the point is to compare the engine against the reference,
and an oracle built by the same guess as the implementation is no oracle at all.

Per case it writes <name>.json (the render context, plus `error` when the template must raise) and
<name>.expected.txt (the exact bytes). tests/chat_inkling_test.cpp renders the same contexts with
limbo::jinja and compares byte for byte. Python is only the reference here; the engine never runs it.

Reads the template from the GGUF's tokenizer.chat_template (the `gguf` package), or from --template FILE.
"""
import argparse
import json
import os
import struct
import sys

DEFAULT_GGUF = "/home/alca/models/inkling/Inkling-2-Mini-UD-IQ2_XXS.gguf"


def read_template_gguf(path):
    """The metadata key out of a GGUF header, with our own reader.

    The python `gguf` package builds its whole tensor table on open and raises on any type id its
    GGMLQuantizationType enum does not know -- it cannot open this file at all (type 42). The header
    before the tensor table is enough for a metadata key, and the format is stable, so read it here.
    """
    fixed = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

    def read_value(f, t):
        if t == 8:
            n = struct.unpack("<Q", f.read(8))[0]
            return f.read(n).decode("utf-8", "replace")
        if t == 9:
            et = struct.unpack("<I", f.read(4))[0]
            n = struct.unpack("<Q", f.read(8))[0]
            for _ in range(n):
                skip_value(f, et)
            return None
        fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<B", 10: "<Q", 11: "<q", 12: "<d"}[t]
        return struct.unpack(fmt, f.read(fixed[t]))[0]

    def skip_value(f, t):
        read_value(f, t)

    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            sys.exit(f"{path}: not a GGUF file")
        struct.unpack("<I", f.read(4))[0]          # version
        struct.unpack("<Q", f.read(8))[0]          # tensor count
        n_kv = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n_kv):
            klen = struct.unpack("<Q", f.read(8))[0]
            key = f.read(klen).decode("utf-8", "replace")
            vtype = struct.unpack("<I", f.read(4))[0]
            value = read_value(f, vtype)
            if key == "tokenizer.chat_template" and isinstance(value, str):
                return value
    sys.exit(f"{path}: no tokenizer.chat_template key")


def read_template(path):
    if path.endswith((".jinja", ".txt", ".template")):
        with open(path, encoding="utf-8") as f:
            return f.read()
    return read_template_gguf(path)


def compile_reference(template):
    """transformers' own compiled template: AssistantTracker + loopcontrols + its filters and globals."""
    from transformers.utils.chat_template_utils import _compile_jinja_template

    return _compile_jinja_template(template)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "units": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {"name": "add", "description": "Add two integers",
                     "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}}},
    },
]


def cases():
    """(name, context) pairs. Every branch of the template that a chat can reach is in here once."""
    yield "01_system_user", {
        "messages": [
            {"role": "system", "content": "You are a careful assistant."},
            {"role": "user", "content": "What is 84 * 3 / 2?"},
        ],
        "add_generation_prompt": True,
    }
    yield "02_user_only", {
        "messages": [{"role": "user", "content": "Hello."}],
        "add_generation_prompt": True,
    }
    yield "03_multi_turn", {
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Name a colour."},
            {"role": "assistant", "content": "Vermilion."},
            {"role": "user", "content": "Another."},
        ],
        "add_generation_prompt": True,
    }
    yield "04_reasoning_content", {
        "messages": [
            {"role": "user", "content": "Why is the sky blue?"},
            {"role": "assistant", "reasoning_content": "Rayleigh scattering.", "content": "Short wavelengths scatter more."},
            {"role": "user", "content": "And at sunset?"},
        ],
        "add_generation_prompt": True,
    }
    yield "05_tools_declared", {
        "messages": [{"role": "user", "content": "Weather in Berlin?"}],
        "tools": TOOLS,
        "add_generation_prompt": True,
    }
    yield "06_tool_call_and_result", {
        "messages": [
            {"role": "user", "content": "Weather in Berlin?"},
            {"role": "assistant", "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "get_weather", "arguments": {"city": "Berlin", "units": "metric"}}},
            ]},
            {"role": "tool", "name": "get_weather", "tool_call_id": "call_1", "content": "{\"temp_c\": 7}"},
            {"role": "user", "content": "In Fahrenheit?"},
        ],
        "tools": TOOLS,
        "add_generation_prompt": True,
    }
    yield "07_effort_string", {
        "messages": [{"role": "user", "content": "Think hard."}],
        "reasoning_effort": "high",
        "add_generation_prompt": True,
    }
    yield "08_effort_float", {
        "messages": [{"role": "user", "content": "Think a little."}],
        "reasoning_effort": 0.2,
        "add_generation_prompt": True,
    }
    yield "09_effort_zero", {
        "messages": [{"role": "user", "content": "Do not think."}],
        "reasoning_effort": "none",
        "add_generation_prompt": True,
    }
    yield "10_content_parts", {
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "Two parts: "},
                {"type": "input_text", "text": "second."},
            ]},
        ],
        "add_generation_prompt": True,
    }
    yield "11_image_part", {
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "What is in this picture?"},
                {"type": "image", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ]},
        ],
        "add_generation_prompt": True,
    }
    yield "12_system_only", {
        "messages": [{"role": "system", "content": "You exist."}],
        "add_generation_prompt": True,
    }
    yield "13_no_generation_prompt", {
        "messages": [{"role": "user", "content": "Scored turn."}],
        "add_generation_prompt": False,
    }
    yield "14_two_tool_calls", {
        "messages": [
            {"role": "user", "content": "Weather in Berlin and 2+2."},
            {"role": "assistant", "tool_calls": [
                {"id": "call_a", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Berlin"}}},
                {"id": "call_b", "type": "function", "function": {"name": "add", "arguments": {"a": 2, "b": 2}}},
            ]},
        ],
        "tools": TOOLS,
        "add_generation_prompt": True,
    }
    yield "15_empty_tools_list", {
        "messages": [{"role": "user", "content": "No tools today."}],
        "tools": [],
        "add_generation_prompt": True,
    }
    # ---- the template's own refusals: the reference raises, so the engine must too -----------------
    yield "90_error_unknown_effort", {
        "messages": [{"role": "user", "content": "hi"}],
        "reasoning_effort": "bogus",
        "add_generation_prompt": True,
    }
    yield "91_error_effort_range", {
        "messages": [{"role": "user", "content": "hi"}],
        "reasoning_effort": 1.5,
        "add_generation_prompt": True,
    }
    yield "92_error_audio_part", {
        "messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}]}],
        "add_generation_prompt": True,
    }
    yield "93_error_image_on_assistant", {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "image", "image_url": {"url": "x"}}]},
        ],
        "add_generation_prompt": True,
    }
    yield "94_error_unknown_part", {
        "messages": [{"role": "user", "content": [{"type": "video", "video_url": {"url": "x"}}]}],
        "add_generation_prompt": True,
    }
    yield "95_error_bad_tool_name", {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "tool_calls": [{"id": "c", "type": "function", "function": {"name": 7, "arguments": {}}}]},
        ],
        "add_generation_prompt": True,
    }
    yield "96_error_args_as_string", {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "tool_calls": [
                {"id": "c", "type": "function", "function": {"name": "add", "arguments": "{\"a\":1}"}}]},
        ],
        "add_generation_prompt": True,
    }
    yield "97_error_unknown_role", {
        "messages": [{"role": "developer", "content": "hi"}],
        "add_generation_prompt": True,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=DEFAULT_GGUF)
    ap.add_argument("--template", default=None, help="read the template from a file instead of a GGUF")
    ap.add_argument("--out", default="bench/fixtures/chat-inkling")
    args = ap.parse_args()

    template = read_template(args.template or args.gguf)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "template.jinja"), "w", encoding="utf-8") as f:
        f.write(template)

    env_template = compile_reference(template)
    import jinja2

    n_ok = n_err = 0
    for name, ctx in cases():
        # The template's own key set: everything the reference is given is what the engine is given.
        render_ctx = dict(ctx)
        try:
            out = env_template.render(**render_ctx)
            err = None
        except jinja2.exceptions.TemplateError as e:
            out, err = None, str(getattr(e, "message", e))
        except Exception as e:  # sandbox/type errors: the engine must refuse these too
            out, err = None, "*"
        rec = {"name": name, "context": render_ctx}
        if err is not None:
            rec["error"] = err
        with open(os.path.join(args.out, name + ".json"), "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=1, ensure_ascii=False)
            f.write("\n")
        exp = os.path.join(args.out, name + ".expected.txt")
        if err is None:
            with open(exp, "w", encoding="utf-8") as f:
                f.write(out)
            n_ok += 1
        else:
            if os.path.exists(exp):
                os.remove(exp)
            n_err += 1
    print(f"inkling_fixtures: {n_ok} rendered + {n_err} refusal cases -> {args.out} "
          f"(template {len(template)} chars)")


if __name__ == "__main__":
    main()
