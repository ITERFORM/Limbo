#!/usr/bin/env python3
# Copyright (c) 2026 ITERFORM. All rights reserved.
"""Golden fixtures for the chat renderer: the model's own chat template, rendered by Python jinja2.

    scripts/chat_fixtures.py [--gguf MODEL.gguf] [--out bench/fixtures/chat]

Reads tokenizer.chat_template from the GGUF, writes it to <out>/template.jinja, renders a fixed set of
conversations with jinja2 and writes, per case, <name>.json (the render context and flags) and
<name>.expected.txt (the exact bytes, or nothing when the template raises). tests/chat_test.cpp renders
the same contexts with limbo::jinja and compares byte for byte.

The jinja2 environment is HuggingFace transformers' (ImmutableSandboxedEnvironment, trim_blocks,
lstrip_blocks, loopcontrols, a `tojson` that keeps key order and does not escape non-ASCII, and a
`raise_exception` that raises TemplateError) because that is the environment the model's template is
written for. jinja_cases.json holds small templates for the interpreter itself.

Python is only the reference here; the engine never runs it.
"""
import argparse
import hashlib
import json
import os
import sys

import jinja2
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

DEFAULT_GGUF = (
    "/home/alca/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-MTP-GGUF/snapshots/"
    "5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
)


def read_template(path):
    try:
        from gguf import GGUFReader
    except ImportError:
        sys.exit("chat_fixtures: the python `gguf` package is needed to read the template")
    r = GGUFReader(path)
    f = r.fields["tokenizer.chat_template"]
    return bytes(f.parts[f.data[0]]).decode("utf-8")


def make_env(trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=False):
    env = ImmutableSandboxedEnvironment(
        trim_blocks=trim_blocks,
        lstrip_blocks=lstrip_blocks,
        keep_trailing_newline=keep_trailing_newline,
        extensions=[loopcontrols],
    )

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    return env


def render(env, source, context):
    """-> (text, None) or (None, error). raise_exception errors carry their message; others are '*'."""
    try:
        return env.from_string(source).render(**context), None
    except jinja2.exceptions.TemplateError as e:
        if type(e) is jinja2.exceptions.TemplateError:
            return None, str(e.message)
        return None, "*"
    except Exception:  # sandbox, type and key errors: the interpreter must also refuse these
        return None, "*"


def normalize(messages):
    """What chat::normalize_messages does: OpenAI tool_calls carry `arguments` as a JSON string."""
    out = []
    for m in messages:
        m = json.loads(json.dumps(m))
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", tc)
            if isinstance(fn.get("arguments"), str):
                try:
                    v = json.loads(fn["arguments"])
                except ValueError:
                    continue
                if isinstance(v, dict):
                    fn["arguments"] = v
        out.append(m)
    return out


# ---- the conversations ------------------------------------------------------------------------------

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city. Returns <b>JSON</b> & uses 'quotes'.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. München or 東京"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}
SEARCH = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "default": 5},
                "fresh": {"type": "boolean"},
                "domains": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["query"],
        },
    },
}


def u(text):
    return {"role": "user", "content": text}


def a(text, **kw):
    m = {"role": "assistant", "content": text}
    m.update(kw)
    return m


def call(name, args, cid="call_1"):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}


def tool_msg(text, cid="call_1"):
    return {"role": "tool", "tool_call_id": cid, "content": text}


CASES = []


def case(name, note, messages, normalize_args=False, **ctx):
    c = {"messages": messages}
    c.update(ctx)
    CASES.append((name, note, c, normalize_args))


case("01_system_user", "system + user, generation prompt (thinking default)",
     [{"role": "system", "content": "You are terse."}, u("Say hello in five words.")], add_generation_prompt=True)
case("02_user_only", "no system message", [u("Hi")], add_generation_prompt=True)
case("03_multi_turn", "three user turns; earlier assistant turns lose their think block",
     [{"role": "system", "content": "Be brief."}, u("What is 2+2?"), a("4"), u("And 3+3?"), a("6"), u("Thanks.")],
     add_generation_prompt=True)
case("04_reasoning_content_dropped", "reasoning_content on an assistant turn before the last user query is dropped",
     [u("Think about 17*3."), a("51", reasoning_content="17*3 = 51."), u("Again for 19*3.")],
     add_generation_prompt=True)
case("05_reasoning_content_kept", "reasoning_content after the last user query is kept (no generation prompt)",
     [u("Think about 17*3."), a("51", reasoning_content="  17*3 = 51.\n\n")], add_generation_prompt=False)
case("06_inline_think_in_content", "the template splits <think>..</think> out of assistant content",
     [u("q1"), a("<think>\nsome thoughts\n</think>\n\nanswer one"), u("q2"),
      a("<think>\nmore thoughts\n</think>\n\nanswer two")], add_generation_prompt=True)
case("07_enable_thinking_true", "enable_thinking true", [u("Hi")], add_generation_prompt=True, enable_thinking=True)
case("08_enable_thinking_false", "enable_thinking false: empty think block in the generation prompt",
     [u("Hi")], add_generation_prompt=True, enable_thinking=False)
case("09_no_generation_prompt", "add_generation_prompt false", [u("Hi"), a("Hello!")], add_generation_prompt=False)
case("10_no_generation_prompt_thinking_false", "add_generation_prompt false with enable_thinking false",
     [u("Hi")], add_generation_prompt=False, enable_thinking=False)
case("11_preserve_thinking", "preserve_thinking keeps the think block of earlier turns",
     [u("q1"), a("answer one", reasoning_content="thought one"), u("q2")],
     add_generation_prompt=True, preserve_thinking=True)
case("12_tools_system_user", "tool definitions merged with the system message",
     [{"role": "system", "content": "You are a helpful agent."}, u("Weather in Berlin?")],
     tools=[WEATHER, SEARCH], add_generation_prompt=True)
case("13_tools_no_system", "tool definitions, no system message", [u("Weather in Berlin?")],
     tools=[WEATHER], add_generation_prompt=True, enable_thinking=False)
case("14_tool_call_roundtrip", "assistant tool_calls with typed arguments, then a tool result, then more",
     [u("Search for rust news and tell me the weather."),
      a("Let me look.", reasoning_content="Two lookups.", tool_calls=[
          call("web_search", {"query": "rust news", "max_results": 3, "fresh": True, "domains": ["a.com", "b.org"]}),
          call("get_weather", {"city": "München", "unit": "celsius"}, "call_2")]),
      tool_msg("headline one\nheadline two", "call_1"), tool_msg('{"temp": 11.5}', "call_2"),
      a("Rust 1.9 is out and Munich is 11.5 C.")],
     tools=[WEATHER, SEARCH], add_generation_prompt=True)
case("15_tool_call_no_content", "tool call with empty content right after the think block",
     [u("Weather?"), a("", reasoning_content="Call the tool.", tool_calls=[call("get_weather", {"city": "Paris"})]),
      tool_msg("sunny")],
     tools=[WEATHER], add_generation_prompt=True)
case("16_tool_call_null_content", "content null (OpenAI shape) with a tool call",
     [u("Weather?"), {"role": "assistant", "content": None, "tool_calls": [call("get_weather", {"city": "Rome"})]},
      tool_msg("rainy")],
     tools=[WEATHER], add_generation_prompt=True, enable_thinking=False)
case("17_argument_types", "argument values of every JSON type, floats and unicode",
     [u("go"), a("", tool_calls=[call("web_search", {
         "query": "ünïcode 日本語 \U0001F600", "n": 5, "x": 1.5, "tiny": 1e-07, "big": 1e22, "whole": 3.0,
         "neg": -2, "flag": False, "nothing": None, "obj": {"k": [1, 2.5, None, True], "s": "a\"b\\c\n"},
         "list": ["x", 2]})])],
     tools=[SEARCH], add_generation_prompt=False)
case("18_openai_string_arguments", "OpenAI wire shape: arguments is a JSON string (chat normalises it)",
     [u("Weather?"), a("", tool_calls=[
         {"id": "call_9", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Oslo\", \"unit\": \"celsius\"}"}}]),
      tool_msg("snow", "call_9")],
     normalize_args=True, tools=[WEATHER], add_generation_prompt=True)
case("19_developer_role", "developer message treated as system",
     [{"role": "developer", "content": "Dev rules."}, u("Hi")], add_generation_prompt=True)
case("20_two_system_messages", "two leading system messages are merged",
     [{"role": "system", "content": "First."}, {"role": "system", "content": "Second."}, u("Hi")],
     add_generation_prompt=True)
case("21_content_parts", "content as a list of text parts",
     [{"role": "system", "content": [{"type": "text", "text": "Sys part."}]},
      {"role": "user", "content": [{"type": "text", "text": "Hello "}, {"type": "text", "text": "world"}]}],
     add_generation_prompt=True)
case("22_trim_unicode_space", "content is trimmed with Python's strip (NBSP, ideographic space, newlines)",
     [u(" 　 padded \n\n"), a("\n answer  \n"), u("  next  ")], add_generation_prompt=True)
case("23_template_syntax_in_content", "braces and tags in user content stay verbatim",
     [u("{{ 7*7 }} {% if x %}y{% endif %} <|im_end|>"), a("<tool_response> not a response")],
     add_generation_prompt=False)
case("24_tool_results_in_user_turn", "a user message that is a tool_response does not count as a query",
     [u("start"), a("calling", tool_calls=[call("get_weather", {"city": "A"})]),
      u("<tool_response>\nhot\n</tool_response>")],
     tools=[WEATHER], add_generation_prompt=True)
case("25_tool_then_user", "tool results followed by a new user query",
     [u("q"), a("c", reasoning_content="r", tool_calls=[call("get_weather", {"city": "B"})]), tool_msg("t"),
      a("done", reasoning_content="r2"), u("next")],
     tools=[WEATHER], add_generation_prompt=True)
case("26_empty_tools_list", "an empty tools list renders no tool block", [u("Hi")], tools=[],
     add_generation_prompt=True)
case("27_assistant_last_no_prompt", "conversation ends with an assistant turn", [u("Hi"), a("Hello", reasoning_content="greet")],
     add_generation_prompt=False)
case("28_tool_content_parts", "tool message content as parts",
     [u("q"), a("", tool_calls=[call("get_weather", {"city": "C"})]),
      {"role": "tool", "content": [{"type": "text", "text": "part one "}, {"type": "text", "text": "part two"}]}],
     tools=[WEATHER], add_generation_prompt=True)

ERROR_CASES = [
    ("90_error_no_messages", "empty messages list", [], "No messages provided."),
    ("91_error_system_image", "system content with an image",
     [{"role": "system", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}, u("Hi")],
     "System message cannot contain images."),
    ("92_error_bad_content_type", "content of an unexpected type", [{"role": "user", "content": 5}],
     "Unexpected content type."),
    ("93_error_bad_part", "a content part that is not text/image/video", [{"role": "user", "content": [{"foo": 1}]}],
     "Unexpected item type in content."),
]

# ---- small templates for the interpreter -----------------------------------------------------------

JINJA_CASES = []


def jc(name, template, context=None, trim=True, lstrip=True, keep_nl=False):
    JINJA_CASES.append({"name": name, "template": template, "context": context or {},
                        "trim_blocks": trim, "lstrip_blocks": lstrip, "keep_trailing_newline": keep_nl})


jc("literals", "{{ 1 }}|{{ 1.0 }}|{{ 1.5e3 }}|{{ 'a' }}|{{ \"b\" }}|{{ true }}|{{ false }}|{{ none }}|{{ True }}|{{ None }}")
jc("string_escapes", "{{ 'a\\nb\\tc\\\\d\\'e\\x41\\u00e9' }}")
jc("arith", "{{ 1+2 }} {{ 7-9 }} {{ 3*4 }} {{ 7/2 }} {{ 7//2 }} {{ -7//2 }} {{ 7%3 }} {{ -7%3 }} {{ 2**10 }} {{ 2**-1 }} {{ 1+2*3 }} {{ (1+2)*3 }} {{ -3|abs }}")
jc("float_repr", "{{ 0.1+0.2 }} {{ 1e16 }} {{ 1e15 }} {{ 1e-5 }} {{ 0.0001 }} {{ 123456789.0 }} {{ 1.0*3 }} {{ 2.5e-10 }} {{ 1/3 }}")
jc("bool_arith", "{{ true + 1 }} {{ true == 1 }} {{ 1 == 1.0 }} {{ 'a' == 'a' }} {{ 'a' != 'b' }}")
jc("concat", "{{ 'a' ~ 1 ~ none ~ true ~ 1.5 }} {{ 'x' + 'y' }} {{ [1] + [2] }} {{ 'ab' * 2 }}")
jc("compare", "{{ 1 < 2 }} {{ 2 <= 2 }} {{ 'a' < 'b' }} {{ 1 < 2 < 3 }} {{ 1 < 3 < 2 }} {{ [1,2] < [1,3] }}")
jc("logic", "{{ 0 or 'x' }} {{ 1 and 'y' }} {{ '' or none }} {{ not 0 }} {{ not(1 and 0) }} {{ 1 if 0 else 2 }} {{ 3 if 1 }}")
jc("conditional_nested", "{{ 'a' if x else 'b' if y else 'c' }}", {"x": False, "y": True})
jc("in_operator", "{{ 'a' in 'abc' }} {{ 'z' not in 'abc' }} {{ 2 in [1,2] }} {{ 'k' in d }} {{ 'q' in d }}", {"d": {"k": 1}})
jc("list_dict_literals", "{{ [1, 'a', [2]] }} {{ {'a': 1, 'b': [1,2]} }}")
jc("repr_strings", "{{ ['it\\'s', 'say \"hi\"', 'new\\nline'] }}")
jc("attr_and_index", "{{ d.a }} {{ d['b'] }} {{ l[0] }} {{ l[-1] }} {{ l[5] is undefined }} {{ d.zz is undefined }} {{ s[1] }}",
   {"d": {"a": "A", "b": "B"}, "l": [10, 20, 30], "s": "héllo"})
jc("slices", "{{ l[1:] }} {{ l[:-1] }} {{ l[::-1] }} {{ l[::2] }} {{ l[-2:] }} {{ l[1:3] }} {{ s[1:3] }} {{ s[::-1] }} {{ l[5:] }}",
   {"l": [0, 1, 2, 3, 4], "s": "héllo wörld"})
jc("if_elif_else", "{% if a %}A{% elif b %}B{% else %}C{% endif %}|{% if not a and not b %}none{% endif %}",
   {"a": 0, "b": 1})
jc("for_basic", "{% for x in l %}{{ loop.index }}/{{ loop.length }}:{{ x }}{% if not loop.last %},{% endif %}{% endfor %}",
   {"l": ["a", "b", "c"]})
jc("for_loop_vars", "{% for x in l %}[{{ loop.index0 }} {{ loop.revindex }} {{ loop.revindex0 }} {{ loop.first }} {{ loop.last }} {{ loop.previtem }} {{ loop.nextitem }}]{% endfor %}",
   {"l": [1, 2, 3]})
jc("for_else", "{% for x in l %}{{ x }}{% else %}empty{% endfor %}|{% for x in none_here %}{{ x }}{% else %}undef{% endfor %}", {"l": []})
jc("for_if_filter", "{% for x in l if x > 1 %}{{ x }}{{ loop.index }}{% endfor %}", {"l": [1, 2, 3]})
jc("for_dict", "{% for k in d %}{{ k }}={{ d[k] }};{% endfor %}{% for k, v in d.items() %}{{ k }}:{{ v }};{% endfor %}", {"d": {"b": 2, "a": 1}})
jc("for_tuple_unpack", "{% for a, b in pairs %}{{ a }}-{{ b }} {% endfor %}", {"pairs": [[1, 2], [3, 4]]})
jc("for_break_continue", "{% for x in l %}{% if x == 2 %}{% continue %}{% endif %}{% if x == 4 %}{% break %}{% endif %}{{ x }}{% endfor %}",
   {"l": [1, 2, 3, 4, 5]})
jc("for_string", "{% for c in s %}[{{ c }}]{% endfor %}", {"s": "aé日"})
jc("for_scope", "{% set x = 1 %}{% for i in l %}{% set x = i %}{% endfor %}{{ x }}", {"l": [5, 6]})
jc("namespace", "{% set ns = namespace(n=0, s='') %}{% for x in l %}{% set ns.n = ns.n + x %}{% set ns.s = ns.s ~ x %}{% endfor %}{{ ns.n }} {{ ns.s }}",
   {"l": [1, 2, 3]})
jc("set_tuple", "{% set a, b = 1, 2 %}{{ a }}{{ b }}{% set c = [1,2] %}{{ c }}")
jc("set_block", "{% set body %}in {{ x }} block{% endset %}[{{ body }}]", {"x": "set"})
jc("if_scope", "{% if true %}{% set y = 5 %}{% endif %}{{ y }}")
jc("macro_basic", "{% macro m(a, b=2, c='z') %}{{ a }}{{ b }}{{ c }}{% endmacro %}{{ m(1) }} {{ m(1, 3) }} {{ m(1, c='q') }} {{ m(a=9, b=8) }}")
jc("macro_scope", "{% set g = 'G' %}{% macro m() %}{{ g }}{{ x }}{% endmacro %}{{ m() }}", {"x": "X"})
jc("macro_recursion", "{% macro fact(n) %}{% if n <= 1 %}1{% else %}{{ n * (fact(n-1)|int) }}{% endif %}{% endmacro %}{{ fact(5) }}")
jc("filter_trim_len", "[{{ '  a b  '|trim }}] {{ 'héllo'|length }} {{ l|length }} {{ d|length }} {{ missing|length }}", {"l": [1, 2], "d": {"a": 1}})
jc("filter_chain_precedence", "{{ l|length - 1 }} {{ l|join(',')|upper }} {{ 'a'|replace('a', 'bb')|length }} {{ -5|abs|string }}", {"l": ["x", "y"]})
jc("filter_default", "{{ missing|default('d') }} {{ none|default('n') }} {{ none|default('n', true) }} {{ ''|default('e', true) }} {{ 0|default(7, true) }} {{ x|d('xd') }}")
jc("filter_join", "{{ [1,2,3]|join }} {{ [1,2,3]|join('-') }} {{ ['a','b']|join(', ') }} {{ items|join('/', attribute='n') }}", {"items": [{"n": "p"}, {"n": "q"}]})
jc("filter_string_forms", "{{ 5|string }} {{ 1.5|string }} {{ true|string }} {{ none|string }} {{ [1,'a']|string }} {{ d|string }}", {"d": {"k": "v"}})
jc("filter_tojson", "{{ d|tojson }} {{ l|tojson }} {{ s|tojson }} {{ 1.0|tojson }} {{ none|tojson }} {{ true|tojson }}",
   {"d": {"b": 1, "a": [1, 2.5, None, True], "c": {"x": "ü<>&'"}}, "l": [], "s": "tab\there \"q\" \\   \x01"})
jc("filter_tojson_indent", "{{ d|tojson(indent=2) }}", {"d": {"a": [1, 2], "b": {"c": None}, "e": [], "f": {}}})
jc("filter_misc", "{{ 'abc'|upper }} {{ 'ABC'|lower }} {{ 'hello world'|title }} {{ 'hello'|capitalize }} {{ l|first }} {{ l|last }} {{ l|list }} {{ l|reverse|list }} {{ 'abc'|reverse }} {{ '3'|int + 1 }} {{ 'x'|int(5) }} {{ '2.5'|float }} {{ [3,1,2]|sort }} {{ [1,1,2]|unique|list }} {{ [1,2,3]|sum }} {{ [3,1,2]|max }}", {"l": [1, 2, 3]})
jc("filter_map_select", "{{ users|map(attribute='name')|join(',') }} {{ users|selectattr('ok')|map(attribute='name')|join(',') }} {{ users|rejectattr('ok')|map(attribute='name')|join(',') }} {{ [1,2,3,4]|select('even')|join(',') }}",
   {"users": [{"name": "a", "ok": True}, {"name": "b", "ok": False}, {"name": "c", "ok": True}]})
jc("filter_items_indent_escape", "{% for k, v in d|items %}{{ k }}={{ v }}{% endfor %} {{ '<a href=\"x\">&</a>'|e }} {{ 'a\\nb\\nc'|indent(2) }}", {"d": {"k": 1}})
jc("tests_basic", "{{ x is defined }} {{ y is defined }} {{ y is undefined }} {{ n is none }} {{ 's' is string }} {{ 1 is number }} {{ 1 is integer }} {{ 1.5 is float }} {{ true is boolean }} {{ l is iterable }} {{ d is mapping }} {{ l is mapping }} {{ 's' is sequence }} {{ l is sequence }} {{ d is sequence }} {{ 5 is sequence }} {{ 1 is not string }}",
   {"x": 1, "n": None, "l": [1], "d": {"a": 1}})
jc("tests_misc", "{{ 4 is even }} {{ 3 is odd }} {{ 6 is divisibleby 3 }} {{ 2 is eq 2 }} {{ 2 is ne 3 }} {{ 1 is lt 2 }} {{ 'a' is in ['a'] }} {{ true is true }} {{ false is false }} {{ 1 is true }} {{ 'ab' is lower }} {{ 'AB' is upper }}")
jc("undefined_behaviour", "[{{ missing }}] [{{ missing.x if false else 'ok' }}] {{ missing is none }} {% for i in missing %}x{% else %}none{% endfor %} {{ missing == missing }} {{ not missing }}")
jc("methods_str", "{{ 'abc'.startswith('a') }} {{ 'abc'.endswith('bc') }} {{ 'a,b,c'.split(',') }} {{ ' a  b '.split() }} {{ 'a,b,c'.split(',', 1) }} {{ 'xxaxx'.strip('x') }} {{ 'xxaxx'.lstrip('x') }} {{ 'xxaxx'.rstrip('x') }} {{ '\\n\\ta \\n'.strip() }} {{ 'aXbX'.replace('X', '-') }} {{ ','.join(['a','b']) }} {{ 'abc'.upper() }} {{ 'abcabc'.count('b') }} {{ 'abc'.find('c') }} {{ 'abc'.find('z') }}")
jc("methods_split_chain", "{{ c.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') }}|{{ c.split('</think>')[-1].lstrip('\\n') }}",
   {"c": "<think>\n\nreasoning here\n</think>\n\nfinal answer"})
jc("methods_dict", "{{ d.keys()|list }} {{ d.values()|list }} {{ d.get('a') }} {{ d.get('z', 'dflt') }} {{ d.get('z') }}", {"d": {"a": 1, "b": 2}})
jc("unicode_strip", "[{{ s|trim }}] [{{ s.lstrip() }}] [{{ s.rstrip() }}]", {"s": "   x y 　\n\x1f"})
jc("ws_ctrl_minus", "a  {%- if true -%}  b  {%- endif -%}  c\n  {{- 'd' -}}  \n e")
jc("ws_ctrl_default", "line1\n{% if true %}\nline2\n{% endif %}\nline3\n")
jc("ws_ctrl_default_no_trim", "line1\n{% if true %}\nline2\n{% endif %}\nline3\n", trim=False, lstrip=False)
jc("ws_lstrip_blocks", "  {% if true %}\n    x\n  {% endif %}\n  {# c #}\ny", lstrip=True, trim=False)
jc("ws_lstrip_plus", "  {%+ if true %}x{% endif %}\n    {%+ if true %}y{% endif %}", lstrip=True, trim=False)
jc("ws_lstrip_var_untouched", "  {{ 'v' }}\n  {% if true %}\nz{% endif %}", lstrip=True, trim=True)
jc("ws_comments", "a{# comment #}b{#- c -#}  d {# multi\nline #}\ne", trim=True, lstrip=True)
jc("ws_trailing_newline_dropped", "text\n")
jc("ws_trailing_newline_kept", "text\n", keep_nl=True)
jc("ws_crlf", "a\r\nb\rc\n{% if true %}\r\nd{% endif %}")
jc("braces_in_dict_literal", "{{ {'a': {'b': 1}} }}|{{ {'k': '}}'} }}|{{ '%}' }}")
jc("loop_nested", "{% for i in a %}{% for j in b %}{{ i }}{{ j }}{{ loop.index }} {% endfor %}{{ loop.index }}|{% endfor %}", {"a": [1, 2], "b": ["x", "y"]})
jc("range_dict_globals", "{{ range(3)|list }} {{ range(1, 6, 2)|list }} {{ dict(a=1, b=2) }}")
jc("string_adjacent_literals", "{{ 'a' 'b' }}")
jc("multiline_expr", "{{ [1,\n 2,\n 3]|sum }}{%- set x = {\n 'a': 1\n} %}{{ x.a }}")

ERROR_JINJA = [
    ("err_undefined_attr", "{{ missing.x }}"),
    ("err_undefined_arith", "{{ missing + 1 }}"),
    ("err_undefined_call", "{{ missing() }}"),
    ("err_unknown_filter", "{{ 1|nosuchfilter }}"),
    ("err_unknown_test", "{{ 1 is nosuchtest }}"),
    ("err_int_str_add", "{{ 1 + 'a' }}"),
    ("err_div_zero", "{{ 1 / 0 }}"),
    ("err_str_in_int", "{{ 1 in 'abc' }}"),
    ("err_unclosed_if", "{% if true %}x"),
    ("err_stray_endfor", "{% endfor %}"),
    ("err_unclosed_var", "{{ 1 "),
    ("err_bad_syntax", "{{ 1 + }}"),
    ("err_include", "{% include 'x' %}"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=DEFAULT_GGUF)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "bench", "fixtures", "chat"))
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    tmpl = read_template(args.gguf)
    with open(os.path.join(out, "template.jinja"), "w", encoding="utf-8", newline="") as f:
        f.write(tmpl)
    with open(os.path.join(out, "template.sha256"), "w") as f:
        f.write(hashlib.sha256(tmpl.encode("utf-8")).hexdigest() + "\n")

    env = make_env()
    n = 0
    for name, note, ctx, norm in CASES:
        c = json.loads(json.dumps(ctx))
        rctx = dict(c)
        if norm:
            rctx["messages"] = normalize(c["messages"])
        text, err = render(env, tmpl, rctx)
        if err is not None:
            sys.exit(f"case {name}: unexpected template error {err!r}")
        with open(os.path.join(out, name + ".json"), "w", encoding="utf-8") as f:
            json.dump({"name": name, "note": note, "normalize": norm, "context": c, "error": None}, f, ensure_ascii=False, indent=2)
            f.write("\n")
        with open(os.path.join(out, name + ".expected.txt"), "wb") as f:
            f.write(text.encode("utf-8"))
        n += 1
    for name, note, msgs, want in ERROR_CASES:
        c = {"messages": msgs, "add_generation_prompt": True}
        text, err = render(env, tmpl, c)
        if err != want:
            sys.exit(f"case {name}: expected error {want!r}, got {err!r} / {text!r}")
        with open(os.path.join(out, name + ".json"), "w", encoding="utf-8") as f:
            json.dump({"name": name, "note": note, "normalize": False, "context": c, "error": want}, f, ensure_ascii=False, indent=2)
            f.write("\n")
        n += 1

    # The interpreter cases. Each is rendered under its own whitespace options.
    cases = []
    for jc_ in JINJA_CASES:
        e = make_env(jc_["trim_blocks"], jc_["lstrip_blocks"], jc_["keep_trailing_newline"])
        text, err = render(e, jc_["template"], jc_["context"])
        if err is not None:
            sys.exit(f"jinja case {jc_['name']}: unexpected error {err!r}")
        d = dict(jc_)
        d["expected"] = text
        d["error"] = None
        cases.append(d)
    for name, template in ERROR_JINJA:
        text, err = render(make_env(), template, {})
        if err is None:
            sys.exit(f"jinja case {name}: expected an error, rendered {text!r}")
        cases.append({"name": name, "template": template, "context": {}, "trim_blocks": True, "lstrip_blocks": True,
                      "keep_trailing_newline": False, "expected": None, "error": "*"})
    with open(os.path.join(out, "jinja_cases.json"), "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"chat_fixtures: template {len(tmpl)} bytes (sha256 {hashlib.sha256(tmpl.encode()).hexdigest()[:16]}), "
          f"{n} chat fixtures, {len(cases)} interpreter cases -> {out}")


if __name__ == "__main__":
    main()
