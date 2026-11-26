// Copyright (c) 2026 ITERFORM. All rights reserved.
// tests/chat_test.cpp — chat rendering, the Jinja subset, JSON, and the output parser. No model needed.
//
//  - every golden fixture in bench/fixtures/chat (scripts/chat_fixtures.py renders them with Python
//    jinja2 from the model's own template) is rendered by limbo::jinja and compared byte for byte;
//  - jinja_cases.json: small templates for the interpreter itself, same reference;
//  - the output parser (reasoning / content / tool calls), whole and fed in arbitrary pieces;
//  - the parse -> re-render round trip the tape depends on.
//
// LIMBO_CHAT_FIXTURES overrides the fixture directory; with LIMBO_TEST_GGUF set the GGUF's own
// template is also compared with the fixture's copy.
#include "limbo/chat.hpp"
#include "limbo/check.hpp"
#include "limbo/gguf.hpp"
#include "limbo/jinja.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using limbo::jinja::Value;

namespace {

std::string slurp(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

Value json(const std::string& text) {
    Value v;
    std::string err;
    if (!limbo::jinja::json_parse(text, &v, &err)) {
        std::printf("FAIL test json did not parse: %s in %s\n", err.c_str(), text.c_str());
        ++limbo::test::failures();
    }
    return v;
}

std::string show(const std::string& s) {
    std::string o;
    for (unsigned char c : s) {
        if (c == '\n') o += "\\n";
        else if (c == '\t') o += "\\t";
        else if (c < 0x20) {
            char b[8];
            std::snprintf(b, sizeof b, "\\x%02x", c);
            o += b;
        } else o += char(c);
    }
    return o;
}

void first_difference(const std::string& got, const std::string& want) {
    size_t i = 0;
    while (i < got.size() && i < want.size() && got[i] == want[i]) ++i;
    const size_t a = i > 30 ? i - 30 : 0;
    std::printf("       first difference at byte %zu (got %zu bytes, want %zu)\n       got  ...%s\n       want ...%s\n", i,
                got.size(), want.size(), show(got.substr(a, 80)).c_str(), show(want.substr(a, 80)).c_str());
}

void check_fixtures(const fs::path& dir, const limbo::chat::Renderer& r) {
    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(dir)) {
        const std::string n = e.path().filename().string();
        if (n.size() > 5 && n.substr(n.size() - 5) == ".json" && n != "jinja_cases.json") files.push_back(e.path());
    }
    std::sort(files.begin(), files.end());
    CHECK(files.size() >= 28, "the golden chat fixtures are present (32 cases)");
    for (const fs::path& f : files) {
        const Value fx = json(slurp(f));
        const std::string name = fx.get_str("name");
        Value ctx = *fx.find("context");
        if (const Value* n = fx.find("normalize"); n && n->truthy()) ctx.set("messages", limbo::chat::normalize_messages(*ctx.find("messages")));
        std::string out, err;
        const bool ok = r.render(ctx, &out, &err);
        const Value* ev = fx.find("error");
        if (ev && ev->is_string()) {
            const bool match = !ok && (ev->s() == "*" || err == ev->s());
            if (match) std::printf("ok   fixture %s: raises \"%s\"\n", name.c_str(), err.c_str());
            else {
                std::printf("FAIL fixture %s: wanted error \"%s\", got ok=%d err=\"%s\"\n", name.c_str(), ev->s().c_str(), ok, err.c_str());
                ++limbo::test::failures();
            }
            continue;
        }
        fs::path exp = f;
        exp.replace_extension(".expected.txt");
        const std::string want = slurp(exp);
        if (ok && out == want) std::printf("ok   fixture %s: %zu bytes identical\n", name.c_str(), out.size());
        else {
            std::printf("FAIL fixture %s: %s\n", name.c_str(), ok ? "output differs" : err.c_str());
            if (ok) first_difference(out, want);
            ++limbo::test::failures();
        }
    }
}

void check_jinja_cases(const fs::path& dir) {
    const Value cases = json(slurp(dir / "jinja_cases.json"));
    CHECK(cases.list().size() >= 60, "the interpreter cases are present");
    int pass = 0;
    for (const Value& c : cases.list()) {
        limbo::jinja::Options o;
        o.trim_blocks = c.find("trim_blocks")->truthy();
        o.lstrip_blocks = c.find("lstrip_blocks")->truthy();
        o.keep_trailing_newline = c.find("keep_trailing_newline")->truthy();
        std::string err, out;
        auto t = limbo::jinja::Template::compile(c.find("template")->s(), o, &err);
        const bool ok = t && t->render(*c.find("context"), &out, &err);
        const Value* want = c.find("expected");
        const std::string name = c.get_str("name");
        bool good;
        if (want && want->is_string()) good = ok && out == want->s();
        else good = !ok;  // the reference raised: so must we
        if (good) ++pass;
        else {
            std::printf("FAIL jinja case %s: %s\n", name.c_str(), ok ? "output differs" : err.c_str());
            if (ok && want && want->is_string()) {
                std::printf("       got  [%s]\n       want [%s]\n", show(out).c_str(), show(want->s()).c_str());
            }
            ++limbo::test::failures();
        }
    }
    std::printf("ok   %d of %zu interpreter cases match Python jinja2\n", pass, cases.list().size());
}

// ---- JSON ----

void check_json() {
    using namespace limbo::jinja;
    Value v;
    std::string err;
    CHECK(json_parse("{\"b\": 1, \"a\": [1, 2.5, null, true, \"x\"], \"c\": {}}", &v, &err), "json: a mixed document parses");
    CHECK(json_compact(v) == "{\"b\":1,\"a\":[1,2.5,null,true,\"x\"],\"c\":{}}", "json: key order and number types survive a round trip");
    CHECK(json_dump(v) == "{\"b\": 1, \"a\": [1, 2.5, null, true, \"x\"], \"c\": {}}", "json: Python's default separators");
    CHECK(json_parse("1.0", &v, &err) && v.is_float() && json_dump(v) == "1.0", "json: 1.0 stays a float");
    CHECK(json_parse("10", &v, &err) && v.is_int(), "json: 10 is an int");
    CHECK(json_parse("1e3", &v, &err) && v.is_float() && json_dump(v) == "1000.0", "json: an exponent makes a float");
    CHECK(json_parse("\"\\u00e9\\ud83d\\ude00\\n\\\"\"", &v, &err) && v.s() == "\xC3\xA9\xF0\x9F\x98\x80\n\"", "json: escapes and surrogate pairs decode");
    CHECK(json_dump(v) == "\"\xC3\xA9\xF0\x9F\x98\x80\\n\\\"\"", "json: non-ASCII is kept as written");
    DumpOptions ascii;
    ascii.ensure_ascii = true;
    CHECK(json_dump(v, ascii) == "\"\\u00e9\\ud83d\\ude00\\n\\\"\"", "json: ensure_ascii escapes like Python");
    CHECK(!json_parse("{\"a\": 1} x", &v, &err), "json: trailing data is an error");
    CHECK(!json_parse("{\"a\": }", &v, &err), "json: a missing value is an error");
    CHECK(!json_parse("[1, 2", &v, &err), "json: an unterminated array is an error");
    CHECK(!json_parse(std::string(400, '[') + std::string(400, ']'), &v, &err), "json: absurd nesting is refused");
    CHECK(json_parse("[]", &v, &err) && json_dump(v) == "[]", "json: an empty array");
    DumpOptions ind;
    ind.indent = 2;
    CHECK(json_parse("{\"a\":[1,{\"b\":null}],\"e\":[]}", &v, &err) &&
              json_dump(v, ind) == "{\n  \"a\": [\n    1,\n    {\n      \"b\": null\n    }\n  ],\n  \"e\": []\n}",
          "json: indent matches Python");
    CHECK(py_float_repr(0.1) == "0.1" && py_float_repr(1e16) == "1e+16" && py_float_repr(1e-5) == "1e-05" &&
              py_float_repr(123456789.0) == "123456789.0" && py_float_repr(-0.0) == "-0.0" && py_float_repr(2.5e-10) == "2.5e-10",
          "py_float_repr follows Python's repr");
}

// ---- the output parser ----

Value weather_tools() {
    return json(R"([
      {"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{
        "city":{"type":"string"},"days":{"type":"integer"},"ratio":{"type":"number"},"metric":{"type":"boolean"},
        "tags":{"type":"array"},"opts":{"type":"object"},"maybe":{"type":["null","string"]}}}}},
      {"type":"function","function":{"name":"web_search","parameters":{"type":"object","properties":{"query":{"type":"string"}}}}}
    ])");
}

std::string call_text(const std::string& name, const std::vector<std::pair<std::string, std::string>>& params) {
    std::string o = "<tool_call>\n<function=" + name + ">\n";
    for (const auto& p : params) o += "<parameter=" + p.first + ">\n" + p.second + "\n</parameter>\n";
    return o + "</function>\n</tool_call>";
}

// Feeds `text` to a parser in pieces of the given sizes (cycled) and merges the events.
limbo::chat::Message parse_pieces(const std::string& text, bool in_think, const Value& tools, bool parse_tools, std::vector<size_t> sizes) {
    limbo::chat::OutputParser p(in_think, tools, parse_tools);
    std::vector<limbo::chat::Event> ev;
    size_t at = 0, k = 0;
    while (at < text.size()) {
        const size_t n = std::min(sizes[k++ % sizes.size()], text.size() - at);
        p.feed(std::string_view(text).substr(at, n), &ev);
        at += n;
    }
    p.finish(&ev);
    limbo::chat::Message m;
    for (auto& e : ev) {
        if (e.kind == limbo::chat::Event::Reasoning) m.reasoning += e.text, m.has_reasoning = true;
        else if (e.kind == limbo::chat::Event::Content) m.content += e.text;
        else m.calls.push_back(e.call);
    }
    return m;
}

bool same_message(const limbo::chat::Message& a, const limbo::chat::Message& b) {
    if (a.reasoning != b.reasoning || a.content != b.content || a.calls.size() != b.calls.size()) return false;
    for (size_t i = 0; i < a.calls.size(); ++i)
        if (a.calls[i].name != b.calls[i].name || !(a.calls[i].arguments == b.calls[i].arguments)) return false;
    return true;
}

void check_parser() {
    using limbo::chat::parse_output;
    const Value tools = weather_tools();
    {
        const auto m = parse_output("Hello there!", false, Value::undefined());
        CHECK(m.content == "Hello there!" && !m.has_reasoning && m.calls.empty(), "parse: plain content when the prompt closed the think block");
    }
    {
        const auto m = parse_output("  \n My thoughts here.\n</think>\n\nThe answer.\n", true, Value::undefined());
        CHECK(m.reasoning == "My thoughts here." && m.content == "The answer." && m.has_reasoning, "parse: reasoning, then content, both trimmed");
    }
    {
        const auto m = parse_output("<think>\nself-opened\n</think>\n\nfine", false, Value::undefined());
        CHECK(m.reasoning == "self-opened" && m.content == "fine", "parse: a think block the model opened itself");
    }
    {
        const auto m = parse_output("thoughts that never end", true, Value::undefined());
        CHECK(m.reasoning == "thoughts that never end" && m.content.empty(), "parse: no </think> (length cut): all reasoning");
    }
    {
        const auto m = parse_output("think\n</think>\n\nLet me check.\n\n" + call_text("get_weather", {{"city", "Paris"}, {"days", "3"}}), true, tools);
        CHECK(m.calls.size() == 1 && m.content == "Let me check.", "parse: content before a tool call, the call split out");
        CHECK(m.calls[0].name == "get_weather" && m.calls[0].arguments.find("city")->s() == "Paris", "parse: string parameter");
        CHECK(m.calls[0].arguments.find("days")->is_int() && m.calls[0].arguments.find("days")->i() == 3, "parse: integer parameter typed by the schema");
        CHECK(limbo::jinja::json_compact(m.calls[0].arguments) == "{\"city\":\"Paris\",\"days\":3}", "parse: parameters keep their order");
    }
    {
        const auto m = parse_output("\n\n" + call_text("get_weather", {{"city", "123"}, {"days", "7"}, {"ratio", "0.5"}, {"metric", "True"},
                                                                       {"tags", "[\"a\", 2]"}, {"opts", "{\"k\": null}"}, {"maybe", "x"},
                                                                       {"extra", "[1, 2]"}, {"note", "plain words"}, {"flag", "true"}, {"str2", "007"}}),
                                    false, tools);
        CHECK(m.calls.size() == 1 && m.content.empty(), "parse: a call with no content before it");
        const Value& a = m.calls[0].arguments;
        CHECK(a.find("city")->is_string() && a.find("city")->s() == "123", "parse: a string-typed parameter stays text even when it looks numeric");
        CHECK(a.find("ratio")->is_float() && a.find("ratio")->d() == 0.5, "parse: number");
        CHECK(a.find("metric")->is_bool() && a.find("metric")->b(), "parse: boolean from Python's True");
        CHECK(a.find("tags")->is_list() && a.find("tags")->list().size() == 2, "parse: array parameter");
        CHECK(a.find("opts")->is_dict(), "parse: object parameter");
        CHECK(a.find("maybe")->is_string(), "parse: [null, string] types as string");
        CHECK(a.find("extra")->is_list(), "parse: no schema, valid JSON: parsed");
        CHECK(a.find("note")->is_string() && a.find("note")->s() == "plain words", "parse: no schema, not JSON: text");
        CHECK(a.find("flag")->is_bool(), "parse: no schema, true: boolean");
        CHECK(a.find("str2")->is_string() && a.find("str2")->s() == "007", "parse: no schema, 007 is not JSON: text");
    }
    {
        const auto m = parse_output("</think>\n\n" + call_text("get_weather", {{"city", "A"}}) + "\n" + call_text("web_search", {{"query", "multi\nline\nvalue"}}), true, tools);
        CHECK(m.calls.size() == 2 && m.calls[1].name == "web_search", "parse: two calls in a row");
        CHECK(m.calls[1].arguments.find("query")->s() == "multi\nline\nvalue", "parse: a multi-line value");
    }
    // inkling2's message family (the GGUF's own template): the turn is
    // <|message_model|><|content_thinking|>...<|end_message|> then <|message_model|><|content_text|>...<|end_message|>;
    // a call carries the function name between the role token and its content marker and its block is
    // {"name": ..., "args": {...}}. The label is header, never content.
    {
        const auto m = parse_output("<|message_model|><|content_thinking|>the plan<|end_message|>"
                                    "<|message_model|><|content_text|>the answer<|end_message|>",
                                    false, Value::undefined());
        CHECK(m.reasoning == "the plan" && m.content == "the answer" && m.has_reasoning,
              "parse: inkling2 reasoning and content split out of the message stream");
    }
    {
        const auto m = parse_output("<|message_model|><|content_text|>Checking now.<|end_message|>"
                                    "<|message_model|>get_weather<|content_invoke_tool_json|>"
                                    "{\"name\":\"get_weather\",\"args\":{\"city\":\"Paris\",\"days\":3}}<|end_message|>",
                                    false, tools);
        CHECK(m.content == "Checking now." && m.calls.size() == 1, "parse: inkling2 content then a call, the label dropped");
        CHECK(m.calls[0].name == "get_weather" && m.calls[0].arguments.find("city")->s() == "Paris",
              "parse: inkling2 invoke-tool-json names the function");
        CHECK(m.calls[0].arguments.find("days")->is_int() && m.calls[0].arguments.find("days")->i() == 3,
              "parse: inkling2 args typed by the schema");
    }
    {
        const std::string t = "<|message_model|><|content_thinking|>step by step<|end_message|>"
                              "<|message_model|>get_weather<|content_invoke_tool_json|>"
                              "{\"name\":\"get_weather\",\"args\":{\"city\":\"Paris\"}}<|end_message|>";
        for (size_t n : {1, 2, 3, 5, 11})
            CHECK(same_message(parse_pieces(t, false, tools, true, {n}), parse_output(t, false, tools)),
                  "parse: inkling2 streamed in odd chunks converges (markers split across feeds)");
    }
    {
        const auto m = parse_output(call_text("get_weather", {{"city", "A"}}), false, tools, false);
        CHECK(m.calls.empty() && m.content.find("<tool_call>") != std::string::npos, "parse: tool_choice none leaves the call text in the content");
    }
    {
        const auto m = parse_output("<tool_call>\n<function=get_weather>\n<parameter=city>\nRome\n</parameter>\n</function>", false, tools);
        CHECK(m.calls.size() == 1 && m.calls[0].arguments.find("city")->s() == "Rome", "parse: a call whose </tool_call> never came is still a call");
    }
    {
        const auto m = parse_output("text\n<tool_call>\n<function=get_weather>\n<parameter=city>\nRo", false, tools);
        CHECK(m.calls.empty() && m.content.find("<tool_call>") != std::string::npos && m.content.find("Ro") != std::string::npos,
              "parse: a call cut off mid-parameter comes back as text");
    }
    // What a quantized model gets wrong in a long call (seen on Qwen3.6-35B-A3B IQ3_XXS writing an SVG through
    // Ariadne, 2026-11-25): a value it never closes, a call it abandons for a new one, a </function> it skips.
    {
        const auto m = parse_output("<tool_call>\n<function=get_weather>\n<parameter=city>\n<svg>\n<path d=\"M 1 2\"/>\n</svg>\n"
                                    "<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>", false, tools);
        CHECK(m.calls.size() == 1 && m.calls[0].arguments.find("city") && m.calls[0].arguments.find("city")->s() == "<svg>\n<path d=\"M 1 2\"/>\n</svg>" &&
                  m.calls[0].arguments.find("days") && m.calls[0].arguments.find("days")->i() == 3,
              "parse: an unclosed value ends at the next parameter the function declares");
    }
    {
        const auto m = parse_output(call_text("get_weather", {{"city", "a\n<parameter=nonsense>\nb"}}), false, tools);
        CHECK(m.calls.size() == 1 && m.calls[0].arguments.find("city")->s() == "a\n<parameter=nonsense>\nb",
              "parse: a parameter look-alike the function does not declare stays in the value");
    }
    const std::string restart = "Writing it.\n<tool_call>\n<function=web_search>\n<parameter=query>\npython3 -c \"open('/home/x_6507\n\n" +
                                call_text("get_weather", {{"city", "Rome"}}) + "\nafter";
    {
        const auto m = parse_output(restart, false, tools);
        CHECK(m.calls.size() == 1 && m.calls[0].name == "get_weather" && m.calls[0].arguments.find("city")->s() == "Rome" &&
                  m.content == "Writing it.after",
              "parse: a call opened inside an unclosed one replaces it; the fragment is dropped, never run");
        bool same = true;
        for (size_t n : {1, 2, 3, 7, 13})
            if (!same_message(parse_pieces(restart, false, tools, true, {n}), m)) same = false;
        CHECK(same, "parse: the restart streamed in pieces gives the same message");
    }
    {
        const auto m = parse_output("<tool_call>\n<function=get_weather>\n<parameter=city>\nA\n</parameter>\n</function>\n" +
                                    call_text("web_search", {{"query", "q"}}), false, tools);
        CHECK(m.calls.size() == 2 && m.calls[0].name == "get_weather" && m.calls[1].name == "web_search",
              "parse: a call that skipped </tool_call> before the next one is still a call");
    }
    {
        const auto m = parse_output("<tool_call>\n<function=get_weather>\n<parameter=city>\nRome\n</parameter>\n<parameter=days>\n2\n</parameter>\n",
                                    false, tools);
        CHECK(m.calls.size() == 1 && m.calls[0].arguments.find("days")->i() == 2,
              "parse: a turn that ends after its last </parameter> is a call");
    }
    {
        const auto m = parse_output("<tool_call>\nnot a function\n</tool_call> after", false, tools);
        CHECK(m.calls.empty() && m.content.find("not a function") != std::string::npos, "parse: a malformed block comes back as text");
    }
    {
        const auto m = parse_output("<tool_call>\n{\"name\": \"web_search\", \"arguments\": {\"query\": \"q\"}}\n</tool_call>", false, tools);
        CHECK(m.calls.size() == 1 && m.calls[0].arguments.find("query")->s() == "q", "parse: the JSON tool-call form");
    }
    {
        const auto m = parse_output("a < b and <b>bold</b> <tool and </think", false, Value::undefined());
        CHECK(m.content == "a < b and <b>bold</b> <tool and </think", "parse: marker look-alikes are plain text");
    }
    {
        const auto m = parse_output("x" + std::string("\xC3\xA9\xE6\x97\xA5") + " y", false, Value::undefined());
        CHECK(m.content == "x\xC3\xA9\xE6\x97\xA5 y", "parse: UTF-8 passes through");
    }

    // The streaming path is the same code: any way of cutting the text gives the same message.
    const std::string long_text =
        "reason one\nreason two\n</think>\n\nBefore the call.\n\n" + call_text("get_weather", {{"city", "K\xC3\xB6ln"}, {"days", "2"}}) + "\n" +
        call_text("web_search", {{"query", "rust"}});
    const auto whole = parse_output(long_text, true, tools);
    bool all_same = true;
    for (size_t step : {size_t(1), size_t(2), size_t(3), size_t(5), size_t(7), size_t(13), size_t(64)})
        if (!same_message(whole, parse_pieces(long_text, true, tools, true, {step}))) all_same = false;
    if (!same_message(whole, parse_pieces(long_text, true, tools, true, {1, 4, 2, 9, 3}))) all_same = false;
    CHECK(all_same, "parse: fed in pieces of 1..64 bytes gives the same message as fed whole");
    CHECK(whole.calls.size() == 2 && whole.content == "Before the call.", "parse: the long text has reasoning, content and two calls");

    // Deltas are held back only while they could be a marker: a streaming consumer sees text early.
    {
        limbo::chat::OutputParser p(false, tools, true);
        std::vector<limbo::chat::Event> ev;
        p.feed("Hello wor", &ev);
        CHECK(ev.size() == 1 && ev[0].text == "Hello wor", "stream: plain text is released at once");
        ev.clear();
        p.feed("ld <tool_ca", &ev);
        CHECK(ev.size() == 1 && ev[0].text == "ld", "stream: a possible marker start (and trailing space) is held back");
        ev.clear();
        p.feed("ll>\n<function=get_weather>\n</function>\n</tool_call>", &ev);
        CHECK(ev.size() == 1 && ev[0].kind == limbo::chat::Event::Call, "stream: the call is emitted when it closes");
    }
}

// ---- normalisation and the round trip ----

void check_roundtrip(const limbo::chat::Renderer& r) {
    using namespace limbo::chat;
    {
        const Value msgs = json(R"([{"role":"assistant","content":null,"tool_calls":[
            {"id":"c1","type":"function","function":{"name":"f","arguments":"{\"a\": 1, \"b\": [1, 2]}"}},
            {"id":"c2","type":"function","function":{"name":"g","arguments":"not json"}}]}])");
        const Value n = normalize_messages(msgs);
        const Value& calls = *n.list()[0].find("tool_calls");
        CHECK(calls.list()[0].find("function")->find("arguments")->is_dict(), "normalize: a JSON-string arguments becomes an object");
        CHECK(calls.list()[0].find("id")->s() == "c1", "normalize: other call fields survive");
        CHECK(calls.list()[1].find("function")->find("arguments")->is_string(), "normalize: arguments that are not JSON are left alone");
        CHECK(msgs.list()[0].find("tool_calls")->list()[0].find("function")->find("arguments")->is_string(), "normalize: the input is not modified");
    }
    CHECK(prompt_opens_think("<|im_start|>assistant\n<think>\n"), "opens_think: a prompt ending in <think>");
    CHECK(!prompt_opens_think("<|im_start|>assistant\n<think>\n\n</think>\n\n"), "opens_think: a closed block does not");
    CHECK(!prompt_opens_think(""), "opens_think: empty");

    // What the model wrote, parsed into an OpenAI message, must render back to the same tokens: the
    // generated tail of the tape is the prefix of the next prompt.
    const Value tools = weather_tools();
    const std::string generated =
        "I should check the weather.\n</think>\n\nLet me look.\n\n" +
        call_text("get_weather", {{"city", "M\xC3\xBCnchen"}, {"days", "3"}, {"tags", "[\"a\", \"b\"]"}}) + "\n" +
        call_text("web_search", {{"query", "line one\nline two"}});
    const Message m = parse_output(generated, true, tools);
    Value asst = Value::object();
    asst.set("role", Value::string("assistant"));
    asst.set("content", Value::string(m.content));
    asst.set("reasoning_content", Value::string(m.reasoning));
    Value calls = Value::array();
    for (const ToolCall& c : m.calls) {
        Value fn = Value::object();
        fn.set("name", Value::string(c.name));
        fn.set("arguments", Value::string(limbo::jinja::json_compact(c.arguments)));  // the wire form
        Value tc = Value::object();
        tc.set("id", Value::string("call_x"));
        tc.set("type", Value::string("function"));
        tc.set("function", fn);
        calls.push(tc);
    }
    asst.set("tool_calls", calls);
    Value msgs = Value::array();
    msgs.push(json(R"({"role":"user","content":"weather?"})"));
    msgs.push(asst);
    std::string head, full, err;
    const Value kw = json("{}");
    CHECK(r.apply(normalize_messages(json(R"([{"role":"user","content":"weather?"}])")), tools, true, kw, Value::undefined(), &head, &err),
          "roundtrip: the generation prompt renders");
    CHECK(r.apply(normalize_messages(msgs), tools, false, kw, Value::undefined(), &full, &err), "roundtrip: the assistant turn renders");
    const std::string want = head + generated + "<|im_end|>\n";
    CHECK(full.size() >= want.size() && full.compare(full.size() - want.size(), want.size(), want) == 0,
          "roundtrip: prompt + generated text + <|im_end|> is exactly what the template re-renders for that turn");
    if (full.size() < want.size() || full.compare(full.size() - want.size(), want.size(), want) != 0) first_difference(full, want);
}

}  // namespace

int main() {
    const char* env = std::getenv("LIMBO_CHAT_FIXTURES");
    const fs::path dir = env && *env ? env : "bench/fixtures/chat";
    if (!fs::exists(dir / "template.jinja")) {
        std::printf("FAIL no fixtures in %s (run scripts/chat_fixtures.py)\n", dir.string().c_str());
        return 1;
    }
    const std::string tmpl = slurp(dir / "template.jinja");
    std::string err;
    auto r = limbo::chat::Renderer::from_source(tmpl, &err);
    CHECK(r != nullptr, "the model's chat template compiles");
    if (!r) {
        std::printf("       %s\n", err.c_str());
        return limbo::test::finish("chat");
    }

    check_json();
    check_fixtures(dir, *r);
    check_jinja_cases(dir);
    check_parser();
    check_roundtrip(*r);

    {  // the bytes of a utf-8 helper pair
        using limbo::chat::utf8_complete_prefix;
        using limbo::chat::utf8_sanitize;
        CHECK(utf8_complete_prefix("ab") == 2 && utf8_complete_prefix("a\xC3") == 1 && utf8_complete_prefix("a\xE6\x97") == 1 &&
                  utf8_complete_prefix("a\xF0\x9F\x98") == 1 && utf8_complete_prefix("\xE6\x97\xA5") == 3,
              "utf8: an incomplete trailing sequence is cut off, a complete one kept");
        CHECK(utf8_sanitize("a\xFF""b\xC3") == "a\xEF\xBF\xBD""b\xEF\xBF\xBD", "utf8: invalid bytes become U+FFFD");
        CHECK(utf8_sanitize("\xC3\xA9\xE6\x97\xA5") == "\xC3\xA9\xE6\x97\xA5", "utf8: valid text is unchanged");
    }

    if (const char* g = std::getenv("LIMBO_TEST_GGUF"); g && *g) {
        const std::string path = std::string(g).substr(0, std::string(g).find(':'));
        auto f = limbo::gguf::File::open(path, &err);
        if (!f) {
            std::printf("FAIL cannot open %s: %s\n", path.c_str(), err.c_str());
            ++limbo::test::failures();
        } else {
            const auto t = f->get_str("tokenizer.chat_template");
            CHECK(t && *t == tmpl, "the GGUF's tokenizer.chat_template is the template the fixtures were rendered from");
            auto rr = limbo::chat::Renderer::from_gguf(*f, &err);
            CHECK(rr != nullptr, "Renderer::from_gguf compiles the GGUF's template");
        }
    } else std::printf("--   LIMBO_TEST_GGUF not set: the GGUF's template is not compared with the fixture copy\n");
    return limbo::test::finish("chat");
}
