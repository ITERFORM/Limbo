// Copyright (c) 2026 ITERFORM. All rights reserved.
// src/chat.cpp — chat rendering over the GGUF's template, message normalisation, output parsing.
#include "limbo/chat.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>

#include "limbo/gguf.hpp"

namespace limbo::chat {

using jinja::Value;

// ---- Renderer -------------------------------------------------------------------------------------

std::unique_ptr<Renderer> Renderer::from_source(std::string_view tmpl, std::string* err) {
    std::unique_ptr<Renderer> r(new Renderer);
    r->src_ = std::string(tmpl);
    std::unique_ptr<jinja::Template> t = jinja::Template::compile(tmpl, jinja::Options{}, err);
    if (!t) return nullptr;
    r->tpl_ = std::move(t);
    return r;
}

std::unique_ptr<Renderer> Renderer::from_gguf(const gguf::File& f, std::string* err) {
    const auto t = f.get_str("tokenizer.chat_template");
    if (!t) {
        if (err) *err = "the GGUF has no tokenizer.chat_template";
        return nullptr;
    }
    return from_source(*t, err);
}

bool Renderer::render(const Value& context, std::string* out, std::string* err) const {
    return tpl_->render(context, out, err);
}

Value Renderer::make_context(const Value& messages, const Value& tools, bool add_generation_prompt,
                             const Value& kwargs, const Value& extra) {
    Value ctx = Value::object();
    ctx.set("messages", messages);
    if (tools.is_list() && !tools.list().empty()) ctx.set("tools", tools);
    ctx.set("add_generation_prompt", Value::boolean(add_generation_prompt));
    for (const Value* src : {&kwargs, &extra}) {
        if (!src->is_dict()) continue;
        for (const auto& kv : src->dict())
            if (!ctx.find(kv.first)) ctx.set(kv.first, kv.second);
    }
    return ctx;
}

bool Renderer::apply(const Value& messages, const Value& tools, bool add_generation_prompt, const Value& kwargs,
                     const Value& extra, std::string* out, std::string* err) const {
    return render(make_context(messages, tools, add_generation_prompt, kwargs, extra), out, err);
}

Value normalize_messages(const Value& messages) {
    if (!messages.is_list()) return messages;
    Value out = Value::array();
    for (const Value& m : messages.list()) {
        const Value* tc = m.is_dict() ? m.find("tool_calls") : nullptr;
        if (!tc || !tc->is_list()) {
            out.push(m);
            continue;
        }
        Value nm = Value::object();
        for (const auto& kv : m.dict()) {
            if (kv.first != "tool_calls") {
                nm.set(kv.first, kv.second);
                continue;
            }
            Value calls = Value::array();
            for (const Value& c : kv.second.list()) {
                const Value* fn = c.is_dict() ? c.find("function") : nullptr;
                const Value* args = fn && fn->is_dict() ? fn->find("arguments") : nullptr;
                if (!args || !args->is_string()) {
                    calls.push(c);
                    continue;
                }
                Value parsed;
                std::string perr;
                if (!jinja::json_parse(args->s(), &parsed, &perr) || !parsed.is_dict()) {
                    calls.push(c);
                    continue;
                }
                Value nc = Value::object();
                for (const auto& ckv : c.dict()) {
                    if (ckv.first != "function") {
                        nc.set(ckv.first, ckv.second);
                        continue;
                    }
                    Value nf = Value::object();
                    for (const auto& fkv : ckv.second.dict()) nf.set(fkv.first, fkv.first == "arguments" ? parsed : fkv.second);
                    nc.set("function", nf);
                }
                calls.push(nc);
            }
            nm.set("tool_calls", calls);
        }
        out.push(nm);
    }
    return out;
}

bool prompt_opens_think(std::string_view p) {
    while (!p.empty() && (p.back() == ' ' || p.back() == '\n' || p.back() == '\t' || p.back() == '\r')) p.remove_suffix(1);
    constexpr std::string_view tag = "<think>";
    return p.size() >= tag.size() && p.substr(p.size() - tag.size()) == tag;
}

// ---- UTF-8 ----------------------------------------------------------------------------------------

size_t utf8_complete_prefix(std::string_view s) {
    const size_t n = s.size();
    size_t i = n;
    // Look back over at most three continuation bytes for the lead byte of a sequence cut off at the end.
    for (size_t back = 1; back <= 4 && back <= n; ++back) {
        const unsigned char c = static_cast<unsigned char>(s[n - back]);
        if ((c & 0xC0) == 0x80) continue;  // continuation: keep looking for the lead
        size_t need = 0;
        if (c >= 0xC2 && c < 0xE0) need = 2;
        else if (c >= 0xE0 && c < 0xF0) need = 3;
        else if (c >= 0xF0 && c < 0xF5) need = 4;
        if (need > back) i = n - back;  // incomplete
        break;
    }
    return i;
}

std::string utf8_sanitize(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        if (c < 0x80) {
            out.push_back(char(c));
            ++i;
            continue;
        }
        size_t need = 0;
        uint32_t v = 0;
        if (c >= 0xC2 && c < 0xE0) need = 2, v = c & 0x1F;
        else if (c >= 0xE0 && c < 0xF0) need = 3, v = c & 0x0F;
        else if (c >= 0xF0 && c < 0xF5) need = 4, v = c & 0x07;
        bool ok = need != 0 && i + need <= s.size();
        for (size_t k = 1; ok && k < need; ++k) {
            const unsigned char cc = static_cast<unsigned char>(s[i + k]);
            if ((cc & 0xC0) != 0x80) ok = false;
            else v = (v << 6) | (cc & 0x3F);
        }
        if (ok && ((need == 3 && (v < 0x800 || (v >= 0xD800 && v <= 0xDFFF))) || (need == 4 && (v < 0x10000 || v > 0x10FFFF))))
            ok = false;
        if (ok) {
            out.append(s.substr(i, need));
            i += need;
        } else {
            out += "\xEF\xBF\xBD";
            ++i;
        }
    }
    return out;
}

// ---- tool blocks ----------------------------------------------------------------------------------

namespace {

bool is_ws(char c) { return c == ' ' || c == '\n' || c == '\t' || c == '\r' || c == '\v' || c == '\f'; }

std::string_view trim_ws(std::string_view s) {
    while (!s.empty() && is_ws(s.front())) s.remove_prefix(1);
    while (!s.empty() && is_ws(s.back())) s.remove_suffix(1);
    return s;
}

std::string lower_ascii(std::string_view s) {
    std::string o(s);
    for (char& c : o)
        if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
    return o;
}

// The schema of one parameter of one tool, or nullptr.
const Value* param_schema(const Value& tools, const std::string& fn_name, const std::string& param) {
    for (const Value& t : tools.list()) {
        if (!t.is_dict()) continue;
        const Value* fn = t.find("function");
        const Value& f = fn && fn->is_dict() ? *fn : t;
        if (f.get_str("name") != fn_name) continue;
        const Value* params = f.find("parameters");
        const Value* props = params && params->is_dict() ? params->find("properties") : nullptr;
        const Value* sch = props && props->is_dict() ? props->find(param) : nullptr;
        return sch && sch->is_dict() ? sch : nullptr;
    }
    return nullptr;
}

// The parameter names a tool declares (its schema's properties).
std::vector<std::string> declared_params(const Value& tools, const std::string& fn_name) {
    std::vector<std::string> out;
    if (!tools.is_list()) return out;
    for (const Value& t : tools.list()) {
        if (!t.is_dict()) continue;
        const Value* fn = t.find("function");
        const Value& f = fn && fn->is_dict() ? *fn : t;
        if (f.get_str("name") != fn_name) continue;
        const Value* params = f.find("parameters");
        const Value* props = params && params->is_dict() ? params->find("properties") : nullptr;
        if (props && props->is_dict())
            for (const auto& kv : props->dict()) out.push_back(kv.first);
        break;
    }
    return out;
}

// The JSON-schema type name to parse a value as: "" when the schema does not pin one.
std::string schema_type(const Value* sch) {
    if (!sch) return "";
    const Value* t = sch->find("type");
    if (!t) return "";
    if (t->is_string()) return t->s();
    if (t->is_list()) {
        for (const Value& x : t->list())
            if (x.is_string() && x.s() != "null") return x.s();
    }
    return "";
}

bool parse_int_exact(std::string_view s, int64_t* out) {
    s = trim_ws(s);
    if (s.empty()) return false;
    const auto r = std::from_chars(s.data(), s.data() + s.size(), *out);
    return r.ec == std::errc() && r.ptr == s.data() + s.size();
}

bool parse_double_exact(std::string_view s, double* out) {
    const std::string t(trim_ws(s));
    if (t.empty()) return false;
    char* end = nullptr;
    *out = std::strtod(t.c_str(), &end);
    return end && *end == 0 && std::isfinite(*out);
}

Value convert_value(const std::string& raw, const std::string& type) {
    if (type == "string") return Value::string(raw);
    if (type == "integer") {
        int64_t v;
        if (parse_int_exact(raw, &v)) return Value::integer(v);
        double d;
        if (parse_double_exact(raw, &d) && d == double(int64_t(d))) return Value::integer(int64_t(d));
        return Value::string(raw);
    }
    if (type == "number") {
        int64_t v;
        if (parse_int_exact(raw, &v)) return Value::integer(v);
        double d;
        if (parse_double_exact(raw, &d)) return Value::number(d);
        return Value::string(raw);
    }
    if (type == "boolean") {
        const std::string l = lower_ascii(trim_ws(raw));
        if (l == "true" || l == "1") return Value::boolean(true);
        if (l == "false" || l == "0") return Value::boolean(false);
        return Value::string(raw);
    }
    if (type == "null") {
        const std::string l = lower_ascii(trim_ws(raw));
        return l == "null" || l == "none" ? Value::null() : Value::string(raw);
    }
    Value v;
    std::string err;
    if (jinja::json_parse(trim_ws(raw), &v, &err)) {
        if (type == "array") return v.is_list() ? v : Value::string(raw);
        if (type == "object") return v.is_dict() ? v : Value::string(raw);
        if (!v.is_string()) return v;  // untyped: numbers, booleans, null, arrays, objects
    }
    return Value::string(raw);
}

}  // namespace

bool parse_tool_block(std::string_view block, const Value& tools, ToolCall* out) {
    const std::string_view t = trim_ws(block);
    if (t.empty()) return false;
    if (t.front() == '{') {  // the JSON form {"name": ..., "arguments": {...}}
        Value v;
        std::string err;
        if (!jinja::json_parse(t, &v, &err) || !v.is_dict()) return false;
        const Value* fn = v.find("function");
        const Value& f = fn && fn->is_dict() ? *fn : v;
        const Value* name = f.find("name");
        if (!name || !name->is_string() || name->s().empty()) return false;
        out->name = name->s();
        const Value* args = f.find("arguments");
        if (!args) args = f.find("parameters");
        if (!args) args = f.find("args");  // inkling2's invoke-tool-json form
        out->arguments = Value::object();
        if (args && args->is_dict()) out->arguments = *args;
        else if (args && args->is_string()) {
            Value pa;
            if (jinja::json_parse(args->s(), &pa, &err) && pa.is_dict()) out->arguments = pa;
        }
        return true;
    }
    constexpr std::string_view fn_open = "<function=";
    size_t p = t.find(fn_open);
    if (p == std::string_view::npos) return false;
    p += fn_open.size();
    const size_t gt = t.find('>', p);
    if (gt == std::string_view::npos) return false;
    const std::string name(trim_ws(t.substr(p, gt - p)));
    if (name.empty()) return false;
    out->name = name;
    out->arguments = Value::object();

    std::string_view rest = t.substr(gt + 1);
    constexpr std::string_view par_open = "<parameter=", par_close = "</parameter>", fn_close = "</function>";
    // A value the model never closed runs into the next parameter: "\n<parameter=NAME>" with NAME one the function
    // declares ends it (a long write's content followed straight by its path lost the path otherwise).
    std::vector<std::string> next_marks;
    for (const std::string& d : declared_params(tools, name)) next_marks.push_back("\n<parameter=" + d + ">");
    for (;;) {
        const size_t po = rest.find(par_open);
        const size_t fe = rest.find(fn_close);
        if (po == std::string_view::npos || (fe != std::string_view::npos && fe < po)) break;
        const size_t ngt = rest.find('>', po + par_open.size());
        if (ngt == std::string_view::npos) break;
        const std::string pname(trim_ws(rest.substr(po + par_open.size(), ngt - po - par_open.size())));
        std::string_view body = rest.substr(ngt + 1);
        size_t pc = body.find(par_close);
        size_t pn = std::string_view::npos;
        for (const std::string& mk : next_marks) {
            const size_t k = body.find(mk);
            if (k != std::string_view::npos && (pn == std::string_view::npos || k < pn)) pn = k;
        }
        std::string_view value;
        if (pn != std::string_view::npos && (pc == std::string_view::npos || pn < pc)) {
            value = body.substr(0, pn);
            rest = body.substr(pn + 1);
        } else if (pc == std::string_view::npos) {  // unterminated: up to </function>, or the end of the block
            const size_t f2 = body.find(fn_close);
            value = body.substr(0, f2);
            rest = f2 == std::string_view::npos ? std::string_view() : body.substr(f2);
        } else {
            value = body.substr(0, pc);
            rest = body.substr(pc + par_close.size());
        }
        if (!value.empty() && value.front() == '\n') value.remove_prefix(1);
        if (!value.empty() && value.back() == '\n') value.remove_suffix(1);
        if (!pname.empty())
            out->arguments.set(pname, convert_value(std::string(value), schema_type(param_schema(tools, name, pname))));
    }
    return true;
}

// ---- the incremental parser -----------------------------------------------------------------------

namespace {

// Emits text with leading whitespace skipped and trailing whitespace held until more text follows.
struct TrimStream {
    bool started = false;
    std::string held;

    void reset() {
        started = false;
        held.clear();
    }
    std::string push(std::string_view t) {
        std::string avail = held;
        avail.append(t);
        held.clear();
        size_t b = 0;
        if (!started) {
            while (b < avail.size() && is_ws(avail[b])) ++b;
            if (b == avail.size()) return "";
            started = true;
        }
        size_t e = avail.size();
        while (e > b && is_ws(avail[e - 1])) --e;
        held = avail.substr(e);
        return avail.substr(b, e - b);
    }
};

// The longest k < marker.size() with s ending in marker[:k].
size_t partial_suffix(std::string_view s, std::string_view marker) {
    for (size_t k = std::min(marker.size() - 1, s.size()); k > 0; --k)
        if (s.substr(s.size() - k) == marker.substr(0, k)) return k;
    return 0;
}

// inkling2's message family (the GGUF's own template): a generated turn is one or more
// <|message_...|><|content_...|> ... <|end_message|> messages, and the text between the role token and
// the content marker is a label (the function name before a call), never content. Qwen's output carries
// none of these markers, so the two families never collide.
constexpr std::string_view kMsgPrefix = "<|message_";
constexpr std::string_view kThink2 = "<|content_thinking|>", kText2 = "<|content_text|>",
                          kXml2 = "<|content_xml|>", kCall2 = "<|content_invoke_tool_json|>",
                          kEndMsg = "<|end_message|>", kEndSample = "<|content_model_end_sampling|>";

// The earliest of `ms` in s: the position (npos when none) and which marker sits there.
size_t find_first(std::string_view s, std::string_view* which, const std::string_view* ms, size_t n) {
    size_t best = std::string_view::npos;
    *which = {};
    for (size_t i = 0; i < n; ++i) {
        const size_t k = s.find(ms[i]);
        if (k != std::string_view::npos && (best == std::string_view::npos || k < best)) {
            best = k;
            *which = ms[i];
        }
    }
    return best;
}

// How many trailing bytes of s could still grow into one of `ms` (streaming hold-back).
size_t hold_any(std::string_view s, const std::string_view* ms, size_t n) {
    size_t h = 0;
    for (size_t i = 0; i < n; ++i) h = std::max(h, partial_suffix(s, ms[i]));
    return h;
}

}  // namespace

struct OutputParser::Impl {
    enum class St { Start, Header, Think, Content, Tool };
    St st;
    bool parse_tools;
    Value tools;
    std::string buf;
    TrimStream trim;
    // Which call family opened the Tool state (finish() completes an unterminated block with its markers).
    std::string_view tool_open = kCall, tool_close = kEndCall;

    static constexpr std::string_view kThink = "<think>", kEndThink = "</think>", kCall = "<tool_call>",
                                      kEndCall = "</tool_call>";
    // a qwen call opening at a line start inside an open block
    static constexpr std::string_view kRestart = "\n<tool_call>\n<function=";

    void emit(Event::Kind k, std::string_view text, std::vector<Event>* out) {
        std::string t = trim.push(text);
        if (t.empty()) return;
        Event e;
        e.kind = k;
        e.text = std::move(t);
        out->push_back(std::move(e));
    }

    void pump(bool final, std::vector<Event>* out) {
        for (;;) {
            switch (st) {
                case St::Start: {
                    size_t b = 0;
                    while (b < buf.size() && is_ws(buf[b])) ++b;
                    buf.erase(0, b);
                    if (buf.empty()) return;
                    if (buf.compare(0, kThink.size(), kThink) == 0) {
                        buf.erase(0, kThink.size());
                        st = St::Think;
                        trim.reset();
                        continue;
                    }
                    // inkling2: a turn opens <|message_...|><|content_...|>; a bare content marker opens
                    // one where the prompt stopped after the role token.
                    if (buf.compare(0, kMsgPrefix.size(), kMsgPrefix) == 0 ||
                        buf.compare(0, kThink2.size(), kThink2) == 0 ||
                        buf.compare(0, kText2.size(), kText2) == 0 || buf.compare(0, kXml2.size(), kXml2) == 0 ||
                        buf.compare(0, kCall2.size(), kCall2) == 0) {
                        st = St::Header;
                        continue;
                    }
                    {
                        const std::string_view hold[] = {kThink, kMsgPrefix, kThink2, kText2, kXml2, kCall2};
                        if (!final && hold_any(buf, hold, 6) == buf.size()) return;
                    }
                    st = St::Content;
                    trim.reset();
                    continue;
                }
                case St::Header: {
                    // Everything up to the content marker is header: the role token and, before a call,
                    // the function name the template puts there. It is never content.
                    std::string_view which;
                    const std::string_view secs[] = {kThink2, kText2, kXml2, kCall2};
                    const size_t k = find_first(buf, &which, secs, 4);
                    if (k == std::string_view::npos) {
                        buf.erase(0, buf.size() - (final ? 0 : hold_any(buf, secs, 4)));
                        return;
                    }
                    buf.erase(0, k + which.size());
                    st = which == kThink2 ? St::Think : which == kCall2 ? St::Tool : St::Content;
                    trim.reset();
                    continue;
                }
                case St::Think: {
                    std::string_view which;
                    const std::string_view term[] = {kEndThink, kEndMsg, kEndSample, kText2, kXml2, kCall2};
                    const size_t k = find_first(buf, &which, term, 6);
                    if (k != std::string_view::npos) {
                        emit(Event::Reasoning, std::string_view(buf).substr(0, k), out);
                        buf.erase(0, k + which.size());
                        trim.reset();
                        if (which == kEndThink) st = St::Content;
                        else if (which == kEndMsg || which == kEndSample) st = St::Start;
                        else st = which == kCall2 ? St::Tool : St::Content;
                        continue;
                    }
                    const size_t hold = final ? 0 : hold_any(buf, term, 6);
                    emit(Event::Reasoning, std::string_view(buf).substr(0, buf.size() - hold), out);
                    buf.erase(0, buf.size() - hold);
                    return;
                }
                case St::Content: {
                    std::string_view which;
                    const std::string_view term[] = {kCall, kCall2, kEndMsg, kEndSample, kThink2};
                    const std::string_view no_tools[] = {kEndMsg, kEndSample, kThink2};
                    const std::string_view* ms = parse_tools ? term : no_tools;
                    const size_t n = parse_tools ? 5 : 3;
                    const size_t k = find_first(buf, &which, ms, n);
                    if (k != std::string_view::npos) {
                        emit(Event::Content, std::string_view(buf).substr(0, k), out);
                        trim.reset();  // whitespace before a marker is formatting, not content
                        buf.erase(0, k + which.size());
                        if (which == kEndMsg || which == kEndSample) st = St::Start;
                        else if (which == kThink2) st = St::Think;
                        else st = St::Tool;
                        continue;
                    }
                    const size_t hold = final ? 0 : hold_any(buf, ms, n);
                    emit(Event::Content, std::string_view(buf).substr(0, buf.size() - hold), out);
                    buf.erase(0, buf.size() - hold);
                    return;
                }
                case St::Tool: {
                    std::string_view which;
                    const std::string_view term[] = {kEndCall, kEndMsg, kRestart};
                    const size_t k = find_first(buf, &which, term, 3);
                    if (k == std::string_view::npos) return;
                    if (which == kRestart) {
                        // A new call opened inside this one: the model gave this block up. One that closed its
                        // </function> is a call whose </tool_call> it skipped; anything less is a broken
                        // fragment and is dropped -- never run, and not left in the history for the model to copy.
                        const std::string_view blk = std::string_view(buf).substr(0, k);
                        if (blk.find("</function>") != std::string_view::npos) complete_call(blk, kCall, kEndCall, false, out);
                        buf.erase(0, k + 1 + kCall.size());
                        tool_open = kCall;
                        tool_close = kEndCall;
                        continue;
                    }
                    complete_call(std::string_view(buf).substr(0, k), which == kEndCall ? kCall : kCall2, which, true, out);
                    buf.erase(0, k + which.size());
                    tool_open = which == kEndCall ? kCall : kCall2;
                    tool_close = which;
                    st = which == kEndCall ? St::Content : St::Start;
                    trim.reset();
                    continue;
                }
            }
        }
    }

    void complete_call(std::string_view block, std::string_view open, std::string_view close, bool closed,
                       std::vector<Event>* out) {
        Event e;
        e.kind = Event::Call;
        if (parse_tool_block(block, tools, &e.call)) {
            out->push_back(std::move(e));
            return;
        }
        std::string raw(open);  // not a call after all: hand the text back as content
        raw.append(block);
        if (closed) raw.append(close);
        trim.reset();
        Event c;
        c.kind = Event::Content;
        c.text = std::move(raw);
        out->push_back(std::move(c));
    }
};

OutputParser::OutputParser(bool in_think, const Value& tools, bool parse_tools) : impl_(new Impl) {
    impl_->st = in_think ? Impl::St::Think : Impl::St::Start;
    impl_->parse_tools = parse_tools;
    impl_->tools = tools;
}
OutputParser::~OutputParser() = default;

void OutputParser::feed(std::string_view text, std::vector<Event>* out) {
    impl_->buf.append(text);
    impl_->pump(false, out);
}

void OutputParser::finish(std::vector<Event>* out) {
    Impl& m = *impl_;
    m.pump(true, out);
    if (m.st == Impl::St::Tool && !m.buf.empty()) {
        // The turn ended inside a tool block. A block that already closed its </function> is a call
        // whose </tool_call> the model skipped, and inkling2's complete JSON object is the same
        // (complete_call hands anything else back as text with its own markers).
        const std::string_view b = m.buf;
        // ... and so is one whose every parameter closed, with only </function> missing.
        const std::string_view bt = trim_ws(b);
        const bool params_closed = bt.size() >= 12 && bt.substr(bt.size() - 12) == "</parameter>";
        if (m.tool_open == kCall2 || params_closed || b.find("</function>") != std::string_view::npos) {
            m.complete_call(b, m.tool_open, m.tool_close, false, out);
        } else {
            Event c;
            c.kind = Event::Content;
            c.text = std::string(Impl::kCall) + std::string(b);
            out->push_back(std::move(c));
        }
        m.buf.clear();
    }
}

Message parse_output(std::string_view text, bool in_think, const Value& tools, bool parse_tools) {
    OutputParser p(in_think, tools, parse_tools);
    std::vector<Event> ev;
    p.feed(text, &ev);
    p.finish(&ev);
    Message m;
    for (Event& e : ev) {
        if (e.kind == Event::Reasoning) {
            m.reasoning += e.text;
            m.has_reasoning = true;
        } else if (e.kind == Event::Content) m.content += e.text;
        else m.calls.push_back(std::move(e.call));
    }
    return m;
}

}  // namespace limbo::chat
