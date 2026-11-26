// Copyright (c) 2026 ITERFORM. All rights reserved.
// src/jinja.cpp — the Jinja subset interpreter and the JSON codec (see include/limbo/jinja.hpp).
//
// Written from the Jinja template language's public documentation and from observed jinja2 3.1
// behaviour (the fixtures in bench/fixtures/chat are rendered by it); no jinja/minja source was read.
#include "limbo/jinja.hpp"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <functional>
#include <stdexcept>
#include <unordered_map>

namespace limbo::jinja {

struct MacroObj {
    const void* node = nullptr;     // the Node of the {% macro %}
    std::weak_ptr<void> closure;    // the Frame it was defined in (weak: the frame owns the macro, so a strong
                                    // pointer back would be a cycle and leak the whole context)
    std::string name;
};

namespace {

// ---- errors ---------------------------------------------------------------------------------------

struct Err : std::runtime_error {  // a template error; line 0 means "not attributed yet"
    int line;
    explicit Err(const std::string& m, int l = 0) : std::runtime_error(m), line(l) {}
};
struct Raised : std::runtime_error {  // raise_exception(): the message is the whole error
    using std::runtime_error::runtime_error;
};
[[noreturn]] void fail(const std::string& m, int line = 0) { throw Err(m, line); }

// ---- UTF-8 ----------------------------------------------------------------------------------------

// Decodes the code point at s[i]; returns its length. Invalid bytes decode as U+FFFD, length 1.
size_t u8dec(std::string_view s, size_t i, uint32_t* cp) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    if (c < 0x80) {
        *cp = c;
        return 1;
    }
    size_t n = 0;
    uint32_t v = 0;
    if (c >= 0xC2 && c < 0xE0) n = 2, v = c & 0x1F;
    else if (c >= 0xE0 && c < 0xF0) n = 3, v = c & 0x0F;
    else if (c >= 0xF0 && c < 0xF5) n = 4, v = c & 0x07;
    if (n == 0 || i + n > s.size()) {
        *cp = 0xFFFD;
        return 1;
    }
    for (size_t k = 1; k < n; ++k) {
        const unsigned char cc = static_cast<unsigned char>(s[i + k]);
        if ((cc & 0xC0) != 0x80) {
            *cp = 0xFFFD;
            return 1;
        }
        v = (v << 6) | (cc & 0x3F);
    }
    if ((n == 3 && (v < 0x800 || (v >= 0xD800 && v <= 0xDFFF))) || (n == 4 && (v < 0x10000 || v > 0x10FFFF))) {
        *cp = 0xFFFD;
        return 1;
    }
    *cp = v;
    return n;
}

void u8enc(uint32_t cp, std::string& out) {
    if (cp < 0x80) out.push_back(char(cp));
    else if (cp < 0x800) {
        out.push_back(char(0xC0 | (cp >> 6)));
        out.push_back(char(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(char(0xE0 | (cp >> 12)));
        out.push_back(char(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(char(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(char(0xF0 | (cp >> 18)));
        out.push_back(char(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(char(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(char(0x80 | (cp & 0x3F)));
    }
}

size_t u8len(std::string_view s) {
    size_t n = 0;
    for (size_t i = 0; i < s.size(); ++n) {
        uint32_t cp;
        i += u8dec(s, i, &cp);
    }
    return n;
}

// Python's str.isspace() set.
bool is_py_space(uint32_t c) {
    return (c >= 9 && c <= 13) || (c >= 0x1C && c <= 0x20) || c == 0x85 || c == 0xA0 || c == 0x1680 ||
           (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

// str.strip / lstrip / rstrip. `chars` null = whitespace.
std::string py_strip(std::string_view s, const std::string* chars, bool left, bool right) {
    std::vector<uint32_t> set;
    if (chars) {
        for (size_t i = 0; i < chars->size();) {
            uint32_t cp;
            i += u8dec(*chars, i, &cp);
            set.push_back(cp);
        }
    }
    auto in_set = [&](uint32_t cp) {
        if (!chars) return is_py_space(cp);
        return std::find(set.begin(), set.end(), cp) != set.end();
    };
    size_t b = 0, e = s.size();
    if (left) {
        while (b < e) {
            uint32_t cp;
            const size_t l = u8dec(s, b, &cp);
            if (!in_set(cp)) break;
            b += l;
        }
    }
    if (right) {
        while (e > b) {
            size_t st = e - 1;
            while (st > b && (static_cast<unsigned char>(s[st]) & 0xC0) == 0x80) --st;
            uint32_t cp;
            const size_t l = u8dec(s, st, &cp);
            if (st + l != e) {  // not a clean sequence: the last byte alone
                st = e - 1;
                cp = 0xFFFD;
            }
            if (!in_set(cp)) break;
            e = st;
        }
    }
    return std::string(s.substr(b, e - b));
}

std::string rstrip_ws(std::string_view s) { return py_strip(s, nullptr, false, true); }

// The byte offset of every code point, plus the end: lets slicing and indexing work in characters.
std::vector<size_t> cp_offsets(std::string_view s) {
    std::vector<size_t> off;
    for (size_t i = 0; i < s.size();) {
        off.push_back(i);
        uint32_t cp;
        i += u8dec(s, i, &cp);
    }
    off.push_back(s.size());
    return off;
}

// ---- numbers --------------------------------------------------------------------------------------

bool is_num(const Value& v) { return v.is_number() || v.is_bool(); }  // Python: bool is an int
bool is_intlike(const Value& v) { return v.is_int() || v.is_bool(); }
int64_t as_i(const Value& v) { return v.is_bool() ? (v.b() ? 1 : 0) : v.i(); }
double as_d(const Value& v) { return v.is_float() ? v.d() : double(as_i(v)); }

}  // namespace

std::string py_float_repr(double d) {
    if (std::isnan(d)) return "nan";
    if (std::isinf(d)) return d < 0 ? "-inf" : "inf";
    char buf[64];
    const auto r = std::to_chars(buf, buf + sizeof buf, d, std::chars_format::scientific);
    const std::string sc(buf, r.ptr);
    size_t p = 0;
    const bool neg = sc[0] == '-';
    if (neg) p = 1;
    const size_t epos = sc.find('e');
    const int exp10 = std::atoi(sc.c_str() + epos + 1);
    std::string digits;
    for (size_t i = p; i < epos; ++i)
        if (sc[i] != '.') digits.push_back(sc[i]);
    const int decpt = exp10 + 1;
    std::string out = neg ? "-" : "";
    if (exp10 >= -4 && exp10 < 16) {
        if (decpt <= 0) {
            out += "0.";
            out.append(size_t(-decpt), '0');
            out += digits;
        } else if (size_t(decpt) >= digits.size()) {
            out += digits;
            out.append(size_t(decpt) - digits.size(), '0');
            out += ".0";
        } else {
            out += digits.substr(0, size_t(decpt));
            out += '.';
            out += digits.substr(size_t(decpt));
        }
    } else {
        out += digits[0];
        if (digits.size() > 1) {
            out += '.';
            out += digits.substr(1);
        }
        char eb[16];
        std::snprintf(eb, sizeof eb, "e%c%02d", exp10 < 0 ? '-' : '+', std::abs(exp10));
        out += eb;
    }
    return out;
}

// ---- Value ----------------------------------------------------------------------------------------

Value Value::undefined(std::string name) {
    Value v;
    v.s_ = std::move(name);
    return v;
}
Value Value::null() {
    Value v;
    v.k_ = Kind::Null;
    return v;
}
Value Value::boolean(bool b) {
    Value v;
    v.k_ = Kind::Bool;
    v.b_ = b;
    return v;
}
Value Value::integer(int64_t i) {
    Value v;
    v.k_ = Kind::Int;
    v.i_ = i;
    return v;
}
Value Value::number(double d) {
    Value v;
    v.k_ = Kind::Float;
    v.d_ = d;
    return v;
}
Value Value::string(std::string s) {
    Value v;
    v.k_ = Kind::String;
    v.s_ = std::move(s);
    return v;
}
Value Value::array() {
    Value v;
    v.k_ = Kind::List;
    v.l_ = std::make_shared<List>();
    return v;
}
Value Value::object() {
    Value v;
    v.k_ = Kind::Dict;
    v.o_ = std::make_shared<Dict>();
    return v;
}
Value Value::ns() {
    Value v;
    v.k_ = Kind::Namespace;
    v.o_ = std::make_shared<Dict>();
    return v;
}
Value Value::builtin(std::string name) {
    Value v;
    v.k_ = Kind::Builtin;
    v.s_ = std::move(name);
    return v;
}
Value Value::macro(std::shared_ptr<MacroObj> m) {
    Value v;
    v.k_ = Kind::Macro;
    v.m_ = std::move(m);
    return v;
}

const List& Value::list() const {
    static const List empty;
    return l_ ? *l_ : empty;
}
List& Value::list() {
    if (!l_) l_ = std::make_shared<List>();
    return *l_;
}
const Dict& Value::dict() const {
    static const Dict empty;
    return o_ ? *o_ : empty;
}
Dict& Value::dict() {
    if (!o_) o_ = std::make_shared<Dict>();
    return *o_;
}
size_t Value::size() const {
    switch (k_) {
        case Kind::List: return list().size();
        case Kind::Dict:
        case Kind::Namespace: return dict().size();
        case Kind::String: return s_.size();
        default: return 0;
    }
}
void Value::push(Value v) { list().push_back(std::move(v)); }
const Value* Value::find(std::string_view key) const {
    if (!o_) return nullptr;
    for (const auto& kv : *o_)
        if (kv.first == key) return &kv.second;
    return nullptr;
}
void Value::set(std::string_view key, Value v) {
    Dict& d = dict();
    for (auto& kv : d) {
        if (kv.first == key) {
            kv.second = std::move(v);
            return;
        }
    }
    d.emplace_back(std::string(key), std::move(v));
}
std::string Value::get_str(std::string_view key, std::string def) const {
    const Value* v = find(key);
    return v && v->is_string() ? v->s() : def;
}
int64_t Value::get_int(std::string_view key, int64_t def) const {
    const Value* v = find(key);
    if (!v) return def;
    if (v->is_int()) return v->i();
    if (v->is_float()) return int64_t(v->d());
    return def;
}

Value Value::clone() const {
    Value v = *this;
    if (k_ == Kind::List) {
        v.l_ = std::make_shared<List>();
        for (const Value& x : list()) v.l_->push_back(x.clone());
    } else if (k_ == Kind::Dict || k_ == Kind::Namespace) {
        v.o_ = std::make_shared<Dict>();
        for (const auto& kv : dict()) v.o_->emplace_back(kv.first, kv.second.clone());
    }
    return v;
}

bool Value::truthy() const {
    switch (k_) {
        case Kind::Undefined:
        case Kind::Null: return false;
        case Kind::Bool: return b_;
        case Kind::Int: return i_ != 0;
        case Kind::Float: return d_ != 0;
        case Kind::String: return !s_.empty();
        case Kind::List: return !list().empty();
        case Kind::Dict: return !dict().empty();
        default: return true;
    }
}

namespace {
std::string py_str_repr(const std::string& s) {
    const bool has_sq = s.find('\'') != std::string::npos, has_dq = s.find('"') != std::string::npos;
    const char q = (has_sq && !has_dq) ? '"' : '\'';
    std::string out(1, q);
    for (unsigned char c : s) {
        if (c == static_cast<unsigned char>(q) || c == '\\') {
            out += '\\';
            out += char(c);
        } else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else if (c < 0x20 || c == 0x7f) {
            char b[8];
            std::snprintf(b, sizeof b, "\\x%02x", c);
            out += b;
        } else out += char(c);
    }
    out += q;
    return out;
}
}  // namespace

std::string Value::repr() const {
    switch (k_) {
        case Kind::String: return py_str_repr(s_);
        case Kind::List: {
            std::string out = "[";
            bool first = true;
            for (const Value& v : list()) {
                if (!first) out += ", ";
                first = false;
                out += v.repr();
            }
            return out + "]";
        }
        case Kind::Dict: {
            std::string out = "{";
            bool first = true;
            for (const auto& kv : dict()) {
                if (!first) out += ", ";
                first = false;
                out += py_str_repr(kv.first) + ": " + kv.second.repr();
            }
            return out + "}";
        }
        case Kind::Namespace: {
            std::string out = "<Namespace {";
            bool first = true;
            for (const auto& kv : dict()) {
                if (!first) out += ", ";
                first = false;
                out += py_str_repr(kv.first) + ": " + kv.second.repr();
            }
            return out + "}>";
        }
        case Kind::Undefined: return "Undefined";
        default: return str();
    }
}

std::string Value::str() const {
    switch (k_) {
        case Kind::Undefined: return "";
        case Kind::Null: return "None";
        case Kind::Bool: return b_ ? "True" : "False";
        case Kind::Int: return std::to_string(i_);
        case Kind::Float: return py_float_repr(d_);
        case Kind::String: return s_;
        case Kind::Macro: return "<Macro " + (m_ ? m_->name : std::string()) + ">";
        case Kind::Builtin: return "<built-in function " + s_ + ">";
        default: return repr();
    }
}

bool operator==(const Value& a, const Value& b) {
    if (is_num(a) && is_num(b)) {
        if (is_intlike(a) && is_intlike(b)) return as_i(a) == as_i(b);
        return as_d(a) == as_d(b);
    }
    if (a.kind() != b.kind()) return false;
    switch (a.kind()) {
        case Value::Kind::Undefined:
        case Value::Kind::Null: return true;
        case Value::Kind::String: return a.s() == b.s();
        case Value::Kind::List: {
            const List &x = a.list(), &y = b.list();
            if (x.size() != y.size()) return false;
            for (size_t i = 0; i < x.size(); ++i)
                if (!(x[i] == y[i])) return false;
            return true;
        }
        case Value::Kind::Dict: {
            if (a.size() != b.size()) return false;
            for (const auto& kv : a.dict()) {
                const Value* o = b.find(kv.first);
                if (!o || !(kv.second == *o)) return false;
            }
            return true;
        }
        default: return false;
    }
}

// ---- JSON -----------------------------------------------------------------------------------------

namespace {

struct JsonParser {
    std::string_view t;
    size_t p = 0;
    std::string err;

    bool bad(const std::string& m) {
        if (err.empty()) err = m + " at byte " + std::to_string(p);
        return false;
    }
    void ws() {
        while (p < t.size() && (t[p] == ' ' || t[p] == '\t' || t[p] == '\n' || t[p] == '\r')) ++p;
    }
    bool lit(std::string_view w) {
        if (t.substr(p, w.size()) == w) {
            p += w.size();
            return true;
        }
        return false;
    }
    bool hex4(uint32_t* out) {
        if (p + 4 > t.size()) return bad("truncated \\u escape");
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = t[p + size_t(i)];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= uint32_t(c - '0');
            else if (c >= 'a' && c <= 'f') v |= uint32_t(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= uint32_t(c - 'A' + 10);
            else return bad("bad \\u escape");
        }
        p += 4;
        *out = v;
        return true;
    }
    bool str(std::string* out) {
        ++p;  // opening quote
        for (;;) {
            if (p >= t.size()) return bad("unterminated string");
            const char c = t[p++];
            if (c == '"') return true;
            if (c != '\\') {
                out->push_back(c);
                continue;
            }
            if (p >= t.size()) return bad("unterminated escape");
            const char e = t[p++];
            switch (e) {
                case '"': out->push_back('"'); break;
                case '\\': out->push_back('\\'); break;
                case '/': out->push_back('/'); break;
                case 'b': out->push_back('\b'); break;
                case 'f': out->push_back('\f'); break;
                case 'n': out->push_back('\n'); break;
                case 'r': out->push_back('\r'); break;
                case 't': out->push_back('\t'); break;
                case 'u': {
                    uint32_t cp;
                    if (!hex4(&cp)) return false;
                    if (cp >= 0xD800 && cp < 0xDC00) {
                        uint32_t lo = 0;
                        if (t.substr(p, 2) == "\\u") {
                            p += 2;
                            if (!hex4(&lo)) return false;
                        }
                        if (lo >= 0xDC00 && lo < 0xE000) cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        else cp = 0xFFFD;  // a lone surrogate has no UTF-8 form
                    } else if (cp >= 0xDC00 && cp < 0xE000) cp = 0xFFFD;
                    u8enc(cp, *out);
                    break;
                }
                default: return bad("bad escape");
            }
        }
    }
    bool value(Value* out, int depth) {
        if (depth > 256) return bad("nesting too deep");
        ws();
        if (p >= t.size()) return bad("unexpected end");
        const char c = t[p];
        if (c == '{') {
            ++p;
            Value o = Value::object();
            ws();
            if (p < t.size() && t[p] == '}') {
                ++p;
                *out = o;
                return true;
            }
            for (;;) {
                ws();
                if (p >= t.size() || t[p] != '"') return bad("expected a string key");
                std::string k;
                if (!str(&k)) return false;
                ws();
                if (p >= t.size() || t[p] != ':') return bad("expected ':'");
                ++p;
                Value v;
                if (!value(&v, depth + 1)) return false;
                o.set(k, std::move(v));
                ws();
                if (p < t.size() && t[p] == ',') {
                    ++p;
                    continue;
                }
                if (p < t.size() && t[p] == '}') {
                    ++p;
                    break;
                }
                return bad("expected ',' or '}'");
            }
            *out = o;
            return true;
        }
        if (c == '[') {
            ++p;
            Value a = Value::array();
            ws();
            if (p < t.size() && t[p] == ']') {
                ++p;
                *out = a;
                return true;
            }
            for (;;) {
                Value v;
                if (!value(&v, depth + 1)) return false;
                a.push(std::move(v));
                ws();
                if (p < t.size() && t[p] == ',') {
                    ++p;
                    continue;
                }
                if (p < t.size() && t[p] == ']') {
                    ++p;
                    break;
                }
                return bad("expected ',' or ']'");
            }
            *out = a;
            return true;
        }
        if (c == '"') {
            std::string s;
            if (!str(&s)) return false;
            *out = Value::string(std::move(s));
            return true;
        }
        if (lit("true")) return *out = Value::boolean(true), true;
        if (lit("false")) return *out = Value::boolean(false), true;
        if (lit("null")) return *out = Value::null(), true;
        if (c == '-' || (c >= '0' && c <= '9')) {
            const size_t s = p;
            if (t[p] == '-') ++p;
            if (p >= t.size() || t[p] < '0' || t[p] > '9') return bad("bad number");
            if (t[p] == '0') ++p;
            else
                while (p < t.size() && t[p] >= '0' && t[p] <= '9') ++p;
            bool is_float = false;
            if (p < t.size() && t[p] == '.') {
                is_float = true;
                ++p;
                if (p >= t.size() || t[p] < '0' || t[p] > '9') return bad("bad number");
                while (p < t.size() && t[p] >= '0' && t[p] <= '9') ++p;
            }
            if (p < t.size() && (t[p] == 'e' || t[p] == 'E')) {
                is_float = true;
                ++p;
                if (p < t.size() && (t[p] == '+' || t[p] == '-')) ++p;
                if (p >= t.size() || t[p] < '0' || t[p] > '9') return bad("bad number");
                while (p < t.size() && t[p] >= '0' && t[p] <= '9') ++p;
            }
            const std::string_view num = t.substr(s, p - s);
            if (!is_float) {
                int64_t iv = 0;
                const auto r = std::from_chars(num.data(), num.data() + num.size(), iv);
                if (r.ec == std::errc() && r.ptr == num.data() + num.size()) {
                    *out = Value::integer(iv);
                    return true;
                }
            }
            double dv = 0;
            const auto r = std::from_chars(num.data(), num.data() + num.size(), dv);
            if (r.ec != std::errc() && r.ec != std::errc::result_out_of_range) return bad("bad number");
            *out = Value::number(dv);
            return true;
        }
        return bad("unexpected character");
    }
};

void json_escape(const std::string& s, bool ascii, std::string& out) {
    out.push_back('"');
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        if (c < 0x80) {
            ++i;
            switch (c) {
                case '"': out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\n': out += "\\n"; break;
                case '\r': out += "\\r"; break;
                case '\t': out += "\\t"; break;
                case '\b': out += "\\b"; break;
                case '\f': out += "\\f"; break;
                default:
                    if (c < 0x20 || (ascii && c == 0x7f)) {
                        char b[8];
                        std::snprintf(b, sizeof b, "\\u%04x", c);
                        out += b;
                    } else out.push_back(char(c));
            }
            continue;
        }
        uint32_t cp;
        const size_t l = u8dec(s, i, &cp);
        if (!ascii) {
            out.append(s, i, l);
        } else if (cp >= 0x10000) {
            const uint32_t v = cp - 0x10000;
            char b[16];
            std::snprintf(b, sizeof b, "\\u%04x\\u%04x", 0xD800 + (v >> 10), 0xDC00 + (v & 0x3FF));
            out += b;
        } else {
            char b[8];
            std::snprintf(b, sizeof b, "\\u%04x", cp);
            out += b;
        }
        i += l;
    }
    out.push_back('"');
}

void dump_rec(const Value& v, const DumpOptions& o, int level, std::string& out) {
    auto newline = [&](int lv) {
        out.push_back('\n');
        out.append(size_t(lv) * size_t(o.indent), ' ');
    };
    switch (v.kind()) {
        case Value::Kind::Null: out += "null"; return;
        case Value::Kind::Bool: out += v.b() ? "true" : "false"; return;
        case Value::Kind::Int: out += std::to_string(v.i()); return;
        case Value::Kind::Float:
            if (std::isnan(v.d())) out += "NaN";
            else if (std::isinf(v.d())) out += v.d() < 0 ? "-Infinity" : "Infinity";
            else out += py_float_repr(v.d());
            return;
        case Value::Kind::String: json_escape(v.s(), o.ensure_ascii, out); return;
        case Value::Kind::List: {
            const List& l = v.list();
            if (l.empty()) {
                out += "[]";
                return;
            }
            out.push_back('[');
            for (size_t i = 0; i < l.size(); ++i) {
                if (i) out += o.indent >= 0 ? "," : o.item_sep;
                if (o.indent >= 0) newline(level + 1);
                dump_rec(l[i], o, level + 1, out);
            }
            if (o.indent >= 0) newline(level);
            out.push_back(']');
            return;
        }
        case Value::Kind::Dict: {
            Dict d = v.dict();
            if (d.empty()) {
                out += "{}";
                return;
            }
            if (o.sort_keys)
                std::stable_sort(d.begin(), d.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
            out.push_back('{');
            bool first = true;
            for (const auto& kv : d) {
                if (!first) out += o.indent >= 0 ? "," : o.item_sep;
                first = false;
                if (o.indent >= 0) newline(level + 1);
                json_escape(kv.first, o.ensure_ascii, out);
                out += o.key_sep;
                dump_rec(kv.second, o, level + 1, out);
            }
            if (o.indent >= 0) newline(level);
            out.push_back('}');
            return;
        }
        default: throw std::runtime_error("Object of this type is not JSON serializable");
    }
}

}  // namespace

bool json_parse(std::string_view text, Value* out, std::string* err) {
    JsonParser jp;
    jp.t = text;
    Value v;
    if (!jp.value(&v, 0)) {
        if (err) *err = jp.err;
        return false;
    }
    jp.ws();
    if (jp.p != text.size()) {
        if (err) *err = "trailing data at byte " + std::to_string(jp.p);
        return false;
    }
    *out = std::move(v);
    return true;
}

std::string json_dump(const Value& v, const DumpOptions& o) {
    std::string out;
    dump_rec(v, o, 0, out);
    return out;
}

std::string json_compact(const Value& v) {
    DumpOptions o;
    o.item_sep = ",";
    o.key_sep = ":";
    return json_dump(v, o);
}

// ---- the template: lexing -------------------------------------------------------------------------

namespace {

struct Tok {  // one piece of the template: text, {{ expr }} or {% block %}
    enum T { Data, Var, Block } t = Data;
    std::string text;
    int line = 1;
};

std::vector<Tok> lex_template(std::string src, const Options& opt) {
    // jinja2 normalises line endings and, unless told otherwise, drops one trailing newline.
    std::string s;
    s.reserve(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        if (src[i] == '\r') {
            s.push_back('\n');
            if (i + 1 < src.size() && src[i + 1] == '\n') ++i;
        } else s.push_back(src[i]);
    }
    if (!opt.keep_trailing_newline && !s.empty() && s.back() == '\n') s.pop_back();

    std::vector<Tok> out;
    const size_t n = s.size();
    size_t pos = 0;
    int line = 1;
    bool line_starting = true;
    auto count_nl = [&](size_t a, size_t b) {
        for (size_t i = a; i < b; ++i)
            if (s[i] == '\n') ++line;
    };
    while (pos < n) {
        size_t st = pos;
        for (;;) {
            st = s.find('{', st);
            if (st == std::string::npos || (st + 1 < n && (s[st + 1] == '%' || s[st + 1] == '{' || s[st + 1] == '#')))
                break;
            ++st;
        }
        if (st == std::string::npos) {
            Tok d;
            d.t = Tok::Data;
            d.text = s.substr(pos);
            d.line = line;
            out.push_back(std::move(d));
            break;
        }
        std::string text = s.substr(pos, st - pos);
        const char kind = s[st + 1];
        size_t q = st + 2;
        char marker = 0;
        if (q < n && (s[q] == '-' || s[q] == '+')) marker = s[q++];
        if (marker == '-') text = rstrip_ws(text);
        else if (marker != '+' && opt.lstrip_blocks && kind != '{') {
            const size_t nl = text.rfind('\n');
            const size_t l_pos = nl == std::string::npos ? 0 : nl + 1;
            if (l_pos > 0 || line_starting) {
                bool only_blank = true;
                for (size_t i = l_pos; i < text.size(); ++i)
                    if (text[i] != ' ' && text[i] != '\t') only_blank = false;
                if (only_blank) text.resize(l_pos);
            }
        }
        const int data_line = line;
        if (!text.empty()) {
            Tok d;
            d.t = Tok::Data;
            d.text = std::move(text);
            d.line = data_line;
            out.push_back(std::move(d));
        }
        count_nl(pos, st);
        const int tag_line = line;
        // The end of the tag: the first closer outside string literals with all brackets balanced.
        size_t e = q;
        if (kind == '#') {
            e = s.find("#}", q);
            if (e == std::string::npos) fail("unterminated comment", tag_line);
        } else {
            int depth = 0;
            bool found = false;
            for (; e < n; ++e) {
                const char c = s[e];
                if (c == '\'' || c == '"') {
                    const char qc = c;
                    for (++e; e < n && s[e] != qc; ++e)
                        if (s[e] == '\\') ++e;
                    continue;
                }
                if (c == '(' || c == '[' || c == '{') {
                    ++depth;
                } else if (c == ')' || c == ']') {
                    --depth;
                } else if (c == '}') {
                    if (depth > 0) --depth;  // closes a dict literal
                    else if (kind == '{' && e + 1 < n && s[e + 1] == '}') {
                        found = true;
                        break;
                    }
                } else if (c == '%' && kind == '%' && depth == 0 && e + 1 < n && s[e + 1] == '}') {
                    found = true;
                    break;
                }
            }
            if (!found) fail(std::string("unterminated '") + (kind == '%' ? "{%" : "{{") + "'", tag_line);
        }
        std::string content = s.substr(q, e - q);
        bool strip_after = false;
        if (!content.empty() && content.back() == '-') {
            strip_after = true;
            content.pop_back();
        }
        size_t after = e + 2;
        if (strip_after) {
            while (after < n) {
                uint32_t cp;
                const size_t l = u8dec(s, after, &cp);
                if (!is_py_space(cp)) break;
                after += l;
            }
        } else if (kind != '{' && opt.trim_blocks && after < n && s[after] == '\n') ++after;
        count_nl(st, after);
        line_starting = after > 0 && s[after - 1] == '\n';
        pos = after;
        if (kind == '#') continue;
        Tok t;
        t.t = kind == '{' ? Tok::Var : Tok::Block;
        t.text = std::move(content);
        t.line = tag_line;
        out.push_back(std::move(t));
    }
    return out;
}

// ---- the template: expression tokens --------------------------------------------------------------

struct ETok {
    enum T { Name, Int, Float, Str, Op, End } t = End;
    std::string s;
    int64_t i = 0;
    double d = 0;
};

// Python's unicode-escape decoding, which jinja applies to string literals.
std::string decode_escapes(std::string_view in, int line) {
    std::string out;
    for (size_t i = 0; i < in.size(); ++i) {
        if (in[i] != '\\' || i + 1 >= in.size()) {
            out.push_back(in[i]);
            continue;
        }
        const char e = in[++i];
        auto hex = [&](size_t len) {
            uint32_t v = 0;
            for (size_t k = 1; k <= len; ++k) {
                if (i + k >= in.size()) fail("bad escape in string literal", line);
                const char c = in[i + k];
                v <<= 4;
                if (c >= '0' && c <= '9') v |= uint32_t(c - '0');
                else if (c >= 'a' && c <= 'f') v |= uint32_t(c - 'a' + 10);
                else if (c >= 'A' && c <= 'F') v |= uint32_t(c - 'A' + 10);
                else fail("bad escape in string literal", line);
            }
            i += len;
            return v;
        };
        switch (e) {
            case 'n': out.push_back('\n'); break;
            case 't': out.push_back('\t'); break;
            case 'r': out.push_back('\r'); break;
            case '\\': out.push_back('\\'); break;
            case '\'': out.push_back('\''); break;
            case '"': out.push_back('"'); break;
            case 'a': out.push_back('\a'); break;
            case 'b': out.push_back('\b'); break;
            case 'f': out.push_back('\f'); break;
            case 'v': out.push_back('\v'); break;
            case '0': out.push_back('\0'); break;
            case 'x': u8enc(hex(2), out); break;
            case 'u': u8enc(hex(4), out); break;
            case 'U': u8enc(hex(8), out); break;
            case '\n': break;  // backslash-newline: line continuation
            default:
                out.push_back('\\');
                out.push_back(e);
        }
    }
    return out;
}

std::vector<ETok> lex_expr(std::string_view src, int line) {
    std::vector<ETok> out;
    size_t i = 0;
    const size_t n = src.size();
    while (i < n) {
        const unsigned char c = static_cast<unsigned char>(src[i]);
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            ++i;
            continue;
        }
        ETok t;
        if (std::isalpha(c) || c == '_') {
            size_t j = i;
            while (j < n && (std::isalnum(static_cast<unsigned char>(src[j])) || src[j] == '_')) ++j;
            t.t = ETok::Name;
            t.s = std::string(src.substr(i, j - i));
            i = j;
        } else if (std::isdigit(c)) {
            size_t j = i;
            while (j < n && std::isdigit(static_cast<unsigned char>(src[j]))) ++j;
            bool fl = false;
            if (j + 1 < n && src[j] == '.' && std::isdigit(static_cast<unsigned char>(src[j + 1]))) {
                fl = true;
                ++j;
                while (j < n && std::isdigit(static_cast<unsigned char>(src[j]))) ++j;
            }
            if (j < n && (src[j] == 'e' || src[j] == 'E')) {
                size_t k = j + 1;
                if (k < n && (src[k] == '+' || src[k] == '-')) ++k;
                if (k < n && std::isdigit(static_cast<unsigned char>(src[k]))) {
                    fl = true;
                    while (k < n && std::isdigit(static_cast<unsigned char>(src[k]))) ++k;
                    j = k;
                }
            }
            const std::string num(src.substr(i, j - i));
            if (fl) {
                t.t = ETok::Float;
                t.d = std::strtod(num.c_str(), nullptr);
            } else {
                t.t = ETok::Int;
                const auto r = std::from_chars(num.data(), num.data() + num.size(), t.i);
                if (r.ec != std::errc()) {
                    t.t = ETok::Float;
                    t.d = std::strtod(num.c_str(), nullptr);
                }
            }
            i = j;
        } else if (c == '\'' || c == '"') {
            size_t j = i + 1;
            while (j < n && src[j] != char(c)) {
                if (src[j] == '\\') ++j;
                ++j;
            }
            if (j >= n) fail("unterminated string literal", line);
            t.t = ETok::Str;
            t.s = decode_escapes(src.substr(i + 1, j - i - 1), line);
            i = j + 1;
        } else {
            t.t = ETok::Op;
            static const char* two[] = {"**", "//", "==", "!=", "<=", ">="};
            bool done = false;
            for (const char* o : two) {
                if (src.substr(i, 2) == o) {
                    t.s = o;
                    i += 2;
                    done = true;
                    break;
                }
            }
            if (!done) {
                if (!std::strchr("+-*/%~<>=()[]{},:.|", char(c))) fail(std::string("unexpected character '") + char(c) + "'", line);
                t.s = std::string(1, char(c));
                ++i;
            }
        }
        out.push_back(std::move(t));
    }
    out.emplace_back();  // End
    return out;
}

// ---- the AST --------------------------------------------------------------------------------------

enum class N : uint8_t {
    Data, Output, If, For, Set, SetBlock, Macro, Generation, Break, Continue,
    Const, Name, List, Tuple, Dict, Attr, Sub, Slice, Call, Filter, Test, Neg, Pos, Not, And, Or, Cond, Bin, Cmp,
};
struct Node;
using NodeP = std::unique_ptr<Node>;
struct Node {
    N t;
    int line = 0;
    std::string s;               // Data text, Name/Attr/Filter/Test name, Bin op, Macro name
    Value v;                     // Const
    bool flag = false;           // Test: negated
    std::vector<NodeP> a;        // operands / positional args (entries may be null: Slice, Cond else, For filter)
    std::vector<std::string> kn; // kwarg names, for-targets, macro params, Cmp operators
    std::vector<NodeP> kv;       // kwarg values, macro defaults (nullable)
    std::vector<NodeP> body, orelse;
    Node(N type, int l) : t(type), line(l) {}
};
NodeP mk(N t, int line) { return std::make_unique<Node>(t, line); }

// ---- the parser -----------------------------------------------------------------------------------

class ExprParser {
public:
    ExprParser(std::vector<ETok> toks, int line) : t_(std::move(toks)), line_(line) {}

    NodeP expr() { return condexpr(); }
    NodeP or_expr() { return or_(); }
    bool at_end() const { return cur().t == ETok::End; }
    const ETok& cur() const { return t_[p_]; }
    bool is_name(const char* n) const { return cur().t == ETok::Name && cur().s == n; }
    bool is_op(const char* o) const { return cur().t == ETok::Op && cur().s == o; }
    void next() {
        if (p_ + 1 < t_.size()) ++p_;
    }
    bool accept_name(const char* n) {
        if (!is_name(n)) return false;
        next();
        return true;
    }
    bool accept_op(const char* o) {
        if (!is_op(o)) return false;
        next();
        return true;
    }
    void expect_op(const char* o) {
        if (!accept_op(o)) fail(std::string("expected '") + o + "'", line_);
    }
    std::string expect_name() {
        if (cur().t != ETok::Name) fail("expected a name", line_);
        std::string s = cur().s;
        next();
        return s;
    }
    void expect_end() {
        if (!at_end()) fail("unexpected '" + (cur().t == ETok::Str ? "string" : cur().s) + "'", line_);
    }
    int line() const { return line_; }

    // targets for {% for %} and {% set %}: names, a.attr, or tuples of names.
    NodeP target() {
        if (accept_op("(")) {
            NodeP t = target_list(")");
            expect_op(")");
            return t;
        }
        NodeP first = target_atom();
        if (is_op(",")) {
            NodeP tup = mk(N::Tuple, line_);
            tup->a.push_back(std::move(first));
            while (accept_op(",")) {
                if (is_name("in") || is_op("=") || at_end()) break;
                tup->a.push_back(target_atom());
            }
            return tup;
        }
        return first;
    }

private:
    std::vector<ETok> t_;
    size_t p_ = 0;
    int line_;
    int depth_ = 0;

    NodeP target_atom() {
        NodeP n = mk(N::Name, line_);
        n->s = expect_name();
        if (is_op(".")) {
            next();
            NodeP at = mk(N::Attr, line_);
            at->a.push_back(std::move(n));
            at->s = expect_name();
            return at;
        }
        return n;
    }
    NodeP target_list(const char* closer) {
        NodeP tup = mk(N::Tuple, line_);
        while (!is_op(closer)) {
            if (!tup->a.empty()) expect_op(",");
            if (is_op(closer)) break;
            tup->a.push_back(target_atom());
        }
        return tup;
    }

    NodeP condexpr() {
        if (++depth_ > 200) fail("expression nested too deeply", line_);
        NodeP e1 = or_();
        while (is_name("if")) {
            next();
            NodeP c = or_();
            NodeP e3;
            if (accept_name("else")) e3 = condexpr();
            NodeP n = mk(N::Cond, line_);
            n->a.push_back(std::move(e1));
            n->a.push_back(std::move(c));
            n->a.push_back(std::move(e3));
            e1 = std::move(n);
        }
        --depth_;
        return e1;
    }
    NodeP or_() {
        NodeP l = and_();
        while (is_name("or")) {
            next();
            NodeP n = mk(N::Or, line_);
            n->a.push_back(std::move(l));
            n->a.push_back(and_());
            l = std::move(n);
        }
        return l;
    }
    NodeP and_() {
        NodeP l = not_();
        while (is_name("and")) {
            next();
            NodeP n = mk(N::And, line_);
            n->a.push_back(std::move(l));
            n->a.push_back(not_());
            l = std::move(n);
        }
        return l;
    }
    NodeP not_() {
        if (is_name("not")) {
            next();
            if (++depth_ > 200) fail("expression nested too deeply", line_);
            NodeP n = mk(N::Not, line_);
            n->a.push_back(not_());
            --depth_;
            return n;
        }
        return compare();
    }
    NodeP compare() {
        NodeP first = concat();
        NodeP cmp;
        for (;;) {
            std::string op;
            if (cur().t == ETok::Op && (cur().s == "==" || cur().s == "!=" || cur().s == "<" || cur().s == ">" ||
                                        cur().s == "<=" || cur().s == ">=")) {
                op = cur().s;
                next();
            } else if (is_name("in")) {
                op = "in";
                next();
            } else if (is_name("not") && p_ + 1 < t_.size() && t_[p_ + 1].t == ETok::Name && t_[p_ + 1].s == "in") {
                op = "notin";
                next();
                next();
            } else break;
            if (!cmp) {
                cmp = mk(N::Cmp, line_);
                cmp->a.push_back(std::move(first));
            }
            cmp->kn.push_back(op);
            cmp->a.push_back(concat());
        }
        return cmp ? std::move(cmp) : std::move(first);
    }
    NodeP bin(const char* op, NodeP l, NodeP r) {
        NodeP n = mk(N::Bin, line_);
        n->s = op;
        n->a.push_back(std::move(l));
        n->a.push_back(std::move(r));
        return n;
    }
    NodeP concat() {
        NodeP l = add();
        while (is_op("~")) {
            next();
            l = bin("~", std::move(l), add());
        }
        return l;
    }
    NodeP add() {
        NodeP l = mul();
        while (is_op("+") || is_op("-")) {
            const char* op = cur().s == "+" ? "+" : "-";
            next();
            l = bin(op, std::move(l), mul());
        }
        return l;
    }
    NodeP mul() {
        NodeP l = pow_();
        while (is_op("*") || is_op("/") || is_op("//") || is_op("%")) {
            const std::string op = cur().s;
            next();
            l = bin(op.c_str(), std::move(l), pow_());
        }
        return l;
    }
    NodeP pow_() {
        NodeP l = unary(true);
        while (is_op("**")) {
            next();
            l = bin("**", std::move(l), unary(true));
        }
        return l;
    }
    NodeP unary(bool with_filter) {
        NodeP n;
        if (is_op("-") || is_op("+")) {
            const bool neg = cur().s == "-";
            next();
            if (++depth_ > 200) fail("expression nested too deeply", line_);
            n = mk(neg ? N::Neg : N::Pos, line_);
            n->a.push_back(unary(false));
            --depth_;
        } else n = primary();
        n = postfix(std::move(n));
        if (with_filter) n = filter_expr(std::move(n));
        return n;
    }
    NodeP primary() {
        const ETok t = cur();
        if (t.t == ETok::Name) {
            next();
            if (t.s == "true" || t.s == "True" || t.s == "false" || t.s == "False") {
                NodeP n = mk(N::Const, line_);
                n->v = Value::boolean(t.s == "true" || t.s == "True");
                return n;
            }
            if (t.s == "none" || t.s == "None") {
                NodeP n = mk(N::Const, line_);
                n->v = Value::null();
                return n;
            }
            NodeP n = mk(N::Name, line_);
            n->s = t.s;
            return n;
        }
        if (t.t == ETok::Str) {
            std::string s = t.s;
            next();
            while (cur().t == ETok::Str) {  // adjacent literals concatenate
                s += cur().s;
                next();
            }
            NodeP n = mk(N::Const, line_);
            n->v = Value::string(std::move(s));
            return n;
        }
        if (t.t == ETok::Int || t.t == ETok::Float) {
            next();
            NodeP n = mk(N::Const, line_);
            n->v = t.t == ETok::Int ? Value::integer(t.i) : Value::number(t.d);
            return n;
        }
        if (t.t == ETok::Op && t.s == "(") {
            next();
            if (accept_op(")")) return mk(N::Tuple, line_);
            if (++depth_ > 200) fail("expression nested too deeply", line_);
            NodeP e = expr();
            --depth_;
            if (is_op(",")) {
                NodeP tup = mk(N::Tuple, line_);
                tup->a.push_back(std::move(e));
                while (accept_op(",")) {
                    if (is_op(")")) break;
                    tup->a.push_back(expr());
                }
                expect_op(")");
                return tup;
            }
            expect_op(")");
            return e;
        }
        if (t.t == ETok::Op && t.s == "[") {
            next();
            NodeP l = mk(N::List, line_);
            while (!is_op("]")) {
                if (!l->a.empty()) expect_op(",");
                if (is_op("]")) break;
                l->a.push_back(expr());
            }
            expect_op("]");
            return l;
        }
        if (t.t == ETok::Op && t.s == "{") {
            next();
            NodeP d = mk(N::Dict, line_);
            while (!is_op("}")) {
                if (!d->a.empty()) expect_op(",");
                if (is_op("}")) break;
                d->a.push_back(expr());
                expect_op(":");
                d->a.push_back(expr());
            }
            expect_op("}");
            return d;
        }
        if (t.t == ETok::End) fail("unexpected end of expression", line_);
        fail("unexpected '" + t.s + "'", line_);
    }
    NodeP postfix(NodeP n) {
        for (;;) {
            if (is_op(".")) {
                next();
                NodeP at = mk(N::Attr, line_);
                at->a.push_back(std::move(n));
                if (cur().t == ETok::Name) at->s = cur().s;
                else if (cur().t == ETok::Int) at->s = std::to_string(cur().i);
                else fail("expected an attribute name after '.'", line_);
                next();
                n = std::move(at);
            } else if (is_op("[")) {
                next();
                n = subscript(std::move(n));
            } else if (is_op("(")) {
                n = call(std::move(n));
            } else break;
        }
        return n;
    }
    NodeP subscript(NodeP obj) {
        NodeP sub = mk(N::Sub, line_);
        sub->a.push_back(std::move(obj));
        NodeP s0, s1, s2;
        bool slice = false;
        if (is_op(":")) {
            slice = true;
        } else {
            s0 = expr();
            if (is_op(":")) slice = true;
        }
        if (slice) {
            next();  // the first ':'
            if (!is_op(":") && !is_op("]")) s1 = expr();
            if (accept_op(":")) {
                if (!is_op("]")) s2 = expr();
            }
            NodeP sl = mk(N::Slice, line_);
            sl->a.push_back(std::move(s0));
            sl->a.push_back(std::move(s1));
            sl->a.push_back(std::move(s2));
            sub->a.push_back(std::move(sl));
        } else sub->a.push_back(std::move(s0));
        expect_op("]");
        return sub;
    }
    void call_args(Node& n) {
        expect_op("(");
        while (!is_op(")")) {
            if (cur().t == ETok::Name && p_ + 1 < t_.size() && t_[p_ + 1].t == ETok::Op && t_[p_ + 1].s == "=") {
                n.kn.push_back(cur().s);
                next();
                next();
                n.kv.push_back(expr());
            } else {
                if (!n.kn.empty()) fail("positional argument follows keyword argument", line_);
                n.a.push_back(expr());
            }
            if (!accept_op(",")) break;
        }
        expect_op(")");
    }
    NodeP call(NodeP callee) {
        NodeP n = mk(N::Call, line_);
        n->a.push_back(std::move(callee));
        call_args(*n);
        return n;
    }
    NodeP filter_expr(NodeP n) {
        for (;;) {
            if (is_op("|")) {
                next();
                NodeP f = mk(N::Filter, line_);
                f->a.push_back(std::move(n));
                f->s = expect_name();
                while (is_op(".")) {
                    next();
                    f->s += "." + expect_name();
                }
                if (is_op("(")) call_args(*f);
                n = std::move(f);
            } else if (is_name("is")) {
                next();
                NodeP tn = mk(N::Test, line_);
                tn->a.push_back(std::move(n));
                if (accept_name("not")) tn->flag = true;
                tn->s = expect_name();
                while (is_op(".")) {
                    next();
                    tn->s += "." + expect_name();
                }
                if (is_op("(")) {
                    call_args(*tn);
                } else if ((cur().t == ETok::Name || cur().t == ETok::Str || cur().t == ETok::Int ||
                            cur().t == ETok::Float || is_op("(") || is_op("[") || is_op("{")) &&
                           !is_name("else") && !is_name("or") && !is_name("and") && !is_name("if") &&
                           !is_name("in") && !is_name("not")) {
                    if (is_name("is")) fail("you cannot chain multiple tests with 'is'", line_);
                    tn->a.push_back(postfix(primary()));
                }
                n = std::move(tn);
            } else if (is_op("(")) {
                n = call(std::move(n));
            } else break;
        }
        return n;
    }
};

// ---- the statement parser -------------------------------------------------------------------------

struct Block {  // a {% block %} tag split into keyword and the rest
    std::string kw, rest;
};
Block split_block(const std::string& content) {
    size_t i = 0;
    while (i < content.size() && std::isspace(static_cast<unsigned char>(content[i]))) ++i;
    size_t j = i;
    while (j < content.size() && (std::isalnum(static_cast<unsigned char>(content[j])) || content[j] == '_')) ++j;
    return {content.substr(i, j - i), content.substr(j)};
}

class StmtParser {
public:
    explicit StmtParser(const std::vector<Tok>& toks) : t_(toks) {}

    std::vector<NodeP> parse_all() {
        std::string stop;
        std::vector<NodeP> body = parse_until({}, &stop);
        if (p_ < t_.size()) fail("unexpected '{% " + split_block(t_[p_].text).kw + " %}'", t_[p_].line);
        return body;
    }

private:
    const std::vector<Tok>& t_;
    size_t p_ = 0;
    int depth_ = 0;

    ExprParser ep(const std::string& text, int line) { return ExprParser(lex_expr(text, line), line); }

    // Parses nodes until a block whose keyword is in `stops` (left unconsumed, its keyword in *stop),
    // or until the tokens run out when `stops` is empty.
    std::vector<NodeP> parse_until(std::initializer_list<const char*> stops, std::string* stop) {
        std::vector<NodeP> out;
        if (++depth_ > 100) fail("blocks nested too deeply", t_.empty() ? 0 : t_[std::min(p_, t_.size() - 1)].line);
        while (p_ < t_.size()) {
            const Tok& tk = t_[p_];
            if (tk.t == Tok::Data) {
                NodeP n = mk(N::Data, tk.line);
                n->s = tk.text;
                out.push_back(std::move(n));
                ++p_;
            } else if (tk.t == Tok::Var) {
                ExprParser e = ep(tk.text, tk.line);
                NodeP n = mk(N::Output, tk.line);
                n->a.push_back(e.expr());
                e.expect_end();
                out.push_back(std::move(n));
                ++p_;
            } else {
                const Block b = split_block(tk.text);
                for (const char* s : stops) {
                    if (b.kw == s) {
                        *stop = b.kw;
                        --depth_;
                        return out;
                    }
                }
                out.push_back(statement(b, tk.line));
            }
        }
        if (stops.size() != 0) fail("unexpected end of template, expected '{% " + std::string(*stops.begin()) + " %}'", t_.empty() ? 0 : t_.back().line);
        --depth_;
        return out;
    }

    void expect_block(const char* kw, int line) {
        if (p_ >= t_.size() || t_[p_].t != Tok::Block || split_block(t_[p_].text).kw != kw)
            fail(std::string("expected '{% ") + kw + " %}'", line);
        ++p_;
    }

    NodeP statement(const Block& b, int line) {
        ++p_;  // the block token
        const std::string& kw = b.kw;
        if (kw == "if") return parse_if(b.rest, line);
        if (kw == "for") return parse_for(b.rest, line);
        if (kw == "set") return parse_set(b.rest, line);
        if (kw == "macro") return parse_macro(b.rest, line);
        if (kw == "generation") return parse_generation(b.rest, line);
        if (kw == "break" || kw == "continue") {
            ep(b.rest, line).expect_end();
            return mk(kw == "break" ? N::Break : N::Continue, line);
        }
        if (kw == "endif" || kw == "endfor" || kw == "endmacro" || kw == "endset" || kw == "endgeneration" ||
            kw == "else" || kw == "elif")
            fail("unexpected '{% " + kw + " %}'", line);
        fail("unsupported statement '{% " + kw + " %}'", line);
    }

    NodeP parse_if(const std::string& rest, int line) {
        ExprParser e = ep(rest, line);
        NodeP n = mk(N::If, line);
        n->a.push_back(e.expr());
        e.expect_end();
        std::string stop;
        n->body = parse_until({"elif", "else", "endif"}, &stop);
        const Tok& st = t_[p_];
        if (stop == "elif") {
            ++p_;
            n->orelse.push_back(parse_if(split_block(st.text).rest, st.line));
        } else if (stop == "else") {
            ++p_;
            n->orelse = parse_until({"endif"}, &stop);
            ++p_;
        } else ++p_;
        return n;
    }

    NodeP parse_for(const std::string& rest, int line) {
        ExprParser e = ep(rest, line);
        NodeP n = mk(N::For, line);
        NodeP tgt = e.target();
        if (tgt->t == N::Name) n->kn.push_back(tgt->s);
        else if (tgt->t == N::Tuple) {
            for (auto& x : tgt->a) {
                if (x->t != N::Name) fail("for-loop target must be names", line);
                n->kn.push_back(x->s);
            }
        } else fail("for-loop target must be names", line);
        if (!e.accept_name("in")) fail("expected 'in' in for-loop", line);
        n->a.push_back(e.or_expr());
        if (e.accept_name("if")) n->a.push_back(e.expr());
        else n->a.push_back(nullptr);
        if (e.is_name("recursive")) fail("recursive loops are not supported", line);
        e.expect_end();
        std::string stop;
        n->body = parse_until({"else", "endfor"}, &stop);
        if (stop == "else") {
            ++p_;
            n->orelse = parse_until({"endfor"}, &stop);
        }
        ++p_;
        return n;
    }

    NodeP parse_set(const std::string& rest, int line) {
        ExprParser e = ep(rest, line);
        NodeP tgt = e.target();
        if (e.accept_op("=")) {
            NodeP n = mk(N::Set, line);
            n->a.push_back(std::move(tgt));
            if (e.at_end()) fail("expected a value after '='", line);
            NodeP val = e.expr();
            if (e.is_op(",")) {  // a tuple value
                NodeP tup = mk(N::Tuple, line);
                tup->a.push_back(std::move(val));
                while (e.accept_op(",")) {
                    if (e.at_end()) break;
                    tup->a.push_back(e.expr());
                }
                val = std::move(tup);
            }
            e.expect_end();
            n->a.push_back(std::move(val));
            return n;
        }
        e.expect_end();
        NodeP n = mk(N::SetBlock, line);
        n->a.push_back(std::move(tgt));
        std::string stop;
        n->body = parse_until({"endset"}, &stop);
        ++p_;
        return n;
    }

    // {% generation %} ... {% endgeneration %} -- the transformers extension that marks the span the
    // assistant generated (return_assistant_tokens_mask). The reference implementation is a jinja2
    // Extension whose block emits its body unchanged and records the character span beside it
    // (transformers/utils/chat_template_utils.py, AssistantTracker), and it discards any argument the
    // tag carries. So for rendering this is a transparent wrapper: emit the body, record nothing. This
    // engine renders prompts; the span is a training-time mask, not a prompt byte.
    NodeP parse_generation(const std::string& /*rest*/, int line) {
        NodeP n = mk(N::Generation, line);
        std::string stop;
        n->body = parse_until({"endgeneration"}, &stop);
        ++p_;
        return n;
    }

    NodeP parse_macro(const std::string& rest, int line) {
        ExprParser e = ep(rest, line);
        NodeP n = mk(N::Macro, line);
        n->s = e.expect_name();
        e.expect_op("(");
        while (!e.is_op(")")) {
            n->kn.push_back(e.expect_name());
            if (e.accept_op("=")) n->kv.push_back(e.expr());
            else n->kv.push_back(nullptr);
            if (!e.accept_op(",")) break;
        }
        e.expect_op(")");
        e.expect_end();
        std::string stop;
        n->body = parse_until({"endmacro"}, &stop);
        ++p_;
        return n;
    }
};

// ---- the interpreter ------------------------------------------------------------------------------

struct Frame {
    std::unordered_map<std::string, Value> vars;
    std::shared_ptr<Frame> parent;
};
using FramePtr = std::shared_ptr<Frame>;
using Args = std::vector<Value>;
using Kwargs = std::vector<std::pair<std::string, Value>>;

enum class Flow { Normal, Break, Continue };

// Python slice bounds over n elements.
struct SliceBounds {
    int64_t b = 0, e = 0, st = 1;
};
SliceBounds slice_bounds(int64_t n, const Value& start, const Value& stop, const Value& step) {
    auto get = [](const Value& v, int64_t def) {
        if (v.is_undefined() || v.is_null()) return def;
        if (!is_intlike(v)) fail("slice indices must be integers");
        return as_i(v);
    };
    SliceBounds r;
    r.st = get(step, 1);
    if (r.st == 0) fail("slice step cannot be zero");
    r.b = get(start, r.st < 0 ? INT64_MAX : 0);
    r.e = get(stop, r.st < 0 ? INT64_MIN : INT64_MAX);
    auto adjust = [&](int64_t& x) {
        if (x < 0) {
            x += n;
            if (x < 0) x = r.st < 0 ? -1 : 0;
        } else if (x >= n) x = r.st < 0 ? n - 1 : n;
    };
    adjust(r.b);
    adjust(r.e);
    return r;
}

class Interp {
public:
    explicit Interp(const std::vector<NodeP>& root) : root_(root) {}

    std::string run(const Value& context) {
        FramePtr top = std::make_shared<Frame>();
        for (const char* g : {"namespace", "raise_exception", "range", "dict", "strftime_now"})
            top->vars[g] = Value::builtin(g);
        for (const auto& kv : context.dict()) top->vars[kv.first] = kv.second;
        std::string out;
        exec_body(root_, top, out);
        return out;
    }

private:
    const std::vector<NodeP>& root_;
    int call_depth_ = 0;

    // ---- statements ----

    Flow exec_body(const std::vector<NodeP>& body, const FramePtr& f, std::string& out) {
        for (const NodeP& n : body) {
            const Flow fl = exec(*n, f, out);
            if (fl != Flow::Normal) return fl;
        }
        return Flow::Normal;
    }

    Flow exec(const Node& n, const FramePtr& f, std::string& out) {
        try {
            return exec_inner(n, f, out);
        } catch (Err& e) {
            if (e.line == 0) e.line = n.line;
            throw;
        }
    }

    std::vector<Value> iter_items(const Value& v) {
        std::vector<Value> r;
        switch (v.kind()) {
            case Value::Kind::List: return v.list();
            case Value::Kind::Dict:
                for (const auto& kv : v.dict()) r.push_back(Value::string(kv.first));
                return r;
            case Value::Kind::String: {
                const std::string& s = v.s();
                for (size_t i = 0; i < s.size();) {
                    uint32_t cp;
                    const size_t l = u8dec(s, i, &cp);
                    r.push_back(Value::string(s.substr(i, l)));
                    i += l;
                }
                return r;
            }
            case Value::Kind::Undefined: return r;
            default: fail("object is not iterable");
        }
    }

    void bind(const Node& tgt_owner, const std::vector<std::string>& names, const Value& item, Frame& fr) {
        if (names.size() == 1) {
            fr.vars[names[0]] = item;
            return;
        }
        const std::vector<Value> parts = iter_items(item);
        if (parts.size() != names.size())
            fail("cannot unpack " + std::to_string(parts.size()) + " values into " + std::to_string(names.size()) + " names", tgt_owner.line);
        for (size_t i = 0; i < names.size(); ++i) fr.vars[names[i]] = parts[i];
    }

    void assign(const Node& tgt, const Value& val, const FramePtr& f) {
        if (tgt.t == N::Name) {
            f->vars[tgt.s] = val;
        } else if (tgt.t == N::Attr) {
            Value obj = eval(*tgt.a[0], f);
            if (obj.kind() != Value::Kind::Namespace) fail("cannot assign attribute '" + tgt.s + "' on a non-namespace object");
            obj.set(tgt.s, val);
        } else if (tgt.t == N::Tuple) {
            const std::vector<Value> parts = iter_items(val);
            if (parts.size() != tgt.a.size()) fail("cannot unpack " + std::to_string(parts.size()) + " values into " + std::to_string(tgt.a.size()) + " names");
            for (size_t i = 0; i < parts.size(); ++i) assign(*tgt.a[i], parts[i], f);
        } else fail("invalid assignment target");
    }

    Flow exec_inner(const Node& n, const FramePtr& f, std::string& out) {
        switch (n.t) {
            case N::Data: out += n.s; return Flow::Normal;
            case N::Output: out += eval(*n.a[0], f).str(); return Flow::Normal;
            case N::If:
                if (eval(*n.a[0], f).truthy()) return exec_body(n.body, f, out);
                return exec_body(n.orelse, f, out);
            case N::For: return exec_for(n, f, out);
            case N::Set: assign(*n.a[0], eval(*n.a[1], f), f); return Flow::Normal;
            case N::SetBlock: {
                std::string cap;
                exec_body(n.body, f, cap);
                assign(*n.a[0], Value::string(std::move(cap)), f);
                return Flow::Normal;
            }
            case N::Macro: {
                auto m = std::make_shared<MacroObj>();
                m->node = &n;
                m->closure = f;
                m->name = n.s;
                f->vars[n.s] = Value::macro(std::move(m));
                return Flow::Normal;
            }
            case N::Generation: return exec_body(n.body, f, out);
            case N::Break: return Flow::Break;
            case N::Continue: return Flow::Continue;
            default: fail("not a statement");
        }
    }

    Flow exec_for(const Node& n, const FramePtr& f, std::string& out) {
        std::vector<Value> items = iter_items(eval(*n.a[0], f));
        if (n.a[1]) {
            std::vector<Value> kept;
            for (const Value& it : items) {
                auto tmp = std::make_shared<Frame>();
                tmp->parent = f;
                bind(n, n.kn, it, *tmp);
                if (eval(*n.a[1], tmp).truthy()) kept.push_back(it);
            }
            items.swap(kept);
        }
        if (items.empty()) return exec_body(n.orelse, f, out);
        const int64_t len = int64_t(items.size());
        for (int64_t idx = 0; idx < len; ++idx) {
            auto fr = std::make_shared<Frame>();
            fr->parent = f;
            bind(n, n.kn, items[size_t(idx)], *fr);
            Value loop = Value::object();
            loop.set("index0", Value::integer(idx));
            loop.set("index", Value::integer(idx + 1));
            loop.set("revindex0", Value::integer(len - idx - 1));
            loop.set("revindex", Value::integer(len - idx));
            loop.set("first", Value::boolean(idx == 0));
            loop.set("last", Value::boolean(idx == len - 1));
            loop.set("length", Value::integer(len));
            loop.set("previtem", idx > 0 ? items[size_t(idx - 1)] : Value::undefined("loop.previtem"));
            loop.set("nextitem", idx + 1 < len ? items[size_t(idx + 1)] : Value::undefined("loop.nextitem"));
            fr->vars["loop"] = std::move(loop);
            const Flow fl = exec_body(n.body, fr, out);
            if (fl == Flow::Break) break;
        }
        return Flow::Normal;
    }

    // ---- expressions ----

    const Value* lookup(const Frame* fr, const std::string& name) const {
        for (; fr; fr = fr->parent.get()) {
            const auto it = fr->vars.find(name);
            if (it != fr->vars.end()) return &it->second;
        }
        return nullptr;
    }

    [[noreturn]] void undefined_error(const Value& v) const {
        fail(v.s().empty() ? "an undefined value was used" : "'" + v.s() + "' is undefined");
    }

    Value attr(const Value& obj, const std::string& name) const {
        switch (obj.kind()) {
            case Value::Kind::Undefined: undefined_error(obj);
            case Value::Kind::Dict:
            case Value::Kind::Namespace: {
                const Value* v = obj.find(name);
                return v ? *v : Value::undefined(name);
            }
            default: return Value::undefined(name);
        }
    }

    Value getitem(const Value& obj, const Value& idx) const {
        switch (obj.kind()) {
            case Value::Kind::Undefined: undefined_error(obj);
            case Value::Kind::List: {
                if (!is_intlike(idx)) return Value::undefined();
                int64_t i = as_i(idx);
                const int64_t n = int64_t(obj.list().size());
                if (i < 0) i += n;
                if (i < 0 || i >= n) return Value::undefined();
                return obj.list()[size_t(i)];
            }
            case Value::Kind::Dict:
            case Value::Kind::Namespace: {
                if (!idx.is_string()) return Value::undefined();
                const Value* v = obj.find(idx.s());
                return v ? *v : Value::undefined(idx.s());
            }
            case Value::Kind::String: {
                if (!is_intlike(idx)) return Value::undefined();
                const std::vector<size_t> off = cp_offsets(obj.s());
                int64_t i = as_i(idx);
                const int64_t n = int64_t(off.size()) - 1;
                if (i < 0) i += n;
                if (i < 0 || i >= n) return Value::undefined();
                return Value::string(obj.s().substr(off[size_t(i)], off[size_t(i) + 1] - off[size_t(i)]));
            }
            default: return Value::undefined();
        }
    }

    Value do_slice(const Value& obj, const Value& a, const Value& b, const Value& c) const {
        if (obj.is_undefined()) undefined_error(obj);
        if (obj.is_list()) {
            const List& l = obj.list();
            const SliceBounds sb = slice_bounds(int64_t(l.size()), a, b, c);
            Value out = Value::array();
            if (sb.st > 0)
                for (int64_t i = sb.b; i < sb.e; i += sb.st) out.push(l[size_t(i)]);
            else
                for (int64_t i = sb.b; i > sb.e; i += sb.st) out.push(l[size_t(i)]);
            return out;
        }
        if (obj.is_string()) {
            const std::vector<size_t> off = cp_offsets(obj.s());
            const SliceBounds sb = slice_bounds(int64_t(off.size()) - 1, a, b, c);
            std::string out;
            auto take = [&](int64_t i) { out.append(obj.s(), off[size_t(i)], off[size_t(i) + 1] - off[size_t(i)]); };
            if (sb.st > 0)
                for (int64_t i = sb.b; i < sb.e; i += sb.st) take(i);
            else
                for (int64_t i = sb.b; i > sb.e; i += sb.st) take(i);
            return Value::string(std::move(out));
        }
        fail("object is not sliceable");
    }

    // ---- operators ----

    Value arith(const std::string& op, const Value& a, const Value& b) const {
        if (a.is_undefined()) undefined_error(a);
        if (b.is_undefined()) undefined_error(b);
        if (op == "~") return Value::string(a.str() + b.str());
        if (op == "+") {
            if (is_num(a) && is_num(b)) {
                if (is_intlike(a) && is_intlike(b)) {
                    int64_t r;
                    if (!__builtin_add_overflow(as_i(a), as_i(b), &r)) return Value::integer(r);
                }
                return Value::number(as_d(a) + as_d(b));
            }
            if (a.is_string() && b.is_string()) return Value::string(a.s() + b.s());
            if (a.is_list() && b.is_list()) {
                Value out = Value::array();
                for (const Value& v : a.list()) out.push(v);
                for (const Value& v : b.list()) out.push(v);
                return out;
            }
        } else if (op == "-") {
            if (is_num(a) && is_num(b)) {
                if (is_intlike(a) && is_intlike(b)) {
                    int64_t r;
                    if (!__builtin_sub_overflow(as_i(a), as_i(b), &r)) return Value::integer(r);
                }
                return Value::number(as_d(a) - as_d(b));
            }
        } else if (op == "*") {
            if (is_num(a) && is_num(b)) {
                if (is_intlike(a) && is_intlike(b)) {
                    int64_t r;
                    if (!__builtin_mul_overflow(as_i(a), as_i(b), &r)) return Value::integer(r);
                }
                return Value::number(as_d(a) * as_d(b));
            }
            const Value* seq = a.is_string() || a.is_list() ? &a : (b.is_string() || b.is_list() ? &b : nullptr);
            const Value* cnt = seq == &a ? &b : &a;
            if (seq && is_intlike(*cnt)) {
                const int64_t k = std::max<int64_t>(0, as_i(*cnt));
                if (seq->is_string()) {
                    std::string s;
                    for (int64_t i = 0; i < k; ++i) s += seq->s();
                    return Value::string(std::move(s));
                }
                Value out = Value::array();
                for (int64_t i = 0; i < k; ++i)
                    for (const Value& v : seq->list()) out.push(v);
                return out;
            }
        } else if (op == "/") {
            if (is_num(a) && is_num(b)) {
                if (as_d(b) == 0) fail("division by zero");
                return Value::number(as_d(a) / as_d(b));
            }
        } else if (op == "//") {
            if (is_num(a) && is_num(b)) {
                if (as_d(b) == 0) fail("division by zero");
                if (is_intlike(a) && is_intlike(b)) {
                    const int64_t x = as_i(a), y = as_i(b);
                    int64_t q = x / y;
                    if ((x % y != 0) && ((x < 0) != (y < 0))) --q;
                    return Value::integer(q);
                }
                return Value::number(std::floor(as_d(a) / as_d(b)));
            }
        } else if (op == "%") {
            if (is_num(a) && is_num(b)) {
                if (as_d(b) == 0) fail("modulo by zero");
                if (is_intlike(a) && is_intlike(b)) {
                    const int64_t x = as_i(a), y = as_i(b);
                    int64_t r = x % y;
                    if (r != 0 && ((r < 0) != (y < 0))) r += y;
                    return Value::integer(r);
                }
                double r = std::fmod(as_d(a), as_d(b));
                if (r != 0 && ((r < 0) != (as_d(b) < 0))) r += as_d(b);
                return Value::number(r);
            }
        } else if (op == "**") {
            if (is_num(a) && is_num(b)) {
                if (is_intlike(a) && is_intlike(b) && as_i(b) >= 0) {
                    int64_t r = 1;
                    bool ok = true;
                    for (int64_t i = 0; i < as_i(b) && ok; ++i) ok = !__builtin_mul_overflow(r, as_i(a), &r);
                    if (ok) return Value::integer(r);
                }
                return Value::number(std::pow(as_d(a), as_d(b)));
            }
        }
        fail("unsupported operand types for '" + op + "'");
    }

    int cmp3(const Value& a, const Value& b) const {  // -1, 0, 1
        if (is_num(a) && is_num(b)) {
            if (is_intlike(a) && is_intlike(b)) return as_i(a) < as_i(b) ? -1 : (as_i(a) > as_i(b) ? 1 : 0);
            return as_d(a) < as_d(b) ? -1 : (as_d(a) > as_d(b) ? 1 : 0);
        }
        if (a.is_string() && b.is_string()) {
            const int c = a.s().compare(b.s());
            return c < 0 ? -1 : (c > 0 ? 1 : 0);
        }
        if (a.is_list() && b.is_list()) {
            const List &x = a.list(), &y = b.list();
            for (size_t i = 0; i < x.size() && i < y.size(); ++i) {
                if (x[i] == y[i]) continue;
                return cmp3(x[i], y[i]);
            }
            return x.size() < y.size() ? -1 : (x.size() > y.size() ? 1 : 0);
        }
        if (a.is_undefined()) undefined_error(a);
        if (b.is_undefined()) undefined_error(b);
        fail("comparison not supported between these types");
    }

    bool contains(const Value& hay, const Value& needle) const {
        switch (hay.kind()) {
            case Value::Kind::String:
                if (!needle.is_string()) fail("'in <string>' requires a string as the left operand");
                return hay.s().find(needle.s()) != std::string::npos;
            case Value::Kind::List:
                for (const Value& v : hay.list())
                    if (v == needle) return true;
                return false;
            case Value::Kind::Dict: return needle.is_string() && hay.find(needle.s()) != nullptr;
            case Value::Kind::Undefined: return false;
            default: fail("argument is not iterable");
        }
    }

    // ---- eval ----

    Value eval(const Node& n, const FramePtr& f) {
        switch (n.t) {
            case N::Const: return n.v;
            case N::Name: {
                const Value* v = lookup(f.get(), n.s);
                return v ? *v : Value::undefined(n.s);
            }
            case N::List: {
                Value l = Value::array();
                for (const NodeP& e : n.a) l.push(eval(*e, f));
                return l;
            }
            case N::Tuple: {
                Value l = Value::array();
                for (const NodeP& e : n.a) l.push(eval(*e, f));
                return l;
            }
            case N::Dict: {
                Value d = Value::object();
                for (size_t i = 0; i + 1 < n.a.size(); i += 2) {
                    const Value k = eval(*n.a[i], f);
                    if (!k.is_string()) fail("only string dictionary keys are supported");
                    d.set(k.s(), eval(*n.a[i + 1], f));
                }
                return d;
            }
            case N::Attr: return attr(eval(*n.a[0], f), n.s);
            case N::Sub: {
                const Value obj = eval(*n.a[0], f);
                const Node& ix = *n.a[1];
                if (ix.t == N::Slice) {
                    auto part = [&](size_t k) { return ix.a[k] ? eval(*ix.a[k], f) : Value::null(); };
                    return do_slice(obj, part(0), part(1), part(2));
                }
                return getitem(obj, eval(ix, f));
            }
            case N::Neg:
            case N::Pos: {
                const Value v = eval(*n.a[0], f);
                if (v.is_undefined()) undefined_error(v);
                if (!is_num(v)) fail("bad operand type for unary operator");
                if (n.t == N::Pos) return is_intlike(v) ? Value::integer(as_i(v)) : v;
                if (is_intlike(v)) return Value::integer(-as_i(v));
                return Value::number(-v.d());
            }
            case N::Not: return Value::boolean(!eval(*n.a[0], f).truthy());
            case N::And: {
                Value l = eval(*n.a[0], f);
                return l.truthy() ? eval(*n.a[1], f) : l;
            }
            case N::Or: {
                Value l = eval(*n.a[0], f);
                return l.truthy() ? l : eval(*n.a[1], f);
            }
            case N::Cond:
                if (eval(*n.a[1], f).truthy()) return eval(*n.a[0], f);
                return n.a[2] ? eval(*n.a[2], f) : Value::undefined();
            case N::Bin: return arith(n.s, eval(*n.a[0], f), eval(*n.a[1], f));
            case N::Cmp: {
                Value left = eval(*n.a[0], f);
                for (size_t i = 0; i < n.kn.size(); ++i) {
                    const Value right = eval(*n.a[i + 1], f);
                    const std::string& op = n.kn[i];
                    bool r;
                    if (op == "==") r = left == right;
                    else if (op == "!=") r = !(left == right);
                    else if (op == "in") r = contains(right, left);
                    else if (op == "notin") r = !contains(right, left);
                    else {
                        const int c = cmp3(left, right);
                        r = op == "<" ? c < 0 : op == ">" ? c > 0 : op == "<=" ? c <= 0 : c >= 0;
                    }
                    if (!r) return Value::boolean(false);
                    left = right;
                }
                return Value::boolean(true);
            }
            case N::Filter: {
                Value in = eval(*n.a[0], f);
                Args args;
                Kwargs kw;
                collect_args(n, 1, f, args, kw);
                return apply_filter(n.s, in, args, kw);
            }
            case N::Test: {
                Value in = eval(*n.a[0], f);
                Args args;
                Kwargs kw;
                collect_args(n, 1, f, args, kw);
                const bool r = apply_test(n.s, in, args);
                return Value::boolean(n.flag ? !r : r);
            }
            case N::Call: return eval_call(n, f);
            default: fail("not an expression");
        }
    }

    void collect_args(const Node& n, size_t first, const FramePtr& f, Args& args, Kwargs& kw) {
        for (size_t i = first; i < n.a.size(); ++i) args.push_back(eval(*n.a[i], f));
        for (size_t i = 0; i < n.kn.size(); ++i) kw.emplace_back(n.kn[i], eval(*n.kv[i], f));
    }

    Value eval_call(const Node& n, const FramePtr& f) {
        const Node& callee = *n.a[0];
        Args args;
        Kwargs kw;
        if (callee.t == N::Attr) {
            Value obj = eval(*callee.a[0], f);
            collect_args(n, 1, f, args, kw);
            return call_method(obj, callee.s, args, kw);
        }
        Value fn = eval(callee, f);
        collect_args(n, 1, f, args, kw);
        return call_value(fn, args, kw);
    }

    Value call_value(const Value& fn, Args& args, Kwargs& kw) {
        if (fn.kind() == Value::Kind::Macro) return call_macro(*fn.macro_obj(), args, kw);
        if (fn.kind() == Value::Kind::Builtin) return call_builtin(fn.s(), args, kw);
        if (fn.is_undefined()) undefined_error(fn);
        fail("object is not callable");
    }

    Value call_macro(const MacroObj& m, Args& args, Kwargs& kw) {
        const Node& mn = *static_cast<const Node*>(m.node);
        if (args.size() > mn.kn.size()) fail("macro '" + mn.s + "' takes at most " + std::to_string(mn.kn.size()) + " arguments");
        if (++call_depth_ > 100) fail("maximum recursion depth exceeded");
        auto fr = std::make_shared<Frame>();
        fr->parent = std::static_pointer_cast<Frame>(m.closure.lock());
        if (!fr->parent) fail("macro '" + mn.s + "' was called after the scope that defined it ended");
        std::vector<bool> given(mn.kn.size(), false);
        for (size_t i = 0; i < args.size(); ++i) {
            fr->vars[mn.kn[i]] = args[i];
            given[i] = true;
        }
        for (auto& kv : kw) {
            size_t k = 0;
            while (k < mn.kn.size() && mn.kn[k] != kv.first) ++k;
            if (k == mn.kn.size()) fail("macro '" + mn.s + "' got an unexpected keyword argument '" + kv.first + "'");
            fr->vars[kv.first] = kv.second;
            given[k] = true;
        }
        for (size_t i = 0; i < mn.kn.size(); ++i) {
            if (given[i]) continue;
            fr->vars[mn.kn[i]] = mn.kv[i] ? eval(*mn.kv[i], fr) : Value::undefined(mn.kn[i]);
        }
        std::string out;
        exec_body(mn.body, fr, out);
        --call_depth_;
        return Value::string(std::move(out));
    }

    Value call_builtin(const std::string& name, Args& args, Kwargs& kw) {
        if (name == "raise_exception") throw Raised(args.empty() ? std::string("raise_exception") : args[0].str());
        if (name == "namespace") {
            Value ns = Value::ns();
            if (!args.empty() && args[0].is_dict())
                for (const auto& kv : args[0].dict()) ns.set(kv.first, kv.second);
            for (auto& kv : kw) ns.set(kv.first, kv.second);
            return ns;
        }
        if (name == "dict") {
            Value d = Value::object();
            if (!args.empty() && args[0].is_dict())
                for (const auto& kv : args[0].dict()) d.set(kv.first, kv.second);
            for (auto& kv : kw) d.set(kv.first, kv.second);
            return d;
        }
        if (name == "range") {
            int64_t a = 0, b = 0, st = 1;
            if (args.size() == 1) b = as_i(args[0]);
            else if (args.size() >= 2) a = as_i(args[0]), b = as_i(args[1]);
            if (args.size() >= 3) st = as_i(args[2]);
            if (st == 0) fail("range() step must not be zero");
            Value l = Value::array();
            for (int64_t i = a; st > 0 ? i < b : i > b; i += st) {
                if (l.list().size() > 1000000) fail("range too large");
                l.push(Value::integer(i));
            }
            return l;
        }
        if (name == "strftime_now") {
            const std::time_t t = std::time(nullptr);
            std::tm tmv;
            localtime_r(&t, &tmv);
            char buf[256];
            const std::string fmt = args.empty() ? "%Y-%m-%d" : args[0].str();
            const size_t n = std::strftime(buf, sizeof buf, fmt.c_str(), &tmv);
            return Value::string(std::string(buf, n));
        }
        fail("unknown function '" + name + "'");
    }

    // ---- tests ----

    bool apply_test(const std::string& name, const Value& v, const Args& args) {
        if (name == "defined") return !v.is_undefined();
        if (name == "undefined") return v.is_undefined();
        if (name == "none") return v.is_null();
        if (name == "string") return v.is_string();
        if (name == "number") return is_num(v);
        if (name == "integer") return v.is_int();
        if (name == "float") return v.is_float();
        if (name == "boolean") return v.is_bool();
        if (name == "true") return v.is_bool() && v.b();
        if (name == "false") return v.is_bool() && !v.b();
        if (name == "mapping") return v.is_dict();
        if (name == "iterable") return v.is_string() || v.is_list() || v.is_dict() || v.is_undefined();
        if (name == "sequence") return v.is_string() || v.is_list() || v.is_dict() || v.is_undefined();
        if (name == "callable") return v.kind() == Value::Kind::Macro || v.kind() == Value::Kind::Builtin;
        if (name == "lower") return v.is_string() && v.s() == lower(v.s()) && v.s() != upper(v.s());
        if (name == "upper") return v.is_string() && v.s() == upper(v.s()) && v.s() != lower(v.s());
        auto arg = [&]() -> const Value& {
            if (args.empty()) fail("test '" + name + "' needs an argument");
            return args[0];
        };
        if (name == "even") return is_intlike(v) && as_i(v) % 2 == 0;
        if (name == "odd") return is_intlike(v) && as_i(v) % 2 != 0;
        if (name == "divisibleby") return is_intlike(v) && as_i(arg()) != 0 && as_i(v) % as_i(arg()) == 0;
        if (name == "eq" || name == "equalto" || name == "==") return v == arg();
        if (name == "ne" || name == "!=") return !(v == arg());
        if (name == "lt" || name == "lessthan" || name == "<") return cmp3(v, arg()) < 0;
        if (name == "le" || name == "<=") return cmp3(v, arg()) <= 0;
        if (name == "gt" || name == "greaterthan" || name == ">") return cmp3(v, arg()) > 0;
        if (name == "ge" || name == ">=") return cmp3(v, arg()) >= 0;
        if (name == "in") return contains(arg(), v);
        if (name == "sameas") return v.kind() == arg().kind() && v == arg();
        fail("unknown test '" + name + "'");
    }

    // ---- string helpers ----

    static std::string lower(std::string s) {
        for (char& c : s)
            if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
        return s;
    }
    static std::string upper(std::string s) {
        for (char& c : s)
            if (c >= 'a' && c <= 'z') c = char(c - 'a' + 'A');
        return s;
    }
    static std::string title(const std::string& s) {
        std::string out = s;
        bool start = true;
        for (char& c : out) {
            const bool alpha = std::isalpha(static_cast<unsigned char>(c)) != 0;
            if (alpha) c = char(start ? std::toupper(static_cast<unsigned char>(c)) : std::tolower(static_cast<unsigned char>(c)));
            start = !alpha;
        }
        return out;
    }

    // str.split semantics.
    static Value split(const std::string& s, const Value& sep, int64_t maxsplit) {
        Value out = Value::array();
        if (sep.is_undefined() || sep.is_null()) {
            size_t i = 0;
            int64_t cnt = 0;
            while (i < s.size()) {
                uint32_t cp;
                size_t l = u8dec(s, i, &cp);
                while (i < s.size() && is_py_space(cp)) {
                    i += l;
                    if (i < s.size()) l = u8dec(s, i, &cp);
                }
                if (i >= s.size()) break;
                if (maxsplit >= 0 && cnt >= maxsplit) {
                    out.push(Value::string(py_strip(s.substr(i), nullptr, false, true)));
                    return out;
                }
                const size_t st = i;
                while (i < s.size()) {
                    l = u8dec(s, i, &cp);
                    if (is_py_space(cp)) break;
                    i += l;
                }
                out.push(Value::string(s.substr(st, i - st)));
                ++cnt;
            }
            return out;
        }
        if (!sep.is_string()) fail("split separator must be a string");
        if (sep.s().empty()) fail("empty separator");
        size_t i = 0;
        int64_t cnt = 0;
        for (;;) {
            const size_t k = (maxsplit >= 0 && cnt >= maxsplit) ? std::string::npos : s.find(sep.s(), i);
            if (k == std::string::npos) {
                out.push(Value::string(s.substr(i)));
                return out;
            }
            out.push(Value::string(s.substr(i, k - i)));
            i = k + sep.s().size();
            ++cnt;
        }
    }

    static std::string replace(const std::string& s, const std::string& from, const std::string& to, int64_t count) {
        if (count == 0) return s;
        if (from.empty()) {  // Python inserts `to` between every character
            std::string out;
            int64_t done = 0;
            const std::vector<size_t> off = cp_offsets(s);
            for (size_t k = 0; k + 1 < off.size(); ++k) {
                if (count < 0 || done < count) {
                    out += to;
                    ++done;
                }
                out.append(s, off[k], off[k + 1] - off[k]);
            }
            if (count < 0 || done < count) out += to;
            return out;
        }
        std::string out;
        size_t i = 0;
        int64_t done = 0;
        for (;;) {
            const size_t k = (count >= 0 && done >= count) ? std::string::npos : s.find(from, i);
            if (k == std::string::npos) {
                out.append(s, i, std::string::npos);
                return out;
            }
            out.append(s, i, k - i);
            out += to;
            i = k + from.size();
            ++done;
        }
    }

    // ---- methods ----

    const Value& arg_or(const Args& a, const Kwargs& kw, size_t i, const char* name) const {
        if (i < a.size()) return a[i];
        for (const auto& kv : kw)
            if (kv.first == name) return kv.second;
        static const Value und;
        return und;
    }

    Value call_method(const Value& obj, const std::string& m, Args& a, Kwargs& kw) {
        if (obj.is_undefined()) undefined_error(obj);
        if (obj.is_string()) {
            const std::string& s = obj.s();
            if (m == "startswith" || m == "endswith") {
                const Value& p = arg_or(a, kw, 0, "prefix");
                std::vector<std::string> opts;
                if (p.is_string()) opts.push_back(p.s());
                else if (p.is_list())
                    for (const Value& v : p.list()) opts.push_back(v.str());
                else fail(m + "() argument must be a string or a list of strings");
                for (const std::string& o : opts) {
                    if (s.size() < o.size()) continue;
                    if (m == "startswith" ? s.compare(0, o.size(), o) == 0 : s.compare(s.size() - o.size(), o.size(), o) == 0)
                        return Value::boolean(true);
                }
                return Value::boolean(false);
            }
            if (m == "strip" || m == "lstrip" || m == "rstrip") {
                const Value& c = arg_or(a, kw, 0, "chars");
                const std::string cs = c.is_string() ? c.s() : std::string();
                return Value::string(py_strip(s, c.is_string() ? &cs : nullptr, m != "rstrip", m != "lstrip"));
            }
            if (m == "split") {
                const Value& sep = arg_or(a, kw, 0, "sep");
                const Value& mx = arg_or(a, kw, 1, "maxsplit");
                return split(s, sep, mx.is_undefined() ? -1 : as_i(mx));
            }
            if (m == "splitlines") {
                Value out = Value::array();
                size_t i = 0;
                while (i < s.size()) {
                    size_t k = s.find('\n', i);
                    if (k == std::string::npos) {
                        out.push(Value::string(s.substr(i)));
                        break;
                    }
                    out.push(Value::string(s.substr(i, k - i)));
                    i = k + 1;
                }
                return out;
            }
            if (m == "replace") {
                const Value& cnt = arg_or(a, kw, 2, "count");
                return Value::string(replace(s, arg_or(a, kw, 0, "old").str(), arg_or(a, kw, 1, "new").str(), cnt.is_undefined() ? -1 : as_i(cnt)));
            }
            if (m == "lower") return Value::string(lower(s));
            if (m == "upper") return Value::string(upper(s));
            if (m == "title") return Value::string(title(s));
            if (m == "capitalize") return Value::string(s.empty() ? s : upper(s.substr(0, 1)) + lower(s.substr(1)));
            if (m == "join") {
                std::string out;
                bool first = true;
                for (const Value& v : iter_items(arg_or(a, kw, 0, "iterable"))) {
                    if (!first) out += s;
                    first = false;
                    out += v.str();
                }
                return Value::string(std::move(out));
            }
            if (m == "find" || m == "rfind" || m == "index") {
                const size_t k = m == "rfind" ? s.rfind(arg_or(a, kw, 0, "sub").str()) : s.find(arg_or(a, kw, 0, "sub").str());
                if (k == std::string::npos) {
                    if (m == "index") fail("substring not found");
                    return Value::integer(-1);
                }
                return Value::integer(int64_t(u8len(std::string_view(s).substr(0, k))));
            }
            if (m == "count") {
                const std::string sub = arg_or(a, kw, 0, "sub").str();
                if (sub.empty()) return Value::integer(int64_t(u8len(s)) + 1);
                int64_t c = 0;
                for (size_t i = s.find(sub); i != std::string::npos; i = s.find(sub, i + sub.size())) ++c;
                return Value::integer(c);
            }
            if (m == "isdigit" || m == "isnumeric" || m == "isdecimal")
                return Value::boolean(!s.empty() && std::all_of(s.begin(), s.end(), [](char c) { return c >= '0' && c <= '9'; }));
            if (m == "isalpha")
                return Value::boolean(!s.empty() && std::all_of(s.begin(), s.end(), [](char c) { return std::isalpha(static_cast<unsigned char>(c)) != 0; }));
            if (m == "isspace") {
                if (s.empty()) return Value::boolean(false);
                for (size_t i = 0; i < s.size();) {
                    uint32_t cp;
                    i += u8dec(s, i, &cp);
                    if (!is_py_space(cp)) return Value::boolean(false);
                }
                return Value::boolean(true);
            }
            fail("str has no method '" + m + "'");
        }
        if (obj.is_dict()) {
            if (m == "items") {
                Value out = Value::array();
                for (const auto& kv : obj.dict()) {
                    Value pair = Value::array();
                    pair.push(Value::string(kv.first));
                    pair.push(kv.second);
                    out.push(std::move(pair));
                }
                return out;
            }
            if (m == "keys") {
                Value out = Value::array();
                for (const auto& kv : obj.dict()) out.push(Value::string(kv.first));
                return out;
            }
            if (m == "values") {
                Value out = Value::array();
                for (const auto& kv : obj.dict()) out.push(kv.second);
                return out;
            }
            if (m == "get") {
                const Value& k = arg_or(a, kw, 0, "key");
                const Value* v = k.is_string() ? obj.find(k.s()) : nullptr;
                if (v) return *v;
                const Value& d = arg_or(a, kw, 1, "default");
                return d.is_undefined() ? Value::null() : d;
            }
            if (const Value* v = obj.find(m)) return call_value(*v, a, kw);
            fail("dict has no method '" + m + "'");
        }
        if (obj.kind() == Value::Kind::Namespace) {
            if (const Value* v = obj.find(m)) return call_value(*v, a, kw);
            fail("namespace has no method '" + m + "'");
        }
        if (obj.is_list()) {
            if (m == "index" || m == "count") {
                const Value& x = arg_or(a, kw, 0, "value");
                int64_t c = 0, first = -1;
                for (size_t i = 0; i < obj.list().size(); ++i) {
                    if (obj.list()[i] == x) {
                        if (first < 0) first = int64_t(i);
                        ++c;
                    }
                }
                if (m == "count") return Value::integer(c);
                if (first < 0) fail("value is not in list");
                return Value::integer(first);
            }
            fail("list has no method '" + m + "'");
        }
        fail("object has no method '" + m + "'");
    }

    // ---- filters ----

    Value apply_filter(const std::string& name, const Value& in, Args& a, Kwargs& kw) {
        if (name == "safe" || name == "forceescape" || name == "markupsafe") return in;
        if (name == "string") return Value::string(in.str());
        if (name == "trim" || name == "strip") {
            const Value& c = arg_or(a, kw, 0, "chars");
            const std::string cs = c.is_string() ? c.s() : std::string();
            return Value::string(py_strip(in.str(), c.is_string() ? &cs : nullptr, true, true));
        }
        if (name == "length" || name == "count") {
            switch (in.kind()) {
                case Value::Kind::String: return Value::integer(int64_t(u8len(in.s())));
                case Value::Kind::List:
                case Value::Kind::Dict: return Value::integer(int64_t(in.size()));
                case Value::Kind::Undefined: return Value::integer(0);
                default: fail("object has no length");
            }
        }
        if (name == "tojson") {
            DumpOptions o;
            const Value& ind = arg_or(a, kw, 0, "indent");
            if (is_intlike(ind) && !ind.is_bool()) o.indent = int(ind.i());
            for (const auto& kv : kw) {
                if (kv.first == "ensure_ascii") o.ensure_ascii = kv.second.truthy();
                else if (kv.first == "sort_keys") o.sort_keys = kv.second.truthy();
                else if (kv.first == "separators" && kv.second.is_list() && kv.second.list().size() == 2) {
                    o.item_sep = kv.second.list()[0].str();
                    o.key_sep = kv.second.list()[1].str();
                }
            }
            return Value::string(json_dump(in, o));
        }
        if (name == "default" || name == "d") {
            const Value& d = arg_or(a, kw, 0, "default_value");
            const Value& b = arg_or(a, kw, 1, "boolean");
            if (in.is_undefined() || (b.truthy() && !in.truthy())) return d.is_undefined() ? Value::string("") : d;
            return in;
        }
        if (name == "lower") return Value::string(lower(in.str()));
        if (name == "upper") return Value::string(upper(in.str()));
        if (name == "capitalize") {
            const std::string s = in.str();
            return Value::string(s.empty() ? s : upper(s.substr(0, 1)) + lower(s.substr(1)));
        }
        if (name == "title") return Value::string(title(in.str()));
        if (name == "join") {
            const std::string sep = arg_or(a, kw, 0, "d").is_undefined() ? "" : arg_or(a, kw, 0, "d").str();
            const Value& at = arg_or(a, kw, 1, "attribute");
            std::string out;
            bool first = true;
            for (const Value& v : iter_items(in)) {
                if (!first) out += sep;
                first = false;
                out += (at.is_string() ? attr(v, at.s()) : v).str();
            }
            return Value::string(std::move(out));
        }
        if (name == "list") {
            Value out = Value::array();
            for (Value& v : iter_items(in)) out.push(std::move(v));
            return out;
        }
        if (name == "first" || name == "last") {
            const std::vector<Value> it = iter_items(in);
            if (it.empty()) return Value::undefined();
            return name == "first" ? it.front() : it.back();
        }
        if (name == "reverse") {
            if (in.is_string()) {
                const std::vector<size_t> off = cp_offsets(in.s());
                std::string out;
                for (size_t k = off.size() - 1; k-- > 0;) out.append(in.s(), off[k], off[k + 1] - off[k]);
                return Value::string(std::move(out));
            }
            std::vector<Value> it = iter_items(in);
            std::reverse(it.begin(), it.end());
            Value out = Value::array();
            for (Value& v : it) out.push(std::move(v));
            return out;
        }
        if (name == "int") {
            const Value& d = arg_or(a, kw, 0, "default");
            const Value fallback = d.is_undefined() ? Value::integer(0) : d;
            if (is_intlike(in)) return Value::integer(as_i(in));
            if (in.is_float()) return Value::integer(int64_t(in.d()));
            if (in.is_string()) {
                const std::string t = py_strip(in.s(), nullptr, true, true);
                int64_t v = 0;
                const auto r = std::from_chars(t.data(), t.data() + t.size(), v);
                if (r.ec == std::errc() && r.ptr == t.data() + t.size()) return Value::integer(v);
                char* end = nullptr;
                const double dv = std::strtod(t.c_str(), &end);
                if (!t.empty() && end && *end == 0) return Value::integer(int64_t(dv));
            }
            return fallback;
        }
        if (name == "float") {
            const Value& d = arg_or(a, kw, 0, "default");
            const Value fallback = d.is_undefined() ? Value::number(0.0) : d;
            if (is_num(in)) return Value::number(as_d(in));
            if (in.is_string()) {
                const std::string t = py_strip(in.s(), nullptr, true, true);
                char* end = nullptr;
                const double dv = std::strtod(t.c_str(), &end);
                if (!t.empty() && end && *end == 0) return Value::number(dv);
            }
            return fallback;
        }
        if (name == "abs") {
            if (is_intlike(in)) return Value::integer(std::llabs(as_i(in)));
            if (in.is_float()) return Value::number(std::fabs(in.d()));
            fail("abs of a non-number");
        }
        if (name == "round") {
            const Value& pr = arg_or(a, kw, 0, "precision");
            const double scale = std::pow(10.0, pr.is_undefined() ? 0 : double(as_i(pr)));
            return Value::number(std::round(as_d(in) * scale) / scale);
        }
        if (name == "replace") {
            const Value& cnt = arg_or(a, kw, 2, "count");
            return Value::string(replace(in.str(), arg_or(a, kw, 0, "old").str(), arg_or(a, kw, 1, "new").str(), cnt.is_undefined() ? -1 : as_i(cnt)));
        }
        if (name == "items") {
            Args none;
            Kwargs nokw;
            return in.is_dict() ? call_method(in, "items", none, nokw) : Value::array();
        }
        if (name == "e" || name == "escape") {
            std::string out;
            for (char c : in.str()) {
                switch (c) {
                    case '&': out += "&amp;"; break;
                    case '<': out += "&lt;"; break;
                    case '>': out += "&gt;"; break;
                    case '\'': out += "&#39;"; break;
                    case '"': out += "&#34;"; break;
                    default: out.push_back(c);
                }
            }
            return Value::string(std::move(out));
        }
        if (name == "indent") {
            const Value& w = arg_or(a, kw, 0, "width");
            const Value& fi = arg_or(a, kw, 1, "first");
            const std::string pad(size_t(w.is_undefined() ? 4 : as_i(w)), ' ');
            const std::string s = in.str();
            std::string out;
            size_t i = 0;
            bool first_line = true;
            while (i <= s.size()) {
                const size_t k = s.find('\n', i);
                const std::string line = s.substr(i, k == std::string::npos ? std::string::npos : k - i);
                if (!first_line) out += '\n';
                if (!line.empty() && (!first_line || fi.truthy())) out += pad;
                out += line;
                first_line = false;
                if (k == std::string::npos) break;
                i = k + 1;
            }
            return Value::string(std::move(out));
        }
        if (name == "sum") {
            Value acc = Value::integer(0);
            for (const Value& v : iter_items(in)) acc = arith("+", acc, v);
            return acc;
        }
        if (name == "min" || name == "max") {
            const std::vector<Value> it = iter_items(in);
            if (it.empty()) return Value::undefined();
            Value best = it[0];
            for (const Value& v : it) {
                const int c = cmp3(v, best);
                if (name == "min" ? c < 0 : c > 0) best = v;
            }
            return best;
        }
        if (name == "sort") {
            std::vector<Value> it = iter_items(in);
            const Value& rev = arg_or(a, kw, 0, "reverse");
            const Value& at = arg_or(a, kw, 2, "attribute");
            std::stable_sort(it.begin(), it.end(), [&](const Value& x, const Value& y) {
                return at.is_string() ? cmp3(attr(x, at.s()), attr(y, at.s())) < 0 : cmp3(x, y) < 0;
            });
            if (rev.truthy()) std::reverse(it.begin(), it.end());
            Value out = Value::array();
            for (Value& v : it) out.push(std::move(v));
            return out;
        }
        if (name == "unique") {
            Value out = Value::array();
            for (const Value& v : iter_items(in)) {
                bool seen = false;
                for (const Value& o : out.list())
                    if (o == v) seen = true;
                if (!seen) out.push(v);
            }
            return out;
        }
        if (name == "map") {
            const Value& at = arg_or(a, kw, 99, "attribute");
            Value out = Value::array();
            for (const Value& v : iter_items(in)) {
                if (at.is_string()) {
                    Value r = attr(v, at.s());
                    const Value& d = arg_or(a, kw, 99, "default");
                    out.push(r.is_undefined() && !d.is_undefined() ? d : r);
                } else if (!a.empty() && a[0].is_string()) {
                    Args rest(a.begin() + 1, a.end());
                    Kwargs nokw;
                    out.push(apply_filter(a[0].s(), v, rest, nokw));
                } else fail("map needs an attribute or a filter name");
            }
            return out;
        }
        if (name == "select" || name == "reject" || name == "selectattr" || name == "rejectattr") {
            const bool by_attr = name == "selectattr" || name == "rejectattr";
            const bool want = name == "select" || name == "selectattr";
            size_t k = 0;
            std::string at;
            if (by_attr) {
                if (a.empty() || !a[0].is_string()) fail(name + " needs an attribute name");
                at = a[0].s();
                k = 1;
            }
            const std::string test = k < a.size() ? a[k].str() : std::string();
            Args targs;
            for (size_t i = k + 1; i < a.size(); ++i) targs.push_back(a[i]);
            Value out = Value::array();
            for (const Value& v : iter_items(in)) {
                const Value x = by_attr ? attr(v, at) : v;
                const bool r = test.empty() ? x.truthy() : apply_test(test, x, targs);
                if (r == want) out.push(v);
            }
            return out;
        }
        if (name == "dictsort") {
            // jinja2's do_dictsort: the mapping's (key, value) pairs, sorted by key (or by value), with
            // case_sensitive=False folding case through lower() exactly as Python does. Inkling's template
            // sorts a tool call's arguments this way, and that is what makes the JSON it emits stable --
            // so the sort has to be Python's, ties included: reverse=True inverts the comparison and
            // leaves equal keys in insertion order, which is what stable_sort with a flipped predicate
            // gives (std::reverse after the sort would not).
            if (!in.is_object()) fail("dictsort expects a mapping");
            const Value& by = arg_or(a, kw, 0, "by");
            const bool by_value = by.is_string() && by.s() == "value";
            if (by.is_string() && !by_value && by.s() != "key") fail("dictsort: 'by' must be 'key' or 'value'");
            bool cs = false, rev = false;
            for (const auto& kv : kw) {
                if (kv.first == "case_sensitive") cs = kv.second.truthy();
                else if (kv.first == "reverse") rev = kv.second.truthy();
            }
            std::vector<std::pair<std::string, Value>> items = in.dict();
            auto key_of = [&](const std::pair<std::string, Value>& it) {
                if (!by_value) return cs ? it.first : lower(it.first);
                if (it.second.is_string()) return cs ? it.second.s() : lower(it.second.s());
                return it.second.str();
            };
            std::stable_sort(items.begin(), items.end(), [&](const auto& x, const auto& y) {
                return rev ? key_of(y) < key_of(x) : key_of(x) < key_of(y);
            });
            Value out = Value::array();
            for (auto& it : items) {
                Value pair = Value::array();
                pair.push(Value::string(it.first));
                pair.push(it.second);
                out.push(std::move(pair));
            }
            return out;
        }
        fail("unknown filter '" + name + "'");
    }
};

}  // namespace

// ---- Template -------------------------------------------------------------------------------------

struct Template::Impl {
    std::vector<NodeP> root;
};

Template::Template() : impl_(new Impl) {}
Template::~Template() = default;

std::unique_ptr<Template> Template::compile(std::string_view source, const Options& opt, std::string* err) {
    try {
        std::vector<Tok> toks = lex_template(std::string(source), opt);
        StmtParser p(toks);
        std::unique_ptr<Template> t(new Template);
        t->impl_->root = p.parse_all();
        return t;
    } catch (const Err& e) {
        if (err) *err = "template line " + std::to_string(e.line) + ": " + e.what();
        return nullptr;
    } catch (const std::exception& e) {
        if (err) *err = std::string("template: ") + e.what();
        return nullptr;
    }
}

bool Template::render(const Value& context, std::string* out, std::string* err) const {
    try {
        Interp in(impl_->root);
        *out = in.run(context);
        return true;
    } catch (const Raised& e) {
        if (err) *err = e.what();
    } catch (const Err& e) {
        if (err) *err = "template line " + std::to_string(e.line) + ": " + e.what();
    } catch (const std::exception& e) {
        if (err) *err = std::string("template: ") + e.what();
    }
    return false;
}

}  // namespace limbo::jinja
