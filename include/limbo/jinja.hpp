// Copyright (c) 2026 ITERFORM. All rights reserved.
#pragma once
// limbo/jinja.hpp — a Jinja subset interpreter and the JSON value it runs on.
//
// Why an interpreter: the chat format of a model is the string the GGUF carries in
// tokenizer.chat_template, and that string is the only source of truth for the prompt bytes. Running
// it as written (instead of a hand-ported renderer per model family) keeps the engine correct when a
// model's template changes. The reference is Python jinja2 3.1 with HuggingFace's environment
// (trim_blocks, lstrip_blocks, loopcontrols, `tojson` without key sorting and without ASCII escaping);
// bench/fixtures/chat holds goldens rendered by exactly that.
//
// Supported: {{ }}, {% if/elif/else %}, {% for %} (else, if-filter, loop.*), {% set %} (names, tuple
// targets, namespace attributes, block form), {% macro %}, {% break %}/{% continue %}, comments,
// whitespace control ({%- -%}, {%+ %}, trim_blocks, lstrip_blocks), the usual operators, `is` tests,
// conditional expressions, slices, string/list/dict methods and the common filters. Python semantics
// are followed where templates can observe them: str() of bool/None/float, `is sequence`, Undefined
// that prints empty but raises on attribute access, floor division, str.strip over Unicode space.
// Anything else (include, import, extends, call blocks, raw) is rejected at compile time with a
// message that names the construct.
//
// The Value type is also the JSON document type the HTTP layer uses: objects keep insertion order and
// numbers keep the int/float distinction of the text they came from.

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace limbo::jinja {

struct MacroObj;
class Value;
using List = std::vector<Value>;
using Dict = std::vector<std::pair<std::string, Value>>;  // insertion-ordered

class Value {
public:
    enum class Kind : uint8_t { Undefined, Null, Bool, Int, Float, String, List, Dict, Namespace, Macro, Builtin };

    Value() = default;  // Undefined
    static Value undefined(std::string name = {});  // `name` is only used in error messages
    static Value null();
    static Value boolean(bool b);
    static Value integer(int64_t i);
    static Value number(double d);
    static Value string(std::string s);
    static Value array();
    static Value object();
    static Value ns();  // a namespace(): an object whose attributes a template may assign
    static Value builtin(std::string name);
    static Value macro(std::shared_ptr<MacroObj> m);

    Kind kind() const { return k_; }
    bool is_undefined() const { return k_ == Kind::Undefined; }
    bool is_null() const { return k_ == Kind::Null; }
    bool is_bool() const { return k_ == Kind::Bool; }
    bool is_int() const { return k_ == Kind::Int; }
    bool is_float() const { return k_ == Kind::Float; }
    bool is_number() const { return k_ == Kind::Int || k_ == Kind::Float; }
    bool is_string() const { return k_ == Kind::String; }
    bool is_list() const { return k_ == Kind::List; }
    bool is_dict() const { return k_ == Kind::Dict; }
    bool is_object() const { return k_ == Kind::Dict || k_ == Kind::Namespace; }

    bool b() const { return b_; }
    int64_t i() const { return i_; }
    double d() const { return d_; }
    double as_double() const { return k_ == Kind::Int ? double(i_) : d_; }
    const std::string& s() const { return s_; }

    // Lists and objects are shared on copy (Python reference semantics).
    const List& list() const;
    List& list();
    const Dict& dict() const;
    Dict& dict();
    size_t size() const;  // list length, object size, string bytes
    void push(Value v);   // list append
    const Value* find(std::string_view key) const;  // object lookup, nullptr when absent
    void set(std::string_view key, Value v);        // object assign (replaces in place, else appends)
    // Convenience readers: the string / number under `key`, or the default.
    std::string get_str(std::string_view key, std::string def = {}) const;
    int64_t get_int(std::string_view key, int64_t def) const;

    Value clone() const;      // a deep copy: lists and objects are shared on plain copy
    bool truthy() const;      // Python bool()
    std::string str() const;  // Python str()
    std::string repr() const; // Python repr()

    const MacroObj* macro_obj() const { return m_.get(); }
    const std::shared_ptr<MacroObj>& macro_ptr() const { return m_; }

private:
    Kind k_ = Kind::Undefined;
    bool b_ = false;
    int64_t i_ = 0;
    double d_ = 0;
    std::string s_;
    std::shared_ptr<List> l_;
    std::shared_ptr<Dict> o_;
    std::shared_ptr<MacroObj> m_;
};

bool operator==(const Value& a, const Value& b);  // Python ==
inline bool operator!=(const Value& a, const Value& b) { return !(a == b); }

// ---- JSON ---------------------------------------------------------------------------------------

// Parses one JSON document (trailing whitespace allowed, anything else after it is an error).
// Integers that fit int64 stay ints; everything else numeric is a double. Depth is limited.
bool json_parse(std::string_view text, Value* out, std::string* err);

struct DumpOptions {
    int indent = -1;           // < 0: one line
    bool ensure_ascii = false;
    bool sort_keys = false;
    std::string item_sep = ", ";  // Python's json.dumps defaults; ignored between items when indent >= 0
    std::string key_sep = ": ";
};
// Python json.dumps over a Value (Undefined, Namespace, Macro and Builtin are not serializable: they
// throw std::runtime_error, which the template layer turns into an error string).
std::string json_dump(const Value& v, const DumpOptions& o = {});
// The wire form: no spaces.
std::string json_compact(const Value& v);
// Python repr(float): shortest round trip, 'e+NN' exponent from 1e16, always a '.0' or an exponent.
std::string py_float_repr(double d);

// ---- templates ----------------------------------------------------------------------------------

struct Options {
    bool trim_blocks = true;   // HuggingFace's environment
    bool lstrip_blocks = true;
    bool keep_trailing_newline = false;
};

class Template {
public:
    // nullptr on failure; *err says what and on which line.
    static std::unique_ptr<Template> compile(std::string_view source, const Options& opt, std::string* err);
    ~Template();
    Template(const Template&) = delete;
    Template& operator=(const Template&) = delete;

    // `context` is an object. A template error (raise_exception included) returns false with *err set;
    // for raise_exception *err is exactly the message the template gave.
    bool render(const Value& context, std::string* out, std::string* err) const;

private:
    Template();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace limbo::jinja
