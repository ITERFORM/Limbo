// Copyright (c) 2026 ITERFORM. All rights reserved.
#pragma once
// limbo/chat.hpp — chat rendering from the model's own template, and parsing of what the model writes back.
//
// Rendering: the GGUF carries the model's chat template in tokenizer.chat_template; Renderer runs it
// with limbo::jinja (the HuggingFace environment, see jinja.hpp) over an OpenAI-shaped conversation.
// Parsing: OutputParser splits generated text into reasoning_content (the think block), content and
// tool calls, in the formats the Qwen3.5/3.6 template teaches the model:
//
//   generation prompt ends in "<think>\n"   ->  the output starts inside the think block:
//       reasoning "\n</think>\n\n" content ["\n\n"] ("<tool_call>\n<function=NAME>\n
//       <parameter=KEY>\nVALUE\n</parameter>\n ... </function>\n</tool_call>")*
//   generation prompt ends in "</think>\n\n" (enable_thinking false)  ->  the output is content directly.
//
// Parameter values are typed by the request's tool schema (string stays text; integer, number, boolean,
// array, object are parsed); without a schema a value that is valid JSON is taken as JSON. The parser
// is incremental: feed() takes the text as it is generated and emits reasoning / content deltas (held
// back only while they could still be the start of a marker or trailing whitespace) and one event per
// complete tool call, so the streaming and non-streaming paths share one implementation.

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "limbo/jinja.hpp"

namespace limbo::gguf {
class File;
}

namespace limbo::chat {

using jinja::Value;

class Renderer {
public:
    static std::unique_ptr<Renderer> from_source(std::string_view tmpl, std::string* err);
    static std::unique_ptr<Renderer> from_gguf(const gguf::File& f, std::string* err);

    const std::string& source() const { return src_; }

    // Renders with `context` exactly as given (an object): the fixtures use this.
    bool render(const Value& context, std::string* out, std::string* err) const;

    // The context a chat template expects: messages (already normalised), tools when non-empty,
    // add_generation_prompt, then `kwargs` (chat_template_kwargs: enable_thinking, preserve_thinking...)
    // and `extra` (bos_token, eos_token), neither of which may replace the first three.
    static Value make_context(const Value& messages, const Value& tools, bool add_generation_prompt,
                              const Value& kwargs, const Value& extra);

    bool apply(const Value& messages, const Value& tools, bool add_generation_prompt, const Value& kwargs,
               const Value& extra, std::string* out, std::string* err) const;

private:
    Renderer() = default;
    std::string src_;
    std::shared_ptr<jinja::Template> tpl_;
};

// OpenAI messages -> what the template reads: tool_calls[].function.arguments arrives as a JSON string
// on the wire and the template needs the object. Anything that does not parse to an object is left
// alone (the template then prints no parameters for it).
Value normalize_messages(const Value& messages);

// True when the rendered prompt leaves the model inside an open think block (ends in "<think>").
bool prompt_opens_think(std::string_view prompt);

struct ToolCall {
    std::string name;
    Value arguments;  // an object, parameters in the order written
};

struct Event {
    enum Kind { Reasoning, Content, Call } kind = Content;
    std::string text;  // Reasoning / Content delta
    ToolCall call;     // Call
};

class OutputParser {
public:
    // in_think: the prompt ended inside <think>. tools: the request's tool list (schemas type the
    // parameters), may be undefined. parse_tools false leaves <tool_call> text in the content.
    OutputParser(bool in_think, const Value& tools, bool parse_tools);
    ~OutputParser();
    void feed(std::string_view text, std::vector<Event>* out);
    void finish(std::vector<Event>* out);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct Message {
    std::string reasoning;
    bool has_reasoning = false;
    std::string content;
    std::vector<ToolCall> calls;
};
Message parse_output(std::string_view text, bool in_think, const Value& tools, bool parse_tools = true);

// One tool block's inner text ("<function=...>...</function>" or a JSON {"name","arguments"}), typed
// by `tools`. False when it is not a well-formed call.
bool parse_tool_block(std::string_view block, const Value& tools, ToolCall* out);

// The longest prefix of `s` that ends on a code point boundary (an incomplete trailing sequence is
// cut off); and `s` with every invalid byte replaced by U+FFFD. Generated tokens are byte-level, so a
// character can arrive in pieces.
size_t utf8_complete_prefix(std::string_view s);
std::string utf8_sanitize(std::string_view s);

}  // namespace limbo::chat
