// Copyright (c) 2026 ITERFORM. All rights reserved.
#pragma once
// limbo/tok.hpp — the tokenizer: the GGUF vocabulary, byte-level BPE, the Qwen pre-split.
//
// What is taken from where (docs/CLEANROOM.md §3 — interoperability facts, not expression):
//  - vocabulary, token types, merges, bos/eos and the pre-tokenizer name: the GGUF's own
//    tokenizer.ggml.* keys;
//  - the pre-split pattern: the model's tokenizer.json (Qwen/Qwen3.6-35B-A3B), whose pre_tokenizer is
//    Sequence[Split(<regex>, Isolated), ByteLevel(add_prefix_space=false, use_regex=false)]. The
//    regex, verbatim from that file:
//      (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}|
//       ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//    tokenizer.ggml.pre == "qwen35" selects it. llama.cpp (the oracle for this gate, not a donor)
//    carries the same pattern with the (?i:...) group written out — src/llama-vocab.cpp, case
//    LLAMA_VOCAB_PRE_TYPE_QWEN35. Byte-exactness against llama-tokenize is measured, never assumed.
//
// Two tokenizer families are implemented, selected by tokenizer.ggml.model:
//  - "gpt2": byte-level BPE with the pre-split above (Qwen);
//  - "gemma4": BPE over the raw UTF-8 text, the way a SentencePiece-style model is cut. The normaliser
//    turns every space into U+2581, the pre-split only separates runs of newlines from the text between
//    them (the pattern `[^\n]+|[\n]+`), merges run per run over whole characters with no byte alphabet, a
//    run that is only newlines is taken whole when the vocabulary has it as a token, and a symbol with no
//    token falls back to its bytes' `<0xXX>` tokens. MEASURED against llama-tokenize, never assumed.
//
// A Vocab holds views into a gguf::File's mapping, so the File must outlive it.

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace limbo::gguf {
class File;
}

namespace limbo::tok {

// llama_token_type values (ggml's enum, as written into tokenizer.ggml.token_type).
enum TokenType : int32_t {
    kUndefined = 0,
    kNormal = 1,
    kUnknown = 2,
    kControl = 3,
    kUserDefined = 4,
    kUnused = 5,
    kByte = 6,
};

struct Vocab {
    std::vector<std::string_view> tokens;              // id -> piece, in the byte-level alphabet
    std::vector<int32_t> types;                        // id -> TokenType
    std::unordered_map<std::string_view, int32_t> id_of;
    std::unordered_map<std::string, int32_t> rank;     // "a\0b" -> merge rank (0 is the first merge)
    std::vector<std::pair<std::string_view, int32_t>> specials;  // control + user-defined, longest first
    std::string_view pre;                              // tokenizer.ggml.pre
    std::string_view model;                            // tokenizer.ggml.model
    int32_t bos = -1, eos = -1, pad = -1, unk = -1;
    bool add_bos = false;
    bool byte_level = true;      // gpt2 byte alphabet; false: raw UTF-8 symbols (gemma4)
    bool escape_spaces = false;  // gemma4: ' ' becomes U+2581 before merging
    // Which pre-tokenizer alternation `split` applies, from tokenizer.ggml.pre. MEASURED: llama.cpp's
    // unicode_regex_split collapses \p{Lu}, \p{Ll}, \p{Lt}, \p{Lm} and \p{Lo} to ONE letter flag
    // (src/unicode.cpp:1500-1504), so GPT-4o's two letter alternatives are the same alternation as
    // Qwen2's, and the whole of GPT-4o's difference is its number rule: \p{N}{1,3} -- one to THREE
    // number codepoints per piece, against Qwen2's one. "1234" is [123][4], not [1][2][3][4].
    enum class PreStyle : uint8_t { Qwen2, Gpt4o };
    PreStyle pre_style = PreStyle::Qwen2;

    static bool load(const gguf::File& f, Vocab& out, std::string* err);
    int32_t id(std::string_view piece) const;
    // Longest special token starting at `at`, if any. MEASURED against llama-tokenize: tokens of type
    // USER_DEFINED (Qwen's <tool_call>, </tool_call>) are taken out of the text even when special
    // parsing is off, while CONTROL tokens (<|im_start|>) are only taken when it is on — which is the
    // same split the model's tokenizer.json makes with the `special` flag on its added_tokens.
    bool match_special(std::string_view text, size_t at, bool allow_control, int32_t* id, size_t* len) const;
};

struct Options {
    bool add_bos = false;        // prepend bos when the model asks for it (tokenizer.ggml.add_bos_token)
    bool parse_special = false;  // take control/user-defined tokens written literally in the text
};

class Tokenizer {
public:
    explicit Tokenizer(const Vocab& v) : v_(&v) {}
    const Vocab& vocab() const { return *v_; }

    // Text -> token ids. `out` is appended to.
    void encode(std::string_view text, std::vector<int32_t>& out, const Options& o = {}) const;
    std::string decode(const std::vector<int32_t>& ids) const;
    std::string decode(const int32_t* ids, size_t n) const;

    // The pre-split: the pieces the pattern cuts the text into, before the byte-level mapping.
    // This is the function to test against the model's own pre_tokenizer.
    std::vector<std::string_view> split(std::string_view text) const;

    // The byte-level BPE of one already-split piece (the byte-level mapping applied inside).
    void bpe(std::string_view piece, std::vector<int32_t>& out) const;

private:
    void bpe_raw(std::string_view piece, std::vector<int32_t>& out) const;
    const Vocab* v_;
};

// The byte-level alphabet (GPT-2's bytes_to_unicode): printable bytes map to themselves, the other
// 68 to codepoints 256..323. byte_to_cp is the map, cp_to_byte its inverse (0xff when absent),
// byte_to_str the UTF-8 string a byte maps to.
uint32_t byte_to_cp(uint8_t b);
uint8_t cp_to_byte(uint32_t cp);
const std::string& byte_to_str(uint8_t b);

}  // namespace limbo::tok
