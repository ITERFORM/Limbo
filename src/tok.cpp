// Copyright (c) 2026 ITERFORM. All rights reserved.
// src/tok.cpp — byte-level BPE over the GGUF vocabulary, with the Qwen pre-split.
//
// The pre-split implements the model's own pattern (include/limbo/tok.hpp has the provenance):
//
//   (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+ | \p{N} |
//    ?[^\s\p{L}\p{M}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
//
// Alternatives are tried in order at each position and the first that matches wins — the semantics of
// a leftmost-first alternation, which is also what llama.cpp's unicode_regex_split does. Each
// alternative is a hand-written matcher, including the backtracking that \s+(?!\S) needs.
#include "limbo/tok.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "limbo/gguf.hpp"
#include "limbo/unicode.hpp"

namespace limbo::tok {

namespace {

// ---- the byte-level alphabet (GPT-2 bytes_to_unicode) ------------------------------------------
struct ByteAlphabet {
    uint32_t to_cp[256] = {};
    uint8_t from_cp[0x400] = {};
    std::string to_str[256];

    ByteAlphabet() {
        std::array<bool, 256> printable{};
        auto mark = [&](int lo, int hi) {
            for (int b = lo; b <= hi; ++b) printable[b] = true;
        };
        mark('!', '~');
        mark(0xA1, 0xAC);
        mark(0xAE, 0xFF);
        int next = 0;
        for (int b = 0; b < 256; ++b) to_cp[b] = printable[b] ? uint32_t(b) : 256u + uint32_t(next++);
        std::fill(std::begin(from_cp), std::end(from_cp), uint8_t(0xff));
        for (int b = 0; b < 256; ++b) {
            const uint32_t cp = to_cp[b];
            if (cp < sizeof from_cp) from_cp[cp] = uint8_t(b);
            std::string s;
            uni::append_utf8(cp, s);
            to_str[b] = std::move(s);
        }
    }
};

const ByteAlphabet& alphabet() {
    static const ByteAlphabet a;
    return a;
}

// ---- the pre-split ----------------------------------------------------------------------------

size_t ws_run_end(std::string_view t, size_t i) {
    size_t q = i;
    while (q < t.size()) {
        uint32_t cp = 0;
        const size_t l = uni::decode(t, q, &cp);
        if (!l || !uni::is_space(cp)) break;
        q += l;
    }
    return q;
}

size_t letter_mark_run(std::string_view t, size_t i) {
    size_t q = i;
    while (q < t.size()) {
        uint32_t cp = 0;
        const size_t l = uni::decode(t, q, &cp);
        if (!l || !(uni::is_letter(cp) || uni::is_mark(cp))) break;
        q += l;
    }
    return q - i;
}

// 1: (?i:'s|'t|'re|'ve|'m|'ll|'d)
size_t alt_contraction(std::string_view t, size_t i) {
    if (t[i] != '\'' || i + 1 >= t.size()) return 0;
    const char c = t[i + 1];
    if (c == 's' || c == 'S' || c == 't' || c == 'T' || c == 'm' || c == 'M' || c == 'd' || c == 'D') return 2;
    if (i + 2 >= t.size()) return 0;
    const char c0 = (c >= 'A' && c <= 'Z') ? char(c + 32) : c;
    const char d = t[i + 2];
    const char d0 = (d >= 'A' && d <= 'Z') ? char(d + 32) : d;
    if ((c0 == 'r' && d0 == 'e') || (c0 == 'v' && d0 == 'e') || (c0 == 'l' && d0 == 'l')) return 3;
    return 0;
}

// 2: [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
size_t alt_letter(std::string_view t, size_t i) {
    uint32_t cp = 0;
    const size_t l = uni::decode(t, i, &cp);
    if (l && cp != '\r' && cp != '\n' && !uni::is_letter(cp) && !uni::is_number(cp)) {
        const size_t run = letter_mark_run(t, i + l);
        if (run) return l + run;
    }
    return letter_mark_run(t, i);
}

// 3: \p{N} — one number codepoint: Qwen splits digits individually.
size_t alt_number(std::string_view t, size_t i) {
    uint32_t cp = 0;
    const size_t l = uni::decode(t, i, &cp);
    return (l && uni::is_number(cp)) ? l : 0;
}

// 3 (GPT-4o): \p{N}{1,3} — one to three number codepoints, greedy. This is the only place the GPT-4o
// pre-tokenizer and the Qwen2 one differ once llama.cpp's category collapse is taken into account, and
// it is worth a whole class of tokens: Inkling-2-Mini carries single tokens for "123", "84" and "126",
// so splitting digits one at a time loses them (measured: "1234" is [70904, 19] in llama.cpp against
// [16,17,18,19] from the Qwen2 rule).
// 2 (GPT-4o): the letter piece carries an optional contraction SUFFIX, where Qwen2 has the contraction
// as an alternative in its own right. MEASURED on ".iß'SsRK": llama.cpp splits it [.iß'S][sRK] and the
// suffix form reproduces its ids exactly ([., iß, ', S, s, RK]); without it the piece is [.iß]['SsRK]
// and the merge table -- which has S+s at rank 72066 but no ('\'','S') pair at all -- then yields
// [., iß, ', Ss, RK] instead. A case-sensitive letter rule (upper-ish = "a letter that is not a-z",
// lower-ish = "a letter that is not A-Z") was tried for the remaining random-byte cases and REVERTED:
// it splits ASCII words that must stay whole (tools.txt went from exact to 14 files differing), so it
// is not the rule llama.cpp applies. See the note in the commit.
size_t alt_letter_contraction(std::string_view t, size_t i) {
    const size_t n = alt_letter(t, i);
    return n ? n + alt_contraction(t, i + n) : 0;
}

size_t alt_number_1_3(std::string_view t, size_t i) {
    size_t n = 0;
    for (int k = 0; k < 3; ++k) {
        uint32_t cp = 0;
        const size_t l = uni::decode(t, i + n, &cp);
        if (!l || !uni::is_number(cp)) break;
        n += l;
    }
    return n;
}

// 4: ` ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*`
size_t alt_symbols(std::string_view t, size_t i) {
    auto run = [&](size_t q) -> size_t {
        size_t n = 0;
        while (q < t.size()) {
            uint32_t cp = 0;
            const size_t l = uni::decode(t, q, &cp);
            if (!l) break;
            if (uni::is_space(cp) || uni::is_letter(cp) || uni::is_mark(cp) || uni::is_number(cp)) break;
            q += l;
            n += l;
        }
        if (!n) return 0;
        while (q < t.size() && (t[q] == '\r' || t[q] == '\n')) {
            ++q;
            ++n;
        }
        return n;
    };
    if (t[i] == ' ') {
        if (const size_t n = run(i + 1)) return 1 + n;  // the leading space is part of the piece
    }
    return run(i);
}

// 5: \s*[\r\n]+ — the whitespace run up to and including its last newline.
// 4 (GPT-4o): ` ?[^\s\p{L}\p{N}]+[\r\n/]*` — the symbol run does NOT exclude \p{M}, so a mark rides
// along with the symbol before it, and the run absorbs trailing '/' as well as \r\n. MEASURED: U+2B24
// followed by U+FE0F ("âĿ¤ï¸ı" byte-encoded) is ONE token in llama.cpp and two in the Qwen2 rule, which
// breaks the run at the mark. This is the second of the two differences that matter; the number rule
// is the first.
size_t alt_symbols_slash(std::string_view t, size_t i) {
    auto run = [&](size_t q) -> size_t {
        size_t n = 0;
        while (q < t.size()) {
            uint32_t cp = 0;
            const size_t l = uni::decode(t, q, &cp);
            if (!l) break;
            if (uni::is_space(cp) || uni::is_letter(cp) || uni::is_number(cp)) break;
            q += l;
            n += l;
        }
        if (!n) return 0;
        while (q < t.size() && (t[q] == '\r' || t[q] == '\n' || t[q] == '/')) {
            ++q;
            ++n;
        }
        return n;
    };
    if (t[i] == ' ') {
        if (const size_t n = run(i + 1)) return 1 + n;  // the leading space is part of the piece
    }
    return run(i);
}

size_t alt_ws_newlines(std::string_view t, size_t i) {
    const size_t e = ws_run_end(t, i);
    if (e == i) return 0;
    size_t after_last_nl = 0;
    for (size_t q = i; q < e; ++q)
        if (t[q] == '\r' || t[q] == '\n') after_last_nl = q + 1;
    return after_last_nl ? after_last_nl - i : 0;
}

// 6: \s+(?!\S) — greedy, then backtrack one codepoint when the run is followed by a non-space.
size_t alt_ws_not_before_word(std::string_view t, size_t i) {
    const size_t e = ws_run_end(t, i);
    if (e == i) return 0;
    if (e == t.size()) return e - i;
    size_t last = i;
    for (size_t q = i; q < e;) {
        uint32_t cp = 0;
        const size_t l = uni::decode(t, q, &cp);
        if (!l) break;
        last = q;
        q += l;
    }
    return last > i ? last - i : 0;
}

// 7: \s+
size_t alt_ws(std::string_view t, size_t i) { return ws_run_end(t, i) - i; }

// The length of the piece starting at i: the first alternative that matches, or 0.
size_t match_here(std::string_view t, size_t i, bool gpt4o) {
    if (gpt4o) {
        // The GPT-4o alternation. The letter alternatives collapse to the Qwen2 letter rule (see
        // Vocab::PreStyle), a contraction is a suffix of a letter piece that measures the same either
        // way ("don't" is [1630, 6, 83] from both), and what is left is the number rule and the symbol
        // run's treatment of marks.
        if (const size_t n = alt_letter_contraction(t, i)) return n;
        if (const size_t n = alt_number_1_3(t, i)) return n;
        if (const size_t n = alt_symbols_slash(t, i)) return n;
        if (const size_t n = alt_ws_newlines(t, i)) return n;
        if (const size_t n = alt_ws_not_before_word(t, i)) return n;
        return alt_ws(t, i);
    }
    if (const size_t n = alt_contraction(t, i)) return n;
    if (const size_t n = alt_letter(t, i)) return n;
    if (const size_t n = alt_number(t, i)) return n;
    if (const size_t n = alt_symbols(t, i)) return n;
    if (const size_t n = alt_ws_newlines(t, i)) return n;
    if (const size_t n = alt_ws_not_before_word(t, i)) return n;
    return alt_ws(t, i);
}

}  // namespace

uint32_t byte_to_cp(uint8_t b) { return alphabet().to_cp[b]; }

uint8_t cp_to_byte(uint32_t cp) { return cp < 0x400 ? alphabet().from_cp[cp] : uint8_t(0xff); }

const std::string& byte_to_str(uint8_t b) { return alphabet().to_str[b]; }

// ---- vocabulary -------------------------------------------------------------------------------

int32_t Vocab::id(std::string_view piece) const {
    const auto it = id_of.find(piece);
    return it == id_of.end() ? -1 : it->second;
}

bool Vocab::match_special(std::string_view text, size_t at, bool allow_control, int32_t* id, size_t* len) const {
    // `specials` is sorted longest first, so the first hit is the longest.
    for (const auto& [piece, tok] : specials) {
        if (piece.size() > text.size() - at) continue;
        if (!allow_control && types[size_t(tok)] == kControl) continue;
        if (text.compare(at, piece.size(), piece) != 0) continue;
        *id = tok;
        *len = piece.size();
        return true;
    }
    return false;
}

bool Vocab::load(const gguf::File& f, Vocab& out, std::string* err) {
    const auto model = f.get_str("tokenizer.ggml.model");
    if (!model) {
        *err = "tokenizer.ggml.model is missing";
        return false;
    }
    out.model = *model;
    if (*model != "gpt2" && *model != "gemma4") {
        *err = "tokenizer.ggml.model is \"" + std::string(*model) +
               "\": only gpt2 (byte-level BPE) and gemma4 (raw UTF-8 BPE) are implemented";
        return false;
    }
    out.byte_level = *model == "gpt2";
    out.escape_spaces = *model == "gemma4";
    const auto* toks = f.get_str_array("tokenizer.ggml.tokens");
    if (!toks || toks->empty()) {
        *err = "tokenizer.ggml.tokens is missing or empty";
        return false;
    }
    out.tokens = *toks;

    auto types = f.get_i32_array("tokenizer.ggml.token_type");
    if (types.size() != out.tokens.size()) {
        *err = "tokenizer.ggml.token_type has " + std::to_string(types.size()) + " entries for " +
               std::to_string(out.tokens.size()) + " tokens";
        return false;
    }
    out.types = std::move(types);

    const auto* merges = f.get_str_array("tokenizer.ggml.merges");
    if (!merges || merges->empty()) {
        *err = "tokenizer.ggml.merges is missing: a gpt2 vocabulary without merges cannot be encoded";
        return false;
    }
    out.rank.reserve(merges->size() * 2);
    for (size_t i = 0; i < merges->size(); ++i) {
        const std::string_view m = (*merges)[i];
        const size_t sp = m.find(' ');
        if (sp == std::string_view::npos || sp == 0 || sp + 1 == m.size()) {
            *err = "merge " + std::to_string(i) + " is not a pair: \"" + std::string(m) + "\"";
            return false;
        }
        std::string key(m.substr(0, sp));
        key.push_back('\0');
        key.append(m.substr(sp + 1));
        out.rank.emplace(std::move(key), int32_t(i));
    }

    // Last definition wins, the same rule the oracle uses when it builds its token -> id map.
    out.id_of.reserve(out.tokens.size() * 2);
    for (size_t i = 0; i < out.tokens.size(); ++i) out.id_of[out.tokens[i]] = int32_t(i);

    out.pre = f.get_str("tokenizer.ggml.pre").value_or("");
    // llama.cpp's table (src/llama-vocab.cpp): these pres are LLAMA_VOCAB_PRE_TYPE_GPT4O.
    for (const std::string_view p : {"gpt-4o", "llama4", "kanana2", "talkie", "inkling2"})
        if (out.pre == p) out.pre_style = Vocab::PreStyle::Gpt4o;
    out.bos = int32_t(f.get_int("tokenizer.ggml.bos_token_id").value_or(-1));
    out.eos = int32_t(f.get_int("tokenizer.ggml.eos_token_id").value_or(-1));
    out.pad = int32_t(f.get_int("tokenizer.ggml.padding_token_id").value_or(-1));
    out.add_bos = f.get_int("tokenizer.ggml.add_bos_token").value_or(0) != 0;

    // The end-of-generation tokens the oracle recognises by their text are control tokens whatever type the
    // file gave them (gemma's <eos>, <turn|> and <|tool_response>); gemma's </s> is one of them only until
    // the vocabulary also has <|tool_response>, which makes it an ordinary token again. MEASURED: without
    // this, "<eos>" in a prompt is tokenised as five characters instead of token 1.
    {
        static const std::string_view kEogTexts[] = {
            "<|eot_id|>", "<|eot|>", "<|im_end|>", "<|end|>", "<|return|>", "<|call|>", "<|flush|>", "<|calls|>",
            "<end_of_turn>", "<|endoftext|>", "</s>", "<|eom_id|>", "<EOT>", "_<EOT>", "[EOT]", "[EOS]",
            "<|end_of_text|>", "<end_of_utterance>", "<eos>", "<turn|>", "<|tool_response>",
            "<\xEF\xBD\x9C" "end\xE2\x96\x81of\xE2\x96\x81sentence\xEF\xBD\x9C>", "[e~[", "<|ifm|im_end|>"};
        int32_t s_id = -1;
        bool has_tool_response = false;
        for (size_t i = 0; i < out.tokens.size(); ++i) {
            if (std::find(std::begin(kEogTexts), std::end(kEogTexts), out.tokens[i]) == std::end(kEogTexts)) continue;
            if (out.tokens[i] == "<|tool_response>") has_tool_response = true;
            if (out.tokens[i] == "</s>") s_id = int32_t(i);
            if (out.types[i] != kControl) out.types[i] = kControl;
        }
        if (has_tool_response && s_id >= 0) out.types[size_t(s_id)] = kNormal;
    }

    for (size_t i = 0; i < out.tokens.size(); ++i) {
        const int32_t t = out.types[i];
        if (t == kControl || t == kUserDefined) out.specials.emplace_back(out.tokens[i], int32_t(i));
        if (t == kUnknown && out.unk < 0) out.unk = int32_t(i);
    }
    std::stable_sort(out.specials.begin(), out.specials.end(),
                     [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
    return true;
}

// ---- encode -----------------------------------------------------------------------------------

std::vector<std::string_view> Tokenizer::split(std::string_view text) const {
    std::vector<std::string_view> out;
    size_t i = 0;
    if (!v_->byte_level) {
        // [^\n]+|[\n]+ : alternating runs of text and of newlines.
        while (i < text.size()) {
            const bool nl = text[i] == '\n';
            size_t j = i;
            while (j < text.size() && (text[j] == '\n') == nl) ++j;
            out.push_back(text.substr(i, j - i));
            i = j;
        }
        return out;
    }
    while (i < text.size()) {
        size_t n = match_here(text, i, v_->pre_style == Vocab::PreStyle::Gpt4o);
        if (!n) n = 1;  // never stall
        out.push_back(text.substr(i, n));
        i += n;
    }
    return out;
}

namespace {
// The length of the UTF-8 character that starts with this byte, as a lead-byte table gives it: 1 for ASCII
// and for a stray continuation byte, 2, 3 or 4 by the lead's high bits.
size_t utf8_len_from_lead(unsigned char b) {
    static const unsigned char by_high_nibble[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
    return by_high_nibble[b >> 4];
}
}  // namespace

// The BPE of one run of the raw-UTF-8 family: symbols are whole characters, merges follow the file's ranks.
void Tokenizer::bpe_raw(std::string_view piece, std::vector<int32_t>& out) const {
    const Vocab& v = *v_;
    std::string s;
    s.reserve(piece.size() + 8);
    for (const char ch : piece) {
        if (ch == ' ' && v.escape_spaces) s += "\xE2\x96\x81";
        else s.push_back(ch);
    }
    // A run of newlines only is one token when the vocabulary has it, whatever its length.
    if (s.find_first_not_of('\n') == std::string::npos) {
        if (const int32_t whole = v.id(s); whole >= 0) {
            out.push_back(whole);
            return;
        }
    }
    std::vector<std::string> sym;
    for (size_t i = 0; i < s.size();) {
        const size_t l = std::min(s.size() - i, utf8_len_from_lead(static_cast<unsigned char>(s[i])));
        sym.emplace_back(s, i, l);
        i += l;
    }
    while (sym.size() > 1) {
        size_t best = 0;
        int32_t best_rank = std::numeric_limits<int32_t>::max();
        std::string key;
        for (size_t i = 0; i + 1 < sym.size(); ++i) {
            key.assign(sym[i]);
            key.push_back('\0');
            key.append(sym[i + 1]);
            const auto it = v.rank.find(key);
            if (it != v.rank.end() && it->second < best_rank) {
                best_rank = it->second;
                best = i;
            }
        }
        if (best_rank == std::numeric_limits<int32_t>::max()) break;
        sym[best] += sym[best + 1];
        sym.erase(sym.begin() + long(best) + 1);
    }
    for (const std::string& p : sym) {
        const int32_t tok = v.id(p);
        if (tok >= 0) {
            out.push_back(tok);
            continue;
        }
        for (const char ch : p) {   // no token: one <0xXX> byte token per byte, when the vocabulary has them
            char name[8];
            std::snprintf(name, sizeof name, "<0x%02X>", unsigned(static_cast<unsigned char>(ch)));
            const int32_t one = v.id(name);
            if (one >= 0) out.push_back(one);
        }
    }
}

void Tokenizer::bpe(std::string_view piece, std::vector<int32_t>& out) const {
    const Vocab& v = *v_;
    if (!v.byte_level) {
        bpe_raw(piece, out);
        return;
    }
    const ByteAlphabet& A = alphabet();
    std::string s;
    s.reserve(piece.size() * 2);
    for (size_t i = 0; i < piece.size(); ++i) s += A.to_str[static_cast<unsigned char>(piece[i])];

    // A piece that is already a token is that token (only a normal token: a control token written in
    // the text is not produced by the byte-level path).
    if (const int32_t whole = v.id(s); whole >= 0 && v.types[whole] == kNormal) {
        out.push_back(whole);
        return;
    }

    std::vector<std::string> sym;
    sym.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        uint32_t cp = 0;
        size_t l = uni::decode(s, i, &cp);
        if (!l) l = 1;
        sym.emplace_back(s, i, l);
        i += l;
    }

    while (sym.size() > 1) {
        size_t best = 0;
        int32_t best_rank = std::numeric_limits<int32_t>::max();
        std::string key;
        for (size_t i = 0; i + 1 < sym.size(); ++i) {
            key.assign(sym[i]);
            key.push_back('\0');
            key.append(sym[i + 1]);
            const auto it = v.rank.find(key);
            if (it != v.rank.end() && it->second < best_rank) {
                best_rank = it->second;
                best = i;
            }
        }
        if (best_rank == std::numeric_limits<int32_t>::max()) break;
        sym[best] += sym[best + 1];
        sym.erase(sym.begin() + long(best) + 1);
    }

    for (const std::string& p : sym) {
        const int32_t tok = v.id(p);
        if (tok >= 0) {
            out.push_back(tok);
            continue;
        }
        // No token for this symbol: fall back to the <0xXX> byte token of each of its codepoints,
        // then to unknown. Every byte-level codepoint of every byte is a token in a gpt2 vocabulary,
        // so this path is a guard, not a route.
        uint32_t cp = 0;
        for (size_t i = 0; i < p.size();) {
            size_t l = uni::decode(p, i, &cp);
            if (!l) l = 1;
            const uint8_t b = cp_to_byte(cp);
            int32_t one = -1;
            if (b != 0xff) {
                char name[8];
                std::snprintf(name, sizeof name, "<0x%02X>", unsigned(b));
                one = v.id(name);
            }
            if (one < 0) one = v.unk;
            if (one >= 0) out.push_back(one);
            i += l;
        }
    }
}

void Tokenizer::encode(std::string_view text, std::vector<int32_t>& out, const Options& o) const {
    const Vocab& v = *v_;
    if (o.add_bos && v.bos >= 0) out.push_back(v.bos);
    // Special tokens written literally in the text are cut out first, then the gaps are encoded — the
    // same shape the oracle uses, so a symbol run cannot swallow a special token. User-defined tokens
    // are always cut out; control tokens only when the caller asked for them (see match_special).
    size_t start = 0;
    size_t i = 0;
    while (i < text.size()) {
        int32_t id = -1;
        size_t len = 0;
        if (v.match_special(text, i, o.parse_special, &id, &len)) {
            for (const std::string_view piece : split(text.substr(start, i - start))) bpe(piece, out);
            out.push_back(id);
            i += len;
            start = i;
        } else {
            ++i;
        }
    }
    for (const std::string_view piece : split(text.substr(start))) bpe(piece, out);
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids) const { return decode(ids.data(), ids.size()); }

std::string Tokenizer::decode(const int32_t* ids, size_t n) const {
    const Vocab& v = *v_;
    std::string out;
    for (size_t i = 0; i < n; ++i) {
        if (ids[i] < 0 || size_t(ids[i]) >= v.tokens.size()) continue;
        const std::string_view piece = v.tokens[size_t(ids[i])];
        if (!v.byte_level) {
            // Raw UTF-8 family: U+2581 is a space, a byte token is its byte, everything else is its text.
            const int32_t ty = v.types[size_t(ids[i])];
            if (ty == kByte && piece.size() == 6 && piece.compare(0, 3, "<0x") == 0 && piece[5] == '>') {
                out.push_back(char(std::strtol(std::string(piece.substr(3, 2)).c_str(), nullptr, 16)));
            } else if (ty == kNormal) {
                for (size_t q = 0; q < piece.size();) {
                    if (piece.compare(q, 3, "\xE2\x96\x81") == 0) {
                        out.push_back(' ');
                        q += 3;
                    } else {
                        out.push_back(piece[q++]);
                    }
                }
            } else {
                out.append(piece);
            }
            continue;
        }
        for (size_t q = 0; q < piece.size();) {
            uint32_t cp = 0;
            size_t l = uni::decode(piece, q, &cp);
            if (!l) l = 1;
            const uint8_t b = cp_to_byte(cp);
            if (b != 0xff) out.push_back(char(b));
            else out.append(piece.substr(q, l));  // not in the byte alphabet: copy verbatim
            q += l;
        }
    }
    return out;
}

}  // namespace limbo::tok
