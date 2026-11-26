// Copyright (c) 2026 ITERFORM. All rights reserved.
// tests/tok_test.cpp — the tokenizer: the byte-level alphabet, the pre-split, and BPE on a
// vocabulary built here. With LIMBO_TEST_GGUF set the real vocabulary is loaded too and the round trip
// decode(encode(text)) == text is checked on text that exercises every alternative of the pattern.
//
// Byte-exactness against the oracle is not a unit test: it is scripts/tok_compare.sh, which diffs
// tok_dump against llama-tokenize over a corpus.
#include "limbo/check.hpp"
#include "limbo/gguf.hpp"
#include "limbo/tok.hpp"

#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>

using limbo::tok::Tokenizer;
using limbo::tok::Vocab;

namespace {

// A vocabulary of the 256 byte-level pieces plus a few merges, so BPE can be tested without a model.
struct MiniVocab {
    std::vector<std::string> storage;
    Vocab v;

    MiniVocab() { storage.reserve(1024); }

    int32_t add(std::string_view piece, int32_t type = limbo::tok::kNormal) {
        storage.emplace_back(piece);
        v.tokens.push_back(storage.back());
        v.types.push_back(type);
        const int32_t id = int32_t(v.tokens.size()) - 1;
        v.id_of[v.tokens.back()] = id;
        return id;
    }
    void merge(std::string_view a, std::string_view b) {
        std::string key(a);
        key.push_back('\0');
        key.append(b);
        v.rank[key] = int32_t(v.rank.size());
    }
    int32_t id(std::string_view piece) const { return v.id(piece); }
};

// Pieces contain spaces, bars and newlines, so the display separator is U+2759 (❙), which no test
// input contains.
std::string join(const std::vector<std::string_view>& v, const char* sep = "❙") {
    std::string out;
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) out += sep;
        out += v[i];
    }
    return out;
}

void check_split(const Tokenizer& t, std::string_view text, const std::string& want) {
    const std::string got = join(t.split(text));
    if (got == want) {
        std::printf("ok   split(\"%s\") -> [%s]\n", std::string(text).c_str(), got.c_str());
        return;
    }
    std::printf("FAIL split(\"%s\")\n       got  [%s]\n       want [%s]\n", std::string(text).c_str(), got.c_str(),
                want.c_str());
    ++limbo::test::failures();
}

}  // namespace

int main() {
    // ---- the byte-level alphabet -------------------------------------------------------------
    for (uint32_t b = 0; b < 256; ++b) {
        const uint32_t cp = limbo::tok::byte_to_cp(uint8_t(b));
        if (limbo::tok::cp_to_byte(cp) != b) {
            std::printf("FAIL byte %u maps to codepoint 0x%X and back to %u\n", b, cp, limbo::tok::cp_to_byte(cp));
            ++limbo::test::failures();
        }
    }
    CHECK(limbo::tok::byte_to_cp(0x20) == 0x120, "space maps to U+0120");
    CHECK(limbo::tok::byte_to_cp(0x0A) == 0x10A, "newline maps to U+010A");
    CHECK(limbo::tok::byte_to_cp('a') == 'a', "a printable byte maps to itself");
    CHECK(limbo::tok::cp_to_byte(0x1234) == 0xff, "a codepoint outside the alphabet has no byte");

    // ---- the pre-split ----------------------------------------------------------------------
    static const Vocab kEmpty;
    const Tokenizer empty(kEmpty);
    check_split(empty, "", "");
    check_split(empty, "hello world", "hello❙ world");
    check_split(empty, "don't", "don❙'t");
    check_split(empty, "DON'T", "DON❙'T");
    check_split(empty, "123", "1❙2❙3");
    check_split(empty, "a  b", "a❙ ❙ b");
    check_split(empty, "x   ", "x❙   ");
    // Two spaces, then " x": at the third position the optional non-letter char takes the space and
    // the letter run takes the x, and that alternative comes before the whitespace ones.
    check_split(empty, "   x", "  ❙ x");
    check_split(empty, "a\n\nb", "a❙\n\n❙b");
    check_split(empty, "x!\n", "x❙!\n");
    check_split(empty, "  \n  ", "  \n❙  ");
    check_split(empty, "it's 12.5% ok", "it❙'s❙ ❙1❙2❙.❙5❙%❙ ok");
    // A letter run keeps its combining marks; CJK and emoji are one piece each.
    check_split(empty, "\xE6\x97\xA5\xE6\x9C\xAC", "\xE6\x97\xA5\xE6\x9C\xAC");
    check_split(empty, "e\xCC\x81", "e\xCC\x81");
    check_split(empty, "\xC2\xBD", "\xC2\xBD");                          // U+00BD is a number (No)
    check_split(empty, "\xE2\x85\xA7", "\xE2\x85\xA7");                  // U+2167 is a number (Nl)
    check_split(empty, "\xF0\x9F\x98\x80", "\xF0\x9F\x98\x80");          // emoji: a symbol run
    // U+00A0 is whitespace, so it cannot be the optional leading char of a letter run: it ends up in
    // front of the letters instead.
    check_split(empty, "a\xC2\xA0""b", "a❙\xC2\xA0""b");
    // A special token is four pieces to the pattern: that is why the vocabulary carries it as a
    // control token and parse_special cuts it out before the split.
    check_split(empty, "<|im_start|>", "<|❙im❙_start❙|>");

    // ---- BPE on a vocabulary built here ------------------------------------------------------
    {
        MiniVocab mv;
        for (uint32_t b = 0; b < 256; ++b) mv.add(limbo::tok::byte_to_str(uint8_t(b)));
        const int32_t ab = mv.add("ab");
        mv.merge("a", "b");
        const int32_t a = mv.id("a");
        const int32_t c = mv.id("c");
        const Tokenizer t(mv.v);

        std::vector<int32_t> out;
        t.encode("ab", out);
        CHECK(out.size() == 1 && out[0] == ab, "a whole piece that is a token is one id");
        out.clear();
        t.encode("aab", out);
        CHECK(out.size() == 2 && out[0] == a && out[1] == ab, "a merge is applied where its rank is lowest");
        out.clear();
        t.encode("abc", out);
        CHECK(out.size() == 2 && out[0] == ab && out[1] == c, "a merge stops when no pair has a rank");
        out.clear();
        t.encode("", out);
        CHECK(out.empty(), "the empty text has no tokens");
        CHECK(t.decode(std::vector<int32_t>{ab, c}) == "abc", "decode reverses the byte-level alphabet");
    }

    // ---- the raw-UTF-8 family (gemma4) on a vocabulary built here ---------------------------------------
    {
        MiniVocab mv;
        mv.v.byte_level = false;
        mv.v.escape_spaces = true;
        for (const char* piece : {"a", "b", "c", "\xE2\x96\x81", "\n", "ab", "\xE2\x96\x81" "ab", "\n\n\n", "\xC3\xA9"})
            mv.add(piece);
        const int32_t byte_a = mv.add("<0x61>", limbo::tok::kByte);
        const int32_t byte_e1 = mv.add("<0xE2>", limbo::tok::kByte);
        const int32_t sp = mv.id("\xE2\x96\x81"), nl = mv.id("\n"), nl3 = mv.id("\n\n\n");
        const int32_t ab = mv.id("ab"), spab = mv.id("\xE2\x96\x81" "ab"), a = mv.id("a"), c = mv.id("c");
        mv.merge("a", "b");
        mv.merge("\xE2\x96\x81", "ab");
        const Tokenizer t(mv.v);
        // The pre-split only separates newline runs from the text between them.
        CHECK(join(t.split("ab c\n\nab")) == "ab c❙\n\n❙ab", "gemma pre-split: runs of newlines against the rest");
        std::vector<int32_t> out;
        t.encode(" ab", out);
        CHECK(out.size() == 1 && out[0] == spab, "a space is U+2581 and merges into the next piece");
        out.clear();
        t.encode("a c", out);
        CHECK(out.size() == 3 && out[0] == a && out[1] == sp && out[2] == c, "a lone space is the U+2581 token");
        out.clear();
        t.encode("\n\n\n", out);
        CHECK(out.size() == 1 && out[0] == nl3, "a run of newlines that is a token is taken whole");
        out.clear();
        t.encode("\n\n", out);
        CHECK(out.size() == 2 && out[0] == nl && out[1] == nl, "a run of newlines that is not a token falls to characters");
        out.clear();
        t.encode("ab\xC3\xA9", out);
        CHECK(out.size() == 2 && out[0] == ab, "symbols are whole UTF-8 characters");
        out.clear();
        // Two characters the vocabulary lacks: each byte of them falls back to its <0xXX> token if it has one.
        t.encode("\xE2\x82\xAC", out);
        CHECK(out.size() == 1 && out[0] == byte_e1, "a character without a token falls back to its bytes' tokens");
        CHECK(byte_a >= 0, "the byte tokens exist");
        CHECK(t.decode(std::vector<int32_t>{spab, nl, c}) == " ab\nc", "decode turns U+2581 into a space");
        CHECK(t.decode(std::vector<int32_t>{byte_a}) == "a", "decode turns a <0xXX> token into its byte");
    }

    // ---- the real vocabulary ------------------------------------------------------------------
    const char* real = std::getenv("LIMBO_TEST_GGUF");
    if (!real || !*real) {
        std::printf("--   LIMBO_TEST_GGUF not set: the real vocabulary is skipped\n");
        return limbo::test::finish("tok");
    }
    const std::string path = std::string(real).substr(0, std::string(real).find(':'));
    std::string err;
    auto f = limbo::gguf::File::open(path, &err);
    if (!f) {
        std::printf("FAIL cannot open %s: %s\n", path.c_str(), err.c_str());
        ++limbo::test::failures();
        return limbo::test::finish("tok");
    }
    Vocab v;
    CHECK(Vocab::load(*f, v, &err), "the real vocabulary loads");
    if (v.tokens.empty()) return limbo::test::finish("tok");

    std::printf("     %.*s: %zu tokens, %zu merges, pre \"%.*s\", bos %d, eos %d, add_bos %d\n",
                (int)v.model.size(), v.model.data(), v.tokens.size(), v.rank.size(), (int)v.pre.size(),
                v.pre.data(), v.bos, v.eos, int(v.add_bos));
    CHECK(v.tokens.size() == v.types.size(), "every token has a type");
    if (v.byte_level) {
        CHECK(v.rank.size() == 247587, "the merge table has every merge");
        int missing_bytes = 0;
        for (uint32_t b = 0; b < 256; ++b)
            if (v.id(limbo::tok::byte_to_str(uint8_t(b))) < 0) ++missing_bytes;
        CHECK(missing_bytes == 0, "every byte-level codepoint is a token (no byte fallback needed)");
    } else {
        CHECK(v.rank.size() > 0, "the merge table is not empty");
        int missing_bytes = 0;
        for (unsigned b = 0; b < 256; ++b) {
            char name[8];
            std::snprintf(name, sizeof name, "<0x%02X>", b);
            if (v.id(name) < 0) ++missing_bytes;
        }
        CHECK(missing_bytes == 0, "every byte has a <0xXX> fallback token");
    }

    int bad_merge = 0;
    for (const auto& [key, rank] : v.rank) {
        const size_t sp = key.find('\0');
        if (sp == std::string::npos) {
            ++bad_merge;
            continue;
        }
        const std::string a = key.substr(0, sp), b = key.substr(sp + 1);
        const int32_t ia = v.id(a), ib = v.id(b);
        if (ia < 0 || ib < 0) ++bad_merge;
        else if (rank < 0 || rank >= int32_t(v.rank.size())) ++bad_merge;
    }
    CHECK(bad_merge == 0, "every merge's two halves are tokens");

    const Tokenizer t(v);
    // The special path: a control token written in the text is one id, and the text around it is
    // encoded normally.
    {
        const char* open_tag = v.byte_level ? "<|im_start|>" : "<|turn>";
        const char* close_tag = v.byte_level ? "<|im_end|>" : "<turn|>";
        const int32_t start = v.id(open_tag);
        const int32_t end = v.id(close_tag);
        CHECK(start >= 0 && end >= 0, "the chat control tokens are in the vocabulary");
        std::vector<int32_t> out;
        limbo::tok::Options o;
        o.parse_special = true;
        t.encode(std::string(open_tag) + "system\nhi" + close_tag, out, o);
        CHECK(!out.empty() && out.front() == start, "parse_special takes the turn-open token as one id");
        CHECK(!out.empty() && out.back() == end, "parse_special takes the turn-close token as one id");
        if (!v.byte_level) {
            // gemma's <eos> is typed normal in the file and is a control token to the oracle
            out.clear();
            t.encode("<eos>", out, o);
            CHECK(out.size() == 1 && out[0] == v.id("<eos>"), "<eos> is one token with parse_special");
            out.clear();
            t.encode("<eos>", out);
            CHECK(out.size() > 1, "<eos> is plain text without parse_special");
        }
    }
    const char* texts[] = {
        "Hello, world!",
        "def f(x):\n    return x * 2\n",
        "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"K\xC3\xB6ln\"}}",
        "  leading and trailing   ",
        "tab\tand\nnewline\r\n",
        "numbers 0123456789 and 12.5%",
        "don't can't it's I'M we're they've he'll she'd",
        "\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E and \xD0\xA0\xD1\x83\xD1\x81\xD1\x81\xD0\xBA\xD0\xB8\xD0\xB9",
        "e\xCC\x81 combining \xF0\x9F\x98\x80 emoji",
        "<|im_start|>system\nYou are Ariadne.<|im_end|>\n",
        "\xC2\xBD and \xE2\x85\xA7 and \xC2\xA0nbsp",
        "code: `x = [i for i in range(10)]`  # comment",
    };
    for (const char* text : texts) {
        std::vector<int32_t> out;
        t.encode(text, out);
        const std::string back = t.decode(out);
        if (back == text) {
            std::printf("ok   round trip (%zu tokens): %.40s\n", out.size(), text);
        } else {
            std::printf("FAIL round trip: in  \"%s\"\n                    out \"%s\"\n", text, back.c_str());
            ++limbo::test::failures();
        }
    }

    return limbo::test::finish("tok");
}
