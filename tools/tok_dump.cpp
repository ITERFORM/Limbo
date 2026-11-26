// Copyright (c) 2026 ITERFORM. All rights reserved.
// tools/tok_dump.cpp — encode text with Limbo's tokenizer, for diffing against the oracle.
//
//   tok_dump -m model.gguf [-f file | --stdin | -p text] [--ids|--pieces|--split]
//              [--no-bos] [--parse-special]
//
// --ids prints the same Python-parseable list llama-tokenize --ids prints, so the two outputs can be
// diffed directly; --split prints the pre-split pieces, for diffing against the model's own
// pre_tokenizer; --pieces prints one token per line with its piece escaped.
#include "limbo/gguf.hpp"
#include "limbo/tok.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

std::string escape(std::string_view s) {
    std::string out;
    for (unsigned char c : s) {
        if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else if (c == '\\') out += "\\\\";
        else if (c < 0x20 || c == 0x7f) {
            char b[8];
            std::snprintf(b, sizeof b, "\\x%02X", c);
            out += b;
        } else out.push_back(char(c));
    }
    return out;
}

bool read_all(FILE* f, std::string& out) {
    char buf[65536];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) out.append(buf, n);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    const char* model = nullptr;
    const char* file = nullptr;
    const char* prompt = nullptr;
    const char* lookup = nullptr;
    const char* decode_ids = nullptr;
    bool ids = false, pieces = false, split_only = false, no_bos = false, parse_special = false, use_stdin = false;

    for (int i = 1; i < argc; ++i) {
        auto is = [&](const char* a) { return std::strcmp(argv[i], a) == 0; };
        if ((is("-m") || is("--model")) && i + 1 < argc) model = argv[++i];
        else if ((is("-f") || is("--file")) && i + 1 < argc) file = argv[++i];
        else if ((is("-p") || is("--prompt")) && i + 1 < argc) prompt = argv[++i];
        else if (is("--ids")) ids = true;
        else if (is("--pieces")) pieces = true;
        else if (is("--split")) split_only = true;
        else if (is("--no-bos")) no_bos = true;
        else if (is("--parse-special")) parse_special = true;
        else if (is("--stdin")) use_stdin = true;
        else if ((is("--piece") || is("--token")) && i + 1 < argc) lookup = argv[++i];
        else if (is("--decode") && i + 1 < argc) decode_ids = argv[++i];
        else {
            std::fprintf(stderr,
                         "usage: tok_dump -m model.gguf [-f file | --stdin | -p text] "
                         "[--ids|--pieces|--split] [--no-bos] [--parse-special]\n"
                         "       tok_dump -m model.gguf --piece ID[,ID...]   (id -> type, piece)\n");
            return 2;
        }
    }
    if (!model) {
        std::fprintf(stderr, "tok_dump: -m model.gguf is required\n");
        return 2;
    }

    std::string text;
    if (prompt) text = prompt;
    else if (file) {
        FILE* f = std::fopen(file, "rb");
        if (!f) {
            std::fprintf(stderr, "tok_dump: cannot open %s\n", file);
            return 1;
        }
        read_all(f, text);
        std::fclose(f);
    } else if (use_stdin) {
        read_all(stdin, text);
    }

    std::string err;
    auto gguf = limbo::gguf::File::open(model, &err);
    if (!gguf) {
        std::fprintf(stderr, "tok_dump: %s\n", err.c_str());
        return 1;
    }
    limbo::tok::Vocab vocab;
    if (!limbo::tok::Vocab::load(*gguf, vocab, &err)) {
        std::fprintf(stderr, "tok_dump: %s\n", err.c_str());
        return 1;
    }
    const limbo::tok::Tokenizer tok(vocab);

    if (decode_ids) {
        std::vector<int32_t> ids;
        const std::string list(decode_ids);
        size_t at = 0;
        while (at <= list.size()) {
            const size_t comma = list.find(',', at);
            const std::string one = list.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
            if (!one.empty()) ids.push_back(std::atoi(one.c_str()));
            if (comma == std::string::npos) break;
            at = comma + 1;
        }
        std::fputs(tok.decode(ids).c_str(), stdout);
        return 0;
    }

    if (lookup) {
        const std::string ids(lookup);
        size_t at = 0;
        while (at <= ids.size()) {
            const size_t comma = ids.find(',', at);
            const std::string one = ids.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
            if (!one.empty()) {
                const int id = std::atoi(one.c_str());
                if (id < 0 || size_t(id) >= vocab.tokens.size()) {
                    std::printf("%8d  (no such token)\n", id);
                } else {
                    const int32_t type = vocab.types[size_t(id)];
                    const char* tname = type == limbo::tok::kNormal      ? "normal"
                                        : type == limbo::tok::kUnknown   ? "unknown"
                                        : type == limbo::tok::kControl   ? "control"
                                        : type == limbo::tok::kUserDefined ? "user"
                                        : type == limbo::tok::kUnused    ? "unused"
                                        : type == limbo::tok::kByte      ? "byte"
                                                                         : "undefined";
                    std::printf("%8d  %-9s \"%s\"\n", id, tname,
                                escape(vocab.tokens[size_t(id)]).c_str());
                }
            }
            if (comma == std::string::npos) break;
            at = comma + 1;
        }
        return 0;
    }

    if (split_only) {
        for (const std::string_view piece : tok.split(text)) std::printf("%s\n", escape(piece).c_str());
        return 0;
    }

    limbo::tok::Options o;
    o.add_bos = !no_bos && vocab.add_bos;
    o.parse_special = parse_special;
    std::vector<int32_t> out;
    tok.encode(text, out, o);

    if (pieces) {
        for (const int32_t id : out) {
            const std::string_view piece =
                (id >= 0 && size_t(id) < vocab.tokens.size()) ? vocab.tokens[size_t(id)] : std::string_view("?");
            std::printf("%7d  %-8s %s\n", id, vocab.types[size_t(id)] == limbo::tok::kControl ? "control" : "normal",
                        escape(piece).c_str());
        }
        return 0;
    }
    (void)ids;
    std::printf("[");
    for (size_t i = 0; i < out.size(); ++i) std::printf("%s%d", i ? ", " : "", out[i]);
    std::printf("]\n");
    return 0;
}
