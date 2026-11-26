// Copyright (c) 2026 ITERFORM. All rights reserved.
// tests/gguf_test.cpp — the GGUF reader: one synthetic file it must accept, and every way it must refuse.
//
// The synthetic file is written byte by byte here, so each refusal case is a real file on disk rather
// than a patched object. With LIMBO_TEST_GGUF set (colon-separated paths) the real ladder files are
// opened too: every span must fit the mapping and the spans must cover the data section, which is what
// proves the type table's block sizes against files llama.cpp wrote.
#include "limbo/check.hpp"
#include "limbo/gguf.hpp"
#include "limbo/types.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace {

using Bytes = std::vector<uint8_t>;

constexpr uint32_t kMagic = 0x46554747;  // "GGUF"
constexpr uint64_t kSlab = 512;          // data-section bytes in every synthetic file

// Little-endian writer, the shape GGUF v2/v3 uses.
struct W {
    Bytes b;
    void raw(const void* p, size_t n) {
        const uint8_t* q = static_cast<const uint8_t*>(p);
        b.insert(b.end(), q, q + n);
    }
    void u32(uint32_t v) { raw(&v, 4); }
    void u64(uint64_t v) { raw(&v, 8); }
    void str(std::string_view s) {
        u64(s.size());
        raw(s.data(), s.size());
    }
    void pad(uint64_t a) {
        while (a && b.size() % a) b.push_back(0);
    }
};

struct Opts {
    uint32_t version = 3;
    bool alignment_key = true;
    uint32_t alignment = 32;
    uint32_t t0_type = 0;  // F32
    uint64_t t0_ne0 = 4;   // row length
    uint32_t t1_type = 2;  // Q4_0
    uint64_t t1_offset = 32;
    uint64_t t1_ne0 = 64;
    bool second_tensor = true;
    size_t truncate_to = 0;
};

Bytes build(const Opts& o) {
    W w;
    w.u32(kMagic);
    w.u32(o.version);
    w.u64(o.second_tensor ? 2 : 1);  // n_tensors
    w.u64(o.alignment_key ? 3 : 2);  // n_kv
    w.str("general.architecture");
    w.u32(8);  // STRING
    w.str("qwen35moe");
    if (o.alignment_key) {
        w.str("general.alignment");
        w.u32(4);  // U32
        w.u32(o.alignment);
    }
    w.str("tokenizer.ggml.tokens");
    w.u32(9);  // ARRAY
    w.u32(8);  // of STRING
    w.u64(2);
    w.str("<s>");
    w.str("hello");

    w.str("blk.0.attn_q.weight");
    w.u32(2);         // n_dims
    w.u64(o.t0_ne0);  // ne[0]
    w.u64(2);         // ne[1]
    w.u32(o.t0_type);
    w.u64(0);         // offset 0

    if (o.second_tensor) {
        w.str("blk.0.ffn_down_exps.weight");
        w.u32(2);
        w.u64(o.t1_ne0);
        w.u64(2);
        w.u32(o.t1_type);
        w.u64(o.t1_offset);
    }
    w.pad(o.alignment_key ? o.alignment : 32);

    Bytes out = std::move(w.b);
    const size_t data = out.size();
    out.resize(data + kSlab);
    for (size_t i = 0; i < kSlab; ++i) out[data + i] = static_cast<uint8_t>((i * 7 + 1) & 0xff);
    if (o.truncate_to) out.resize(o.truncate_to);
    return out;
}

// Writes the buffer under build/tmp/gguf and returns the path.
std::string put(const std::string& name, const Bytes& b) {
    const std::filesystem::path p = limbo::test::tmp_dir("gguf") / name;
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
    f.close();
    return p.string();
}

// Opens a buffer and demands failure, with `want` somewhere in the message when it is not empty.
void must_refuse(const std::string& name, const Bytes& b, const char* want) {
    const std::string path = put(name, b);
    std::string err;
    auto f = limbo::gguf::File::open(path, &err);
    if (f) {
        std::printf("FAIL %s: opened, but it must be refused\n", name.c_str());
        ++limbo::test::failures();
        return;
    }
    if (err.empty()) {
        std::printf("FAIL %s: refused without a reason\n", name.c_str());
        ++limbo::test::failures();
        return;
    }
    if (*want && err.find(want) == std::string::npos) {
        std::printf("FAIL %s: refused with \"%s\", wanted \"%s\"\n", name.c_str(), err.c_str(), want);
        ++limbo::test::failures();
        return;
    }
    std::printf("ok   %-14s refused: %s\n", name.c_str(), err.c_str());
}

}  // namespace

int main() {
    // ---- the file it must accept -------------------------------------------------------------
    const std::string good = put("good.gguf", build(Opts{}));
    std::string err;
    auto f = limbo::gguf::File::open(good, &err);
    CHECK(f != nullptr, "a well-formed file opens");
    if (!f) {
        std::printf("     %s\n", err.c_str());
        return limbo::test::finish("gguf");
    }
    CHECK(f->version() == 3, "version 3");
    CHECK(f->arch() == "qwen35moe", "general.architecture reads back");
    CHECK(f->alignment() == 32, "general.alignment reads back");
    CHECK(f->data_offset() % 32 == 0, "the data section is aligned");
    CHECK(f->tensors().size() == 2, "two tensors");

    const limbo::gguf::Tensor* t0 = f->tensor("blk.0.attn_q.weight");
    CHECK(t0 != nullptr, "tensor 0 is found by name");
    if (t0) {
        CHECK(t0->n_elements() == 8, "tensor 0 has 8 elements");
        CHECK(t0->nbytes == 32, "tensor 0 is 32 bytes of F32");
        CHECK(t0->ne[0] == 4 && t0->ne[1] == 2, "tensor 0 shape is 4x2");
        CHECK(t0->data == f->base() + f->data_offset(), "tensor 0 points into the mapping");
        if (t0->data) CHECK(t0->data[0] == 1, "tensor 0 data is the written pattern");
    }
    const limbo::gguf::Tensor* t1 = f->tensor("blk.0.ffn_down_exps.weight");
    CHECK(t1 != nullptr, "tensor 1 is found by name");
    if (t1) {
        CHECK(t1->nbytes == 72, "tensor 1 is 72 bytes of Q4_0");
        CHECK(t1->type == 2, "tensor 1 carries its type id");
        CHECK(t1->data == f->base() + f->data_offset() + 32, "tensor 1 sits at its offset");
        if (t1->data) CHECK(t1->data[0] == static_cast<uint8_t>((32 * 7 + 1) & 0xff), "tensor 1 data matches");
    }
    CHECK(f->find("general.alignment") != nullptr, "kv lookup finds a key");
    CHECK(f->get_int("general.alignment") == 32, "get_int on a u32");
    CHECK(f->get_int("no.such.key") == std::nullopt, "a missing key is nullopt");
    CHECK(f->get_str("general.architecture") == std::optional<std::string_view>("qwen35moe"), "get_str on a string");
    const auto* toks = f->get_str_array("tokenizer.ggml.tokens");
    CHECK(toks != nullptr && toks->size() == 2, "a string array reads back");
    if (toks && toks->size() == 2) CHECK((*toks)[1] == "hello", "string array element 1");
    CHECK(f->get_i32_array("general.alignment").empty(), "a non-array key yields no array");

    // ---- the files it must refuse ------------------------------------------------------------
    { Opts o; o.version = 1; must_refuse("v1.gguf", build(o), "unsupported GGUF version"); }
    { Opts o; o.alignment = 24; must_refuse("align24.gguf", build(o), "not a power of two"); }
    { Opts o; o.t0_type = 999; must_refuse("badtype.gguf", build(o), "unknown type id"); }
    { Opts o; o.t0_type = 2; o.t0_ne0 = 30; must_refuse("partial.gguf", build(o), "whole number of blocks"); }
    { Opts o; o.alignment = 16; o.t1_offset = 16; must_refuse("overlap.gguf", build(o), "overlaps"); }
    { Opts o; o.t1_offset = 1u << 20; must_refuse("past.gguf", build(o), "past the end of the file"); }
    { Opts o; o.truncate_to = 20; must_refuse("tiny.gguf", build(o), "too small"); }
    { Opts o; o.truncate_to = 90; must_refuse("cut.gguf", build(o), ""); }
    {
        Opts o;
        o.second_tensor = false;
        Bytes b = build(o);
        b[0] = 0x00;
        must_refuse("magic.gguf", b, "bad magic");
    }

    // ---- the real ladder files ---------------------------------------------------------------
    const char* real = std::getenv("LIMBO_TEST_GGUF");
    if (real && *real) {
        const std::string list(real);
        size_t at = 0;
        while (at <= list.size()) {
            const size_t colon = list.find(':', at);
            const std::string path = list.substr(at, colon == std::string::npos ? std::string::npos : colon - at);
            if (!path.empty()) {
                std::string rerr;
                auto g = limbo::gguf::File::open(path, &rerr);
                if (!g) {
                    std::printf("FAIL %s: %s\n", path.c_str(), rerr.c_str());
                    ++limbo::test::failures();
                } else {
                    uint64_t sum = 0;
                    std::map<uint32_t, std::pair<uint64_t, uint64_t>> hist;  // type id -> (tensors, bytes)
                    for (const auto& t : g->tensors()) {
                        sum += t.nbytes;
                        auto& h = hist[t.type];
                        h.first += 1;
                        h.second += t.nbytes;
                    }
                    const uint64_t data = g->size() - g->data_offset();
                    const double cover = data ? static_cast<double>(sum) / static_cast<double>(data) : 0.0;
                    std::printf("     %s\n       %.*s: %llu tensors, %.2f GiB file, data %.2f GiB, coverage %.4f\n",
                                path.c_str(), (int)g->arch().size(), g->arch().data(),
                                (unsigned long long)g->tensors().size(),
                                static_cast<double>(g->size()) / 1073741824.0,
                                static_cast<double>(data) / 1073741824.0, cover);
                    for (const auto& [id, h] : hist) {
                        const limbo::TypeInfo* ti = limbo::type_info(id);
                        std::printf("         %-8s %6llu tensors %9.1f MiB\n", ti ? ti->name.data() : "?",
                                    (unsigned long long)h.first, static_cast<double>(h.second) / 1048576.0);
                    }
                    CHECK(!g->tensors().empty(), "the real file has tensors");
                    // A continuation shard carries no architecture and no tokenizer: it is one
                    // continuation of a split model, identified by split.no/split.count, and only the
                    // primary shard describes the model. Rung 3's shard 2 is exactly this case (one
                    // tensor, the PLE table). Joining them is the geometry layer's job, not the
                    // reader's, but the contract is worth asserting here.
                    const auto split_no = g->get_int("split.no");
                    const auto split_count = g->get_int("split.count");
                    const bool continuation = split_no && split_count && *split_count > 1 && *split_no > 0;
                    if (continuation) {
                        std::printf("         (continuation shard %lld of %lld: no architecture key, by design)\n",
                                    (long long)*split_no, (long long)*split_count);
                        CHECK(g->find("split.tensors.count") != nullptr,
                              "a continuation shard carries split.tensors.count");
                    } else {
                        CHECK(!g->arch().empty(), "the real file names its architecture");
                    }
                    CHECK(cover > 0.95, "the tensor spans cover the data section (the type table is right)");
                }
            }
            if (colon == std::string::npos) break;
            at = colon + 1;
        }
    } else {
        std::printf("--   LIMBO_TEST_GGUF not set: the real ladder files are skipped\n");
    }

    return limbo::test::finish("gguf");
}
