// Copyright (c) 2026 ITERFORM. All rights reserved.
// tests/safetensors_test.cpp — the .safetensors reader: a file written here is read back, and bad headers are refused.
#include <cstdio>
#include <cstring>
#include <fstream>

#include "limbo/check.hpp"
#include "limbo/safetensors.hpp"

using namespace limbo;

static void write_file(const std::string& path, const std::string& header, const std::string& data, uint64_t claimed = ~0ull) {
    std::ofstream f(path, std::ios::binary);
    const uint64_t n = claimed == ~0ull ? header.size() : claimed;
    for (int i = 0; i < 8; ++i) f.put(char((n >> (8 * i)) & 255));
    f << header << data;
}

int main() {
    const auto dir = test::tmp_dir("safetensors");
    const std::string p = (dir / "a.safetensors").string();
    float v[6] = {1, 2, 3, 4, 5, 6};
    const std::string data(reinterpret_cast<const char*>(v), sizeof v);
    const std::string hdr = R"({"__metadata__":{"x":"y"},"w":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}})";
    write_file(p, hdr, data);
    std::string err;
    auto f = st::File::open(p, &err);
    CHECK(f != nullptr, "opens a valid file");
    if (f) {
        CHECK(f->tensors().size() == 1, "one tensor, metadata skipped");
        const st::Tensor* t = f->find("w");
        CHECK(t && t->dtype == "F32" && t->shape.size() == 2 && t->shape[0] == 2 && t->shape[1] == 3 && t->nbytes == 24, "dtype, shape, span");
        CHECK(t && std::memcmp(t->data, v, 24) == 0, "data");
        CHECK(f->find("nope") == nullptr, "missing name");
    }
    const std::string bad_span = R"({"w":{"dtype":"F32","shape":[2,3],"data_offsets":[0,20]}})";
    write_file(p, bad_span, data);
    CHECK(st::File::open(p, &err) == nullptr && err.find("span") != std::string::npos, "span != shape x dtype refused");
    const std::string past = R"({"w":{"dtype":"F32","shape":[2,3],"data_offsets":[8,32]}})";
    write_file(p, past, data);
    CHECK(st::File::open(p, &err) == nullptr && err.find("outside") != std::string::npos, "offsets past the data refused");
    write_file(p, hdr, data, 1u << 20);
    CHECK(st::File::open(p, &err) == nullptr, "oversized header length refused");
    const std::string dt = R"({"w":{"dtype":"Q9","shape":[1],"data_offsets":[0,4]}})";
    write_file(p, dt, data);
    CHECK(st::File::open(p, &err) == nullptr && err.find("dtype") != std::string::npos, "unknown dtype refused");
    CHECK(st::File::open((dir / "missing").string(), &err) == nullptr, "missing file");
    return test::finish("safetensors_test");
}
