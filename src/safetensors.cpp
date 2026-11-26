// Copyright (c) 2026 ITERFORM. All rights reserved.
// src/safetensors.cpp — the .safetensors reader (include/limbo/safetensors.hpp).
#include "limbo/safetensors.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

#include "limbo/jinja.hpp"

namespace limbo::st {

size_t dtype_bytes(const std::string& d) {
    if (d == "F64" || d == "I64" || d == "U64") return 8;
    if (d == "F32" || d == "I32" || d == "U32") return 4;
    if (d == "F16" || d == "BF16" || d == "I16" || d == "U16") return 2;
    if (d == "I8" || d == "U8" || d == "BOOL" || d == "F8_E4M3" || d == "F8_E5M2") return 1;
    return 0;
}

std::unique_ptr<File> File::open(const std::string& path, std::string* err) {
    auto fail = [&](const std::string& m) -> std::unique_ptr<File> {
        if (err) *err = path + ": " + m;
        return nullptr;
    };
    std::unique_ptr<File> f(new File);
    f->fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (f->fd_ < 0) return fail(std::string("open: ") + std::strerror(errno));
    struct stat st {};
    if (::fstat(f->fd_, &st) != 0 || st.st_size < 8) return fail("not a safetensors file (shorter than 8 bytes)");
    f->size_ = uint64_t(st.st_size);
    void* m = ::mmap(nullptr, f->size_, PROT_READ, MAP_SHARED, f->fd_, 0);
    if (m == MAP_FAILED) return fail(std::string("mmap: ") + std::strerror(errno));
    f->base_ = static_cast<const uint8_t*>(m);
    uint64_t hn = 0;
    for (int i = 7; i >= 0; --i) hn = hn << 8 | f->base_[i];
    if (hn > f->size_ - 8 || hn > (1ull << 28)) return fail("header length " + std::to_string(hn) + " does not fit the file");
    const uint64_t data0 = 8 + hn, data_n = f->size_ - data0;
    jinja::Value doc;
    std::string jerr;
    if (!jinja::json_parse(std::string_view(reinterpret_cast<const char*>(f->base_ + 8), size_t(hn)), &doc, &jerr)) return fail("header: " + jerr);
    if (!doc.is_dict()) return fail("header is not an object");
    for (const auto& kv : doc.dict()) {
        if (kv.first == "__metadata__") continue;
        const jinja::Value& v = kv.second;
        const jinja::Value* dt = v.find("dtype");
        const jinja::Value* sh = v.find("shape");
        const jinja::Value* of = v.find("data_offsets");
        if (!dt || !dt->is_string() || !sh || !sh->is_list() || !of || !of->is_list() || of->list().size() != 2)
            return fail("tensor " + kv.first + ": malformed entry");
        Tensor t;
        t.name = kv.first;
        t.dtype = dt->s();
        for (const jinja::Value& d : sh->list()) {
            if (!d.is_int() || d.i() < 0) return fail("tensor " + kv.first + ": bad shape");
            t.shape.push_back(d.i());
        }
        const jinja::Value &b = of->list()[0], &e = of->list()[1];
        if (!b.is_int() || !e.is_int() || b.i() < 0 || e.i() < b.i() || uint64_t(e.i()) > data_n)
            return fail("tensor " + kv.first + ": data_offsets outside the file");
        const size_t eb = dtype_bytes(t.dtype);
        if (eb == 0) return fail("tensor " + kv.first + ": unknown dtype " + t.dtype);
        t.nbytes = uint64_t(e.i() - b.i());
        double expect = double(eb);
        for (int64_t d : t.shape) expect *= double(d);
        if (expect != double(t.nbytes)) return fail("tensor " + kv.first + ": span does not match shape x dtype");
        t.data = f->base_ + data0 + uint64_t(b.i());
        f->tensors_.push_back(std::move(t));
    }
    return f;
}

File::~File() {
    if (base_) ::munmap(const_cast<uint8_t*>(base_), size_);
    if (fd_ >= 0) ::close(fd_);
}

const Tensor* File::find(const std::string& name) const {
    for (const Tensor& t : tensors_)
        if (t.name == name) return &t;
    return nullptr;
}

}  // namespace limbo::st
