// Copyright (c) 2026 ITERFORM. All rights reserved.
// src/gguf.cpp — GGUF v2/v3 reader over one read-only mmap.
#include "limbo/gguf.hpp"

#include <algorithm>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "limbo/types.hpp"

namespace limbo::gguf {

namespace {

constexpr uint32_t kMagic = 0x46554747;  // "GGUF" little-endian

size_t scalar_bytes(VType t) {
    switch (t) {
        case VType::U8: case VType::I8: case VType::BOOL: return 1;
        case VType::U16: case VType::I16: return 2;
        case VType::U32: case VType::I32: case VType::F32: return 4;
        case VType::U64: case VType::I64: case VType::F64: return 8;
        default: return 0;
    }
}

template <typename T> T load(const uint8_t* p) { T v; std::memcpy(&v, p, sizeof v); return v; }

// A bounds-checked cursor; the first failure sticks and records where it happened.
struct Cursor {
    const uint8_t* base;
    uint64_t size;
    uint64_t pos = 0;
    std::string fail;

    bool need(uint64_t n, const char* what) {
        if (!fail.empty()) return false;
        if (n > size - pos) {
            fail = std::string("truncated reading ") + what + " at byte " + std::to_string(pos);
            return false;
        }
        return true;
    }
    template <typename T> bool get(T& out, const char* what) {
        if (!need(sizeof(T), what)) return false;
        out = load<T>(base + pos);
        pos += sizeof(T);
        return true;
    }
    bool str(std::string_view& out, const char* what) {
        uint64_t n = 0;
        if (!get(n, what) || !need(n, what)) return false;
        out = std::string_view(reinterpret_cast<const char*>(base + pos), n);
        pos += n;
        return true;
    }
};

bool read_value(Cursor& c, VType t, Value& v) {
    v.type = t;
    if (t == VType::STRING) {
        std::string_view s;
        if (!c.str(s, "string value")) return false;
        v.strs.push_back(s);
        return true;
    }
    if (t == VType::ARRAY) {
        uint32_t et = 0;
        if (!c.get(et, "array type") || !c.get(v.count, "array count")) return false;
        v.elem = static_cast<VType>(et);
        if (v.elem == VType::STRING) {
            if (v.count > c.size) { c.fail = "string array count past the file"; return false; }
            v.strs.reserve(v.count);
            for (uint64_t i = 0; i < v.count; ++i) {
                std::string_view s;
                if (!c.str(s, "string array element")) return false;
                v.strs.push_back(s);
            }
            return true;
        }
        const size_t eb = scalar_bytes(v.elem);
        if (eb == 0) { c.fail = "array of unsupported element type " + std::to_string(et); return false; }
        if (v.count > (c.size - c.pos) / eb) { c.fail = "numeric array past the file at byte " + std::to_string(c.pos); return false; }
        v.raw = c.base + c.pos;
        c.pos += v.count * eb;
        return true;
    }
    const size_t n = scalar_bytes(t);
    if (n == 0) { c.fail = "unknown value type " + std::to_string(static_cast<uint32_t>(t)); return false; }
    if (!c.need(n, "scalar value")) return false;
    v.raw = c.base + c.pos;
    c.pos += n;
    return true;
}

}  // namespace

std::optional<int64_t> Value::as_int() const {
    if (type == VType::ARRAY || type == VType::STRING || !raw) return std::nullopt;
    switch (type) {
        case VType::U8: return load<uint8_t>(raw);
        case VType::I8: return load<int8_t>(raw);
        case VType::BOOL: return load<uint8_t>(raw) != 0;
        case VType::U16: return load<uint16_t>(raw);
        case VType::I16: return load<int16_t>(raw);
        case VType::U32: return load<uint32_t>(raw);
        case VType::I32: return load<int32_t>(raw);
        case VType::U64: return static_cast<int64_t>(load<uint64_t>(raw));
        case VType::I64: return load<int64_t>(raw);
        default: return std::nullopt;
    }
}

std::optional<double> Value::as_float() const {
    if (type == VType::F32 && raw) return load<float>(raw);
    if (type == VType::F64 && raw) return load<double>(raw);
    if (auto i = as_int()) return static_cast<double>(*i);
    return std::nullopt;
}

std::optional<std::string_view> Value::as_str() const {
    if (type == VType::STRING && !strs.empty()) return strs[0];
    return std::nullopt;
}

std::unique_ptr<File> File::open(const std::string& path, std::string* err) {
    std::unique_ptr<File> f(new File());
    f->path_ = path;
    f->fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (f->fd_ < 0) { if (err) *err = path + ": " + std::strerror(errno); return nullptr; }
    struct stat st {};
    if (::fstat(f->fd_, &st) != 0 || st.st_size < 24) {
        if (err) *err = path + ": not a GGUF file (too small)";
        return nullptr;
    }
    f->size_ = static_cast<uint64_t>(st.st_size);
    void* m = ::mmap(nullptr, f->size_, PROT_READ, MAP_SHARED, f->fd_, 0);
    if (m == MAP_FAILED) { if (err) *err = path + ": mmap: " + std::strerror(errno); return nullptr; }
    f->base_ = static_cast<const uint8_t*>(m);
    std::string why;
    if (!f->parse(&why)) { if (err) *err = path + ": " + why; return nullptr; }
    return f;
}

void File::drop_pages() const {
    if (base_) ::madvise(const_cast<uint8_t*>(base_), size_, MADV_DONTNEED);
}

File::~File() {
    if (base_) ::munmap(const_cast<uint8_t*>(base_), size_);
    if (fd_ >= 0) ::close(fd_);
}

bool File::parse(std::string* err) {
    Cursor c;
    c.base = base_;
    c.size = size_;
    uint32_t magic = 0;
    uint64_t n_tensors = 0, n_kv = 0;
    if (!c.get(magic, "magic") || magic != kMagic) { *err = "bad magic"; return false; }
    if (!c.get(version_, "version") || (version_ != 2 && version_ != 3)) {
        *err = "unsupported GGUF version " + std::to_string(version_);
        return false;
    }
    if (!c.get(n_tensors, "tensor count") || !c.get(n_kv, "kv count")) { *err = c.fail; return false; }
    if (n_kv > size_ || n_tensors > size_) { *err = "header counts past the file"; return false; }

    kv_.reserve(n_kv);
    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string_view key;
        uint32_t t = 0;
        Value v;
        if (!c.str(key, "key") || !c.get(t, "value type") || !read_value(c, static_cast<VType>(t), v)) {
            *err = c.fail.empty() ? "bad metadata" : c.fail;
            return false;
        }
        kv_.emplace_back(key, std::move(v));
    }
    if (auto a = get_int("general.alignment")) {
        if (*a <= 0 || (*a & (*a - 1)) != 0) { *err = "general.alignment is not a power of two"; return false; }
        alignment_ = static_cast<uint64_t>(*a);
    }

    tensors_.reserve(n_tensors);
    for (uint64_t i = 0; i < n_tensors; ++i) {
        Tensor t;
        if (!c.str(t.name, "tensor name") || !c.get(t.n_dims, "tensor dims")) { *err = c.fail; return false; }
        if (t.n_dims == 0 || t.n_dims > 4) { *err = "tensor " + std::string(t.name) + " has " + std::to_string(t.n_dims) + " dims"; return false; }
        for (uint32_t d = 0; d < t.n_dims; ++d) {
            uint64_t ne = 0;
            if (!c.get(ne, "tensor shape")) { *err = c.fail; return false; }
            if (ne == 0 || ne > (1ull << 40)) { *err = "tensor " + std::string(t.name) + " has a bad dimension"; return false; }
            t.ne[d] = static_cast<int64_t>(ne);
        }
        if (!c.get(t.type, "tensor type") || !c.get(t.offset, "tensor offset")) { *err = c.fail; return false; }
        if (!type_info(t.type)) { *err = "tensor " + std::string(t.name) + " has unknown type id " + std::to_string(t.type); return false; }
        t.nbytes = type_bytes(t.type, static_cast<uint64_t>(t.n_elements()));
        if (t.nbytes == 0) { *err = "tensor " + std::string(t.name) + ": row length is not a whole number of blocks"; return false; }
        if (t.offset % alignment_ != 0) { *err = "tensor " + std::string(t.name) + " is not aligned"; return false; }
        tensors_.push_back(t);
    }
    data_offset_ = (c.pos + alignment_ - 1) / alignment_ * alignment_;
    if (data_offset_ > size_) { *err = "data section starts past the file"; return false; }

    // Spans must fit the file and must not overlap: a type id misread (old 42/43) shows up here.
    std::vector<const Tensor*> by_off;
    by_off.reserve(tensors_.size());
    for (auto& t : tensors_) {
        if (t.offset > size_ - data_offset_ || t.nbytes > size_ - data_offset_ - t.offset) {
            *err = "tensor " + std::string(t.name) + " runs past the end of the file";
            return false;
        }
        t.data = base_ + data_offset_ + t.offset;
        by_off.push_back(&t);
    }
    std::sort(by_off.begin(), by_off.end(), [](const Tensor* a, const Tensor* b) { return a->offset < b->offset; });
    for (size_t i = 1; i < by_off.size(); ++i) {
        if (by_off[i - 1]->offset + by_off[i - 1]->nbytes > by_off[i]->offset) {
            *err = "tensor " + std::string(by_off[i - 1]->name) + " overlaps " + std::string(by_off[i]->name) +
                   " (its type's block size does not match the file)";
            return false;
        }
    }
    return true;
}

const Value* File::find(std::string_view key) const {
    for (const auto& [k, v] : kv_)
        if (k == key) return &v;
    return nullptr;
}

std::optional<int64_t> File::get_int(std::string_view key) const {
    const Value* v = find(key);
    return v ? v->as_int() : std::nullopt;
}

std::optional<double> File::get_float(std::string_view key) const {
    const Value* v = find(key);
    return v ? v->as_float() : std::nullopt;
}

std::optional<std::string_view> File::get_str(std::string_view key) const {
    const Value* v = find(key);
    return v ? v->as_str() : std::nullopt;
}

std::string_view File::arch() const { return get_str("general.architecture").value_or(""); }

namespace {
template <typename T>
std::vector<T> typed_array(const Value* v, VType want) {
    std::vector<T> out;
    if (!v || v->type != VType::ARRAY || v->elem != want || !v->raw) return out;
    out.resize(v->count);
    std::memcpy(out.data(), v->raw, v->count * sizeof(T));
    return out;
}
}  // namespace

std::vector<int32_t> File::get_i32_array(std::string_view key) const { return typed_array<int32_t>(find(key), VType::I32); }
std::vector<uint32_t> File::get_u32_array(std::string_view key) const { return typed_array<uint32_t>(find(key), VType::U32); }
std::vector<float> File::get_f32_array(std::string_view key) const { return typed_array<float>(find(key), VType::F32); }

const std::vector<std::string_view>* File::get_str_array(std::string_view key) const {
    const Value* v = find(key);
    if (!v || v->type != VType::ARRAY || v->elem != VType::STRING) return nullptr;
    return &v->strs;
}

const Tensor* File::tensor(std::string_view name) const {
    for (const auto& t : tensors_)
        if (t.name == name) return &t;
    return nullptr;
}

}  // namespace limbo::gguf
