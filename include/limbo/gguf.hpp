// Copyright (c) 2026 ITERFORM. All rights reserved.
#pragma once
// limbo/gguf.hpp — a read-only GGUF file: metadata and tensor table over one mmap.
//
// Zero-copy: names, strings and tensor data are views into the mapping and live as long as the
// File. Every offset and length is checked against the file size at open; a tensor's byte span
// is derived from its type and shape and must fit before the next tensor and inside the file.
// One shard per File; a split model is several Files.

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace limbo::gguf {

enum class VType : uint32_t {
    U8 = 0, I8 = 1, U16 = 2, I16 = 3, U32 = 4, I32 = 5, F32 = 6, BOOL = 7,
    STRING = 8, ARRAY = 9, U64 = 10, I64 = 11, F64 = 12,
};

struct Value {
    VType type = VType::U8;
    VType elem = VType::U8;      // element type when type == ARRAY
    uint64_t count = 0;          // element count when type == ARRAY
    const uint8_t* raw = nullptr;  // scalar or first array element, inside the mapping
    std::vector<std::string_view> strs;  // filled for STRING scalars (one) and STRING arrays

    std::optional<int64_t> as_int() const;     // any integer or bool scalar
    std::optional<double> as_float() const;    // any numeric scalar
    std::optional<std::string_view> as_str() const;
};

struct Tensor {
    std::string_view name;
    uint32_t type = 0;           // limbo::Type id
    uint32_t n_dims = 0;
    int64_t ne[4] = {1, 1, 1, 1};  // ne[0] is the contiguous (row) dimension
    uint64_t offset = 0;         // relative to the data section
    uint64_t nbytes = 0;         // derived from type and shape
    const uint8_t* data = nullptr;
    int64_t n_elements() const { return ne[0] * ne[1] * ne[2] * ne[3]; }
};

class File {
public:
    // nullptr on failure; *err names what and where (byte offset).
    static std::unique_ptr<File> open(const std::string& path, std::string* err);
    ~File();
    File(const File&) = delete;
    File& operator=(const File&) = delete;

    const std::string& path() const { return path_; }
    uint32_t version() const { return version_; }
    uint64_t size() const { return size_; }
    uint64_t alignment() const { return alignment_; }
    uint64_t data_offset() const { return data_offset_; }
    const uint8_t* base() const { return base_; }
    // Hands this process's resident pages of the mapping back to the page cache (madvise DONTNEED): the data stays
    // valid and is faulted in again on the next touch. Keeps a process's RSS small over a file bigger than RAM.
    void drop_pages() const;

    const std::vector<std::pair<std::string_view, Value>>& kv() const { return kv_; }
    const Value* find(std::string_view key) const;
    std::optional<int64_t> get_int(std::string_view key) const;
    std::optional<double> get_float(std::string_view key) const;
    std::optional<std::string_view> get_str(std::string_view key) const;
    std::string_view arch() const;   // general.architecture, "" if absent

    // Typed copies of numeric arrays (empty if the key is missing or of another element type).
    std::vector<int32_t> get_i32_array(std::string_view key) const;
    std::vector<uint32_t> get_u32_array(std::string_view key) const;
    std::vector<float> get_f32_array(std::string_view key) const;
    const std::vector<std::string_view>* get_str_array(std::string_view key) const;

    const std::vector<Tensor>& tensors() const { return tensors_; }
    const Tensor* tensor(std::string_view name) const;

private:
    File() = default;
    bool parse(std::string* err);

    std::string path_;
    int fd_ = -1;
    const uint8_t* base_ = nullptr;
    uint64_t size_ = 0;
    uint32_t version_ = 0;
    uint64_t alignment_ = 32;
    uint64_t data_offset_ = 0;
    std::vector<std::pair<std::string_view, Value>> kv_;
    std::vector<Tensor> tensors_;
};

}  // namespace limbo::gguf
