// Copyright (c) 2026 ITERFORM. All rights reserved.
#pragma once
// limbo/safetensors.hpp — a read-only .safetensors file over one mmap: an 8-byte little-endian header length, a JSON header
// (name -> dtype, shape, data_offsets), then the raw tensors. Zero-copy: tensor data are views into the mapping and live as long
// as the File. Every offset is checked against the file size at open and each span must equal shape x element size.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace limbo::st {

struct Tensor {
    std::string name;
    std::string dtype;           // "BF16", "F16", "F32", ...
    std::vector<int64_t> shape;  // row-major, slowest dimension first
    uint64_t nbytes = 0;
    const uint8_t* data = nullptr;
    int64_t n_elements() const {
        int64_t n = 1;
        for (int64_t d : shape) n *= d;
        return n;
    }
};

// Bytes of one element of a dtype name; 0 for one this reader does not know.
size_t dtype_bytes(const std::string& dtype);

class File {
public:
    // nullptr on failure; *err names what and where.
    static std::unique_ptr<File> open(const std::string& path, std::string* err);
    ~File();
    File(const File&) = delete;
    File& operator=(const File&) = delete;

    const std::vector<Tensor>& tensors() const { return tensors_; }
    const Tensor* find(const std::string& name) const;
    uint64_t size() const { return size_; }

private:
    File() = default;
    int fd_ = -1;
    const uint8_t* base_ = nullptr;
    uint64_t size_ = 0;
    std::vector<Tensor> tensors_;
};

}  // namespace limbo::st
