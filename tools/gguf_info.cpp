// Copyright (c) 2026 ITERFORM. All rights reserved.
// tools/gguf_info.cpp — header facts, metadata and the tensor type histogram of a GGUF file.
//
//   gguf_info <file.gguf> [key-filter]
//
// The first block is the same set of facts `llama-gguf <file> r` prints (version, alignment, data
// offset, kv count, tensor count), so the two can be diffed line by line: that diff is the W0 gate
// "the tensor tables of all three ladder files agree with llama.cpp's".
#include "limbo/gguf.hpp"
#include "limbo/types.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>

namespace {

void print_value(const limbo::gguf::Value& v) {
    switch (v.type) {
        case limbo::gguf::VType::STRING:
            std::printf("= \"%.*s\"", (int)v.strs[0].size(), v.strs[0].data());
            return;
        case limbo::gguf::VType::ARRAY: {
            const char* en = "?";
            switch (v.elem) {
                case limbo::gguf::VType::U8: en = "u8"; break;
                case limbo::gguf::VType::I8: en = "i8"; break;
                case limbo::gguf::VType::U16: en = "u16"; break;
                case limbo::gguf::VType::I16: en = "i16"; break;
                case limbo::gguf::VType::U32: en = "u32"; break;
                case limbo::gguf::VType::I32: en = "i32"; break;
                case limbo::gguf::VType::F32: en = "f32"; break;
                case limbo::gguf::VType::BOOL: en = "bool"; break;
                case limbo::gguf::VType::STRING: en = "string"; break;
                case limbo::gguf::VType::U64: en = "u64"; break;
                case limbo::gguf::VType::I64: en = "i64"; break;
                case limbo::gguf::VType::F64: en = "f64"; break;
                default: break;
            }
            std::printf("= array[%s] x %llu", en, (unsigned long long)v.count);
            if (v.elem == limbo::gguf::VType::STRING && !v.strs.empty()) {
                std::printf("  first: \"%.*s\"", (int)v.strs[0].size(), v.strs[0].data());
                if (v.strs.size() > 1) std::printf(", \"%.*s\"", (int)v.strs[1].size(), v.strs[1].data());
            } else if (v.elem == limbo::gguf::VType::I32 && v.raw && v.count) {
                const auto* p = reinterpret_cast<const int32_t*>(v.raw);
                if (v.count <= 16) {
                    std::printf("  all:");
                    for (uint64_t i = 0; i < v.count; ++i) std::printf(" %d", p[i]);
                } else {
                    std::printf("  first: %d", p[0]);
                    if (v.count > 1) std::printf(", %d", p[1]);
                }
            } else if (v.elem == limbo::gguf::VType::F32 && v.raw && v.count) {
                const auto* p = reinterpret_cast<const float*>(v.raw);
                std::printf("  first: %g", (double)p[0]);
                if (v.count > 1) std::printf(", %g", (double)p[1]);
            }
            return;
        }
        default: break;
    }
    if (auto i = v.as_int()) std::printf("= %lld", (long long)*i);
    else if (auto d = v.as_float()) std::printf("= %g", *d);
    else std::printf("= ?");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: gguf_info <file.gguf> [key-substring | --tensor NAME]\n");
        return 2;
    }
    const bool want_tensor = argc > 2 && std::strcmp(argv[2], "--tensor") == 0;
    const char* tensor_name = want_tensor && argc > 3 ? argv[3] : nullptr;
    const bool want_type = argc > 2 && std::strcmp(argv[2], "--type") == 0;
    const int type_id = want_type && argc > 3 ? std::atoi(argv[3]) : -1;
    const bool want_roles = argc > 2 && std::strcmp(argv[2], "--roles") == 0;
    if (want_tensor && !tensor_name) {
        std::fprintf(stderr, "gguf_info: --tensor needs a name\n");
        return 2;
    }
    const char* filter = (argc > 2 && !want_tensor) ? argv[2] : nullptr;
    std::string err;
    auto f = limbo::gguf::File::open(argv[1], &err);
    if (!f) {
        std::fprintf(stderr, "gguf_info: %s\n", err.c_str());
        return 1;
    }
    const auto& a = f->arch();
    if (want_roles) {
        // The role set of an architecture, taken from the file rather than from a list someone typed:
        // "blk.<N>.<role>" collapses to <role>, everything else keeps its name.
        std::map<std::string, std::pair<uint64_t, uint64_t>> roles;  // role -> (tensors, bytes)
        for (const auto& t : f->tensors()) {
            std::string name(t.name);
            std::string role = name;
            if (name.rfind("blk.", 0) == 0) {
                const size_t dot = name.find('.', 4);
                if (dot != std::string::npos) role = name.substr(dot + 1);
            }
            auto& r = roles[role];
            r.first += 1;
            r.second += t.nbytes;
        }
        std::printf("%zu roles in %s:\n", roles.size(), f->path().c_str());
        for (const auto& [role, r] : roles)
            std::printf("  %-40s %5llu tensors %10.2f MiB\n", role.c_str(), (unsigned long long)r.first,
                        double(r.second) / 1048576.0);
        return 0;
    }
    if (want_type) {
        const limbo::TypeInfo* ti = limbo::type_info(uint32_t(type_id));
        std::printf("type %d (%s):\n", type_id, ti ? ti->name.data() : "?");
        for (const auto& t : f->tensors()) {
            if (int(t.type) != type_id) continue;
            std::printf("  %-44.*s shape", (int)t.name.size(), t.name.data());
            for (uint32_t d = t.n_dims; d > 0; --d) std::printf(" %lld", (long long)t.ne[d - 1]);
            std::printf("  %.2f MiB\n", double(t.nbytes) / 1048576.0);
        }
        return 0;
    }
    if (want_tensor) {
        const limbo::gguf::Tensor* t = f->tensor(tensor_name);
        if (!t) {
            std::printf("%s: no such tensor\n", tensor_name);
            return 1;
        }
        const limbo::TypeInfo* ti = limbo::type_info(t->type);
        std::printf("tensor:      %.*s\n", (int)t->name.size(), t->name.data());
        std::printf("type:        %s (%u)\n", ti ? ti->name.data() : "?", t->type);
        std::printf("n_dims:      %u\n", t->n_dims);
        std::printf("shape:      ");
        for (uint32_t d = t->n_dims; d > 0; --d) std::printf(" %lld", (long long)t->ne[d - 1]);
        std::printf("\n");
        std::printf("n_elements:  %lld\n", (long long)t->n_elements());
        std::printf("nbytes:      %llu\n", (unsigned long long)t->nbytes);
        std::printf("offset:      %llu\n", (unsigned long long)t->offset);
        return 0;
    }
    std::printf("path:        %s\n", f->path().c_str());
    std::printf("version:     %u\n", f->version());
    std::printf("alignment:   %llu\n", (unsigned long long)f->alignment());
    std::printf("data offset: %llu\n", (unsigned long long)f->data_offset());
    std::printf("file size:   %llu\n", (unsigned long long)f->size());
    std::printf("kv count:    %zu\n", f->kv().size());
    std::printf("n_tensors:   %zu\n", f->tensors().size());
    std::printf("arch:        %.*s\n", (int)a.size(), a.data());

    if (filter) {
        std::printf("\n-- metadata keys containing \"%s\" --\n", filter);
        for (const auto& [k, v] : f->kv()) {
            if (k.find(filter) == std::string_view::npos) continue;
            std::printf("  %-42.*s ", (int)k.size(), k.data());
            print_value(v);
            std::printf("\n");
        }
    } else {
        std::printf("\n-- metadata --\n");
        for (const auto& [k, v] : f->kv()) {
            std::printf("  %-42.*s ", (int)k.size(), k.data());
            print_value(v);
            std::printf("\n");
        }
    }

    std::printf("\n-- tensor types --\n");
    std::map<uint32_t, std::pair<uint64_t, uint64_t>> hist;
    uint64_t sum = 0;
    for (const auto& t : f->tensors()) {
        auto& h = hist[t.type];
        h.first += 1;
        h.second += t.nbytes;
        sum += t.nbytes;
    }
    for (const auto& [id, h] : hist) {
        const limbo::TypeInfo* ti = limbo::type_info(id);
        std::printf("  %-8s %6llu tensors %10.2f MiB\n", ti ? ti->name.data() : "?", (unsigned long long)h.first,
                    (double)h.second / 1048576.0);
    }
    const uint64_t data = f->size() - f->data_offset();
    std::printf("  %-8s %6zu tensors %10.2f MiB  coverage %.6f of the data section\n", "total", f->tensors().size(),
                (double)sum / 1048576.0, data ? (double)sum / (double)data : 0.0);
    return 0;
}
