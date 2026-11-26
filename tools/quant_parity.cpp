// Copyright (c) 2026 ITERFORM. All rights reserved.
// tools/quant_parity.cpp — Limbo's weight decoder against ggml's own, on real tensors.
//
// This links the operator's libggml-base (the oracle) as well as liblimbo, and is not part of the
// engine: it exists so that every format's decode is compared, value by value, against the reference
// implementation on the actual bytes of the actual ladder files. A decoder that has only been eyeballed
// is a decoder nobody should trust, and a wrong weight decode is invisible until the tokens differ.
//
//   quant_parity <file.gguf> [more.gguf ...]
//
// For each supported tensor it decodes the first row with both implementations and reports the largest
// absolute difference. Anything above zero is a failure.
#include "limbo/gguf.hpp"
#include "limbo/quants.hpp"
#include "limbo/types.hpp"

#include <ggml.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

struct Stat {
    uint64_t tensors = 0;
    uint64_t values = 0;
    double max_abs = 0;
    std::string worst_tensor;
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: quant_parity <file.gguf> [more.gguf ...]\n");
        return 2;
    }

    std::map<uint32_t, Stat> per_type;
    int failures = 0;

    for (int a = 1; a < argc; ++a) {
        std::string err;
        auto f = limbo::gguf::File::open(argv[a], &err);
        if (!f) {
            std::fprintf(stderr, "quant_parity: %s\n", err.c_str());
            ++failures;
            continue;
        }
        std::printf("-- %s\n", argv[a]);
        // One result slot per tensor, filled in parallel (the tensors are independent and both decoders are
        // pure functions), then reported in file order so the output does not depend on the schedule.
        struct Res {
            bool supported = false, skip = false, decode_fail = false;
            uint64_t values = 0;
            double worst = 0;
        };
        const auto& ts = f->tensors();
        std::vector<Res> res(ts.size());
#pragma omp parallel for schedule(dynamic, 4) if (ts.size() > 32)
        for (size_t k = 0; k < ts.size(); ++k) {
            const auto& t = ts[k];
            Res& r = res[k];
            if (!limbo::quants::supported(t.type)) continue;
            r.supported = true;
            // Sample rows ACROSS the tensor, not just the first one. A [4, 8192] tensor has 4 elements
            // in its first row, so checking only ne[0] would verify 4 of 32768 values — which is how
            // the conv weight got through this check unexamined.
            const int64_t n = t.ne[0];
            const int64_t n_rows = t.n_elements() / n;
            std::vector<int64_t> rows;
            if (n_rows <= 8) {
                for (int64_t i = 0; i < n_rows; ++i) rows.push_back(i);
            } else {
                rows = {0, n_rows / 4, n_rows / 2, (3 * n_rows) / 4, n_rows - 1};
            }
            const auto* traits = ggml_get_type_traits(static_cast<ggml_type>(t.type));
            // ggml registers no to_float for F32: the decode is the identity, so the comparison is
            // against a straight copy. Any other type without one is skipped, and said to be skipped.
            const bool plain_copy = !traits || !traits->to_float;
            if (plain_copy && t.type != 0) {
                r.skip = true;
                continue;
            }
            std::vector<float> mine(static_cast<size_t>(n)), theirs(static_cast<size_t>(n));
            const uint64_t row_bytes = limbo::type_bytes(t.type, uint64_t(n));
            for (int64_t row : rows) {
                const uint8_t* p = t.data + uint64_t(row) * row_bytes;
                if (!limbo::quants::dequant_row(t.type, p, mine.data(), uint64_t(n))) {
                    r.decode_fail = true;
                    break;
                }
                if (plain_copy) std::memcpy(theirs.data(), p, size_t(n) * sizeof(float));
                else traits->to_float(p, theirs.data(), n);
                for (int64_t i = 0; i < n; ++i) {
                    const double d = std::fabs(double(mine[size_t(i)]) - double(theirs[size_t(i)]));
                    if (d > r.worst) r.worst = d;
                }
            }
            r.values = uint64_t(n) * uint64_t(rows.size());
        }
        for (size_t k = 0; k < ts.size(); ++k) {
            const auto& t = ts[k];
            const Res& r = res[k];
            if (!r.supported) continue;
            if (r.skip) {
                std::printf("   skip %-36s type %u: the oracle has no to_float\n", std::string(t.name).c_str(),
                            t.type);
                continue;
            }
            if (r.decode_fail) ++failures;
            auto& s = per_type[t.type];
            s.tensors += 1;
            s.values += r.values;
            if (r.worst > s.max_abs) {
                s.max_abs = r.worst;
                s.worst_tensor = std::string(t.name);
            }
            if (r.worst > 0) {
                std::printf("   FAIL %-36s type %-8s max|diff| %.9g\n", std::string(t.name).c_str(),
                            limbo::type_info(t.type) ? limbo::type_info(t.type)->name.data() : "?", r.worst);
                ++failures;
            }
        }
    }

    std::printf("\n%-10s %8s %12s %12s  %s\n", "format", "tensors", "values", "max|diff|", "worst tensor");
    for (const auto& [type, s] : per_type) {
        const limbo::TypeInfo* ti = limbo::type_info(type);
        std::printf("%-10s %8llu %12llu %12.9g  %s\n", ti ? ti->name.data() : "?", (unsigned long long)s.tensors,
                    (unsigned long long)s.values, s.max_abs, s.worst_tensor.c_str());
    }
    std::printf("\nquant_parity: %zu formats checked, %d difference(s)\n", per_type.size(), failures);
    return failures ? 1 : 0;
}
