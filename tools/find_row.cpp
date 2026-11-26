// Copyright (c) 2026 ITERFORM. All rights reserved.
// tools/find_row.cpp — which row of a tensor holds these values?
//
// A diagnostic for exactly one question: when a decoded row does not match the oracle's, is the decode
// wrong or is the row index wrong? It scans a tensor's rows with Limbo's decoder and reports the row
// whose first values match the ones asked for.
//
//   find_row <file.gguf> <tensor-name> v0,v1,v2[,v3[,v4]] [tolerance]
#include "limbo/gguf.hpp"
#include "limbo/model.hpp"
#include "limbo/quants.hpp"
#include "limbo/types.hpp"

#include <ggml.h>



#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: find_row <file.gguf> <tensor> v0,v1,v2[,...] [tol]\n");
        return 2;
    }
    const double tol = argc > 4 ? std::atof(argv[4]) : 1e-4;
    std::vector<double> want;
    {
        std::string s(argv[3]);
        size_t at = 0;
        while (at <= s.size()) {
            const size_t c = s.find(',', at);
            want.push_back(std::atof(s.substr(at, c == std::string::npos ? std::string::npos : c - at).c_str()));
            if (c == std::string::npos) break;
            at = c + 1;
        }
    }
    std::string err;
    auto m = limbo::model::Model::open({argv[1]}, &err);
    if (!m) {
        std::fprintf(stderr, "find_row: %s\n", err.c_str());
        return 1;
    }
    const limbo::gguf::Tensor* t = m->tensor(argv[2]);
    if (!t) {
        std::fprintf(stderr, "find_row: no tensor %s\n", argv[2]);
        return 1;
    }
    const int n_in = int(t->ne[0]);
    const int n_rows = t->n_dims > 1 ? int(t->ne[1]) : 1;
    const uint64_t row_bytes = limbo::type_bytes(t->type, uint64_t(n_in));
    std::printf("tensor %.*s: %s, %d elements per row, %d rows, %llu bytes per row\n",
                (int)t->name.size(), t->name.data(), limbo::type_info(t->type)->name.data(), n_in, n_rows,
                (unsigned long long)row_bytes);
    std::vector<float> row;
    row.resize(size_t(n_in));
    // --row N: decode that row with Limbo AND with ggml's own to_float, and report where they differ.
    int row_arg = -1;
    for (int i = 3; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], "--row") == 0) row_arg = std::atoi(argv[i + 1]);
    if (row_arg >= 0) {
        const int r = row_arg;
        std::vector<float> mine, theirs;
        mine.resize(size_t(n_in));
        theirs.resize(size_t(n_in));
        limbo::quants::dequant_row(t->type, t->data + uint64_t(r) * row_bytes, mine.data(), uint64_t(n_in));
        const auto* traits = ggml_get_type_traits(static_cast<ggml_type>(t->type));
        if (traits && traits->to_float) traits->to_float(t->data + uint64_t(r) * row_bytes, theirs.data(), n_in);
        else std::memcpy(theirs.data(), t->data + uint64_t(r) * row_bytes, size_t(n_in) * 4);
        double worst = 0;
        int64_t first = -1;
        for (int64_t i = 0; i < n_in; ++i) {
            const double d = std::fabs(double(mine[size_t(i)]) - double(theirs[size_t(i)]));
            if (d > worst) worst = d;
            if (first < 0 && d > 1e-9) first = i;
        }
        std::printf("row %d: max|mine - ggml| = %.9g", r, worst);
        if (first >= 0) std::printf(", first difference at index %lld (mine %.6f vs ggml %.6f)", (long long)first,
                                    mine[size_t(first)], theirs[size_t(first)]);
        std::printf("\n  mine[0..3] = %.4f %.4f %.4f %.4f   ggml[0..3] = %.4f %.4f %.4f %.4f\n", mine[0], mine[1],
                    mine[2], mine[3], theirs[0], theirs[1], theirs[2], theirs[3]);
        std::printf("  mine[1920..1923] = %.6f %.6f %.6f %.6f   ggml[1920..1923] = %.6f %.6f %.6f %.6f\n",
                    mine[1920], mine[1921], mine[1922], mine[1923], theirs[1920], theirs[1921], theirs[1922],
                    theirs[1923]);
        return worst > 0 ? 1 : 0;
    }
    int hits = 0;
    for (int r = 0; r < n_rows; ++r) {
        limbo::quants::dequant_row(t->type, t->data + uint64_t(r) * row_bytes, row.data(), uint64_t(n_in));
        bool ok = true;
        for (size_t i = 0; i < want.size(); ++i)
            if (std::fabs(double(row[i]) - want[i]) > tol) ok = false;
        if (ok) {
            std::printf("  row %d matches", r);
            for (size_t i = 0; i < want.size(); ++i) std::printf(" %.4f", row[i]);
            std::printf("\n");
            if (++hits >= 8) break;
        }
    }
    std::printf("find_row: %d matching row(s)\n", hits);
    return hits ? 0 : 1;
}
