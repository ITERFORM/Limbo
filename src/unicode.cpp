// Copyright (c) 2026 ITERFORM. All rights reserved.
// src/unicode.cpp — class lookup and UTF-8, over the generated tables in src/unicode_tables.cpp.
#include "limbo/unicode.hpp"

namespace limbo::uni {

bool in(const Range* ranges, size_t n, uint32_t cp) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        if (cp < ranges[mid].lo) hi = mid;
        else if (cp > ranges[mid].hi) lo = mid + 1;
        else return true;
    }
    return false;
}

size_t decode(std::string_view s, size_t at, uint32_t* cp) {
    if (at >= s.size()) {
        *cp = 0xFFFD;
        return 0;
    }
    const auto b0 = static_cast<unsigned char>(s[at]);
    size_t n;
    uint32_t v;
    if (b0 < 0x80) {
        n = 1;
        v = b0;
    } else if ((b0 & 0xE0) == 0xC0) {
        n = 2;
        v = b0 & 0x1Fu;
    } else if ((b0 & 0xF0) == 0xE0) {
        n = 3;
        v = b0 & 0x0Fu;
    } else if ((b0 & 0xF8) == 0xF0) {
        n = 4;
        v = b0 & 0x07u;
    } else {
        *cp = 0xFFFD;
        return 1;
    }
    if (at + n > s.size()) {
        *cp = 0xFFFD;
        return 1;
    }
    for (size_t i = 1; i < n; ++i) {
        const auto c = static_cast<unsigned char>(s[at + i]);
        if ((c & 0xC0) != 0x80) {
            *cp = 0xFFFD;
            return 1;
        }
        v = (v << 6) | (c & 0x3Fu);
    }
    *cp = v;
    return n;
}

void append_utf8(uint32_t cp, std::string& out) {
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

}  // namespace limbo::uni
