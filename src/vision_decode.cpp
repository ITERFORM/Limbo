// Copyright (c) 2026 ITERFORM. All rights reserved.
// src/vision_decode.cpp — data urls and PNG / JPEG decode (stb_image, third_party/stb, public domain: THIRD_PARTY_NOTICES).
#include <cstring>

#include "limbo/vision.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO
#define STBI_NO_LINEAR
#define STBI_NO_HDR
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_MAX_DIMENSIONS (1 << 15)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wimplicit-fallthrough"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wcast-function-type"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wtype-limits"
#pragma GCC diagnostic ignored "-Wshift-negative-value"
#include "../third_party/stb/stb_image.h"
#pragma GCC diagnostic pop

namespace limbo::vision {

bool decode_data_url(std::string_view url, std::vector<uint8_t>* bytes, std::string* err) {
    auto bad = [&](const char* m) {
        *err = m;
        return false;
    };
    if (url.substr(0, 5) != "data:") {
        if (url.substr(0, 7) == "http://" || url.substr(0, 8) == "https://") return bad("image urls must be data: urls (http(s) is not fetched)");
        return bad("the image url must be a data: url");
    }
    const size_t comma = url.find(',');
    if (comma == std::string_view::npos) return bad("malformed data url (no comma)");
    const std::string_view head = url.substr(5, comma - 5);
    if (head.size() < 7 || head.substr(head.size() - 7) != ";base64") return bad("the data url must be base64");
    const std::string_view b = url.substr(comma + 1);
    auto val = [](unsigned char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+' || c == '-') return 62;  // url-safe alphabet too
        if (c == '/' || c == '_') return 63;
        return -1;
    };
    bytes->clear();
    bytes->reserve(b.size() / 4 * 3 + 3);
    uint32_t acc = 0;
    int nbits = 0;
    for (unsigned char c : b) {
        if (c == '=' || c == '\n' || c == '\r' || c == ' ') continue;
        const int v = val(c);
        if (v < 0) return bad("invalid base64 in the data url");
        acc = (acc << 6) | uint32_t(v);
        nbits += 6;
        if (nbits >= 8) {
            nbits -= 8;
            bytes->push_back(uint8_t(acc >> nbits));
            acc &= (1u << nbits) - 1;
        }
    }
    if (bytes->empty()) return bad("the data url holds no data");
    return true;
}

bool decode_image(const uint8_t* data, size_t n, Image* out, std::string* err, int64_t max_pixels) {
    int w = 0, h = 0, c = 0;
    if (n == 0 || n > size_t(1) << 30) {
        *err = "empty or oversized image";
        return false;
    }
    if (!stbi_info_from_memory(data, int(n), &w, &h, &c)) {
        *err = std::string("not a PNG or JPEG image: ") + stbi_failure_reason();
        return false;
    }
    if (w < 1 || h < 1 || int64_t(w) * h > max_pixels) {
        *err = "the image is too large (" + std::to_string(w) + "x" + std::to_string(h) + ")";
        return false;
    }
    unsigned char* px = stbi_load_from_memory(data, int(n), &w, &h, &c, 3);
    if (!px) {
        *err = std::string("cannot decode the image: ") + stbi_failure_reason();
        return false;
    }
    out->w = w;
    out->h = h;
    out->rgb.assign(px, px + size_t(w) * size_t(h) * 3);
    stbi_image_free(px);
    return true;
}

}  // namespace limbo::vision
