// Copyright (c) 2026 ITERFORM. All rights reserved.
// tools/vision_check.cpp — Limbo's CPU reference tower against llama.cpp's projector output (oracle/oracle_vision).
//
//   vision_check --mmproj mmproj.gguf --image a.png --oracle a.emb [--max-tokens N] [--threads N] [--dump ours.emb]
//
// Prints the token grid and the errors of our embeddings against the oracle's: the relative L2 error of the whole matrix, the
// worst per-token relative L2 error, and the largest element error in units of the oracle's RMS.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "limbo/vision.hpp"

using namespace limbo::vision;

// --nodes-dir: compare every intermediate the oracle saved (oracle_vision --node-dir) with ours, name by name.
static std::string g_nodes;
static void sink(const char* name, const float* d, size_t n, void*) {
    std::ifstream f(g_nodes + "/" + name + ".f32", std::ios::binary);
    if (!f) return;
    std::vector<float> ref(n);
    f.read(reinterpret_cast<char*>(ref.data()), std::streamsize(n * sizeof(float)));
    if (!f) {
        std::printf("  node %-14s  (size mismatch)\n", name);
        return;
    }
    double num = 0, den = 0;
    for (size_t i = 0; i < n; ++i) {
        num += double(d[i] - ref[i]) * double(d[i] - ref[i]);
        den += double(ref[i]) * double(ref[i]);
    }
    std::printf("  node %-14s  rel L2 %.3e\n", name, std::sqrt(num / den));
}

int main(int argc, char** argv) {
    std::string mm, image, oracle, dump;
    int max_tokens = 1024, threads = 8;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--mmproj") mm = next();
        else if (a == "--image") image = next();
        else if (a == "--oracle") oracle = next();
        else if (a == "--dump") dump = next();
        else if (a == "--max-tokens") max_tokens = std::atoi(next().c_str());
        else if (a == "--threads") threads = std::atoi(next().c_str());
        else if (a == "--oracle-rounding") set_oracle_rounding(true);
        else if (a == "--nodes-dir") g_nodes = next();
    }
    if (mm.empty() || image.empty()) {
        std::fprintf(stderr, "usage: vision_check --mmproj f --image f [--oracle f.emb] [--max-tokens N] [--threads N] [--dump f]\n");
        return 2;
    }
    std::string err;
    auto pj = Projector::open(mm, &err);
    if (!pj) return std::fprintf(stderr, "%s\n", err.c_str()), 1;
    std::ifstream fi(image, std::ios::binary);
    const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(fi)), std::istreambuf_iterator<char>());
    Image im;
    if (!decode_image(bytes.data(), bytes.size(), &im, &err)) return std::fprintf(stderr, "%s\n", err.c_str()), 1;
    Limits lim;
    lim.max_tokens = max_tokens;
    Prepared pr;
    if (!preprocess(im, pj->hp(), lim, &pr, &err)) return std::fprintf(stderr, "%s\n", err.c_str()), 1;
    std::printf("image %dx%d -> %dx%d px, %d x %d tokens (%d)\n", im.w, im.h, pr.w, pr.h, pr.nx, pr.ny, pr.n_tokens());
    if (!g_nodes.empty()) set_debug_sink(sink, nullptr);
    std::vector<float> ours;
    if (!encode_ref(*pj, pr, threads, &ours, &err)) return std::fprintf(stderr, "%s\n", err.c_str()), 1;
    if (!dump.empty()) {
        FILE* f = std::fopen(dump.c_str(), "wb");
        std::fwrite("LVE1", 1, 4, f);
        const int32_t h[4] = {pr.n_tokens(), pj->hp().out_dim, pr.nx, pr.ny};
        std::fwrite(h, sizeof h, 1, f);
        std::fwrite(ours.data(), sizeof(float), ours.size(), f);
        std::fclose(f);
    }
    if (oracle.empty()) return 0;
    std::ifstream fo(oracle, std::ios::binary);
    char magic[4];
    int32_t h[4];
    fo.read(magic, 4);
    fo.read(reinterpret_cast<char*>(h), sizeof h);
    if (!fo || std::memcmp(magic, "LVE1", 4)) return std::fprintf(stderr, "not an oracle capture\n"), 1;
    if (h[0] != pr.n_tokens() || h[1] != pj->hp().out_dim || h[2] != pr.nx || h[3] != pr.ny) {
        std::printf("GRID MISMATCH: oracle %d tokens (%d x %d), ours %d (%d x %d)\n", h[0], h[2], h[3], pr.n_tokens(), pr.nx, pr.ny);
        return 1;
    }
    std::vector<float> ref(size_t(h[0]) * size_t(h[1]));
    fo.read(reinterpret_cast<char*>(ref.data()), std::streamsize(ref.size() * sizeof(float)));
    if (!fo) return std::fprintf(stderr, "short oracle capture\n"), 1;
    const size_t D = size_t(h[1]);
    double num = 0, den = 0, worst_tok = 0, max_el = 0;
    for (size_t t = 0; t < size_t(h[0]); ++t) {
        double tn = 0, td = 0;
        for (size_t d = 0; d < D; ++d) {
            const double a = ours[t * D + d], b = ref[t * D + d], e = a - b;
            tn += e * e;
            td += b * b;
            max_el = std::max(max_el, std::fabs(e));
        }
        num += tn;
        den += td;
        worst_tok = std::max(worst_tok, std::sqrt(tn / td));
    }
    const double rms = std::sqrt(den / double(ours.size()));
    std::printf("vs oracle: rel L2 %.3e   worst token rel L2 %.3e   max |err| %.3e (= %.3e x rms %.3f)\n", std::sqrt(num / den), worst_tok, max_el,
                max_el / rms, rms);
    return 0;
}
