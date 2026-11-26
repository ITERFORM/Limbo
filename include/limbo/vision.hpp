// Copyright (c) 2026 ITERFORM. All rights reserved.
#pragma once
// limbo/vision.hpp — image input for the qwen3vl-merger projector (docs/vision.md): the mmproj file, image decode,
// preprocessing, the token grid, the pseudo tokens an image becomes on the tape, an LRU of what was computed, and a CPU
// reference tower (the GPU tower in limbo/cuda/vision.hpp is tested against it, and it against llama.cpp's).
#include <cstdint>
#include <list>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "limbo/gguf.hpp"

namespace limbo::vision {

struct Hparams {
    int patch = 16, embd = 1152, ffn = 4304, layers = 27, heads = 16, merge = 2, out_dim = 5120, pos_side = 48;
    float eps = 1e-6f;
    float mean[3] = {0.5f, 0.5f, 0.5f}, stdev[3] = {0.5f, 0.5f, 0.5f};
    int head_dim() const { return embd / heads; }
    int unit() const { return patch * merge; }  // pixels per token side
};

// A weight matrix as the file stores it: n rows of k BF16 (k contiguous), so y[n] = sum_k w[n][k] x[k].
struct Mat {
    const uint16_t* p = nullptr;
    int64_t n = 0, k = 0;
};

struct Layer {
    Mat qkv, out, up, down;
    const float *qkv_b = nullptr, *out_b = nullptr, *up_b = nullptr, *down_b = nullptr;
    const float *ln1_w = nullptr, *ln1_b = nullptr, *ln2_w = nullptr, *ln2_b = nullptr;
};

// The mmproj GGUF, mapped. Refuses anything but the qwen3vl_merger projector with the shapes the tower is written for.
class Projector {
public:
    static std::unique_ptr<Projector> open(const std::string& path, std::string* err);
    const Hparams& hp() const { return hp_; }
    const std::vector<Layer>& layers() const { return layers_; }
    Mat mm0, mm2;
    const float *mm0_b = nullptr, *mm2_b = nullptr, *post_w = nullptr, *post_b = nullptr;
    const float* patch_w0 = nullptr;  // f32 [1152][3][16][16], the two temporal taps
    const float* patch_w1 = nullptr;
    const float* patch_b = nullptr;
    const float* pos_embd = nullptr;  // f32 [pos_side^2][embd]
    uint64_t weight_bytes() const { return file_->size(); }

private:
    Projector() = default;
    std::unique_ptr<gguf::File> file_;
    Hparams hp_;
    std::vector<Layer> layers_;
};

struct Image {
    int w = 0, h = 0;
    std::vector<uint8_t> rgb;  // h * w * 3, interleaved
};

// "data:image/png;base64,...." -> the bytes. http(s) urls and every other scheme are refused (no fetch).
bool decode_data_url(std::string_view url, std::vector<uint8_t>* bytes, std::string* err);
// PNG or JPEG to 8-bit RGB (alpha dropped). Refuses more than max_pixels.
bool decode_image(const uint8_t* data, size_t n, Image* out, std::string* err, int64_t max_pixels = 64'000'000);

struct Limits {
    int min_tokens = 8;
    int max_tokens = 1024;  // image tokens (32 x 32 px each) the tower may produce; the server lowers it to what the scratch holds
};

// The resized size of docs/vision.md §2 (a multiple of patch * merge on both sides). False for an empty image.
bool target_size(int w, int h, const Hparams& hp, const Limits& lim, int* tw, int* th);

struct Prepared {
    int w = 0, h = 0;      // resized pixels
    int px = 0, py = 0;    // patches
    int nx = 0, ny = 0;    // tokens (after the 2x2 merge)
    uint64_t key = 0;      // identity of (file bytes, grid): the tape's pseudo tokens and the caches use it
    std::vector<uint8_t> rgb;  // the resized image, h * w * 3
    int n_tokens() const { return nx * ny; }
    int n_pos() const { return nx > ny ? nx : ny; }  // text positions the image consumes
    size_t bytes() const { return rgb.size() + sizeof(*this); }
};

bool preprocess(const Image& img, const Hparams& hp, const Limits& lim, Prepared* out, std::string* err);
// The planar normalised input of the patch conv: out[3][h][w] floats, (px/255 - mean)/std.
void normalise_chw(const Prepared& p, const Hparams& hp, float* out);

uint64_t hash_bytes(const uint8_t* p, size_t n, uint64_t seed = 0);

// ---- the tape and the positions --------------------------------------------------------------------------------
// An image is n tokens on the tape: ids with the top bit set, a function of (key, row). Text ids are small and non-negative.
inline bool is_image_token(int32_t t) { return t < 0; }
int32_t image_token(uint64_t key, int row);
// The three M-RoPE positions of row r of an image whose first token is at text position p0 (docs/vision.md §5).
inline void image_positions(int p0, int nx, int r, int32_t out[3]) {
    out[0] = p0;
    out[1] = p0 + r / nx;
    out[2] = p0 + r % nx;
}
// Rows the cache holds minus positions consumed: the shift of every later text token's rope position (<= 0).
inline int rope_delta_of(int nx, int ny) { return (nx > ny ? nx : ny) - nx * ny; }

// ---- an LRU of what the server computed per image --------------------------------------------------------------
// Prepared images by file hash (so a re-sent image skips decode and resize), and projected embeddings by key.
class Store {
public:
    Store(const Hparams& hp, Limits lim, size_t prepared_bytes = size_t(256) << 20, size_t embd_bytes = size_t(768) << 20)
        : hp_(hp), lim_(lim), cap_prep_(prepared_bytes), cap_embd_(embd_bytes) {}
    const Limits& limits() const { return lim_; }
    void set_max_tokens(int n) { lim_.max_tokens = n; }
    // The prepared image of a data url (decoded and resized now, or from the cache).
    std::shared_ptr<const Prepared> add(std::string_view data_url, std::string* err);
    std::shared_ptr<const Prepared> find(uint64_t key) const;
    // Projected embeddings of an image, n_tokens x out_dim floats, if kept.
    std::shared_ptr<const std::vector<float>> embeddings(uint64_t key);
    void put_embeddings(uint64_t key, std::shared_ptr<const std::vector<float>> e);

private:
    Hparams hp_;
    Limits lim_;
    size_t cap_prep_, cap_embd_, used_prep_ = 0, used_embd_ = 0;
    struct PEnt {
        uint64_t raw;
        std::shared_ptr<const Prepared> p;
    };
    std::list<PEnt> prep_;  // most recent first
    std::unordered_map<uint64_t, std::list<PEnt>::iterator> by_raw_, by_key_;
    struct EEnt {
        uint64_t key;
        std::shared_ptr<const std::vector<float>> e;
    };
    std::list<EEnt> embd_;
    std::unordered_map<uint64_t, std::list<EEnt>::iterator> by_ekey_;
};

// ---- the CPU reference tower -------------------------------------------------------------------------------------
// fp32 arithmetic, the BF16 weights widened exactly: the numbers the GPU tower is compared to. out = n_tokens x out_dim.
// Test hook: round every matmul input to BF16 and q, k, v to F16 (what llama.cpp's CPU path does) so the tower can be told apart
// from the oracle's own rounding noise. Off by default.
void set_oracle_rounding(bool on);
// Test hook: called with named intermediates (llama.cpp's node names: inp_pos_emb, ln1-N, Qcur_rope-N, attn_out-N, ffn_inp-N,
// layer_out-N, ...) as n_patches x width floats in merge-group order.
using DebugSink = void (*)(const char* name, const float* data, size_t n, void* user);
void set_debug_sink(DebugSink f, void* user);
bool encode_ref(const Projector& pj, const Prepared& img, int threads, std::vector<float>* out, std::string* err);
// The intermediate after the patch embedding and the positions (n_patches x embd, merge-group order), for tests.
bool embed_patches_ref(const Projector& pj, const Prepared& img, int threads, std::vector<float>* out, std::string* err);

}  // namespace limbo::vision
