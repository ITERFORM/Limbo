// Copyright (c) 2026 ITERFORM. All rights reserved.
// src/vision.cpp — the mmproj file, preprocessing, the tape's image tokens and the LRU (docs/vision.md §1, §2, §4-§6).
// Our own code from the spec. The reference tower is in vision_ref.cpp.
#include "limbo/vision.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "limbo/types.hpp"

namespace limbo::vision {

namespace {

uint64_t mix64(uint64_t x) {
    x ^= x >> 32;
    x *= 0xD6E8FEB86659FD93ull;
    x ^= x >> 32;
    x *= 0xD6E8FEB86659FD93ull;
    x ^= x >> 32;
    return x;
}

}  // namespace

uint64_t hash_bytes(const uint8_t* p, size_t n, uint64_t seed) {
    uint64_t h = mix64(seed ^ (uint64_t(n) * 0x9E3779B97F4A7C15ull));
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint64_t w;
        std::memcpy(&w, p + i, 8);
        h = mix64(h ^ w) + 0x9E3779B97F4A7C15ull;
    }
    uint64_t t = 0;
    std::memcpy(&t, p + i, n - i);
    return mix64(h ^ t ^ (uint64_t(n - i) << 56));
}

int32_t image_token(uint64_t key, int row) {
    const uint32_t v = 0x80000000u | uint32_t(mix64(key + uint64_t(row) * 0x9E3779B97F4A7C15ull) >> 33);
    return v == 0xFFFFFFFFu ? int32_t(0xFFFFFFFEu) : int32_t(v);  // -1 is the server's "none"
}

// ---- the projector file ----------------------------------------------------------------------------------------

std::unique_ptr<Projector> Projector::open(const std::string& path, std::string* err) {
    auto fail = [&](const std::string& m) -> std::unique_ptr<Projector> {
        *err = "mmproj " + path + ": " + m;
        return nullptr;
    };
    std::unique_ptr<gguf::File> f = gguf::File::open(path, err);
    if (!f) return nullptr;
    if (f->arch() != "clip") return fail("not a clip projector (architecture '" + std::string(f->arch()) + "')");
    if (f->get_str("clip.projector_type").value_or("") != "qwen3vl_merger") return fail("only the qwen3vl_merger projector is supported");
    auto geti = [&](const char* k, int* v) {
        const auto x = f->get_int(k);
        if (x) *v = int(*x);
        return x.has_value();
    };
    std::unique_ptr<Projector> pj(new Projector());
    Hparams& h = pj->hp_;
    if (!geti("clip.vision.patch_size", &h.patch) || !geti("clip.vision.embedding_length", &h.embd) ||
        !geti("clip.vision.feed_forward_length", &h.ffn) || !geti("clip.vision.block_count", &h.layers) ||
        !geti("clip.vision.attention.head_count", &h.heads) || !geti("clip.vision.projection_dim", &h.out_dim))
        return fail("a clip.vision.* key is missing");
    geti("clip.vision.spatial_merge_size", &h.merge);
    if (const auto e = f->get_float("clip.vision.attention.layer_norm_epsilon")) h.eps = float(*e);
    const std::vector<float> mean = f->get_f32_array("clip.vision.image_mean"), sd = f->get_f32_array("clip.vision.image_std");
    if (mean.size() == 3 && sd.size() == 3)
        for (int i = 0; i < 3; ++i) {
            h.mean[i] = mean[size_t(i)];
            h.stdev[i] = sd[size_t(i)];
        }
    if (h.layers < 1 || h.heads < 1 || h.embd % h.heads || h.head_dim() % 2 || h.merge != 2 || h.patch < 2 || h.embd < 1) return fail("unsupported geometry");
    if (const gguf::Value* ds = f->find("clip.vision.is_deepstack_layers"); ds && ds->type == gguf::VType::ARRAY)
        for (uint64_t i = 0; i < ds->count; ++i)
            if (ds->raw[i]) return fail("deepstack layers are not supported");

    auto mat = [&](const std::string& name, int64_t n, int64_t k, Mat* m) {
        const gguf::Tensor* t = f->tensor(name);
        if (!t || t->type != uint32_t(Type::BF16) || t->ne[0] != k || t->ne[1] != n) return false;
        m->p = reinterpret_cast<const uint16_t*>(t->data);
        m->n = n;
        m->k = k;
        return true;
    };
    auto vec = [&](const std::string& name, int64_t n, const float** v) {
        const gguf::Tensor* t = f->tensor(name);
        if (!t || t->type != uint32_t(Type::F32) || t->n_elements() != n) return false;
        *v = reinterpret_cast<const float*>(t->data);
        return true;
    };
    const int64_t E = h.embd, F = h.ffn, M4 = E * 4;
    bool ok = mat("mm.0.weight", M4, M4, &pj->mm0) && mat("mm.2.weight", h.out_dim, M4, &pj->mm2) && vec("mm.0.bias", M4, &pj->mm0_b) &&
              vec("mm.2.bias", h.out_dim, &pj->mm2_b) && vec("v.post_ln.weight", E, &pj->post_w) && vec("v.post_ln.bias", E, &pj->post_b) &&
              vec("v.patch_embd.bias", E, &pj->patch_b);
    const int64_t pk = int64_t(h.patch) * h.patch * 3;
    ok = ok && vec("v.patch_embd.weight", pk * E, &pj->patch_w0) && vec("v.patch_embd.weight.1", pk * E, &pj->patch_w1);
    if (const gguf::Tensor* t = f->tensor("v.position_embd.weight")) {
        const int64_t np = t->ne[1];
        int side = int(std::lround(std::sqrt(double(np))));
        if (t->type == uint32_t(Type::F32) && t->ne[0] == E && int64_t(side) * side == np) {
            h.pos_side = side;
            pj->pos_embd = reinterpret_cast<const float*>(t->data);
        } else ok = false;
    } else ok = false;
    if (!ok) return fail("a global tensor is missing or has an unexpected shape or type");
    pj->layers_.resize(size_t(h.layers));
    for (int l = 0; l < h.layers; ++l) {
        const std::string b = "v.blk." + std::to_string(l) + ".";
        Layer& y = pj->layers_[size_t(l)];
        ok = mat(b + "attn_qkv.weight", 3 * E, E, &y.qkv) && mat(b + "attn_out.weight", E, E, &y.out) && mat(b + "ffn_up.weight", F, E, &y.up) &&
             mat(b + "ffn_down.weight", E, F, &y.down) && vec(b + "attn_qkv.bias", 3 * E, &y.qkv_b) && vec(b + "attn_out.bias", E, &y.out_b) &&
             vec(b + "ffn_up.bias", F, &y.up_b) && vec(b + "ffn_down.bias", E, &y.down_b) && vec(b + "ln1.weight", E, &y.ln1_w) &&
             vec(b + "ln1.bias", E, &y.ln1_b) && vec(b + "ln2.weight", E, &y.ln2_w) && vec(b + "ln2.bias", E, &y.ln2_b);
        if (!ok) return fail("block " + std::to_string(l) + " is missing a tensor or has an unexpected shape");
    }
    pj->file_ = std::move(f);
    return pj;
}

// ---- preprocessing -----------------------------------------------------------------------------------------------

bool target_size(int w, int h, const Hparams& hp, const Limits& lim, int* tw, int* th) {
    if (w < 1 || h < 1) return false;
    const int u = hp.unit();
    const int64_t px_tok = int64_t(u) * u;
    const int64_t lo = int64_t(std::max(1, lim.min_tokens)) * px_tok, hi = int64_t(std::max(1, lim.max_tokens)) * px_tok;
    auto round_u = [&](double x) { return int(std::lround(x / u)) * u; };  // halves away from zero, like round()
    auto ceil_u = [&](double x) { return int(std::ceil(x / u)) * u; };
    auto floor_u = [&](double x) { return int(std::floor(x / u)) * u; };
    int wb = std::max(u, round_u(w)), hb = std::max(u, round_u(h));
    if (int64_t(wb) * hb > hi) {
        const float beta = std::sqrt(float(h) * float(w) / float(hi));
        hb = std::max(u, floor_u(float(h) / beta));
        wb = std::max(u, floor_u(float(w) / beta));
    } else if (int64_t(wb) * hb < lo) {
        const float beta = std::sqrt(float(lo) / (float(h) * float(w)));
        hb = ceil_u(float(h) * beta);
        wb = ceil_u(float(w) * beta);
    }
    *tw = wb;
    *th = hb;
    return true;
}

namespace {

// Taps of a separable Pillow-style bicubic resample (a = -0.5) from `in` to `out` samples, as 22-bit fixed point.
struct Taps {
    int ksize = 0;
    std::vector<int> first, count;
    std::vector<int32_t> w;  // out * ksize
};

double cubic(double x) {
    constexpr double a = -0.5;
    x = std::fabs(x);
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    if (x < 2.0) return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
    return 0.0;
}

Taps make_taps(int in, int out) {
    Taps t;
    const double scale = double(in) / out, fscale = std::max(1.0, scale), support = 2.0 * fscale;
    t.ksize = int(std::ceil(support)) * 2 + 1;
    t.first.assign(size_t(out), 0);
    t.count.assign(size_t(out), 0);
    t.w.assign(size_t(out) * size_t(t.ksize), 0);
    std::vector<double> tmp(size_t(t.ksize));
    for (int o = 0; o < out; ++o) {
        const double c = (o + 0.5) * scale;
        const int lo = std::max(0, int(c - support + 0.5)), hi = std::min(in, int(c + support + 0.5));
        const int n = hi - lo;
        double sum = 0;
        for (int i = 0; i < n; ++i) {
            tmp[size_t(i)] = cubic((i + lo - c + 0.5) / fscale);
            sum += tmp[size_t(i)];
        }
        t.first[size_t(o)] = lo;
        t.count[size_t(o)] = n;
        for (int i = 0; i < n; ++i) {
            const double v = sum != 0 ? tmp[size_t(i)] / sum : tmp[size_t(i)];
            t.w[size_t(o) * size_t(t.ksize) + size_t(i)] = int32_t(v * double(1 << 22) + (v < 0 ? -0.5 : 0.5));  // truncates toward zero
        }
    }
    return t;
}

inline uint8_t clamp8(int32_t v) { return v < 0 ? 0 : v > 255 ? 255 : uint8_t(v); }

}  // namespace

bool preprocess(const Image& img, const Hparams& hp, const Limits& lim, Prepared* out, std::string* err) {
    int tw = 0, th = 0;
    if (img.w < 1 || img.h < 1 || img.rgb.size() != size_t(img.w) * size_t(img.h) * 3 || !target_size(img.w, img.h, hp, lim, &tw, &th)) {
        *err = "empty image";
        return false;
    }
    std::vector<uint8_t> cur = img.rgb;
    int cw = img.w, ch = img.h;
    if (tw != cw) {  // horizontal pass
        const Taps t = make_taps(cw, tw);
        std::vector<uint8_t> nx(size_t(tw) * size_t(ch) * 3);
        for (int y = 0; y < ch; ++y) {
            const uint8_t* row = cur.data() + size_t(y) * size_t(cw) * 3;
            uint8_t* dst = nx.data() + size_t(y) * size_t(tw) * 3;
            for (int x = 0; x < tw; ++x) {
                const int32_t* k = &t.w[size_t(x) * size_t(t.ksize)];
                const uint8_t* p = row + size_t(t.first[size_t(x)]) * 3;
                int32_t s0 = 1 << 21, s1 = 1 << 21, s2 = 1 << 21;
                for (int i = 0; i < t.count[size_t(x)]; ++i, p += 3) {
                    s0 += int32_t(p[0]) * k[i];
                    s1 += int32_t(p[1]) * k[i];
                    s2 += int32_t(p[2]) * k[i];
                }
                dst[x * 3] = clamp8(s0 >> 22);
                dst[x * 3 + 1] = clamp8(s1 >> 22);
                dst[x * 3 + 2] = clamp8(s2 >> 22);
            }
        }
        cur.swap(nx);
        cw = tw;
    }
    if (th != ch) {  // vertical pass
        const Taps t = make_taps(ch, th);
        const size_t rowb = size_t(cw) * 3;
        std::vector<uint8_t> nx(rowb * size_t(th));
        std::vector<int32_t> acc(rowb);
        for (int y = 0; y < th; ++y) {
            std::fill(acc.begin(), acc.end(), 1 << 21);
            const int32_t* k = &t.w[size_t(y) * size_t(t.ksize)];
            for (int i = 0; i < t.count[size_t(y)]; ++i) {
                const uint8_t* src = cur.data() + size_t(t.first[size_t(y)] + i) * rowb;
                for (size_t j = 0; j < rowb; ++j) acc[j] += int32_t(src[j]) * k[i];
            }
            for (size_t j = 0; j < rowb; ++j) nx[size_t(y) * rowb + j] = clamp8(acc[j] >> 22);
        }
        cur.swap(nx);
        ch = th;
    }
    out->w = cw;
    out->h = ch;
    out->px = cw / hp.patch;
    out->py = ch / hp.patch;
    out->nx = out->px / hp.merge;
    out->ny = out->py / hp.merge;
    out->rgb = std::move(cur);
    uint64_t k = hash_bytes(out->rgb.data(), out->rgb.size(), 0x1234abcd);
    k = mix64(k ^ (uint64_t(uint32_t(out->nx)) << 32 | uint32_t(out->ny)));
    out->key = k;
    return true;
}

void normalise_chw(const Prepared& p, const Hparams& hp, float* out) {
    const size_t plane = size_t(p.w) * size_t(p.h);
    for (size_t i = 0; i < plane; ++i)
        for (int c = 0; c < 3; ++c) out[size_t(c) * plane + i] = (float(p.rgb[i * 3 + size_t(c)]) / 255.0f - hp.mean[c]) / hp.stdev[c];
}

// ---- the LRU -----------------------------------------------------------------------------------------------------

std::shared_ptr<const Prepared> Store::add(std::string_view url, std::string* err) {
    std::vector<uint8_t> bytes;
    if (!decode_data_url(url, &bytes, err)) return nullptr;
    const uint64_t raw = mix64(hash_bytes(bytes.data(), bytes.size(), 7) ^ (uint64_t(uint32_t(lim_.max_tokens)) << 20 | uint32_t(lim_.min_tokens)));
    if (auto it = by_raw_.find(raw); it != by_raw_.end()) {
        prep_.splice(prep_.begin(), prep_, it->second);
        return it->second->p;
    }
    Image im;
    if (!decode_image(bytes.data(), bytes.size(), &im, err)) return nullptr;
    auto p = std::make_shared<Prepared>();
    if (!preprocess(im, hp_, lim_, p.get(), err)) return nullptr;
    prep_.push_front({raw, p});
    by_raw_[raw] = prep_.begin();
    by_key_[p->key] = prep_.begin();
    used_prep_ += p->bytes();
    while (used_prep_ > cap_prep_ && prep_.size() > 1) {
        const PEnt& e = prep_.back();
        used_prep_ -= e.p->bytes();
        by_raw_.erase(e.raw);
        if (auto k = by_key_.find(e.p->key); k != by_key_.end() && k->second == std::prev(prep_.end())) by_key_.erase(k);
        prep_.pop_back();
    }
    return p;
}

std::shared_ptr<const Prepared> Store::find(uint64_t key) const {
    const auto it = by_key_.find(key);
    return it == by_key_.end() ? nullptr : it->second->p;
}

std::shared_ptr<const std::vector<float>> Store::embeddings(uint64_t key) {
    const auto it = by_ekey_.find(key);
    if (it == by_ekey_.end()) return nullptr;
    embd_.splice(embd_.begin(), embd_, it->second);
    return it->second->e;
}

void Store::put_embeddings(uint64_t key, std::shared_ptr<const std::vector<float>> e) {
    if (by_ekey_.count(key)) return;
    used_embd_ += e->size() * sizeof(float);
    embd_.push_front({key, std::move(e)});
    by_ekey_[key] = embd_.begin();
    while (used_embd_ > cap_embd_ && embd_.size() > 1) {
        const EEnt& b = embd_.back();
        used_embd_ -= b.e->size() * sizeof(float);
        by_ekey_.erase(b.key);
        embd_.pop_back();
    }
}

}  // namespace limbo::vision
