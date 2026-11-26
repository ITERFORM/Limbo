// Copyright (c) 2026 ITERFORM. All rights reserved.
// tests/cuda_paged_test.cu — the paged K/V kernels (docs/paged-kv.md) against the contiguous ones, bit for bit, in all three formats.
//
// A cache of random rows is built twice: contiguous (the layout of include/limbo/cuda/kv.hpp, page table null) and paged (a pool of
// 11 pages, the 6 chunks of the sequence on shuffled pages, the other pages holding other rows). The same launches then run on both
// and must agree to the bit:
//   1. decode attention: one row at the last position, an 8-row window at the end, an 8-row window across a chunk boundary;
//   2. the decode prep: 8 rows across a chunk boundary and one row, then the whole cache read back through the table equals the
//      contiguous one (every byte written lands where the table says, none elsewhere);
//   3. the prompt: the batched prep across chunk boundaries, then the tensor-core attention: the f16 pool read in place through the
//      table against the plain whole-cache kernel, the Q8 and Q4 pools dequantised into f16 windows of 512 through the table against
//      the same windows made from the contiguous planes.
// No GPU: SKIP, exit 0.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "../src/cuda/dec_ops.hpp"
#include "../src/cuda/ops.hpp"
#include "limbo/check.hpp"
#include "limbo/cuda/kv.hpp"

namespace {

using limbo::cuda::KvFormat;
namespace dec = limbo::cuda::dec;
namespace ops = limbo::cuda::ops;

#define CUDA_OK(call)                                                                                \
    do {                                                                                             \
        cudaError_t e_ = (call);                                                                     \
        if (e_ != cudaSuccess) {                                                                     \
            std::printf("FAIL %s: %s (%s:%d)\n", #call, cudaGetErrorString(e_), __FILE__, __LINE__);  \
            ++limbo::test::failures();                                                               \
            return;                                                                                  \
        }                                                                                            \
    } while (0)

constexpr int NH = 16, NKV = 2, HD = 256, ROPE = 64;
constexpr int N_CHUNKS = 6, N_CTX = N_CHUNKS * 256, N_PAGES = 11, VPITCH = N_CTX;  // N_CTX is a multiple of 64

uint32_t g_seed = 20261121u;
float frand() {
    g_seed = g_seed * 1664525u + 1013904223u;
    return static_cast<float>(g_seed >> 8) / 16777216.0f * 2.f - 1.f;
}
uint32_t urand() {
    g_seed = g_seed * 1664525u + 1013904223u;
    return g_seed >> 8;
}

// One layer's cache as bytes: K, V, K scales, V scales. Contiguous: rows [pos][kv head]; f16 V [kv head][256][VPITCH].
struct Planes {
    std::vector<uint8_t> k, v, ks, vs;
};

size_t row_bytes(int f) { return f == 0 ? HD * 2 : f == 1 ? HD : HD / 2; }
constexpr size_t kScaleRow = HD / 32 * 2;

Planes make_planes(int f, int rows_pos, bool paged) {
    Planes p;
    const size_t pos = size_t(rows_pos);
    if (f == 0) {
        p.k.assign(pos * NKV * row_bytes(0), 0);
        p.v.assign(paged ? pos * NKV * row_bytes(0) : size_t(NKV) * HD * VPITCH * 2, 0);
    } else {
        p.k.assign(pos * NKV * row_bytes(f), 0);
        p.v.assign(pos * NKV * row_bytes(f), 0);
        p.ks.assign(pos * NKV * kScaleRow, 0);
        p.vs.assign(pos * NKV * kScaleRow, 0);
    }
    return p;
}

void randomise(int f, Planes* c) {
    if (f == 0) {
        for (size_t i = 0; i < c->k.size() / 2; ++i) reinterpret_cast<__half*>(c->k.data())[i] = __float2half(frand());
        for (size_t i = 0; i < c->v.size() / 2; ++i) reinterpret_cast<__half*>(c->v.data())[i] = __float2half(frand() * 1.7f);
        return;
    }
    for (auto* v : {&c->k, &c->v})
        for (uint8_t& b : *v) b = uint8_t(urand());
    for (auto* v : {&c->ks, &c->vs})
        for (size_t i = 0; i < v->size() / 2; ++i) reinterpret_cast<__half*>(v->data())[i] = __float2half(0.004f + 0.01f * (urand() % 16) / 16.f);
}

// Where a position's rows live: the contiguous row and the pool row (page table pt), and the f16 V element.
struct Map {
    const std::vector<int>& pt;
    size_t crow(int pos, int kh) const { return size_t(pos) * NKV + kh; }
    size_t prow(int pos, int kh) const { return (size_t(pt[size_t(pos / 256)]) * 256 + size_t(pos % 256)) * NKV + kh; }
    size_t cv(int kh, int d, int pos) const { return (size_t(kh) * HD + d) * VPITCH + size_t(pos); }
    size_t pv(int kh, int d, int pos) const { return size_t(pt[size_t(pos / 256)]) * NKV * HD * 256 + (size_t(kh) * HD + d) * 256 + size_t(pos % 256); }
};

// Copies the sequence's rows between a contiguous and a pooled cache; to_pool: contiguous -> pool.
void convert(int f, const Map& m, Planes* c, Planes* p, bool to_pool) {
    const size_t rb = row_bytes(f);
    for (int pos = 0; pos < N_CTX; ++pos)
        for (int kh = 0; kh < NKV; ++kh) {
            auto move = [&](std::vector<uint8_t>& cv, std::vector<uint8_t>& pv, size_t n) {
                uint8_t* a = cv.data() + m.crow(pos, kh) * n;
                uint8_t* b = pv.data() + m.prow(pos, kh) * n;
                if (to_pool) std::memcpy(b, a, n);
                else std::memcpy(a, b, n);
            };
            move(c->k, p->k, rb);
            if (f == 0) {
                for (int d = 0; d < HD; ++d) {
                    uint8_t* a = c->v.data() + m.cv(kh, d, pos) * 2;
                    uint8_t* b = p->v.data() + m.pv(kh, d, pos) * 2;
                    if (to_pool) std::memcpy(b, a, 2);
                    else std::memcpy(a, b, 2);
                }
            } else {
                move(c->v, p->v, rb);
                move(c->ks, p->ks, kScaleRow);
                move(c->vs, p->vs, kScaleRow);
            }
        }
}

struct DevPlanes {
    void *k = nullptr, *v = nullptr, *ks = nullptr, *vs = nullptr;
    void upload(const Planes& p) {
        for (auto [d, h] : {std::pair<void**, const std::vector<uint8_t>*>{&k, &p.k}, {&v, &p.v}, {&ks, &p.ks}, {&vs, &p.vs}}) {
            if (h->empty()) continue;
            cudaMalloc(d, h->size());
            cudaMemcpy(*d, h->data(), h->size(), cudaMemcpyHostToDevice);
        }
    }
    void download(Planes* p) const {
        for (auto [d, h] : {std::pair<void* const*, std::vector<uint8_t>*>{&k, &p->k}, {&v, &p->v}, {&ks, &p->ks}, {&vs, &p->vs}})
            if (*d) cudaMemcpy(h->data(), *d, h->size(), cudaMemcpyDeviceToHost);
    }
    ~DevPlanes() {
        for (void* d : {k, v, ks, vs}) cudaFree(d);
    }
};

template <typename T>
struct Dev {
    T* p = nullptr;
    explicit Dev(const std::vector<T>& h) {
        cudaMalloc(reinterpret_cast<void**>(&p), h.size() * sizeof(T));
        cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    }
    explicit Dev(size_t n) {
        cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T));
        cudaMemset(p, 0, n * sizeof(T));
    }
    ~Dev() { cudaFree(p); }
    std::vector<T> get(size_t n) const {
        std::vector<T> h(n);
        cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost);
        return h;
    }
};

std::vector<float> randv(size_t n, float s) {
    std::vector<float> v(n);
    for (float& x : v) x = frand() * s;
    return v;
}

bool same(const std::vector<float>& a, const std::vector<float>& b) { return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0; }

// The decode attention of T rows from pos0, on a cache and on its paged twin.
bool decode_attn(int f, const DevPlanes& c, const DevPlanes& p, const int* dpt, int pos0, int T) {
    std::vector<float> q = randv(size_t(T) * NH * HD, 3.f), qfull = randv(size_t(T) * NH * 2 * HD, 2.f);
    Dev<float> dq(q), dqf(qfull), out_c(size_t(T) * NH * HD), out_p(size_t(T) * NH * HD), scr(dec::attn_scratch_floats(N_CTX, 8, NH));
    Dev<int> dpos(std::vector<int>{pos0});
    for (int paged = 0; paged < 2; ++paged) {
        const DevPlanes& d = paged ? p : c;
        const int* pt = paged ? dpt : nullptr;
        float* out = paged ? out_p.p : out_c.p;
        if (f == 0)
            dec::attention(dq.p, static_cast<const __half*>(d.k), static_cast<const __half*>(d.v), dqf.p, out, scr.p, dpos.p, T, N_CTX, VPITCH, NH, NKV, HD,
                           1.0f / 16.0f, nullptr, pt);
        else
            (f == 1 ? dec::attention_q8 : dec::attention_q4)(dq.p, static_cast<const int8_t*>(d.k), static_cast<const uint16_t*>(d.ks),
                                                               static_cast<const int8_t*>(d.v), static_cast<const uint16_t*>(d.vs), dqf.p, out, scr.p, dpos.p,
                                                               T, N_CTX, NH, NKV, HD, 1.0f / 16.0f, nullptr, pt);
    }
    cudaDeviceSynchronize();
    return same(out_c.get(q.size()), out_p.get(q.size()));
}

// The decode prep of T rows from pos0 on both caches; true when q_out and the whole caches (read back through the table) agree.
bool decode_prep(int f, const Map& m, DevPlanes& c, DevPlanes& p, const int* dpt, Planes* hc, Planes* hp, int pos0, int T) {
    const std::vector<float> qfull = randv(size_t(T) * NH * 2 * HD, 2.f), kb = randv(size_t(T) * NKV * HD, 2.f), vb = randv(size_t(T) * NKV * HD, 2.f),
                             qn = randv(HD, 1.f), kn = randv(HD, 1.f);
    Dev<float> dqf(qfull), dkb(kb), dvb(vb), dqn(qn), dkn(kn), qo_c(size_t(T) * NH * HD), qo_p(size_t(T) * NH * HD);
    Dev<int> dpos(std::vector<int>{pos0});
    for (int paged = 0; paged < 2; ++paged) {
        DevPlanes& d = paged ? p : c;
        const int* pt = paged ? dpt : nullptr;
        float* qo = paged ? qo_p.p : qo_c.p;
        if (f == 0)
            dec::attn_prep(dqf.p, dkb.p, dvb.p, dqn.p, dkn.p, qo, static_cast<__half*>(d.k), static_cast<__half*>(d.v), dpos.p, T, VPITCH, 1e-6, 1e7, ROPE, NH,
                           NKV, HD, nullptr, pt);
        else
            (f == 1 ? dec::attn_prep_q8 : dec::attn_prep_q4)(dqf.p, dkb.p, dvb.p, dqn.p, dkn.p, qo, static_cast<int8_t*>(d.k), static_cast<uint16_t*>(d.ks),
                                                               static_cast<int8_t*>(d.v), static_cast<uint16_t*>(d.vs), dpos.p, T, 1e-6, 1e7, ROPE, NH, NKV, HD,
                                                               nullptr, pt);
    }
    cudaDeviceSynchronize();
    c.download(hc);
    p.download(hp);
    Planes back = make_planes(f, N_CTX, false);
    convert(f, m, &back, hp, false);  // the pool read back through the table, as a contiguous cache
    const bool caches = back.k == hc->k && back.v == hc->v && back.ks == hc->ks && back.vs == hc->vs;
    return caches && same(qo_c.get(size_t(T) * NH * HD), qo_p.get(size_t(T) * NH * HD));
}

// The prompt: the batched prep of T rows at pos0 on both caches, then the windowed attention; true when the caches and the output agree.
bool prompt(int f, const Map& m, DevPlanes& c, DevPlanes& p, const int* dpt, Planes* hc, Planes* hp, int pos0, int T) {
    constexpr int WP = 512;
    const int total = pos0 + T;
    const std::vector<float> qfull = randv(size_t(T) * NH * 2 * HD, 2.f), kb = randv(size_t(T) * NKV * HD, 2.f), vb = randv(size_t(T) * NKV * HD, 2.f),
                             qn = randv(HD, 1.f), kn = randv(HD, 1.f);
    Dev<float> dqf(qfull), dkb(kb), dvb(vb), dqn(qn), dkn(kn), qo_c(size_t(T) * NH * HD), qo_p(size_t(T) * NH * HD), cs(size_t(T) * ROPE + 64), out_c(size_t(T) * NH * HD),
        out_p(size_t(T) * NH * HD), ml(size_t(T) * NH * 2);
    Dev<__half> kw(size_t(WP) * NKV * HD), vw(size_t(WP) * NKV * HD);
    ops::rope_table(cs.p, pos0, T, 1e7, ROPE, nullptr);
    for (int paged = 0; paged < 2; ++paged) {
        DevPlanes& d = paged ? p : c;
        const int* pt = paged ? dpt : nullptr;
        float* qo = paged ? qo_p.p : qo_c.p;
        float* out = paged ? out_p.p : out_c.p;
        if (f == 0)
            ops::attn_prep_rows(dqf.p, dkb.p, dvb.p, dqn.p, dkn.p, qo, static_cast<__half*>(d.k), static_cast<__half*>(d.v), cs.p, pos0, T, VPITCH, 1e-6, ROPE, NH,
                                NKV, HD, nullptr, pt);
        else
            (f == 1 ? ops::attn_prep_rows_q8 : ops::attn_prep_rows_q4)(dqf.p, dkb.p, dvb.p, dqn.p, dkn.p, qo, static_cast<int8_t*>(d.k), static_cast<uint16_t*>(d.ks),
                                                                         static_cast<int8_t*>(d.v), static_cast<uint16_t*>(d.vs), cs.p, pos0, T, 1e-6, ROPE, NH,
                                                                         NKV, HD, nullptr, pt);
        if (f == 0) {  // the f16 pool is read in place, one pass: the plain whole-cache kernel on the contiguous layout against the table
            ops::attention_rows(qo, static_cast<const __half*>(d.k), static_cast<const __half*>(d.v), dqf.p, out, pos0, T, N_CTX, VPITCH, NH, NKV, HD, 1.0f / 16.0f,
                                nullptr, true, pt);
            continue;
        }
        for (int k0 = 0; k0 < total; k0 += WP) {
            const int rows = std::min(WP, (total - k0 + 63) / 64 * 64);
            (f == 1 ? ops::kv_expand_window : ops::kv_expand_window_q4)(static_cast<const int8_t*>(d.k), static_cast<const uint16_t*>(d.ks),
                                                                        static_cast<const int8_t*>(d.v), static_cast<const uint16_t*>(d.vs), kw.p, vw.p, k0,
                                                                        rows, total, NKV, WP, nullptr, pt);
            const int flags = (k0 == 0 ? 1 : 0) | (k0 + WP >= total ? 2 : 0);
            ops::attention_rows_window(qo, kw.p, vw.p, dqf.p, out, pos0, T, k0, rows, WP, NH, NKV, HD, 1.0f / 16.0f, flags, ml.p, nullptr);
        }
    }
    cudaDeviceSynchronize();
    c.download(hc);
    p.download(hp);
    Planes back = make_planes(f, N_CTX, false);
    convert(f, m, &back, hp, false);
    const bool caches = back.k == hc->k && back.v == hc->v && back.ks == hc->ks && back.vs == hc->vs;
    return caches && same(qo_c.get(size_t(T) * NH * HD), qo_p.get(size_t(T) * NH * HD)) && same(out_c.get(size_t(T) * NH * HD), out_p.get(size_t(T) * NH * HD));
}

void run(int f) {
    const char* name = f == 0 ? "f16" : f == 1 ? "q8" : "q4";
    // the sequence's 6 chunks on shuffled pages of an 11-page pool
    std::vector<int> ids(N_PAGES);
    for (int i = 0; i < N_PAGES; ++i) ids[size_t(i)] = i;
    for (int i = N_PAGES - 1; i > 0; --i) std::swap(ids[size_t(i)], ids[urand() % uint32_t(i + 1)]);
    std::vector<int> pt(ids.begin(), ids.begin() + N_CHUNKS);
    std::vector<int32_t> pt32(N_CHUNKS + 1, -1);
    for (int i = 0; i < N_CHUNKS; ++i) pt32[size_t(i)] = pt[size_t(i)];
    const Map m{pt};
    Dev<int32_t> dpt(pt32);
    char what[160];

    auto fresh = [&](Planes* hc, Planes* hp) {
        *hc = make_planes(f, N_CTX, false);
        *hp = make_planes(f, N_PAGES * 256, true);
        randomise(f, hc);
        randomise(f, hp);  // pages the sequence does not own hold other rows
        convert(f, m, hc, hp, true);
    };
    {
        Planes hc, hp;
        fresh(&hc, &hp);
        DevPlanes c, p;
        c.upload(hc);
        p.upload(hp);
        std::snprintf(what, sizeof what, "%s: decode attention, 1 row at the last position, paged = contiguous bit for bit", name);
        CHECK(decode_attn(f, c, p, dpt.p, N_CTX - 70, 1), what);
        std::snprintf(what, sizeof what, "%s: decode attention, an 8-row window at the end", name);
        CHECK(decode_attn(f, c, p, dpt.p, N_CTX - 70 - 7, 8), what);
        std::snprintf(what, sizeof what, "%s: decode attention, an 8-row window across the chunk boundary 256", name);
        CHECK(decode_attn(f, c, p, dpt.p, 252, 8), what);
        std::snprintf(what, sizeof what, "%s: decode attention, one row at a chunk's first and last position", name);
        CHECK(decode_attn(f, c, p, dpt.p, 512, 1) && decode_attn(f, c, p, dpt.p, 767, 1), what);
    }
    {
        Planes hc, hp;
        fresh(&hc, &hp);
        DevPlanes c, p;
        c.upload(hc);
        p.upload(hp);
        std::snprintf(what, sizeof what, "%s: decode prep, 8 rows across a chunk boundary: the rows land where the table says", name);
        CHECK(decode_prep(f, m, c, p, dpt.p, &hc, &hp, 252, 8), what);
        std::snprintf(what, sizeof what, "%s: decode prep, one row in a later chunk", name);
        CHECK(decode_prep(f, m, c, p, dpt.p, &hc, &hp, 1100, 1), what);
    }
    for (auto [pos0, T] : {std::pair<int, int>{200, 130}, {1000, 90}}) {
        Planes hc, hp;
        fresh(&hc, &hp);
        DevPlanes c, p;
        c.upload(hc);
        p.upload(hp);
        std::snprintf(what, sizeof what, "%s: prompt prep + attention, %d rows at %d: caches and output equal the contiguous path", name, T, pos0);
        CHECK(prompt(f, m, c, p, dpt.p, &hc, &hp, pos0, T), what);
    }
}

}  // namespace

int main() {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        std::printf("SKIP cuda_paged_test: no CUDA device\n");
        return 0;
    }
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    if (free_b < (512ull << 20)) {
        std::printf("SKIP cuda_paged_test: %.2f GB free\n", double(free_b) / 1e9);
        return 0;
    }
    for (int f = 0; f < 3; ++f) run(f);
    return limbo::test::finish("cuda_paged_test");
}
