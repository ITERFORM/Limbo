// Copyright (c) 2026 ITERFORM. All rights reserved.
// src/vision_ref.cpp — the CPU reference of the vision tower (docs/vision.md §3): fp32 arithmetic, BF16 weights widened exactly.
// It is the yardstick for the GPU tower and is itself measured against llama.cpp (oracle/oracle_vision.cpp, tools/vision_check).
#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <thread>

#include "limbo/vision.hpp"

namespace limbo::vision {

namespace {

template <class F>
void parallel_for(int n, int threads, F&& f) {  // f(lo, hi) over a static split of [0, n)
    threads = std::max(1, std::min(threads, n));
    if (threads == 1) {
        f(0, n);
        return;
    }
    std::vector<std::thread> pool;
    for (int t = 1; t < threads; ++t) pool.emplace_back([&, t] { f(int(int64_t(n) * t / threads), int(int64_t(n) * (t + 1) / threads)); });
    f(0, int(int64_t(n) / threads));
    for (std::thread& th : pool) th.join();
}

bool g_round = false;
DebugSink g_sink = nullptr;
void* g_sink_user = nullptr;
void emit(const std::string& name, const std::vector<float>& v) {
    if (g_sink) g_sink(name.c_str(), v.data(), v.size(), g_sink_user);
}

// Round-to-nearest-even to BF16 and to F16, back to float.
inline float round_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fff + ((u >> 16) & 1);
    u &= 0xffff0000u;
    std::memcpy(&f, &u, 4);
    return f;
}
inline float round_f16(float f) { return static_cast<float>(static_cast<_Float16>(f)); }
void round_all(std::vector<float>& v, float (*fn)(float)) {
    for (float& x : v) x = fn(x);
}

inline float bf16(uint16_t b) {
    const uint32_t u = uint32_t(b) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

std::vector<float> widen(const Mat& m, int threads) {
    std::vector<float> w(size_t(m.n) * size_t(m.k));
    parallel_for(int(m.n), threads, [&](int lo, int hi) {
        for (size_t i = size_t(lo) * size_t(m.k); i < size_t(hi) * size_t(m.k); ++i) w[i] = bf16(m.p[i]);
    });
    return w;
}

__attribute__((target("avx2,fma"))) inline float hsum8(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}

// c[m][n] = sum_k a[m][k] * w[n][k] for n in [n0, n1), every m: 4 rows by 2 columns per step on AVX2.
__attribute__((target("avx2,fma"))) void gemm_cols_avx2(const float* a, int M, int K, const float* w, int n0, int n1, float* c, int ldc) {
    const int K8 = K & ~7;
    for (int n = n0; n < n1; n += 2) {
        const int nn = std::min(2, n1 - n);
        const float* w0 = w + size_t(n) * size_t(K);
        const float* w1 = nn > 1 ? w0 + K : w0;
        for (int m = 0; m < M; m += 4) {
            const int mm = std::min(4, M - m);
            const float* a0 = a + size_t(m) * size_t(K);
            const float* ar[4] = {a0, mm > 1 ? a0 + K : a0, mm > 2 ? a0 + 2 * size_t(K) : a0, mm > 3 ? a0 + 3 * size_t(K) : a0};
            __m256 acc[4][2];
            for (auto& r : acc) r[0] = r[1] = _mm256_setzero_ps();
            for (int k = 0; k < K8; k += 8) {
                const __m256 b0 = _mm256_loadu_ps(w0 + k), b1 = _mm256_loadu_ps(w1 + k);
                for (int i = 0; i < 4; ++i) {
                    const __m256 x = _mm256_loadu_ps(ar[i] + k);
                    acc[i][0] = _mm256_fmadd_ps(x, b0, acc[i][0]);
                    acc[i][1] = _mm256_fmadd_ps(x, b1, acc[i][1]);
                }
            }
            for (int i = 0; i < mm; ++i)
                for (int j = 0; j < nn; ++j) {
                    float s = hsum8(acc[i][j]);
                    const float* wr = j ? w1 : w0;
                    for (int k = K8; k < K; ++k) s += ar[i][k] * wr[k];
                    c[size_t(m + i) * size_t(ldc) + size_t(n + j)] = s;
                }
        }
    }
}

void gemm_cols_generic(const float* a, int M, int K, const float* w, int n0, int n1, float* c, int ldc) {
    for (int n = n0; n < n1; ++n)
        for (int m = 0; m < M; ++m) {
            float s = 0;
            for (int k = 0; k < K; ++k) s += a[size_t(m) * size_t(K) + size_t(k)] * w[size_t(n) * size_t(K) + size_t(k)];
            c[size_t(m) * size_t(ldc) + size_t(n)] = s;
        }
}

// c = a * w^T + bias; a is M x K, w is N x K (already widened), c is M x N.
void gemm(const float* a_in, int M, int K, const float* w, int N, const float* bias, float* c, int threads) {
    std::vector<float> rounded;
    const float* a = a_in;
    if (g_round) {
        rounded.assign(a_in, a_in + size_t(M) * size_t(K));
        round_all(rounded, round_bf16);
        a = rounded.data();
    }
    static const bool avx = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    const int chunks = (N + 15) / 16;
    parallel_for(chunks, threads, [&](int lo, int hi) {
        const int n0 = lo * 16, n1 = std::min(N, hi * 16);
        if (n0 >= n1) return;
        if (avx) gemm_cols_avx2(a, M, K, w, n0, n1, c, N);
        else gemm_cols_generic(a, M, K, w, n0, n1, c, N);
    });
    if (bias)
        parallel_for(M, threads, [&](int lo, int hi) {
            for (int m = lo; m < hi; ++m)
                for (int n = 0; n < N; ++n) c[size_t(m) * size_t(N) + size_t(n)] += bias[n];
        });
}

void layer_norm(const float* x, int rows, int n, const float* w, const float* b, float eps, float* out, int threads) {
    parallel_for(rows, threads, [&](int lo, int hi) {
        for (int r = lo; r < hi; ++r) {
            const float* v = x + size_t(r) * size_t(n);
            double s = 0;
            for (int i = 0; i < n; ++i) s += v[i];
            const float mean = float(s / n);
            double q = 0;
            for (int i = 0; i < n; ++i) q += double(v[i] - mean) * double(v[i] - mean);
            const float scale = 1.0f / std::sqrt(float(q / n) + eps);
            float* o = out + size_t(r) * size_t(n);
            for (int i = 0; i < n; ++i) o[i] = (v[i] - mean) * scale * w[i] + b[i];
        }
    });
}

inline float gelu_exact_tanh(float x) { return 0.5f * x * (1.0f + std::tanh(0.79788456080286535588f * x * (1.0f + 0.044715f * x * x))); }
// llama.cpp's CPU gelu looks the value up in an F16 table (input and output rounded to F16, saturated beyond +-10).
inline float gelu_tanh(float x) {
    if (!g_round) return gelu_exact_tanh(x);
    if (x <= -10.0f) return 0.0f;
    if (x >= 10.0f) return x;
    return round_f16(gelu_exact_tanh(round_f16(x)));
}

// Token t (merge-group order) -> patch column x and row y.
inline void token_xy(int t, int nx, int* x, int* y) {
    const int g = t / 4, w = t % 4;
    *y = (g / nx) * 2 + w / 2;
    *x = (g % nx) * 2 + w % 2;
}

}  // namespace

void set_oracle_rounding(bool on) { g_round = on; }
void set_debug_sink(DebugSink f, void* user) {
    g_sink = f;
    g_sink_user = user;
}

bool embed_patches_ref(const Projector& pj, const Prepared& img, int threads, std::vector<float>* out, std::string* err) {
    const Hparams& hp = pj.hp();
    const int E = hp.embd, ps = hp.patch, P = img.px * img.py, K = ps * ps * 3;
    if (img.px < 2 || img.py < 2 || img.px % 2 || img.py % 2 || img.w != img.px * ps || img.h != img.py * ps) {
        *err = "the prepared image is not a whole number of merge groups";
        return false;
    }
    std::vector<float> chw(size_t(3) * size_t(img.w) * size_t(img.h));
    normalise_chw(img, hp, chw.data());
    // patch vectors in (c, kh, kw) order, tokens in merge-group order
    std::vector<float> a(size_t(P) * size_t(K));
    parallel_for(P, threads, [&](int lo, int hi) {
        for (int t = lo; t < hi; ++t) {
            int x, y;
            token_xy(t, img.nx, &x, &y);
            float* d = a.data() + size_t(t) * size_t(K);
            for (int c = 0; c < 3; ++c)
                for (int kh = 0; kh < ps; ++kh)
                    for (int kw = 0; kw < ps; ++kw)
                        *d++ = chw[(size_t(c) * size_t(img.h) + size_t(y * ps + kh)) * size_t(img.w) + size_t(x * ps + kw)];
        }
    });
    std::vector<float> w(size_t(E) * size_t(K));
    for (size_t i = 0; i < w.size(); ++i) w[i] = pj.patch_w0[i] + pj.patch_w1[i];  // both taps see the same image
    out->assign(size_t(P) * size_t(E), 0.f);
    gemm(a.data(), P, K, w.data(), E, pj.patch_b, out->data(), threads);
    // + the position table, bilinear with align-corners to py x px
    const int S = hp.pos_side;
    auto coord = [&](int i, int n, int* i0, int* i1, float* f) {
        const float sf = n > 1 && S > 1 ? float(n - 1) / float(S - 1) : float(n) / float(S);
        const float s = float(i) / sf;
        int a0 = int(std::floor(s));
        int a1 = a0 + 1;
        a0 = std::max(0, std::min(a0, S - 1));
        a1 = std::max(0, std::min(a1, S - 1));
        *i0 = a0;
        *i1 = a1;
        *f = std::max(0.0f, std::min(s - float(a0), 1.0f));
    };
    parallel_for(P, threads, [&](int lo, int hi) {
        for (int t = lo; t < hi; ++t) {
            int x, y, x0, x1, y0, y1;
            float dx, dy;
            token_xy(t, img.nx, &x, &y);
            coord(x, img.px, &x0, &x1, &dx);
            coord(y, img.py, &y0, &y1, &dy);
            const float* ta = pj.pos_embd + (size_t(y0) * size_t(S) + size_t(x0)) * size_t(E);
            const float* tb = pj.pos_embd + (size_t(y0) * size_t(S) + size_t(x1)) * size_t(E);
            const float* tc = pj.pos_embd + (size_t(y1) * size_t(S) + size_t(x0)) * size_t(E);
            const float* td = pj.pos_embd + (size_t(y1) * size_t(S) + size_t(x1)) * size_t(E);
            float* o = out->data() + size_t(t) * size_t(E);
            for (int e = 0; e < E; ++e) o[e] += ta[e] * (1 - dx) * (1 - dy) + tb[e] * dx * (1 - dy) + tc[e] * (1 - dx) * dy + td[e] * dx * dy;
        }
    });
    return true;
}

bool encode_ref(const Projector& pj, const Prepared& img, int threads, std::vector<float>* out, std::string* err) {
    const Hparams& hp = pj.hp();
    std::vector<float> x;
    if (!embed_patches_ref(pj, img, threads, &x, err)) return false;
    const int E = hp.embd, P = img.px * img.py, H = hp.heads, D = hp.head_dim(), F = hp.ffn, half = D / 2, quarter = half / 2;
    // rope tables per token: pairs (i, i + half); i < quarter rotates by the row, else by the column
    std::vector<float> cs(size_t(P) * size_t(half)), sn(size_t(P) * size_t(half));
    parallel_for(P, threads, [&](int lo, int hi) {
        for (int t = lo; t < hi; ++t) {
            int px_, py_;
            token_xy(t, img.nx, &px_, &py_);
            for (int i = 0; i < half; ++i) {
                const int pos = i < quarter ? py_ : px_;
                const double th = double(pos) * std::pow(10000.0, -2.0 * double(i % quarter) / double(half));
                cs[size_t(t) * size_t(half) + size_t(i)] = float(std::cos(th));
                sn[size_t(t) * size_t(half) + size_t(i)] = float(std::sin(th));
            }
        }
    });
    emit("inp_pos_emb", x);
    std::vector<float> xn(size_t(P) * size_t(E)), qkv(size_t(P) * size_t(3 * E)), att(size_t(P) * size_t(E)), hid(size_t(P) * size_t(F)),
        tmp(size_t(P) * size_t(E));
    const float scale = 1.0f / std::sqrt(float(D));
    int il = -1;
    for (const Layer& y : pj.layers()) {
        ++il;
        layer_norm(x.data(), P, E, y.ln1_w, y.ln1_b, hp.eps, xn.data(), threads);
        emit("ln1-" + std::to_string(il), xn);
        gemm(xn.data(), P, E, widen(y.qkv, threads).data(), 3 * E, y.qkv_b, qkv.data(), threads);
        parallel_for(P, threads, [&](int lo, int hi) {
            for (int t = lo; t < hi; ++t)
                for (int which = 0; which < 2; ++which)  // q and k
                    for (int h = 0; h < H; ++h) {
                        float* v = qkv.data() + size_t(t) * size_t(3 * E) + size_t(which) * size_t(E) + size_t(h) * size_t(D);
                        for (int i = 0; i < half; ++i) {
                            const float c = cs[size_t(t) * size_t(half) + size_t(i)], s = sn[size_t(t) * size_t(half) + size_t(i)];
                            const float a = v[i], b = v[i + half];
                            v[i] = a * c - b * s;
                            v[i + half] = b * c + a * s;
                        }
                    }
        });
        if (g_sink) {
            std::vector<float> qr(size_t(P) * size_t(E));
            for (int t = 0; t < P; ++t) std::copy_n(qkv.data() + size_t(t) * size_t(3 * E), E, qr.data() + size_t(t) * size_t(E));
            emit("Qcur_rope-" + std::to_string(il), qr);
        }
        if (g_round)
            parallel_for(P, threads, [&](int lo, int hi) {
                for (int t = lo; t < hi; ++t)
                    for (int i = 0; i < 3 * E; ++i) {
                        float& v = qkv[size_t(t) * size_t(3 * E) + size_t(i)];
                        v = round_f16(v);
                    }
            });
        parallel_for(P * H, threads, [&](int lo, int hi) {
            std::vector<float> sc(static_cast<size_t>(P), 0.0f);
            for (int qh = lo; qh < hi; ++qh) {
                const int t = qh / H, h = qh % H;
                const float* q = qkv.data() + size_t(t) * size_t(3 * E) + size_t(h) * size_t(D);
                float mx = -INFINITY;
                for (int j = 0; j < P; ++j) {
                    const float* k = qkv.data() + size_t(j) * size_t(3 * E) + size_t(E) + size_t(h) * size_t(D);
                    float s = 0;
                    for (int d = 0; d < D; ++d) s += q[d] * k[d];
                    sc[size_t(j)] = s * scale;
                    mx = std::max(mx, sc[size_t(j)]);
                }
                float sum = 0;
                for (int j = 0; j < P; ++j) {
                    sc[size_t(j)] = std::exp(sc[size_t(j)] - mx);
                    sum += sc[size_t(j)];
                }
                float* o = att.data() + size_t(t) * size_t(E) + size_t(h) * size_t(D);
                for (int d = 0; d < D; ++d) o[d] = 0;
                for (int j = 0; j < P; ++j) {
                    const float* v = qkv.data() + size_t(j) * size_t(3 * E) + size_t(2 * E) + size_t(h) * size_t(D);
                    const float p = sc[size_t(j)] / sum;
                    for (int d = 0; d < D; ++d) o[d] += p * v[d];
                }
            }
        });
        gemm(att.data(), P, E, widen(y.out, threads).data(), E, y.out_b, tmp.data(), threads);
        emit("attn_out-" + std::to_string(il), tmp);
        for (size_t i = 0; i < x.size(); ++i) x[i] += tmp[i];
        emit("ffn_inp-" + std::to_string(il), x);
        layer_norm(x.data(), P, E, y.ln2_w, y.ln2_b, hp.eps, xn.data(), threads);
        gemm(xn.data(), P, E, widen(y.up, threads).data(), F, y.up_b, hid.data(), threads);
        parallel_for(P, threads, [&](int lo, int hi) {
            for (size_t i = size_t(lo) * size_t(F); i < size_t(hi) * size_t(F); ++i) hid[i] = gelu_tanh(hid[i]);
        });
        gemm(hid.data(), P, F, widen(y.down, threads).data(), E, y.down_b, tmp.data(), threads);
        for (size_t i = 0; i < x.size(); ++i) x[i] += tmp[i];
        emit("layer_out-" + std::to_string(il), x);
    }
    layer_norm(x.data(), P, E, pj.post_w, pj.post_b, hp.eps, xn.data(), threads);
    emit("norm_b-" + std::to_string(hp.layers), xn);
    // merger: four consecutive rows are one 4E vector
    const int M = P / 4, K4 = 4 * E;
    std::vector<float> m0(size_t(M) * size_t(K4));
    gemm(xn.data(), M, K4, widen(pj.mm0, threads).data(), K4, pj.mm0_b, m0.data(), threads);
    parallel_for(M, threads, [&](int lo, int hi) {
        for (size_t i = size_t(lo) * size_t(K4); i < size_t(hi) * size_t(K4); ++i) m0[i] = gelu_tanh(m0[i]);
    });
    out->assign(size_t(M) * size_t(hp.out_dim), 0.f);
    gemm(m0.data(), M, K4, widen(pj.mm2, threads).data(), hp.out_dim, pj.mm2_b, out->data(), threads);
    return true;
}

}  // namespace limbo::vision
