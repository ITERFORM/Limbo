// Copyright (c) 2026 ITERFORM. All rights reserved.
// src/cuda/gpu_prefill.cu — the batched prompt pass on the int8 tensor cores.
// Owner: the prefill agent. Private state: gpu_impl.hpp (prefill section).

#include <cstdlib>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "gpu_impl.hpp"
#include "limbo/cuda/gemm.hpp"
#include "limbo/cuda/gemv.hpp"
#include "limbo/quants.hpp"
#include "limbo/types.hpp"
#include "ops.hpp"

namespace limbo::cuda {

namespace {
constexpr int kPfWin = 4096;  // positions of the cache expanded to f16 at a time when it is Q8 or Q4 (kwin / vwin: 4 MB each)
}  // namespace

bool Gpu::Impl::ensure_prefill(int cap) {
    if (pf.cap >= cap && pf.blob) return true;
    if (pf.blob) {
        cudaStreamSynchronize(stream);
        cudaFree(pf.blob);
        pf = Prefill();
    }
    const size_t E = size_t(c.n_embd), vn = size_t(c.n_v_head) * c.gdn_head, qw = size_t(c.n_head) * c.hd,
                 kvw = size_t(c.n_head_kv) * c.hd, T = size_t(cap), pairs = T * size_t(c.n_used);
    // the widest activation row a product reads (K of act_quant): the planes, their scales and the K-quants' sums are sized by it
    const size_t kmax = std::max({E, vn, qw, size_t(c.n_ff), size_t(c.n_ff_exp), size_t(c.n_ff_shexp), size_t(4096)});
    const size_t n_ffw = std::max(size_t(c.n_ff_shexp), size_t(c.n_ff));  // the shared expert's width, or the dense FFN's
    struct Carve {
        float** dst;
        size_t n;
    };
    Prefill& P = pf;
    // Three groups share one region: a layer's delta-net buffers, a layer's attention buffers and the routed experts' buffers are
    // never live at once (the out product has read the first two before the MoE norm, and stream2's share of them is joined
    // into the main stream before that; the next layer writes them after the combine). The rest lives through the whole chunk.
    const Carve keep[] = {
        {&P.x, T * E},          {&P.xn, T * E},         {&P.rlog, T * size_t(c.n_expert)}, {&P.post, T * E}, {&P.wts, pairs},
        {&P.gs, T * n_ffw}, {&P.us, T * n_ffw}, {&P.hs, T * n_ffw},
        {&P.shared, T * E},     {&P.sg, T},             {&P.xs, T * (kmax / 32) + 64}, {&P.xsum, T * (kmax / 16) + 64},
        {&P.rope, T * size_t(c.rope_dim) + 64},
    };
    const Carve gdn[] = {
        {&P.qkv, T * size_t(c.qkv_w)}, {&P.conv, T * size_t(c.qkv_w)}, {&P.qk, T * 2 * size_t(c.n_k_head) * c.gdn_head},
        {&P.alpha, T * size_t(c.dt_rank)}, {&P.beta, T * size_t(c.dt_rank)}, {&P.o, T * vn}, {&P.z, T * vn}, {&P.gated, T * vn},
    };
    const Carve att[] = {
        {&P.qfull, T * 2 * qw}, {&P.kbuf, T * kvw},     {&P.vbuf, T * kvw},      {&P.qbuf, T * qw},       {&P.attn, T * qw},
        {&P.aml, T * size_t(c.n_head) * 2 + 64}, {&P.kwin, size_t(kPfWin) * kvw / 2 + 64}, {&P.vwin, size_t(kPfWin) * kvw / 2 + 64},
    };
    const Carve moe[] = {{&P.h, pairs * size_t(c.n_ff_exp)}, {&P.dn, pairs * E}};
    auto span = [](const Carve* b, const Carve* e) {
        size_t n = 0;
        for (; b != e; ++b) n += (b->n + 63) / 64 * 64;
        return n;
    };
    const size_t n_keep = span(std::begin(keep), std::end(keep));
    const size_t n_shared = std::max({span(std::begin(gdn), std::end(gdn)), span(std::begin(att), std::end(att)), span(std::begin(moe), std::end(moe))});
    const size_t total = n_keep + n_shared;
    const size_t tiles_small = size_t(moe_max_tiles_small(c.n_expert)), tiles_big = size_t(moe_max_tiles_big(int(pairs), c.n_expert));
    auto up4 = [](size_t n) { return (n + 3) / 4 * 4; };
    const size_t ints = 3 * up4(pairs) + up4(T) + up4(size_t(c.n_expert)) + up4(size_t(c.n_expert) + 1) + 64 +
                        4 * tiles_small + 4 * tiles_big;
    const size_t xq_bytes = 3 * T * kmax + 256;  // planes x rows x K
    const size_t bytes = total * sizeof(float) + ints * sizeof(int32_t) + xq_bytes;
    if (cudaMalloc(&P.blob, bytes) != cudaSuccess) {
        cudaGetLastError();
        P = Prefill();
        return false;
    }
    P.bytes = bytes;
    P.cap = cap;
    float* f = static_cast<float*>(P.blob);
    auto lay = [&](const Carve* b, const Carve* e, size_t at) {
        for (; b != e; ++b) {
            *b->dst = f + at;
            at += (b->n + 63) / 64 * 64;
        }
    };
    lay(std::begin(keep), std::end(keep), 0);
    lay(std::begin(gdn), std::end(gdn), n_keep);
    lay(std::begin(att), std::end(att), n_keep);
    lay(std::begin(moe), std::end(moe), n_keep);
    int32_t* ip = reinterpret_cast<int32_t*>(f + total);
    int32_t* const ip0 = ip;
    auto take = [&](size_t n) {
        int32_t* r = ip;
        ip += up4(n);
        return r;
    };
    P.ids = take(pairs);
    P.L.list_tok = take(pairs);
    P.L.pos_of = take(pairs);
    P.tok = take(T);
    P.L.cnt = take(size_t(c.n_expert));
    P.L.start = take(size_t(c.n_expert) + 1);
    P.L.n_tiles = take(64);
    P.L.tiles_small = reinterpret_cast<int4*>(ip);
    ip += 4 * tiles_small;
    P.L.tiles_big = reinterpret_cast<int4*>(ip);
    ip += 4 * tiles_big;
    P.xq = reinterpret_cast<int8_t*>(ip0 + ints);
    return ip <= ip0 + ints;
}

bool Gpu::Impl::run_chunk(const int32_t* toks, int T, int pos0, int df_r0) {
    cudaStream_t s = stream;
    const int E = c.n_embd, nt = pf_planes;
    const bool q8 = kv.fmt != KvFormat::F16, q4 = kv.fmt == KvFormat::Q4;  // q8: a quantised cache (Q8 or Q4)
    const int vn = c.n_v_head * c.gdn_head, qw = c.n_head * c.hd, kvw = c.n_head_kv * c.hd;
    const int pairs = T * c.n_used;
    const float epsf = static_cast<float>(c.eps);
    const float gdn_scale = static_cast<float>(1.0 / std::sqrt(double(c.gdn_head)));
    const float gdn_eps = static_cast<float>(c.eps / c.gdn_head);
    const float attn_scale = static_cast<float>(1.0 / std::sqrt(double(c.hd)));
    const int* pt = this->pt();  // the live slot's page table
    Prefill& P = pf;
    bool ok = true;
    std::vector<std::pair<const char*, cudaEvent_t>> evs;
    auto mark = [&](const char* name) {
        if (!pf_prof) return;
        cudaEvent_t e;
        cudaEventCreate(&e);
        cudaEventRecord(e, s);
        evs.emplace_back(name, e);
    };
    mark("start");
    if (cudaMemcpyAsync(P.tok, toks, size_t(T) * sizeof(int32_t), cudaMemcpyHostToDevice, s) != cudaSuccess) return false;
    ok &= embed_rows(embd.type, embd.p, P.tok, T, P.x, E, s);
    mark("embed");
    ops::rope_table(P.rope, pos0, T, c.rope_base, c.rope_dim, s);
    mark("rope table");

    // The second stream carries the work that has no dependency on the critical path: the delta net's alpha/beta
    // and z products beside the qkv product and the scan, the shared expert beside the router and the routed
    // experts. Profiling runs everything on the main stream so that every phase is timed alone.
    struct Ev {
        cudaEvent_t e[5] = {};
        Ev() {
            for (cudaEvent_t& x : e) cudaEventCreateWithFlags(&x, cudaEventDisableTiming);
        }
        ~Ev() {
            for (cudaEvent_t x : e) cudaEventDestroy(x);
        }
    } ev;
    enum { kFork, kAb, kZ, kFork2, kShared };
    const bool par = pf_overlap && !pf_prof;
    cudaStream_t s2 = par ? stream2 : s;
    auto fork = [&](int which) {
        if (!par) return;
        cudaEventRecord(ev.e[which], s);
        cudaStreamWaitEvent(s2, ev.e[which], 0);
    };

    // y = W q for the activation rows that were just quantised into pf.xq / pf.xs
    auto mm = [&](const W& w, float* out, int n_out, int n_in, bool acc) {
        return gemm(w.type, w.p, P.xq, P.xs, nt, T, out, n_out, n_in, acc, s, P.xsum);
    };
    // the K-quants' minimum term needs the activation block sums: redo the quantisation with them when a product of the group does
    auto sums = [&](std::initializer_list<const W*> ws, const float* src, int K) {
        bool need = false;
        for (const W* w : ws) need = need || (w->p && gemm_needs_sums(w->type));
        return !need || act_quant_sums(src, T, K, nt, P.xq, P.xs, P.xsum, s);
    };
    for (int l = 0; l < c.n_layer_main; ++l) {
        const Layer& y = L[size_t(l)];
        if (df && df_r0 >= 0) df->capture(l, P.x + size_t(df_r0) * size_t(E), T - df_r0, 1, s);  // the drafter reads some layers' inputs
        ok &= rmsnorm_quant(P.x, y.attn_norm, P.xn, T, E, epsf, nt, P.xq, P.xs, s);
        ok &= y.full ? sums({&y.q, &y.k, &y.v}, P.xn, E) : sums({&y.gate, &y.qkv}, P.xn, E);
        mark("norm+quant");
        if (y.full) {
            ok &= mm(y.q, P.qfull, 2 * qw, E, false);
            ok &= mm(y.k, P.kbuf, kvw, E, false);
            ok &= mm(y.v, P.vbuf, kvw, E, false);
            mark("attn qkv gemm");
            if (q8)
                (q4 ? ops::attn_prep_rows_q4 : ops::attn_prep_rows_q8)(P.qfull, P.kbuf, P.vbuf, y.qn, y.kn, P.qbuf, kv.kq[size_t(l)],
                                                                      kv.ks[size_t(l)], kv.vq[size_t(l)], kv.vs[size_t(l)], P.rope, pos0, T,
                                                                      c.eps, c.rope_dim, c.n_head, c.n_head_kv, c.hd, s, pt);
            else
                ops::attn_prep_rows(P.qfull, P.kbuf, P.vbuf, y.qn, y.kn, P.qbuf, kc[size_t(l)], vc[size_t(l)], P.rope, pos0, T, 0, c.eps,
                                    c.rope_dim, c.n_head, c.n_head_kv, c.hd, s, pt);
            mark("attn prep");
            if (!q8) {
                // the f16 pool is read in place through the page table: one pass, the bits of the contiguous kernel
                ops::attention_rows(P.qbuf, kc[size_t(l)], vc[size_t(l)], P.qfull, P.attn, pos0, T, n_ctx, 0, c.n_head, c.n_head_kv, c.hd, attn_scale, s,
                                    nt >= 2, pt);
            } else {
                // a Q8 / Q4 pool is dequantised into f16 windows: expand 4096 positions, attend them, the next window continues the softmax
                const int total = pos0 + T;
                for (int k0 = 0; k0 < total; k0 += kPfWin) {
                    const int rows = std::min(kPfWin, (total - k0 + 63) / 64 * 64);
                    (q4 ? ops::kv_expand_window_q4 : ops::kv_expand_window)(kv.kq[size_t(l)], kv.ks[size_t(l)], kv.vq[size_t(l)], kv.vs[size_t(l)],
                                                                             reinterpret_cast<__half*>(P.kwin), reinterpret_cast<__half*>(P.vwin), k0, rows,
                                                                             total, c.n_head_kv, kPfWin, s, pt);
                    const int flags = (k0 == 0 ? 1 : 0) | (k0 + kPfWin >= total ? 2 : 0);
                    ops::attention_rows_window(P.qbuf, reinterpret_cast<const __half*>(P.kwin), reinterpret_cast<const __half*>(P.vwin), P.qfull, P.attn,
                                               pos0, T, k0, rows, kPfWin, c.n_head, c.n_head_kv, c.hd, attn_scale, flags, P.aml, s, nt >= 2);
                }
            }
            mark("attention");
            ok &= act_quant(P.attn, T, qw, nt, P.xq, P.xs, s);
            ok &= sums({&y.o}, P.attn, qw);
            mark("act_quant");
            ok &= mm(y.o, P.x, E, qw, true);
            mark("attn out gemm");
        } else {
            fork(kFork);  // alpha, beta and z beside the qkv product
            ok &= gemm_f32(reinterpret_cast<const float*>(y.alpha.p), P.xn, P.alpha, T, c.dt_rank, E, s2);
            ok &= gemm_f32(reinterpret_cast<const float*>(y.beta.p), P.xn, P.beta, T, c.dt_rank, E, s2);
            if (!par) mark("gdn alpha/beta f32");
            if (par) cudaEventRecord(ev.e[kAb], s2);
            ok &= gemm(y.gate.type, y.gate.p, P.xq, P.xs, nt, T, P.z, vn, E, false, s2, P.xsum);
            if (!par) mark("gdn z gemm");
            if (par) {
                cudaEventRecord(ev.e[kZ], s2);
            }
            ok &= mm(y.qkv, P.qkv, c.qkv_w, E, false);
            mark("gdn qkv gemm");
            ops::conv_silu_rows(P.qkv, hist[size_t(l)], y.conv, P.conv, c.qkv_w, T, s);
            mark("gdn conv");
            ops::head_norm_rows(P.conv, c.qkv_w, P.qk, 2 * c.n_k_head * c.gdn_head, T, 2 * c.n_k_head, c.gdn_head, gdn_eps,
                                gdn_scale, s);
            mark("gdn head norm");
            if (par) cudaStreamWaitEvent(s, ev.e[kAb], 0);
            ops::gdn_coef_rows(P.alpha, P.beta, c.dt_rank, y.dtb, y.ssm_a, T, c.dt_rank, s);
            ops::gdn_scan_rows(S[size_t(l)], P.qk, 2 * c.n_k_head * c.gdn_head, P.conv + 2 * c.n_k_head * c.gdn_head,
                               c.qkv_w, P.alpha, P.beta, c.dt_rank, P.o, vn, T, c.n_v_head, c.n_k_head, c.gdn_head,
                               c.dt_rank, gdn_scale, s);
            mark("gdn scan");
            if (par) cudaStreamWaitEvent(s, ev.e[kZ], 0);
            ops::gdn_gate_rows(P.o, P.z, y.ssm_norm, P.gated, T * c.n_v_head, c.gdn_head, c.eps, s);
            mark("gdn gate");
            ok &= act_quant(P.gated, T, vn, nt, P.xq, P.xs, s);
            ok &= sums({&y.out}, P.gated, vn);
            mark("gdn quant");
            ok &= mm(y.out, P.x, E, vn, true);
            mark("gdn out gemm");
        }

        // the FFN on the post-attention norm
        ok &= rmsnorm_quant(P.x, y.post_norm, P.post, T, E, epsf, nt, P.xq, P.xs, s);
        if (c.n_ff > 0) {
            // dense SwiGLU: x += down(silu(gate(post)) * up(post))
            ok &= sums({&y.fgate, &y.fup}, P.post, E);
            mark("ffn norm+quant");
            ok &= mm(y.fgate, P.gs, c.n_ff, E, false);
            ok &= mm(y.fup, P.us, c.n_ff, E, false);
            mark("ffn gate/up");
            ops::swiglu(P.gs, P.us, P.hs, T * c.n_ff, s);
            ok &= act_quant(P.hs, T, c.n_ff, nt, P.xq, P.xs, s);
            ok &= sums({&y.fdown}, P.hs, c.n_ff);
            ok &= mm(y.fdown, P.x, E, c.n_ff, true);
            mark("ffn down");
            if (trace) {
                if (trace->size() < size_t(c.n_layer_main)) trace->resize(size_t(c.n_layer_main));
                std::vector<float>& t = (*trace)[size_t(l)];
                t.resize(size_t(E));
                cudaStreamSynchronize(s);
                cudaMemcpy(t.data(), P.x + size_t(T - 1) * size_t(E), size_t(E) * sizeof(float), cudaMemcpyDeviceToHost);
            }
            continue;
        }
        mark("moe norm+quant");
        fork(kFork2);  // the shared expert beside the router and the routed experts
        ok &= gemm(y.sgate.type, y.sgate.p, P.xq, P.xs, nt, T, P.gs, c.n_ff_shexp, E, false, s2);
        ok &= gemm(y.sup.type, y.sup.p, P.xq, P.xs, nt, T, P.us, c.n_ff_shexp, E, false, s2);
        ops::swiglu(P.gs, P.us, P.hs, T * c.n_ff_shexp, s2);
        ops::shexp_gate_rows(y.shgate, P.post, P.sg, E, T, s2);
        if (!par) mark("shexp gate/up");
        if (par) cudaEventRecord(ev.e[kShared], s2);
        ok &= gemm_f32(reinterpret_cast<const float*>(y.router.p), P.post, P.rlog, T, c.n_expert, E, s);
        mark("router f32");
        ops::router_topk_rows(P.rlog, c.n_expert, c.n_used, T, P.ids, P.wts, s);
        ok &= moe_build(P.ids, T, c.n_expert, c.n_used, P.L, s);
        mark("topk+lists");
        ok &= moe_gate_up(y.ge.type, y.ge.p, y.ue.p, y.ge_bytes, P.xq, P.xs, nt, T, pairs, P.L, c.n_expert, P.h,
                          c.n_ff_exp, E, s);
        mark("moe gate/up");
        if (par) cudaStreamWaitEvent(s, ev.e[kShared], 0);  // the shared expert has read xq; h quant overwrites it
        ok &= act_quant(P.h, pairs, c.n_ff_exp, nt, P.xq, P.xs, s);
        mark("h quant");
        ok &= moe_down(y.de.type, y.de.p, y.de_bytes, P.xq, P.xs, nt, pairs, P.L, c.n_expert, P.dn, E, c.n_ff_exp, s);
        mark("moe down");
        ok &= act_quant(P.hs, T, c.n_ff_shexp, nt, P.xq, P.xs, s);
        ok &= mm(y.sdown, P.shared, E, c.n_ff_shexp, false);
        mark("shexp down");
        ops::moe_combine_rows(P.x, P.dn, P.L.pos_of, P.wts, P.shared, P.sg, E, c.n_used, T, s);
        mark("combine");
        if (trace) {
            if (trace->size() < size_t(c.n_layer_main)) trace->resize(size_t(c.n_layer_main));
            std::vector<float>& t = (*trace)[size_t(l)];
            t.resize(size_t(E));
            cudaStreamSynchronize(s);
            cudaMemcpy(t.data(), P.x + size_t(T - 1) * size_t(E), size_t(E) * sizeof(float), cudaMemcpyDeviceToHost);
        }
    }
    if (pf_prof) {
        cudaStreamSynchronize(s);
        for (size_t i = 1; i < evs.size(); ++i) {
            float ms = 0;
            cudaEventElapsedTime(&ms, evs[i - 1].second, evs[i].second);
            bool found = false;
            for (auto& kv : *pf_prof)
                if (kv.first == evs[i].first) {
                    kv.second += ms;
                    found = true;
                }
            if (!found) pf_prof->emplace_back(evs[i].first, ms);
        }
        for (auto& e : evs) cudaEventDestroy(e.second);
    }
    return ok && cudaGetLastError() == cudaSuccess;
}

void Gpu::set_prefill_streams(bool on) { p_->pf_overlap = on; }

bool Gpu::set_prefill(int planes, int chunk) {
    if (planes < 0 || planes > 3 || chunk < 16 || chunk > 4096) return false;
    p_->pf_planes = planes;
    p_->pf_chunk = chunk;
    return true;
}
int Gpu::prefill_planes() const { return p_->pf_planes; }
int Gpu::prefill_chunk() const { return p_->q4e ? Impl::kQ4ePfMax : p_->pf_chunk; }   // q4e: the layer-major pass takes whole pieces

bool Gpu::prefill(const int32_t* tokens, int n, int pos0, bool head_on_last) {
    Impl& I = *p_;
    if (n < 0 || pos0 < 0 || pos0 + n > I.n_ctx) return false;
    if (n == 0) return true;
    if (!I.pages_for(pos0, pos0 + n)) return false;
    // The scratch is sized for a whole chunk at the first batched call whatever n is, so a caller that feeds short
    // pieces first does not make the next long one reallocate it.
    const bool batched = I.pf_planes > 0 && I.pf_able && n >= 8 && I.ensure_prefill(I.pf_chunk);
    if (!batched) {
        // qwen4exp has no int8 prefill recipe -- pf_able is false for it, because that path has no HC, PLE or GDN
        // scan -- but it does have the verify window, and the window IS a batched prefill: T rows with every dense
        // weight read once, one host round trip a layer instead of T, and rows the gate says are the sequential
        // rows (T=1 and T=2 bit-identical, T=4 4.26e-05, commit(n) exact). So a prompt goes through it a window at
        // a time: 60 rows is 8 passes, not 60 round trips through the whole 48-layer stack.
        //
        // The window's own guard is the same one the single-row step applies, evaluated at its LAST row -- it
        // refuses when (pos0 + T) / idx_kpool exceeds the sparse selection's budget, and the single step at
        // pos0 + T - 1 computes that same quotient and refuses with it. So the window never refuses where the loop
        // below would have succeeded, and there is nothing to fall back from.
        if (I.q4e) {
            // the layer-major prompt pass for all but the last row when it needs the head (that one takes the window)
            int done = 0;
            const int body = head_on_last ? n - 1 : n;
            // The layer-major pass streams every non-resident expert through the stage (~17 GB on the Coder, ~2.9 s at
            // 6.5 GB/s) whatever the piece's size; below LIMBO_Q4E_PF_MIN rows (256) the verify windows are cheaper: they
            // read only the experts the rows route to, the misses on the lane (120 tokens: 2.9 s through the pass).
            static const int pf_min = std::getenv("LIMBO_Q4E_PF_MIN") ? std::max(1, std::atoi(std::getenv("LIMBO_Q4E_PF_MIN"))) : 256;
            if (body >= pf_min && !std::getenv("LIMBO_Q4E_NO_PF") && I.q4e_lend()) {
                for (; done < body;) {
                    const int T = std::min(Impl::kQ4ePfMax, body - done);
                    const int nn = std::min(Impl::kQ4ePfMax, body - done - T);
                    if (!I.prefill_q4e(tokens + done, T, pos0 + done, nn >= 64 ? tokens + done + T : nullptr, nn >= 64 ? nn : 0))
                        return false;
                    // the MTP block's K/V over the piece: its final residuals and the token after each
                    if (I.q4mtp.ready && !I.q4e_mtp_prompt(I.qpf.st, tokens + done + 1, std::min(T, n - 1 - done), pos0 + done)) return false;
                    done += T;
                }
            }
            if (done > 0) I.q4e_rebalance();
            tokens += done;
            pos0 += done;
            n -= done;
            for (int off = 0; off < n; off += Impl::kQ4eRows) {
                const int T = std::min(Impl::kQ4eRows, n - off);
                if (!I.step_q4e_rows(tokens + off, T, pos0 + off, true)) return false;
                if (I.q4mtp.ready && off + 1 < n &&
                    !I.q4e_mtp_front(I.qrw.st, tokens + off + 1, std::min(T, n - 1 - off), pos0 + off))
                    return false;
                if (head_on_last && off + T == n) {
                    // The window computes every row's head (acceptance needs all of them); the caller reads the
                    // last row's, which is what the single-step loop left in I.logits too.
                    const size_t nv = size_t(I.c.n_vocab);
                    if (cudaMemcpyAsync(I.logits, I.qrw.logits + size_t(T - 1) * nv, nv * sizeof(float),
                                        cudaMemcpyDeviceToDevice, I.stream) != cudaSuccess)
                        return false;
                }
            }
            return cudaGetLastError() == cudaSuccess;
        }
        for (int i = 0; i < n; ++i)
            if (!step(tokens[i], pos0 + i, head_on_last && i == n - 1)) return false;
        return true;
    }
    const int step_t = std::min(I.pf_chunk, I.pf.cap);
    for (int off = 0; off < n; off += step_t) {
        const int T = std::min(step_t, n - off);
        // the drafter reads the window below the end of the prompt: the rows of this chunk at or past df_from, at most its stage
        const int p0 = pos0 + off;
        const int r0 = I.df ? std::max({0, I.df_from - p0, T - I.df_stage_rows}) : T;
        if (!I.run_chunk(tokens + off, T, p0, r0 < T ? r0 : -1)) return false;
        if (I.df && r0 < T) I.df_ingest(1, 0, T - r0, p0 + r0);
        if (off + T == n && head_on_last) {
            const Cfg& c = I.c;
            ops::rmsnorm(I.pf.x + size_t(T - 1) * size_t(c.n_embd), I.out_norm, I.rn, c.n_embd, static_cast<float>(c.eps), I.stream);
            if (!I.gemv(I.head, I.rn, I.logits, c.n_vocab, c.n_embd)) return false;
        }
    }
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace limbo::cuda
