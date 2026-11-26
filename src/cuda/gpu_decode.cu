// Copyright (c) 2026 ITERFORM. All rights reserved.
// src/cuda/gpu_decode.cu — one token per step: the decode pass, replayed as a captured CUDA graph.
// Owner: the decode agent. Private state: gpu_impl.hpp (decode section).

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
#include "dec_ops.hpp"
#include "ops.hpp"

namespace limbo::cuda {

namespace {
constexpr uint32_t F32 = static_cast<uint32_t>(Type::F32);
}

bool Gpu::Impl::launch_step(bool head) {
    Impl& I = *this;
    const Cfg& c = I.c;
    cudaStream_t s = I.stream;
    const float epsf = static_cast<float>(c.eps);
    const float gdn_scale = static_cast<float>(1.0 / std::sqrt(double(c.gdn_head)));
    const float gdn_eps = static_cast<float>(c.eps / c.gdn_head);
    const float attn_scale = static_cast<float>(1.0 / std::sqrt(double(c.hd)));
    const int vn = c.n_v_head * c.gdn_head;
    const int qw = c.n_head * c.hd;
    const int* pt = I.pt();  // the live slot's page table (the graph bakes the pointer in; the slot's graphs are its own)
    bool ok = true;
    std::vector<std::pair<const char*, cudaEvent_t>> evs;
    auto mark = [&](const char* name) {
        if (!dec_prof) return;
        cudaEvent_t e;
        cudaEventCreate(&e);
        cudaEventRecord(e, s);
        evs.emplace_back(name, e);
    };
    // Parallel branches (graph forks): off while a profile or a trace is on, since those synchronise inside the pass.
    const bool par = I.da.s2 && !I.dec_prof && !I.trace && !I.moe_trace;
    cudaStream_t s2 = par ? I.da.s2 : s;
    int evi = 0;
    auto fork = [&]() {  // work queued on s2 from here on waits for everything queued on s so far
        if (!par) return;
        cudaEvent_t e = I.da.ev[evi++ & 7];
        cudaEventRecord(e, s);
        cudaStreamWaitEvent(s2, e, 0);
    };
    auto join = [&]() {  // work queued on s from here on waits for everything queued on s2 so far
        if (!par) return;
        cudaEvent_t e = I.da.ev[evi++ & 7];
        cudaEventRecord(e, s2);
        cudaStreamWaitEvent(s, e, 0);
    };
    mark("start");
    ok &= embed_row(I.embd.type, I.embd.p, I.d_args, I.x, c.n_embd, s);
    mark("embed");

    for (int l = 0; l < c.n_layer_main; ++l) {
        const Layer& y = I.L[size_t(l)];
        if (I.df) I.df->capture(l, I.x, 1, 0, s);  // the drafter reads some layers' inputs
        ok &= dec::rmsnorm(I.x, y.attn_norm, I.xn, c.n_embd, epsf, 1, s);
        mark("norm");
        if (y.full) {
            ok &= I.gemv(y.q, I.xn, I.qfull, 2 * qw, c.n_embd);
            ok &= I.gemv(y.k, I.xn, I.kbuf, c.n_head_kv * c.hd, c.n_embd);
            ok &= I.gemv(y.v, I.xn, I.vbuf, c.n_head_kv * c.hd, c.n_embd);
            mark("attn q/k/v gemv");
            if (I.kv.fmt == KvFormat::Q4) {
                dec::attn_prep_q4(I.qfull, I.kbuf, I.vbuf, y.qn, y.kn, I.qbuf, I.kv.kq[size_t(l)], I.kv.ks[size_t(l)], I.kv.vq[size_t(l)],
                                  I.kv.vs[size_t(l)], I.d_args + 1, 1, c.eps, c.rope_base, c.rope_dim, c.n_head, c.n_head_kv, c.hd, s, pt);
                ok &= dec::attention_q4(I.qbuf, I.kv.kq[size_t(l)], I.kv.ks[size_t(l)], I.kv.vq[size_t(l)], I.kv.vs[size_t(l)], I.qfull, I.attn,
                                        I.attn_scr, I.d_args + 1, 1, I.n_ctx, c.n_head, c.n_head_kv, c.hd, attn_scale, s, pt);
            } else if (I.kv.fmt == KvFormat::Q8) {
                dec::attn_prep_q8(I.qfull, I.kbuf, I.vbuf, y.qn, y.kn, I.qbuf, I.kv.kq[size_t(l)], I.kv.ks[size_t(l)], I.kv.vq[size_t(l)],
                                  I.kv.vs[size_t(l)], I.d_args + 1, 1, c.eps, c.rope_base, c.rope_dim, c.n_head, c.n_head_kv, c.hd, s, pt);
                ok &= dec::attention_q8(I.qbuf, I.kv.kq[size_t(l)], I.kv.ks[size_t(l)], I.kv.vq[size_t(l)], I.kv.vs[size_t(l)], I.qfull, I.attn,
                                        I.attn_scr, I.d_args + 1, 1, I.n_ctx, c.n_head, c.n_head_kv, c.hd, attn_scale, s, pt);
            } else {
                dec::attn_prep(I.qfull, I.kbuf, I.vbuf, y.qn, y.kn, I.qbuf, I.kc[size_t(l)], I.vc[size_t(l)], I.d_args + 1, 1,
                               0, c.eps, c.rope_base, c.rope_dim, c.n_head, c.n_head_kv, c.hd, s, pt);
                ok &= dec::attention(I.qbuf, I.kc[size_t(l)], I.vc[size_t(l)], I.qfull, I.attn, I.attn_scr, I.d_args + 1, 1, I.n_ctx,
                                     0, c.n_head, c.n_head_kv, c.hd, attn_scale, s, pt);
            }
            mark("attn prep+attention");
            ok &= I.gemv(y.o, I.attn, I.x, c.n_embd, qw, true);
            mark("attn o gemv");
        } else {
            fork();
            ok &= limbo::cuda::gemv(y.gate.type, y.gate.p, I.xn, I.z, vn, c.n_embd, false, s2);  // the z product beside the scan chain
            ok &= I.gemv(y.qkv, I.xn, I.qkv, c.qkv_w, c.n_embd);
            mark("gdn qkv gemv");
            ops::conv_silu(I.qkv, I.hist[size_t(l)], y.conv, I.conv, c.qkv_w, s);
            ops::head_norm(I.conv, I.qk, 2 * c.n_k_head, c.gdn_head, gdn_eps, gdn_scale, s);
            mark("gdn conv+norm");
            if (y.alpha.type == F32 && y.beta.type == F32 && c.n_embd <= 2048 && c.dt_rank <= 64)
                ok &= dec::f32_cols(reinterpret_cast<const float*>(y.alpha.p), reinterpret_cast<const float*>(y.beta.p), I.xn, c.n_embd, I.alpha,
                                    I.beta, 64, c.dt_rank, c.n_embd, 1, 2, s);
            else if (y.alpha.type == y.beta.type)
                ok &= gemv_dual(y.alpha.type, y.alpha.p, y.beta.p, I.xn, I.alpha, I.beta, c.dt_rank, c.n_embd, s);
            else {
                ok &= I.gemv(y.alpha, I.xn, I.alpha, c.dt_rank, c.n_embd);
                ok &= I.gemv(y.beta, I.xn, I.beta, c.dt_rank, c.n_embd);
            }
            mark("gdn alpha/beta");
            ops::gdn_scan(I.S[size_t(l)], I.qk, I.conv + 2 * c.n_k_head * c.gdn_head, I.alpha, I.beta, y.dtb, y.ssm_a,
                          I.o, c.n_v_head, c.n_k_head, c.gdn_head, c.dt_rank, gdn_scale, s);
            mark("gdn scan");
            join();
            mark("gdn z gemv");
            ops::gdn_gate(I.o, I.z, y.ssm_norm, I.gated, c.n_v_head, c.gdn_head, c.eps, s);
            ok &= I.gemv(y.out, I.gated, I.x, c.n_embd, vn, true);
            mark("gdn gate+out gemv");
        }

        // MoE on the post-attention norm: the router, the top-8 and the experts all stay on the device.
        ok &= dec::rmsnorm(I.x, y.post_norm, I.post, c.n_embd, epsf, 1, s);
        mark("norm");
        if (c.n_ff > 0) {
            // dense SwiGLU: x += down(silu(gate(post)) * up(post))
            if (y.fgate.type == y.fup.type)
                ok &= gemv_dual(y.fgate.type, y.fgate.p, y.fup.p, I.post, I.gbuf, I.ubuf, c.n_ff, c.n_embd, s);
            else {
                ok &= I.gemv(y.fgate, I.post, I.gbuf, c.n_ff, c.n_embd);
                ok &= I.gemv(y.fup, I.post, I.ubuf, c.n_ff, c.n_embd);
            }
            mark("ffn gate/up");
            ops::swiglu(I.gbuf, I.ubuf, I.hbuf, c.n_ff, s);
            ok &= I.gemv(y.fdown, I.hbuf, I.x, c.n_embd, c.n_ff, true);
            mark("ffn down");
            if (I.trace) {
                if (I.trace->size() < size_t(c.n_layer_main)) I.trace->resize(size_t(c.n_layer_main));
                std::vector<float>& t = (*I.trace)[size_t(l)];
                t.resize(size_t(c.n_embd));
                cudaStreamSynchronize(s);
                cudaMemcpy(t.data(), I.x, size_t(c.n_embd) * sizeof(float), cudaMemcpyDeviceToHost);
            }
            continue;
        }
        fork();
        // the shared expert beside the router and the routed experts
        if (y.sgate.type == y.sup.type)
            ok &= gemv_dual(y.sgate.type, y.sgate.p, y.sup.p, I.post, I.gs, I.us, c.n_ff_shexp, c.n_embd, s2);
        else {
            ok &= limbo::cuda::gemv(y.sgate.type, y.sgate.p, I.post, I.gs, c.n_ff_shexp, c.n_embd, false, s2);
            ok &= limbo::cuda::gemv(y.sup.type, y.sup.p, I.post, I.us, c.n_ff_shexp, c.n_embd, false, s2);
        }
        ok &= gemv_swiglu(y.sdown.type, y.sdown.p, I.gs, I.us, I.shared, c.n_embd, c.n_ff_shexp, false, s2);
        ops::shexp_gate(y.shgate, I.post, I.sg, c.n_embd, s2);
        mark("shared expert");
        if (y.router.type == F32)
            ok &= dec::f32_cols(reinterpret_cast<const float*>(y.router.p), nullptr, I.post, c.n_embd, I.rlog, nullptr, c.n_expert, c.n_expert,
                                c.n_embd, 1, 1, s);
        else
            ok &= I.gemv(y.router, I.post, I.rlog, c.n_expert, c.n_embd);
        mark("router gemv");
        ok &= dec::topk(I.rlog, c.n_expert, c.n_used, 1, I.ids, I.wts, s);
        mark("topk");
        if (I.moe_trace) {
            if (I.moe_trace->size() < size_t(c.n_layer_main)) I.moe_trace->resize(size_t(c.n_layer_main));
            std::vector<float>& t = (*I.moe_trace)[size_t(l)];
            t.assign(size_t(2 * c.n_used + c.n_expert), 0.f);
            int32_t ids[32];
            cudaStreamSynchronize(s);
            cudaMemcpy(ids, I.ids, sizeof(int32_t) * size_t(c.n_used), cudaMemcpyDeviceToHost);
            for (int i = 0; i < c.n_used; ++i) t[size_t(i)] = float(ids[i]);
            cudaMemcpy(t.data() + c.n_used, I.wts, sizeof(float) * size_t(c.n_used), cudaMemcpyDeviceToHost);
            cudaMemcpy(t.data() + 2 * c.n_used, I.rlog, sizeof(float) * size_t(c.n_expert), cudaMemcpyDeviceToHost);
        }
        // gate/up with the SwiGLU fused where the type has a fast kernel (IQ2_S, 2048 columns), else the generic pair
        if (!gemv_experts_gate_up(y.ge.type, y.ge.p, y.ue.p, y.ge_bytes, I.ids, c.n_used, I.post, I.hbuf, c.n_ff_exp, c.n_embd, s)) {
            ok &= gemv_experts_dual(y.ge.type, y.ge.p, y.ue.p, y.ge_bytes, I.ids, c.n_used, I.post, I.gbuf, I.ubuf, c.n_ff_exp,
                                    c.n_embd, s);
            ops::swiglu(I.gbuf, I.ubuf, I.hbuf, c.n_used * c.n_ff_exp, s);
        }
        mark("experts gate/up");
        if (!gemv_experts_down_fast(y.de.type, y.de.p, y.de_bytes, I.ids, c.n_used, I.hbuf, I.dn, c.n_embd, c.n_ff_exp, s))
            ok &= gemv_experts(y.de.type, y.de.p, y.de_bytes, I.ids, c.n_used, I.hbuf, c.n_ff_exp, I.dn, c.n_embd, c.n_ff_exp, s);
        mark("experts down");
        join();
        ops::moe_combine(I.x, I.dn, I.wts, I.shared, I.sg, c.n_embd, c.n_used, s);
        mark("combine");

        if (I.trace) {
            if (I.trace->size() < size_t(c.n_layer_main)) I.trace->resize(size_t(c.n_layer_main));
            std::vector<float>& t = (*I.trace)[size_t(l)];
            t.resize(size_t(c.n_embd));
            cudaStreamSynchronize(s);
            cudaMemcpy(t.data(), I.x, size_t(c.n_embd) * sizeof(float), cudaMemcpyDeviceToHost);
        }
    }

    if (head) {
        ok &= dec::rmsnorm(I.x, I.out_norm, I.rn, c.n_embd, epsf, 1, s);
        ok &= I.gemv(I.head, I.rn, I.logits, c.n_vocab, c.n_embd);
        mark("head");
    }
    if (dec_prof) {
        cudaStreamSynchronize(s);
        for (size_t i = 1; i < evs.size(); ++i) {
            float ms = 0;
            cudaEventElapsedTime(&ms, evs[i - 1].second, evs[i].second);
            bool found = false;
            for (auto& kv : *dec_prof)
                if (kv.first == evs[i].first) {
                    kv.second += ms;
                    found = true;
                }
            if (!found) dec_prof->emplace_back(evs[i].first, ms);
        }
        for (auto& e : evs) cudaEventDestroy(e.second);
    }
    return ok && cudaGetLastError() == cudaSuccess;
}

bool Gpu::Impl::ensure_dec_scratch() {
    if (attn_scr) return true;
    attn_scr = dalloc<float>(dec::attn_scratch_floats(n_ctx, 1, c.n_head));  // one row; the T-row pass has rw.attn_scr
    if (!attn_scr) return false;
    if (cudaStreamCreateWithFlags(&da.s2, cudaStreamNonBlocking) != cudaSuccess) return false;
    if (!q4e && !frag_reserve(da.s2, nullptr)) return false;
    for (cudaEvent_t& e : da.ev)
        if (cudaEventCreateWithFlags(&e, cudaEventDisableTiming) != cudaSuccess) return false;
    return true;
}

bool Gpu::step(int32_t token, int pos, bool head) {
    Impl& I = *p_;
    if (!I.step_run(token, pos, head)) return false;
    if (I.df && pos >= I.df_from) I.df_ingest(0, 0, 1, pos);  // a failure here only costs the drafter its context
    return true;
}

bool Gpu::Impl::step_run(int32_t token, int pos, bool head) {
    Impl& I = *this;
    if (pos < 0 || pos >= I.n_ctx) return false;
    if (I.q4e) return I.step_q4e(token, pos, head);
    if (!I.ensure_dec_scratch()) return false;
    if (!I.pages_for(pos, pos + 1)) return false;
    cudaStream_t s = I.stream;
    // The token and the position reach the device through a ring of pinned slots: a slot is reused only after
    // the copy that read it has run.
    const int slot = I.ring++ % Impl::kRing;
    if (cudaEventSynchronize(I.pin_ev[slot]) != cudaSuccess) return false;
    I.pin[2 * slot] = token;
    I.pin[2 * slot + 1] = pos;
    if (cudaMemcpyAsync(I.d_args, I.pin + 2 * slot, 2 * sizeof(int32_t), cudaMemcpyHostToDevice, s) != cudaSuccess) return false;
    if (cudaEventRecord(I.pin_ev[slot], s) != cudaSuccess) return false;
    // The debug traces synchronise inside the pass: they cannot be captured.
    if (!I.graphs || I.trace || I.moe_trace || I.dec_prof) return I.launch_step(head);
    const int v = head ? 1 : 0;
    if (!I.gexec[v]) {
        // The first pass of each variant runs eagerly so every kernel's module is loaded before a capture
        // begins; the next one is captured (and then replayed, not run).
        if (I.warm[v]++ == 0) return I.launch_step(head);
        cudaGraph_t g = nullptr;
        if (cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal) != cudaSuccess) return false;
        const bool ok = I.launch_step(head);
        if (cudaStreamEndCapture(s, &g) != cudaSuccess || !ok || !g) {
            if (g) cudaGraphDestroy(g);
            cudaGetLastError();
            return false;
        }
        const cudaError_t e = cudaGraphInstantiate(&I.gexec[v], g, 0);
        cudaGraphDestroy(g);
        if (e != cudaSuccess) {
            I.gexec[v] = nullptr;
            I.graphs = false;  // fall back to launching kernel by kernel
            cudaGetLastError();
            return I.launch_step(head);
        }
    }
    return cudaGraphLaunch(I.gexec[v], s) == cudaSuccess;
}

void Gpu::set_graphs(bool on) { p_->graphs = on; }

}  // namespace limbo::cuda
