// Copyright (c) 2026 ITERFORM. All rights reserved.
// tests/model_test.cpp — geometry and tensor map: what the file says the model is.
//
// LIMBO_TEST_GGUF holds one or more model specs separated by ';', each spec a colon-separated shard
// list (primary first), for example:
//
//   LIMBO_TEST_GGUF="rung1.gguf;rung3-00001.gguf:rung3-00002.gguf" make test
//
// The assertions are structural, so they hold for any file rather than for one quant: the full and
// linear layer sets must partition the block list, the roles must cover exactly all/full/linear/global,
// every tensor must be claimed, and the expert tensors' shapes must agree with the expert counts in the
// metadata. A map that has learned the wrong structure fails here instead of feeding a kernel.
#include "limbo/check.hpp"
#include "limbo/model.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>

namespace {

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    size_t at = 0;
    while (at <= s.size()) {
        const size_t p = s.find(sep, at);
        const std::string one = s.substr(at, p == std::string::npos ? std::string::npos : p - at);
        if (!one.empty()) out.push_back(one);
        if (p == std::string::npos) break;
        at = p + 1;
    }
    return out;
}

// LIMBO_TEST_GGUF is a flat list of files for the other suites, so a ':' list may mix whole models
// with the shards of one model. Group them: a file whose split.no says it is a continuation belongs to
// the model before it, anything else starts a new one.
std::vector<std::vector<std::string>> group_models(const std::vector<std::string>& files) {
    std::vector<std::vector<std::string>> out;
    for (const std::string& f : files) {
        std::string err;
        auto g = limbo::gguf::File::open(f, &err);
        const bool continuation =
            g && g->get_int("split.count").value_or(1) > 1 && g->get_int("split.no").value_or(0) > 0;
        if (!continuation || out.empty()) out.push_back({});
        out.back().push_back(f);
    }
    return out;
}

void check_model(const std::vector<std::string>& shards) {
    std::string err;
    auto m = limbo::model::Model::open(shards, &err);
    if (!m) {
        std::printf("FAIL cannot open %s: %s\n", shards[0].c_str(), err.c_str());
        ++limbo::test::failures();
        return;
    }
    const limbo::model::Geometry& g = m->geometry();
    std::printf("     %s: %d layers, embd %d, %d heads / %d kv, ctx %d, vocab %d\n", g.arch.c_str(), g.n_layer,
                g.n_embd, g.n_head, g.n_head_kv, g.n_ctx, g.n_vocab);
    std::printf("       experts: %d used %d, ff %d, shared ff %d; dense ff %d%s\n", g.n_expert, g.n_expert_used,
                g.n_ff_exp, g.n_ff_shexp, g.n_ff, g.ff_l.empty() ? "" : " (per layer)");
    if (!g.swa_l.empty()) {
        int n_swa = 0;
        for (uint8_t f : g.swa_l) n_swa += f != 0;
        std::printf("       sliding window %d on %d of %d layers (head %d, rope %d, base %.0f); shared-KV tail %d; "
                    "per-layer input %d; softcap %.0f\n",
                    g.sliding_window, n_swa, g.n_layer, g.head_dim_swa, g.rope_dim_swa, g.rope_base_swa,
                    g.shared_kv_layers, g.n_embd_per_layer, g.logit_softcap);
    }
    std::printf("       attention: %zu full layers, %zu linear layers, interval %d, head_dim %d, rope %d (base %.0f)\n",
                m->full_layers().size(), m->linear_layers().size(), g.full_attn_interval, g.head_dim, g.rope_dim,
                g.rope_base);
    if (g.ssm_inner) std::printf("       gdn: conv %d, state %d, groups %d, dt rank %d, inner %d\n", g.ssm_conv,
                                 g.ssm_state, g.ssm_groups, g.ssm_dt_rank, g.ssm_inner);
    std::printf("       capabilities: mtp %d, ple %d, hyper %d, indexer %d; %zu tensors, %.2f GiB\n",
                int(g.has_mtp), int(g.has_ple), int(g.has_hyper), int(g.has_indexer), m->tensors().size(),
                double(m->total_bytes()) / 1073741824.0);

    CHECK(g.n_layer > 0, "the model has layers");
    CHECK(g.n_embd > 0, "the model has an embedding width");
    CHECK(g.n_vocab > 0, "the vocabulary size comes from token_embd");
    CHECK(m->unclaimed().empty(), "every tensor in every shard is claimed by a role");
    // Against the *main* layers: when the file carries an MTP head it is its own block (rung 1's
    // blk.40), it is not part of the full/linear partition, and asserting against n_layer would fail on
    // a correct file.
    CHECK(m->full_layers().size() + m->linear_layers().size() == size_t(g.n_layer_main),
          "the full and linear layer sets partition the main block list");
    CHECK(g.n_layer_main + (g.has_mtp ? 1 : 0) == g.n_layer, "main layers plus the MTP block are all the blocks");

    // The expert tensors must agree with the metadata: gate/up are [embd, ff, experts], down is
    // [ff, embd, experts].
    const limbo::gguf::Tensor* gate = m->tensor(0, "ffn_gate_exps.weight");
    const limbo::gguf::Tensor* down = m->tensor(0, "ffn_down_exps.weight");
    if (gate && down && g.n_expert > 0) {
        CHECK(gate->ne[0] == g.n_embd && gate->ne[1] == g.n_ff_exp && gate->ne[2] == g.n_expert,
              "ffn_gate_exps is [embd, ff_exp, experts]");
        CHECK(down->ne[0] == g.n_ff_exp && down->ne[1] == g.n_embd && down->ne[2] == g.n_expert,
              "ffn_down_exps is [ff_exp, embd, experts]");
        std::printf("       expert pool: %.2f GiB across %d layers (%.2f MiB per expert per layer)\n",
                    double(m->role_bytes("ffn_gate_exps.weight") + m->role_bytes("ffn_up_exps.weight") +
                           m->role_bytes("ffn_down_exps.weight")) /
                        1073741824.0,
                    g.n_layer,
                    double(m->role_bytes("ffn_gate_exps.weight") + m->role_bytes("ffn_up_exps.weight") +
                           m->role_bytes("ffn_down_exps.weight")) /
                        double(g.n_layer) / double(g.n_expert) / 1048576.0);
    } else {
        std::printf("       (no routed experts in this file)\n");
    }

    // The dense FFN, the sliding-window pattern, the shared-KV tail, the per-layer inputs and the fused
    // expert tensor, each against the geometry the metadata declared.
    if (const limbo::gguf::Tensor* ff = m->tensor(0, "ffn_gate.weight")) {
        CHECK(ff->ne[0] == g.n_embd && ff->ne[1] == g.n_ff_at(0), "ffn_gate is [embd, n_ff of its layer]");
        bool all_ok = true;
        for (int l = 0; l < g.n_layer_main; ++l) {
            const limbo::gguf::Tensor* t = m->tensor(l, "ffn_gate.weight");
            if (!t || t->ne[0] != g.n_embd || t->ne[1] != g.n_ff_at(l)) all_ok = false;
        }
        CHECK(all_ok, "every layer's ffn_gate matches its own n_ff");
    }
    if (!g.swa_l.empty()) {
        CHECK(int(g.swa_l.size()) == g.n_layer, "one sliding-window flag per layer");
        CHECK(g.sliding_window > 0 && g.head_dim_swa > 0, "a sliding-window model carries its window and head width");
        bool q_ok = true, kv_ok = true, share_ok = true;
        for (int l = 0; l < g.n_layer; ++l) {
            const int hd = g.is_swa(l) ? g.head_dim_swa : g.head_dim;
            const limbo::gguf::Tensor* q = m->tensor(l, "attn_q.weight");
            if (!q || q->ne[1] != int64_t(g.n_head) * hd) q_ok = false;
            if (g.has_own_kv(l)) {
                const limbo::gguf::Tensor* k = m->tensor(l, "attn_k.weight");
                if (!k || k->ne[1] != int64_t(g.n_head_kv_at(l)) * hd) kv_ok = false;
            } else {
                const int src = g.kv_source(l);
                if (src < 0 || !g.has_own_kv(src) || g.is_swa(src) != g.is_swa(l)) share_ok = false;
            }
        }
        CHECK(q_ok, "attn_q is n_head x the head width of its layer's kind");
        CHECK(kv_ok, "attn_k is n_head_kv(layer) x the head width of its layer's kind, on every layer that owns KV");
        CHECK(share_ok, "a shared-KV layer reads an own-KV layer of the same attention kind");
    }
    if (g.n_embd_per_layer > 0 && g.ple_ngram > 0) {
        // PLE as an n-gram hash table: rows of n_embd_per_layer values, addressed by the metadata's head ranges.
        const limbo::gguf::Tensor* ple = m->tensor(-1, "per_layer_token_embd.weight");
        CHECK(ple && ple->ne[0] == g.n_embd_per_layer, "the hash table's rows are one per-layer-input wide");
        const int n_heads = (g.ple_ngram - 1) * g.ple_heads_per_ngram;
        CHECK(int(g.ple_head_offsets.size()) >= n_heads && int(g.ple_head_vocab.size()) >= n_heads &&
                  int(g.ple_multipliers.size()) >= g.ple_ngram,
              "the hash carries a multiplier per n-gram position and a row range per head");
        uint64_t rows = 0;
        for (int h = 0; h < n_heads && h < int(g.ple_head_offsets.size()); ++h)
            rows = std::max<uint64_t>(rows, g.ple_head_offsets[size_t(h)] + g.ple_head_vocab[size_t(h)]);
        CHECK(ple && uint64_t(ple->ne[1]) >= rows, "the table has a row for every head range");
    } else if (g.n_embd_per_layer > 0) {
        const limbo::gguf::Tensor* ple = m->tensor(-1, "per_layer_token_embd.weight");
        const limbo::gguf::Tensor* proj = m->tensor(-1, "per_layer_model_proj.weight");
        CHECK(ple && ple->ne[0] == int64_t(g.n_embd_per_layer) * g.n_layer && ple->ne[1] == g.n_vocab,
              "per_layer_token_embd is [n_layer * per-layer width, vocab]");
        CHECK(proj && proj->ne[0] == g.n_embd && proj->ne[1] == int64_t(g.n_embd_per_layer) * g.n_layer,
              "per_layer_model_proj is [embd, n_layer * per-layer width]");
    }
    if (const limbo::gguf::Tensor* gu = m->tensor(0, "ffn_gate_up_exps.weight")) {
        const limbo::gguf::Tensor* dn = m->tensor(0, "ffn_down_exps.weight");
        CHECK(gu->ne[0] == g.n_embd && gu->ne[1] == 2 * int64_t(g.n_ff_exp) && gu->ne[2] == g.n_expert,
              "ffn_gate_up_exps is [embd, 2 * ff_exp, experts]");
        CHECK(dn && dn->ne[0] == g.n_ff_exp && dn->ne[1] == g.n_embd && dn->ne[2] == g.n_expert,
              "ffn_down_exps is [ff_exp, embd, experts]");
    }
    if (g.ssm_inner > 0 && !m->linear_layers().empty()) {
        const int l = m->linear_layers()[0];
        const limbo::gguf::Tensor* qkv = m->tensor(l, "attn_qkv.weight");
        const limbo::gguf::Tensor* out = m->tensor(l, "ssm_out.weight");
        CHECK(qkv && qkv->ne[1] == 2 * int64_t(g.ssm_groups) * g.ssm_state + g.ssm_inner,
              "attn_qkv is [embd, 2 * key heads * state + inner]");
        CHECK(out && out->ne[0] == g.ssm_inner && out->ne[1] == g.n_embd, "ssm_out is [inner, embd]");
        CHECK(g.ssm_dt_rank * g.ssm_state == g.ssm_inner, "inner size is value heads x state");
    }

    // The attention tensors must agree with the head geometry.
    const limbo::gguf::Tensor* q = m->tensor(m->full_layers().empty() ? 0 : m->full_layers()[0], "attn_q.weight");
    if (q) {
        // Not n_head * key_length: rung 3's attn_q is 12288 wide for 24 heads against key_length 256,
        // so the q projection is wider than the KV geometry. Assert what is structural (the row count
        // divides by the head count) and print the implied width next to key_length, so a mismatch is
        // visible instead of assumed away.
        CHECK(q->ne[0] == g.n_embd, "attn_q columns are the embedding width");
        CHECK(g.n_head > 0 && q->ne[1] % g.n_head == 0, "attn_q rows divide by the head count");
        if (g.n_head > 0) std::printf("       attn_q: %lld rows = %d heads x %lld (key_length %d)\n",
                                      (long long)q->ne[1], g.n_head, (long long)(q->ne[1] / g.n_head), g.head_dim);
    }
    const limbo::gguf::Tensor* kv = m->tensor(m->linear_layers().empty() ? 0 : m->linear_layers()[0], "attn_qkv.weight");
    if (kv) std::printf("       attn_qkv (linear layer %d): shape", m->linear_layers()[0]),
        std::printf(" %lld %lld\n", (long long)kv->ne[0], (long long)kv->ne[1]);
}

}  // namespace

int main() {
    const char* env = std::getenv("LIMBO_TEST_GGUF");
    if (!env || !*env) {
        std::printf("--   LIMBO_TEST_GGUF not set: the model map is skipped\n");
        return limbo::test::finish("model");
    }
    for (const std::string& spec : split(env, ';'))
        for (const std::vector<std::string>& model : group_models(split(spec, ':'))) check_model(model);
    return limbo::test::finish("model");
}
