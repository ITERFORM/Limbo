// Copyright (c) 2026 ITERFORM. All rights reserved.
#pragma once
// limbo/step.hpp — the reference forward pass: exact, slow, and the oracle for every kernel.
//
// Built from bench/rung1-graph-spec.md, which is the oracle's own graph for this model measured node by
// node. The reference is deliberately single-token-per-call, unoptimised and readable: its job is to be
// *right*, so that every CUDA kernel later has something exact to be compared against (PLAN.md §4:
// "every fast kernel has a parity bench against a plain reference"), and so that W1's gate — token-exact
// against the captures — can be measured at all.
//
// It owns no cleverness: one scratch buffer set, a dense f16 KV cache (matching the oracle's cache
// dtype, because a numeric difference there is a real difference), the GDN state per layer, and a
// thread pool that splits matrix rows.

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "limbo/model.hpp"

namespace limbo::step {

// ---- recipes: what a model is, read from its file -----------------------------------------------------
// A model is a list of layers. Each layer is a mixer (gated delta net, full attention, sliding-window
// attention), an FFN (a gated MLP with its activation, or routed experts with an optional shared expert or
// a parallel dense MLP) and the placement of its norms; around the list sit the model-level parts
// (embedding scale, per-layer embeddings, final norm, tied or separate head, logit soft-capping). Ref::open
// fills the recipe from the GGUF (metadata keys and which tensors exist) and then runs it; nothing in the
// forward pass looks at the architecture name. The GPU port is built on the same recipe.
enum class Mixer : uint8_t { Gdn, Attention };
enum class Act : uint8_t { Silu, Gelu };
enum class FfnKind : uint8_t {
    Dense,        // one gated MLP: down(act(gate(x)) * up(x))
    MoeShared,    // routed experts plus a shared expert behind a scalar sigmoid gate (qwen35moe)
    DenseAndMoe,  // a dense MLP and routed experts side by side, each with its own norms (gemma4 26B-A4B)
};

struct AttnRecipe {
    bool sliding = false;       // only the last `window` positions are visible (window includes the token itself)
    int window = 0;
    int n_head = 0, n_head_kv = 0, head_dim = 0;
    int rope_dim = 0;           // rotary width, in the first rope_dim dims of each head, NEOX pairing
    double rope_base = 0;
    bool rope_factors = false;  // per-pair frequency divisors from rope_freqs.weight (gemma's global layers)
    bool q_gate = false;        // q is double width: [q | gate] per head, sigmoid(gate) multiplies the output
    bool own_kv = true;         // false: computes no K/V and reads the cache of layer kv_src
    int kv_src = -1;
    bool v_from_k = false;      // no attn_v: V is the raw K projection (before K's norm and rope)
    bool v_norm = false;        // V gets an unweighted RMS norm per head
    int qsa_top_k = 0;          // sparse attention (qwen4exp): each query attends to this many cells (plus the tail)
    int qsa_kpool = 0, qsa_heads = 0, qsa_dim = 0;   // the indexer: block size of its pooled keys, its heads and key width
    float scale = 1.0f;         // multiplies q.k before the softmax
};

struct FfnRecipe {
    FfnKind kind = FfnKind::Dense;
    Act act = Act::Silu;
    int n_ff = 0;                // the dense MLP's width (also gemma's parallel dense branch)
    int n_expert = 0, n_used = 0, n_ff_exp = 0, n_ff_shexp = 0;
    bool fused_gate_up = false;  // the experts' gate and up are one tensor, gate rows first
    bool expert_scale = false;   // a learned per-expert multiplier on the down projection's output
    bool router_scaled = false;  // router input = rms_norm(x) / sqrt(n_embd) * a learned vector, not the FFN input
};

struct LayerRecipe {
    Mixer mixer = Mixer::Attention;
    AttnRecipe attn;
    bool post_mixer_norm = false;  // a norm on the mixer's output before it joins the residual (gemma)
    FfnRecipe ffn;
    bool post_ffn_norm = false;    // the same for the FFN's output
    bool ple = false;              // a per-layer-embedding block after the FFN
    bool ple_hash = false;         // an n-gram-hash embedding block before the mixer (qwen4exp)
    bool gdn_sigmoid_gate = false; // the GDN's output gate is sigmoid(z), not silu(z)
    bool out_scale = false;        // a learned scalar on the layer's output
};

struct Config {
    int n_layer = 0, n_layer_main = 0;
    int n_embd = 0, n_head = 0, n_head_kv = 0, head_dim = 0, rope_dim = 0;
    int n_ctx = 0, n_vocab = 0;
    int n_expert = 0, n_expert_used = 0, n_ff_exp = 0, n_ff_shexp = 0;
    // The linear (GDN) side, all measured from the graph.
    int n_k_head = 0, n_v_head = 0, gdn_head = 0, conv_kernel = 0, dt_rank = 0, qkv_width = 0;
    double rms_eps = 1e-6, rope_base = 1e7;
    std::vector<int> full_layers;  // layers with attention (full or sliding); the rest are linear
    // The recipe.
    std::string arch;
    std::vector<LayerRecipe> layers;   // n_layer_main entries
    float embed_scale = 1.0f;          // multiplies the token embedding (gemma: sqrt(n_embd))
    int n_embd_per_layer = 0;          // per-layer token embeddings: one input of this width per layer
    bool tied_head = false;            // no output matrix: the head is token_embd
    float logit_softcap = 0.0f;        // logits = cap * tanh(logits / cap), 0 = off
    // Hyper-connections: the residual is hc_count parallel streams; each mixer and FFN reads them through a gated
    // low-rank mixer (rank hc_rank) and writes back to each stream with its own weight; the head reads them the same way.
    int hc_count = 0, hc_rank = 0;
    // The n-gram hash embedding: n_embd_per_layer-wide rows, ple_heads of them per token, from a table addressed by
    // hashing the token with its predecessors.
    int ple_ngram = 0, ple_heads_per_ngram = 0, ple_conv_kernel = 0;
    int64_t ple_eos = -1;
    std::vector<uint64_t> ple_multipliers, ple_head_offsets, ple_head_vocab;
    // inkling2_loop: no RoPE, a learned relative bias instead, four short convolutions per layer, L2
    // clips, per-head attention sinks, a depth-mixing residual bank, and a looped core. See
    // docs/inkling2.md -- every constant here is the file's own key.
    struct Inkling {
        int rel_extent = 0, rel_length = 0;          // attention.relative_extent / _length
        float qk_clip = 0, rel_clip = 0;             // attention.qk_clip / .relative_clip
        int log_floor = 0;                           // attention.log_scale_floor
        float log_alpha = 0;                         // attention.log_scale_alpha
        float situ_beta = 0, situ_lin_beta = 0;      // activation.situ_beta / _linear_beta
        int sconv = 0;                               // shortconv.l_cache
        float route_scale = 1.0f;                    // expert_weights_scale
        int loop_start = 0, loop_end = 0, loop_count = 0, gate_rank = 0;
        int res_block = 0;                           // attn_res.block_size
        int vocab_unpadded = 0;                      // the head keeps only this many of the padded rows
        bool on = false;
    } ink;
};

// One step's activations, kept per layer so a test can compare a named node against the oracle's dump.
struct Trace {
    bool on = false;
    // With all_pos, every position's per-layer outputs are kept as well: [pos][layer] (the fields below
    // keep their position-0-only meaning, so existing callers read what they always read).
    bool all_pos = false;
    std::vector<std::vector<std::vector<float>>> attn_out_pos, l_out_pos;
    // The routed experts of every (pos, layer) with all_pos: ids by descending router weight, then the
    // renormalised weights of the same experts, 8 + 8 values.
    std::vector<std::vector<std::vector<float>>> moe_pos;
    // The most recent MoE call's routed experts (ids by descending router weight), whenever `on`.
    std::vector<int> moe_last_ids;
    std::vector<float> moe_last_p;   // the router probabilities of the 8 chosen and of the 9th, descending
    std::vector<std::vector<float>> attn_out;   // attn_residual per layer
    std::vector<std::vector<float>> l_out;      // the layer's output per layer
    std::vector<float> result_norm, logits;
    // Layer 0's stages, in the oracle's node order, so a mismatch is localised to a node.
    std::vector<float> x0, xn0, qkv0, conv0, qhat0, khat0, gdn0, z0, gated0, linattn0, moe_logits0;
    // With `on` and node_layer >= 0, the named intermediates of that layer's FFN half of the latest call (the
    // oracle's graph names where it has them: ffn_norm, ffn_mlp, ffn_moe_out, ffn_moe, ffn_moe_combined,
    // ffn_post_norm), for isolating a mismatch to one node.
    int node_layer = -1;
    std::map<std::string, std::vector<float>> nodes;
};

class Ref {
public:
    static std::unique_ptr<Ref> open(const std::vector<std::string>& shards, int n_ctx, int threads,
                                     std::string* err);
    ~Ref();

    // Tensors of the file that the recipe does not run, with why (the MTP block; the K/V projections of
    // layers that read another layer's cache). open() fails on any other tensor it does not read.
    const std::vector<std::string>& unused_tensors() const { return unused_; }

    const Config& config() const { return cfg_; }
    const model::Model& model() const { return *model_; }
    int n_vocab() const { return cfg_.n_vocab; }

    // Runs one token at `pos` and returns n_vocab logits (owned by the Ref, valid until the next call).
    // want_logits=false skips the output head (the biggest single matvec) and returns nullptr: a prompt
    // token whose logits nobody reads still has to run the layers, because they carry the state.
    const float* forward(int32_t token, int pos, bool want_logits = true);
    void reset();  // clears the caches and the GDN state

    // One layer on its own, for isolating a layer's error from everything upstream of it: embed() writes the
    // token embedding (n_embd floats), forward_layer() runs layer l in place on x at position pos, updating
    // that layer's own state (GDN state, conv history, KV cache). Positions must be fed in order per layer.
    void embed(int32_t token, float* x);
    void forward_layer(int l, int pos, float* x);
    // The output norm and the LM head on a final hidden state x (n_embd floats): the logits, owned by the Ref.
    const float* head(const float* x);
    // A layer's two halves on their own: forward_layer_attn is x += attention(norm(x)) (and updates the
    // layer's state), forward_layer_ffn is x += moe(post_norm(x)). forward_layer is the two in a row.
    void forward_layer_attn(int l, int pos, float* x);
    void forward_layer_ffn(int l, float* x);
    // The experts (ids, best first) layer l's router would choose if its FFN-side hyper-connection read were fed `stack`
    // (hc_count x n_embd) instead of the real post-attention stack: how well an earlier state predicts the routing.
    std::vector<int> route_from_stack(int l, const float* stack);
    // Records the token at a position for the n-gram hash (forward() does it itself; the layer-by-layer path does not).
    void note_token(int pos, int32_t token);

    Trace& trace() { return trace_; }
    // Stops after n layers: for bisecting a slow or wrong layer, and for the per-node comparison.
    void set_layer_limit(int n) { layer_limit_ = n; }
    // Rounds every matmul's activation the way llama.cpp's CPU path does (Q8_K / Q8_0 / f16 per weight
    // type) and the attention's q and softmax weights to f16, as the oracle's non-flash path does. Off by
    // default: the default arithmetic keeps fp32 activations, which is the exact one.
    void set_oracle_arith(bool on) { oracle_arith_ = on; }
    // A file bigger than RAM: before a token, if the process holds more than `bytes` of mapped file pages, they go back
    // to the page cache (0 = never). The weights are mapped, never copied: this is what keeps the RSS flat.
    void set_rss_budget(uint64_t bytes) { rss_budget_ = bytes; }

private:
    Ref();
    struct Weights;   // the tensors of every layer, resolved once at open (src/step.cpp)
    struct Proj;
    struct Hc;        // a hyper-connection mixer's weights
    void mv(const Proj& p, int n_out, int n_in, const float* x, float* y);
    void layer_linear(int l, const float* xn, float* out);
    void layer_attention(int l, int pos, const float* xn, float* out);
    void layer_ffn(int l, float* x);
    void ffn_dense(int l, const float* pre, float* out);
    void moe(int l, const float* route_in, const float* post, float* out);
    void embed_token(int32_t token, float* x);
    void trim_resident();
    // The hyper-connection path: the residual is a stack of hc_count streams (stack_), see Config.
    const float* forward_hc(int32_t token, int pos, bool want_logits);
    // inkling2_loop (docs/inkling2.md): a depth-mixed residual bank, four short convolutions per layer,
    // relative-bias attention with per-head sinks and L2 clips, a routed MoE, and a looped core whose
    // later passes gate the previous pass's bank into the layer inputs. One call runs all
    // n_layer + (loop_count-1)*(loop_end-loop_start) visits; the weights are the physical layers', the
    // KV/conv state is per visit slot.
    const float* forward_inkling(int32_t token, int pos, bool want_logits);
    void ink_visit(int il, int ip, int pos, float* cur, std::vector<std::vector<float>>& blocks,
                   const std::vector<float>* messages);
    void ink_mix(const std::vector<std::vector<float>>& blocks, const float* cur, const float* query,
                 int q_count, float* out);
    void ink_convolve(int slot, int offset, const float* cur, int n_ch, int n_heads, const float* dyn,
                      const gguf::Tensor* weight, float* out);
    void ink_moe(int il, const float* post, float* out);
    void ffn_block(int l, const float* x, const float* pre, float* out);
    void ple_hash_block(int l, int pos, float* stack);
    void layer_hc(int l, int pos, float* stack);
    void layer_hc_attn(int l, int pos, float* stack);   // ple block, mixer and its combine
    void layer_hc_ffn(int l, float* stack);              // routed + shared experts and their combine
    void hc_note(int l, int pos, const char* name, const float* v, size_t n);
    void hc_mix(const Hc& m, const float* stack, float* mixed, float* inject);
    void hc_combine(float* stack, const float* block_out, const float* inject);
    void ple_block(int l, float* x);
    bool qsa_select(int l, int pos, const float* xn, std::vector<int>& sel);
    bool resolve(std::string* err);

    std::unique_ptr<model::Model> model_;
    std::unique_ptr<Weights> w_;
    std::vector<std::string> unused_;
    std::vector<float> ple_in_;    // this token's per-layer input, n_layer_main x n_embd_per_layer
    std::vector<float> stack_;     // hyper-connection residual: hc_count streams of n_embd
    std::vector<int32_t> tok_hist_;     // the tokens fed so far, by position (the n-gram hash reads predecessors)
    std::vector<float> ple_hist_;       // the n-gram block's conv input history, ((kernel-1) * dilation) x hc_count * n_embd, newest last
    Config cfg_;
    int threads_ = 8;
    int layer_limit_ = 0;  // 0 = all
    bool oracle_arith_ = false;
    uint64_t rss_budget_ = 0;
    bool attn_only_ = false;   // forward_layer_attn: stop before the MoE half
    int cur_pos_ = 0;      // the position being computed, for the trace
    Trace trace_;

    // scratch, all sized once
    std::vector<float> x_, xn_, q_, k_, v_, attn_, gate_, up_, down_, h_, logits_, tmp_;
    std::vector<std::vector<float>> gdn_state_;   // per layer: n_v_head * gdn_head * gdn_head
    std::vector<std::vector<float>> conv_state_;  // per layer: conv_kernel-1 rows of qkv_width
    std::vector<std::vector<uint16_t>> idx_raw_, idx_pool_;   // the indexer's per-position keys and pooled block keys, f16
    std::vector<std::vector<uint16_t>> k_cache_, v_cache_;  // per layer, [n_ctx][n_head_kv * head_dim], f16
    // inkling2_loop: the state is per visit slot (weights shared, state not): one f16 K/V pair and one
    // short-conv history of (sconv-1) rows of the shared channels (k, v, attention output, FFN output)
    // per slot; docs/inkling2.md §2/§3.
    std::vector<std::vector<uint16_t>> ink_k_cache_, ink_v_cache_;  // per slot, [n_ctx][n_head_kv * head_dim], f16
    std::vector<std::vector<float>> ink_conv_hist_;                 // per slot, (sconv-1) rows of n_sconv_channels
};

}  // namespace limbo::step
