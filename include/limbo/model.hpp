// Copyright (c) 2026 ITERFORM. All rights reserved.
#pragma once
// limbo/model.hpp — the geometry and the tensor map: what a GGUF file says the model is.
//
// This is the layer every wave stands on: the planner needs the geometry, the kernels need to know
// which tensor is which role and which layer, and the reference forward pass needs both. It is built to
// work for any Qwen-family file without a hand-typed role list, because a hand-typed list drifts from
// the file and then quietly feeds a kernel the wrong tensor:
//
//  - the geometry comes from the metadata keys the file itself carries;
//  - the roles come from the tensor names ("blk.<N>.<role>", everything else global);
//  - the structure is then *asserted* against the geometry: the layers that carry a full-attention role
//    and the layers that carry a linear-attention role must partition the block list exactly, and every
//    tensor in every shard must be claimed by a role. An unclaimed tensor is an error, not a detail:
//    it means the map does not know what the file is.
//
// A split model is several shards: the primary carries the architecture and the tokenizer, the
// continuation carries only split.no/split.count and its own tensors (rung 3's shard 2 is one tensor,
// the PLE table). Model::open takes them in order and joins them.

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "limbo/gguf.hpp"

namespace limbo::model {

struct Geometry {
    std::string arch;        // "qwen35moe", "qwen4exp", ...
    int n_layer = 0;
    int n_embd = 0;
    int n_head = 0;
    int n_head_kv = 0;
    int n_ctx = 0;
    int head_dim = 0;        // attention.key_length
    int rope_dim = 0;        // rope.dimension_count
    double rope_base = 0;
    double rms_eps = 0;
    int n_expert = 0;
    int n_expert_used = 0;
    int n_ff_exp = 0;
    int n_ff_shexp = 0;
    int full_attn_interval = 0;  // 0 when every layer is full attention
    // The linear-attention (GDN) side, present when the file has ssm.* keys.
    int ssm_conv = 0, ssm_state = 0, ssm_groups = 0, ssm_dt_rank = 0, ssm_inner = 0;
    // Capabilities, read off the file rather than assumed from the architecture name.
    bool has_mtp = false;      // a nextn.* head
    bool has_ple = false;      // per-layer token embeddings
    bool has_hyper = false;    // hyper-connections
    bool has_indexer = false;  // a sparse-attention indexer
    int n_vocab = 0;           // from token_embd's row count
    int n_vocab_unpadded = 0;  // vocab_size_unpadded, when the file declares one (inkling2_loop: the head keeps only these rows)

    // The dense FFN width, and the per-layer lists the file carries when a width or a head count is not
    // the same for every layer (gemma4: its shared-KV layers have a double-wide FFN, its global layers
    // fewer KV heads). Empty means "the scalar above for every layer".
    int n_ff = 0;
    std::vector<int> ff_l, head_kv_l;
    std::vector<int> head_l;   // per-layer head counts, when the file gives the key as an array
    // Sliding-window attention: the file's own per-layer pattern (1 = sliding window), the window, and the
    // head width / rotary width / rope base the sliding layers use when they differ from the global ones.
    std::vector<uint8_t> swa_l;
    int sliding_window = 0;
    int head_dim_swa = 0, rope_dim_swa = 0;
    double rope_base_swa = 0;
    int shared_kv_layers = 0;    // the last N layers read the K/V cache of an earlier layer
    int n_embd_per_layer = 0;    // per-layer token embeddings (gemma's PLE): width of one layer's input
    double logit_softcap = 0;    // final_logit_softcapping, 0 = none

    // Hyper-connections (qwen4exp): parallel residual streams and the low rank of the mixers that read them.
    int hc_count = 0, hc_rank = 0;
    // The sparse-attention indexer: heads, key width, and the number of cells a query attends to.
    int indexer_heads = 0, indexer_dim = 0, indexer_top_k = 0;
    std::vector<int> compress_ratios;   // per layer; non-zero on the sparse-attention layers
    // PLE as an n-gram hash table (qwen4exp): n-gram size, heads per n-gram, conv kernel, the id that resets the
    // window, and the per-position multipliers and per-head row ranges of the hash.
    int ple_ngram = 0, ple_heads_per_ngram = 0, ple_conv_kernel = 0;
    int64_t ple_eos = -1;
    std::vector<uint64_t> ple_multipliers, ple_head_offsets, ple_head_vocab;
    double expert_weights_scale = 0;   // 0 = unset (no scaling)

    // The looped core (inkling2_loop): layers loop_start..loop_end are run loop_count times, each pass
    // gated by the rank-`loop_gate_rank` controllers the layers carry. loop_count 0 = the file does not loop.
    int loop_start = 0, loop_end = 0, loop_count = 0, loop_gate_rank = 0;
    // The attention-residual subset (inkling2_loop): the layers that carry attn_res_query. Layer 0 has no
    // attention residual to read -- the file says so by carrying ffn_res_query on blk.0 and no
    // attn_res_query -- so that role covers layers 1..n_layer-1, which is NOT a union of structural
    // classes, and this list is the declared target the role assertion accepts.
    std::vector<int> attn_res_l;
    // The layers whose routed-expert down projection is NVFP4: those carry a per-expert .scale tensor
    // beside the weights (and so does their fused gate/up), and no other layer does. A property of the
    // file's quantisation, read from the tensor TYPES -- not from the role map it then constrains.
    std::vector<int> scale_l;
    // The looped core's layers (loop_start..loop_end-1): every loop_* role covers exactly these, which is
    // a range the file declares in its own metadata (inkling2_loop.loop.start/end), not a class union.
    std::vector<int> loop_l;
    // inkling2_loop's attention and activation constants, read from the file (docs/inkling2.md).
    int rel_length = 0, rel_extent = 0, sconv = 0, attn_res_block = 0, log_scale_floor = 0;
    double qk_clip = 0, rel_clip = 0, log_scale_alpha = 0, situ_beta = 0, situ_linear_beta = 0;

    bool is_swa(int l) const { return l >= 0 && size_t(l) < swa_l.size() && swa_l[size_t(l)] != 0; }
    int n_head_kv_at(int l) const { return head_kv_l.empty() ? n_head_kv : head_kv_l[size_t(l)]; }
    int n_head_at(int l) const { return head_l.empty() ? n_head : head_l[size_t(l)]; }
    int n_ff_at(int l) const { return ff_l.empty() ? n_ff : ff_l[size_t(l)]; }
    bool has_own_kv(int l) const { return l < n_layer - shared_kv_layers; }
    // The layer whose K/V cache layer l reads: itself when it owns one, else the last own-KV layer of the
    // same attention kind (sliding or global). -1 when the file has no such layer.
    int kv_source(int l) const;

    // MEASURED against the files: among the main layers, full attention is (layer + 1) % interval == 0.
    // Rung 1 (qwen35moe) has 41 blocks of which the last is the MTP block, so its main layers are 0..39
    // and the full ones are 3, 7, ... 39; rung 3 (qwen4exp) has 48 layers, full at 3, 7, ... 47.
    bool is_full_attention(int layer) const;
    int n_layer_main = 0;   // n_layer, or n_layer - 1 when the last block is an MTP head
    int mtp_layer = -1;     // the block that carries nextn.*, -1 when the file has no MTP head
};

class Model {
public:
    // Shards in order (primary first). nullptr and *err on any inconsistency.
    static std::unique_ptr<Model> open(const std::vector<std::string>& shards, std::string* err);

    const Geometry& geometry() const { return geo_; }
    const std::vector<std::unique_ptr<gguf::File>>& shards() const { return files_; }
    const std::vector<const gguf::Tensor*>& tensors() const { return tensors_; }

    // layer < 0 for a global role. nullptr when this file has no such tensor.
    const gguf::Tensor* tensor(int layer, std::string_view role) const;
    const gguf::Tensor* tensor(std::string_view name) const;

    // The layers that carry a full-attention role and the layers that carry a linear-attention one.
    const std::vector<int>& full_layers() const { return full_layers_; }
    const std::vector<int>& linear_layers() const { return linear_layers_; }

    // Names of tensors no role claims: empty after a successful open.
    const std::vector<std::string>& unclaimed() const { return unclaimed_; }

    // Returns the resident pages of every shard's mapping to the page cache (see gguf::File::drop_pages).
    void drop_resident() const { for (const auto& f : files_) f->drop_pages(); }

    // Total bytes of every tensor, and of the ones in one role.
    uint64_t total_bytes() const { return total_bytes_; }
    uint64_t role_bytes(std::string_view role) const;

private:
    Model() = default;
    bool build(std::string* err);

    std::vector<std::unique_ptr<gguf::File>> files_;
    std::vector<const gguf::Tensor*> tensors_;
    std::map<std::pair<int, std::string>, const gguf::Tensor*> by_role_;
    std::map<std::string, const gguf::Tensor*> by_name_;
    std::vector<int> full_layers_, linear_layers_;
    std::vector<std::string> unclaimed_;
    Geometry geo_;
    uint64_t total_bytes_ = 0;
};

}  // namespace limbo::model
