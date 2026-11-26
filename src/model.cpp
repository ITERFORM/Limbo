// Copyright (c) 2026 ITERFORM. All rights reserved.
// src/model.cpp — geometry and tensor map over one or more GGUF shards.
#include "limbo/model.hpp"

#include "limbo/types.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace limbo::model {

namespace {

// "blk.<N>.<role>" -> (layer, role); anything else is a global role with layer -1.
bool split_name(std::string_view name, int* layer, std::string_view* role) {
    if (name.rfind("blk.", 0) != 0) {
        *layer = -1;
        *role = name;
        return true;
    }
    size_t i = 4;
    int n = 0;
    while (i < name.size() && name[i] >= '0' && name[i] <= '9') n = n * 10 + (name[i++] - '0');
    if (i >= name.size() || name[i] != '.') return false;
    *layer = n;
    *role = name.substr(i + 1);
    return true;
}

int meta_int(const gguf::File& f, const std::string& arch, const char* suffix, int fallback) {
    const std::string key = arch + "." + suffix;
    if (auto v = f.get_int(key)) return int(*v);
    return fallback;
}

double meta_float(const gguf::File& f, const std::string& arch, const char* suffix, double fallback) {
    const std::string key = arch + "." + suffix;
    if (auto v = f.get_float(key)) return *v;
    return fallback;
}

bool has_meta(const gguf::File& f, const std::string& arch, const char* suffix) {
    return f.find(arch + "." + suffix) != nullptr;
}

}  // namespace

int Geometry::kv_source(int l) const {
    if (has_own_kv(l)) return l;
    for (int k = n_layer - shared_kv_layers - 1; k >= 0; --k)
        if (is_swa(k) == is_swa(l)) return k;
    return -1;
}

bool Geometry::is_full_attention(int layer) const {
    if (full_attn_interval <= 0) return true;  // every layer is full attention
    return (layer + 1) % full_attn_interval == 0;
}

const gguf::Tensor* Model::tensor(int layer, std::string_view role) const {
    const auto it = by_role_.find({layer, std::string(role)});
    return it == by_role_.end() ? nullptr : it->second;
}

const gguf::Tensor* Model::tensor(std::string_view name) const {
    const auto it = by_name_.find(std::string(name));
    return it == by_name_.end() ? nullptr : it->second;
}

uint64_t Model::role_bytes(std::string_view role) const {
    uint64_t sum = 0;
    for (const auto& [key, t] : by_role_)
        if (key.second == role) sum += t->nbytes;
    return sum;
}

// One path that names the first shard of a split model (`...-00001-of-00002.gguf`) stands for the whole set: the
// continuation shards are the same name with the shard number counted up. Anything else is returned as it came.
static std::vector<std::string> with_siblings(const std::vector<std::string>& given) {
    if (given.size() != 1) return given;
    const std::string& path = given[0];
    const size_t of = path.rfind("-of-");
    if (of == std::string::npos || of < 6 || path.size() < of + 4 + 5 + 5) return given;
    const std::string no = path.substr(of - 5, 5), total = path.substr(of + 4, 5);
    if (no != "00001" || path.compare(of + 9, std::string::npos, ".gguf") != 0) return given;
    for (char c : total) if (c < '0' || c > '9') return given;
    const int n = std::atoi(total.c_str());
    std::vector<std::string> out{path};
    for (int k = 2; k <= n; ++k) {
        char num[16];
        std::snprintf(num, sizeof num, "%05d", k);
        out.push_back(path.substr(0, of - 5) + num + path.substr(of));
    }
    return out;
}

std::unique_ptr<Model> Model::open(const std::vector<std::string>& given, std::string* err) {
    const std::vector<std::string> shards = with_siblings(given);
    if (shards.empty()) {
        *err = "no shards given";
        return nullptr;
    }
    std::unique_ptr<Model> m(new Model());
    for (size_t i = 0; i < shards.size(); ++i) {
        std::string e;
        auto f = gguf::File::open(shards[i], &e);
        if (!f) {
            *err = e;
            return nullptr;
        }
        // A continuation shard must say so, and must not claim to be the primary.
        const auto no = f->get_int("split.no");
        const auto count = f->get_int("split.count");
        if (count && *count > 1) {
            if (!no || *no != int64_t(i)) {
                *err = shards[i] + ": split.no is " + (no ? std::to_string(*no) : "absent") + " but it is shard " +
                       std::to_string(i);
                return nullptr;
            }
        } else if (i != 0) {
            *err = shards[i] + ": a second shard without split.count";
            return nullptr;
        }
        m->files_.push_back(std::move(f));
    }
    if (!m->build(err)) return nullptr;
    return m;
}

bool Model::build(std::string* err) {
    const gguf::File& primary = *files_[0];
    geo_.arch = std::string(primary.arch());
    if (geo_.arch.empty()) {
        *err = primary.path() + ": no general.architecture";
        return false;
    }
    const std::string& a = geo_.arch;
    geo_.n_layer = meta_int(primary, a, "block_count", 0);
    geo_.n_embd = meta_int(primary, a, "embedding_length", 0);
    geo_.n_head = meta_int(primary, a, "attention.head_count", 0);
    geo_.n_head_kv = meta_int(primary, a, "attention.head_count_kv", 0);
    geo_.n_ctx = meta_int(primary, a, "context_length", 0);
    geo_.head_dim = meta_int(primary, a, "attention.key_length", 0);
    geo_.rope_dim = meta_int(primary, a, "rope.dimension_count", 0);
    geo_.rope_base = meta_float(primary, a, "rope.freq_base", 0);
    geo_.rms_eps = meta_float(primary, a, "attention.layer_norm_rms_epsilon", 0);
    geo_.n_expert = meta_int(primary, a, "expert_count", 0);
    geo_.n_expert_used = meta_int(primary, a, "expert_used_count", 0);
    geo_.n_ff_exp = meta_int(primary, a, "expert_feed_forward_length", 0);
    geo_.n_ff_shexp = meta_int(primary, a, "expert_shared_feed_forward_length", 0);
    geo_.full_attn_interval = meta_int(primary, a, "full_attention_interval", 0);
    geo_.ssm_conv = meta_int(primary, a, "ssm.conv_kernel", 0);
    geo_.ssm_state = meta_int(primary, a, "ssm.state_size", 0);
    geo_.ssm_groups = meta_int(primary, a, "ssm.group_count", 0);
    geo_.ssm_dt_rank = meta_int(primary, a, "ssm.time_step_rank", 0);
    geo_.ssm_inner = meta_int(primary, a, "ssm.inner_size", 0);
    geo_.n_ff = meta_int(primary, a, "feed_forward_length", 0);
    geo_.sliding_window = meta_int(primary, a, "attention.sliding_window", 0);
    geo_.head_dim_swa = meta_int(primary, a, "attention.key_length_swa", 0);
    geo_.rope_dim_swa = meta_int(primary, a, "rope.dimension_count_swa", 0);
    geo_.rope_base_swa = meta_float(primary, a, "rope.freq_base_swa", 0);
    geo_.shared_kv_layers = meta_int(primary, a, "attention.shared_kv_layers", 0);
    geo_.n_embd_per_layer = meta_int(primary, a, "embedding_length_per_layer_input", 0);
    geo_.logit_softcap = meta_float(primary, a, "final_logit_softcapping", 0);
    geo_.hc_count = meta_int(primary, a, "hyper_connection.count", 0);
    geo_.hc_rank = meta_int(primary, a, "hyper_connection.low_rank", 0);
    geo_.indexer_heads = meta_int(primary, a, "attention.indexer.head_count", 0);
    geo_.indexer_dim = meta_int(primary, a, "attention.indexer.key_length", 0);
    geo_.indexer_top_k = meta_int(primary, a, "attention.indexer.top_k", 0);
    // A test hook: a smaller selection budget makes the sparse path run on short sequences (CPU and GPU alike).
    if (const char* k = std::getenv("LIMBO_QSA_TOPK"); k && geo_.indexer_top_k > 0) geo_.indexer_top_k = std::atoi(k);
    geo_.ple_ngram = meta_int(primary, a, "ple.ngram_size", 0);
    geo_.ple_heads_per_ngram = meta_int(primary, a, "ple.heads_per_ngram", 0);
    geo_.ple_conv_kernel = meta_int(primary, a, "ple.conv_kernel", 0);
    geo_.ple_eos = meta_int(primary, a, "ple.eos_token_id", -1);
    geo_.expert_weights_scale = meta_float(primary, a, "expert_weights_scale", 0);
    {
        const std::vector<int32_t> cr = primary.get_i32_array(a + ".attention.compress_ratios");
        geo_.compress_ratios.assign(cr.begin(), cr.end());
        auto u64s = [&](const char* suffix, std::vector<uint64_t>* out) {
            const gguf::Value* v = primary.find(a + "." + suffix);
            if (!v || v->type != gguf::VType::ARRAY || v->elem != gguf::VType::U64) return;
            out->resize(v->count);
            std::memcpy(out->data(), v->raw, v->count * sizeof(uint64_t));
        };
        u64s("ple.layer_multipliers", &geo_.ple_multipliers);
        u64s("ple.head_offsets", &geo_.ple_head_offsets);
        u64s("ple.head_vocab_sizes", &geo_.ple_head_vocab);
    }

    if (geo_.n_layer <= 0 || geo_.n_embd <= 0) {
        *err = a + ": block_count or embedding_length is missing";
        return false;
    }

    // A width or a head count the file gives as one value per layer (the key is an array, so the scalar
    // read above found nothing): keep the list, and make the scalar the first entry for callers that
    // only want one number.
    auto per_layer = [&](const char* suffix, int* scalar, std::vector<int>* list) {
        const std::vector<int32_t> arr = primary.get_i32_array(a + "." + suffix);
        if (arr.empty()) return true;
        if (int(arr.size()) != geo_.n_layer) {
            *err = a + "." + suffix + " has " + std::to_string(arr.size()) + " entries for " +
                   std::to_string(geo_.n_layer) + " layers";
            return false;
        }
        list->assign(arr.begin(), arr.end());
        if (*scalar == 0) *scalar = arr[0];
        return true;
    };
    if (!per_layer("feed_forward_length", &geo_.n_ff, &geo_.ff_l)) return false;
    if (!per_layer("attention.head_count", &geo_.n_head, &geo_.head_l)) return false;
    if (!per_layer("attention.head_count_kv", &geo_.n_head_kv, &geo_.head_kv_l)) return false;
    if (const gguf::Value* v = primary.find(a + ".attention.sliding_window_pattern")) {
        if (v->type != gguf::VType::ARRAY || v->elem != gguf::VType::BOOL || int(v->count) != geo_.n_layer) {
            *err = a + ".attention.sliding_window_pattern is not one bool per layer";
            return false;
        }
        geo_.swa_l.assign(v->raw, v->raw + v->count);
    }
    if (geo_.shared_kv_layers < 0 || geo_.shared_kv_layers >= geo_.n_layer) {
        *err = a + ".attention.shared_kv_layers is out of range";
        return false;
    }
    // The looped core, read off the file rather than assumed from the name.
    geo_.loop_start = meta_int(primary, a, "loop.start", 0);
    geo_.loop_end = meta_int(primary, a, "loop.end", 0);
    geo_.loop_count = meta_int(primary, a, "loop.count", 0);
    geo_.loop_gate_rank = meta_int(primary, a, "loop.gate_rank", 0);
    geo_.rel_length = meta_int(primary, a, "attention.relative_length", 0);
    geo_.rel_extent = meta_int(primary, a, "attention.relative_extent", 0);
    geo_.qk_clip = meta_float(primary, a, "attention.qk_clip", 0);
    geo_.rel_clip = meta_float(primary, a, "attention.relative_clip", 0);
    geo_.log_scale_floor = meta_int(primary, a, "attention.log_scale_floor", 0);
    geo_.log_scale_alpha = meta_float(primary, a, "attention.log_scale_alpha", 0);
    geo_.situ_beta = meta_float(primary, a, "activation.situ_beta", 0);
    geo_.situ_linear_beta = meta_float(primary, a, "activation.situ_linear_beta", 0);
    geo_.sconv = meta_int(primary, a, "shortconv.l_cache", 0);
    geo_.attn_res_block = meta_int(primary, a, "attn_res.block_size", 0);
    if (geo_.loop_count > 0 && (geo_.loop_start < 0 || geo_.loop_end <= geo_.loop_start || geo_.loop_end > geo_.n_layer)) {
        *err = a + ".loop: start/end are out of range (" + std::to_string(geo_.loop_start) + ".." +
               std::to_string(geo_.loop_end) + " of " + std::to_string(geo_.n_layer) + " layers)";
        return false;
    }
    if (geo_.arch == "inkling2_loop")
        for (int l = 1; l < geo_.n_layer; ++l) geo_.attn_res_l.push_back(l);
    for (int l = geo_.loop_start; l < geo_.loop_end; ++l) geo_.loop_l.push_back(l);

    // Every tensor of every shard, in shard order.
    for (const auto& f : files_)
        for (const auto& t : f->tensors()) {
            tensors_.push_back(&t);
            total_bytes_ += t.nbytes;
            int layer = 0;
            std::string_view role;
            if (!split_name(t.name, &layer, &role)) {
                unclaimed_.push_back(std::string(t.name));
                continue;
            }
            by_role_[{layer, std::string(role)}] = &t;
            by_name_[std::string(t.name)] = &t;
        }

    // Capabilities, read off the file. The MTP head is a *block* in rung 1 (blk.40.nextn.*), so it is
    // detected by the layer that carries nextn.enorm.weight, not by a global name.
    for (const auto& [key, t] : by_role_) {
        (void)t;
        if (key.first >= 0 && key.second == "nextn.enorm.weight") geo_.mtp_layer = key.first;
    }
    geo_.has_mtp = geo_.mtp_layer >= 0;
    geo_.n_layer_main = geo_.has_mtp ? geo_.n_layer - 1 : geo_.n_layer;
    geo_.has_ple = by_role_.count({-1, "per_layer_token_embd.weight"}) > 0 ||
                   by_role_.count({geo_.mtp_layer >= 0 ? 1 : -1, "ple_key.weight"}) > 0 ||
                   by_role_.count({1, "ple_key.weight"}) > 0;
    geo_.has_hyper = by_role_.count({-1, "output_hc_up.weight"}) > 0;
    geo_.has_indexer = has_meta(primary, a, "attention.indexer.top_k");
    if (const gguf::Tensor* t = tensor(-1, "token_embd.weight")) geo_.n_vocab = int(t->ne[1]);
    geo_.n_vocab_unpadded = meta_int(primary, a, "vocab_size_unpadded", 0);
    // The layers that carry a per-expert scale beside their routed experts. A presence fact of the file:
    // its quantisation recipe put scales on blk.0..19 and none on blk.20..23, and that does NOT follow the
    // down projection's type (17 layers are Q2_0 and 7 are NVFP4 here). Read from the inventory, and only
    // the two .scale roles are allowed to use it -- see the role assertion below.
    for (int l = 0; l < geo_.n_layer; ++l)
        if (by_role_.count({l, "ffn_down_exps.scale"})) geo_.scale_l.push_back(l);

    // A declared layer subset: rung 3's PLE tensors are per-layer (blk.1.ple_*) and the metadata says
    // which layers they belong to, so that list is a legal coverage target alongside all/full/linear.
    std::vector<int> declared_subset;
    for (int32_t l : primary.get_i32_array(a + ".ple.layers"))
        if (l >= 0 && l < geo_.n_layer) declared_subset.push_back(int(l));
    std::sort(declared_subset.begin(), declared_subset.end());

    // The full/linear partition, derived from the file: full-attention layers carry attn_q, linear ones
    // carry attn_qkv. Everything else must agree with this split, which is the assertion that catches a
    // map that has quietly learned the wrong structure.
    for (int l = 0; l < geo_.n_layer; ++l) {
        if (l == geo_.mtp_layer) continue;  // the MTP block carries its own attention
        // A file with GDN keys splits into attention and linear layers by which projection a layer carries; a
        // file without them (gemma) is all attention, whether it ships q, k and v or one fused qkv.
        const bool gdn_file = geo_.ssm_inner > 0;
        const bool full = by_role_.count({l, "attn_q.weight"}) > 0 ||
                          (!gdn_file && by_role_.count({l, "attn_qkv.weight"}) > 0);
        const bool linear = gdn_file && by_role_.count({l, "attn_qkv.weight"}) > 0;
        if (full == linear) {
            *err = "layer " + std::to_string(l) + " is " + (full ? "both full and linear" : "neither full nor linear");
            return false;
        }
        (full ? full_layers_ : linear_layers_).push_back(l);
    }
    if (geo_.full_attn_interval > 0) {
        for (int l = 0; l < geo_.n_layer; ++l) {
            if (l == geo_.mtp_layer) continue;
            const bool declared = geo_.is_full_attention(l);
            const bool actual = std::find(full_layers_.begin(), full_layers_.end(), l) != full_layers_.end();
            if (declared != actual) {
                *err = "layer " + std::to_string(l) + " is " + (actual ? "full" : "linear") +
                       " in the file but full_attention_interval=" + std::to_string(geo_.full_attn_interval) +
                       " says otherwise";
                return false;
            }
        }
    }

    // Every per-layer role must cover whole structural classes, never part of one. A class is the set of
    // layers that share (kind, sliding window or global, owns its K/V cache or reads another's): the
    // full-attention layers, the linear ones, the MTP block, gemma's sliding and global layers, its
    // shared-KV tail. That is the assertion that catches a map which has quietly learned the wrong
    // structure, and it is what caught the MTP block and the PLE layer here.
    auto kind_of = [&](int l) {
        if (l == geo_.mtp_layer) return 2;
        return std::find(linear_layers_.begin(), linear_layers_.end(), l) != linear_layers_.end() ? 1 : 0;
    };
    auto class_of = [&](int l) {
        return kind_of(l) * 4 + (geo_.is_swa(l) ? 2 : 0) + (geo_.has_own_kv(l) ? 1 : 0);
    };
    std::map<std::string, std::vector<int>> role_layers;
    for (const auto& [key, t] : by_role_) {
        (void)t;
        role_layers[key.second].push_back(key.first);
    }
    for (auto& [role, layers] : role_layers) {
        std::sort(layers.begin(), layers.end());
        const bool global = layers.size() == 1 && layers[0] == -1;
        bool whole = !global && layers.front() >= 0;
        if (whole) {
            std::map<int, int> have, total;
            for (int l : layers) ++have[class_of(l)];
            for (int l = 0; l < geo_.n_layer; ++l) ++total[class_of(l)];
            for (const auto& [cls, n] : have)
                if (n != total[cls]) whole = false;
        }
        const bool ok = global || whole || (!declared_subset.empty() && layers == declared_subset);
        // Two declared targets beyond whole-class coverage, each tied to the role it belongs to so the
        // assertion stays sharp for every other role: the layers the file gives attn_res_query (layer 0
        // has no attention residual to read), and the layers it gives a per-expert scale (a quantisation
        // fact -- the tensor is either there or not).
        const bool declared_role =
            (role == "attn_res_query.weight" && !geo_.attn_res_l.empty() && layers == geo_.attn_res_l) ||
            ((role == "ffn_down_exps.scale" || role == "ffn_gate_up_exps.scale") && !geo_.scale_l.empty() &&
             layers == geo_.scale_l) ||
            (role.rfind("loop_", 0) == 0 && !geo_.loop_l.empty() && layers == geo_.loop_l);
        const bool ok2 = ok || declared_role;
        if (!ok2) {
            std::string which;
            for (int l : layers) which += std::to_string(l) + " ";
            *err = "role " + role + " covers layers [" + which +
                   "] which is not a union of the structural classes (kind, sliding, owns-KV), not a global role, "
                   "and not the declared subset";
            return false;
        }
    }
    if (!unclaimed_.empty()) {
        *err = std::to_string(unclaimed_.size()) + " tensor(s) no role claims, first: " + unclaimed_.front();
        return false;
    }
    return true;
}

}  // namespace limbo::model
