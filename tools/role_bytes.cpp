// Copyright (c) 2026 ITERFORM. All rights reserved.
// tools/role_bytes.cpp — what a model's roles weigh: bytes per role over all layers, the size of one routed expert at
// each layer (gate + up + down rows of one expert) and the tensors' element counts by group.
//
//   role_bytes shard1.gguf [shard2.gguf ...]
#include "limbo/model.hpp"

#include <cstdio>
#include <map>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: role_bytes shard1.gguf [shard2.gguf ...]\n");
        return 2;
    }
    std::string err;
    auto m = limbo::model::Model::open(std::vector<std::string>(argv + 1, argv + argc), &err);
    if (!m) {
        std::fprintf(stderr, "role_bytes: %s\n", err.c_str());
        return 1;
    }
    std::map<std::string, uint64_t> bytes;
    std::map<std::string, int> count;
    double hc = 0, head = 0, experts = 0, dense = 0;   // element counts
    for (const auto* t : m->tensors()) {
        const std::string n(t->name);
        std::string role = n;
        if (n.rfind("blk.", 0) == 0) role = "blk.*" + n.substr(n.find('.', 4));
        bytes[role] += t->nbytes;
        count[role]++;
        const double e = double(t->n_elements());
        if (n.find("_norm") != std::string::npos || n.find("norm.") != std::string::npos) continue;
        if (n.find("hc_") != std::string::npos) hc += e;
        else if (n == "output.weight") head += e;
        else if (n.find("_exps") != std::string::npos) experts += e;
        else if (n != "token_embd.weight" && n != "per_layer_token_embd.weight") dense += e;
    }
    for (const auto& kv : bytes) std::printf("%-40s n=%3d  %10.2f MiB\n", kv.first.c_str(), count[kv.first], double(kv.second) / 1048576.0);
    const auto& g = m->geometry();
    double total = 0;
    for (int l = 0; l < g.n_layer; ++l) {
        double b = 0;
        for (const char* r : {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_gate_up_exps.weight",
                              "ffn_down_exps.weight"})
            if (const auto* t = m->tensor(l, r)) b += double(t->nbytes) / double(g.n_expert);
        total += b * g.n_expert;
        std::printf("layer %2d: one expert %.3f MiB\n", l, b / 1048576.0);
    }
    std::printf("routed experts: %.2f GiB\n", total / 1073741824.0);
    std::printf("elements (G): hyper-connections %.3f, output head %.3f, other dense %.3f, routed experts %.3f (%d of %d per token: %.3f)\n", hc / 1e9,
                head / 1e9, dense / 1e9, experts / 1e9, g.n_expert_used, g.n_expert, experts / 1e9 * g.n_expert_used / std::max(1, g.n_expert));
    return 0;
}
