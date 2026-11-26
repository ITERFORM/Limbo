// Copyright (c) 2026 ITERFORM. All rights reserved.
// tools/ref_forward.cpp — run the reference forward pass and print what the oracle prints.
//
//   ref_forward -m model.gguf -f bench/prompts/code.txt [--name code] [--n-predict 4] [--threads 8]
//   ref_forward -m model.gguf --cmp-dump eval-callback.txt [-t 8] [--oracle-arith]
//   ref_forward -m model.gguf --layer-gate l_out.bin [--attn-residual attn_residual.bin] [-t 8] [--oracle-arith]
//               [--verbose] [--ctx N] [--first-layer A] [--last-layer B]
//
// --layer-gate reads a binary capture of every l_out-N node in full (written by a scratch program that links
// libllama and runs the prompt as one CPU batch) and runs EACH layer from the oracle's own input to that layer,
// over every position in order, comparing the result with the oracle's output of the same layer on all 2048
// elements. Nothing accumulates from one layer to the next, so a structural error in one layer shows as a spike
// there instead of vanishing into the drift of the whole stack.
//
// --cmp-dump reads the output of `llama-eval-callback -p <short text> -n 1` (it carries the prompt's token
// ids and, per node, the first and last three values of every token row plus the whole tensor's sum),
// runs the same ids through the reference and prints, per layer, how far attn_residual-N and l_out-N are
// from the oracle's on every row the dump shows. That separates arithmetic noise (a smooth growth with
// depth) from a structural bug (a jump at one layer, or at one position).
//
// The prompt file is the same one oracle/oracle_dump reads (a line "@@ <name>" then the text), so the
// two outputs can be compared directly: greedy token ids and, per step, the top-2 ids with their gap in
// nats, which is what PLAN.md §3.3's near-tie rule needs.
#include "limbo/gguf.hpp"
#include "limbo/model.hpp"
#include "limbo/step.hpp"
#include "limbo/tok.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <sstream>
#include <array>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

std::string read_prompt(const char* path, const std::string& want) {
    std::ifstream in(path, std::ios::binary);
    std::string line, cur_name, cur_text;
    bool have = false;
    std::string found;
    while (std::getline(in, line)) {
        if (line.rfind("@@ ", 0) == 0) {
            if (have && (want.empty() || cur_name == want)) found = cur_text;
            cur_name = line.substr(3);
            cur_text.clear();
            have = true;
            continue;
        }
        if (have) {
            cur_text += line;
            cur_text += '\n';
        }
    }
    if (have && (want.empty() || cur_name == want)) found = cur_text;
    return found;
}


// ---- the eval-callback dump ----
struct DumpNode {
    std::vector<int> ne;                        // {n_embd, rows, 1, 1}
    std::vector<std::vector<float>> rows;       // per row: 3 head + 3 tail values (or all when short)
    double sum = 0;
};

struct Dump {
    std::vector<int32_t> ids;
    std::map<std::string, DumpNode> nodes;      // the LAST occurrence of a name wins (l_out-N is unique)
};

bool parse_dump(const char* path, Dump& d) {
    std::ifstream in(path);
    if (!in) return false;
    std::string line;
    bool in_ids = false;
    DumpNode* cur = nullptr;
    while (std::getline(in, line)) {
        if (line.find("number of input tokens") != std::string::npos) {
            in_ids = true;
            continue;
        }
        if (in_ids) {
            const size_t p = line.find(" I   ");
            if (p != std::string::npos) {
                d.ids.push_back(int32_t(std::atoi(line.c_str() + p + 5)));
                continue;
            }
            in_ids = false;
        }
        const size_t pc = line.find("common_debug_cb_eval:");
        if (pc != std::string::npos) {
            std::istringstream ss(line.substr(pc + 21));
            std::string name;
            ss >> name;
            const size_t pb = line.rfind("= {");
            DumpNode n;
            if (pb != std::string::npos) {
                std::string dims = line.substr(pb + 3);
                for (char& ch : dims) if (ch == ',' || ch == '}') ch = ' ';
                std::istringstream ds(dims);
                int v;
                while (ds >> v) n.ne.push_back(v);
            }
            cur = &(d.nodes[name] = n);
            continue;
        }
        if (!cur) continue;
        const size_t ps = line.find("sum = ");
        if (ps != std::string::npos) {
            cur->sum = std::atof(line.c_str() + ps + 6);
            continue;
        }
        // a value row: "[   v,  v,  v,   ...,   v,  v,  v  ],"
        const size_t lb = line.find('[');
        const size_t rb = line.find(']');
        if (lb == std::string::npos || rb == std::string::npos || line.find("...,", lb) == std::string::npos) {
            if (lb != std::string::npos && rb != std::string::npos && rb > lb + 1 && cur->ne.size() && cur->ne[0] <= 6) {
                // a short row: all values, no ellipsis
            } else {
                continue;
            }
        }
        std::string body = line.substr(lb + 1, rb - lb - 1);
        const size_t el = body.find("...");
        if (el != std::string::npos) body.erase(el, 3);
        for (char& ch : body) if (ch == ',') ch = ' ';
        std::istringstream bs(body);
        std::vector<float> row;
        float v;
        while (bs >> v) row.push_back(v);
        if (!row.empty()) cur->rows.push_back(row);
    }
    return !d.ids.empty();
}

// Prints, per layer and node, the largest |limbo - oracle| over the six shown values of each row, next to the
// oracle's largest |value| on that row (a scale), and the relative error of the tensor sum.
void compare_dump(const Dump& d, const limbo::step::Trace& tr, int n_main) {
    const int n_pos = int(d.ids.size());
    std::printf("%-6s %-14s %-4s %12s %12s %12s %14s %14s\n", "layer", "node", "row", "max|d|", "oracle|max|", "rel", "sum oracle",
                "sum limbo");
    for (int l = 0; l < n_main; ++l) {
        for (int which = 0; which < 2; ++which) {
            const std::string name = std::string(which ? "l_out-" : "attn_residual-") + std::to_string(l);
            auto it = d.nodes.find(name);
            if (it == d.nodes.end()) continue;
            const DumpNode& nd = it->second;
            const int n_rows = int(nd.rows.size());
            if (n_rows == 0) continue;
            double lsum = 0;
            const int first_pos = n_pos - n_rows;   // a cropped last layer shows only the final rows
            for (int r = 0; r < n_rows; ++r) {
                const int pos = first_pos + r;
                const auto& vv = which ? tr.l_out_pos[size_t(pos)][size_t(l)] : tr.attn_out_pos[size_t(pos)][size_t(l)];
                for (float x : vv) lsum += x;
                const std::vector<float>& o = nd.rows[size_t(r)];
                double md = 0, mo = 0;
                for (size_t i = 0; i < o.size(); ++i) {
                    const size_t idx = i < 3 ? i : vv.size() - (o.size() - i);
                    md = std::max(md, std::fabs(double(vv[idx]) - double(o[i])));
                    mo = std::max(mo, std::fabs(double(o[i])));
                }
                std::printf("%-6d %-14s %-4d %12.5f %12.4f %12.5f\n", l, which ? "l_out" : "attn_residual", pos, md, mo, md / (mo > 0 ? mo : 1));
            }
            std::printf("%-6d %-14s %-4s %12s %12s %12.5f %14.5f %14.5f\n", l, which ? "l_out" : "attn_residual", "sum", "", "",
                        std::fabs(lsum - nd.sum) / std::max(1e-9, std::fabs(nd.sum)), nd.sum, lsum);
        }
    }
}

// The routed experts: the dump's ffn_moe_topk-N holds the 8 chosen ids per token, of which it prints three at
// each end. Reports every (layer, row) whose shown ids are not all among the reference's 8 (a different
// expert in a layer is a different layer output, whatever the arithmetic), and the same-set-different-order
// count, which is harmless.
void compare_routing(const Dump& d, const limbo::step::Trace& tr, int n_main) {
    const int n_pos = int(d.ids.size());
    int checked = 0, bad = 0, reordered = 0;
    for (int l = 0; l < n_main; ++l) {
        auto it = d.nodes.find("ffn_moe_topk-" + std::to_string(l));
        if (it == d.nodes.end()) continue;
        const DumpNode& nd = it->second;
        const int n_rows = int(nd.rows.size());
        for (int r = 0; r < n_rows; ++r) {
            const int pos = n_pos - n_rows + r;
            const std::vector<float>& mt = tr.moe_pos[size_t(pos)][size_t(l)];
            const std::vector<float>& o = nd.rows[size_t(r)];
            ++checked;
            bool all_in = true, same_order = true;
            for (size_t i = 0; i < o.size(); ++i) {
                bool in = false;
                for (int k = 0; k < 8; ++k) in = in || int(mt[size_t(k)]) == int(o[i]);
                all_in = all_in && in;
                const size_t idx = i < 3 ? i : 8 - (o.size() - i);
                same_order = same_order && int(mt[idx]) == int(o[i]);
            }
            if (!all_in) {
                ++bad;
                std::printf("routing DIFF layer %d row %d: oracle", l, pos);
                for (float x : o) std::printf(" %d", int(x));
                std::printf("  limbo");
                for (int k = 0; k < 8; ++k) std::printf(" %d", int(mt[size_t(k)]));
                std::printf("  (weights");
                for (int k = 8; k < 16; ++k) std::printf(" %.4f", mt[size_t(k)]);
                std::printf(")\n");
            } else if (!same_order) {
                ++reordered;
            }
        }
    }
    std::printf("routing: %d (layer,row) pairs checked, %d with a shown expert missing from the reference's 8, %d same set in a different order\n",
                checked, bad, reordered);
}

// ---- the layer gate ----
//
// A capture of one node family (l_out-N or attn_residual-N) of every layer, in full, written by a scratch program
// that links libllama (bench/w1-cpu-gate-2026-11-19/scratch/oracle_nodes.cpp): int32 n_tokens, n_layers, n_embd; the
// prompt ids; per layer int32 layer, rows, rows*n_embd floats (the last layer is cropped to the final row); then
// the routed experts per layer (int32 n, then layer, rows, rows*8 ids) and the last row's logits.
struct Nodes {
    int n = 0, ne = 0;
    std::vector<int32_t> ids;
    std::vector<std::vector<float>> v;            // [layer] rows*ne, empty when the layer was not captured
    std::vector<int> rows;
    std::vector<std::vector<int32_t>> topk;       // [layer] rows*n_used
    std::vector<int> topk_rows;
    std::vector<float> logits;

    // The row of `layer` that belongs to position p, or nullptr (a cropped or absent layer).
    const float* row(int layer, int p) const {
        if (layer < 0 || layer >= int(v.size()) || v[size_t(layer)].empty()) return nullptr;
        const int r = rows[size_t(layer)];
        if (p < n - r) return nullptr;
        return v[size_t(layer)].data() + size_t(p - (n - r)) * size_t(ne);
    }
};

bool read_nodes(const char* path, Nodes& nd, int n_layers, int n_used = 8) {
    std::FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    int32_t hdr[3];
    if (std::fread(hdr, 4, 3, f) != 3) return false;
    nd.n = hdr[0];
    nd.ne = hdr[2];
    nd.ids.resize(static_cast<size_t>(nd.n));
    if (std::fread(nd.ids.data(), 4, nd.ids.size(), f) != nd.ids.size()) return false;
    nd.v.assign(static_cast<size_t>(n_layers), {});
    nd.rows.assign(static_cast<size_t>(n_layers), 0);
    nd.topk.assign(static_cast<size_t>(n_layers), {});
    nd.topk_rows.assign(static_cast<size_t>(n_layers), 0);
    for (int i = 0; i < hdr[1]; ++i) {
        int32_t layer, r;
        if (std::fread(&layer, 4, 1, f) != 1 || std::fread(&r, 4, 1, f) != 1) return false;
        if (layer < 0 || layer >= n_layers) {
            std::fprintf(stderr, "ref_forward: the capture has a node of layer %d, the model has %d main layers\n", layer, n_layers);
            return false;
        }
        nd.v[static_cast<size_t>(layer)].resize(static_cast<size_t>(r) * static_cast<size_t>(nd.ne));
        nd.rows[static_cast<size_t>(layer)] = r;
        if (std::fread(nd.v[static_cast<size_t>(layer)].data(), 4, nd.v[static_cast<size_t>(layer)].size(), f) !=
            nd.v[static_cast<size_t>(layer)].size())
            return false;
    }
    int32_t nt = 0;
    if (std::fread(&nt, 4, 1, f) == 1) {
        for (int i = 0; i < nt; ++i) {
            int32_t layer, r;
            if (std::fread(&layer, 4, 1, f) != 1 || std::fread(&r, 4, 1, f) != 1) break;
            if (layer < 0 || layer >= n_layers) break;
            auto& t = nd.topk[static_cast<size_t>(layer)];
            t.resize(static_cast<size_t>(r) * static_cast<size_t>(n_used));
            nd.topk_rows[static_cast<size_t>(layer)] = r;
            if (std::fread(t.data(), 4, t.size(), f) != t.size()) break;
        }
    }
    int32_t nl = 0;
    if (std::fread(&nl, 4, 1, f) == 1 && nl > 0) {
        nd.logits.resize(static_cast<size_t>(nl));
        if (std::fread(nd.logits.data(), 4, nd.logits.size(), f) != nd.logits.size()) nd.logits.clear();
    }
    std::fclose(f);
    return true;
}

// Error of x against the oracle's o, relative to the oracle's output and to the step's update (o - in).
struct Err {
    double rel_out = 0, rel_upd = 0;
};

Err compare_vec(const float* x, const float* o, const float* in, int ne) {
    double d2 = 0, o2 = 0, u2 = 0;
    for (int i = 0; i < ne; ++i) {
        const double d = double(x[i]) - double(o[i]);
        const double u = double(o[i]) - double(in[i]);
        d2 += d * d;
        o2 += double(o[i]) * double(o[i]);
        u2 += u * u;
    }
    return {std::sqrt(d2 / std::max(o2, 1e-30)), std::sqrt(d2 / std::max(u2, 1e-30))};
}

struct Acc {
    double s_out = 0, s_upd = 0, m_out = 0, m_upd = 0;
    int cnt = 0, worst = -1;
    double bsum[4] = {0, 0, 0, 0};   // mean rel_upd by position: [0,128) [128,512) [512,2048) [2048,..)
    int bcnt[4] = {0, 0, 0, 0};
    void add(const Err& e, int p) {
        s_out += e.rel_out;
        s_upd += e.rel_upd;
        ++cnt;
        if (e.rel_upd > m_upd) { m_upd = e.rel_upd; worst = p; }
        m_out = std::max(m_out, e.rel_out);
        const int b = p < 128 ? 0 : p < 512 ? 1 : p < 2048 ? 2 : 3;
        bsum[b] += e.rel_upd;
        ++bcnt[b];
    }
};

void print_acc(const char* label, int l, const char* kind, const Acc& a, int flips, int n_pos) {
    std::printf("%-5d %-4s %-6s %12.6f %12.6f %12.6f %12.6f %8d %6d\n", l, kind, label, a.cnt ? a.s_out / a.cnt : 0, a.m_out,
                a.cnt ? a.s_upd / a.cnt : 0, a.m_upd, a.worst, flips);
    if (n_pos > 600) {
        std::printf("                    mean rel_upd by position:");
        const char* names[4] = {"[0,128)", "[128,512)", "[512,2048)", "[2048,..)"};
        for (int b = 0; b < 4; ++b)
            if (a.bcnt[b]) std::printf("  %s %.5f", names[b], a.bsum[b] / a.bcnt[b]);
        std::printf("\n");
    }
}

// Every layer fed the oracle's own input. `lo` holds l_out-N; `ar` (optional) holds attn_residual-N, which also
// splits each layer into its attention half (l_out-(N-1) -> attn_residual-N) and its MoE half (attn_residual-N ->
// l_out-N), so an error is attributed to one half instead of to the layer.
int layer_gate(const Nodes& lo, const Nodes* ar, limbo::step::Ref& ref, bool verbose, int first_layer, int last_layer) {
    const int n = lo.n, ne = lo.ne;
    std::printf("layer-gate: %d tokens, n_embd %d%s\n", n, ne, ar ? ", attention/MoE split" : "");
    std::printf("%-5s %-4s %-6s %12s %12s %12s %12s %8s %6s\n", "layer", "kind", "part", "mean rel_out", "max rel_out", "mean rel_upd",
                "max rel_upd", "worst row", "flips");
    const auto& layers = ref.config().layers;
    const int n_layers = ref.config().n_layer_main;
    const bool hc = ref.config().hc_count > 0;   // rows are the whole stack of hyper-connection streams
    const bool has_ple = ref.config().n_embd_per_layer > 0 && !hc;
    const int n_used = ref.config().n_expert_used;
    const int e = ref.config().n_embd;
    std::vector<float> x(static_cast<size_t>(ne)), in(static_cast<size_t>(ne)), tok_embd(static_cast<size_t>(ne));
    int routing_rows = 0, routing_flips = 0, n_same = 0, n_flip = 0;
    double upd_same = 0, upd_flip = 0;
    ref.reset();
    for (int l = std::max(0, first_layer); l < n_layers && (last_layer < 0 || l <= last_layer); ++l) {
        if (lo.v[static_cast<size_t>(l)].empty() || (l > 0 && lo.v[static_cast<size_t>(l - 1)].empty())) continue;
        const limbo::step::LayerRecipe& lr = layers[static_cast<size_t>(l)];
        const char* kind = lr.mixer == limbo::step::Mixer::Gdn ? "GDN" : lr.attn.sliding ? "SWA" : "ATT";
        Acc whole, half_a, half_f;
        int flips_here = 0;
        for (int p = 0; p < n; ++p) {
            // embed() also loads the token's per-layer inputs, which a layer of a per-layer-embedding model reads.
            if (hc) ref.note_token(p, lo.ids[static_cast<size_t>(p)]);
            if (l == 0) {
                ref.embed(lo.ids[static_cast<size_t>(p)], in.data());
                for (int s = 1; hc && s * e < ne; ++s) std::copy(in.begin(), in.begin() + e, in.begin() + static_cast<size_t>(s) * e);
            } else {
                // A model that prunes rows before its last layers (gemma's shared-KV tail) captures only the
                // final row there: a layer without its input row is skipped (such a layer keeps no state).
                if (!lo.row(l - 1, p)) continue;
                if (has_ple) ref.embed(lo.ids[static_cast<size_t>(p)], tok_embd.data());
                std::copy(lo.row(l - 1, p), lo.row(l - 1, p) + ne, in.begin());
            }
            const float* o = lo.row(l, p);
            const float* a = ar ? ar->row(l, p) : nullptr;
            // The layer's state advances once per position, so a split run never calls forward_layer as well.
            x = in;
            Err e_whole, e_a, e_f;
            bool have_split = false;
            if (a && o) {
                ref.forward_layer_attn(l, p, x.data());            // attention half from the oracle's input
                e_a = compare_vec(x.data(), a, in.data(), ne);
                ref.forward_layer_ffn(l, x.data());                 // ... then MoE on Limbo's own attention output
                e_whole = compare_vec(x.data(), o, in.data(), ne);
                std::copy(a, a + ne, x.begin());                    // MoE half from the oracle's attn_residual
                ref.forward_layer_ffn(l, x.data());
                e_f = compare_vec(x.data(), o, a, ne);
                have_split = true;
            } else {
                ref.forward_layer(l, p, x.data());
                if (o) e_whole = compare_vec(x.data(), o, in.data(), ne);
            }
            if (!o) continue;
            // Did Limbo route to the same 8 experts as the oracle on this row? (In a split run the last MoE call
            // is the isolated one, which is the fair comparison.)
            bool flip = false;
            const int tr = lo.topk_rows[static_cast<size_t>(l)];
            if (tr > 0 && p >= n - tr) {
                const int32_t* oi = lo.topk[static_cast<size_t>(l)].data() + static_cast<size_t>(p - (n - tr)) * static_cast<size_t>(n_used);
                const std::vector<int>& li = ref.trace().moe_last_ids;
                for (int k = 0; k < n_used && !flip; ++k) {
                    bool hit = false;
                    for (size_t j = 0; j < li.size(); ++j) hit = hit || li[j] == oi[k];
                    flip = !hit;
                }
                ++routing_rows;
                routing_flips += flip;
                flips_here += flip;
                (flip ? upd_flip : upd_same) += e_whole.rel_upd;
                (flip ? n_flip : n_same) += 1;
            }
            whole.add(e_whole, p);
            if (verbose) {
                std::printf("  layer %d row %d rel_out %.6f rel_upd %.6f", l, p, e_whole.rel_out, e_whole.rel_upd);
                const std::vector<float>& lp = ref.trace().moe_last_p;
                if (flip && int(lp.size()) > n_used)
                    std::printf("  ROUTING DIFFERS (limbo's last chosen and next router probabilities %.5f %.5f, gap %.2f%%)", lp[size_t(n_used) - 1],
                                lp[size_t(n_used)], 100.0 * (lp[size_t(n_used) - 1] - lp[size_t(n_used)]) / lp[size_t(n_used) - 1]);
                std::printf("\n");
            }
            if (have_split) {
                half_a.add(e_a, p);
                half_f.add(e_f, p);
            }
        }
        print_acc("layer", l, kind, whole, flips_here, n);
        if (ar && half_a.cnt) {
            print_acc("attn", l, kind, half_a, 0, n);
            print_acc("moe", l, kind, half_f, 0, n);
        }
    }
    // The head on its own: the oracle's last hidden state (l_out-39, last row) through Limbo's output norm and LM head.
    if (!lo.logits.empty() && lo.row(n_layers - 1, n - 1)) {
        const float* lg = ref.head(lo.row(n_layers - 1, n - 1));
        const size_t nv = std::min(lo.logits.size(), static_cast<size_t>(ref.n_vocab()));
        double d2 = 0, o2 = 0, md = 0;
        size_t a_l = 0, a_o = 0;
        for (size_t i = 0; i < nv; ++i) {
            const double d = double(lg[i]) - double(lo.logits[i]);
            d2 += d * d;
            o2 += double(lo.logits[i]) * double(lo.logits[i]);
            md = std::max(md, std::fabs(d));
            if (lg[i] > lg[a_l]) a_l = i;
            if (lo.logits[i] > lo.logits[a_o]) a_o = i;
        }
        std::printf("head (output norm + LM head on the oracle's last hidden state): max|dlogit| %.5f, rel L2 %.6f, argmax limbo %zu oracle %zu\n",
                    md, std::sqrt(d2 / std::max(o2, 1e-30)), a_l, a_o);
    }
    if (routing_rows > 0)
        std::printf("routing: %d (layer,row) pairs, %d with a different expert set (%.2f%%); mean rel_upd %.5f over the %d rows where the set agrees, %.5f over the %d where it differs\n",
                    routing_rows, routing_flips, 100.0 * routing_flips / routing_rows, n_same ? upd_same / n_same : 0.0, n_same,
                    n_flip ? upd_flip / n_flip : 0.0, n_flip);
    return 0;
}


// How early can the router's choice be known? For every layer m and every k, layer m's own router is fed the stack that
// entered layer m-k (the oracle's l_last of layer m-k-1; the embedding stack for m-k = 0), through layer m's FFN-side
// hyper-connection read, and its 10 experts are compared with the experts the oracle's run really chose at layer m.
int route_ahead(const Nodes& lo, limbo::step::Ref& ref) {
    const int n = lo.n, ne = lo.ne, nl = ref.config().n_layer_main, e = ref.config().n_embd, K = ref.config().n_expert_used;
    std::printf("route-ahead: %d tokens, stack row %d, %d experts per token\n", n, ne, K);
    std::printf("%4s %14s %14s\n", "k", "recall@10", "positions");
    for (int k : {0, 1, 2, 3, 4, 6, 8}) {
        long hit = 0, tot = 0;
        for (int m = std::max(1, k + 1); m < nl; ++m) {
            const int tr = lo.topk_rows[static_cast<size_t>(m)];
            if (tr < n) continue;   // a layer the oracle cropped has no routing for every row
            for (int p = 0; p < n; ++p) {
                const float* st = k == m ? nullptr : lo.row(m - k - 1, p);
                if (!st) continue;
                const std::vector<int> pred = ref.route_from_stack(m, st);
                const int32_t* oi = lo.topk[static_cast<size_t>(m)].data() + static_cast<size_t>(p) * static_cast<size_t>(K);
                for (int j = 0; j < K; ++j) {
                    ++tot;
                    for (int q : pred) if (q == oi[j]) { ++hit; break; }
                }
            }
        }
        std::printf("%4d %13.1f%% %14ld\n", k, tot ? 100.0 * double(hit) / double(tot) : 0.0, tot / K);
    }
    (void)e;
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const char* model = nullptr;
    const char* prompt_file = nullptr;
    std::string name;
    int n_predict = 4;
    int threads = 8;
    int n_ctx = 512;
    int layers = 0;
    bool dump_nodes = false;
    bool oracle_arith = false;
    const char* cmp_dump = nullptr;
    const char* layer_file = nullptr;
    const char* attn_file = nullptr;
    bool route_ahead_mode = false;
    bool verbose = false;
    int first_layer = 0, last_layer = -1;

    for (int i = 1; i < argc; ++i) {
        auto is = [&](const char* a) { return std::strcmp(argv[i], a) == 0; };
        if ((is("-m") || is("--model")) && i + 1 < argc) model = argv[++i];
        else if ((is("-f") || is("--file")) && i + 1 < argc) prompt_file = argv[++i];
        else if (is("--name") && i + 1 < argc) name = argv[++i];
        else if (is("--n-predict") && i + 1 < argc) n_predict = std::atoi(argv[++i]);
        else if ((is("-t") || is("--threads")) && i + 1 < argc) threads = std::atoi(argv[++i]);
        else if (is("--ctx") && i + 1 < argc) n_ctx = std::atoi(argv[++i]);
        else if (is("--layers") && i + 1 < argc) layers = std::atoi(argv[++i]);
        else if (is("--dump-nodes")) dump_nodes = true;
        else if (is("--cmp-dump") && i + 1 < argc) cmp_dump = argv[++i];
        else if (is("--layer-gate") && i + 1 < argc) layer_file = argv[++i];
        else if (is("--attn-residual") && i + 1 < argc) attn_file = argv[++i];
        else if (is("--route-ahead")) route_ahead_mode = true;
        else if (is("--verbose")) verbose = true;
        else if (is("--first-layer") && i + 1 < argc) first_layer = std::atoi(argv[++i]);
        else if (is("--last-layer") && i + 1 < argc) last_layer = std::atoi(argv[++i]);
        else if (is("--oracle-arith")) oracle_arith = true;
        else {
            std::fprintf(stderr,
                         "usage: ref_forward -m model.gguf -f prompts.txt [--name NAME] [--n-predict N] "
                         "[-t threads] [--ctx N]\n");
            return 2;
        }
    }
    if (!model || (!prompt_file && !cmp_dump && !layer_file)) {
        std::fprintf(stderr, "ref_forward: -m and -f are required\n");
        return 2;
    }

    std::string err;
    auto gguf = limbo::gguf::File::open(model, &err);
    if (!gguf) {
        std::fprintf(stderr, "ref_forward: %s\n", err.c_str());
        return 1;
    }
    limbo::tok::Vocab vocab;
    if (!limbo::tok::Vocab::load(*gguf, vocab, &err)) {
        std::fprintf(stderr, "ref_forward: %s\n", err.c_str());
        return 1;
    }
    const limbo::tok::Tokenizer tok(vocab);

    auto ref = limbo::step::Ref::open({model}, n_ctx, threads, &err);
    if (!ref) {
        std::fprintf(stderr, "ref_forward: %s\n", err.c_str());
        return 1;
    }
    if (layers > 0) ref->set_layer_limit(layers);
    ref->set_oracle_arith(oracle_arith);

    if (layer_file) {
        ref->trace().on = true;   // the routing of each call is read back from the trace
        Nodes lo, ar;
        if (!read_nodes(layer_file, lo, ref->config().n_layer_main, ref->config().n_expert_used)) {
            std::fprintf(stderr, "ref_forward: cannot read %s\n", layer_file);
            return 1;
        }
        if (attn_file && !read_nodes(attn_file, ar, ref->config().n_layer_main, ref->config().n_expert_used)) {
            std::fprintf(stderr, "ref_forward: cannot read %s\n", attn_file);
            return 1;
        }
        if (route_ahead_mode) return route_ahead(lo, *ref);
        return layer_gate(lo, attn_file ? &ar : nullptr, *ref, verbose, first_layer, last_layer);
    }

    if (cmp_dump) {
        Dump d;
        if (!parse_dump(cmp_dump, d)) {
            std::fprintf(stderr, "ref_forward: cannot read a dump from %s\n", cmp_dump);
            return 1;
        }
        std::fprintf(stderr, "ref_forward: dump has %zu prompt tokens, %zu nodes\n", d.ids.size(), d.nodes.size());
        ref->trace().on = true;
        ref->trace().all_pos = true;
        const float* lg = nullptr;
        for (size_t i = 0; i < d.ids.size(); ++i) lg = ref->forward(d.ids[i], int(i));
        compare_dump(d, ref->trace(), ref->config().n_layer_main);
        compare_routing(d, ref->trace(), ref->config().n_layer_main);
        // The last row's logits: the dump prints the first and last three of the 248,320.
        auto it = d.nodes.find("result_output");
        if (it != d.nodes.end() && !it->second.rows.empty() && lg) {
            const std::vector<float>& o = it->second.rows.back();
            double md = 0;
            for (size_t i = 0; i < o.size(); ++i) {
                const size_t idx = i < 3 ? i : size_t(ref->n_vocab()) - (o.size() - i);
                md = std::max(md, std::fabs(double(lg[idx]) - double(o[i])));
            }
            std::printf("result_output (first/last 3 logits of the last row) max|d| %.5f\n", md);
        }
        return 0;
    }
    const int n_vocab = ref->n_vocab();

    const std::string text = read_prompt(prompt_file, name);
    std::vector<int32_t> ids;
    limbo::tok::Options o;
    o.parse_special = true;  // the same flags the oracle's capture used
    tok.encode(text, ids, o);
    std::fprintf(stderr, "ref_forward: %zu prompt tokens, %d threads, n_ctx %d\n", ids.size(), threads, n_ctx);

    if (dump_nodes) ref->trace().on = true;
    std::vector<int32_t> greedy;
    std::vector<std::array<double, 5>> top2;
    int pos = 0;
    const float* logits = nullptr;
    for (size_t i = 0; i < ids.size(); ++i, ++pos) logits = ref->forward(ids[i], pos);

    for (int step = 0; step < n_predict; ++step) {
        int i1 = 0, i2 = 0;
        float v1 = -INFINITY, v2 = -INFINITY;
        for (int v = 0; v < n_vocab; ++v) {
            const float x = logits[v];
            if (x > v1) {
                v2 = v1;
                i2 = i1;
                v1 = x;
                i1 = v;
            } else if (x > v2) {
                v2 = x;
                i2 = v;
            }
        }
        greedy.push_back(i1);
        top2.push_back({double(i1), double(v1), double(i2), double(v2), double(v1) - double(v2)});
        logits = ref->forward(i1, pos);
        ++pos;
    }

    if (dump_nodes) {
        auto pr = [](const char* n, const std::vector<float>& v) {
            std::fprintf(stderr, "%-22s", n);
            for (size_t i = 0; i < v.size() && i < 12; ++i) std::fprintf(stderr, " %9.4f", v[i]);
            std::fprintf(stderr, "\n");
        };
        const limbo::step::Trace& tr = ref->trace();
        pr("input_embed", tr.x0);
        pr("attn_norm", tr.xn0);
        pr("qkv (node_15)", tr.qkv0);
        pr("conv_output_silu", tr.conv0);
        pr("q_conv_predelta", tr.qhat0);
        pr("k_conv_predelta", tr.khat0);
        pr("attn_output (gdn out)", tr.gdn0);
        pr("z (attn_gate)", tr.z0);
        pr("gated (node_59)", tr.gated0);
        pr("linear_attn_out", tr.linattn0);
        if (!tr.attn_out.empty()) pr("attn_residual-0", tr.attn_out[0]);
        pr("ffn_moe_probs-0", tr.moe_logits0);
        for (size_t l = 0; l < tr.l_out.size(); ++l) {
            const std::vector<float>& v = tr.l_out[l];
            if (v.size() < 128) continue;
            std::fprintf(stderr, "l_out-%-14zu head %9.4f %9.4f %9.4f   tail %9.4f %9.4f %9.4f\n", l, v[0], v[1],
                         v[2], v[v.size() - 3], v[v.size() - 2], v[v.size() - 1]);
        }
        pr("result_output", tr.logits);
    }
    std::printf("{\"name\":\"%s\",\"n_prompt\":%zu,\"prompt_ids\":[", name.c_str(), ids.size());
    for (size_t i = 0; i < ids.size(); ++i) std::printf("%s%d", i ? "," : "", ids[i]);
    std::printf("],\"n_predict\":%zu,\"greedy\":[", greedy.size());
    for (size_t i = 0; i < greedy.size(); ++i) std::printf("%s%d", i ? "," : "", greedy[i]);
    std::printf("],\"top2\":[");
    for (size_t i = 0; i < top2.size(); ++i)
        std::printf("%s[%d,%.6f,%d,%.6f,%.6f]", i ? "," : "", int(top2[i][0]), top2[i][1], int(top2[i][2]), top2[i][3],
                    top2[i][4]);
    std::printf("]}\n");
    return 0;
}
