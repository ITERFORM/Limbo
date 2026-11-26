// Copyright (c) 2026 ITERFORM. All rights reserved.
// tools/routing_stats.cpp — how local is a MoE model's expert routing? Reads the `ffn_moe_topk` sections that
// oracle_nodes writes (`--node none` keeps only those: per layer, per token, the experts the router chose) and
// answers what a VRAM expert cache needs to hold:
//
//   routing_stats -m model.gguf train.bin [test.bin ...]
//
// Every capture is one sequence in token order. The first is the training sequence (the profile is ranked on it);
// all are test sequences. Printed: how concentrated each layer's routing is; the hit rate of a static profile
// (ranked on the training sequence, and on the test sequence itself as the upper bound) for a range of slot
// counts; per-layer LRU by stack distance (exact); a global LRU and a decayed-frequency cache with swap
// counts on the training sequence; the distinct experts per layer a window of W tokens touches; and how far
// apart in tokens an expert's uses are. Slot sizes are the real per-layer expert sizes of the model.
#include "limbo/model.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <vector>

namespace {

struct Seq {
    std::string name;
    int n = 0;                                  // tokens
    std::vector<std::vector<int32_t>> topk;     // [layer] rows * n_used, rows == n
};

bool read_capture(const char* path, int n_layers, int n_used, Seq& s) {
    std::FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    int32_t hdr[3];
    if (std::fread(hdr, 4, 3, f) != 3) return false;
    s.n = hdr[0];
    s.name = path;
    if (std::fseek(f, long(4 * s.n), SEEK_CUR) != 0) return false;
    for (int i = 0; i < hdr[1]; ++i) {   // node sections (none with --node none): skip
        int32_t layer, rows;
        if (std::fread(&layer, 4, 1, f) != 1 || std::fread(&rows, 4, 1, f) != 1) return false;
        if (std::fseek(f, long(4) * rows * hdr[2], SEEK_CUR) != 0) return false;
    }
    int32_t nt = 0;
    if (std::fread(&nt, 4, 1, f) != 1) return false;
    s.topk.assign(size_t(n_layers), {});
    for (int i = 0; i < nt; ++i) {
        int32_t layer, rows;
        if (std::fread(&layer, 4, 1, f) != 1 || std::fread(&rows, 4, 1, f) != 1) return false;
        std::vector<int32_t> v(size_t(rows) * size_t(n_used));
        if (std::fread(v.data(), 4, v.size(), f) != v.size()) return false;
        if (layer >= 0 && layer < n_layers) s.topk[size_t(layer)] = std::move(v);
    }
    std::fclose(f);
    return true;   // a layer with fewer rows (the last layer: llama.cpp crops it to the output rows) is dropped by the caller
}

}  // namespace

int main(int argc, char** argv) {
    const char* model_path = nullptr;
    std::vector<const char*> caps;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-m") && i + 1 < argc) model_path = argv[++i];
        else caps.push_back(argv[i]);
    }
    if (!model_path || caps.empty()) {
        std::fprintf(stderr, "usage: routing_stats -m model.gguf train.bin [test.bin ...]\n");
        return 2;
    }
    std::string err;
    auto model = limbo::model::Model::open({model_path}, &err);
    if (!model) {
        std::fprintf(stderr, "routing_stats: %s\n", err.c_str());
        return 1;
    }
    const auto& g = model->geometry();
    int L = g.n_layer_main > 0 ? g.n_layer_main : g.n_layer;
    const int E = g.n_expert, K = g.n_expert_used;
    // bytes of one expert (gate + up + down rows) at each layer, from the file's own tensors
    std::vector<double> ebytes(static_cast<size_t>(L));
    double total_bytes = 0;
    for (int l = 0; l < L; ++l) {
        double b = 0;
        for (const char* r : {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"})
            if (const auto* t = model->tensor(l, r)) b += double(t->nbytes) / double(E);
        ebytes[size_t(l)] = b;
        total_bytes += b * E;
    }
    const double MiB = 1048576.0, GiB = 1073741824.0;
    std::printf("model: %d layers, %d experts, %d used, expert pool %.2f GiB (%.3f MiB per expert on average)\n", L, E, K, total_bytes / GiB,
                total_bytes / double(L * E) / MiB);

    std::vector<Seq> seqs(caps.size());
    for (size_t i = 0; i < caps.size(); ++i)
        if (!read_capture(caps[i], L, K, seqs[i])) {
            std::fprintf(stderr, "routing_stats: cannot read %s\n", caps[i]);
            return 1;
        }
    // keep the layers every sequence has all rows of
    {
        std::vector<int> keep;
        for (int l = 0; l < L; ++l) {
            bool ok = true;
            for (auto& s : seqs) ok = ok && s.topk[size_t(l)].size() == size_t(s.n) * size_t(K);
            if (ok) keep.push_back(l);
        }
        if (int(keep.size()) != L) {
            std::printf("  layers without a row per token are dropped (%d of %d kept)\n", int(keep.size()), L);
            std::vector<double> eb;
            for (int l : keep) eb.push_back(ebytes[size_t(l)]);
            for (auto& s : seqs) {
                std::vector<std::vector<int32_t>> t;
                for (int l : keep) t.push_back(std::move(s.topk[size_t(l)]));
                s.topk = std::move(t);
            }
            ebytes = eb;
            total_bytes = 0;
            for (double b : ebytes) total_bytes += b * E;
            L = int(keep.size());
        }
    }
    for (auto& s : seqs) std::printf("  sequence %s: %d tokens, %lld lookups\n", s.name.c_str(), s.n, (long long)s.n * L * K);

    auto counts_of = [&](const Seq& s) {
        std::vector<double> c(size_t(L) * E, 0.0);
        for (int l = 0; l < L; ++l)
            for (int t = 0; t < s.n; ++t)
                for (int k = 0; k < K; ++k) c[size_t(l) * E + size_t(s.topk[size_t(l)][size_t(t) * K + k])] += 1.0;
        return c;
    };

    // ---- concentration: the share of a layer's lookups its m most-used experts take, averaged over layers
    {
        const Seq& s = seqs[0];
        const auto c = counts_of(s);
        std::printf("\nconcentration on %s: share of a layer's lookups taken by its m most-used experts (mean over layers; min..max)\n", s.name.c_str());
        for (int m : {8, 16, 32, 64, 128}) {
            double sum = 0, lo = 1, hi = 0;
            for (int l = 0; l < L; ++l) {
                std::vector<double> v(c.begin() + long(l) * E, c.begin() + long(l + 1) * E);
                std::sort(v.begin(), v.end(), std::greater<double>());
                const double tot = std::accumulate(v.begin(), v.end(), 0.0);
                const double top = std::accumulate(v.begin(), v.begin() + m, 0.0);
                sum += top / tot;
                lo = std::min(lo, top / tot);
                hi = std::max(hi, top / tot);
            }
            std::printf("  top %3d of %d: %5.1f %%  (%.1f..%.1f)\n", m, E, 100.0 * sum / L, 100.0 * lo, 100.0 * hi);
        }
    }

    // ---- static profile: the N most-used (layer, expert) pairs of the training sequence stay resident
    const std::vector<int> slot_grid = {1000, 2000, 3000, 3608, 4500, 6000, 7500, 9000, 10500, 12288};
    {
        std::printf("\nstatic profile: hit rate of the N pairs ranked on the training sequence; (self) = ranked on the test sequence itself\n");
        std::printf("%8s %8s", "slots", "GiB");
        for (auto& s : seqs) std::printf("  %-14.14s", s.name.substr(s.name.find_last_of('/') + 1).c_str());
        std::printf("\n");
        const auto train = counts_of(seqs[0]);
        auto rank_of = [&](const std::vector<double>& c) {
            std::vector<int> o(c.size());
            std::iota(o.begin(), o.end(), 0);
            std::stable_sort(o.begin(), o.end(), [&](int a, int b) { return c[size_t(a)] > c[size_t(b)]; });
            return o;
        };
        const auto order = rank_of(train);
        for (int N : slot_grid) {
            double bytes = 0;
            for (int i = 0; i < N && i < L * E; ++i) bytes += ebytes[size_t(order[size_t(i)] / E)];
            std::printf("%8d %8.2f", N, bytes / GiB);
            for (auto& s : seqs) {
                const auto c = counts_of(s);
                const auto self = rank_of(c);
                std::vector<char> in(c.size(), 0), in_self(c.size(), 0);
                for (int i = 0; i < N && i < L * E; ++i) { in[size_t(order[size_t(i)])] = 1; in_self[size_t(self[size_t(i)])] = 1; }
                double hit = 0, hit_self = 0, tot = 0;
                for (size_t p = 0; p < c.size(); ++p) {
                    tot += c[p];
                    if (in[p]) hit += c[p];
                    if (in_self[p]) hit_self += c[p];
                }
                std::printf("  %5.1f%% (%5.1f%%)", 100 * hit / tot, 100 * hit_self / tot);
            }
            std::printf("\n");
        }
    }

    // ---- per-layer LRU, exact, by stack distance: a lookup hits a cache of c experts when fewer than c distinct
    // experts of that layer were used since this expert's last use (cold lookups miss)
    {
        std::printf("\nper-layer LRU by stack distance (c experts in every layer; slots = c x %d); hit rate per sequence\n", L);
        std::printf("%6s %8s %8s", "c", "slots", "GiB");
        for (auto& s : seqs) std::printf("  %-8.8s", s.name.substr(s.name.find_last_of('/') + 1).c_str());
        std::printf("\n");
        const std::vector<int> cs = {8, 16, 24, 32, 48, 64, 96, 128, 192, 256};
        std::vector<std::vector<std::vector<long>>> hist(seqs.size());   // [seq][layer][distance] counts, distance E = cold
        for (size_t si = 0; si < seqs.size(); ++si) {
            hist[si].assign(size_t(L), std::vector<long>(size_t(E) + 1, 0));
            for (int l = 0; l < L; ++l) {
                std::vector<int> stack;   // most recent first
                for (int t = 0; t < seqs[si].n; ++t)
                    for (int k = 0; k < K; ++k) {
                        const int e = seqs[si].topk[size_t(l)][size_t(t) * K + k];
                        auto it = std::find(stack.begin(), stack.end(), e);
                        if (it == stack.end()) { hist[si][size_t(l)][size_t(E)]++; stack.insert(stack.begin(), e); }
                        else {
                            hist[si][size_t(l)][size_t(it - stack.begin())]++;
                            stack.erase(it);
                            stack.insert(stack.begin(), e);
                        }
                    }
            }
        }
        for (int c : cs) {
            double bytes = 0;
            for (int l = 0; l < L; ++l) bytes += ebytes[size_t(l)] * c;
            std::printf("%6d %8d %8.2f", c, c * L, bytes / GiB);
            for (size_t si = 0; si < seqs.size(); ++si) {
                double hit = 0, tot = 0;
                for (int l = 0; l < L; ++l)
                    for (int d = 0; d <= E; ++d) {
                        tot += double(hist[si][size_t(l)][size_t(d)]);
                        if (d < c) hit += double(hist[si][size_t(l)][size_t(d)]);
                    }
                std::printf("  %6.1f%%", 100 * hit / tot);
            }
            std::printf("\n");
        }
        // reuse distance in tokens (consecutive uses of the same expert in the same layer), pooled over layers
        std::printf("\nreuse distance in tokens (share of re-uses within d tokens of the previous use of the same expert), per sequence\n");
        std::printf("%6s", "d");
        for (auto& s : seqs) std::printf("  %-8.8s", s.name.substr(s.name.find_last_of('/') + 1).c_str());
        std::printf("\n");
        const std::vector<int> ds = {1, 2, 4, 8, 16, 32, 64, 128, 512};
        for (int d : ds) {
            std::printf("%6d", d);
            for (size_t si = 0; si < seqs.size(); ++si) {
                // the stamp moves after the token's own lookups, so two picks of one token are not a reuse
                long within = 0, reuse = 0;
                for (int l = 0; l < L; ++l) {
                    std::vector<int> last(size_t(E), -1);
                    for (int t = 0; t < seqs[si].n; ++t) {
                        for (int k = 0; k < K; ++k) {
                            const int e = seqs[si].topk[size_t(l)][size_t(t) * K + k];
                            if (last[size_t(e)] >= 0) { ++reuse; if (t - last[size_t(e)] <= d) ++within; }
                        }
                        for (int k = 0; k < K; ++k) last[size_t(seqs[si].topk[size_t(l)][size_t(t) * K + k])] = t;
                    }
                }
                std::printf("  %6.1f%%", reuse ? 100.0 * double(within) / double(reuse) : 0.0);
            }
            std::printf("\n");
        }
    }

    // ---- windows: distinct experts per layer touched by W consecutive tokens (what a verify window must have)
    {
        std::printf("\ndistinct experts per layer touched by a window of W consecutive tokens (mean over layers and windows)\n");
        for (size_t si = 0; si < seqs.size(); ++si) {
            std::printf("  %-28.28s", seqs[si].name.substr(seqs[si].name.find_last_of('/') + 1).c_str());
            for (int W : {1, 2, 3, 4, 6, 8, 16}) {
                double sum = 0;
                long n = 0;
                for (int l = 0; l < L; ++l)
                    for (int t0 = 0; t0 + W <= seqs[si].n; t0 += std::max(1, W)) {
                        std::set<int> u;
                        for (int t = t0; t < t0 + W; ++t)
                            for (int k = 0; k < K; ++k) u.insert(seqs[si].topk[size_t(l)][size_t(t) * K + k]);
                        sum += double(u.size());
                        ++n;
                    }
                std::printf("  W=%-2d %5.1f", W, sum / double(n));
            }
            std::printf("\n");
        }
    }

    // ---- adaptive caches on the training sequence (the longest one): a global LRU and a decayed-frequency cache,
    // both with the real slot sizes; swaps = admissions, the bytes a PCIe link has to carry
    {
        const Seq& s = seqs[0];
        std::printf("\nadaptive caches on %s (cold start, so the first tokens fill them: hit rate over all tokens | over the second half)\n",
                    s.name.substr(s.name.find_last_of('/') + 1).c_str());
        std::printf("%-26s %8s %8s %18s %14s\n", "policy", "slots", "GiB", "hit all | 2nd half", "MiB/token in");
        for (int N : {3608, 4800, 6000, 9000, 12288}) {
            // global LRU over (layer, expert) pairs
            {
                std::vector<long> stamp(size_t(L) * E, -1);
                std::set<std::pair<long, int>> by_age;
                long hit = 0, tot = 0, hit2 = 0, tot2 = 0;
                double in_bytes = 0;
                long clk = 0;
                for (int t = 0; t < s.n; ++t)
                    for (int l = 0; l < L; ++l)
                        for (int k = 0; k < K; ++k) {
                            const int id = l * E + s.topk[size_t(l)][size_t(t) * K + k];
                            ++clk;
                            const bool h = stamp[size_t(id)] >= 0;
                            if (h) by_age.erase({stamp[size_t(id)], id});
                            else {
                                in_bytes += ebytes[size_t(l)];
                                if (int(by_age.size()) >= N) { stamp[size_t(by_age.begin()->second)] = -1; by_age.erase(by_age.begin()); }
                            }
                            stamp[size_t(id)] = clk;
                            by_age.insert({clk, id});
                            ++tot; hit += h;
                            if (t >= s.n / 2) { ++tot2; hit2 += h; }
                        }
                std::printf("%-26s %8d %8.2f %8.1f%% | %5.1f%% %14.1f\n", "global LRU", N, double(N) * total_bytes / double(L * E) / GiB, 100.0 * double(hit) / double(tot),
                            100.0 * double(hit2) / double(tot2), in_bytes / MiB / s.n);
            }
            // decayed frequency: score = sum of gamma^age over uses; a missed expert enters when its score beats the weakest resident's
            for (double gamma : {0.99, 0.998}) {
                std::vector<double> score(size_t(L) * E, 0.0);
                std::vector<char> res(size_t(L) * E, 0);
                std::set<std::pair<double, int>> weakest;   // residents by score (stale scores are refreshed on use)
                long hit = 0, tot = 0, hit2 = 0, tot2 = 0, nres = 0;
                double in_bytes = 0;
                // start from the training prefix profile: the first 64 tokens rank the initial residents
                std::vector<double> init(size_t(L) * E, 0.0);
                for (int t = 0; t < std::min(64, s.n); ++t)
                    for (int l = 0; l < L; ++l)
                        for (int k = 0; k < K; ++k) init[size_t(l * E + s.topk[size_t(l)][size_t(t) * K + k])] += 1.0;
                std::vector<int> ord(init.size());
                std::iota(ord.begin(), ord.end(), 0);
                std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return init[size_t(a)] > init[size_t(b)]; });
                for (int i = 0; i < N && init[size_t(ord[size_t(i)])] > 0; ++i) { res[size_t(ord[size_t(i)])] = 1; ++nres; score[size_t(ord[size_t(i)])] = init[size_t(ord[size_t(i)])]; }
                for (int id = 0; id < L * E; ++id) if (res[size_t(id)]) weakest.insert({score[size_t(id)], id});
                double scale = 1.0;   // scores live in units of gamma^-t so decay is one multiply per token
                for (int t = 0; t < s.n; ++t) {
                    scale /= gamma;
                    if (scale > 1e100) {   // renormalise
                        for (auto& v : score) v /= scale;
                        weakest.clear();
                        for (int id = 0; id < L * E; ++id) if (res[size_t(id)]) weakest.insert({score[size_t(id)], id});
                        scale = 1.0;
                    }
                    for (int l = 0; l < L; ++l)
                        for (int k = 0; k < K; ++k) {
                            const int id = l * E + s.topk[size_t(l)][size_t(t) * K + k];
                            const bool h = res[size_t(id)] != 0;
                            if (h) weakest.erase({score[size_t(id)], id});
                            score[size_t(id)] += scale;
                            if (h) weakest.insert({score[size_t(id)], id});
                            else if (nres < N) { res[size_t(id)] = 1; ++nres; weakest.insert({score[size_t(id)], id}); in_bytes += ebytes[size_t(l)]; }
                            else if (!weakest.empty() && score[size_t(id)] > weakest.begin()->first) {
                                const int victim = weakest.begin()->second;
                                weakest.erase(weakest.begin());
                                res[size_t(victim)] = 0;
                                res[size_t(id)] = 1;
                                weakest.insert({score[size_t(id)], id});
                                in_bytes += ebytes[size_t(l)];
                            }
                            ++tot; hit += h;
                            if (t >= s.n / 2) { ++tot2; hit2 += h; }
                        }
                }
                char name[64];
                std::snprintf(name, sizeof(name), "decayed LFU, gamma %.3f", gamma);
                std::printf("%-26s %8d %8.2f %8.1f%% | %5.1f%% %14.1f\n", name, N, double(N) * total_bytes / double(L * E) / GiB, 100.0 * double(hit) / double(tot),
                            100.0 * double(hit2) / double(tot2), in_bytes / MiB / s.n);
            }
        }
        std::printf("(slots x average expert size: 3608 = %.2f GiB, 6000 = %.2f GiB, 9000 = %.2f GiB, 12288 = %.2f GiB; 4800 = 9.1 GiB)\n",
                    3608 * total_bytes / double(L * E) / GiB, 6000 * total_bytes / double(L * E) / GiB, 9000 * total_bytes / double(L * E) / GiB, total_bytes / GiB);
    }
    return 0;
}
