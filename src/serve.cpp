// Copyright (c) 2026 ITERFORM. All rights reserved.
// src/serve.cpp — the OpenAI-compatible service (see include/limbo/serve.hpp).
#include "limbo/serve.hpp"
#include "limbo/hf.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <climits>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <strings.h>
#include <ctime>
#include <deque>
#include <numeric>
#include <random>

#include "limbo/gguf.hpp"
#include "limbo/vlog.hpp"

namespace limbo::serve {

using jinja::Value;

namespace {

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

uint64_t splitmix(uint64_t& x) {
    x += 0x9E3779B97F4A7C15ull;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
uint64_t rotl(uint64_t v, int k) { return (v << k) | (v >> (64 - k)); }

// Holds back text that could still become a stop sequence; releases the rest.
class StopMatcher {
public:
    explicit StopMatcher(std::vector<std::string> stops) : stops_(std::move(stops)) {}

    std::string push(std::string_view t, bool* hit) {
        pend_.append(t);
        size_t first = std::string::npos;
        for (const std::string& s : stops_) {
            const size_t k = pend_.find(s);
            if (k != std::string::npos && k < first) first = k;
        }
        if (first != std::string::npos) {
            std::string out = pend_.substr(0, first);
            pend_.clear();
            *hit = true;
            return out;
        }
        size_t hold = 0;
        for (const std::string& s : stops_) {
            for (size_t k = std::min(s.size() - 1, pend_.size()); k > hold; --k) {
                if (pend_.compare(pend_.size() - k, k, s, 0, k) == 0) {
                    hold = k;
                    break;
                }
            }
        }
        std::string out = pend_.substr(0, pend_.size() - hold);
        pend_.erase(0, pend_.size() - hold);
        return out;
    }
    std::string flush() {
        std::string out;
        out.swap(pend_);
        return out;
    }

private:
    std::vector<std::string> stops_;
    std::string pend_;
};

std::string sse(const Value& v) { return "data: " + jinja::json_compact(v) + "\n\n"; }

bool as_number(const Value* v, double* out) {
    if (!v) return false;
    if (v->is_int()) *out = double(v->i());
    else if (v->is_float()) *out = v->d();
    else return false;
    return true;
}

}  // namespace

// ---- Backend defaults -----------------------------------------------------------------------------

const float* Backend::forward(int32_t token, int pos, bool want_logits) {
    const float* l = forward(token, pos);
    return want_logits ? l : nullptr;
}

int32_t Backend::forward_greedy(int32_t token, int pos) { return Sampler::argmax(forward(token, pos), n_vocab()); }

void Backend::prefill(const int32_t* tokens, int n, int pos0) {
    for (int i = 0; i < n; ++i) forward(tokens[i], pos0 + i, false);
}

int Backend::decode_spec(int32_t fed, int pos, int, const SpecPick& pick, int32_t* out) {
    if (pick.greedy) {
        out[0] = forward_greedy(fed, pos);
    } else {
        bool kept = false;
        out[0] = pick.row(forward(fed, pos, true), -1, &kept);
    }
    return 1;
}

// ---- Sampler --------------------------------------------------------------------------------------

int32_t Sampler::argmax(const float* lg, int n) {
    int best = 0;
    float lmax = -INFINITY;
    for (int i = 0; i < n; ++i) {
        if (lg[i] > lmax) {
            lmax = lg[i];
            best = i;
        }
    }
    return best;
}

Sampler::Sampler(const SamplerParams& p, uint64_t fallback_seed) : p_(p) {
    uint64_t x = p.has_seed ? p.seed : fallback_seed;
    for (uint64_t& s : s_) s = splitmix(x);
}

uint64_t Sampler::next() {
    const uint64_t result = rotl(s_[1] * 5, 7) * 9;
    const uint64_t t = s_[1] << 17;
    s_[2] ^= s_[0];
    s_[3] ^= s_[1];
    s_[1] ^= s_[2];
    s_[0] ^= s_[3];
    s_[2] ^= t;
    s_[3] = rotl(s_[3], 45);
    return result;
}

double Sampler::uniform() { return double(next() >> 11) * (1.0 / 9007199254740992.0); }

int32_t Sampler::sample(const float* lg, int n) {
    const int best = argmax(lg, n);
    const float lmax = lg[best];
    if (p_.temperature <= 0 || n <= 1 || !std::isfinite(lmax)) return best;
    const double inv_t = 1.0 / p_.temperature;
    auto weight = [&](int i) { return std::exp((double(lg[i]) - double(lmax)) * inv_t); };

    const bool truncate = p_.top_k > 0 || p_.top_p < 1.0 || p_.min_p > 0.0;
    if (!truncate) {  // an exact draw from the full softmax
        double z = 0;
        for (int i = 0; i < n; ++i) z += weight(i);
        const double u = uniform() * z;
        double acc = 0;
        for (int i = 0; i < n; ++i) {
            acc += weight(i);
            if (acc > u) return i;
        }
        return best;
    }

    double acc = 0;
    const size_t kept = kept_set(lg, n, lmax, &acc);
    const double u = uniform() * acc;
    double c = 0;
    for (size_t r = 0; r < kept; ++r) {
        c += weight(idx_[r]);
        if (c > u) return idx_[r];
    }
    return idx_[kept - 1];
}

size_t Sampler::kept_set(const float* lg, int n, float lmax, double* total) {
    const double inv_t = 1.0 / p_.temperature;
    auto weight = [&](int i) { return std::exp((double(lg[i]) - double(lmax)) * inv_t); };
    idx_.resize(size_t(n));
    std::iota(idx_.begin(), idx_.end(), 0);
    auto better = [&](int a, int b) { return lg[a] > lg[b] || (lg[a] == lg[b] && a < b); };
    const size_t universe = p_.top_k > 0 ? std::min<size_t>(size_t(p_.top_k), size_t(n)) : size_t(n);
    size_t sorted = p_.top_k > 0 ? universe : std::min<size_t>(256, size_t(n));
    std::partial_sort(idx_.begin(), idx_.begin() + long(sorted), idx_.end(), better);
    double z = 0;
    if (p_.top_k > 0) {
        for (size_t r = 0; r < universe; ++r) z += weight(idx_[r]);
    } else {
        for (int i = 0; i < n; ++i) z += weight(i);
    }
    size_t kept = 0;
    double acc = 0;
    for (size_t r = 0; r < universe; ++r) {
        if (r == sorted) {  // the cut was not reached inside the sorted prefix: sort the rest too
            std::sort(idx_.begin() + long(sorted), idx_.end(), better);
            sorted = universe;
        }
        const double w = weight(idx_[r]);
        if (r > 0 && p_.min_p > 0 && w < p_.min_p) break;  // w is relative to the top token's weight (1)
        ++kept;
        acc += w;
        if (p_.top_p < 1.0 && acc / z >= p_.top_p) break;
    }
    *total = acc;
    return kept;
}

int32_t Sampler::sample_draft(const float* lg, int n, int32_t draft, bool* kept) {
    *kept = false;
    const int best = argmax(lg, n);
    const float lmax = lg[best];
    if (p_.temperature <= 0 || n <= 1 || !std::isfinite(lmax)) {
        *kept = best == draft;
        return best;
    }
    if (draft < 0 || draft >= n) return sample(lg, n);
    const double inv_t = 1.0 / p_.temperature;
    auto weight = [&](int i) { return std::exp((double(lg[i]) - double(lmax)) * inv_t); };
    const bool truncate = p_.top_k > 0 || p_.top_p < 1.0 || p_.min_p > 0.0;
    double z = 0, wd = 0;
    size_t nk = 0;
    if (!truncate) {
        for (int i = 0; i < n; ++i) z += weight(i);
        wd = weight(draft);
    } else {
        nk = kept_set(lg, n, lmax, &z);
        for (size_t r = 0; r < nk; ++r)
            if (idx_[r] == draft) wd = weight(draft);
    }
    if (uniform() * z < wd) {
        *kept = true;
        return draft;
    }
    // the same distribution without the draft
    const double u = uniform() * (z - wd);
    double c = 0;
    if (!truncate) {
        for (int i = 0; i < n; ++i) {
            if (i == draft) continue;
            c += weight(i);
            if (c > u) return i;
        }
        return best != draft ? best : (draft == 0 ? 1 : 0);
    }
    int32_t last = -1;
    for (size_t r = 0; r < nk; ++r) {
        if (idx_[r] == draft) continue;
        last = idx_[r];
        c += weight(idx_[r]);
        if (c > u) return idx_[r];
    }
    return last >= 0 ? last : best;
}

// ---- Service --------------------------------------------------------------------------------------

Service::Service(Backend& backend, const tok::Tokenizer& tok, const chat::Renderer& tpl, Config cfg)
    : be_(backend), tok_(tok), tpl_(tpl), cfg_(std::move(cfg)) {
    vl_ = cfg_.log ? std::max(0, cfg_.verbose) : 0;
    be_.set_verbose(vl_);
    const tok::Vocab& v = tok_.vocab();
    auto add = [&](int32_t id) {
        if (id >= 0 && std::find(eog_.begin(), eog_.end(), id) == eog_.end()) eog_.push_back(id);
    };
    add(v.eos);
    add(v.id("<|im_end|>"));
    add(v.id("<|endoftext|>"));

    // The tape: checkpoints need a backend that can snapshot, and a model that marks its messages.
    boundary_tok_ = v.id("<|im_start|>");

    // Slots first (a second conversation kept whole is the point), then checkpoint slots; both leave the
    // reserve free for the desktop. The numbers go in the note so the decision can be read back.
    const double mib = 1048576.0;
    char buf[480];
    be_.set_vram_reserve(uint64_t(std::max(0, cfg_.vram_reserve_mib)) << 20);
    const uint64_t slot_b = be_.slot_bytes();
    // A slot without checkpoint slots loses its tape at every new turn (the template re-renders the last
    // assistant turn), so a second slot only counts as fitting when every slot keeps kMinCk checkpoint slots beside it.
    const uint64_t snap_b = (cfg_.checkpoints > 0 && boundary_tok_ >= 0) ? be_.snapshot_bytes() : 0;
    const int kMinCk = snap_b > 0 ? std::min(cfg_.checkpoints, 2) : 0;
    int want_slots = cfg_.slots;
    if (want_slots > 1 && slot_b > 0) {
        const uint64_t free_now = be_.vram_free();
        const uint64_t reserve = uint64_t(std::max(0, cfg_.vram_reserve_mib)) << 20;
        const uint64_t budget = free_now > reserve ? free_now - reserve : 0;
        uint64_t need = uint64_t(kMinCk) * snap_b;  // slot 0's checkpoint slots
        int fit = 1;
        while (fit < want_slots && need + slot_b + uint64_t(kMinCk) * snap_b <= budget) {
            need += slot_b + uint64_t(kMinCk) * snap_b;
            ++fit;
        }
        want_slots = fit;
    }
    int n_slots = 1;
    if (cfg_.slots > 1) {
        const uint64_t free0 = be_.vram_free();
        n_slots = want_slots > 1 ? std::max(1, be_.set_slots(want_slots)) : 1;
        if (slot_b > 0) {
            const uint64_t free1 = be_.vram_free();
            std::snprintf(buf, sizeof buf, "slots: %d of %d asked (%.0f MiB free before, %.1f MiB per slot, %d MiB reserve, %.0f MiB free after)", n_slots,
                          cfg_.slots, double(free0) / mib, double(slot_b) / mib, cfg_.vram_reserve_mib, double(free1) / mib);
            note_ += buf;
            if (n_slots < cfg_.slots) {
                std::snprintf(buf, sizeof buf, "; running with %d: slot %d needs %.1f MiB plus %d checkpoint slots per slot (%.1f MiB each) and the %d MiB reserve, %.0f MiB were free",
                              n_slots, n_slots + 1, double(slot_b) / mib, kMinCk, double(snap_b) / mib, cfg_.vram_reserve_mib, double(free0) / mib);
                note_ += buf;
            }
            note_ += "\n";
        }
    } else if (slot_b > 0) {
        std::snprintf(buf, sizeof buf, "slots: 1 (--slots 1; a second would take %.1f MiB)\n", double(slot_b) / mib);
        note_ += buf;
    }
    slots_.resize(size_t(n_slots));
    if (const Backend::Pool pl = be_.pool(); pl.positions > 0) {
        std::snprintf(buf, sizeof buf, "pool: %d positions in %d pages of %d shared by %d slot%s\n", pl.positions, pl.total_pages, pl.page, n_slots,
                      n_slots == 1 ? "" : "s");
        note_ += buf;
    }
    if (cfg_.checkpoints > 0 && boundary_tok_ >= 0) {
        n_snaps_ = be_.set_snapshot_slots(cfg_.checkpoints);
        for (int k = n_snaps_ - 1; k >= 0; --k) free_snaps_.push_back(k);
        if (be_.snapshot_bytes() > 0) {
            std::snprintf(buf, sizeof buf, "checkpoints: %d of %d asked, %.1f MiB each (%.0f MiB free after)\n", n_snaps_, cfg_.checkpoints,
                          double(be_.snapshot_bytes()) / mib, double(be_.vram_free()) / mib);
            note_ += buf;
        }
    }
    if (vl_ >= 1) startup_block();
    else if (cfg_.log) {
        for (size_t at = 0; at < note_.size();) {
            const size_t e = note_.find('\n', at);
            std::fprintf(stderr, "serve: %.*s\n", int(e - at), note_.c_str() + at);
            at = e + 1;
        }
        if (!cfg_.api_keys.empty()) std::fprintf(stderr, "serve: API key required on every endpoint but /health (%zu configured)\n", cfg_.api_keys.size());
    }
}

// -v: what the server and the backend decided at start-up, then where the device memory went.
void Service::startup_block() {
    using vlog::bytes;
    using vlog::fmt;
    using vlog::thou;
    for (size_t at = 0; at < note_.size();) {  // "tag: text" lines of the constructor
        const size_t e = note_.find('\n', at);
        const std::string ln = note_.substr(at, e - at);
        const size_t c = ln.find(": ");
        if (c != std::string::npos) vlog::line(ln.substr(0, c) == "checkpoints" ? "ckpt" : ln.substr(0, c).c_str(), ln.substr(c + 2));
        else vlog::line("boot", ln);
        at = e + 1;
    }
    for (const std::string& ln : be_.startup_report()) {
        const size_t c = ln.find(": ");
        if (c != std::string::npos && c < 10) vlog::line(ln.substr(0, c).c_str(), ln.substr(c + 2));
        else vlog::line("boot", ln);
    }
    vlog::line("http", "POST /v1/chat/completions; GET /health /v1/models /props /slots /metrics (llama-server's llamacpp: counters); POST /slots/N?action=erase");
    vlog::line("http", cfg_.api_keys.empty() ? "no API key: every endpoint is open (--api-key KEY to require Authorization: Bearer)"
                                             : vlog::fmt("API key required (%zu configured) on every endpoint but /health", cfg_.api_keys.size()),
               0, cfg_.api_keys.empty() ? vlog::Tone::Warn : vlog::Tone::Good);
    const std::vector<Backend::MemUse> mem = be_.memory_report();
    if (mem.empty()) return;
    uint64_t sum = 0;
    size_t w = 8;
    for (const Backend::MemUse& m : mem) w = std::max(w, m.what.size());
    for (const Backend::MemUse& m : mem) {
        sum += m.bytes;
        vlog::line("vram", fmt("%-*s %12s  %s", int(w), m.what.c_str(), bytes(m.bytes).c_str(), m.note.c_str()));
    }
    const uint64_t reserve = uint64_t(std::max(0, cfg_.vram_reserve_mib)) << 20, free_b = be_.vram_free();
    vlog::line("vram", fmt("%-*s %12s  %s", int(w), "reserve", bytes(reserve).c_str(), "kept free for the desktop (--vram-reserve-mib)"));
    vlog::line("vram", fmt("%-*s %12s  %s", int(w), "free", bytes(free_b).c_str(), free_b >= reserve ? "above the reserve" : "BELOW the reserve"),
               0, free_b >= reserve ? vlog::Tone::Good : vlog::Tone::Warn);
    vlog::line("vram", fmt("%-*s %12s  %s", int(w), "total", bytes(sum + reserve + free_b).c_str(), "the sum: the card's free memory when the engine started loading"));
}

// ---- the verbose log ----------------------------------------------------------------------------------

void Service::vline(const char* tag, const std::string& text, vlog::Tone tone) const {
    if (vl_ >= 1) vlog::line(tag, text, rid_, tone);
}

const char* Service::kind_name(Kind k) { return k == Kind::Head ? "head" : k == Kind::Root ? "root" : "plain"; }

// "pool 182/412 pages, 3 shared" (empty without a pool)
std::string Service::pool_note() const {
    const Backend::Pool pl = be_.pool();
    if (pl.positions <= 0) return {};
    return vlog::fmt("pool %d/%d pages, %d shared", pl.used_pages, pl.total_pages, pl.shared_pages);
}

// The live slot's positions >= keep go back to the pool; says how many pages that freed.
void Service::cut_(int keep) {
    const int before = vl_ >= 1 ? be_.slot_pages(cur_) : 0;
    be_.cut(keep);
    if (vl_ >= 1)
        if (const int freed = before - be_.slot_pages(cur_); freed > 0)
            vline("pages", vlog::fmt("slot %d cut back to %s positions: %d page%s freed  (%s)", cur_, vlog::thou(keep).c_str(), freed, freed == 1 ? "" : "s", pool_note().c_str()));
}

// ---- routing, slots and checkpoints -------------------------------------------------------------------

Service::Reuse Service::plan_reuse(const Slot& h, const std::vector<int32_t>& toks) const {
    Reuse r;
    if (h.tape.empty() || toks.empty()) return r;
    const size_t last = toks.size() - 1;
    r.lcp = size_t(std::mismatch(h.tape.begin(), h.tape.end(), toks.begin(), toks.end()).first - h.tape.begin());
    if (r.lcp == h.tape.size() && h.tape.size() <= last) {
        r.keep = h.tape.size();
        r.cont = true;
        return r;
    }
    const size_t d = std::min(r.lcp, last);
    for (const Ckpt& c : h.ck)
        if (size_t(c.pos) <= d && c.pos > 0 && (!r.ck || c.pos > r.ck->pos)) r.ck = &c;
    if (r.ck) r.keep = size_t(r.ck->pos);
    return r;
}

// The slot a request runs in. A pin (id_slot, a number, or a label bound by its first request) wins. Otherwise the slot
// that can reuse the most of the prompt, provided that is more than the opening (the first message boundary:
// what every conversation of an agent shares). Reusing more than the opening means sharing a whole message
// beyond it (a checkpoint sits at every message boundary that matters), or the prompt extending the slot's
// whole tape; a coincidence of a few tokens never counts. Ties go to the longer common prefix, then the more
// recently used. Nobody qualifies: the least recently used slot (a slot never used counts as the oldest).
int Service::route(const http::Request& rq, const std::vector<int32_t>& toks, size_t opening, int id_slot, const char** how) {
    const int n = int(slots_.size());
    *how = "only";
    if (n == 1) return 0;
    if (id_slot >= 0) {
        if (id_slot < n) {
            *how = "id_slot";
            return id_slot;
        }
        if (cfg_.log) std::fprintf(stderr, "serve: id_slot %d, the backend has %d slots; routing by prefix\n", id_slot, n);
    }
    const std::string* hv = cfg_.session_header.empty() ? nullptr : rq.header(cfg_.session_header);
    const std::string label = hv ? hv->substr(0, 64) : std::string();
    bool numeric = false;
    if (!label.empty()) {
        char* end = nullptr;
        const long v = std::strtol(label.c_str(), &end, 10);
        numeric = end != label.c_str() && *end == 0;
        if (numeric) {
            if (v >= 0 && v < n) {
                *how = "pin";
                return int(v);
            }
            if (cfg_.log) std::fprintf(stderr, "serve: %s names slot %ld, the backend has %d; routing by prefix\n", cfg_.session_header.c_str(), v, n);
        } else {
            for (const auto& pin : pins_)
                if (pin.first == label && pin.second < n) {
                    *how = "pin";
                    return pin.second;
                }
        }
    }
    int best = -1;
    Reuse best_ru;
    for (int i = 0; i < n; ++i) {
        const Reuse ru = plan_reuse(slots_[size_t(i)], toks);
        if (ru.keep <= opening) continue;
        const bool better = best < 0 || ru.keep > best_ru.keep || (ru.keep == best_ru.keep && ru.lcp > best_ru.lcp) ||
                            (ru.keep == best_ru.keep && ru.lcp == best_ru.lcp && slots_[size_t(i)].used > slots_[size_t(best)].used);
        if (better) {
            best = i;
            best_ru = ru;
        }
    }
    *how = "prefix";
    if (best < 0) {
        *how = "lru";
        best = 0;
        for (int i = 1; i < n; ++i)
            if (slots_[size_t(i)].used < slots_[size_t(best)].used) best = i;
    }
    if (!label.empty() && !numeric) {
        if (pins_.size() >= 64) pins_.erase(pins_.begin());
        pins_.emplace_back(label, best);
    }
    return best;
}

void Service::release_ckpts(Slot& s, int above_pos) {
    for (size_t i = 0; i < s.ck.size();) {
        if (s.ck[i].pos > above_pos) {
            free_snaps_.push_back(s.ck[i].snap);
            s.ck.erase(s.ck.begin() + long(i));
        } else ++i;
    }
}

// A free slot, or one taken from the checkpoint that is worth least: the oldest plain one, then the oldest user-turn
// head but the newest (a new head may replace even that one: it is the newer). The root is never taken. Slots of the
// other slots are only taken when this one has nothing to give.
int Service::acquire_snap(Slot& s, Kind kind) {
    if (!free_snaps_.empty()) {
        const int k = free_snaps_.back();
        free_snaps_.pop_back();
        return k;
    }
    auto victim = [&](Slot& x, bool may_take_newest_head) -> int {  // index into x.ck, or -1
        int old_plain = -1, old_head = -1, new_head = -1;
        for (size_t i = 0; i < x.ck.size(); ++i) {
            const Ckpt& c = x.ck[i];
            if (c.kind == Kind::Plain) {
                if (old_plain < 0 || c.stamp < x.ck[size_t(old_plain)].stamp) old_plain = int(i);
            } else if (c.kind == Kind::Head) {
                if (old_head < 0 || c.stamp < x.ck[size_t(old_head)].stamp) old_head = int(i);
                if (new_head < 0 || c.stamp > x.ck[size_t(new_head)].stamp) new_head = int(i);
            }
        }
        if (old_plain >= 0) return old_plain;
        if (old_head >= 0 && old_head != new_head) return old_head;
        return old_head >= 0 && may_take_newest_head ? old_head : -1;
    };
    auto take = [&](Slot& x, int i) {
        const int k = x.ck[size_t(i)].snap;
        vline("slot", vlog::fmt("evicted checkpoint @%s (%s) of slot %d for a newer one", vlog::thou(x.ck[size_t(i)].pos).c_str(), kind_name(x.ck[size_t(i)].kind),
                                int(&x - slots_.data())));
        x.ck.erase(x.ck.begin() + i);
        return k;
    };
    if (const int i = victim(s, kind != Kind::Plain); i >= 0) return take(s, i);
    for (Slot& o : slots_)
        if (&o != &s)
            if (const int i = victim(o, false); i >= 0) return take(o, i);
    return -1;
}

void Service::take_ckpt(Slot& s, size_t pos, Kind kind) {
    for (const Ckpt& c : s.ck)
        if (size_t(c.pos) == pos) return;
    const int slot = acquire_snap(s, kind);
    if (slot < 0) {
        vline("slot", vlog::fmt("checkpoint @%s (%s) skipped: no snapshot slot free and none worth taking", vlog::thou(long(pos)).c_str(), kind_name(kind)), vlog::Tone::Warn);
        return;
    }
    if (!be_.snapshot(slot)) {
        free_snaps_.push_back(slot);
        return;
    }
    vline("slot", vlog::fmt("checkpoint @%s (%s) in snapshot slot %d", vlog::thou(long(pos)).c_str(), kind_name(kind), slot));
    Ckpt c;
    c.pos = int(pos);
    c.snap = slot;
    c.kind = kind;
    c.stamp = ++stamp_;
    s.ck.push_back(c);
    ++stats_.checkpoints;
}

void Service::select_slot_(int s) {
    if (s != cur_ && be_.select_slot(s)) cur_ = s;
}

// The slot that can lend the most of `toks` (see the top of serve.hpp): its newest checkpoint at or before the common prefix, or its
// tape's end when the whole tape is a prefix of the prompt. Only a slot that lends more than `own_keep` counts.
Service::Lend Service::find_lender(int slot, const std::vector<int32_t>& toks, size_t own_keep) const {
    Lend best;
    if (toks.empty()) return best;
    const size_t last = toks.size() - 1;
    size_t best_pos = own_keep;
    for (int t = 0; t < int(slots_.size()); ++t) {
        if (t == slot || slots_[size_t(t)].tape.empty()) continue;
        const Slot& o = slots_[size_t(t)];
        const size_t lcp = size_t(std::mismatch(o.tape.begin(), o.tape.end(), toks.begin(), toks.end()).first - o.tape.begin());
        const size_t d = std::min(lcp, last);
        Lend cand;
        cand.from = t;
        for (const Ckpt& c : o.ck)
            if (size_t(c.pos) <= d && c.pos > 0 && (!cand.ck || c.pos > cand.ck->pos)) cand.ck = &c;
        cand.pos = cand.ck ? size_t(cand.ck->pos) : 0;
        if (lcp == o.tape.size() && o.tape.size() <= last && o.tape.size() > cand.pos) {  // its live state at the end of its tape
            cand.pos = o.tape.size();
            cand.ck = nullptr;
        }
        if (cand.pos > best_pos) {
            best = cand;
            best_pos = cand.pos;
        }
    }
    return best;
}

// `slot` (the backend's live one) takes `l.pos` positions of the lender's K/V by reference and the matching recurrent snapshot; it
// then reads only what comes after. On failure the slot is empty (reset) or untouched, and its tape says so.
bool Service::adopt(int slot, const Lend& l) {
    Slot& me = slots_[size_t(slot)];
    Slot& from = slots_[size_t(l.from)];
    int snap = l.ck ? l.ck->snap : -1;
    if (!l.ck) {  // the lender's live state at its tape's end becomes a checkpoint of its own
        if (free_snaps_.empty()) release_ckpts(me, -1);  // this slot's are about to go anyway: the snapshot needs a slot to land in
        select_slot_(l.from);
        take_ckpt(from, l.pos, Kind::Plain);
        select_slot_(slot);
        for (const Ckpt& c : from.ck)
            if (size_t(c.pos) == l.pos) snap = c.snap;
        if (snap < 0) return false;
    }
    be_.cut(0);
    me.tape.clear();
    release_ckpts(me, -1);
    const Backend::Pool p0 = vl_ >= 1 ? be_.pool() : Backend::Pool{};
    if (!be_.share(l.from, int(l.pos)) || !be_.restore(snap)) {
        be_.reset();
        vline("slot", vlog::fmt("adopt slot %d -> %d of %s positions FAILED; reading from the start", l.from, slot, vlog::thou(long(l.pos)).c_str()), vlog::Tone::Warn);
        return false;
    }
    me.tape.assign(from.tape.begin(), from.tape.begin() + long(l.pos));
    if (vl_ >= 1) {
        const Backend::Pool p1 = be_.pool();
        const int copied = int(p1.copies - p0.copies), pages = be_.slot_pages(slot);
        vline("slot", vlog::fmt("adopt  slot %d -> %d  %s positions from %s: %d page%s by reference, %d copied%s  %s", l.from, slot, vlog::thou(long(l.pos)).c_str(),
                                l.ck ? vlog::fmt("checkpoint @%s", vlog::thou(l.ck->pos).c_str()).c_str() : "the lender's live tail", pages - copied, pages - copied == 1 ? "" : "s", copied,
                                copied > 0 ? " (the cut falls inside a page: copy-on-write)" : "", pool_note().c_str()));
    }
    return true;
}

// GET /props: what Ariadne's endpoint probe reads (n_ctx, total_slots, the template) in llama-server's shape.
void Service::handle_props(http::Conn& conn) {
    int n_ctx = be_.n_ctx();
    if (const Backend::Pool pl = be_.pool(); pl.positions > 0) n_ctx = n_ctx > 0 ? std::min(n_ctx, pl.positions) : pl.positions;
    Value gen = Value::object();
    gen.set("id", Value::integer(0));
    gen.set("n_ctx", Value::integer(n_ctx));
    Value v = Value::object();
    v.set("default_generation_settings", gen);
    v.set("total_slots", Value::integer(int(slots_.size())));
    v.set("n_ctx", Value::integer(n_ctx));
    v.set("model_alias", Value::string(cfg_.model_id));
    v.set("model_path", Value::string(cfg_.model_id));
    v.set("chat_template", Value::string(tpl_.source()));
    v.set("build_info", Value::string("Limbo"));
    v.set("is_sleeping", Value::boolean(false));
    conn.respond(200, "application/json", jinja::json_compact(v));
}

// GET /slots: one entry per slot; n_past and n_prompt_tokens are what the slot's tape holds.
void Service::handle_slots(const http::Request&, http::Conn& conn) {
    Value arr = Value::array();
    for (int i = 0; i < int(slots_.size()); ++i) {
        Value o = Value::object();
        o.set("id", Value::integer(i));
        o.set("n_ctx", Value::integer(be_.n_ctx()));
        o.set("n_prompt_tokens", Value::integer(int(slots_[size_t(i)].tape.size())));
        o.set("n_past", Value::integer(int(slots_[size_t(i)].tape.size())));
        o.set("is_processing", Value::boolean(busy_ == i));
        o.set("pages", Value::integer(be_.slot_pages(i)));
        arr.push(o);
    }
    conn.respond(200, "application/json", jinja::json_compact(arr));
}

// POST /slots/{id}?action=erase: the slot's tape, checkpoints and pages go.
void Service::erase_slot(int id, http::Conn& conn) {
    Slot& s = slots_[size_t(id)];
    const size_t n = s.tape.size();
    const size_t nck = s.ck.size();
    select_slot_(id);
    const int pages = vl_ >= 1 ? be_.slot_pages(id) : 0;
    be_.reset();
    s.tape.clear();
    release_ckpts(s, -1);
    vline("slot", vlog::fmt("erase  slot %d: %s tokens, %zu checkpoint%s, %d page%s freed  (%s)", id, vlog::thou(long(n)).c_str(), nck, nck == 1 ? "" : "s", pages,
                            pages == 1 ? "" : "s", pool_note().c_str()));
    Value v = Value::object();
    v.set("id_slot", Value::integer(id));
    v.set("n_erased", Value::integer(int(n)));
    conn.respond(200, "application/json", jinja::json_compact(v));
}

bool Service::is_eog(int32_t t) const { return std::find(eog_.begin(), eog_.end(), t) != eog_.end(); }

std::string Service::new_id(const char* prefix) {
    uint64_t x = ++counter_ * 0x9E3779B97F4A7C15ull ^ uint64_t(std::chrono::steady_clock::now().time_since_epoch().count());
    char buf[64];
    std::snprintf(buf, sizeof buf, "%s%016llx%08x", prefix, (unsigned long long)splitmix(x), unsigned(splitmix(x) & 0xFFFFFFFFu));
    return buf;
}

static std::string error_json(const std::string& msg, const char* type) {
    Value e = Value::object();
    Value inner = Value::object();
    inner.set("message", Value::string(msg));
    inner.set("type", Value::string(type));
    inner.set("param", Value::null());
    inner.set("code", Value::null());
    e.set("error", inner);
    return jinja::json_compact(e);
}

void Service::reply_error(http::Conn& conn, int status, const std::string& msg, const char* type) {
    conn.respond(status, "application/json", error_json(msg, type));
    ++stats_.rejected;
    if (vl_ >= 1) vlog::line(status >= 500 ? "error" : "warn", vlog::fmt("%d %s", status, msg.c_str()), rid_, vlog::Tone::Warn);
    else if (cfg_.log) std::fprintf(stderr, "serve: %d %s\n", status, msg.c_str());
}

void Service::preempt(const http::Request& rq, http::Conn& conn, const Slot& s, int n_prompt, int cached, double t_start) {
    const size_t read = s.tape.size();
    cut_(int(read));  // the pages past the tape go back to the pool for the request that waits
    stats_.last_prompt_tokens = n_prompt;
    stats_.last_cached_tokens = cached;
    stats_.last_completion_tokens = 0;
    stats_.tape_tokens = int(read);
    ++stats_.preempted;
    const std::string msg = "preempted by a waiting request; the " + std::to_string(read) + " tokens read so far stay cached on slot " + std::to_string(cur_);
    // 503 is in Ariadne's retryable set (retry.cpp retryable_status), so prime() sends the prompt again after Retry-After.
    conn.respond(503, "application/json", error_json(msg, "server_error"), {{"Retry-After", "1"}});
    if (vl_ >= 1)
        vline("slot", vlog::fmt("preempt  slot %d read %s of %s tokens for a waiting request, kept; 503 Retry-After 1 s  (after %s)", cur_, vlog::thou(long(read)).c_str(),
                                vlog::thou(n_prompt).c_str(), vlog::ms((now_s() - t_start) * 1000).c_str()),
              vlog::Tone::Warn);
    else if (cfg_.log)
        std::fprintf(stderr, "serve: %s preempted=1 read=%zu prompt=%d cached=%d slot=%d after %.2fs\n", rq.path.c_str(), read, n_prompt, cached, cur_, now_s() - t_start);
}

bool Service::parse_request(const Value& body, ChatRequest* out, std::string* err) const {
    auto bad = [&](const std::string& m) {
        *err = m;
        return false;
    };
    if (!body.is_dict()) return bad("the request body must be a JSON object");
    const Value* msgs = body.find("messages");
    if (!msgs || !msgs->is_list()) return bad("'messages' is required and must be an array");
    for (const Value& m : msgs->list()) {
        if (!m.is_dict() || !m.find("role") || !m.find("role")->is_string()) return bad("every message needs a string 'role'");
    }
    out->messages = *msgs;

    out->tools = Value::undefined();
    if (const Value* t = body.find("tools"); t && !t->is_null()) {
        if (!t->is_list()) return bad("'tools' must be an array");
        for (const Value& x : t->list())
            if (!x.is_dict()) return bad("every tool must be an object");
        out->tools = *t;
    }
    bool choice_none = false;
    if (const Value* tc = body.find("tool_choice"); tc && !tc->is_null()) {
        if (tc->is_string()) {
            if (tc->s() == "none") choice_none = true;
            else if (tc->s() != "auto" && tc->s() != "required") return bad("'tool_choice' must be auto, none, required or an object");
        } else if (!tc->is_dict()) return bad("'tool_choice' must be a string or an object");
    }
    out->tools_on = out->tools.is_list() && !out->tools.list().empty() && !choice_none;

    out->kwargs = Value::object();
    if (const Value* k = body.find("chat_template_kwargs"); k && !k->is_null()) {
        if (!k->is_dict()) return bad("'chat_template_kwargs' must be an object");
        out->kwargs = *k;
    }
    if (const Value* et = body.find("enable_thinking"); et && et->is_bool() && !out->kwargs.find("enable_thinking"))
        out->kwargs.set("enable_thinking", *et);

    if (const Value* s = body.find("stream"); s && !s->is_null()) {
        if (!s->is_bool()) return bad("'stream' must be a boolean");
        out->stream = s->b();
    }
    out->id_slot = -1;
    if (const Value* is = body.find("id_slot"); is && !is->is_null()) {
        double d = 0;
        if (!as_number(is, &d) || d != std::floor(d) || d < -1 || d > 1e6) return bad("'id_slot' must be an integer >= -1");
        out->id_slot = int(d);
    }
    out->cache_prompt = true;
    if (const Value* cp = body.find("cache_prompt"); cp && !cp->is_null()) {
        if (!cp->is_bool()) return bad("'cache_prompt' must be a boolean");
        out->cache_prompt = cp->b();
    }
    if (const Value* so = body.find("stream_options"); so && so->is_dict()) {
        const Value* iu = so->find("include_usage");
        out->include_usage = iu && iu->is_bool() && iu->b();
    }
    if (const Value* n = body.find("n"); n && !n->is_null()) {
        if (!n->is_int() || n->i() != 1) return bad("only n = 1 is supported");
    }
    out->max_tokens = 0;
    for (const char* key : {"max_tokens", "max_completion_tokens"}) {
        const Value* v = body.find(key);
        if (!v || v->is_null()) continue;
        double d;
        if (!as_number(v, &d) || d < 1 || d != std::floor(d) || d > 1e9) return bad(std::string("'") + key + "' must be a positive integer");
        out->max_tokens = int(d);
        break;
    }
    out->sampling = cfg_.sampling;
    double d;
    if (const Value* v = body.find("temperature"); v && !v->is_null()) {
        if (!as_number(v, &d) || d < 0 || d > 10) return bad("'temperature' must be a number in [0, 10]");
        out->sampling.temperature = d;
    }
    if (const Value* v = body.find("top_p"); v && !v->is_null()) {
        if (!as_number(v, &d) || d < 0 || d > 1) return bad("'top_p' must be a number in [0, 1]");
        out->sampling.top_p = d;
    }
    if (const Value* v = body.find("min_p"); v && !v->is_null()) {
        if (!as_number(v, &d) || d < 0 || d > 1) return bad("'min_p' must be a number in [0, 1]");
        out->sampling.min_p = d;
    }
    if (const Value* v = body.find("top_k"); v && !v->is_null()) {
        if (!as_number(v, &d) || d < 0 || d != std::floor(d) || d > 1e9) return bad("'top_k' must be a non-negative integer");
        out->sampling.top_k = int(d);
    }
    if (const Value* v = body.find("seed"); v && !v->is_null()) {
        if (!v->is_int()) return bad("'seed' must be an integer");
        out->sampling.has_seed = true;
        out->sampling.seed = uint64_t(v->i());
    }
    out->stop.clear();
    if (const Value* v = body.find("stop"); v && !v->is_null()) {
        if (v->is_string()) {
            if (!v->s().empty()) out->stop.push_back(v->s());
        } else if (v->is_list()) {
            if (v->list().size() > 16) return bad("'stop' takes at most 16 sequences");
            for (const Value& s : v->list()) {
                if (!s.is_string()) return bad("'stop' must be a string or an array of strings");
                if (!s.s().empty()) out->stop.push_back(s.s());
            }
        } else return bad("'stop' must be a string or an array of strings");
    }
    out->model = body.get_str("model");
    return true;
}

// What identifies an assistant message on its way back: its text (trimmed) and its calls (name and arguments, compact JSON),
// as the client sends them and as this server answered them.
static std::string thinking_key(const Value& content, const Value& calls) {
    std::string k;
    if (content.is_string()) {
        std::string_view t = content.s();
        while (!t.empty() && std::isspace(static_cast<unsigned char>(t.front()))) t.remove_prefix(1);
        while (!t.empty() && std::isspace(static_cast<unsigned char>(t.back()))) t.remove_suffix(1);
        k.assign(t);
    }
    if (calls.is_list())
        for (const Value& c : calls.list()) {
            const Value* f = c.is_dict() ? c.find("function") : nullptr;
            if (!f || !f->is_dict()) continue;
            k += "\x1f" + f->get_str("name") + "\x1e";
            const Value* a = f->find("arguments");
            Value parsed;
            std::string perr;
            if (a && a->is_string() && jinja::json_parse(a->s(), &parsed, &perr)) k += jinja::json_compact(parsed);
            else if (a && a->is_dict()) k += jinja::json_compact(*a);
            else if (a && a->is_string()) k += a->s();
        }
    return k;
}

// One message as it identifies the history after it: role, name / call id, and the thinking_key of its text and calls.
static std::string message_sig(const Value& m) {
    if (!m.is_dict()) return {};
    const Value* c = m.find("content");
    const Value* tc = m.find("tool_calls");
    return m.get_str("role") + "\x1d" + m.get_str("tool_call_id") + "\x1d" + m.get_str("name") + "\x1d" +
           thinking_key(c ? *c : Value::undefined(), tc ? *tc : Value::undefined());
}

// FNV-1a over the signatures, a separator between messages.
static uint64_t sig_fold(uint64_t h, const std::string& sig) {
    for (unsigned char ch : sig) h = (h ^ ch) * 1099511628211ull;
    return (h ^ 0x1cu) * 1099511628211ull;
}
constexpr uint64_t kSigSeed = 1469598103934665603ull;

void Service::remember_thinking(const Value& history, const std::string& key, const std::string& reasoning) {
    if (!cfg_.preserve_thinking || reasoning.empty()) return;
    uint64_t h = kSigSeed;
    if (history.is_list())
        for (const Value& m : history.list()) h = sig_fold(h, message_sig(m));
    if (recall_.size() >= kRecall) recall_.erase(recall_.begin());
    recall_.push_back({h, key, reasoning});
}

Value Service::with_thinking(const Value& messages) const {
    if (!cfg_.preserve_thinking || recall_.empty() || !messages.is_list()) return messages;
    Value out = Value::array();
    uint64_t before = kSigSeed;   // the history up to this message
    for (const Value& m : messages.list()) {
        const uint64_t here = before;
        before = sig_fold(before, message_sig(m));
        const Value* r = m.is_dict() ? m.find("reasoning_content") : nullptr;
        const Value* c = m.is_dict() ? m.find("content") : nullptr;
        // reasoning_content absent (or null): the client does not carry thinking, so it is refilled; present and "" : the client
        // cut it on purpose (Ariadne ages old reasoning out) and it stays cut -- rendered as if there never was any
        const bool bare = m.is_dict() && m.get_str("role") == "assistant" && (!r || r->is_null()) &&
                          !(c && c->is_string() && c->s().find("</think>") != std::string::npos);
        if (!bare) {
            out.push(m);
            continue;
        }
        const Value* tc = m.find("tool_calls");
        const std::string key = thinking_key(c ? *c : Value::undefined(), tc ? *tc : Value::undefined());
        // the answer at this place in this history; else the one answer with this text, when only one has it (a client that
        // reworded its earlier history); else nothing -- never another place's thinking
        const Recall* hit = nullptr;
        const Recall* only = nullptr;
        int same = 0;
        for (auto it = recall_.rbegin(); it != recall_.rend() && !hit; ++it) {
            if (it->key != key) continue;
            if (it->before == here) hit = &*it;
            ++same;
            only = &*it;
        }
        if (!hit && same == 1 && !key.empty()) hit = only;
        if (!hit) {
            out.push(m);
            continue;
        }
        Value nm = Value::object();   // a new message: the request's own stays as it came
        for (const auto& kv : m.dict()) nm.set(kv.first, kv.second);
        nm.set("reasoning_content", Value::string(hit->reasoning));
        out.push(nm);
    }
    return out;
}

bool Service::build_prompt(const ChatRequest& rq, Prompt* out, std::string* err) const {
    const tok::Vocab& v = tok_.vocab();
    Value extra = Value::object();
    extra.set("bos_token", Value::string(v.add_bos && v.bos >= 0 ? tok_.decode(&v.bos, 1) : std::string()));
    extra.set("eos_token", Value::string(v.eos >= 0 ? tok_.decode(&v.eos, 1) : std::string()));
    const Value messages = chat::normalize_messages(with_thinking(rq.messages));
    Value kwargs = rq.kwargs;
    if (cfg_.preserve_thinking && kwargs.is_dict() && !kwargs.find("preserve_thinking")) kwargs.set("preserve_thinking", Value::boolean(true));
    if (!tpl_.apply(messages, rq.tools_on ? rq.tools : Value::undefined(), true, kwargs, extra, &out->text, err)) return false;
    out->tokens.clear();
    tok::Options o;
    o.add_bos = v.add_bos;
    o.parse_special = true;
    tok_.encode(out->text, out->tokens, o);
    out->opens_think = chat::prompt_opens_think(out->text);
    return true;
}

// llama-server's --api-key: `Authorization: Bearer <key>`; the comparison does not stop at the first differing byte.
bool Service::authorized(const http::Request& rq) const {
    if (cfg_.api_keys.empty()) return true;
    const std::string* h = rq.header("authorization");
    if (!h || h->size() < 8 || strncasecmp(h->c_str(), "Bearer ", 7) != 0) return false;
    const std::string tok = h->substr(7);
    bool ok = false;
    for (const std::string& k : cfg_.api_keys) {
        unsigned diff = unsigned(k.size() ^ tok.size());
        for (size_t i = 0; i < k.size(); ++i) diff |= unsigned(uint8_t(k[i]) ^ uint8_t(i < tok.size() ? tok[i] : 0));
        ok |= diff == 0;
    }
    return ok;
}

void Service::handle(const http::Request& rq, http::Conn& conn) {
    const std::string& p = rq.path;
    auto ok_json = [&](const Value& v) { conn.respond(200, "application/json", jinja::json_compact(v)); };
    const bool open_path = p == "/health" || p == "/v1/health";
    if (!open_path && !authorized(rq)) {  // before any routing: an unknown path is no more informative than a known one
        conn.respond(401, "application/json", error_json("Invalid API Key", "authentication_error"), {{"WWW-Authenticate", "Bearer"}});
        ++stats_.rejected;
        if (vl_ >= 1) vlog::line("warn", vlog::fmt("401 %s %s: missing or wrong API key", rq.method.c_str(), p.substr(0, 80).c_str()), 0, vlog::Tone::Warn);
        else if (cfg_.log) std::fprintf(stderr, "serve: 401 %s %s: missing or wrong API key\n", rq.method.c_str(), p.substr(0, 80).c_str());
        return;
    }
    if (p == "/health" || p == "/v1/health") {
        if (rq.method != "GET" && rq.method != "HEAD") return reply_error(conn, 405, "use GET");
        Value v = Value::object();
        v.set("status", Value::string("ok"));
        return ok_json(v);
    }
    if (p == "/metrics") {  // Prometheus text, llama-server's `llamacpp:` names (ariadne limbo watch reads them)
        if (rq.method != "GET") return reply_error(conn, 405, "use GET");
        const double pt = double(stats_.prompt_tokens), ps = double(stats_.prompt_us) / 1e6;
        const double gt = double(stats_.predicted_tokens), gs = double(stats_.predicted_us) / 1e6;
        std::string out;
        auto put = [&](const char* name, const char* type, const char* help, double v) {
            char line[96];
            std::snprintf(line, sizeof line, "%.10g", v);
            out += std::string("# HELP ") + name + " " + help + "\n# TYPE " + name + " " + type + "\n" + name + " " + line + "\n";
        };
        put("llamacpp:prompt_tokens_total", "counter", "Number of prompt tokens processed.", pt);
        put("llamacpp:prompt_seconds_total", "counter", "Prompt process time", ps);
        put("llamacpp:tokens_predicted_total", "counter", "Number of generation tokens processed.", gt);
        put("llamacpp:tokens_predicted_seconds_total", "counter", "Predict process time", gs);
        put("llamacpp:n_decode_total", "counter", "Total number of llama_decode() calls", gt);
        put("llamacpp:prompt_tokens_seconds", "gauge", "Average prompt throughput in tokens/s.", ps > 0 ? pt / ps : 0.0);
        put("llamacpp:predicted_tokens_seconds", "gauge", "Average generation throughput in tokens/s.", gs > 0 ? gt / gs : 0.0);
        put("llamacpp:requests_processing", "gauge", "Number of requests processing.", busy_ >= 0 ? 1.0 : 0.0);
        put("llamacpp:requests_deferred", "gauge", "Number of requests deferred.", 0.0);
        put("limbo:requests_total", "counter", "Requests answered.", double(stats_.completed.load()));
        put("limbo:checkpoints_total", "counter", "Recurrent-state checkpoints taken.", double(stats_.checkpoints.load()));
        put("limbo:adoptions_total", "counter", "Requests that took another slot's prefix.", double(stats_.adoptions.load()));
        put("limbo:preempted_total", "counter", "Primes that stepped aside for a waiting request.", double(stats_.preempted.load()));
        conn.respond(200, "text/plain; version=0.0.4", out);
        return;
    }
    if (p == "/v1/models" || p == "/models") {
        if (rq.method != "GET") return reply_error(conn, 405, "use GET");
        Value m = Value::object();
        m.set("id", Value::string(cfg_.model_id));
        m.set("object", Value::string("model"));
        m.set("created", Value::integer(int64_t(std::time(nullptr))));
        m.set("owned_by", Value::string("limbo"));
        if (be_.n_ctx() > 0) m.set("context_length", Value::integer(be_.n_ctx()));
        Value data = Value::array();
        data.push(m);
        Value v = Value::object();
        v.set("object", Value::string("list"));
        v.set("data", data);
        return ok_json(v);
    }
    if (p == "/props" || p == "/v1/props") {
        if (rq.method != "GET") return reply_error(conn, 405, "use GET");
        return handle_props(conn);
    }
    if (p == "/slots") {
        if (rq.method != "GET") return reply_error(conn, 405, "use GET");
        return handle_slots(rq, conn);
    }
    if (p.rfind("/slots/", 0) == 0) {
        if (rq.method != "POST") return reply_error(conn, 405, "use POST");
        char* end = nullptr;
        const long id = std::strtol(p.c_str() + 7, &end, 10);
        if (end == p.c_str() + 7 || *end != 0 || id < 0 || id >= long(slots_.size())) return reply_error(conn, 404, "no such slot: " + p.substr(7), "not_found_error");
        if (rq.query.find("action=erase") == std::string::npos) return reply_error(conn, 400, "only ?action=erase is supported");
        return erase_slot(int(id), conn);
    }
    if (p == "/v1/chat/completions" || p == "/chat/completions") {
        if (rq.method != "POST") return reply_error(conn, 405, "use POST");
        return chat_completions(rq, conn);
    }
    reply_error(conn, 404, "no such endpoint: " + p, "not_found_error");
}

void Service::chat_completions(const http::Request& rq, http::Conn& conn) {
    ++stats_.requests;
    const double t_start = now_s();
    rid_ = long(stats_.requests.load());
    vlog::set_request(rid_);
    struct RidGuard {
        long& r;
        ~RidGuard() {
            r = 0;
            vlog::set_request(0);
        }
    } rid_guard{rid_};
    Value body;
    std::string err;
    if (!jinja::json_parse(rq.body, &body, &err)) return reply_error(conn, 400, "invalid JSON: " + err);
    ChatRequest cr;
    if (!parse_request(body, &cr, &err)) return reply_error(conn, 400, err);
    Prompt pr;
    if (!build_prompt(cr, &pr, &err)) return reply_error(conn, 400, err);
    const int n_prompt = int(pr.tokens.size());
    const int n_ctx = be_.n_ctx();
    if (n_prompt == 0) return reply_error(conn, 400, "the prompt is empty");
    if (n_ctx > 0 && n_prompt >= n_ctx)
        return reply_error(conn, 400, "the prompt has " + std::to_string(n_prompt) + " tokens, the context holds " + std::to_string(n_ctx));
    if (cfg_.log_prompts) std::fprintf(stderr, "serve: prompt (%d tokens):\n%s\n---\n", n_prompt, pr.text.c_str());

    int max_new = cr.max_tokens > 0 ? cr.max_tokens : cfg_.default_max_tokens > 0 ? cfg_.default_max_tokens : n_ctx > 0 ? n_ctx : 4096;
    if (n_ctx > 0) max_new = std::min(max_new, n_ctx - n_prompt);

    // The slot this request runs in (routed by prefix, or pinned by id_slot or the header) and what it can reuse.
    size_t opening = 0;  // the first message boundary after position 0: the opening every conversation of an agent shares
    if (boundary_tok_ >= 0)
        for (size_t b = 1; b < pr.tokens.size() && !opening; ++b)
            if (pr.tokens[b] == boundary_tok_) opening = b;
    const char* how = "only";
    if (const int hi = route(rq, pr.tokens, opening, cr.id_slot, &how); hi != cur_) {
        if (!be_.select_slot(hi)) return reply_error(conn, 500, "the backend cannot select that slot", "server_error");
        cur_ = hi;
    }
    Slot& ses = slots_[size_t(cur_)];
    if (vl_ >= 1) {
        const size_t nmsg = cr.messages.is_list() ? cr.messages.list().size() : 0;
        const size_t ntools = cr.tools_on ? cr.tools.list().size() : 0;
        vline("req", vlog::fmt("%s %s  slot %d via %s  prompt %s tok  max %s  temp %.2f  %s  %zu msgs, %zu tools  tape %s", rq.method.c_str(), rq.path.c_str(), cur_, how,
                               vlog::thou(n_prompt).c_str(), vlog::thou(max_new).c_str(), cr.sampling.temperature, cr.stream ? "stream" : "no-stream", nmsg, ntools,
                               vlog::thou(long(ses.tape.size())).c_str()));
    }
    struct Busy {  // GET /slots answers while this request runs
        int& b;
        ~Busy() { b = -1; }
    } busy_guard{busy_};
    busy_ = cur_;
    std::vector<int32_t>& tape = ses.tape;
    ses.used = ++use_clock_;
    stats_.last_slot = cur_;

    // The tape. Continue when the slot has consumed a strict prefix of this prompt. When the prompt diverges
    // inside the tape (or is shorter than it), restore the newest checkpoint at or before the divergence and
    // read on from there; K/V past it is overwritten as it is read, never shifted. Another slot that has read more of the prompt
    // lends it (its pages by reference, its checkpoint's recurrent state copied) when that beats the slot's own tape. With no
    // reuse at all, reset and read everything (a slot taken over from another conversation keeps only the opening it shares).
    const size_t last = size_t(n_prompt) - 1;
    const bool pooled = be_.pool().positions > 0;
    size_t keep = 0;
    int restored_from = -1, adopted_from = -1;
    const size_t tape_before = tape.size();
    if (cr.cache_prompt) {
        const Lend ln = pooled && slots_.size() > 1 ? find_lender(cur_, pr.tokens, tape.empty() ? 0 : plan_reuse(ses, pr.tokens).keep) : Lend{};
        if (ln.from >= 0 && adopt(cur_, ln)) {
            keep = ln.pos;
            adopted_from = ln.from;
            ++stats_.adoptions;
        } else if (!tape.empty()) {
            const Reuse ru = plan_reuse(ses, pr.tokens);  // (after a failed adoption: its checkpoints may be gone)
            if (ru.cont) keep = ru.keep;
            else if (ru.ck && be_.restore(ru.ck->snap)) {
                keep = ru.keep;
                restored_from = int(ru.keep);
                ++stats_.restores;
                tape.resize(keep);
                cut_(int(keep));
                const size_t nck = ses.ck.size();
                release_ckpts(ses, restored_from);  // the checkpoints past it belong to the branch we left
                vline("slot", vlog::fmt("restore  ckpt @%s (%s): the prompt diverges from the tape at %s, tape %s -> %s; %zu later checkpoint%s released",
                                        vlog::thou(long(keep)).c_str(), kind_name(ru.ck->kind), vlog::thou(long(ru.lcp)).c_str(), vlog::thou(long(tape_before)).c_str(),
                                        vlog::thou(long(keep)).c_str(), nck - ses.ck.size(), nck - ses.ck.size() == 1 ? "" : "s"));
            }
        }
    }
    if (keep == 0) {
        const int pg = vl_ >= 1 ? be_.slot_pages(cur_) : 0;
        be_.reset();
        tape.clear();
        release_ckpts(ses, -1);
        if (pg > 0) vline("pages", vlog::fmt("slot %d reset: %d page%s freed  (%s)", cur_, pg, pg == 1 ? "" : "s", pool_note().c_str()));
    }
    stats_.last_restored_from = restored_from;
    stats_.last_adopted_from = adopted_from;
    const int cached = int(keep);
    if (vl_ >= 1) {
        const char* src = adopted_from >= 0 ? "adopted from another slot" : restored_from >= 0 ? "checkpoint restore"
                          : keep > 0 ? "the slot's own tape, continued" : !cr.cache_prompt ? "cache_prompt is false" : tape_before ? "diverged with no checkpoint: reset" : "empty slot";
        vline("cache", vlog::fmt("cached %s of %s (%s)  new %s  via %s%s", vlog::thou(cached).c_str(), vlog::thou(n_prompt).c_str(),
                                 vlog::pct(cached, n_prompt).c_str(), vlog::thou(n_prompt - cached).c_str(), src,
                                 adopted_from >= 0 ? vlog::fmt("  adopted-from %d", adopted_from).c_str() : ""));
    }
    // The pool: the pages for the whole prompt are taken now, so a prompt that does not fit is refused before anything is sent, and
    // the answer cannot outgrow what is left (the window of a speculation cycle writes up to 8 positions past the accepted ones).
    if (pooled) {
        auto refuse = [&](const std::string& why) {
            cut_(int(tape.size()));
            const Backend::Pool pl = be_.pool();
            const int others = std::max(0, pl.used_pages - be_.slot_pages(cur_)) * pl.page;
            return reply_error(conn, 400, why + ": the K/V pool holds " + std::to_string(pl.positions) + " positions in all and the other slots take " +
                                              std::to_string(others) + " of them (--ctx sizes the pool; there is no host tier yet)");
        };
        const int pg0 = vl_ >= 1 ? be_.slot_pages(cur_) : 0;
        if (!be_.reserve(n_prompt)) return refuse("the prompt needs " + std::to_string(n_prompt) + " positions");
        if (vl_ >= 1)
            vline("pages", vlog::fmt("slot %d takes %d page%s for the prompt (%d -> %d)  %s", cur_, be_.slot_pages(cur_) - pg0, be_.slot_pages(cur_) - pg0 == 1 ? "" : "s", pg0,
                                     be_.slot_pages(cur_), pool_note().c_str()));
        const int room = be_.room(n_prompt) - 8;
        if (room < 1) return refuse("the prompt needs " + std::to_string(n_prompt) + " positions and leaves none to answer in");
        max_new = std::min(max_new, room);
    }

    const std::string id = new_id("chatcmpl-");
    const int64_t created = int64_t(std::time(nullptr));
    const std::string model = cfg_.model_id;
    const bool stream = cr.stream;

    auto base_chunk = [&](const char* object) {
        Value c = Value::object();
        c.set("id", Value::string(id));
        c.set("object", Value::string(object));
        c.set("created", Value::integer(created));
        c.set("model", Value::string(model));
        return c;
    };
    auto delta_chunk = [&](Value delta, const Value& finish) {
        Value c = base_chunk("chat.completion.chunk");
        Value ch = Value::object();
        ch.set("index", Value::integer(0));
        ch.set("delta", std::move(delta));
        ch.set("finish_reason", finish);
        Value choices = Value::array();
        choices.push(ch);
        c.set("choices", choices);
        return c;
    };

    double last_write = now_s();
    bool aborted = false;
    auto send_chunk = [&](const Value& v) {
        if (!conn.write_chunk(sse(v))) aborted = true;
        last_write = now_s();
    };
    if (stream) {
        if (!conn.begin_stream()) aborted = true;
        Value d = Value::object();
        d.set("role", Value::string("assistant"));
        d.set("content", Value::string(""));
        if (!aborted) send_chunk(delta_chunk(d, Value::null()));
    }

    // ---- the message being built ----
    chat::OutputParser parser(pr.opens_think, cr.tools, cr.tools_on);
    std::vector<chat::Event> events;
    chat::Message msg;
    Value tool_calls_json = Value::array();
    int call_index = 0;
    auto call_json = [&](const chat::ToolCall& tc, bool with_index, int index) {
        Value f = Value::object();
        f.set("name", Value::string(tc.name));
        f.set("arguments", Value::string(jinja::json_compact(tc.arguments)));
        Value c = Value::object();
        if (with_index) c.set("index", Value::integer(index));
        c.set("id", Value::string(new_id("call_").substr(0, 5 + 24)));
        c.set("type", Value::string("function"));
        c.set("function", f);
        return c;
    };
    auto drain = [&]() {
        for (chat::Event& e : events) {
            if (e.kind == chat::Event::Call) {
                msg.calls.push_back(e.call);
                if (stream) {
                    Value d = Value::object();
                    Value arr = Value::array();
                    arr.push(call_json(e.call, true, call_index));
                    d.set("tool_calls", arr);
                    if (!aborted) send_chunk(delta_chunk(d, Value::null()));
                } else tool_calls_json.push(call_json(e.call, false, call_index));
                ++call_index;
                continue;
            }
            std::string& acc = e.kind == chat::Event::Reasoning ? msg.reasoning : msg.content;
            acc += e.text;
            if (e.kind == chat::Event::Reasoning) msg.has_reasoning = true;
            if (stream && !aborted) {
                Value d = Value::object();
                d.set(e.kind == chat::Event::Reasoning ? "reasoning_content" : "content", Value::string(e.text));
                send_chunk(delta_chunk(d, Value::null()));
            }
        }
        events.clear();
    };

    // ---- prefill ----
    double t_prefill = 0, t_decode = 0;
    int n_gen = 0;
    std::string finish;
    try {
        const double t0 = now_s();
        Sampler sampler(cr.sampling, uint64_t(std::chrono::steady_clock::now().time_since_epoch().count()) ^ counter_);
        const bool greedy = cr.sampling.temperature <= 0;
        const int nv = be_.n_vocab();
        // One backend step on `tok` at `pos`, and the token chosen from its logits. Temperature 0 asks the
        // backend for the greedy token alone (a GPU backend then never copies the logits to the host).
        auto choose = [&](int32_t tok, int pos) -> int32_t {
            if (greedy) return be_.forward_greedy(tok, pos);
            return sampler.sample(be_.forward(tok, pos, true), nv);
        };
        // Between backend calls: let other connections be served, notice a client that left, keep the stream alive.
        auto yield_ = [&]() {
            const int n = conn.yield();
            if (vl_ >= 2 && n > 0) vline("yield", vlog::fmt("answered %d light request%s (health, props, slots) while working", n, n == 1 ? "" : "s"));
        };
        auto client_ok = [&]() {
            yield_();
            if (conn.peer_gone()) {
                aborted = true;
                return false;
            }
            if (stream && cfg_.keepalive_ms > 0 && (now_s() - last_write) * 1000 > cfg_.keepalive_ms) {
                if (!conn.write_chunk(": keep-alive\n\n")) aborted = true;
                last_write = now_s();
            }
            return !aborted;
        };

        // Where to take checkpoints while reading: message boundaries from the restore point on. Every cut costs a
        // pass over the weights, so a plain boundary is skipped when it would leave a piece shorter than
        // checkpoint_gap (and never shorter than the backend's batch minimum: that piece would run token by token,
        // slower and with other rounding) on either side. Two are always taken: the generation prompt of a user
        // turn (the checkpoint the next user turn restores) and the end of the opening (the first boundary).
        // Reading is cut at every checkpoint position, so the pieces depend on the prompt alone and a restored
        // read splits like a cold one.
        std::vector<size_t> cuts;
        std::vector<Kind> cut_kind;
        if (n_snaps_ > 0 && boundary_tok_ >= 0) {
            const Value* lm = cr.messages.is_list() && !cr.messages.list().empty() ? &cr.messages.list().back() : nullptr;
            const bool user_turn = lm && lm->get_str("role") == "user";
            const size_t min_piece = size_t(std::max({1, be_.prefill_batch_min(), cfg_.checkpoint_gap}));
            std::vector<size_t> bounds;
            for (size_t b = 1; b <= last; ++b)
                if (pr.tokens[b] == boundary_tok_) bounds.push_back(b);
            size_t prev = keep;
            for (size_t k = 0; k < bounds.size(); ++k) {
                const size_t b = bounds[k];
                if (b < keep) continue;
                const Kind kind = user_turn && k + 1 == bounds.size() ? Kind::Head : k == 0 ? Kind::Root : Kind::Plain;
                if (b != keep) {
                    // The user turn's own checkpoint only needs the piece before it to be a batch; the others need both sides.
                    const size_t before = kind == Kind::Head ? size_t(std::max(1, be_.prefill_batch_min())) : min_piece;
                    const size_t after = kind == Kind::Head ? 0 : min_piece;
                    if (b - prev < before || last - b < after) continue;
                }
                cuts.push_back(b);
                cut_kind.push_back(kind);
                prev = b;
            }
        }

        // The prompt without its logits, in the backend's chunks; the last token runs the head and gives the first sample.
        const size_t chunk = size_t(std::max(1, be_.prefill_chunk()));
        be_.prompt_end(int(last) + 1);
        size_t next_cut = 0;
        // -v: the prompt is reported in pieces (up to each checkpoint, or 16,384 tokens when they are further apart). A piece is
        // closed with a sync, so its time is its own; the extra syncs only exist while the log is on.
        size_t n_pieces = 0, piece_no = 0, piece_from = keep;
        double piece_t0 = now_s();
        if (vl_ >= 1 && last > keep) {
            n_pieces = 1;
            for (size_t c : cuts)
                if (c > keep && c < last) ++n_pieces;
        }
        auto piece_done = [&](size_t upto, bool at_cut) {
            be_.sync();
            const double t = now_s();
            const size_t n = upto - piece_from;
            if (at_cut || upto == last) ++piece_no;
            vline("prefill", vlog::fmt("%s  %s tok  pos %s..%s  %s  %s", at_cut || upto == last ? vlog::fmt("piece %zu/%zu", piece_no, n_pieces).c_str() : vlog::fmt("piece %zu/%zu...", piece_no + 1, n_pieces).c_str(),
                                       vlog::thou(long(n)).c_str(), vlog::thou(long(piece_from)).c_str(), vlog::thou(long(upto)).c_str(), vlog::ms((t - piece_t0) * 1000).c_str(),
                                       vlog::rate(double(n), t - piece_t0).c_str()));
            piece_from = upto;
            piece_t0 = t;
        };
        // A prime (max_tokens 1, not streamed: the 503 needs unsent headers) steps aside for a waiting request after any piece.
        const bool yields = cr.max_tokens == 1 && !stream;
        for (size_t i = keep; i < last && client_ok();) {
            while (next_cut < cuts.size() && cuts[next_cut] < i) ++next_cut;
            if (vl_ >= 1 && i > piece_from && next_cut < cuts.size() && cuts[next_cut] == i) piece_done(i, true);
            if (next_cut < cuts.size() && cuts[next_cut] == i) {
                take_ckpt(ses, i, cut_kind[next_cut]);
                ++next_cut;
            }
            if (yields && i > keep) {  // the tape is exactly what was read: consecutive, last piece whole
                const bool w = conn.waiting();
                if (vl_ >= 2 && !w) vline("yield", vlog::fmt("after %s of %s prompt tokens: no request waiting, reading on", vlog::thou(long(i)).c_str(), vlog::thou(n_prompt).c_str()));
                if (w) {
                    preempt(rq, conn, ses, n_prompt, cached, t_start);
                    return;
                }
            }
            const size_t stop = next_cut < cuts.size() ? std::min(cuts[next_cut], last) : last;
            const size_t n = std::min(chunk, stop - i);
            be_.prefill(pr.tokens.data() + i, int(n), int(i));
            tape.insert(tape.end(), pr.tokens.begin() + long(i), pr.tokens.begin() + long(i + n));
            i += n;
            if (yields) be_.sync();  // the check above then sees the device at the tape's end, not a few pieces behind it
            if (vl_ >= 1 && i - piece_from >= 16384 && !(next_cut < cuts.size() && cuts[next_cut] == i)) piece_done(i, false);
        }
        int32_t t = -1;
        if (!aborted && client_ok()) {
            if (vl_ >= 1 && last > piece_from) piece_done(last, true);
            if (next_cut < cuts.size() && cuts[next_cut] == last) take_ckpt(ses, last, cut_kind[next_cut]);
            const double t_head = now_s();
            t = choose(pr.tokens[last], int(last));
            tape.push_back(pr.tokens[last]);
            if (vl_ >= 1) vline("prefill", vlog::fmt("last token + first sample  %s", vlog::ms((now_s() - t_head) * 1000).c_str()));
        }
        t_prefill = now_s() - t0;

        // ---- decode ----
        const double t1 = now_s();
        StopMatcher stops(cr.stop);
        std::string utf8_pending;
        bool stop_hit = false;
        // Speculation: a run of the backend can choose several tokens at once. `known` holds those after t; all but the last
        // were read already (they are on the tape), so while it is not empty t is not fed again.
        const bool spec = cfg_.speculate && be_.spec_ready();
        Backend::SpecPick pick;
        pick.greedy = greedy;
        pick.row = [&](const float* lg, int32_t draft, bool* kept) { return sampler.sample_draft(lg, nv, draft, kept); };
        pick.stop = [&](int32_t tok) { return is_eog(tok); };
        std::deque<int32_t> known;
        std::vector<int32_t> run(64);
        while (!aborted) {
            ++n_gen;
            if (is_eog(t)) {
                finish = "stop";
                break;
            }
            utf8_pending += tok_.decode(&t, 1);
            const size_t k = chat::utf8_complete_prefix(utf8_pending);
            std::string text = chat::utf8_sanitize(std::string_view(utf8_pending).substr(0, k));
            utf8_pending.erase(0, k);
            const std::string released = stops.push(text, &stop_hit);
            if (!released.empty()) parser.feed(released, &events);
            drain();
            if (stop_hit) {
                finish = "stop";
                break;
            }
            if (n_gen >= max_new) {
                finish = "length";
                break;
            }
            yield_();
            if (aborted || conn.peer_gone()) {
                aborted = true;
                break;
            }
            if (!known.empty()) {
                t = known.front();
                known.pop_front();
                continue;
            }
            const int32_t fed = t;
            if (spec) {
                const int room = std::max(1, std::min(max_new - n_gen, int(run.size())));
                const int n = be_.decode_spec(fed, int(tape.size()), room, pick, run.data());
                tape.push_back(fed);
                for (int j = 0; j + 1 < n; ++j) tape.push_back(run[size_t(j)]);
                for (int j = 1; j < n; ++j) known.push_back(run[size_t(j)]);
                t = run[0];
            } else {
                t = choose(fed, int(tape.size()));
                tape.push_back(fed);
            }
        }
        t_decode = now_s() - t1;
        if (!aborted) {
            if (!stop_hit) {
                if (!utf8_pending.empty()) {
                    bool dummy = false;
                    const std::string rest = stops.push(chat::utf8_sanitize(utf8_pending), &dummy);
                    if (!rest.empty()) parser.feed(rest, &events);
                    if (dummy) stop_hit = true;
                }
                if (!stop_hit) {
                    const std::string rest = stops.flush();
                    if (!rest.empty()) parser.feed(rest, &events);
                }
            }
            parser.finish(&events);
            drain();
        }
    } catch (const std::exception& e) {
        be_.reset();
        tape.clear();
        release_ckpts(ses, -1);
        stats_.tape_tokens = 0;
        ++stats_.aborted;
        if (vl_ >= 1) vline("error", vlog::fmt("backend failure: %s; slot %d reset", e.what(), cur_), vlog::Tone::Bad);
        else if (cfg_.log) std::fprintf(stderr, "serve: backend failure: %s\n", e.what());
        if (!conn.headers_sent()) conn.respond(500, "application/json", "{\"error\":{\"message\":\"backend failure\",\"type\":\"server_error\",\"param\":null,\"code\":null}}");
        return;
    }

    stats_.last_prompt_tokens = n_prompt;
    stats_.last_cached_tokens = cached;
    stats_.last_completion_tokens = n_gen;
    stats_.tape_tokens = int(tape.size());
    stats_.prompt_tokens += uint64_t(std::max(0, n_prompt - cached));
    stats_.prompt_us += uint64_t(std::max(0.0, t_prefill) * 1e6);
    stats_.predicted_tokens += uint64_t(std::max(0, n_gen));
    stats_.predicted_us += uint64_t(std::max(0.0, t_decode) * 1e6);
    if (aborted) {
        ++stats_.aborted;
        if (vl_ >= 1)
            vline("warn", vlog::fmt("client left: prompt %s cached %s gen %s after %s  (slot %d keeps %s tokens)", vlog::thou(n_prompt).c_str(), vlog::thou(cached).c_str(),
                                    vlog::thou(n_gen).c_str(), vlog::ms((now_s() - t_start) * 1000).c_str(), cur_, vlog::thou(long(tape.size())).c_str()),
                  vlog::Tone::Warn);
        else if (cfg_.log)
            std::fprintf(stderr, "serve: client left: prompt=%d cached=%d gen=%d after %.2fs\n", n_prompt, cached, n_gen, now_s() - t_start);
        return;
    }
    ++stats_.completed;
    if (finish == "stop" && !msg.calls.empty()) finish = "tool_calls";
    if (msg.has_reasoning)
        remember_thinking(cr.messages,
                          thinking_key(msg.content.empty() && !msg.calls.empty() ? Value::null() : Value::string(msg.content), tool_calls_json),
                          msg.reasoning);

    // ---- the answer ----
    const int n_new = n_prompt - cached;
    auto rate = [](int n, double s) { return s > 0 ? n / s : 0.0; };
    Value timings = Value::object();
    timings.set("slot", Value::integer(cur_));
    if (adopted_from >= 0) timings.set("adopted_from", Value::integer(adopted_from));
    timings.set("cache_n", Value::integer(cached));
    timings.set("prompt_n", Value::integer(n_new));
    timings.set("prompt_ms", Value::number(t_prefill * 1000));
    timings.set("prompt_per_token_ms", Value::number(n_new > 0 ? t_prefill * 1000 / n_new : 0.0));
    timings.set("prompt_per_second", Value::number(rate(n_new, t_prefill)));
    timings.set("predicted_n", Value::integer(n_gen));
    timings.set("predicted_ms", Value::number(t_decode * 1000));
    timings.set("predicted_per_token_ms", Value::number(n_gen > 0 ? t_decode * 1000 / n_gen : 0.0));
    timings.set("predicted_per_second", Value::number(rate(n_gen, t_decode)));
    Value usage = Value::object();
    usage.set("prompt_tokens", Value::integer(n_prompt));
    usage.set("completion_tokens", Value::integer(n_gen));
    usage.set("total_tokens", Value::integer(n_prompt + n_gen));
    Value det = Value::object();
    det.set("cached_tokens", Value::integer(cached));
    usage.set("prompt_tokens_details", det);

    if (stream) {
        Value fin = delta_chunk(Value::object(), Value::string(finish));
        fin.set("usage", usage);
        fin.set("timings", timings);
        send_chunk(fin);
        if (cr.include_usage && !aborted) {
            Value u = base_chunk("chat.completion.chunk");
            u.set("choices", Value::array());
            u.set("usage", usage);
            u.set("timings", timings);
            send_chunk(u);
        }
        if (!aborted && !conn.write_chunk("data: [DONE]\n\n")) aborted = true;
        if (!aborted) conn.end_stream();
    } else {
        Value m = Value::object();
        m.set("role", Value::string("assistant"));
        if (msg.content.empty() && !msg.calls.empty()) m.set("content", Value::null());
        else m.set("content", Value::string(msg.content));
        if (msg.has_reasoning) m.set("reasoning_content", Value::string(msg.reasoning));
        if (!msg.calls.empty()) m.set("tool_calls", tool_calls_json);
        Value ch = Value::object();
        ch.set("index", Value::integer(0));
        ch.set("message", m);
        ch.set("finish_reason", Value::string(finish));
        Value choices = Value::array();
        choices.push(ch);
        Value r = base_chunk("chat.completion");
        r.set("choices", choices);
        r.set("usage", usage);
        r.set("timings", timings);
        conn.respond(200, "application/json", jinja::json_compact(r));
    }
    if (vl_ >= 1) {
        const std::string sp = be_.spec_stats();
        vline("decode", vlog::fmt("%s tok in %s  %s  finish %s%s%s", vlog::thou(n_gen).c_str(), vlog::ms(t_decode * 1000).c_str(), vlog::rate(n_gen, t_decode).c_str(),
                                  finish.c_str(), sp.empty() ? "" : "  spec ", sp.c_str()));
        const Backend::Pool pl = be_.pool();
        vline("done", vlog::fmt("%s  slot %d  prompt %s (cached %s) + gen %s  wall %s  tape %s  ckpts %zu%s%s", rq.path.c_str(), cur_, vlog::thou(n_prompt).c_str(),
                                vlog::thou(cached).c_str(), vlog::thou(n_gen).c_str(), vlog::ms((now_s() - t_start) * 1000).c_str(), vlog::thou(long(tape.size())).c_str(),
                                ses.ck.size(), pl.positions > 0 ? "  " : "", pool_note().c_str()),
              vlog::Tone::Good);
    } else if (cfg_.log) {
        char tapeinfo[200] = "";
        if (n_snaps_ > 0 || slots_.size() > 1) {
            int n = std::snprintf(tapeinfo, sizeof tapeinfo, " slot=%d via=%s ckpts=%zu restored_at=%d", cur_, how, ses.ck.size(), restored_from);
            if (adopted_from >= 0 && n > 0 && size_t(n) < sizeof tapeinfo) n += std::snprintf(tapeinfo + n, sizeof tapeinfo - size_t(n), " adopted_from=%d", adopted_from);
            if (const Backend::Pool pl = be_.pool(); pl.positions > 0 && n > 0 && size_t(n) < sizeof tapeinfo)
                std::snprintf(tapeinfo + n, sizeof tapeinfo - size_t(n), " pool=%d/%dpg shared=%d", pl.used_pages, pl.total_pages, pl.shared_pages);
        }
        const std::string sp = be_.spec_stats();
        std::fprintf(stderr,
                     "serve: %s stream=%d prompt=%d cached=%d gen=%d finish=%s prefill=%.2fs (%.2f tok/s) decode=%.2fs (%.2f tok/s)%s%s%s\n",
                     rq.path.c_str(), stream ? 1 : 0, n_prompt, cached, n_gen, finish.c_str(), t_prefill, rate(n_new, t_prefill), t_decode,
                     rate(n_gen, t_decode), tapeinfo, sp.empty() ? "" : " spec=", sp.c_str());
    }
}

// ---- the server's main ----------------------------------------------------------------------------

namespace {

std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop.store(true); }

void usage(const MainSpec& spec) {
    std::fprintf(stderr,
                 "usage: %s -m model.gguf | -hf org/repo[:QUANT]  [--host 127.0.0.1] [--port 8080] [--ctx auto|N|max] [--threads 6]\n"
                 "                   [--model-name NAME] [--max-tokens N (0: to the context end)] [--temperature T] [--top-k K]\n"
                 "                   [--top-p P] [--min-p P] [--log-prompts] [--quiet] [-v | -vv]\n"
                 "                   [--api-key KEY (repeatable; /health stays open, the rest wants Authorization: Bearer KEY)]\n"
                 "                   [--checkpoints 8] [--checkpoint-gap 64] [--preserve-thinking on|off]\n"
                 "                   [--slots 1 (alias --sessions)] [--vram-reserve-mib 768]%s\n",
                 spec.prog, spec.gpu ? " [--min-free-gb G]\n                   [--kv auto|f16|q8|q4] [--prefill-planes 2]\n                   [--spec auto|mtp|ngram|dflash|dspark|off] [--ngram-n N] [--ngram-max K]\n                   [--draft-model DIR|FILE|none (default: the zoo's drafter for this model)] [--draft-quant auto|native|q8|bf16]\n                   [--draft-vocab N] [--draft-k K]" : "");
}

}  // namespace

int run_main(int argc, char** argv, const MainSpec& spec) {
    MainOptions o;
    bool have_model = false;
    for (int i = 1; i < argc; ++i) {
        auto is = [&](const char* a) { return std::strcmp(argv[i], a) == 0; };
        auto val = [&]() -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };
        const char* v = nullptr;
        if (is("-m") || is("--model")) {
            if ((v = val())) {
                o.model = v;
                have_model = true;
            }
        } else if ((is("-hf") || is("--hf")) && (v = val())) {
            o.hf = v;
            have_model = true;
        } else if (is("--host") && (v = val())) o.host = v;
        else if (is("--port") && (v = val())) o.port = std::atoi(v);
        else if (is("--ctx") && (v = val())) o.ctx = std::string(v) == "max" ? 0 : std::string(v) == "auto" ? -1 : std::max(1, std::atoi(v));
        else if ((is("-t") || is("--threads")) && (v = val())) o.threads = std::atoi(v);
        else if (is("--model-name") && (v = val())) o.name = v;
        else if (is("--max-tokens") && (v = val())) o.cfg.default_max_tokens = std::atoi(v);
        else if (is("--temperature") && (v = val())) o.cfg.sampling.temperature = std::atof(v);
        else if (is("--top-k") && (v = val())) o.cfg.sampling.top_k = std::atoi(v);
        else if (is("--top-p") && (v = val())) o.cfg.sampling.top_p = std::atof(v);
        else if (is("--min-p") && (v = val())) o.cfg.sampling.min_p = std::atof(v);
        else if (spec.gpu && is("--min-free-gb") && (v = val())) o.min_free_gb = std::atof(v);
        else if (spec.gpu && is("--kv") && (v = val())) o.kv = v;
        else if (spec.gpu && is("--prefill-planes") && (v = val())) o.prefill_planes = std::atoi(v);
        else if (spec.gpu && is("--spec") && (v = val())) o.spec = v;
        else if (spec.gpu && is("--ngram-n") && (v = val())) o.ngram_n = std::atoi(v);
        else if (spec.gpu && is("--ngram-max") && (v = val())) o.ngram_max = std::atoi(v);
        else if (is("--preserve-thinking") && (v = val())) o.cfg.preserve_thinking = std::string(v) != "off";
        else if (spec.gpu && (is("--draft-model") || is("--dflash")) && (v = val())) {
            o.no_draft = std::string(v) == "none";
            o.draft_model = o.no_draft ? std::string() : std::string(v);
        }
        else if (spec.gpu && is("--draft-quant") && (v = val())) o.draft_quant = v;
        else if (spec.gpu && is("--draft-vocab") && (v = val())) o.draft_vocab = std::atoi(v);
        else if (spec.gpu && is("--draft-k") && (v = val())) o.draft_k = std::atoi(v);
        else if (is("--checkpoints") && (v = val())) o.cfg.checkpoints = std::atoi(v);
        else if (is("--checkpoint-gap") && (v = val())) o.cfg.checkpoint_gap = std::atoi(v);
        else if ((is("--slots") || is("--sessions")) && (v = val())) o.cfg.slots = std::atoi(v);
        else if (is("--vram-reserve-mib") && (v = val())) o.cfg.vram_reserve_mib = std::atoi(v);
        else if (is("--api-key") && (v = val())) {
            if (!*v) {
                std::fprintf(stderr, "%s: --api-key needs a non-empty key\n", spec.prog);
                return 2;
            }
            o.cfg.api_keys.emplace_back(v);
        } else if (is("--log-prompts")) o.cfg.log_prompts = true;
        else if (is("-v") || is("--verbose")) o.cfg.verbose = std::max(o.cfg.verbose, 1);
        else if (is("-vv")) o.cfg.verbose = 2;
        else if (is("--quiet")) o.cfg.log = false;
        else {
            usage(spec);
            return 2;
        }
    }
    if (spec.gpu && o.spec != "auto" && o.spec != "mtp" && o.spec != "ngram" && o.spec != "dflash" && o.spec != "dspark" &&
        o.spec != "off") {
        std::fprintf(stderr, "%s: --spec takes auto, mtp, ngram, dflash, dspark or off, not %s\n", spec.prog, o.spec.c_str());
        return 2;
    }
    if (!have_model || o.ngram_n < 0 || o.ngram_n == 1 || o.ngram_n > 64 || o.ngram_max < 0 || o.ngram_max > 7 || (o.ctx > 0 && o.ctx < 16) || (o.ctx == 0 && !spec.gpu) || o.threads < 1 || o.port < 0 || o.port > 65535 || o.min_free_gb < 0 || o.cfg.checkpoints < 0 ||
        o.cfg.checkpoints > 4096 || o.cfg.checkpoint_gap < 1 || o.cfg.slots < 1 || o.cfg.slots > 64 || o.cfg.vram_reserve_mib < 0 || o.draft_k < 0 || o.draft_k > 7 || o.draft_vocab < 0 ||
        (o.draft_quant != "q8" && o.draft_quant != "bf16" && o.draft_quant != "native" && o.draft_quant != "auto")) {
        usage(spec);
        return 2;
    }

    std::string err;
    if (!o.hf.empty()) {   // the model by its Hugging Face name: the local folders, the cache, or a download
        if (!hf::model(o.hf, &o.model, &err)) {
            std::fprintf(stderr, "%s: %s\n", spec.prog, err.c_str());
            return 1;
        }
        std::fprintf(stderr, "%s: -hf %s -> %s\n", spec.prog, o.hf.c_str(), o.model.c_str());
    }
    auto gguf = gguf::File::open(o.model, &err);
    if (!gguf) {
        std::fprintf(stderr, "%s: %s\n", spec.prog, err.c_str());
        return 1;
    }
    // --ctx auto: the stack never needs more than ~90K and above 64K is luxury, so 65536 (or the model's own context when it
    // is shorter); the VRAM a longer context would take goes to the drafters, the experts and the K/V precision.
    if (o.ctx < 0) {
        o.ctx_auto = true;
        const int64_t trained = gguf->get_int(std::string(gguf->arch()) + ".context_length").value_or(65536);
        o.ctx = spec.gpu ? int(std::clamp<int64_t>(trained, 16, 65536)) : 4096;
    }
    // The drafter the zoo holds for what this model is (its architecture and geometry, so a fine-tune gets its base's),
    // unless one was named or none was asked for. A zoo that cannot fetch it starts without it.
    if (spec.gpu && o.draft_model.empty() && !o.no_draft && o.spec != "off" && o.spec != "mtp" && o.spec != "ngram") {
        hf::Drafter d;
        std::string zerr;
        if (!hf::zoo(*gguf, &d, &zerr)) {
            std::fprintf(stderr, "%s: no drafter: %s\n", spec.prog, zerr.c_str());
        } else if (!d.path.empty()) {
            o.draft_model = d.path;
            o.draft_alts = d.alts;
            std::fprintf(stderr, "%s: drafter %s -> %s%s\n", spec.prog, d.from.c_str(), d.path.c_str(),
                         d.alts.empty() ? "" : " (and smaller quants for a tight card)");
        }
    }
    tok::Vocab vocab;
    if (!tok::Vocab::load(*gguf, vocab, &err)) {
        std::fprintf(stderr, "%s: %s\n", spec.prog, err.c_str());
        return 1;
    }
    const tok::Tokenizer tokenizer(vocab);
    auto renderer = chat::Renderer::from_gguf(*gguf, &err);
    if (!renderer) {
        std::fprintf(stderr, "%s: chat template: %s\n", spec.prog, err.c_str());
        return 1;
    }
    std::unique_ptr<Backend> backend;
    if (o.cfg.log && o.cfg.verbose >= 1) {
        vlog::set_color(vlog::stderr_wants_color());
        vlog::line("boot", vlog::fmt("%s  verbose %d  model %s  ctx %s  slots %d  checkpoints %d", spec.prog, o.cfg.verbose, o.model.c_str(), vlog::thou(o.ctx).c_str(), o.cfg.slots,
                                     o.cfg.checkpoints));
    }
    const double t_open = now_s();
    if (const int rc = spec.open(o, &backend, &err); rc != 0 || !backend) {
        std::fprintf(stderr, "%s: %s\n", spec.prog, err.empty() ? "the backend did not open" : err.c_str());
        return rc != 0 ? rc : 1;
    }
    if (o.ctx == 0 || o.ctx_auto) o.ctx = backend->n_ctx();   // --ctx max: the context the backend fitted
    if (o.name.empty()) {
        const auto n = gguf->get_str("general.name");
        o.name = n ? std::string(*n) : o.model;
        if (!n) o.name = o.name.substr(o.name.find_last_of('/') == std::string::npos ? 0 : o.name.find_last_of('/') + 1);
    }
    o.cfg.model_id = o.name;
    if (o.cfg.log && o.cfg.verbose >= 1) vlog::line("boot", vlog::fmt("backend open in %s", vlog::ms((now_s() - t_open) * 1000).c_str()));

    Service service(*backend, tokenizer, *renderer, o.cfg);
    auto server = http::Server::listen(o.host, o.port, &err);
    if (!server) {
        std::fprintf(stderr, "%s: %s\n", spec.prog, err.c_str());
        return 1;
    }

    std::signal(SIGPIPE, SIG_IGN);
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    const std::string what = backend->describe();
    std::printf("%s: %s ctx %d%s%s, listening on http://%s:%d\n", spec.prog, o.name.c_str(), o.ctx, what.empty() ? "" : ", ", what.c_str(),
                o.host.c_str(), server->port());
    std::fflush(stdout);
    server->run([&](const http::Request& rq, http::Conn& c) { service.handle(rq, c); }, g_stop);
    std::fprintf(stderr, "%s: stopped\n", spec.prog);
    return 0;
}

}  // namespace limbo::serve
