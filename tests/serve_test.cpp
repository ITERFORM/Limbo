// Copyright (c) 2026 ITERFORM. All rights reserved.
// tests/serve_test.cpp — the OpenAI-compatible server end to end, over real sockets on 127.0.0.1, with a
// fake Backend that emits a scripted token sequence. No model needed.
//
// The fake predicts, after the forward pass at position p, script[p + 1 - gen_from] (then the end token),
// and counts every call, so the tests can say what the server fed the backend: how many forward passes,
// how many resets, and that positions were always consecutive.
#include "limbo/check.hpp"
#include "limbo/chat.hpp"
#include "limbo/http.hpp"
#include "limbo/jinja.hpp"
#include "limbo/kvpool.hpp"
#include "limbo/ngram.hpp"
#include "limbo/serve.hpp"
#include "limbo/tok.hpp"
#include "limbo/vlog.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using limbo::jinja::Value;
namespace serve = limbo::serve;

namespace {

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

template <class F>
bool wait_until(F f, int timeout_ms) {
    const double end = now_ms() + timeout_ms;
    while (now_ms() < end) {
        if (f()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return f();
}

// ---- a vocabulary without a model: 256 byte pieces and the Qwen chat/think/tool special tokens ----

struct MiniVocab {
    std::vector<std::string> storage;
    limbo::tok::Vocab v;
    int32_t im_start = 0, im_end = 0;

    MiniVocab() {
        storage.reserve(512);
        for (uint32_t b = 0; b < 256; ++b) add(limbo::tok::byte_to_str(uint8_t(b)), limbo::tok::kNormal);
        const int32_t eot = add("<|endoftext|>", limbo::tok::kControl);
        im_start = add("<|im_start|>", limbo::tok::kControl);
        im_end = add("<|im_end|>", limbo::tok::kControl);
        for (const char* s : {"<think>", "</think>", "<tool_call>", "</tool_call>"}) add(s, limbo::tok::kUserDefined);
        v.bos = eot;
        v.eos = im_end;
        v.add_bos = false;
        v.pre = "qwen35";
        v.model = "gpt2";
        for (size_t i = 0; i < v.tokens.size(); ++i)
            if (v.types[i] == limbo::tok::kControl || v.types[i] == limbo::tok::kUserDefined) v.specials.emplace_back(v.tokens[i], int32_t(i));
        std::stable_sort(v.specials.begin(), v.specials.end(), [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
    }
    int32_t add(std::string_view piece, int32_t type) {
        storage.emplace_back(piece);
        v.tokens.push_back(storage.back());
        v.types.push_back(type);
        const int32_t id = int32_t(v.tokens.size()) - 1;
        v.id_of[v.tokens.back()] = id;
        return id;
    }
};

struct Fake : serve::Backend {
    int nv;
    int32_t eog;
    std::vector<float> logits;
    std::vector<int32_t> script;  // written only while the server is idle
    std::atomic<int> gen_from{0}, n_forward{0}, n_reset{0}, n_bad_pos{0}, sleep_us{0}, ctx{0};
    // The hooks the server calls instead of forward(token, pos): prompt tokens without logits, the greedy
    // token, and batched prompt reads of at most `chunk` tokens.
    std::atomic<int> n_nologits{0}, n_greedy{0}, n_prefill{0}, n_bad_chunk{0}, chunk{1};
    int expect_pos = 0;

    Fake(int vocab, int32_t end) : nv(vocab), eog(end), logits(size_t(vocab), 0.f) {}
    int n_vocab() const override { return nv; }
    void reset() override {
        expect_pos = 0;
        ++n_reset;
    }
    int n_ctx() const override { return ctx.load(); }
    const float* forward(int32_t token, int pos, bool want_logits) override {
        if (!want_logits) ++n_nologits;
        const float* l = forward(token, pos);
        return want_logits ? l : nullptr;
    }
    int32_t forward_greedy(int32_t token, int pos) override {
        ++n_greedy;
        return serve::Sampler::argmax(forward(token, pos), nv);
    }
    int prefill_chunk() const override { return chunk.load(); }
    std::atomic<int> n_sync{0};
    void sync() override { ++n_sync; }
    void prefill(const int32_t* tokens, int n, int pos0) override {
        ++n_prefill;
        if (n > chunk.load() || n < 1) ++n_bad_chunk;
        for (int i = 0; i < n; ++i) forward(tokens[i], pos0 + i, false);
    }
    const float* forward(int32_t, int pos) override {
        if (pos != expect_pos) ++n_bad_pos;
        expect_pos = pos + 1;
        ++n_forward;
        if (sleep_us.load() > 0) std::this_thread::sleep_for(std::chrono::microseconds(sleep_us.load()));
        const int k = pos + 1 - gen_from.load();
        const int32_t next = (k >= 0 && size_t(k) < script.size()) ? script[size_t(k)] : eog;
        std::fill(logits.begin(), logits.end(), 0.f);
        logits[size_t(next)] = 10.f;
        return logits.data();
    }
};

// A Fake that speculates: decode_spec takes 1..3 tokens a call in a fixed rotation (as a backend whose drafts hit and missed would),
// reads every token but the last at consecutive positions, and never reads a token pick.stop names. Logs every read.
struct SpecFake : Fake {
    std::vector<std::pair<int32_t, int>> reads;  // forward(): token, position
    std::vector<int> lens{1, 3, 2, 3, 1, 2};
    int calls = 0, by_len[4] = {0, 0, 0, 0}, n_stopped = 0;
    SpecFake(int vocab, int32_t end) : Fake(vocab, end) {}
    const float* forward(int32_t t, int pos) override {
        reads.emplace_back(t, pos);
        return Fake::forward(t, pos);
    }
    bool spec_ready() const override { return true; }
    int decode_spec(int32_t fed, int pos, int max_out, const SpecPick& pick, int32_t* out) override {
        const int want = std::min(lens[size_t(calls++) % lens.size()], max_out);
        const float* lg = forward(fed, pos);
        for (int n = 1;; ++n) {
            bool kept = false;
            const int32_t t = pick.greedy ? serve::Sampler::argmax(lg, nv) : pick.row(lg, -1, &kept);
            out[n - 1] = t;
            const bool stop = pick.stop && pick.stop(t);
            n_stopped += stop;
            if (n == want || stop) {
                ++by_len[n];
                return n;
            }
            lg = forward(t, pos + n);
        }
    }
};

// A Fake whose state is a hash chain over the tokens it consumed (and a position), per home, that can be
// snapshotted into slots and restored. After any request the test compares the live home's chain with the
// chain of the tokens a cold read of that conversation would have consumed: equal only if the server
// restored the right state and read exactly the right tokens after it.
struct SnapFake : Fake {
    struct St {
        int pos = 0;
        uint64_t h = 0;
    };
    struct Slot {
        St st;
        bool ok = false;
    };
    std::vector<St> home;
    std::vector<Slot> snaps;
    int cur = 0, n_slots_max = 2, slots_max = 4, min_piece = 1;
    int n_snapshot = 0, n_restore = 0, n_slot_calls = 0;
    bool fail_restore = false;
    std::vector<int> pieces;  // the length of every prefill() call

    SnapFake(int vocab, int32_t end) : Fake(vocab, end), home(1) {}
    static uint64_t mix(uint64_t h, int32_t t) { return (h ^ (uint64_t(uint32_t(t)) + 0x9E3779B97F4A7C15ull)) * 0x100000001B3ull + 0x7F4A7C15ull; }
    static uint64_t chain(const std::vector<int32_t>& a, const std::vector<int32_t>& b = {}) {
        uint64_t h = 0;
        for (int32_t t : a) h = mix(h, t);
        for (int32_t t : b) h = mix(h, t);
        return h;
    }
    void reset() override {
        Fake::reset();
        home[size_t(cur)] = St{};
    }
    const float* forward(int32_t t, int pos) override {
        St& s = home[size_t(cur)];
        expect_pos = s.pos;
        const float* l = Fake::forward(t, pos);
        s.pos = expect_pos;
        s.h = mix(s.h, t);
        return l;
    }
    void prefill(const int32_t* tokens, int n, int pos0) override {
        pieces.push_back(n);
        Fake::prefill(tokens, n, pos0);
    }
    // The card: free bytes, what an extra home and a snapshot slot take, and the reserve the server asked to keep.
    uint64_t free_b = 64ull << 30, reserve = 0, slot_b = 100ull << 20, snap_b = 10ull << 20;
    int set_snapshot_slots(int want) override {
        ++n_slot_calls;
        size_t n = 0;
        while (int(n) < std::min(want, slots_max) && free_b >= snap_b + reserve) {
            free_b -= snap_b;
            ++n;
        }
        snaps.assign(n, Slot{});
        return int(n);
    }
    void set_vram_reserve(uint64_t b) override { reserve = b; }
    uint64_t vram_free() const override { return free_b; }
    uint64_t slot_bytes() const override { return slot_b; }
    uint64_t snapshot_bytes() const override { return snap_b; }
    int snapshot_slots() const override { return int(snaps.size()); }
    bool snapshot(int slot) override {
        if (slot < 0 || size_t(slot) >= snaps.size()) return false;
        snaps[size_t(slot)] = Slot{home[size_t(cur)], true};
        ++n_snapshot;
        return true;
    }
    bool restore(int slot) override {
        if (fail_restore || slot < 0 || size_t(slot) >= snaps.size() || !snaps[size_t(slot)].ok) return false;
        home[size_t(cur)] = snaps[size_t(slot)].st;
        ++n_restore;
        return true;
    }
    int prefill_batch_min() const override { return min_piece; }
    int set_slots(int want) override {
        while (int(home.size()) < std::min(want, n_slots_max) && free_b >= slot_b + reserve) {
            free_b -= slot_b;
            home.emplace_back();
        }
        return int(home.size());
    }
    int slots() const override { return int(home.size()); }
    bool select_slot(int h) override {
        if (h < 0 || size_t(h) >= home.size()) return false;
        cur = h;
        return true;
    }
};

// A SnapFake with a paged K/V pool: pages are counted by limbo::kv::PagePool (the bookkeeping the GPU engine uses), the K/V contents
// are not modelled (the state chain is). Sharing a prefix then shows as pages held once.
struct PoolFake : SnapFake {
    limbo::kv::PagePool pp;
    PoolFake(int vocab, int32_t end, int pages, int n_ctx) : SnapFake(vocab, end), pp(pages, (n_ctx + 255) / 256, 8) { ctx = n_ctx; }
    void reset() override {
        SnapFake::reset();
        pp.cut(cur, 0);
    }
    const float* forward(int32_t t, int pos) override {
        pp.write(cur, pos, pos + 1, nullptr);
        return SnapFake::forward(t, pos);
    }
    Pool pool() const override {
        Pool p;
        p.page = 256;
        p.total_pages = pp.pages();
        p.positions = pp.pages() * 256;
        p.used_pages = pp.used_pages();
        p.shared_pages = pp.shared_pages();
        p.free_positions = pp.free_pages() * 256;
        p.copies = n_copies;
        return p;
    }
    int slot_pages(int s) const override { return pp.held(s); }
    void cut(int keep) override { pp.cut(cur, keep); }
    bool reserve(int n) override { return pp.fill(cur, n); }
    uint64_t n_copies = 0;
    bool share(int from, int n) override {
        std::vector<limbo::kv::PageCopy> cp;
        const bool ok = pp.share(cur, from, n, &cp);
        n_copies += cp.size();
        return ok;
    }
    int room(int at) const override { return pp.room(cur, at); }
};

// A PoolFake that drafts from the slots' tapes (limbo::ngram, wired to the backend calls as GpuBackend wires it) in decode_spec: every
// draft is checked against the scripted model, so a hit reads the drafted token at the next position and a miss ends the run.
struct NgramFake : PoolFake {
    limbo::ngram::Slots ix;
    bool in_prefill = false, in_spec = false;
    int cycles = 0, drafted = 0, kept = 0, plain = 0;
    NgramFake(int vocab, int32_t end, int pages, int n_ctx, int n) : PoolFake(vocab, end, pages, n_ctx) { ix.configure(n, n_ctx, 1); }
    int set_slots(int want) override {
        const int n = PoolFake::set_slots(want);
        ix.resize(n);
        return n;
    }
    bool select_slot(int h) override { return PoolFake::select_slot(h) && ix.select(h); }
    void reset() override {
        PoolFake::reset();
        ix.cur().clear();
    }
    void cut(int keep) override {
        PoolFake::cut(keep);
        ix.cur().truncate(size_t(keep));
    }
    bool share(int from, int n) override {
        const bool ok = PoolFake::share(from, n);
        if (ok) ix.share(from, size_t(n));
        return ok;
    }
    void prefill(const int32_t* tokens, int n, int pos0) override {
        ix.cur().put(pos0, tokens, n, false);
        in_prefill = true;
        PoolFake::prefill(tokens, n, pos0);
        in_prefill = false;
    }
    const float* forward(int32_t t, int pos) override {
        if (!in_prefill && !in_spec) ix.cur().put(pos, &t, 1, true);
        return PoolFake::forward(t, pos);
    }
    bool spec_ready() const override { return true; }
    int decode_spec(int32_t fed, int pos, int max_out, const SpecPick& pick, int32_t* out) override {
        ix.cur().put(pos, &fed, 1, true);
        const limbo::ngram::Draft dr = ix.cur().propose(std::min(max_out - 1, 7));
        in_spec = true;
        ++cycles;
        if (dr.n == 0) ++plain;
        const float* lg = forward(fed, pos);
        for (int n = 1;; ++n) {
            const int32_t draft = n - 1 < dr.n ? dr.tok[n - 1] : -1;
            bool k = false;
            const int32_t t = pick.greedy ? serve::Sampler::argmax(lg, nv) : pick.row(lg, draft, &k);
            if (pick.greedy) k = t == draft;
            out[n - 1] = t;
            if (n - 1 < dr.n) {
                ++drafted;
                kept += k;
            }
            if (!(draft >= 0 && k) || (pick.stop && pick.stop(t)) || n == max_out) {
                in_spec = false;
                if (n > 1) ix.cur().put(pos + 1, out, n - 1, true);
                return n;
            }
            lg = forward(t, pos + n);
        }
    }
};

// A PoolFake with a start-up report and a memory table, and a note of the verbosity it was told.
struct VerboseFake : PoolFake {
    int verbose_level = -1;
    VerboseFake(int vocab, int32_t end, int pages, int n_ctx) : PoolFake(vocab, end, pages, n_ctx) {}
    void set_verbose(int l) override { verbose_level = l; }
    std::vector<std::string> startup_report() const override { return {"model: fake model, 2 layers", "kv: fake K/V, 40 pages"}; }
    std::vector<MemUse> memory_report() const override { return {{"weights", 1ull << 30, "tensors"}, {"K/V pool", 300ull << 20, "pages"}}; }
    uint64_t vram_free() const override { return 2ull << 30; }
};

// ---- a small blocking HTTP client ----

struct Sock {
    int fd = -1;
    std::string buf;
    bool eof = false;
    ~Sock() {
        if (fd >= 0) ::close(fd);
    }
    bool dial(int port) {
        fd = ::socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_port = htons(uint16_t(port));
        ::inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
        return ::connect(fd, reinterpret_cast<sockaddr*>(&a), sizeof a) == 0;
    }
    bool send_all(std::string_view d) {
        size_t off = 0;
        while (off < d.size()) {
            const ssize_t n = ::send(fd, d.data() + off, d.size() - off, MSG_NOSIGNAL);
            if (n <= 0) return false;
            off += size_t(n);
        }
        return true;
    }
    bool fill(int timeout_ms) {
        pollfd p{fd, POLLIN, 0};
        if (::poll(&p, 1, timeout_ms) <= 0) return false;
        char tmp[16384];
        const ssize_t n = ::recv(fd, tmp, sizeof tmp, 0);
        if (n <= 0) {
            eof = true;
            return false;
        }
        buf.append(tmp, size_t(n));
        return true;
    }
    bool read_until(const std::string& marker, std::string* out, int timeout_ms) {
        for (;;) {
            const size_t k = buf.find(marker);
            if (k != std::string::npos) {
                out->assign(buf, 0, k + marker.size());
                buf.erase(0, k + marker.size());
                return true;
            }
            if (!fill(timeout_ms)) return false;
        }
    }
    bool read_n(size_t n, std::string* out, int timeout_ms) {
        while (buf.size() < n)
            if (!fill(timeout_ms)) return false;
        out->assign(buf, 0, n);
        buf.erase(0, n);
        return true;
    }
    // One chunk of a chunked body; false at the terminating chunk or on error.
    bool read_chunk(std::string* out, int timeout_ms) {
        std::string line;
        if (!read_until("\r\n", &line, timeout_ms)) return false;
        const size_t n = std::strtoul(line.c_str(), nullptr, 16);
        if (n == 0) {
            std::string tail;
            read_until("\r\n", &tail, timeout_ms);
            return false;
        }
        std::string data, crlf;
        if (!read_n(n, &data, timeout_ms) || !read_n(2, &crlf, timeout_ms)) return false;
        *out = data;
        return true;
    }
};

struct Resp {
    int status = 0;
    std::string head, body;
    std::vector<std::string> chunks;
    bool chunked = false;
    bool ok = false;

    std::string header(const std::string& lower_name) const {
        std::string h = head;
        for (char& c : h)
            if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
        const size_t k = h.find("\r\n" + lower_name + ":");
        if (k == std::string::npos) return "";
        const size_t b = k + 3 + lower_name.size();
        const size_t e = h.find("\r\n", b);
        std::string v = head.substr(b, e - b);
        while (!v.empty() && v.front() == ' ') v.erase(0, 1);
        return v;
    }
};

bool read_head(Sock& s, Resp* r, int timeout_ms) {
    if (!s.read_until("\r\n\r\n", &r->head, timeout_ms)) return false;
    r->status = std::atoi(r->head.c_str() + 9);
    r->chunked = r->header("transfer-encoding") == "chunked";
    return true;
}

Resp read_response(Sock& s, int timeout_ms = 15000) {
    Resp r;
    if (!read_head(s, &r, timeout_ms)) return r;
    if (r.chunked) {
        std::string c;
        while (s.read_chunk(&c, timeout_ms)) {
            r.chunks.push_back(c);
            r.body += c;
        }
        r.ok = true;
    } else {
        const std::string cl = r.header("content-length");
        if (!cl.empty()) r.ok = s.read_n(size_t(std::atol(cl.c_str())), &r.body, timeout_ms);
        else {
            while (s.fill(timeout_ms)) {}
            r.body = s.buf;
            r.ok = true;
        }
    }
    return r;
}

Resp request(int port, const std::string& method, const std::string& path, const std::string& body = "",
             const std::string& extra = "") {
    Sock s;
    Resp r;
    if (!s.dial(port)) return r;
    std::string req = method + " " + path + " HTTP/1.1\r\nHost: 127.0.0.1\r\n" + extra;
    if (method == "POST") req += "Content-Type: application/json\r\nContent-Length: " + std::to_string(body.size()) + "\r\n";
    req += "\r\n" + body;
    s.send_all(req);
    return read_response(s);
}

Value json(const std::string& text) {
    Value v;
    std::string err;
    if (!limbo::jinja::json_parse(text, &v, &err)) {
        std::printf("FAIL test json did not parse (%s): %.200s\n", err.c_str(), text.c_str());
        ++limbo::test::failures();
    }
    return v;
}

// v["a"][0]["b"] style access; undefined when any step is missing.
Value at(const Value& v, std::initializer_list<const char*> path) {
    Value cur = v;
    for (const char* p : path) {
        if (cur.is_list()) {
            const size_t i = size_t(std::atoi(p));
            if (i >= cur.list().size()) return Value::undefined();
            cur = cur.list()[i];
        } else if (cur.is_dict()) {
            const Value* n = cur.find(p);
            if (!n) return Value::undefined();
            cur = *n;
        } else return Value::undefined();
    }
    return cur;
}

std::string str_at(const Value& v, std::initializer_list<const char*> path) {
    const Value x = at(v, path);
    return x.is_string() ? x.s() : std::string("<") + (x.is_undefined() ? "missing" : x.is_null() ? "null" : "not a string") + ">";
}

// The SSE events of a streamed body: the JSON of every `data:` line, plus whether [DONE] came.
struct Sse {
    std::vector<Value> events;
    bool done = false;
    bool keepalive = false;
    bool malformed = false;
};
Sse parse_sse(const std::string& body) {
    Sse s;
    size_t at_ = 0;
    while (at_ < body.size()) {
        size_t e = body.find("\n\n", at_);
        if (e == std::string::npos) {
            s.malformed = true;
            break;
        }
        const std::string ev = body.substr(at_, e - at_);
        at_ = e + 2;
        if (ev.rfind(": ", 0) == 0) {
            s.keepalive = true;
            continue;
        }
        if (ev.rfind("data: ", 0) != 0) {
            s.malformed = true;
            continue;
        }
        const std::string d = ev.substr(6);
        if (d == "[DONE]") s.done = true;
        else s.events.push_back(json(d));
    }
    return s;
}

std::string json_str(const Value& v) { return limbo::jinja::json_compact(v); }

const char* kWeatherTools = R"([{"type":"function","function":{"name":"get_weather","description":"Weather for a city.",
  "parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}},"required":["city"]}}}])";

std::string call_text(const std::string& name, const std::vector<std::pair<std::string, std::string>>& params) {
    std::string o = "<tool_call>\n<function=" + name + ">\n";
    for (const auto& p : params) o += "<parameter=" + p.first + ">\n" + p.second + "\n</parameter>\n";
    return o + "</function>\n</tool_call>";
}

}  // namespace

int main() {
    MiniVocab mv;
    const limbo::tok::Tokenizer tokenizer(mv.v);

    // The model's own template, from the fixtures.
    const char* fx = std::getenv("LIMBO_CHAT_FIXTURES");
    const fs::path dir = fx && *fx ? fx : "bench/fixtures/chat";
    std::ifstream tf(dir / "template.jinja", std::ios::binary);
    std::stringstream tss;
    tss << tf.rdbuf();
    std::string err;
    auto renderer = limbo::chat::Renderer::from_source(tss.str(), &err);
    CHECK(renderer != nullptr, "serve: the chat template compiles");
    if (!renderer) return limbo::test::finish("serve");

    Fake fake(int(mv.v.tokens.size()), mv.im_end);
    serve::Config cfg;
    cfg.model_id = "limbo-test";
    cfg.log = false;
    cfg.keepalive_ms = 0;
    CHECK(serve::Config().slots == 1, "Config: the default is ONE slot");
    cfg.slots = 2;  // the routing tests below are about two conversations
    CHECK(serve::Config().preserve_thinking, "Config: thinking is preserved by default");
    // The tape, home and pool tests below are about a history that diverges from the tape (the template dropping earlier
    // think blocks) and what a restore does then; preserving the thinking takes that divergence away, so they run without
    // it. Its own block (preserve:) is after the tape's.
    cfg.preserve_thinking = false;
    cfg.vram_reserve_mib = 768;   // the slot and VRAM-table tests below are written against a 768 MiB reserve (the default is 256)
    serve::Service svc(fake, tokenizer, *renderer, cfg);

    limbo::http::ServerOptions so;
    so.header_timeout_ms = 400;
    so.body_timeout_ms = 2000;
    so.max_body_bytes = 1u << 20;
    auto server = limbo::http::Server::listen("127.0.0.1", 0, &err, so);
    CHECK(server != nullptr, "serve: the server binds an ephemeral port on 127.0.0.1");
    if (!server) return limbo::test::finish("serve");
    const int port = server->port();
    std::atomic<bool> stop{false};
    std::thread th([&] { server->run([&](const limbo::http::Request& rq, limbo::http::Conn& c) { svc.handle(rq, c); }, stop); });

    auto enc = [&](const std::string& text) {
        std::vector<int32_t> ids;
        limbo::tok::Options o;
        o.parse_special = true;
        tokenizer.encode(text, ids, o);
        return ids;
    };
    auto prompt_len = [&](const Value& body) {
        serve::ChatRequest cr;
        serve::Prompt p;
        std::string e;
        if (!svc.parse_request(body, &cr, &e) || !svc.build_prompt(cr, &p, &e)) {
            std::printf("FAIL prompt_len: %s\n", e.c_str());
            ++limbo::test::failures();
            return 0;
        }
        return int(p.tokens.size());
    };
    // Arms the fake for `body`: it will write `text` after the prompt.
    auto arm = [&](const Value& body, const std::string& text) {
        fake.script = enc(text);
        fake.gen_from = prompt_len(body);
    };
    auto chat = [&](const Value& body) { return request(port, "POST", "/v1/chat/completions", json_str(body)); };

    // ---- the cheap endpoints ----
    {
        const Resp r = request(port, "GET", "/health");
        CHECK(r.status == 200 && json(r.body).get_str("status") == "ok", "GET /health -> 200 {status: ok}");
        CHECK(r.header("content-type").find("application/json") == 0, "GET /health is JSON");
        const Resp m = request(port, "GET", "/v1/models");
        const Value mj = json(m.body);
        CHECK(m.status == 200 && str_at(mj, {"object"}) == "list" && str_at(mj, {"data", "0", "id"}) == "limbo-test" &&
                  str_at(mj, {"data", "0", "object"}) == "model",
              "GET /v1/models lists the model");
        CHECK(request(port, "GET", "/nope").status == 404, "an unknown path is 404");
        CHECK(request(port, "GET", "/v1/chat/completions").status == 405, "GET on the chat endpoint is 405");
        CHECK(request(port, "POST", "/health", "{}").status == 405, "POST on /health is 405");
        const Resp mt = request(port, "GET", "/metrics");
        CHECK(mt.status == 200 && mt.header("content-type").find("text/plain") == 0 &&
                  mt.body.find("\nllamacpp:tokens_predicted_total ") != std::string::npos &&
                  mt.body.find("\nllamacpp:tokens_predicted_seconds_total ") != std::string::npos &&
                  mt.body.find("# TYPE llamacpp:prompt_tokens_total counter") != std::string::npos,
              "GET /metrics: Prometheus text with llama-server's llamacpp: counters (ariadne limbo watch reads them)");
    }

    // ---- request validation ----
    {
        struct Bad { const char* body; const char* needle; };
        const Bad bads[] = {
            {"{not json", "invalid JSON"},
            {"[]", "JSON object"},
            {"{}", "'messages'"},
            {R"({"messages":[]})", "No messages provided."},
            {R"({"messages":[{"content":"x"}]})", "'role'"},
            {R"({"messages":[{"role":"user","content":"x"}],"n":2})", "n = 1"},
            {R"({"messages":[{"role":"user","content":"x"}],"temperature":-1})", "temperature"},
            {R"({"messages":[{"role":"user","content":"x"}],"stop":5})", "stop"},
            {R"({"messages":[{"role":"user","content":"x"}],"max_tokens":0})", "max_tokens"},
            {R"({"messages":[{"role":"user","content":"x"}],"tools":"nope"})", "tools"},
            {R"({"messages":[{"role":"user","content":"x"}],"tool_choice":"sometimes"})", "tool_choice"},
            {R"({"messages":[{"role":"user","content":"x"}],"chat_template_kwargs":5})", "chat_template_kwargs"},
            {R"({"messages":[{"role":"system","content":[{"type":"image_url","image_url":{"url":"x"}}]},{"role":"user","content":"x"}]})", "System message cannot contain images."},
        };
        int good = 0;
        for (const Bad& b : bads) {
            const Resp r = request(port, "POST", "/v1/chat/completions", b.body);
            const Value j = json(r.body);
            if (r.status == 400 && str_at(j, {"error", "message"}).find(b.needle) != std::string::npos && str_at(j, {"error", "type"}) == "invalid_request_error") ++good;
            else std::printf("FAIL bad request %.60s: status %d body %.200s\n", b.body, r.status, r.body.c_str()), ++limbo::test::failures();
        }
        std::printf("ok   %d of %zu malformed requests get a 400 with an OpenAI-shaped error\n", good, sizeof bads / sizeof bads[0]);
    }

    // ---- a plain completion ----
    const Value hello = json(R"({"model":"x","messages":[{"role":"user","content":"Say hello."}],"max_tokens":64,"temperature":0,
                                 "chat_template_kwargs":{"enable_thinking":false}})");
    {
        arm(hello, "Hello there, friend!");
        const int prompt_n = fake.gen_from;
        const Resp r = chat(hello);
        const Value j = json(r.body);
        CHECK(r.status == 200 && r.header("content-type").find("application/json") == 0, "completion: 200 application/json");
        CHECK(str_at(j, {"object"}) == "chat.completion" && str_at(j, {"id"}).rfind("chatcmpl-", 0) == 0 && str_at(j, {"model"}) == "limbo-test",
              "completion: object, id and model");
        CHECK(at(j, {"created"}).is_int(), "completion: created is an integer");
        CHECK(str_at(j, {"choices", "0", "message", "role"}) == "assistant" && str_at(j, {"choices", "0", "message", "content"}) == "Hello there, friend!",
              "completion: the scripted text is the message content");
        CHECK(at(j, {"choices", "0", "message", "reasoning_content"}).is_undefined(), "completion: no reasoning_content when the think block was closed by the prompt");
        CHECK(str_at(j, {"choices", "0", "finish_reason"}) == "stop", "completion: finish_reason stop");
        CHECK(at(j, {"usage", "prompt_tokens"}).i() == prompt_n && at(j, {"usage", "completion_tokens"}).i() == int64_t(fake.script.size()) + 1 &&
                  at(j, {"usage", "total_tokens"}).i() == prompt_n + int64_t(fake.script.size()) + 1,
              "completion: usage counts the prompt and the completion (the end token included)");
        CHECK(at(j, {"usage", "prompt_tokens_details", "cached_tokens"}).is_int() && at(j, {"usage", "prompt_tokens_details", "cached_tokens"}).i() == 0,
              "completion: the first request caches nothing");
        CHECK(at(j, {"timings", "predicted_n"}).i() == int64_t(fake.script.size()) + 1 && at(j, {"timings", "prompt_n"}).i() == prompt_n &&
                  at(j, {"timings", "cache_n"}).i() == 0 && at(j, {"timings", "predicted_ms"}).is_number() && at(j, {"timings", "prompt_ms"}).is_number(),
              "completion: a timings block");
        CHECK(fake.n_reset.load() == 1 && fake.n_forward.load() == prompt_n + int(fake.script.size()) && fake.n_bad_pos.load() == 0,
              "completion: one reset, then prompt + generated tokens forwarded at consecutive positions");
        CHECK(svc.tape_size() == size_t(prompt_n) + fake.script.size(), "completion: the tape holds the prompt and everything generated but the end token");
    }

    // ---- thinking and reasoning_content ----
    const std::string think_tail = "I should greet.\n</think>\n\nHi!";
    {
        const Value body = json(R"({"messages":[{"role":"user","content":"Hi"}],"max_tokens":200,"temperature":0})");
        arm(body, think_tail);
        const Value j = json(chat(body).body);
        CHECK(str_at(j, {"choices", "0", "message", "reasoning_content"}) == "I should greet." && str_at(j, {"choices", "0", "message", "content"}) == "Hi!",
              "thinking: the text before </think> is reasoning_content, after it content");
    }

    // ---- tool calls ----
    const std::string tool_text = "Checking.\n</think>\n\nLet me look.\n\n" + call_text("get_weather", {{"city", "Paris"}, {"days", "2"}});
    const Value tool_body = [&] {
        Value b = json(R"({"messages":[{"role":"user","content":"Weather in Paris?"}],"max_tokens":300,"temperature":0})");
        b.set("tools", json(kWeatherTools));
        return b;
    }();
    Value tool_assistant_msg;
    {
        arm(tool_body, tool_text);
        const Resp r = chat(tool_body);
        const Value j = json(r.body);
        CHECK(str_at(j, {"choices", "0", "finish_reason"}) == "tool_calls", "tool call: finish_reason tool_calls");
        CHECK(str_at(j, {"choices", "0", "message", "content"}) == "Let me look." && str_at(j, {"choices", "0", "message", "reasoning_content"}) == "Checking.",
              "tool call: content and reasoning around the call");
        CHECK(str_at(j, {"choices", "0", "message", "tool_calls", "0", "type"}) == "function" &&
                  str_at(j, {"choices", "0", "message", "tool_calls", "0", "function", "name"}) == "get_weather" &&
                  str_at(j, {"choices", "0", "message", "tool_calls", "0", "id"}).rfind("call_", 0) == 0,
              "tool call: an OpenAI tool_calls entry with id, type and function.name");
        const Value args = json(str_at(j, {"choices", "0", "message", "tool_calls", "0", "function", "arguments"}));
        CHECK(args.is_dict() && args.find("city") && args.find("city")->s() == "Paris" && args.find("days") && args.find("days")->is_int() && args.find("days")->i() == 2,
              "tool call: function.arguments is a JSON string of the typed parameters");
        tool_assistant_msg = *at(j, {"choices", "0"}).find("message");

        // No tool call text without tools: it stays content.
        Value plain = json(R"({"messages":[{"role":"user","content":"Weather?"}],"max_tokens":300,"temperature":0})");
        arm(plain, tool_text);
        const Value pj = json(chat(plain).body);
        CHECK(at(pj, {"choices", "0", "message", "tool_calls"}).is_undefined() && str_at(pj, {"choices", "0", "message", "content"}).find("<tool_call>") != std::string::npos &&
                  str_at(pj, {"choices", "0", "finish_reason"}) == "stop",
              "tool call: without tools the text is left as content");
        // tool_choice none: the tools are not offered and calls are not parsed.
        Value none = tool_body.clone();
        none.set("tool_choice", Value::string("none"));
        const int with_tools = prompt_len(tool_body), without = prompt_len(none);
        CHECK(without < with_tools, "tool_choice none: the tool block is not in the prompt");
        arm(none, tool_text);
        const Value nj = json(chat(none).body);
        CHECK(at(nj, {"choices", "0", "message", "tool_calls"}).is_undefined() && str_at(nj, {"choices", "0", "finish_reason"}) == "stop", "tool_choice none: the call is not parsed");
    }

    // ---- streaming ----
    {
        Value body = json(R"({"messages":[{"role":"user","content":"Hi"}],"max_tokens":200,"temperature":0,"stream":true})");
        arm(body, think_tail);
        const Resp r = chat(body);
        CHECK(r.status == 200 && r.header("content-type").find("text/event-stream") == 0 && r.chunked, "stream: 200, text/event-stream, chunked");
        const Sse s = parse_sse(r.body);
        CHECK(!s.malformed && s.done, "stream: well-formed `data:` events and a final [DONE]");
        std::string reasoning, content;
        bool ids_same = true, objects_ok = true;
        std::string id0 = s.events.empty() ? "" : str_at(s.events[0], {"id"});
        for (const Value& e : s.events) {
            if (str_at(e, {"id"}) != id0) ids_same = false;
            if (str_at(e, {"object"}) != "chat.completion.chunk") objects_ok = false;
            const Value d = at(e, {"choices", "0", "delta"});
            if (d.is_dict()) {
                if (const Value* x = d.find("reasoning_content")) reasoning += x->s();
                if (const Value* x = d.find("content")) content += x->s();
            }
        }
        CHECK(ids_same && objects_ok, "stream: every chunk has the same id and object chat.completion.chunk");
        CHECK(str_at(s.events[0], {"choices", "0", "delta", "role"}) == "assistant", "stream: the first chunk carries the role");
        CHECK(reasoning == "I should greet." && content == "Hi!", "stream: reasoning_content and content deltas add up to the message");
        const Value& last = s.events.back();
        CHECK(str_at(last, {"choices", "0", "finish_reason"}) == "stop" && at(last, {"usage", "completion_tokens"}).i() == int64_t(fake.script.size()) + 1 &&
                  at(last, {"timings", "predicted_n"}).is_int(),
              "stream: the last chunk has finish_reason, usage and timings");
        CHECK(s.events.size() > 5, "stream: the text arrives in several deltas, not one lump");
        bool finish_early = false;
        for (size_t i = 0; i + 1 < s.events.size(); ++i)
            if (!at(s.events[i], {"choices", "0", "finish_reason"}).is_null()) finish_early = true;
        CHECK(!finish_early, "stream: finish_reason is null until the last chunk");

        // include_usage adds a final chunk with empty choices.
        body.set("stream_options", json(R"({"include_usage":true})"));
        arm(body, think_tail);
        const Sse su = parse_sse(chat(body).body);
        CHECK(su.done && at(su.events.back(), {"choices"}).is_list() && at(su.events.back(), {"choices"}).list().empty() && at(su.events.back(), {"usage", "total_tokens"}).is_int(),
              "stream: include_usage ends with an empty-choices usage chunk");
    }
    {
        Value body = tool_body.clone();
        body.set("stream", Value::boolean(true));
        arm(body, tool_text);
        const Sse s = parse_sse(chat(body).body);
        int n_calls = 0;
        std::string name, args, content;
        for (const Value& e : s.events) {
            const Value d = at(e, {"choices", "0", "delta"});
            if (const Value* tc = d.find("tool_calls")) {
                ++n_calls;
                name = str_at(*tc, {"0", "function", "name"});
                args = str_at(*tc, {"0", "function", "arguments"});
                CHECK(at(*tc, {"0", "index"}).is_int() && at(*tc, {"0", "index"}).i() == 0 && str_at(*tc, {"0", "id"}).rfind("call_", 0) == 0 && str_at(*tc, {"0", "type"}) == "function",
                      "stream: a tool call delta has index, id and type");
            }
            if (const Value* x = d.find("content")) content += x->s();
        }
        CHECK(n_calls == 1 && name == "get_weather" && json(args).find("city")->s() == "Paris", "stream: the tool call arrives as one delta with name and arguments");
        CHECK(content == "Let me look." && str_at(s.events.back(), {"choices", "0", "finish_reason"}) == "tool_calls", "stream: content before the call, finish_reason tool_calls");
    }

    // ---- stop sequences, length ----
    {
        Value b = hello.clone();
        b.set("stop", Value::string("STOP"));
        arm(b, "alpha beta STOP gamma");
        const Value j = json(chat(b).body);
        CHECK(str_at(j, {"choices", "0", "message", "content"}) == "alpha beta" && str_at(j, {"choices", "0", "finish_reason"}) == "stop", "stop: the text ends before the stop sequence");
        CHECK(at(j, {"usage", "completion_tokens"}).i() == int64_t(std::string("alpha beta STOP").size()), "stop: generation ends at the token that completes the sequence");

        b.set("stop", json(R"(["zz", "gamma", "beta"])"));
        arm(b, "alpha beta STOP gamma");
        CHECK(str_at(json(chat(b).body), {"choices", "0", "message", "content"}) == "alpha", "stop: the earliest of several sequences wins");

        b.set("stop", json(R"(["STOP"])"));
        b.set("stream", Value::boolean(true));
        arm(b, "alpha STO beta");
        std::string content;
        Sse s = parse_sse(chat(b).body);
        for (const Value& e : s.events)
            if (const Value* x = at(e, {"choices", "0", "delta"}).find("content")) content += x->s();
        CHECK(content == "alpha STO beta" && str_at(s.events.back(), {"choices", "0", "finish_reason"}) == "stop", "stop: text that only looks like the start of a stop sequence is delivered");
        arm(b, "alpha STOP beta");
        content.clear();
        s = parse_sse(chat(b).body);
        for (const Value& e : s.events)
            if (const Value* x = at(e, {"choices", "0", "delta"}).find("content")) content += x->s();
        CHECK(content == "alpha", "stop: streamed text stops before a completed stop sequence (and holds the partial one back)");

        Value l = hello.clone();
        l.set("max_tokens", Value::integer(5));
        arm(l, "abcdefghijklmnop");
        const Value lj = json(chat(l).body);
        CHECK(str_at(lj, {"choices", "0", "message", "content"}) == "abcde" && str_at(lj, {"choices", "0", "finish_reason"}) == "length" &&
                  at(lj, {"usage", "completion_tokens"}).i() == 5,
              "length: max_tokens ends the completion with finish_reason length");
        fake.ctx = 0;
    }

    // ---- the context window ----
    {
        arm(hello, "abcdefghijklmnopqrstuvwxyz");
        const int n = fake.gen_from;
        fake.ctx = n + 6;
        const Value j = json(chat(hello).body);
        CHECK(at(j, {"usage", "completion_tokens"}).i() == 6 && str_at(j, {"choices", "0", "finish_reason"}) == "length", "context: generation stops with `length` when the window is full");
        fake.ctx = n;
        const Resp r = chat(hello);
        CHECK(r.status == 400 && str_at(json(r.body), {"error", "message"}).find("context") != std::string::npos, "context: a prompt that does not fit is a 400");
        fake.ctx = 0;
    }

    // ---- the tape: prefix reuse ----
    {
        const int resets0 = fake.n_reset.load();
        arm(tool_body, tool_text);
        const Value a = json(chat(tool_body).body);  // fresh conversation: the tape was another prompt, so a reset
        CHECK(at(a, {"usage", "prompt_tokens_details", "cached_tokens"}).i() == 0 && fake.n_reset.load() == resets0 + 1, "tape: a new conversation resets and reads the whole prompt");
        const size_t tape_a = svc.tape_size();
        CHECK(tape_a == size_t(at(a, {"usage", "prompt_tokens"}).i()) + fake.script.size(), "tape: after the turn it holds prompt + generated tokens");

        // The harness's next request: the same conversation + the assistant message it received + a tool result.
        Value msgs = Value::array();
        for (const Value& m : tool_body.find("messages")->list()) msgs.push(m);
        msgs.push(*at(a, {"choices", "0"}).find("message"));
        Value tool = Value::object();
        tool.set("role", Value::string("tool"));
        tool.set("tool_call_id", at(a, {"choices", "0", "message", "tool_calls", "0", "id"}));
        tool.set("content", Value::string("sunny, 21 C"));
        msgs.push(tool);
        Value b = tool_body.clone();
        b.set("messages", msgs);
        const std::string b_tail = "Fine.\n</think>\n\nIt is sunny in Paris.";
        arm(b, b_tail);
        const int fwd0 = fake.n_forward.load(), nolog0 = fake.n_nologits.load(), greedy0 = fake.n_greedy.load();
        const Value bj = json(chat(b).body);
        const int b_prompt = fake.gen_from;
        CHECK(str_at(bj, {"choices", "0", "message", "content"}) == "It is sunny in Paris.", "tape: the follow-up request is answered");
        CHECK(at(bj, {"usage", "prompt_tokens_details", "cached_tokens"}).i() == int64_t(tape_a), "tape: the follow-up reuses every token of the previous turn (cached_tokens == tape size)");
        CHECK(fake.n_reset.load() == resets0 + 1, "tape: no reset for the follow-up");
        CHECK(fake.n_forward.load() - fwd0 == (b_prompt - int(tape_a)) + int(fake.script.size()), "tape: only the new prompt tokens and the generated ones went through the backend");
        CHECK(at(bj, {"timings", "cache_n"}).i() == int64_t(tape_a) && at(bj, {"timings", "prompt_n"}).i() == b_prompt - int64_t(tape_a), "tape: timings split cached and processed prompt tokens");
        CHECK(fake.n_nologits.load() - nolog0 == (b_prompt - int(tape_a)) - 1 && fake.n_greedy.load() - greedy0 == int(fake.script.size()) + 1,
              "tape: the new prompt tokens but the last skip the head, and the last one plus every generated token ask for the greedy token");
        const size_t tape_b = svc.tape_size();

        // A new user turn: the template drops the earlier think blocks, so the prompt diverges from the tape.
        Value msgs2 = Value::array();
        for (const Value& m : msgs.list()) msgs2.push(m);
        msgs2.push(*at(bj, {"choices", "0"}).find("message"));
        msgs2.push(json(R"({"role":"user","content":"Thanks!"})"));
        Value c = tool_body.clone();
        c.set("messages", msgs2);
        // With preserve_thinking the template keeps them, and the tape is a prefix again.
        Value cp = c.clone();
        cp.set("chat_template_kwargs", json(R"({"preserve_thinking":true})"));
        arm(cp, "Welcome.\n</think>\n\nYou are welcome.");
        const Value cpj = json(chat(cp).body);
        CHECK(at(cpj, {"usage", "prompt_tokens_details", "cached_tokens"}).i() == int64_t(tape_b) && fake.n_reset.load() == resets0 + 1,
              "tape: with preserve_thinking a new user turn still continues from the tape");
        // Without it the earlier assistant turns lose their think block: a mismatch, so reset and re-read.
        arm(c, "Welcome.\n</think>\n\nYou are welcome.");
        const Value cj = json(chat(c).body);
        CHECK(at(cj, {"usage", "prompt_tokens_details", "cached_tokens"}).i() == 0 && fake.n_reset.load() == resets0 + 2 &&
                  str_at(cj, {"choices", "0", "message", "content"}) == "You are welcome.",
              "tape: a prompt that diverges from the tape resets the backend and re-reads it (and still answers)");
        // The same request again: the prompt is shorter than the tape, so it is re-read, never rewound.
        arm(c, "Welcome.\n</think>\n\nYou are welcome.");
        const Value cj2 = json(chat(c).body);
        CHECK(at(cj2, {"usage", "prompt_tokens_details", "cached_tokens"}).i() == 0 && fake.n_reset.load() == resets0 + 3 && str_at(cj2, {"choices", "0", "message", "content"}) == "You are welcome.",
              "tape: a repeated request re-reads its prompt (nothing is rewound)");
        CHECK(fake.n_bad_pos.load() == 0, "tape: through all of it the backend saw consecutive positions from 0");
    }

    // ---- the backend hooks: no logits for prompt tokens, the greedy token, batched prompt reads ----
    {
        struct Counters { int fwd, nolog, greedy, prefill, resets; };
        auto snap = [&] { return Counters{fake.n_forward.load(), fake.n_nologits.load(), fake.n_greedy.load(), fake.n_prefill.load(), fake.n_reset.load()}; };
        const std::string text = "Hook text, checked.";
        const int n_gen = int(enc(text).size()) + 1;  // the end token is sampled too

        // Temperature 0: every prompt token but the last skips the head; the last and each generated token come back greedy.
        Value g = json(R"({"messages":[{"role":"user","content":"Greedy hooks."}],"max_tokens":200,"temperature":0,
                           "chat_template_kwargs":{"enable_thinking":false}})");
        arm(g, text);
        const int n_prompt = fake.gen_from;
        fake.chunk = 1;
        Counters c0 = snap();
        const Value gj = json(chat(g).body);
        Counters c1 = snap();
        CHECK(str_at(gj, {"choices", "0", "message", "content"}) == text && at(gj, {"usage", "completion_tokens"}).i() == n_gen,
              "hooks: temperature 0 through forward_greedy gives the scripted text");
        CHECK(c1.nolog - c0.nolog == n_prompt - 1, "hooks: a fresh prompt reads all but its last token without logits");
        CHECK(c1.greedy - c0.greedy == n_gen, "hooks: the last prompt token and every generated token but the last come back through forward_greedy");
        CHECK(c1.fwd - c0.fwd == n_prompt + n_gen - 1 && c1.resets - c0.resets == 1 && fake.n_bad_pos.load() == 0,
              "hooks: the same forward passes at the same consecutive positions as before");
        CHECK(c1.prefill - c0.prefill == n_prompt - 1, "hooks: with a chunk of 1 the server hands the backend the prompt one token at a time");

        // Temperature > 0: the sampler needs the logits, so the greedy hook stays unused (a tiny temperature keeps the fake's one-hot winner).
        Value t = g.clone();
        t.set("temperature", Value::number(0.1));
        t.set("seed", Value::integer(7));
        t.set("messages", json(R"([{"role":"user","content":"Sampled hooks."}])"));
        arm(t, text);
        const int n_prompt_t = fake.gen_from;
        c0 = snap();
        const Value tj = json(chat(t).body);
        c1 = snap();
        CHECK(str_at(tj, {"choices", "0", "message", "content"}) == text && c1.greedy == c0.greedy, "hooks: temperature above 0 samples the logits and never calls forward_greedy");
        CHECK(c1.nolog - c0.nolog == n_prompt_t - 1 && c1.fwd - c0.fwd == n_prompt_t + n_gen - 1, "hooks: the prompt still skips the head when sampling");

        // A batched prompt read: chunks of at most 7, consecutive positions, the last token left for forward.
        fake.chunk = 7;
        Value k = g.clone();
        k.set("messages", json(R"([{"role":"user","content":"Chunked prompt reading, a longer one so that several chunks are needed."}])"));
        arm(k, text);
        const int n_prompt_k = fake.gen_from;
        c0 = snap();
        const Value kj = json(chat(k).body);
        c1 = snap();
        CHECK(str_at(kj, {"choices", "0", "message", "content"}) == text, "hooks: a batched prompt read gives the same answer");
        CHECK(c1.prefill - c0.prefill == (n_prompt_k - 1 + 6) / 7 && fake.n_bad_chunk.load() == 0 && c1.nolog - c0.nolog == n_prompt_k - 1,
              "hooks: prefill is called with at most prefill_chunk() tokens, ceil((prompt - 1) / chunk) times");
        CHECK(fake.n_bad_pos.load() == 0, "hooks: batched reads keep the positions consecutive");
        fake.chunk = 1;
    }

    // ---- hanging up ----
    {
        fake.sleep_us = 2000;
        Value body = json(R"({"messages":[{"role":"user","content":"Go on."}],"max_tokens":400,"temperature":0,"stream":true,
                              "chat_template_kwargs":{"enable_thinking":false}})");
        arm(body, std::string(300, 'a'));
        const uint64_t aborted0 = svc.stats().aborted.load();
        {
            Sock s;
            s.dial(port);
            const std::string b = json_str(body);
            s.send_all("POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: " + std::to_string(b.size()) + "\r\n\r\n" + b);
            Resp r;
            read_head(s, &r, 5000);
            std::string chunk;
            bool got_content = false;
            for (int i = 0; i < 100 && !got_content; ++i) {
                if (!s.read_chunk(&chunk, 5000)) break;
                got_content = chunk.find("\"content\":\"a") != std::string::npos;
            }
            CHECK(r.status == 200 && got_content, "hangup: a streaming client receives content");
            linger lg{1, 0};  // close with a reset, as a crashed client would
            ::setsockopt(s.fd, SOL_SOCKET, SO_LINGER, &lg, sizeof lg);
        }
        CHECK(wait_until([&] { return svc.stats().aborted.load() == aborted0 + 1; }, 3000), "hangup: the server noticed the client was gone and counted the request as aborted");
        CHECK(svc.stats().last_completion_tokens.load() < 300, "hangup: generation stopped early (not all 300 tokens)");
        CHECK(request(port, "GET", "/health").status == 200, "hangup: the server answers the next request");
        // Non-streaming: the client leaves while the model is still reading the prompt. 5 ms a token: the
        // 300 tokens would take 1.5 s, so the abort has to be noticed in a fraction of that.
        {
            fake.sleep_us = 5000;
            Value b2 = hello.clone();
            b2.set("max_tokens", Value::integer(400));
            arm(b2, std::string(300, 'b'));
            const double t0 = now_ms();
            {
                Sock s;
                s.dial(port);
                const std::string b = json_str(b2);
                s.send_all("POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: " + std::to_string(b.size()) + "\r\n\r\n" + b);
                std::this_thread::sleep_for(std::chrono::milliseconds(60));
            }
            CHECK(wait_until([&] { return svc.stats().aborted.load() == aborted0 + 2; }, 3000) && now_ms() - t0 < 1000,
                  "hangup: a non-streaming client that left is noticed within a few tokens, not after the whole completion");
        }
        fake.sleep_us = 0;
        // And a normal request still works with the tape in whatever state the aborts left it.
        arm(hello, "Back again.");
        const Value j = json(chat(hello).body);
        CHECK(str_at(j, {"choices", "0", "message", "content"}) == "Back again." && fake.n_bad_pos.load() == 0, "hangup: the next completion is correct and positions stayed consecutive");
    }

    // ---- hanging clients and bad framing ----
    {
        const double t0 = now_ms();
        Sock silent;
        silent.dial(port);  // connects and says nothing
        const Resp ok = request(port, "GET", "/health");
        const double dt = now_ms() - t0;
        CHECK(ok.status == 200 && dt < 3000, "framing: a silent connection is dropped after the read deadline and does not block the next request");
        Resp sr;
        read_head(silent, &sr, 2000);
        CHECK(sr.status == 408, "framing: the silent client is told 408");
        {
            Sock s;
            s.dial(port);
            s.send_all("POST /v1/chat/completions HTTP/1.1\r\nContent-Len");  // half a header, then gone
        }
        {
            Sock s;
            s.dial(port);
            s.send_all("POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 100\r\n\r\n{\"mess");  // short body, then gone
        }
        CHECK(request(port, "GET", "/health").status == 200, "framing: clients that vanish mid-request do not hurt the server");
        {
            Sock s;
            s.dial(port);
            s.send_all("FOO\r\n\r\n");
            CHECK(read_response(s).status == 400, "framing: garbage is a 400");
        }
        {
            Sock s;
            s.dial(port);
            s.send_all("POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 99999999\r\n\r\n");
            CHECK(read_response(s).status == 413, "framing: a body over the limit is a 413 without reading it");
        }
        {
            Sock s;
            s.dial(port);
            s.send_all("POST /v1/chat/completions HTTP/1.1\r\n\r\n");
            CHECK(read_response(s).status == 411, "framing: a POST without a length is a 411");
        }
        // A chunked request body, with Expect: 100-continue.
        {
            arm(hello, "chunky");
            const std::string b = json_str(hello);
            const size_t half = b.size() / 2;
            char h1[32], h2[32];
            std::snprintf(h1, sizeof h1, "%zx", half);
            std::snprintf(h2, sizeof h2, "%zx", b.size() - half);
            Sock s;
            s.dial(port);
            s.send_all("POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\nExpect: 100-continue\r\n\r\n");
            std::string cont;
            const bool got100 = s.read_until("\r\n\r\n", &cont, 3000) && cont.rfind("HTTP/1.1 100", 0) == 0;
            s.send_all(std::string(h1) + "\r\n" + b.substr(0, half) + "\r\n" + h2 + "\r\n" + b.substr(half) + "\r\n0\r\n\r\n");
            const Resp r = read_response(s);
            CHECK(got100 && r.status == 200 && str_at(json(r.body), {"choices", "0", "message", "content"}) == "chunky", "framing: Expect: 100-continue and a chunked request body");
        }
    }

    // ---- other requests are answered while one runs; the rest wait their turn ----
    {
        fake.sleep_us = 3000;
        Value body = json(R"({"messages":[{"role":"user","content":"Long one."}],"max_tokens":400,"temperature":0,"stream":true,
                              "chat_template_kwargs":{"enable_thinking":false}})");
        arm(body, std::string(150, 'c'));
        Sock s1;
        s1.dial(port);
        const std::string b = json_str(body);
        s1.send_all("POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: " + std::to_string(b.size()) + "\r\n\r\n" + b);
        Resp r1;
        read_head(s1, &r1, 5000);
        std::string chunk;
        s1.read_chunk(&chunk, 5000);  // the role chunk: the request is being served

        Sock s3;  // a second completion request: it must wait for the first
        s3.dial(port);
        s3.send_all("POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 8\r\n\r\n{not js}");
        std::this_thread::sleep_for(std::chrono::milliseconds(20));

        const double t0 = now_ms();
        const Resp h = request(port, "GET", "/health");
        const double t_health = now_ms() - t0;
        double t_done = 0;
        std::string rest;
        while (s1.read_chunk(&chunk, 8000)) {
            rest += chunk;
            if (rest.find("[DONE]") != std::string::npos) t_done = now_ms() - t0;
        }
        const Resp r3 = read_response(s3);
        CHECK(h.status == 200 && t_health < 300, "busy: /health is answered within 300 ms while a completion is running");
        CHECK(t_done > t_health + 100, "busy: ... and well before that completion finishes");
        CHECK(rest.find("[DONE]") != std::string::npos, "busy: the running completion is not disturbed");
        CHECK(r3.status == 400, "busy: a second completion request waits and is then served");
        fake.sleep_us = 0;
    }

    stop = true;
    th.join();

    // ---- the sampler ----
    {
        const int n = 1000;
        std::vector<float> lg(size_t(n), 0.f);
        lg[17] = 5.f;
        lg[900] = 5.f;
        serve::SamplerParams g;
        g.temperature = 0;
        serve::Sampler sg(g, 1);
        CHECK(sg.sample(lg.data(), n) == 17, "sampler: greedy takes the argmax, the lowest id on a tie");
        serve::SamplerParams p;
        p.temperature = 0.8;
        p.has_seed = true;
        p.seed = 42;
        serve::Sampler a(p, 0), b(p, 99), c(p, 0);
        p.seed = 43;
        serve::Sampler d(p, 0);
        std::vector<int32_t> sa, sb, sd;
        std::vector<float> flat(static_cast<size_t>(n), 0.f);
        for (int i = 0; i < n; ++i) flat[size_t(i)] = float((i * 37) % 11) * 0.3f;
        for (int i = 0; i < 40; ++i) {
            sa.push_back(a.sample(flat.data(), n));
            sb.push_back(b.sample(flat.data(), n));
            sd.push_back(d.sample(flat.data(), n));
        }
        CHECK(sa == sb, "sampler: the same seed gives the same draws whatever the fallback seed");
        CHECK(sa != sd, "sampler: another seed gives other draws");
        serve::SamplerParams k1;
        k1.temperature = 1.0;
        k1.top_k = 1;
        serve::Sampler sk(k1, 5);
        bool all17 = true;
        for (int i = 0; i < 50; ++i) all17 = all17 && sk.sample(lg.data(), n) == 17;
        CHECK(all17, "sampler: top_k 1 is greedy");
        serve::SamplerParams tp;
        tp.temperature = 1.0;
        tp.top_p = 0.01;
        serve::Sampler st(tp, 5);
        bool top = true;
        for (int i = 0; i < 50; ++i) top = top && st.sample(lg.data(), n) == 17;
        CHECK(top, "sampler: a tiny top_p keeps only the top token");
        std::vector<float> three = {0.f, -10.f, -10.f, 0.1f};
        serve::SamplerParams mp;
        mp.temperature = 1.0;
        mp.min_p = 0.5;
        serve::Sampler sm(mp, 6);
        bool only03 = true;
        for (int i = 0; i < 300; ++i) {
            const int t = sm.sample(three.data(), 4);
            only03 = only03 && (t == 0 || t == 3);
        }
        CHECK(only03, "sampler: min_p drops tokens far below the best one");
        std::vector<float> two = {std::log(3.f), 0.f};
        serve::SamplerParams tt;
        tt.temperature = 1.0;
        serve::Sampler su(tt, 7);
        int zero = 0;
        for (int i = 0; i < 6000; ++i) zero += su.sample(two.data(), 2) == 0;
        CHECK(zero > 4350 && zero < 4650, "sampler: a plain draw follows the softmax (3:1 -> about 75%)");
        serve::SamplerParams tk;
        tk.temperature = 1.0;
        tk.top_k = 2;
        serve::Sampler sk2(tk, 8);
        std::vector<float> four = {3.f, 2.f, 1.f, 0.f};
        bool in2 = true;
        for (int i = 0; i < 500; ++i) {
            const int t = sk2.sample(four.data(), 4);
            in2 = in2 && t <= 1;
        }
        CHECK(in2, "sampler: top_k 2 never leaves the two best");
    }

    // ---- the tape with checkpoints and sessions: a backend that snapshots its state ----
    {
        SnapFake sf(int(mv.v.tokens.size()), mv.im_end);
        sf.chunk = 7;
        serve::Config c2 = cfg;
        c2.checkpoints = 4;
        c2.checkpoint_gap = 1;  // every message boundary
        c2.slots = 1;        // one conversation at a time: the single-home behaviour; routing is tested further down
        serve::Service s2(sf, tokenizer, *renderer, c2);
        CHECK(s2.checkpoint_slots() == 4 && sf.n_slot_calls == 1 && sf.slots() == 1, "tape+: the service asks the backend for 4 snapshot slots and keeps one home when --sessions is 1");
        auto server2 = limbo::http::Server::listen("127.0.0.1", 0, &err, so);
        CHECK(server2 != nullptr, "tape+: a second server binds");
        if (server2) {
            const int port2 = server2->port();
            std::atomic<bool> stop2{false};
            std::thread th2([&] { server2->run([&](const limbo::http::Request& rq, limbo::http::Conn& c) { s2.handle(rq, c); }, stop2); });
            auto toks = [&](const Value& body) {
                serve::ChatRequest cr;
                serve::Prompt p;
                std::string e;
                if (!s2.parse_request(body, &cr, &e) || !s2.build_prompt(cr, &p, &e)) {
                    std::printf("FAIL tape+ prompt: %s\n", e.c_str());
                    ++limbo::test::failures();
                }
                return p.tokens;
            };
            auto last_boundary = [&](const std::vector<int32_t>& t) {
                size_t b = 0;
                for (size_t i = 1; i < t.size(); ++i)
                    if (t[i] == mv.im_start) b = i;
                return int(b);
            };
            auto lcp = [](const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
                size_t i = 0;
                while (i < a.size() && i < b.size() && a[i] == b[i]) ++i;
                return int(i);
            };
            // One request: arms the fake to say `text` after the prompt, returns the answer.
            auto ask = [&](const Value& body, const std::string& text, const std::string& hdr = "") {
                sf.script = enc(text);
                sf.gen_from = int(toks(body).size());
                return request(port2, "POST", "/v1/chat/completions", json_str(body), hdr);
            };
            auto cached_of = [&](const Resp& r) { return int(at(json(r.body), {"usage", "prompt_tokens_details", "cached_tokens"}).i()); };
            // The chain a cold read of `prompt` + the scripted text would leave.
            auto cold_chain = [&](const Value& body) { return SnapFake::chain(toks(body), sf.script); };

            // turn 1: a user turn with tools
            const Resp r1 = ask(tool_body, tool_text);
            const Value j1 = json(r1.body);
            const std::vector<int32_t> p1 = toks(tool_body);
            const int head1 = last_boundary(p1);
            CHECK(cached_of(r1) == 0 && sf.home[0].h == cold_chain(tool_body) && sf.n_bad_pos.load() == 0, "tape+: turn 1 reads the prompt cold and the backend's state is the cold chain");
            CHECK(s2.checkpoints_held() >= 2 && sf.n_snapshot >= 2, "tape+: turn 1 took checkpoints at the message boundaries");
            CHECK(sf.pieces.size() > 3, "tape+: the prompt was read in pieces cut at the boundaries");

            // turn 2: the tool result; the tape is a prefix, so nothing is restored
            Value msgs = Value::array();
            for (const Value& m : tool_body.find("messages")->list()) msgs.push(m);
            msgs.push(*at(j1, {"choices", "0"}).find("message"));
            Value tool = Value::object();
            tool.set("role", Value::string("tool"));
            tool.set("tool_call_id", at(j1, {"choices", "0", "message", "tool_calls", "0", "id"}));
            tool.set("content", Value::string("sunny, 21 C"));
            msgs.push(tool);
            Value b2 = tool_body.clone();
            b2.set("messages", msgs);
            const size_t tape1 = s2.tape_size();
            const Resp r2 = ask(b2, "Fine.\n</think>\n\nIt is sunny in Paris.");
            const Value j2 = json(r2.body);
            CHECK(cached_of(r2) == int(tape1) && s2.stats().restores.load() == 0 && sf.n_restore == 0, "tape+: a tool-result turn continues from the tape: all of it cached, nothing restored");
            CHECK(sf.home[0].h == cold_chain(b2), "tape+: ... and the state equals a cold read of that prompt");
            const std::vector<int32_t> tape2 = [&] {
                std::vector<int32_t> t = toks(b2);
                t.insert(t.end(), sf.script.begin(), sf.script.end());
                return t;
            }();

            // turn 3: a new user turn; the template drops the earlier think blocks, so the prompt diverges inside the tape
            Value msgs3 = Value::array();
            for (const Value& m : msgs.list()) msgs3.push(m);
            msgs3.push(*at(j2, {"choices", "0"}).find("message"));
            msgs3.push(json(R"({"role":"user","content":"Thanks!"})"));
            Value b3 = tool_body.clone();
            b3.set("messages", msgs3);
            const int div3 = lcp(tape2, toks(b3));
            CHECK(div3 > head1 && size_t(div3) < tape2.size(), "tape+: turn 3 diverges from the tape after turn 1's generation prompt and before its end");
            const int fwd0 = sf.n_forward.load(), resets0 = sf.n_reset.load();
            const Resp r3 = ask(b3, "Welcome.\n</think>\n\nYou are welcome.");
            const int n3 = int(toks(b3).size());
            CHECK(cached_of(r3) == head1 && s2.stats().last_restored_from.load() == head1 && sf.n_restore == 1,
                  "tape+: turn 3 restores the checkpoint at turn 1's generation prompt (the newest at or before the divergence) instead of resetting");
            CHECK(sf.n_reset.load() == resets0 && sf.n_forward.load() - fwd0 == (n3 - head1) + int(sf.script.size()),
                  "tape+: ... and only the prompt from there plus the generated tokens went through the backend");
            CHECK(sf.home[0].h == cold_chain(b3) && sf.n_bad_pos.load() == 0, "tape+: ... the state after turn 3 equals a cold read of its prompt, positions consecutive");
            CHECK(str_at(json(r3.body), {"choices", "0", "message", "content"}) == "You are welcome.", "tape+: ... and the answer is right");

            // a repeated request: the prompt is shorter than the tape; restore its generation prompt, read a few tokens
            const int head3 = last_boundary(toks(b3));
            const Resp r3b = ask(b3, "Welcome.\n</think>\n\nYou are welcome.");
            CHECK(cached_of(r3b) == head3 && sf.home[0].h == cold_chain(b3), "tape+: a repeated request restores its own generation prompt and reads only what follows it");

            // cache_prompt false: a full cold read
            const int resets1 = sf.n_reset.load();
            Value nc = b3.clone();
            nc.set("cache_prompt", Value::boolean(false));
            const Resp r3c = ask(nc, "Welcome.\n</think>\n\nYou are welcome.");
            CHECK(cached_of(r3c) == 0 && sf.n_reset.load() == resets1 + 1 && sf.home[0].h == cold_chain(b3), "tape+: cache_prompt false resets and reads everything");

            // a change before the first checkpoint: nothing to restore, a cold read
            Value edited = b3.clone();
            const std::vector<Value>& old_msgs = b3.find("messages")->list();
            Value first = old_msgs[0].clone();
            first.set("content", Value::string("Weather in Rome?"));
            Value em2 = Value::array();
            em2.push(first);
            for (size_t i = 1; i < old_msgs.size(); ++i) em2.push(old_msgs[i]);
            edited.set("messages", em2);
            const Resp r4 = ask(edited, "Ok.\n</think>\n\nDone.");
            int root = 0;
            {
                const std::vector<int32_t> pe = toks(edited);
                for (size_t i = 1; i < pe.size() && !root; ++i)
                    if (pe[i] == mv.im_start) root = int(i);
            }
            CHECK(root > 0 && cached_of(r4) == root && sf.home[0].h == cold_chain(edited),
                  "tape+: an edit in the first message restores the root checkpoint (the end of the system prompt and tools) and reads from there");
            // an edit inside the tools block comes before every checkpoint: a cold read
            Value retooled = edited.clone();
            Value tl = json(kWeatherTools);
            Value fn = tl.list()[0].clone();
            Value fn2 = fn.find("function")->clone();
            fn2.set("description", Value::string("Weather for a city, said differently."));
            fn.set("function", fn2);
            Value tl2 = Value::array();
            tl2.push(fn);
            retooled.set("tools", tl2);
            const Resp r4b = ask(retooled, "Ok.\n</think>\n\nDone.");
            CHECK(cached_of(r4b) == 0 && sf.home[0].h == cold_chain(retooled), "tape+: a divergence before every checkpoint is a cold read");

            // a failing restore falls back to a reset
            {
                const Resp rr = ask(b3, "Welcome.\n</think>\n\nYou are welcome.");  // back to b3: a cold read (the tape is `edited`)
                (void)rr;
                const Resp rc = ask(b3, "Welcome.\n</think>\n\nYou are welcome.");  // restores its generation prompt
                CHECK(cached_of(rc) == head3, "tape+: (setup) the repeat restores");
                sf.fail_restore = true;
                const Resp rf = ask(b3, "Welcome.\n</think>\n\nYou are welcome.");
                sf.fail_restore = false;
                CHECK(cached_of(rf) == 0 && sf.home[0].h == cold_chain(b3) && str_at(json(rf.body), {"choices", "0", "message", "content"}) == "You are welcome.",
                      "tape+: a restore the backend refuses falls back to a reset and a full read, and the answer is right");
            }
            CHECK(s2.checkpoints_held() <= 4 && s2.checkpoints_held() >= 1, "tape+: never more checkpoints than slots");

            stop2 = true;
            th2.join();
        }
    }

    // ---- preserve: the thinking kept, so a new turn continues the tape -- also from a client that drops reasoning_content ----
    {
        SnapFake sf(int(mv.v.tokens.size()), mv.im_end);
        sf.chunk = 7;
        serve::Config cp = cfg;
        cp.preserve_thinking = true;
        cp.checkpoints = 4;
        cp.checkpoint_gap = 1;
        cp.slots = 1;
        serve::Service sp(sf, tokenizer, *renderer, cp);
        auto serverp = limbo::http::Server::listen("127.0.0.1", 0, &err, so);
        CHECK(serverp != nullptr, "preserve: a server binds");
        if (serverp) {
            const int portp = serverp->port();
            std::atomic<bool> stopp{false};
            std::thread thp([&] { serverp->run([&](const limbo::http::Request& rq, limbo::http::Conn& c) { sp.handle(rq, c); }, stopp); });
            auto toks = [&](const Value& body) {
                serve::ChatRequest cr;
                serve::Prompt p;
                std::string e;
                if (!sp.parse_request(body, &cr, &e) || !sp.build_prompt(cr, &p, &e)) {
                    std::printf("FAIL preserve prompt: %s\n", e.c_str());
                    ++limbo::test::failures();
                }
                return p.tokens;
            };
            auto ask = [&](const Value& body, const std::string& text) {
                sf.script = enc(text);
                sf.gen_from = int(toks(body).size());
                return request(portp, "POST", "/v1/chat/completions", json_str(body));
            };
            auto cached_of = [&](const Resp& r) { return int(at(json(r.body), {"usage", "prompt_tokens_details", "cached_tokens"}).i()); };
            // what a client that keeps only content and tool_calls sends back of an answer
            auto bare = [](const Value& m) {
                Value o = Value::object();
                for (const auto& kv : m.dict())
                    if (kv.first != "reasoning_content") o.set(kv.first, kv.second);
                return o;
            };
            const Resp r1 = ask(tool_body, tool_text);
            const Value j1 = json(r1.body);
            CHECK(!str_at(j1, {"choices", "0", "message", "reasoning_content"}).empty(), "preserve: (setup) turn 1 answers with its thinking");
            Value msgs = Value::array();
            for (const Value& m : tool_body.find("messages")->list()) msgs.push(m);
            msgs.push(bare(*at(j1, {"choices", "0"}).find("message")));
            Value tool = Value::object();
            tool.set("role", Value::string("tool"));
            tool.set("tool_call_id", at(j1, {"choices", "0", "message", "tool_calls", "0", "id"}));
            tool.set("content", Value::string("sunny, 21 C"));
            msgs.push(tool);
            Value b2 = tool_body.clone();
            b2.set("messages", msgs);
            const size_t tape1 = sp.tape_size();
            const Resp r2 = ask(b2, "Fine.\n</think>\n\nIt is sunny in Paris.");
            const Value j2 = json(r2.body);
            CHECK(cached_of(r2) == int(tape1) && sf.n_restore == 0,
                  "preserve: the tool-result turn continues the tape although the client dropped turn 1's reasoning_content");
            const size_t tape2 = sp.tape_size();
            Value msgs3 = Value::array();
            for (const Value& m : msgs.list()) msgs3.push(m);
            msgs3.push(bare(*at(j2, {"choices", "0"}).find("message")));
            msgs3.push(json(R"({"role":"user","content":"Thanks!"})"));
            Value b3 = tool_body.clone();
            b3.set("messages", msgs3);
            const Resp r3 = ask(b3, "Glad.\n</think>\n\nYou are welcome.");
            CHECK(cached_of(r3) == int(tape2) && sf.n_restore == 0 && sp.stats().restores.load() == 0,
                  "preserve: a new user turn continues the whole tape (both answers' thinking re-rendered from the server's memory): nothing restored, nothing re-read");
            CHECK(sf.home[0].h == SnapFake::chain(toks(b3), sf.script) && sf.n_bad_pos.load() == 0,
                  "preserve: ... and the state equals a cold read of that prompt");
            CHECK(str_at(json(r3.body), {"choices", "0", "message", "content"}) == "You are welcome.", "preserve: ... and the answer is right");
            {   // a client that cut a reasoning on purpose sends reasoning_content "": it stays cut
                Value cut = Value::array();
                for (const Value& m : msgs3.list()) {
                    if (m.is_dict() && m.get_str("role") == "assistant") {
                        Value o = Value::object();
                        for (const auto& kv : m.dict()) o.set(kv.first, kv.second);
                        o.set("reasoning_content", Value::string(""));
                        cut.push(o);
                    } else cut.push(m);
                }
                Value bc = tool_body.clone();
                bc.set("messages", cut);
                CHECK(toks(bc).size() + 4 < toks(b3).size(), "preserve: reasoning_content \"\" (cut by the client) is not refilled: the prompt renders without it");
            }
            stopp = true;
            thp.join();
        }
    }

    // ---- homes: two conversations, no header, routed by prefix ----
    {
        // The start-up decision, with its numbers.
        {
            SnapFake a(int(mv.v.tokens.size()), mv.im_end);
            serve::Config c = cfg;  // sessions 2, reserve 768 MiB
            serve::Service sa(a, tokenizer, *renderer, c);
            CHECK(sa.slot_count() == 2 && sa.startup_note().find("slots: 2 of 2 asked") != std::string::npos &&
                      sa.startup_note().find("100.0 MiB per slot") != std::string::npos && sa.startup_note().find("768 MiB reserve") != std::string::npos,
                  "homes: by default two homes are made when they fit, and the note carries the numbers");
            CHECK(sa.startup_note().find("checkpoints: 4 of 8 asked") != std::string::npos, "homes: the checkpoint slots are reported with their numbers");

            SnapFake t(int(mv.v.tokens.size()), mv.im_end);
            t.free_b = (768ull + 50) << 20;  // a home needs 100 MiB and the reserve: 818 MiB free is not enough
            serve::Service st(t, tokenizer, *renderer, c);
            CHECK(st.slot_count() == 1 && t.slots() == 1 && st.startup_note().find("running with 1: slot 2 needs 100.0 MiB plus 2 checkpoint slots per slot") != std::string::npos &&
                      st.startup_note().find("768 MiB reserve, 818 MiB were free") != std::string::npos,
                  "homes: when a second one does not fit the server runs with one and says why, in numbers");

            SnapFake r(int(mv.v.tokens.size()), mv.im_end);
            r.free_b = (768ull + 50) << 20;
            serve::Config c0 = cfg;
            c0.vram_reserve_mib = 0;
            serve::Service sr(r, tokenizer, *renderer, c0);
            CHECK(sr.slot_count() == 2, "homes: --vram-reserve-mib 0 lets the same card hold two");

            SnapFake o(int(mv.v.tokens.size()), mv.im_end);
            serve::Config c1 = cfg;
            c1.slots = 1;
            serve::Service so1(o, tokenizer, *renderer, c1);
            CHECK(so1.slot_count() == 1 && so1.startup_note().find("slots: 1 (--slots 1") != std::string::npos, "homes: --sessions 1 disables the second home");

            SnapFake m(int(mv.v.tokens.size()), mv.im_end);
            serve::Config c3 = cfg;
            c3.slots = 3;
            m.n_slots_max = 4;
            serve::Service sm(m, tokenizer, *renderer, c3);
            CHECK(sm.slot_count() == 3, "homes: --sessions 3 raises it");

            Fake plain(int(mv.v.tokens.size()), mv.im_end);
            serve::Service sp(plain, tokenizer, *renderer, cfg);
            CHECK(sp.slot_count() == 1 && sp.startup_note().empty(), "homes: a backend without homes or snapshots runs one, silently");
        }

        SnapFake sf(int(mv.v.tokens.size()), mv.im_end);
        sf.chunk = 7;
        serve::Config c3 = cfg;
        c3.checkpoints = 6;
        c3.checkpoint_gap = 1;
        serve::Service s3(sf, tokenizer, *renderer, c3);
        auto server3 = limbo::http::Server::listen("127.0.0.1", 0, &err, so);
        CHECK(server3 != nullptr && s3.slot_count() == 2, "homes: a server with two homes binds");
        if (server3 && s3.slot_count() == 2) {
            const int port3 = server3->port();
            std::atomic<bool> stop3{false};
            std::thread th3([&] { server3->run([&](const limbo::http::Request& rq, limbo::http::Conn& c) { s3.handle(rq, c); }, stop3); });
            auto toks = [&](const Value& body) {
                serve::ChatRequest cr;
                serve::Prompt p;
                std::string e;
                if (!s3.parse_request(body, &cr, &e) || !s3.build_prompt(cr, &p, &e)) {
                    std::printf("FAIL homes prompt: %s\n", e.c_str());
                    ++limbo::test::failures();
                }
                return p.tokens;
            };
            auto boundary_after = [&](const std::vector<int32_t>& t, int nth) {  // index of the nth `<|im_start|>` after position 0 (1-based)
                int k = 0;
                for (size_t i = 1; i < t.size(); ++i)
                    if (t[i] == mv.im_start && ++k == nth) return int(i);
                return 0;
            };
            auto last_boundary = [&](const std::vector<int32_t>& t) {
                int b = 0;
                for (size_t i = 1; i < t.size(); ++i)
                    if (t[i] == mv.im_start) b = int(i);
                return b;
            };
            auto ask = [&](const Value& body, const std::string& text, const std::string& hdr = "") {
                sf.script = enc(text);
                sf.gen_from = int(toks(body).size());
                return request(port3, "POST", "/v1/chat/completions", json_str(body), hdr);
            };
            auto cached_of = [&](const Resp& r) { return int(at(json(r.body), {"usage", "prompt_tokens_details", "cached_tokens"}).i()); };
            auto home_of = [&](const Resp& r) { return int(at(json(r.body), {"timings", "slot"}).i()); };
            auto cold = [&](const Value& body) { return SnapFake::chain(toks(body), sf.script); };
            auto chat_body = [&](const char* system, const char* user) {
                Value b = json(R"({"max_tokens":60,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}})");
                Value m = Value::array();
                Value sm = Value::object();
                sm.set("role", Value::string("system"));
                sm.set("content", Value::string(system));
                m.push(sm);
                Value um = Value::object();
                um.set("role", Value::string("user"));
                um.set("content", Value::string(user));
                m.push(um);
                b.set("messages", m);
                return b;
            };
            auto follow = [&](const Value& body, const Resp& r, const char* next_user) {  // the conversation so far + the reply + a new user turn
                Value m = Value::array();
                for (const Value& x : body.find("messages")->list()) m.push(x);
                m.push(*at(json(r.body), {"choices", "0"}).find("message"));
                Value um = Value::object();
                um.set("role", Value::string("user"));
                um.set("content", Value::string(next_user));
                m.push(um);
                Value b = body.clone();
                b.set("messages", m);
                return b;
            };
            const char* kSysY = "You write very short titles for text. Reply with the title only, no quotes.";
            const char* kSysZ = "You are a pirate. Answer in one short sentence, with arrrs.";

            // X: the tool conversation (tools in its system prompt); Y: a short unrelated one. No header anywhere.
            const Resp x1 = ask(tool_body, tool_text);
            const uint64_t chain_x1 = sf.home[size_t(sf.cur)].h;
            CHECK(home_of(x1) == 0 && cached_of(x1) == 0 && chain_x1 == cold(tool_body), "homes: the first conversation takes home 0 (nothing used yet) and reads cold");
            const Value y1_body = chat_body(kSysY, "Title for: the clean room rules of the Limbo engine.");
            const Resp y1 = ask(y1_body, "Clean Room Rules");
            CHECK(home_of(y1) == 1 && cached_of(y1) == 0 && sf.home[1].h == cold(y1_body) && sf.home[0].h == chain_x1,
                  "homes: an unrelated second conversation takes the other home; home 0's state is untouched");

            // X's tool round: the prompt extends home 0's whole tape
            Value msgs = Value::array();
            for (const Value& m : tool_body.find("messages")->list()) msgs.push(m);
            msgs.push(*at(json(x1.body), {"choices", "0"}).find("message"));
            Value tool = Value::object();
            tool.set("role", Value::string("tool"));
            tool.set("tool_call_id", at(json(x1.body), {"choices", "0", "message", "tool_calls", "0", "id"}));
            tool.set("content", Value::string("sunny, 21 C"));
            msgs.push(tool);
            Value x2_body = tool_body.clone();
            x2_body.set("messages", msgs);
            const size_t tape_x = s3.tape_size(0);
            const Resp x2 = ask(x2_body, "Fine.\n</think>\n\nIt is sunny in Paris.");
            CHECK(home_of(x2) == 0 && cached_of(x2) == int(tape_x) && sf.home[0].h == cold(x2_body),
                  "homes: after the other conversation ran, X's tool round goes back to home 0 and reuses its whole tape");

            // Y's next user turn: the template re-renders Y's reply, the prompt diverges inside home 1's tape, and its Head checkpoint serves it
            const Value y2_body = follow(y1_body, y1, "Title for: how the tape keeps a conversation.");
            const Resp y2 = ask(y2_body, "The Tape");
            CHECK(home_of(y2) == 1 && cached_of(y2) == last_boundary(toks(y1_body)) && sf.home[1].h == cold(y2_body),
                  "homes: Y's new user turn is routed to home 1 by its prefix and restores its generation prompt checkpoint");

            // X's new user turn
            const Value x3_body = follow(x2_body, x2, "Thanks!");
            const int head_x1 = last_boundary(toks(tool_body));
            const Resp x3 = ask(x3_body, "Welcome.\n</think>\n\nYou are welcome.");
            const uint64_t chain_x3 = sf.home[0].h;
            CHECK(home_of(x3) == 0 && cached_of(x3) == head_x1 && chain_x3 == cold(x3_body), "homes: X's new user turn restores its checkpoint in home 0 while Y keeps home 1");
            CHECK(s3.tape_size(0) > 0 && s3.tape_size(1) > 0 && s3.checkpoints_held(0) >= 2 && s3.checkpoints_held(1) >= 1, "homes: both conversations hold a tape and checkpoints");

            // W: the same tools and system prompt as X, another question. It shares only the opening, so it is not X's:
            // the least recently used home (Y's, older than X's) is taken over, and X's home is not touched.
            Value w1_body = tool_body.clone();
            {
                Value m = Value::array();
                m.push(json(R"({"role":"user","content":"Weather in Rome?"})"));
                w1_body.set("messages", m);
            }
            const Resp w1 = ask(w1_body, "Checking.\n</think>\n\nSunny.");
            CHECK(home_of(w1) == 1 && sf.home[0].h == chain_x3 && sf.home[1].h == cold(w1_body),
                  "homes: a conversation that shares only the opening with X does not take X's home: the least recently used home is reset and taken over");
            const Resp x4 = ask(x3_body, "Welcome.\n</think>\n\nYou are welcome.");  // X again (a repeat of its last request)
            CHECK(home_of(x4) == 0 && cached_of(x4) == last_boundary(toks(x3_body)) && sf.home[0].h == cold(x3_body), "homes: X, untouched in home 0, still restores its checkpoint");

            // V: also the same opening; now home 1 (W's) is the least recently used, and a taken-over home keeps the opening it shares
            Value v1_body = tool_body.clone();
            {
                Value m = Value::array();
                m.push(json(R"({"role":"user","content":"Weather in Oslo?"})"));
                v1_body.set("messages", m);
            }
            const Resp v1 = ask(v1_body, "Checking.\n</think>\n\nSnowing.");
            CHECK(home_of(v1) == 1 && cached_of(v1) == boundary_after(toks(v1_body), 1) && sf.home[1].h == cold(v1_body),
                  "homes: the taken-over home keeps the opening (system prompt and tools) it shares with the new conversation: only the question is read");

            // a third, unrelated conversation: X (home 0) was used before V (home 1), so X's home is the least recently used
            const uint64_t chain_v1 = sf.home[1].h;
            const Value z1_body = chat_body(kSysZ, "Where is the treasure?");
            const Resp z1 = ask(z1_body, "Arrr, under the palm.");
            CHECK(home_of(z1) == 0 && cached_of(z1) == 0 && sf.home[0].h == cold(z1_body) && sf.home[1].h == chain_v1,
                  "homes: a third conversation takes the least recently used home (X's, used before V) and reads cold; V in home 1 is intact");

            // pins
            const Value q1_body = chat_body(kSysY, "Title for: the weather.");
            const Resp pin0 = ask(q1_body, "Weather", "X-Limbo-Session: 0\r\n");
            CHECK(home_of(pin0) == 0 && sf.home[0].h == cold(q1_body), "homes: X-Limbo-Session: 0 pins the request to home 0 (and takes it over)");
            const Value q2_body = chat_body(kSysZ, "Where is the rum?");
            const Resp lab1 = ask(q2_body, "Gone.", "X-Limbo-Session: alpha\r\n");
            const int alpha_home = home_of(lab1);
            const Resp lab2 = ask(chat_body(kSysY, "Title for: an unrelated thing."), "Thing", "X-Limbo-Session: alpha\r\n");
            CHECK(home_of(lab2) == alpha_home, "homes: a label is bound to the home its first request was routed to, whatever the prompt");
            const Resp pin_other = ask(q1_body, "Weather", "X-Limbo-Session: 9\r\n");
            CHECK(pin_other.status == 200 && (home_of(pin_other) == 0 || home_of(pin_other) == 1), "homes: a pin to a home that does not exist is routed by prefix");

            // cache_prompt false keeps the routing but reads cold
            const int resets0 = sf.n_reset.load();
            Value nc = q1_body.clone();
            nc.set("cache_prompt", Value::boolean(false));
            const Resp ncr = ask(nc, "Weather");
            CHECK(cached_of(ncr) == 0 && sf.n_reset.load() == resets0 + 1 && sf.home[size_t(home_of(ncr))].h == cold(q1_body), "homes: cache_prompt false resets the home it was routed to and reads everything");
            // a tiny conversation, two turns back to back: its restore is far below any token minimum, and it must still stay in its home
            const Value t1_body = chat_body("Be brief.", "Hi");
            const Resp t1 = ask(t1_body, "Hello.");
            const int th = home_of(t1);
            const uint64_t other_chain = sf.home[size_t(1 - th)].h;
            const Value t2_body = follow(t1_body, t1, "Bye");
            const Resp t2 = ask(t2_body, "Bye.");
            CHECK(last_boundary(toks(t1_body)) < 64 && home_of(t2) == th && cached_of(t2) == last_boundary(toks(t1_body)) && sf.home[size_t(1 - th)].h == other_chain && sf.home[size_t(th)].h == cold(t2_body),
                  "homes: a tiny conversation's next user turn (a restore of under 64 tokens) stays in its home with nothing else in between, the other home untouched");
            CHECK(sf.n_bad_pos.load() == 0 && s3.checkpoints_held(0) <= 6 && s3.checkpoints_held(1) <= 6, "homes: positions stayed consecutive in both homes, never more checkpoints than slots");

            stop3 = true;
            th3.join();
        }
    }

    // ---- no piece shorter than the backend's batch minimum, except the user turn's checkpoint ----
    {
        SnapFake sf(int(mv.v.tokens.size()), mv.im_end);
        sf.chunk = 64;
        sf.min_piece = 8;
        serve::Config c2 = cfg;
        c2.checkpoints = 4;
        c2.checkpoint_gap = 8;
        serve::Service s2(sf, tokenizer, *renderer, c2);
        auto server2 = limbo::http::Server::listen("127.0.0.1", 0, &err, so);
        if (server2) {
            std::atomic<bool> stop2{false};
            const int port2 = server2->port();
            std::thread th2([&] { server2->run([&](const limbo::http::Request& rq, limbo::http::Conn& c) { s2.handle(rq, c); }, stop2); });
            Value b = json(R"({"messages":[{"role":"system","content":"You are a careful assistant with a rather long system prompt."},
                                          {"role":"user","content":"Hello there, how are you doing today, my friend?"}],"max_tokens":50,"temperature":0,
                                          "chat_template_kwargs":{"enable_thinking":false}})");
            sf.script = enc("Fine.");
            {
                serve::ChatRequest cr;
                serve::Prompt p;
                std::string e;
                s2.parse_request(b, &cr, &e);
                s2.build_prompt(cr, &p, &e);
                sf.gen_from = int(p.tokens.size());
            }
            const Resp r = request(port2, "POST", "/v1/chat/completions", json_str(b));
            CHECK(r.status == 200, "pieces: the request is served");
            int small = 0;
            for (int n : sf.pieces)
                if (n < 8) ++small;
            // user turn: boundaries at the user message (long enough on both sides) and the generation prompt (the head)
            CHECK(s2.stats().checkpoints.load() == 2, "pieces: a user turn gets a checkpoint at the end of the system prompt and one at its generation prompt");
            CHECK(small <= 1, "pieces: only the piece after the user turn's checkpoint is shorter than the batch minimum");
            stop2 = true;
            th2.join();
        }
    }

    // ---- speculation in the server loop: a backend that returns 1..3 tokens a call ----
    {
        SpecFake sf(int(mv.v.tokens.size()), mv.im_end);
        serve::Config cs = cfg, cp = cfg;
        cp.speculate = false;
        serve::Service spec_svc(sf, tokenizer, *renderer, cs), plain_svc(sf, tokenizer, *renderer, cp);
        auto srv_s = limbo::http::Server::listen("127.0.0.1", 0, &err, so);
        auto srv_p = limbo::http::Server::listen("127.0.0.1", 0, &err, so);
        CHECK(srv_s != nullptr && srv_p != nullptr, "spec: two servers bind");
        if (srv_s && srv_p) {
            const int port_s = srv_s->port(), port_p = srv_p->port();
            std::atomic<bool> stop_s{false}, stop_p{false};
            std::thread th_s([&] { srv_s->run([&](const limbo::http::Request& rq, limbo::http::Conn& c) { spec_svc.handle(rq, c); }, stop_s); });
            std::thread th_p([&] { srv_p->run([&](const limbo::http::Request& rq, limbo::http::Conn& c) { plain_svc.handle(rq, c); }, stop_p); });
            struct Run {
                std::string content, finish;
                int64_t n = 0;
                std::vector<std::pair<int32_t, int>> fed;  // the reads after the prompt
                int calls = 0;
            };
            auto go = [&](int port, const std::string& text, int max_tokens, double temperature) {
                Value body = json(R"({"messages":[{"role":"user","content":"Spec."}],"temperature":0,"cache_prompt":false,"chat_template_kwargs":{"enable_thinking":false}})");
                body.set("max_tokens", Value::integer(max_tokens));
                body.set("temperature", Value::number(temperature));
                if (temperature > 0) body.set("top_k", Value::integer(1));
                sf.script = enc(text);
                const int plen = prompt_len(body);
                sf.gen_from = plen;
                sf.reads.clear();
                sf.calls = 0;
                const Value j = json(request(port, "POST", "/v1/chat/completions", json_str(body)).body);
                Run r;
                r.content = str_at(j, {"choices", "0", "message", "content"});
                r.finish = str_at(j, {"choices", "0", "finish_reason"});
                r.n = at(j, {"usage", "completion_tokens"}).i();
                for (const auto& rd : sf.reads)
                    if (rd.second >= plen) r.fed.push_back(rd);
                r.calls = sf.calls;
                return r;
            };
            const int plen0 = [&] {
                Value b = json(R"({"messages":[{"role":"user","content":"Spec."}],"max_tokens":8,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}})");
                return prompt_len(b);
            }();
            bool same = true, once = true, no_eog = true, cap = true, ran = true;
            int checked = 0;
            const std::string alpha = "abcdefghijklmnopqrstuvwxyz";
            for (int n : {1, 2, 3, 4, 5, 6, 7, 8, 11, 20}) {  // natural end after n tokens, at every alignment of the rotation
                const Run p = go(port_p, alpha.substr(0, size_t(n)), 100, 0), q = go(port_s, alpha.substr(0, size_t(n)), 100, 0);
                same = same && p.content == q.content && p.content == alpha.substr(0, size_t(n)) && p.finish == "stop" && q.finish == "stop" && p.n == q.n && p.n == n + 1 && p.fed == q.fed;
                ran = ran && p.calls == 0 && q.calls > 0;
                once = once && int(q.fed.size()) == n;
                for (size_t i = 0; i < q.fed.size(); ++i) once = once && q.fed[i].second == plen0 + int(i) && q.fed[i].first == int32_t(alpha[i]);
                for (const auto& rd : q.fed) no_eog = no_eog && rd.first != mv.im_end;
                ++checked;
            }
            CHECK(checked == 10 && same, "spec: greedy text, finish_reason, token count and every read equal the plain loop's (n = 1..20)");
            CHECK(ran && sf.by_len[1] > 0 && sf.by_len[2] > 0 && sf.by_len[3] > 0, "spec: decode_spec ran with runs of 1, 2 and 3 tokens; the plain loop does not call it");
            CHECK(once, "spec: every token is read exactly once, at consecutive positions after the prompt, the last one only to learn the end");
            CHECK(no_eog && sf.n_stopped > 0, "spec: an end-of-generation token that closes a run is never read");
            for (int m : {1, 2, 3, 4, 5, 6, 7, 11}) {  // max_tokens inside runs
                const Run p = go(port_p, alpha, m, 0), q = go(port_s, alpha, m, 0);
                cap = cap && q.n == m && q.finish == "length" && q.content == alpha.substr(0, size_t(m)) && q.fed == p.fed && int(q.fed.size()) == m - 1;
            }
            CHECK(cap, "spec: max_tokens is respected exactly (no token past it is read or returned), as in the plain loop");
            const Run sm = go(port_s, alpha.substr(0, 9), 100, 0.7);
            bool sm_ok = sm.content == alpha.substr(0, 9) && sm.n == 10 && sm.calls > 0 && sm.fed.size() == 9;
            for (size_t i = 0; i < sm.fed.size(); ++i) sm_ok = sm_ok && sm.fed[i].second == plen0 + int(i);
            CHECK(sm_ok, "spec: a sampled request (temperature 0.7, top_k 1) runs through decode_spec with the same reads");
            CHECK(sf.n_bad_pos.load() == 0, "spec: positions stayed consecutive in every request");
            stop_s = true;
            stop_p = true;
            th_s.join();
            th_p.join();
        }
    }

    // ---- the paged pool through the server: /props, /slots, id_slot, erase, prefix sharing both ways, the 400 ----
    {
        PoolFake pf(int(mv.v.tokens.size()), mv.im_end, 40, 4096);
        pf.chunk = 64;
        serve::Config cp = cfg;
        cp.checkpoints = 6;
        cp.checkpoint_gap = 1;
        cp.slots = 2;
        serve::Service sp(pf, tokenizer, *renderer, cp);
        auto server4 = limbo::http::Server::listen("127.0.0.1", 0, &err, so);
        CHECK(server4 != nullptr && sp.slot_count() == 2 && sp.startup_note().find("pool: 10240 positions in 40 pages of 256 shared by 2 slots") != std::string::npos,
              "pool: two slots over one pool, the start-up note carries the numbers");
        if (server4 && sp.slot_count() == 2) {
            const int port4 = server4->port();
            std::atomic<bool> stop4{false};
            std::thread th4([&] { server4->run([&](const limbo::http::Request& rq, limbo::http::Conn& c) { sp.handle(rq, c); }, stop4); });
            auto toks = [&](const Value& body) {
                serve::ChatRequest cr;
                serve::Prompt p;
                std::string e;
                if (!sp.parse_request(body, &cr, &e) || !sp.build_prompt(cr, &p, &e)) {
                    std::printf("FAIL pool prompt: %s\n", e.c_str());
                    ++limbo::test::failures();
                }
                return p.tokens;
            };
            auto last_boundary = [&](const std::vector<int32_t>& t) {
                int b = 0;
                for (size_t i = 1; i < t.size(); ++i)
                    if (t[i] == mv.im_start) b = int(i);
                return b;
            };
            auto with_slot = [&](Value b, int id) {
                b.set("id_slot", Value::integer(id));
                return b;
            };
            auto ask = [&](const Value& body, const std::string& text) {
                pf.script = enc(text);
                pf.gen_from = int(toks(body).size());
                return request(port4, "POST", "/v1/chat/completions", json_str(body));
            };
            auto cached_of = [&](const Resp& r) { return int(at(json(r.body), {"usage", "prompt_tokens_details", "cached_tokens"}).i()); };
            auto slot_of = [&](const Resp& r) { return int(at(json(r.body), {"timings", "slot"}).i()); };
            auto adopted_of = [&](const Resp& r) { return at(json(r.body), {"timings", "adopted_from"}).is_int() ? int(at(json(r.body), {"timings", "adopted_from"}).i()) : -1; };
            auto cold = [&](const Value& body) { return SnapFake::chain(toks(body), pf.script); };
            auto conv = [&](const std::string& user, bool follow_up) {
                Value b = json(R"({"max_tokens":40,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}})");
                Value m = Value::array();
                auto msg = [&](const char* role, const std::string& content) {
                    Value x = Value::object();
                    x.set("role", Value::string(role));
                    x.set("content", Value::string(content));
                    m.push(x);
                };
                msg("system", "You work on one long document.");
                msg("user", user);
                if (follow_up) {
                    msg("assistant", "ok");
                    msg("user", "now answer in one word");
                }
                b.set("messages", m);
                return b;
            };
            auto text_of = [](char c, int n) { return std::string(size_t(n), c); };

            // /props and /slots: what Ariadne's endpoint probe reads
            const Resp props = request(port4, "GET", "/props");
            const Value pj = json(props.body);
            CHECK(props.status == 200 && at(pj, {"total_slots"}).i() == 2 && at(pj, {"default_generation_settings", "n_ctx"}).i() == 4096 && at(pj, {"n_ctx"}).i() == 4096 &&
                      str_at(pj, {"model_alias"}) == "limbo-test" && str_at(pj, {"chat_template"}).size() > 20 && str_at(pj, {"build_info"}).rfind("Limbo", 0) == 0,
                  "/props: n_ctx, total_slots, the model and its template in llama-server's shape");
            const Resp sl0 = request(port4, "GET", "/slots");
            const Value sj0 = json(sl0.body);
            CHECK(sl0.status == 200 && sj0.is_list() && sj0.list().size() == 2 && at(sj0, {"1", "id"}).i() == 1 && at(sj0, {"0", "n_past"}).i() == 0 &&
                      at(sj0, {"0", "is_processing"}).is_bool() && !at(sj0, {"0", "is_processing"}).b() && at(sj0, {"1", "n_prompt_tokens"}).i() == 0,
                  "/slots: one entry per slot with id, n_prompt_tokens, n_past, is_processing");
            CHECK(request(port4, "POST", "/props", "{}").status == 405 && request(port4, "GET", "/slots/0").status == 405, "/props and /slots take GET, /slots/{id} takes POST");

            // the sidekick prewarms slot 0 with the main's next prefix; the main's request (slot 1) is that prefix + a short tail
            const std::string L = text_of('a', 900);
            const Value s_body = with_slot(conv(L, false), 0), m_body = with_slot(conv(L, true), 1);
            const Resp s1 = ask(s_body, "ok");
            const std::vector<int32_t> s_toks = toks(s_body);
            CHECK(s1.status == 200 && slot_of(s1) == 0 && cached_of(s1) == 0 && pf.home[0].h == cold(s_body) && sp.tape_size(0) >= s_toks.size(),
                  "pool: id_slot 0 runs the request in slot 0 and reads it cold");
            const int pages_s = pf.slot_pages(0);
            CHECK(pages_s == int((s_toks.size() + 255) / 256) && pf.pool().used_pages == pages_s && pf.pool().shared_pages == 0, "pool: the sidekick's prompt holds ceil(n / 256) pages, none shared");
            const Value sj1 = json(request(port4, "GET", "/slots").body);
            CHECK(at(sj1, {"0", "n_past"}).i() == int(sp.tape_size(0)) && at(sj1, {"0", "n_past"}).i() > 900 && at(sj1, {"1", "n_past"}).i() == 0, "/slots: slot 0 holds the prompt, slot 1 nothing");
            const int b_s = last_boundary(s_toks);  // the sidekick's Head checkpoint: its generation prompt
            const Resp m1 = ask(m_body, "yes");
            const std::vector<int32_t> m_toks = toks(m_body);
            CHECK(m1.status == 200 && slot_of(m1) == 1 && adopted_of(m1) == 0 && cached_of(m1) == b_s && pf.home[1].h == cold(m_body) && pf.home[0].h != 0,
                  "pool: the main takes the sidekick's prefix up to its checkpoint (cached = the history), reads only the rest, and its state equals a cold read");
            const int pages_m = pf.slot_pages(1);
            CHECK(pf.pool().shared_pages == b_s / 256 && b_s / 256 >= 3 && pf.pool().used_pages == pf.slot_pages(0) + pages_m - b_s / 256 && pf.pool().used_pages < pf.slot_pages(0) + pages_m,
                  "pool: the prefix is held once: the whole pages of the history are shared (refcount 2), the pool counts them once");
            std::printf("     pool: sidekick %zu tokens (%d pages), main %zu tokens (%d pages), cached %d; pool holds %d pages, %d shared\n", s_toks.size(), pages_s, m_toks.size(), pages_m,
                        cached_of(m1), pf.pool().used_pages, pf.pool().shared_pages);
            CHECK(sp.stats().adoptions.load() == 1 && pf.n_bad_pos.load() == 0, "pool: one adoption, positions consecutive");

            // the other way: the main reads a prefix first, the sidekick's request (slot 0) takes it
            const std::string L2 = text_of('b', 700);
            const Value r_body = with_slot(conv(L2, false), 1), q_body = with_slot(conv(L2, true), 0);
            const Resp r1 = ask(r_body, "ok");
            CHECK(r1.status == 200 && slot_of(r1) == 1 && pf.home[1].h == cold(r_body), "pool: the main reads another document");
            const int b_r = last_boundary(toks(r_body));
            const Resp q1 = ask(q_body, "no");
            CHECK(q1.status == 200 && slot_of(q1) == 0 && adopted_of(q1) == 1 && cached_of(q1) == b_r && pf.home[0].h == cold(q_body),
                  "pool: and the other way round: the sidekick takes the main's prefix");
            CHECK(pf.pool().shared_pages == b_r / 256 && pf.pool().used_pages == pf.slot_pages(0) + pf.slot_pages(1) - b_r / 256, "pool: that prefix is held once too (the old document's pages went back)");

            // the lender has read exactly the prompt's prefix and has no checkpoint at its tape's end (a tool round extends the whole tape):
            // its live state is snapshotted there and the other slot takes everything it has read
            const Resp x1 = ask(with_slot(tool_body, 0), tool_text);
            const size_t tape_x = sp.tape_size(0);
            Value msgs = Value::array();
            for (const Value& m : tool_body.find("messages")->list()) msgs.push(m);
            msgs.push(*at(json(x1.body), {"choices", "0"}).find("message"));
            Value tool = Value::object();
            tool.set("role", Value::string("tool"));
            tool.set("tool_call_id", at(json(x1.body), {"choices", "0", "message", "tool_calls", "0", "id"}));
            tool.set("content", Value::string("sunny, 21 C"));
            msgs.push(tool);
            Value x2_body = with_slot(tool_body, 1);
            x2_body.set("messages", msgs);
            const Resp x2 = ask(x2_body, "Fine.\n</think>\n\nIt is sunny in Paris.");
            CHECK(x1.status == 200 && x2.status == 200 && slot_of(x1) == 0 && slot_of(x2) == 1 && adopted_of(x2) == 0 && cached_of(x2) == int(tape_x) && pf.home[1].h == cold(x2_body),
                  "pool: a prompt that continues the other slot's whole tape takes all of it (a snapshot of the lender's live state), nothing is read twice");
            std::printf("     pool: slot 1 took %d of the %zu positions slot 0 had read\n", cached_of(x2), tape_x);

            // erase and id_slot errors
            const int used_before = pf.pool().used_pages;
            const Resp er = request(port4, "POST", "/slots/1?action=erase", "{}");
            CHECK(er.status == 200 && at(json(er.body), {"id_slot"}).i() == 1 && at(json(er.body), {"n_erased"}).i() > 0 && sp.tape_size(1) == 0 && pf.slot_pages(1) == 0 &&
                      pf.pool().used_pages < used_before,
                  "erase: POST /slots/1?action=erase empties the slot's tape and gives its pages back");
            CHECK(request(port4, "POST", "/slots/7?action=erase", "{}").status == 404 && request(port4, "POST", "/slots/x?action=erase", "{}").status == 404 &&
                      request(port4, "POST", "/slots/0", "{}").status == 400,
                  "erase: a slot that does not exist is 404, a missing action 400");
            Value bad = with_slot(conv("hello", false), 0);
            bad.set("id_slot", Value::string("one"));
            CHECK(request(port4, "POST", "/v1/chat/completions", json_str(bad)).status == 400, "id_slot: a non-number is 400");
            const Resp far = ask(with_slot(conv("hello there", false), 9), "ok");
            CHECK(far.status == 200 && (slot_of(far) == 0 || slot_of(far) == 1), "id_slot: a slot that does not exist is routed by prefix");
            CHECK(pf.pp.consistent() && pf.n_bad_pos.load() == 0, "pool: every count still equals its holders after all of it");
            stop4 = true;
            th4.join();
        }
    }

    // ---- the verbose log: formats, then -v / -vv through a server with two slots over one pool ----
    {
        namespace vlog = limbo::vlog;
        CHECK(vlog::thou(0) == "0" && vlog::thou(999) == "999" && vlog::thou(41923) == "41,923" && vlog::thou(1234567) == "1,234,567" && vlog::thou(-12345) == "-12,345",
              "vlog: thousands separators");
        CHECK(vlog::bytes(512) == "512 B" && vlog::bytes(4096) == "4.0 KiB" && vlog::bytes(312ull << 20) == "312 MiB" && vlog::bytes(3ull << 30) == "3.00 GiB" &&
                  vlog::bytes(1536ull << 20) == "1.50 GiB",
              "vlog: byte units");
        CHECK(vlog::ms(412) == "412 ms" && vlog::ms(31.25) == "31.2 ms" && vlog::ms(3210) == "3.21 s" && vlog::ms(12000) == "12.0 s" && vlog::rate(4424, 1) == "4,424 tok/s" &&
                  vlog::rate(72.6, 1) == "72.6 tok/s" && vlog::pct(957, 1000) == "95.7%",
              "vlog: time, rate and percent units");
        std::vector<std::string> got;
        std::mutex got_mu;
        vlog::set_sink([&](const std::string& l) {
            std::lock_guard<std::mutex> lk(got_mu);
            got.push_back(l);
        });
        vlog::set_color(false);
        vlog::line("req", "plain", 7);
        vlog::set_color(true);
        vlog::line("req", "coloured", 7);
        vlog::set_color(false);
        CHECK(got.size() == 2 && std::regex_match(got[0], std::regex(R"(\d\d:\d\d:\d\d\.\d{3}  #7    req      plain\n)")), "vlog: HH:MM:SS.mmm, the request number, the tag padded to 8 columns, the text");
        CHECK(got[0].find('\x1b') == std::string::npos && got[1].find("\x1b[") != std::string::npos, "vlog: colour codes only when colour is on");
        got.clear();

        VerboseFake vf(int(mv.v.tokens.size()), mv.im_end, 40, 4096);
        vf.chunk = 64;
        serve::Config cv = cfg;
        cv.checkpoints = 6;
        cv.checkpoint_gap = 1;
        cv.slots = 2;
        cv.log = true;
        cv.verbose = 2;
        {
            serve::Service sv(vf, tokenizer, *renderer, cv);
            auto has = [&](const std::string& needle) {
                std::lock_guard<std::mutex> lk(got_mu);
                return std::any_of(got.begin(), got.end(), [&](const std::string& l) { return l.find(needle) != std::string::npos; });
            };
            CHECK(vf.verbose_level == 2, "-v: the backend is told the level (Backend::set_verbose)");
            CHECK(has("slots   ") && has("pool    pool: ") == false && has("pool     10,240") == false, "-v start-up: lines are tagged by topic");
            CHECK(has("http     POST /v1/chat/completions; GET /health") && has("/metrics (llama-server") && has("pool     10240 positions in 40 pages") && has("ckpt     4 of 6 asked") && has("model    fake model, 2 layers") && has("kv       fake K/V"), "-v start-up: slots, pool, checkpoints, the backend's own lines");
            CHECK(has("vram     weights") && has("vram     K/V pool") && has("vram     reserve") && has("vram     free") && has("1.00 GiB") && has("768 MiB"),
                  "-v start-up: the VRAM table lists the backend's items, the reserve and what is free, with units");
            auto srv = limbo::http::Server::listen("127.0.0.1", 0, &err, so);
            CHECK(srv != nullptr, "-v: a server binds");
            if (srv) {
                const int pv = srv->port();
                std::atomic<bool> stop_v{false};
                std::thread th_v([&] { srv->run([&](const limbo::http::Request& rq, limbo::http::Conn& c) { sv.handle(rq, c); }, stop_v); });
                auto body_of = [&](const std::string& user, int id_slot, int max_tokens) {
                    Value b = json(R"({"temperature":0,"chat_template_kwargs":{"enable_thinking":false}})");
                    b.set("max_tokens", Value::integer(max_tokens));
                    b.set("id_slot", Value::integer(id_slot));
                    Value m = Value::array();
                    Value sys = Value::object();
                    sys.set("role", Value::string("system"));
                    sys.set("content", Value::string("You work on one long document."));
                    m.push(sys);
                    Value u = Value::object();
                    u.set("role", Value::string("user"));
                    u.set("content", Value::string(user));
                    m.push(u);
                    b.set("messages", m);
                    return b;
                };
                auto ask = [&](const Value& body, const std::string& text) {
                    serve::ChatRequest cr;
                    serve::Prompt p;
                    std::string e;
                    sv.parse_request(body, &cr, &e);
                    sv.build_prompt(cr, &p, &e);
                    vf.script = enc(text);
                    vf.gen_from = int(p.tokens.size());
                    return request(pv, "POST", "/v1/chat/completions", json_str(body));
                };
                const std::string doc(900, 'a');
                const Resp a1 = ask(body_of(doc, 0, 40), "ok");
                CHECK(a1.status == 200, "-v: a request is served");
                const Resp a2 = ask(body_of(doc + " and more", 1, 40), "yes");  // slot 1 takes slot 0's prefix
                CHECK(a2.status == 200, "-v: a second request is served");
                const Resp pr = ask(body_of(doc + " and prime", 0, 1), "ok");  // a prime (max_tokens 1) checks for a waiting request after each piece
                CHECK(pr.status == 200, "-v: a prime is served");
                const Resp er = request(pv, "POST", "/slots/1?action=erase", "{}");
                CHECK(er.status == 200 && request(pv, "POST", "/v1/chat/completions", "{").status == 400, "-v: erase and a malformed body are answered");
                stop_v = true;
                th_v.join();
                std::vector<std::string> all;
                {
                    std::lock_guard<std::mutex> lk(got_mu);
                    all = got;
                }
                bool shape = !all.empty();
                for (const std::string& l : all) {
                    const bool ok = std::regex_match(l, std::regex(R"(\d\d:\d\d:\d\d\.\d{3}  (#\d+ *|     ) [a-z]+ +\S.*\n)")) && l.size() > 30 && l[28] == ' ' && l[29] != ' ';
                    if (!ok) std::printf("     bad line: %s", l.c_str());
                    shape &= ok;
                }
                CHECK(shape, "-v: every line is HH:MM:SS.mmm, a request-number column, a tag padded to 8, the text, in aligned columns");
                auto pre = [](int id, const char* tag) {
                    char b[64];
                    std::snprintf(b, sizeof b, "#%-4d %-8s ", id, tag);
                    return std::string(b);
                };
                CHECK(has(pre(1, "req") + "POST /v1/chat/completions  slot 0 via id_slot  prompt ") && has("temp 0.00  no-stream  2 msgs, 0 tools"), "-v: the request header: method, path, slot, route, prompt tokens, settings");
                CHECK(has(pre(1, "cache") + "cached 0 of ") && has("via empty slot"), "-v: a cold request says it read everything");
                CHECK(has(pre(1, "pages") + "slot 0 takes ") && has("pool 4/40 pages, 0 shared"), "-v: pages taken for the prompt, with the pool's state");
                CHECK(has(pre(1, "prefill") + "piece 1/3  ") && has("tok/s") && has("pos 0.."), "-v: prompt progress per piece (i/N, tokens, positions, ms, tok/s)");
                CHECK(has("checkpoint @") && has("(root) in snapshot slot") && has("(head) in snapshot slot"), "-v: every checkpoint taken is named with its kind and snapshot slot");
                CHECK(has(pre(1, "decode") + "") && has(" tok in ") && has("finish stop") && has(pre(1, "done") + "/v1/chat/completions  slot 0  prompt "), "-v: the decode summary and the closing line");
                CHECK(has(pre(2, "slot") + "adopt  slot 0 -> 1  ") && has(" by reference, 1 copied (the cut falls inside a page") && has(pre(2, "cache") + "cached ") && has("adopted-from 0"),
                      "-v: an adoption names lender, borrower, positions, pages by reference and copied, and the request header's adopted-from");
                CHECK(has(pre(3, "yield") + "after ") && has("no request waiting, reading on"), "-vv: a prime logs its check for a waiting request after each piece");
                CHECK(has("slot     erase  slot 1: ") && has("pages freed") && has("warn     400 "), "-v: erase and a refused request are events too");
                std::printf("     verbose: %zu lines, e.g.\n", all.size());
                for (size_t i = 0, shown = 0; i < all.size() && shown < 6; ++i)
                    if (all[i].find(" req ") != std::string::npos || all[i].find(" prefill ") != std::string::npos || all[i].find(" adopt ") != std::string::npos) {
                        std::printf("       %s", all[i].c_str());
                        ++shown;
                    }
            }
        }
        // -v off: the service writes nothing through vlog (the one-line log stays on stderr)
        {
            got.clear();
            serve::Config c0 = cv;
            c0.verbose = 0;
            VerboseFake v0(int(mv.v.tokens.size()), mv.im_end, 40, 4096);
            serve::Service s0(v0, tokenizer, *renderer, c0);
            CHECK(got.empty() && v0.verbose_level == 0, "-v off: no verbose lines at start-up and the backend is told 0");
        }
        vlog::set_sink(nullptr);
    }

    // ---- ngram drafts through the server: the text is the plain loop's, and each slot's index is its tape through rewinds, adoptions and erases ----
    {
        NgramFake on(int(mv.v.tokens.size()), mv.im_end, 40, 4096, 3), off(int(mv.v.tokens.size()), mv.im_end, 40, 4096, 3);
        on.chunk = off.chunk = 64;
        serve::Config cn = cfg, cf = cfg;
        cn.checkpoints = cf.checkpoints = 6;
        cn.checkpoint_gap = cf.checkpoint_gap = 1;
        cn.slots = cf.slots = 2;
        cf.speculate = false;
        serve::Service sn(on, tokenizer, *renderer, cn), sf(off, tokenizer, *renderer, cf);
        auto srv_n = limbo::http::Server::listen("127.0.0.1", 0, &err, so);
        auto srv_f = limbo::http::Server::listen("127.0.0.1", 0, &err, so);
        CHECK(srv_n && srv_f && sn.slot_count() == 2 && sf.slot_count() == 2, "ngram: two servers with two slots each");
        if (srv_n && srv_f) {
            const int pn = srv_n->port(), pf = srv_f->port();
            std::atomic<bool> stop_n{false}, stop_f{false};
            std::thread tn([&] { srv_n->run([&](const limbo::http::Request& rq, limbo::http::Conn& c) { sn.handle(rq, c); }, stop_n); });
            std::thread tf([&] { srv_f->run([&](const limbo::http::Request& rq, limbo::http::Conn& c) { sf.handle(rq, c); }, stop_f); });
            std::string doc;
            for (int i = 0; i < 26; ++i) doc += "item " + std::to_string(i * 7 % 13) + " = compute(" + char('a' + i % 26) + ");\n";
            std::string ans = doc;
            ans[200] = 'Z';  // a renamed identifier in the middle of a quoted passage
            ans[420] = 'Q';
            auto body_of = [&](bool follow_up, bool second, int slot, double temp) {
                Value b = json(R"({"max_tokens":400,"chat_template_kwargs":{"enable_thinking":false}})");
                b.set("temperature", Value::number(temp));
                if (temp > 0) {
                    b.set("top_k", Value::integer(1));
                    b.set("seed", Value::integer(3));
                }
                b.set("id_slot", Value::integer(slot));
                Value m = Value::array();
                auto msg = [&](const char* role, const std::string& content) {
                    Value x = Value::object();
                    x.set("role", Value::string(role));
                    x.set("content", Value::string(content));
                    m.push(x);
                };
                msg("system", "You work on one long document.");
                msg("user", second ? "Another document: " + doc : "Quote the document: " + doc);
                if (follow_up) {
                    msg("assistant", "ok");
                    msg("user", "now quote it again");
                }
                b.set("messages", m);
                return b;
            };
            auto toks = [&](const Value& body) {
                serve::ChatRequest cr;
                serve::Prompt p;
                std::string e;
                sn.parse_request(body, &cr, &e);
                sn.build_prompt(cr, &p, &e);
                return p.tokens;
            };
            auto content_of = [&](const Resp& r) { return str_at(json(r.body), {"choices", "0", "message", "content"}); };
            bool same = true, tape_ok = true, state_ok = true;
            auto run = [&](const Value& body, const std::string& answer, const char* what) {
                const size_t np = toks(body).size();
                on.script = off.script = enc(answer);
                on.gen_from = off.gen_from = int(np);
                const Resp a = request(pn, "POST", "/v1/chat/completions", json_str(body));
                const Resp b = request(pf, "POST", "/v1/chat/completions", json_str(body));
                const bool eq = a.status == 200 && b.status == 200 && content_of(a) == content_of(b) && content_of(a).size() > 100;
                same &= eq;
                for (int sl = 0; sl < 2; ++sl) {
                    tape_ok &= on.ix.at(sl).size() == sn.tape_size(sl) && off.ix.at(sl).size() == sf.tape_size(sl);
                    tape_ok &= SnapFake::chain(on.ix.at(sl).tape()) == on.home[size_t(sl)].h && SnapFake::chain(off.ix.at(sl).tape()) == off.home[size_t(sl)].h;
                }
                state_ok &= on.home[size_t(int(at(body, {"id_slot"}).i()))].h == off.home[size_t(int(at(body, {"id_slot"}).i()))].h;
                std::printf("     ngram: %-28s same text %d, drafted %d kept %d over %d cycles so far\n", what, eq ? 1 : 0, on.drafted, on.kept, on.cycles);
            };
            run(body_of(false, false, 0, 0), ans, "cold read, slot 0");
            const int kept1 = on.kept;
            CHECK(on.cycles > 0 && on.drafted > 0 && kept1 > 20 && on.kept < on.drafted, "ngram: the quoted document is drafted from the prompt; hits and misses (the substituted tokens)");
            run(body_of(true, false, 1, 0), ans, "follow-up on slot 1 (adopts)");
            run(body_of(true, false, 0, 0), ans, "slot 0 again (restore, cut)");
            run(body_of(false, true, 1, 0), ans, "another document, slot 1");
            CHECK(request(pn, "POST", "/slots/1?action=erase", "{}").status == 200 && request(pf, "POST", "/slots/1?action=erase", "{}").status == 200 && on.ix.at(1).size() == 0 &&
                      on.ix.at(0).size() == sn.tape_size(0),
                  "ngram: erasing a slot empties its index and leaves the other's");
            run(body_of(false, false, 1, 0.7), ans, "sampled (top_k 1), slot 1");
            CHECK(same, "ngram: the text equals the plain loop's after every request, greedy and sampled");
            CHECK(tape_ok, "ngram: every slot's index is its tape (and the backend's consumed tokens) after every request: rewinds, adoptions, erases keep it exact");
            CHECK(state_ok, "ngram: the backend's state chain equals the plain loop's");
            CHECK(on.n_bad_pos.load() == 0 && on.pp.consistent(), "ngram: positions consecutive, pool counts consistent");
            stop_n = true;
            stop_f = true;
            tn.join();
            tf.join();
        }
    }

    // ---- --api-key: /health is open, everything else wants Authorization: Bearer <key> ----
    {
        Fake ka(int(mv.v.tokens.size()), mv.im_end);
        serve::Config ck = cfg;
        ck.api_keys = {"s3cret-key", "second-key"};
        serve::Service sk(ka, tokenizer, *renderer, ck);
        auto srv_k = limbo::http::Server::listen("127.0.0.1", 0, &err, so);
        CHECK(srv_k != nullptr, "api-key: a server binds");
        if (srv_k) {
            const int pk = srv_k->port();
            std::atomic<bool> stop_k{false};
            std::thread tk([&] { srv_k->run([&](const limbo::http::Request& rq, limbo::http::Conn& c) { sk.handle(rq, c); }, stop_k); });
            const std::string chat = R"({"messages":[{"role":"user","content":"Hi."}],"max_tokens":4,"temperature":0})";
            auto auth = [](const std::string& v) { return "Authorization: " + v + "\r\n"; };
            ka.script = enc("ok");
            ka.gen_from = prompt_len(json(chat));
            const Resp h0 = request(pk, "GET", "/health");
            CHECK(h0.status == 200 && request(pk, "GET", "/v1/health").status == 200, "api-key: /health and /v1/health stay open");
            const Resp n0 = request(pk, "GET", "/v1/models");
            CHECK(n0.status == 401 && str_at(json(n0.body), {"error", "type"}) == "authentication_error" && n0.header("www-authenticate") == "Bearer", "api-key: no header is 401, authentication_error, WWW-Authenticate: Bearer");
            CHECK(request(pk, "GET", "/SDK/webLanguage").status == 401 && request(pk, "GET", "/props").status == 401 && request(pk, "GET", "/slots").status == 401 &&
                      request(pk, "GET", "/metrics").status == 401 && request(pk, "POST", "/slots/0?action=erase", "{}").status == 401,
                  "api-key: a scanner's path, /props, /slots, /metrics and erase are all 401 (no 404 that says what exists)");
            CHECK(request(pk, "POST", "/v1/chat/completions", chat).status == 401 && ka.n_forward.load() == 0, "api-key: a completion without the key is 401 and never reaches the backend");
            CHECK(request(pk, "GET", "/v1/models", "", auth("Bearer wrong-key")).status == 401 && request(pk, "GET", "/v1/models", "", auth("Bearer s3cret-ke")).status == 401 &&
                      request(pk, "GET", "/v1/models", "", auth("Bearer s3cret-key2")).status == 401 && request(pk, "GET", "/v1/models", "", auth("Basic s3cret-key")).status == 401 &&
                      request(pk, "GET", "/v1/models", "", auth("s3cret-key")).status == 401 && request(pk, "GET", "/v1/models", "", auth("Bearer ")).status == 401,
                  "api-key: a wrong key, a prefix of it, the key plus a byte, another scheme, no scheme, an empty token: all 401");
            CHECK(request(pk, "GET", "/v1/models", "", auth("Bearer s3cret-key")).status == 200 && request(pk, "GET", "/v1/models", "", auth("bearer second-key")).status == 200,
                  "api-key: either configured key opens the endpoints (the scheme's case does not matter)");
            const Resp ok = request(pk, "POST", "/v1/chat/completions", chat, auth("Bearer s3cret-key"));
            CHECK(ok.status == 200 && request(pk, "GET", "/props", "", auth("Bearer s3cret-key")).status == 200 && request(pk, "GET", "/metrics", "", auth("Bearer s3cret-key")).status == 200,
                  "api-key: with the key a completion, /props and /metrics are served");
            stop_k = true;
            tk.join();
        }
        // off unless set: the default service answers without a header (every test above relies on it too)
        serve::Service s0(ka, tokenizer, *renderer, cfg);
        auto srv_0 = limbo::http::Server::listen("127.0.0.1", 0, &err, so);
        if (srv_0) {
            std::atomic<bool> stop_0{false};
            std::thread t0([&] { srv_0->run([&](const limbo::http::Request& rq, limbo::http::Conn& c) { s0.handle(rq, c); }, stop_0); });
            CHECK(serve::Config().api_keys.empty() && request(srv_0->port(), "GET", "/v1/models").status == 200 && request(srv_0->port(), "GET", "/props").status == 200,
                  "api-key: off unless set");
            stop_0 = true;
            t0.join();
        }
    }

    // ---- a pool too small for a prompt: 400 with the numbers; the answer is cut to what is left ----
    {
        PoolFake pf(int(mv.v.tokens.size()), mv.im_end, 3, 4096);  // 768 positions
        pf.chunk = 64;
        serve::Config cp = cfg;
        cp.slots = 1;
        serve::Service sp(pf, tokenizer, *renderer, cp);
        auto server5 = limbo::http::Server::listen("127.0.0.1", 0, &err, so);
        if (server5) {
            const int port5 = server5->port();
            std::atomic<bool> stop5{false};
            std::thread th5([&] { server5->run([&](const limbo::http::Request& rq, limbo::http::Conn& c) { sp.handle(rq, c); }, stop5); });
            auto body_of = [&](int n_user, int max_tokens) {
                Value b = json(R"({"temperature":0,"chat_template_kwargs":{"enable_thinking":false}})");
                b.set("max_tokens", Value::integer(max_tokens));
                Value m = Value::array();
                Value x = Value::object();
                x.set("role", Value::string("user"));
                x.set("content", Value::string(std::string(size_t(n_user), 'z')));
                m.push(x);
                b.set("messages", m);
                return b;
            };
            const Value big = body_of(900, 10);
            const Resp r = request(port5, "POST", "/v1/chat/completions", json_str(big));
            const std::string msg = str_at(json(r.body), {"error", "message"});
            CHECK(r.status == 400 && msg.find("768 positions") != std::string::npos && msg.find("the prompt needs") != std::string::npos && pf.slot_pages(0) <= 3,
                  "pool: a prompt that does not fit the pool is refused with 400 and the numbers");
            std::printf("     pool: %s\n", msg.c_str());
            const Value fits = body_of(600, 5000);
            serve::ChatRequest cr;
            serve::Prompt pr;
            std::string e;
            sp.parse_request(fits, &cr, &e);
            sp.build_prompt(cr, &pr, &e);
            pf.script = enc(std::string(400, 'q'));
            pf.gen_from = int(pr.tokens.size());
            const Resp r2 = request(port5, "POST", "/v1/chat/completions", json_str(fits));
            const Value j2 = json(r2.body);
            CHECK(r2.status == 200 && str_at(j2, {"choices", "0", "finish_reason"}) == "length" && at(j2, {"usage", "completion_tokens"}).i() == 768 - int(pr.tokens.size()) - 8,
                  "pool: a prompt that fits is answered, and the answer stops at what the pool has left (less the speculation window)");
            stop5 = true;
            th5.join();
        }
    }

    // ---- a prime (max_tokens 1) steps aside for a waiting request, keeps its tape, and its retry reads only the rest ----
    {
        PoolFake pf(int(mv.v.tokens.size()), mv.im_end, 40, 4096);
        pf.chunk = 64;
        serve::Config cp = cfg;
        cp.checkpoints = 6;
        cp.checkpoint_gap = 1;
        cp.slots = 2;
        serve::Service sp(pf, tokenizer, *renderer, cp);
        auto srv = limbo::http::Server::listen("127.0.0.1", 0, &err, so);
        CHECK(srv != nullptr && sp.slot_count() == 2, "prime: two slots over one pool");
        if (srv && sp.slot_count() == 2) {
            const int pp = srv->port();
            std::atomic<bool> stop_y{false};
            std::thread th_y([&] { srv->run([&](const limbo::http::Request& rq, limbo::http::Conn& c) { sp.handle(rq, c); }, stop_y); });
            auto body_of = [&](char fill, int n, int max_tokens, bool stream, int slot) {
                Value b = json(R"({"temperature":0,"chat_template_kwargs":{"enable_thinking":false}})");
                b.set("max_tokens", Value::integer(max_tokens));
                if (stream) b.set("stream", Value::boolean(true));
                if (slot >= 0) b.set("id_slot", Value::integer(slot));
                Value m = Value::array();
                for (auto [role, text] : {std::pair<const char*, std::string>{"system", "You work on one long document."}, {"user", std::string(size_t(n), fill)}}) {
                    Value x = Value::object();
                    x.set("role", Value::string(role));
                    x.set("content", Value::string(text));
                    m.push(x);
                }
                b.set("messages", m);
                return b;
            };
            auto toks = [&](const Value& body) {
                serve::ChatRequest cr;
                serve::Prompt p;
                std::string e;
                if (!sp.parse_request(body, &cr, &e) || !sp.build_prompt(cr, &p, &e)) ++limbo::test::failures();
                return p.tokens;
            };
            auto post = [&](const Value& body) { return request(pp, "POST", "/v1/chat/completions", json_str(body)); };
            auto cached_of = [&](const Resp& r) { return int(at(json(r.body), {"usage", "prompt_tokens_details", "cached_tokens"}).i()); };
            auto slot_of = [&](const Resp& r) { return int(at(json(r.body), {"timings", "slot"}).i()); };
            auto n_past = [&](int slot) { return int(at(json(request(pp, "GET", "/slots").body), {std::to_string(slot).c_str(), "n_past"}).i()); };
            pf.script = enc("ok");
            pf.gen_from = 1 << 20;  // the scripted text never starts: any generated token is the end token

            // nobody waits: a prime reads its whole prompt
            const Value calm = body_of('a', 1500, 1, false, 0);
            const int n_calm = int(toks(calm).size());
            const Resp r0 = post(calm);
            CHECK(r0.status == 200 && cached_of(r0) == 0 && sp.stats().preempted.load() == 0 && n_past(0) == n_calm,
                  "prime: nobody waits -> it reads the whole prompt and answers 200");
            CHECK(pf.n_sync.load() >= n_calm / 64 - 1, "prime: the backend is synced after every piece it reads");

            // a long prime on slot 0; 100 ms later a main request on slot 1
            pf.sleep_us = 1000;
            const Value prime = body_of('b', 1500, 1, false, 0);
            const std::vector<int32_t> ptoks = toks(prime);
            const int n_prime = int(ptoks.size());
            const Value main_body = body_of('m', 40, 8, false, 1);
            Resp rp, rm;
            const int fwd0 = pf.n_forward.load();
            std::thread tp([&] { rp = post(prime); });
            wait_until([&] { return pf.n_forward.load() > fwd0 + 100; }, 3000);
            const double tm0 = now_ms();
            std::thread tm([&] { rm = post(main_body); });
            tp.join();
            const double t_prime_end = now_ms();
            tm.join();
            const double t_main = now_ms() - tm0;
            pf.sleep_us = 0;
            const Value ej = json(rp.body);
            const std::string emsg = str_at(ej, {"error", "message"});
            const int read = n_past(0);
            CHECK(rp.status == 503 && rp.header("retry-after") == "1" && str_at(ej, {"error", "type"}) == "server_error" &&
                      emsg.find("preempted by a waiting request; the " + std::to_string(read) + " tokens read so far stay cached on slot 0") == 0,
                  "prime: a waiting request makes it answer 503, Retry-After: 1 and an OpenAI-shaped error naming what stays cached");
            std::printf("     prime: %s\n", emsg.c_str());
            CHECK(read > 100 && read < n_prime - 1 && sp.stats().preempted.load() == 1, "prime: it stopped inside the prompt");
            std::printf("     prime: read %d of %d tokens\n", read, n_prime);
            CHECK(rm.status == 200 && slot_of(rm) == 1, "prime: the waiting request is then served, on its own slot");
            CHECK(t_prime_end - tm0 < 400 && t_main < 1500, "prime: the waiting request is served long before the prime would have finished reading");
            std::printf("     prime: main waited %.0f ms in all; the prime stopped %.0f ms after it arrived\n", t_main, t_prime_end - tm0);

            // the slot's tape is its state: consecutive positions, the chain over exactly the tokens read
            const std::vector<int32_t> head(ptoks.begin(), ptoks.begin() + read);
            CHECK(pf.n_bad_pos == 0 && pf.home[0].pos == read && pf.home[0].h == SnapFake::chain(head),
                  "prime: after the 503 the slot's state is the chain over its tape (consecutive positions, last piece whole)");

            // the retry: the same prompt, unpinned, continues from the tape
            const int fwd1 = pf.n_forward.load();
            const Resp rr = post(prime);
            CHECK(rr.status == 200 && slot_of(rr) == 0 && cached_of(rr) == read && pf.n_forward.load() - fwd1 == n_prime - read,
                  "prime: the retry goes to the same slot, cached = what was read, and only the rest is read");
            CHECK(pf.n_bad_pos == 0 && n_past(0) == n_prime, "prime: the retry's tape is the whole prompt");

            // never yielding: a normal request, and a streamed prime, with someone waiting
            const uint64_t pre0 = sp.stats().preempted.load();
            const int sync0 = pf.n_sync.load();
            pf.sleep_us = 1000;
            for (const char kind : {'n', 's'}) {
                const Value b = kind == 'n' ? body_of('c', 1200, 4, false, 0) : body_of('d', 1200, 1, true, 0);
                Resp rb, rw;
                const int f0 = pf.n_forward.load();
                std::thread tb([&] { rb = post(b); });
                wait_until([&] { return pf.n_forward.load() > f0 + 100; }, 3000);
                std::thread tw([&] { rw = post(body_of('w', 20, 4, false, 1)); });
                tb.join();
                tw.join();
                CHECK(rb.status == 200 && rw.status == 200 && sp.stats().preempted.load() == pre0,
                      kind == 'n' ? "prime: a request with max_tokens > 1 never yields" : "prime: a streamed prime never yields (its headers are out)");
                if (kind == 'n') CHECK(pf.n_sync.load() == sync0, "prime: a request with max_tokens > 1 never syncs between pieces");
            }
            pf.sleep_us = 0;
            stop_y = true;
            th_y.join();
        }
    }

    // ---- the Backend defaults: a backend that only implements forward(token, pos) still serves ----
    {
        struct Plain : serve::Backend {
            std::vector<float> lg = std::vector<float>(8, 0.f);
            std::vector<std::pair<int32_t, int>> calls;
            int n_vocab() const override { return 8; }
            void reset() override {}
            const float* forward(int32_t t, int pos) override {
                calls.emplace_back(t, pos);
                std::fill(lg.begin(), lg.end(), 0.f);
                lg[3] = lg[5] = 2.f;  // a tie
                return lg.data();
            }
        } p;
        serve::Backend& b = p;
        CHECK(b.forward(11, 0, false) == nullptr && b.forward(12, 1, true) == p.lg.data(), "Backend default: forward(.., want_logits) is nullptr when skipped, the logits otherwise");
        CHECK(b.forward_greedy(13, 2) == 3, "Backend default: forward_greedy takes the argmax of the logits, the lowest id on a tie");
        const int32_t toks[3] = {21, 22, 23};
        b.prefill(toks, 3, 3);
        CHECK(b.prefill_chunk() == 1 && p.calls.size() == 6 && p.calls[3] == std::make_pair(int32_t(21), 3) && p.calls[5] == std::make_pair(int32_t(23), 5),
              "Backend default: prefill is forward() per token at consecutive positions, one token per chunk");
        std::vector<float> tie(100, 0.f);
        tie[40] = tie[9] = 1.5f;
        CHECK(serve::Sampler::argmax(tie.data(), 100) == 9, "Sampler::argmax: the lowest id on a tie");
    }

    return limbo::test::finish("serve");
}
