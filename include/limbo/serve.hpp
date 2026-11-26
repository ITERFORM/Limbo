// Copyright (c) 2026 ITERFORM. All rights reserved.
#pragma once
// limbo/serve.hpp — `limbo serve`: an OpenAI-compatible chat endpoint over a forward pass.
//
//   POST /v1/chat/completions   stream false and true (SSE chunks, then `data: [DONE]`)
//   GET  /v1/models, GET /health
//   GET  /props, GET /slots, POST /slots/{id}?action=erase   llama-server's, which Ariadne reads (see Service::handle)
//
// A request is turned into a prompt by the model's own chat template (chat.hpp), tokenized, run through
// a Backend (the prompt without its logits, then one token at a time) and the generated text is split into
// reasoning_content, content and tool_calls. The server keeps the tokens the backend has consumed (the
// tape) and when a new prompt starts with all of them it continues from that position instead of
// re-reading; usage.prompt_tokens_details.cached_tokens reports how many positions were reused.
//
// A prompt that diverges inside the tape (Qwen3.6's template drops the reasoning of every assistant turn
// before the last user query, so every new user turn does) is not re-read from the start when the backend
// can snapshot its recurrent state: while it reads a prompt the server takes a snapshot at message
// boundaries (the token where `<|im_start|>` of a message begins), keeps the most recent ones with their
// positions, and on a divergence restores the newest one at or before it and reads on from there. K/V is
// positional, so nothing is copied or shifted: the rows past the restore point are overwritten as they are
// read. Without snapshot support a mismatch resets the backend and re-reads the prompt.
//
// Slots: when the backend can hold several conversations (Backend::set_slots) the server asks for Config::slots of them (default 1;
// as many as fit, leaving Config::vram_reserve_mib free). The slots are peers: each has its own tape and checkpoints, and they
// share one K/V pool (docs/paged-kv.md). A request goes to the slot that can reuse the most of its prompt (the whole tape
// if the prompt extends it, else the newest checkpoint at or before the divergence) provided that is more
// than the opening every conversation of an agent shares (the system prompt: up to the first message
// boundary); otherwise to the least recently used slot, which is taken over. A pin wins: the request field `id_slot` (llama-server's),
// or the header (Config::session_header, X-Limbo-Session): a number is a slot, any other value is a label bound to the slot its
// first request was routed to. Which slot served a request is in timings.slot.
//
// Prefix sharing: before a slot reads a prompt, the server looks at the other slots' tapes. A slot whose tape shares a prefix with the
// prompt lends it up to its newest checkpoint at or before the common prefix, or, when its whole tape is a prefix of the prompt, up to
// its tape's end (a snapshot of its live state is taken there): the borrowing slot points its page table at the lender's pages
// (a reference count, no copy but the page the cut falls in), restores the recurrent snapshot and reads only the rest. It works
// both ways: the sidekick reads the main's prefix, and the main adopts what the sidekick has read ahead.

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "limbo/chat.hpp"
#include "limbo/http.hpp"
#include "limbo/jinja.hpp"
#include "limbo/step.hpp"
#include "limbo/tok.hpp"
#include "limbo/vlog.hpp"

namespace limbo::serve {

// A forward pass the server can drive. Positions are consecutive from 0 after reset(); the server
// never skips or repeats one. The returned logits are valid until the next call.
//
// Only forward(token, pos) and n_vocab()/reset() are required. The rest are hooks with correct defaults
// that a backend overrides when it can do better: skip the output head for prompt tokens, return just the
// greedy token without copying the logits to the host, take the prompt in batches.
struct Backend {
    virtual ~Backend() = default;
    virtual int n_vocab() const = 0;
    virtual void reset() = 0;
    virtual const float* forward(int32_t token, int pos) = 0;
    virtual int n_ctx() const { return 0; }  // 0: no limit known
    virtual std::string describe() const { return {}; }  // for the startup line: "6 threads", "RTX 4060 Ti"

    // The same pass for a caller that will not read the logits (every prompt token but the last): the
    // layers run (they carry the state), the output head need not. Returns nullptr when skipped.
    virtual const float* forward(int32_t token, int pos, bool want_logits);

    // The same pass, and the greedy token of its logits (the lowest id on a tie) instead of the logits:
    // what the server calls at temperature 0. The default takes the argmax of forward()'s logits.
    virtual int32_t forward_greedy(int32_t token, int pos);

    // Reads tokens[0..n) at positions pos0.. with no logits. The server calls it with at most
    // prefill_chunk() tokens and looks after the client between calls. The default is forward(.., false)
    // per token; a backend with a batched prompt pass raises prefill_chunk() and overrides it.
    virtual void prefill(const int32_t* tokens, int n, int pos0);
    virtual int prefill_chunk() const { return 1; }
    // Said once before the prompt is read: it ends at position `end` (the last token, which runs the head, included). A backend whose
    // drafter reads only the last positions of a prompt may skip the earlier ones; the default ignores it.
    virtual void prompt_end(int) {}
    // Returns when every pass queued so far has run. A backend that queues work returns from prefill() before it is done; the server
    // waits for a prime's piece (default: nothing queued) so that stopping after a piece means the device is idle at the tape's end.
    virtual void sync() {}

    // ---- the tape: recurrent-state checkpoints and slots (defaults: unsupported, so a mismatch resets) ----
    //
    // A backend whose state has a part that cannot be rewound can keep copies of it in numbered slots.
    // set_snapshot_slots(want) makes `want` slots exist and returns how many do (0: none, the default).
    // snapshot(slot) copies the state as it is after every token consumed so far; restore(slot) puts it back,
    // and the next forward/prefill must be at the position the snapshot was taken at. Both return false when
    // they cannot, and then the server carries on as if the checkpoint did not exist.
    virtual int set_snapshot_slots(int) { return 0; }
    virtual int snapshot_slots() const { return 0; }
    virtual bool snapshot(int) { return false; }
    virtual bool restore(int) { return false; }
    // The smallest prompt piece the backend reads faster, and with the same arithmetic, as a batch. The server
    // does not cut a prompt into pieces shorter than this for a checkpoint nobody asked to be exact.
    virtual int prefill_batch_min() const { return 1; }

    // Slots: independent conversations (page table, recurrent state) that switch without a copy. set_slots(n) makes
    // n exist and returns how many do (1: only the one every backend has). select_slot(h) makes h the one
    // the other calls act on.
    virtual int set_slots(int) { return 1; }
    virtual int slots() const { return 1; }
    virtual bool select_slot(int h) { return h == 0; }
    // Optional state (extra slots, snapshot slots) is only taken while this many device bytes stay free; the
    // numbers behind the choice are reported so the server can say why it made what it made (0: no device memory).
    virtual void set_vram_reserve(uint64_t) {}
    virtual uint64_t vram_free() const { return 0; }
    virtual uint64_t slot_bytes() const { return 0; }      // device bytes of one extra slot
    virtual uint64_t snapshot_bytes() const { return 0; }  // device bytes of one snapshot slot

    // ---- the paged K/V pool (docs/paged-kv.md; defaults: no pool, every slot holds n_ctx on its own) ----
    struct Pool {
        int positions = 0;       // the pool holds this many positions for all slots together (0: no pool)
        int free_positions = 0;  // ... of which no slot has taken a page for
        int used_pages = 0, shared_pages = 0, total_pages = 0;
        int page = 256;          // positions to a page
        uint64_t copies = 0;     // pages copied since the start (a prefix cut inside a page, a write into a shared page)
    };
    virtual Pool pool() const { return {}; }
    virtual int slot_pages(int) const { return 0; }  // pages the slot's table holds (any slot)
    // The live slot keeps its positions < keep; the pages past them go back to the pool. Called whenever the server cuts a tape back.
    virtual void cut(int) {}
    // The live slot has pages for its first n positions (false: the pool has none left; what it took stays).
    virtual bool reserve(int) { return true; }
    // The live slot, which must be empty, starts as the first n positions of slot `from` (K/V only: whole pages are shared, the page the
    // cut falls in is copied); the caller restores the matching recurrent snapshot. False when it cannot.
    virtual bool share(int, int) { return false; }
    // Positions the live slot can still hold from position `at` on (its own pages' tail plus the pool's free pages).
    virtual int room(int) const { return 1 << 30; }

    // ---- speculative decoding (default: none) ----
    //
    // decode_spec(fed, pos, max_out, pick, out) reads `fed` at `pos` as forward() would, and may go further: it drafts tokens,
    // checks them in one pass and keeps the prefix `pick` accepts. It returns n in 1..max_out and fills out[0..n): out[0..n-1)
    // were read after `fed` (the state is at pos + n), out[n-1] is the next token and was not read. pick.row chooses the token
    // of a checked row from its logits and that row's draft (-1: none) and says whether it kept the draft; with pick.greedy the
    // backend takes the highest logit itself (no logits leave the device). A draft pick.stop names (an end of generation) is
    // never read: it ends the run as out[n-1]. Any other call on the backend ends what a run knows about the state.
    struct SpecPick {
        bool greedy = true;
        std::function<int32_t(const float* logits, int32_t draft, bool* kept)> row;
        std::function<bool(int32_t)> stop;
    };
    virtual bool spec_ready() const { return false; }
    virtual int decode_spec(int32_t fed, int pos, int max_out, const SpecPick& pick, int32_t* out);
    virtual std::string spec_stats() const { return {}; }  // for the request log line

    // ---- the verbose log (-v, -vv; include/limbo/vlog.hpp) ----
    // set_verbose(level): 0 off, 1 events, 2 adds a trace per speculation cycle (the backend writes it through vlog::line).
    virtual void set_verbose(int) {}
    // Lines for the start-up block of -v (geometry, K/V format and why, speculation costs): text without time stamp, one per line.
    virtual std::vector<std::string> startup_report() const { return {}; }
    // Device bytes by use, in the order of a table (weights, K/V pool, slots' state, checkpoints, MTP, scratch, ...); the server
    // adds the reserve and what is free. Empty: no device memory.
    struct MemUse {
        std::string what;
        uint64_t bytes = 0;
        std::string note;
    };
    virtual std::vector<MemUse> memory_report() const { return {}; }
};

// The CPU reference forward pass as a Backend. Borrows the Ref, or owns it (the second constructor).
class RefBackend final : public Backend {
public:
    explicit RefBackend(step::Ref& ref) : ref_(&ref) {}
    RefBackend(std::unique_ptr<step::Ref> own, std::string label) : own_(std::move(own)), ref_(own_.get()), label_(std::move(label)) {}
    int n_vocab() const override { return ref_->n_vocab(); }
    void reset() override { ref_->reset(); }
    const float* forward(int32_t token, int pos) override { return ref_->forward(token, pos); }
    const float* forward(int32_t token, int pos, bool want_logits) override { return ref_->forward(token, pos, want_logits); }
    int n_ctx() const override { return ref_->config().n_ctx; }
    std::string describe() const override { return label_; }

private:
    std::unique_ptr<step::Ref> own_;
    step::Ref* ref_;
    std::string label_;
};

struct SamplerParams {
    double temperature = 1.0;  // 0 (or less) = greedy
    int top_k = 0;             // 0 = off
    double top_p = 1.0;
    double min_p = 0.0;
    bool has_seed = false;
    uint64_t seed = 0;
};

// Temperature first, then top_k, top_p, min_p over the remaining distribution (the order Hugging Face
// and vLLM use). Greedy ties go to the lowest token id.
class Sampler {
public:
    Sampler(const SamplerParams& p, uint64_t fallback_seed);
    int32_t sample(const float* logits, int n_vocab);
    // A draw from the same distribution for a row that carries a draft (Backend::decode_spec): the draft is kept with its
    // probability p(draft), otherwise the token is drawn from the distribution without the draft, so the result is distributed
    // as sample()'s. *kept says which. Temperature 0: the highest logit, kept when it is the draft. draft < 0: sample().
    int32_t sample_draft(const float* logits, int n_vocab, int32_t draft, bool* kept);
    // What sample() returns at temperature 0: the highest logit, the lowest id on a tie.
    static int32_t argmax(const float* logits, int n_vocab);

private:
    SamplerParams p_;
    uint64_t s_[4];
    uint64_t next();
    double uniform();  // [0, 1)
    std::vector<int32_t> idx_;
    // top_k / top_p / min_p: idx_[0..kept) the ids that stay, in the order of their logits; *total their weight (each
    // exp((logit - lmax) / temperature)). Returns kept.
    size_t kept_set(const float* lg, int n, float lmax, double* total);
};

struct Config {
    std::string model_id = "limbo";
    int default_max_tokens = 0;     // when the request names none (0: until the context is full, as llama-server); clamped to it
    SamplerParams sampling;         // defaults for fields a request leaves out (OpenAI's)
    bool log = true;                // one line per request on stderr
    bool log_prompts = false;       // also the rendered prompt
    int verbose = 0;                // 1 (-v): start-up block, per-request header, prompt pieces, decode summary, slot events; 2 (-vv): + every speculation cycle and yield
    int keepalive_ms = 5000;        // SSE comment while the prompt is being read
    int checkpoints = 8;            // recurrent-state snapshot slots to ask the backend for (0: none, mismatch = reset)
    int checkpoint_gap = 64;        // tokens: a plain message boundary closer than this to the last cut or to the end of the prompt is not
                                    // checkpointed (a cut costs a pass over the weights, ~40 ms on rung 1: it must save more than that)
    int slots = 1;                  // slots to ask the backend for: as many as fit, up to this (1: one conversation at a time)
    int vram_reserve_mib = 256;     // device memory left free for the desktop when slots and snapshot slots are taken
    bool speculate = true;          // decode through Backend::decode_spec when the backend has it
    std::vector<std::string> api_keys;  // --api-key (repeatable): when any is set, every endpoint but /health wants `Authorization: Bearer <key>` (401 otherwise)
    std::string session_header = "x-limbo-session";  // lower case: the http layer lower-cases header names
    // --preserve-thinking (on): the template's preserve_thinking is true unless the request sets it (Qwen3.6/3.8 then render an
    // earlier answer's thinking, not only the current tool round's), and an assistant message that comes back without its
    // reasoning_content gets the thinking this server generated for it. The history then reads as the model wrote it -- its
    // own reasoning kept, and the prompt equal to the slot's tape, so the turn is read from the cache instead of again.
    bool preserve_thinking = true;
};

// What the server has done, for tests and the log. Written by the serving thread, readable anywhere.
struct Stats {
    std::atomic<uint64_t> requests{0}, completed{0}, aborted{0}, rejected{0};
    std::atomic<int> last_prompt_tokens{0}, last_cached_tokens{0}, last_completion_tokens{0};
    std::atomic<int> tape_tokens{0};
    std::atomic<uint64_t> checkpoints{0}, restores{0};  // snapshots taken, restores that served a diverging prompt
    std::atomic<int> last_restored_from{-1};            // position the last request restored to (-1: it did not)
    std::atomic<int> last_slot{0};                      // the slot that served the last request
    std::atomic<int> last_adopted_from{-1};             // the slot whose prefix the last request took (-1: none)
    std::atomic<uint64_t> adoptions{0};                 // requests that took another slot's prefix
    // /metrics (llama-server's `llamacpp:` names): tokens read and generated, and the microseconds spent on each
    std::atomic<uint64_t> prompt_tokens{0}, prompt_us{0}, predicted_tokens{0}, predicted_us{0};
    std::atomic<uint64_t> preempted{0};                 // primes that stopped reading for a waiting request (503, Retry-After)
};

struct ChatRequest {
    jinja::Value messages;
    jinja::Value tools;       // the list as sent (undefined when none)
    bool tools_on = false;    // tools were sent and tool_choice is not "none"
    jinja::Value kwargs;      // chat_template_kwargs
    bool stream = false;
    bool include_usage = false;
    bool cache_prompt = true;  // llama-server's field: false reads the whole prompt from a reset backend
    int id_slot = -1;          // llama-server's field: the slot to run in (-1: routed by prefix)
    int max_tokens = 0;       // 0 = the server default
    SamplerParams sampling;
    std::vector<std::string> stop;
    std::string model;
};

struct Prompt {
    std::string text;
    std::vector<int32_t> tokens;
    bool opens_think = false;  // the model starts inside a think block
};

class Service {
public:
    Service(Backend& backend, const tok::Tokenizer& tok, const chat::Renderer& tpl, Config cfg);

    // The http handler: routes the three endpoints.
    void handle(const http::Request& rq, http::Conn& conn);

    // The two steps that turn a request body into a prompt, public so a test can compute what a
    // request will feed the backend.
    bool parse_request(const jinja::Value& body, ChatRequest* out, std::string* err) const;
    bool build_prompt(const ChatRequest& rq, Prompt* out, std::string* err) const;

    const Stats& stats() const { return stats_; }
    size_t tape_size(int slot = 0) const { return size_t(slot) < slots_.size() ? slots_[size_t(slot)].tape.size() : 0; }
    int checkpoints_held(int slot = 0) const { return size_t(slot) < slots_.size() ? int(slots_[size_t(slot)].ck.size()) : 0; }
    int checkpoint_slots() const { return n_snaps_; }  // what the backend gave
    int slot_count() const { return int(slots_.size()); }
    // What was decided at start-up about slots and snapshot slots, with the numbers (also printed when cfg.log).
    const std::string& startup_note() const { return note_; }

private:
    // The thinking of this server's recent answers, by what the client will send back of them: the answer (content and tool
    // calls) and the history before it (each message's role, text and calls, its thinking left out), so the same answer at two
    // places in a conversation stays two answers. The serving thread's own, newest last, kRecall at most.
    static constexpr size_t kRecall = 256;
    struct Recall {
        uint64_t before = 0;
        std::string key, reasoning;
    };
    std::vector<Recall> recall_;
    void remember_thinking(const jinja::Value& history, const std::string& key, const std::string& reasoning);
    jinja::Value with_thinking(const jinja::Value& messages) const;

    void chat_completions(const http::Request& rq, http::Conn& conn);
    void reply_error(http::Conn& conn, int status, const std::string& msg, const char* type = "invalid_request_error");
    bool is_eog(int32_t t) const;
    bool authorized(const http::Request& rq) const;  // true when no key is configured or the bearer token is one of them
    std::string new_id(const char* prefix);

    // One checkpoint: the backend's state after the first `pos` tokens of the slot's tape, in snapshot slot `snap`.
    // kind: Plain (any other message boundary), Head (the generation prompt of a user turn: where the next user
    // turn's re-render diverges), Root (the end of the opening, the first message boundary: a new conversation
    // with the same system prompt restores it).
    enum class Kind : uint8_t { Plain, Head, Root };
    struct Ckpt {
        int pos = 0, snap = -1;
        Kind kind = Kind::Plain;
        uint64_t stamp = 0;
    };
    struct Slot {
        std::vector<int32_t> tape;  // the tokens the slot has consumed since its last reset
        std::vector<Ckpt> ck;
        uint64_t used = 0;          // use_clock_ at its last request (0: never)
    };
    // What a slot can reuse of a prompt: `cont` when the prompt extends the whole tape (keep = tape size), else
    // `ck`, the newest checkpoint at or before the divergence (keep = its position), else nothing.
    struct Reuse {
        size_t keep = 0, lcp = 0;
        const Ckpt* ck = nullptr;
        bool cont = false;
    };
    Reuse plan_reuse(const Slot& h, const std::vector<int32_t>& toks) const;
    int route(const http::Request& rq, const std::vector<int32_t>& toks, size_t opening, int id_slot, const char** how);
    int acquire_snap(Slot& s, Kind kind);
    void release_ckpts(Slot& s, int above_pos);  // gives back every checkpoint with pos > above_pos (-1: all)
    void take_ckpt(Slot& s, size_t pos, Kind kind);
    // A prime (max_tokens 1) that stops reading because another request waits: the slot keeps its tape, the client gets a 503 to retry.
    void preempt(const http::Request& rq, http::Conn& conn, const Slot& s, int n_prompt, int cached, double t_start);
    // Prefix sharing between slots (see the top of this file). `from`: the lender (-1: none), `pos`: how many positions it lends,
    // `ck`: the checkpoint it lends at, or null when it must first snapshot its live state at its tape's end.
    struct Lend {
        int from = -1;
        size_t pos = 0;
        const Ckpt* ck = nullptr;
    };
    Lend find_lender(int slot, const std::vector<int32_t>& toks, size_t own_keep) const;
    bool adopt(int slot, const Lend& l);  // `slot` is the backend's live slot; false: nothing was taken, the slot is as it was or reset
    void select_slot_(int s);             // be_.select_slot + cur_
    void startup_block();                 // -v: the start-up lines
    static const char* kind_name(Kind k);
    std::string pool_note() const;        // "pool 182/412 pages, 3 shared"
    void cut_(int keep);                  // Backend::cut, and says how many pages went back
    void vline(const char* tag, const std::string& text, vlog::Tone tone = vlog::Tone::Plain) const;  // -v: one line of the request being served
    void handle_props(http::Conn& conn);
    void handle_slots(const http::Request& rq, http::Conn& conn);
    void erase_slot(int id, http::Conn& conn);

    Backend& be_;
    const tok::Tokenizer& tok_;
    const chat::Renderer& tpl_;
    Config cfg_;
    std::vector<Slot> slots_;        // as many as the backend made
    std::vector<std::pair<std::string, int>> pins_;  // X-Limbo-Session labels -> slot
    std::string note_;
    int vl_ = 0;                     // verbosity (Config::verbose, 0 when the log is off)
    long rid_ = 0;                   // the request being served (its number in the log; 0: none)
    int cur_ = 0;                    // the slot the backend has selected
    int busy_ = -1;                  // the slot a request is running in (GET /slots answers while it runs)
    int n_snaps_ = 0;                // checkpoint (snapshot) slots the backend gave
    uint64_t use_clock_ = 0;
    std::vector<int> free_snaps_;
    int32_t boundary_tok_ = -1;      // <|im_start|>; none: no checkpoints
    uint64_t stamp_ = 0;
    std::vector<int32_t> eog_;
    Stats stats_;
    uint64_t counter_ = 0;
};

// ---- the server's main, shared by every entry point (CPU `limbo_serve`, GPU `limbo_serve_gpu`) ----

// What the command line asked for; the backend factory reads it.
struct MainOptions {
    std::string model;
    bool ctx_auto = false;      // --ctx auto was in force (the backend may shrink it to keep its drafters)
    std::string hf;             // -hf org/repo[:QUANT]: the model by its Hugging Face name (include/limbo/hf.hpp)
    std::string host = "127.0.0.1";
    std::string name;
    int port = 8080;
    int ctx = -1;               // -1 auto: the GPU entry 65536 (the stack's budget) or the model's own context if shorter; the CPU entry 4096
    int threads = 6;
    double min_free_gb = 0;     // the GPU entry's VRAM floor (--min-free-gb; 0: the model's weights + 1 GiB); the CPU entry has none
    std::string kv = "auto";    // the GPU entry's K/V cache format (--kv auto|f16|q8|q4; limbo::cuda::GpuOptions)
    int prefill_planes = 2;     // the GPU entry's batched-prefill activation planes (--prefill-planes 1|2|3)
    std::string spec = "auto";  // the GPU entry's speculation (--spec auto|mtp|ngram|off; auto: both, per cycle: MTP when the model has the block and it fits)
    int ngram_n = 0;            // --ngram-n: tokens that must repeat before an ngram draft is made (0: the backend's default)
    int ngram_max = 0;          // --ngram-max: the longest ngram draft, 1..7 (0: 7)
    std::string draft_quant = "auto";  // --draft-quant auto|native|q8|bf16
    int draft_vocab = 0;        // --draft-vocab N
    int draft_k = 0;            // --draft-k N
    std::string draft_model;    // --draft-model / --dflash: the DFlash2 drafter (a directory with config.json and model.safetensors, or the .safetensors file)
    bool no_draft = false;      // --draft-model none: not even the zoo's
    std::vector<std::string> draft_alts;   // the zoo's smaller quants of the drafter, best first (the backend fits the best one)
    Config cfg;
};

struct MainSpec {
    const char* prog = "limbo_serve";
    bool gpu = false;  // accepts --min-free-gb
    // Opens the forward pass for opts.model. Returns 0 with *out set, or an exit status with *err set (the
    // message is printed as `<prog>: <err>`). The Backend owns everything it needs.
    std::function<int(const MainOptions&, std::unique_ptr<Backend>* out, std::string* err)> open;
};

// Parses argv, opens the GGUF (tokenizer, chat template, name), the backend, and serves until SIGINT or
// SIGTERM. Returns the process exit status: 0 clean stop, 2 usage, 1 any load failure the backend does not
// name itself.
int run_main(int argc, char** argv, const MainSpec& spec);

}  // namespace limbo::serve
