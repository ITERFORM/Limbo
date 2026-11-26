// Copyright (c) 2026 ITERFORM. All rights reserved.
// tools/limbo_serve_gpu.cu — `limbo serve` on the GPU: the same server and flags as limbo_serve, with the
// rung-1 forward pass on the CUDA device (limbo::cuda::Gpu) instead of the CPU reference.
//
//   limbo_serve_gpu -m model.gguf [--host 127.0.0.1] [--port 8090] [--ctx 4096] [--model-name NAME]
//                   [--max-tokens N] [--temperature T] [--top-k K] [--top-p P] [--min-p P]
//                   [--log-prompts] [--quiet] [-v | -vv] [--min-free-gb G] [--kv auto|f16|q8|q4] [--prefill-planes 2]
//                   [--spec auto|mtp|ngram|dflash|dspark|off] [--ngram-n N] [--ngram-max K] [--draft-model DIR|FILE] [--slots 1] [--checkpoints 8] [--vram-reserve-mib 768]
//
// One K/V pool (docs/paged-kv.md) holds every slot's cache: --ctx positions in all, like llama.cpp's -kvu. --slots N (default 1;
// --sessions is an alias) makes N conversations that share it, a slot that reads a prompt another slot has read takes its pages by
// reference; Ariadne runs --slots 2 (slot 1 the main, slot 0 the sidekick) and finds /props, /slots, POST /slots/{id}?action=erase and
// the request field id_slot as llama-server has them. --kv picks the pool's format; auto (the default) takes the most exact of f16,
// q8 and q4 whose pool covers --ctx with the MTP block, two checkpoint slots per slot and the reserve beside it, and when even q4 does
// not, the pool is what fits (the start-up line says how many positions) and a request needing more gets a 400
// (include/limbo/cuda/backend.hpp GpuOptions). --ctx goes to 262144.
// --prefill-planes: 2 (the default) agrees with the llama.cpp captures as often as the exact 3 and is a fifth faster; 1 is
// llama.cpp's own activation rounding and faster again (bench/long-context-2026-11-21.md).
// --spec auto (the default) decodes speculatively from two sources, chosen per cycle by expected tokens per millisecond: the model's MTP
// block when the file has it and it fits (after the slots, before the cache's precision), and ngram drafts, a copy of what followed
// the same --ngram-n tokens earlier in the slot's own tape (a function reproduced with a rename, a file quoted back; src/ngram.cpp).
// --draft-model DIR (alias --dflash; a directory with config.json + model.safetensors of incoai/Qwen3.6-35B-A3B-DFlash2, or the file) adds the
// DFlash2 block drafter as a third source of --spec auto (docs/dflash2.md): 7 drafts in one pass over a context made of the target's
// hidden states; --spec dflash uses it alone. It takes ~0.8 GiB + 48 MiB a slot of VRAM, counted by --kv auto like the MTP block.
// Either way the draft is checked in one window of at most 8 rows: exact for greedy and for sampling. mtp / ngram use one source, off none.
// -v prints a readable event log on stderr (start-up block with the VRAM breakdown, per request a header, prompt pieces, decode summary,
// slot events); -vv adds a line per speculation cycle and per yield. Colour only on a terminal (NO_COLOR is honoured); without -v
// the one line per request stays as it was (include/limbo/vlog.hpp).
// --api-key KEY (repeatable) makes every endpoint but /health want `Authorization: Bearer KEY` (llama-server's flag): 401 otherwise,
// before any routing. Off unless set; the key is visible in `ps`, so a file-based key is for later.
// --threads is accepted and ignored. The engine copies the weights to VRAM at start-up and refuses to load, exit status 3,
// when the card has less free than --min-free-gb (default: the weights + 1 GiB): the desktop and other jobs share it. Nothing here takes a lock; wrap the whole process in `flock` where another job needs the card.
#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>

#include "limbo/cuda/backend.hpp"
#include "limbo/model.hpp"
#include "limbo/serve.hpp"

int main(int argc, char** argv) {
    limbo::serve::MainSpec spec;
    spec.prog = "limbo_serve_gpu";
    spec.gpu = true;
    spec.open = [](const limbo::serve::MainOptions& o, std::unique_ptr<limbo::serve::Backend>* out, std::string* err) {
        std::string e;
        // An inkling2_loop file runs its own device pass (InkGpu), not the Qwen family's Gpu.
        {
            std::unique_ptr<limbo::model::Model> probe = limbo::model::Model::open({o.model}, &e);
            if (!probe) {
                *err = "the model refused to load: " + e;
                return 3;
            }
            if (probe->geometry().arch == "inkling2_loop") {
                // --ctx max (0) fits the context to the card, keeping --vram-reserve-mib free
                // --draft-model (with --spec auto or dspark): Inkling-2's own DSpark drafter
                const bool want_draft = !o.draft_model.empty() && o.spec != "off";
                auto b = limbo::cuda::InkBackend::open(
                    {o.model}, o.ctx, static_cast<uint64_t>(o.min_free_gb * 1e9), &e,
                    static_cast<uint64_t>(std::max(0, o.cfg.vram_reserve_mib)) << 20,
                    want_draft ? o.draft_model : std::string(), o.draft_k > 0 ? o.draft_k : 3, o.kv == "q8");
                if (!b) {
                    *err = "the inkling2 device pass refused to load: " + e;
                    return 3;
                }
                *out = std::move(b);
                return 0;
            }
        }
        if (o.ctx <= 0) {
            *err = "--ctx max is the inkling2 pass's fit; this architecture sizes its pool with --kv auto: give --ctx N (up to 262144)";
            return 2;
        }
        limbo::cuda::GpuOptions plan;
        plan.want = o.kv;
        plan.slots = o.cfg.slots;
        plan.checkpoints = std::min(o.cfg.checkpoints, 2);  // what serve::Service keeps per slot before it adds a second
        plan.reserve = static_cast<uint64_t>(std::max(0, o.cfg.vram_reserve_mib)) << 20;
        plan.ctx_floor = o.ctx_auto ? 49152 : 0;   // an agent's session reaches ~37K; below 48K a drafter is not worth the 400s
        plan.prefill_planes = o.prefill_planes;
        plan.spec = o.spec;
        plan.ngram_n = o.ngram_n;
        plan.ngram_max = o.ngram_max;
        plan.draft_model = o.draft_model;
        plan.draft_alts = o.draft_alts;
        plan.draft_quant = o.draft_quant;
        plan.draft_vocab = o.draft_vocab;
        plan.draft_k = o.draft_k;
        auto b = limbo::cuda::GpuBackend::open({o.model}, o.ctx, static_cast<uint64_t>(o.min_free_gb * 1e9), &e, plan);
        if (!b) {
            *err = "the GPU engine refused to load: " + e;
            return 3;
        }
        *out = std::move(b);
        return 0;
    };
    return limbo::serve::run_main(argc, argv, spec);
}
