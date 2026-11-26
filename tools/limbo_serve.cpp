// Copyright (c) 2026 ITERFORM. All rights reserved.
// tools/limbo_serve.cpp — `limbo serve`: the OpenAI-compatible endpoint over the CPU reference forward pass.
//
//   limbo_serve -m model.gguf [--host 127.0.0.1] [--port 8090] [--ctx 4096] [--threads 6]
//               [--model-name NAME] [--max-tokens N] [--temperature T] [--top-k K] [--top-p P] [--min-p P]
//               [--log-prompts] [--quiet]
//
// The chat template is the GGUF's own tokenizer.chat_template. SIGINT/SIGTERM stop the server cleanly.
// The server itself is limbo::serve::run_main (src/serve.cpp), shared with the GPU entry limbo_serve_gpu.
#include "limbo/serve.hpp"
#include "limbo/step.hpp"

int main(int argc, char** argv) {
    limbo::serve::MainSpec spec;
    spec.prog = "limbo_serve";
    spec.open = [](const limbo::serve::MainOptions& o, std::unique_ptr<limbo::serve::Backend>* out, std::string* err) {
        auto ref = limbo::step::Ref::open({o.model}, o.ctx, o.threads, err);
        if (!ref) return 1;
        *out = std::make_unique<limbo::serve::RefBackend>(std::move(ref), std::to_string(o.threads) + " threads");
        return 0;
    };
    return limbo::serve::run_main(argc, argv, spec);
}
