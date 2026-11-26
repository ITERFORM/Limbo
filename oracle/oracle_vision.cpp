// Copyright (c) 2026 ITERFORM. All rights reserved.
// oracle/oracle_vision.cpp — the image embeddings llama.cpp's mtmd computes, written to a file.
//
// A measuring device, not engine code (docs/CLEANROOM.md rule 0): it links the operator's llama.cpp build and is never linked
// into liblimbo.a. Only its numbers cross the line: tools/vision_check compares them to Limbo's tower.
//
//   oracle_vision --model llm.gguf --mmproj mmproj.gguf --image a.png --out a.emb [--max-tokens N] [-t threads] [--gpu]
//
// The LLM is opened vocab-only (mtmd needs it for the embedding width), nothing of it is read. Output: "LVE1", then int32
// n_tokens, n_embd, nx, ny, then n_tokens * n_embd floats (the projector output, one row per image token).
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"

struct Nodes {
    std::vector<std::string> want;
    std::string dir;
};

// Saves the named graph nodes (f32, contiguous) as <dir>/<name>.f32 when the projector's graph computes them.
static bool node_cb(struct ggml_tensor* t, bool ask, void* user) {
    Nodes* n = static_cast<Nodes*>(user);
    const std::string name = ggml_get_name(t);
    bool hit = false;
    for (const std::string& w : n->want) hit = hit || w == name;
    if (ask) return hit;
    if (!hit || t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) return true;
    std::vector<float> buf(ggml_nelements(t));
    ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));
    FILE* f = std::fopen((n->dir + "/" + name + ".f32").c_str(), "wb");
    if (f) {
        std::fwrite(buf.data(), sizeof(float), buf.size(), f);
        std::fclose(f);
    }
    return true;
}

int main(int argc, char** argv) {
    std::string model, mmproj, image, out, node_dir, node_list;
    int max_tokens = 1024, threads = 8, n_embd_arg = 5120;
    bool gpu = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--model") model = next();
        else if (a == "--mmproj") mmproj = next();
        else if (a == "--image") image = next();
        else if (a == "--out") out = next();
        else if (a == "--max-tokens") max_tokens = std::atoi(next().c_str());
        else if (a == "-t") threads = std::atoi(next().c_str());
        else if (a == "--n-embd") n_embd_arg = std::atoi(next().c_str());
        else if (a == "--node-dir") node_dir = next();
        else if (a == "--nodes") node_list = next();
        else if (a == "--gpu") gpu = true;
    }
    if (model.empty() || mmproj.empty() || image.empty() || out.empty()) {
        std::fprintf(stderr, "usage: oracle_vision --model llm.gguf --mmproj mmproj.gguf --image f --out f.emb [--max-tokens N] [-t N] [--gpu]\n");
        return 2;
    }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    mp.n_gpu_layers = 0;
    llama_model* lm = llama_model_load_from_file(model.c_str(), mp);
    if (!lm) {
        std::fprintf(stderr, "cannot open %s\n", model.c_str());
        return 1;
    }
    mtmd_context_params cp = mtmd_context_params_default();
    cp.use_gpu = gpu;
    cp.n_threads = threads;
    cp.print_timings = true;
    cp.warmup = false;
    cp.image_max_tokens = max_tokens;
    Nodes nodes;
    if (!node_dir.empty()) {
        nodes.dir = node_dir;
        for (size_t b = 0; b <= node_list.size();) {
            const size_t e = std::min(node_list.find(',', b), node_list.size());
            if (e > b) nodes.want.push_back(node_list.substr(b, e - b));
            b = e + 1;
        }
        cp.cb_eval = node_cb;
        cp.cb_eval_user_data = &nodes;
    }
    mtmd_context* ctx = mtmd_init_from_file(mmproj.c_str(), lm, cp);
    if (!ctx) {
        std::fprintf(stderr, "cannot open the projector %s\n", mmproj.c_str());
        return 1;
    }
    mtmd_helper_bitmap_wrapper bw = mtmd_helper_bitmap_init_from_file(ctx, image.c_str(), false, mtmd_helper_init_opt_default());
    if (!bw.bitmap) {
        std::fprintf(stderr, "cannot read the image %s\n", image.c_str());
        return 1;
    }
    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    const std::string text = std::string("<|vision_start|>") + mtmd_default_marker() + "<|vision_end|>";
    mtmd_input_text it{text.c_str(), text.size(), false, true};
    const mtmd_bitmap* bms[1] = {bw.bitmap};
    if (mtmd_tokenize(ctx, chunks, &it, bms, 1) != 0) {
        std::fprintf(stderr, "tokenize failed\n");
        return 1;
    }
    const mtmd_input_chunk* img = nullptr;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
        const mtmd_input_chunk* c = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_IMAGE) img = c;
    }
    if (!img) {
        std::fprintf(stderr, "no image chunk\n");
        return 1;
    }
    const mtmd_image_tokens* tk = mtmd_input_chunk_get_tokens_image(img);
    const int n = int(mtmd_image_tokens_get_n_tokens(tk));
    const mtmd_decoder_pos last = mtmd_image_tokens_get_decoder_pos(tk, 0, size_t(n - 1));
    const int ny = int(last.y) + 1, nx = n / ny;
    if (mtmd_encode_chunk(ctx, img) != 0) {
        std::fprintf(stderr, "encode failed\n");
        return 1;
    }
    int n_embd = llama_model_n_embd(lm);
    if (n_embd <= 0) n_embd = n_embd_arg;  // a vocab-only model reports none
    const float* e = mtmd_get_output_embd(ctx);
    FILE* f = std::fopen(out.c_str(), "wb");
    if (!f) return 1;
    std::fwrite("LVE1", 1, 4, f);
    const int32_t hdr[4] = {n, n_embd, nx, ny};
    std::fwrite(hdr, sizeof hdr, 1, f);
    std::fwrite(e, sizeof(float), size_t(n) * size_t(n_embd), f);
    std::fclose(f);
    std::printf("oracle_vision: %s -> %d tokens (%d x %d), n_pos %d, %d floats each\n", image.c_str(), n, nx, ny, int(mtmd_image_tokens_get_n_pos(tk)), n_embd);
    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bw.bitmap);
    mtmd_free(ctx);
    llama_model_free(lm);
    return 0;
}
