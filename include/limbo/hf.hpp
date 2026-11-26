// Copyright (c) 2026 ITERFORM. All rights reserved.
// include/limbo/hf.hpp -- a model by its Hugging Face name, and the drafters that belong to it.
//
//   limbo -hf org/repo[:QUANT]
//
// The GGUF is looked up in the local model folders (LIMBO_MODEL_DIRS, colon-separated, default ~/models: a file of the same
// name anywhere below them), then in the Hugging Face cache, and downloaded into the cache only when neither has it (after a
// check that the disk holds it). huggingface_hub does the listing and the download (python3 -P; its token, cache and resume).
// LIMBO_HF_DRY=1: say what would be downloaded and stop.
//
// The drafter zoo matches a model by what it is, not by its name: the architecture and its geometry (embedding width,
// vocabulary), so a fine-tune of a model the zoo knows gets that model's drafter. The speculation policy weighs every
// source per cycle by tokens per ms, so a drafter that suits a fine-tune less is simply chosen less.
#pragma once

#include <string>
#include <vector>

namespace limbo::gguf {
class File;
}

namespace limbo::hf {

// "org/repo[:QUANT]" -> the local path of the model's GGUF (its first shard; every shard present). QUANT picks the file whose
// name ends in it (IQ3_XXS, Q4_K_M, IQ1_M, ...); a file that adds "-mtp" (the same weights with their MTP block) is preferred.
// Without QUANT the repo must hold one model file, or one that is already here.
bool model(const std::string& spec, std::string* path, std::string* err);

// One named file of a repo (a drafter GGUF), or a whole small repo (a safetensors drafter: its directory).
bool file(const std::string& repo, const std::string& name, std::string* path, std::string* err);
bool repo_dir(const std::string& repo, std::string* path, std::string* err);

// The drafter the zoo holds for this model ("" kind: none; the model's own MTP block and the ngram tape are not drafters
// here, the backend finds those itself). Fetched like a model; *from says where it came from.
struct Drafter {
    std::string kind;   // "dflash" or "dspark"
    std::string path;   // the best of them
    std::vector<std::string> alts;   // the same drafter in smaller quants, best first: the backend takes the best that fits
    std::string from;
};
bool zoo(const gguf::File& f, Drafter* d, std::string* err);

}  // namespace limbo::hf
