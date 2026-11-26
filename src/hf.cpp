// Copyright (c) 2026 ITERFORM. All rights reserved.
// src/hf.cpp -- `-hf org/repo[:QUANT]` and the drafter zoo (include/limbo/hf.hpp).
#include "limbo/hf.hpp"

#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <regex>
#include <string>
#include <vector>

#include "limbo/gguf.hpp"

namespace limbo::hf {

namespace {

// The helper huggingface_hub runs: mode, repo, argument in argv (never pasted into code); the answer is one path on
// stdout, every message on stderr (the download's progress included), the exit status 0 or 2.
constexpr const char* kHelper = R"PY(
import os, re, shutil, sys
mode, repo, arg = sys.argv[1], sys.argv[2], sys.argv[3]
def fail(msg):
    print("limbo -hf: " + msg, file=sys.stderr)
    sys.exit(2)
try:
    from huggingface_hub import HfApi, hf_hub_download, try_to_load_from_cache, constants
except ImportError:
    fail("needs huggingface_hub (pip install --user huggingface_hub)")
dirs = [os.path.expanduser(d) for d in os.environ.get("LIMBO_MODEL_DIRS", "~/models").split(":") if d]
def local(name):
    base = os.path.basename(name)
    for d in dirs:
        for root, _, files in os.walk(d, followlinks=True):
            if base in files:
                return os.path.realpath(os.path.join(root, base))
    return None
def listing():
    try:
        return {s.rfilename: (s.size or 0) for s in HfApi().model_info(repo, files_metadata=True).siblings}
    except Exception as e:
        root = os.path.join(constants.HF_HUB_CACHE, "models--" + repo.replace("/", "--"), "snapshots")
        out = {}
        for snap in sorted(os.listdir(root)) if os.path.isdir(root) else []:
            base = os.path.join(root, snap)
            for r, _, fs in os.walk(base):
                for f in fs:
                    out.setdefault(os.path.relpath(os.path.join(r, f), base), 0)
        if not out:
            fail("%s: cannot list it (%s) and the cache holds nothing of it" % (repo, type(e).__name__))
        return out
def have(f, look_local):
    p = local(f) if look_local else None
    if p:
        return p
    c = try_to_load_from_cache(repo, f)
    return c if isinstance(c, str) and os.path.exists(c) else None
def fetch(files, sizes, look_local=True):
    missing = [f for f in files if not have(f, look_local)]
    if missing:
        need = sum(sizes.get(f, 0) for f in missing)
        if os.environ.get("LIMBO_HF_DRY"):
            print("limbo -hf: would download %.1f GiB from %s: %s" % (need / 2**30, repo, ", ".join(missing)), file=sys.stderr)
            sys.exit(3)
        os.makedirs(constants.HF_HUB_CACHE, exist_ok=True)
        free = shutil.disk_usage(constants.HF_HUB_CACHE).free
        if need + (2 << 30) > free:
            fail("%s: %.1f GiB to download, %.1f GiB free on the cache's disk (2 GiB stay free)" % (repo, need / 2**30, free / 2**30))
        print("limbo -hf: downloading %d file(s), %.1f GiB, from %s" % (len(missing), need / 2**30, repo), file=sys.stderr)
    return [have(f, look_local) or hf_hub_download(repo, f) for f in files]
sizes = listing()
shard = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")
if mode == "model":
    skip = re.compile(r"mmproj|dflash|dspark|imatrix", re.I)
    firsts = [f for f in sizes if f.endswith(".gguf") and not skip.search(os.path.basename(f))
              and (not shard.search(f) or shard.search(f).group(1) == "00001")]
    def stem(f):
        b = os.path.basename(f)
        return shard.sub("", b) if shard.search(b) else b[:-5]
    q = arg.lower()
    if q:
        exact = [f for f in firsts if stem(f).lower().endswith(q) or stem(f).lower().endswith(q + "-mtp")]
        cand = exact or [f for f in firsts if q in f.lower()]
        mtp = [f for f in cand if stem(f).lower().endswith("-mtp")]
        cand = mtp or cand
    else:
        cand = firsts
    if len(cand) > 1:
        here = [f for f in cand if have(f, True)]
        cand = here if here else cand
    if not cand:
        fail("%s has no model file%s; it holds: %s" % (repo, " ending in " + arg if arg else "", ", ".join(sorted(os.path.basename(f) for f in firsts)) or "no GGUF"))
    if len(cand) > 1:
        fail("%s holds %d model files; name one with :QUANT (%s)" % (repo, len(cand), ", ".join(sorted(stem(f) for f in cand))))
    first = cand[0]
    m = shard.search(first)
    files = [first] if not m else [first[:m.start()] + "-%05d-of-%s.gguf" % (k, m.group(2)) for k in range(1, int(m.group(2)) + 1)]
    print(fetch(files, sizes)[0])
elif mode == "file":
    print(fetch([arg], sizes)[0])
elif mode == "repo":
    files = [f for f in sizes if f.endswith((".json", ".safetensors"))]
    if "config.json" not in files:
        fail("%s holds no config.json" % repo)
    paths = fetch(files, sizes, look_local=False)
    print(os.path.dirname(paths[files.index("config.json")]))
)PY";

bool valid_repo(const std::string& r) {
    static const std::regex re(R"(^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$)");
    return std::regex_match(r, re);
}

// python3 -P -c kHelper mode repo arg: stdout's last line, or false with *err.
bool helper(const char* mode, const std::string& repo, const std::string& arg, std::string* out, std::string* err) {
    int fd[2];
    if (pipe(fd) != 0) {
        if (err) *err = "-hf: pipe failed";
        return false;
    }
    const pid_t pid = fork();
    if (pid < 0) {
        close(fd[0]);
        close(fd[1]);
        if (err) *err = "-hf: fork failed";
        return false;
    }
    if (pid == 0) {
        dup2(fd[1], 1);
        close(fd[0]);
        close(fd[1]);
        std::vector<char*> av = {const_cast<char*>("python3"), const_cast<char*>("-P"), const_cast<char*>("-c"),
                                 const_cast<char*>(kHelper), const_cast<char*>(mode), const_cast<char*>(repo.c_str()),
                                 const_cast<char*>(arg.c_str()), nullptr};
        execvp("python3", av.data());
        std::fprintf(stderr, "limbo -hf: cannot run python3\n");
        _exit(127);
    }
    close(fd[1]);
    std::string text;
    char buf[4096];
    for (ssize_t n; (n = read(fd[0], buf, sizeof buf)) > 0;) text.append(buf, size_t(n));
    close(fd[0]);
    int st = 0;
    waitpid(pid, &st, 0);
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) text.pop_back();
    const size_t nl = text.rfind('\n');
    const std::string last = nl == std::string::npos ? text : text.substr(nl + 1);
    if (WIFEXITED(st) && WEXITSTATUS(st) == 3) {
        if (err) *err = "-hf " + repo + ": LIMBO_HF_DRY is set, nothing downloaded";
        return false;
    }
    if (!WIFEXITED(st) || WEXITSTATUS(st) != 0 || last.empty()) {
        if (err) *err = "-hf " + repo + ": the lookup failed (its message is above)";
        return false;
    }
    *out = last;
    return true;
}

// What the zoo knows, by architecture and geometry (0: any). A fine-tune keeps its base model's geometry.
struct Entry {
    const char* arch;
    int64_t n_embd;
    int64_t n_vocab;
    const char* kind;
    const char* repo;
    const char* file;   // nullptr: the whole repo (a safetensors drafter); "a|b|c": the same drafter in quants, best first
    const char* base;   // the model it was trained against, for the log
};
constexpr Entry kZoo[] = {
    {"qwen35moe", 2048, 248320, "dflash", "incoai/Qwen3.6-35B-A3B-DFlash2", nullptr, "Qwen3.6-35B-A3B"},
    {"qwen35", 5120, 248320, "dflash", "Anbeeld/Qwen3.8-27B-DFlash2-GGUF",
     "Qwen3.8-27B-DFlash2-Q6_K.gguf|Qwen3.8-27B-DFlash2-Q4_K_M.gguf|Qwen3.8-27B-DFlash2-Q2_K.gguf", "Qwen3.8-27B"},
    {"inkling2_loop", 0, 0, "dspark", "unsloth/Inkling-2-Mini-GGUF", "dspark-Inkling-2-Mini-Q8_0.gguf", "Inkling-2-Mini"},
};

}  // namespace

bool model(const std::string& spec, std::string* path, std::string* err) {
    const size_t c = spec.find(':');
    const std::string repo = spec.substr(0, c), quant = c == std::string::npos ? std::string() : spec.substr(c + 1);
    static const std::regex qre(R"(^[A-Za-z0-9._-]*$)");
    if (!valid_repo(repo) || !std::regex_match(quant, qre)) {
        if (err) *err = "-hf takes org/repo[:QUANT], not " + spec;
        return false;
    }
    return helper("model", repo, quant, path, err);
}

bool file(const std::string& repo, const std::string& name, std::string* path, std::string* err) {
    if (!valid_repo(repo)) {
        if (err) *err = "not a repo: " + repo;
        return false;
    }
    return helper("file", repo, name, path, err);
}

bool repo_dir(const std::string& repo, std::string* path, std::string* err) {
    if (!valid_repo(repo)) {
        if (err) *err = "not a repo: " + repo;
        return false;
    }
    return helper("repo", repo, "", path, err);
}

bool zoo(const gguf::File& f, Drafter* d, std::string* err) {
    *d = {};
    const std::string arch(f.arch());
    const int64_t n_embd = f.get_int(arch + ".embedding_length").value_or(0);
    const auto* toks = f.get_str_array("tokenizer.ggml.tokens");
    const int64_t n_vocab = toks ? int64_t(toks->size()) : 0;
    for (const Entry& e : kZoo) {
        if (arch != e.arch || (e.n_embd && e.n_embd != n_embd) || (e.n_vocab && e.n_vocab != n_vocab)) continue;
        std::vector<std::string> paths;
        if (!e.file) {
            std::string p;
            if (!repo_dir(e.repo, &p, err)) return false;
            paths.push_back(p);
        } else {   // every quant listed: the backend picks the best one beside the target (a file that cannot be had is left out)
            const std::string list = e.file;
            for (size_t a = 0; a <= list.size();) {
                size_t b = list.find('|', a);
                if (b == std::string::npos) b = list.size();
                std::string p, ferr;
                if (b > a && file(e.repo, list.substr(a, b - a), &p, &ferr)) paths.push_back(p);
                else if (b > a && err) *err = ferr;
                a = b + 1;
            }
            if (paths.empty()) return false;
        }
        d->kind = e.kind;
        d->path = paths[0];
        d->alts.assign(paths.begin() + 1, paths.end());
        d->from = std::string(e.repo) + " (" + e.base + "'s " + e.kind + " drafter, " + std::to_string(paths.size()) + " quant" +
                  (paths.size() > 1 ? "s" : "") + ")";
        return true;
    }
    return true;
}

}  // namespace limbo::hf
