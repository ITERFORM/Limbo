// Copyright (c) 2026 ITERFORM. All rights reserved.
// tests/chat_inkling_test.cpp — the Inkling-2 chat template, byte-exact against its reference environment.
//
// The template carries `{% generation %}`, which plain jinja2 refuses: it is a transformers extension,
// and transformers' AssistantTracker (utils/chat_template_utils.py) renders the block's body UNCHANGED
// while recording the character span of the assistant's turns beside it. So the oracle cannot be plain
// jinja2 — the reference IS the environment that implements the tag. scripts/inkling_fixtures.py renders
// the cases with transformers' own compiled environment (ImmutableSandboxedEnvironment, loopcontrols,
// AssistantTracker, a non-escaping `tojson`, raise_exception) and writes the exact bytes; this test
// renders the same contexts with limbo::jinja and compares byte for byte, refusals included.
//
// The fixtures are a model fact — the GGUF's tokenizer.chat_template, 6868 chars, from
// unsloth/Inkling-2-Mini-GGUF's Mainline_Support files — not a build input, so with no fixtures the
// suite says so and passes, the way the LIMBO_TEST_GGUF cases do. LIMBO_INKLING_FIXTURES overrides the
// directory (deliberately not LIMBO_CHAT_FIXTURES: that one points at rung 1's template).
#include "limbo/chat.hpp"
#include "limbo/check.hpp"
#include "limbo/jinja.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using limbo::jinja::Value;

namespace {

std::string slurp(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

Value json(const std::string& text) {
    Value v;
    std::string err;
    if (!limbo::jinja::json_parse(text, &v, &err)) {
        std::printf("FAIL test json did not parse: %s in %s\n", err.c_str(), text.c_str());
        ++limbo::test::failures();
    }
    return v;
}

std::string show(const std::string& s) {
    std::string o;
    for (unsigned char c : s) {
        if (c == '\n') o += "\\n";
        else if (c == '\t') o += "\\t";
        else if (c < 0x20) {
            char b[8];
            std::snprintf(b, sizeof b, "\\x%02x", c);
            o += b;
        } else o += char(c);
    }
    return o;
}

void first_difference(const std::string& got, const std::string& want) {
    size_t i = 0;
    while (i < got.size() && i < want.size() && got[i] == want[i]) ++i;
    const size_t a = i > 30 ? i - 30 : 0;
    std::printf("       first difference at byte %zu (got %zu bytes, want %zu)\n       got  ...%s\n       want ...%s\n", i,
                got.size(), want.size(), show(got.substr(a, 80)).c_str(), show(want.substr(a, 80)).c_str());
}

}  // namespace

int main() {
    const char* env = std::getenv("LIMBO_INKLING_FIXTURES");
    const fs::path dir = env && *env ? env : "bench/fixtures/chat-inkling";
    if (!fs::exists(dir / "template.jinja")) {
        std::printf("--   no inkling fixtures in %s (run scripts/inkling_fixtures.py)\n", dir.string().c_str());
        return limbo::test::finish("chat_inkling");
    }

    const std::string tmpl = slurp(dir / "template.jinja");
    std::string err;
    auto r = limbo::chat::Renderer::from_source(tmpl, &err);
    CHECK(r != nullptr, "the Inkling-2 chat template compiles ({% generation %} and all)");
    if (!r) {
        std::printf("       %s\n", err.c_str());
        return limbo::test::finish("chat_inkling");
    }

    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(dir)) {
        const std::string n = e.path().filename().string();
        if (n.size() > 5 && n.substr(n.size() - 5) == ".json") files.push_back(e.path());
    }
    std::sort(files.begin(), files.end());
    CHECK(files.size() >= 23, "the inkling fixtures are present (23 cases: 15 rendered, 8 refusals)");

    int rendered = 0, refused = 0;
    for (const fs::path& f : files) {
        const Value fx = json(slurp(f));
        const std::string name = fx.get_str("name");
        const Value ctx = *fx.find("context");
        std::string out;
        std::string rerr;
        const bool ok = r->render(ctx, &out, &rerr);
        const Value* ev = fx.find("error");
        if (ev && ev->is_string()) {
            const bool match = !ok && (ev->s() == "*" || rerr == ev->s());
            if (match) {
                ++refused;
                std::printf("ok   %s: refused \"%s\"\n", name.c_str(), rerr.c_str());
            } else {
                std::printf("FAIL %s: the reference refused with \"%s\", the engine ok=%d err=\"%s\"\n", name.c_str(),
                            ev->s().c_str(), int(ok), rerr.c_str());
                ++limbo::test::failures();
            }
            continue;
        }
        fs::path exp = f;
        exp.replace_extension(".expected.txt");
        const std::string want = slurp(exp);
        if (ok && out == want) {
            ++rendered;
            std::printf("ok   %s: %zu bytes identical to the reference\n", name.c_str(), out.size());
        } else {
            std::printf("FAIL %s: %s\n", name.c_str(), ok ? "output differs" : rerr.c_str());
            if (ok) first_difference(out, want);
            ++limbo::test::failures();
        }
    }
    std::printf("     %d rendered, %d refused, %zu bytes of template\n", rendered, refused, tmpl.size());
    return limbo::test::finish("chat_inkling");
}
