// Copyright (c) 2026 ITERFORM. All rights reserved.
#pragma once
// limbo/check.hpp — the test harness: standalone binaries, a CHECK macro, a scratch directory.
//
// A test binary returns non-zero when any CHECK failed. Scratch files go under
// $LIMBO_TEST_TMP (default build/tmp), never /tmp: /tmp is RAM shared with the desktop.

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

namespace limbo::test {

inline int& failures() { static int n = 0; return n; }

inline std::filesystem::path tmp_dir(const std::string& sub) {
    const char* env = std::getenv("LIMBO_TEST_TMP");
    std::filesystem::path p = std::filesystem::absolute(env && *env ? env : "build/tmp") / sub;
    std::filesystem::create_directories(p);
    return p;
}

inline int finish(const char* suite) {
    if (failures()) std::printf("%s: %d check(s) FAILED\n", suite, failures());
    else std::printf("%s: all passed\n", suite);
    return failures() ? 1 : 0;
}

}  // namespace limbo::test

#define CHECK(cond, what)                                                                  \
    do {                                                                                   \
        if (cond) std::printf("ok   %s\n", what);                                          \
        else { std::printf("FAIL %s  (%s:%d: %s)\n", what, __FILE__, __LINE__, #cond);     \
               ++limbo::test::failures(); }                                               \
    } while (0)
