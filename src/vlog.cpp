// Copyright (c) 2026 ITERFORM. All rights reserved.
// src/vlog.cpp — limbo::vlog, the verbose log's formatting (include/limbo/vlog.hpp).
#include "limbo/vlog.hpp"

#include <time.h>
#include <unistd.h>

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <mutex>

namespace limbo::vlog {

namespace {
std::mutex g_mu;
bool g_color = false;
std::function<void(const std::string&)> g_sink;
std::atomic<long> g_request{0};

const char* kReset = "\x1b[0m";
const char* kDim = "\x1b[2m";

// A colour per kind of event; unknown tags are plain.
const char* tag_color(const char* tag) {
    const std::string t = tag;
    if (t == "boot" || t == "http" || t == "vram" || t == "kv" || t == "mtp" || t == "ngram") return "\x1b[36m";  // cyan: start-up
    if (t == "req" || t == "done") return "\x1b[1;37m";                                            // bold: request edges
    if (t == "route" || t == "cache") return "\x1b[37m";
    if (t == "prefill") return "\x1b[34m";
    if (t == "decode") return "\x1b[32m";
    if (t == "slot" || t == "pages") return "\x1b[35m";
    if (t == "spec" || t == "yield") return "\x1b[33m";
    if (t == "warn") return "\x1b[1;33m";
    if (t == "error") return "\x1b[1;31m";
    return "";
}
}  // namespace

void set_color(bool on) { g_color = on; }
bool color() { return g_color; }
bool stderr_wants_color() {
    const char* nc = std::getenv("NO_COLOR");
    return ::isatty(2) && !(nc && *nc);
}
void set_sink(std::function<void(const std::string&)> sink) {
    std::lock_guard<std::mutex> lk(g_mu);
    g_sink = std::move(sink);
}

std::string dim(const std::string& s) { return g_color ? std::string(kDim) + s + kReset : s; }

void set_request(long id) { g_request = id; }

void line(const char* tag, const std::string& text, long id, Tone tone) {
    if (id < 0) id = g_request;
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    tm lt;
    const time_t sec = ts.tv_sec;
    localtime_r(&sec, &lt);
    char head[48];
    std::snprintf(head, sizeof head, "%02d:%02d:%02d.%03d", lt.tm_hour, lt.tm_min, lt.tm_sec, int(ts.tv_nsec / 1000000));
    char idc[32] = "     ";
    if (id > 0) std::snprintf(idc, sizeof idc, "#%-4ld", id);
    char tg[16];
    std::snprintf(tg, sizeof tg, "%-8s", tag);
    std::string out;
    out.reserve(text.size() + 64);
    if (g_color) {
        out += kDim;
        out += head;
        out += "  ";
        out += idc;
        out += kReset;
        out += ' ';
        out += tag_color(tag);
        out += tg;
        out += kReset;
        out += ' ';
        const char* tc = tone == Tone::Good ? "\x1b[32m" : tone == Tone::Warn ? "\x1b[33m" : tone == Tone::Bad ? "\x1b[31m" : "";
        out += tc;
        out += text;
        if (*tc) out += kReset;
    } else {
        out += head;
        out += "  ";
        out += idc;
        out += ' ';
        out += tg;
        out += ' ';
        out += text;
    }
    out += '\n';
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_sink) g_sink(out);
    else {
        std::fputs(out.c_str(), stderr);
        std::fflush(stderr);
    }
}

std::string fmt(const char* f, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, f);
    const int n = std::vsnprintf(buf, sizeof buf, f, ap);
    va_end(ap);
    if (n < 0) return {};
    if (size_t(n) < sizeof buf) return std::string(buf, size_t(n));
    std::string big(size_t(n) + 1, '\0');
    va_start(ap, f);
    std::vsnprintf(big.data(), big.size(), f, ap);
    va_end(ap);
    big.resize(size_t(n));
    return big;
}

std::string thou(long long v) {
    const bool neg = v < 0;
    std::string d = std::to_string(neg ? -v : v), out;
    for (size_t i = 0; i < d.size(); ++i) {
        if (i && (d.size() - i) % 3 == 0) out += ',';
        out += d[i];
    }
    return neg ? "-" + out : out;
}

std::string bytes(uint64_t b) {
    const double x = double(b);
    if (b >= (1ull << 30)) return fmt("%.2f GiB", x / double(1ull << 30));
    if (b >= (1ull << 20)) return fmt("%s MiB", thou(static_cast<long long>(std::llround(x / double(1ull << 20)))).c_str());
    if (b >= (1ull << 10)) return fmt("%.1f KiB", x / 1024.0);
    return fmt("%llu B", static_cast<unsigned long long>(b));
}

std::string ms(double m) {
    if (m >= 10000) return fmt("%.1f s", m / 1000);
    if (m >= 1000) return fmt("%.2f s", m / 1000);
    if (m >= 100) return fmt("%.0f ms", m);
    return fmt("%.1f ms", m);
}

std::string rate(double n, double seconds) {
    if (seconds <= 0) return "- tok/s";
    const double r = n / seconds;
    return r >= 100 ? fmt("%s tok/s", thou(static_cast<long long>(std::llround(r))).c_str()) : fmt("%.1f tok/s", r);
}

std::string pct(double part, double whole) { return whole > 0 ? fmt("%.1f%%", 100.0 * part / whole) : std::string("-"); }

}  // namespace limbo::vlog
