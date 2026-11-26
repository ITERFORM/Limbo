// Copyright (c) 2026 ITERFORM. All rights reserved.
#pragma once
// limbo/vlog.hpp — the server's verbose log (`-v` events, `-vv` adds per-cycle traces): one line per event on stderr,
//   HH:MM:SS.mmm  #id  tag      text
// Colour only when stderr is a terminal and NO_COLOR is unset. Callers gate on their own verbosity; this only formats and writes.
#include <cstdint>
#include <functional>
#include <string>

namespace limbo::vlog {

enum class Tone : uint8_t { Plain, Good, Warn, Bad };

// Colour on or off (run_main: isatty(2) && !NO_COLOR). Off by default, so a test sees plain text.
void set_color(bool on);
bool color();
bool stderr_wants_color();  // isatty(2) and NO_COLOR unset or empty
// Where lines go (a test captures them); an empty function: stderr.
void set_sink(std::function<void(const std::string&)> sink);

// The request a line without an id of its own belongs to (the server sets it while it serves one; 0: none).
void set_request(long id);
// One line. `tag` is padded to 8 columns; id > 0 prints `#id` in its own column (the request the line belongs to); id < 0: the current request.
void line(const char* tag, const std::string& text, long id = -1, Tone tone = Tone::Plain);
// The same text with an escape-free dim style for secondary words, empty when colour is off.
std::string dim(const std::string& s);

std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
std::string thou(long long v);          // 41923 -> "41,923"
std::string bytes(uint64_t b);          // 1.5 GiB / 312 MiB / 4.0 KiB
std::string ms(double milliseconds);    // 412 ms / 3.21 s
std::string rate(double n, double seconds);  // "4,424 tok/s"
std::string pct(double part, double whole);  // "95.7%"

}  // namespace limbo::vlog
