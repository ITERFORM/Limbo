// Copyright (c) 2026 ITERFORM. All rights reserved.
// src/http.cpp — the HTTP/1.1 server (see include/limbo/http.hpp).
#include "limbo/http.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>

namespace limbo::http {

namespace {

int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

const char* reason(int status) {
    switch (status) {
        case 100: return "Continue";
        case 200: return "OK";
        case 400: return "Bad Request";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 408: return "Request Timeout";
        case 411: return "Length Required";
        case 413: return "Payload Too Large";
        case 431: return "Request Header Fields Too Large";
        case 500: return "Internal Server Error";
        case 501: return "Not Implemented";
        case 503: return "Service Unavailable";
        default: return "Status";
    }
}

std::string json_error_body(const std::string& msg, const char* type) {
    std::string o = "{\"error\":{\"message\":\"";
    for (char c : msg) {
        if (c == '"' || c == '\\') {
            o += '\\';
            o += c;
        } else if (static_cast<unsigned char>(c) < 0x20) o += ' ';
        else o += c;
    }
    o += "\",\"type\":\"";
    o += type;
    o += "\",\"param\":null,\"code\":null}}";
    return o;
}

enum class Rd { Ok, Eof, Timeout, Error };

// Buffered, deadline-bound reading from a socket.
struct Reader {
    int fd;
    std::string buf;
    size_t pos = 0;  // consumed up to here

    // Reads at least one more byte into buf.
    Rd fill(int64_t deadline) {
        for (;;) {
            const int64_t left = deadline - now_ms();
            if (left <= 0) return Rd::Timeout;
            pollfd p{fd, POLLIN, 0};
            const int r = ::poll(&p, 1, int(std::min<int64_t>(left, 1000)));
            if (r < 0) {
                if (errno == EINTR) continue;
                return Rd::Error;
            }
            if (r == 0) continue;
            char tmp[16384];
            const ssize_t n = ::recv(fd, tmp, sizeof tmp, 0);
            if (n > 0) {
                buf.append(tmp, size_t(n));
                return Rd::Ok;
            }
            if (n == 0) return Rd::Eof;
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) continue;
            return Rd::Error;
        }
    }
    Rd need(size_t n, int64_t deadline) {
        while (buf.size() - pos < n) {
            const Rd r = fill(deadline);
            if (r != Rd::Ok) return r;
        }
        return Rd::Ok;
    }
    // One line ending in CRLF (the CRLF is dropped).
    Rd line(std::string* out, int64_t deadline, size_t limit) {
        size_t from = pos;
        for (;;) {
            const size_t k = buf.find("\r\n", from);
            if (k != std::string::npos) {
                out->assign(buf, pos, k - pos);
                pos = k + 2;
                return Rd::Ok;
            }
            if (buf.size() - pos > limit) return Rd::Error;
            from = buf.size() > 0 ? buf.size() - 1 : 0;
            if (from < pos) from = pos;
            const Rd r = fill(deadline);
            if (r != Rd::Ok) return r;
        }
    }
};

std::string lower(std::string s) {
    for (char& c : s)
        if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
    return s;
}

std::string trim(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && (s[b] == ' ' || s[b] == '\t')) ++b;
    while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t')) --e;
    return s.substr(b, e - b);
}

bool send_all(int fd, std::string_view d, int timeout_ms) {
    const int64_t deadline = now_ms() + timeout_ms;
    size_t off = 0;
    while (off < d.size()) {
        const ssize_t n = ::send(fd, d.data() + off, d.size() - off, MSG_NOSIGNAL);
        if (n > 0) {
            off += size_t(n);
            continue;
        }
        if (n < 0 && errno == EINTR) continue;
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            const int64_t left = deadline - now_ms();
            if (left <= 0) return false;
            pollfd p{fd, POLLOUT, 0};
            ::poll(&p, 1, int(std::min<int64_t>(left, 1000)));
            continue;
        }
        return false;
    }
    return true;
}

// Reads one request. On failure returns false with *status (0 = drop silently) and *msg.
bool read_request(int fd, const ServerOptions& opt, int header_timeout_ms, Request* req, int* status, std::string* msg) {
    Reader rd{fd, {}, 0};
    const int64_t hdr_deadline = now_ms() + header_timeout_ms;
    auto fail = [&](int st, const char* m) {
        *status = st;
        *msg = m;
        return false;
    };
    auto rd_fail = [&](Rd r, bool started) {
        if (r == Rd::Timeout) return fail(408, "request timeout");
        if (r == Rd::Eof && !started) return fail(0, "closed");
        if (r == Rd::Eof) return fail(0, "closed mid-request");
        return fail(0, "read error");
    };

    // the head: everything up to the blank line
    size_t hend = std::string::npos;
    for (;;) {
        // tolerate stray CRLFs before the request line
        while (rd.pos < rd.buf.size() && (rd.buf[rd.pos] == '\r' || rd.buf[rd.pos] == '\n')) ++rd.pos;
        hend = rd.buf.find("\r\n\r\n", rd.pos);
        if (hend != std::string::npos) break;
        if (rd.buf.size() - rd.pos > opt.max_header_bytes) return fail(431, "request header too large");
        const Rd r = rd.fill(hdr_deadline);
        if (r != Rd::Ok) return rd_fail(r, rd.buf.size() > rd.pos);
    }
    if (hend - rd.pos > opt.max_header_bytes) return fail(431, "request header too large");
    const std::string head = rd.buf.substr(rd.pos, hend - rd.pos);
    rd.pos = hend + 4;

    size_t le = head.find("\r\n");
    const std::string rl = head.substr(0, le);
    const size_t s1 = rl.find(' '), s2 = rl.rfind(' ');
    if (s1 == std::string::npos || s2 == s1) return fail(400, "malformed request line");
    req->method = rl.substr(0, s1);
    req->target = rl.substr(s1 + 1, s2 - s1 - 1);
    req->version = rl.substr(s2 + 1);
    if (req->method.empty() || req->target.empty() || req->version.rfind("HTTP/1.", 0) != 0) return fail(400, "malformed request line");
    for (char c : req->method)
        if (c < 'A' || c > 'Z') return fail(400, "malformed method");
    const size_t q = req->target.find('?');
    req->path = req->target.substr(0, q);
    req->query = q == std::string::npos ? "" : req->target.substr(q + 1);

    size_t p = le == std::string::npos ? head.size() : le + 2;
    while (p < head.size()) {
        size_t e = head.find("\r\n", p);
        if (e == std::string::npos) e = head.size();
        const std::string ln = head.substr(p, e - p);
        p = e + 2;
        const size_t c = ln.find(':');
        if (c == std::string::npos || c == 0 || ln[c - 1] == ' ' || ln[c - 1] == '\t') return fail(400, "malformed header");
        req->headers.emplace_back(lower(ln.substr(0, c)), trim(ln.substr(c + 1)));
    }

    const std::string* te = req->header("transfer-encoding");
    const std::string* cl = req->header("content-length");
    const std::string* ex = req->header("expect");
    const int64_t body_deadline = now_ms() + opt.body_timeout_ms;
    const bool chunked = te && lower(*te) == "chunked";
    if (te && !chunked && lower(*te) != "identity") return fail(501, "unsupported transfer-encoding");

    if (chunked) {
        if (ex && lower(*ex) == "100-continue") send_all(fd, "HTTP/1.1 100 Continue\r\n\r\n", 1000);
        for (;;) {
            std::string ln;
            Rd r = rd.line(&ln, body_deadline, 1024);
            if (r != Rd::Ok) return rd_fail(r, true);
            const size_t sc = ln.find(';');
            const std::string hex = trim(ln.substr(0, sc));
            if (hex.empty() || hex.size() > 8) return fail(400, "bad chunk size");
            size_t n = 0;
            for (char c : hex) {
                int v;
                if (c >= '0' && c <= '9') v = c - '0';
                else if (c >= 'a' && c <= 'f') v = c - 'a' + 10;
                else if (c >= 'A' && c <= 'F') v = c - 'A' + 10;
                else return fail(400, "bad chunk size");
                n = n * 16 + size_t(v);
            }
            if (n == 0) {  // trailers up to the blank line
                for (;;) {
                    r = rd.line(&ln, body_deadline, 8192);
                    if (r != Rd::Ok) return rd_fail(r, true);
                    if (ln.empty()) break;
                }
                break;
            }
            if (req->body.size() + n > opt.max_body_bytes) return fail(413, "request body too large");
            r = rd.need(n + 2, body_deadline);
            if (r != Rd::Ok) return rd_fail(r, true);
            req->body.append(rd.buf, rd.pos, n);
            rd.pos += n;
            if (rd.buf.compare(rd.pos, 2, "\r\n") != 0) return fail(400, "bad chunk terminator");
            rd.pos += 2;
        }
        return true;
    }
    if (cl) {
        size_t n = 0;
        if (cl->empty() || cl->size() > 12) return fail(400, "bad content-length");
        for (char c : *cl) {
            if (c < '0' || c > '9') return fail(400, "bad content-length");
            n = n * 10 + size_t(c - '0');
        }
        if (n > opt.max_body_bytes) return fail(413, "request body too large");
        if (n > rd.buf.size() - rd.pos && ex && lower(*ex) == "100-continue") send_all(fd, "HTTP/1.1 100 Continue\r\n\r\n", 1000);
        const Rd r = rd.need(n, body_deadline);
        if (r != Rd::Ok) return rd_fail(r, true);
        req->body.assign(rd.buf, rd.pos, n);
        return true;
    }
    if (req->method == "POST" || req->method == "PUT" || req->method == "PATCH") return fail(411, "content-length required");
    return true;
}

}  // namespace

const std::string* Request::header(std::string_view lower_name) const {
    for (const auto& h : headers)
        if (h.first == lower_name) return &h.second;
    return nullptr;
}

// ---- Conn -----------------------------------------------------------------------------------------

Conn::Conn(int fd, Server* srv, int write_timeout_ms) : fd_(fd), srv_(srv), write_timeout_ms_(write_timeout_ms) {}

Conn::~Conn() {
    if (fd_ >= 0) {
        ::shutdown(fd_, SHUT_WR);
        ::close(fd_);
    }
}

bool Conn::write_all(std::string_view data) {
    if (broken_) return false;
    if (!send_all(fd_, data, write_timeout_ms_)) broken_ = true;
    return !broken_;
}

bool Conn::respond(int status, std::string_view content_type, std::string_view body,
                   const std::vector<std::pair<std::string, std::string>>& extra) {
    std::string h = "HTTP/1.1 " + std::to_string(status) + " " + reason(status) + "\r\n";
    h += "Content-Type: " + std::string(content_type) + "\r\n";
    h += "Content-Length: " + std::to_string(body.size()) + "\r\n";
    h += "Connection: close\r\n";
    for (const auto& kv : extra) h += kv.first + ": " + kv.second + "\r\n";
    h += "\r\n";
    h.append(body);
    headers_sent_ = true;
    return write_all(h);
}

bool Conn::begin_stream(int status, std::string_view content_type) {
    std::string h = "HTTP/1.1 " + std::to_string(status) + " " + reason(status) + "\r\n";
    h += "Content-Type: " + std::string(content_type) + "\r\n";
    h += "Cache-Control: no-cache\r\n";
    h += "Connection: close\r\n";
    h += "Transfer-Encoding: chunked\r\n";
    h += "X-Accel-Buffering: no\r\n\r\n";
    headers_sent_ = true;
    streaming_ = true;
    return write_all(h);
}

bool Conn::write_chunk(std::string_view data) {
    if (data.empty()) return !broken_;
    char hex[32];
    const int n = std::snprintf(hex, sizeof hex, "%zx\r\n", data.size());
    std::string c(hex, size_t(n));
    c.append(data);
    c += "\r\n";
    return write_all(c);
}

bool Conn::end_stream() {
    streaming_ = false;
    return write_all("0\r\n\r\n");
}

bool Conn::peer_gone() {
    if (broken_) return true;
    pollfd p{fd_, POLLIN, 0};
    const int r = ::poll(&p, 1, 0);
    if (r <= 0) return false;
    if (p.revents & (POLLHUP | POLLERR | POLLNVAL)) return true;
    if (p.revents & POLLIN) {
        char c;
        const ssize_t n = ::recv(fd_, &c, 1, MSG_PEEK | MSG_DONTWAIT);
        if (n == 0) return true;
        if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) return true;
    }
    return false;
}

int Conn::yield() { return srv_ ? srv_->service_light() : 0; }

bool Conn::waiting() { return srv_ && srv_->has_waiting(); }

// ---- Server ---------------------------------------------------------------------------------------

std::unique_ptr<Server> Server::listen(const std::string& host, int port, std::string* err, const ServerOptions& opt) {
    const int fd = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
    if (fd < 0) {
        if (err) *err = std::string("socket: ") + std::strerror(errno);
        return nullptr;
    }
    const int one = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(uint16_t(port));
    if (::inet_pton(AF_INET, host == "localhost" ? "127.0.0.1" : host.c_str(), &a.sin_addr) != 1) {
        if (err) *err = "not an IPv4 address: " + host;
        ::close(fd);
        return nullptr;
    }
    if (::bind(fd, reinterpret_cast<sockaddr*>(&a), sizeof a) < 0 || ::listen(fd, 64) < 0) {
        if (err) *err = "bind/listen " + host + ":" + std::to_string(port) + ": " + std::strerror(errno);
        ::close(fd);
        return nullptr;
    }
    socklen_t len = sizeof a;
    ::getsockname(fd, reinterpret_cast<sockaddr*>(&a), &len);
    std::unique_ptr<Server> s(new Server);
    s->listen_fd_ = fd;
    s->port_ = ntohs(a.sin_port);
    s->opt_ = opt;
    return s;
}

Server::~Server() {
    for (int fd : parked_) ::close(fd);
    if (listen_fd_ >= 0) ::close(listen_fd_);
}

void Server::handle_fd(int fd, const Handler& h) {
    Conn conn(fd, this, opt_.write_timeout_ms);
    Request req;
    int status = 0;
    std::string msg;
    if (!read_request(fd, opt_, opt_.header_timeout_ms, &req, &status, &msg)) {
        if (status) conn.respond(status, "application/json", json_error_body(msg, "invalid_request_error"));
        return;
    }
    const Handler* prev = handler_;
    handler_ = &h;
    ++depth_;
    try {
        h(req, conn);
    } catch (const std::exception& e) {
        if (!conn.headers_sent()) conn.respond(500, "application/json", json_error_body(e.what(), "server_error"));
    }
    --depth_;
    handler_ = prev;
}

int Server::service_light() {
    if (!handler_ || in_light_) return 0;
    in_light_ = true;
    int served = 0;
    auto is_get = [](int fd, bool* closed) {
        char peek[4];
        const ssize_t n = ::recv(fd, peek, sizeof peek, MSG_PEEK | MSG_DONTWAIT);
        *closed = n == 0;
        return n == 4 && std::memcmp(peek, "GET ", 4) == 0;
    };
    auto serve_light = [&](int fd) {
        ++served;
        Conn conn(fd, this, opt_.write_timeout_ms);
        Request req;
        int status = 0;
        std::string msg;
        if (!read_request(fd, opt_, 300, &req, &status, &msg)) {
            if (status) conn.respond(status, "application/json", json_error_body(msg, "invalid_request_error"));
            return;
        }
        try {
            (*handler_)(req, conn);
        } catch (const std::exception& e) {
            if (!conn.headers_sent()) conn.respond(500, "application/json", json_error_body(e.what(), "server_error"));
        }
    };
    // Parked connections that turned out to be cheap GETs, or that hung up.
    for (size_t i = 0; i < parked_.size();) {
        bool closed = false;
        const int fd = parked_[i];
        if (is_get(fd, &closed)) {
            parked_.erase(parked_.begin() + long(i));
            serve_light(fd);
        } else if (closed) {
            parked_.erase(parked_.begin() + long(i));
            ::close(fd);
        } else ++i;
    }
    for (int k = 0; k < 8; ++k) {
        pollfd pfd{listen_fd_, POLLIN, 0};
        if (::poll(&pfd, 1, 0) <= 0) break;
        const int fd = ::accept4(listen_fd_, nullptr, nullptr, SOCK_NONBLOCK | SOCK_CLOEXEC);
        if (fd < 0) break;
        bool closed = false;
        if (is_get(fd, &closed)) serve_light(fd);
        else if (closed) ::close(fd);
        else parked_.push_back(fd);
    }
    in_light_ = false;
    return served;
}

bool Server::has_waiting() {
    service_light();
    for (int fd : parked_) {
        char c;
        if (::recv(fd, &c, 1, MSG_PEEK | MSG_DONTWAIT) > 0) return true;
    }
    return false;
}

bool Server::serve_one(const Handler& h, int timeout_ms) {
    if (!parked_.empty()) {
        const int fd = parked_.front();
        parked_.pop_front();
        handle_fd(fd, h);
        return true;
    }
    pollfd pfd{listen_fd_, POLLIN, 0};
    if (::poll(&pfd, 1, timeout_ms) <= 0) return false;
    const int fd = ::accept4(listen_fd_, nullptr, nullptr, SOCK_NONBLOCK | SOCK_CLOEXEC);
    if (fd < 0) return false;
    handle_fd(fd, h);
    return true;
}

void Server::run(const Handler& h, const std::atomic<bool>& stop) {
    while (!stop.load(std::memory_order_relaxed)) serve_one(h, 50);
}

}  // namespace limbo::http
