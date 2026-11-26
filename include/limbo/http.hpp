// Copyright (c) 2026 ITERFORM. All rights reserved.
#pragma once
// limbo/http.hpp — a small HTTP/1.1 server: sockets and poll, one request at a time.
//
// Scope is what an OpenAI-compatible endpoint needs and no more: request line, headers, a body by
// Content-Length or chunked, `Expect: 100-continue`, responses that close the connection, and
// chunked streaming for server-sent events. Every read has a deadline, every write has a deadline,
// and a client that hangs up never takes the server down (SIGPIPE is never raised: sends use
// MSG_NOSIGNAL and a failed send marks the connection broken).
//
// One request is served at a time, but a long request does not blind the server: while a handler
// runs it may call Conn::yield(), which accepts waiting connections and answers the cheap GETs
// (health, models) on the spot; any other request is parked and served, in arrival order, when the
// running one finishes. No mutex anywhere: everything runs on the thread that calls run().
//
// The listener binds the address it is given (the caller decides; limbo serve defaults to 127.0.0.1).

#include <atomic>
#include <deque>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace limbo::http {

struct Request {
    std::string method, target, path, query, version;
    std::vector<std::pair<std::string, std::string>> headers;  // names lower-cased
    std::string body;

    const std::string* header(std::string_view lower_name) const;
};

struct ServerOptions {
    int header_timeout_ms = 10000;  // from the start of a request to the end of its headers
    int body_timeout_ms = 60000;    // from the end of the headers to the end of the body
    int write_timeout_ms = 30000;   // one blocked write
    size_t max_header_bytes = 64 * 1024;
    size_t max_body_bytes = 256u << 20;
};

class Server;

class Conn {
public:
    ~Conn();
    Conn(const Conn&) = delete;
    Conn& operator=(const Conn&) = delete;

    // A complete response with Content-Length; the connection closes afterwards.
    bool respond(int status, std::string_view content_type, std::string_view body,
                 const std::vector<std::pair<std::string, std::string>>& extra = {});

    // Chunked streaming (server-sent events). begin_stream sends the headers; each write_chunk is one
    // HTTP chunk; end_stream sends the terminating chunk.
    bool begin_stream(int status = 200, std::string_view content_type = "text/event-stream");
    bool write_chunk(std::string_view data);
    bool end_stream();

    bool headers_sent() const { return headers_sent_; }
    bool broken() const { return broken_; }  // a write failed: the peer is gone
    // True when the peer has closed or reset (without waiting). A request already read cannot say
    // anything more, so end-of-stream on the socket means the client gave up.
    bool peer_gone();
    // Let the server answer other connections while this handler works (see the header comment).
    int yield();  // returns how many waiting connections it answered
    // True when another connection has a request waiting behind this one (cheap GETs are answered first, a silent
    // connection does not count). A long handler asks it between steps to step aside.
    bool waiting();

private:
    friend class Server;
    Conn(int fd, Server* srv, int write_timeout_ms);
    bool write_all(std::string_view data);

    int fd_;
    Server* srv_;
    int write_timeout_ms_;
    bool headers_sent_ = false;
    bool broken_ = false;
    bool streaming_ = false;
};

class Server {
public:
    using Handler = std::function<void(const Request&, Conn&)>;
    // Binds host:port (port 0 picks a free one); nullptr and *err on failure.
    static std::unique_ptr<Server> listen(const std::string& host, int port, std::string* err,
                                          const ServerOptions& opt = {});
    ~Server();
    Server(const Server&) = delete;
    Server& operator=(const Server&) = delete;

    int port() const { return port_; }

    // Serves connections until `stop` turns true (checked every 50 ms while idle).
    void run(const Handler& h, const std::atomic<bool>& stop);
    // Serves at most one waiting connection, waiting up to timeout_ms for it. True when one was served.
    bool serve_one(const Handler& h, int timeout_ms);

private:
    friend class Conn;
    Server() = default;
    void handle_fd(int fd, const Handler& h);
    int service_light();
    bool has_waiting();

    int listen_fd_ = -1;
    int port_ = 0;
    ServerOptions opt_;
    const Handler* handler_ = nullptr;  // set while a handler runs, for yield()
    std::deque<int> parked_;
    bool in_light_ = false;
    int depth_ = 0;
};

}  // namespace limbo::http
