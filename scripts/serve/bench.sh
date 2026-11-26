#!/usr/bin/env bash
# Copyright (c) 2026 ITERFORM. All rights reserved.
# scripts/serve/bench.sh -- start a model the way scripts/serve starts it and measure it the way it will be used.
#
#   scripts/serve/bench.sh <model> [N] [TOKENS]
#     <model>  qwen3.8-flash-next | qwen3.6-35b-a3b | qwen3.8-27b | inkling-2-mini (a scripts/serve/<model>.sh)
#     N        clients at once (default 1); the server gets SLOTS=N unless SLOTS is set
#     TOKENS   tokens each request generates (default 300)
#
#   env  PROMPT  the prompt file (bench/prompts/code.txt)       SAME=1   every client sends the same prompt (shared pages)
#        TURNS   requests a client sends, each on its own history with the reply appended (1; 2 shows its slot's cache)
#        THINK=0 sends enable_thinking false                    URL      measure a server already running (no start)
#        and every start knob of _common.sh: CTX KV SLOTS CKPT DRAFT RESERVE (e.g. SLOTS=4 CTX=65536 DRAFT=0)
#
# Prints each request (prompt tokens read, from the slot's cache, prefill and decode rates, wall), the aggregate (tokens out
# per second of wall over all clients), then the server's own lines: its config, the speculation, the pool's pages.
set -u
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
name=${1:?usage: bench.sh <model> [N] [TOKENS]}
N=${2:-1}
TOKENS=${3:-300}
[ -x "$here/$name.sh" ] || { echo "no $here/$name.sh" >&2; exit 1; }
PROMPT=${PROMPT:-$root/bench/prompts/code.txt}
log=""
if [ -z "${URL:-}" ]; then
    # a server already on the card (started as `limbo` or as limbo_serve_gpu) owns the VRAM: measure it with URL= instead
    if pgrep -x limbo >/dev/null || pgrep -x limbo_serve_gpu >/dev/null; then
        echo "a limbo server is running: stop it, or measure it with URL=http://127.0.0.1:8080" >&2
        exit 1
    fi
    PORT=$(python3 -c "import socket; s = socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1])")
    URL="http://127.0.0.1:$PORT"
    log=$(mktemp "${TMPDIR:-$root/build}/bench-$name.XXXX.log")
    SLOTS=${SLOTS:-$N} PORT=$PORT setsid "$here/$name.sh" >"$log" 2>&1 &
    pid=$!
    stop() { kill -TERM -- -"$pid" 2>/dev/null; sleep 2; kill -KILL -- -"$pid" 2>/dev/null; }
    trap stop EXIT INT TERM
    echo "starting $name (log $log) ..."
    for i in $(seq 1 600); do
        curl -sf "$URL/health" >/dev/null 2>&1 && break
        kill -0 "$pid" 2>/dev/null || { echo "the server died:"; tail -8 "$log"; exit 1; }
        sleep 1
    done
    curl -sf "$URL/health" >/dev/null 2>&1 || { echo "the server never came up:"; tail -8 "$log"; exit 1; }
    echo "up after ${i} s"
fi

python3 - "$URL" "$N" "$TOKENS" "$PROMPT" "${SAME:-0}" "${TURNS:-1}" "${THINK:-}" <<'PY'
import json, sys, threading, time, urllib.request
url, n, tokens, prompt, same, turns, think = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5] == "1", int(sys.argv[6]), sys.argv[7]
text = open(prompt).read()

def ask(msgs, max_tokens=tokens):
    body = {"messages": msgs, "max_tokens": max_tokens, "temperature": 0, "top_k": 1}
    if think == "0":
        body["enable_thinking"] = False
    rq = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(rq, timeout=3600) as r:
        out = json.load(r)
    return out, time.time() - t0

# a 16-token warm-up, so the first client does not pay the first request's one-time costs
ask([{"role": "user", "content": "hi"}], 16)
rows, lock = [], threading.Lock()
def client(k):
    head = "" if same else f"[agent {k}] "
    msgs = [{"role": "user", "content": head + text}]
    for turn in range(turns):
        out, wall = ask(msgs)
        t = out.get("timings", {})
        m = out["choices"][0]["message"]
        with lock:
            rows.append((k, turn + 1, t.get("prompt_n", 0), t.get("cache_n", 0), t.get("prompt_per_second", 0.0),
                         t.get("predicted_n", 0), t.get("predicted_per_second", 0.0), wall))
        msgs += [{"role": "assistant", "content": m.get("content") or ""}, {"role": "user", "content": "Go on: the next part, in the same style."}]

t0 = time.time()
th = [threading.Thread(target=client, args=(k,)) for k in range(n)]
for x in th: x.start()
for x in th: x.join()
span = time.time() - t0
print(f"{'client':>6} {'turn':>4} {'read':>7} {'cached':>7} {'prefill/s':>10} {'gen':>5} {'decode/s':>9} {'wall s':>7}")
for r in sorted(rows):
    print(f"{r[0]:>6} {r[1]:>4} {r[2]:>7} {r[3]:>7} {r[4]:>10.0f} {r[5]:>5} {r[6]:>9.1f} {r[7]:>7.1f}")
gen = sum(r[5] for r in rows)
print(f"aggregate: {gen} tokens out in {span:.1f} s = {gen / span:.1f} tok/s over {n} client(s) x {turns} turn(s); "
      f"mean single-request decode {sum(r[6] for r in rows) / max(1, len(rows)):.1f} tok/s")
PY

if [ -n "$log" ]; then
    echo "--- server"
    grep -E ": -hf |: drafter |: no drafter " "$log" | sed 's|/home/[^/]*/|~/|g' | cut -c1-240
    grep -E "listening on" "$log" | sed 's/^limbo_serve_gpu: //' | cut -c1-400
    grep -E "  (slots|kv|ckpt) " "$log" | head -4 | cut -c1-220
    grep -E "decode .* spec" "$log" | sed 's/^.*decode/decode/' | cut -c1-240 | tail -"$(( ${N} * ${TURNS:-1} + 1 ))"
    grep -E "pool [0-9]+/[0-9]+ pages" "$log" | tail -1 | sed 's/.*\(pool [0-9]*\/[0-9]* pages, [0-9]* shared\).*/\1/'
fi
