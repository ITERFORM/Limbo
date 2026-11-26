#!/usr/bin/env python3
# Copyright (c) 2026 ITERFORM. All rights reserved.
"""The Ariadne shape on limbo_serve_gpu: --slots 2 over one K/V pool, slot 1 the main, slot 0 the sidekick that only prewarms.

  scripts/ariadne_shape.py [--ctx 65536] [--tokens 30000] [--kv auto] [--server build/bin/tools/limbo_serve_gpu] [--port 18095]

1. /props and /slots answer as llama-server's do (what Ariadne's endpoint probe reads).
2. The sidekick (id_slot 0) prewarms a long prompt; the main (id_slot 1) then asks with that prompt, the sidekick's answer and a short
   tail: the main must adopt the sidekick's pages and checkpoint (cached = the prewarmed history), read only the tail, and the pool must
   hold the history once.
3. The other way: the main reads a second document, the sidekick's request (slot 0) takes it.
4. POST /slots/0?action=erase empties slot 0 and its pages go back.

Starts and stops only its own server. Run it under `flock build/gpu.lock`. Exit 0 when every gate holds."""
import argparse, json, os, pathlib, re, signal, subprocess, sys, time, urllib.error, urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from needle_serve import MODEL, haystack  # noqa: E402

SYSTEM = "You answer questions about the text the user gives you. Be brief and exact."

def call(port, path, body=None, timeout=3600):
    rq = urllib.request.Request(f"http://127.0.0.1:{port}{path}", None if body is None else json.dumps(body).encode(),
                                {"content-type": "application/json"}, method="GET" if body is None else "POST")
    try:
        with urllib.request.urlopen(rq, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode(errors="replace") or "{}")

def chat(port, messages, slot, max_tokens=1):
    return call(port, "/v1/chat/completions", {"model": "limbo", "messages": messages, "max_tokens": max_tokens, "temperature": 0, "id_slot": slot,
                                               "chat_template_kwargs": {"enable_thinking": False}})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=65536)
    ap.add_argument("--tokens", type=int, default=30000, help="approximate prompt length of each document")
    ap.add_argument("--kv", default="auto")
    ap.add_argument("--server", default=str(ROOT / "build/bin/tools/limbo_serve_gpu"))
    ap.add_argument("--port", type=int, default=18095)
    ap.add_argument("--extra", default="", help="more server flags, space separated")
    a = ap.parse_args()
    tmp = pathlib.Path(os.environ.get("LIMBO_TEST_TMP", ROOT / "build/tmp"))
    logp = tmp / f"ariadne-shape-{a.ctx}.log"
    log = open(logp, "w")
    srv = subprocess.Popen([a.server, "-m", MODEL[0], "--port", str(a.port), "--ctx", str(a.ctx), "--kv", a.kv, "--slots", "2"] + a.extra.split(),
                           stdout=log, stderr=subprocess.STDOUT)
    ok = True

    def gate(cond, what):
        nonlocal ok
        ok &= bool(cond)
        print(("PASS " if cond else "FAIL ") + what, flush=True)

    def request_line(n):  # the n-th request line of the server log (it prints after the answer is sent)
        for _ in range(50):
            lines = [l for l in logp.read_text().splitlines() if "/v1/chat/completions" in l and "prompt=" in l]
            if len(lines) >= n:
                return lines[n - 1]
            time.sleep(0.1)
        return ""

    try:
        for _ in range(600):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=1)
                break
            except Exception:
                if srv.poll() is not None:
                    print("server exited:", logp.read_text()[-2000:])
                    return 2
                time.sleep(1)
        for l in logp.read_text().splitlines():
            if "serve:" in l and ("slots" in l or "pool" in l or "checkpoints" in l) or "listening" in l:
                print("  " + l[:300])
        st, props = call(a.port, "/props")
        gate(st == 200 and props.get("total_slots") == 2 and props["default_generation_settings"]["n_ctx"] > 0 and props.get("build_info") and props.get("chat_template"),
             f"/props: total_slots {props.get('total_slots')}, n_ctx {props.get('default_generation_settings', {}).get('n_ctx')}, build_info {props.get('build_info')!r}")
        st, slots = call(a.port, "/slots")
        gate(st == 200 and [s["id"] for s in slots] == [0, 1] and all(not s["is_processing"] and s["n_past"] == 0 for s in slots), f"/slots: {slots}")

        # ---- the sidekick prewarms, the main adopts ----
        hay = haystack(int(a.tokens * 3.1), salt=3)
        history = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": hay + "\n\nSummarise the text in one word."}]
        t0 = time.time()
        st, side = chat(a.port, history, 0)
        side_wall = time.time() - t0
        n_side = side["usage"]["prompt_tokens"]
        gate(st == 200 and side["usage"]["prompt_tokens_details"]["cached_tokens"] == 0 and side["timings"]["slot"] == 0,
             f"sidekick (slot 0) prewarms {n_side} tokens cold: prefill {side['timings']['prompt_ms'] / 1000:.2f} s")
        print("  " + request_line(1)[:330])
        main_msgs = history + [{"role": "assistant", "content": side["choices"][0]["message"].get("content") or "Done"},
                               {"role": "user", "content": "Now name the first file mentioned in the text."}]
        t0 = time.time()
        st, mn = chat(a.port, main_msgs, 1)
        main_wall = time.time() - t0
        n_main, cached = mn["usage"]["prompt_tokens"], mn["usage"]["prompt_tokens_details"]["cached_tokens"]
        tail = n_main - cached
        gate(st == 200 and mn["timings"]["slot"] == 1 and mn["timings"].get("adopted_from") == 0, f"main (slot 1) adopts slot 0's prefix: adopted_from {mn['timings'].get('adopted_from')}")
        gate(cached >= 0.97 * n_side and cached <= n_side, f"main: prompt {n_main}, cached {cached} (sidekick read {n_side}): reads only {tail} tokens")
        gate(mn["timings"]["prompt_ms"] < 0.1 * side["timings"]["prompt_ms"], f"main prefill {mn['timings']['prompt_ms'] / 1000:.2f} s against the sidekick's {side['timings']['prompt_ms'] / 1000:.2f} s (wall {main_wall:.2f} s)")
        line = request_line(2)
        print("  " + line[:360])
        m = re.search(r"pool=(\d+)/(\d+)pg shared=(\d+)", line)
        used, total, shared = (int(x) for x in m.groups()) if m else (0, 0, 0)
        pages = lambda n: (n + 255) // 256
        gate(m and shared >= cached // 256 - 1 and used <= pages(n_main) + 2 and used < 0.7 * (pages(n_side) + pages(n_main)),
             f"the pool holds the prefix once: {used} of {total} pages used, {shared} shared (two copies would be {pages(n_side) + pages(n_main)})")
        st, slots = call(a.port, "/slots")
        gate(slots[0]["n_past"] >= n_side and slots[1]["n_past"] >= n_main and slots[0]["pages"] + slots[1]["pages"] - shared == used,
             f"/slots: slot 0 n_past {slots[0]['n_past']} ({slots[0]['pages']} pages), slot 1 n_past {slots[1]['n_past']} ({slots[1]['pages']} pages)")

        # ---- the other way: the main reads a second document, the sidekick takes it ----
        hay2 = haystack(int(a.tokens * 0.6 * 3.1), salt=11)
        h2 = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": hay2 + "\n\nSummarise the text in one word."}]
        st, r2 = chat(a.port, h2, 1)
        n2 = r2["usage"]["prompt_tokens"]
        st, q2 = chat(a.port, h2 + [{"role": "assistant", "content": r2["choices"][0]["message"].get("content") or "Done"},
                                    {"role": "user", "content": "Name the last file mentioned."}], 0)
        gate(st == 200 and q2["timings"]["slot"] == 0 and q2["timings"].get("adopted_from") == 1 and q2["usage"]["prompt_tokens_details"]["cached_tokens"] >= 0.97 * n2,
             f"the sidekick (slot 0) adopts the main's {n2}-token prefix: cached {q2['usage']['prompt_tokens_details']['cached_tokens']}, adopted_from {q2['timings'].get('adopted_from')}")
        print("  " + request_line(4)[:360])

        # ---- erase ----
        st, er = call(a.port, "/slots/0?action=erase", {})
        st2, slots = call(a.port, "/slots")
        gate(st == 200 and er.get("n_erased", 0) > 0 and slots[0]["n_past"] == 0 and slots[0]["pages"] == 0 and slots[1]["pages"] > 0, f"erase slot 0: {er}, /slots now {[(s['n_past'], s['pages']) for s in slots]}")
        return 0 if ok else 1
    finally:
        srv.send_signal(signal.SIGTERM)
        try:
            srv.wait(timeout=30)
        except subprocess.TimeoutExpired:
            srv.kill()
        log.close()

if __name__ == "__main__":
    sys.exit(main())
