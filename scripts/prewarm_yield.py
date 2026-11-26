#!/usr/bin/env python3
# Copyright (c) 2026 ITERFORM. All rights reserved.
"""A prewarm (max_tokens 1) steps aside for a waiting main request, on limbo_serve_gpu --slots 2.

  scripts/prewarm_yield.py [--ctx 65536] [--tokens 30000] [--kv auto] [--server build/bin/tools/limbo_serve_gpu] [--port 18096]

1. A main request on its own (slot 1) is timed on the idle server: the floor.
2. A ~30k-token prime (max_tokens 1, stream false) goes to slot 0; --delay (1 s) later the same short main request goes to slot 1. The main's
   wall time must stay near the floor (it was the rest of the prime's read, ~15 s, before the prime yielded). The prime answers 503 with
   Retry-After: 1 and the number of tokens it had read, which /slots shows as slot 0's n_past.
3. The prime is sent again, as Ariadne's prime() does: cached = what was read before, only the rest is read, and its answer is the token an
   uninterrupted prime of the same prompt gives (slot 0 is erased and the prompt read once more, nobody waiting, for the reference).

Starts and stops only its own server. Run it under `flock build/gpu.lock`. Exit 0 when every gate holds."""
import argparse, json, os, pathlib, re, signal, subprocess, sys, threading, time, urllib.error, urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from needle_serve import MODEL, haystack  # noqa: E402

SYSTEM = "You answer questions about the text the user gives you. Be brief and exact."

def post(port, body, timeout=3600):
    """(status, Retry-After header, parsed JSON body, wall seconds)"""
    rq = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"}, method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(rq, timeout=timeout) as r:
            return r.status, r.headers.get("Retry-After"), json.load(r), time.time() - t0
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Retry-After"), json.loads(e.read().decode(errors="replace") or "{}"), time.time() - t0

def get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=30) as r:
        return json.load(r)

def body_of(messages, slot, max_tokens):
    return {"model": "limbo", "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": False, "id_slot": slot,
            "chat_template_kwargs": {"enable_thinking": False}}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=65536)
    ap.add_argument("--tokens", type=int, default=30000, help="approximate prompt length of the prime")
    ap.add_argument("--kv", default="auto")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds the prime reads before the main arrives")
    ap.add_argument("--server", default=str(ROOT / "build/bin/tools/limbo_serve_gpu"))
    ap.add_argument("--port", type=int, default=18096)
    ap.add_argument("--extra", default="", help="more server flags, space separated")
    a = ap.parse_args()
    tmp = pathlib.Path(os.environ.get("LIMBO_TEST_TMP", ROOT / "build/tmp"))
    tmp.mkdir(parents=True, exist_ok=True)
    logp = tmp / f"prewarm-yield-{a.ctx}.log"
    log = open(logp, "w")
    srv = subprocess.Popen([a.server, "-m", MODEL[0], "--port", str(a.port), "--ctx", str(a.ctx), "--kv", a.kv, "--slots", "2"] + a.extra.split(),
                           stdout=log, stderr=subprocess.STDOUT)
    ok = True

    def gate(cond, what):
        nonlocal ok
        ok &= bool(cond)
        print(("PASS " if cond else "FAIL ") + what, flush=True)

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
        main_msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Name one prime number between 10 and 20."}]
        main_body = body_of(main_msgs, 1, 16)
        post(a.port, main_body)  # warm-up: kernels, allocator
        main_body["messages"][1]["content"] = "Name one colour of the rainbow, and nothing else."
        st, _, mj, floor = post(a.port, main_body)
        gate(st == 200, f"main on the idle server: {floor * 1000:.0f} ms ({mj['usage']['completion_tokens']} tokens)")

        hay = haystack(int(a.tokens * 3.1), salt=5)
        prime_body = body_of([{"role": "system", "content": SYSTEM}, {"role": "user", "content": hay + "\n\nSummarise the text in one word."}], 0, 1)
        res = {}

        def run_prime():
            res["t0"] = time.time()
            res["r"] = post(a.port, prime_body)
            res["t1"] = time.time()
        tp = threading.Thread(target=run_prime)
        tp.start()
        time.sleep(a.delay)
        main_body["messages"][1]["content"] = "Name one animal that lives in water, and nothing else."
        t_arrive = time.time()
        st, _, mj, main_wall = post(a.port, main_body)
        gate(st == 200 and mj["timings"]["slot"] == 1, f"main (slot 1) while the prime reads on slot 0: wall {main_wall * 1000:.0f} ms (idle floor {floor * 1000:.0f} ms)")
        gate(main_wall < 2.0, f"main wall {main_wall:.2f} s < 2 s")
        tp.join()
        st, retry_after, ej, first_wall = res["r"]
        msg = ej.get("error", {}).get("message", "")
        m = re.match(r"preempted by a waiting request; the (\d+) tokens read so far stay cached on slot (\d+)", msg)
        read = int(m.group(1)) if m else -1
        gate(st == 503 and retry_after == "1" and m and m.group(2) == "0" and ej["error"].get("type") == "server_error",
             f"prime: {st}, Retry-After {retry_after!r}, {msg!r} (stopped {res['t1'] - t_arrive:.2f} s after the main arrived, {first_wall:.2f} s after it started)")
        slots = get(a.port, "/slots")
        gate(read > 0 and slots[0]["n_past"] == read and not slots[0]["is_processing"], f"slot 0 keeps its tape: n_past {slots[0]['n_past']} == read {read} ({slots[0]['pages']} pages)")

        st, _, rj, retry_wall = post(a.port, prime_body)  # what prime() does on a 503
        n_prompt = rj["usage"]["prompt_tokens"]
        cached, tl = rj["usage"]["prompt_tokens_details"]["cached_tokens"], rj["timings"]
        gate(st == 200 and tl["slot"] == 0 and cached == read and tl["prompt_n"] == n_prompt - read,
             f"retry: prompt {n_prompt}, cached {cached} (read before: {read}), reads only {tl['prompt_n']}: prefill {tl['prompt_ms'] / 1000:.2f} s, wall {retry_wall:.2f} s")
        answer = rj["choices"][0]["message"]

        # reference: slot 0 emptied, the same prompt read once with nobody waiting
        er = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{a.port}/slots/0?action=erase", b"{}", {"content-type": "application/json"}, method="POST"), timeout=30))
        st, _, fj, full_wall = post(a.port, prime_body)
        ftl = fj["timings"]
        gate(st == 200 and fj["usage"]["prompt_tokens_details"]["cached_tokens"] == 0 and fj["choices"][0]["message"] == answer,
             f"uninterrupted prime (reference): cached 0, prefill {ftl['prompt_ms'] / 1000:.2f} s, wall {full_wall:.2f} s; same answer as the interrupted one: {answer.get('content')!r}")
        gate(first_wall + retry_wall < full_wall * 1.15 + 1.0, f"interrupted prime in all {first_wall + retry_wall:.2f} s (first read {first_wall:.2f} + retry {retry_wall:.2f}) against {full_wall:.2f} s uninterrupted")
        for l in logp.read_text().splitlines():
            if "preempted=1" in l:
                print("  " + l[:300])
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
