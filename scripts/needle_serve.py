#!/usr/bin/env python3
# Copyright (c) 2026 ITERFORM. All rights reserved.
"""Long-context check of limbo_serve_gpu: two conversations, each a haystack of the repo's own text with one needle, asked
twice (the follow-up must come from the cache), interleaved so both slots are used (pass `--extra "--slots 2"`; the pool holds --ctx positions in all, so --ctx must cover both prompts).

  scripts/needle_serve.py --ctx 65536 --tokens 60000 [--kv auto] [--gen N] [--server build/bin/tools/limbo_serve_gpu] [--port 18091]

Starts the server (and stops only that process), prints per request: prompt and cached tokens, prefill and decode speed, home,
and whether the answer holds the needle. Exit 0 when every answer does. Run it under `flock build/gpu.lock`."""
import argparse, glob, json, os, pathlib, re, signal, subprocess, sys, time, urllib.error, urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-MTP-GGUF/snapshots/*/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"))

def haystack(chars, salt):
    files = sorted(glob.glob(str(ROOT / "docs/**/*.md"), recursive=True)) + sorted(glob.glob(str(ROOT / "src/**/*.c*"), recursive=True)) \
        + sorted(glob.glob(str(ROOT / "include/**/*.hpp"), recursive=True)) + sorted(glob.glob(str(ROOT / "bench/*.md")))
    files = files[salt % len(files):] + files[:salt % len(files)]
    out, n = [], 0
    while n < chars:
        for f in files:
            t = f"\n=== {os.path.relpath(f, ROOT)} ===\n" + pathlib.Path(f).read_text(errors="replace")
            out.append(t)
            n += len(t)
            if n >= chars:
                break
    return "".join(out)[:chars]

def ask(port, messages, max_tokens=48):
    body = json.dumps({"model": "limbo", "messages": messages, "max_tokens": max_tokens, "temperature": 0,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    rq = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body, {"content-type": "application/json"})
    with urllib.request.urlopen(rq, timeout=3600) as r:
        return json.load(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=65536)
    ap.add_argument("--tokens", type=int, default=60000, help="approximate prompt length of each conversation")
    ap.add_argument("--kv", default="auto")
    ap.add_argument("--server", default=str(ROOT / "build/bin/tools/limbo_serve_gpu"))
    ap.add_argument("--port", type=int, default=18091)
    ap.add_argument("--one", action="store_true", help="one conversation only")
    ap.add_argument("--gen", type=int, default=0, help="after the needle turns, one answer of this many tokens in conversation 0 (decode speed at the long context)")
    ap.add_argument("--extra", default="", help="more server flags, space separated")
    a = ap.parse_args()
    log = open(pathlib.Path(os.environ.get("LIMBO_TEST_TMP", ROOT / "build/tmp")) / f"needle-{a.ctx}-{a.kv}.log", "w")
    srv = subprocess.Popen([a.server, "-m", MODEL[0], "--port", str(a.port), "--ctx", str(a.ctx), "--kv", a.kv] + a.extra.split(),
                           stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(300):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=1)
                break
            except Exception:
                if srv.poll() is not None:
                    print("server exited:", open(log.name).read()[-2000:]); return 2
                time.sleep(1)
        convs = []
        for k, (code, depth) in enumerate([("7391-ALPHA-ZEBRA", 0.25), ("4410-DELTA-MANGO", 0.6)][: 1 if a.one else 2]):
            hay = haystack(int(a.tokens * 3.1), salt=17 * k)
            needle = f"\n\nIMPORTANT NOTE: the launch code of project number {k + 1} is {code}. Remember it.\n\n"
            question = f"\n\nWhat is the launch code of project number {k + 1}? Answer with the code only."
            def build(n_chars, hay=hay, needle=needle, question=question, depth=depth):
                h = hay[:n_chars]
                cut = int(len(h) * depth)
                return h[:cut] + needle + h[cut:] + question
            convs.append({"code": code, "build": build, "chars": len(hay), "msgs": [
                {"role": "system", "content": "You answer questions about the text the user gives you. Be brief and exact."},
                {"role": "user", "content": build(len(hay))}]})
        order = [(0, 0), (1, 0), (0, 1), (1, 1)] if len(convs) == 2 else [(0, 0), (0, 1)]
        ok_all = True
        for ci, turn in order:
            c = convs[ci]
            if turn == 1:
                c["msgs"] += [{"role": "assistant", "content": c["last"]},
                              {"role": "user", "content": "Repeat that code once more, then name the file the note sat in or near."}]
            t0 = time.time()
            for _ in range(4):
                try:
                    d = ask(a.port, c["msgs"])
                    break
                except urllib.error.HTTPError as e:
                    m = re.search(r"prompt has (\d+) tokens, the context holds (\d+)", e.read().decode(errors="replace"))
                    if not m or turn != 0:
                        raise
                    have, room = int(m.group(1)), int(m.group(2))
                    c["chars"] = int(c["chars"] * min(a.tokens, room - 512) / have)
                    c["msgs"][1]["content"] = c["build"](c["chars"])
            wall = time.time() - t0
            if "error" in d:
                print(f"conv {ci} turn {turn}: ERROR {d['error']}"); ok_all = False; continue
            u, t = d["usage"], d.get("timings", {})
            ans = d["choices"][0]["message"].get("content", "")
            c["last"] = ans
            hit = c["code"] in ans
            ok_all &= hit
            print(f"conv {ci} turn {turn}: prompt {u['prompt_tokens']:6d} cached {u.get('prompt_tokens_details', {}).get('cached_tokens', 0):6d} "
                  f"prefill {t.get('prompt_ms', 0) / 1000:7.2f} s ({t.get('prompt_per_second', 0):6.0f} tok/s) decode {t.get('predicted_per_second', 0):5.1f} tok/s "
                  f"slot {t.get('slot')} wall {wall:6.1f} s  {'OK ' if hit else 'MISS'} {ans[:90]!r}", flush=True)
        if a.gen:
            c = convs[0]
            c["msgs"] += [{"role": "assistant", "content": c["last"]},
                          {"role": "user", "content": "Now describe in plain prose, at length, what kinds of documents the text contained and what each was about."}]
            d = ask(a.port, c["msgs"], max_tokens=a.gen)
            u, t = d["usage"], d.get("timings", {})
            print(f"gen: {u['completion_tokens']} tokens at {u['prompt_tokens']} prompt tokens ({u.get('prompt_tokens_details', {}).get('cached_tokens', 0)} cached): "
                  f"decode {t.get('predicted_per_second', 0):5.1f} tok/s", flush=True)
        return 0 if ok_all else 1
    finally:
        srv.send_signal(signal.SIGTERM)
        try:
            srv.wait(timeout=30)
        except subprocess.TimeoutExpired:
            srv.kill()
        log.close()
        print("".join(l for l in open(log.name) if "slots:" in l or "pool:" in l or "checkpoints:" in l or "listening" in l or "refused" in l or "spec=" in l), end="")

if __name__ == "__main__":
    sys.exit(main())
