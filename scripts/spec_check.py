#!/usr/bin/env python3
# Copyright (c) 2026 ITERFORM. All rights reserved.
"""limbo_serve_gpu with and without MTP speculation: the same greedy requests must give the same text, and the decode speed of
each is printed. Then one sampled request per mode (temperature 0.7) to see the sampled path run.

  scripts/spec_check.py [--model file.gguf] [--ctx 32768] [--kv auto|f16|q8|q4] [--server build/bin/tools/limbo_serve_gpu] [--port 18093] [--max-tokens 256]

Starts and stops only its own server processes. Run it under `flock build/gpu.lock`."""
import argparse, glob, json, os, pathlib, signal, subprocess, sys, time, urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-MTP-GGUF/snapshots/*/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"))
PROMPTS = [
    "Write a short story (about 150 words) about a lighthouse keeper who finds a message in a bottle.",
    "Write a Python function that parses an ISO 8601 date string without using datetime, with a docstring and two tests.",
    "Explain in plain words how a CPU cache works, then list three ways a program can make better use of it.",
]

def ask(port, text, max_tokens, temperature, seed=None):
    body = {"model": "limbo", "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens, "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": False}}
    if seed is not None:
        body["seed"] = seed
    rq = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"})
    with urllib.request.urlopen(rq, timeout=600) as r:
        return json.load(r)

def run(a, spec, kv):
    log = open(pathlib.Path(os.environ.get("LIMBO_TEST_TMP", ROOT / "build/tmp")) / f"spec-check-{spec}.log", "w")
    srv = subprocess.Popen([a.server, "-m", a.model or MODEL[0], "--port", str(a.port), "--ctx", str(a.ctx), "--spec", spec, "--kv", a.kv], stdout=log, stderr=subprocess.STDOUT)
    out = []
    try:
        for _ in range(300):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=1)
                break
            except Exception:
                if srv.poll() is not None:
                    raise SystemExit("server exited: " + open(log.name).read()[-2000:])
                time.sleep(1)
        for p in PROMPTS:
            d = ask(a.port, p, a.max_tokens, 0)
            out.append((d["choices"][0]["message"]["content"], d["usage"]["completion_tokens"], d["timings"]["predicted_per_second"]))
        d = ask(a.port, PROMPTS[0], a.max_tokens, 0.7, seed=7)
        out.append((d["choices"][0]["message"]["content"], d["usage"]["completion_tokens"], d["timings"]["predicted_per_second"]))
    finally:
        srv.send_signal(signal.SIGTERM)
        try:
            srv.wait(timeout=30)
        except subprocess.TimeoutExpired:
            srv.kill()
        log.close()
    for l in open(log.name):
        if "listening" in l or "spec=" in l:
            print(f"  [{spec}] {l.strip()[:250]}")
        if "listening" in l:
            kv[spec] = l.split("K/V ")[1].split(" ")[0].strip(",")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="", help="the GGUF to serve (default: the rung-1 file)")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--server", default=str(ROOT / "build/bin/tools/limbo_serve_gpu"))
    ap.add_argument("--kv", default="q8", help="K/V cache format: f16, q8, q4 or auto (auto may pick another format with speculation on: texts then differ by K/V precision)")
    ap.add_argument("--port", type=int, default=18093)
    ap.add_argument("--dump", default="", help="write the greedy and sampled texts of both modes here (json), to compare builds")
    ap.add_argument("--max-tokens", type=int, default=256)
    a = ap.parse_args()
    kv = {}
    off = run(a, "off", kv)
    on = run(a, "auto", kv)
    if kv["off"] != kv["auto"]:
        print(f"NOTE: the K/V formats differ ({kv['off']} without speculation, {kv['auto']} with): greedy texts are not comparable, pass --kv f16|q8|q4")
    if a.dump:
        pathlib.Path(a.dump).write_text(json.dumps({"off": [o[0] for o in off], "auto": [o[0] for o in on]}))
    ok = True
    for i, (x, y) in enumerate(zip(off[:3], on[:3])):
        same = x[0] == y[0]
        ok &= same
        print(f"greedy {i}: {x[1]} tokens; plain {x[2]:6.1f} tok/s, speculative {y[2]:6.1f} tok/s (x{y[2] / x[2]:.2f}); text {'IDENTICAL' if same else 'DIFFERS'}")
        if not same:
            k = next((j for j in range(min(len(x[0]), len(y[0]))) if x[0][j] != y[0][j]), min(len(x[0]), len(y[0])))
            print(f"   first difference at char {k}: {x[0][max(0, k - 40):k + 40]!r} vs {y[0][max(0, k - 40):k + 40]!r}")
    print(f"sampled (t 0.7): plain {off[3][1]} tokens at {off[3][2]:.1f} tok/s, speculative {on[3][1]} tokens at {on[3][2]:.1f} tok/s")
    print("   plain:      ", off[3][0][:160].replace("\n", " "))
    print("   speculative:", on[3][0][:160].replace("\n", " "))
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
