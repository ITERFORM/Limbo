#!/usr/bin/env python3
# Copyright (c) 2026 ITERFORM. All rights reserved.
"""limbo_serve_gpu under each --spec mode with the DFlash2 drafter: decode speed, tokens per pass and whether the greedy text equals --spec off's.

  scripts/dflash_bench.py --model file.gguf --draft DIR [--modes off,mtp,ngram,dflash,auto] [--tasks prose1,prose2,code,repeat,edit] [--kv q8]
                          [--ctx 32768] [--slots 2] [--max-tokens 400] [--extra "--draft-vocab 65536"] [--dump out.json]

A mode may carry flags after a colon (dflash:--draft-quant bf16). Tasks: two prose prompts, a code prompt, the ngram_bench repeat and edit tasks.
Starts and stops only its own server processes; run it under `flock build/gpu.lock`."""
import argparse, json, os, pathlib, re, signal, subprocess, sys, time, urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
TMP = pathlib.Path(os.environ.get("LIMBO_TEST_TMP", ROOT / "build/tmp"))

def snap(path, a=None, b=None):
    t = subprocess.run(["git", "-C", str(ROOT), "show", f"50af5ec:{path}"], capture_output=True, text=True, check=True).stdout.splitlines()
    return "\n".join(t[a:b] if a is not None else t)

def tasks():
    fn, hdr = snap("src/serve.cpp", 960, 1160), snap("include/limbo/kvpool.hpp")
    return {
        "prose1": "Write a short story (about 200 words) about a lighthouse keeper who finds a message in a bottle.",
        "prose2": "Explain in plain words how a CPU cache works, then list three ways a program can make better use of it.",
        "code": "Write a Python function that parses an ISO 8601 date string without using datetime, with a docstring and two tests.",
        "repeat": "Below is a C++ function. Reproduce it exactly, line for line, except: rename every `t_prefill` to `prefill_secs` and every `t_decode` to `decode_secs`. "
                  "Output only the code in one fenced block.\n\n```cpp\n" + fn + "\n```",
        "edit": "Here is a header. Add a method `bool empty() const { return held_total() == 0; }` right after `held`, and output the complete new file in one fenced block, "
                "changing nothing else.\n\n```cpp\n" + hdr + "\n```",
    }

def ask(port, text, max_tokens, temperature=0, seed=None):
    body = {"model": "limbo", "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens, "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": False}, "cache_prompt": False}
    if seed is not None: body["seed"] = seed
    rq = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"})
    with urllib.request.urlopen(rq, timeout=1800) as r:
        return json.load(r)

def run(a, mode):
    name, _, flags = mode.partition(":")
    log = open(TMP / f"dflash-bench-{name}.log", "w")
    cmd = [a.server, "-m", a.model, "--port", str(a.port), "--ctx", str(a.ctx), "--spec", name, "--kv", a.kv, "--slots", str(a.slots), "-v"]
    if name in ("dflash", "auto") and a.draft: cmd += ["--draft-model", a.draft]
    cmd += a.extra.split() + flags.split()
    srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    out = {}
    try:
        for _ in range(900):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=1); break
            except Exception:
                if srv.poll() is not None: raise SystemExit("server exited: " + open(log.name).read()[-2000:])
                time.sleep(1)
        ask(a.port, "Say hi.", 8)
        ts = tasks()
        for t in a.tasks.split(","):
            d = ask(a.port, ts[t], a.max_tokens)
            out[t] = dict(text=d["choices"][0]["message"]["content"], n=d["usage"]["completion_tokens"], tps=d["timings"]["predicted_per_second"])
        if a.sampled:
            d = ask(a.port, ts["prose1"], a.max_tokens, 0.7, seed=7)
            out["sampled"] = dict(text=d["choices"][0]["message"]["content"], n=d["usage"]["completion_tokens"], tps=d["timings"]["predicted_per_second"])
    finally:
        srv.send_signal(signal.SIGTERM)
        try: srv.wait(timeout=60)
        except subprocess.TimeoutExpired: srv.kill()
        log.close()
    spec = [l.split("spec ")[1].strip() for l in open(log.name) if " decode " in l and "spec " in l]
    names = ["warm"] + a.tasks.split(",") + (["sampled"] if a.sampled else [])
    for n, s in zip(names[1:], spec[1:]):
        m = re.match(r"(\d+)/(\d+) \(([\d.]+) tokens a pass, (\d+) of (\d+) drafts kept", s)
        if m and n in out: out[n]["tpp"], out[n]["kept"], out[n]["drafted"] = float(m.group(3)), int(m.group(4)), int(m.group(5))
    out["_start"] = [l.strip()[:300] for l in open(log.name) if "speculation" in l and "listening" in l]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--draft", default="")
    ap.add_argument("--modes", default="off,mtp,ngram,dflash,auto")
    ap.add_argument("--tasks", default="prose1,prose2,code,repeat,edit")
    ap.add_argument("--server", default=str(ROOT / "build/bin/tools/limbo_serve_gpu"))
    ap.add_argument("--kv", default="q8"); ap.add_argument("--ctx", type=int, default=32768); ap.add_argument("--port", type=int, default=18097)
    ap.add_argument("--slots", type=int, default=2); ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--extra", default=""); ap.add_argument("--dump", default=""); ap.add_argument("--sampled", action="store_true")
    a = ap.parse_args()
    res = {}
    for mode in a.modes.split(","):
        print(f"[{mode}] running ...", flush=True)
        res[mode] = run(a, mode)
    base = res.get("off")
    print(f"\n{'task':8} {'mode':28} {'tokens':>6} {'tok/s':>7} {'vs off':>7} {'tok/pass':>8} {'kept':>9} {'text':>6}")
    ok = True
    for name in a.tasks.split(",") + (["sampled"] if a.sampled else []):
        for mode, r in res.items():
            if name not in r: continue
            x = r[name]
            same = "-" if name == "sampled" or not base or name not in base else ("same" if x["text"] == base[name]["text"] else "DIFF")
            ok &= same != "DIFF"
            sp = f"{x['tps'] / base[name]['tps']:.2f}x" if base and name in base and base[name]["tps"] > 0 else "-"
            tpp = f"{x['tpp']:.2f}" if "tpp" in x else "-"
            kept = f"{x['kept']}/{x['drafted']}" if "kept" in x else "-"
            print(f"{name:8} {mode:28} {x['n']:6d} {x['tps']:7.1f} {sp:>7} {tpp:>8} {kept:>9} {same:>6}")
    for mode, r in res.items():
        print(f"[{mode}] " + " | ".join(r["_start"]))
    if a.dump: pathlib.Path(a.dump).write_text(json.dumps({m: {k: v for k, v in r.items() if k != "_start"} for m, r in res.items()}))
    print("greedy texts identical across modes" if ok else "GREEDY TEXTS DIFFER")
    sys.exit(0 if ok else 1)

main()
