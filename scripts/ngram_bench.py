#!/usr/bin/env python3
# Copyright (c) 2026 ITERFORM. All rights reserved.
"""limbo_serve_gpu under each --spec mode on repetition-heavy and prose tasks (greedy): speed, spec stats, and whether the text equals --spec off's.

  scripts/ngram_bench.py --model file.gguf [--modes off,mtp,ngram,auto] [--kv q4] [--ctx 32768] [--max-tokens 1200] [--ngram-n N] [--extra "-vv"]

Tasks: repeat (a 200-line C++ function reproduced with two renames), edit (a header quoted back with one method added), prose x2.
The code is a fixed snapshot (git show 50af5ec:...). Starts and stops only its own server processes; run it under `flock build/gpu.lock`."""
import argparse, json, os, pathlib, signal, subprocess, sys, time, urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
TMP = pathlib.Path(os.environ.get("LIMBO_TEST_TMP", ROOT / "build/tmp"))

def snap(path, a=None, b=None):
    t = subprocess.run(["git", "-C", str(ROOT), "show", f"50af5ec:{path}"], capture_output=True, text=True, check=True).stdout.splitlines()
    return "\n".join(t[a:b] if a is not None else t)

def tasks():
    fn = snap("src/serve.cpp", 960, 1160)
    hdr = snap("include/limbo/kvpool.hpp")
    return [
        ("repeat", "Below is a C++ function. Reproduce it exactly, line for line, except: rename every `t_prefill` to `prefill_secs` and every `t_decode` to `decode_secs`. "
                   "Output only the code in one fenced block.\n\n```cpp\n" + fn + "\n```"),
        ("edit", "Here is a header. Add a method `bool empty() const { return held_total() == 0; }` right after `held`, and output the complete new file in one fenced block, "
                 "changing nothing else.\n\n```cpp\n" + hdr + "\n```"),
        ("prose1", "Write a short story (about 200 words) about a lighthouse keeper who finds a message in a bottle."),
        ("prose2", "Explain in plain words how a CPU cache works, then list three ways a program can make better use of it."),
    ]

def ask(port, text, max_tokens, temperature=0, seed=None, slot=None):
    body = {"model": "limbo", "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens, "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": False}, "cache_prompt": False}
    if seed is not None: body["seed"] = seed
    if slot is not None: body["id_slot"] = slot
    rq = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"})
    with urllib.request.urlopen(rq, timeout=1800) as r:
        return json.load(r)

def run(a, mode):
    log = open(TMP / f"ngram-bench-{mode}.log", "w")
    cmd = [a.server, "-m", a.model, "--port", str(a.port), "--ctx", str(a.ctx), "--spec", mode, "--kv", a.kv, "--slots", str(a.slots), "-v"]
    if a.ngram_n: cmd += ["--ngram-n", str(a.ngram_n)]
    if a.ngram_max: cmd += ["--ngram-max", str(a.ngram_max)]
    cmd += a.extra.split()
    srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    out = {}
    try:
        for _ in range(600):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=1); break
            except Exception:
                if srv.poll() is not None: raise SystemExit("server exited: " + open(log.name).read()[-2000:])
                time.sleep(1)
        ask(a.port, "Say hi.", 8)  # warm: the graphs are captured on first use
        for name, text in tasks():
            if a.only and name not in a.only.split(","): continue
            t0 = time.time()
            d = ask(a.port, text, a.max_tokens)
            t = d["timings"]
            out[name] = dict(text=d["choices"][0]["message"]["content"], n=d["usage"]["completion_tokens"], tps=t["predicted_per_second"], pps=t["prompt_per_second"],
                             wall=time.time() - t0)
        if not a.only or "sampled" in a.only.split(","):
            d = ask(a.port, tasks()[2][1], a.max_tokens, 0.7, seed=7)  # one sampled request: must run and be sane
            out["sampled"] = dict(text=d["choices"][0]["message"]["content"], n=d["usage"]["completion_tokens"], tps=d["timings"]["predicted_per_second"], pps=0, wall=0)
    finally:
        srv.send_signal(signal.SIGTERM)
        try: srv.wait(timeout=60)
        except subprocess.TimeoutExpired: srv.kill()
        log.close()
    out["_spec"] = [l.split("spec=")[1].strip()[:150] for l in open(log.name) if "spec=" in l]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--modes", default="off,mtp,ngram,auto")
    ap.add_argument("--server", default=str(ROOT / "build/bin/tools/limbo_serve_gpu"))
    ap.add_argument("--kv", default="q4"); ap.add_argument("--ctx", type=int, default=32768); ap.add_argument("--port", type=int, default=18097)
    ap.add_argument("--slots", type=int, default=2); ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--ngram-n", type=int, default=0); ap.add_argument("--ngram-max", type=int, default=0)
    ap.add_argument("--extra", default="", help="more server flags, e.g. -vv")
    ap.add_argument("--dump", default="")
    ap.add_argument("--only", default="", help="comma list of tasks to run (repeat, edit, prose1, prose2, sampled)")
    a = ap.parse_args()
    res = {}
    for mode in a.modes.split(","):
        print(f"[{mode}] running ...", flush=True)
        res[mode] = run(a, mode)
        for k, v in res[mode]["_spec"][:3]: pass
    base = res.get("off")
    print(f"\n{'task':8} {'mode':6} {'tokens':>6} {'decode tok/s':>13} {'vs off':>7} {'same text as off':>17}")
    ok = True
    for name in [t[0] for t in tasks()] + ["sampled"]:
        for mode, r in res.items():
            if name not in r: continue
            x = r[name]
            same = "-" if name == "sampled" or not base or name not in base else ("yes" if x["text"] == base[name]["text"] else "NO")
            ok &= same != "NO"
            sp = f"{x['tps'] / base[name]['tps']:.2f}x" if base and name in base and base[name]["tps"] > 0 else "-"
            print(f"{name:8} {mode:6} {x['n']:6d} {x['tps']:13.1f} {sp:>7} {same:>17}")
    for mode, r in res.items():
        print(f"[{mode}] spec lines: " + " | ".join(r["_spec"][:6]))
    if a.dump: pathlib.Path(a.dump).write_text(json.dumps({m: {k: v for k, v in r.items() if k != "_spec"} for m, r in res.items()}))
    print("greedy texts identical across modes" if ok else "GREEDY TEXTS DIFFER")
    sys.exit(0 if ok else 1)

main()
