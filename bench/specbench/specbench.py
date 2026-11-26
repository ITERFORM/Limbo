#!/usr/bin/env python3
# Copyright (c) 2026 ITERFORM. All rights reserved.
"""
specbench.py — the fitting harness: which limbo configuration is the best practical one for a given
model, machine and workload, measured on real prompts against the real server.

Migrated from our specbench project (the llama.cpp-adaptive-kv-streaming harness) and adapted to
limbo: the lifecycle, sweep and ranking machinery is specbench's; the server surface is
`limbo_serve` / `limbo_serve_gpu`, and every timing comes from the response's own `timings` block
rather than from scraped logs.

Two ways to run it:

  1. ATTACHED  — benchmark a limbo server you already started yourself:
        bench/specbench/specbench.py --url http://127.0.0.1:8090 --profile tools

  2. MANAGED   — specbench starts/stops limbo_serve_gpu itself and sweeps the fitting space:
        bench/specbench/specbench.py --launch --model MODEL.gguf --profile tools \
            --sweep kv=f16,q8,q4 --sweep spec=off,auto --sweep prefill-planes=2,3

The combination space is the cartesian product of the sweeps (--sweep FLAG=v1,v2 or --sweep-file
grid.json); `--sweep-limbo` is the full practical grid (kv x spec x prefill-planes). Every
configuration runs warmup + N completions on the selected profile's real prompt at temperature 0.
`--dry-run` prints the plan without running; `--resume` continues a killed campaign, re-running only
rows that are missing or failed.

"Survives contact with reality" is a set of gates, not a claim:

  * CORRECTNESS — the greedy text of every speculative configuration must equal its plain twin
    (same numerics, --spec off). Missing twins are run automatically. A configuration that diverges
    is marked `correct: false`, excluded from the ranking, and printed first in the report.
  * STABILITY — every run of one configuration must produce the same greedy text; a configuration
    that flip-flops between runs is marked `stable: false` and likewise excluded.
  * REALITY — refusals, OOMs, server deaths and request errors become the row's `status` / `oom`
    fields with their logs, never dropped rows. results.json / results.csv are rewritten after every
    row. The server file descriptor lifecycle is leak-free across long campaigns.
  * THE MACHINE — every row carries the fingerprint (GPU, driver, CPU, RAM, kernel, repo HEAD), and
    per-row min/max/stddev are recorded so the report can state ties instead of inventing winners.

`--self-test` exercises the whole planning/correctness/report machinery on synthetic rows (the
harness's own gate: it must be able to fail before it is trusted to pass).

The report ranks by --objective (default: the server's eval tok/s) and prints the winning launch
command verbatim, plus one fittings table per profile.
"""

from __future__ import annotations

import argparse
import atexit
import csv
import hashlib
import itertools
import json
import math
import os
import re
import shlex
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent                      # bench/specbench/ -> repo root
PROMPTS = REPO / "bench" / "prompts"           # the real-workload prompt set
DEFAULT_SERVER = str(REPO / "build" / "bin" / "tools" / "limbo_serve_gpu")

# The flags the limbo server accepts (managed mode validates sweeps against this before the first
# boot, so a typo fails fast instead of costing a server lifecycle).
SERVER_FLAGS = {
    "-m", "--model", "--host", "--port", "--model-name", "--max-tokens", "--temperature", "--top-k",
    "--top-p", "--min-p", "--log-prompts", "--quiet", "--threads", "--min-free-gb", "--kv",
    "--prefill-planes", "--spec", "--ngram-n", "--ngram-max", "--draft-model", "--draft-quant",
    "--draft-vocab", "--draft-k", "--slots", "--checkpoints", "--vram-reserve-mib", "--api-key",
    "--ctx", "-v", "-vv",
}
# Flags whose values change numerics (the correctness family of a config).
NUMERIC_FLAGS = {"--kv", "--prefill-planes", "--ctx"}
# Flags that only change speculation (the correctness twin is this config with them off).
SPEC_FLAGS = {"--spec", "--ngram-n", "--ngram-max", "--draft-model", "--draft-quant",
              "--draft-vocab", "--draft-k"}


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #


def last(xs):
    return xs[-1] if xs else None


def avg(xs):
    return sum(xs) / len(xs) if xs else None


def med(xs):
    return statistics.median(xs) if xs else None


def stdev(xs):
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


def fmt(x, digits=2):
    return "-" if x is None else f"{x:.{digits}f}"


def http_json(url: str, payload: dict | None = None, timeout: float = 600.0):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} from {url}: {e.read()[:400].decode(errors='replace')}") from e


def is_healthy(base: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(base + "/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def wait_ready(base: str, proc: subprocess.Popen, timeout: float) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        if is_healthy(base):
            return True
        time.sleep(0.4)
    return False


def gpu_mem_used_mib() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip()
        return int(out.splitlines()[0])
    except Exception:
        return None


def gpu_mem_free_mib() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip()
        return int(out.splitlines()[0])
    except Exception:
        return None


def machine_fingerprint() -> dict:
    def sh(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout.strip()
        except Exception:
            return ""
    mem = ""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                mem = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass
    cpu = ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass
    return {
        "gpu": sh(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                   "--format=csv,noheader"]),
        "cpu": cpu,
        "ram": mem,
        "kernel": sh(["uname", "-sr"]),
        "repo_head": sh(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"]),
        "vram_used_at_start_mib": gpu_mem_used_mib(),
    }


# --------------------------------------------------------------------------- #
# prompt sizing: real prompts filled to the depth the benchmark wants
# --------------------------------------------------------------------------- #


_TOK_COUNT_CACHE: dict = {}


def token_count(model: str | None, text: str) -> int | None:
    """Exact token count through tools/tok_dump (offline: the GGUF's own tokenizer, no server).
    None when the tool or the model is unavailable."""
    if not model:
        return None
    key = (model, hashlib.sha256(text.encode()).hexdigest()[:16])
    if key in _TOK_COUNT_CACHE:
        return _TOK_COUNT_CACHE[key]
    try:
        out = subprocess.run([str(REPO / "build" / "bin" / "tools" / "tok_dump"), "-m", model, "--stdin"],
                             input=text, capture_output=True, text=True, timeout=300).stdout
        n = len(json.loads(out.strip().splitlines()[-1]))
    except Exception:
        n = None
    _TOK_COUNT_CACHE[key] = n
    return n


def fill_prompt(base: str, model: str | None, target: int, counter=None) -> str:
    """Repeat the real prompt to ~target tokens (the bench/prompts/long.txt idiom: repeated to
    length). Never exceeds the target: the server's context limit is hard. Token counts come from
    the GGUF's own tokenizer (tools/tok_dump, offline); the row records the depth actually reached
    either way, so an estimate is visible rather than assumed."""
    count = counter or token_count
    if target <= 0:
        return base
    unit = count(model, base)
    if not unit:
        est = max(1, int(target / max(1.0, len(base) / 3.5)))
        return "\n\n".join([base] * est)
    if unit > target:
        return base  # the base alone already exceeds the target: it is the shortest honest prompt
    repeats = max(1, target // unit)   # floor: whole repeats may never overshoot the hard limit
    text = "\n\n".join([base] * repeats)
    while repeats * unit + unit // 4 <= target:   # grow only while the estimate stays in range
        cand = "\n\n".join([base] * (repeats + 1))
        n = count(model, cand)
        if n is None or n > target:
            break
        repeats += 1
        text = cand
    return text


def prompt_fits(base: str, model: str | None, target: int, counter=None) -> int | None:
    """The base prompt's token count when it cannot fit the target (the row is refused without a
    server boot); None when it fits. The fitting question includes 'can this workload run here at
    all', and the answer is recorded, not assumed."""
    count = counter or token_count
    unit = count(model, base)
    return unit if (target > 0 and unit and unit > target) else None


# --------------------------------------------------------------------------- #
# completions: one request, measured client-side; the server's own timings ride the response
# --------------------------------------------------------------------------- #


def completion(base: str, prompt: str, max_tokens: int, temperature: float, timeout: float) -> dict:
    body = {"model": "limbo", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature, "stream": True}
    t0 = time.time()
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    ttft = None
    parts = []
    fin = {}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                # A wall-clock deadline, not a socket timeout: the server's SSE keepalives trickle
                # every few seconds and would defeat a per-read timeout while a run wedges.
                if time.time() - t0 > timeout:
                    raise RuntimeError(f"wall-clock deadline ({timeout:.0f}s) exceeded mid-stream")
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                if "timings" in chunk:
                    fin = chunk
                    continue
                d = (chunk.get("choices") or [{}])[0].get("delta") or {}
                piece = (d.get("reasoning_content") or "") + (d.get("content") or "")
                if piece and ttft is None:
                    ttft = time.time() - t0
                if piece:
                    parts.append(piece)
                if d.get("tool_calls"):
                    parts.append(json.dumps(d["tool_calls"], sort_keys=True))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read()[:400].decode(errors='replace')}") from e
    wall = time.time() - t0
    if not fin:
        raise RuntimeError("the stream ended without a timings chunk (no completion recorded)")
    t = fin.get("timings", {})
    usage = fin.get("usage", {})
    out = usage.get("completion_tokens") or 0
    return {"wall": wall, "ttft": ttft, "text": "".join(parts),
            "output_tokens": out or None,
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached": usage.get("prompt_tokens_details", {}).get("cached_tokens", 0),
            "client_tps": out / wall if wall > 0 and out else None,
            "eval_tps": t.get("predicted_per_second"),
            "prompt_tps": t.get("prompt_per_second"),
            "finish": (fin.get("choices") or [{}])[0].get("finish_reason")}


# --------------------------------------------------------------------------- #
# server lifecycle (managed mode)
# --------------------------------------------------------------------------- #


def pids_on_port(port: int) -> list[int]:
    try:
        out = subprocess.run(["ss", "-ltnp", f"sport = :{port}"], capture_output=True,
                             text=True, timeout=5).stdout
        return [int(m) for m in re.findall(r"pid=(\d+)", out)]
    except Exception:
        return []


def kill_stale_on_port(port: int) -> None:
    for pid in pids_on_port(port):
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace")
        except Exception:
            continue
        if "limbo_serve" in cmd:  # never kill anything that is not ours
            print(f"    killing stale limbo_serve pid {pid} on port {port}")
            os.kill(pid, signal.SIGTERM)
    t0 = time.time()
    while pids_on_port(port) and time.time() - t0 < 15:
        time.sleep(0.3)


_PROC = {"proc": None}


def start_server(tokens: list[str], logfile: Path) -> subprocess.Popen:
    log = open(logfile, "ab")
    try:
        return subprocess.Popen(tokens, stdout=log, stderr=subprocess.STDOUT)
    finally:
        log.close()  # the child owns its dup; the parent must not leak the handle per config


def stop_server(proc: subprocess.Popen | None, grace: float = 10.0) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    t0 = time.time()
    while proc.poll() is None and time.time() - t0 < grace:
        time.sleep(0.2)
    if proc.poll() is None:
        proc.kill()
        proc.wait()


def _cleanup():
    stop_server(_PROC["proc"])


atexit.register(_cleanup)


# --------------------------------------------------------------------------- #
# sweep planning, correctness families and commands
# --------------------------------------------------------------------------- #


def normalise_flag(flag: str) -> str:
    f = flag.strip()
    return f if f.startswith("-") else "--" + f


def label_for(overrides: dict) -> str:
    return "-".join(f"{k.lstrip('-')}={v}" for k, v in overrides.items()) or "baseline"


def family_key(overrides: dict) -> tuple:
    """Everything that changes numerics: the correctness twin of a config shares this key."""
    return tuple(sorted((k, v) for k, v in overrides.items() if k in NUMERIC_FLAGS))


def spec_flags_off(overrides: dict) -> dict:
    """This config with speculation off — its correctness reference."""
    return {k: v for k, v in overrides.items() if k not in SPEC_FLAGS} | {"--spec": "off"}


def compose_configs(args) -> list[tuple[str, dict]]:
    dims: dict[str, list[str]] = {}
    for spec in args.sweep or []:
        if "=" not in spec:
            sys.exit(f"bad --sweep spec (need FLAG=v1,v2,...): {spec!r}")
        flag, _, vals = spec.partition("=")
        dims[normalise_flag(flag)] = [v.strip() for v in vals.split(",") if v.strip()]
    if args.sweep_file:
        obj = json.loads(Path(args.sweep_file).read_text())
        for flag, vals in obj.items():
            dims[normalise_flag(flag)] = [str(v) for v in vals]
    if args.sweep_limbo:
        dims.setdefault("--kv", ["f16", "q8", "q4"])
        dims.setdefault("--spec", ["off", "mtp", "ngram", "auto"])
        dims.setdefault("--prefill-planes", ["2", "3"])
    for flag in dims:
        if flag not in SERVER_FLAGS:
            sys.exit(f"unknown server flag in sweep: {flag} (known: {' '.join(sorted(SERVER_FLAGS))})")

    configs: list[tuple[str, dict]] = [("baseline", {})]
    keys = list(dims)
    for combo in itertools.product(*(dims[k] for k in keys)):
        ov = dict(zip(keys, combo))
        configs.append((label_for(ov), ov))

    # correctness needs every config's plain twin: add the missing ones (marked with an -off suffix
    # when the twin is not itself part of the grid).
    have = {json.dumps(ov, sort_keys=True) for _, ov in configs}
    for label, ov in list(configs):
        twin = spec_flags_off(ov)
        if json.dumps(twin, sort_keys=True) not in have:
            configs.append((label_for(twin), twin))
            have.add(json.dumps(twin, sort_keys=True))
    return configs


def command_for(args, overrides: dict) -> list[str]:
    # Defaults are placeholders the overrides replace: a swept flag appears exactly once.
    opts = {"--ctx": str(args.ctx), "--slots": str(args.slots)}
    if args.server.endswith("limbo_serve_gpu"):
        opts["--min-free-gb"] = str(args.min_free_gb)
        opts["--vram-reserve-mib"] = str(args.vram_reserve_mib)
    opts.update(overrides)
    tokens = [args.server, "-m", args.model, "--host", "127.0.0.1", "--port", str(args.port)]
    for k, v in opts.items():
        tokens += [k, v]
    if args.api_key:
        tokens += ["--api-key", args.api_key]
    return tokens


# --------------------------------------------------------------------------- #
# measurement, summarising and the correctness engine
# --------------------------------------------------------------------------- #


def summarise(results: list[dict], pre: dict | None = None) -> dict:
    def col(key):
        return [r[key] for r in results if r.get(key) is not None]
    ev, cl = col("eval_tps"), col("client_tps")
    texts = [r["text"] for r in results if r.get("text") is not None]
    probe = pre or (results[0] if results else {}) or {}
    return {
        "n_runs": len(results),
        "depth": probe.get("prompt_tokens"),
        "prefill_tps": probe.get("prompt_tps"),
        "ttft_uncached": pre.get("ttft") if pre else None,
        "ttft_med": med(col("ttft")),
        "eval_tps_mean": avg(ev),
        "eval_tps_med": med(ev),
        "eval_tps_min": min(ev) if ev else None,
        "eval_tps_max": max(ev) if ev else None,
        "eval_tps_std": round(stdev(ev), 3) if ev else None,
        "client_tps_mean": avg(cl),
        "client_tps_med": med(cl),
        "client_tps_min": min(cl) if cl else None,
        "client_tps_max": max(cl) if cl else None,
        "prompt_tps_mean": avg(col("prompt_tps")),
        "cached_max": max(col("cached"), default=None),
        "out_tokens_med": med(col("output_tokens")),
        "stable": all(t == texts[0] for t in texts) if texts else None,
        "text_sha": hashlib.sha256(texts[0].encode()).hexdigest()[:16] if texts else None,
    }


def correctness_pass(rows: list[dict]) -> None:
    """Set `correct` on every row: same family (numerics), compared to the family's --spec off row,
    through the recorded text hashes so the verdict is derivable from the artifact alone."""
    by_family: dict[tuple, dict] = {}
    for r in rows:
        r.setdefault("correct", None)  # the verdict key always exists: None = no twin to compare
        if r.get("overrides", {}).get("--spec") == "off":
            by_family[(r["profile"],) + family_key(r.get("overrides", {}))] = r
    for r in rows:
        twin = by_family.get((r["profile"],) + family_key(r.get("overrides", {})))
        if twin is not None and r is not twin and r.get("text_sha") and twin.get("text_sha"):
            r["correct"] = r["text_sha"] == twin["text_sha"]
        elif r is not twin and twin is not None:
            r["correct"] = None  # no text recorded (the row failed) — never a pass


def load_profiles(which: str) -> dict[str, str]:
    wanted = ["tools", "code", "prose", "long", "hi"] if which == "all" else which.split(",")
    out = {}
    for name in wanted:
        p = PROMPTS / f"{name}.txt"
        if not p.exists():
            sys.exit(f"no such profile prompt: {p}")
        out[name] = p.read_text()
    return out


def write_results(outdir: Path, header: dict, rows: list[dict]) -> None:
    header = dict(header, finished=time.strftime("%Y-%m-%d %H:%M:%S"), n_rows=len(rows))
    (outdir / "results.json").write_text(json.dumps({"header": header, "rows": rows}, indent=1))
    if rows:
        keys = sorted({k for r in rows for k in r})
        with open(outdir / "results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v
                            for k, v in r.items()})


# --------------------------------------------------------------------------- #
# the campaign
# --------------------------------------------------------------------------- #


def run_campaign(args, outdir: Path, header: dict) -> list[dict]:
    base = f"http://127.0.0.1:{args.port}"
    prompts = load_profiles(args.profile)
    configs = compose_configs(args)
    rows: list[dict] = []

    if args.resume and (outdir / "results.json").exists():
        old = json.loads((outdir / "results.json").read_text()).get("rows", [])
        rows = [r for r in old if r.get("status") in ("ok", "prompt-too-long") and
                (r.get("n_runs") or r.get("status") == "prompt-too-long")]
        done = {(r["profile"], r["label"]) for r in rows}
        print(f"-- resume: keeping {len(rows)} finished rows")
    else:
        done = set()

    (outdir / "logs").mkdir(parents=True, exist_ok=True)
    write_results(outdir, header, rows)

    if args.dry_run:
        n_rows = len(configs) * len(prompts)
        print(f"-- dry run: {len(configs)} configs x {len(prompts)} profiles = {n_rows} rows, "
              f"~{n_rows * (args.warmup + args.runs) * args.max_tokens} generated tokens")
        for label, ov in configs:
            print(f"   {label}")
        return rows

    interrupted = False
    try:
        for profile, prompt in prompts.items():
            print(f"\n=== profile {profile} ({len(prompt)} chars of real prompt) ===")
            for label, overrides in configs:
                if (profile, label) in done:
                    print(f"\n-- {profile}__{label}: already recorded, skipped")
                    continue
                name = f"{profile}__{label}"
                eff_ctx = int(overrides.get("--ctx", args.ctx))
                headroom = args.max_tokens + 1024
                target = min(args.prompt_tokens or (eff_ctx - headroom), max(eff_ctx - headroom, 0))
                print(f"\n-- {name}  (prompt filled to ~{target} tokens)")
                too_big = prompt_fits(prompts[profile], args.model, target)
                if too_big:
                    row = {"label": label, "profile": profile, "overrides": overrides,
                           "command": "(not launched)", "status": "prompt-too-long", "oom": False,
                           "base_tokens": too_big, "target_tokens": target}
                    print(f"    the base prompt is {too_big} tokens, over the {target}-token target: "
                          f"recorded without booting a server")
                    rows.append(row)
                    write_results(outdir, header, rows)
                    continue
                prompt = fill_prompt(prompts[profile], args.model, target)
                kill_stale_on_port(args.port)
                logfile = outdir / "logs" / f"{name}.log"
                tokens = command_for(args, overrides)
                print("    " + " ".join(shlex.quote(t) for t in tokens))
                vram_before = gpu_mem_used_mib()
                t0 = time.time()
                proc = start_server(tokens, logfile)
                _PROC["proc"] = proc
                row = {"label": label, "profile": profile, "overrides": overrides,
                       "command": " ".join(shlex.quote(t) for t in tokens), "status": "ok",
                       "oom": False, "vram_used_mib": None, "load_s": None,
                       # The pool sizing is sensitive to ~100 MiB of desktop VRAM jitter (a full-256k
                       # pool has been seen fit and not fit at the same flags): record what the row
                       # had to work with, so a marginal fit is explainable from results.json alone.
                       "vram_free_at_boot_mib": gpu_mem_free_mib()}
                if not wait_ready(base, proc, args.boot_timeout):
                    rc = proc.poll()
                    log = logfile.read_text(errors="replace")[-2000:]
                    row["status"] = "died" if rc is not None else "no-health"
                    row["oom"] = bool(re.search(r"out of (device )?memory|cudaMalloc", log, re.I))
                    if rc == 3:
                        row["status"] = "refused"
                    print(f"    server did not come up (rc={rc}, oom={row['oom']}) — recorded")
                    stop_server(proc)
                    rows.append(row)
                    write_results(outdir, header, rows)
                    continue
                row["load_s"] = round(time.time() - t0, 2)
                after = gpu_mem_used_mib()
                row["vram_used_mib"] = (after - vram_before) if (
                    after is not None and vram_before is not None) else None

                results = []
                pre = None
                try:
                    pre = completion(base, prompt, args.max_tokens, args.temperature, args.request_timeout)
                    print(f"    [probe] depth={pre['prompt_tokens']} "
                          f"prefill={fmt(pre['prompt_tps'])} tok/s "
                          f"ttft_uncached={fmt(pre['ttft'])}s")
                except Exception as e:
                    print(f"    [probe] ERROR: {e}")
                    row["status"] = "error"
                    row["error"] = str(e)
                    stop_server(proc)
                    rows.append(row)
                    write_results(outdir, header, rows)
                    continue
                for i in range(args.warmup):
                    try:
                        completion(base, prompt, args.max_tokens, args.temperature,
                                   args.request_timeout)
                    except Exception as e:
                        print(f"    [warmup {i+1}] ERROR: {e}")
                for i in range(args.runs):
                    try:
                        r = completion(base, prompt, args.max_tokens, args.temperature,
                                       args.request_timeout)
                    except Exception as e:
                        print(f"    [run {i+1}] ERROR: {e}")
                        row["status"] = "error"
                        row["error"] = str(e)
                        break
                    results.append(r)
                    print(f"    [run {i+1}/{args.runs}] ttft={fmt(r['ttft'])} "
                          f"out={r['output_tokens']} client={fmt(r['client_tps'])} "
                          f"eval={fmt(r['eval_tps'])} cached={r['cached']}")

                row.update(summarise(results, pre))
                stop_server(proc)
                rows.append(row)
                write_results(outdir, header, rows)
    except KeyboardInterrupt:
        interrupted = True
        print("\n-- interrupted: stopping the server, keeping every recorded row")

    correctness_pass(rows)
    write_results(outdir, header, rows)
    if interrupted:
        print("-- partial campaign; run again with --resume to finish it")
    return rows


# --------------------------------------------------------------------------- #
# report: rank, disclose, and print the winning command verbatim
# --------------------------------------------------------------------------- #


OBJECTIVE_KEY = {"eval_tps": "eval_tps_mean", "client_tps": "client_tps_mean",
                 "ttft": "ttft_med", "prompt_tps": "prompt_tps_mean"}


def report(rows: list[dict], args, outdir: Path, header: dict) -> None:
    key = OBJECTIVE_KEY[args.objective]
    lines = [f"# specbench fittings — {time.strftime('%Y-%m-%d')}", ""]
    m = header.get("machine", {})
    lines += [f"machine: {m.get('gpu')} · {m.get('cpu')} · {m.get('ram')} · {m.get('kernel')}",
              f"model: {header.get('model')} · repo {m.get('repo_head')} · "
              f"objective: {args.objective} · {args.runs} runs x {args.max_tokens} tokens",
              "", "Correctness: every speculative config must match its plain twin's greedy text; "
              "stability: all runs of a config must agree. Failed rows leave the ranking and lead "
              "the report. One campaign is one sample: treat gaps inside the run spread as ties.", ""]
    for profile in sorted({r["profile"] for r in rows}):
        pr = [r for r in rows if r["profile"] == profile]
        bad = [r for r in pr if r.get("correct") is False or r.get("stable") is False]
        good = [r for r in pr if r.get("status") == "ok" and r.get("correct") is not False
                and r.get("stable") is not False]
        good.sort(key=lambda r: (r.get(key) is not None, r.get(key) or 0),
                  reverse=args.objective != "ttft")
        lines.append(f"## profile {profile}")
        if bad:
            lines.append("**Excluded (failed the gates):**")
            for r in bad:
                why = []
                if r.get("correct") is False:
                    why.append("greedy text diverged from its plain twin")
                if r.get("stable") is False:
                    why.append("runs disagree with each other")
                lines.append(f"- `{r['label']}`: {'; '.join(why)}")
            lines.append("")
        if not good:
            lines.append("no configuration completed.\n")
        else:
            lines.append("| rank | config | objective | depth | prefill tok/s | eval tok/s (min-max) | "
                         "client | ttft | out | kv/planes/spec | correct | stable |")
            lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
            for i, r in enumerate(good, 1):
                o = r.get("overrides", {})
                knob = f"{o.get('--kv', 'auto')}/{o.get('--prefill-planes', '2')}/{o.get('--spec', 'auto')}"
                spread = (f"{fmt(r.get('eval_tps_min'))}-{fmt(r.get('eval_tps_max'))}"
                          if r.get("eval_tps_min") is not None else "-")
                corr = r.get("correct")
                lines.append(f"| {i} | `{r['label']}` | {fmt(r.get(key))} "
                             f"| {r.get('depth') or '-'} | {fmt(r.get('prefill_tps'))} | {spread} "
                             f"| {fmt(r.get('client_tps_mean'))} | {fmt(r.get('ttft_med'))} "
                             f"| {r.get('out_tokens_med')} | {knob} "
                             f"| {'-' if corr is None else corr} | {r.get('stable')} |")
        w = good[0] if good else None
        if w:
            lines += ["", f"**WINNER ({profile}):** `{w['label']}` — {fmt(w.get(key))} {args.objective}",
                      "```bash", w["command"], "```", ""]
        unmeasured = [r for r in pr if r.get("status") != "ok" and
                      not (r.get("correct") is False or r.get("stable") is False)]
        if unmeasured:
            lines.append("**Not measured (reality rows — the boundaries, kept as recorded):**")
            for r in unmeasured:
                why = r.get("status", "?")
                if why == "prompt-too-long":
                    why += (f" — the base prompt is {r.get('base_tokens')} tokens, over the "
                            f"{r.get('target_tokens')}-token target")
                elif why == "error" and r.get("error"):
                    why += f" — {str(r['error'])[:120]}"
                lines.append(f"- `{r['label']}`: {why}")
            lines.append("")
        misses = [r for r in pr if r.get("depth") and r.get("cached_max") == 0]
        if misses:
            lines.append("**Tape misses (the cache served nothing — every run re-prefilled the full "
                         "prompt; a divergence that lost the tape, worth an engine look):**")
            for r in misses:
                lines.append(f"- `{r['label']}`: depth {r['depth']}, cached 0 of it, "
                             f"ttft_uncached {fmt(r.get('ttft_uncached'))}s")
            lines.append("")
    text = "\n".join(lines)
    print("\n" + text)
    (outdir / "fittings.md").write_text(text + "\n")


# --------------------------------------------------------------------------- #
# the harness's own gate
# --------------------------------------------------------------------------- #


def self_test() -> int:
    """Exercise planning, correctness, stability and reporting on synthetic rows. Every check must
    be able to fail: the mutations at the end prove the gate is not decorative."""
    fails = 0

    def check(ok, what):
        nonlocal fails
        print(("ok   " if ok else "FAIL ") + what)
        fails += 0 if ok else 1

    class A:
        sweep = ["kv=f16,q8", "spec=off,auto"]
        sweep_file = None
        sweep_limbo = False
    configs = compose_configs(A())
    check(len(configs) == 6, "grid: 2x2 plus the baseline, plus the baseline's missing plain twin")
    twin_missing = [c for c in configs if c[1].get("--spec") == "off"]
    check(len(twin_missing) == 3, "grid: a --spec off row exists for every numerics family")

    class Args:
        server = DEFAULT_SERVER
        model = "m.gguf"
        port = 8090
        ctx = 4096
        slots = 1
        min_free_gb = 0.0
        vram_reserve_mib = 768
        api_key = ""
    cmd = command_for(Args(), {"--ctx": "262144", "--kv": "q4"})
    check(cmd.count("--ctx") == 1 and "262144" in cmd and "4096" not in cmd,
          "command: a swept flag replaces the default exactly once")
    check(cmd.count("--kv") == 1 and "q4" in cmd, "command: swept values ride along")

    filled = fill_prompt("word " * 100, None, 500)   # no tokenizer: the documented estimate path
    check(len(filled) > 400, "fill: the estimate path reaches the requested scale")
    check(fill_prompt("word " * 100, None, 0) == "word " * 100, "fill: target 0 keeps the prompt as is")

    # the hard-limit invariants, on a stub counter: one prompt unit = 100 tokens
    stub = lambda model, text: 100 * (text.count("BASE") or 1) + 9 * text.count("\n\n")
    check(prompt_fits("BASE", None, 50, counter=stub) == 100,
          "fit: a base prompt over the target is reported before any boot")
    check(prompt_fits("BASE", None, 250, counter=stub) is None, "fit: a base prompt that fits passes")
    check(fill_prompt("BASE", None, 150, counter=stub).count("BASE") == 1,
          "fill: a single repeat is the floor — never a partial prompt, never an overshoot")
    grown = fill_prompt("BASE", None, 350, counter=stub)
    check(stub(None, grown) <= 350, "fill: growing whole repeats never exceeds the hard target")

    rows = []
    for label, ov in configs:
        rows.append({"profile": "tools", "label": label, "overrides": ov, "status": "ok",
                     "n_runs": 2, "eval_tps_mean": 100.0, "client_tps_mean": 90.0,
                     "ttft_med": 0.05, "out_tokens_med": 64, "text_sha": "A" if ov.get("--kv") == "f16" else "B",
                     "stable": True, "command": "cmd"})
    correctness_pass(rows)
    check(all(r["correct"] in (True, None) for r in rows), "correctness: identical family texts pass")

    # mutation 1: a speculative row's text drifts -> it must be excluded and lead the report
    mut = json.loads(json.dumps(rows))
    for r in mut:
        if r["label"] == "kv=f16-spec=auto":
            r["text_sha"] = "X"
    correctness_pass(mut)
    check(next(r for r in mut if r["label"] == "kv=f16-spec=auto")["correct"] is False,
          "mutation: a drifted twin is caught")
    # mutation 2: runs disagree -> unstable rows leave the ranking
    for r in mut:
        if r["label"] == "kv=q8-spec=auto":
            r["stable"] = False
    import io
    import contextlib
    class B:
        objective = "eval_tps"
        runs = 2
        max_tokens = 64
    buf = io.StringIO()
    out = HERE / "selftest.tmp"
    out.mkdir(exist_ok=True)
    with contextlib.redirect_stdout(buf):
        report(mut, B(), out, {"machine": {"gpu": "test"}, "model": "test"})
    text = (out / "fittings.md").read_text()
    check("Excluded (failed the gates)" in text and "kv=f16-spec=auto" in text.split("## profile")[1],
          "report: the failed twin leads the report")
    check("| 1 | `kv=q8-spec=off`" in text or "| 1 | `baseline`" in text,
          "report: unstable and diverged rows are not rankable")
    check("WINNER" in text and "```bash" in text, "report: the winning command is printed verbatim")

    # the reality rows must stay in the report: a boundary is a result, not a gap
    for r in mut:
        if r["label"] == "kv=q8-spec=off":
            r.update({"status": "prompt-too-long", "base_tokens": 4977, "target_tokens": 2560})
        if r["label"] == "kv=q8-spec=off-prefill-planes=2":
            r.update({"status": "refused"})
    with contextlib.redirect_stdout(buf):
        report(mut, B(), out, {"machine": {"gpu": "test"}, "model": "test"})
    text = (out / "fittings.md").read_text()
    check("Not measured (reality rows" in text and "prompt-too-long" in text and "4977" in text,
          "report: reality rows stay in the table with their numbers")
    check("`kv=q8-spec=off`" in text.split("Not measured")[1], "report: the refused boundary is listed")
    for r in mut:
        if r["label"] == "kv=f16-spec=off":
            r.update({"status": "ok", "correct": True, "stable": True, "depth": 258813,
                      "cached_max": 0, "ttft_uncached": 506.3})
    with contextlib.redirect_stdout(buf):
        report(mut, B(), out, {"machine": {"gpu": "test"}, "model": "test"})
    text = (out / "fittings.md").read_text()
    check("Tape misses" in text and "cached 0 of it" in text,
          "report: a tape miss (every run re-prefilled) is disclosed")
    import shutil
    shutil.rmtree(out)
    print(f"self-test: {'FAIL' if fails else 'all green'}")
    return 1 if fails else 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="limbo fitting harness (see bench/specbench/README.md)")
    ap.add_argument("--url", default="http://127.0.0.1:8090", help="ATTACHED mode: server to hit")
    ap.add_argument("--launch", action="store_true", help="MANAGED mode: start/stop the server per config")
    ap.add_argument("--server", default=DEFAULT_SERVER, help="the server binary (managed mode)")
    ap.add_argument("--model", help="the GGUF (managed mode)")
    ap.add_argument("--out", help="results directory (default bench/specbench/results/<stamp>)")
    ap.add_argument("--profile", default="tools", help="prompt set: tools, code, prose, long, hi, all")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--prompt-tokens", type=int, default=0,
                    help="fill the real prompt to N tokens (0: fill the window — ctx minus the "
                         "generation headroom); the exact depth reached is recorded per row")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--slots", type=int, default=1)
    ap.add_argument("--min-free-gb", type=float, default=0.0)
    ap.add_argument("--vram-reserve-mib", type=int, default=768)
    ap.add_argument("--api-key", default="")
    ap.add_argument("--request-timeout", type=float, default=600.0,
                    help="per completion, seconds (a hung request becomes a recorded error)")
    ap.add_argument("--boot-timeout", type=float, default=180.0,
                    help="seconds to wait for /health after a launch")
    ap.add_argument("--sweep", action="append", metavar="FLAG=v1,v2,...")
    ap.add_argument("--sweep-file", help="json grid {flag: [values]}")
    ap.add_argument("--sweep-limbo", action="store_true",
                    help="the full practical grid: kv x spec x prefill-planes")
    ap.add_argument("--objective", default="eval_tps",
                    choices=["eval_tps", "client_tps", "ttft", "prompt_tps"])
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted campaign: keep finished rows, re-run the rest")
    ap.add_argument("--report-only", action="store_true",
                    help="re-render fittings.md from an existing results.json without running")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    ap.add_argument("--self-test", action="store_true", help="run the harness's own gate and exit")
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    if args.self_test:
        return self_test()
    if args.launch and not args.model:
        sys.exit("--launch needs --model")

    stamp = time.strftime("%Y-%m-%d") + "-" + (Path(args.model).stem if args.model else "attached")
    outdir = Path(args.out) if args.out else HERE / "results" / stamp
    outdir.mkdir(parents=True, exist_ok=True)

    prompts = load_profiles(args.profile)
    header = {"objective": args.objective, "runs": args.runs, "warmup": args.warmup,
              "max_tokens": args.max_tokens, "temperature": args.temperature,
              "model": args.model or args.url, "machine": machine_fingerprint(),
              "started": time.strftime("%Y-%m-%d %H:%M:%S"),
              "profiles": {n: {"chars": len(p), "sha256": hashlib.sha256(p.encode()).hexdigest()[:16]}
                           for n, p in prompts.items()}}

    if args.report_only:
        data = json.loads((outdir / "results.json").read_text())
        report(data["rows"], args, outdir, data.get("header", header))
        print(f"\nresults: {outdir}")
        return 0

    if not args.launch:
        print(f"ATTACHED against {args.url}")
        rows = []
        for profile, prompt in prompts.items():
            rs = [completion(args.url, prompt, args.max_tokens, args.temperature,
                             args.request_timeout) for _ in range(args.runs)]
            row = {"label": "attached", "profile": profile, "overrides": {}, "status": "ok",
                   "command": f"(server at {args.url})"}
            row.update(summarise(rs))
            rows.append(row)
            print(f"  {profile}: eval={fmt(row['eval_tps_mean'])} "
                  f"client={fmt(row['client_tps_mean'])} ttft={fmt(row['ttft_med'])}")
        write_results(outdir, header, rows)
    else:
        rows = run_campaign(args, outdir, header)
    report(rows, args, outdir, header)
    print(f"\nresults: {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
