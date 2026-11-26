#!/usr/bin/env bash
# Copyright (c) 2026 ITERFORM. All rights reserved.
# scripts/model_pick.sh -- read models.json and say what runs, with what.
#
#   scripts/model_pick.sh --list              one line a model: name, arch, status
#   scripts/model_pick.sh <name>              the engine and a ready-to-run command line
#   scripts/model_pick.sh <name> --json       the raw entry
#   scripts/model_pick.sh --runnable          only the entries whose status is WORKS
#   scripts/model_pick.sh --support           the arch / type coverage tables
#
# The config is models.json next to this script's parent. It is data, not documentation: every
# status in it came from running the engine on this box.
set -u
root="$(cd "$(dirname "$0")/.." && pwd)"
cfg="$root/models.json"
[ -f "$cfg" ] || { echo "model_pick: no $cfg" >&2; exit 1; }

python3 - "$cfg" "$@" <<'PY'
import json, shlex, sys
import glob as globmod

cfg_path, args = sys.argv[1], sys.argv[2:]
doc = json.load(open(cfg_path))
models = doc["models"]

def resolve(p):
    # HF snapshot paths carry a sha directory that changes on re-download, so the config stores a `*`.
    if not p or "*" not in p:
        return p, True
    hits = sorted(globmod.glob(p))
    return (hits[0], True) if hits else (p, False)

def by_name(n):
    for m in models:
        if m["name"] == n:
            return m
    return None

if not args or args[0] == "--list":
    for m in models:
        eng = m.get("engine") or "-"
        print(f"{m['status']:<14} {m['name']:<38} {m.get('arch','?'):<16} {eng}")
    print()
    for g in doc.get("ladder_missing", []):
        print(f"{g['status']:<14} {g['model']:<38} {g['arch']:<16} {g.get('engine','-')}")
    sys.exit(0)

if args[0] == "--runnable":
    for m in models:
        if m["status"] == "WORKS":
            print(m["name"])
    sys.exit(0)

if args[0] == "--support":
    s = doc["support"]
    for key, title in (("cpu_serve_arch", "CPU   arch"), ("gpu_serve_arch", "GPU   arch"),
                       ("tokenizer_models", "tokenizer"), ("gpu_gemv_types", "GPU   gemv types"),
                       ("gpu_gemm_types", "GPU   gemm types"), ("cpu_lane_types", "CPU   lane types")):
        v = s.get(key)
        if isinstance(v, list):
            print(f"{title:<18} {len(v):>2}  {' '.join(v)}")
        else:
            print(f"{title:<18} {v}")
    print()
    print(f"{'batched prefill':<18} {s['batched_prefill']}")
    print(f"{'GPU geometry gate':<18} {s['gpu_geometry_gate']}")
    sys.exit(0)

m = by_name(args[0])
if m is None:
    print(f"model_pick: no model named {args[0]!r}. Try --list.", file=sys.stderr)
    sys.exit(1)

if len(args) > 1 and args[1] == "--json":
    print(json.dumps(m, indent=2))
    sys.exit(0)

print(f"name     {m['name']}")
print(f"status   {m['status']}")
print(f"arch     {m.get('arch','?')}")
resolved, found = resolve(m.get("path") or "")
if m.get("path"):
    print(f"path     {resolved}")
    if not found:
        print("         !! NOT PRESENT ON DISK -- the glob matched nothing (never fetched, or moved)")
if m.get("role"):
    print(f"role     {m['role']}")
if m.get("engine"):
    print(f"engine   {m['engine']}")
if m.get("blocker"):
    print(f"blocker  {m['blocker']}")
if m.get("notes"):
    print(f"notes    {m['notes']}")
if m.get("measured"):
    print(f"measured {m['measured']}")
if m.get("spec"):
    print(f"spec     {m['spec']}")

env = m.get("env") or {}
real = {k: v for k, v in env.items() if not str(v).startswith("(unset)")}
if real:
    print("\nenv (set explicitly):")
    for k, v in real.items():
        print(f"  {k}={v}")
if env:
    print("env (defaults, leave unset unless measuring):")
    for k, v in env.items():
        if str(v).startswith("(unset)"):
            print(f"  {k}  {v}")

argv = m.get("argv") or []
if m.get("engine") and argv:
    binp = f"$root/build-gpuserve/bin/tools/{m['engine']}" if m["engine"].startswith("limbo_serve") else m["engine"]
    prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in real.items())
    argv = [resolved if a == "{path}" else a for a in argv]
    print("\ncommand:")
    print(f"  {prefix + ' ' if prefix else ''}{binp} " + " ".join(shlex.quote(a) for a in argv))
    if any(k == "LIMBO_Q4E_TIME" for k, v in env.items() if not str(v).startswith("(unset)")):
        print("  # with LIMBO_Q4E_TIME=1 the engine prints a per-token line at exit: dense/sync/lane/ple/host,")
        print("  # launch/syncwait, majf/minf, resident%, and how many experts the lane read from ram vs file.")
PY
