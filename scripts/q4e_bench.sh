#!/usr/bin/env bash
# Copyright (c) 2026 ITERFORM. All rights reserved.
# scripts/q4e_bench.sh -- q4e measurements on one engine start: limbo_serve_gpu on the Coder IQ1_M pair, one completion per
# prompt file in order (the same file twice = the repeated-text case), the engine's own clock kept. Scratch in build/tmp.
#
#   scripts/q4e_bench.sh <tag> <max-tokens> <prompt-file>... [-- extra serve flags...]
#
# The standard config (Strata's for this model): ctx 131072, K/V q8, one slot; speculation is the model's own (MTP + ngram),
# never a --spec of the bench's choosing. Env passes through (LIMBO_Q4E_*); CTX and BIN override; THINK=0 sends enable_thinking false.
set -u
cd "$(dirname "$0")/.."
tag="$1"; mtok="$2"; shift 2
prompts=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do prompts+=("$1"); shift; done
[ "${1:-}" = "--" ] && shift

ROOT=build/tmp/bench
mkdir -p "$ROOT"
M=/home/alca/projects/Strata-data/models/coder-IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf
BIN=${BIN:-build/bin/tools/limbo_serve_gpu}
CTX=${CTX:-131072}
LOG="$ROOT/$tag.log"
PORT=$(python3 -c "import socket; s = socket.socket(); s.bind((\"127.0.0.1\", 0)); print(s.getsockname()[1])")

# Own process group, so a kill of this script takes the engine with it (a stray engine holds the card).
setsid env LIMBO_Q4E_TIME=1 LIMBO_Q4E_PF_LOG=1 "$BIN" -m "$M" --host 127.0.0.1 --port "$PORT" --ctx "$CTX" \
    --kv q8 --slots 1 --checkpoints 0 --vram-reserve-mib 256 -v "$@" >"$LOG" 2>&1 &
pid=$!
stop() { kill -TERM -- -$pid 2>/dev/null; sleep 1; kill -KILL -- -$pid 2>/dev/null; }
trap stop EXIT INT TERM

for i in $(seq 1 400); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    kill -0 $pid 2>/dev/null || { echo "engine died:"; tail -30 "$LOG"; exit 1; }
    sleep 1
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || { echo "engine never came up:"; tail -30 "$LOG"; exit 1; }
grep -E "q4e experts|warm" "$LOG" | tail -2

n=0
for p in "${prompts[@]}"; do
    n=$((n + 1))
    from=$(wc -l <"$LOG")
    curl -s -X POST "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "$(python3 - "$p" "$mtok" <<'PY'
import json,os,sys
p=open(sys.argv[1]).read()
b={"messages":[{"role":"user","content":p}],"max_tokens":int(sys.argv[2]),"temperature":0,"top_k":1}
if os.environ.get("THINK") == "0": b["enable_thinking"]=False
print(json.dumps(b))
PY
)" >"$ROOT/$tag.$n.json" 2>/dev/null || echo "request $n failed"
    python3 - "$ROOT/$tag.$n.json" "$LOG" "$from" "$(basename "$p")" <<'PY'
import json,re,sys
r=json.load(open(sys.argv[1])); t=r.get("timings",{})
lines=open(sys.argv[2]).read().splitlines()[int(sys.argv[3]):]
pat=re.compile(r"\[q4e (\S+)\s+pos\s+\d+\]\s+([\d.]+) ms = dense\s+([\d.]+) \+ sync\s+([\d.]+) \+ lane\s+([\d.]+) \+ ple\s+([\d.]+) \+ host\s+(-?[\d.]+).*resident\s+([\d.]+)%")
rows=[m.groups() for m in map(pat.search,lines) if m]
st=[list(map(float,g[1:])) for g in rows if g[0]=="step"][8:]
print(f"{sys.argv[4]:>10}: prompt {t.get('prompt_n')} tok {t.get('prompt_per_second',0):7.1f} tok/s | "
      f"decode {t.get('predicted_n')} tok {t.get('predicted_per_second',0):6.2f} tok/s ({t.get('predicted_per_token_ms',0):.1f} ms)")
if st:
    k=len(st); a=[sum(c)/k for c in zip(*st)]
    print(f"{'':>10}  step x{k}: {a[0]:.1f} ms = dense {a[1]:.1f} + sync {a[2]:.1f} + lane {a[3]:.1f} + ple {a[4]:.1f} + host {a[5]:.1f} | hit {a[6]:.1f}%")
w=[g for g in rows if g[0].startswith("w")]
if w: print(f"{'':>10}  windows: {len(w)}")
PY
done
stop; trap - EXIT INT TERM
grep -E "rebalance|prefill" "$LOG" | tail -4
echo "log: $LOG"
