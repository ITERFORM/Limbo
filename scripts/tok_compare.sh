#!/usr/bin/env bash
# Copyright (c) 2026 ITERFORM. All rights reserved.
# scripts/tok_compare.sh — the W0 tokenizer gate: Limbo's ids must equal llama-tokenize's ids, byte
# for byte, over the corpus in bench/corpus/ (including the tool-call payloads).
#
#   scripts/tok_compare.sh <model.gguf> [oracle-binary] [corpus-dir]
#
# Two passes per file: raw text (--no-parse-special, the default of both tools) and with the special
# tokens parsed, which is how a rendered chat prompt arrives.
set -u

MODEL="${1:?usage: tok_compare.sh <model.gguf> [llama-tokenize] [corpus-dir]}"
ORACLE="${2:-/home/alca/projects/llama.cpp/build/bin/llama-tokenize}"
CORPUS="${3:-$(dirname "$0")/../bench/corpus}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LIMBO="${TOK_DUMP:-$ROOT/build/bin/tools/tok_dump}"
JOBS="${TOK_JOBS:-$(( $(nproc) / 2 ))}"

[ -x "$LIMBO" ] || { echo "tok_compare: build first: make tools (or set TOK_DUMP)" >&2; exit 2; }
[ -x "$ORACLE" ] || { echo "tok_compare: no oracle at $ORACLE" >&2; exit 2; }

# One job per (file, mode); the jobs are independent, so they run $JOBS at a time and report in order.
# --no-escape: llama-tokenize processes \n, \t and \\ in the input file by default (-e, default true),
# which would tokenize a different byte string than the one on disk. Limbo always reads the file as-is.
compare_one() {
    local f="$1" mode="$2" extra oextra want got n
    # raw: neither tool takes special tokens out of the text (the oracle parses by default).
    # special: both do.
    if [ "$mode" = special ]; then extra="--parse-special"; oextra=""; else extra=""; oextra="--no-parse-special"; fi
    want="$("$ORACLE" -m "$MODEL" -f "$f" --ids --no-bos --no-escape $oextra 2>/dev/null | tail -1)"
    got="$("$LIMBO" -m "$MODEL" -f "$f" --ids --no-bos $extra 2>/dev/null | tail -1)"
    if [ -z "$want" ]; then
        echo "SKIP $(basename "$f") [$mode]: the oracle printed nothing"
        return
    fi
    if [ "$want" = "$got" ]; then
        n="$(printf '%s' "$want" | tr -cd ',' | wc -c)"
        echo "ok   $(basename "$f") [$mode]: $((n + 1)) tokens"
    else
        echo "FAIL $(basename "$f") [$mode]"
        python3 - "$want" "$got" <<'PY'
import sys
w = [int(x) for x in sys.argv[1].strip().strip('[]').split(',') if x.strip()]
g = [int(x) for x in sys.argv[2].strip().strip('[]').split(',') if x.strip()]
print(f"       oracle {len(w)} tokens, limbo {len(g)} tokens")
for i in range(max(len(w), len(g))):
    a = w[i] if i < len(w) else None
    b = g[i] if i < len(g) else None
    if a != b:
        lo = max(0, i - 4)
        print(f"       first difference at {i}: oracle {w[lo:i+4]} vs limbo {g[lo:i+4]}")
        break
PY
    fi
}

mkdir -p "$ROOT/build/tmp"   # never the shared /tmp
OUT="$(mktemp -d "$ROOT/build/tmp/tok_compare.XXXXXX")"
i=0
for f in "$CORPUS"/*.txt; do
    for mode in raw special; do
        while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
        compare_one "$f" "$mode" > "$OUT/$(printf '%04d' "$i").txt" &
        i=$((i + 1))
    done
done
wait
cat "$OUT"/*.txt
pass="$(cat "$OUT"/*.txt | grep -c '^ok')"
fail="$(cat "$OUT"/*.txt | grep -c '^FAIL')"
rm -rf "$OUT"

echo
echo "tok_compare: $pass matched, $fail differed"
[ "$fail" -eq 0 ] || exit 1
