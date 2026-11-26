# Copyright (c) 2026 ITERFORM. All rights reserved.
# scripts/serve/_common.sh -- sourced by scripts/serve/<model>.sh: `limbo -hf <repo:QUANT>` at this box's address. The rest is the
# server's own fit at start (include/limbo/hf.hpp, --ctx auto): the model from ~/models, the HF cache or a download; context
# 65536 (the stack's budget: never more than ~90K, above 64K luxury); the drafters the zoo holds for what the model is (MTP block,
# ngram, DFlash2 / DSpark), then the most exact K/V format beside them.
#
# Knobs, only when set:  CTX  KV  SLOTS  CKPT  RESERVE (MiB, default 256)  DRAFT=0 (no external drafter)
#                        BIN (this checkout's build/bin/limbo)  HOST (127.0.0.1)  PORT (8080)  V=2 (-vv)
# Arguments after the script's name go to the server as they are.
set -eu
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN=${BIN:-$root/build/bin/limbo}
[ -x "$BIN" ] || { echo "no server binary at $BIN (make tools)" >&2; exit 1; }
args=(--host "${HOST:-127.0.0.1}" --port "${PORT:-8080}" --vram-reserve-mib "${RESERVE:-256}")
if [ -n "${CTX:-}" ]; then args+=(--ctx "$CTX"); fi
if [ -n "${KV:-}" ]; then args+=(--kv "$KV"); fi
if [ -n "${SLOTS:-}" ]; then args+=(--slots "$SLOTS"); fi
if [ -n "${CKPT:-}" ]; then args+=(--checkpoints "$CKPT"); fi
if [ "${DRAFT:-1}" = 0 ]; then args+=(--draft-model none); fi
if [ "${V:-1}" = 2 ]; then args+=(-vv); else args+=(-v); fi
