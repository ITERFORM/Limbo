#!/usr/bin/env bash
# Copyright (c) 2026 ITERFORM. All rights reserved.
# Qwen3.8-Flash-Next Coder (qwen4exp): its MTP layer + ngram, K/V q8, the experts VRAM does not hold in pinned RAM.
. "$(dirname "$0")/_common.sh"
exec "$BIN" -hf ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF:IQ1_M "${args[@]}" "$@"
