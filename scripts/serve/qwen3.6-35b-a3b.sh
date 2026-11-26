#!/usr/bin/env bash
# Copyright (c) 2026 ITERFORM. All rights reserved.
# Qwen3.6-35B-A3B (qwen35moe): MTP block + ngram + DFlash2, the whole model in VRAM.
. "$(dirname "$0")/_common.sh"
exec "$BIN" -hf unsloth/Qwen3.6-35B-A3B-MTP-GGUF:IQ3_XXS "${args[@]}" "$@"
