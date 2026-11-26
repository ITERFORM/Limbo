#!/usr/bin/env bash
# Copyright (c) 2026 ITERFORM. All rights reserved.
# Qwen3.8-27B dense (qwen35, the -mtp file): MTP block + ngram + DFlash2.
. "$(dirname "$0")/_common.sh"
exec "$BIN" -hf ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF:IQ3_XXS "${args[@]}" "$@"
