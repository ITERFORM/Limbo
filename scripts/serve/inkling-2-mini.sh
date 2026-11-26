#!/usr/bin/env bash
# Copyright (c) 2026 ITERFORM. All rights reserved.
# Inkling-2-Mini (inkling2_loop, its own device pass, one conversation): DSpark.
. "$(dirname "$0")/_common.sh"
exec "$BIN" -hf unsloth/Inkling-2-Mini-GGUF:IQ2_XXS "${args[@]}" "$@"
