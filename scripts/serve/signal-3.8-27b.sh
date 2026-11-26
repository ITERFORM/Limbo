#!/usr/bin/env bash
# Copyright (c) 2026 ITERFORM. All rights reserved.
# Signal-3.8-27B (agentionai, a Qwen3.8-27B fine-tune: qwen35, its MTP block in the file): MTP block + ngram + the 27B's DFlash2.
. "$(dirname "$0")/_common.sh"
exec "$BIN" -hf agentionai/Signal-3.8-27B-GGUF:IQ3_S "${args[@]}" "$@"
