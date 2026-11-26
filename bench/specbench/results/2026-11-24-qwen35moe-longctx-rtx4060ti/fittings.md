# specbench fittings — 2026-11-24

machine: NVIDIA GeForce RTX 4060 Ti, 16380 MiB, 615.71.09 · AMD Ryzen 7 3800X 8-Core Processor · 32802056 kB · Linux 7.2.7-zen1-1-zen
model: /home/alca/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-MTP-GGUF/snapshots/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf · repo ccbba52 · objective: eval_tps · 5 runs x 512 tokens

Correctness: every speculative config must match its plain twin's greedy text; stability: all runs of a config must agree. Failed rows leave the ranking and lead the report. One campaign is one sample: treat gaps inside the run spread as ties.

## profile long
| rank | config | objective | depth | prefill tok/s | eval tok/s (min-max) | client | ttft | out | kv/planes/spec | correct | stable |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `ctx=32768-spec=auto-kv=auto` | 103.94 | 29871 | 1716.29 | 103.04-105.33 | 102.31 | 0.08 | 512 | auto/2/auto | True | True |
| 2 | `ctx=32768-spec=off-kv=auto` | 80.10 | 29871 | 1708.64 | 79.98-80.24 | 79.11 | 0.08 | 512 | auto/2/off | - | True |

**WINNER (long):** `ctx=32768-spec=auto-kv=auto` — 103.94 eval_tps
```bash
/home/alca/projects/limbo-engine/build/bin/tools/limbo_serve_gpu -m /home/alca/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-MTP-GGUF/snapshots/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf --host 127.0.0.1 --port 8090 --ctx 32768 --slots 1 --min-free-gb 0.0 --vram-reserve-mib 768 --spec auto --kv auto
```

**Not measured (reality rows — the boundaries, kept as recorded):**
- `baseline`: prompt-too-long — the base prompt is 4977 tokens, over the 2560-token target
- `ctx=4096-spec=off-kv=auto`: prompt-too-long — the base prompt is 4977 tokens, over the 2560-token target
- `ctx=4096-spec=auto-kv=auto`: prompt-too-long — the base prompt is 4977 tokens, over the 2560-token target
- `ctx=262144-spec=off-kv=auto`: error
- `ctx=262144-spec=auto-kv=auto`: error
- `ctx=300000-spec=off-kv=auto`: refused
- `ctx=300000-spec=auto-kv=auto`: refused
- `spec=off`: prompt-too-long — the base prompt is 4977 tokens, over the 2560-token target

