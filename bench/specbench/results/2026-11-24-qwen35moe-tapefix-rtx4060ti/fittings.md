# specbench fittings — 2026-11-24

machine: NVIDIA GeForce RTX 4060 Ti, 16380 MiB, 615.71.09 · AMD Ryzen 7 3800X 8-Core Processor · 32802056 kB · Linux 7.2.7-zen1-1-zen
model: /home/alca/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-MTP-GGUF/snapshots/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf · repo 2eba89f · objective: eval_tps · 3 runs x 512 tokens

Correctness: every speculative config must match its plain twin's greedy text; stability: all runs of a config must agree. Failed rows leave the ranking and lead the report. One campaign is one sample: treat gaps inside the run spread as ties.

## profile long
| rank | config | objective | depth | prefill tok/s | eval tok/s (min-max) | client | ttft | out | kv/planes/spec | correct | stable |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `ctx=262144-vram-reserve-mib=768-spec=auto` | 46.33 | 258813 | 531.78 | 46.00-46.67 | 45.12 | 0.30 | 512 | auto/2/auto | True | True |
| 2 | `ctx=262144-vram-reserve-mib=768-spec=off` | 44.86 | 258813 | 513.71 | 44.60-45.01 | 43.72 | 0.30 | 512 | auto/2/off | - | True |

**WINNER (long):** `ctx=262144-vram-reserve-mib=768-spec=auto` — 46.33 eval_tps
```bash
/home/alca/projects/limbo-engine/build/bin/tools/limbo_serve_gpu -m /home/alca/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-MTP-GGUF/snapshots/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf --host 127.0.0.1 --port 8090 --ctx 262144 --slots 1 --min-free-gb 0.0 --vram-reserve-mib 768 --spec auto
```

**Not measured (reality rows — the boundaries, kept as recorded):**
- `baseline`: prompt-too-long — the base prompt is 4977 tokens, over the 2560-token target
- `spec=off`: prompt-too-long — the base prompt is 4977 tokens, over the 2560-token target

