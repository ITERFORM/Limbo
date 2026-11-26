# specbench fittings — 2026-11-24

machine: NVIDIA GeForce RTX 4060 Ti, 16380 MiB, 615.71.09 · AMD Ryzen 7 3800X 8-Core Processor · 32802056 kB · Linux 7.2.7-zen1-1-zen
model: /home/alca/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-MTP-GGUF/snapshots/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf · repo 4126b1d · objective: eval_tps · 5 runs x 512 tokens

Correctness: every speculative config must match its plain twin's greedy text; stability: all runs of a config must agree. Failed rows leave the ranking and lead the report. One campaign is one sample: treat gaps inside the run spread as ties.

## profile long
**Excluded (failed the gates):**
- `ctx=262144-vram-reserve-mib=768-spec=auto`: greedy text diverged from its plain twin

| rank | config | objective | depth | prefill tok/s | eval tok/s (min-max) | client | ttft | out | kv/planes/spec | correct | stable |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `ctx=262144-vram-reserve-mib=256-spec=auto` | 45.91 | 258813 | 517.19 | 45.69-46.15 | 44.72 | 0.30 | 512 | auto/2/auto | True | True |
| 2 | `ctx=262144-vram-reserve-mib=256-spec=off` | 45.44 | 258813 | 511.41 | 43.13-47.07 | 44.30 | 0.29 | 512 | auto/2/off | True | True |
| 3 | `ctx=262144-vram-reserve-mib=768-spec=off` | 44.68 | 258813 | 509.60 | 44.47-44.94 | 43.55 | 0.30 | 512 | auto/2/off | - | True |

**WINNER (long):** `ctx=262144-vram-reserve-mib=256-spec=auto` — 45.91 eval_tps
```bash
/home/alca/projects/limbo-engine/build/bin/tools/limbo_serve_gpu -m /home/alca/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-MTP-GGUF/snapshots/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf --host 127.0.0.1 --port 8090 --ctx 262144 --slots 1 --min-free-gb 0.0 --vram-reserve-mib 256 --spec auto
```

**Not measured (reality rows — the boundaries, kept as recorded):**
- `baseline`: prompt-too-long — the base prompt is 4977 tokens, over the 2560-token target
- `spec=off`: prompt-too-long — the base prompt is 4977 tokens, over the 2560-token target

**Tape misses (the cache served nothing — every run re-prefilled the full prompt; a divergence that lost the tape, worth an engine look):**
- `ctx=262144-vram-reserve-mib=768-spec=auto`: depth 258813, cached 0 of it, ttft_uncached 503.78s

