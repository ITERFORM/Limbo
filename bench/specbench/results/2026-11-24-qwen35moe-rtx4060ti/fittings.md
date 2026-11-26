# specbench fittings — 2026-11-24

machine: NVIDIA GeForce RTX 4060 Ti, 16380 MiB, 615.71.09 · AMD Ryzen 7 3800X 8-Core Processor · 32802056 kB · Linux 7.2.7-zen1-1-zen
model: /home/alca/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-MTP-GGUF/snapshots/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf · repo eb327bf · objective: eval_tps · 3 runs x 128 tokens

Correctness: every speculative config must match its plain twin's greedy text; stability: all runs of a config must agree. Failed rows leave the ranking and lead the report. One campaign is one sample: treat gaps inside the run spread as ties.

## profile code
| rank | config | objective | eval tok/s (min-max) | client | ttft | out | kv/planes/spec | correct | stable |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `kv=f16-spec=auto-prefill-planes=2` | 179.52 | 178.44-180.77 | 168.98 | 0.04 | 128 | f16/2/auto | True | True |
| 2 | `kv=q8-spec=auto-prefill-planes=2` | 179.47 | 176.90-181.02 | 168.97 | 0.04 | 128 | q8/2/auto | True | True |
| 3 | `kv=q4-spec=auto-prefill-planes=2` | 167.64 | 165.68-169.23 | 158.48 | 0.04 | 128 | q4/2/auto | True | True |
| 4 | `baseline` | 166.85 | 163.34-169.80 | 157.40 | 0.05 | 128 | auto/2/auto | True | True |
| 5 | `kv=f16-spec=off-prefill-planes=2` | 101.36 | 101.17-101.45 | 97.96 | 0.04 | 128 | f16/2/off | - | True |
| 6 | `kv=q4-spec=off-prefill-planes=2` | 100.49 | 100.30-100.64 | 97.13 | 0.04 | 128 | q4/2/off | - | True |
| 7 | `kv=q8-spec=off-prefill-planes=2` | 99.99 | 98.97-100.57 | 96.67 | 0.04 | 128 | q8/2/off | - | True |
| 8 | `spec=off` | 98.08 | 97.31-99.39 | 94.77 | 0.05 | 128 | auto/2/off | - | True |

**WINNER (code):** `kv=f16-spec=auto-prefill-planes=2` — 179.52 eval_tps
```bash
/home/alca/projects/limbo-engine/build/bin/tools/limbo_serve_gpu -m /home/alca/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-MTP-GGUF/snapshots/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf --host 127.0.0.1 --port 8090 --ctx 4096 --min-free-gb 0.0 --vram-reserve-mib 768 --slots 1 --kv f16 --spec auto --prefill-planes 2
```

## profile tools
| rank | config | objective | eval tok/s (min-max) | client | ttft | out | kv/planes/spec | correct | stable |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `kv=f16-spec=auto-prefill-planes=2` | 136.65 | 135.35-138.67 | 130.15 | 0.04 | 128 | f16/2/auto | True | True |
| 2 | `kv=q4-spec=auto-prefill-planes=2` | 136.61 | 136.55-136.70 | 130.47 | 0.04 | 128 | q4/2/auto | True | True |
| 3 | `baseline` | 134.60 | 133.65-136.08 | 128.11 | 0.05 | 128 | auto/2/auto | True | True |
| 4 | `kv=q8-spec=auto-prefill-planes=2` | 133.08 | 132.92-133.25 | 126.92 | 0.05 | 128 | q8/2/auto | True | True |
| 5 | `spec=off` | 101.41 | 101.30-101.55 | 97.98 | 0.04 | 128 | auto/2/off | - | True |
| 6 | `kv=q4-spec=off-prefill-planes=2` | 98.24 | 96.24-100.07 | 95.01 | 0.04 | 128 | q4/2/off | - | True |
| 7 | `kv=f16-spec=off-prefill-planes=2` | 95.02 | 93.01-96.90 | 91.60 | 0.05 | 128 | f16/2/off | - | True |
| 8 | `kv=q8-spec=off-prefill-planes=2` | 94.62 | 94.52-94.69 | 91.45 | 0.05 | 128 | q8/2/off | - | True |

**WINNER (tools):** `kv=f16-spec=auto-prefill-planes=2` — 136.65 eval_tps
```bash
/home/alca/projects/limbo-engine/build/bin/tools/limbo_serve_gpu -m /home/alca/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-MTP-GGUF/snapshots/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf --host 127.0.0.1 --port 8090 --ctx 4096 --min-free-gb 0.0 --vram-reserve-mib 768 --slots 1 --kv f16 --spec auto --prefill-planes 2
```

