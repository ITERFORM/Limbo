# specbench — the fitting harness

Which limbo configuration is the best practical one **for a given model, machine and workload** —
measured on real prompts against the real server, with the correctness gate that makes a
recommendation survivable.

Migrated from our specbench project (the llama.cpp-adaptive-kv-streaming benchmark) and adapted to
limbo: the managed lifecycle, cartesian sweeps and ranking are specbench's; the server surface is
`limbo_serve` / `limbo_serve_gpu`, and every timing comes from the response's own `timings` block.

## The fitting space

A *fitting* is a point in the product of four axes. The harness exercises the product and ranks it:

| axis | swept by | values |
|---|---|---|
| model | `--model` | any supported GGUF (qwen35moe, qwen35, qwen4exp, inkling2_loop) |
| machine | (captured per row) | GPU + driver, CPU, RAM, kernel, repo HEAD, VRAM at start |
| workload (the user) | `--profile` + `--prompt-tokens` | the real prompts in `bench/prompts/`: `tools` (agent turn with a tool schema), `code` (quoted function), `prose`, `long`, `hi` — filled to the depth under test (see below) |
| context depth | `--sweep ctx=...` | a ladder up to the model's limit and **past it**: a row whose context does not fit is recorded (`refused` / `oom`), so the boundary is measured, never assumed |
| engine knobs | `--sweep FLAG=v1,v2` | `--kv auto\|f16\|q8\|q4`, `--spec auto\|mtp\|ngram\|dflash\|dspark\|off`, `--prefill-planes 1\|2\|3`, `--ngram-n`, `--ngram-max`, `--slots`, `--checkpoints`, `--draft-quant`, `--draft-vocab`, `--draft-k`, `--ctx` |

`--sweep-limbo` is the full practical grid (`kv` × `spec` × `prefill-planes`); `--sweep-file
grid.json` runs a campaign file's product (the `campaign/` shape). Sweeps are validated against the
server's real flag set **before the first boot** — a typo fails fast instead of costing a server
lifecycle. Missing plain twins (a config's numerics with `--spec off`) are run automatically: the
correctness gate needs them, so the grid always contains them.

## Running

Attached — against a server you started yourself:

```bash
bench/specbench/specbench.py --url http://127.0.0.1:8090 --profile tools
```

Managed — the harness starts and stops `limbo_serve_gpu` per configuration:

```bash
bench/specbench/specbench.py --launch \
    --model /models/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf \
    --profile long \
    --sweep ctx=4096,32768,262144,300000 --sweep spec=off,auto --sweep kv=auto \
    --prompt-tokens 0 --runs 5 --max-tokens 512 --request-timeout 1200 \
    --out bench/specbench/results/2026-11-24-qwen35moe-longctx
```

**Depth.** `--prompt-tokens N` fills the profile's real prompt to N tokens (repeated to length, the
`bench/prompts/long.txt` idiom); `--prompt-tokens 0` (the default) fills the *window*: each row's
context minus the generation headroom, so `--sweep ctx=...,262144` benchmarks a full 256k-class
context end to end. Token counts come from the GGUF's own tokenizer (`tools/tok_dump`, offline — no
server round trip), the fill never exceeds the target (the server's limit is hard), and the depth
actually reached is recorded per row. When the base prompt alone is larger than the target, the row
is recorded `prompt-too-long` without booting a server.

**Budgets.** Defaults are real: `--runs 5 --max-tokens 512`. Each row pays exactly one full prefill
(the recorded **probe** — the only uncached request; it carries `depth`, `prefill_tps` and
`ttft_uncached`), then the measured runs are cached-prompt decodes whose `eval_tps` / `client_tps`
/ `ttft` are decode-only numbers. A full-window row at 262k therefore costs one ~7-minute prefill
plus its decode runs, not one prefill per run.

Per configuration the managed mode kills any stale `limbo_serve*` on the port, launches the command
with the swept flags applied, waits for `/health` (bounded by `--boot-timeout`), keeps the server
log, runs warmup + measured completions at temperature 0 (each bounded by `--request-timeout`, enforced
as a wall-clock deadline across the SSE stream — the server's keepalives defeat socket timeouts; a
wedged request becomes a recorded error), records TTFT / client tok/s / the server's own
`predicted_per_second` and `prompt_per_second` / VRAM delta / load time, stops the server and waits
for the port to free. `results.json` and `results.csv` are rewritten after every row: a crash loses
the last row at worst, and `--resume` continues the campaign from exactly there, re-running only
rows that are missing or failed. `--dry-run` prints the plan and the token budget without running
anything.

The report ranks by `--objective` (default `eval_tps`, the server's decode rate; also `client_tps`,
`ttft`, `prompt_tps`) and prints **the winning launch command verbatim**, plus one fittings table
per profile with each row's run spread. `fittings.md` lands beside the results.

## The harness's own gate

`--self-test` runs the planning, correctness, stability and reporting machinery on synthetic rows,
including deliberate mutations (a drifted twin's text, runs that disagree with each other) that the
gate must catch. Sabotaging the harness itself must turn it red — run both directions before
trusting a green one:

```bash
bench/specbench/specbench.py --self-test
```

## Surviving contact with reality

The rules the harness enforces, not promises:

- **Correctness before speed.** At temperature 0 the greedy text of every speculative configuration
  must equal its plain twin — the same `--kv` / `--prefill-planes` with `--spec off`. Missing twins
  are run automatically. A configuration that diverges is marked `correct: false`, excluded from the
  ranking, and printed at the top of the report. This is limbo's own spec contract (the verify
  window accepts only the plain loop's tokens); the harness refuses to rank a config that breaks it.
- **Stability is part of correctness.** All runs of one configuration must produce the same greedy
  text; a configuration that flip-flops between runs is marked `stable: false` and excluded exactly
  like a diverging one. The verdicts are derived from the recorded text hashes, so they are
  re-derivable from `results.json` alone.
- **Numerics are grouped, not compared.** `--kv q4` is allowed to differ from `--kv f16` (documented
  numerics); each group's `--spec off` row is its own reference. The `text_sha` column makes every
  group's drift visible.
- **Every row carries the machine.** GPU, driver, CPU, RAM, kernel and repo HEAD are recorded in the
  results header. A fitting is claimed only for the machine that measured it.
- **Reality is recorded, not smoothed.** A refusal to load (exit 3), an OOM, a server death, a
  request error or a **prompt that cannot fit the context at all** becomes the row's `status` field
  (`refused` / `oom` / `error` / `prompt-too-long`) with its numbers and logs — never a dropped row.
  A `prompt-too-long` row is decided offline by the model's own tokenizer before any server boots:
  the workload cannot run at that depth on any configuration, and that is a fitting fact. The worse
  runs stay in the table.
- **The card is shared.** Run managed campaigns under `flock` when other jobs use the GPU; the
  harness itself never assumes it owns the machine.
- **One campaign is one sample.** Between-campaign spread on a shared desktop is real (observed
  winners differing by ~7% between two runs of the same grid): treat gaps inside that as ties and
  re-run before claiming a winner.

## Outputs

| file | what |
|---|---|
| `results.json` | header (objective, machine fingerprint, prompt digests) + every row, written incrementally |
| `results.csv` | the same rows, flat |
| `fittings.md` | per-profile ranked tables, the correctness failures first, the winning command verbatim |
| `logs/<profile>__<config>.log` | the server's own output per configuration |

Results directories under `bench/specbench/results/` are the artifacts of record — the same claims
discipline as the rest of `bench/`: a number that exists only in prose is a number nobody can check.
