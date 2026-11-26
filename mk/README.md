# mk/

One fragment per subsystem whose build does not fit the generic rules in `Makefile`
(`src/*.cpp` → `liblimbo.a`, `tests/*_test.cpp` and `tools/*.cpp` → binaries linking it).
The Makefile includes `mk/*.mk` before its rules, so a fragment may append to `TEST_BINS`
and `TOOL_BINS` and define explicit rules (an explicit rule wins over the pattern rules).

| fragment | owns |
|---|---|
| `cuda.mk` | `src/cuda/*.cu`, `tools/*.cu`, `tests/*_test.cu` — nvcc, sm_89 |
| `oracle.mk` | `oracle/oracle_noise.cpp`, `oracle/oracle_nodes.cpp` — link llama.cpp, never liblimbo |
| `serve_gpu.mk` | `tools/limbo_serve_gpu.cu` — `limbo serve` on the GPU (links `liblimbo_cuda.a`) |

Every agent builds in its own root (`make BUILD=build-<name>`): parallel makes sharing one `build/` race on `liblimbo.a`.
Fragments use `$(BUILD)`, `$(OBJ)`, `$(BIN)`, never a literal `build/`.
