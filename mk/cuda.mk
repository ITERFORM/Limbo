# mk/cuda.mk — the CUDA side of Limbo: src/cuda/*.cu -> liblimbo_cuda.a, tests/cuda_*_test.cu, tools/gpu_forward.cu.
#
# nvcc builds for sm_89 (the RTX 4060 Ti); host code is held to the same -Wall -Wextra -Werror as the
# C++ side. Everything here lives under $(BUILD), so parallel agents with their own BUILD never collide.
# `make test` builds the CUDA tests and runs them with the rest; a test that cannot get a GPU (or enough
# free VRAM beside the desktop) says SKIP and exits 0. `make tools` builds gpu_forward.

NVCC       ?= $(firstword $(wildcard /opt/cuda/bin/nvcc) nvcc)
NVCCFLAGS  ?= -std=c++20 -O2 -arch=sm_89 -lineinfo -Xcompiler -Wall,-Wextra,-Werror

# No nvcc, no CUDA targets: the CPU build, tests and tools are unaffected.
ifneq ($(shell command -v $(NVCC) 2>/dev/null),)

CUDA_SRCS := $(wildcard src/cuda/*.cu)
CUDA_OBJS := $(patsubst src/cuda/%.cu,$(OBJ)/cuda/%.o,$(CUDA_SRCS))
CUDA_LIB  := $(BUILD)/liblimbo_cuda.a

CUDA_TEST_SRCS := $(wildcard tests/cuda_*_test.cu)
CUDA_TEST_BINS := $(patsubst tests/%.cu,$(BIN)/tests/%,$(CUDA_TEST_SRCS))

TEST_BINS += $(CUDA_TEST_BINS)
# Every tools/*.cu is a CUDA tool (an explicit rule elsewhere, e.g. mk/serve_gpu.mk, wins over the pattern).
CUDA_TOOL_SRCS := $(wildcard tools/*.cu)
CUDA_TOOL_BINS := $(patsubst tools/%.cu,$(BIN)/tools/%,$(CUDA_TOOL_SRCS))
TOOL_BINS += $(CUDA_TOOL_BINS)

.PHONY: cuda
cuda: $(CUDA_LIB) $(CUDA_TEST_BINS) $(CUDA_TOOL_BINS)

$(OBJ)/cuda/%.o: src/cuda/%.cu $(wildcard include/limbo/*.hpp) $(wildcard include/limbo/cuda/*.hpp) $(wildcard src/cuda/*.hpp src/cuda/*.cuh)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -c $< -o $@

$(CUDA_LIB): $(CUDA_OBJS)
	@mkdir -p $(dir $@)
	$(AR) rcs $@ $^

$(BIN)/tests/cuda_%_test: tests/cuda_%_test.cu $(CUDA_LIB) $(LIB) $(wildcard include/limbo/cuda/*.hpp)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) $< $(CUDA_LIB) $(LIB) -o $@

$(BIN)/tools/%: tools/%.cu $(CUDA_LIB) $(LIB) $(wildcard include/limbo/cuda/*.hpp)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) $< $(CUDA_LIB) $(LIB) -o $@

endif
