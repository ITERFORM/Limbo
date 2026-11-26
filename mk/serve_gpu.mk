# mk/serve_gpu.mk — `limbo serve` on the GPU: tools/limbo_serve_gpu.cu (links liblimbo_cuda.a, which holds
# src/cuda/gpu_backend.cu beside the engine). Uses mk/cuda.mk's NVCC, NVCCFLAGS and CUDA_LIB; with no nvcc
# CUDA_LIB is undefined and this fragment adds nothing. `make serve-gpu` builds just this tool; `make install` puts `limbo` on
# the PATH.

ifdef CUDA_LIB

TOOL_BINS += $(BIN)/tools/limbo_serve_gpu

.PHONY: serve-gpu
serve-gpu: $(BIN)/tools/limbo_serve_gpu

$(BIN)/tools/limbo_serve_gpu: tools/limbo_serve_gpu.cu $(CUDA_LIB) $(LIB) $(wildcard include/limbo/cuda/*.hpp)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) $< $(CUDA_LIB) $(LIB) -o $@

# `limbo`: the same server under its short name (`limbo -hf org/repo:QUANT`). build/bin/limbo beside the tools; `make install`
# links it into $(PREFIX)/bin (~/.local/bin), pointing at this checkout's build.
PREFIX ?= $(HOME)/.local
TOOL_BINS += $(BIN)/limbo
$(BIN)/limbo: $(BIN)/tools/limbo_serve_gpu
	ln -sf tools/limbo_serve_gpu $@

.PHONY: install
install: $(BIN)/limbo
	@mkdir -p $(PREFIX)/bin
	ln -sf $(abspath $(BIN)/tools/limbo_serve_gpu) $(PREFIX)/bin/limbo

endif
