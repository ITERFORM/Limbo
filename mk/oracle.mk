# mk/oracle.mk — the measuring devices beside oracle_dump. They link the operator's llama.cpp build,
# never liblimbo (docs/CLEANROOM.md rule 0).
#
#   oracle_noise   llama.cpp against itself: the capture's tokens, teacher-forced, batch vs one-token decode
#   oracle_nodes   llama.cpp's l_out-N / attn_residual-N nodes in full, for ref_forward --layer-gate

ORACLE_EXTRA := $(BIN)/oracle/oracle_noise $(BIN)/oracle/oracle_nodes
oracle: $(ORACLE_EXTRA)

$(BIN)/oracle/oracle_noise: oracle/oracle_noise.cpp
	@mkdir -p $(dir $@)
	$(CXX) -std=c++20 -O2 -g -Wall -Wextra -I$(LLAMA_DIR)/include -I$(LLAMA_DIR)/ggml/include $< -L$(LLAMA_LIB_DIR) -lllama \
		-Wl,-rpath,$(LLAMA_LIB_DIR) -o $@

$(BIN)/oracle/oracle_nodes: oracle/oracle_nodes.cpp
	@mkdir -p $(dir $@)
	$(CXX) -std=c++20 -O2 -g -Wall -Wextra -I$(LLAMA_DIR)/include -I$(LLAMA_DIR)/ggml/include $< -L$(LLAMA_LIB_DIR) -lllama \
		-lggml-base -lggml -Wl,-rpath,$(LLAMA_LIB_DIR) -o $@

# The projector's embeddings (mtmd): the yardstick of tools/vision_check. Links libmtmd and libllama, never liblimbo.
ORACLE_EXTRA += $(BIN)/oracle/oracle_vision
$(BIN)/oracle/oracle_vision: oracle/oracle_vision.cpp
	@mkdir -p $(dir $@)
	$(CXX) -std=c++20 -O2 -g -Wall -Wextra -I$(LLAMA_DIR)/include -I$(LLAMA_DIR)/ggml/include -I$(LLAMA_DIR)/tools/mtmd $< \
		-L$(LLAMA_LIB_DIR) -lmtmd -lllama -lggml-base -lggml -Wl,-rpath,$(LLAMA_LIB_DIR) -o $@
