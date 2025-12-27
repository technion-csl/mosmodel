MODULE_NAME := analysis/perf_mem/test_findHalfWeightWindow
SUBMODULES :=

TEST_INP_FILES := $(wildcard $(MODULE_NAME)/input*)
TEST_OUT_FILES := $(subst input,output,$(TEST_INP_FILES))

$(MODULE_NAME): $(TEST_OUT_FILES)

$(TEST_OUT_FILES): $(MODULE_NAME)/output%: $(MODULE_NAME)/input%
	$(FIND_WINDOW) --input=$< --output=$@
	diff $@ $(subst output,expected_output,$@)
	rm -f $@

DELETE_TARGETS := $(addsuffix /delete,$(TEST_OUT_FILES))
$(MODULE_NAME)/clean: $(DELETE_TARGETS)

include $(ROOT_DIR)/common.mk


