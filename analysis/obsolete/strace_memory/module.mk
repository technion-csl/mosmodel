MODULE_NAME := analysis/strace_memory
SUBMODULES := 

include $(ROOT_DIR)/common.mk

CALCULATE_POOL_SIZES := $(ROOT_DIR)/$(MODULE_NAME)/calculatePoolSizes.py

CSV_FILE_NAME := pool_sizes.csv
CSV_FILES := $(addprefix $(MODULE_NAME)/,$(INTERESTING_BENCHMARKS))
CSV_FILES := $(addsuffix /$(CSV_FILE_NAME),$(CSV_FILES))

$(MODULE_NAME): $(CSV_FILES)

$(CSV_FILES): $(MODULE_NAME)/%/$(CSV_FILE_NAME): | experiments/strace_memory/%
	$(CALCULATE_POOL_SIZES) --input_dir=$| --output_dir=$(dir $@)

DELETE_TARGETS := $(addsuffix /delete,$(CSV_FILES))
$(MODULE_NAME)/clean: $(DELETE_TARGETS)

