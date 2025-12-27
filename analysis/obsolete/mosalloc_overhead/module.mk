MODULE_NAME := analysis/mosalloc_overhead
SUBMODULES := 

RESULT_FILE := $(MODULE_NAME)/mosalloc_glibc_overhead.csv
.PHONY: $(MODULE_NAME)

$(MODULE_NAME): results/single_page_size/mean.csv analysis/general_metrics/glibc_malloc/mean.csv
	$(BUILD_OVERHEAD) \
		--mosalloc_4k_mean=results/single_page_size/mean.csv \
		--glibc_malloc_mean=analysis/general_metrics/glibc_malloc/mean.csv \
		--benchmarks=$(INTERESTING_BENCHMARKS_LIST) \
		--output $(RESULT_FILE)

DELETE_TARGETS := $(addsuffix /delete,$(RESULT_FILE))
$(MODULE_NAME)/clean: $(DELETE_TARGETS)

include $(ROOT_DIR)/common.mk

