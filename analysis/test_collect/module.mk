MODULE_NAME := analysis/test_collect
SUBMODULES :=

$(MODULE_NAME): $(COLLECT_RESULTS)
	$(COLLECT_RESULTS) -e analysis -c test_collect \
		-b my_gups/1GB,my_gups/4GB -r 3 -o $@/results/
	diff -r $@/results $@/expected_results && rm -rf $@/results

include $(ROOT_DIR)/common.mk

