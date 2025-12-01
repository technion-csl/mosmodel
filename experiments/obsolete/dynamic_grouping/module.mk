MODULE_NAME := experiments/dynamic_grouping

DYNAMIC_GROUPING_EXPERIMENT := $(MODULE_NAME)
DYNAMIC_GROUPING_RESULTS := $(subst experiments,results,$(DYNAMIC_GROUPING_EXPERIMENT))
DYNAMIC_GROUPING_NUM_OF_REPEATS := $(NUMBER_OF_SOCKETS)
NUM_LAYOUTS := 55
NUM_OF_REPEATS := $(DYNAMIC_GROUPING_NUM_OF_REPEATS)
undefine LAYOUTS #allow the template to create new layouts based on the new NUM_LAYOUTS

include $(EXPERIMENTS_TEMPLATE)

CREATE_DYNAMIC_GROUPING_LAYOUTS_SCRIPT := $(MODULE_NAME)/createLayouts.py
$(LAYOUT_FILES): $(DYNAMIC_GROUPING_EXPERIMENT)/layouts/%.csv: $(MEMORY_FOOTPRINT_FILE) analysis/pebs_tlb_miss_trace/mem_bins_2mb.csv
	mkdir -p results/dynamic_auto_mosalloc
	$(COLLECT_RESULTS) --experiments_root=$(DYNAMIC_GROUPING_EXPERIMENT) --repeats=$(NUM_OF_REPEATS) \
		--output_dir=$(DYNAMIC_GROUPING_RESULTS) --remove_outliers
	$(CREATE_DYNAMIC_GROUPING_LAYOUTS_SCRIPT) \
		--memory_footprint=$(MEMORY_FOOTPRINT_FILE) \
		--pebs_mem_bins=$(MEM_BINS_2MB_CSV_FILE) \
		--layout=$* \
		--exp_dir=$(dir $@)/.. \
		--mean_file=$(DYNAMIC_GROUPING_RESULTS)/mean.csv

override undefine NUM_LAYOUTS
override undefine NUM_OF_REPEATS
override undefine LAYOUTS
