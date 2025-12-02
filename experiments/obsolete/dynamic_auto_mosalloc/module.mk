MODULE_NAME := experiments/dynamic_auto_mosalloc

DYNAMIC_AUTO_MOSALLOC_EXPERIMENT := $(MODULE_NAME)
DYNAMIC_AUTO_MOSALLOC_RESULTS := $(subst experiments,results,$(DYNAMIC_AUTO_MOSALLOC_EXPERIMENT))
DYNAMIC_AUTO_MOSALLOC_NUM_OF_REPEATS := 4
NUM_LAYOUTS := 50
NUM_OF_REPEATS := $(DYNAMIC_AUTO_MOSALLOC_NUM_OF_REPEATS)
undefine LAYOUTS #allow the template to create new layouts based on the new NUM_LAYOUTS

include $(EXPERIMENTS_TEMPLATE)

CREATE_DYNAMIC_AUTO_MOSALLOC_LAYOUTS_SCRIPT := $(MODULE_NAME)/createLayouts.py
$(LAYOUT_FILES): $(DYNAMIC_AUTO_MOSALLOC_EXPERIMENT)/layouts/%.csv: $(MEMORY_FOOTPRINT_FILE) analysis/pebs_tlb_miss_trace/mem_bins_2mb.csv
	mkdir -p results/dynamic_auto_mosalloc
	$(COLLECT_RESULTS) --experiments_root=$(DYNAMIC_AUTO_MOSALLOC_EXPERIMENT) --repeats=$(NUM_OF_REPEATS) \
		--output_dir=$(DYNAMIC_AUTO_MOSALLOC_RESULTS)
	$(CREATE_DYNAMIC_AUTO_MOSALLOC_LAYOUTS_SCRIPT) \
		--memory_footprint=$(MEMORY_FOOTPRINT_FILE) \
		--pebs_mem_bins=$(MEM_BINS_2MB_CSV_FILE) \
		--layout=$* \
		--exp_dir=$(dir $@)/.. \
		--results_file=$(DYNAMIC_AUTO_MOSALLOC_RESULTS)/median.csv

override undefine NUM_LAYOUTS
override undefine NUM_OF_REPEATS
override undefine LAYOUTS
